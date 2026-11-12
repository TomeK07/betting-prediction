"""T10 for one team: last 30 days of news → two models → Python checks → vesti.md.

Sources (free, robots.txt respected, cached in data/raw/vesti/ and data/timovi/…/izvori/):
  all      Eurohoops team search (title must name the team)
  EL       Eurohoops round injury reports (the team's own section), Sportando daily EuroLeague injury report (Italian,
           the game paragraph), RotoWire Euro player news (free text only, one page a day)
  NBA      ESPN team news (headline/description must name the team or a roster player — general league stories are
           dropped BEFORE a model call, 31.10) and ESPN injury notes
Before the models: every source is filtered for relevance (saves the free quota).
Python checks every item: the quote is in the text; a named player is on the team's OFFICIAL roster (otherwise:
other team / former player → rejected); the source is from the last 30 days (= season 2026-27). Only items both
models report from the same source go to vesti.md (PROVERENO); the rest are listed as rejected / single-model.
Injury/lineup news from the last 24 h also go through T1 (status) → data/vesti/<date>.yaml for `picks`.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import yaml

from core.config import Settings
from model.team_dossier import VESTI_HEADER, read_news
from research.llm_client import LLMClient
from research.tasks import t10_vesti
from research.testset_build import EL_SEARCHES, ESPN_INJURIES, Fetcher, _eurohoops, _roster_team, _el_frames, norm, plain, surname
from research.validate import has_forbidden, quote_in_text

TEAMS_DIR = Path(__file__).resolve().parents[1] / "data" / "timovi"
VESTI_DIR = Path(__file__).resolve().parents[1] / "data" / "vesti"
ESPN_NEWS = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/news"
ESPN_STORY = "https://content.core.api.espn.com/v1/sports/news/{id}"
ESPN_TYPES = {"Story", "HeadlineNews", "Recap", "Preview"}
TEXT_CHARS = 6000
DAYS = 30
# a box-score number in any news type ("scoring 15 points", "won 84-82") → rejected; "Rounds 5 and 6" stays
STAT_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:points|pts|rebounds|assists|steals|blocks|turnovers|minutes|3-pointers)\b"
                     r"|\b\d{2,3}\s*[-–]\s*\d{2,3}\b", re.I)


def _src(url: str, published: str, title: str, text: str) -> dict:
    return {"izvor_url": url, "objaveno": published[:16], "naslov": title, "tekst": text[:TEXT_CHARS]}


SPORTANDO_API = "https://sportando.basketball/wp-json/wp/v2/posts"
ROTOWIRE_EURO = "https://www.rotowire.com/euro/news.php"
USE_ROTOWIRE = False      # Tome 31.10: off — terms of use page not found (parser kept, tested)
NAMES_FILE = Path(__file__).resolve().parents[1] / "config" / "timovi_iminja.yaml"


def team_names(league: str) -> dict:
    return (yaml.safe_load(NAMES_FILE.read_text(encoding="utf-8")) or {}).get(league) or {}


def league_frames(settings: Settings, league: str):
    """(team_logs, player_logs, roster) of the current season of any league."""
    import pandas as pd
    folder = settings.paths.processed / league / settings.league(league).season
    read = lambda n: pd.read_csv(folder / n, dtype={"game_code": str, "player_code": str})
    return read("team_logs.csv"), read("player_logs.csv"), read("roster.csv")


def _full_names(roster, code: str) -> list[str]:
    """'NAVARRO, IBON' / 'Ibon Navarro' → 'Ibon Navarro' (first + last), for finding a player in a text."""
    out = []
    for p in roster[roster["team"] == code]["player"].dropna():
        out.append(f"{p.split(',')[1].strip()} {p.split(',')[0].strip()}".title() if "," in p else p)
    return out


def _mentions(text: str, full_names: list[str]) -> bool:
    t = norm(text)
    return any(len(n.split()) >= 2 and f" {norm(n)} " in f" {t} " for n in full_names)


USE_EXCERPT = False                 # ON only after the T10 test with excerpts passes (Tome 1.11)
EXCERPT_FROM = 1500                 # chars; shorter texts go whole
EXCERPT_MAX = 3000


def excerpt(text: str, keys) -> str:
    """Tome 1.11 (quota): only the title + sentences naming the team or a roster player go to
    the model (1 sentence before, 2 after). Sentences are verbatim, so a quote is still checked against the full source text."""
    if len(text) <= EXCERPT_FROM:
        return text
    pats = [re.compile(rf"\b{re.escape(norm(k))}\b") for k in keys if k and len(norm(k)) > 3]
    parts = [p for p in re.split(r"(?<=[.!?])\s+|\n+", text) if p.strip()]
    hits = [i for i, p in enumerate(parts) if any(r.search(norm(p)) for r in pats)]
    if not hits:
        return text[:EXCERPT_FROM]
    keep = sorted({0} | {j for i in hits for j in range(i - 1, i + 3) if 0 <= j < len(parts)})   # 1 before, 2 after (1.11: ROTATION items lost with 0/+1)
    return "\n".join(parts[j] for j in keep)[:EXCERPT_MAX]


_KEYS_CACHE: dict = {}


def excerpt_keys(league: str, team: str) -> list[str]:
    """Team name words + official roster surnames, from a team code or full name (test set rows use the name)."""
    if (league, team) not in _KEYS_CACHE:
        from core.config import load_settings
        names = team_names(league)
        code = team if team in names else next((c for c, v in names.items()
                                                if norm(v["ime"]) == norm(team) or norm(team) in map(norm, v["zborovi"])), None)
        keys = [team, *team.split()]
        if code:
            roster = league_frames(load_settings(), league)[2]
            keys += [*names[code]["zborovi"], names[code]["ime"], *(surname(n) for n in _full_names(roster, code))]
        _KEYS_CACHE[(league, team)] = keys
    return _KEYS_CACHE[(league, team)]


def relevant(title: str, desc: str, words: list[str], roster_surnames: set) -> bool:
    """Junk filter BEFORE a model call (31.10): the headline/description names the team or one of its players."""
    head = f"{title} {desc}"
    return (any(re.search(rf"\b{re.escape(w)}\b", head, re.I) for w in words)
            or any(len(s) > 3 and re.search(rf"\b{re.escape(s)}\b", norm(head)) for s in roster_surnames))


def _sportando(f: Fetcher, raw: Path, since: str, today: date, full_names: list[str]) -> list[dict]:
    """Sportando's daily EuroLeague injury report: one paragraph per game; kept if it names one of OUR players."""
    body = f.get(SPORTANDO_API, raw / "vesti" / "sportando" / f"injury_{since}_{today}.json",
                 params={"search": "EuroLeague injury report", "per_page": 30, "after": since + "T00:00:00",
                         "_fields": "id,date,link,title,content"})
    out = []
    for p in (json.loads(body) if body else []):
        if "injury report" not in p["title"]["rendered"].lower():
            continue
        for par in re.findall(r"<p[^>]*>(.*?)</p>", p["content"]["rendered"], re.S):
            text = plain(par)
            if _mentions(text, full_names):
                out.append(_src(p["link"], p["date"], html.unescape(p["title"]["rendered"]), text))
    return out


def _rotowire(f: Fetcher, raw: Path, since: str, today: date, full_names: list[str]) -> list[dict]:
    """RotoWire Euro player news (free text, never the paid analysis): the team's items, one source per day."""
    body = f.get(ROTOWIRE_EURO, raw / "vesti" / "rotowire" / f"euro_news_{today}.html")
    if not body:
        return []
    page = body.decode("utf-8", "replace")
    by_day: dict[str, list[str]] = {}
    for block in re.split(r'<div class="news-update(?: [^"_][^"]*)?">', page)[1:]:     # not news-update__* parts
        who = re.search(r'news-update__player-link[^>]*>(.*?)</a>', block)
        when = re.search(r'news-update__timestamp">(.*?)</div>', block)
        news = re.search(r'news-update__news">(.*?)</div>', block, re.S)
        if not (who and when and news):
            continue
        name = html.unescape(who.group(1)).strip()
        if not _mentions(name, full_names):
            continue
        try:
            day = datetime.strptime(when.group(1).strip(), "%B %d, %Y").date().isoformat()
        except ValueError:
            continue
        if day >= since:
            head = re.search(r'news-update__headline">(.*?)</div>', block)
            by_day.setdefault(day, []).append(f"{name}: {plain(head.group(1)) if head else ''}. {plain(news.group(1))}")
    return [_src(f"{ROTOWIRE_EURO}#{day}", day, f"RotoWire Euro news {day}", "\n".join(lines))
            for day, lines in by_day.items()]


def sources(settings: Settings, league: str, code: str, name: str, queries: list[str], words: list[str],
            today: date, days: int = DAYS) -> list[dict]:
    f, raw = Fetcher(settings), settings.paths.raw
    since = (today - timedelta(days=days)).isoformat()
    _, _, roster = league_frames(settings, league)
    full = _full_names(roster, code)
    surnames = {surname(n) for n in full}
    out, seen = [], set()
    for q in queries:
        for p in _eurohoops(f, raw, q, since, pages=2):
            title = html.unescape(p["title"]["rendered"])
            if p["id"] in seen or "/en/" not in p["link"] or "injury report" in title.lower():
                continue
            if not any(re.search(rf"\b{re.escape(w)}\b", title, re.I) for w in words):
                continue
            seen.add(p["id"])
            out.append(_src(p["link"], p["date"], title, plain(p["content"]["rendered"])))
    if league == "euroleague":
        out += _sportando(f, raw, since, today, full) + (_rotowire(f, raw, since, today, full) if USE_ROTOWIRE else [])
    if league != "nba":                                   # the team's section of the official round injury reports
        for q in EL_SEARCHES if league == "euroleague" else []:
            for p in _eurohoops(f, raw, q, since):
                if p["id"] in seen or "/en/euroleague/" not in p["link"] or "injur" not in p["title"]["rendered"].lower():
                    continue
                seen.add(p["id"])
                hdr = ""
                for part in re.split(r"(?=<h[34]>)", p["content"]["rendered"]):
                    if part.startswith("<h3>"):
                        hdr = plain(part.split("</h3>")[0])
                    elif part.startswith("<h4>"):
                        names = re.findall(r"[A-Z][\w'.-]+ [A-Z][\w'.-]+", plain(part))
                        if _roster_team(names, roster) == code:
                            out.append(_src(p["link"], p["date"], html.unescape(p["title"]["rendered"]), f"{hdr}\n{plain(part)}"))
    else:
        espn = (team_names("nba").get(code) or {}).get("espn") or code.lower()
        body = f.get(ESPN_NEWS, raw / "vesti" / "espn" / f"news_{code.lower()}_{today}.json",
                     params={"team": espn, "limit": 50})
        for a in (json.loads(body).get("articles") or []) if body else []:
            if a.get("type") not in ESPN_TYPES or (a.get("published") or "")[:10] < since:
                continue
            if not relevant(a.get("headline") or "", a.get("description") or "", words, surnames):
                continue                                 # general league story (Cup format, how to watch …)
            sid = (a.get("links", {}).get("api", {}).get("self", {}) or {}).get("href", "").rstrip("/").split("/")[-1]
            if not sid.isdigit():
                continue
            st = f.get(ESPN_STORY.format(id=sid), raw / "vesti" / "espn" / f"story_{sid}.json")
            h = (json.loads(st).get("headlines") or [{}])[0] if st else {}
            text = plain(h.get("story") or "")
            if len(text) > 300:
                web = (a.get("links", {}).get("web", {}) or {}).get("href") or ESPN_STORY.format(id=sid)
                out.append(_src(web, a["published"], a.get("headline", ""), text))
        inj = f.get(ESPN_INJURIES, raw / "vesti" / "espn" / f"injuries_{today}.json")
        for t in (json.loads(inj).get("injuries") or []) if inj else []:
            if surname(t.get("displayName", "")) != surname(name):
                continue
            for i in t["injuries"]:
                if i.get("shortComment") and i["date"][:10] >= since:
                    out.append(_src(f"{ESPN_INJURIES}#{i['athlete']['displayName'].replace(' ', '_')}", i["date"],
                                    f"ESPN injury note: {i['athlete']['displayName']}",
                                    f"{t['displayName']} — injuries\n{i['shortComment']}\n{i.get('longComment', '')}"))
    return sorted(out, key=lambda s: s["objaveno"], reverse=True)


def _cache_source(folder: Path, s: dict) -> None:
    d = folder / "izvori"
    d.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(s["izvor_url"].encode()).hexdigest()[:12]
    (d / f"{s['objaveno'][:10]}_{key}.json").write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")


def _items(ans, src: dict, roster: set, labels: dict, model: str) -> tuple[list[dict], list[str]]:
    if not isinstance(ans, dict) or has_forbidden(ans):
        return [], [f"{model}: забрането поле / не е објект — одговорот отфрлен"]
    kept, bad = [], []
    for it in ans.get("items") or []:
        if not isinstance(it, dict):
            continue
        tip = str(it.get("tip", "")).upper()
        if tip in getattr(t10_vesti, "REJECTED_TYPES", {}):
            bad.append(f"{model}: тип „{tip}“ = анализа/мислење — LLM само extract (отфрлено)")
            continue
        if tip not in t10_vesti.TYPES:
            bad.append(f"{model}: непознат тип „{tip}“")
            continue
        if not it.get("citat") or not quote_in_text(it["citat"], src["tekst"]):
            bad.append(f"{model}: „{(it.get('sto') or '')[:60]}“ — цитатот го нема во текстот")
            continue
        sto = it.get("sto") or ""
        if (t10_vesti.TYPES[tip] in ("ФОРМА", "ДРУГО") and re.search(r"\d", sto)) or STAT_RE.search(sto):
            bad.append(f"{model}: „{(it.get('sto') or '')[:60]}“ — статистика/резултат: бројките се од нашиот box score, не од AI")
            continue
        who = it.get("igrac")
        if who and surname(who) not in roster:
            bad.append(f"{model}: играч „{who}“ не е во официјалниот состав (друг тим / поранешен)")
            continue
        vl = it.get("vlijanie") or "not stated"
        if t10_vesti.TYPES[tip] == "РАСПОРЕД":            # a schedule fact is never a strength / effect (31.10)
            vl = "not stated"
        kept.append({"tip": t10_vesti.TYPES[tip], "igrac": labels.get(surname(who), who) if who else None,
                     "sto": (it.get("sto") or "").strip(), "vlijanie": vl,
                     "citat": it["citat"].strip(), "model": model, **{k: src[k] for k in ("izvor_url", "objaveno")}})
    return kept, bad


def _same(a: dict, b: dict) -> bool:
    if a["tip"] != b["tip"]:
        return False
    if a["igrac"] or b["igrac"]:
        return bool(a["igrac"] and b["igrac"]) and surname(a["igrac"]) == surname(b["igrac"])
    qa, qb = set(norm(a["citat"]).split()), set(norm(b["citat"]).split())
    return bool(qa and qb) and len(qa & qb) / min(len(qa), len(qb)) >= 0.6


def run_news(settings: Settings, league: str, code: str, name: str, queries: list[str], words: list[str],
             models: list[str], today: date, roster: set, labels: dict, days: int = DAYS) -> dict:
    folder = TEAMS_DIR / league / code
    srcs = sources(settings, league, code, name, queries, words, today, days)
    clients = [LLMClient(m, timeout_s=90, retries=5) for m in models[:2]]
    stats = {"izvori": len(srcs), "povici": 0, "od_kesh": 0, "tokeni": 0, "sekundi": 0.0, "bez_odgovor": []}
    rejected, single, agreed = [], [], []
    agree = {m: {"odgovori": 0, "stavki": 0, "potvrdeni": 0} for m in models[:2]}
    keys = [*words, name, *roster]
    for s in srcs:
        _cache_source(folder, s)
        got = {}
        for c in clients:
            try:
                text = excerpt(s["tekst"], keys) if USE_EXCERPT else s["tekst"]
                ans, meta = c.ask_json_meta(t10_vesti.SYSTEM_PROMPT,
                                            t10_vesti.user_text(name, league, s["izvor_url"], s["objaveno"], text),
                                            t10_vesti.EXCERPT_VERSION if text != s["tekst"] else t10_vesti.PROMPT_VERSION)
            except Exception as exc:                  # limit / provider down → recorded, never filled in
                stats["bez_odgovor"].append(f"{c.model} · {s['izvor_url']}: {str(exc)[:100]}")
                continue
            stats["povici"] += 1
            stats["od_kesh"] += int(bool(meta.get("od_kesh")))
            stats["tokeni"] += meta.get("tokeni") or 0
            stats["sekundi"] += meta.get("sekundi") or 0
            got[c.model], bad = _items(ans, s, roster, labels, c.model)
            rejected += [f"{s['objaveno'][:10]} {b} ({s['izvor_url']})" for b in bad]
        a, b = (got.get(m, []) for m in models[:2])
        if len(got) == 2:                         # agreement is only measurable when both answered
            for m, mine, other in ((models[0], a, b), (models[1], b, a)):
                agree[m]["odgovori"] += 1
                agree[m]["stavki"] += len(mine)
                agree[m]["potvrdeni"] += sum(any(_same(x, y) for y in other) for x in mine)
        for x in a:
            twin = next((y for y in b if _same(x, y)), None)
            (agreed if twin else single).append({**x, "modeli": f"{models[0]}+{models[1]}" if twin else x["model"]})
        single += [{**y, "modeli": y["model"]} for y in b if not any(_same(x, y) for x in a)]
    # one entry per event: same player + type within 3 days → one record, all sources
    merged: list[dict] = []
    for x in sorted(agreed, key=lambda r: r["objaveno"], reverse=True):
        twin = next((m for m in merged if m["tip"] == x["tip"] and x["igrac"] and m["igrac"] == x["igrac"]
                     and abs((datetime.fromisoformat(m["objaveno"][:10]) - datetime.fromisoformat(x["objaveno"][:10])).days) <= 3),
                    None)
        if twin:
            if x["izvor_url"] not in twin["izvori"]:
                twin["izvori"].append(x["izvor_url"])
        else:
            merged.append({**x, "izvori": [x["izvor_url"]]})
    write_vesti(folder, code, merged)
    stats["soglasuvanje"] = agree
    log_agreement("T10", league, code, today, agree)
    return {"zapisi": merged, "eden_model": single, "otfrleno": rejected, "stats": stats, "izvori": srcs}


def log_agreement(task: str, league: str, code: str, today: date, agree: dict) -> None:
    """Tome 31.10: the checker model is below the threshold (status СЛЕДИ) → its agreement rate is logged per run."""
    import sqlite3
    from research.llm_eval import LOG_DB
    LOG_DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(LOG_DB)
    con.execute("CREATE TABLE IF NOT EXISTS llm_soglasuvanje (datum TEXT, zadaca TEXT, liga TEXT, tim TEXT, model TEXT, "
                "odgovori INTEGER, stavki INTEGER, potvrdeni INTEGER, zapishano TEXT)")
    for m, v in agree.items():
        con.execute("INSERT INTO llm_soglasuvanje VALUES (?,?,?,?,?,?,?,?,?)",
                    (today.isoformat(), task, league, code, m, v.get("odgovori", 0), v.get("stavki", v.get("fakti", 0)),
                     v["potvrdeni"], datetime.now().isoformat(timespec="seconds")))
    con.commit()
    con.close()


def agreement_summary() -> str:
    import sqlite3
    from research.llm_eval import LOG_DB
    if not LOG_DB.exists():
        return "Нема записи за согласување."
    con = sqlite3.connect(LOG_DB)
    try:
        rows = con.execute("SELECT zadaca, model, SUM(odgovori), SUM(stavki), SUM(potvrdeni) FROM llm_soglasuvanje "
                           "GROUP BY zadaca, model ORDER BY zadaca, model").fetchall()
    except sqlite3.OperationalError:
        rows = []
    con.close()
    out = ["| Задача | Модел | извори (двата одговориле) | ставки | потврдени од другиот | % |", "|---|---|---|---|---|---|"]
    out += [f"| {z} | {m} | {o} | {s} | {p} | {100 * p / s:.0f}% |" if s else f"| {z} | {m} | {o} | 0 | 0 | — |"
            for z, m, o, s, p in rows]
    return "\n".join(out)


def _cell(t: str) -> str:
    return re.sub(r"\s+", " ", str(t or "")).replace("|", "/").strip()


def write_vesti(folder: Path, team: str, rows: list[dict]) -> None:
    path = folder / "vesti.md"
    old = read_news(folder)
    keys = {(r["datum"], r["tip"], r["igrac"], r["citat"][:40]) for r in old}
    lines = [f"| {r['datum']} | {r['tip']} | {r['igrac']} | {r['sto']} | {r['vlijanie']} | {r['izvor']} | {r['citat']} | {r['modeli']} |"
             for r in old]
    for r in rows:
        k = (r["objaveno"][:10], r["tip"], r["igrac"] or "—", _cell(r["citat"])[:40])
        if k in keys:
            continue
        vl = "нема (не е наведено)" if r["vlijanie"].lower().startswith("not stated") else _cell(r["vlijanie"])
        src = " ; ".join(f"[{i + 1}]({u})" for i, u in enumerate(r["izvori"]))
        lines.append(f"| {r['objaveno'][:10]} | {r['tip']} | {r['igrac'] or '—'} | {_cell(r['sto'])} | {vl} | {src} | "
                     f"„{_cell(r['citat'])}“ | {r['modeli']} |")
    lines.sort(key=lambda l: l[2:12], reverse=True)
    path.write_text(VESTI_HEADER.format(team=team) + "\n".join(lines) + "\n", encoding="utf-8")


def injuries_to_vesti(settings: Settings, league: str, name: str, result: dict, t1_models: list[str], now: datetime) -> int:
    """Injury / lineup news from the last 24 h → T1 (status, two models) → data/vesti/<date>.yaml (for `picks`)."""
    from research.tasks.t1_povredi import run_t1, to_vesti
    fresh = {r["izvor_url"] for r in result["zapisi"] if r["tip"] in ("ПОВРЕДА", "СОСТАВ")
             and datetime.fromisoformat(r["objaveno"][:16]).replace(tzinfo=timezone.utc) >= now - timedelta(hours=24)}
    if not fresh:
        return 0
    clients = [LLMClient(m, timeout_s=90, retries=3) for m in t1_models[:2]]
    items = []
    for s in result["izvori"]:
        if s["izvor_url"] in fresh:
            pub = datetime.fromisoformat(s["objaveno"][:16]).replace(tzinfo=timezone.utc)
            items += to_vesti(run_t1(s["tekst"], s["izvor_url"], pub, clients, now, league))
    if not items:
        return 0
    path = VESTI_DIR / f"{now.date().isoformat()}.yaml"
    old = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else []
    have = {(o.get("liga"), surname(o.get("igrac", ""))) for o in old or []}
    new = [{**i, "vnesol": "T1 агенти (2 модела, PROVERENO)"} for i in items if (i["liga"], surname(i["igrac"])) not in have]
    if new:
        VESTI_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump((old or []) + new, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return len(new)
