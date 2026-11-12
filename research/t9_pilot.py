"""T9 for one team: season 2026-27 articles that mention the coach / GM → two models → Python checks →
PROPOSAL in data/timovi/{league}/{team}/predlozi/<date>_T9.yaml (+ .md to read). Nothing goes into profil.yaml
until Tome runs `python main.py profil odobri --prifati …`.

Tome 31.10: only the head coach (name, since when, short philosophy) and the GM — the style is numbers (team_dossier).
Python checks every fact: the quote is in the article; the head coach is the official one; the philosophy has no
numbers and is checked against our ranks (numbers win). A fact both models report from the same article is
PROVERENO, otherwise NEPROVERENO (kept, marked).
"""
from __future__ import annotations

import html
import re
from datetime import date
from pathlib import Path

import yaml

from core.config import Settings
from research.llm_client import LLMClient
from research.t10_news import STAT_RE, league_frames, log_agreement
from research.tasks import t9_profil
from research.testset_build import Fetcher, _eurohoops, _nba_coaches, norm, plain, surname
from research.validate import has_forbidden, quote_in_text

TEAMS_DIR = Path(__file__).resolve().parents[1] / "data" / "timovi"
MAX_ARTICLES = 6                        # coach/GM only (31.10) — the quota goes to T10 news first
TEXT_CHARS = 6000


def _official(settings: Settings, f: Fetcher, league: str, code: str) -> tuple[str | None, set[str], dict]:
    team_logs, players, roster = league_frames(settings, league)
    if league != "nba":                                  # EuroLeague / EuroCup: the coach is on the box score
        c = team_logs[team_logs["team"] == code]["coach"].dropna() if "coach" in team_logs else []
        coach = c.iloc[-1] if len(c) else None
        coach = f"{coach.split(',')[1].strip().title()} {coach.split(',')[0].strip().title()}" if coach and "," in coach else coach
    else:
        coach = _nba_coaches(settings, f, code)
    names = set(roster[roster["team"] == code]["player"]) | set(players[players["team"] == code]["player"])
    labels = {surname(p): lbl for p, lbl in zip(players["player"], players["label"]) if p in names}
    return coach, {surname(p) for p in names}, labels


def articles(settings: Settings, f: Fetcher, queries: list[str], title_words: list[str], since: date) -> list[dict]:
    seen, out = set(), []
    for q in queries:
        for p in _eurohoops(f, settings.paths.raw, q, since.isoformat(), pages=8):
            title = html.unescape(p["title"]["rendered"])
            if p["id"] in seen or "/en/" not in p["link"] or "injury report" in title.lower():
                continue
            if not any(re.search(rf"\b{re.escape(w)}\b", title, re.I) for w in title_words):
                continue
            seen.add(p["id"])
            out.append({"naslov": title, "izvor_url": p["link"], "objaveno": p["date"][:10],
                        "tekst": plain(p["content"]["rendered"])[:TEXT_CHARS]})
    return out


def pick_articles(arts: list[dict]) -> list[dict]:
    """Season 2026-27 only (Tome 30.10); only texts that mention the coach / GM role (31.10); the longest first."""
    seen, out = set(), []
    for a in sorted(arts, key=lambda a: (-len(a["tekst"]), a["objaveno"])):
        if not re.search(t9_profil.ROLE_WORDS, a["tekst"], re.I):
            continue
        if a["izvor_url"] not in seen and not a["izvor_url"].startswith("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"):
            seen.add(a["izvor_url"])
            out.append(a)
    return sorted(out[:MAX_ARTICLES], key=lambda a: a["objaveno"], reverse=True)


# claim words → the number that confirms or contradicts it (rank 1 = most / best)
CLAIMS = [
    (r"\b(fast|up-tempo|uptempo|run and gun|transition)\b", "tempo", "high", "темпо"),
    (r"\b(slow|grind|half-court|methodical|deliberate)\b", "tempo", "low", "темпо"),
    (r"\b(three-point|3-point|from deep|beyond the arc|perimeter shooting|shooters?)\b", "3par", "high", "3PAr"),
    (r"\b(elite defen\w*|best defen\w*|stingy|lockdown|top defen\w*|strong defen\w*)\b", "drtg", "high", "DRtg"),
    (r"\b(poor defen\w*|bad defen\w*|struggl\w* (on|at) defen\w*|leaky)\b", "drtg", "low", "DRtg"),
    (r"\b(rebound\w*|glass|boards|offensive rebounding)\b", "oreb_strategija", "high", "OREB%"),
    (r"\b(elite offen\w*|best offen\w*|explosive offen\w*|potent offen\w*)\b", "ortg", "high", "ORtg"),
]


def check_numbers(text: str, par: dict, n_teams: int) -> tuple[str, str]:
    """A description vs OUR numbers (numbers win; the difference is written down)."""
    notes, status = [], "НЕУТРАЛНО"
    for rx, key, want, mk in CLAIMS:
        if not re.search(rx, text or "", re.I):
            continue
        v = (par.get(key) or {}).get("vrednost") or {}
        rank = v.get("rang") if key != "oreb_strategija" else v.get("rang_oreb")
        if not rank:
            continue
        top, bottom = rank <= n_teams / 3, rank > 2 * n_teams / 3
        ok = top if want == "high" else bottom
        bad = bottom if want == "high" else top
        notes.append(f"{mk} ранг {rank}/{n_teams}")
        if bad:
            status = "СЕ КОСИ"
        elif ok and status != "СЕ КОСИ":
            status = "ПОТВРДЕНО"
    return status, "; ".join(notes)


def _facts(ans: dict, art: dict, coach: str | None, roster: set[str], model: str) -> tuple[list[dict], list[str]]:
    """Validated facts of one answer: (kept, rejected reasons)."""
    if not isinstance(ans, dict) or has_forbidden(ans):
        return [], [f"{model}: забрането поле / не е објект — одговорот отфрлен"]
    text, kept, bad = art["tekst"], [], []
    base = {"izvor_url": art["izvor_url"], "objaveno": art["objaveno"], "model": model}

    def add(pole, vrednost, citat, **extra):
        if not citat or not quote_in_text(citat, text):
            bad.append(f"{model}: {pole} „{vrednost}“ — цитатот го нема во текстот")
            return
        kept.append({"pole": pole, "vrednost": vrednost, "citat": citat, **extra, **base})

    tr = ans.get("trener") if isinstance(ans.get("trener"), dict) else None
    if tr and tr.get("ime"):
        if coach and surname(tr["ime"]) == surname(coach):
            add("trener", tr["ime"], tr.get("citat"), od_koga=tr.get("od_koga"))
        else:
            bad.append(f"{model}: тренер „{tr['ime']}“ ≠ официјален „{coach}“ (стар тренер / друг тим?)")
    for it in ans.get("filozofija") or []:
        if not isinstance(it, dict) or not it.get("opis"):
            continue
        opis = str(it["opis"])
        if re.search(r"\d", opis) or STAT_RE.search(str(it.get("citat") or "")):   # a result / stat line, not a philosophy
            bad.append(f"{model}: филозофија „{opis[:60]}“ — резултат/статистика, не опис (бројките = box score)")
        elif len(opis.split()) > 25:
            bad.append(f"{model}: филозофија има {len(opis.split())} зборови (> 25)")
        else:
            add("trener_filozofija", opis, it.get("citat"))
    gm = ans.get("gm") if isinstance(ans.get("gm"), dict) else None
    if gm and gm.get("ime"):
        if surname(gm["ime"]) in roster:
            bad.append(f"{model}: ГМ „{gm['ime']}“ е играч од составот — не е ГМ")
        else:
            add("gm", gm["ime"], gm.get("citat"), uloga=gm.get("uloga"))
    return kept, bad


def _same(a: dict, b: dict) -> bool:
    if a["pole"] != b["pole"] or a["izvor_url"] != b["izvor_url"]:
        return False
    if a["pole"] in ("trener", "gm"):
        return surname(a["vrednost"]) == surname(b["vrednost"])
    qa, qb = set(norm(a["citat"]).split()), set(norm(b["citat"]).split())
    return bool(qa and qb) and len(qa & qb) / min(len(qa), len(qb)) >= 0.6      # same passage quoted


def run_team(settings: Settings, league: str, code: str, name: str, queries: list[str], title_words: list[str],
             models: list[str], since: date, today: date, extra: list[dict] | None = None) -> tuple[Path, dict]:
    """→ (proposal path, stats {napisi, povici, od_kesh, tokeni, sekundi, kosenje})."""
    if len(models) < 2:
        raise ValueError("T9 бара два различни модели (LLM_ISTRAZUVACI §1.5)")
    f = Fetcher(settings)
    coach, roster, labels = _official(settings, f, league, code)
    arts = pick_articles(articles(settings, f, queries, title_words, since)
                         + [a for a in (extra or []) if a["objaveno"][:10] >= since.isoformat()])
    clients = [LLMClient(m, timeout_s=90, retries=5) for m in models[:2]]
    facts, rejected, failed = {m: [] for m in models[:2]}, [], []
    stats = {"napisi": len(arts), "povici": 0, "od_kesh": 0, "tokeni": 0, "sekundi": 0.0}
    for art in arts:
        user = t9_profil.user_text(name, league, art["izvor_url"], art["objaveno"], art["tekst"])
        for c in clients:
            try:
                ans, meta = c.ask_json_meta(t9_profil.SYSTEM_PROMPT, user, t9_profil.PROMPT_VERSION)
            except Exception as exc:                  # provider down / limit → recorded, never filled in
                failed.append(f"{c.model} · {art['izvor_url']}: {str(exc)[:120]}")
                continue
            stats["povici"] += 1
            stats["od_kesh"] += int(bool(meta.get("od_kesh")))
            stats["tokeni"] += meta.get("tokeni") or 0
            stats["sekundi"] += meta.get("sekundi") or 0
            kept, bad = _facts(ans, art, coach, roster, c.model)
            facts[c.model] += kept
            rejected += [f"{art['objaveno'][:10]} {r}" for r in bad]
    a, b = facts[models[0]], facts[models[1]]
    merged = []
    for x in a:
        twin = next((y for y in b if _same(x, y)), None)
        merged.append({**x, "model": f"{models[0]}+{models[1]}" if twin else x["model"],
                       "proverka": "PROVERENO" if twin else "NEPROVERENO"})
    merged += [{**y, "proverka": "NEPROVERENO"} for y in b if not any(_same(x, y) for x in a)]
    merged.sort(key=lambda r: (r["pole"], r["objaveno"]), reverse=False)

    folder = TEAMS_DIR / league / code
    prof = yaml.safe_load((folder / "profil.yaml").read_text(encoding="utf-8")) if (folder / "profil.yaml").exists() else {}
    n_teams = len([p for p in (TEAMS_DIR / league).glob("*") if p.is_dir()]) or 20
    kos = []
    for r in merged:                                  # descriptions vs OUR numbers — numbers win, the clash is written down
        if r["pole"] == "trener_filozofija":
            st, note = check_numbers(f"{r['vrednost']} {r['citat']}", prof.get("parametri") or {}, n_teams)
            if note:
                r["proverka_brojki"] = f"{st}: {note}"
                if st == "СЕ КОСИ":
                    kos.append(f"„{r['citat'][:90]}“ → {note}")
    stats["kosenje"] = kos
    stats["soglasuvanje"] = agreement(facts, models[:2])
    log_agreement("T9", league, code, today, stats["soglasuvanje"])
    proposal = {
        "tip": "ОПИС", "tim": code, "liga": league, "predlog_datum": today.isoformat(),
        "izvor_url": "; ".join(a["izvor_url"] for a in arts[:3]) + (" …" if len(arts) > 3 else ""),
        "citat": next((r["citat"] for r in merged if r["pole"] == "trener"), merged[0]["citat"] if merged else ""),
        "prompt_verzija": t9_profil.PROMPT_VERSION, "modeli": models[:2],
        "oficijalno": {"trener": coach, "izvor": "box score Евролига" if league == "euroleague" else "nba_api CommonTeamRoster"},
        "napisi": [{k: a[k] for k in ("objaveno", "naslov", "izvor_url")} for a in arts],
        "fakti": [{k: v for k, v in r.items() if v is not None} for r in merged],
        "otfrleno_od_python": rejected, "bez_odgovor": failed, "se_kosi_so_brojki": kos,
    }
    out = folder / "predlozi" / f"{today.isoformat()}_T9.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(proposal, allow_unicode=True, sort_keys=False, width=140), encoding="utf-8")
    out.with_suffix(".md").write_text(render(proposal), encoding="utf-8")
    return out, stats


FIELD_MK = {"trener": "Тренер", "trener_filozofija": "Филозофија на тренерот (≤ 25 зборови)", "gm": "ГМ / спортски директор"}


def agreement(facts: dict[str, list[dict]], models: list[str]) -> dict:
    """How often each model's validated facts are confirmed by the other one (Tome 31.10: logged per model)."""
    a, b = (facts.get(m) or [] for m in models)
    conf_a = sum(any(_same(x, y) for y in b) for x in a)
    conf_b = sum(any(_same(y, x) for x in a) for y in b)
    return {models[0]: {"fakti": len(a), "potvrdeni": conf_a}, models[1]: {"fakti": len(b), "potvrdeni": conf_b}}


def render(p: dict) -> str:
    out = [f"# ПРЕДЛОГ T9 — {p['tim']} ({p['liga']}) · {p['predlog_datum']}", "",
           f"*Само предлог од агентите ({' + '.join(p['modeli'])}, {p['prompt_verzija']}). Python ги провери цитатите, "
           f"тренерот и составот. Ништо не е во profil.yaml пред `profil odobri`.* ✅ = двата модели, ⚪ = еден модел.", "",
           f"**Официјален тренер:** {p['oficijalno']['trener']} ({p['oficijalno']['izvor']}) · написи: {len(p['napisi'])}", ""]
    for pole, mk in FIELD_MK.items():
        rows = [r for r in p["fakti"] if r["pole"] == pole]
        if not rows:
            continue
        out += [f"## {mk}", ""]
        for r in rows:
            mark = "✅" if r["proverka"] == "PROVERENO" else "⚪"
            nb = f" · бројки: {r['proverka_brojki']}" if r.get("proverka_brojki") else ""
            out.append(f"- {mark} **{r['vrednost']}** — „{r['citat']}“ ([{r['objaveno'][:10]}]({r['izvor_url']})){nb}")
        out.append("")
    out += ["## Написи", ""] + [f"- {a['objaveno']} [{a['naslov']}]({a['izvor_url']})" for a in p["napisi"]]
    if p.get("se_kosi_so_brojki"):
        out += ["", "## Се коси со нашите бројки (важат бројките)", ""] + [f"- {r}" for r in p["se_kosi_so_brojki"]]
    if p["otfrleno_od_python"]:
        out += ["", f"## Отфрлено од Python ({len(p['otfrleno_od_python'])})", ""] + [f"- {r}" for r in p["otfrleno_od_python"]]
    if p["bez_odgovor"]:
        out += ["", f"## Без одговор ({len(p['bez_odgovor'])})", ""] + [f"- {r}" for r in p["bez_odgovor"]]
    return "\n".join(out) + "\n"
