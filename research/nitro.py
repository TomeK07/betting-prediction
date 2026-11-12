"""НИТРО (Tome 1.11): Claude searches the web itself (WebSearch/WebFetch, no FreeLLMAPI) for the teams of the named
games and writes the facts it found to an input file; THIS module does everything else, so the T1/T10 rules hold:

  input   data/raw/vesti/nitro/<date>/<label>_vlez.json
          {"liga": "nba", "natprevari": ["LAL-TOR"], "fakti": [{"tim", "tip", "igrac", "sto", "status"?, "prichina"?,
           "izvor_url", "citat", "objaveno"}]}
  1. Python downloads every source ITSELF (Claude's WebFetch text is a summary, not the page) → data/raw/vesti/nitro/
  2. checks: no forbidden field (probability/pick/…); known type; the quote literally in the downloaded text; a named
     player on the team's OFFICIAL roster; a date; no statistics in `sto`. Without a source → not written.
  3. vesti.md of the team with modeli = IZVOR=CLAUDE_NITRO; statuses (OUT/Q/…) of the last 48 h → data/vesti/<date>.yaml
  4. povredi → picks for the league; short summary for Tome (OUT/Q per team, who gets the minutes, picks changes).
No number in the summary comes from Claude: statuses are quoted, minutes/picks are read from the Python reports.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import yaml

from core.config import Settings
from research.t10_news import STAT_RE, TEAMS_DIR, VESTI_DIR, team_names, write_vesti
from research.tasks import t10_vesti
from research.testset_build import Fetcher, plain, surname
from research.validate import T1_STATUSES, has_forbidden, quote_in_text

IZVOR = "IZVOR=CLAUDE_NITRO"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "vesti" / "nitro"
STATUS_HOURS = 48                      # Tome 1.11: the last 48 h
MAX_SEARCHES = 15                      # Tome's Claude quota per НИТРО (Claude counts, Python records it)
TIPOVI = set(t10_vesti.TYPES) | set(t10_vesti.TYPES.values())


def _text(raw: bytes) -> str:
    t = raw.decode("utf-8", errors="replace")
    t = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", t)
    return plain(t)


def download(f: Fetcher, url: str) -> tuple[str | None, Path]:
    key = hashlib.sha1(url.encode()).hexdigest()[:12]
    path = RAW / "izvori" / f"{key}.html"
    try:
        raw = f.get(url, path)
    except Exception:
        return None, path
    if raw is None:
        return None, path
    txt = path.with_suffix(".txt")
    text = _text(raw)
    txt.write_text(f"{url}\n\n{text}", encoding="utf-8")
    return text, txt


def _when(s: str | None) -> datetime | None:
    try:
        t = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def check(fact: dict, text: str | None, roster: set, now: datetime) -> list[str]:
    bad = []
    if has_forbidden(fact):
        bad.append("забрането поле (веројатност/избор/…)")
    if not fact.get("izvor_url"):
        bad.append("нема извор")
    elif text is None:
        bad.append("изворот не можеше да се симне → цитатот не може да се провери")
    elif not fact.get("citat") or not quote_in_text(fact["citat"], text):
        bad.append("цитатот го нема во симнатиот текст")
    if str(fact.get("tip", "")).upper() not in TIPOVI and fact.get("tip") not in TIPOVI:
        bad.append(f"непознат тип „{fact.get('tip')}“")
    t = _when(fact.get("objaveno"))
    if t is None:
        bad.append("нема датум на објава")
    elif t > now + timedelta(hours=1):
        bad.append("датумот е во иднина")
    if fact.get("igrac") and surname(fact["igrac"]) not in roster:
        bad.append(f"„{fact['igrac']}“ не е во официјалниот состав")
    if STAT_RE.search(fact.get("sto") or ""):
        bad.append("статистика во „што“ — бројките се од нашиот box score")
    st = fact.get("status")
    if st and str(st).upper() not in T1_STATUSES:
        bad.append(f"непознат статус „{st}“")
    return bad


def _tip(t: str) -> str:
    return t10_vesti.TYPES.get(str(t).upper(), t)


def to_vesti_yaml(day: date, items: list[dict]) -> int:
    """Statuses → data/vesti/<day>.yaml (read by `povredi`). A newer NITRO status replaces an older one of the player."""
    path = VESTI_DIR / f"{day.isoformat()}.yaml"
    old = (yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else None) or []
    n = 0
    for it in items:
        k = (it["liga"], surname(it["igrac"]))
        same = [o for o in old if (o.get("liga"), surname(o.get("igrac", ""))) == k]
        if any(str(o.get("objaveno", "")) >= it["objaveno"] for o in same):
            continue                               # the archive already has this or a newer status
        old = [o for o in old if o not in same] + [it]
        n += 1
    if n:
        VESTI_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(old, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return n


def _section(text: str | None, home: str, away: str) -> str:
    if not text:
        return ""
    m = re.search(rf"^## {home} – {away} .*?(?=^## |\Z)", text, re.M | re.S)
    return m.group(0) if m else ""


def _rows(section: str) -> set[str]:
    return {" | ".join(c.strip() for c in l.split("|")[1:3]) for l in section.splitlines()
            if l.startswith("| ") and not l.startswith("| Избор") and not l.startswith("| Играч")}


def summary(povredi: str, before: str, after: str, home: str, away: str) -> list[str]:
    sec = _section(povredi, home, away)
    out = [f"**{home}–{away}**"]
    for team in (home, away):
        part = re.search(rf"\*\*{team}:\*\*(.*?)(?=\*\*[A-Z]{{3}}:\*\*|\Z)", sec, re.S)
        part = part.group(1) if part else ""
        st = [re.match(r"\| (.+?) \((?:[A-Z]{3})\) \| (\S+)", l) for l in part.splitlines()]
        st = [f"{m.group(1)} {m.group(2)[0] if m.group(2) != 'OUT' else 'OUT'}" for m in st
              if m and m.group(2) in ("OUT", "QUESTIONABLE", "DOUBTFUL") and "СТАРО" not in m.string]
        gain = [l.split("|")[1].strip() + " " + l.split("|")[3].strip() for l in part.splitlines()
                if re.match(r"\| .+ \| [\d.]+ \| \+", l)][:3]
        out.append(f"- {team}: OUT/Q (ново): {', '.join(st) or 'нема'} · минути: {', '.join(gain) or 'нема ефект'}")
    b, a = _rows(_section(before, home, away)), _rows(_section(after, home, away))
    if before is None:
        out.append(f"- picks: пред НИТРО немаше picks за денес · сега {len(a)} кандидати (сите на хартија)")
        return out
    out.append(f"- picks: +{len(a - b)} / −{len(b - a)}" + (f" · нови: {'; '.join(sorted(a - b)[:3])}" if a - b else "")
               + (f" · отпаднати: {'; '.join(sorted(b - a)[:3])}" if b - a else ""))
    return out


def cmd_nitro(settings: Settings, vlez: str, today: date, picks: bool = True) -> int:
    from pipeline.injuries_cmd import cmd_povredi
    from research.t9_pilot import _official
    import contextlib
    import io

    path = Path(vlez)
    data = json.loads(path.read_text(encoding="utf-8"))
    lg, now = data["liga"], datetime.now(timezone.utc)
    f = Fetcher(settings)
    names = team_names(lg)
    rosters: dict[str, tuple[set, dict]] = {}
    ok_by_team: dict[str, list[dict]] = {}
    statuses, rejected, log = [], [], []
    for fact in data.get("fakti", []):
        code = fact.get("tim")
        if code not in rosters:
            _, roster, labels = _official(settings, f, lg, code)
            rosters[code] = (roster, labels)
        roster, labels = rosters[code]
        text, saved = download(f, fact.get("izvor_url") or "") if fact.get("izvor_url") else (None, None)
        bad = check(fact, text, roster, now)
        log.append({**fact, "zacuvano": str(saved) if saved else None, "ok": not bad, "pricina": "; ".join(bad)})
        if bad:
            rejected.append(f"{code} · {fact.get('igrac') or '—'} · {fact.get('sto', '')[:60]} → {'; '.join(bad)}")
            continue
        who = fact.get("igrac")
        lbl = labels.get(surname(who), who) if who else None
        ok_by_team.setdefault(code, []).append({
            "tip": _tip(fact["tip"]), "igrac": lbl, "sto": fact.get("sto", ""), "vlijanie": "not stated",
            "citat": fact["citat"], "objaveno": _when(fact["objaveno"]).isoformat(), "izvori": [fact["izvor_url"]],
            "modeli": IZVOR})
        t = _when(fact["objaveno"])
        if fact.get("status") and who and now - t <= timedelta(hours=STATUS_HOURS):
            statuses.append({"liga": lg, "tim": names.get(code, {}).get("ime", code), "igrac": who,
                             "status": str(fact["status"]).upper(), "prichina": fact.get("prichina", ""),
                             "izvor_url": fact["izvor_url"], "citat": fact["citat"], "objaveno": t.isoformat(),
                             "model": "claude-nitro", "prompt_verzija": "nitro-v1", "vnesol": IZVOR})
    for code, rows in ok_by_team.items():
        write_vesti(TEAMS_DIR / lg / code, code, rows)
    n_status = to_vesti_yaml(today, statuses)
    path.with_name(path.stem.replace("_vlez", "") + "_proverka.json").write_text(
        json.dumps({"vreme": now.isoformat(timespec="seconds"), "prebaruvanja": data.get("prebaruvanja"),
                    "fakti": log}, ensure_ascii=False, indent=1), encoding="utf-8")

    rep = settings.paths.reports
    pick_file = rep / f"picks_{today}_{lg}.md"
    before = pick_file.read_text(encoding="utf-8") if pick_file.exists() else None
    with contextlib.redirect_stdout(io.StringIO()):
        cmd_povredi(settings, lg, today)
        if picks:
            from pipeline.picks_cmd import cmd_picks
            if not (settings.paths.raw / "mozzart" / today.isoformat()).exists():   # picks needs today's lines
                from main import cmd_mozzart
                cmd_mozzart(settings)
            cmd_picks(settings, lg, today)
    pov = rep / f"povredi_{today}_{lg}.md"
    povredi = pov.read_text(encoding="utf-8") if pov.exists() else None
    after = pick_file.read_text(encoding="utf-8") if pick_file.exists() else None

    n_ok = sum(len(r) for r in ok_by_team.values())
    lines = [f"НИТРО {lg} · {today} · пребарувања {data.get('prebaruvanja', '?')}/{MAX_SEARCHES} · "
             f"факти: прифатени {n_ok}, одбиени {len(rejected)} · → data/vesti {n_status}"]
    for g in data.get("natprevari", []):
        home, away = g.split("-")
        lines += summary(povredi, before, after, home, away)
    lines += [f"- одбиено: {r}" for r in rejected[:4]]
    out = rep / "nitro" / f"{today}_{path.stem.replace('_vlez', '')}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# " + "\n".join(lines) + "\n\n## Сите одбиени\n" + "\n".join(f"- {r}" for r in rejected) + "\n",
                   encoding="utf-8")
    print("\n".join(lines[:15]))
    return 0
