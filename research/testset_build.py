"""Build the LLM test sets (LLM_ISTRAZUVACI §6): tests/llm/T1.jsonl and tests/llm/T9.jsonl.

The correct answer ALWAYS comes from official data, never from an AI (Tome 30.10):
- T1 NBA: ESPN injury note (the text) ↔ the official NBA injury report PDF published right after the note
  (status for that player; team submitted but player not listed → AVAILABLE).
- T1 EL: one team section of a Eurohoops round injury article (the text) ↔ the official EuroLeague statement
  in it + the box score of that game (an official OUT must be missing from the 12-man game sheet).
- T9: Eurohoops team articles (the text) ↔ official head coach (EL box score / NBA CommonTeamRoster)
  and the official roster of the season.
Examples where the text contradicts the official answer (the news was overtaken) are dropped, never "fixed".
Every download is cached under data/raw/vesti/ (or data/raw/nba/injury/) and never fetched twice.
"""
from __future__ import annotations

import html
import json
import logging
import random
import re
import time
import unicodedata
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from core.config import ROOT, Settings
from model.injury_nba import parse_pdf, split_name

log = logging.getLogger(__name__)

OUT_DIR = ROOT / "tests" / "llm"
ET = ZoneInfo("America/New_York")
ESPN_INJURIES = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
EUROHOOPS_API = "https://www.eurohoops.net/wp-json/wp/v2/posts"
NBA_PDF = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{stamp}.pdf"
OFFICIAL_STATUS = {"Out": "OUT", "Doubtful": "DOUBTFUL", "Questionable": "QUESTIONABLE",
                   "Probable": "PROBABLE", "Available": "AVAILABLE"}
# explicit status words in a news text — used only to DROP examples the official report contradicts
TEXT_STATUS = [("OUT", r"\b(ruled out|will miss|won't play|will not play|is out|are out|remains? out|sidelined)\b"),
               ("DOUBTFUL", r"\bdoubtful\b"),
               ("QUESTIONABLE", r"\b(questionable|game-time decision)\b"),
               ("PROBABLE", r"\bprobable\b"),
               ("AVAILABLE", r"\b(available|will play|cleared to play|is active|will return|returns? to action)\b")]
EL_STATEMENT = [("OUT", r"(?:is|are) (?:the )?only (?:[\w’']+ )?(?:players? )?(?:ruled out|unavailable|not available)"),
                ("OUT", r"(?:is|are) (?:still )?ruled out"), ("OUT", r"(?:is|are) out\b"), ("OUT", r"only absentees?"),
                ("OUT", r"(?:is|are) (?:the )?absentees?"), ("OUT", r"will miss"), ("OUT", r"remains? sidelined"),
                ("OUT", r"(?:is|are) (?:not available|unavailable)"),
                ("QUESTIONABLE", r"(?:is|are) (?:listed as |upgraded to |downgraded to )?questionable"),
                ("QUESTIONABLE", r"game-time decisions?"), ("DOUBTFUL", r"(?:is|are) (?:listed as )?doubtful"),
                ("AVAILABLE", r"(?:is|are) (?:all )?(?:listed as |upgraded to )?available")]
EL_SEARCHES = ["EuroLeague injury report", "EuroLeague official injury", "injury report Round"]


# ---------- small helpers ----------

def norm(t: str) -> str:
    t = unicodedata.normalize("NFKD", str(t or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]", " ", t).strip()


def surname(name: str) -> str:
    """'Porter Jr., Michael' / 'Michael Porter Jr.' / 'NAVARRO, IBON' → 'porter' / 'navarro'."""
    if "," in str(name):
        name = str(name).split(",")[0]
    else:
        name = str(name).split(" ", 1)[-1] if " " in str(name) else str(name)
    toks = [x for x in norm(name).split() if x not in ("jr", "sr", "ii", "iii", "iv")]
    return toks[-1] if toks else ""


def plain(fragment: str) -> str:
    text = re.sub(r"<li[^>]*>", "\n- ", fragment)
    text = re.sub(r"</(p|h\d|li|ul)>", "\n", text)
    text = html.unescape(re.sub(r"<[^>]+>", "", text))
    return re.sub(r"\n\s*\n+", "\n", text).strip()


class Fetcher:
    """Polite cached GET (pause between requests, 404 remembered as a marker file)."""

    def __init__(self, settings: Settings):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": settings.http.user_agent})
        self.pause, self.timeout, self._last = settings.http.pause_s, settings.http.timeout_s, 0.0

    def get(self, url: str, path: Path, params: Optional[dict] = None) -> Optional[bytes]:
        missing = path.with_name(path.name + ".missing")
        if path.exists():
            return path.read_bytes()
        if missing.exists():
            return None
        time.sleep(max(0.0, self.pause - (time.monotonic() - self._last)))
        self._last = time.monotonic()
        resp = self.s.get(url, params=params, timeout=self.timeout)
        path.parent.mkdir(parents=True, exist_ok=True)
        if resp.status_code in (403, 404):
            missing.touch()
            return None
        resp.raise_for_status()
        path.write_bytes(resp.content)
        return resp.content


# ---------- T1 NBA ----------

def _slot_stamp(t: datetime) -> str:
    h12 = t.hour % 12 or 12
    return f"{t:%Y-%m-%d}_{h12:02d}_{t:%M}{'AM' if t.hour < 12 else 'PM'}"


def _report_slots(note_et: datetime) -> list[datetime]:
    """The first reports published after the note (15-minute grid), then the usual afternoon reports of the
    next two days (the team may not play on the day of the note)."""
    t0 = note_et.replace(second=0, microsecond=0) + timedelta(minutes=(15 - note_et.minute % 15) % 15 or 15)
    slots = [t0 + timedelta(minutes=15 * i) for i in range(4)]
    for d in range(3):
        day = (note_et + timedelta(days=d)).date()
        for hh in (13, 17):
            t = datetime(day.year, day.month, day.day, hh, 0, tzinfo=ET)
            if t > t0:
                slots.append(t)
    return sorted(set(slots))


def _official_status(f: Fetcher, raw: Path, note_et: datetime, team: str, player: str) -> Optional[dict]:
    for t in _report_slots(note_et):
        stamp = _slot_stamp(t)
        path = raw / "nba" / "injury" / f"{t:%Y-%m-%d}" / f"Injury-Report_{stamp}.pdf"
        if not f.get(NBA_PDF.format(stamp=stamp), path):
            continue
        rep = parse_pdf(path)
        rows = rep[rep["team"] == team]
        if rows.empty or (rows["status"] == "NOT_SUBMITTED").all():
            continue
        hit = rows[rows["player"].fillna("").map(lambda p: surname(p) == surname(player) and
                                                   norm(split_name(p)[1])[:1] == norm(player)[:1])]
        status = OFFICIAL_STATUS.get(hit["status"].iloc[0], hit["status"].iloc[0]) if len(hit) else "AVAILABLE"
        why = hit["reason"].iloc[0] if len(hit) else "не е на листата (тимот поднесе извештај)"
        return {"status": status, "izvor": f"NBA official injury report {stamp.replace('_', ' ', 1)} ET", "prichina": why}
    return None


def text_statuses(text: str) -> set[str]:
    low = text.lower()
    return {s for s, rx in TEXT_STATUS if re.search(rx, low)}


def build_t1_nba(settings: Settings, f: Fetcher, n: int, since: date, seed: int = 7) -> tuple[list[dict], dict]:
    raw = settings.paths.raw
    data = json.loads(f.get(ESPN_INJURIES, raw / "vesti" / "espn" / f"injuries_{date.today()}.json"))
    notes = [(t["displayName"], i) for t in data["injuries"] for i in t["injuries"]
             if i.get("shortComment") and i["date"][:10] >= since.isoformat()]
    random.Random(seed).shuffle(notes)
    out, stats = [], {"бележки": len(notes), "без_официјален": 0, "конфликт": 0}
    for team, i in notes:
        if len(out) >= n:
            break
        name = i["athlete"]["displayName"]
        note_et = datetime.fromisoformat(i["date"].replace("Z", "+00:00")).astimezone(ET)
        truth = _official_status(f, raw, note_et, team, name)
        if truth is None:
            stats["без_официјален"] += 1
            continue
        # on the ESPN page the team is the heading above the note → part of the text the model reads
        text = f"{team} — injuries\n{i['shortComment']}" + ("\n" + i["longComment"] if i.get("longComment") else "")
        said = text_statuses(i["shortComment"])
        if said and truth["status"] not in said:
            stats["конфликт"] += 1                 # the news was overtaken by the official report → not a fair test
            continue
        out.append({"id": f"T1-NBA-{len(out) + 1:03d}", "zadaca": "T1", "liga": "nba", "tekst": text,
                    "izvor_url": ESPN_INJURIES, "objaveno": i["date"],
                    "ocekuvano": [{"igrac": name, "tim": team, "status": truth["status"]}],
                    "vistina": f"{truth['izvor']}: {truth['status']} ({truth['prichina']})"})
    return out, stats


# ---------- T1 EuroLeague ----------

def _eurohoops(f: Fetcher, raw: Path, search: str, after: str, pages: int = 1) -> list[dict]:
    posts = []
    for page in range(1, pages + 1):
        key = re.sub(r"[^a-z0-9]+", "_", search.lower())
        body = f.get(EUROHOOPS_API, raw / "vesti" / "eurohoops" / f"search_{key}_{after}_{page}.json",
                     params={"search": search, "per_page": 50, "page": page, "after": after + "T00:00:00",
                             "_fields": "id,date,link,title,content"})
        if not body:
            break
        batch = json.loads(body)
        posts += batch
        if len(batch) < 50:
            break
    return posts


def _el_frames(settings: Settings):
    folder = settings.paths.processed / "euroleague" / settings.league("euroleague").season
    read = lambda n: pd.read_csv(folder / n, dtype={"game_code": str, "player_code": str})
    return read("team_logs.csv"), read("player_logs.csv"), read("roster.csv")


def _roster_team(names: list[str], roster: pd.DataFrame) -> Optional[str]:
    votes = {}
    for nm in names:
        last = surname(nm)
        hits = roster[roster["player"].map(lambda p: surname(p) == last and norm(p.split(",")[-1])[:1] == norm(nm)[:1])]
        for t in hits["team"].unique():
            votes[t] = votes.get(t, 0) + 1
    return max(votes, key=votes.get) if votes else None


def _statement(par: str) -> list[tuple[str, str]]:
    """The club's official sentence → [(player as written, status)]. 'X is available while Y is ruled out'
    is split into clauses; a clause without a recognised pattern or names gives nothing (never guessed)."""
    out = []
    for clause in re.split(r"\s+while\s+|;\s*", par):
        for status, rx in EL_STATEMENT:
            m = re.search(rx, clause)
            if m:
                subject = clause[:m.start()]
                # 'Georgios Papagiannis remain unavailable and' → the leading capitalised name only
                for piece in re.split(r",\s*|\s+and\s+", subject):
                    nm = re.match(r"[A-Z][\w'.-]+(?: [A-Z][\w'.-]+)+", piece.strip())
                    if nm:
                        out.append((nm.group(0), status))
                break
    return out


def build_t1_el(settings: Settings, f: Fetcher, n: int, since: date) -> tuple[list[dict], dict]:
    team_logs, player_logs, roster = _el_frames(settings)
    seen, posts = set(), []
    for q in EL_SEARCHES:
        for p in _eurohoops(f, settings.paths.raw, q, since.isoformat()):
            if p["id"] not in seen and "/en/euroleague/" in p["link"] and "injur" in p["title"]["rendered"].lower():
                seen.add(p["id"])
                posts.append(p)
    out, stats = [], {"написи": len(posts), "делови": 0, "без_изјава": 0, "без_натпревар": 0, "конфликт": 0}
    for p in sorted(posts, key=lambda x: x["date"]):
        day = date.fromisoformat(p["date"][:10])
        game_hdr = ""
        for part in re.split(r"(?=<h[34]>)", p["content"]["rendered"]):
            if part.startswith("<h3>"):
                game_hdr = plain(part.split("</h3>")[0])
                continue
            if not part.startswith("<h4>"):
                continue
            stats["делови"] += 1
            par = plain(re.search(r"<p>(.*?)</p>", part, re.S).group(1)) if "<p>" in part else ""
            st = _statement(par)
            if not st:
                stats["без_изјава"] += 1
                continue
            team = _roster_team([nm for nm, _ in st], roster)
            games = team_logs[(team_logs["team"] == team) & (pd.to_datetime(team_logs["game_date"]).dt.date >= day)
                              & (pd.to_datetime(team_logs["game_date"]).dt.date <= day + timedelta(days=3))]
            if team is None or games.empty:
                stats["без_натпревар"] += 1
                continue
            g = games.sort_values("game_date").iloc[0]
            sheet = player_logs[(player_logs["game_code"] == g["game_code"]) & (player_logs["team"] == team)]
            on_sheet = {surname(x) for x in sheet["player"]}
            on_roster = {surname(x) for x in roster[roster["team"] == team]["player"]}
            exp, bad = [], False
            for nm, status in st:
                if surname(nm) not in on_roster:       # 'Coach Peppe Poeta', a typo … → not scored
                    continue
                played = surname(nm) in on_sheet
                if (status == "OUT" and played) or (status == "AVAILABLE" and not played):
                    bad = True                         # the official word and the game sheet disagree → do not use
                exp.append({"igrac": nm, "tim": team, "status": status,
                            "box_score": "на листата" if played else "НЕ е на листата"})
            if bad:
                stats["конфликт"] += 1
                continue
            text = f"{game_hdr}\n{plain(part)}"
            out.append({"id": f"T1-EL-{len(out) + 1:03d}", "zadaca": "T1", "liga": "euroleague", "tekst": text,
                        "izvor_url": p["link"], "objaveno": p["date"], "ocekuvano": exp,
                        "vistina": f"официјален извештај на Евролига (изјава на клубот) + box score {g['game_date']} "
                                   f"(натпревар {g['game_code']}, {team}–{g['opp']})"})
    random.Random(11).shuffle(out)
    out = out[:n]
    for k, ex in enumerate(out, 1):
        ex["id"] = f"T1-EL-{k:03d}"
    return out, stats


# ---------- T9 ----------

def _nba_coaches(settings: Settings, f: Fetcher, abbr: str) -> Optional[str]:
    from nba_api.stats.endpoints import commonteamroster
    from nba_api.stats.static import teams
    path = settings.paths.raw / "nba" / "coaches" / f"{settings.league('nba').season}_{abbr}.json"
    if not path.exists():
        team = teams.find_team_by_abbreviation(abbr)
        time.sleep(settings.nba_api.pause_s)
        res = commonteamroster.CommonTeamRoster(team_id=team["id"], season=settings.league("nba").season, timeout=60)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(res.get_dict()), encoding="utf-8")
    d = json.loads(path.read_text(encoding="utf-8"))
    rs = next(r for r in d["resultSets"] if r["name"] == "Coaches")
    rows = [dict(zip(rs["headers"], r)) for r in rs["rowSet"]]
    head = [r for r in rows if r.get("COACH_TYPE") == "Head Coach"]
    return f"{head[0]['FIRST_NAME']} {head[0]['LAST_NAME']}" if head else None


T9_TEAMS = {
    "euroleague": {"MAD": "Real Madrid", "PAR": "Partizan", "BAR": "Barcelona", "OLY": "Olympiacos",
                   "PAN": "Panathinaikos", "ULK": "Fenerbahce", "ZAL": "Zalgiris", "IST": "Anadolu Efes",
                   "TEL": "Maccabi", "MIL": "Olimpia Milan", "MUN": "Bayern", "BAS": "Baskonia", "DUB": "Dubai"},
    "nba": {"LAL": "Lakers", "BOS": "Celtics", "GSW": "Warriors", "NYK": "Knicks", "DEN": "Nuggets",
            "OKC": "Thunder", "CHI": "Bulls", "MIA": "Heat", "PHI": "76ers", "SAS": "Spurs", "TOR": "Raptors",
            "MIL": "Bucks", "CLE": "Cavaliers"},
}


def build_t9(settings: Settings, f: Fetcher, n_per_league: int, since: date) -> tuple[list[dict], dict]:
    el_team_logs, el_players, el_roster = _el_frames(settings)
    nba_folder = settings.paths.processed / "nba" / settings.league("nba").season
    nba_roster = pd.read_csv(nba_folder / "roster.csv", dtype={"player_code": str})
    nba_players = pd.read_csv(nba_folder / "player_logs.csv", dtype={"player_code": str})
    out, stats = [], {}
    for lg, teams in T9_TEAMS.items():
        cands = []
        for code, name in teams.items():
            if lg == "euroleague":
                c = el_team_logs[el_team_logs["team"] == code]["coach"].dropna()
                coach = c.iloc[-1] if len(c) else None
                names = sorted(set(el_roster[el_roster["team"] == code]["player"]) |
                               set(el_players[el_players["team"] == code]["player"]))
                names = [f"{p.split(',')[1].strip().title()} {p.split(',')[0].strip().title()}" if "," in p else p
                         for p in names]
                coach = f"{coach.split(',')[1].strip().title()} {coach.split(',')[0].strip().title()}" if coach and "," in coach else coach
            else:
                coach = _nba_coaches(settings, f, code)
                names = sorted(set(nba_roster[nba_roster["team"] == code]["player"]) |
                               set(nba_players[nba_players["team"] == code]["player"]))
            posts = _eurohoops(f, settings.paths.raw, name, since.isoformat())
            for p in posts:
                title = html.unescape(p["title"]["rendered"])
                body = plain(p["content"]["rendered"])
                if ("/en/" not in p["link"] or not re.search(rf"\b{re.escape(name)}\b", title, re.I)
                        or "injury report" in title.lower() or len(body) < 1200):
                    continue
                cands.append({"liga": lg, "tim_kod": code, "tim": name, "naslov": title, "tekst": body[:6000],
                              "izvor_url": p["link"], "objaveno": p["date"],
                              "ocekuvano": {"trener": coach, "sostav": names},
                              "vistina": ("тренер: box score на Евролига (последен натпревар); состав: официјален "
                                          "состав E2026 + box score") if lg == "euroleague" else
                                         ("тренер: nba_api CommonTeamRoster (Head Coach); состав: официјален "
                                          "roster + box score 2026-27")})
        stats[lg] = len(cands)
        rnd = random.Random(23)
        rnd.shuffle(cands)
        picked, per_team = [], {}
        for c in cands:                                # spread across teams: at most 3 articles per team
            if per_team.get(c["tim_kod"], 0) < 3 and len(picked) < n_per_league:
                picked.append(c)
                per_team[c["tim_kod"]] = per_team.get(c["tim_kod"], 0) + 1
        out += picked
    for k, ex in enumerate(out, 1):
        ex.update(id=f"T9-{ex['liga'][:3].upper()}-{k:03d}", zadaca="T9")
    return out, stats


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_all(settings: Settings, since_t1: date, since_t9: date) -> dict:
    f = Fetcher(settings)
    nba, s_nba = build_t1_nba(settings, f, 35, since_t1)
    el, s_el = build_t1_el(settings, f, 50 - len(nba) if len(nba) < 35 else 15, since_t1 - timedelta(days=14))
    t9, s_t9 = build_t9(settings, f, 25, since_t9)
    write_jsonl(nba + el, OUT_DIR / "T1.jsonl")
    write_jsonl(t9, OUT_DIR / "T9.jsonl")
    return {"T1": len(nba) + len(el), "T1_nba": s_nba | {"избрани": len(nba)}, "T1_el": s_el | {"избрани": len(el)},
            "T9": len(t9), "T9_kandidati": s_t9}
