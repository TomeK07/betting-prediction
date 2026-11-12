"""Grade waiting picks (status CEKA) from official box scores (BUILD_PROMPT §6, PROCES §3 step 0).

Pure logic: works on the processed logs (no network). Never touches p_model.
A pick that cannot be matched unambiguously (team, game or player) stays CEKA with a reason —
rule 2 (error 16.10): never guess who the player is.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import pandas as pd

PLAYER_STATS = {
    "poeni": ["pts"], "asistencii": ["ast"], "skokovi": ["reb"], "trojki": ["fgm3"],
    "pra": ["pts", "reb", "ast"], "poeni+skokovi": ["pts", "reb"], "skokovi+asistencii": ["reb", "ast"],
}
TEAM_MARKETS = {"vkupno", "tim_vkupno", "hendikep", "1x2", "dvoen_sans"}
NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}
MATCHUP_SPLIT = re.compile(r"\s+(?:-|@|vs\.?|–)\s+")


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9 \-]", " ", text.lower()).strip()


@dataclass
class LeagueData:
    schedule: pd.DataFrame
    team_logs: pd.DataFrame
    player_logs: pd.DataFrame
    aliases: dict[str, str]


@dataclass
class GradeResult:
    status: str                       # DA | NE | VOID | CEKA (unresolved)
    reason: str
    value: Optional[float] = None
    minutes: Optional[float] = None
    game_code: Optional[str] = None


# ---------- teams & games ----------

def resolve_team(alias: Optional[str], data: LeagueData) -> Optional[str]:
    if not alias:
        return None
    key = norm(re.sub(r"\(.*?\)", "", alias))
    if key in data.aliases:
        return data.aliases[key]
    codes = set(data.schedule["home"]) | set(data.schedule["away"])
    if key.upper() in codes:
        return key.upper()
    names = pd.concat([data.schedule[["home", "home_name"]].set_axis(["code", "name"], axis=1),
                       data.schedule[["away", "away_name"]].set_axis(["code", "name"], axis=1)]).drop_duplicates()
    tokens = key.split()
    hits = set()
    for code, name in names.itertuples(index=False):
        words = norm(name).split()
        # every alias token must be the start of a name word, in order
        i = 0
        for w in words:
            if i < len(tokens) and w.startswith(tokens[i]):
                i += 1
        if i == len(tokens):
            hits.add(code)
    return hits.pop() if len(hits) == 1 else None


def candidate_dates(datum: str, vreme: Optional[str], league: str) -> list[str]:
    """Legacy datum is a Skopje date; for NBA the game's ET date is that day or the day before.

    `vreme` is NOT used: in the legacy archive it is often the time the pick was written, not tip-off
    (verified 29.10 on MIL 23/24.10). If the team plays on both days the game is ambiguous → manual.
    """
    d = date.fromisoformat(datum)
    if league != "nba":
        return [d.isoformat()]
    return [(d - timedelta(days=1)).isoformat(), d.isoformat()]


def find_game(team: str, dates: list[str], data: LeagueData) -> tuple[Optional[pd.Series], str]:
    s = data.schedule
    games = s[((s["home"] == team) | (s["away"] == team)) & s["game_date"].isin(dates)]
    if len(games) == 0:
        return None, f"нема натпревар на {team} на {'/'.join(dates)}"
    if len(games) > 1:
        return None, (f"{team} игра повеќе натпревари на {'/'.join(dates)} "
                      f"({', '.join(games['home'] + '–' + games['away'])}) — рачно")
    g = games.iloc[0]
    if g["status"] != "result":
        return None, f"натпреварот {g['home']}–{g['away']} уште не е завршен во податоците (пушти fetch --refresh)"
    return g, ""


def teams_of_pick(pick: dict, data: LeagueData) -> list[str]:
    """All team codes named by the pick (tim column, matchup text, handicap side)."""
    found = []
    for alias in [pick.get("tim"), *MATCHUP_SPLIT.split(pick.get("natprevar") or "")]:
        code = resolve_team(alias, data)
        if code and code not in found:
            found.append(code)
    return found


# ---------- players ----------

def _split_legacy_name(name: str) -> tuple[str, str]:
    """'Don. Mitchell' -> ('don', 'mitchell'); 'J. Clark III' -> ('j', 'clark'); 'K. A. Towns' -> ('k', 'towns')."""
    tokens = [t for t in norm(name.replace(".", " ")).split()]
    while len(tokens) > 1 and tokens[-1] in NAME_SUFFIXES:
        tokens.pop()
    if len(tokens) == 1:
        return "", tokens[0]
    return tokens[0], tokens[-1]


def _player_names(row: pd.Series) -> tuple[str, list[str]]:
    """(given names, surname tokens) from a log row — EL 'LAST, FIRST' or NBA 'First Last'."""
    full = str(row["player"])
    if "," in full:
        last, first = full.split(",", 1)
    else:
        last = re.sub(r"\s+\w\.\s+\(.*\)$", "", str(row["label"]))   # label 'Towns K. (NYK)'
        first = full.replace(last, "").strip()
    surname_tokens = [t for t in norm(last).replace("-", " ").split() if t not in NAME_SUFFIXES]
    return norm(first), surname_tokens + [norm(last)]


def match_player(name: str, rows: pd.DataFrame) -> tuple[Optional[pd.Series], str]:
    given, surname = _split_legacy_name(name)
    surname_parts = surname.replace("-", " ").split()
    hits = []
    for _, r in rows.iterrows():
        first, last_tokens = _player_names(r)
        if not (surname in last_tokens or all(p in last_tokens for p in surname_parts)):
            continue
        if given and not first.startswith(given):
            continue
        hits.append(r)
    if len(hits) == 1:
        return hits[0], ""
    if not hits:
        return None, f"играчот „{name}“ не е најден"
    return None, f"„{name}“ одговара на повеќе играчи: {', '.join(h['label'] for h in hits)}"


# ---------- outcome ----------

def settle(value: float, line: float, side: str) -> str:
    if value == line:
        return "VOID"                     # push
    over = value > line
    return "DA" if (over if side == "nad" else not over) else "NE"


def _game_by_code(pick: dict, data: LeagueData) -> tuple[Optional[pd.Series], str]:
    """New picks (Phase 7) carry the official game code → no date guessing."""
    s = data.schedule[data.schedule["game_code"].astype(str) == str(pick["game_code"])]
    if s.empty:
        return None, f"натпревар {pick['game_code']} не е во распоредот"
    g = s.iloc[0]
    if g["status"] != "result":
        return None, f"натпреварот {g['home']}–{g['away']} уште не е завршен во податоците"
    return g, ""


def _player_code(pick: dict) -> Optional[str]:
    try:
        return (json.loads(pick.get("vlezovi_json") or "{}") or {}).get("player_code")
    except (TypeError, ValueError):
        return None


def grade_pick(pick: dict, data: LeagueData) -> GradeResult:
    market = pick["pazar"]
    dates = candidate_dates(pick["datum"], pick.get("vreme"), pick["liga"])
    by_code = bool(pick.get("game_code"))

    if market in PLAYER_STATS:
        team = resolve_team(pick.get("tim"), data)
        if not team:
            return GradeResult("CEKA", f"тимот „{pick.get('tim')}“ не е препознаен")
        game, why = _game_by_code(pick, data) if by_code else find_game(team, dates, data)
        if game is None:
            return GradeResult("CEKA", why)
        rows = data.player_logs[(data.player_logs["game_code"].astype(str) == str(game["game_code"]))
                                & (data.player_logs["team"] == team)]
        code = _player_code(pick)
        if code:                                   # new picks: official player id, never a name match
            hit = rows[rows["player_code"].astype(str) == str(code)]
            player, why = (hit.iloc[0], "") if len(hit) else (None, "играчот не е во записникот")
            if player is None:
                return GradeResult("VOID", "играчот не е во записникот (не играл)", game_code=str(game["game_code"]))
        else:
            player, why = match_player(pick["igrac"], rows)
        if player is None:
            # Not on this game's sheet: VOID only if he is a known player of the team this season
            season_rows = data.player_logs[data.player_logs["team"] == team]
            known, _ = match_player(pick["igrac"], season_rows.drop_duplicates("player_code"))
            if known is not None:
                return GradeResult("VOID", f"{known['label']} не е во записникот (не играл)",
                                   game_code=str(game["game_code"]))
            return GradeResult("CEKA", why)
        if player["dnp"] or not player["minutes"]:
            reason = player.get("absence_reason") if isinstance(player.get("absence_reason"), str) else "DNP"
            return GradeResult("VOID", f"{player['label']} не играл ({reason})", minutes=0.0,
                               game_code=str(game["game_code"]))
        value = float(sum(player[c] for c in PLAYER_STATS[market]))
        if pick.get("linija") is None or pick.get("strana") not in ("nad", "pod"):
            return GradeResult("CEKA", "нема линија/страна за оценување", value=value)
        return GradeResult(settle(value, pick["linija"], pick["strana"]),
                           f"{player['label']}: {value:g} ({player['minutes']:.1f} мин)",
                           value=value, minutes=float(player["minutes"]), game_code=str(game["game_code"]))

    if market in TEAM_MARKETS:
        teams = teams_of_pick(pick, data)
        side_team = resolve_team(pick.get("strana"), data) if market == "hendikep" else None
        anchor = side_team or (teams[0] if teams else None)
        if not anchor and not by_code:
            return GradeResult("CEKA", f"тимовите во „{pick.get('natprevar')}“ не се препознаени")
        game, why = _game_by_code(pick, data) if by_code else find_game(anchor, dates, data)
        if game is None:
            return GradeResult("CEKA", why)
        if teams and not set(teams) <= {game["home"], game["away"]}:
            return GradeResult("CEKA", f"тимовите {teams} не се совпаѓаат со {game['home']}–{game['away']}")
        tl = data.team_logs[data.team_logs["game_code"].astype(str) == str(game["game_code"])].set_index("team")
        home, away = game["home"], game["away"]
        h, a = float(tl.loc[home, "pts"]), float(tl.loc[away, "pts"])
        score = f"{home} {h:g}–{a:g} {away}" + (" (прод.)" if tl.loc[home, "overtimes"] else "")
        gc = str(game["game_code"])
        if market in ("vkupno", "tim_vkupno") and (pick.get("linija") is None or pick.get("strana") not in ("nad", "pod")):
            return GradeResult("CEKA", f"нема линија/страна за оценување ({score})", game_code=gc)
        if market == "vkupno":
            return GradeResult(settle(h + a, pick["linija"], pick["strana"]), f"{score} = {h + a:g}", h + a, game_code=gc)
        if market == "tim_vkupno":
            team = resolve_team(pick.get("tim") or (pick.get("natprevar") or ""), data)
            if team not in (home, away):
                return GradeResult("CEKA", "не е јасно за кој тим е тимскиот тотал")
            pts = h if team == home else a
            return GradeResult(settle(pts, pick["linija"], pick["strana"]), f"{score}; {team} {pts:g}", pts, game_code=gc)
        if market == "hendikep":
            if not side_team or pick.get("linija") is None:
                return GradeResult("CEKA", f"хендикеп без јасен тим/линија („{pick.get('strana')}“)")
            margin = (h - a) if side_team == home else (a - h)
            adjusted = margin + pick["linija"]
            status = "VOID" if adjusted == 0 else ("DA" if adjusted > 0 else "NE")
            return GradeResult(status, f"{score}; {side_team} {pick['linija']:+g} → {adjusted:+g}", margin, game_code=gc)
        # 1X2 / double chance: regulation time only — an overtime game was a draw after 40/48 min
        if tl.loc[home, "overtimes"]:
            outcome = "X"
        else:
            outcome = "1" if h > a else "2"
        wanted = str(pick.get("strana") or "").upper()
        if not wanted:
            return GradeResult("CEKA", "нема страна за 1X2")
        return GradeResult("DA" if outcome in wanted else "NE", f"{score}; регуларно = {outcome}", game_code=gc)

    return GradeResult("CEKA", f"пазарот „{market}“ не се оценува автоматски")
