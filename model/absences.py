"""Absences: who is out, NEW vs OLD (BUILD_PROMPT §2.4, rule 3), and the effect on teammates (M2).

NEW absence = the player played (minutes > 0) in at least one of the team's last 3 games and is out now.
OLD absence = he did not play in any of the last 3 → already inside the averages, no edge (rule 3).
Decided by code from box scores (error 22.10: old absences were used as arguments).

Status policy:
- NBA: Out / Doubtful → does not play; Questionable → ЗАВИСИ_ОД_СОСТАВ (not in the main ticket);
  Probable / Available → plays.
- EuroLeague / EuroCup: questionable / GTD / "hoping" = PLAYS (error 27.10: Hifi, Simmons).
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

import pandas as pd

from model.injury_nba import split_name
from model.logs_common import SUFFIX_RE, clean_suffix
from model.player_model import STATS, PlayerParams, add_features, prepare
from model.team_model import REGULATION_MINUTES

NEW_LOOKBACK = 3
ROTATION_MIN = {"nba": 20.0, "euroleague": 15.0, "eurocup": 15.0}
OUT_STATUSES = {"OUT", "DOUBTFUL"}
DEPENDS = "ЗАВИСИ_ОД_СОСТАВ"


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower()
    text = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", text)
    return re.sub(r"[^a-z]", "", text)


def team_code_map(schedule: pd.DataFrame) -> dict[str, str]:
    names = pd.concat([schedule[["home", "home_name"]].set_axis(["c", "n"], axis=1),
                       schedule[["away", "away_name"]].set_axis(["c", "n"], axis=1)]).drop_duplicates()
    return {_norm(n): c for c, n in names.itertuples(index=False)} | {_norm(c): c for c in names["c"]}


def match_player(last: str, first: str, team: str, roster: pd.DataFrame) -> tuple[Optional[str], str]:
    """Rule 2: surname + first name + team, never surname alone. Returns (player_code, note)."""
    r = roster[roster["team"] == team]
    full = r["player"].fillna("").map(clean_suffix)
    # EL style "LAST, FIRST" only when most names have a comma: one "Ernest Udeh, Jr." flipped all of CLE (2.11)
    if full.str.contains(",").mean() > 0.5:
        lasts = full.str.split(",").str[0]
        firsts = full.str.split(",").str[1].fillna("")
    else:                                                # NBA style "First Last"
        firsts = full.str.split(" ").str[0]
        lasts = full.str.split(" ", n=1).str[1].fillna("")
    bare = lambda s: _norm(SUFFIX_RE.sub("", str(s)))   # "Udeh Jr." == "Udeh"
    ok_last = lasts.map(bare) == bare(last)
    ok_first = firsts.map(_norm).str.startswith(_norm(first)[:3]) if first else True
    hits = r[ok_last & ok_first]
    if len(hits) > 1 and first:                          # Jalen / Jaylin: the full first name decides
        exact = hits[firsts[hits.index].map(_norm) == _norm(first)]
        hits = exact if len(exact) == 1 else hits
    if len(hits) == 1:
        return str(hits["player_code"].iat[0]), ""
    if hits.empty:
        return None, f"„{first} {last}“ не е во составот на {team}"
    return None, f"„{first} {last}“ — повеќе играчи во {team}"


def classify(player_code: str, team: str, logs: pd.DataFrame, before: str) -> dict:
    """NEW / OLD from the team's last 3 games before `before` (ISO date)."""
    t = logs[(logs["team"] == team) & (logs["game_date"] < before)]
    games = t.drop_duplicates("game_code").sort_values(["date_utc", "game_code"])["game_code"].tolist()
    last = set(games[-NEW_LOOKBACK:])
    mine = t[(t["player_code"] == player_code) & (t["minutes"].fillna(0) > 0)]
    played_recent = mine["game_code"].isin(last).any()
    return {"novo": bool(played_recent), "tip": "НОВО" if played_recent else "СТАРО",
            "minuti_sezona": float(mine["minutes"].mean()) if len(mine) else 0.0,
            "natprevari": int(len(mine))}


def plays(status: str, league: str) -> Optional[bool]:
    """True = plays, False = out, None = depends on lineup (NBA questionable)."""
    s = (status or "").upper()
    if s in OUT_STATUSES:
        return False
    if s == "NEPOTVRDEN":                            # manual: club sent no report, missed the last game (Tome 30.10)
        return None
    if s == "QUESTIONABLE":
        return None if league == "nba" else True      # EL/EC: questionable = plays (error 27.10)
    return True


def last_game_absences(logs: pd.DataFrame, roster: pd.DataFrame, team: str, before: str, league: str) -> pd.DataFrame:
    """EL/EC (no official PDF): rotation players who did not play the team's last game (DNP or not listed)."""
    t = logs[(logs["team"] == team) & (logs["game_date"] < before)]
    if t.empty:
        return pd.DataFrame()
    last_code = t.sort_values(["date_utc", "game_code"])["game_code"].iat[-1]
    played_last = set(t[(t["game_code"] == last_code) & (t["minutes"].fillna(0) > 0)]["player_code"])
    season = t[t["minutes"].fillna(0) > 0].groupby("player_code")["minutes"].mean()
    rot = season[season >= ROTATION_MIN[league]].index
    missing = [p for p in rot if p not in played_last]
    rows = []
    for p in missing:
        info = classify(p, team, logs, before)
        name = roster.loc[roster["player_code"] == p, "label"]
        rows.append({"team": team, "player_code": p, "label": name.iat[0] if len(name) else p,
                     "status": "ПРОПУШТИЛ ПОСЛЕДЕН", "reason": "box score", **info})
    return pd.DataFrame(rows)


def pregame_minutes(logs: pd.DataFrame, roster: pd.DataFrame, league: str, team: str, before: str,
                    out_codes: set[str], params: PlayerParams, recent_games: int = 10,
                    old_out: frozenset | set = frozenset()) -> pd.DataFrame:
    """Tonight's projected minutes/stats for a team (state after each player's last game), M2-scaled.

    min_bez = everyone in `out_codes` removed (the projection picks uses). min_site = baseline WITHOUT the OLD
    absences (already in the averages, rule 3) → efekt_* = what the NEW absences change (Tome 2.11: before, the
    old ones were counted too, e.g. SGA +11.6 when only Caruso was new).
    """
    cur = prepare(logs[logs["game_date"] < before], roster)
    if "season_code" not in cur:
        cur["season_code"] = "cur"
    feats = add_features(cur, league, params)
    t = feats[feats["team"] == team]
    if t.empty:
        return pd.DataFrame()
    recent = t.drop_duplicates("game_code").sort_values(["date_utc", "game_code"])["game_code"].tolist()[-recent_games:]
    active = t[t["game_code"].isin(recent)]["player_code"].unique()
    last = (t[t["player_code"].isin(active)].sort_values(["date_utc", "game_code"]).groupby("player_code").tail(1)
            .set_index("player_code"))
    reg = REGULATION_MINUTES[league] * 5
    out = pd.DataFrame({"label": last["label"], "pos": last["pos"], "min_ewm": last["min_ewm_after"]})
    for s in STATS:
        out[f"{s}_rate"] = last[f"{s}_rate_after"]

    def scaled(mask: pd.Series) -> pd.Series:
        m = out["min_ewm"].where(mask, 0.0)
        return (m * reg / m.sum()).clip(upper={"nba": 44.0}.get(league, 38.0)) if m.sum() else m

    out["min_site"] = scaled(~out.index.isin(set(old_out)))
    out["min_bez"] = scaled(~out.index.isin(out_codes))
    out["efekt_min"] = out["min_bez"] - out["min_site"]
    for s in STATS:
        out[f"proj_{s}"] = out["min_bez"] * out[f"{s}_rate"]
        out[f"efekt_{s}"] = (out["min_bez"] - out["min_site"]) * out[f"{s}_rate"]
    out["out"] = out.index.isin(out_codes)
    return out.reset_index().sort_values("min_bez", ascending=False)
