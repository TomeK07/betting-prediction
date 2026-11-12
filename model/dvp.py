"""DvP — defence vs position (BUILD_PROMPT §3.1).

For every defending team and position: what opposing players of that position produce per 36
minutes against it, relative to the league average for that position (+10% = allows 10% more).
Only games before `before` are used (no look-ahead). Positions come from the official roster:
EuroLeague/EuroCup Guard/Forward/Center, NBA G/F/C (first letter of 'G-F', 'F-C').
Small samples are shrunk toward the league (pseudo-minutes).
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

STATS = ["pts", "reb", "ast", "fgm3"]
POSITION_MAP = {"guard": "G", "forward": "F", "center": "C", "g": "G", "f": "F", "c": "C"}


def primary_position(raw: Optional[str]) -> Optional[str]:
    if not isinstance(raw, str) or not raw.strip():
        return None
    return POSITION_MAP.get(raw.strip().lower().split("-")[0])


def dvp_table(player_logs: pd.DataFrame, roster: pd.DataFrame, before: Optional[str] = None,
              shrink_minutes: float = 150.0) -> pd.DataFrame:
    """Rows: defending team × position; columns per stat: per-36 allowed, league per-36, % vs league."""
    logs = player_logs.copy()
    if before is not None:
        logs = logs[logs["game_date"] < before]
    logs = logs[(~logs["dnp"].astype(bool)) & (logs["minutes"] > 0)]
    pos = roster.assign(pos=roster["position"].map(primary_position)).drop_duplicates("player_code")
    logs = logs.merge(pos[["player_code", "pos"]], on="player_code", how="left")
    unknown_share = float(logs["pos"].isna().mean()) if len(logs) else 0.0
    logs = logs[logs["pos"].notna()]

    league = logs.groupby("pos")[STATS + ["minutes"]].sum()
    lg36 = league[STATS].div(league["minutes"], axis=0) * 36
    grp = logs.groupby(["opp", "pos"])[STATS + ["minutes"]].sum()
    out = grp[["minutes"]].rename(columns={"minutes": "minuti"}).copy()
    for s in STATS:
        lg = grp.index.get_level_values("pos").map(lg36[s])
        # shrink: add `shrink_minutes` of league-average production
        per36 = (grp[s] + lg * shrink_minutes / 36) / (grp["minutes"] + shrink_minutes) * 36
        out[f"{s}_36"] = per36.values
        out[f"{s}_liga_36"] = lg.values
        out[f"{s}_vs_liga_pct"] = ((per36 / lg) - 1).values * 100
    out.index = out.index.set_names(["tim", "pozicija"])
    out.attrs["unknown_position_share"] = unknown_share
    return out.reset_index()
