"""Closing price and CLV for paper/real picks (decision 29.10: no scheduled jobs).

Closing price = the last snapshot taken BEFORE tip-off (only exists if Tome ran `fetch`/`linii` before the
game), same bookmaker, same match, same market, side and LINE. If the line moved and ours is no longer
offered, there is no closing price → kvota_zatvaracka and clv stay empty (stats uses only picks that have it).
CLV = our price / closing price − 1  (> 0 = we beat the closing line).
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from functools import lru_cache
from typing import Optional

import pandas as pd

from core.config import Settings
from model.lines import build_day


@lru_cache(maxsize=16)
def _day_lines(raw_root: str, day: str) -> pd.DataFrame:
    from pathlib import Path
    return build_day(Path(raw_root), day)


def closing_price(settings: Settings, pick: dict) -> Optional[tuple[float, str]]:
    try:
        vlez = json.loads(pick.get("vlezovi_json") or "{}")
    except ValueError:
        return None
    book, nat = vlez.get("kladilnica"), (vlez.get("nat_ids") or {}).get(vlez.get("kladilnica"))
    if not book or not nat or pick.get("pazar") == "hendikep":     # handicap sides need team mapping — later
        return None
    d = date.fromisoformat(pick["datum"])
    frames = [_day_lines(str(settings.paths.raw), (d + timedelta(days=k)).isoformat()) for k in (0, 1)]
    L = pd.concat([f for f in frames if len(f)], ignore_index=True) if any(len(f) for f in frames) else pd.DataFrame()
    if L.empty:
        return None
    markets = {"tim_vkupno": ["tim_vkupno_d", "tim_vkupno_g"]}.get(pick["pazar"], [pick["pazar"]])
    rows = L[(L["kladilnica"] == book) & (L["nat_id"] == str(nat)) & L["pazar"].isin(markets)
             & (L["strana"] == pick["strana"]) & (L["linija"] == float(pick["linija"]))]
    if rows.empty:
        return None
    start = pd.to_datetime(rows["pocetok_utc"].iat[0], utc=True)
    snap = pd.to_datetime(rows["snimka"], utc=True)
    before = rows[snap < start]
    if before.empty:
        return None
    last = before.loc[pd.to_datetime(before["snimka"], utc=True).idxmax()]
    return float(last["kvota"]), str(last["snimka"])
