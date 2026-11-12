"""Normalized odds table from raw bookmaker snapshots (Mozzart, evrotip) — decision 29.10.

One row = one price: bookmaker, snapshot time, league, match, (player), market, side, line, odds, main-line flag.
Uses: line movement through the day, margin per bookmaker × market, de-vig, best price per pick (Phase 7).
Market names follow the archive: vkupno, tim_vkupno_d, tim_vkupno_g, hendikep, 1x2, poeni, asistencii,
skokovi, trojki, pra. Side: nad / pod / 1 / X / 2.
"""
from __future__ import annotations

import glob
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import brentq
import pandas as pd

COLUMNS = ["kladilnica", "snimka", "liga", "nat_id", "pocetok_utc", "domakjin", "gostin", "igrac", "tim_igrac",
           "pazar", "strana", "linija", "kvota", "glavna", "napomena"]

# ---------------- Mozzart ----------------
MOZZART_MARKETS = {           # gameShortName -> (market, main line?)
    "вп": ("vkupno", True), "впт1": ("tim_vkupno_d", True), "впт2": ("tim_vkupno_g", True),
    "кт. хен.": ("hendikep", True), "кт": ("1x2", True),
    "идп": ("poeni", True), "мбпи": ("poeni", False), "вбпи": ("poeni", False),
    "ива": ("asistencii", True), "ивс": ("skokovi", True), "ипса": ("pra", True), "ит": ("trojki", True),
}
MOZZART_SIDES = {"помалку": "pod", "повеќе": "nad", "1": "1", "2": "2", "x": "X"}


def _ts(ms) -> Optional[str]:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat(timespec="minutes") if ms else None


def parse_mozzart(payload: dict) -> pd.DataFrame:
    labels = payload.get("competitions", {})
    matches = {m["id"]: m for m in payload.get("matches", [])}
    rows = []
    for o in payload.get("odds", []):
        m = matches.get(o.get("id"))
        if not m:
            continue
        parts = [p.get("name") for p in m.get("participants", [])]
        player_market = len(parts) == 1
        liga = labels.get(str(m["competition"]["id"]), "?").replace("_igraci", "")
        for k in (o.get("kodds") or {}).values():
            if not k or not k.get("subGame"):
                continue
            sg = k["subGame"]
            short = sg.get("gameShortName")
            if short not in MOZZART_MARKETS:
                continue
            market, main = MOZZART_MARKETS[short]
            side = MOZZART_SIDES.get(str(sg.get("subGameName", "")).strip().lower())
            if side is None:
                continue
            try:
                odds = float(k["value"])
                line = float(k["specialOddValue"]) if k.get("specialOddValue") not in (None, "") else None
            except (TypeError, ValueError):
                continue
            rows.append({"kladilnica": "mozzart", "snimka": payload.get("fetched_at"), "liga": liga,
                         "nat_id": str(m["id"]), "pocetok_utc": _ts(m.get("startTime")),
                         "domakjin": None if player_market else parts[0], "gostin": None if player_market else parts[1],
                         "igrac": parts[0] if player_market else None, "pazar": market, "strana": side,
                         "linija": line, "kvota": odds, "glavna": main,
                         "napomena": "хендикеп: линијата е за домаќинот" if market == "hendikep" else None})
    return pd.DataFrame(rows, columns=COLUMNS)


# ---------------- evrotip ----------------
EVROTIP_TEAM = {      # tipTypeName prefix -> (market, side, params key, main?)
    "G_PLUS": ("vkupno", "nad", "overUnder", True), "G_MINUS": ("vkupno", "pod", "overUnder", True),
    "GH_PLUS": ("tim_vkupno_d", "nad", "homeOverUnder", True), "GH_MINUS": ("tim_vkupno_d", "pod", "homeOverUnder", True),
    "GA_PLUS": ("tim_vkupno_g", "nad", "awayOverUnder", True), "GA_MINUS": ("tim_vkupno_g", "pod", "awayOverUnder", True),
    "KI_1": ("1x2", "1", None, True), "KI_X": ("1x2", "X", None, True), "KI_2": ("1x2", "2", None, True),
    "H_1": ("hendikep", "1", "hd2", True), "H_2": ("hendikep", "2", "hd2", True),
}
for n in range(2, 8):
    EVROTIP_TEAM[f"OU{n}_PLUS"] = ("vkupno", "nad", f"overUnder{n}", False)
    EVROTIP_TEAM[f"OU{n}_MINUS"] = ("vkupno", "pod", f"overUnder{n}", False)
EVROTIP_PLAYER = {
    "PL_POINTS_OVER": ("poeni", "nad", "ouPlPoints", True), "PL_POINTS_UNDER": ("poeni", "pod", "ouPlPoints", True),
    "PL_POINTS_OVER_2": ("poeni", "nad", "ouPlP2", False), "PL_POINTS_UNDER_2": ("poeni", "pod", "ouPlP2", False),
    "PL_POINTS_OVER_3": ("poeni", "nad", "ouPlP3", False), "PL_POINTS_UNDER_3": ("poeni", "pod", "ouPlP3", False),
    "PL_ASSISTS_OVER": ("asistencii", "nad", "ouPlAssists", True), "PL_ASSISTS_UNDER": ("asistencii", "pod", "ouPlAssists", True),
    "PL_REBOUNDS_OVER": ("skokovi", "nad", "ouPlRebounds", True), "PL_REBOUNDS_UNDER": ("skokovi", "pod", "ouPlRebounds", True),
    "PL_3PTS_OVER": ("trojki", "nad", "ouPl3Points", True), "PL_3PTS_UNDER": ("trojki", "pod", "ouPl3Points", True),
    "PL_PRA_OVER": ("pra", "nad", "ouPlTPRA", True), "PL_PRA_UNDER": ("pra", "pod", "ouPlTPRA", True),
}


def parse_evrotip(payload: dict, dictionary: dict) -> pd.DataFrame:
    bpm = dictionary.get("betPickMap", {})
    rows = []
    for m in payload.get("matches", []):
        sport = m.get("sport")
        label = m.get("_league", "?")
        player_market = sport == "SK"
        table = EVROTIP_PLAYER if player_market else EVROTIP_TEAM
        params = m.get("params") or {}
        for code, odds in (m.get("odds") or {}).items():
            pick = bpm.get(f"{code}_{sport}")
            if not pick or pick.get("tipTypeName") not in table:
                continue
            market, side, key, main = table[pick["tipTypeName"]]
            line = None
            if key:
                try:
                    line = float(params[key])
                except (KeyError, TypeError, ValueError):
                    continue
            rows.append({"kladilnica": "evrotip", "snimka": payload.get("fetched_at"),
                         "liga": label.replace("_igraci", ""), "nat_id": str(m["id"]),
                         "pocetok_utc": _ts(m.get("kickOffTime")),
                         "domakjin": None if player_market else m.get("home"),
                         "gostin": None if player_market else m.get("away"),
                         "tim_igrac": m.get("away") if player_market else None,   # player markets: the player's OWN team
                         "igrac": m.get("home") if player_market else None,
                         "pazar": market, "strana": side, "linija": line, "kvota": float(odds), "glavna": main,
                         "napomena": ("хендикеп: знакот НЕ е потврден (evrotip hd2)" if market == "hendikep"
                                      else None)})
    return pd.DataFrame(rows, columns=COLUMNS)


# ---------------- table, margin, de-vig, movement ----------------

def _read(path: str):
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_day(raw_root: Path, day: str) -> pd.DataFrame:
    """All snapshots of one day from both bookmakers → one normalized table."""
    frames = []
    for path in sorted(glob.glob(str(raw_root / "mozzart" / day / "*.json*"))):
        frames.append(parse_mozzart(_read(path)))
    dict_path = raw_root / "evrotip" / day / "ttg_lang.json.gz"
    if dict_path.exists():
        dictionary = _read(str(dict_path))
        for path in sorted(glob.glob(str(raw_root / "evrotip" / day / "[0-9]*.json.gz"))):
            frames.append(parse_evrotip(_read(path), dictionary))
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    return mark_main_lines(pd.concat(frames, ignore_index=True))


def mark_main_lines(lines: pd.DataFrame) -> pd.DataFrame:
    """Mozzart lists every alternative line under the same market: the MAIN line is the one whose two
    prices are closest to each other (≈1.90/1.90, BUILD_PROMPT §2.1). Other books keep their own flag."""
    df = lines.copy()
    multi = (df["kladilnica"] == "mozzart") & df["pazar"].isin(["vkupno", "tim_vkupno_d", "tim_vkupno_g", "hendikep"])
    sub = df[multi]
    if sub.empty:
        return df
    keys = ["kladilnica", "snimka", "nat_id", "pazar", "linija"]
    piv = sub.pivot_table(index=keys, columns="strana", values="kvota", aggfunc="first")
    col = lambda c: piv[c] if c in piv.columns else pd.Series(np.nan, index=piv.index)
    piv["_a"] = col("nad").combine_first(col("1"))        # totals: НАД/ПОД, handicap: side 1 / side 2
    piv["_b"] = col("pod").combine_first(col("2"))
    piv = piv.dropna(subset=["_a", "_b"])
    if piv.empty:
        return df
    piv["gap"] = (1 / piv["_a"] - 1 / piv["_b"]).abs()
    best = piv.reset_index().sort_values("gap").drop_duplicates(["kladilnica", "snimka", "nat_id", "pazar"])
    main_keys = set(map(tuple, best[keys].astype(str).values))
    df.loc[multi, "glavna"] = [tuple(map(str, r)) in main_keys for r in df.loc[multi, keys].values]
    return df


PAIR_KEYS = ["kladilnica", "snimka", "nat_id", "pazar", "linija"]


def margins(lines: pd.DataFrame) -> pd.DataFrame:
    """Overround of two-way markets (НАД/ПОД, 1/2 handicap): 1/k1 + 1/k2 − 1, per bookmaker × market."""
    two = lines[lines["strana"].isin(["nad", "pod"]) | ((lines["pazar"] == "hendikep") & lines["strana"].isin(["1", "2"]))]
    g = two.groupby(PAIR_KEYS + ["igrac"], dropna=False)
    pairs = g.filter(lambda x: len(x) == 2)
    over = pairs.groupby(PAIR_KEYS + ["igrac"], dropna=False)["kvota"].apply(lambda k: float((1 / k).sum() - 1))
    out = over.reset_index().rename(columns={"kvota": "marza"})
    meta = pairs.drop_duplicates(PAIR_KEYS + ["igrac"])[PAIR_KEYS + ["igrac", "glavna"]]
    return out.merge(meta, on=PAIR_KEYS + ["igrac"], how="left")


DEVIG_METHODS = ("proporcionalen", "power", "shin")
DEVIG_METHOD = "proporcionalen"      # chosen by `backtest --devig` (see docs/PLAN.md); report always names it


def devig_two_way(k_side: float, k_other: float, method: Optional[str] = None) -> float:
    """Fair probability of `side` with the bookmaker margin removed.

    proporcionalen: a / (a + b) — the margin is taken evenly; overrates longshots when prices are very unequal.
    power:          p = a^k with a^k + b^k = 1 — takes more margin from the longshot (favourite-longshot bias).
    shin:           Shin (1993) insider model, two outcomes — also takes more from the longshot.
    (a = 1/k_side, b = 1/k_other)
    """
    method = method or DEVIG_METHOD
    a, b = 1 / k_side, 1 / k_other
    if method == "proporcionalen" or abs(a + b - 1) < 1e-12:
        return a / (a + b)
    if method == "power":
        if a + b < 1:                                   # no margin (or negative): fall back
            return a / (a + b)
        k = brentq(lambda k: a ** k + b ** k - 1, 1.0, 50.0)
        return a ** k
    if method == "shin":
        s = a + b

        def probs(z):
            return [(np.sqrt(z * z + 4 * (1 - z) * x * x / s) - z) / (2 * (1 - z)) for x in (a, b)]
        if s < 1:
            return a / s
        z = brentq(lambda z: sum(probs(z)) - 1, 0.0, 0.99)
        return float(probs(z)[0])
    raise ValueError(method)


def movement(lines: pd.DataFrame) -> pd.DataFrame:
    """First vs last snapshot of each main line: how far the line and the price moved during the day."""
    main = lines[lines["glavna"]].sort_values("snimka")
    keys = ["kladilnica", "nat_id", "igrac", "pazar", "strana"]
    g = main.groupby(keys, dropna=False)
    out = g.agg(prva=("snimka", "first"), posledna=("snimka", "last"), linija_prva=("linija", "first"),
                linija_posledna=("linija", "last"), kvota_prva=("kvota", "first"), kvota_posledna=("kvota", "last"),
                snimki=("snimka", "nunique")).reset_index()
    out["pomestuvanje"] = out["linija_posledna"] - out["linija_prva"]
    return out
