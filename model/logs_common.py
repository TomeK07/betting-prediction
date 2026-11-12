"""Helpers shared by all leagues' game logs.

Common log schema (team and player rows): season, game_code, game_date (official league date:
local for Europe, US Eastern for NBA), date_utc, date_local (Skopje), team, opp, home, and box
stats pts, fgm, fga, fgm2, fga2, fgm3, fga3, ftm, fta, oreb, dreb, reb, ast, stl, tov, blk, pf.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd


def parse_minutes(value: Optional[str]) -> Optional[float]:
    """'14:08' -> 14.1333; 'DNP' -> 0.0; missing/empty -> None (never guessed)."""
    if value is None:
        return None
    text = str(value).strip()
    if text.upper() == "DNP":
        return 0.0
    if ":" not in text:
        return None
    mins, secs = text.split(":", 1)
    try:
        return int(mins) + float(secs) / 60.0
    except ValueError:
        return None


SUFFIX_RE = re.compile(r"(?:\s*,\s*|\s+)(Jr|Sr|II|III|IV)\.?\s*$", re.I)


def clean_suffix(name: str) -> str:
    """'Udeh, Jr.' / 'Udeh Jr' → 'Udeh Jr.'; 'Murphy III' stays (2.11: the comma broke roster matching for CLE)."""
    name = str(name or "").strip()
    m = SUFFIX_RE.search(name)
    if not m:
        return name
    suf = m.group(1)
    suf = suf.upper() if suf.upper().strip("IV") == "" else suf.capitalize() + "."
    return f"{name[:m.start()].rstrip(' ,')} {suf}"


def disambiguate_labels(*frames: pd.DataFrame) -> None:
    """Rule 2 label 'Williams J. (OKC)' must name ONE player: two players of a team with the same surname + initial
    (Jalen / Jaylin Williams, 2.11) get a longer first-name prefix: 'Williams Jal. (OKC)', 'Williams Jay. (OKC)'.
    Same player_code → same label in every frame (in place)."""
    parts = [f[["player_code", "player", "label"]] for f in frames if len(f) and "label" in f]
    if not parts:
        return
    allp = pd.concat(parts).dropna(subset=["player_code"]).drop_duplicates("player_code")
    new = {}
    for lbl, g in allp.groupby("label"):
        if len(g) < 2:
            continue
        firsts = {c: (p.split(",", 1)[1] if "," in p and not SUFFIX_RE.search(p) else p.split(" ")[0]).strip()
                  for c, p in zip(g["player_code"], g["player"].fillna(""))}
        if len({f.lower() for f in firsts.values()}) < len(firsts):   # same person under two codes (EL data) → keep
            continue
        head, team = lbl.rsplit(" (", 1)
        last = head[:-3] if re.search(r" \w\.$", head) else head
        for n in range(2, 12):
            pre = {c: f[:n] for c, f in firsts.items()}
            if len(set(pre.values())) == len(pre):
                break
        for c, f in pre.items():
            new[c] = f"{last} {f}. ({team}"
    for f in frames:
        if len(f) and "label" in f:
            m = f["player_code"].map(new)
            f["label"] = m.where(m.notna(), f["label"])


def make_label(last: str, first: str, team: str) -> str:
    """Rule 2: surname + initial + team, e.g. 'Motley J. (RED)'."""
    initial = f" {first.strip()[0]}." if first and first.strip() else ""
    return f"{last.strip()}{initial} ({team})"


def local_datetime(iso_utc: str, tz: str) -> datetime:
    dt = datetime.fromisoformat(iso_utc.replace("Z", "+00:00"))
    return dt.astimezone(ZoneInfo(tz))


def add_shot_totals(row: dict, prefix: str = "") -> None:
    """Fill fgm/fga from 2s+3s, or 2s from fgm/fga−3s, whichever the source lacks."""
    p = prefix
    if row.get(f"{p}fgm") is None and row.get(f"{p}fgm2") is not None:
        row[f"{p}fgm"] = row[f"{p}fgm2"] + row[f"{p}fgm3"]
        row[f"{p}fga"] = row[f"{p}fga2"] + row[f"{p}fga3"]
    elif row.get(f"{p}fgm2") is None and row.get(f"{p}fgm") is not None:
        row[f"{p}fgm2"] = row[f"{p}fgm"] - row[f"{p}fgm3"]
        row[f"{p}fga2"] = row[f"{p}fga"] - row[f"{p}fga3"]


def sort_logs(df: pd.DataFrame) -> pd.DataFrame:
    """Chronological order: date, then game code (rule 1, error 27.10)."""
    keys = [c for c in ("date_utc", "game_code", "team", "player_code") if c in df.columns]
    return df.sort_values(keys).reset_index(drop=True)


def last_n_games(logs: pd.DataFrame, n: int, before: Optional[str] = None,
                 include_dnp: bool = False) -> pd.DataFrame:
    """Most recent n games by DATE (newest first). `before` = ISO game_date, exclusive."""
    df = logs
    if before is not None:
        df = df[df["game_date"] < before]
    if not include_dnp and "dnp" in df.columns:
        df = df[~df["dnp"]]
    return df.sort_values(["date_utc", "game_code"], ascending=False).head(n)


def percent_columns_to_numbers(df: pd.DataFrame) -> pd.DataFrame:
    """'53.7%' -> 53.7 so the columns are numeric."""
    for col in df.columns:
        if (df[col].dtype == object or pd.api.types.is_string_dtype(df[col])) \
                and df[col].astype(str).str.endswith("%").any():
            df[col] = pd.to_numeric(df[col].astype(str).str.rstrip("%"), errors="coerce")
    return df
