"""SQLite picks archive (BUILD_PROMPT §6): archive/izbori.db + CSV export archive/izbori.csv."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from core.config import ROOT

DB_PATH = ROOT / "archive" / "izbori.db"
CSV_PATH = ROOT / "archive" / "izbori.csv"

# DUPLIKAT: same pick recorded more than once — kept for the record, never counted (decision 29.10)
STATUSES = ("CEKA", "DA", "NE", "VOID", "ODBIEN", "POVLECEN", "DUPLIKAT")
ERROR_TYPES = ("VARIJANSA", "PODATOK", "VEST", "MODEL", "PRAVILO")

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS izbori (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    datum TEXT NOT NULL,              -- ISO date as written by the analyst (Skopje)
    vreme TEXT,
    liga TEXT NOT NULL,               -- euroleague | eurocup | nba
    natprevar TEXT,
    tim TEXT,
    protivnik TEXT,
    igrac TEXT,
    pazar TEXT NOT NULL,
    linija REAL,
    strana TEXT,                      -- nad | pod | team code (handicap/1X2) | 1 | X | 2 | 1X | X2
    kvota REAL,
    kvota_zatvaracka REAL,
    p_model REAL,                     -- 0..1, locked once written (never changed after the game)
    p_kalibriran REAL,
    p_pazar REAL,
    prednost_pp REAL,
    ev REAL,
    ulog_pct REAL,
    pravila_primeneti TEXT,
    status TEXT NOT NULL CHECK (status IN {STATUSES}),
    rezultat_broj REAL,
    rezultat_tekst TEXT,
    minuti TEXT,
    tip_greska TEXT CHECK (tip_greska IS NULL OR tip_greska IN {ERROR_TYPES}),
    clv REAL,
    vlezovi_json TEXT,
    ekspert TEXT,
    zabeleska TEXT,
    verzija_model TEXT,
    izvor TEXT,
    legacy_json TEXT,                 -- original legacy CSV row, verbatim
    duplikat_od TEXT,                 -- izvor of the counted record when status = DUPLIKAT
    graded_at TEXT,
    graded_by TEXT
);
-- BUILD_PROMPT §6 unique key, for every record that counts (duplicates are kept but excluded)
CREATE UNIQUE INDEX IF NOT EXISTS ux_izbori_kluc
    ON izbori (datum, IFNULL(natprevar, ''), IFNULL(igrac, ''), pazar, IFNULL(linija, -999), IFNULL(strana, ''))
    WHERE status != 'DUPLIKAT';
CREATE TABLE IF NOT EXISTS beleski (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    datum TEXT,
    vreme TEXT,
    liga TEXT,
    naslov TEXT,
    tekst TEXT,
    izvor TEXT
);
CREATE TABLE IF NOT EXISTS log_izmeni (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vreme TEXT DEFAULT (datetime('now')),
    izbor_id INTEGER,
    akcija TEXT,
    detali TEXT
);
"""


# Columns added after the first migration (29.10.2026). ALTER TABLE keeps existing data.
ADDED_COLUMNS = {
    "kvota_proverena": "INTEGER",   # 1 = real bookmaker price; 0 = orientational/unverified → not in ROI
    "uplateno": "INTEGER",          # 1 = Tome actually placed it, 0 = no, NULL = unknown (legacy)
    "uplateno_iznos": "REAL",       # stake actually placed (денари)
    "game_code": "TEXT",            # official game id (new picks, Phase 7) → grading without date guessing
    # Tome 31.10: a REJECTED pick is graded too (status stays ODBIEN) → "do the filters add value?"
    "ishod": "TEXT",                # DA / NE / VOID from the box score, for every pick (chosen or rejected)
    "odbien_od": "TEXT",            # МОДЕЛ / ФИЛТЕР / СОСТАВ / РАЧНО — who rejected it (first reason that applies)
}


def rejected_by(pravila: str | None) -> str:
    """The rejection reasons written by model/picks.py → who rejected: the model's own edge (EV), a filter rule,
    the lineup (news), or Tome by hand (no automatic reason)."""
    p = pravila or ""
    if "EV_ПОД_4%" in p or "ЛИНИЈА_НЕ_ПОСТОИ" in p:
        return "МОДЕЛ"
    if any(k in p for k in ("ПРОВЕРИ_ПОДАТОЦИ", "МИНУТИ <", "ПРВ_СТРЕЛЕЦ", "ПОЕНИ_НАД", "ПОЕНИ НАД", "САМО_ХАРТИЈА")):
        return "ФИЛТЕР"
    if "ЗАВИСИ_ОД_СОСТАВ" in p or "NEPOTVRDEN" in p:
        return "СОСТАВ"
    return "РАЧНО"


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    _ensure_columns(con)
    return con


def _ensure_columns(con: sqlite3.Connection) -> None:
    existing = {r["name"] for r in con.execute("PRAGMA table_info(izbori)")}
    for name, sql_type in ADDED_COLUMNS.items():
        if name in existing:
            continue
        con.execute(f"ALTER TABLE izbori ADD COLUMN {name} {sql_type}")
        if name == "kvota_proverena":
            from store.migrate import kvota_verified
            rows = con.execute("SELECT id, zabeleska, kvota FROM izbori").fetchall()
            for r in rows:
                con.execute("UPDATE izbori SET kvota_proverena=? WHERE id=?",
                            [None if r["kvota"] is None else int(kvota_verified(r["zabeleska"])), r["id"]])
        if name == "ishod":                  # graded picks already have their result in `status`
            con.execute("UPDATE izbori SET ishod=status WHERE status IN ('DA','NE','VOID')")
        con.execute("INSERT INTO log_izmeni (akcija, detali) VALUES ('SEMA', ?)", [f"додадена колона {name}"])
    # who rejected: derived from the stored reasons (never changes the decision itself)
    for r in con.execute("SELECT id, pravila_primeneti FROM izbori WHERE status='ODBIEN' AND odbien_od IS NULL").fetchall():
        con.execute("UPDATE izbori SET odbien_od=? WHERE id=?", [rejected_by(r["pravila_primeneti"]), r["id"]])
    con.commit()


def picks_frame(con: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM izbori ORDER BY datum, vreme, id", con)


def export_csv(con: sqlite3.Connection, path: Path = CSV_PATH) -> None:
    picks_frame(con).to_csv(path, index=False, encoding="utf-8")
