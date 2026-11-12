"""`migrate` and `stats` commands."""
from __future__ import annotations

import logging
import shutil
from datetime import date, datetime, timedelta

from core.config import ROOT
from store import db
from store.legacy_csv import read_legacy_csv
from store.migrate import load_corrections, load_legacy, migrate_rows, write_to_db


def load_legacy_raw(folder) -> list[dict]:
    rows = []
    for path in sorted(folder.glob("*.csv")):
        rows += read_legacy_csv(path)
    return migrate_rows(rows).picks
from store.stats import render_report

log = logging.getLogger(__name__)


def cmd_migrate(force: bool = False) -> int:
    if db.DB_PATH.exists():
        if not force:
            print(f"Архивата веќе постои ({db.DB_PATH}). Миграцијата е еднократна; "
                  f"со --force старата база се зачувува како резервна копија и се прави нова.")
            return 2
        backup = db.DB_PATH.with_name(f"izbori_backup_{datetime.now():%Y%m%d_%H%M%S}.db")
        shutil.move(db.DB_PATH, backup)
        print(f"Старата база е зачувана како {backup.name}")
    folder = ROOT / "data" / "archive"
    corrections = load_corrections(ROOT / "config" / "arhiva_korekcii.yaml")
    migrated = load_legacy(folder, corrections)
    con = db.connect()
    inserted = write_to_db(con, migrated)
    db.export_csv(con)
    dups = [p for p in migrated.picks if p["status"] == "DUPLIKAT"]
    print(f"Мигрирани записи: {inserted} (од нив DUPLIKAT: {len(dups)}) · белешки: {len(migrated.notes)} · "
          f"корекции: {len(migrated.corrections)} · технички поправки во изворот: {len(migrated.repairs)}")
    for r in migrated.repairs:
        print(f"  поправка: {r}")
    for c in migrated.corrections:
        print(f"  корекција: {c}")
    for d in dups:
        print(f"  DUPLIKAT: {d['izvor']} → се брои {d['duplikat_od']}")
    print()
    print(explain_vs_manual(folder, migrated))
    print(f"\nЗачувано: {db.DB_PATH} и {db.CSV_PATH}")
    return 0


def explain_vs_manual(folder, migrated) -> str:
    """Old hand counts (raw CSV, DA/NE as written) vs the cleaned archive, record by record."""
    raw = {p["izvor"]: p for p in load_legacy_raw(folder)}
    lines = ["СПОРЕДБА со рачните бројки (DA/NE како што беа запишани во CSV):"]
    for lg in ("nba", "euroleague", "eurocup"):
        old = [p for p in raw.values() if p["liga"] == lg and p["status"] in ("DA", "NE")]
        new = [p for p in migrated.picks if p["liga"] == lg and p["status"] in ("DA", "NE") and p["p_model"] is not None]
        if not old and not new:
            continue
        o_hits, n_hits = sum(p["status"] == "DA" for p in old), sum(p["status"] == "DA" for p in new)
        lines.append(f"  {lg}: рачно {o_hits}/{len(old)} → ново {n_hits}/{len(new)} (пред grade)")
        for p in migrated.picks:
            before = raw[p["izvor"]]["status"]
            if p["liga"] == lg and before in ("DA", "NE") and p["status"] != before:
                lines.append(f"    − {p['izvor']}: {p['igrac'] or p['natprevar']} {p['pazar']} {p['strana']} "
                             f"{p['linija']} — беше {before}, сега {p['status']}"
                             + (f" (се брои {p['duplikat_od']})" if p["duplikat_od"] else ""))
    return "\n".join(lines)


def cmd_stats(league: str | None = None, week: bool = False) -> int:
    if not db.DB_PATH.exists():
        print("Нема архива. Прво: python main.py migrate")
        return 2
    con = db.connect()
    since = (date.today() - timedelta(days=7)).isoformat() if week else None
    print(render_report(db.picks_frame(con), league=league, since=since))
    if week:
        from core.config import load_settings
        from pipeline.backtest_cmd import team_model_week_monitor
        for line in team_model_week_monitor(load_settings()):
            print(line)
    return 0


def cmd_izbori(datum: str | None = None, league: str | None = None, status: str | None = None) -> int:
    """List picks with their ids (needed for `uplata`)."""
    con = db.connect()
    df = db.picks_frame(con)
    if datum:
        df = df[df["datum"] == datum]
    if league:
        df = df[df["liga"] == league]
    if status:
        df = df[df["status"] == status.upper()]
    if df.empty:
        print("Нема избори за тој филтер.")
        return 0
    df = df.astype(object).where(df.notna(), None)
    for r in df.itertuples():
        who = r.igrac or r.natprevar or "?"
        paid = {1: f"уплатено {r.uplateno_iznos:g} ден." if r.uplateno_iznos else "уплатено",
                0: "не уплатено"}.get(r.uplateno, "уплата: ?")
        price = "" if r.kvota is None else f" @{r.kvota:g}" + ("" if r.kvota_proverena != 0 else " (непроверена)")
        line = "" if r.linija is None else f" {r.linija:g}"
        print(f"#{r.id:<4} {r.datum} {r.liga:<10} {who} — {r.pazar} {r.strana or ''}{line}{price} · {r.status} · {paid}")
    return 0


def cmd_uplata(pick_id: int, iznos: float | None, ne: bool = False) -> int:
    """Record whether a pick was actually placed (and the stake) — basis of "ROI уплатено"."""
    con = db.connect()
    row = con.execute("SELECT id, datum, igrac, natprevar, pazar, uplateno, uplateno_iznos FROM izbori WHERE id=?",
                      [pick_id]).fetchone()
    if row is None:
        print(f"Нема избор #{pick_id}. Види: python main.py izbori --datum YYYY-MM-DD")
        return 2
    if not ne and (iznos is None or iznos <= 0):
        print("Внеси износ: --iznos 200 (или --ne ако не е уплатено)")
        return 2
    new = (0, None) if ne else (1, iznos)
    con.execute("UPDATE izbori SET uplateno=?, uplateno_iznos=? WHERE id=?", [*new, pick_id])
    con.execute("INSERT INTO log_izmeni (izbor_id, akcija, detali) VALUES (?, 'UPLATA', ?)",
                [pick_id, f"{row['uplateno']}/{row['uplateno_iznos']} → {new[0]}/{new[1]}"])
    con.commit()
    db.export_csv(con)
    who = row["igrac"] or row["natprevar"]
    print(f"#{pick_id} {row['datum']} {who} ({row['pazar']}): " + ("НЕ е уплатено" if ne else f"уплатено {iznos:g} ден."))
    return 0
