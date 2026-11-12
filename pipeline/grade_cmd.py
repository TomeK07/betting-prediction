"""`python main.py grade` — fill results of waiting picks from official box scores."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

from core.config import ROOT, Settings
from pipeline.clv import closing_price
from pipeline.fetch import fetch_league
from scrapers.http_client import SourceError
from store import db
from store.grade import LeagueData, grade_pick, norm

log = logging.getLogger(__name__)
GRADER = "grade/boxscore-v1"


def load_league_data(settings: Settings, league: str) -> LeagueData:
    folder = settings.paths.processed / league / settings.league(league).season
    read = lambda name, **kw: pd.read_csv(folder / name, dtype={"game_code": str, "player_code": str}, **kw)
    with open(ROOT / "config" / "timovi.yaml", encoding="utf-8") as f:
        aliases = {norm(k): v for k, v in ((yaml.safe_load(f) or {}).get(league) or {}).items()}
    return LeagueData(read("schedule.csv"), read("team_logs.csv"), read("player_logs.csv"), aliases)


def cmd_grade(settings: Settings, only_date: Optional[str] = None, dry_run: bool = False) -> int:
    if not db.DB_PATH.exists():
        print("Нема архива. Прво: python main.py migrate")
        return 2
    con = db.connect()
    picks = db.picks_frame(con)
    # Tome 31.10: rejected picks get their outcome too (`ishod`) — the status stays ODBIEN, never a decision change
    waiting = picks[(picks["status"] == "CEKA") | ((picks["status"] == "ODBIEN") & picks["ishod"].isna())]
    if only_date:
        waiting = waiting[waiting["datum"] == only_date]
    if waiting.empty:
        print("Нема избори што чекаат резултат.")
        return 0

    today = datetime.now(ZoneInfo(settings.timezone)).date()
    data: dict[str, LeagueData] = {}
    for league in sorted(waiting["liga"].unique()):
        try:
            fetch_league(settings, league, today)
        except SourceError as exc:
            log.error("%s: не можам да ги освежам податоците — %s (оценувам од постоечките)", league, exc)
        data[league] = load_league_data(settings, league)

    now = datetime.now(ZoneInfo(settings.timezone)).isoformat(timespec="seconds")
    counts: dict[str, int] = {}
    n_rej = int((waiting["status"] == "ODBIEN").sum())
    print(f"{'(ПРОБА — ништо не се запишува) ' if dry_run else ''}Оценување на {len(waiting) - n_rej} избори "
          f"+ {n_rej} одбиени (само исход, статусот останува ODBIEN):")
    rej_counts: dict[str, int] = {}
    for pick in waiting.to_dict("records"):
        pick = {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in pick.items()}
        res = grade_pick(pick, data[pick["liga"]])
        rejected = pick["status"] == "ODBIEN"
        who = pick["igrac"] or pick["natprevar"]
        line = f"{pick['pazar']} {pick['strana'] or ''} {'' if pick['linija'] is None else pick['linija']}".strip()
        if rejected:
            rej_counts[res.status] = rej_counts.get(res.status, 0) + 1
            if dry_run or res.status == "CEKA":
                continue
            con.execute("""UPDATE izbori SET ishod=?, rezultat_broj=?, rezultat_tekst=?, minuti=?, graded_at=?, graded_by=?
                           WHERE id=? AND status='ODBIEN' AND ishod IS NULL""",
                        [res.status, res.value, res.reason, None if res.minutes is None else f"{res.minutes:.1f}",
                         now, GRADER, pick["id"]])
            con.execute("INSERT INTO log_izmeni (izbor_id, akcija, detali) VALUES (?, 'GRADE_ODBIEN', ?)",
                        [pick["id"], json.dumps({"ishod": res.status, "reason": res.reason}, ensure_ascii=False)])
            continue
        counts[res.status] = counts.get(res.status, 0) + 1
        icon = {"DA": "✅", "NE": "❌", "VOID": "⚪", "CEKA": "⏳"}[res.status]
        print(f"  {icon} {pick['datum']} {pick['liga']:<10} {who} — {line} → {res.status}: {res.reason}")
        if dry_run or res.status == "CEKA":
            continue
        vlezovi = json.loads(pick["vlezovi_json"]) if pick.get("vlezovi_json") else {}
        if pick.get("minuti") and "minuti_pred_natprevar" not in vlezovi:
            vlezovi["minuti_pred_natprevar"] = pick["minuti"]       # analyst's pre-game estimate
        close = closing_price(settings, pick) if (pick.get("verzija_model") or "").startswith("py") else None
        k_close = close[0] if close else None
        clv = (pick["kvota"] / k_close - 1) if (k_close and pick.get("kvota")) else None
        if close:
            vlezovi["zatvaracka_snimka"] = close[1]
        con.execute(
            """UPDATE izbori SET status=?, ishod=?, rezultat_broj=?, rezultat_tekst=?, minuti=?, vlezovi_json=?,
                   graded_at=?, graded_by=?, kvota_zatvaracka=?, clv=? WHERE id=? AND status='CEKA'""",
            [res.status, res.status, res.value, res.reason,
             None if res.minutes is None else f"{res.minutes:.1f}",
             json.dumps(vlezovi, ensure_ascii=False) if vlezovi else None, now, GRADER, k_close, clv, pick["id"]])
        con.execute("INSERT INTO log_izmeni (izbor_id, akcija, detali) VALUES (?, 'GRADE', ?)",
                    [pick["id"], json.dumps({"status": res.status, "reason": res.reason,
                                             "game_code": res.game_code}, ensure_ascii=False)])
    if not dry_run:
        con.commit()
        db.export_csv(con)
    print("Вкупно: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
          + ("" if not counts.get("CEKA") else " (⏳ = останува CEKA, види причина)"))
    if rej_counts:
        print("Одбиени (исход, не одлука): " + ", ".join(f"{k} {v}" for k, v in sorted(rej_counts.items())))
    return 0
