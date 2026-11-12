"""`python main.py povredi` — Phase 6: absences, NEW vs OLD, effect on teammates (Macedonian report)."""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from core.config import ROOT, Settings
from model import absences as ab
from model.injury_nba import parse_pdf, split_name
from model.player_model import params_from_config
from pipeline.fetch import fetch_league
from scrapers.http_client import SourceError

log = logging.getLogger(__name__)
VESTI_DIR = ROOT / "data" / "vesti"
T1_STATUSES = {"OUT", "DOUBTFUL", "QUESTIONABLE", "PROBABLE", "AVAILABLE", "RETURN", "NEPOTVRDEN"}   # NEPOTVRDEN: manual only


def load_vesti(day: date, league: str) -> list[dict]:
    """Manual / T1 news: data/vesti/<date>.yaml — list of {liga, tim, igrac, status, izvor_url, citat, objaveno}.
    Entries without a source are ignored (LLM_ISTRAZUVACI §3: no izvor_url + citat = rejected)."""
    path = VESTI_DIR / f"{day.isoformat()}.yaml"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        items = yaml.safe_load(f) or []
    out = []
    for it in items:
        if it.get("liga") != league:
            continue
        if not it.get("izvor_url") or not it.get("citat") or str(it.get("status", "")).upper() not in T1_STATUSES:
            log.warning("Вест без извор/цитат или со непознат статус — игнорирана: %s", it)
            continue
        out.append(it)
    return out


def _load(settings: Settings, league: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    folder = settings.paths.processed / league / settings.league(league).season
    logs = pd.read_csv(folder / "player_logs.csv", dtype={"player_code": str, "game_code": str})
    roster = pd.read_csv(folder / "roster.csv", dtype={"player_code": str})
    schedule = pd.read_csv(folder / "schedule.csv", dtype={"game_code": str})
    return logs, roster, schedule


def league_absences(settings: Settings, league: str, day: date) -> tuple[pd.DataFrame, list[str]]:
    """All known absences for teams playing on `day`, classified NEW/OLD."""
    logs, roster, schedule = _load(settings, league)
    games = schedule[schedule["game_date"] == day.isoformat()]
    teams = set(games["home"]) | set(games["away"])
    notes, rows = [], []
    if league == "nba":
        from scrapers.nba_injury import latest_report
        path = latest_report(settings, day)
        if path is None:
            notes.append("NBA injury report за денес уште не е објавен.")
        else:
            rep = parse_pdf(path)
            notes.append(f"Официјален NBA injury report: {rep.attrs.get('report_time')} ET ({Path(path).name}).")
            codes = ab.team_code_map(schedule)
            pending = rep[rep["status"] == "NOT_SUBMITTED"]["team"].map(lambda n: codes.get(ab._norm(n), n))
            if len(pending):
                notes.append("Уште НЕ предале извештај: " + ", ".join(sorted(set(pending))) + ".")
            notes.append("__pending__:" + ",".join(sorted(set(pending))))
            for r in rep[rep["player"].notna()].itertuples():
                team = codes.get(ab._norm(r.team))
                if team not in teams:
                    continue
                last, first = split_name(r.player)
                code, note = ab.match_player(last, first, team, roster)
                info = ab.classify(code, team, logs, day.isoformat()) if code else \
                    {"novo": None, "tip": "НЕПОЗНАТ ИГРАЧ", "minuti_sezona": None, "natprevari": 0}
                rows.append({"team": team, "player_code": code, "label": f"{first} {last} ({team})",
                             "status": r.status.upper(), "reason": r.reason, "izvor": "NBA injury report",
                             "zabeleska": note, **info})
    else:
        for team in sorted(teams):
            df = ab.last_game_absences(logs, roster, team, day.isoformat(), league)
            if len(df):
                df["izvor"] = "box score (последен натпревар)"
                df["zabeleska"] = "статус за вечерва непотврден — провери вести"
                rows += df.to_dict("records")
        for v in load_vesti(day, league):
            team = ab.team_code_map(schedule).get(ab._norm(v["tim"]), v["tim"])
            last, first = split_name(v["igrac"])
            code, note = ab.match_player(last, first, team, roster)
            info = ab.classify(code, team, logs, day.isoformat()) if code else \
                {"novo": None, "tip": "НЕПОЗНАТ ИГРАЧ", "minuti_sezona": None, "natprevari": 0}
            rows = [r for r in rows if r.get("player_code") != code]     # news overrides box-score guess
            rows.append({"team": team, "player_code": code, "label": f"{v['igrac']} ({team})",
                         "status": str(v["status"]).upper(), "reason": v.get("prichina", ""),
                         "izvor": v["izvor_url"], "zabeleska": f"„{v['citat']}“ ({v.get('objaveno', '?')}) {note}".strip(),
                         **info})
    return pd.DataFrame(rows), notes


def injury_report(settings: Settings, league: str, day: date) -> Optional[str]:
    cfg = settings.league(league)
    try:
        fetch_league(settings, league, day)
    except SourceError as exc:
        log.error("%s: податоците не се освежени — %s", league, exc)
    logs, roster, schedule = _load(settings, league)
    games = schedule[schedule["game_date"] == day.isoformat()]
    if games.empty:
        return None
    absent, notes = league_absences(settings, league, day)
    unknown = absent[absent["tip"] == "НЕПОЗНАТ ИГРАЧ"] if len(absent) else absent
    if league == "nba" and len(unknown):                  # 2.11: not on our roster → refresh it once, match again
        from pipeline.fetch import refresh_nba_roster
        try:
            n_new = refresh_nba_roster(settings, league, day)
        except Exception as exc:                          # source down → report as is, never guessed
            n_new = None
            log.error("составите не се освежени: %s", exc)
        logs, roster, schedule = _load(settings, league)
        absent, notes = league_absences(settings, league, day)
        left = int((absent["tip"] == "НЕПОЗНАТ ИГРАЧ").sum()) if len(absent) else 0
        notes.insert(0, f"Состави освежени автоматски ({len(unknown)} играчи ги немаше): "
                        f"{'нови во составот ' + str(n_new) if n_new is not None else 'изворот не одговара'} · "
                        f"уште непознати {left}.")
    pending = set()
    for n in [n for n in notes if n.startswith("__pending__:")]:
        pending |= set(filter(None, n.split(":", 1)[1].split(",")))
    notes = [n for n in notes if not n.startswith("__pending__:")]
    params = params_from_config(cfg.player_model)
    lines = [f"# Отсуства и ефект — {cfg.name_mk} · {day}", "", *[f"- {n}" for n in notes],
             "- НОВО = играл во барем еден од последните 3 натпревари на тимот (носи предност); "
             "СТАРО = веќе во просеците (правило 3). Ефект = M2 минути без отсутните наспроти сите.",
             "- " + ("NBA: Questionable → **ЗАВИСИ_ОД_СОСТАВ** (не во главен тикет)." if league == "nba"
                     else "Евролига/Еврокуп: questionable = **игра** (грешка 27.10)."), ""]
    for g in games.itertuples():
        lines += [f"## {g.home} – {g.away} ({g.time_local})", ""]
        for team in (g.home, g.away):
            a = absent[absent["team"] == team] if len(absent) else pd.DataFrame()
            if a.empty:
                msg = ("⏳ уште НЕ предал извештај за повреди — непознато (пушти повторно подоцна)"
                       if team in pending else "нема познати отсуства")
                lines += [f"**{team}:** {msg}.", ""]
                continue
            lines += [f"**{team}:**", "", "| Играч | Статус | Тип | Мин. (сез.) | Причина / извор |", "|---|---|---|---|---|"]
            for r in a.sort_values("minuti_sezona", ascending=False, na_position="last").itertuples():
                st = r.status
                verdict = ab.plays(st, league)
                st_txt = st + (" → " + ab.DEPENDS if verdict is None else "")
                mins = "—" if r.minuti_sezona is None or pd.isna(r.minuti_sezona) else f"{r.minuti_sezona:.1f}"
                extra = f" — {r.zabeleska}" if isinstance(r.zabeleska, str) and r.zabeleska else ""
                lines.append(f"| {r.label} | {st_txt} | **{r.tip}** | {mins} | {r.reason} · {r.izvor}{extra} |")
            out_codes = {c for c, s in zip(a["player_code"], a["status"])
                         if c and ab.plays(s, league) is False}
            if league != "nba":                      # EL/EC: missed last game but no news → assume out, flagged
                out_codes |= {c for c, s in zip(a["player_code"], a["status"]) if s == "ПРОПУШТИЛ ПОСЛЕДЕН"}
            old = {c for c, t in zip(a["player_code"], a["tip"]) if c in out_codes and t == "СТАРО"}
            eff = ab.pregame_minutes(logs, roster, league, team, day.isoformat(), out_codes, params, old_out=old)
            gain = eff[(~eff["out"]) & (eff["efekt_min"] >= 1.5)] if len(eff) else eff
            if len(gain):
                lines += ["", "Ефект (M2): кој добива минути — само од НОВИТЕ отсуства (старите се веќе во просеците)", "",
                          "| Играч | Мин. без отсутните | Δ мин | Δ поени | Δ ast | Δ скок |", "|---|---|---|---|---|---|"]
                for r in gain.itertuples():
                    lines.append(f"| {r.label} | {r.min_bez:.1f} | +{r.efekt_min:.1f} | {r.efekt_pts:+.1f} | "
                                 f"{r.efekt_ast:+.1f} | {r.efekt_reb:+.1f} |")
            lines.append("")
    lines += ["*Ефектот е проекција на минутите (M2), не веројатност; пазарот (Mozzart) влегува дури во Фаза 7.*"]
    return "\n".join(lines)


def cmd_povredi(settings: Settings, league: str, day: date) -> int:
    leagues = ["euroleague", "eurocup", "nba"] if league == "all" else [league]
    settings.paths.reports.mkdir(parents=True, exist_ok=True)
    for lg in leagues:
        text = injury_report(settings, lg, day)
        if text is None:
            print(f"{settings.league(lg).name_mk}: нема натпревари на {day}.")
            continue
        path = settings.paths.reports / f"povredi_{day}_{lg}.md"
        path.write_text(text, encoding="utf-8")
        print(text)
        print(f"Извештај: {path}\n")
    return 0
