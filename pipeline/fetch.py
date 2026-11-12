"""`python main.py fetch` — download, cache and turn into date-sorted logs."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

from core.cache import RawCache
from core.config import Settings
from model import euroleague_logs as el
from model import nba_logs
from model.logs_common import disambiguate_labels
from scrapers.euroleague import STAT_TABLES as EL_STAT_TABLES, EuroleagueScraper
from scrapers.http_client import HttpClient
from scrapers.nba import STAT_TABLES as NBA_STAT_TABLES, NbaScraper

log = logging.getLogger(__name__)


@dataclass
class FetchSummary:
    league: str
    games_finished: int
    boxscores: int
    boxscores_missing: list
    team_log_rows: int
    player_log_rows: int
    games_on_day: list[str]
    out_dir: str


def fetch_league(settings: Settings, league_key: str, day: date, refresh: bool = False,
                 refresh_box: bool = False, season: Optional[str] = None) -> FetchSummary:
    """`season` = a past season (e.g. E2025, U2025, 2025-26) for backtests; default = current."""
    current = settings.league(league_key)
    league = current.model_copy(update={"season": season}) if season else current
    history = league.season != current.season
    tz = settings.timezone
    today = datetime.now(ZoneInfo(tz)).date()
    if day != today and not history:
        log.warning("Датумот %s не е денес (%s): сезонските статистики и составите се моментална "
                    "слика, не од тој ден. Box score логовите се филтрираат до %s.", day, today, day)

    cache = RawCache(settings.paths.raw)
    if league.source == "euroleague":
        collected = _collect_euroleague(settings, league_key, league, day, cache, refresh, refresh_box)
    elif league.source == "nba" and history:
        collected = _collect_nba_history(settings, league_key, league, cache, refresh)
    elif league.source == "nba":
        collected = _collect_nba(settings, league_key, league, day, cache, refresh, refresh_box)
    else:
        raise NotImplementedError(f"Непознат извор '{league.source}'")
    schedule, tables, roster, boxscores, missing, build = collected

    team_logs, player_logs = build(schedule, boxscores)
    finished = (team_logs["game_code"].nunique() if len(team_logs) else 0) + len(missing)
    out = settings.paths.processed / league_key / league.season
    _write_outputs(out, schedule, team_logs, player_logs, roster, tables, {
        "fetched_at": datetime.now(ZoneInfo(tz)).isoformat(timespec="seconds"),
        "as_of_date": day.isoformat(), "season": league.season,
        "games_finished": finished, "boxscores_missing": missing})

    on_day = schedule[schedule["game_date"] == day.isoformat()]
    games_on_day = [_describe_game(r, day) for r in on_day.itertuples()]
    return FetchSummary(league_key, finished, finished - len(missing), missing,
                        len(team_logs), len(player_logs), games_on_day, str(out))


def _finished_games(schedule: pd.DataFrame, day: date) -> pd.DataFrame:
    return schedule[(schedule["status"] == "result") & (schedule["game_date"] <= day.isoformat())]


def _download_boxscores(name_mk: str, codes, fetch_one) -> tuple[dict, list]:
    boxscores, missing = {}, []
    for code in codes:
        box = fetch_one(code)
        if box is None or box.get("Live"):
            missing.append(code)
        else:
            boxscores[code] = box
    if missing:
        log.warning("%s: недостасува box score за натпревари %s — исклучени од логовите (не се погодуваат)",
                    name_mk, missing)
    return boxscores, missing


def _collect_euroleague(settings, key, league, day, cache, refresh, refresh_box):
    scraper = EuroleagueScraper(key, league, settings.euroleague_api, HttpClient(settings.http), cache)
    log.info("%s: распоред…", league.name_mk)
    schedule = el.schedule_frame(scraper.schedule(day, refresh), settings.timezone)
    log.info("%s: сезонски статистики и состави…", league.name_mk)
    tables = {f"{scope}_{kind}": el.stat_table_frame(scraper.stat_table(day, scope, kind, refresh), scope)
              for scope, kind in EL_STAT_TABLES}
    roster = el.roster_frame(scraper.people(day, refresh))
    finished = _finished_games(schedule, day)
    log.info("%s: box score за %d завршени натпревари (кешираните не се влечат пак)…",
             league.name_mk, len(finished))
    boxscores, missing = _download_boxscores(
        league.name_mk, [int(c) for c in finished["game_code"]],
        lambda c: scraper.boxscore(c, refresh=refresh_box))
    return schedule, tables, roster, boxscores, missing, el.build_logs


def _collect_nba_history(settings, key, league, cache, refresh):
    """Past NBA season: the two season-wide LeagueGameLog tables (team + player) instead of
    1230 per-game box scores. Player minutes there are rounded and starters/DNP are missing —
    fine for the team model and backtests, not for minute models of the current season."""
    scraper = NbaScraper(key, league, settings.nba_api, settings.http, cache)
    log.info("%s %s: сезонски логови (LeagueGameLog)…", league.name_mk, league.season)
    team_payload = scraper.league_game_log("T", refresh)
    player_payload = scraper.league_game_log("P", refresh)
    schedule, team_logs, player_logs = nba_logs.from_league_game_log(team_payload, player_payload)
    roster = nba_logs.roster_frame(scraper.player_index(date.today(), refresh, historical=True))
    roster = roster[roster["player_code"].isin(set(player_logs["player_code"]))]
    prebuilt = (team_logs, player_logs)
    return schedule, {}, roster, {"_all": None}, [], lambda s, b: prebuilt


def _collect_nba(settings, key, league, day, cache, refresh, refresh_box):
    scraper = NbaScraper(key, league, settings.nba_api, settings.http, cache)
    log.info("%s: распоред…", league.name_mk)
    schedule = nba_logs.schedule_frame(scraper.schedule(day, refresh), settings.timezone,
                                       settings.nba_api.game_types)
    codes = nba_logs.team_id_map(schedule)
    log.info("%s: сезонски статистики и состави…", league.name_mk)
    tables = {f"{scope}_{measure.lower().replace(' ', '_')}":
              nba_logs.stat_table_frame(scraper.stat_table(day, scope, measure, refresh), scope, codes)
              for scope, measure in NBA_STAT_TABLES}
    roster = nba_logs.roster_frame(scraper.player_index(day, refresh))
    finished = _finished_games(schedule, day)
    log.info("%s: box score за %d завршени натпревари (кешираните не се влечат пак)…",
             league.name_mk, len(finished))
    boxscores, missing = _download_boxscores(
        league.name_mk, list(finished["game_code"]),
        lambda c: scraper.boxscore(c, refresh=refresh_box))
    return schedule, tables, roster, boxscores, missing, nba_logs.build_logs


def refresh_nba_roster(settings: Settings, league_key: str, day: date) -> int:
    """2.11: a player on the official injury report who is not on our roster → fresh PlayerIndex (signings, two-way
    call-ups) → roster.csv rewritten (labels kept consistent with the logs). Returns the number of new players."""
    league = settings.league(league_key)
    out = settings.paths.processed / league_key / league.season
    scraper = NbaScraper(league_key, league, settings.nba_api, settings.http, RawCache(settings.paths.raw))
    roster = nba_logs.roster_frame(scraper.player_index(day, refresh=True))
    old = pd.read_csv(out / "roster.csv", dtype={"player_code": str})
    logs = pd.read_csv(out / "player_logs.csv", dtype={"player_code": str, "game_code": str})
    disambiguate_labels(roster, logs)
    roster.to_csv(out / "roster.csv", index=False, encoding="utf-8")
    return len(set(roster["player_code"]) - set(old["player_code"]))


def _write_outputs(out: Path, schedule, team_logs, player_logs, roster, tables, meta: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    disambiguate_labels(roster, player_logs)          # 2.11: Jalen / Jaylin Williams were both 'Williams J. (OKC)'
    schedule.to_csv(out / "schedule.csv", index=False, encoding="utf-8")
    team_logs.to_csv(out / "team_logs.csv", index=False, encoding="utf-8")
    player_logs.to_csv(out / "player_logs.csv", index=False, encoding="utf-8")
    roster.to_csv(out / "roster.csv", index=False, encoding="utf-8")
    for name, df in tables.items():
        df.to_csv(out / f"season_{name}.csv", index=False, encoding="utf-8")
    with open(out / "_fetch_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def _describe_game(r, day: date) -> str:
    when = r.time_local if r.date_local == day.isoformat() else f"{r.date_local[8:10]}.{r.date_local[5:7]} {r.time_local}"
    label = getattr(r, "label", None)
    extra = f"коло {r.round}" if pd.notna(r.round) else (label if pd.notna(label) and label else r.phase)
    return f"{when} {r.home} – {r.away} ({extra})"
