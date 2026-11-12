"""NBA via nba_api (official stats.nba.com, free) — BUILD_PROMPT §2.3."""
from __future__ import annotations

import logging
import time
from datetime import date
from typing import Any, Callable, Optional

from nba_api.stats import endpoints as nba

from core.cache import RawCache
from core.config import HttpSettings, LeagueSettings, NbaApi
from scrapers.http_client import SourceError

log = logging.getLogger(__name__)

# (scope, measure type) pairs downloaded on every fetch
STAT_TABLES = [
    ("teams", "Base"),
    ("teams", "Advanced"),
    ("teams", "Four Factors"),
    ("teams", "Opponent"),
    ("players", "Base"),
    ("players", "Advanced"),
]


class NbaScraper:
    def __init__(self, league_key: str, league: LeagueSettings, api: NbaApi, http: HttpSettings,
                 cache: RawCache, sleep: Callable[[float], None] = time.sleep):
        self.key = league_key
        self.season = league.season
        self.api = api
        self.http = http
        self.cache = cache
        self._sleep = sleep
        self._last_request = 0.0

    # --- snapshots ---

    def schedule(self, day: date, refresh: bool = False) -> dict:
        return self._snapshot(day, "schedule", refresh, nba.ScheduleLeagueV2, season=self.season)

    def stat_table(self, day: date, scope: str, measure: str, refresh: bool = False) -> dict:
        cls = nba.LeagueDashTeamStats if scope == "teams" else nba.LeagueDashPlayerStats
        name = f"{scope}_{measure.lower().replace(' ', '_')}"
        return self._snapshot(day, name, refresh, cls, season=self.season,
                              season_type_all_star=self.api.season_type,
                              per_mode_detailed="PerGame", measure_type_detailed_defense=measure)

    def league_game_log(self, kind: str, refresh: bool = False) -> dict:
        """Whole-season game log in one call: kind 'T' (team rows) or 'P' (player rows)."""
        return self._snapshot(date.today(), f"league_game_log_{kind}", refresh, nba.LeagueGameLog,
                              season=self.season, player_or_team_abbreviation=kind,
                              season_type_all_star=self.api.season_type)

    def player_index(self, day: date, refresh: bool = False, historical: bool = False) -> dict:
        """Positions. historical=True also lists players no longer on a roster (past seasons:
        the plain index for 2025-26 covered only 22% of that season's players, 29.10.2026)."""
        if historical:
            return self._snapshot(day, "player_index_historical", refresh, nba.PlayerIndex,
                                  season=self.season, historical_nullable="1")
        return self._snapshot(day, "player_index", refresh, nba.PlayerIndex, season=self.season)

    # --- immutable per-game files ---

    def boxscore(self, game_id: str, refresh: bool = False) -> Optional[dict]:
        path = self.cache.boxscore_path(self.key, self.season, game_id)
        if not refresh and (cached := self.cache.read(path)) is not None:
            return cached
        payload = self._call(nba.BoxScoreTraditionalV3, game_id=game_id)
        box = (payload or {}).get("boxScoreTraditional") or {}
        if not box.get("homeTeam", {}).get("players"):
            log.warning("NBA: празен box score за %s", game_id)
            return None
        self.cache.write(path, payload)
        return payload

    def _snapshot(self, day: date, name: str, refresh: bool, cls, **params) -> Any:
        path = self.cache.snapshot_path(self.key, day, f"{self.season}_{name}")
        if not refresh and (cached := self.cache.read(path)) is not None:
            return cached
        payload = self._call(cls, **params)
        self.cache.write(path, payload)
        return payload

    def _call(self, cls, **params) -> dict:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.http.retries + 1):
            wait = self.api.pause_s - (time.monotonic() - self._last_request)
            if wait > 0:
                self._sleep(wait)
            self._last_request = time.monotonic()
            try:
                return cls(timeout=self.http.timeout_s, **params).get_dict()
            except Exception as exc:  # nba_api raises requests errors, JSON errors, KeyErrors
                last_error = exc
                log.warning("NBA %s обид %d/%d неуспешен: %s", cls.__name__, attempt, self.http.retries, exc)
                if attempt < self.http.retries:
                    self._sleep(self.api.retry_wait_s * attempt)
        raise SourceError(f"stats.nba.com не одговара по {self.http.retries} обиди: {cls.__name__} "
                          f"{params} ({last_error})")
