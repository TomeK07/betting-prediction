"""EuroLeague / EuroCup official feeds (BUILD_PROMPT §2.2). Same API, competition code E or U."""
from __future__ import annotations

import logging
from datetime import date
from typing import Any, Optional

from core.cache import RawCache
from core.config import EuroleagueApi, LeagueSettings
from scrapers.http_client import HttpClient

log = logging.getLogger(__name__)

# (scope, kind) pairs downloaded on every fetch
STAT_TABLES = [
    ("players", "traditional"),
    ("players", "advanced"),
    ("teams", "traditional"),
    ("teams", "opponentsTraditional"),
    ("teams", "advanced"),
]


class EuroleagueScraper:
    def __init__(self, league_key: str, league: LeagueSettings, api: EuroleagueApi,
                 client: HttpClient, cache: RawCache):
        self.key = league_key
        self.comp = league.competition_code
        self.season = league.season
        self.api = api
        self.client = client
        self.cache = cache

    # --- snapshots (re-downloaded only with refresh=True) ---

    def schedule(self, day: date, refresh: bool = False) -> list[dict]:
        url = f"{self.api.feeds_base}/v2/competitions/{self.comp}/seasons/{self.season}/games"
        payload = self._snapshot(day, "schedule", url, {"limit": self.api.schedule_limit}, refresh)
        meta = payload.get("metadata", {})
        if meta.get("totalPages", 1) > 1:
            raise RuntimeError(f"Распоредот има {meta['totalPages']} страници — зголеми schedule_limit")
        return payload["data"]

    def stat_table(self, day: date, scope: str, kind: str, refresh: bool = False) -> dict:
        url = f"{self.api.feeds_base}/v3/competitions/{self.comp}/statistics/{scope}/{kind}"
        params = {"seasonMode": "Single", "limit": self.api.stats_limit,
                  "seasonCode": self.season, "statisticMode": "perGame"}
        payload = self._snapshot(day, f"{scope}_{kind}", url, params, refresh)
        rows = payload.get(scope, [])
        if payload.get("total", len(rows)) > len(rows):
            raise RuntimeError(f"{scope}/{kind}: добиени {len(rows)} од {payload['total']} — зголеми stats_limit")
        return payload

    def people(self, day: date, refresh: bool = False) -> list[dict]:
        """Season roster for all clubs, incl. player positions (Guard/Forward/Center)."""
        url = f"{self.api.feeds_base}/v2/competitions/{self.comp}/seasons/{self.season}/people"
        payload = self._snapshot(day, "people", url, {"limit": 2000}, refresh)
        return payload["data"] if isinstance(payload, dict) else payload

    # --- immutable per-game files ---

    def boxscore(self, game_code: int, refresh: bool = False) -> Optional[dict]:
        """Box score of a finished game; cached forever once the game is final."""
        path = self.cache.boxscore_path(self.key, self.season, game_code)
        if not refresh and (cached := self.cache.read(path)) is not None:
            return cached
        payload = self.client.get_json(f"{self.api.live_base}/Boxscore",
                                       {"gamecode": game_code, "seasoncode": self.season})
        if not payload or not payload.get("Stats"):
            log.warning("%s: празен box score за натпревар %d", self.key, game_code)
            return None
        if payload.get("Live"):
            log.warning("%s: натпревар %d уште е во тек — не се кешира", self.key, game_code)
            return payload
        self.cache.write(path, payload)
        return payload

    def _snapshot(self, day: date, name: str, url: str, params: dict, refresh: bool) -> Any:
        path = self.cache.snapshot_path(self.key, day, f"{self.season}_{name}")
        if not refresh and (cached := self.cache.read(path)) is not None:
            log.debug("Од кеш: %s", path)
            return cached
        payload = self.client.get_json(url, params)
        self.cache.write(path, payload)
        return payload
