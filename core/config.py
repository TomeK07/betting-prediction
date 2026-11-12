"""Typed access to config/settings.yaml."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
SETTINGS_PATH = ROOT / "config" / "settings.yaml"


class Paths(BaseModel):
    raw: Path
    processed: Path
    reports: Path


class HttpSettings(BaseModel):
    timeout_s: float
    retries: int
    backoff_s: float
    pause_s: float
    rate_limit_max_wait_s: float = 600
    user_agent: str


class EuroleagueApi(BaseModel):
    feeds_base: str
    live_base: str
    schedule_limit: int
    stats_limit: int


class MozzartApi(BaseModel):
    base: str
    sport_id: int
    page_size: int
    odds_batch: int
    offer_name: str


class EvrotipApi(BaseModel):
    base: str
    leagues: dict[str, Optional[list]]


class NbaApi(BaseModel):
    pause_s: float
    retry_wait_s: float
    season_type: str
    game_types: list[str]


class LeagueSettings(BaseModel):
    name_mk: str
    source: str
    season: str
    prev_season: Optional[str] = None
    prev2_season: Optional[str] = None
    team_model: dict = {}
    player_model: dict = {}
    competition_code: Optional[str] = None
    home_advantage_pts: float
    min_expected_minutes: float
    mozzart_competition_id: int
    mozzart_players_competition_id: int


class Settings(BaseModel):
    timezone: str
    paths: Paths
    http: HttpSettings
    euroleague_api: EuroleagueApi
    nba_api: NbaApi
    mozzart: MozzartApi
    evrotip: EvrotipApi
    leagues: dict[str, LeagueSettings]
    thresholds: dict[str, float]
    stake: dict[str, float]
    limits: dict[str, int]

    def league(self, key: str) -> LeagueSettings:
        if key not in self.leagues:
            raise KeyError(f"Непозната лига '{key}'. Можни: {', '.join(self.leagues)}")
        return self.leagues[key]


def load_settings(path: Path = SETTINGS_PATH) -> Settings:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    settings = Settings(**data)
    # Resolve relative data paths against the project root so the CLI works from any cwd.
    settings.paths = Paths(**{k: (ROOT / v if not Path(v).is_absolute() else Path(v))
                              for k, v in settings.paths.model_dump().items()})
    return settings
