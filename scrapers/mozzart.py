"""Mozzart odds snapshots (BUILD_PROMPT §2.1) — CACHE ONLY.

Decision 29.10.2026: start saving lines every day for backtests and CLV, but nothing in the model
may read them before Phase 7 (anti-anchoring). This module only downloads and stores raw JSON.
Each call writes a NEW file data/raw/mozzart/<YYYY-MM-DD>/<HHMMSS>.json.gz (never overwritten), so the
last snapshot before tip-off can later serve as the closing price.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from core.config import Settings
from scrapers.http_client import HttpClient

log = logging.getLogger(__name__)

HEADERS_EXTRA = {"Content-Type": "application/json", "Accept": "application/json, text/plain, */*"}


def tracked_competitions(settings: Settings) -> dict[int, str]:
    """competition.id → label, for the three leagues (team + player markets)."""
    out = {}
    for key, lg in settings.leagues.items():
        out[lg.mozzart_competition_id] = f"{key}"
        out[lg.mozzart_players_competition_id] = f"{key}_igraci"
    return out


def snapshot(settings: Settings, client: HttpClient | None = None) -> tuple[Path, dict]:
    cfg = settings.mozzart
    client = client or HttpClient(settings.http)
    headers = {**HEADERS_EXTRA, "Origin": cfg.base, "Referer": cfg.base + "/"}
    now = datetime.now(ZoneInfo(settings.timezone))

    games = client.get_json(f"{cfg.base}/getAllGames", headers=headers)
    offer = [g for g in games.get(str(cfg.sport_id), []) if g.get("name") == cfg.offer_name]
    if not offer:
        raise RuntimeError(f"Mozzart: нема „{cfg.offer_name}“ за спорт {cfg.sport_id}")
    subgames = list(dict.fromkeys(offer[0]["subgameIds"]))     # API returns duplicates

    tracked = tracked_competitions(settings)
    matches, offset = [], 0
    while True:
        page = client.post_json(f"{cfg.base}/betOffer2", {
            "date": "all_days", "sort": "bytime", "pageSize": cfg.page_size, "offset": offset,
            "sportIds": [cfg.sport_id], "search": "", "matchTypeId": 0}, headers=headers).get("matches", [])
        if not page:
            break
        matches += [m for m in page if m.get("competition", {}).get("id") in tracked]
        offset += cfg.page_size

    odds = []
    ids = [m["id"] for m in matches]
    for i in range(0, len(ids), cfg.odds_batch):
        # an empty `subgames` list returns no odds — the full list is required (BUILD_PROMPT §2.1)
        odds += client.post_json(f"{cfg.base}/getBettingOdds",
                                 {"matchIds": ids[i:i + cfg.odds_batch], "subgames": subgames}, headers=headers)

    payload = {"fetched_at": now.isoformat(timespec="seconds"), "source": cfg.base,
               "competitions": {str(k): v for k, v in tracked.items()},
               "matches": matches, "odds": odds}
    from scrapers.evrotip import write_gz
    path = settings.paths.raw / "mozzart" / now.date().isoformat() / f"{now:%H%M%S}.json.gz"
    write_gz(path, payload)                      # compressed (decision 29.10); older .json files stay as they are

    by_comp = Counter(tracked[m["competition"]["id"]] for m in matches)
    kodds = sum(len(o.get("kodds", {})) for o in odds)
    summary = {"matches": len(matches), "kodds": kodds, "by_competition": dict(by_comp),
               "without_odds": len(ids) - sum(1 for o in odds if o.get("kodds"))}
    return path, summary
