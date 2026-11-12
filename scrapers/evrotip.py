"""evrotip.mk odds snapshots — CACHE ONLY (decision 29.10: second bookmaker for line comparison/CLV).

Endpoints (checked 29.10.2026, plain requests, no blocking; robots.txt not published):
  /restapi/offer/mk/ttg_lang                         bet-code dictionary (betPickMap: '<code>_<sport>' → tipTypeName)
  /restapi/offer/mk/sport/<B|SK>/league/<id>/mob     matches of a league (B = team markets, SK = player markets)
  /restapi/offer/mk/match/<id>                       all odds of one match + `params` (the lines)
Each snapshot = new gzip file data/raw/evrotip/<YYYY-MM-DD>/<HHMMSS>.json.gz (never overwritten).
No model reads these before Phase 7 (anti-anchoring).
"""
from __future__ import annotations

import gzip
import json
import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from core.config import Settings
from scrapers.http_client import HttpClient

log = logging.getLogger(__name__)
QUERY = {"annex": 1, "mobileVersion": "2.48", "locale": "mk"}


def write_gz(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def read_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def snapshot(settings: Settings, client: HttpClient | None = None) -> tuple[Path, dict]:
    cfg = settings.evrotip
    client = client or HttpClient(settings.http)
    headers = {"Referer": cfg.base + "/", "Accept": "application/json"}
    now = datetime.now(ZoneInfo(settings.timezone))
    folder = settings.paths.raw / "evrotip" / now.date().isoformat()

    dict_path = folder / "ttg_lang.json.gz"          # bet-code dictionary: once per day
    if not dict_path.exists():
        ttg = client.get_json(f"{cfg.base}/restapi/offer/mk/ttg_lang", QUERY, headers=headers)
        write_gz(dict_path, {"betPickMap": ttg.get("betPickMap", {}), "betMap": ttg.get("betMap", {})})

    matches, counts = [], {}
    for label, spec in cfg.leagues.items():
        if not spec:
            continue
        sport, league_id = spec
        lst = client.get_json(f"{cfg.base}/restapi/offer/mk/sport/{sport}/league/{league_id}/mob", QUERY,
                              headers=headers).get("esMatches") or []
        counts[label] = len(lst)
        for m in lst:
            full = client.get_json(f"{cfg.base}/restapi/offer/mk/match/{m['id']}", QUERY, headers=headers)
            full["_league"] = label
            matches.append(full)
    payload = {"fetched_at": now.isoformat(timespec="seconds"), "source": cfg.base, "matches": matches}
    path = folder / f"{now:%H%M%S}.json.gz"
    write_gz(path, payload)
    return path, {"matches": len(matches), "odds": sum(len(m.get("odds") or {}) for m in matches),
                  "by_league": counts}
