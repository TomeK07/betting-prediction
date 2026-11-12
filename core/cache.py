"""Raw download cache.

Two kinds of files:
- snapshots (schedule, season stats, rosters) change daily -> data/raw/<league>/<YYYY-MM-DD>/<name>.json
- immutable game files (finished box scores) -> data/raw/<league>/boxscore/<season>/<game>.json

Nothing is ever deleted: refreshing a snapshot renames the old file with a timestamp suffix.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)


class RawCache:
    def __init__(self, root: Path):
        self.root = Path(root)

    def snapshot_path(self, league: str, day: date, name: str) -> Path:
        return self.root / league / day.isoformat() / f"{name}.json"

    def boxscore_path(self, league: str, season: str, game_code: int | str) -> Path:
        name = f"{game_code:03d}" if isinstance(game_code, int) else str(game_code)
        return self.root / league / "boxscore" / season / f"{name}.json"

    @staticmethod
    def read(path: Path) -> Optional[Any]:
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def write(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            stamp = datetime.fromtimestamp(path.stat().st_mtime).strftime("%H%M%S")
            archived = path.with_name(f"{path.stem}__{stamp}{path.suffix}")
            path.replace(archived)
            log.debug("Стара верзија зачувана како %s", archived.name)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        tmp.replace(path)
