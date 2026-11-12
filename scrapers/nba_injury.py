"""Official NBA injury report (BUILD_PROMPT §2.4) — download only.

The index page lists every report PDF (one every 15 minutes on game days). We take the latest one
for the requested date and cache it forever (a published report never changes):
data/raw/nba/injury/<YYYY-MM-DD>/Injury-Report_<date>_<time>.pdf
"""
from __future__ import annotations

import logging
import re
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import requests

from core.config import Settings

log = logging.getLogger(__name__)

INDEX_URL = "https://official.nba.com/nba-injury-report-{season}-season/"
PDF_BASE = "https://ak-static.cms.nba.com/referee/injury/"
LINK_RE = re.compile(r"Injury-Report_(\d{4}-\d{2}-\d{2})_(\d{2})_(\d{2})(AM|PM)\.pdf")


def _report_time(m: re.Match) -> datetime:
    day, hh, mm, ampm = m.groups()
    hour = int(hh) % 12 + (12 if ampm == "PM" else 0)
    return datetime.fromisoformat(f"{day}T{hour:02d}:{mm}:00")


def latest_report(settings: Settings, day: date, session: Optional[requests.Session] = None) -> Optional[Path]:
    """Download (or reuse) the latest report published for `day`. None if there is none yet."""
    session = session or requests.Session()
    season = settings.league("nba").season
    headers = {"User-Agent": settings.http.user_agent}
    html = session.get(INDEX_URL.format(season=season), headers=headers, timeout=settings.http.timeout_s).text
    links = {m.group(0): _report_time(m) for m in LINK_RE.finditer(html) if m.group(1) == day.isoformat()}
    if not links:
        log.warning("NBA injury report: нема објавен извештај за %s", day)
        return None
    name = max(links, key=links.get)
    folder = settings.paths.raw / "nba" / "injury" / day.isoformat()
    path = folder / name
    if path.exists():
        return path
    resp = session.get(PDF_BASE + name, headers=headers, timeout=settings.http.timeout_s)
    resp.raise_for_status()
    if not resp.content.startswith(b"%PDF"):
        raise RuntimeError(f"NBA injury report: {name} не е PDF")
    folder.mkdir(parents=True, exist_ok=True)
    path.write_bytes(resp.content)
    log.info("NBA injury report: %s (%s)", name, links[name].strftime("%H:%M ET"))
    return path
