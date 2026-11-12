"""HTTP GET with timeout, retry and a polite pause between requests."""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

import requests

from core.config import HttpSettings

log = logging.getLogger(__name__)


class SourceError(RuntimeError):
    """A data source failed after all retries — never swallowed silently."""


class HttpClient:
    def __init__(self, cfg: HttpSettings, session: Optional[requests.Session] = None,
                 sleep=time.sleep):
        self.cfg = cfg
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": cfg.user_agent, "Accept": "application/json"})
        self._sleep = sleep
        self._last_request = 0.0

    def get_json(self, url: str, params: Optional[dict] = None, headers: Optional[dict] = None) -> Any:
        return self._request("GET", url, params=params, headers=headers)

    def post_json(self, url: str, body: Any, headers: Optional[dict] = None) -> Any:
        return self._request("POST", url, json=body, headers=headers)

    def _request(self, method: str, url: str, **kwargs) -> Any:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.cfg.retries + 1):
            self._throttle()
            try:
                resp = self.session.request(method, url, timeout=self.cfg.timeout_s, **kwargs)
                if resp.status_code == 429 and attempt < self.cfg.retries:
                    wait = self._retry_after(resp)
                    log.info("Изворот бара пауза (HTTP 429) — чекам %.0f сек…", wait)
                    self._sleep(wait)
                    continue
                if resp.status_code >= 500 or resp.status_code == 429:
                    raise SourceError(f"HTTP {resp.status_code}")
                resp.raise_for_status()
                return resp.json()
            except (requests.RequestException, ValueError, SourceError) as exc:
                last_error = exc
                log.warning("Обид %d/%d неуспешен за %s: %s", attempt, self.cfg.retries, url, exc)
                if attempt < self.cfg.retries:
                    self._sleep(self.cfg.backoff_s * attempt)
        raise SourceError(f"Изворот не одговара по {self.cfg.retries} обиди: {url} ({last_error})")

    def _retry_after(self, resp) -> float:
        """Server-requested wait (Retry-After seconds), capped; falls back to backoff."""
        try:
            wait = float(resp.headers.get("Retry-After"))
        except (TypeError, ValueError):
            wait = self.cfg.backoff_s * 10
        return min(wait + 1, self.cfg.rate_limit_max_wait_s)

    def _throttle(self) -> None:
        wait = self.cfg.pause_s - (time.monotonic() - self._last_request)
        if wait > 0:
            self._sleep(wait)
        self._last_request = time.monotonic()
