from datetime import date

import pytest
import requests

from core.cache import RawCache
from core.config import HttpSettings, load_settings
from scrapers.http_client import HttpClient, SourceError


def test_settings_load():
    s = load_settings()
    assert s.league("euroleague").season == "E2026"
    assert s.league("eurocup").competition_code == "U"
    assert s.paths.raw.is_absolute()
    with pytest.raises(KeyError):
        s.league("abl")


def test_cache_never_deletes_old_snapshot(tmp_path):
    cache = RawCache(tmp_path)
    path = cache.snapshot_path("euroleague", date(2026, 10, 29), "schedule")
    cache.write(path, {"v": 1})
    cache.write(path, {"v": 2})
    assert cache.read(path) == {"v": 2}
    archived = [p for p in path.parent.iterdir() if p.name.startswith("schedule__")]
    assert len(archived) == 1 and cache.read(archived[0]) == {"v": 1}


def test_boxscore_path():
    cache = RawCache("data/raw")
    assert cache.boxscore_path("eurocup", "U2026", 7).as_posix().endswith("eurocup/boxscore/U2026/007.json")


class _Resp:
    def __init__(self, status, payload=None):
        self.status_code = status
        self._payload = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def json(self):
        return self._payload


class _Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.headers = {}
        self.calls = 0

    def request(self, method, url, params=None, timeout=None, json=None, headers=None):
        self.calls += 1
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


CFG = HttpSettings(timeout_s=1, retries=3, backoff_s=0, pause_s=0, user_agent="test")


def test_http_retries_then_succeeds():
    session = _Session([requests.ConnectionError("down"), _Resp(503), _Resp(200, {"ok": 1})])
    client = HttpClient(CFG, session=session, sleep=lambda s: None)
    assert client.get_json("http://x") == {"ok": 1}
    assert session.calls == 3


def test_http_429_waits_retry_after():
    waits = []
    r429 = _Resp(429)
    r429.headers = {"Retry-After": "272"}
    session = _Session([r429, _Resp(200, {"ok": 1})])
    client = HttpClient(CFG, session=session, sleep=waits.append)
    assert client.get_json("http://x") == {"ok": 1}
    assert 273 in waits


def test_http_fails_loudly_after_retries():
    session = _Session([_Resp(500), _Resp(500), _Resp(500)])
    client = HttpClient(CFG, session=session, sleep=lambda s: None)
    with pytest.raises(SourceError):
        client.get_json("http://x")


class _FakeMozzart:
    """Minimal stand-in for the Mozzart API."""
    def __init__(self, n_matches):
        self.matches = [{"id": i, "competition": {"id": 26 if i % 2 else 999}} for i in range(n_matches)]
        self.odds_calls = []

    def get_json(self, url, params=None, headers=None):
        return {"2": [{"name": "Комплетна понуда", "subgameIds": [1, 1, 2]}]}

    def post_json(self, url, body, headers=None):
        if url.endswith("/betOffer2"):
            return {"matches": self.matches[body["offset"]: body["offset"] + body["pageSize"]]}
        self.odds_calls.append(body)
        return [{"id": i, "kodds": {"x": {}}} for i in body["matchIds"]]


def test_mozzart_snapshot_filters_batches_and_never_overwrites(tmp_path):
    from scrapers.mozzart import snapshot
    s = load_settings()
    s.paths.raw = tmp_path
    fake = _FakeMozzart(130)                                     # 65 tracked (EL id 26), 65 other
    path, summary = snapshot(s, client=fake)
    assert summary["matches"] == 65 and summary["by_competition"] == {"euroleague": 65}
    assert [len(c["matchIds"]) for c in fake.odds_calls] == [30, 30, 5]
    assert fake.odds_calls[0]["subgames"] == [1, 2]               # deduplicated, never empty
    assert path.parent.parent.name == "mozzart" and path.exists() and path.name.endswith(".json.gz")
