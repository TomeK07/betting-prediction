from datetime import date, datetime, timezone

import yaml

from research import nitro

NOW = datetime(2026, 11, 1, 21, 0, tzinfo=timezone.utc)
TEXT = "Ziaire Williams has been listed as questionable for tonight's game against Toronto."
FACT = {"tim": "LAL", "tip": "INJURY", "igrac": "Ziaire Williams", "sto": "questionable", "status": "QUESTIONABLE",
        "izvor_url": "https://example.com/a", "citat": "Ziaire Williams has been listed as questionable", "objaveno": "2026-11-01"}


def test_valid_fact_passes():
    assert nitro.check(FACT, TEXT, {"williams"}, NOW) == []


def test_rules_like_t1_t10():
    assert any("цитатот" in b for b in nitro.check({**FACT, "citat": "Williams is out for the season"}, TEXT, {"williams"}, NOW))
    assert any("симне" in b for b in nitro.check(FACT, None, {"williams"}, NOW))                  # no download = no check
    assert any("нема извор" in b for b in nitro.check({**FACT, "izvor_url": ""}, TEXT, {"williams"}, NOW))
    assert any("состав" in b for b in nitro.check(FACT, TEXT, {"james"}, NOW))
    assert any("забрането" in b for b in nitro.check({**FACT, "verojatnost": 0.6}, TEXT, {"williams"}, NOW))
    assert any("статистика" in b for b in nitro.check({**FACT, "sto": "scored 22 points"}, TEXT, {"williams"}, NOW))
    assert any("датум" in b for b in nitro.check({**FACT, "objaveno": None}, TEXT, {"williams"}, NOW))


def test_newer_status_replaces_older_and_keeps_newest(tmp_path, monkeypatch):
    monkeypatch.setattr(nitro, "VESTI_DIR", tmp_path)
    day = date(2026, 11, 1)
    base = {"liga": "nba", "tim": "Los Angeles Lakers", "igrac": "Ziaire Williams", "izvor_url": "u", "citat": "c"}
    (tmp_path / "2026-11-01.yaml").write_text(yaml.safe_dump([{**base, "status": "QUESTIONABLE",
                                                              "objaveno": "2026-11-01T10:00:00+00:00"}]), encoding="utf-8")
    assert nitro.to_vesti_yaml(day, [{**base, "status": "OUT", "objaveno": "2026-11-01T20:00:00+00:00"}]) == 1
    assert nitro.to_vesti_yaml(day, [{**base, "status": "PROBABLE", "objaveno": "2026-11-01T05:00:00+00:00"}]) == 0
    rows = yaml.safe_load((tmp_path / "2026-11-01.yaml").read_text(encoding="utf-8"))
    assert [r["status"] for r in rows] == ["OUT"]
