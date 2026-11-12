from datetime import date, datetime

from pipeline import dnevno_cmd as dn


def test_report_urgent_first_line_and_1600_update(tmp_path, monkeypatch):
    monkeypatch.setattr(dn, "OUT", tmp_path)
    day = date(2026, 11, 1)
    p = dn.write_report(day, "## Главна 14:00\nA", ["14:00 падна: fetch (GRESKI.log)"], first=True)
    assert p.read_text(encoding="utf-8").startswith("⚠️ ИТНО: 14:00 падна: fetch")
    dn.write_report(day, "## Ажурирање 20:00\nB", ["1 избор(и) чекаат твоја одлука пред уплата"], first=False)
    text = p.read_text(encoding="utf-8")
    assert text.count("⚠️ ИТНО") == 1 and "fetch" in text.splitlines()[0] and "одлука" in text.splitlines()[0]
    assert text.index("Главна 14:00") < text.index("Ажурирање 20:00")
    p2 = dn.write_report(date(2026, 11, 2), "## Главна 14:00\nC", [], first=True)
    assert p2.read_text(encoding="utf-8").startswith("# Дневен извештај")


def test_failed_step_is_logged_and_does_not_raise(tmp_path, monkeypatch):
    monkeypatch.setattr(dn, "OUT", tmp_path)
    monkeypatch.setattr(dn, "ERRORS", tmp_path / "GRESKI.log")
    ok = dn.run_step(["nema_takva_komanda"], "14:00", datetime(2026, 11, 1, 14, 0))
    assert ok is False
    line = (tmp_path / "GRESKI.log").read_text(encoding="utf-8")
    assert "14:00 | nema_takva_komanda | exit 2" in line
    assert [f.name for f in (tmp_path / "logs").iterdir()] == ["2026-11-01_1400_1400_nema_takva_komanda.log"]


def test_llm_site_does_not_start_twice(tmp_path, monkeypatch, capsys):
    from pipeline import llm_cmd, profile_cmd
    monkeypatch.setattr(profile_cmd, "TEAMS_DIR", tmp_path)
    (tmp_path / llm_cmd.LOCK).write_text("123", encoding="utf-8")           # fresh heartbeat of another run
    assert llm_cmd.run_all(None, date(2026, 11, 1), None) == 0
    assert "веќе работи" in capsys.readouterr().out
    assert (tmp_path / llm_cmd.LOCK).exists()                              # the other run's lock is untouched


def test_run_lock_serializes_runs_and_clears_stale(tmp_path, monkeypatch):
    import os
    monkeypatch.setattr(dn, "OUT", tmp_path)
    monkeypatch.setattr(dn, "RUN_LOCK", tmp_path / ".dnevno.lock")
    assert dn.acquire_run_lock(wait_s=0) is True
    assert dn.acquire_run_lock(wait_s=0) is False        # 1.11: 14:00/20:00/CLV must not run together
    old = dn.RUN_LOCK.stat().st_mtime - dn.RUN_LOCK_STALE_S - 60
    os.utime(dn.RUN_LOCK, (old, old))
    assert dn.acquire_run_lock(wait_s=0) is True         # a killed run's lock does not block forever
