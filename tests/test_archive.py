import pandas as pd
import pytest

from store.grade import LeagueData, _split_legacy_name, candidate_dates, grade_pick, match_player, resolve_team, settle
from store.legacy_csv import LegacyRow, _split_glued, read_legacy_csv
from store.migrate import apply_corrections, mark_duplicates, migrate_rows, parse_datum
from store.stats import calibration_table, decided, summarize, wilson_interval

HEADER = "datum,liga,natprevar_ili_igrac,tim,pazar,linija,strana,kvota,p_model,p_pazar,rezultat,pogodok,minuti,zabeleska\n"


# ---------- legacy CSV repair ----------

def test_unquoted_comma_in_note_is_rejoined(tmp_path):
    f = tmp_path / "a.csv"
    f.write_text(HEADER + "23.10 20:15,Evroliga,OLY - FEN,,hendikep,2.5,Fener +2.5,1.90,55,50,FEN +3,DA,,Fournier OUT, Montero ?\n",
                 encoding="utf-8")
    [row] = read_legacy_csv(f)
    assert row.values["zabeleska"] == "Fournier OUT, Montero ?"
    assert row.repairs


def test_two_records_glued_on_one_line_are_split():
    a = ["28.10 09:20", "Evroliga", "ANALIZA", "", "", "", "", "", "", "", "", "BELESKA", "", "text tim.28.10 19:00"]
    b = ["Evroliga", "V. Micic", "Hapoel", "asistencii", "4.5", "pod", "1.65", "68", "58", "", "CEKA", "~24", "note"]
    first, second = _split_glued("x.csv", 27, a + b)
    assert first.values["zabeleska"] == "text tim."
    assert second.values["datum"] == "28.10 19:00" and second.values["natprevar_ili_igrac"] == "V. Micic"


def test_parse_datum():
    assert parse_datum("27.10 20:15") == ("2026-10-27", "20:15")
    assert parse_datum("24.10 ~17:05") == ("2026-10-24", "17:05")
    assert parse_datum("28.10") == ("2026-10-28", None)


def _row(**kw):
    base = dict(datum="25.10 02:00", liga="NBA", natprevar_ili_igrac="WAS - CHI", tim="", pazar="vkupno",
                linija="243.5", strana="under", kvota="1.90", p_model="56", p_pazar="50", rezultat="",
                pogodok="CEKA", minuti="", zabeleska="")
    base.update(kw)
    return LegacyRow("x.csv", 2, base)


def test_migration_normalizes_sides_leagues_and_percentages():
    m = migrate_rows([_row(), _row(liga="Euroleague", strana="poveke", natprevar_ili_igrac="J. Loyd", tim="Efes",
                               pazar="asistencii", linija="2.5")])
    total, player = m.picks
    assert total["strana"] == "pod" and total["liga"] == "nba" and total["p_model"] == pytest.approx(0.56)
    assert player["liga"] == "euroleague" and player["strana"] == "nad" and player["igrac"] == "J. Loyd"
    assert total["legacy_json"]  # original kept verbatim


def test_migration_handicap_side_and_ceka_in_wrong_column():
    m = migrate_rows([_row(pazar="hendikep", linija="2.5", strana="Fener +2.5"),
                      _row(pogodok="", rezultat="CEKA")])
    assert (m.picks[0]["strana"], m.picks[0]["linija"]) == ("Fener", 2.5)
    assert m.picks[1]["status"] == "CEKA" and m.picks[1]["rezultat_tekst"] is None


def test_beleska_goes_to_notes_not_picks():
    m = migrate_rows([_row(pogodok="BELESKA", pazar="", zabeleska="lesson")])
    assert not m.picks and m.notes[0]["tekst"] == "lesson"


def test_duplicates_kept_as_records_but_marked():
    first = migrate_rows([_row(pogodok="DA", p_model="58", kvota="2.05")]).picks[0]
    second = {**first, "p_model": 0.55, "kvota": 2.06, "izvor": "x.csv:9"}
    rejected = {**first, "status": "ODBIEN", "izvor": "x.csv:1"}
    picks = [rejected, first, second]
    marked = mark_duplicates(picks)
    assert len(picks) == 3                                                # nothing is deleted
    assert first["status"] == "DA" and first["p_model"] == pytest.approx(0.58)  # p_model locked at first write
    assert rejected["status"] == "DUPLIKAT" and second["status"] == "DUPLIKAT"
    assert {m["duplikat_od"] for m in marked} == {first["izvor"]}


def test_same_pick_other_line_is_duplicate_but_bought_line_is_not():
    a, b, c = migrate_rows([
        _row(pogodok="NE", linija="186.5", natprevar_ili_igrac="Paris - Valencia"),
        _row(pogodok="NE", linija="187.5", natprevar_ili_igrac="Paris - Valencia"),     # Megawin copy
        _row(pogodok="DA", linija="9.5", pazar="poeni (Mozzart kupeno mbpi)", natprevar_ili_igrac="W. Kessler",
             strana="over"),
    ]).picks
    d = migrate_rows([_row(pogodok="DA", linija="11.5", pazar="poeni", natprevar_ili_igrac="W. Kessler",
                           strana="over")]).picks[0]
    for i, p in enumerate((a, b, c, d)):
        p["izvor"] = f"x.csv:{i}"
    mark_duplicates([a, b, c, d])
    assert a["status"] == "NE" and b["status"] == "DUPLIKAT"
    assert c["status"] == "DA" and d["status"] == "DA"     # main + bought line = two picks (decision 29.10)


def test_corrections_applied_and_logged():
    m = migrate_rows([_row()])
    m.picks[0]["izvor"] = "x.csv:2"
    apply_corrections(m, {"x.csv:2": {"set": {"status": "POVLECEN"}, "prichina": "заменет со ревизија"}})
    assert m.picks[0]["status"] == "POVLECEN" and "КОРЕКЦИЈА" in m.picks[0]["zabeleska"]
    assert m.corrections
    with pytest.raises(ValueError):
        apply_corrections(m, {"x.csv:99": {"set": {}, "prichina": ""}})


# ---------- grading ----------

def _league():
    schedule = pd.DataFrame([
        {"game_code": "1", "game_date": "2026-10-24", "status": "result", "home": "CHI", "away": "MIL",
         "home_name": "Chicago Bulls", "away_name": "Milwaukee Bucks"},
        {"game_code": "2", "game_date": "2026-10-25", "status": "result", "home": "MIL", "away": "PHI",
         "home_name": "Milwaukee Bucks", "away_name": "Philadelphia 76ers"},
    ])
    team_logs = pd.DataFrame([
        {"game_code": "1", "team": "CHI", "pts": 110, "overtimes": 0}, {"game_code": "1", "team": "MIL", "pts": 115, "overtimes": 0},
        {"game_code": "2", "team": "MIL", "pts": 100, "overtimes": 1}, {"game_code": "2", "team": "PHI", "pts": 100 + 5, "overtimes": 1},
    ])
    player_logs = pd.DataFrame([
        {"game_code": "1", "team": "CHI", "player_code": "10", "player": "Josh Giddey", "label": "Giddey J. (CHI)",
         "dnp": False, "minutes": 33.0, "pts": 12, "reb": 8, "ast": 9, "fgm3": 1, "absence_reason": None},
        {"game_code": "1", "team": "CHI", "player_code": "11", "player": "Jalen Smith", "label": "Smith J. (CHI)",
         "dnp": True, "minutes": 0.0, "pts": 0, "reb": 0, "ast": 0, "fgm3": 0, "absence_reason": "DNP - Coach's Decision"},
        {"game_code": "1", "team": "CHI", "player_code": "12", "player": "Jevon Smith", "label": "Smith J. (CHI)",
         "dnp": False, "minutes": 10.0, "pts": 4, "reb": 1, "ast": 0, "fgm3": 0, "absence_reason": None},
    ])
    return LeagueData(schedule, team_logs, player_logs, {"bulls": "CHI"})


def _pick(**kw):
    base = dict(datum="2026-10-25", vreme="01:00", liga="nba", natprevar=None, tim="CHI", igrac=None,
                pazar="asistencii", linija=8.5, strana="nad")
    base.update(kw)
    return base


def test_settle_over_under_push():
    assert settle(9, 8.5, "nad") == "DA" and settle(8, 8.5, "nad") == "NE"
    assert settle(8, 8.5, "pod") == "DA" and settle(8, 8.0, "pod") == "VOID"


def test_nba_window_and_back_to_back_is_ambiguous():
    assert candidate_dates("2026-10-25", "22:00", "nba") == ["2026-10-24", "2026-10-25"]
    assert candidate_dates("2026-10-25", None, "euroleague") == ["2026-10-25"]
    res = grade_pick(_pick(tim="MIL", igrac="X. Y", pazar="poeni"), _league())
    assert res.status == "CEKA" and "рачно" in res.reason        # MIL played 24 and 25 → never guess


def test_player_prop_graded_from_box_score():
    res = grade_pick(_pick(igrac="J. Giddey"), _league())
    assert res.status == "DA" and res.value == 9


def test_combined_market_and_alias():
    res = grade_pick(_pick(tim="Bulls", igrac="Jo. Giddey", pazar="pra", linija=30.5, strana="pod"), _league())
    assert res.status == "DA" and res.value == 29


def test_same_surname_and_initial_is_never_guessed():
    # Error 16.10 (Carsen vs Kessler Edwards): two "Smith J." on CHI → stays CEKA
    res = grade_pick(_pick(igrac="J. Smith", pazar="poeni", linija=2.5), _league())
    assert res.status == "CEKA" and "повеќе" in res.reason
    assert grade_pick(_pick(igrac="Jev. Smith", pazar="poeni", linija=2.5), _league()).status == "DA"


def test_dnp_is_void():
    res = grade_pick(_pick(igrac="Jal. Smith", pazar="poeni", linija=2.5), _league())
    assert res.status == "VOID" and "Coach" in res.reason


def test_team_markets_total_handicap_and_1x2_regulation():
    lg = _league()
    tot = grade_pick(_pick(natprevar="CHI - MIL", tim=None, pazar="vkupno", linija=224.5, strana="pod"), lg)
    assert tot.status == "NE" and tot.value == 225          # 110 + 115 = 225 > 224.5
    assert grade_pick(_pick(natprevar="CHI - MIL", tim=None, pazar="vkupno", linija=225.5, strana="pod"),
                      lg).status == "DA"
    hcp = grade_pick(_pick(natprevar="CHI - MIL", tim=None, pazar="hendikep", linija=4.5, strana="CHI"), lg)
    assert hcp.status == "NE"  # CHI lost by 5, +4.5 → −0.5
    # game 2 went to overtime → 1X2 (regulation) is a draw
    x = grade_pick(_pick(datum="2026-10-25", natprevar="MIL - PHI", tim=None, pazar="1x2", linija=None, strana="2"),
                   LeagueData(lg.schedule[lg.schedule.game_code == "2"], lg.team_logs, lg.player_logs, {}))
    assert x.status == "NE" and "X" in x.reason


def test_name_parsing():
    assert _split_legacy_name("Don. Mitchell") == ("don", "mitchell")
    assert _split_legacy_name("J. Clark III") == ("j", "clark")
    assert _split_legacy_name("K. A. Towns") == ("k", "towns")


def test_resolve_team_prefix_and_alias():
    lg = _league()
    assert resolve_team("Milwaukee", lg) == "MIL"
    assert resolve_team("OKC (timski)", lg) is None


# ---------- stats ----------

def test_wilson_interval_matches_proces_example():
    # PROCES §2: 13/16 → about 57–93%
    lo, hi = wilson_interval(13, 16)
    assert 0.55 < lo < 0.60 and 0.90 < hi < 0.95


def test_summary_brier_roi_and_calibration():
    picks = pd.DataFrame({
        "status": ["DA", "NE", "DA", "CEKA", "DA"], "p_model": [0.6, 0.6, 0.7, 0.6, None],
        "kvota": [2.0, 2.0, None, 1.9, 1.9], "pazar": ["poeni"] * 5, "strana": ["nad"] * 5, "clv": [None] * 5,
    })
    d = decided(picks)
    assert len(d) == 3                                    # CEKA and missing p_model excluded
    s = summarize(d)
    assert s.hits == 2 and s.brier == pytest.approx((0.16 + 0.36 + 0.09) / 3)
    assert s.roi == pytest.approx(0.0) and s.roi_n == 2   # +1 and −1 with flat stakes
    assert s.clv_n == 0                                   # CLV only where a closing price exists
    cal = calibration_table(d)
    assert list(cal["ниво"]) == ["60–65%", "70%+"]


def test_roi_model_excludes_unverified_and_roi_paid_uses_real_stakes():
    picks = pd.DataFrame({
        "status": ["DA", "NE", "DA", "NE"], "p_model": [0.6] * 4, "kvota": [2.0, 2.0, 3.0, 1.9],
        "kvota_proverena": [1, 1, 0, 1], "uplateno": [1, 1, None, 0], "uplateno_iznos": [300, 100, None, None],
        "pazar": ["poeni"] * 4, "strana": ["nad"] * 4, "clv": [None] * 4,
    })
    s = summarize(decided(picks))
    assert s.roi_n == 3 and s.roi == pytest.approx((1 - 1 - 1) / 3)        # the 3.00 unverified price is out
    assert s.roi_paid_n == 2 and s.paid_stake == 400
    assert s.roi_paid == pytest.approx((300 * 1 - 100) / 400)              # +300 won, −100 lost


def test_new_picks_graded_by_game_code_and_player_code():
    lg = _league()
    pick = _pick(igrac="Smith J. (CHI)", pazar="poeni", linija=2.5, strana="nad", game_code="1",
                 vlezovi_json='{"player_code": "12"}', datum="2026-10-24")
    res = grade_pick(pick, lg)
    assert res.status == "DA" and res.value == 4          # Jevon Smith by id — no ambiguity with Jalen Smith
    missing = grade_pick({**pick, "vlezovi_json": '{"player_code": "99"}'}, lg)
    assert missing.status == "VOID"
    team = grade_pick(_pick(natprevar="CHI - MIL", tim=None, pazar="vkupno", linija=224.5, strana="pod",
                            game_code="1", datum="2026-10-24"), lg)
    assert team.status == "NE"


def test_paper_review_needs_100_decided_and_10_nights_and_top_picks_shown_apart():
    from store.stats import paper_section
    n = 120                                                       # 120 decided picks but all from 3 nights
    picks = pd.DataFrame({
        "status": ["DA", "NE"] * (n // 2), "p_model": [0.6] * n, "p_kalibriran": [0.6] * n, "kvota": [1.9] * n,
        "pazar": ["poeni"] * n, "strana": ["pod"] * n, "liga": ["nba"] * n, "clv": [None] * n,
        "datum": [f"2026-10-{29 + i % 3:02d}" for i in range(n)], "verzija_model": ["py-v1"] * n,
        "vlezovi_json": ['{"tiket": true}' if i < 4 else '{"tiket": false}' for i in range(n)],
    })
    text = "\n".join(paper_section(picks))
    assert "одлучени 120 / 100 И вечери 3 / 10 → уште не" in text
    assert "ТОП избори (би влегле во тикет): 2/4" in text and "Сите на хартија (учење): 60/120" in text


def test_rejected_by():
    from store.db import rejected_by
    assert rejected_by("EV_ПОД_4% (+1.1%)") == "МОДЕЛ"
    assert rejected_by("ПРОВЕРИ_ПОДАТОЦИ (EV +20% > +15%); ЗНАК: ЗАВИСИ_ОД_СОСТАВ") == "ФИЛТЕР"
    assert rejected_by("ЗНАК: ЗАВИСИ_ОД_СОСТАВ") == "СОСТАВ"
    assert rejected_by(None) == "РАЧНО"
