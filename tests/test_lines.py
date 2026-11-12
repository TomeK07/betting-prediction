import math

import pytest

from model.lines import devig_two_way, margins, mark_main_lines, movement, parse_evrotip, parse_mozzart


def _k(short, sub, line, odds):
    return {"subGame": {"gameShortName": short, "subGameName": sub}, "specialOddValue": line, "value": odds}


MOZ = {"fetched_at": "2026-10-29T15:31:00+01:00", "competitions": {"26": "euroleague", "8578": "euroleague_igraci"},
       "matches": [{"id": 1, "competition": {"id": 26}, "startTime": 1793300400000,
                    "participants": [{"name": "Crvena Zvezda"}, {"name": "Bayern"}]},
                   {"id": 2, "competition": {"id": 8578}, "startTime": 1793300400000, "participants": [{"name": "A.Obst"}]}],
       "odds": [{"id": 1, "kodds": {"a": _k("вп", "помалку", "168.5", "1.90"), "b": _k("вп", "повеќе", "168.5", "1.90"),
                                    "c": _k("вп", "помалку", "160.5", "3.10"), "d": _k("вп", "повеќе", "160.5", "1.35"),
                                    "e": _k("кт. хен.", "1", "-6.5", "1.85"), "f": None}},
                {"id": 2, "kodds": {"g": _k("идп", "повеќе", "11.5", "1.90"), "h": _k("идп", "помалку", "11.5", "1.90"),
                                    "i": _k("мбпи", "помалку", "9.5", "2.41")}}]}

ET_DICT = {"betPickMap": {"227_B": {"tipTypeName": "G_PLUS"}, "228_B": {"tipTypeName": "G_MINUS"},
                          "201_B": {"tipTypeName": "H_1"},
                          "51679_SK": {"tipTypeName": "PL_POINTS_OVER"}, "51681_SK": {"tipTypeName": "PL_POINTS_UNDER"}}}
ET = {"fetched_at": "2026-10-29T16:56:07+01:00", "matches": [
    {"id": 5, "sport": "B", "_league": "nba", "home": "ATL Hawks", "away": "CLE Cavaliers", "kickOffTime": 1793314800000,
     "params": {"overUnder": "237.5", "hd2": "1.5"}, "odds": {"227": 1.93, "228": 1.93, "201": 1.98, "999": 3.0}},
    {"id": 6, "sport": "SK", "_league": "nba_igraci", "home": "CJ McCollum", "away": "ATL Hawks", "kickOffTime": 1793314800000,
     "params": {"ouPlPoints": "16.5"}, "odds": {"51679": 1.97, "51681": 1.82}}]}


def test_parse_mozzart_team_player_and_main_line():
    df = mark_main_lines(parse_mozzart(MOZ))
    tot = df[df["pazar"] == "vkupno"]
    assert set(tot["linija"]) == {168.5, 160.5}
    assert set(tot[tot["glavna"]]["linija"]) == {168.5}           # 1.90/1.90 is the main line, 160.5 is bought
    p = df[df["igrac"] == "A.Obst"]
    assert set(p["pazar"]) == {"poeni"} and p[p["linija"] == 9.5]["glavna"].iat[0] == False  # noqa: E712
    assert df[df["pazar"] == "hendikep"]["napomena"].iat[0].startswith("хендикеп: линијата е за домаќинот")


def test_parse_evrotip_lines_from_params_and_flags():
    df = parse_evrotip(ET, ET_DICT)
    tot = df[df["pazar"] == "vkupno"].set_index("strana")
    assert tot.loc["nad", "linija"] == 237.5 and tot.loc["pod", "kvota"] == 1.93
    assert "НЕ е потврден" in df[df["pazar"] == "hendikep"]["napomena"].iat[0]
    pl = df[df["igrac"] == "CJ McCollum"]
    assert set(pl["linija"]) == {16.5} and pl["tim_igrac"].iat[0] == "ATL Hawks"  # the player's OWN team
    assert len(df) == 5                                                         # unknown code 999 ignored


def test_margin_and_devig():
    df = mark_main_lines(parse_mozzart(MOZ))
    m = margins(df)
    main = m[(m["glavna"]) & (m["pazar"] == "vkupno")]["marza"].iat[0]
    assert main == pytest.approx(2 / 1.90 - 1)                                  # 5.26 %
    assert devig_two_way(1.90, 1.90) == pytest.approx(0.5)
    assert devig_two_way(1.62, 2.16) == pytest.approx((1 / 1.62) / (1 / 1.62 + 1 / 2.16))


def test_movement_first_to_last_snapshot():
    a = mark_main_lines(parse_mozzart(MOZ))
    later = {**MOZ, "fetched_at": "2026-10-29T16:56:00+01:00",
             "odds": [{"id": 1, "kodds": {"a": _k("вп", "помалку", "169.5", "1.90"), "b": _k("вп", "повеќе", "169.5", "1.90")}}]}
    b = mark_main_lines(parse_mozzart(later))
    import pandas as pd
    mv = movement(pd.concat([a, b]))
    row = mv[(mv["pazar"] == "vkupno") & (mv["strana"] == "nad")].iloc[0]
    assert row["linija_prva"] == 168.5 and row["linija_posledna"] == 169.5 and row["pomestuvanje"] == 1.0


def test_closing_price_is_last_snapshot_before_tipoff(tmp_path, monkeypatch):
    import json as _json
    import pandas as pd
    from pipeline import clv
    rows = pd.DataFrame([
        {"kladilnica": "mozzart", "snimka": "2026-10-29T15:00:00+01:00", "nat_id": "7", "pazar": "poeni", "strana": "pod",
         "linija": 11.5, "kvota": 1.90, "pocetok_utc": "2026-10-29T19:00+00:00"},
        {"kladilnica": "mozzart", "snimka": "2026-10-29T19:30:00+01:00", "nat_id": "7", "pazar": "poeni", "strana": "pod",
         "linija": 11.5, "kvota": 1.75, "pocetok_utc": "2026-10-29T19:00+00:00"},
        {"kladilnica": "mozzart", "snimka": "2026-10-29T20:30:00+01:00", "nat_id": "7", "pazar": "poeni", "strana": "pod",
         "linija": 11.5, "kvota": 1.60, "pocetok_utc": "2026-10-29T19:00+00:00"},   # after tip-off → ignored
    ])
    monkeypatch.setattr(clv, "_day_lines", lambda root, day: rows if day == "2026-10-29" else pd.DataFrame())

    class S:
        class paths:
            raw = tmp_path
    pick = {"datum": "2026-10-29", "pazar": "poeni", "strana": "pod", "linija": 11.5,
            "vlezovi_json": _json.dumps({"kladilnica": "mozzart", "nat_ids": {"mozzart": "7"}})}
    k, snap = clv.closing_price(S, pick)
    assert k == 1.75 and snap.startswith("2026-10-29T19:30")             # 19:30 local = 18:30 UTC < 19:00 tip-off
    assert clv.closing_price(S, {**pick, "linija": 12.5}) is None          # our line no longer offered → no CLV


def test_devig_methods_power_and_shin_take_more_from_the_longshot():
    for m in ("proporcionalen", "power", "shin"):
        assert devig_two_way(1.90, 1.90, m) == pytest.approx(0.5)
        assert devig_two_way(1.30, 3.40, m) + devig_two_way(3.40, 1.30, m) == pytest.approx(1.0)
    prop, power, shin = (devig_two_way(1.30, 3.40, m) for m in ("proporcionalen", "power", "shin"))
    assert prop == pytest.approx((1 / 1.30) / (1 / 1.30 + 1 / 3.40))
    assert prop < shin < power                                  # favourite gets more under power/Shin
    # power: p = a^k with a^k + b^k = 1
    a, b = 1 / 1.30, 1 / 3.40
    k = math.log(power) / math.log(a)
    assert a ** k + b ** k == pytest.approx(1.0)
