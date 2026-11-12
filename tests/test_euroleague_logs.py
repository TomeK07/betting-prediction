import pandas as pd
import pytest

from model import euroleague_logs as el


def _player(pid, team, name, minutes, pts=0, ast=0, starter=0):
    return {"Player_ID": f"P{pid}   ", "IsStarter": starter, "Team": team, "Dorsal": "1",
            "Player": name, "Minutes": minutes, "Points": pts, "Assistances": ast,
            "FieldGoalsMade2": 0, "FieldGoalsAttempted2": 0, "FieldGoalsMade3": 0,
            "FieldGoalsAttempted3": 0, "FreeThrowsMade": 0, "FreeThrowsAttempted": 0,
            "OffensiveRebounds": 0, "DefensiveRebounds": 0, "TotalRebounds": 0, "Steals": 0,
            "Turnovers": 0, "BlocksFavour": 0, "BlocksAgainst": 0, "FoulsCommited": 0,
            "FoulsReceived": 0, "Valuation": 0, "Plusminus": 0}


def _side(team, players, pts, minutes="200:00"):
    tot = {k: 0 for k in el.BOX_FIELDS}
    tot.update({"Minutes": minutes, "Points": pts})
    tmr = {"OffensiveRebounds": 1, "DefensiveRebounds": 2, "Turnovers": 1}
    return {"Team": team, "Coach": "X", "PlayersStats": players, "totr": tot, "tmr": tmr}


def _box(home_pts=80, away_pts=75, minutes="200:00"):
    return {"Live": False, "Stats": [
        _side("RED", [_player("011157", "RED", "MOTLEY, JOHNATHAN", "14:08", pts=10, starter=1),
                      _player("014094", "RED", "MILJENOVIC, STEFAN", "DNP")], home_pts, minutes),
        _side("ZAL", [_player("002222", "ZAL", "WEILER-BABB, NICK", "30:30", pts=20, ast=5)],
              away_pts, minutes),
    ]}


def _schedule(rows):
    games = [{"code": c, "season": {"code": "E2026"}, "round": {"round": r}, "phaseType": {"code": "RS"},
              "date": d, "status": "result",
              "home": {"code": "RED", "name": "Zvezda", "score": 80},
              "away": {"code": "ZAL", "name": "Zalgiris", "score": 75}} for c, r, d in rows]
    return el.schedule_frame(games, "Europe/Skopje")


@pytest.mark.parametrize("raw,expected", [
    ("14:08", 14 + 8 / 60), ("00:20", 20 / 60), ("200:00", 200.0), ("DNP", 0.0), (None, None), ("abc", None),
])
def test_parse_minutes(raw, expected):
    assert el.parse_minutes(raw) == (pytest.approx(expected) if expected is not None else None)


def test_player_label_is_surname_initial_team():
    assert el.player_label("MOTLEY, JOHNATHAN", "RED") == "Motley J. (RED)"
    assert el.player_label("WEILER-BABB, NICK", "ZAL") == "Weiler-Babb N. (ZAL)"
    # Error 16.10: same surname, different initial must give different labels
    assert el.player_label("EDWARDS, CARSEN", "MUN") != el.player_label("EDWARDS, KESSLER", "MUN")


def test_normalize_player_code():
    assert el.normalize_player_code("P011157   ") == "011157"
    assert el.normalize_player_code("004194") == "004194"


def test_local_date_crosses_midnight():
    # 22:30 UTC on 29.10 is 23:30 in Skopje (CET, after DST ended 25.10) — same day
    assert el.local_datetime("2026-10-29T22:30:00.000Z", "Europe/Skopje").date().isoformat() == "2026-10-29"
    # 23:30 UTC is already 30.10 in Skopje
    assert el.local_datetime("2026-10-29T23:30:00.000Z", "Europe/Skopje").date().isoformat() == "2026-10-30"


def test_parse_boxscore_team_and_player_rows():
    game = _schedule([(1, 1, "2026-09-24T18:00:00.000Z")]).iloc[0].to_dict()
    teams, players = el.parse_boxscore(_box(), game)
    red = next(t for t in teams if t["team"] == "RED")
    assert red["home"] is True and red["opp"] == "ZAL"
    assert red["pts"] == 80 and red["opp_pts"] == 75
    assert red["overtimes"] == 0 and red["team_tov"] == 1
    dnp = next(p for p in players if p["player_code"] == "014094")
    assert dnp["dnp"] is True and dnp["minutes"] == 0.0
    assert red["fgm"] == red["fgm2"] + red["fgm3"]  # common schema has totals too
    motley = next(p for p in players if p["player_code"] == "011157")
    assert motley["label"] == "Motley J. (RED)" and motley["starter"] is True
    assert motley["minutes"] == pytest.approx(14.1333, abs=1e-3)


def test_overtime_detected():
    game = _schedule([(1, 1, "2026-09-24T18:00:00.000Z")]).iloc[0].to_dict()
    teams, _ = el.parse_boxscore(_box(minutes="225:00"), game)
    assert teams[0]["overtimes"] == 1


def test_boxscore_team_mismatch_raises():
    game = _schedule([(1, 1, "2026-09-24T18:00:00.000Z")]).iloc[0].to_dict()
    game["away"] = "PAR"
    with pytest.raises(ValueError):
        el.parse_boxscore(_box(), game)


def test_logs_sorted_by_date_not_by_round_or_code():
    # Feed order and game codes deliberately out of date order (a postponed round-1 game played last)
    sched = _schedule([(5, 1, "2026-10-20T18:00:00.000Z"),
                       (9, 3, "2026-10-01T18:00:00.000Z"),
                       (7, 2, "2026-09-25T18:00:00.000Z")])
    team_logs, player_logs = el.build_logs(sched, {5: _box(), 9: _box(), 7: _box()})
    assert list(team_logs.drop_duplicates("game_code")["game_code"]) == [7, 9, 5]
    assert player_logs["date_utc"].is_monotonic_increasing


def test_last_n_games_takes_most_recent_by_date_and_skips_dnp():
    # Error 27.10: rounds 1–5 were taken instead of the latest games
    logs = pd.DataFrame({
        "game_code": [1, 2, 3, 4, 5, 6],
        "date_utc": ["2026-10-0%dT18:00:00Z" % d for d in (1, 2, 3, 4, 5, 6)],
        "game_date": ["2026-10-0%d" % d for d in (1, 2, 3, 4, 5, 6)],
        "dnp": [False, False, False, False, True, False],
        "pts": [1, 2, 3, 4, 5, 6],
    }).sample(frac=1, random_state=0)
    last3 = el.last_n_games(logs, 3)
    assert list(last3["game_code"]) == [6, 4, 3]
    assert list(el.last_n_games(logs, 2, before="2026-10-04")["game_code"]) == [3, 2]
    assert list(el.last_n_games(logs, 2, include_dnp=True)["game_code"]) == [6, 5]


def test_stat_table_percent_strings_become_numbers():
    payload = {"total": 1, "teams": [{"teamRanking": 1, "team": {"code": "PAR", "name": "Partizan"},
                                      "gamesPlayed": 8.0, "twoPointersPercentage": "53.7%"}]}
    df = el.stat_table_frame(payload, "teams")
    assert df.loc[0, "twoPointersPercentage"] == pytest.approx(53.7)
    assert df.loc[0, "team"] == "PAR"


def test_roster_keeps_players_only():
    people = [
        {"type": "J", "positionName": "Guard", "dorsal": "8", "active": True,
         "person": {"code": "015400", "name": "FRANKS, ROBERT", "height": 203}, "club": {"code": "PAR"}},
        {"type": "E", "positionName": "", "person": {"code": "CTB", "name": "COLLET, VINCENT"},
         "club": {"code": "PAR"}},
    ]
    df = el.roster_frame(people)
    assert list(df["player_code"]) == ["015400"]
    assert df.loc[0, "position"] == "Guard" and df.loc[0, "label"] == "Franks R. (PAR)"
