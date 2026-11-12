import pytest

from model import nba_logs as nl

TZ = "Europe/Skopje"


def _game(gid, utc, est_date, status=3, home=("DET", 1), away=("BOS", 2), hs=121, as_=113):
    team = lambda code, tid, score: {"teamId": tid, "teamTricode": code, "teamCity": code, "teamName": "X",
                                     "score": score}
    return {"gameId": gid, "gameStatus": status, "gameDateTimeUTC": utc, "gameDateEst": f"{est_date}T00:00:00Z",
            "gameLabel": "", "postponedStatus": "N",
            "homeTeam": team(*home, hs), "awayTeam": team(*away, as_)}


def _schedule(games):
    return nl.schedule_frame({"leagueSchedule": {"gameDates": [{"games": games}]}}, TZ, ["002", "006"])


def _stats(pts, fgm=40, fga=88, fg3m=15, fg3a=40, minutes="240:00"):
    return {"minutes": minutes, "points": pts, "fieldGoalsMade": fgm, "fieldGoalsAttempted": fga,
            "threePointersMade": fg3m, "threePointersAttempted": fg3a, "freeThrowsMade": 10,
            "freeThrowsAttempted": 12, "reboundsOffensive": 10, "reboundsDefensive": 30, "reboundsTotal": 40,
            "assists": 25, "steals": 7, "turnovers": 13, "blocks": 5, "foulsPersonal": 20, "plusMinusPoints": 0}


def _player(pid, first, last, minutes, position="", comment="", pts=0):
    st = _stats(pts, 0, 0, 0, 0, minutes)
    return {"personId": pid, "firstName": first, "familyName": last, "position": position,
            "comment": comment, "jerseyNum": "1", "statistics": st}


def _box(minutes="240:00"):
    return {"boxScoreTraditional": {
        "homeTeam": {"teamTricode": "DET", "statistics": _stats(121, minutes=minutes), "players": [
            _player(1631, "Ausar", "Thompson", "33:15", "F", pts=11),
            _player(1632, "Taurean", "Prince", "", comment="DNP - Coach's Decision"),
            _player(1633, "Ugonna", "Onyenso", "", comment="Inactive - G League - Two-Way")]},
        "awayTeam": {"teamTricode": "BOS", "statistics": _stats(113, minutes=minutes), "players": [
            _player(1640, "Jaylen", "Brown", "38:00", "G", pts=30)]},
    }}


def test_schedule_drops_preseason_and_uses_us_date():
    sched = _schedule([
        _game("0012600001", "2026-10-05T23:00:00Z", "2026-10-05"),               # preseason -> dropped
        _game("0022600010", "2026-10-29T23:30:00Z", "2026-10-29", status=1),     # 19:30 ET
        _game("0022600011", "2026-10-30T02:00:00Z", "2026-10-29", status=1),     # 22:00 ET, 03:00 Skopje
    ])
    assert list(sched["game_code"]) == ["0022600010", "0022600011"]
    late = sched.iloc[1]
    assert late["game_date"] == "2026-10-29"      # NBA/Mozzart day
    assert late["date_local"] == "2026-10-30"     # Skopje calendar day
    assert late["time_local"] == "03:00"
    assert late["status"] == "scheduled" and late["home_score"] is None


def test_parse_boxscore_minutes_starters_and_absences():
    game = _schedule([_game("0022600001", "2026-10-20T23:30:00Z", "2026-10-20")]).iloc[0].to_dict()
    teams, players = nl.parse_boxscore(_box(), game)
    det = next(t for t in teams if t["team"] == "DET")
    assert det["home"] and det["opp"] == "BOS" and det["pts"] == 121 and det["opp_pts"] == 113
    # 2-pointers derived: 40 FGM − 15 3PM = 25
    assert det["fgm2"] == 25 and det["fga2"] == 48 and det["overtimes"] == 0
    thompson = next(p for p in players if p["player_code"] == "1631")
    assert thompson["label"] == "Thompson A. (DET)" and thompson["starter"] and not thompson["dnp"]
    assert thompson["minutes"] == pytest.approx(33.25)
    prince = next(p for p in players if p["player_code"] == "1632")
    assert prince["dnp"] and prince["minutes"] == 0.0 and not prince["starter"]
    assert prince["absence_reason"] == "DNP - Coach's Decision"
    assert next(p for p in players if p["player_code"] == "1633")["absence_reason"].startswith("Inactive")


def test_overtime_nba():
    game = _schedule([_game("0022600001", "2026-10-20T23:30:00Z", "2026-10-20")]).iloc[0].to_dict()
    teams, _ = nl.parse_boxscore(_box(minutes="265:00"), game)
    assert teams[0]["overtimes"] == 1


def test_logs_sorted_by_date():
    sched = _schedule([_game("0022600009", "2026-10-25T23:30:00Z", "2026-10-25"),
                       _game("0022600001", "2026-10-20T23:30:00Z", "2026-10-20")])
    team_logs, player_logs = nl.build_logs(sched, {"0022600009": _box(), "0022600001": _box()})
    assert list(team_logs.drop_duplicates("game_code")["game_code"]) == ["0022600001", "0022600009"]
    assert player_logs["date_utc"].is_monotonic_increasing


def test_team_mismatch_raises():
    game = _schedule([_game("0022600001", "2026-10-20T23:30:00Z", "2026-10-20", away=("NYK", 3))]).iloc[0].to_dict()
    with pytest.raises(ValueError):
        nl.parse_boxscore(_box(), game)


def test_stat_table_maps_team_id_to_tricode_and_drops_ranks():
    payload = {"resultSets": [{"headers": ["TEAM_ID", "TEAM_NAME", "PACE", "PACE_RANK"],
                               "rowSet": [[1, "Detroit Pistons", 99.5, 12]]}]}
    df = nl.stat_table_frame(payload, "teams", {1: "DET"})
    assert df.loc[0, "team"] == "DET" and "PACE_RANK" not in df.columns
