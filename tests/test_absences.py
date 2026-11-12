import pandas as pd
import pytest

from model import absences as ab
from model.euroleague_logs import player_label
from model.injury_nba import parse_words, split_name
from model.player_model import PlayerParams


def _w(text, x, top):
    return {"text": text, "x0": x, "top": top}


HEADER = [_w("Game", 23, 100), _w("Date", 53, 100), _w("Game", 119, 100), _w("Time", 149, 100),
          _w("Matchup", 200, 100), _w("Team", 264, 100), _w("Player", 425, 100), _w("Name", 457, 100),
          _w("Current", 585, 100), _w("Status", 624, 100), _w("Reason", 666, 100)]


def test_injury_pdf_words_carry_forward_wrapped_reason_and_footer():
    page = HEADER + [
        _w("10/29/2026", 24, 120), _w("08:00", 120, 120), _w("(ET)", 145, 120), _w("SAC@NOP", 201, 120),
        _w("Sacramento", 265, 120), _w("Kings", 316, 120), _w("Murray,", 426, 120), _w("Keegan", 461, 120),
        _w("Out", 587, 120), _w("Injury/Illness", 667, 120),
        _w("Right", 667, 135),                                   # wrapped reason, line above the next player
        _w("Simmons,", 426, 142), _w("Ben", 470, 142), _w("Out", 587, 142),
        _w("Achilles", 667, 150),                                # …and below
        _w("New", 265, 165), _w("Orleans", 290, 165), _w("Pelicans", 330, 165),
        _w("NOT", 667, 165), _w("YET", 687, 165), _w("SUBMITTED", 704, 165),
        _w("Page", 380, 560), _w("1", 400, 560), _w("of", 410, 560), _w("2", 420, 560),
    ]
    df = parse_words([page])
    assert list(df["player"].fillna("-")) == ["Murray, Keegan", "Simmons, Ben", "-"]
    assert df.iloc[1]["team"] == "Sacramento Kings" and df.iloc[1]["matchup"] == "SAC@NOP"   # carried forward
    assert "Right" in df.iloc[1]["reason"] and "Achilles" in df.iloc[1]["reason"]
    assert df.iloc[2]["status"] == "NOT_SUBMITTED" and df.iloc[2]["team"] == "New Orleans Pelicans"
    assert df.iloc[0]["game_date"] == "2026-10-29"


def test_split_name_and_roman_numerals():
    assert split_name("Porter Jr., Michael") == ("Porter Jr.", "Michael")
    assert player_label("CLARK III, JIMMY", "TEL") == "Clark III J. (TEL)"
    assert player_label("IVANOVIC, IVAN", "RED") == "Ivanovic I. (RED)"     # 'Iv' inside a word stays


def _logs():
    rows = []
    # team T played games 1..5; star S played 1-4, missed 5; old O played only game 1
    for i in range(1, 6):
        day = f"2026-10-0{i}"
        for pid, mins in (("G", 30), ("F", 25), ("S", 35), ("O", 20)):
            if (pid == "S" and i == 5) or (pid == "O" and i > 1):
                continue
            rows.append({"game_code": str(i), "game_date": day, "date_utc": day + "T18:00Z", "team": "T", "opp": "X",
                         "player_code": pid, "minutes": float(mins), "dnp": False, "pts": 10, "ast": 2, "reb": 3, "fgm3": 1,
                         "season_code": "S", "label": f"{pid} (T)"})
    return pd.DataFrame(rows)


def test_new_vs_old_absence():
    logs = _logs()
    assert ab.classify("S", "T", logs, "2026-10-09")["tip"] == "НОВО"     # played games 2–4 of the last 3
    old = ab.classify("O", "T", logs, "2026-10-09")
    assert old["tip"] == "СТАРО" and old["minuti_sezona"] == 20.0


def test_status_policy_by_league():
    assert ab.plays("Out", "nba") is False and ab.plays("Doubtful", "nba") is False
    assert ab.plays("Questionable", "nba") is None                       # ЗАВИСИ_ОД_СОСТАВ
    assert ab.plays("Questionable", "euroleague") is True               # error 27.10
    assert ab.plays("Probable", "nba") is True


def test_match_player_rule_2():
    nba = pd.DataFrame({"player_code": ["1", "2", "3"], "team": ["SAC", "SAC", "HOU"],
                        "player": ["Keegan Murray", "Kris Murray", "Keegan Murray"]})
    assert ab.match_player("Murray", "Keegan", "SAC", nba) == ("1", "")
    assert ab.match_player("Murray", "", "SAC", nba)[0] is None           # surname alone is never enough
    el = pd.DataFrame({"player_code": ["9"], "team": ["PRS"], "player": ["ETIENNE, TYSON"]})
    assert ab.match_player("Etienne", "Tyson", "PRS", el) == ("9", "")


def test_last_game_absences_and_effect_keeps_team_minutes():
    logs = _logs()
    roster = pd.DataFrame({"player_code": ["G", "F", "S", "O"], "position": ["Guard", "Forward", "Guard", "Center"],
                           "label": ["G (T)", "F (T)", "S (T)", "O (T)"]})
    miss = ab.last_game_absences(logs, roster, "T", "2026-10-09", "euroleague")
    assert set(miss["player_code"]) == {"S", "O"}
    eff = ab.pregame_minutes(logs, roster, "euroleague", "T", "2026-10-09", {"S"}, PlayerParams()).set_index("player_code")
    # S out → G, F and O (O played within the last 10) are scaled toward 200 min and all hit the 38-min cap
    assert eff.loc["S", "min_bez"] == 0
    assert all(eff.loc[p, "min_bez"] == pytest.approx(38.0) for p in ("G", "F", "O"))
    assert (eff.loc[["G", "F", "O"], "efekt_min"] >= 0).all()
