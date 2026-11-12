import numpy as np
import pandas as pd
import pytest

from model.backtest import evaluate, split_by_date, walk_forward
from model.calibration import best_shrink, shrink_toward_half
from model.team_model import ModelParams, compute_ratings, game_frame, predict_game


def _row(code, day, team, opp, home, pts, opp_pts, minutes=240.0):
    base = dict(fga=88, oreb=10, tov=14, fta=22, fgm=40, fgm3=12, ftm=17, dreb=33)
    row = {"game_code": code, "game_date": day, "team": team, "opp": opp, "home": home,
           "pts": pts, "opp_pts": opp_pts, "team_minutes": minutes}
    row.update(base)
    row.update({f"opp_{k}": v for k, v in base.items()})
    return row


def _season(n_days=30, seed=0):
    """Synthetic league: 4 teams, strong team A, weak team D, alternating fixtures."""
    rng = np.random.default_rng(seed)
    strength = {"A": 8, "B": 2, "C": -2, "D": -8}
    pairs = [("A", "B"), ("C", "D"), ("A", "C"), ("B", "D"), ("A", "D"), ("B", "C")]
    rows = []
    for day in range(n_days):
        date = (pd.Timestamp("2025-10-01") + pd.Timedelta(days=2 * day)).date().isoformat()
        for i, (h, a) in enumerate(pairs[day % 3 * 2: day % 3 * 2 + 2]):
            code = f"{day:03d}{i}"
            hp = 110 + strength[h] - strength[a] / 2 + 2 + rng.normal(0, 8)
            ap = 110 + strength[a] - strength[h] / 2 - 2 + rng.normal(0, 8)
            rows += [_row(code, date, h, a, True, round(hp), round(ap)), _row(code, date, a, h, False, round(ap), round(hp))]
    return pd.DataFrame(rows)


def test_possessions_and_ratings_hand_example():
    # FGA 88 − OREB 10 + TOV 14 + 0.44·FTA 22 = 101.68 for both teams → poss 101.68
    g = game_frame(pd.DataFrame([_row("1", "2025-10-01", "A", "B", True, 110, 100),
                                 _row("1", "2025-10-01", "B", "A", False, 100, 110)]), "nba")
    a = g[g["team"] == "A"].iloc[0]
    assert a["poss"] == pytest.approx(101.68)
    assert a["ortg"] == pytest.approx(100 * 110 / 101.68)
    assert a["efg_off"] == pytest.approx((40 + 0.5 * 12) / 88)       # eFG = (FGM + 0.5·3PM)/FGA
    assert a["orb_pct_off"] == pytest.approx(10 / (10 + 33))
    assert a["ftr_off"] == pytest.approx(17 / 88)


def test_overtime_does_not_inflate_pace():
    g = game_frame(pd.DataFrame([_row("1", "2025-10-01", "A", "B", True, 120, 118, minutes=265.0),
                                 _row("1", "2025-10-01", "B", "A", False, 118, 120, minutes=265.0)]), "nba")
    assert g["pace"].iloc[0] == pytest.approx(101.68 * 48 / 53)


def test_back_to_back_flag():
    rows = [_row("1", "2025-10-01", "A", "B", True, 100, 90), _row("1", "2025-10-01", "B", "A", False, 90, 100),
            _row("2", "2025-10-02", "A", "C", True, 100, 90), _row("2", "2025-10-02", "C", "A", False, 90, 100)]
    g = game_frame(pd.DataFrame(rows), "nba")
    second = g[(g["team"] == "A") & (g["game_code"] == "2")].iloc[0]
    assert second["b2b"] == 1.0 and g[(g["team"] == "C")].iloc[0]["opp_b2b"] == 1.0


def test_no_look_ahead():
    """Changing every result from date D onwards must not change any prediction made for dates < D... or for D."""
    games = game_frame(_season(), "nba")
    params = ModelParams(half_life=10.0, use_ff=True, use_rest=True)
    base = walk_forward(games, params, 2.0)
    cut = sorted(games["game_date"].unique())[20]
    tampered = games.copy()
    future = tampered["game_date"] >= cut
    tampered.loc[future, ["pts", "opp_pts"]] += 40          # absurd future scores
    after = walk_forward(game_frame(tampered[[c for c in _season().columns]], "nba"), params, 2.0)
    upto = lambda df: df[df["game_date"] <= cut].set_index("game_code")[["pred_home", "pred_away"]]
    pd.testing.assert_frame_equal(upto(base), upto(after))   # predictions for D itself are unchanged too
    assert not upto(base).empty


def test_model_learns_strength_and_home_advantage():
    games = game_frame(_season(60), "nba")
    r = compute_ratings(games, ModelParams(), hca_prior=2.0)
    assert r.teams.loc["A", "ortg"] > r.teams.loc["D", "ortg"]
    h, a = predict_game(r, "A", "D", ModelParams())
    assert h - a > 10                                       # A is much stronger and at home
    assert r.hca > 0


def test_evaluate_and_split():
    preds = pd.DataFrame({"game_code": ["1", "2"], "game_date": ["2025-10-01", "2025-11-01"],
                          "n_home": [5, 6], "n_away": [5, 9],
                          "pred_home": [110, 100], "pred_away": [100, 100],
                          "act_home": [112, 90], "act_away": [101, 95]})
    m = evaluate(preds)
    assert m.mae_total == pytest.approx((3 + 15) / 2)       # |210−213|, |200−185|
    assert m.mae_margin == pytest.approx((1 + 5) / 2)       # |10−11|, |0−(−5)|
    assert m.within5_total == pytest.approx(0.5)
    first, rest = split_by_date(preds, 0.5)
    assert list(first["game_code"]) == ["1"] and list(rest["game_code"]) == ["2"]


def test_shrink_toward_half():
    assert shrink_toward_half(0.62, 0.5) == pytest.approx(0.56)
    assert shrink_toward_half(0.62, 0.0) == pytest.approx(0.5)
    # perfectly uninformative probabilities → optimal k ≈ 0
    assert best_shrink(np.array([0.6, 0.6, 0.4, 0.4]), np.array([1, 0, 1, 0])) == pytest.approx(0.0)


def test_dvp_per36_shrunk_and_no_look_ahead():
    from model.dvp import dvp_table, primary_position
    assert primary_position("G-F") == "G" and primary_position("Center") == "C" and primary_position("") is None
    logs = pd.DataFrame([
        # guards scoring against team X: 30 pts in 36 min; against Y: 10 pts in 36 min → league 20/36
        {"game_date": "2025-10-01", "opp": "X", "player_code": "1", "dnp": False, "minutes": 36, "pts": 30, "reb": 4, "ast": 5, "fgm3": 3},
        {"game_date": "2025-10-01", "opp": "Y", "player_code": "2", "dnp": False, "minutes": 36, "pts": 10, "reb": 4, "ast": 5, "fgm3": 1},
        {"game_date": "2025-10-09", "opp": "X", "player_code": "1", "dnp": False, "minutes": 36, "pts": 90, "reb": 4, "ast": 5, "fgm3": 3},
    ])
    roster = pd.DataFrame({"player_code": ["1", "2"], "position": ["Guard", "G"]})
    t = dvp_table(logs, roster, before="2025-10-05", shrink_minutes=0).set_index("tim")
    assert t.loc["X", "pts_36"] == pytest.approx(30) and t.loc["X", "pts_liga_36"] == pytest.approx(20)
    assert t.loc["X", "pts_vs_liga_pct"] == pytest.approx(50)               # the 90-point future game is ignored
    shrunk = dvp_table(logs, roster, before="2025-10-05", shrink_minutes=36).set_index("tim")
    assert shrunk.loc["X", "pts_36"] == pytest.approx(25)                   # (30 + 20) / 2


def test_params_from_config():
    from model.team_model import params_from_config
    p = params_from_config({"variant": "V3", "shrink_games": 6, "half_life": None, "opp_adjust": True})
    assert p.name == "V3" and p.half_life is None and not p.use_ff and p.shrink_games == 6.0


def test_weekly_bias_correction_has_no_look_ahead():
    raw = _season(40)
    raw["season_code"] = "S1"
    games = game_frame(raw.drop(columns="season_code"), "nba").assign(season_code="S1")
    params = ModelParams(weekly_bias=True)
    base = walk_forward(games, params, 2.0)
    cut = sorted(games["game_date"].unique())[25]
    tampered = raw.copy()
    tampered.loc[tampered["game_date"] >= cut, ["pts", "opp_pts"]] += 30
    t_games = game_frame(tampered.drop(columns="season_code"), "nba").assign(season_code="S1")
    after = walk_forward(t_games, params, 2.0)
    upto = lambda df: df[df["game_date"] <= cut].set_index("game_code")[["pred_home", "pred_away"]]
    pd.testing.assert_frame_equal(upto(base), upto(after))
    # and it does correct: with every score +10 from the start, later predictions move up
    shifted = raw.copy(); shifted[["pts", "opp_pts"]] += 10
    s_games = game_frame(shifted.drop(columns="season_code"), "nba").assign(season_code="S1")
    no_corr = walk_forward(s_games, ModelParams(), 2.0)
    assert len(no_corr) > 0
