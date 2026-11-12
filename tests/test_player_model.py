import numpy as np
import pandas as pd
import pytest

from model.player_model import (PlayerParams, add_features, dvp_by_game, fit_dispersion, new_absences, p_over,
                                prepare, project)


def _logs():
    """Team T: star S (35 min) plays games 1-3, misses game 4; guard G and forward F play every game."""
    rows = []
    for i, day in enumerate(["2025-10-01", "2025-10-03", "2025-10-05", "2025-10-07"], start=1):
        code = str(i)
        players = [("G", 30, 10, 5, 3, 1), ("F", 25, 8, 2, 6, 0)]
        if i < 4:
            players.append(("S", 35, 25, 4, 7, 2))
        for pid, mins, pts, ast, reb, fg3 in players:
            rows.append({"season_code": "X", "game_date": day, "game_code": code, "team": "T", "opp": "O",
                         "player_code": pid, "dnp": False, "minutes": float(mins), "pts": pts, "ast": ast,
                         "reb": reb, "fgm3": fg3})
    roster = pd.DataFrame({"player_code": ["G", "F", "S"], "position": ["Guard", "Forward", "Guard"]})
    return prepare(pd.DataFrame(rows), roster)


def test_features_use_only_earlier_games():
    d = add_features(_logs(), "euroleague", PlayerParams())
    g = d[d["player_code"] == "G"].sort_values("game_date")
    assert np.isnan(g["min_last5"].iloc[0])                      # nothing known before the first game
    assert g["min_last5"].iloc[1] == pytest.approx(30)
    # change a LATER game: earlier features must not move
    logs = _logs()
    logs.loc[(logs["player_code"] == "G") & (logs["game_code"] == "4"), ["minutes", "pts"]] = [5.0, 99]
    d2 = add_features(logs, "euroleague", PlayerParams())
    pd.testing.assert_series_equal(
        d[d["game_code"] != "4"].set_index(["game_code", "player_code"])["pts_rate"].sort_index(),
        d2[d2["game_code"] != "4"].set_index(["game_code", "player_code"])["pts_rate"].sort_index())
    # and the features OF game 4 itself do not see game 4
    row = lambda df: df[(df["player_code"] == "G") & (df["game_code"] == "4")].iloc[0]
    assert row(d)["min_ewm"] == pytest.approx(row(d2)["min_ewm"])


def test_new_absence_detected_and_minutes_redistributed_to_same_position():
    d = add_features(_logs(), "euroleague", PlayerParams())
    absent = new_absences(d)
    assert list(absent["player_code"]) == ["S"] and list(absent["game_code"]) == ["4"]
    proj = project(d, "euroleague", PlayerParams(same_pos_share=0.7))
    g4 = proj[(proj["game_code"] == "4")].set_index("player_code")
    freed = absent["min_ewm_after"].iloc[0]                       # 35
    # guard G: 70% (same position, only guard left) + his share of the other 30%
    share_g = 30 / 55
    # 30 + 24.5 + 5.7 = 60.2 → capped at 38 minutes (EuroLeague cap)
    assert g4.loc["G", "min_m3"] == pytest.approx(min(30 + 0.7 * freed + 0.3 * freed * share_g, 38.0))
    assert g4.loc["F", "min_m3"] == pytest.approx(25 + 0.3 * freed * (25 / 55))
    assert g4.loc["G", "proj_pts"] > g4.loc["G", "min_m1"] * g4.loc["G", "pts_rate"]   # more minutes + usage


def test_dvp_by_game_uses_earlier_dates_only():
    d = _logs()
    dv = dvp_by_game(d)
    first = dv[dv["game_code"] == "1"]
    assert first["pts_vs_liga_pct"].isna().all() or (first["pts_vs_liga_pct"].fillna(0) == 0).all()


def test_negative_binomial_dispersion_and_p_over():
    rng = np.random.default_rng(1)
    mu = np.full(20000, 6.0)
    r_true = 4.0
    y = rng.negative_binomial(r_true, r_true / (r_true + mu))
    assert fit_dispersion(mu, y) == pytest.approx(r_true, rel=0.1)
    # P(X > 5.5) for Poisson(6) = 1 − P(X ≤ 5) ≈ 0.5543
    assert p_over(np.array([6.0]), np.array([5.5]), 1e6)[0] == pytest.approx(0.5543, abs=1e-3)
    # more dispersion → probability mass spreads → P(over the mean-ish line) drops
    assert p_over(np.array([6.0]), np.array([5.5]), 2.0)[0] < 0.5543


def test_dispersion_by_mean_range_finds_wider_tail_for_small_means():
    from model.player_model import dispersion_for, fit_dispersion_by_mu
    rng = np.random.default_rng(3)
    mu = rng.uniform(0.3, 6.0, 20000)
    r_true = np.where(mu < 2.0, 1.0, 20.0)                        # small means much more dispersed
    y = rng.negative_binomial(r_true, r_true / (r_true + mu))
    rb = fit_dispersion_by_mu(mu, y, [1.5, 2.5])
    assert rb[0] <= 2.0 and rb[1] <= 2.0 and rb[3] >= 8.0
    assert list(dispersion_for(np.array([0.5, 1.5, 2.5, 4.0, 7.0]), {"r_po_mu": [1, 2, 3, 4, 5]})) == [1, 2, 3, 4, 5]
    # vector r in p_over == scalar r row by row
    assert p_over(np.array([1.0, 3.0]), np.array([1.5, 2.5]), np.array([2.0, 9.0]))[1] == pytest.approx(
        p_over(np.array([3.0]), np.array([2.5]), 9.0)[0])
