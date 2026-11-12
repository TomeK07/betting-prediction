"""Статистички предикции без квоти (Томе 12.11)."""
from __future__ import annotations

import pandas as pd

from model.stats_preds import (
    EWMA_MIN,
    build_stat_preds,
    choose_secondary,
    format_stat_section,
    synthetic_line,
)


def test_synthetic_line_is_x5_not_book():
    assert synthetic_line(22.3) == 22.5
    assert synthetic_line(0.2) == 0.5
    assert synthetic_line(5.0) == 5.5


def test_choose_secondary_prefers_higher_dispersion():
    # lower r → more dispersion → prefer that market
    assert choose_secondary(3.0, 8.0, {"r_po_mu": [1, 1, 1, 1, 1]}, {"r_po_mu": [40, 40, 40, 40, 40]}) == "ast"
    assert choose_secondary(3.0, 8.0, {"r_po_mu": [40, 40, 40, 40, 40]}, {"r_po_mu": [1, 1, 1, 1, 1]}) == "reb"


def test_build_stat_preds_covers_ewma_floor_without_odds():
    players = pd.DataFrame([
        {"player_code": "1", "label": "Star A. (AAA)", "team": "AAA", "opp": "BBB", "game_code": "g1",
         "min_ewm": 28.0, "min_bez": 30.0, "out": False,
         "mu_pts": 22.0, "mu_ast": 5.0, "mu_reb": 4.0},
        {"player_code": "2", "label": "Bench B. (AAA)", "team": "AAA", "opp": "BBB", "game_code": "g1",
         "min_ewm": 10.0, "min_bez": 9.0, "out": False,
         "mu_pts": 4.0, "mu_ast": 1.0, "mu_reb": 2.0},
    ])
    logs = pd.DataFrame([
        {"player_code": "1", "game_code": f"x{i}", "game_date": "2026-10-0" + str(i + 1), "minutes": 30, "dnp": False}
        for i in range(6)
    ])
    disp = {"pts": 5.0, "ast": 3.0, "reb": 8.0}
    preds, odb = build_stat_preds("nba", "2026-11-12", players, logs, disp)
    codes = {p.player_code for p in preds}
    assert "1" in codes
    assert "2" not in codes
    assert any(o["label"].startswith("Bench") for o in odb)
    assert all(p.pazar in ("pts", "ast", "reb") for p in preds)
    # two markets per shown player (pts + secondary)
    assert sum(1 for p in preds if p.player_code == "1") == 2
    text = "\n".join(format_stat_section(preds, odb, "NBA"))
    assert "Сите активни играчи — статистички предикции" in text
    assert "Без Mozzart" in text
    assert EWMA_MIN["nba"] == 16.0
