"""Walk-forward backtest of the team model — no look-ahead.

For every game date D the ratings are rebuilt from games with game_date < D only; games on D are
predicted, and only then do they become history. Season averages that include the predicted game
can therefore never leak in (the error Tome warned about).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import pandas as pd

from model.team_model import ModelParams, compute_ratings, predict_game, week_start

MIN_HISTORY_ROWS = 20          # team-game rows needed before the first prediction of a season
REST_MIN_ROWS = 150            # residual rows needed before the rest effect is fitted


def walk_forward(games: pd.DataFrame, params: ModelParams, hca_prior: float,
                 only_dates: Optional[set] = None) -> pd.DataFrame:
    """One prediction per game (home perspective). `games` = model.team_model.game_frame output.
    `only_dates`: predict just these days (history before them is still used in full)."""
    out, resid_rows = [], []
    season_resid: list[tuple] = []      # (date, season, residual points of one team) for the weekly bias
    for day in sorted(games["game_date"].unique()):
        if only_dates is not None and day not in only_dates and not params.use_rest:
            continue
        past = games[games["game_date"] < day]
        today = games[(games["game_date"] == day) & games["home"]]
        if len(past) < MIN_HISTORY_ROWS or today.empty:
            continue
        season = today["season_code"].iat[0] if "season_code" in today and len(today) else None
        ratings = compute_ratings(past, params, hca_prior, season=season, as_of=day)
        coef = _rest_coef(resid_rows) if params.use_rest else None
        n_games = past.groupby("team").size()
        bias = 0.0
        if params.weekly_bias and season is not None:
            ws = week_start(day)
            r = [x for d_, s_, x in season_resid if s_ == season and d_ < ws]
            bias = sum(r) / (len(r) + params.bias_prior_games) if r else 0.0
        for g in today.itertuples():
            base_h, base_a = predict_game(ratings, g.team, g.opp, params)
            rest = (0.0, 0.0)
            # features of a team row: own B2B, own away-away B2B, opponent B2B, opponent away-away B2B
            f_home = np.array([g.b2b, g.b2b_aa, g.opp_b2b, g.opp_b2b_aa], float)
            f_away = np.array([g.opp_b2b, g.opp_b2b_aa, g.b2b, g.b2b_aa], float)
            if coef is not None:
                rest = (float(coef @ f_home), float(coef @ f_away))
            out.append({"game_code": g.game_code, "game_date": day, "home": g.team, "away": g.opp,
                        "n_home": int(n_games.get(g.team, 0)), "n_away": int(n_games.get(g.opp, 0)),
                        "pred_home": base_h + rest[0] + bias, "pred_away": base_a + rest[1] + bias,
                        "act_home": g.pts, "act_away": g.opp_pts})
            # residuals of the rest-free projection feed the rest fit for LATER dates only
            resid_rows.append((g.pts - base_h, *f_home))
            resid_rows.append((g.opp_pts - base_a, *f_away))
            season_resid.append((day, season, g.pts - base_h))
            season_resid.append((day, season, g.opp_pts - base_a))
    return pd.DataFrame(out)


def _rest_coef(rows: list) -> Optional[np.ndarray]:
    """Points effect of each back-to-back category, fitted on residuals of earlier games only.

    Category means (not a linear "days of rest" slope): the residual of rows with the flag minus
    the residual of rows without it, shrunk toward 0 with 50 pseudo-rows (rare categories stay small).
    """
    if len(rows) < REST_MIN_ROWS:
        return None
    arr = np.asarray(rows, float)
    y, X = arr[:, 0], arr[:, 1:]
    coef = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        on = X[:, j] == 1
        if on.sum() == 0:
            continue
        diff = y[on].mean() - y[~on].mean()
        coef[j] = diff * on.sum() / (on.sum() + 50.0)
    # away-away is a subset of B2B: keep only its extra effect on top of plain B2B
    coef[1] -= coef[0] * (coef[1] != 0)
    coef[3] -= coef[2] * (coef[3] != 0)
    return coef


def rest_effects(preds_resid: list) -> Optional[np.ndarray]:
    """Public wrapper for reports (coefficients from a list of residual rows)."""
    return _rest_coef(preds_resid)


@dataclass
class Metrics:
    n: int
    mae_total: float
    mae_margin: float
    mae_team: float
    bias_total: float
    within5_total: float
    within5_margin: float
    within5_team: float


def evaluate(preds: pd.DataFrame, burn_in: int = 5) -> Optional[Metrics]:
    p = preds[(preds["n_home"] >= burn_in) & (preds["n_away"] >= burn_in)]
    if p.empty:
        return None
    err_total = (p["pred_home"] + p["pred_away"]) - (p["act_home"] + p["act_away"])
    err_margin = (p["pred_home"] - p["pred_away"]) - (p["act_home"] - p["act_away"])
    err_team = pd.concat([p["pred_home"] - p["act_home"], p["pred_away"] - p["act_away"]])
    return Metrics(
        n=len(p),
        mae_total=float(err_total.abs().mean()), mae_margin=float(err_margin.abs().mean()),
        mae_team=float(err_team.abs().mean()), bias_total=float(err_total.mean()),
        within5_total=float((err_total.abs() <= 5).mean()), within5_margin=float((err_margin.abs() <= 5).mean()),
        within5_team=float((err_team.abs() <= 5).mean()),
    )


def split_by_date(preds: pd.DataFrame, share: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """First `share` of the season's dates (tuning) vs the rest (out-of-sample evaluation)."""
    dates = sorted(preds["game_date"].unique())
    cut = dates[int(len(dates) * share)] if dates else None
    return preds[preds["game_date"] < cut], preds[preds["game_date"] >= cut]


def tune(games: pd.DataFrame, base: ModelParams, hca_prior: float, share: float = 0.6,
         shrink_grid=(3.0, 6.0, 10.0), half_life_grid=(None, 10.0, 20.0, 40.0)) -> tuple[ModelParams, pd.DataFrame]:
    """Pick shrinkage and recency on the tuning part only (MAE total + MAE margin)."""
    rows = []
    for k in shrink_grid:
        for hl in half_life_grid:
            params = replace(base, shrink_games=k, half_life=hl)
            tune_part, _ = split_by_date(walk_forward(games, params, hca_prior), share)
            m = evaluate(tune_part)
            if m:
                rows.append({"shrink_games": k, "half_life": hl, "mae_total": m.mae_total,
                             "mae_margin": m.mae_margin, "score": m.mae_total + m.mae_margin})
    grid = pd.DataFrame(rows).sort_values("score")
    best = grid.iloc[0]
    hl = None if pd.isna(best["half_life"]) else float(best["half_life"])
    return replace(base, shrink_games=float(best["shrink_games"]), half_life=hl), grid
