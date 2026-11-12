"""Candidate parameters (decision 29.10) — tested walk-forward on last season, like the new-absence test.

A) FIRST SCORER when a star is missing: a NEW absence of a player with usage ≥ 25% → does the first
   remaining scorer get more points/shots than M2 predicts? Candidate: + c × the absent star's
   expected points (c fitted on the tuning part).
B) TEAMMATE RETURNS from injury: players who covered his minutes ("coverers": their minutes rose
   ≥ 2 during his absence) — do they lose more minutes than M2 predicts? Candidate: move their
   minutes back toward the pre-absence level by α (fitted on the tuning part). A possible UNDER.

Verdict: CONFIRMED only if the candidate lowers MAE on the out-of-sample part; else REJECTED.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from core.config import Settings
from model.player_model import add_features, dvp_by_game, params_from_config, project
from pipeline.player_backtest import MIN_PRIOR_GAMES, _cut_date, _load, _mae
from store.stats import wilson_interval  # noqa: F401  (kept for symmetry with other reports)

log = logging.getLogger(__name__)
STAR_USG = 0.25
COVER_GAIN_MIN = 2.0


def add_usage(d: pd.DataFrame, half_life: float) -> pd.DataFrame:
    """USG% per game (BUILD_PROMPT §3.2) and its prior (earlier games only) / after-state."""
    d = d.copy()
    for c in ("fga", "fta", "tov"):
        d[c] = d[c].fillna(0)
    poss = d["fga"] + 0.44 * d["fta"] + d["tov"]
    team = d.groupby(["game_code", "team"])
    tm_poss = team["fga"].transform("sum") + 0.44 * team["fta"].transform("sum") + team["tov"].transform("sum")
    tm_min = team["minutes"].transform("sum")
    d["usg"] = (poss * (tm_min / 5) / (d["minutes"] * tm_poss)).clip(0, 1)
    key = ["season_code", "player_code"]
    g = d.groupby(key, sort=False)
    d["usg_prior"] = g["usg"].transform(lambda s: s.ewm(halflife=half_life, min_periods=1).mean().shift(1))
    d["usg_after"] = g["usg"].transform(lambda s: s.ewm(halflife=half_life, min_periods=1).mean())
    d["fga_prior_rate"] = (g["fga"].transform(lambda s: s.ewm(halflife=half_life, min_periods=1).mean().shift(1))
                           / d["min_ewm"])
    return d


def _team_games(d: pd.DataFrame) -> pd.DataFrame:
    tg = d[["season_code", "game_date", "game_code", "team"]].drop_duplicates().sort_values(["game_date", "game_code"])
    tg["team_game_no"] = tg.groupby(["season_code", "team"]).cumcount()
    return tg


def first_scorer_rows(d: pd.DataFrame) -> pd.DataFrame:
    """Rows of the first remaining scorer in games with a NEW absence of a high-usage player."""
    from model.player_model import new_absences
    absent = new_absences(d)
    usg_last = d[["game_code", "team", "player_code", "usg_after"]].rename(columns={"game_code": "prev_game"})
    absent = absent.merge(usg_last, on=["prev_game", "team", "player_code"], how="left")
    stars = absent[absent["usg_after"] >= STAR_USG]
    if stars.empty:
        return d.iloc[0:0]
    stars = stars.assign(star_pts=stars["min_ewm_after"] * stars["pts_rate_after"])
    lost = stars.groupby(["game_code", "team"]).agg(star_pts=("star_pts", "sum"), n_stars=("player_code", "size"))
    cand = d.merge(lost.reset_index(), on=["game_code", "team"])
    cand = cand[cand["n_prior"] >= MIN_PRIOR_GAMES].copy()
    cand["exp_pts"] = cand["min_ewm"] * cand["pts_rate"]
    first = cand.loc[cand.groupby(["game_code", "team"])["exp_pts"].idxmax()]
    return first


def returning_rows(d: pd.DataFrame) -> pd.DataFrame:
    """Coverers in games where a rotation player returns after missing ≥ 1 team game."""
    tg = _team_games(d)
    d = d.merge(tg[["game_code", "team", "team_game_no"]], on=["game_code", "team"])
    d = d.sort_values(["season_code", "player_code", "game_date"])
    d["prev_team_game_no"] = d.groupby(["season_code", "player_code"])["team_game_no"].shift(1)
    ret = d[(d["team_game_no"] - d["prev_team_game_no"] >= 2) & (d["min_season"] >= 20)]
    rows = []
    state = d.set_index(["season_code", "team", "team_game_no", "player_code"])["min_ewm_after"]
    for r in ret.itertuples():
        pre_no = int(r.prev_team_game_no)          # last team game the returning player played
        mates = d[(d["game_code"] == r.game_code) & (d["team"] == r.team) & (d["player_code"] != r.player_code)]
        for m in mates.itertuples():
            pre = state.get((r.season_code, r.team, pre_no, m.player_code))
            if pre is None or pd.isna(m.min_ewm) or m.min_ewm - pre < COVER_GAIN_MIN:
                continue
            rows.append({"game_code": m.game_code, "player_code": m.player_code, "pre_min": float(pre),
                         "returner": r.player_code, "missed": int(r.team_game_no - pre_no - 1)})
    if not rows:
        return d.iloc[0:0]
    cov = pd.DataFrame(rows).drop_duplicates(["game_code", "player_code"])
    return cov


def candidate_report(settings: Settings, league: str) -> Optional[str]:
    cfg = settings.league(league)
    season = cfg.prev_season
    df = _load(settings, league, season)
    if df is None:
        return None
    p = params_from_config(cfg.player_model)
    cut = _cut_date(df)
    d = project(add_features(df, league, p), league, p, dvp_by_game(df))
    d = add_usage(d, p.half_life)
    d["proj_fga"] = d["min_proj"] * d["fga_prior_rate"]
    lines = [f"# Параметри-кандидати — {cfg.name_mk} {season}", "",
             f"*{date.today()} · walk-forward (само претходни натпревари) · штимање: датуми < {cut}, оценување: од {cut}. "
             "Пресуда: ПОТВРДЕН само ако MAE вон примерок е помал од M2; инаку ОТФРЛЕН.*", ""]

    # ---- A) first scorer ----
    first = first_scorer_rows(d)
    tune_a, test_a = first[first["game_date"] < cut], first[first["game_date"] >= cut]
    lines += [f"## А) Прв стрелец кога фали ѕвезда (USG ≥ {STAR_USG * 100:.0f}%, ново отсуство)", ""]
    if len(tune_a) < 20 or len(test_a) < 20:
        lines += [f"Премалку случаи (штимање {len(tune_a)}, тест {len(test_a)}) → не може да се заклучи (останува кандидат)."]
        verdict_a = "НЕДОВОЛНО"
    else:
        c = float((tune_a["pts"] - tune_a["proj_pts"]).sum() / tune_a["star_pts"].sum())
        cand = test_a["proj_pts"] + c * test_a["star_pts"]
        mae_m2, mae_c = _mae(test_a["proj_pts"], test_a["pts"]), _mae(cand, test_a["pts"])
        verdict_a = "ПОТВРДЕН" if mae_c < mae_m2 else "ОТФРЛЕН"
        lines += [f"Случаи: штимање {len(tune_a)}, тест {len(test_a)}. Штимано c = {c:+.3f} (дел од очекуваните поени на ѕвездата).", "",
                  "| Тест (вон примерок) | наивно (посл. 5) | M2 | M2 + кандидат |", "|---|---|---|---|",
                  f"| MAE поени | {_mae(test_a['pts_last5'], test_a['pts']):.2f} | {mae_m2:.2f} | {mae_c:.2f} |",
                  f"| пристрасност поени (предвидено − реално) | {(test_a['pts_last5'] - test_a['pts']).mean():+.2f} "
                  f"| {(test_a['proj_pts'] - test_a['pts']).mean():+.2f} | {(cand - test_a['pts']).mean():+.2f} |",
                  f"| пристрасност шутови (FGA), M2 | — | {(test_a['proj_fga'] - test_a['fga']).mean():+.2f} | — |", "",
                  f"**Пресуда: {verdict_a}** (MAE {mae_c - mae_m2:+.2f} наспроти M2)."]

    # ---- B) returning teammate ----
    cov = returning_rows(d)
    lines += ["", "## Б) Соиграч се враќа од повреда → „покривачите“ губат минути (можен ПОД)", ""]
    if cov.empty:
        lines += ["Нема случаи."]
        verdict_b = "НЕДОВОЛНО"
    else:
        rows = d.merge(cov, on=["game_code", "player_code"])
        rows = rows[rows["n_prior"] >= MIN_PRIOR_GAMES]
        tune_b, test_b = rows[rows["game_date"] < cut], rows[rows["game_date"] >= cut]
        if len(tune_b) < 20 or len(test_b) < 20:
            lines += [f"Премалку случаи (штимање {len(tune_b)}, тест {len(test_b)})."]
            verdict_b = "НЕДОВОЛНО"
        else:
            gap = tune_b["pre_min"] - tune_b["min_proj"]
            alpha = float(np.clip(((tune_b["minutes"] - tune_b["min_proj"]) * gap).sum() / max((gap * gap).sum(), 1e-9), 0, 1))
            cand_min = test_b["min_proj"] + alpha * (test_b["pre_min"] - test_b["min_proj"])
            ratio = (cand_min / test_b["min_proj"]).replace([np.inf, -np.inf], 1).fillna(1)
            cand_pts = test_b["proj_pts"] * ratio
            m2_min, c_min = _mae(test_b["min_proj"], test_b["minutes"]), _mae(cand_min, test_b["minutes"])
            m2_pts, c_pts = _mae(test_b["proj_pts"], test_b["pts"]), _mae(cand_pts, test_b["pts"])
            verdict_b = "ПОТВРДЕН" if (c_min < m2_min and c_pts <= m2_pts) else "ОТФРЛЕН"
            under = (test_b["pts"] < np.floor(test_b["pts_last5"]) + 0.5).mean()
            lines += [f"Случаи (покривачи): штимање {len(tune_b)}, тест {len(test_b)}. Штимано α = {alpha:.2f} "
                      f"(0 = M2, 1 = целосно назад на минутите пред отсуството).", "",
                      "| Тест (вон примерок) | наивно (посл. 5) | M2 | M2 + кандидат |", "|---|---|---|---|",
                      f"| MAE минути | {_mae(test_b['min_last5'], test_b['minutes']):.2f} | {m2_min:.2f} | {c_min:.2f} |",
                      f"| пристрасност минути | {(test_b['min_last5'] - test_b['minutes']).mean():+.2f} "
                      f"| {(test_b['min_proj'] - test_b['minutes']).mean():+.2f} | {(cand_min - test_b['minutes']).mean():+.2f} |",
                      f"| MAE поени | {_mae(test_b['pts_last5'], test_b['pts']):.2f} | {m2_pts:.2f} | {c_pts:.2f} |",
                      f"| пристрасност поени | {(test_b['pts_last5'] - test_b['pts']).mean():+.2f} "
                      f"| {(test_b['proj_pts'] - test_b['pts']).mean():+.2f} | {(cand_pts - test_b['pts']).mean():+.2f} |", "",
                      f"Колку често покривачот паднал ПОД „вжештената“ линија (просек посл. 5 + 0.5): {under * 100:.0f}% "
                      f"(n={len(test_b)}; праг на исплатливост на 1.90 = 52.6%; *вештачка линија — не е доказ против Mozzart*).", "",
                      f"**Пресуда: {verdict_b}** (минути {c_min - m2_min:+.2f}, поени {c_pts - m2_pts:+.2f} наспроти M2)."]
    lines += ["", f"**Резиме: А = {verdict_a}, Б = {verdict_b}.**"]
    return "\n".join(lines)
