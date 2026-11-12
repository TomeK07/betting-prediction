"""Team model (BUILD_PROMPT §3.1, §4.1): pace × efficiency → projected score.

Everything here is pure: `compute_ratings` sees only the games it is given. The backtest
(model/backtest.py) guarantees those are strictly earlier than the game being predicted.

Per game (team row):
    poss  = mean of both teams' estimates  FGA − OREB + TOV + 0.44·FTA
    ORtg  = 100·pts/poss,  DRtg = 100·opp_pts/poss
    pace  = poss scaled to regulation minutes (overtime would inflate it)
Four Factors (Dean Oliver): eFG = (FGM + 0.5·3PM)/FGA · TOV% = TOV/poss ·
    OREB% = OREB/(OREB + opp DREB) · FTr = FTM/FGA.
Projection (A home vs B):
    poss_AB = pace_A + pace_B − league pace
    pts_A   = poss_AB · (ORtg_A + DRtg_B − league ORtg)/100 + HCA/2   (+ rest, + overtime allowance)
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np
import pandas as pd

REGULATION_MINUTES = {"nba": 48.0, "euroleague": 40.0, "eurocup": 40.0}
FACTORS = ["efg", "tov_pct", "orb_pct", "ftr"]


@dataclass(frozen=True)
class ModelParams:
    """One model variant. The ablation ladder switches these on one by one."""
    name: str = "full"
    baseline: Optional[str] = None      # "league" | "team_avg" (naive models) | None (pace × efficiency)
    use_home: bool = True
    use_pace: bool = True               # team-specific pace (else league pace for every game)
    opp_adjust: bool = True             # SRS-style correction for strength of schedule
    half_life: Optional[float] = None   # recency weighting in games (None = all games equal)
    use_ff: bool = False                # efficiency from Four Factors regression instead of raw ORtg/DRtg
    use_rest: bool = False              # back-to-back / rest adjustment fitted on past residuals
    shrink_games: float = 5.0           # pseudo-games of league average added to every team
    hca_prior_games: float = 30.0       # pseudo-games behind the configured home advantage
    # Season transition (decision 29.10: NBA total bias +4.7 when last season was simply pooled in)
    season_mode: str = "pooled"         # pooled | current_only | carryover
    carry: float = 0.5                  # share of last season's deviation from the league kept
    prior_games: float = 10.0           # weight of that carried prior, in pseudo-games
    lg_prior_games: float = 60.0        # last season's league average, in pseudo team-games
    weekly_bias: bool = False           # add this season's mean residual up to the start of the week
    bias_prior_games: float = 20.0      # shrinkage of that correction (pseudo team-games at 0)


def game_frame(team_logs: pd.DataFrame, league: str) -> pd.DataFrame:
    """Team-game rows with possessions, ratings and Four Factors; sorted by date (rule 1)."""
    t = team_logs.copy()
    t["game_code"] = t["game_code"].astype(str)
    est = lambda p: t[f"{p}fga"] - t[f"{p}oreb"] + t[f"{p}tov"] + 0.44 * t[f"{p}fta"]
    t["poss"] = 0.5 * (est("") + est("opp_"))
    game_minutes = t["team_minutes"] / 5.0
    t["pace"] = t["poss"] * REGULATION_MINUTES[league] / game_minutes
    t["reg_share"] = REGULATION_MINUTES[league] / game_minutes          # 1.0 without overtime
    t["ortg"] = 100 * t["pts"] / t["poss"]
    t["drtg"] = 100 * t["opp_pts"] / t["poss"]
    for p, side in (("", "off"), ("opp_", "def")):
        o = "opp_" if p == "" else ""
        t[f"efg_{side}"] = (t[f"{p}fgm"] + 0.5 * t[f"{p}fgm3"]) / t[f"{p}fga"]
        t[f"tov_pct_{side}"] = t[f"{p}tov"] / t["poss"]
        t[f"orb_pct_{side}"] = t[f"{p}oreb"] / (t[f"{p}oreb"] + t[f"{o}dreb"])
        t[f"ftr_{side}"] = t[f"{p}ftm"] / t[f"{p}fga"]
    t = t.sort_values(["game_date", "game_code", "home"], ascending=[True, True, False]).reset_index(drop=True)
    # rest: days since the team's previous game (known before tip-off)
    prev = t.groupby("team")["game_date"].shift(1)
    t["rest_days"] = (pd.to_datetime(t["game_date"]) - pd.to_datetime(prev)).dt.days - 1
    t["b2b"] = (t["rest_days"] == 0).astype(float)
    # back-to-back where yesterday AND today are away games (travel + no home rest)
    prev_home = t.groupby("team")["home"].shift(1)
    t["b2b_aa"] = ((t["b2b"] == 1) & (~t["home"]) & (prev_home == False)).astype(float)  # noqa: E712
    opp = t[["game_code", "team", "b2b", "b2b_aa"]].rename(
        columns={"team": "opp", "b2b": "opp_b2b", "b2b_aa": "opp_b2b_aa"})
    return t.merge(opp, on=["game_code", "opp"], how="left")


@dataclass
class Ratings:
    teams: pd.DataFrame                 # index team: n, pace, ortg, drtg, ppg, papg, ff off/def
    lg_pace: float
    lg_ortg: float
    lg_ppg: float
    hca: float
    ot_bonus: float
    ff_coef: Optional[np.ndarray] = None
    lg_ff: dict = field(default_factory=dict)


def _weights(past: pd.DataFrame, half_life: Optional[float]) -> pd.Series:
    if not half_life:
        return pd.Series(1.0, index=past.index)
    games_ago = past.groupby("team").cumcount(ascending=False)
    return 0.5 ** (games_ago / half_life)


def _wmean(df: pd.DataFrame, col: str, w: pd.Series) -> pd.Series:
    return (df[col] * w).groupby(df["team"]).sum() / w.groupby(df["team"]).sum()


def _shrink(values: pd.Series, n: pd.Series, prior: float, k: float) -> pd.Series:
    return (values * n + prior * k) / (n + k)


_PREV_CACHE: dict = {}


def week_start(day: str) -> str:
    d = pd.Timestamp(day)
    return (d - pd.Timedelta(days=d.weekday())).date().isoformat()


def compute_ratings(past: pd.DataFrame, params: ModelParams, hca_prior: float,
                    season: Optional[str] = None, as_of: Optional[str] = None) -> Ratings:
    """Team ratings from `past` only (caller guarantees all rows are before the predicted game).

    season_mode != "pooled" needs the predicted game's `season` and date `as_of`:
      - league averages = this season's games up to the start of the current week (updated weekly),
        blended with last season's averages (lg_prior_games pseudo team-games);
      - team deviation = (n·current deviation + prior_games·carry·last-season deviation) / (n + prior_games + shrink)
        ("carryover"), or current season only ("current_only").
    """
    if params.season_mode != "pooled" and season is not None and "season_code" in past:
        return _seasonal_ratings(past, params, hca_prior, season, as_of)
    w = _weights(past, params.half_life)
    n = w.groupby(past["team"]).sum()
    lg_pace = float(past["pace"].mean())
    lg_ortg = float(100 * past["pts"].sum() / past["poss"].sum())
    lg_ppg = float(past["pts"].mean())

    home_rows = past[past["home"]]
    if params.use_home and len(home_rows):
        margin = (home_rows["pts"] - home_rows["opp_pts"])
        hca = float((margin.sum() + hca_prior * params.hca_prior_games) / (len(margin) + params.hca_prior_games))
    else:
        hca = 0.0
    totals = past["pts"] + past["opp_pts"]
    ot_bonus = float((totals - totals * past["reg_share"]).mean() / 2)  # per team

    sign = np.where(past["home"], 1.0, -1.0)
    hca_eff = 100 * hca / lg_pace
    ortg_g = past["ortg"] - sign * hca_eff / 2          # neutral-court efficiency
    drtg_g = past["drtg"] + sign * hca_eff / 2
    pace_g = past["pace"]

    o = _wmean(past.assign(v=ortg_g), "v", w)
    d = _wmean(past.assign(v=drtg_g), "v", w)
    pc = _wmean(past, "pace", w)
    if params.opp_adjust:
        for _ in range(10):
            opp_d = past["opp"].map(d).fillna(lg_ortg) - lg_ortg
            opp_o = past["opp"].map(o).fillna(lg_ortg) - lg_ortg
            opp_p = past["opp"].map(pc).fillna(lg_pace) - lg_pace
            o = _wmean(past.assign(v=ortg_g - opp_d), "v", w)
            d = _wmean(past.assign(v=drtg_g - opp_o), "v", w)
            pc = _wmean(past.assign(v=pace_g - opp_p), "v", w)
    k = params.shrink_games
    teams = pd.DataFrame({
        "n": n,
        "ortg": _shrink(o, n, lg_ortg, k),
        "drtg": _shrink(d, n, lg_ortg, k),
        "pace": _shrink(pc, n, lg_pace, k) if params.use_pace else lg_pace,
        "ppg": _shrink(_wmean(past, "pts", w), n, lg_ppg, k),
        "papg": _shrink(_wmean(past, "opp_pts", w), n, lg_ppg, k),
    })
    ratings = Ratings(teams, lg_pace, lg_ortg, lg_ppg, hca, ot_bonus)

    if params.use_ff and len(past) >= 60:
        lg_ff = {}
        for f in FACTORS:
            lg_ff[f] = float(past[f"{f}_off"].mean())
            teams[f"{f}_off"] = _shrink(_wmean(past, f"{f}_off", w), n, lg_ff[f], k)
            teams[f"{f}_def"] = _shrink(_wmean(past, f"{f}_def", w), n, lg_ff[f], k)
        X = np.column_stack([np.ones(len(past))] + [past[f"{f}_off"].values for f in FACTORS])
        y = past["ortg"].values
        ratings.ff_coef = np.linalg.lstsq(X, y, rcond=None)[0]
        ratings.lg_ff = lg_ff
    return ratings


def _team(r: Ratings, team: str) -> pd.Series:
    if team in r.teams.index:
        return r.teams.loc[team]
    return pd.Series({"n": 0.0, "ortg": r.lg_ortg, "drtg": r.lg_ortg, "pace": r.lg_pace, "ppg": r.lg_ppg,
                      "papg": r.lg_ppg, **{f"{f}_{s}": r.lg_ff.get(f, np.nan) for f in FACTORS for s in ("off", "def")}})


def _efficiency(r: Ratings, att: pd.Series, dfn: pd.Series, params: ModelParams) -> float:
    if params.use_ff and r.ff_coef is not None:
        factors = [att[f"{f}_off"] + dfn[f"{f}_def"] - r.lg_ff[f] for f in FACTORS]
        return float(r.ff_coef[0] + np.dot(r.ff_coef[1:], factors))
    return float(att["ortg"] + dfn["drtg"] - r.lg_ortg)


def predict_game(r: Ratings, home: str, away: str, params: ModelParams,
                 rest_adj: tuple[float, float] = (0.0, 0.0)) -> tuple[float, float]:
    """Projected (home points, away points), overtime allowance included."""
    h, a = _team(r, home), _team(r, away)
    half = r.hca / 2
    if params.baseline == "league":
        return r.lg_ppg + half, r.lg_ppg - half
    if params.baseline == "team_avg":
        # the manual "scenario" method: (attack A + defence B) / 2
        return (h["ppg"] + a["papg"]) / 2 + half, (a["ppg"] + h["papg"]) / 2 - half
    poss = h["pace"] + a["pace"] - r.lg_pace
    pts_h = poss * _efficiency(r, h, a, params) / 100 + half + r.ot_bonus + rest_adj[0]
    pts_a = poss * _efficiency(r, a, h, params) / 100 - half + r.ot_bonus + rest_adj[1]
    return pts_h, pts_a


def ladder() -> list[ModelParams]:
    """Ablation ladder: each step adds one ingredient."""
    v2 = ModelParams(name="V2 темпо×ефикасност", opp_adjust=False)
    v3 = replace(v2, name="V3 + противници", opp_adjust=True)
    v4 = replace(v3, name="V4 + форма", half_life=15.0)
    v5 = replace(v4, name="V5 + Four Factors", use_ff=True)
    v6 = replace(v5, name="V6 + одмор", use_rest=True)
    return [ModelParams(name="V0 просек лига", baseline="league"),
            ModelParams(name="V1 (напад+одбрана)/2", baseline="team_avg"), v2, v3, v4, v5, v6]


def params_from_config(cfg: dict) -> ModelParams:
    """The per-league model chosen by backtest (config/settings.yaml → leagues.<league>.team_model)."""
    return ModelParams(name=cfg.get("variant", "configured"),
                       opp_adjust=bool(cfg.get("opp_adjust", True)),
                       half_life=cfg.get("half_life"),
                       use_ff=bool(cfg.get("use_ff", False)),
                       use_rest=bool(cfg.get("use_rest", False)),
                       shrink_games=float(cfg.get("shrink_games", 5.0)),
                       season_mode=cfg.get("season_mode", "pooled"),
                       carry=float(cfg.get("carry", 0.5)),
                       prior_games=float(cfg.get("prior_games", 10.0)),
                       lg_prior_games=float(cfg.get("lg_prior_games", 60.0)),
                       weekly_bias=bool(cfg.get("weekly_bias", False)))


def _seasonal_ratings(past: pd.DataFrame, params: ModelParams, hca_prior: float,
                      season: str, as_of: Optional[str]) -> Ratings:
    raw = replace(params, season_mode="pooled", shrink_games=0.0, use_ff=False)
    cur = past[past["season_code"] == season]
    older = past[past["season_code"] != season]
    prev = older[older["season_code"] == older["season_code"].max()] if len(older) else older

    prev_r = None
    if len(prev):
        key = (prev["season_code"].iat[0], len(prev), raw)
        if key not in _PREV_CACHE:
            _PREV_CACHE[key] = compute_ratings(prev, raw, hca_prior)
        prev_r = _PREV_CACHE[key]
    cur_r = compute_ratings(cur, raw, hca_prior) if len(cur) >= 2 else None

    # league averages: this season up to the start of the current week, blended with last season
    weekly = cur[cur["game_date"] < week_start(as_of)] if as_of else cur
    def league(col_cur: float, col_prev: Optional[float]) -> float:
        n = len(weekly)
        if col_prev is None:
            return col_cur
        if n == 0:
            return col_prev
        return (n * col_cur + params.lg_prior_games * col_prev) / (n + params.lg_prior_games)
    wk_pace = float(weekly["pace"].mean()) if len(weekly) else np.nan
    wk_ortg = float(100 * weekly["pts"].sum() / weekly["poss"].sum()) if len(weekly) else np.nan
    wk_ppg = float(weekly["pts"].mean()) if len(weekly) else np.nan
    lg_pace = league(wk_pace, prev_r.lg_pace if prev_r else None)
    lg_ortg = league(wk_ortg, prev_r.lg_ortg if prev_r else None)
    lg_ppg = league(wk_ppg, prev_r.lg_ppg if prev_r else None)
    if np.isnan(lg_pace):                     # nothing at all yet
        lg_pace, lg_ortg, lg_ppg = float(past["pace"].mean()), float(100 * past["pts"].sum() / past["poss"].sum()), float(past["pts"].mean())

    teams = sorted(set(past["team"]))
    out = pd.DataFrame(index=teams, columns=["n", "ortg", "drtg", "pace", "ppg", "papg"], dtype=float)
    k, P = params.shrink_games, params.prior_games if params.season_mode == "carryover" else 0.0
    for t in teams:
        n = float(cur_r.teams.loc[t, "n"]) if cur_r is not None and t in cur_r.teams.index else 0.0
        for col, lg_cur_ref, lg_prev_ref, lg_new in (
                ("ortg", cur_r.lg_ortg if cur_r else 0, prev_r.lg_ortg if prev_r else 0, lg_ortg),
                ("drtg", cur_r.lg_ortg if cur_r else 0, prev_r.lg_ortg if prev_r else 0, lg_ortg),
                ("pace", cur_r.lg_pace if cur_r else 0, prev_r.lg_pace if prev_r else 0, lg_pace),
                ("ppg", cur_r.lg_ppg if cur_r else 0, prev_r.lg_ppg if prev_r else 0, lg_ppg),
                ("papg", cur_r.lg_ppg if cur_r else 0, prev_r.lg_ppg if prev_r else 0, lg_ppg)):
            dev_cur = (cur_r.teams.loc[t, col] - lg_cur_ref) if n else 0.0
            has_prev = prev_r is not None and t in prev_r.teams.index
            dev_prev = (prev_r.teams.loc[t, col] - lg_prev_ref) if has_prev else 0.0
            p_w = P if has_prev else 0.0
            out.loc[t, col] = lg_new + (n * dev_cur + p_w * params.carry * dev_prev) / (n + p_w + k) if (n + p_w + k) else lg_new
        out.loc[t, "n"] = n
    if not params.use_pace:
        out["pace"] = lg_pace
    pooled = compute_ratings(past, replace(params, season_mode="pooled"), hca_prior)   # HCA and overtime: all history
    return Ratings(out, lg_pace, lg_ortg, lg_ppg, pooled.hca, pooled.ot_bonus)
