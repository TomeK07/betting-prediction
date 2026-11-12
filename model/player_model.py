"""Player model (BUILD_PROMPT §3.2, §4.3–4.4; docs/znaenje/kosharka-igraci-parametri.md).

Minutes first, then per-minute rates, then a distribution → P(over line).
Every feature of a game row is computed from the player's EARLIER games only (groupby + shift(1)),
so the walk-forward property holds by construction (test: tests/test_player_model.py).

Статистичкиот модел не гледа квоти. Anchoring е забранет.
(Odds / Mozzart are attached only later in pipeline picks — never inside μ.)

Assumption (documented): who plays in a game is known before tip-off (injury report / lineups).
In history we take "appeared in the box score" as that knowledge. Coach's-decision DNPs are not
knowable in advance — a small optimistic bias, the same for the naive model and ours.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as st

from model.dvp import primary_position
from model.team_model import REGULATION_MINUTES

STATS = ["pts", "ast", "reb", "fgm3"]
MARKET_MK = {"pts": "поени", "ast": "асистенции", "reb": "скокови", "fgm3": "тројки"}
ROTATION_MIN = 20.0            # "rotation player" for the new-absence test (Tome, 29.10)
MAX_MINUTES = {"nba": 44.0, "euroleague": 38.0, "eurocup": 38.0}
DVP_CAP = {"nba": 0.15, "euroleague": 0.12, "eurocup": 0.12}   # PROCES §3 step 5


@dataclass(frozen=True)
class PlayerParams:
    half_life: float = 8.0          # games, recency of minutes and rates (backtest 29.10)
    rate_prior_minutes: float = 75.0    # pseudo-minutes of position-average production
    same_pos_share: float = 0.7     # freed minutes to the same position (BUILD_PROMPT §3.2)
    usage_boost: bool = False       # redistribute the absent player's production beyond minutes (did not help)
    dvp_weight: float = 0.5         # share of the DvP deviation applied (capped)
    minutes_model: str = "m2"       # which minutes feed the stat projection: m1 | m2 | m3 | m4 (chosen by backtest)


def prepare(player_logs: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    """Played rows only (minutes > 0), with position; sorted by date."""
    df = player_logs.copy()
    df["player_code"] = df["player_code"].astype(str)
    df["game_code"] = df["game_code"].astype(str)
    df = df[(df["minutes"].fillna(0) > 0) & (~df["dnp"].fillna(False).astype(bool))]
    pos = roster.assign(pos=roster["position"].map(primary_position)).drop_duplicates("player_code")
    pos["player_code"] = pos["player_code"].astype(str)
    df = df.merge(pos[["player_code", "pos"]], on="player_code", how="left")
    df["pos"] = df["pos"].fillna("F")          # unknown → forward (logged by caller via share)
    return df.sort_values(["game_date", "game_code", "team", "player_code"]).reset_index(drop=True)


def _prior(g: pd.core.groupby.SeriesGroupBy, how: str, **kw) -> pd.Series:
    """Statistic of a player's earlier games only (shift(1) inside each player's sequence)."""
    if how == "ewm":
        return g.transform(lambda s: s.ewm(halflife=kw["halflife"], min_periods=1).mean().shift(1))
    if how == "rolling":
        return g.transform(lambda s: s.rolling(kw["window"], min_periods=1).mean().shift(1))
    if how == "expanding":
        return g.transform(lambda s: s.expanding().mean().shift(1))
    if how == "cumsum":
        return g.transform(lambda s: s.cumsum().shift(1))
    raise ValueError(how)


def add_features(df: pd.DataFrame, league: str, p: PlayerParams) -> pd.DataFrame:
    """Prior-only features per player-game (current season; roles change between seasons)."""
    d = df.copy()
    key = ["season_code", "player_code"] if "season_code" in d else ["player_code"]
    g = d.groupby(key, sort=False)
    d["n_prior"] = g.cumcount()
    d["min_last5"] = _prior(g["minutes"], "rolling", window=5)
    d["min_ewm"] = _prior(g["minutes"], "ewm", halflife=p.half_life)
    d["min_season"] = _prior(g["minutes"], "expanding")
    d["cum_min"] = _prior(g["minutes"], "cumsum").fillna(0)
    # state AFTER each game (used for teammates who are absent in a later game)
    d["min_ewm_after"] = g["minutes"].transform(lambda s: s.ewm(halflife=p.half_life, min_periods=1).mean())
    d["min_season_after"] = g["minutes"].transform(lambda s: s.expanding().mean())

    # league per-minute rate by position, from earlier DATES only
    daily = d.groupby(["game_date", "pos"])[STATS + ["minutes"]].sum().sort_index()
    cum = daily.groupby(level="pos").cumsum().groupby(level="pos").shift(1)
    lg_rate = cum[STATS].div(cum["minutes"], axis=0)
    lg_rate = lg_rate.groupby(level="pos").ffill()
    d = d.merge(lg_rate.add_prefix("lg_rate_").reset_index(), on=["game_date", "pos"], how="left")

    g = d.groupby(key, sort=False)
    for s in STATS:
        d[f"{s}_last5"] = _prior(g[s], "rolling", window=5)
        ewm_s = _prior(g[s], "ewm", halflife=p.half_life)
        rate_ewm = ewm_s / d["min_ewm"]
        lg = d[f"lg_rate_{s}"].fillna(rate_ewm)
        k = p.rate_prior_minutes
        d[f"{s}_rate"] = (rate_ewm.fillna(lg) * d["cum_min"] + lg * k) / (d["cum_min"] + k)
        d[f"{s}_rate_after"] = g[s].transform(lambda x: x.ewm(halflife=p.half_life, min_periods=1).mean()) / d["min_ewm_after"]
    return d


def new_absences(d: pd.DataFrame) -> pd.DataFrame:
    """Rotation players (season avg ≥ 20 min) who played the team's previous game but not this one.

    Returns one row per (game_code, team, absent player) with the absent player's state after his
    last game (expected minutes, position, per-minute rates) — all known before tip-off.
    """
    tg = d[["season_code", "game_date", "game_code", "team"]].drop_duplicates().sort_values(["game_date", "game_code"])
    tg["prev_game"] = tg.groupby(["season_code", "team"])["game_code"].shift(1)
    last_state = d[["game_code", "team", "player_code", "pos", "min_ewm_after", "min_season_after"]
                   + [f"{s}_rate_after" for s in STATS]]
    prev_players = tg.dropna(subset=["prev_game"]).merge(
        last_state.rename(columns={"game_code": "prev_game"}), on=["prev_game", "team"])
    prev_players = prev_players[prev_players["min_season_after"] >= ROTATION_MIN]
    today = d[["game_code", "team", "player_code"]].assign(plays=True)
    out = prev_players.merge(today, on=["game_code", "team", "player_code"], how="left")
    return out[out["plays"].isna()].drop(columns="plays")


def project(d: pd.DataFrame, league: str, p: PlayerParams, dvp: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Minutes (several variants) and stat projections for every played row with history."""
    d = d.copy()
    reg_total = REGULATION_MINUTES[league] * 5
    cap = MAX_MINUTES[league]
    d["min_m1"] = d["min_ewm"]

    team_sum = d.groupby(["game_code", "team"])["min_m1"].transform("sum")
    d["min_m2"] = (d["min_m1"] * reg_total / team_sum).clip(upper=cap)

    # M3: freed minutes of NEW absences → 70% same position, 30% everyone (proportional to minutes)
    absent = new_absences(d)
    d["min_m3"] = d["min_m1"]
    d["absent_min"] = 0.0
    for c in STATS:
        d[f"absent_{c}"] = 0.0
    if len(absent):
        freed = absent.groupby(["game_code", "team"]).agg(freed=("min_ewm_after", "sum"))
        freed_pos = absent.groupby(["game_code", "team", "pos"]).agg(freed_pos=("min_ewm_after", "sum"))
        d = d.merge(freed.reset_index(), on=["game_code", "team"], how="left")
        d = d.merge(freed_pos.reset_index(), on=["game_code", "team", "pos"], how="left")
        d[["freed", "freed_pos"]] = d[["freed", "freed_pos"]].fillna(0.0)
        w_all = d["min_m1"] / d.groupby(["game_code", "team"])["min_m1"].transform("sum")
        pos_sum = d.groupby(["game_code", "team", "pos"])["min_m1"].transform("sum")
        w_pos = d["min_m1"] / pos_sum
        same = p.same_pos_share * d["freed_pos"] * w_pos
        # the remaining pool is shared by the whole team: freed minus what went to same-position players
        # (positions with nobody left keep their 70% in the shared pool)
        same_total = (d.drop_duplicates(["game_code", "team", "pos"])
                      .groupby(["game_code", "team"])["freed_pos"].sum().rename("same_total"))
        d = d.merge(same_total.reset_index(), on=["game_code", "team"], how="left")
        pool = d["freed"] - p.same_pos_share * d["same_total"]
        rest = pool * w_all
        d["min_m3"] = (d["min_m1"] + same.fillna(0) + rest.fillna(0)).clip(upper=cap)
        d["absent_min"] = d["freed"]
        prod = absent.assign(**{f"exp_{s}": absent["min_ewm_after"] * absent[f"{s}_rate_after"] for s in STATS})
        lost = prod.groupby(["game_code", "team"])[[f"exp_{s}" for s in STATS]].sum().reset_index()
        d = d.merge(lost, on=["game_code", "team"], how="left")
        for s in STATS:
            d[f"absent_{s}"] = d[f"exp_{s}"].fillna(0.0)

    # M4: redistribution, then the whole team scaled to regulation minutes
    team_sum3 = d.groupby(["game_code", "team"])["min_m3"].transform("sum")
    d["min_m4"] = (d["min_m3"] * reg_total / team_sum3).clip(upper=cap)
    d["min_proj"] = d[f"min_{p.minutes_model}"]

    for s in STATS:
        base = d["min_proj"] * d[f"{s}_rate"]
        if p.usage_boost and len(absent):
            # production of the absent player not already covered by teammates' extra minutes
            gained = ((d["min_proj"] - d["min_m1"]) * d[f"{s}_rate"]).groupby([d["game_code"], d["team"]]).transform("sum")
            extra = (d[f"absent_{s}"] - gained).clip(lower=0)
            share = base / base.groupby([d["game_code"], d["team"]]).transform("sum")
            base = base + 0.5 * extra * share.fillna(0)       # half: usage does not fully transfer
        if dvp is not None and f"{s}_vs_liga_pct" in dvp:
            f = d.merge(dvp, left_on=["game_code", "opp", "pos"], right_on=["game_code", "tim", "pozicija"],
                        how="left")[f"{s}_vs_liga_pct"].fillna(0).values / 100
            f = np.clip(f * p.dvp_weight, -DVP_CAP[league], DVP_CAP[league])
            base = base * (1 + f)
        d[f"proj_{s}"] = base
    return d


def dvp_by_game(df: pd.DataFrame, shrink_minutes: float = 150.0) -> pd.DataFrame:
    """DvP per (game, defending team, position) from EARLIER dates only (cumulative sums shifted by date)."""
    daily = df.groupby(["opp", "pos", "game_date"])[STATS + ["minutes"]].sum().sort_index()
    cum = daily.groupby(level=["opp", "pos"]).cumsum().groupby(level=["opp", "pos"]).shift(1).fillna(0)
    lg_daily = df.groupby(["pos", "game_date"])[STATS + ["minutes"]].sum().sort_index()
    lg_cum = lg_daily.groupby(level="pos").cumsum().groupby(level="pos").shift(1)
    out = cum.reset_index().merge(lg_cum.reset_index(), on=["pos", "game_date"], suffixes=("", "_lg"))
    for s in STATS:
        lg36 = out[f"{s}_lg"] / out["minutes_lg"] * 36
        per36 = (out[s] + lg36 * shrink_minutes / 36) / (out["minutes"] + shrink_minutes) * 36
        out[f"{s}_vs_liga_pct"] = (per36 / lg36 - 1) * 100
    keys = df[["game_code", "game_date", "opp", "pos"]].drop_duplicates()
    out = keys.merge(out, on=["opp", "pos", "game_date"], how="left")
    return out.rename(columns={"opp": "tim", "pos": "pozicija"})[
        ["game_code", "tim", "pozicija"] + [f"{s}_vs_liga_pct" for s in STATS]]


# ---------------- distributions ----------------

def fit_dispersion(mu: np.ndarray, y: np.ndarray) -> float:
    """Negative binomial size r by method of moments: Var = mu + mu²/r (r → ∞ = Poisson)."""
    mu, y = np.asarray(mu, float), np.asarray(y, float)
    ok = mu > 0
    excess = ((y[ok] - mu[ok]) ** 2 - mu[ok]).sum()
    denom = (mu[ok] ** 2).sum()
    if excess <= 0:
        return 1e6
    return float(denom / excess)


def p_over(mu: np.ndarray, line: np.ndarray, r) -> np.ndarray:
    """P(X > line) for X ~ NegBin(mean mu, size r); lines are x.5 so no pushes. r: scalar or one per row."""
    mu = np.maximum(np.asarray(mu, float), 1e-6)
    k = np.floor(np.asarray(line, float))
    r = np.broadcast_to(np.asarray(r, float), mu.shape)
    out = np.empty(mu.shape)
    pois = r >= 1e5
    out[pois] = st.poisson.sf(np.broadcast_to(k, mu.shape)[pois], mu[pois])
    nb = ~pois
    out[nb] = st.nbinom.sf(np.broadcast_to(k, mu.shape)[nb], r[nb], r[nb] / (r[nb] + mu[nb]))
    return out


# Low-line backtest 29.10 (reports/backtest_niski_*.md): one r per market makes the distribution too narrow for
# small means (NB variance mu + mu²/r ≈ Poisson when mu is small) → P(ПОД) overestimated by 3–5 pp at 80%+.
# A separate r per range of the projected mean fixes it; zero-inflation was tested and did not help.
MU_BINS = [1.0, 2.0, 3.5, 6.0]               # mean ranges: <1, 1–2, 2–3.5, 3.5–6, ≥6
R_GRID = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0, 40.0, 1e6]
LOW_LINES = {"fgm3": [0.5, 1.5], "reb": [2.5, 3.5], "ast": [1.5, 2.5]}


def dispersion_for(mu, spec) -> np.ndarray:
    """spec from settings.yaml: a number (one r) or {"r_po_mu": [r per MU_BINS range]}."""
    mu = np.asarray(mu, float)
    if isinstance(spec, dict):
        return np.asarray(spec["r_po_mu"], float)[np.digitize(mu, MU_BINS)]
    return np.full(mu.shape, float(spec))


def threshold_log_loss(mu: np.ndarray, y: np.ndarray, lines: list[float], r) -> float:
    """Mean log loss of P(ПОД) summed over the given lines — the quantity the low-line markets are priced on."""
    tot = 0.0
    for line in lines:
        pu = np.clip(1 - p_over(mu, np.full(len(mu), line), r), 1e-6, 1 - 1e-6)
        under = np.asarray(y, float) <= line
        tot += -float(np.mean(np.where(under, np.log(pu), np.log(1 - pu))))
    return tot


def fit_dispersion_by_mu(mu: np.ndarray, y: np.ndarray, lines: list[float], min_rows: int = 50) -> list[float]:
    """One r per mean range, chosen on the grid by log loss at the given lines (fallback: the single r)."""
    mu, y = np.asarray(mu, float), np.asarray(y, float)
    r0 = fit_dispersion(mu, y)
    b = np.digitize(mu, MU_BINS)
    out = []
    for i in range(len(MU_BINS) + 1):
        sel = b == i
        out.append(min(R_GRID, key=lambda r: threshold_log_loss(mu[sel], y[sel], lines, r))
                   if sel.sum() >= min_rows else r0)
    return out


def params_from_config(cfg: dict) -> PlayerParams:
    """Per-league player model chosen by backtest (settings.yaml → leagues.<league>.player_model)."""
    return PlayerParams(half_life=float(cfg.get("half_life", 8.0)),
                        rate_prior_minutes=float(cfg.get("rate_prior_minutes", 75.0)),
                        minutes_model=cfg.get("minutes_model", "m2"),
                        usage_boost=bool(cfg.get("usage_boost", False)),
                        dvp_weight=float(cfg.get("dvp_weight", 0.5)))
