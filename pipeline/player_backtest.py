"""`python main.py backtest --igraci` — Phase 5 player-model backtest on last season (walk-forward).

Order (decision 29.10): minutes first — if minutes are not better than the naive average of the
last 5 games, the rest is not trusted. Then the new-absence test, then distributions/calibration.
Tuning on the first 60% of the season's dates, evaluation on the last 40%.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from core.config import Settings
from model.player_model import (LOW_LINES, MARKET_MK, MU_BINS, R_GRID, STATS, PlayerParams, add_features,
                                dispersion_for, dvp_by_game, fit_dispersion, fit_dispersion_by_mu, new_absences,
                                p_over, params_from_config, prepare, project, threshold_log_loss)
from store.stats import wilson_interval

log = logging.getLogger(__name__)
TUNE_SHARE = 0.6
MIN_PRIOR_GAMES = 3


def _load(settings: Settings, league: str, season: str) -> Optional[pd.DataFrame]:
    folder = settings.paths.processed / league / season
    if not (folder / "player_logs.csv").exists():
        return None
    logs = pd.read_csv(folder / "player_logs.csv", dtype={"player_code": str, "game_code": str})
    roster = pd.read_csv(folder / "roster.csv", dtype={"player_code": str})
    logs["season_code"] = season
    return prepare(logs, roster)


def _cut_date(d: pd.DataFrame) -> str:
    dates = sorted(d["game_date"].unique())
    return dates[int(len(dates) * TUNE_SHARE)]


def _mae(a, b) -> float:
    return float(np.nanmean(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def _run(df: pd.DataFrame, league: str, p: PlayerParams, dvp: Optional[pd.DataFrame]) -> pd.DataFrame:
    return project(add_features(df, league, p), league, p, dvp)


def player_report(settings: Settings, league: str) -> Optional[str]:
    cfg = settings.league(league)
    season = cfg.prev_season
    df = _load(settings, league, season)
    if df is None:
        print(f"{cfg.name_mk}: нема играчки логови за {season}")
        return None
    cut = _cut_date(df)
    dvp = dvp_by_game(df)
    lines = [f"# Backtest играчки модел — {cfg.name_mk} {season}", "",
             f"*{date.today()} · `python main.py backtest --igraci --league {league}` · walk-forward: секоја бројка на "
             f"играчот е од неговите ПРЕТХОДНИ натпревари. Штимање: датуми < {cut}; оценување: од {cut}. "
             f"Само играчи со ≥ {MIN_PRIOR_GAMES} претходни натпревари во сезоната.*", "",
             "> ⚠️ **Калибрацијата подолу е на вештачки линии (блиску до недавниот просек на играчот). "
             "НЕ е доказ за предност против Mozzart.** Вистинскиот тест е Фаза 7, со зачуваните Mozzart линии "
             "(се собираат од 29.10.2026).", "",
             f"- Одиграни играчки натпревари: {len(df)}; играчи: {df['player_code'].nunique()}.",
             "- Претпоставка: кој игра е познато пред натпреварот (извештај за повреди). Во историјата: "
             "„се појавил во box score“ — исто за наивниот модел и за нашиот.", ""]

    # ---- tuning (first 60%) ----
    best, best_score = None, None
    for hl in (3.0, 5.0, 8.0):
        for k in (75.0, 150.0, 300.0):
            p = PlayerParams(half_life=hl, rate_prior_minutes=k, dvp_weight=0.0)
            d = _run(df, league, p, None)
            t = d[(d["game_date"] < cut) & (d["n_prior"] >= MIN_PRIOR_GAMES)]
            score = _mae(t["min_m3"], t["minutes"]) / t["minutes"].mean() + sum(
                _mae(t[f"proj_{s}"], t[s]) / max(t[s].mean(), 0.1) for s in STATS)
            if best_score is None or score < best_score:
                best, best_score = p, score
    # minutes model and usage boost, chosen on the tuning part
    d = _run(df, league, best, None)
    t = d[(d["game_date"] < cut) & (d["n_prior"] >= MIN_PRIOR_GAMES)]
    best = replace(best, minutes_model=min(("m1", "m2", "m3", "m4"), key=lambda c: _mae(t[f"min_{c}"], t["minutes"])))
    usage_scores = {}
    for ub in (False, True):
        d = _run(df, league, replace(best, usage_boost=ub), None)
        t = d[(d["game_date"] < cut) & (d["n_prior"] >= MIN_PRIOR_GAMES)]
        usage_scores[ub] = sum(_mae(t[f"proj_{s}"], t[s]) / max(t[s].mean(), 0.1) for s in STATS)
    best = replace(best, usage_boost=min(usage_scores, key=usage_scores.get))
    dvp_scores = {}
    for w in (0.0, 0.5, 1.0):
        d = _run(df, league, replace(best, dvp_weight=w), dvp)
        t = d[(d["game_date"] < cut) & (d["n_prior"] >= MIN_PRIOR_GAMES)]
        dvp_scores[w] = sum(_mae(t[f"proj_{s}"], t[s]) / max(t[s].mean(), 0.1) for s in STATS)
    best = replace(best, dvp_weight=min(dvp_scores, key=dvp_scores.get))
    lines += [f"Штимано: форма half-life = {best.half_life:g} натпревари, prior за стапка = {best.rate_prior_minutes:g} мин, "
              f"DvP тежина = {best.dvp_weight:g} (кандидати 0 / 0.5 / 1), минути за проекција = {best.minutes_model.upper()}, "
              f"usage-поттик при отсуство = {'да' if best.usage_boost else 'не'}.", ""]

    full = _run(df, league, best, dvp)
    no_dvp = _run(df, league, replace(best, dvp_weight=0.0), None)
    test = full[(full["game_date"] >= cut) & (full["n_prior"] >= MIN_PRIOR_GAMES)].copy()
    test_nd = no_dvp.set_index(["game_code", "player_code"]).loc[
        test.set_index(["game_code", "player_code"]).index].reset_index()
    tune = full[(full["game_date"] < cut) & (full["n_prior"] >= MIN_PRIOR_GAMES)]

    # ---- 1. minutes ----
    m_naive = _mae(test["min_last5"], test["minutes"])
    m = {"M1 форма (ewm)": _mae(test["min_m1"], test["minutes"]),
         "M2 + тимот = 240/200 мин": _mae(test["min_m2"], test["minutes"]),
         "M3 + пренасочување (ново отсуство)": _mae(test["min_m3"], test["minutes"]),
         "M4 = M3 + тимот = 240/200 мин": _mae(test["min_m4"], test["minutes"])}
    best_min = min(m, key=m.get)
    lines += ["## 1. Минути (прво)", "", "| Модел | MAE минути | во ±4 мин |", "|---|---|---|",
              f"| Наивно: просек последни 5 | {m_naive:.2f} | {(abs(test['min_last5'] - test['minutes']) <= 4).mean() * 100:.0f}% |"]
    for name, col in (("M1 форма (ewm)", "min_m1"), ("M2 + тимот = 240/200 мин", "min_m2"),
                      ("M3 + пренасочување (ново отсуство)", "min_m3"), ("M4 = M3 + тимот = 240/200 мин", "min_m4")):
        lines.append(f"| {name} | {m[name]:.2f} | {(abs(test[col] - test['minutes']) <= 4).mean() * 100:.0f}% |")
    gate = m[best_min] < m_naive
    lines += ["", f"**{'✅' if gate else '❌'} Најдобар: {best_min} ({m[best_min] - m_naive:+.2f} мин наспроти наивното).**"]
    if not gate:
        lines += ["", "Минутите НЕ се подобри од наивното → по одлуката 29.10 не се оди понатаму за оваа лига."]
        return "\n".join(lines)

    # ---- 2. new absence ----
    absent = new_absences(add_features(df, league, best))
    events = absent[absent["game_code"].isin(set(test["game_code"]))]
    aff = test.merge(events[["game_code", "team"]].drop_duplicates(), on=["game_code", "team"])
    aff_nd = test_nd.merge(events[["game_code", "team"]].drop_duplicates(), on=["game_code", "team"])
    lines += ["", "## 2. Ново отсуство (ротациски играч ≥ 20 мин пропуштил, а го играл претходниот натпревар)", "",
              f"Настани во делот за оценување: {len(events)} отсуства, {len(aff)} соиграчки натпревари.", "",
              "| Соиграчи: | наивно (просек посл. 5) | обичен просек (M1) | M2 тим=240/200 | формула BUILD_PROMPT (M3) | M4 = M3 + тим |",
              "|---|---|---|---|---|---|",
              f"| MAE минути | {_mae(aff['min_last5'], aff['minutes']):.2f} | {_mae(aff['min_m1'], aff['minutes']):.2f} "
              f"| {_mae(aff['min_m2'], aff['minutes']):.2f} | {_mae(aff['min_m3'], aff['minutes']):.2f} "
              f"| {_mae(aff['min_m4'], aff['minutes']):.2f} |",
              f"| пристрасност минути | {(aff['min_last5'] - aff['minutes']).mean():+.2f} | {(aff['min_m1'] - aff['minutes']).mean():+.2f} "
              f"| {(aff['min_m2'] - aff['minutes']).mean():+.2f} | {(aff['min_m3'] - aff['minutes']).mean():+.2f} "
              f"| {(aff['min_m4'] - aff['minutes']).mean():+.2f} |"]
    lines += ["", f"Статистика на соиграчите (избран модел: минути {best.minutes_model.upper()}, usage "
                  f"{'да' if best.usage_boost else 'не'}):", "",
              "| Пазар | наивно (посл. 5) | обичен просек (M1 × стапка) | избран модел |", "|---|---|---|---|"]
    for s in STATS:
        plain = aff["min_m1"] * aff[f"{s}_rate"]
        lines.append(f"| {MARKET_MK[s]} | {_mae(aff[f'{s}_last5'], aff[s]):.2f} | {_mae(plain, aff[s]):.2f} "
                     f"| {_mae(aff_nd[f'proj_{s}'], aff_nd[s]):.2f} |")
    lines += ["", "*Негативна пристрасност = соиграчите добиле повеќе минути од предвиденото.*"]

    # ---- 3. markets: projection error ----
    lines += ["", "## 3. Проекција по пазар (вон примерок)", "",
              "| Пазар | n | MAE наивно | MAE модел (без DvP) | MAE модел (со DvP) | просек реално |", "|---|---|---|---|---|---|"]
    for s in STATS:
        lines.append(f"| {MARKET_MK[s]} | {len(test)} | {_mae(test[f'{s}_last5'], test[s]):.2f} "
                     f"| {_mae(test_nd[f'proj_{s}'], test_nd[s]):.2f} | {_mae(test[f'proj_{s}'], test[s]):.2f} "
                     f"| {test[s].mean():.2f} |")

    # ---- 4. distributions ----
    lines += ["", "## 4. Распределба → P(НАД линија): калибрација", "",
              "Линија = (наивен просек на последните 5, заокружен надолу) + 0.5 — таму каде што кладилницата обично ја става. "
              "Дисперзијата (negative binomial) е штимана на првите 60%.", ""]
    for s in STATS:
        r = fit_dispersion(tune[f"proj_{s}"].values, tune[s].values)
        t = test[test[f"{s}_last5"].notna()]
        line = np.floor(t[f"{s}_last5"].values) + 0.5
        p_mod = p_over(t[f"proj_{s}"].values, line, r)
        p_nv = p_over(t[f"{s}_last5"].values, line, 1e6)       # naive: Poisson around the last-5 mean
        over = (t[s].values > line).astype(float)
        q = np.maximum(p_mod, 1 - p_mod)                        # the side the model would pick
        hit = np.where(p_mod >= 0.5, over, 1 - over)
        lines += [f"### {MARKET_MK[s].capitalize()} — дисперзија r = {'∞ (Poisson)' if r >= 1e5 else f'{r:.1f}'}", "",
                  f"Brier: модел {np.mean((p_mod - over) ** 2):.4f} · наивно (Poisson околу посл. 5) "
                  f"{np.mean((p_nv - over) ** 2):.4f} · 50/50 = 0.2500", "",
                  "| Модел вели | НАД: n | НАД: паѓа (95%) | ПОД: n | ПОД: паѓа (95%) |", "|---|---|---|---|---|"]
        is_over = p_mod >= 0.5
        for lo, hi in ((0.5, 0.55), (0.55, 0.6), (0.6, 0.65), (0.65, 0.7), (0.7, 1.01)):
            cells = []
            for side_mask in (is_over, ~is_over):
                sel = (q >= lo) & (q < hi) & side_mask
                n = int(sel.sum())
                if not n:
                    cells += ["0", "—"]
                    continue
                h = int(hit[sel].sum())
                ci = wilson_interval(h, n)
                ok = "✅" if lo - 0.05 <= h / n <= min(hi, 1) + 0.05 else "❌"
                cells += [str(n), f"{h / n * 100:.1f}% {ok} ({ci[0] * 100:.0f}–{ci[1] * 100:.0f}%)"]
            lines.append(f"| {lo * 100:.0f}–{min(hi, 1) * 100:.0f}% | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["> ⚠️ Повторно: вештачки линии → **не е доказ за предност против Mozzart**. Вистинскиот тест е Фаза 7."]
    return "\n".join(lines)


# ---------------- low lines (Tome 29.10: EV +15–25% on low ПОД lines = alarm) ----------------

LL_MIN_GAIN = 0.001          # smaller out-of-sample log-loss gains are noise → no change (PROCES §6.4)


def _zinb_under(mu: np.ndarray, line: float, r: float, pi: float) -> np.ndarray:
    """Zero-inflated NB: P(X ≤ line) with an extra share pi of zeros (overall mean kept at mu)."""
    m = mu / (1 - pi)
    return pi + (1 - pi) * (1 - p_over(m, np.full(len(m), line), r))


def low_lines_report(settings: Settings, league: str) -> Optional[str]:
    """P(ПОД) from the model vs the real frequency at the low lines, last season, walk-forward.
    Distribution variants are chosen on the first 60% of dates and judged on the last 40% (PROCES §6.4)."""
    cfg = settings.league(league)
    df = _load(settings, league, cfg.prev_season)
    if df is None:
        print(f"{cfg.name_mk}: нема играчки логови за {cfg.prev_season}")
        return None
    p = params_from_config(cfg.player_model or {})
    d = project(add_features(df, league, p), league, p, dvp_by_game(df))
    d = d[d["n_prior"] >= MIN_PRIOR_GAMES]
    cut = _cut_date(df)
    tune, test = d[d["game_date"] < cut], d[d["game_date"] >= cut]
    lines = [f"# Backtest ниски линии — {cfg.name_mk} {cfg.prev_season}", "",
             f"*{date.today()} · `python main.py backtest --niski --league {league}` · walk-forward (проекцијата е од "
             f"претходните натпревари, параметри од settings.yaml). Избор на распределба: датуми < {cut} "
             f"({len(tune)} реда); оценување: од {cut} ({len(test)} реда).*", "",
             "За секој играч/натпревар: P(ПОД) од моделот наспроти реалната фреквенција (ПОД = резултат ≤ линијата). "
             "**+ = моделот го преценува ПОД** (опасно: лажен EV на ПОД).", "",
             "Варијанти: **A** = еден r по пазар (досега) · **B** = посебен r по опсег на проекцијата "
             f"(граници {MU_BINS}) · **C** = zero-inflated NB (вишок нули).", ""]
    chosen = {}
    for s, ths in LOW_LINES.items():
        mt, yt = tune[f"proj_{s}"].values, tune[s].values.astype(float)
        me, ye = test[f"proj_{s}"].values, test[s].values.astype(float)
        r0 = fit_dispersion(mt, yt)
        rb = fit_dispersion_by_mu(mt, yt, ths)
        zgrid = [(r, pi) for r in R_GRID for pi in (0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15)]
        zinb_ll = lambda mu, y, r, pi: sum(-float(np.mean(np.where(
            y <= th, np.log(np.clip(_zinb_under(mu, th, r, pi), 1e-6, 1)),
            np.log(np.clip(1 - _zinb_under(mu, th, r, pi), 1e-6, 1))))) for th in ths)
        rc, pic = min(zgrid, key=lambda x: zinb_ll(mt, yt, *x))
        under = {"A": lambda th: 1 - p_over(me, np.full(len(me), th), r0),
                 "B": lambda th: 1 - p_over(me, np.full(len(me), th), dispersion_for(me, {"r_po_mu": rb})),
                 "C": lambda th: _zinb_under(me, th, rc, pic)}
        ll = {"A": threshold_log_loss(me, ye, ths, r0),
              "B": threshold_log_loss(me, ye, ths, dispersion_for(me, {"r_po_mu": rb})),
              "C": zinb_ll(me, ye, rc, pic)}
        best = "B" if ll["A"] - ll["B"] >= LL_MIN_GAIN else "A"   # C only reported (tested, see table)
        chosen[s] = (best, r0, rb)
        lines += [f"## {MARKET_MK[s].capitalize()} — линии {', '.join(f'{t:g}' for t in ths)}", "",
                  f"A: r = {r0:.1f} · B: r по опсег = {', '.join(f'{x:g}' for x in np.round(rb, 1))} · "
                  f"C: r = {rc:g}, вишок нули π = {pic:g}", "",
                  f"Log loss (тест, помало = подобро): A {ll['A']:.4f} · B {ll['B']:.4f} · C {ll['C']:.4f} → "
                  f"**{'B (посебен r за мал λ)' if best == 'B' else 'A (без промена)'}** (праг за промена: подобрување ≥ {LL_MIN_GAIN})", "",
                  "| Линија | P(ПОД) модел | n | реално | A грешка | B грешка | C грешка |", "|---|---|---|---|---|---|---|"]
        for th in ths:
            pa, pb, pc = under["A"](th), under["B"](th), under["C"](th)
            real = (ye <= th).astype(float)
            for lo, hi in ((0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)):
                m = (pa >= lo) & (pa < hi)          # bucket by the CURRENT model so rows are the same for A/B/C
                n = int(m.sum())
                if n < 20:
                    continue
                h = int(real[m].sum())
                ci = wilson_interval(h, n)
                lines.append(f"| {th:g} | {lo * 100:.0f}–{min(hi, 1) * 100:.0f}% | {n} | {h / n * 100:.1f}% "
                             f"({ci[0] * 100:.0f}–{ci[1] * 100:.0f}%) | {(pa[m].mean() - h / n) * 100:+.1f} pp "
                             f"| {(pb[m].mean() - h / n) * 100:+.1f} pp | {(pc[m].mean() - h / n) * 100:+.1f} pp |")
        lines.append("")
    lines += ["## Одлука (за settings.yaml → player_model.dispersion)", ""]
    for s, (best, r0, rb) in chosen.items():
        val = f"{{r_po_mu: [{', '.join(f'{x:g}' for x in rb)}]}}" if best == "B" else f"{r0:.1f}"
        lines.append(f"- {MARKET_MK[s]} (`{s}`): {val}")
    lines += ["", "> Поени не се менуваат (линиите се високи, λ голем). Ова е калибрација на РАСПРЕДЕЛБАТА; не е "
              "доказ за предност против Mozzart — EV се смета дури по оваа поправка, а EV > 15% останува аларм "
              "(ПРОВЕРИ_ПОДАТОЦИ)."]
    return "\n".join(lines)
