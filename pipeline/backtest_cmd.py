"""`python main.py backtest` — walk-forward backtest of the team model (Phase 4) + Macedonian report."""
from __future__ import annotations

import logging
from dataclasses import replace
from datetime import date
from typing import Optional

import pandas as pd

from core.config import ROOT, Settings
from model.backtest import evaluate, split_by_date, tune, walk_forward
from model.team_model import ModelParams, game_frame, ladder
from store.stats import wilson_interval

log = logging.getLogger(__name__)
TUNE_SHARE = 0.6
MARKET_FILE = ROOT / "data" / "raw" / "nba" / "market_history" / "nba_2025_26_closing_lines_ats.csv"


def _load_games(settings: Settings, league: str, season: str) -> Optional[pd.DataFrame]:
    path = settings.paths.processed / league / season / "team_logs.csv"
    if not path.exists():
        return None
    logs = pd.read_csv(path, dtype={"game_code": str})
    games = game_frame(logs, league)
    games["season_code"] = season
    # EL/EC game codes restart every season (1, 2, 3…) → make them unique before mixing seasons
    games["game_code"] = season + "_" + games["game_code"]
    return games


def _nba_market(settings: Settings) -> Optional[pd.DataFrame]:
    """Consensus closing lines (Hugging Face, CC-BY-4.0) — benchmark only, never a model input."""
    if not MARKET_FILE.exists():
        return None
    m = pd.read_csv(MARKET_FILE)
    sched = pd.read_csv(settings.paths.processed / "nba" / settings.league("nba").season / "schedule.csv")
    names = pd.concat([sched[["home", "home_name"]].set_axis(["code", "name"], axis=1),
                       sched[["away", "away_name"]].set_axis(["code", "name"], axis=1)]).drop_duplicates()
    by_nick = {n.split()[-1]: c for c, n in names.itertuples(index=False)}
    m["home"] = m["home_team"].str.split().str[-1].map(by_nick)
    m["away"] = m["away_team"].str.split().str[-1].map(by_nick)
    if m[["home", "away"]].isna().any().any():
        log.warning("Пазар: непознати тимови %s", sorted(set(m.loc[m["home"].isna(), "home_team"])))
    return m.rename(columns={"date_et": "game_date"})[
        ["game_date", "home", "away", "home_score", "away_score", "close_total", "close_spread_home"]]


def _fmt(m) -> str:
    if m is None:
        return "| — | — | — | — | — | — | — |"
    return (f"| {m.n} | {m.mae_total:.2f} | {m.within5_total * 100:.0f}% | {m.mae_margin:.2f} | "
            f"{m.within5_margin * 100:.0f}% | {m.mae_team:.2f} | {m.bias_total:+.2f} |")


HEADER = ("| Модел | n | MAE тотал | тотал ±5 | MAE разлика | разлика ±5 | MAE тимски поени | пристрасност тотал |\n"
          "|---|---|---|---|---|---|---|---|")


def _market_section(preds: pd.DataFrame, market: pd.DataFrame, label: str) -> list[str]:
    p = preds.merge(market, on=["game_date", "home", "away"], how="inner")
    p = p[(p["n_home"] >= 5) & (p["n_away"] >= 5)]
    if p.empty:
        return [f"*{label}: нема натпревари со затворачка линија.*"]
    bad = (p["home_score"] != p["act_home"]) | (p["away_score"] != p["act_away"])
    p = p[~bad]
    act_total = p["act_home"] + p["act_away"]
    act_margin = p["act_home"] - p["act_away"]
    mod_total = p["pred_home"] + p["pred_away"]
    mod_margin = p["pred_home"] - p["pred_away"]
    mkt_margin = -p["close_spread_home"]
    lines = [f"**{label}** — {len(p)} натпревари со затворачка линија"
             + (f" ({int(bad.sum())} исклучени: резултатот не се совпаѓа)" if bad.any() else ""), "",
             "| | MAE тотал | тотал ±5 | MAE разлика | разлика ±5 |", "|---|---|---|---|---|",
             f"| Модел | {(mod_total - act_total).abs().mean():.2f} | {((mod_total - act_total).abs() <= 5).mean() * 100:.0f}% "
             f"| {(mod_margin - act_margin).abs().mean():.2f} | {((mod_margin - act_margin).abs() <= 5).mean() * 100:.0f}% |",
             f"| Затворачка линија (пазар) | {(p['close_total'] - act_total).abs().mean():.2f} "
             f"| {((p['close_total'] - act_total).abs() <= 5).mean() * 100:.0f}% "
             f"| {(mkt_margin - act_margin).abs().mean():.2f} | {((mkt_margin - act_margin).abs() <= 5).mean() * 100:.0f}% |",
             "", "Кога моделот се разликува од линијата — на која страна паѓа резултатот (информативно, не за уплата):", "",
             "| Разлика модел−линија | тотал: n | тотал: погодок (95%) | хендикеп: n | хендикеп: погодок (95%) |",
             "|---|---|---|---|---|"]
    for thr in (2, 4, 6):
        dt = mod_total - p["close_total"]
        sel = dt.abs() >= thr
        res_t = act_total[sel] - p["close_total"][sel]
        hits_t = int(((res_t > 0) == (dt[sel] > 0))[res_t != 0].sum()); n_t = int((res_t != 0).sum())
        dm = mod_margin - mkt_margin
        sel_m = dm.abs() >= thr
        res_m = act_margin[sel_m] - mkt_margin[sel_m]
        hits_m = int(((res_m > 0) == (dm[sel_m] > 0))[res_m != 0].sum()); n_m = int((res_m != 0).sum())
        ci = lambda h, n: (f"{h / n * 100:.1f}% ({wilson_interval(h, n)[0] * 100:.0f}–{wilson_interval(h, n)[1] * 100:.0f}%)"
                           if n else "—")
        lines.append(f"| ≥ {thr} поени | {n_t} | {ci(hits_t, n_t)} | {n_m} | {ci(hits_m, n_m)} |")
    lines.append("")
    lines.append("*Праг на исплатливост на квота 1.90 = 52.6%. Интервалот покажува колку е сигурно.*")
    return lines


def backtest_league(settings: Settings, league: str) -> Optional[str]:
    cfg = settings.league(league)
    prev = _load_games(settings, league, cfg.prev_season) if cfg.prev_season else None
    cur = _load_games(settings, league, cfg.season)
    if prev is None:
        print(f"{cfg.name_mk}: нема податоци за минатата сезона — прво `python main.py fetch --league {league} "
              f"--season {cfg.prev_season}`")
        return None
    hca_prior = cfg.home_advantage_pts
    lines = [f"# Backtest тимски модел — {cfg.name_mk}", "",
             f"*Направено {date.today()} · `python main.py backtest --league {league}` · walk-forward: секоја проекција "
             f"користи само натпревари со датум ПРЕД тој натпревар.*", "",
             f"- Минатата сезона **{cfg.prev_season}**: {prev['game_code'].nunique()} натпревари "
             f"({prev['game_date'].min()} – {prev['game_date'].max()}).",
             f"- Штимање на параметрите: првите {int(TUNE_SHARE * 100)}% од датумите; **оценување: последните "
             f"{100 - int(TUNE_SHARE * 100)}%** (моделот не ги видел) и тековната сезона {cfg.season}.",
             "- Оценуваат се натпревари каде двата тима имаат ≥ 5 претходни натпревари.", ""]

    # 1) tune shrinkage + recency on the tuning part of last season
    full = [p for p in ladder() if p.name.startswith("V6")][0]
    tuned, grid = tune(prev, full, hca_prior, TUNE_SHARE)
    lines += ["## 1. Штимање (само на првите 60%)", "",
              f"Најдобро: shrinkage = {tuned.shrink_games:g} натпревари кон просекот на лигата, "
              f"форма (half-life) = {tuned.half_life if tuned.half_life else 'без'} натпревари.", "",
              "| shrink | half-life | MAE тотал | MAE разлика |", "|---|---|---|---|"]
    for r in grid.head(6).itertuples():
        lines.append(f"| {r.shrink_games:g} | {'без' if pd.isna(r.half_life) else f'{r.half_life:g}'} "
                     f"| {r.mae_total:.2f} | {r.mae_margin:.2f} |")

    # 2) ablation ladder with tuned settings
    variants = []
    for p in ladder():
        p = replace(p, shrink_games=tuned.shrink_games)
        if p.half_life is not None:
            p = replace(p, half_life=tuned.half_life)
        variants.append(p)
    results = {}
    combined = pd.concat([prev, cur]) if cur is not None else prev
    for p in variants:
        log.info("%s: %s…", cfg.name_mk, p.name)
        pr_prev = walk_forward(prev, p, hca_prior)
        pr_all = walk_forward(combined, p, hca_prior) if cur is not None else None
        results[p.name] = (pr_prev, pr_all)
    lines += ["", "## 2. Скалила — што додава секој параметар", "",
              f"### Минатата сезона, последните {100 - int(TUNE_SHARE * 100)}% (вон примерок)", "", HEADER]
    for name, (pr_prev, _) in results.items():
        lines.append(f"| {name} " + _fmt(evaluate(split_by_date(pr_prev, TUNE_SHARE)[1])))
    if cur is not None:
        cur_codes = set(cur["game_code"])
        lines += ["", f"### Тековна сезона {cfg.season} (минатата сезона служи како историја)", "", HEADER]
        for name, (_, pr_all) in results.items():
            lines.append(f"| {name} " + _fmt(evaluate(pr_all[pr_all["game_code"].isin(cur_codes)])))
        lines.append("")
        lines.append("*Тековната сезона има малку натпревари → само насока.*")

    # 3) drop-one from the full model on the out-of-sample part
    best_name = min(results, key=lambda n: (lambda m: m.mae_total + m.mae_margin if m else 1e9)(
        evaluate(split_by_date(results[n][0], TUNE_SHARE)[1])))
    full_tuned = variants[-1]
    drops = {"без дома/гости": replace(full_tuned, use_home=False),
             "без темпо на тимот": replace(full_tuned, use_pace=False),
             "без прилагодување за противници": replace(full_tuned, opp_adjust=False),
             "без форма": replace(full_tuned, half_life=None),
             "без Four Factors": replace(full_tuned, use_ff=False),
             "без одмор": replace(full_tuned, use_rest=False)}
    base_m = evaluate(split_by_date(results[full_tuned.name][0], TUNE_SHARE)[1])
    lines += ["", "## 3. Кој параметар најмногу помага (V6 без еден параметар, вон примерок)", "",
              "Позитивна промена = грешката расте кога параметарот се вади → параметарот помага.", "",
              "| Без | Δ MAE тотал | Δ MAE разлика | Δ MAE тимски поени |", "|---|---|---|---|"]
    deltas = {}
    for label, p in drops.items():
        m = evaluate(split_by_date(walk_forward(prev, p, hca_prior), TUNE_SHARE)[1])
        deltas[label] = (m.mae_total - base_m.mae_total) + (m.mae_margin - base_m.mae_margin)
        lines.append(f"| {label} | {m.mae_total - base_m.mae_total:+.2f} | {m.mae_margin - base_m.mae_margin:+.2f} "
                     f"| {m.mae_team - base_m.mae_team:+.2f} |")
    lines += ["", "*DvP (одбрана по позиција) не влегува во тимски тотал — се тестира во Фаза 5 на играчки пазари.*"]

    # 4) market benchmark
    lines += ["", "## 4. Споредба со пазарот (затворачка линија)", ""]
    if league == "nba" and (market := _nba_market(settings)) is not None:
        best_prev = results[best_name][0]
        lines += [f"Модел: **{best_name}**. Пазарот е само мерило — не е влез во моделот.", ""]
        lines += _market_section(split_by_date(best_prev, TUNE_SHARE)[1], market, "Последните 40% од 2025-26")
        lines += [""] + _market_section(best_prev, market, "Цела сезона 2025-26 (вкл. делот за штимање)")
        lines += ["", "Извор на линиите: Hugging Face `Cameronkelly92/nba-closing-lines-ats-2025-26` (CC-BY-4.0), "
                      "консензус затворачки линии."]
    else:
        lines.append("Нема бесплатни историски затворачки линии за оваа лига → споредба само со наивните модели "
                     "(V0, V1). Од Фаза 7 линиите од Mozzart се зачувуваат секојдневно → споредбата ќе стане можна.")

    oos = {n: evaluate(split_by_date(results[n][0], TUNE_SHARE)[1]) for n in results}
    v1 = next(m for n, m in oos.items() if n.startswith("V1"))
    best = oos[best_name]
    helps = [k.replace("без ", "") for k, v in sorted(deltas.items(), key=lambda kv: -kv[1]) if v > 0.05]
    useless = [k.replace("без ", "") for k, v in deltas.items() if v <= 0.05]
    lines += ["", "## 5. Заклучок", "",
              f"- Најдобар вон примерок (MAE тотал + разлика): **{best_name}** — тотал {best.mae_total:.2f}, "
              f"разлика {best.mae_margin:.2f}.",
              f"- Наспроти рачниот метод „(напад+одбрана)/2“ (V1): тотал {best.mae_total - v1.mae_total:+.2f}, "
              f"разлика {best.mae_margin - v1.mae_margin:+.2f} поени MAE (негативно = подобро).",
              f"- Помага (по големина): {', '.join(helps) if helps else 'ништо значајно'}.",
              f"- Не помага / шум (≤ 0.05): {', '.join(useless) if useless else '—'}."]
    out_dir = settings.paths.processed / league / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, (pr_prev, pr_all) in results.items():
        safe = name.split()[0]
        pr_prev.to_csv(out_dir / f"{cfg.prev_season}_{safe}.csv", index=False)
    return "\n".join(lines)


def cmd_backtest(settings: Settings, league: str, kind: str = "tim") -> int:
    leagues = ["euroleague", "eurocup", "nba"] if league == "all" else [league]
    reports = settings.paths.reports
    reports.mkdir(parents=True, exist_ok=True)
    for lg in leagues:
        from pipeline.candidates import candidate_report
        from pipeline.devig_report import devig_report
        from pipeline.player_backtest import low_lines_report, player_report
        special = {"pocetok": season_start_report, "odmor": rest_report, "igraci": player_report,
                   "kandidati": candidate_report, "pristrasnost": bias_report, "niski": low_lines_report}
        if kind == "devig":                   # not per league: all snapshots at once
            if lg != leagues[0]:
                continue
            special["devig"] = lambda s, _lg: devig_report(s)
            lg = "site"
        text = special[kind](settings, lg) if kind in special else backtest_league(settings, lg)
        if kind in special and text is not None:
            path = reports / f"backtest_{kind}_{lg}_{date.today()}.md"
            path.write_text(text, encoding="utf-8")
            print(text)
            print(f"Извештај: {path}")
            continue
        if text is None:
            continue
        path = reports / f"backtest_tim_{lg}_{date.today()}.md"
        path.write_text(text, encoding="utf-8")
        print(text)
        print(f"\nИзвештај: {path}\n")
    return 0


# ---------------- season start (decision 29.10: fix the between-season shift first) ----------------

def _start_dates(games: pd.DataFrame, season: str, days: int = 28) -> set:
    g = games[games["season_code"] == season]
    first = pd.Timestamp(g["game_date"].min())
    return set(g.loc[pd.to_datetime(g["game_date"]) < first + pd.Timedelta(days=days), "game_date"])


def season_start_report(settings: Settings, league: str) -> Optional[str]:
    from model.team_model import params_from_config
    cfg = settings.league(league)
    seasons = [cfg.prev2_season, cfg.prev_season, cfg.season]
    data = {s: _load_games(settings, league, s) for s in seasons if s}
    missing = [s for s in seasons if data.get(s) is None]
    if missing:
        print(f"{cfg.name_mk}: недостасуваат сезони {missing} — `python main.py fetch --league {league} --season ...`")
        return None
    base = params_from_config(cfg.team_model)
    hca = cfg.home_advantage_pts
    hist_a = pd.concat([data[cfg.prev2_season], data[cfg.prev_season]])
    days_a = _start_dates(hist_a, cfg.prev_season)
    hist_b = pd.concat([data[cfg.prev_season], data[cfg.season]])
    days_b = set(data[cfg.season]["game_date"])

    variants = {"спојено (досегашно)": replace(base, season_mode="pooled"),
                "само тековна сезона": replace(base, season_mode="current_only")}
    for carry in (0.3, 0.5, 0.7):
        for prior in (5.0, 10.0):
            for lg_prior in (15.0, 60.0):
                variants[f"пренос carry={carry} P={prior:g} лига={lg_prior:g}"] = replace(
                    base, season_mode="carryover", carry=carry, prior_games=prior, lg_prior_games=lg_prior)
    rows = []
    for name, p in variants.items():
        log.info("%s почеток на сезона: %s…", cfg.name_mk, name)
        ma = evaluate(walk_forward(hist_a, p, hca, only_dates=days_a).query("game_date in @days_a"), burn_in=0)
        mb = evaluate(walk_forward(hist_b, p, hca, only_dates=days_b).query("game_date in @days_b"), burn_in=0)
        rows.append((name, ma, mb))
    tuned = min((r for r in rows if r[0].startswith("пренос")), key=lambda r: r[1].mae_total + r[1].mae_margin)
    head = ("| Начин | n | MAE тотал | пристрасност тотал | MAE разлика | MAE тимски поени |\n|---|---|---|---|---|---|")
    fmt = lambda m: f"| {m.n} | {m.mae_total:.2f} | {m.bias_total:+.2f} | {m.mae_margin:.2f} | {m.mae_team:.2f} |"
    lines = [f"# Почеток на сезона — {cfg.name_mk}", "",
             f"*{date.today()} · walk-forward, без гледање во иднина. Базен модел: {base.name} (од settings.yaml).*", "",
             f"## А) Првите 4 недели од {cfg.prev_season} (историја: {cfg.prev2_season}) — тука се штима", "", head]
    lines += [f"| {n} " + fmt(ma) for n, ma, _ in rows]
    lines += ["", f"## Б) Тековна сезона {cfg.season} досега (историја: {cfg.prev_season}) — вон примерок", "", head]
    lines += [f"| {n} " + fmt(mb) for n, _, mb in rows]
    pooled = next(r for r in rows if r[0].startswith("спојено"))
    lines += ["", "## Заклучок", "",
              f"- Избран на А: **{tuned[0]}** (лигин просек се ажурира секоја недела).",
              f"- Б (вон примерок): тотал MAE {tuned[2].mae_total:.2f} наспроти спојено {pooled[2].mae_total:.2f}; "
              f"пристрасност {tuned[2].bias_total:+.2f} наспроти {pooled[2].bias_total:+.2f}."]
    return "\n".join(lines)


# ---------------- rest / back-to-back (decision 29.10: categories, not linear days of rest) ----------------

def rest_report(settings: Settings, league: str) -> Optional[str]:
    from model.backtest import rest_effects
    from model.team_model import params_from_config
    cfg = settings.league(league)
    base = replace(params_from_config(cfg.team_model), use_rest=False)
    hca = cfg.home_advantage_pts
    lines = [f"# Одмор / back-to-back — {cfg.name_mk}", "",
             f"*{date.today()} · категории: B2B (втор натпревар во 2 дена) и B2B гостин-гостин; ефектот се штима "
             f"само од претходни натпревари (walk-forward), со shrinkage кон 0.*", ""]
    head = "| Сезона / дел | n | MAE тотал без / со | MAE разлика без / со | MAE тимски поени без / со |\n|---|---|---|---|---|"
    table, coef_lines = [head], []
    for season, part in ((cfg.prev2_season, "цела"), (cfg.prev_season, "последни 40%")):
        games = _load_games(settings, league, season) if season else None
        if games is None:
            continue
        n_b2b = int(games["b2b"].sum()); n_aa = int(games["b2b_aa"].sum())
        lines.append(f"- {season}: {len(games)} тимски натпревари, B2B {n_b2b} ({n_b2b / len(games) * 100:.1f}%), "
                     f"од нив гостин-гостин {n_aa}.")
        if n_b2b < 30:
            continue
        p0 = walk_forward(games, base, hca)
        p1 = walk_forward(games, replace(base, use_rest=True), hca)
        if part != "цела":
            p0, p1 = split_by_date(p0, TUNE_SHARE)[1], split_by_date(p1, TUNE_SHARE)[1]
        m0, m1 = evaluate(p0), evaluate(p1)
        table.append(f"| {season} ({part}) | {m0.n} | {m0.mae_total:.2f} / {m1.mae_total:.2f} | "
                     f"{m0.mae_margin:.2f} / {m1.mae_margin:.2f} | {m0.mae_team:.2f} / {m1.mae_team:.2f} |")
        # descriptive effect on the whole season (residuals of the rest-free model)
        full = walk_forward(games, base, hca).merge(
            games[games["home"]][["game_code", "b2b", "b2b_aa", "opp_b2b", "opp_b2b_aa"]], on="game_code")
        rows = [(r.act_home - r.pred_home, r.b2b, r.b2b_aa, r.opp_b2b, r.opp_b2b_aa) for r in full.itertuples()]
        rows += [(r.act_away - r.pred_away, r.opp_b2b, r.opp_b2b_aa, r.b2b, r.b2b_aa) for r in full.itertuples()]
        c = rest_effects(rows)
        if c is not None:
            coef_lines.append(f"| {season} | {c[0]:+.2f} | {c[1]:+.2f} | {c[2]:+.2f} | {c[3]:+.2f} |")
    if len(table) == 1:
        lines += ["", "Нема доволно B2B натпревари (< 30) → одморот не се тестира и останува исклучен."]
        return "\n".join(lines)
    lines += ["", *table]
    lines += ["", "Ефект во поени на тимот (резидуал на моделот без одмор, со shrinkage):", "",
              "| Сезона | сопствен B2B | дополнително гостин-гостин | противник B2B | противник гостин-гостин |",
              "|---|---|---|---|---|", *coef_lines,
              "", "*Ако „со“ не е подобро од „без“ на двете сезони → одморот останува исклучен (одлука 29.10).*"]
    return "\n".join(lines)


def bias_report(settings: Settings, league: str) -> Optional[str]:
    """Season start: faster league average (лига=15) and/or weekly bias correction (decision 29.10)."""
    from model.team_model import params_from_config
    cfg = settings.league(league)
    seasons = [cfg.prev2_season, cfg.prev_season, cfg.season]
    data = {s: _load_games(settings, league, s) for s in seasons}
    if any(v is None for v in data.values()):
        return None
    base = params_from_config(cfg.team_model)
    if base.season_mode == "pooled":
        base_seasonal = replace(base, season_mode="carryover", carry=0.7, prior_games=5.0)
    else:
        base_seasonal = base
    hist_a = pd.concat([data[cfg.prev2_season], data[cfg.prev_season]]); days_a = _start_dates(hist_a, cfg.prev_season)
    hist_b = pd.concat([data[cfg.prev_season], data[cfg.season]]); days_b = set(data[cfg.season]["game_date"])
    variants = {"тековна конфигурација": base,
                "лига=15": replace(base_seasonal, lg_prior_games=15.0),
                "неделна корекција на пристрасност": replace(base_seasonal, weekly_bias=True),
                "лига=15 + неделна корекција": replace(base_seasonal, lg_prior_games=15.0, weekly_bias=True)}
    lines = [f"# Пристрасност на почеток на сезона — {cfg.name_mk}", "",
             f"*{date.today()} · walk-forward · А = првите 4 недели од {cfg.prev_season}, Б = {cfg.season} досега (вон примерок).*", "",
             "| Варијанта | А MAE тотал | А пристрасност | А MAE разлика | Б MAE тотал | Б пристрасност | Б MAE разлика |",
             "|---|---|---|---|---|---|---|"]
    for name, p in variants.items():
        a = evaluate(walk_forward(hist_a, p, cfg.home_advantage_pts, only_dates=days_a).query("game_date in @days_a"), burn_in=0)
        b = evaluate(walk_forward(hist_b, p, cfg.home_advantage_pts, only_dates=days_b).query("game_date in @days_b"), burn_in=0)
        lines.append(f"| {name} | {a.mae_total:.2f} | {a.bias_total:+.2f} | {a.mae_margin:.2f} | {b.mae_total:.2f} "
                     f"| {b.bias_total:+.2f} | {b.mae_margin:.2f} |")
    lines += ["", "*Пресуда: промена само ако А (штимање) е подобро и Б не е полошо.*"]
    return "\n".join(lines)


def team_model_week_monitor(settings: Settings, days: int = 7) -> list[str]:
    """Weekly check (decision 29.10): how biased was the team model on last week's finished games?"""
    from model.team_model import params_from_config
    out = ["", "ТИМСКИ МОДЕЛ — последните 7 дена (walk-forward, тековна конфигурација):"]
    for league in ("nba", "euroleague", "eurocup"):
        cfg = settings.league(league)
        cur = _load_games(settings, league, cfg.season)
        prev = _load_games(settings, league, cfg.prev_season) if cfg.prev_season else None
        if cur is None:
            continue
        hist = pd.concat([prev, cur]) if prev is not None else cur
        last = pd.Timestamp(cur["game_date"].max())
        week = set(cur.loc[pd.to_datetime(cur["game_date"]) > last - pd.Timedelta(days=days), "game_date"])
        preds = walk_forward(hist, params_from_config(cfg.team_model), cfg.home_advantage_pts, only_dates=week)
        m = evaluate(preds[preds["game_date"].isin(week)], burn_in=0)
        if m is None:
            continue
        flag = " ⚠️" if abs(m.bias_total) >= 3 else ""
        out.append(f"  {cfg.name_mk:<9} n={m.n:<3} MAE тотал {m.mae_total:.2f} · пристрасност тотал {m.bias_total:+.2f}{flag} "
                   f"· MAE разлика {m.mae_margin:.2f}")
    return out
