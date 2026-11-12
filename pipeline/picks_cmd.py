"""`python main.py picks` — Phase 7, PAPER mode by default (decision 29.10): model → both sides → market →
filters → ticket → everything recorded (status CEKA, uplateno = 0). Nothing is ever staked automatically.
"""
from __future__ import annotations

import json
import logging
from dataclasses import replace
from datetime import date, datetime
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from core.config import Settings
from model import absences as ab
from model import picks as pk
from model.calibration import league_shrink, shrink_toward_half
from model.dvp import dvp_table, primary_position
from model.lines import DEVIG_METHOD, build_day
from model.player_model import params_from_config as player_params
from model.scenarios import PlayerContext, player_scenarios, team_scenarios
from model.team_model import compute_ratings, params_from_config as team_params, predict_game
from pipeline.backtest_cmd import _load_games
from pipeline.fetch import fetch_league
from pipeline.injuries_cmd import league_absences
from pipeline.lines_cmd import attach_team_codes
from pipeline.profile_cmd import TEAMS_DIR
from pipeline.team_section import team_markets_section
from scrapers.http_client import SourceError
from store import db

log = logging.getLogger(__name__)
MODEL_VERSION = "py-v1"
DEFAULT_SIGMA = {"nba": (13.2, 18.5, 13.0), "euroleague": (10.0, 15.5, 10.5), "eurocup": (10.5, 17.0, 12.0)}
STATS = {"poeni": "pts", "asistencii": "ast", "skokovi": "reb", "trojki": "fgm3"}
DVP_CAP = {"nba": 0.15, "euroleague": 0.12, "eurocup": 0.12}


def team_sigmas(settings: Settings, league: str) -> tuple[float, float, float]:
    """σ of team points / total / margin from last season's walk-forward residuals (Phase 4 backtest files)."""
    cfg = settings.league(league)
    variant = (cfg.team_model or {}).get("variant", "V4")
    path = settings.paths.processed / league / "backtest" / f"{cfg.prev_season}_{variant}.csv"
    if not path.exists():
        return DEFAULT_SIGMA[league]
    p = pd.read_csv(path)
    p = p[(p["n_home"] >= 5) & (p["n_away"] >= 5)]
    team = pd.concat([p["pred_home"] - p["act_home"], p["pred_away"] - p["act_away"]])
    total = (p["pred_home"] + p["pred_away"]) - (p["act_home"] + p["act_away"])
    margin = (p["pred_home"] - p["pred_away"]) - (p["act_home"] - p["act_away"])
    return float(team.std()), float(total.std()), float(margin.std())


def team_projections(settings: Settings, league: str, day: date, games: pd.DataFrame) -> dict:
    cfg = settings.league(league)
    cur = _load_games(settings, league, cfg.season)
    prev = _load_games(settings, league, cfg.prev_season) if cfg.prev_season else None
    hist = pd.concat([prev, cur]) if prev is not None else cur
    hist = hist[hist["game_date"] < day.isoformat()]
    params = team_params(cfg.team_model)
    r = compute_ratings(hist, params, cfg.home_advantage_pts, season=cfg.season, as_of=day.isoformat())
    # tied after regulation = went to overtime (league history) → P(X) of the 1X2 market
    p_x = float((hist["overtimes"].fillna(0) > 0).mean()) if "overtimes" in hist else None
    out = {}
    for g in games.itertuples():
        mh, ma = predict_game(r, g.home, g.away, params)
        th = r.teams.loc[g.home] if g.home in r.teams.index else None
        ta = r.teams.loc[g.away] if g.away in r.teams.index else None
        poss = (th["pace"] + ta["pace"] - r.lg_pace) if th is not None and ta is not None else r.lg_pace
        out[str(g.game_code)] = {"home": g.home, "away": g.away, "mu_home": mh, "mu_away": ma,
                                 "poss": float(poss), "p_x": p_x,
                                 "pace_h": (th["pace"] - r.lg_pace) if th is not None else 0.0,
                                 "pace_a": (ta["pace"] - r.lg_pace) if ta is not None else 0.0,
                                 "drtg_h": (th["drtg"] - r.lg_ortg) if th is not None else 0.0,
                                 "drtg_a": (ta["drtg"] - r.lg_ortg) if ta is not None else 0.0,
                                 "time": getattr(g, "time_local", ""), "date_utc": g.date_utc}
    return out


def _names(roster: pd.DataFrame) -> pd.DataFrame:
    full = roster["player"].fillna("")
    el = full.str.contains(",")
    first = np.where(el, full.str.split(",").str[1].fillna("").str.strip(), full.str.split(" ").str[0])
    last = np.where(el, full.str.split(",").str[0], full.str.split(" ", n=1).str[1].fillna(""))
    return roster.assign(first=first, last=last)


def player_projections(settings: Settings, league: str, day: date, games: pd.DataFrame, absent: pd.DataFrame,
                       logs: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    cfg = settings.league(league)
    params = player_params(cfg.player_model)
    names = _names(roster).drop_duplicates("player_code").set_index("player_code")
    dvp = dvp_table(logs, roster, before=day.isoformat())
    rows = []
    for g in games.itertuples():
        for team, opp in ((g.home, g.away), (g.away, g.home)):
            a = absent[absent["team"] == team] if len(absent) else pd.DataFrame(columns=["player_code", "status", "novo", "minuti_sezona"])
            pairs = list(zip(a["player_code"], a["status"]))
            out_codes = {c for c, st in pairs if c and ab.plays(st, league) is False}
            if league != "nba":           # EL/EC: missed the last game, no news → treated as out (flagged in povredi)
                out_codes |= {c for c, st in pairs if c and st == "ПРОПУШТИЛ ПОСЛЕДЕН"}
            q_codes = {c for c, st in pairs if c and ab.plays(st, league) is None}
            new_out = a[(a["novo"] == True) & a["player_code"].isin(out_codes)]  # noqa: E712
            old = set(a[(a["novo"] == False) & a["player_code"].isin(out_codes)]["player_code"])  # noqa: E712
            eff = ab.pregame_minutes(logs, roster, league, team, day.isoformat(), out_codes, params, old_out=old)
            if eff.empty:
                continue
            eff = eff[~eff["out"]].copy()
            eff["game_code"], eff["team"], eff["opp"] = str(g.game_code), team, opp
            eff["home"] = team == g.home
            eff["first"] = eff["player_code"].map(names["first"])
            eff["last"] = eff["player_code"].map(names["last"])
            eff["q_self"] = eff["player_code"].isin(q_codes)
            eff["q_team"] = bool(q_codes)
            eff["novo_otsustvo"] = len(new_out) > 0
            for s, col in STATS.items():
                d = dvp[(dvp["tim"] == opp)].set_index("pozicija")[f"{col}_vs_liga_pct"] if len(dvp) else pd.Series(dtype=float)
                f = eff["pos"].map(d).fillna(0) / 100 * params.dvp_weight
                eff[f"dvp_{col}"] = eff["pos"].map(d)
                eff[f"mu_{col}"] = eff[f"proj_{col}"] * (1 + f.clip(-DVP_CAP[league], DVP_CAP[league]))
            # rule 4: first scorer of a team with a NEW rotation absence
            eff["prv_strelec"] = False
            if len(new_out) and (new_out["minuti_sezona"].fillna(0) >= 20).any():
                eff.loc[eff["mu_pts"].idxmax(), "prv_strelec"] = True
            rows.append(eff)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _latest(lines: pd.DataFrame) -> pd.DataFrame:
    last = lines.groupby("kladilnica")["snimka"].transform("max")
    return lines[(lines["snimka"] == last) & lines["glavna"].astype(bool)]


def build_candidates(settings: Settings, league: str, day: date, games: pd.DataFrame, teamp: dict,
                     players: pd.DataFrame, lines: pd.DataFrame) -> list[pk.Candidate]:
    cfg = settings.league(league)
    s_team, s_total, s_margin = team_sigmas(settings, league)
    disp = (cfg.player_model or {}).get("dispersion", {})
    shrink = league_shrink(league)
    L = _latest(lines[lines["liga"] == league])
    cands: dict[tuple, pk.Candidate] = {}

    def add(key, make, book, k_side, k_other, nat_id=None, p_kal=None):
        c = cands.get(key)
        if c is None:
            c = make()
            if p_kal is None:
                pk.calibrate(c, shrink)
            else:                                     # three-way market: calibrated by the caller
                c.p_kal = p_kal
            cands[key] = c
        pk.price(c, book, k_side, k_other, nat_id)

    # ---- team markets ----
    fair_1x2: dict[tuple, float] = {}
    for gc, t in teamp.items():
        g = L[(L["tim_d"] == t["home"]) & (L["tim_g"] == t["away"])]
        nat = f"{t['home']} - {t['away']}"
        for (book, pazar, line), grp in g.groupby(["kladilnica", "pazar", "linija"]):
            sides = grp.set_index("strana")["kvota"].to_dict()
            if pazar in ("vkupno", "tim_vkupno_d", "tim_vkupno_g") and {"nad", "pod"} <= set(sides):
                mu, sig = ((t["mu_home"] + t["mu_away"], s_total) if pazar == "vkupno" else
                           (t["mu_home"], s_team) if pazar == "tim_vkupno_d" else (t["mu_away"], s_team))
                tim = None if pazar == "vkupno" else (t["home"] if pazar == "tim_vkupno_d" else t["away"])
                p_ov = pk.p_normal_over(mu, sig, line)
                for side, p_side in (("nad", p_ov), ("pod", 1 - p_ov)):
                    other = "pod" if side == "nad" else "nad"
                    mk = "vkupno" if pazar == "vkupno" else "tim_vkupno"
                    add((gc, mk, tim, side, line),
                        lambda: pk.Candidate(league, gc, nat, tim, None, None, None, mk, side, float(line), mu, p_side),
                        book, sides[side], sides[other], grp["nat_id"].iat[0])
            elif pazar == "1x2" and {"1", "X", "2"} <= set(sides) and t.get("p_x") is not None:
                # regulation time: P(X) = league overtime rate; the rest split by the margin distribution
                # (the league shrink toward 50% applies to "who leads after regulation", never to P(X))
                mu_m = t["mu_home"] - t["mu_away"]
                q = pk.p_handicap_cover(mu_m, s_margin, 0.0)
                q_k = shrink_toward_half(q, shrink) if shrink is not None else q
                px = t["p_x"]
                probs = {"1": (1 - px) * q, "X": px, "2": (1 - px) * (1 - q)}
                probs_k = {"1": (1 - px) * q_k, "X": px, "2": (1 - px) * (1 - q_k)}
                inv = {k: 1 / sides[k] for k in ("1", "X", "2")}
                for side, team in (("1", t["home"]), ("X", None), ("2", t["away"])):
                    add((gc, "1x2", None, side, -1.0),
                        lambda s=side, tm=team: pk.Candidate(league, gc, nat, tm, None, None, None, "1x2", s, -1.0,
                                                            mu_m if s != "2" else -mu_m, probs[s]),
                        book, sides[side], None, grp["nat_id"].iat[0], p_kal=probs_k[side])
                    fair_1x2[(gc, side, book)] = inv[side] / sum(inv.values())     # proportional 3-way de-vig
            elif pazar == "hendikep" and book == "mozzart" and {"1", "2"} <= set(sides):
                mu_m = t["mu_home"] - t["mu_away"]
                p_home = pk.p_handicap_cover(mu_m, s_margin, line)          # Mozzart line = home team's handicap
                for side, team, opp, tl, p_side, o in (("1", t["home"], t["away"], line, p_home, "2"),
                                                        ("2", t["away"], t["home"], -line, 1 - p_home, "1")):
                    add((gc, "hendikep", team, team, tl),
                        lambda: pk.Candidate(league, gc, nat, team, opp, None, None, "hendikep", team, float(tl),
                                             mu_m if side == "1" else -mu_m, p_side),
                        book, sides[side], sides[o], grp["nat_id"].iat[0])

    for c in cands.values():                          # 1X2 is three-way: market P from the best book's three prices
        if c.pazar == "1x2":
            c.p_pazar = fair_1x2.get((c.game_code, c.strana, c.kladilnica))

    # ---- player markets ----
    if len(players):
        pl = L[L["igrac"].notna() & L["pazar"].isin(STATS)]
        sched = games.assign(start=pd.to_datetime(games["date_utc"], utc=True))
        for (book, name, pazar, line), grp in pl.groupby(["kladilnica", "igrac", "pazar", "linija"]):
            sides = grp.set_index("strana")["kvota"].to_dict()
            if not {"nad", "pod"} <= set(sides):
                continue
            r0 = grp.iloc[0]
            if book == "evrotip":
                own = attach_team_codes(pd.DataFrame([{"liga": league, "domakjin": None, "gostin": r0["tim_igrac"]}]),
                                        settings)["tim_g"].iat[0]          # evrotip: `away` = the player's own team
                gm = games[(games["home"] == own) | (games["away"] == own)]
            else:
                start = pd.to_datetime(r0["pocetok_utc"], utc=True)
                gm = sched[(sched["start"] - start).abs() <= pd.Timedelta(minutes=45)]
            pool = players[players["game_code"].isin(gm["game_code"].astype(str))]
            if book == "evrotip" and len(gm):
                pool = pool[pool["team"] == own]
            hit = pk.match_book_player(name, pool.to_dict("records"))
            if hit is None:
                continue
            col = STATS[pazar]
            mu = float(hit[f"mu_{col}"])
            r = disp.get(col, 8.0)                     # number or {"r_po_mu": [...]} (low-line backtest 29.10)
            p_ov = pk.p_count_over(mu, r, line)
            for side, p_side in (("nad", p_ov), ("pod", 1 - p_ov)):
                other = "pod" if side == "nad" else "nad"
                key = (hit["game_code"], pazar, hit["player_code"], side, line)

                def make(h=hit, s=side, p_s=p_side, m=mu):
                    t = teamp.get(h["game_code"], {})
                    margin = (t.get("mu_home", 0) - t.get("mu_away", 0)) * (1 if h["home"] else -1)
                    ctx = PlayerContext(league=league, position=h["pos"], starter=float(h["min_bez"]) >= 24,
                                        team_is_favorite=margin > 0, expected_margin=abs(margin),
                                        dvp_pct=h.get(f"dvp_{col}"), team_pace_vs_lg=t.get("pace_h" if h["home"] else "pace_a", 0),
                                        opp_pace_vs_lg=t.get("pace_a" if h["home"] else "pace_h", 0),
                                        team_drtg_vs_lg=t.get("drtg_h" if h["home"] else "drtg_a", 0),
                                        opp_drtg_vs_lg=t.get("drtg_a" if h["home"] else "drtg_h", 0),
                                        fouls_per36=None, bench_teammate_gain=0.0, returning_teammate=False,
                                        new_teammate_absence=bool(h["novo_otsustvo"]),
                                        first_scorer_of_depleted=bool(h["prv_strelec"]))
                    c = pk.Candidate(league, h["game_code"], f"{h['team']} vs {h['opp']}", h["team"], h["opp"],
                                     h["label"], h["player_code"], pazar, s, float(line), m, p_s,
                                     min_proj=float(h["min_bez"]), zavisi=bool(h["q_self"] or h["q_team"]))
                    c.scenarija = [x for x in player_scenarios(ctx) if x.side == s]
                    return c
                add(key, make, book, sides[side], sides[other], r0["nat_id"])

    out = list(cands.values())
    for c in out:
        blocked = any(s.code == "PRV_STRELEC" for s in c.scenarija) or (
            c.igrac is not None and bool(players.loc[players["player_code"] == c.player_code, "prv_strelec"].any()))
        pk.apply_filters(c, league, blocked)
    return out


def record(cands: list[pk.Candidate], ticket: list[pk.Candidate], day: date, teamp: dict) -> int:
    """Paper mode: candidates the model favours (p_kal ≥ 50%) are written — passing ones CEKA, rejected with
    EV > 0 as ODBIEN (with the rule). p_model is locked: an existing record is never updated."""
    con = db.connect()
    n = 0
    in_ticket = {id(c) for c in ticket}
    # same pick on a later run with a moved line = the same pick → keep the first (p_model locked, decision 29.10)
    existing = {(r["natprevar"], r["igrac"], r["pazar"], r["strana"]) for r in con.execute(
        "SELECT natprevar, igrac, pazar, strana FROM izbori WHERE datum=? AND verzija_model=?", [day.isoformat(), MODEL_VERSION])}
    for c in cands:
        if c.p_kal < 0.5 or c.kvota is None:
            continue
        if c.odbien and (c.ev or -1) <= 0:
            continue
        if (c.natprevar, c.igrac, c.pazar, c.strana) in existing:
            continue
        existing.add((c.natprevar, c.igrac, c.pazar, c.strana))
        status = "ODBIEN" if c.odbien else "CEKA"
        t = teamp.get(c.game_code, {})
        vlez = {"mu": round(c.mu, 2), "player_code": c.player_code, "min_proj": c.min_proj, "kladilnica": c.kladilnica,
                "kvoti": {b: list(v) for b, v in c.kvoti.items()}, "nat_ids": c.nat_ids,
                "book_pazar": {"mozzart": c.pazar, "evrotip": c.pazar}, "scenarija": [s.code for s in c.scenarija],
                "tiket": id(c) in in_ticket, "hartija": True}
        stake = pk.kelly_fraction(c.p_kal, c.kvota) * 100 if id(c) in in_ticket else None
        cur = con.execute(
            """INSERT OR IGNORE INTO izbori (datum, vreme, liga, natprevar, tim, protivnik, igrac, pazar, linija, strana,
               kvota, p_model, p_kalibriran, p_pazar, prednost_pp, ev, ulog_pct, pravila_primeneti, status, vlezovi_json,
               zabeleska, verzija_model, izvor, kvota_proverena, uplateno, game_code, odbien_od)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [day.isoformat(), t.get("time"), c.liga, c.natprevar, c.tim, c.protivnik, c.igrac, c.pazar, c.linija, c.strana,
             c.kvota, round(c.p_model, 4), round(c.p_kal, 4), None if c.p_pazar is None else round(c.p_pazar, 4),
             None if c.p_pazar is None else round((c.p_kal - c.p_pazar) * 100, 1), round(c.ev, 4), stake,
             "; ".join(c.pravila) or None, status, json.dumps(vlez, ensure_ascii=False),
             ("ТИКЕТ (хартија)" if id(c) in in_ticket else None), MODEL_VERSION, f"picks/{c.kladilnica}", 1, 0,
             str(c.game_code), db.rejected_by("; ".join(c.pravila)) if status == "ODBIEN" else None])
        n += cur.rowcount
    con.commit()
    db.export_csv(con)
    return n


BLOWOUT_WARN = 8.0                  # Tome 31.10: ⚠️ for players of a favourite by ≥ 8 (warning only, no model change)


def _fmt_c(c: pk.Candidate, fav8: frozenset = frozenset()) -> str:
    who = c.igrac or (c.tim if c.tim else c.natprevar)
    if c.igrac and (c.game_code, c.tim) in fav8:
        who += " ⚠️ blowout"
    other = ", ".join(f"{b} {v[0]:.2f}" for b, v in c.kvoti.items())
    why = "; ".join(s.text_mk for s in c.scenarija) or "—"
    pp = "—" if c.p_pazar is None else f"{c.p_pazar * 100:.0f}%"
    flag = (" · ❌ " + "; ".join(c.pravila)) if c.pravila else ""
    return (f"| {who} | {c.pazar} {c.strana.upper()} {c.linija:g} | {c.mu:.1f} | {c.p_kal * 100:.0f}% | {pp} | "
            f"{c.ev * 100:+.1f}% | {other} | {why}{flag} |")


def observed_gap(league: str) -> float:
    """Paper record of the Python model in this league: mean(hit) − mean(p_model) over decided picks (0 if none)."""
    from store.stats import decided
    if not db.DB_PATH.exists():
        return 0.0
    con = db.connect()
    d = decided(db.picks_frame(con))
    con.close()
    d = d[(d["liga"] == league) & d["verzija_model"].fillna("").str.startswith("py")]
    return float(d["hit"].mean() - d["p_model"].mean()) if len(d) else 0.0


def picks_report(settings: Settings, league: str, day: date, write: bool = True) -> Optional[str]:
    cfg = settings.league(league)
    try:
        fetch_league(settings, league, day)
    except SourceError as exc:
        log.error("%s: податоците не се освежени — %s", league, exc)
    folder = settings.paths.processed / league / cfg.season
    read = lambda n: pd.read_csv(folder / n, dtype={"game_code": str, "player_code": str})
    schedule, logs, roster = read("schedule.csv"), read("player_logs.csv"), read("roster.csv")
    games = schedule[schedule["game_date"] == day.isoformat()]
    if games.empty:
        return None
    lines = build_day(settings.paths.raw, datetime.now(ZoneInfo(settings.timezone)).date().isoformat())
    if lines.empty:
        return f"# {cfg.name_mk} {day}\n\nНема снимки на линии — прво `python main.py linii`."
    lines = attach_team_codes(lines, settings)
    absent, notes = league_absences(settings, league, day)
    teamp = team_projections(settings, league, day, games)
    players = player_projections(settings, league, day, games, absent, logs, roster)
    cands = build_candidates(settings, league, day, games, teamp, players, lines)
    ticket = pk.build_ticket(cands)
    n_written = record(cands, ticket, day, teamp) if write else 0

    head = "| Избор | Пазар | Проекција | P (кал.) | P пазар | EV | Квоти | Причина / сценарио |\n|---|---|---|---|---|---|---|---|"
    out = [f"# Избори — {cfg.name_mk} · {day}", "",
           "> 📝 **РЕЖИМ НА ХАРТИЈА** — ништо не се уплаќа; сè е запишано (CEKA, uplateno = не). "
           "Не е доказ за предност пред 100 одлучени избори И 10 вечери на хартија (PROCES §4А).", "",
           *[f"- {n}" for n in [x for x in notes if not x.startswith("__")]],
           f"- Кандидати: {len(cands)} · запишани во архивата: {n_written} · модел {MODEL_VERSION}", ""]
    s_team, s_total, s_margin = team_sigmas(settings, league)
    fav8 = frozenset((gc, t["home"] if t["mu_home"] > t["mu_away"] else t["away"]) for gc, t in teamp.items()
                     if abs(t["mu_home"] - t["mu_away"]) >= BLOWOUT_WARN)
    fmt = lambda c: _fmt_c(c, fav8)
    for gc, t in teamp.items():
        tot = t["mu_home"] + t["mu_away"]
        mar = t["mu_home"] - t["mu_away"]
        sc = team_scenarios(league, t["pace_h"], t["pace_a"], t["drtg_h"], t["drtg_a"], abs(mar))
        out += [f"## {t['home']} – {t['away']} ({t['time']})", "",
                f"**Сценарио:** проекција {t['mu_home']:.1f} – {t['mu_away']:.1f} (тотал {tot:.1f} ± {s_total:.0f}, "
                f"разлика {mar:+.1f} ± {s_margin:.0f}). " + ("; ".join(s.text_mk for s in sc) or "без посебно сценарио."), ""]
        mine = [c for c in cands if c.game_code == gc and c.p_kal >= 0.5 and c.ev is not None]
        for side, title in (("nad", "Најдобри НАД"), ("pod", "Најдобри ПОД")):
            sel = [c for c in mine if c.strana == side and c.pazar != "hendikep"]
            sel = sorted(sel, key=lambda c: (not c.odbien, c.ev), reverse=True)[:5]
            out += [f"**{title}:**", ""]
            out += [head, *[fmt(c) for c in sel]] if sel else ["нема кандидат со линија."]
            out.append("")
        hc = [c for c in mine if c.pazar == "hendikep"]
        if hc:
            out += ["**Хендикеп:**", "", head, *[fmt(c) for c in sorted(hc, key=lambda c: c.ev, reverse=True)[:2]], ""]
    out += team_markets_section(league, teamp, cands, _latest(lines[lines["liga"] == league]), absent, TEAMS_DIR, day)
    out += ["## 🎟️ Тикет (хартија)", ""]
    if ticket:
        k = 1.0
        for c in ticket:
            k *= c.kvota
        p = float(np.prod([c.p_kal for c in ticket]))
        out += [head, *[fmt(c) for c in ticket], "",
                f"Квота на тикетот {k:.2f} · шанса (модел) {p * 100:.1f}% · праг {100 / k:.1f}% · "
                f"¼ Kelly по избор: " + ", ".join(f"{pk.kelly_fraction(c.p_kal, c.kvota) * 100:.1f}%" for c in ticket)
                + " од банката (само за евиденција — хартија)."]
    else:
        out.append("Нема 2–3 избори што ги поминуваат сите правила → **без тикет вечерва**.")
    gap = observed_gap(league)
    out += ["", f"## 🎟️ Предлог тикети (мал / среден / голем · еден играч најмногу на {pk.MAX_TICKETS_PER_PLAYER} тикети · "
            "1 избор по натпревар)", "",
            f"*Шанса = производ од P (кал.) на изборите (различни натпревари → независни). Корекција: набљудувана разлика "
            f"погодок − очекувано на хартија во оваа лига = {gap * 100:+.1f} pp по избор (`stats`).*", ""]
    for name, tk, why in pk.propose_tickets(cands):
        if not tk:
            out += [f"**{name}:** {why}", ""]
            continue
        k = float(np.prod([c.kvota for c in tk]))
        p = float(np.prod([c.p_kal for c in tk]))
        p_adj = float(np.prod([min(0.99, max(0.01, c.p_kal + gap)) for c in tk]))
        out += [f"**{name} ({len(tk)}):**", "", head, *[fmt(c) for c in tk], "",
                f"Квота {k:.2f} · шанса (модел) {p * 100:.1f}% · со корекција {p_adj * 100:.1f}% · праг {100 / k:.1f}% "
                f"· EV тикет (корекција) {(p_adj * k - 1) * 100:+.1f}%", ""]
    rej: dict[str, int] = {}
    for c in cands:
        for r in c.pravila:
            key = r.split(" (")[0]
            rej[key] = rej.get(key, 0) + 1
    out += ["", "## Одбиени по правило (број на кандидати)", ""] + [f"- {k}: {v}" for k, v in sorted(rej.items(), key=lambda kv: -kv[1])]
    out += ["", "*Линиите (Mozzart, evrotip) влегоа дури по проекцијата (anti-anchoring). Калибрација: ЕЛ стегање k=0.5 (СЛЕДИ); "
            "isotonic по страна чека нови резултати (неделно). "
            f"P пазар = de-vig метод **{DEVIG_METHOD}** (избран со `backtest --devig`; EV не зависи од методот). "
            "Распределба: NB, скокови/асистенции со посебен r за мал λ (`backtest --niski`). "
            "EV > +15% или разлика > 15 pp → ПРОВЕРИ_ПОДАТОЦИ (не во тикет).*"]
    return "\n".join(out)


def cmd_picks(settings: Settings, league: str, day: date) -> int:
    leagues = ["euroleague", "eurocup", "nba"] if league == "all" else [league]
    settings.paths.reports.mkdir(parents=True, exist_ok=True)
    for lg in leagues:
        text = picks_report(settings, lg, day)
        if text is None:
            print(f"{settings.league(lg).name_mk}: нема натпревари на {day}.")
            continue
        path = settings.paths.reports / f"picks_{day}_{lg}.md"
        path.write_text(text, encoding="utf-8")
        print(text)
        print(f"Извештај: {path}\n")
    return 0
