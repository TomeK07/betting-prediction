"""`python main.py pregled` (Tome 2.11) — an OVERVIEW of every game with every option Mozzart offers, sorted into three
levels by the MARKET's own (de-vigged) probability. Tome decides. This is NOT a recommendation and changes no ticket
rule: nothing is written to the archive (player picks stay САМО_ХАРТИЈА there).

Per game: expectation (model vs market implied from handicap + total vs experts), then every option incl. alternative
(bought) lines and the main players' props, each with: market % (de-vig of the two prices of THAT line; Mozzart first,
evrotip when Mozzart has no pair), Mozzart odds (+ evrotip when better), model ✅/➖/❌ and experts as marks only,
risks (⚠️ lineup / blowout / low minutes). Levels: 🟢 ≥75% · 🟡 65–75% · 🔴 55–65% · below 55% not shown.
At the end: example tickets (different games only, chance = product of the market %).
"""
from __future__ import annotations

import math
from datetime import date, datetime
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

from core.config import Settings
from model import picks as pk
from model.lines import DEVIG_METHOD, build_day, devig_two_way
from pipeline.injuries_cmd import league_absences
from pipeline.lines_cmd import attach_team_codes
from pipeline.picks_cmd import (BLOWOUT_WARN, build_candidates, player_projections, team_projections)

LEVELS = [("🟢 СИГУРНИ", 0.75), ("🟡 СРЕДНИ", 0.65), ("🔴 СЛАБИ", 0.55)]
MIN_P = 0.55
PLAYERS_PER_TEAM = 5
LOW_MIN = 20.0
AGREE_PP = 0.05                       # model within 5 pp below the market = ➖ (neither agrees nor contradicts)
MK = {"vkupno": "тотал", "tim_vkupno": "тимски тотал", "hendikep": "хендикеп", "1x2": "победник (рег.)",
      "poeni": "поени", "skokovi": "скокови", "asistencii": "асистенции", "trojki": "тројки"}
SIDE = {"nad": "НАД", "pod": "ПОД"}


def market_p(c: pk.Candidate) -> tuple[Optional[float], Optional[str]]:
    """De-vig of THIS line's two prices: Mozzart first, evrotip when Mozzart has no pair."""
    for book in ("mozzart", "evrotip"):
        k = c.kvoti.get(book)
        if k and k[1]:
            return devig_two_way(k[0], k[1]), book
    return None, None


def fair_1x2(L: pd.DataFrame, home: str, away: str) -> dict:
    """Mozzart three-way (regulation time), proportional de-vig. {'1': p, 'X': p, '2': p} or {}."""
    g = L[(L["kladilnica"] == "mozzart") & (L["pazar"] == "1x2") & (L["tim_d"] == home) & (L["tim_g"] == away)]
    k = g.set_index("strana")["kvota"].to_dict()
    if not {"1", "X", "2"} <= set(k):
        return {}
    inv = {s: 1 / k[s] for s in ("1", "X", "2")}
    return {s: v / sum(inv.values()) for s, v in inv.items()}


def market_expectation(L: pd.DataFrame, home: str, away: str) -> Optional[dict]:
    """Mozzart MAIN lines: total T and home handicap H (−11.5 = home by 11.5) → home (T−H)/2, away (T+H)/2."""
    g = L[(L["kladilnica"] == "mozzart") & (L["tim_d"] == home) & (L["tim_g"] == away) & L["igrac"].isna()
          & L["glavna"].astype(bool)]
    line = lambda p: g[g["pazar"] == p]["linija"].dropna()
    t, h = line("vkupno"), line("hendikep")
    if t.empty or h.empty:
        return None
    T, H = float(t.iat[0]), float(h.iat[0])
    td, tg = line("tim_vkupno_d"), line("tim_vkupno_g")
    return {"T": T, "H": H, "home": (T - H) / 2, "away": (T + H) / 2,
            "tt_home": float(td.iat[0]) if len(td) else None, "tt_away": float(tg.iat[0]) if len(tg) else None}


def describe(c: pk.Candidate) -> str:
    side = SIDE.get(c.strana, c.strana)
    if c.igrac:
        return f"{c.igrac} — {MK.get(c.pazar, c.pazar)} {side} {c.linija:g}"
    if c.pazar == "hendikep":
        return f"хендикеп {c.tim} {c.linija:+g}"
    if c.pazar == "1x2":
        return f"победник (рег. време) {c.tim or 'X'}"
    who = f"{c.tim} " if c.tim else ""
    return f"{MK.get(c.pazar, c.pazar)} {who}{side} {c.linija:g}"


def options(cands: list[pk.Candidate], fair: dict, players: pd.DataFrame, fav8: frozenset) -> list[dict]:
    keep, q_self = set(), set()
    if len(players):
        q_self = set(players[players["q_self"]]["player_code"].astype(str))
        top = players.sort_values("min_bez", ascending=False).groupby("team").head(PLAYERS_PER_TEAM)
        keep = set(top["player_code"].astype(str))
    out = []
    for c in cands:
        if c.player_code and str(c.player_code) not in keep:
            continue
        if c.pazar == "1x2":
            p, book = fair.get(c.strana), "mozzart"
        else:
            p, book = market_p(c)
        if p is None or p < MIN_P:
            continue
        risks = []
        own_q = bool(c.player_code) and str(c.player_code) in q_self
        if own_q:
            risks.append("⚠️ состав (самиот е Q → ЗАВИСИ_ОД_СОСТАВ)")
        elif c.zavisi:
            risks.append("⚠️ соиграч Q (минутите зависат од составот)")
        if c.player_code and (c.game_code, c.tim) in fav8 and c.strana == "nad":
            risks.append(f"⚠️ blowout (тимот фаворит ≥{BLOWOUT_WARN:g})")
        if c.player_code and c.min_proj is not None and c.min_proj < LOW_MIN:
            risks.append(f"⚠️ мали минути ({c.min_proj:.0f})")
        level = next(name for name, lo in LEVELS if p >= lo)
        if own_q:
            level = LEVELS[-1][0]                     # Tome: the player himself ЗАВИСИ_ОД_СОСТАВ → at most 🔴
        d = p - c.p_kal
        mz = c.kvoti.get("mozzart", (None,))[0]
        ev = c.kvoti.get("evrotip", (None,))[0]
        out.append({"c": c, "p": p, "own_q": own_q, "book_p": book, "level": level,
                    "model": "✅" if c.p_kal >= p else ("➖" if d <= AGREE_PP else "❌"),
                    "kvota": mz, "evrotip": ev if ev and (mz is None or ev > mz) else None, "risks": risks})
    return sorted(out, key=lambda o: -o["p"])


def _row(o: dict) -> str:
    c = o["c"]
    odds = f"{o['kvota']:.2f}" if o["kvota"] else "— (нема на Mozzart)"
    if o["evrotip"]:
        odds += f" · подобро на evrotip {o['evrotip']:.2f}"
    src = "" if o["book_p"] == "mozzart" else " (evrotip)"
    return (f"| {describe(c)} | **{o['p'] * 100:.0f}%**{src} | {odds} | {o['model']} | нема податок | "
            f"{' '.join(o['risks']) or '—'} |")


def game_block(t: dict, opts: list[dict], mkt: Optional[dict]) -> list[str]:
    h, a = t["home"], t["away"]
    tot, mar = t["mu_home"] + t["mu_away"], t["mu_home"] - t["mu_away"]
    out = [f"## {h} – {a}", "", "**Очекување**", "",
           f"- Модел: {h} {t['mu_home']:.1f} – {a} {t['mu_away']:.1f} · тотал {tot:.1f} · разлика {mar:+.1f}"]
    if mkt:
        tt = (f" · тимски тотали {h} {mkt['tt_home']:g} / {a} {mkt['tt_away']:g}"
              if mkt["tt_home"] is not None and mkt["tt_away"] is not None else "")
        out.append(f"- Пазар (Mozzart, од хендикеп {mkt['H']:+g} и тотал {mkt['T']:g}): {h} {mkt['home']:.1f} – "
                   f"{a} {mkt['away']:.1f}{tt}")
    else:
        out.append("- Пазар: нема главна линија за хендикеп/тотал на Mozzart")
    out += ["- Експерти: нема податок (T2 уште не е изграден; NITRO вестите немаат став за тотал/хендикеп)", ""]
    for name, _ in LEVELS:
        rows = [o for o in opts if o["level"] == name]
        out += [f"**{name}** ({len(rows)})", ""]
        if rows:
            out += ["| Опција | Пазар % | Квота Mozzart | Модел | Експерти | Ризик |", "|---|---|---|---|---|---|",
                    *[_row(o) for o in rows]]
        else:
            out.append("- нема")
        out.append("")
    return out


def example_tickets(by_game: dict[str, list[dict]]) -> list[str]:
    """One option per game (different games → independent); chance = product of the market %; never ⚠️ lineup."""
    def best(levels: set, exclude=()) -> list[dict]:
        picks = []
        for gc, opts in by_game.items():
            ok = [o for o in opts if o["level"] in levels and not o["own_q"] and o["kvota"] and gc not in exclude]
            if ok:
                picks.append(max(ok, key=lambda o: o["p"]))
        return sorted(picks, key=lambda o: -o["p"])

    def fmt(name: str, sel: list[dict], want: str) -> list[str]:
        if not sel:
            return [f"**{name}:** не може — нема доволно опции од различни натпревари ({want}).", ""]
        p = math.prod(o["p"] for o in sel)
        k = math.prod(o["kvota"] for o in sel)
        lines = [f"**{name}** — шанса на тикетот **{p * 100:.0f}%** · квота {k:.2f} (Mozzart) · "
                 f"фер квота за таа шанса {1 / p:.2f}", ""]
        lines += [f"- {o['c'].natprevar}: {describe(o['c'])} · {o['p'] * 100:.0f}% · {o['kvota']:.2f}" for o in sel]
        return lines + [""]

    green = best({LEVELS[0][0]})
    mixed = best({LEVELS[0][0], LEVELS[1][0]})
    out = ["## 🎟️ Пример тикети (од прегледот — НЕ е препорака)", "",
           "*Еден избор по натпревар (различни натпревари → шансата = производ). Квота пониска од фер квотата = "
           "кладилницата ја зема маржата; тоа е нормално и значи дека тикетот НЕ е во плус.*", ""]
    out += fmt("3 сигурни 🟢", green[:3] if len(green) >= 3 else [], "треба 3 натпревари со 🟢")
    n = 5 if len(mixed) >= 5 else 4
    out += fmt(f"{n} мешано 🟢+🟡", mixed[:n] if len(mixed) >= 4 else [], "треба 4–5 натпревари со 🟢/🟡")
    return out


def pregled_report(settings: Settings, league: str, day: date, only: Optional[str] = None) -> Optional[str]:
    cfg = settings.league(league)
    folder = settings.paths.processed / league / cfg.season
    read = lambda n: pd.read_csv(folder / n, dtype={"game_code": str, "player_code": str})
    schedule, logs, roster = read("schedule.csv"), read("player_logs.csv"), read("roster.csv")
    games = schedule[schedule["game_date"] == day.isoformat()]
    if only:
        h, a = only.upper().split("-")
        games = games[(games["home"] == h) & (games["away"] == a)]
    if games.empty:
        return None
    lines = build_day(settings.paths.raw, datetime.now(ZoneInfo(settings.timezone)).date().isoformat())
    if lines.empty:
        return f"# Преглед — {cfg.name_mk} · {day}\n\nНема снимки на линии — прво `python main.py linii`."
    lines = attach_team_codes(lines, settings)
    absent, _ = league_absences(settings, league, day)
    teamp = team_projections(settings, league, day, games)
    players = player_projections(settings, league, day, games, absent, logs, roster)
    every = lines.assign(glavna=True)                 # alternative (bought) lines too, not just the main one
    cands = build_candidates(settings, league, day, games, teamp, players, every)
    last = lines.groupby("kladilnica")["snimka"].transform("max")
    L = lines[lines["snimka"] == last]
    fav8 = frozenset((gc, t["home"] if t["mu_home"] > t["mu_away"] else t["away"]) for gc, t in teamp.items()
                     if abs(t["mu_home"] - t["mu_away"]) >= BLOWOUT_WARN)
    snap = ", ".join(f"{b} {s[11:16]}" for b, s in L.groupby("kladilnica")["snimka"].max().items())
    out = [f"# Преглед — {cfg.name_mk} · {day}", "",
           "> 📋 **ПРЕГЛЕД, НЕ ПРЕПОРАКА.** Процентот = пазарот на таа линија (de-vig " + DEVIG_METHOD + "), не моделот. "
           "Моделот (✅ се согласува / ➖ блиску / ❌ не) и експертите се само ознаки. Backtest: моделот нема предност "
           "пред пазарот → ниту една опција тука не е „во плус“. Ништо не е уплатено ни запишано во архивата.", "",
           f"- Снимка на линии: {snap} · играчи: главните {PLAYERS_PER_TEAM} по тим (по проектирани минути) · "
           f"опции под {MIN_P * 100:.0f}% не се прикажани.", ""]
    by_game: dict[str, list[dict]] = {}
    for gc, t in teamp.items():
        opts = options([c for c in cands if c.game_code == gc], fair_1x2(L, t["home"], t["away"]),
                       players[players["game_code"] == gc] if len(players) else players, fav8)
        by_game[gc] = opts
        out += game_block(t, opts, market_expectation(L, t["home"], t["away"]))
    out += example_tickets(by_game)
    return "\n".join(out) + "\n"


def cmd_pregled(settings: Settings, league: str, day: date, only: Optional[str] = None) -> int:
    leagues = ["euroleague", "eurocup", "nba"] if league == "all" else [league]
    for lg in leagues:
        text = pregled_report(settings, lg, day, only)
        if text is None:
            print(f"{settings.league(lg).name_mk}: нема натпревари на {day}.")
            continue
        suffix = f"_{only.upper()}" if only else ("" if lg == "nba" else f"_{lg}")
        path = settings.paths.reports / f"pregled_{day}{suffix}.md"
        path.write_text(text, encoding="utf-8")
        print(f"Преглед: {path}")
    return 0
