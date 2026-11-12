"""`picks` report section ТИМСКИ ПАЗАРИ (Tome 30.10): per game — expected score, both teams' numbers (profiles),
the main team lines of both books, our projection vs each line, and how much each market is backed by a backtest."""
from __future__ import annotations

from pathlib import Path

from datetime import date

import pandas as pd
import yaml

from model import picks as pk
from model.team_dossier import read_news

MARKET_MK = {"1x2": "Победник (регуларно)", "hendikep": "Хендикеп", "vkupno": "Тотал", "tim_vkupno": "Тимски тотал"}


def trust(league: str, pazar: str) -> str:
    """How much the market is backed by our backtests (PLAN: Phase 4 = NBA totals/handicaps vs closing lines)."""
    if league != "nba":
        return "непроверено (нема историски линии)"
    if pazar in ("hendikep", "vkupno"):
        return "backtest: без предност против пазарот"
    if pazar == "1x2":
        return "backtest: без предност (изведено од хендикеп); X = просек на лигата"
    return "непроверено (тимски тотал без историски линии)"


def _profile(teams_dir: Path, league: str, team: str) -> dict:
    p = teams_dir / league / team / "profil.yaml"
    return ((yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("brojki") or {}) if p.exists() else {}


def profile_use(teams_dir: Path, league: str, team: str, today: date) -> list[str]:
    """What the decision may use from the dossier (BROJKA / POTVRDEN_PARAMETAR) vs explanation only (OPIS),
    plus the team's news of the last 7 days (vesti.md)."""
    folder = teams_dir / league / team
    y = folder / "profil.yaml"
    par = ((yaml.safe_load(y.read_text(encoding="utf-8")) or {}).get("parametri") or {}) if y.exists() else {}
    used = [k for k, v in par.items() if v.get("status") in ("BROJKA", "POTVRDEN_PARAMETAR") and v.get("vrednost") is not None]
    opis = [k for k, v in par.items() if v.get("status") == "OPIS" and v.get("vrednost") is not None]
    out = [f"  - профил: BROJKA/POTVRDEN {len(used)} полиња (моделот ги пресметува од истиот box score) · "
           f"OPIS {len(opis)} одобрени (само објаснување{': ' + ', '.join(opis) if opis else ''})"]
    stil = (par.get("stil") or {}).get("vrednost") or {}
    if stil:
        from model.team_dossier import style_line
        out.append(f"  - стил (од бројки): {style_line(stil)}")
    news = read_news(folder, 7, today)
    out.append(f"  - вести (7 дена): {len(news)}" + ("" if news else " — нема одобрени вести"))
    out += [f"    - {n['datum']} · {n['tip']} · {n['igrac']} · {n['sto'][:110]}" for n in news[:4]]
    return out


def identity(b: dict, n_teams: int) -> str:
    """ѕид = the defence ranks clearly better than the attack; шутаут = the opposite; else среден."""
    e = b.get("efikasnost") or {}
    if not e.get("rang_napad") or not e.get("rang_odbrana"):
        return "—"
    d = e["rang_napad"] - e["rang_odbrana"]
    thr = max(2, n_teams // 4)
    return "ѕид (одбрана)" if d >= thr else "шутаут (напад)" if d <= -thr else "среден"


def team_line(team: str, b: dict, n_teams: int, absent: pd.DataFrame) -> str:
    if not b:
        return f"**{team}** — нема профил (`python main.py profil`)"
    e, t, f = b.get("efikasnost") or {}, b.get("tempo") or {}, b.get("forma_posledni_5") or {}
    dm, gs = b.get("doma") or {}, b.get("gosti") or {}
    parts = [f"ORtg {e.get('ortg_prilagoden')} ({e.get('rang_napad')}.)", f"DRtg {e.get('drtg_prilagoden')} ({e.get('rang_odbrana')}.)",
             f"темпо {t.get('poseda_po_natprevar')} ({t.get('rang')}.)",
             f"форма {f.get('pobedi')}–{f.get('porazi')} ({f.get('poeni')}–{f.get('primeni')})",
             f"дома {dm.get('poeni')}–{dm.get('primeni')} ({dm.get('n')}) / гости {gs.get('poeni')}–{gs.get('primeni')} ({gs.get('n')})",
             f"идентитет: {identity(b, n_teams)}"]
    out = f"**{team}** ({b.get('natprevari')} нат., до {b.get('do_datum')}) — " + " · ".join(parts)
    if len(absent):
        a = absent[(absent["team"] == team) & (absent["status"].astype(str).str.upper() != "AVAILABLE")]
        a = a.sort_values("minuti_sezona", ascending=False, na_position="last").head(5)
        if len(a):
            out += " · отсуства: " + ", ".join(
                f"{r.label} ({r.status}, {r.tip}, {0 if pd.isna(r.minuti_sezona) else r.minuti_sezona:.0f} мин)" for r in a.itertuples())
    return out


def _main_lines(L: pd.DataFrame, home: str, away: str) -> dict:
    """{(market, team, line)} that a book lists as its main line, in candidate terms."""
    g = L[(L["tim_d"] == home) & (L["tim_g"] == away) & L["igrac"].isna() & (L["glavna"] == True)]  # noqa: E712
    keys = set()
    for r in g.itertuples():
        if r.pazar == "vkupno":
            keys.add(("vkupno", None, float(r.linija)))
        elif r.pazar == "tim_vkupno_d":
            keys.add(("tim_vkupno", home, float(r.linija)))
        elif r.pazar == "tim_vkupno_g":
            keys.add(("tim_vkupno", away, float(r.linija)))
        elif r.pazar == "hendikep" and r.kladilnica == "mozzart":
            keys |= {("hendikep", home, float(r.linija)), ("hendikep", away, -float(r.linija))}
        elif r.pazar == "1x2":
            keys.add(("1x2", None, -1.0))
    return keys


def _row(c: pk.Candidate, league: str) -> str:
    side = {"nad": "НАД", "pod": "ПОД"}.get(c.strana, c.strana)
    what = (f"{side} {c.linija:g}" if c.pazar in ("vkupno", "tim_vkupno") else
            f"{c.strana} {c.linija:+g}" if c.pazar == "hendikep" else side)
    if c.pazar == "tim_vkupno":
        what = f"{c.tim} {what}"
    k = lambda b: f"{c.kvoti[b][0]:.2f}" if b in c.kvoti else "—"
    proj = f"{c.mu:+.1f}" if c.pazar in ("hendikep", "1x2") else f"{c.mu:.1f}"
    flags = [r for r in c.pravila if not r.startswith("EV_ПОД")]
    return (f"| {MARKET_MK[c.pazar]} | {what} | {k('mozzart')} | {k('evrotip')} | {proj} | {c.p_kal * 100:.0f}% | "
            f"{'—' if c.p_pazar is None else f'{c.p_pazar * 100:.0f}%'} | {c.ev * 100:+.1f}% | {c.kladilnica} | "
            f"{trust(league, c.pazar)}{' · ' + '; '.join(flags) if flags else ''} |")


def team_markets_section(league: str, teamp: dict, cands: list[pk.Candidate], L: pd.DataFrame, absent: pd.DataFrame,
                         teams_dir: Path, today: date | None = None) -> list[str]:
    n_teams = len([p for p in (teams_dir / league).glob("*") if p.is_dir()]) or 20
    team_c = [c for c in cands if c.igrac is None and c.pazar in MARKET_MK and c.ev is not None]
    out = ["## 🏀 ТИМСКИ ПАЗАРИ", "",
           "*Проекција = тимскиот модел (темпо × ефикасност). Победник = регуларно време (X = стапка на продолженија во "
           "лигата). Хендикеп само Mozzart (знакот на evrotip уште не е потврден). Ознаката на доверба кажува колку "
           "пазарот е проверен со backtest — EV без проверка не е доказ.*", ""]
    best_all = []
    for gc, t in teamp.items():
        tot, mar = t["mu_home"] + t["mu_away"], t["mu_home"] - t["mu_away"]
        out += [f"### {t['home']} – {t['away']} ({t['time']})", "",
                f"**Очекуван резултат:** {t['mu_home']:.1f} – {t['mu_away']:.1f} · тотал {tot:.1f} · разлика {mar:+.1f} · "
                f"темпо {t.get('poss', float('nan')):.1f} поседи", ""]
        for team in (t["home"], t["away"]):
            out.append("- " + team_line(team, _profile(teams_dir, league, team), n_teams, absent))
            out += profile_use(teams_dir, league, team, today or date.today())
        main = _main_lines(L, t["home"], t["away"])
        mine = [c for c in team_c if c.game_code == gc and (c.pazar, c.tim if c.pazar != "1x2" else None, c.linija) in main]
        mine.sort(key=lambda c: (list(MARKET_MK).index(c.pazar), c.tim or "", c.linija, c.strana))
        out.append("")
        if mine:
            out += ["| Пазар | Избор | Mozzart | evrotip | Проекција | P (кал.) | P пазар | EV | Подобра | Доверба / ознаки |",
                    "|---|---|---|---|---|---|---|---|---|---|", *[_row(c, league) for c in mine]]
        else:
            out.append("Нема главни тимски линии кај кладилниците.")
        ok = [c for c in team_c if c.game_code == gc and not c.odbien and not c.zavisi and c.ev >= pk.MIN_EV]
        if ok:
            b = max(ok, key=lambda c: c.ev)
            best_all.append(b)
            out += ["", f"**Најдобар тимски избор:** {_short(b)} @ {b.kladilnica} {b.kvota:.2f} · P {b.p_kal * 100:.0f}% · "
                        f"EV {b.ev * 100:+.1f}% · {trust(league, b.pazar)}"
                        + (" · САМО ХАРТИЈА" if b.samo_hartija else ""), ""]
        else:
            out += ["", "**Најдобар тимски избор:** нема вредност (сите под EV 4% или одбиени со правило).", ""]
    if best_all:
        out += ["### ТОП 5 тимски избори (по EV)", "",
                "| # | Натпревар | Избор | Квота | P (кал.) | P пазар | EV | Доверба |", "|---|---|---|---|---|---|---|---|"]
        for i, c in enumerate(sorted(best_all, key=lambda c: c.ev, reverse=True)[:5], 1):
            out.append(f"| {i} | {c.natprevar} | {_short(c)} | {c.kladilnica} {c.kvota:.2f} | {c.p_kal * 100:.0f}% | "
                       f"{'—' if c.p_pazar is None else f'{c.p_pazar * 100:.0f}%'} | {c.ev * 100:+.1f}% | "
                       f"{trust(league, c.pazar)}{' · САМО ХАРТИЈА' if c.samo_hartija else ''} |")
        out.append("")
    return out


def _short(c: pk.Candidate) -> str:
    side = {"nad": "НАД", "pod": "ПОД"}.get(c.strana, c.strana)
    if c.pazar == "vkupno":
        return f"тотал {side} {c.linija:g}"
    if c.pazar == "tim_vkupno":
        return f"{c.tim} тимски тотал {side} {c.linija:g}"
    if c.pazar == "hendikep":
        return f"{c.strana} {c.linija:+g}"
    return f"победник {c.strana}" + (f" ({c.tim})" if c.tim else "")
