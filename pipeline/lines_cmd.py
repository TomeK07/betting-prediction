"""`python main.py linii --izvestaj` — normalized line table of a day + margins, movement, Mozzart vs evrotip."""
from __future__ import annotations

from datetime import date

import pandas as pd

from core.config import Settings
from model.lines import build_day, margins, movement
from pipeline.grade_cmd import load_league_data
from store.grade import resolve_team


def attach_team_codes(lines: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Bookmaker team names -> our official codes (config/timovi.yaml + name matching, never guessed)."""
    df = lines.copy()
    df["tim_d"], df["tim_g"] = None, None
    for league in df["liga"].dropna().unique():
        if league not in settings.leagues:
            continue
        data = load_league_data(settings, league)
        m = df["liga"] == league
        cache: dict = {}
        res = lambda n: cache.setdefault(n, resolve_team(n, data)) if isinstance(n, str) else None
        df.loc[m, "tim_d"] = df.loc[m, "domakjin"].map(res)
        df.loc[m, "tim_g"] = df.loc[m, "gostin"].map(res)
    return df


def lines_report(settings: Settings, day: date) -> str:
    L = build_day(settings.paths.raw, day.isoformat())
    if L.empty:
        return f"Нема снимки на линии за {day}."
    L = attach_team_codes(L, settings)
    out = settings.paths.processed / "linii"
    out.mkdir(parents=True, exist_ok=True)
    L.to_csv(out / f"{day}.csv", index=False, encoding="utf-8")
    snaps = L.groupby("kladilnica")["snimka"].nunique().to_dict()
    lines = [f"# Линии — {day}", "", f"Снимки: " + ", ".join(f"{k} {v}" for k, v in snaps.items())
             + f" · цени: {len(L)} · табела: data/processed/linii/{day}.csv", "",
             "## Маржа по кладилница × пазар (главни линии, медијана)", "",
             "| Пазар | Mozzart | evrotip |", "|---|---|---|"]
    mg = margins(L)
    tab = mg[mg["glavna"]].groupby(["pazar", "kladilnica"])["marza"].median().unstack()
    for pazar, r in tab.iterrows():
        cell = lambda k: "—" if k not in r or pd.isna(r[k]) else f"{r[k] * 100:.1f}%"
        lines.append(f"| {pazar} | {cell('mozzart')} | {cell('evrotip')} |")
    # main total: compare both books per match
    main = L[(L["pazar"] == "vkupno") & L["glavna"] & L["tim_d"].notna()].sort_values("snimka")
    last = main.groupby(["kladilnica", "tim_d", "tim_g", "strana"]).tail(1)
    piv = last.pivot_table(index=["liga", "tim_d", "tim_g"], columns=["kladilnica", "strana"], values=["linija", "kvota"],
                           aggfunc="first")
    if len(piv):
        lines += ["", "## Главен тотал: Mozzart наспроти evrotip (последна снимка)", "",
                  "| Натпревар | Mozzart линија (НАД/ПОД) | evrotip линија (НАД/ПОД) | подобро НАД | подобро ПОД |",
                  "|---|---|---|---|---|"]
        for (lg, h, a), r in piv.iterrows():
            def g(field, book, side):
                try:
                    v = r[(field, book, side)]
                    return None if pd.isna(v) else float(v)
                except KeyError:
                    return None
            mz = (g("linija", "mozzart", "nad"), g("kvota", "mozzart", "nad"), g("kvota", "mozzart", "pod"))
            et = (g("linija", "evrotip", "nad"), g("kvota", "evrotip", "nad"), g("kvota", "evrotip", "pod"))
            fmt = lambda t: "—" if t[0] is None else f"{t[0]:g} ({t[1]:.2f}/{t[2]:.2f})"
            def better(side):
                cands = [(b, t) for b, t in (("Mozzart", mz), ("evrotip", et)) if t[0] is not None]
                if len(cands) < 2:
                    return "—"
                (b1, t1), (b2, t2) = cands
                if t1[0] != t2[0]:
                    low = min(cands, key=lambda c: c[1][0]) if side == "nad" else max(cands, key=lambda c: c[1][0])
                    return f"{low[0]} (подобра линија)"
                i = 1 if side == "nad" else 2
                return "исто" if t1[i] == t2[i] else (b1 if t1[i] > t2[i] else b2)
            lines.append(f"| {h}–{a} | {fmt(mz)} | {fmt(et)} | {better('nad')} | {better('pod')} |")
    mv = movement(L)
    moved = mv[(mv["snimki"] > 1) & (mv["pomestuvanje"].fillna(0) != 0)]
    lines += ["", f"## Движење на линиите денес: {len(moved)} главни линии се поместиле "
                  f"(од {int((mv['snimki'] > 1).sum())} со повеќе снимки)."]
    for r in moved.sort_values("pomestuvanje", key=abs, ascending=False).head(10).itertuples():
        who = r.igrac if isinstance(r.igrac, str) else r.nat_id
        lines.append(f"- {r.kladilnica} · {who} · {r.pazar} {r.strana}: {r.linija_prva:g} → {r.linija_posledna:g} "
                     f"({r.prva[11:16]} → {r.posledna[11:16]})")
    lines += ["", "*Само кеш и споредба — моделот не ги користи линиите при пресметка (anti-anchoring); "
                  "влегуваат дури на крајот од `picks` (Фаза 7).*"]
    return "\n".join(lines)
