"""`python main.py backtest --devig` — which de-vig method to use (Tome 29.10).

Test without results: the two bookmakers price the same match / market / line with DIFFERENT margins
(evrotip lower on team markets). If a method removes the margin correctly, the two fair probabilities agree.
Measured on all saved snapshots (last snapshot of each day, both books), split by how unequal the prices are —
the favourite-longshot bias of the proportional method shows only on unequal pairs.
When graded paper picks accumulate, the same comparison is repeated against results (log loss) in `stats`.
"""
from __future__ import annotations

import glob
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats as st

from core.config import Settings
from model.lines import DEVIG_METHOD, DEVIG_METHODS, build_day, devig_two_way
from model.picks import split_book_name
from pipeline.lines_cmd import attach_team_codes

TWO_WAY = ["vkupno", "tim_vkupno_d", "tim_vkupno_g", "poeni", "asistencii", "skokovi", "trojki", "pra"]


def _pairs(L: pd.DataFrame) -> pd.DataFrame:
    """One row per bookmaker × game × (player) × market × line with both prices (last snapshot of the day)."""
    L = L[L["pazar"].isin(TWO_WAY) & L["strana"].isin(["nad", "pod"])].copy()
    L = L[L["snimka"] == L.groupby("kladilnica")["snimka"].transform("max")]
    # player key: first letter of the first name + surname (both books write names differently)
    nm = L["igrac"].fillna("").map(lambda n: "" if not n else "{0[0]:.1}|{0[1]}".format(split_book_name(n)))
    L["kluc"] = nm
    # evrotip player markets: `away` = the player's own team → the game is found via either team code
    L["gm"] = L["tim_d"].fillna("") + "-" + L["tim_g"].fillna("")
    pl = L["igrac"].notna()
    L.loc[pl, "gm"] = ""                                     # players: the name + league + line is the key
    w = L.pivot_table(index=["kladilnica", "liga", "gm", "kluc", "pazar", "linija"], columns="strana",
                      values="kvota", aggfunc="first").dropna().reset_index()
    return w


def _ladders(frames: list[pd.DataFrame]) -> list[tuple]:
    """Per bookmaker × market: RMS distance (pp) of each method's fair P from a probit-linear fit per game."""
    rows = []
    L = pd.concat(frames, ignore_index=True)
    L = L[L["snimka"] == L.groupby(["den", "kladilnica"])["snimka"].transform("max")]
    for (book, paz), g in L[L["pazar"].isin(["vkupno", "hendikep"])].groupby(["kladilnica", "pazar"]):
        a, b = ("nad", "pod") if paz == "vkupno" else ("1", "2")
        w = g.pivot_table(index=["den", "nat_id", "linija"], columns="strana", values="kvota", aggfunc="first")
        if a not in w or b not in w:
            continue
        w = w[[a, b]].dropna().reset_index()
        res, n_games, ps = {x: [] for x in DEVIG_METHODS}, 0, []
        for _, gg in w.groupby(["den", "nat_id"]):
            if len(gg) < 5:
                continue
            n_games += 1
            x = gg["linija"].values
            for meth in DEVIG_METHODS:
                p = np.array([devig_two_way(k1, k2, meth) for k1, k2 in zip(gg[a], gg[b])])
                z = st.norm.ppf(np.clip(p, 1e-4, 1 - 1e-4))
                fit = st.norm.cdf(np.polyval(np.polyfit(x, z, 1), x))
                res[meth] += list((p - fit) * 100)
            ps += list(gg[a].rdiv(1) / (gg[a].rdiv(1) + gg[b].rdiv(1)))
        if n_games:
            rows.append((f"{book} · {paz}", n_games, len(ps), f"{min(ps) * 100:.0f}–{max(ps) * 100:.0f}%",
                         [float(np.sqrt(np.mean(np.square(res[x])))) for x in DEVIG_METHODS]))
    return rows


def devig_report(settings: Settings) -> str:
    days = sorted({p.split("\\")[-1].split("/")[-1] for p in glob.glob(str(settings.paths.raw / "*" / "20*"))})
    frames, frames_raw = [], []
    for d in days:
        L = build_day(settings.paths.raw, d)
        if len(L):
            frames.append(_pairs(attach_team_codes(L, settings)).assign(den=d))
            frames_raw.append(L.assign(den=d))
    out = [f"# De-vig: пропорционален / power / Shin", "",
           f"*{date.today()} · `python main.py backtest --devig` · снимки: {', '.join(days) or '—'}*", ""]
    if not frames:
        return "\n".join(out + ["Нема снимки — прво `python main.py linii`."])
    w = pd.concat(frames, ignore_index=True)
    key = ["den", "liga", "gm", "kluc", "pazar", "linija"]
    m = w[w["kladilnica"] == "mozzart"].merge(w[w["kladilnica"] == "evrotip"], on=key, suffixes=("_m", "_e"))
    if m.empty:
        return "\n".join(out + ["Нема ист пазар/линија кај двете кладилници."])
    for meth in DEVIG_METHODS:
        m[f"pm_{meth}"] = [devig_two_way(a, b, meth) for a, b in zip(m["nad_m"], m["pod_m"])]
        m[f"pe_{meth}"] = [devig_two_way(a, b, meth) for a, b in zip(m["nad_e"], m["pod_e"])]
        m[f"d_{meth}"] = (m[f"pm_{meth}"] - m[f"pe_{meth}"]).abs()
    m["marza_m"] = 1 / m["nad_m"] + 1 / m["pod_m"] - 1
    m["marza_e"] = 1 / m["nad_e"] + 1 / m["pod_e"] - 1
    skew = (m["pm_proporcionalen"] - 0.5).abs()
    m["grupa"] = pd.cut(skew, [-0.01, 0.10, 0.20, 0.30, 0.51],
                        labels=["рамни (40–60%)", "60–70% / 30–40%", "70–80% / 20–30%", "> 80% / < 20%"])
    out += ["Тест без резултати: истиот натпревар, пазар и линија кај Mozzart и evrotip. Кладилниците имаат "
            "РАЗЛИЧНА маржа, па методот што правилно ја вади маржата треба да ги доближи двете „фер“ веројатности. "
            "Бројка = просечна разлика |P Mozzart − P evrotip| во процентни поени (помало = подобро).", "",
            f"Парови: {len(m)} (тимски {int((m['kluc'] == '').sum())}, играчки {int((m['kluc'] != '').sum())}) · "
            f"просечна маржа Mozzart {m['marza_m'].mean() * 100:.1f}%, evrotip {m['marza_e'].mean() * 100:.1f}%", "",
            "| Нерамност (P НАД) | n | пропорционален | power | Shin |", "|---|---|---|---|---|"]
    for g, grp in m.groupby("grupa", observed=True):
        out.append(f"| {g} | {len(grp)} | " + " | ".join(f"{grp[f'd_{x}'].mean() * 100:.2f}" for x in DEVIG_METHODS) + " |")
    out.append("| **сите** | " + f"{len(m)} | " + " | ".join(f"**{m[f'd_{x}'].mean() * 100:.2f}**" for x in DEVIG_METHODS) + " |")
    out += ["", "### Тест 2: мазност на скалата на алтернативни линии (една кладилница)", "",
            "Иста кладилница, ист натпревар, ≥ 5 алтернативни линии (тотал / хендикеп). Фер P мора да лежи на мазна "
            "крива (нормална распределба = права во probit). Бројка = RMS отстапување од кривата во pp (помало = "
            "методот ја вади маржата онака како што кладилницата ја става).", "",
            "| Кладилница · пазар | натпревари | парови | опсег P | пропорционален | power | Shin |", "|---|---|---|---|---|---|---|"]
    ladder_best = {}
    for row in _ladders(frames_raw):
        out.append("| " + " | ".join(str(x) for x in row[:4]) + " | " + " | ".join(f"{v:.3f}" for v in row[4]) + " |")
        for meth, v in zip(DEVIG_METHODS, row[4]):
            ladder_best[meth] = ladder_best.get(meth, 0) + v * row[2]
    if ladder_best:
        out += ["", f"**Тест 2 — најмазно: {min(ladder_best, key=ladder_best.get)}.**"]
    uneq = m[skew > 0.20]
    best_all = min(DEVIG_METHODS, key=lambda x: m[f"d_{x}"].mean())
    best_uneq = min(DEVIG_METHODS, key=lambda x: uneq[f"d_{x}"].mean()) if len(uneq) else None
    ex = m.loc[skew.idxmax()]
    out += ["", f"Пример (најнерамен пар): {ex['liga']} {ex['pazar']} {ex['linija']:g} {ex['kluc'] or ex['gm']} — "
            f"Mozzart {ex['nad_m']:.2f}/{ex['pod_m']:.2f}, evrotip {ex['nad_e']:.2f}/{ex['pod_e']:.2f} → P(НАД) Mozzart: "
            + ", ".join(f"{x} {ex[f'pm_{x}'] * 100:.1f}%" for x in DEVIG_METHODS), "",
            f"**Најдобар на сите парови: {best_all}; на нерамни (> 70/30): {best_uneq or '—'} (n={len(uneq)}).** "
            f"Во употреба: `{DEVIG_METHOD}` (model/lines.py → DEVIG_METHOD).", "",
            "> Ограничување: една кладилница може да ја става маржата различно од друга — ова е посреден тест. "
            "Конечната проверка е log loss против резултатите на хартија (по 100 одлучени избори)."]
    return "\n".join(out)
