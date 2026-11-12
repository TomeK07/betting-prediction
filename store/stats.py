"""Pick statistics (PROCES §2): hit rate vs expected, calibration by level, Brier, log loss, ROI, CLV.

Only decided picks (DA/NE) with a p_model enter the accuracy metrics (PROCES §6.5).
ROI only where a price exists; CLV only where a closing price exists (decision 29.10).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from model.calibration import best_shrink, league_shrink, shrink_toward_half

# p_model levels for calibration (PROCES §2, §8): [lo, hi)
LEVELS = [(0.0, 0.55, "<55%"), (0.55, 0.60, "55–60%"), (0.60, 0.65, "60–65%"),
          (0.65, 0.70, "65–70%"), (0.70, 1.01, "70%+")]
LEAGUE_MK = {"nba": "NBA", "euroleague": "Евролига", "eurocup": "Еврокуп"}


def wilson_interval(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a hit rate — honest about small samples."""
    if n == 0:
        return (0.0, 1.0)
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def sample_label(n: int) -> str:
    if n < 30:
        return "само следи"
    if n < 100:
        return "насока"
    return "заклучок"


@dataclass
class Summary:
    n: int
    hits: int
    hit_rate: Optional[float]
    expected: Optional[float]
    ci: tuple[float, float]
    brier: Optional[float]
    log_loss: Optional[float]
    roi: Optional[float]            # "ROI модел": every pick with a real (verified) price, flat stake
    roi_n: int
    roi_paid: Optional[float]       # "ROI уплатено": only what Tome actually placed, real stakes
    roi_paid_n: int
    paid_stake: float
    clv_mean: Optional[float]
    clv_positive_share: Optional[float]
    clv_n: int


def decided(picks: pd.DataFrame) -> pd.DataFrame:
    d = picks[picks["status"].isin(["DA", "NE"]) & picks["p_model"].notna()].copy()
    d["hit"] = (d["status"] == "DA").astype(int)
    return d


def summarize(d: pd.DataFrame) -> Summary:
    n = len(d)
    hits = int(d["hit"].sum()) if n else 0
    if n:
        p = d["p_model"].clip(1e-6, 1 - 1e-6)
        brier = float(((p - d["hit"]) ** 2).mean())
        log_loss = float(-(d["hit"] * p.apply(math.log) + (1 - d["hit"]) * (1 - p).apply(math.log)).mean())
    else:
        brier = log_loss = None
    verified = d["kvota_proverena"].fillna(1) == 1 if "kvota_proverena" in d else True
    priced = d[d["kvota"].notna() & verified]
    profit = (priced["hit"] * (priced["kvota"] - 1) - (1 - priced["hit"])).sum()
    paid_profit, paid_stake, paid = 0.0, 0.0, d.iloc[0:0]
    if {"uplateno", "uplateno_iznos"} <= set(d.columns):
        paid = d[(d["uplateno"] == 1) & d["uplateno_iznos"].notna() & d["kvota"].notna()]
        stake = paid["uplateno_iznos"]
        paid_profit = float((paid["hit"] * stake * (paid["kvota"] - 1) - (1 - paid["hit"]) * stake).sum())
        paid_stake = float(stake.sum())
    clv = d["clv"].dropna() if "clv" in d else pd.Series(dtype=float)
    return Summary(
        n=n, hits=hits,
        hit_rate=hits / n if n else None,
        expected=float(d["p_model"].mean()) if n else None,
        ci=wilson_interval(hits, n),
        brier=brier, log_loss=log_loss,
        roi=float(profit / len(priced)) if len(priced) else None, roi_n=len(priced),
        roi_paid=float(paid_profit / paid_stake) if paid_stake else None, roi_paid_n=len(paid), paid_stake=paid_stake,
        clv_mean=float(clv.mean()) if len(clv) else None,
        clv_positive_share=float((clv > 0).mean()) if len(clv) else None,
        clv_n=len(clv),
    )


def calibration_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lo, hi, label in LEVELS:
        part = d[(d["p_model"] >= lo) & (d["p_model"] < hi)]
        if part.empty:
            continue
        n, hits = len(part), int(part["hit"].sum())
        rows.append({"ниво": label, "n": n, "погодени": hits, "погодок": hits / n,
                     "очекувано": float(part["p_model"].mean()),
                     "разлика_pp": (hits / n - float(part["p_model"].mean())) * 100})
    return pd.DataFrame(rows)


def side_lines(d: pd.DataFrame) -> list[str]:
    """Both sides, always (decision 29.10): hit rate, expected and Brier for НАД and ПОД separately."""
    sided = d[d["strana"].isin(["nad", "pod"])]
    if sided.empty:
        return []
    out = ["  ДВЕТЕ СТРАНИ — НАД наспроти ПОД (вкупно и по пазар):"]
    for side in ("nad", "pod"):
        part = sided[sided["strana"] == side]
        if part.empty:
            continue
        n, h = len(part), int(part["hit"].sum())
        lo, hi = wilson_interval(h, n)
        brier = float(((part["p_model"] - part["hit"]) ** 2).mean())
        out.append(f"    {side.upper():<4} вкупно: {h}/{n} = {h / n * 100:.0f}% (95% {lo * 100:.0f}–{hi * 100:.0f}%), "
                   f"очекувано {part['p_model'].mean() * 100:.0f}%, Brier {brier:.3f} · {sample_label(n)}")
    grp = sided.groupby(["pazar", "strana"]).agg(n=("hit", "size"), hits=("hit", "sum"), p=("p_model", "mean"))
    for (m, side), r in grp.iterrows():
        out.append(f"      {m:<20} {side.upper():<4} {int(r.hits)}/{int(r.n)} = {r.hits / r.n * 100:.0f}% "
                   f"(очекувано {r.p * 100:.0f}%)")
    return out


def _pct(x: Optional[float]) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


PAPER_REVIEW_AT = 100            # PROCES §4А: after 100 paper picks → review of the real edge vs Mozzart
PAPER_MIN_NIGHTS = 10            # Tome 29.10: AND ≥ 10 nights — picks of one night are correlated, not independent


def _in_ticket(vlez: object) -> bool:
    try:
        return bool(json.loads(vlez or "{}").get("tiket"))
    except (TypeError, ValueError):
        return False


def paper_section(picks: pd.DataFrame) -> list[str]:
    """Python model on paper (verzija py-*): progress to the 100-pick review, calibration by league × market × side,
    CLV. A REPORT only — nothing is changed automatically (PROCES §4А, §6.4)."""
    py = picks[picks["verzija_model"].fillna("").str.startswith("py")]
    if py.empty:
        return []
    d = decided(py)
    waiting = int((py["status"] == "CEKA").sum())
    nights = int(d["datum"].nunique())
    ready = len(d) >= PAPER_REVIEW_AT and nights >= PAPER_MIN_NIGHTS
    lines = ["", "ХАРТИЈА — Python модел (учење)", "=" * 30,
             f"Преглед на вистинската предност: одлучени {len(d)} / {PAPER_REVIEW_AT} И вечери {nights} / "
             f"{PAPER_MIN_NIGHTS} → {'✅ спремно за преглед' if ready else 'уште не'} · чекаат: {waiting} · "
             f"одбиени (запишани): {int((py['status'] == 'ODBIEN').sum())}",
             "(изборите од една вечер се меѓусебно поврзани — не се независни докази; затоа и услов за вечери)"]
    if d.empty:
        return lines + ["(уште нема одлучени избори на хартија)"]

    def summary_line(x: pd.DataFrame, title: str) -> str:
        if x.empty:
            return f"{title}: уште нема одлучени"
        s = summarize(x)
        return (f"{title}: {s.hits}/{s.n} = {_pct(s.hit_rate)} (95% {_pct(s.ci[0])}–{_pct(s.ci[1])}) · очекувано "
                f"{_pct(s.expected)} · Brier {s.brier:.3f} · ROI модел {_pct(s.roi)} · вечери {x['datum'].nunique()} · "
                + (f"CLV просек {_pct(s.clv_mean)}, позитивен {_pct(s.clv_positive_share)} (n={s.clv_n})"
                   if s.clv_n else "CLV: нема затворачки квоти (пушти `linii` пред натпреварите)"))
    top = d[d["vlezovi_json"].map(_in_ticket)] if "vlezovi_json" in d else d.iloc[0:0]
    lines.append(summary_line(top, "ТОП избори (би влегле во тикет)"))
    lines.append(summary_line(d, "Сите на хартија (учење)"))
    lines.append("Калибрација по лига × пазар × страна (n, погодок, очекувано):")
    grp = d.groupby(["liga", "pazar", "strana"]).agg(n=("hit", "size"), h=("hit", "sum"), p=("p_kalibriran", "mean"))
    for (lg, m, side), r in grp.iterrows():
        flag = "" if r.n < 30 else (" ✅" if abs(r.h / r.n - r.p) <= 0.05 else " ❌")
        lines.append(f"  {lg:<10} {m:<12} {str(side).upper():<4} {int(r.h)}/{int(r.n)} = {r.h / r.n * 100:.0f}% "
                     f"(очекувано {r.p * 100:.0f}%){flag}{'' if r.n >= 30 else ' · мал примерок'}")
    lines.append("*Нова калибрација (isotonic) по лига × пазар × страна се предлага кога групата ќе има ≥ 50 избори; "
                 "се применува само по backtest и одобрение (ништо „само од себе“).*")
    return lines


def filter_section(picks: pd.DataFrame) -> list[str]:
    """Tome 31.10: do the filters add value? Rejected picks are graded too (`ishod`, status stays ODBIEN).
    If the rejected lose MORE often than the chosen (vs their own p_model) → the filters help."""
    if "ishod" not in picks:
        return []
    py = picks[picks["verzija_model"].fillna("").str.startswith("py") & picks["ishod"].isin(["DA", "NE"])
               & picks["p_model"].notna()].copy()
    if py.empty:
        return []
    py["hit"] = (py["ishod"] == "DA").astype(int)
    chosen = py[py["status"].isin(["DA", "NE"])]
    groups = [("ИЗБРАНИ (поминаа сè)", chosen[chosen["odbien_od"].fillna("") != "РАЧНО"]),
              ("  рачно прескокнати (Томе)", chosen[chosen["odbien_od"] == "РАЧНО"])]
    rej = py[py["status"] == "ODBIEN"]
    groups.append(("ОДБИЕНИ — сите", rej))
    for who in ("МОДЕЛ", "ФИЛТЕР", "СОСТАВ", "РАЧНО"):
        groups.append((f"  одбиени од: {who}", rej[rej["odbien_od"] == who]))
    lines = ["", "ДАЛИ ФИЛТРИТЕ ДОДАВААТ ВРЕДНОСТ — одбиени наспроти избрани (исход од box score)", "=" * 30,
             "| Група | n | погодок | очекувано (p_model) | разлика | ROI рамен улог |", "|---|---|---|---|---|---|"]
    for title, g in groups:
        if g.empty:
            continue
        s = summarize(g)
        lines.append(f"| {title} | {s.n} | {_pct(s.hit_rate)} (95% {_pct(s.ci[0])}–{_pct(s.ci[1])}) | {_pct(s.expected)} | "
                     f"{(s.hit_rate - s.expected) * 100:+.1f} pp | {_pct(s.roi)} (n={s.roi_n}) |")
    lines.append("*Филтерот помага ако одбиените паѓаат ПОЧЕСТО од избраните. Одбиените со ПРОВЕРИ_ПОДАТОЦИ често имаат "
                 "превисок p_model → гледај ја и колоната „разлика“. Мал примерок = само насока.*")
    return lines


def render_report(picks: pd.DataFrame, league: Optional[str] = None, since: Optional[str] = None) -> str:
    """Macedonian text report (console / reports/)."""
    df = picks
    if league:
        df = df[df["liga"] == league]
    if since:
        df = df[df["datum"] >= since]
    lines = []
    title = "СТАТИСТИКА НА ИЗБОРИТЕ" + (f" од {since}" if since else "")
    lines += [title, "=" * len(title)]
    counts = df["status"].value_counts().to_dict()
    lines.append("Статуси: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    waiting = df[df["status"] == "CEKA"]
    if len(waiting):
        lines.append(f"⏳ Чекаат резултат: {len(waiting)} (пушти `python main.py grade`)")

    for lg in [l for l in ("nba", "euroleague", "eurocup") if l in set(df["liga"])]:
        d = decided(df[df["liga"] == lg])
        s = summarize(d)
        lines += ["", f"## {LEAGUE_MK[lg]} — {s.n} одлучени ({sample_label(s.n)})"]
        if not s.n:
            lines.append("  нема одлучени избори со p_model")
            continue
        lines.append(f"  Погодок: {s.hits}/{s.n} = {_pct(s.hit_rate)}  "
                     f"(95% интервал {_pct(s.ci[0])}–{_pct(s.ci[1])})  · очекувано (просек p_model): {_pct(s.expected)}  "
                     f"· разлика {((s.hit_rate - s.expected) * 100):+.1f} pp")
        lines.append(f"  Brier: {s.brier:.3f} (50/50 = 0.250, цел < 0.24)  · Log loss: {s.log_loss:.3f}")
        lines.append(f"  ROI модел (рамен улог, само проверени квоти): {_pct(s.roi)} на {s.roi_n} избори"
                     + ("" if s.roi_n == s.n else f" ({s.n - s.roi_n} без квота или со ориентациона/непроверена — надвор)"))
        lines.append("  ROI уплатено: " + (f"{_pct(s.roi_paid)} на {s.roi_paid_n} уплатени ({s.paid_stake:.0f} ден.)"
                                           if s.roi_paid_n else "нема запишани уплати (`python main.py uplata`)"))
        k_rule = league_shrink(lg)
        if k_rule is not None:
            k_best = best_shrink(d["p_model"].values, d["hit"].values)
            shr = d["p_model"].apply(lambda p: shrink_toward_half(p, k_rule))
            lines.append(f"  Привремено стегање кон 50% (k={k_rule}, СЛЕДИ): Brier {((shr - d['hit']) ** 2).mean():.3f} "
                         f"наместо {s.brier:.3f} · Brier-оптимален k на архивата: {k_best:+.2f}")
        lines.append("  CLV: " + (f"просек {_pct(s.clv_mean)}, позитивен кај {_pct(s.clv_positive_share)} "
                                  f"(n={s.clv_n})" if s.clv_n else "нема избори со затворачка квота"))
        cal = calibration_table(d)
        lines.append("  Калибрација (кога велиме X%, колку паѓа):")
        for r in cal.itertuples():
            flag = "✅" if abs(r.разлика_pp) <= 5 else "❌"
            lines.append(f"    {r.ниво:>7}: {r.погодени}/{r.n} = {r.погодок * 100:.0f}% (очекувано {r.очекувано * 100:.0f}%, "
                         f"{r.разлика_pp:+.0f} pp) {flag}{'' if r.n >= 30 else ' · мал примерок'}")
        by_market = d.groupby("pazar").agg(n=("hit", "size"), hits=("hit", "sum"), p=("p_model", "mean"))
        lines.append("  По пазар:")
        for m, r in by_market.sort_values("n", ascending=False).iterrows():
            lines.append(f"    {m:<20} {int(r.hits)}/{int(r.n)} = {r.hits / r.n * 100:.0f}% (очекувано {r.p * 100:.0f}%)")
        lines += side_lines(d)
    lines += paper_section(df)
    lines += filter_section(df)
    return "\n".join(lines)
