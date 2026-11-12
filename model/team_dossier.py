"""Team dossier (Tome 30.10): decision parameters, the 13-section profil.md, player cards, vesti.md, league index.

Python only — numbers from our box scores and archive. Every parameter is
    {vrednost, izvor: python|agent, datum, status: BROJKA | OPIS | POTVRDEN_PARAMETAR}
Only BROJKA and POTVRDEN_PARAMETAR may ever feed a calculation; OPIS (agents, approved by Tome) only explains.
Agent fields stay `None` until a T9 proposal is approved (`profil odobri`) — never filled here.
"""
from __future__ import annotations

import re
import sqlite3
import unicodedata
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

CARD_MIN_MINUTES = 15.0
CLOSE, BLOWOUT = 6, 15                  # final margin ≤ 6 = close game, ≥ 15 = big margin
# Tome 31.10: the AI description was too weak — style comes from numbers (`stil`), agents only for the coach and GM
AGENT_FIELDS = ["trener_od_koga", "trener_filozofija", "gm"]
NEWS_TYPES = ["ПОВРЕДА", "СОСТАВ", "ТРЕНЕР-ИЗЈАВА", "РОТАЦИЈА", "ТРЕЈД", "ДОГОВОР", "РАСПОРЕД", "ФОРМА", "АНАЛИЗА", "ДРУГО"]
POS_MK = {"G": "бекови", "F": "крила", "C": "центри"}
BALANCED_PP = 3.0                        # scoring share within ±3 pp of the league for every position = balanced


def _r(x, nd=1):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def field(value, today: date, source: str = "python", status: str = "BROJKA", note: Optional[str] = None) -> dict:
    out = {"vrednost": value, "izvor": source, "datum": today.isoformat(), "status": status}
    if note:
        out["zabeleska"] = note
    return out


def slug(label: str) -> str:
    t = unicodedata.normalize("NFKD", label).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9]+", "_", t.split(" (")[0]).strip("_")


# ---------- league-wide team table (ranks need all teams) ----------

def team_table(games: pd.DataFrame) -> pd.DataFrame:
    g = games.groupby("team")
    s = lambda c: g[c].sum()
    t = pd.DataFrame({
        "n": g.size(), "pace": g["pace"].mean(),
        "fga": s("fga"), "fga3": s("fga3"), "fgm3": s("fgm3"), "fta": s("fta"), "fga2": s("fga2"), "fgm2": s("fgm2"),
        "fgm": s("fgm"), "o_fga": s("opp_fga"), "o_fga3": s("opp_fga3"), "o_fgm3": s("opp_fgm3"), "o_fta": s("opp_fta"),
        "o_fga2": s("opp_fga2"), "o_fgm2": s("opp_fgm2"), "o_fgm": s("opp_fgm"),
        "oreb": s("oreb"), "dreb": s("dreb"), "o_oreb": s("opp_oreb"), "o_dreb": s("opp_dreb"),
        "pf": g["pf"].mean(), "blk": g["blk"].mean(), "o_fta_g": g["opp_fta"].mean(), "b2b": g["b2b"].sum()})
    t["3par"] = 100 * t["fga3"] / t["fga"]
    t["ftr"] = 100 * t["fta"] / t["fga"]
    t["p2"] = 100 * t["fgm2"] / t["fga2"]
    t["p3"] = 100 * t["fgm3"] / t["fga3"]
    t["efg"] = 100 * (t["fgm"] + 0.5 * t["fgm3"]) / t["fga"]
    t["o_3par"] = 100 * t["o_fga3"] / t["o_fga"]
    t["o_ftr"] = 100 * t["o_fta"] / t["o_fga"]
    t["o_p2"] = 100 * t["o_fgm2"] / t["o_fga2"]
    t["o_p3"] = 100 * t["o_fgm3"] / t["o_fga3"]
    t["o_efg"] = 100 * (t["o_fgm"] + 0.5 * t["o_fgm3"]) / t["o_fga"]
    t["oreb_pct"] = 100 * t["oreb"] / (t["oreb"] + t["o_dreb"])
    t["dreb_pct"] = 100 * t["dreb"] / (t["dreb"] + t["o_oreb"])
    return t


def _rank(t: pd.DataFrame, col: str, team: str, high_is_1: bool = True) -> Optional[int]:
    if team not in t.index or pd.isna(t.loc[team, col]):
        return None
    return int(t[col].rank(ascending=not high_is_1, method="min")[team])


# ---------- per-team parameters ----------

def rotation_by_margin(team: str, games: pd.DataFrame, players: pd.DataFrame) -> dict:
    """Minutes of the top-5 (by season minutes) in close games vs big-margin games; top-3 = the stars."""
    t = games[games["team"] == team].assign(margin=lambda d: (d["pts"] - d["opp_pts"]).abs())
    p = players[(players["team"] == team) & (players["minutes"].fillna(0) > 0)]
    if t.empty or p.empty:
        return {}
    top = p.groupby("player_code")["minutes"].mean().sort_values(ascending=False)
    p = p.merge(t[["game_code", "margin"]], on="game_code")
    out = {}
    for name, codes in (("top5", list(top.index[:5])), ("top3", list(top.index[:3]))):
        q = p[p["player_code"].isin(codes)]
        close, blow = q[q["margin"] <= CLOSE], q[q["margin"] >= BLOWOUT]
        out[name] = {"blizok_min": _r(close["minutes"].mean()), "blizok_n": int(close["game_code"].nunique()),
                     "razlika_min": _r(blow["minutes"].mean()), "razlika_n": int(blow["game_code"].nunique())}
    return out


def missed_games(team: str, games: pd.DataFrame, players: pd.DataFrame, league: str) -> dict:
    """Top-5 players: team games they did not play, by reason (NBA box score reason; EL/EC = not on the sheet)."""
    t = games[games["team"] == team]
    p = players[players["team"] == team]
    played = p[p["minutes"].fillna(0) > 0]
    top = played.groupby("player_code")["minutes"].mean().sort_values(ascending=False).head(5).index
    label = p.drop_duplicates("player_code").set_index("player_code")["label"]
    out = []
    for code in top:
        got = set(played[played["player_code"] == code]["game_code"])
        miss = [g for g in t["game_code"] if g not in got]
        if not miss:
            continue
        if league == "nba" and "absence_reason" in p:
            rows = p[(p["player_code"] == code) & p["game_code"].isin(miss)]
            reasons = rows["absence_reason"].fillna("не е во записник").value_counts().to_dict()
            if len(miss) > len(rows):                    # no row at all on that sheet
                reasons["не е во записник"] = reasons.get("не е во записник", 0) + len(miss) - len(rows)
        else:
            reasons = {"не е на листата / не играл": len(miss)}
        out.append({"igrac": label.get(code, code), "propushteni": len(miss), "prichini": reasons})
    return {"top5_propushteni": out, "b2b_natprevari": int(t["b2b"].sum()) if "b2b" in t else None}


def schedule_load(team: str, games: pd.DataFrame, today: date) -> dict:
    t = games[games["team"] == team].sort_values("game_date")
    if t.empty:
        return {}
    away_run, best = 0, 0
    for h in t["home"]:
        away_run = 0 if h else away_run + 1
        best = max(best, away_run)
    last7 = t[pd.to_datetime(t["game_date"]) >= pd.Timestamp(today) - pd.Timedelta(days=7)]
    return {"natprevari_posledni_7_dena": int(len(last7)), "b2b_sezona": int(t["b2b"].sum()) if "b2b" in t else None,
            "najdolga_gostinska_serija": int(best)}


def model_errors(team: str, league: str, archive: Path) -> dict:
    """Paper/real picks that involve the team (graded): hit rate vs what the model expected."""
    if not archive.exists():
        return {"n": 0}
    con = sqlite3.connect(archive)
    d = pd.read_sql("SELECT pazar, strana, status, p_kalibriran, tim, natprevar FROM izbori "
                    "WHERE liga=? AND status IN ('DA','NE')", con, params=[league])
    con.close()
    d = d[(d["tim"] == team) | d["natprevar"].fillna("").str.contains(rf"\b{team}\b")]
    if d.empty:
        return {"n": 0}
    hit = (d["status"] == "DA").mean()
    by = {k: f"{(g['status'] == 'DA').sum()}/{len(g)}" for k, g in d.groupby("pazar")}
    withp = d[d["p_kalibriran"].notna()]                 # legacy manual picks may have no probability
    exp = withp["p_kalibriran"].mean() if len(withp) else np.nan
    hit_p = (withp["status"] == "DA").mean() if len(withp) else np.nan
    return {"n": int(len(d)), "pogodok_pct": _r(100 * hit), "ochekuvano_pct": _r(100 * exp),
            "razlika_pp": _r(100 * (hit_p - exp)), "po_pazar": by}


# ---------- style tags from numbers (Tome 31.10) ----------

def style_table(t: pd.DataFrame, games: pd.DataFrame, players: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    """League-wide columns the style tags are ranked on (team_table + rotation depth, star sitting, scoring by position)."""
    from model.dvp import primary_position
    s = t[["n", "pace", "3par", "oreb_pct", "pf", "o_3par", "o_p2", "o_ftr"]].copy()
    p = players[players["minutes"].fillna(0) > 0]
    mins = p.groupby(["team", "player_code"])["minutes"].sum()
    avg = mins / mins.index.get_level_values("team").map(s["n"]).values      # missed games count as 0 minutes
    s["dlabocina"] = (avg >= CARD_MIN_MINUTES).groupby(level="team").sum()
    sit = {}
    for team in s.index:
        t3 = rotation_by_margin(team, games, players).get("top3") or {}
        sit[team] = (t3["blizok_min"] - t3["razlika_min"]) if t3.get("blizok_min") and t3.get("razlika_min") else np.nan
    s["vadi_zvezdi"] = pd.Series(sit)
    pos = roster.assign(pos=roster["position"].map(primary_position)).drop_duplicates("player_code")[["player_code", "pos"]]
    pts = p.merge(pos, on="player_code", how="left").groupby(["team", "pos"])["pts"].sum().unstack(fill_value=0)
    share = 100 * pts.div(pts.sum(axis=1), axis=0)
    for c in POS_MK:
        s[f"udel_{c}"] = share[c] if c in share else np.nan
    return s


def _tier(s: pd.DataFrame, col: str, team: str, high: str, mid: str, low: str) -> tuple[Optional[str], Optional[int], int]:
    """Tercile of the league rank (1 = highest value): top third → `high`, bottom third → `low`."""
    col_s = s[col].dropna()
    n = len(col_s)
    if team not in col_s.index or n < 3:
        return None, None, n
    r = int(col_s.rank(ascending=False, method="min")[team])
    return (high if r <= n / 3 else low if r > 2 * n / 3 else mid), r, n


def style_tags(team: str, s: pd.DataFrame, dvp: Optional[dict]) -> dict:
    """Every tag = {oznaka, brojki}: the word is only a label for the number next to it (rank / league size)."""
    if team not in s.index:
        return {}
    r = s.loc[team]
    out = {}
    lab, rk, n = _tier(s, "pace", team, "брзо", "средно", "бавно")
    out["tempo"] = {"oznaka": lab, "brojki": {"poseda": _r(r["pace"]), "rang": rk, "od": n}}
    lab, rk, n = _tier(s, "3par", team, "многу тројки", "просечно тројки", "малку тројки")
    out["trojki"] = {"oznaka": lab, "brojki": {"3par_pct": _r(r["3par"]), "rang": rk, "od": n}}
    shares = {c: _r(r[f"udel_{c}"]) for c in POS_MK}
    league = {c: _r(s[f"udel_{c}"].mean()) for c in POS_MK}
    diffs = {c: shares[c] - league[c] for c in POS_MK if shares[c] is not None and league[c] is not None}
    top = max(diffs, key=diffs.get) if diffs else None
    out["napad_preku"] = {"oznaka": None if top is None else ("избалансиран" if diffs[top] < BALANCED_PP else f"преку {POS_MK[top]}"),
                          "brojki": {"udel_poeni_pct": {POS_MK[c]: shares[c] for c in POS_MK},
                                     "liga_pct": {POS_MK[c]: league[c] for c in POS_MK}}}
    lab, rk, n = _tier(s, "oreb_pct", team, "напаѓа офанзивен скок", "средно", "се враќа во одбрана")
    out["ofanziven_skok"] = {"oznaka": lab, "brojki": {"oreb_pct": _r(r["oreb_pct"]), "rang": rk, "od": n}}
    lab, rk, n = _tier(s, "dlabocina", team, "длабока", "средна", "кратка")
    out["rotacija"] = {"oznaka": lab, "brojki": {"igraci_15_min": int(r["dlabocina"]) if pd.notna(r["dlabocina"]) else None,
                                                 "rang": rk, "od": n}}
    lab, rk, n = _tier(s, "vadi_zvezdi", team, "ги вади ѕвездите при голема разлика", "умерено", "ги држи ѕвездите")
    out["vadenje_zvezdi"] = {"oznaka": lab, "brojki": {"top3_min_pomalku": _r(r["vadi_zvezdi"]), "rang": rk, "od": n}}
    lab, rk, n = _tier(s, "pf", team, "фаулира многу", "средно", "фаулира малку")
    lab2, rk2, _ = _tier(s, "o_ftr", team, "праќа многу на линија", "средно праќа на линија", "праќа малку на линија")
    out["fauli"] = {"oznaka": "; ".join(x for x in (lab, lab2) if x) or None, "brojki": {"pf_po_natprevar": _r(r["pf"]), "rang": rk,
                                                                  "ftr_protivnik": _r(r["o_ftr"]), "rang_ftr": rk2, "od": n}}
    lab3, rk3, _ = _tier(s, "o_3par", team, "пушта многу тројки", "средно тројки", "ги брани тројките")
    lab2p, rk2p, _ = _tier(s, "o_p2", team, "слаба заштита на кош", "средна заштита на кош", "силна заштита на кош")
    worst = max((dvp or {}).items(), key=lambda kv: kv[1].get("poeni_pct") or -999, default=None)
    dvp_lab = None
    if worst and worst[1].get("poeni_pct") is not None:
        dvp_lab = (f"пушта над лигата на {worst[0]} ({worst[1]['poeni_pct']:+.0f}% поени)" if worst[1]["poeni_pct"] >= 5
                   else "под или околу лигата на сите позиции" if worst[1]["poeni_pct"] < 0 else None)
    out["dozvoluva"] = {
        "oznaka": lab3 and "; ".join(x for x in (lab3, lab2p, dvp_lab) if x),
        "brojki": {"3par_protivnik": _r(r["o_3par"]), "rang_3par": rk3, "2p_pct_protivnik": _r(r["o_p2"]),
                   "rang_2p": rk2p, "od": n, "dvp_poeni_pct": {k: v.get("poeni_pct") for k, v in (dvp or {}).items()}}}
    return out


def style_line(stil: dict) -> str:
    """One readable line: label (number, rank/n)."""
    def one(k, fmt):
        x = stil.get(k) or {}
        return f"{x['oznaka']} ({fmt(x['brojki'])})" if x.get("oznaka") else None
    parts = [one("tempo", lambda b: f"{b['poseda']} поседи, {b['rang']}/{b['od']}"),
             one("trojki", lambda b: f"3PAr {b['3par_pct']}%, {b['rang']}/{b['od']}"),
             one("napad_preku", lambda b: ", ".join(f"{k} {v}%" for k, v in b["udel_poeni_pct"].items())),
             one("ofanziven_skok", lambda b: f"OREB {b['oreb_pct']}%, {b['rang']}/{b['od']}"),
             one("rotacija", lambda b: f"{b['igraci_15_min']} играчи ≥ 15 мин, {b['rang']}/{b['od']}")]
    return " · ".join(p for p in parts if p) or "—"


def parametri(team: str, league: str, b: dict, t: pd.DataFrame, games: pd.DataFrame, players: pd.DataFrame,
              today: date, archive: Path, old: Optional[dict], stil: Optional[dict] = None) -> dict:
    F = lambda v, **kw: field(v, today, **kw)
    if team not in t.index:
        return {}
    r = t.loc[team]
    rk = lambda col, hi=True: _rank(t, col, team, hi)
    e = b.get("efikasnost") or {}
    out = {
        "tempo": F({"poseda": _r(r["pace"]), "rang": rk("pace")}),
        "ortg": F({"prilagoden": e.get("ortg_prilagoden"), "surov": e.get("ortg_surov"), "rang": e.get("rang_napad")}),
        "drtg": F({"prilagoden": e.get("drtg_prilagoden"), "surov": e.get("drtg_surov"), "rang": e.get("rang_odbrana")}),
        "3par": F({"pct": _r(r["3par"]), "rang": rk("3par")}),
        "profil_sut": F({"3par": _r(r["3par"]), "ftr": _r(r["ftr"]), "2p_pct": _r(r["p2"]), "3p_pct": _r(r["p3"]),
                         "efg": _r(r["efg"]), "rang_efg": rk("efg")}),
        "profil_sut_dozvoluva": F({"3par": _r(r["o_3par"]), "ftr": _r(r["o_ftr"]), "2p_pct": _r(r["o_p2"]),
                                   "3p_pct": _r(r["o_p3"]), "efg": _r(r["o_efg"]), "rang_efg": rk("o_efg", False)}),
        "dvp": F(b.get("dvp")),
        "oreb_strategija": F({"oreb_pct": _r(r["oreb_pct"]), "rang_oreb": rk("oreb_pct"), "dreb_pct": _r(r["dreb_pct"]),
                              "rang_dreb": rk("dreb_pct")}, note="бројки; „стратегија“ (оди/не оди на скок) = опис од агенти"),
        "fauli_tim": F({"pf_po_natprevar": _r(r["pf"]), "rang_najmalku": rk("pf", False)}),
        "fta_dozvoluva": F({"fta_po_natprevar": _r(r["o_fta_g"]), "ftr_dozvoluva": _r(r["o_ftr"]),
                            "rang_najmalku": rk("o_ftr", False)}),
        "zastitnik_kos": F({"blk_po_natprevar": _r(r["blk"]), "rang_blk": rk("blk"), "2p_pct_protivnik": _r(r["o_p2"]),
                            "rang_2p_odbrana": rk("o_p2", False)}, note="кој е заштитникот = опис од агенти"),
    }
    rot = rotation_by_margin(team, games, players)
    out["rotacija_blizok"] = F({k: {"min": v["blizok_min"], "n": v["blizok_n"]} for k, v in rot.items()})
    out["rotacija_razlika"] = F({k: {"min": v["razlika_min"], "n": v["razlika_n"]} for k, v in rot.items()})
    t3 = rot.get("top3") or {}
    out["razlika_vadenje_zvezdi"] = F(
        {"top3_min_blizok": t3.get("blizok_min"), "top3_min_razlika": t3.get("razlika_min"),
         "razlika_min": _r(t3["blizok_min"] - t3["razlika_min"]) if t3.get("blizok_min") and t3.get("razlika_min") else None},
        note=f"блиску = финална разлика ≤ {CLOSE}, голема ≥ {BLOWOUT}; мал примерок на почеток на сезона")
    out["load_management"] = F(missed_games(team, games, players, league))
    out["patuvanje"] = F(schedule_load(team, games, today))
    out["dvizenje_linii"] = F(None, note="треба ≥ 2 недели снимки (`linii`) по тим — уште нема")
    out["model_greshki"] = F(model_errors(team, league, archive))
    out["stil"] = F(stil or {}, note="ознаки од бројки (терцили од рангот во лигата); зборот е само име за бројката. "
                                     "Позиција = прва буква од официјалниот состав (F-C → крило), иста конвенција како DvP")
    old_p = (old or {}).get("parametri") or {}
    for k in AGENT_FIELDS:
        kept = old_p.get(k)
        out[k] = kept if kept and kept.get("izvor") == "agent" else F(None, source="agent", status="OPIS",
                                                                          note="чека T9/T10 + одобрение (`profil odobri`)")
    return out


# ---------- player cards ----------

def player_cards(team: str, games: pd.DataFrame, players: pd.DataFrame, league: str, today: date) -> dict[str, str]:
    p = players[players["team"] == team].copy()
    played = p[p["minutes"].fillna(0) > 0]
    if played.empty:
        return {}
    t = games[games["team"] == team].set_index("game_code")
    g = played.groupby("game_code")
    played = played.assign(tm_min=g["minutes"].transform("sum"), tm_reb=g["reb"].transform("sum"),
                           tm_fgm=g["fgm"].transform("sum"),
                           tm_poss=g["fga"].transform("sum") + 0.44 * g["fta"].transform("sum") + g["tov"].transform("sum"))
    played["opp_reb"] = played["game_code"].map(t["opp_reb"]) if "opp_reb" in t else np.nan
    mins = played.groupby("player_code")["minutes"].mean()
    star = played.assign(u=played["fga"] + 0.44 * played["fta"] + played["tov"]).groupby("player_code")["u"].sum().idxmax()
    star_games = set(played[played["player_code"] == star]["game_code"])
    cards = {}
    foul_out = 6 if league == "nba" else 5
    for code in mins[mins >= CARD_MIN_MINUTES].index:
        q = played[played["player_code"] == code].sort_values("game_date")
        if len(q) < 2:
            continue
        lbl = q["label"].iat[-1]
        tot = q[["minutes", "pts", "reb", "ast", "fgm3", "pf"]].sum()
        usg = 100 * ((q["fga"] + 0.44 * q["fta"] + q["tov"]) * q["tm_min"] / 5).sum() / (q["minutes"] * q["tm_poss"]).sum()
        ast_den = ((q["minutes"] / (q["tm_min"] / 5)) * q["tm_fgm"] - q["fgm"]).sum()
        reb_pct = 100 * (q["reb"] * q["tm_min"] / 5).sum() / (q["minutes"] * (q["tm_reb"] + q["opp_reb"])).sum()
        per36 = lambda c: _r(36 * tot[c] / tot["minutes"])
        home, away = q[q["home"] == True], q[q["home"] != True]  # noqa: E712
        miss = [gc for gc in t.index if gc not in set(q["game_code"])]
        if league == "nba" and "absence_reason" in p:
            rows = p[(p["player_code"] == code) & p["game_code"].isin(miss)]
            reasons = rows["absence_reason"].fillna("не е во записник").value_counts().to_dict()
            if len(miss) > len(rows):
                reasons["не е во записник"] = reasons.get("не е во записник", 0) + len(miss) - len(rows)
        else:
            reasons = {"не е на листата / не играл": len(miss)} if miss else {}
        if code != star:
            with_s, without_s = q[q["game_code"].isin(star_games)], q[~q["game_code"].isin(star_games)]
            star_line = (f"со {played[played['player_code'] == star]['label'].iat[0]}: {with_s['minutes'].mean():.1f} мин, "
                         f"{with_s['pts'].mean():.1f} поени ({len(with_s)}) · без: "
                         + (f"{without_s['minutes'].mean():.1f} мин, {without_s['pts'].mean():.1f} поени ({len(without_s)})"
                            if len(without_s) else "нема натпревар без него"))
        else:
            star_line = "тој е главниот носител на тимот (највисок USG)"
        last5 = q.tail(5)
        lines = [f"# {lbl}", "", f"*БРОЈКИ: Python (box score), {len(q)} натпревари до {q['game_date'].max()} · "
                 f"ажурирано {today.isoformat()}. Опис (улога, против каков чувар) = агенти, по одобрение.*", "",
                 "## Улога", f"- Стартер: {100 * q['starter'].fillna(False).astype(bool).mean():.0f}% од натпреварите · "
                 f"USG {usg:.1f}% · AST% {100 * q['ast'].sum() / ast_den:.1f} · REB% {reb_pct:.1f}",
                 f"- Кога ѕвездата седи: {star_line}", "- Опис на улогата (агенти): —", "",
                 "## Минути", f"- Просек {q['minutes'].mean():.1f} ± {q['minutes'].std():.1f} (мин {q['minutes'].min():.0f}, "
                 f"макс {q['minutes'].max():.0f})", "",
                 "## На 36 минути", f"- поени {per36('pts')} · скокови {per36('reb')} · асистенции {per36('ast')} · "
                 f"тројки {per36('fgm3')}",
                 f"- Дома: {home['pts'].mean():.1f} поени / {home['minutes'].mean():.1f} мин ({len(home)}) · Гости: "
                 + (f"{away['pts'].mean():.1f} поени / {away['minutes'].mean():.1f} мин ({len(away)})" if len(away) else "—"), "",
                 "## Ризик фаули", f"- {per36('pf')} фаули на 36 мин · натпревари со {foul_out - 1}+ фаули: "
                 f"{int((q['pf'] >= foul_out - 1).sum())} · исфрлен ({foul_out}): {int((q['pf'] >= foul_out).sum())}", "",
                 "## Отсуства оваа сезона", "- " + (", ".join(f"{k}: {v}" for k, v in reasons.items()) if reasons else "нема"),
                 "- Ограничување на минути: — (агенти / вести)", "",
                 "## Против каков тип одбрана / чувар", "- — (агенти; наши бројки по тип одбрана уште нема)", "",
                 "## Последни 5", "", "| Датум | Против | Мин | Поени | Скок | Аст | Тројки | Фаули |", "|---|---|---|---|---|---|---|---|"]
        for r in last5.itertuples():
            lines.append(f"| {r.game_date} | {r.opp} | {r.minutes:.0f} | {r.pts:.0f} | {r.reb:.0f} | {r.ast:.0f} | "
                         f"{r.fgm3:.0f} | {r.pf:.0f} |")
        lines += ["", "## Последни вести за улогата", "", "(од vesti.md — само одобрени)", ""]
        cards[slug(lbl)] = "\n".join(lines) + "\n"
    return cards


# ---------- markdown ----------

def _v(par: dict, key: str):
    return ((par or {}).get(key) or {}).get("vrednost")


def _opis_block(opis: Optional[dict], fields: list[str]) -> list[str]:
    facts = [f for f in ((opis or {}).get("fakti") or []) if f.get("pole") in fields]
    if not facts:
        return ["- Опис (агенти): — чека T9 + одобрение"]
    return [f"- {'✅' if f.get('proverka') == 'PROVERENO' else '⚪'} {f.get('vrednost')} — „{f.get('citat')}“ "
            f"([{f.get('objaveno')}]({f.get('izvor_url')}))" for f in facts]


STYLE_MK = {"tempo": "Темпо", "trojki": "Тројки", "napad_preku": "Напад преку", "ofanziven_skok": "Офанзивен скок",
            "rotacija": "Ротација", "vadenje_zvezdi": "Вадење ѕвезди", "fauli": "Фаули", "dozvoluva": "Дозволува"}
NO_AI = "- не се пополнува (Томе 31.10: AI-опис исклучен; стилот = ознаки од бројки)"


def _tags(par: dict, keys: list[str]) -> list[str]:
    stil = _v(par, "stil") or {}
    out = []
    for k in keys:
        x = stil.get(k) or {}
        if x.get("oznaka"):
            flat = lambda v: "/".join(f"{k2} {v2}" for k2, v2 in v.items()) if isinstance(v, dict) else v
            nums = ", ".join(f"{a} {flat(b)}" for a, b in (x.get("brojki") or {}).items() if b is not None)
            out.append(f"- **{STYLE_MK[k]}: {x['oznaka']}** ({nums})"
                       + (" · позиција = прва буква од официјалниот состав (F-C → крило)" if k == "napad_preku" else ""))
    return out


def render_dossier_md(league_mk: str, team: str, b: dict, par: dict, opis: Optional[dict], news: list[dict],
                      coach_official: Optional[str], today: date) -> str:
    tp, o, d = _v(par, "tempo") or {}, _v(par, "ortg") or {}, _v(par, "drtg") or {}
    ps, pd_ = _v(par, "profil_sut") or {}, _v(par, "profil_sut_dozvoluva") or {}
    reb, pf, fta = _v(par, "oreb_strategija") or {}, _v(par, "fauli_tim") or {}, _v(par, "fta_dozvoluva") or {}
    zk, lm, pt = _v(par, "zastitnik_kos") or {}, _v(par, "load_management") or {}, _v(par, "patuvanje") or {}
    rb, rr, rz = _v(par, "rotacija_blizok") or {}, _v(par, "rotacija_razlika") or {}, _v(par, "razlika_vadenje_zvezdi") or {}
    mg = _v(par, "model_greshki") or {}
    f5 = b.get("forma_posledni_5") or {}
    lines = [f"# {team} — {league_mk}", "",
             f"*Ажурирано {today.isoformat()} · БРОЈКИ = Python ({b.get('natprevari')} натпревари до {b.get('do_datum')}) · "
             "**стил** = ознаки од бројките (rang 1 = највисока вредност во лигата, od = број тимови; горна третина / "
             "средна / долна) · ОПИС = само тренер/ГМ (T9), по одобрение од Томе · "
             "✅ = двата модели, ⚪ = еден модел.*", "",
             "## 1. Основно",
             f"- Тренер (официјално: {'box score' if league_mk != 'NBA' else 'nba_api'}): **{coach_official or b.get('trener') or '—'}**",
             *_opis_block(opis, ["trener", "trener_od_koga", "trener_filozofija"]),
             *(_opis_block(opis, ["gm"]) if any(f.get("pole") == "gm" for f in (opis or {}).get("fakti") or []) else ["- ГМ: — (T9, по одобрение)"]), "",
             "## 2. Промени оваа сезона", "- Види `promeni.md` (автоматски од box score + одобрени ТРЕЈД/ТРЕНЕР/ДОЛГА ПОВРЕДА).", "",
             "## 3. Напад",
             f"- ORtg {o.get('prilagoden')} (ранг {o.get('rang')}) · темпо {tp.get('poseda')} поседи (ранг {tp.get('rang')})",
             f"- Шут: 3PAr {ps.get('3par')}% · FTr {ps.get('ftr')} · 2P {ps.get('2p_pct')}% · 3P {ps.get('3p_pct')}% · "
             f"eFG {ps.get('efg')}% (ранг {ps.get('rang_efg')})",
             *_tags(par, ["tempo", "trojki", "napad_preku"]), "",
             "## 4. Одбрана",
             f"- DRtg {d.get('prilagoden')} (ранг {d.get('rang')}) · противникот: 3PAr {pd_.get('3par')}%, FTr {pd_.get('ftr')}, "
             f"2P {pd_.get('2p_pct')}%, 3P {pd_.get('3p_pct')}%, eFG {pd_.get('efg')}% (ранг {pd_.get('rang_efg')})",
             f"- Заштита на кош: {zk.get('blk_po_natprevar')} блокади (ранг {zk.get('rang_blk')}), 2P на противникот "
             f"{zk.get('2p_pct_protivnik')}% (ранг {zk.get('rang_2p_odbrana')})",
             *_tags(par, ["dozvoluva"]), "",
             "## 5. Скокови и фаули",
             f"- OREB% {reb.get('oreb_pct')} (ранг {reb.get('rang_oreb')}) · DREB% {reb.get('dreb_pct')} (ранг {reb.get('rang_dreb')})",
             f"- Фаули {pf.get('pf_po_natprevar')}/нат. (ранг најмалку {pf.get('rang_najmalku')}) · дозволува "
             f"{fta.get('fta_po_natprevar')} слободни/нат. (ранг најмалку {fta.get('rang_najmalku')})",
             *_tags(par, ["ofanziven_skok", "fauli"]), "",
             "## 6. Тренер и ротација",
             f"- Топ-5 минути: близок натпревар {(rb.get('top5') or {}).get('min')} ({(rb.get('top5') or {}).get('n')}) · "
             f"голема разлика {(rr.get('top5') or {}).get('min')} ({(rr.get('top5') or {}).get('n')})",
             f"- Ѕвездите (топ-3): {rz.get('top3_min_blizok')} мин во близок наспроти {rz.get('top3_min_razlika')} при голема "
             f"разлика → вади ги за {rz.get('razlika_min')} мин",
             "- Пропуштени натпревари (топ-5): " + ("; ".join(f"{x['igrac']} {x['propushteni']} ({', '.join(f'{k} {v}' for k, v in x['prichini'].items())})"
                                                      for x in lm.get("top5_propushteni") or []) or "нема"),
             f"- Back-to-back оваа сезона: {pt.get('b2b_sezona')}", *_tags(par, ["rotacija", "vadenje_zvezdi"]), "",
             "## 7. Распоред и контекст",
             f"- Натпревари последни 7 дена: {pt.get('natprevari_posledni_7_dena')} · најдолга гостинска серија: "
             f"{pt.get('najdolga_gostinska_serija')}",
             f"- Дома {b.get('doma', {}).get('poeni')}–{b.get('doma', {}).get('primeni')} ({b.get('doma', {}).get('n')}) · "
             f"гости {b.get('gosti', {}).get('poeni')}–{b.get('gosti', {}).get('primeni')} ({b.get('gosti', {}).get('n')}) · "
             f"форма {f5.get('pobedi')}–{f5.get('porazi')}",
             "",
             "## 8. Мотивација", NO_AI, "",
             "## 9. Пазар", "- Јавен тим / движење на линии: — (треба ≥ 2 недели снимки `linii`)", "",
             "## 10. Што велат аналитичарите оваа сезона", NO_AI, "",
             "## 11. За кладење (DvP — што пушта на противникот, на 36 мин наспроти лигата)", "",
             "| Позиција | поени | ast | скок | тројки |", "|---|---|---|---|---|"]
    for pos, x in (b.get("dvp") or {}).items():
        lines.append(f"| {pos} | {x['poeni_pct']:+.0f}% | {x['ast_pct']:+.0f}% | {x['skok_pct']:+.0f}% | {x['trojki_pct']:+.0f}% |")
    lines += ["", "## 12. Каде греши нашиот модел за овој тим",
              (f"- Одлучени избори: {mg.get('n')} · погодок {mg.get('pogodok_pct')}% наспроти очекувано {mg.get('ochekuvano_pct')}% "
               f"({mg.get('razlika_pp') if mg.get('razlika_pp') is not None else '—'} pp) · по пазар: {mg.get('po_pazar')}" if mg.get("n") else "- Уште нема одлучени избори."), "",
              "## 13. Извори и последно ажурирање",
              f"- Бројки: box score ({b.get('do_datum')}), архива на избори · опис: `predlozi/` → одобрено · вести: `vesti.md` "
              f"({len(news)} од последните 7 дена; сите: `python main.py vesti {team}`)", f"- Ажурирано: {today.isoformat()}", ""]
    lines += ["## Последни вести (7 дена)", ""] + ([f"- {n['datum']} · {n['tip']} · {n.get('igrac') or '—'} · {n['sto']}"
                                                   for n in news[:7]] or ["- нема проверени вести во последните 7 дена"])
    return "\n".join(lines) + "\n"


VESTI_HEADER = ("# Вести — {team}\n\n*Најновите горе. Секој запис: датум | тип | играч | што | влијание | извор | цитат | "
                "модели што се согласиле. Само проверено (2 модела) и само сезона 2026-27.*\n\n"
                "| Датум | Тип | Играч | Што | Влијание | Извор | Цитат | Модели |\n|---|---|---|---|---|---|---|---|\n")


def read_news(folder: Path, days: Optional[int] = None, today: Optional[date] = None) -> list[dict]:
    path = folder / "vesti.md"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\| (\d{4}-\d{2}-\d{2}) \| ([^|]+) \| ([^|]*) \| ([^|]+) \| ([^|]*) \| ([^|]*) \| ([^|]*) \| ([^|]*) \|$", line)
        if m:
            rows.append(dict(zip(["datum", "tip", "igrac", "sto", "vlijanie", "izvor", "citat", "modeli"],
                                 [x.strip() for x in m.groups()])))
    if days is not None and today is not None:
        lim = (pd.Timestamp(today) - pd.Timedelta(days=days)).date().isoformat()
        rows = [r for r in rows if r["datum"] >= lim]
    return sorted(rows, key=lambda r: r["datum"], reverse=True)


def write_dossier(folder: Path, league: str, league_mk: str, team: str, b: dict, t: pd.DataFrame, games: pd.DataFrame,
                  players: pd.DataFrame, today: date, archive: Path, coach_official: Optional[str] = None,
                  stil: Optional[dict] = None) -> None:
    yml = folder / "profil.yaml"
    data = yaml.safe_load(yml.read_text(encoding="utf-8")) if yml.exists() else {}
    data["parametri"] = parametri(team, league, b, t, games, players, today, archive, data, stil)
    yml.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    if not (folder / "vesti.md").exists():
        (folder / "vesti.md").write_text(VESTI_HEADER.format(team=team), encoding="utf-8")
    news = read_news(folder, 7, today)
    (folder / "profil.md").write_text(render_dossier_md(league_mk, team, b, data["parametri"], data.get("opis"), news,
                                                        coach_official, today), encoding="utf-8")
    cards_dir = folder / "igraci"
    cards_dir.mkdir(exist_ok=True)
    for name, text in player_cards(team, games, players, league, today).items():
        (cards_dir / f"{name}.md").write_text(text, encoding="utf-8")


def write_index(root: Path, league: str, league_mk: str, today: date) -> None:
    rows = []
    for folder in sorted(p for p in (root / league).glob("*") if p.is_dir()):
        y = folder / "profil.yaml"
        if not y.exists():
            continue
        d = yaml.safe_load(y.read_text(encoding="utf-8")) or {}
        b = d.get("brojki") or {}
        n_news = len(read_news(folder))
        n_cards = len(list((folder / "igraci").glob("*.md"))) if (folder / "igraci").exists() else 0
        approved = "да" if d.get("opis") else "не"
        rows.append(f"| [{folder.name}]({folder.name}/profil.md) | {b.get('natprevari')} | {b.get('do_datum')} | "
                    f"{d.get('azurirano')} | {n_cards} | {n_news} | {approved} |")
    (root / league / "README.md").write_text(
        f"# {league_mk} — индекс на тимови\n\n*Ажурирано {today.isoformat()} (`python main.py profil`).*\n\n"
        "| Тим | Натпревари | Податоци до | Профил ажуриран | Картички | Вести | Опис одобрен |\n"
        "|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n", encoding="utf-8")
