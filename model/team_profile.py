"""Team profiles — the NUMBERS part (decision 29.10.2026), rebuilt weekly by `python main.py profil`.

data/timovi/<league>/<team>/
  profil.yaml   structured: `brojki` (this module, overwritten) + `opis` (T9, only after Tome approves — never touched here)
  profil.md     readable for Tome
  promeni.md    change log (date, what changed, source) — appended, never rewritten
  izvori/       cached source texts (T9/T10)
Detected automatically from box scores: new starting five, new coach (EL/EC box score), new player on the sheet.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from model.dvp import dvp_table
from model.team_dossier import style_table, style_tags, team_table, write_dossier, write_index
from model.team_model import ModelParams, compute_ratings, game_frame

POS_MK = {"G": "бекови", "F": "крила", "C": "центри"}


def _r(x, nd=1):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def player_shares(players: pd.DataFrame, team: str) -> pd.DataFrame:
    """Per player: minutes, usage (USG%) and assist rate (AST%) over the season — who runs the offence."""
    p = players[(players["team"] == team) & (players["minutes"].fillna(0) > 0)].copy()
    if p.empty:
        return p
    for c in ("fga", "fta", "tov", "fgm", "ast"):
        p[c] = p[c].fillna(0)
    g = p.groupby("game_code")
    tm_min, tm_fgm = g["minutes"].transform("sum"), g["fgm"].transform("sum")
    tm_poss = g["fga"].transform("sum") + 0.44 * g["fta"].transform("sum") + g["tov"].transform("sum")
    p["usg_num"] = (p["fga"] + 0.44 * p["fta"] + p["tov"]) * (tm_min / 5)
    p["usg_den"] = p["minutes"] * tm_poss
    p["ast_den"] = (p["minutes"] / (tm_min / 5)) * tm_fgm - p["fgm"]
    agg = p.groupby(["player_code", "label"]).agg(
        n=("game_code", "nunique"), minuti=("minutes", "mean"), poeni=("pts", "mean"), ast=("ast", "mean"),
        skok=("reb", "mean"), usg_num=("usg_num", "sum"), usg_den=("usg_den", "sum"), ast_sum=("ast", "sum"),
        ast_den=("ast_den", "sum"), starter=("starter", lambda s: float(pd.Series(s).fillna(False).astype(bool).mean())))
    agg["usg"] = 100 * agg["usg_num"] / agg["usg_den"]
    agg["ast_pct"] = 100 * agg["ast_sum"] / agg["ast_den"].where(agg["ast_den"] > 0)
    return agg.reset_index().sort_values("minuti", ascending=False)


def team_numbers(team: str, games: pd.DataFrame, players: pd.DataFrame, roster: pd.DataFrame,
                 ratings, league_games: pd.DataFrame, before: str) -> dict:
    t = games[games["team"] == team].sort_values(["game_date", "game_code"])
    if t.empty:
        return {}
    rank = lambda col, asc: int(ratings.teams[col].rank(ascending=asc, method="min").get(team, np.nan))
    tr = ratings.teams.loc[team] if team in ratings.teams.index else None
    home, away = t[t["home"]], t[~t["home"]]
    last5 = t.tail(5)
    fga3 = lambda df, p="": float(df[f"{p}fga3"].sum() / df[f"{p}fga"].sum()) if df[f"{p}fga"].sum() else np.nan
    reb = {"oreb_pct": float(t["oreb"].sum() / (t["oreb"].sum() + t["opp_dreb"].sum())),
           "dreb_pct": float(t["dreb"].sum() / (t["dreb"].sum() + t["opp_oreb"].sum()))}
    shares = player_shares(players, team)
    dvp = dvp_table(players, roster, before=before)
    dvp_t = dvp[dvp["tim"] == team].set_index("pozicija")
    lg = league_games
    starters = (players[(players["team"] == team) & players.get("starter", pd.Series(False, index=players.index)).fillna(False).astype(bool)]
                .sort_values(["game_date", "game_code"]))
    last_starters = sorted(starters[starters["game_code"].isin(t.tail(5)["game_code"])]
                           .groupby("label")["game_code"].nunique().sort_values(ascending=False).head(5).index)
    coach = t["coach"].dropna().iloc[-1] if "coach" in t and t["coach"].notna().any() else None
    return {
        "natprevari": int(len(t)), "do_datum": t["game_date"].max(),
        "tempo": {"poseda_po_natprevar": _r(t["pace"].mean()), "rang": rank("pace", False),
                  "liga": _r(lg["pace"].mean())},
        "efikasnost": {
            "ortg_surov": _r(100 * t["pts"].sum() / t["poss"].sum()),
            "drtg_surov": _r(100 * t["opp_pts"].sum() / t["poss"].sum()),
            "ortg_prilagoden": _r(tr["ortg"]) if tr is not None else None,
            "drtg_prilagoden": _r(tr["drtg"]) if tr is not None else None,
            "rang_napad": rank("ortg", False), "rang_odbrana": rank("drtg", True),
            "liga_ortg": _r(ratings.lg_ortg)},
        "dvp": {POS_MK.get(pos, pos): {"poeni_pct": _r(r["pts_vs_liga_pct"]), "ast_pct": _r(r["ast_vs_liga_pct"]),
                                        "skok_pct": _r(r["reb_vs_liga_pct"]), "trojki_pct": _r(r["fgm3_vs_liga_pct"])}
                for pos, r in dvp_t.iterrows()},
        "nositeli": [{"igrac": r.label, "minuti": _r(r.minuti), "usg_pct": _r(r.usg), "ast_pct": _r(r.ast_pct),
                      "poeni": _r(r.poeni), "ast": _r(r.ast)}
                     for r in shares[(shares["minuti"] >= 15) & (shares["n"] >= 3)]
                     .sort_values("usg", ascending=False).head(4).itertuples()] if len(shares) else [],
        "rotacija": [{"igrac": r.label, "minuti": _r(r.minuti), "natprevari": int(r.n), "starter_pct": _r(100 * r.starter, 0)}
                     for r in shares.head(10).itertuples()] if len(shares) else [],
        "startna_petorka_posledni_5": last_starters,
        "trener": coach,
        "doma": {"n": int(len(home)), "poeni": _r(home["pts"].mean()), "primeni": _r(home["opp_pts"].mean())},
        "gosti": {"n": int(len(away)), "poeni": _r(away["pts"].mean()), "primeni": _r(away["opp_pts"].mean())},
        "forma_posledni_5": {"pobedi": int((last5["pts"] > last5["opp_pts"]).sum()), "porazi": int((last5["pts"] < last5["opp_pts"]).sum()),
                             "poeni": _r(last5["pts"].mean()), "primeni": _r(last5["opp_pts"].mean()),
                             "neto_rejting": _r(100 * (last5["pts"].sum() - last5["opp_pts"].sum()) / last5["poss"].sum())},
        "trojki": {"3par_napad": _r(100 * fga3(t), 1), "3par_dozvoluva": _r(100 * fga3(t, "opp_"), 1),
                   "3p_pct": _r(100 * t["fgm3"].sum() / t["fga3"].sum()), "liga_3par": _r(100 * fga3(lg), 1)},
        "skokovi": {k: _r(100 * v) for k, v in reb.items()},
        "igraci_vo_zapisnik": sorted(set(players.loc[players["team"] == team, "label"])),
    }


def detect_changes(old: Optional[dict], new: dict) -> list[str]:
    """Changes visible in box scores since the previous profile (written to promeni.md)."""
    if not old:
        return []
    out = []
    if old.get("trener") and new.get("trener") and old["trener"] != new["trener"]:
        out.append(f"НОВ ТРЕНЕР: {old['trener']} → {new['trener']}")
    o, n = set(old.get("startna_petorka_posledni_5") or []), set(new.get("startna_petorka_posledni_5") or [])
    if o and n and o != n:
        out.append(f"СТАРТНА ПЕТОРКА: излегле {sorted(o - n)}, влегле {sorted(n - o)}")
    new_players = set(new.get("igraci_vo_zapisnik") or []) - set(old.get("igraci_vo_zapisnik") or [])
    if new_players:
        out.append(f"НОВ ИГРАЧ ВО ЗАПИСНИК: {sorted(new_players)}")
    return out


def render_md(league_mk: str, team: str, b: dict, opis: Optional[dict]) -> str:
    e, tp, f = b["efikasnost"], b["tempo"], b["forma_posledni_5"]
    lines = [f"# {team} — {league_mk}", "", f"*БРОЈКИ: Python, {b['natprevari']} натпревари до {b['do_datum']}. "
             "ОПИС: само одобрено од Томе (T9).*", "",
             "## Бројки",
             f"- **Темпо:** {tp['poseda_po_natprevar']} поседи (лига {tp['liga']}), ранг {tp['rang']}.",
             f"- **Напад / одбрана (прилагодени):** ORtg {e['ortg_prilagoden']} (ранг {e['rang_napad']}), "
             f"DRtg {e['drtg_prilagoden']} (ранг {e['rang_odbrana']}); лига {e['liga_ortg']}.",
             f"- **Дома:** {b['doma']['poeni']}–{b['doma']['primeni']} ({b['doma']['n']}) · "
             f"**Гости:** {b['gosti']['poeni']}–{b['gosti']['primeni']} ({b['gosti']['n']}).",
             f"- **Форма (посл. 5):** {f['pobedi']}–{f['porazi']}, {f['poeni']}–{f['primeni']}, нето {f['neto_rejting']}.",
             f"- **Тројки:** 3PAr {b['trojki']['3par_napad']}% (лига {b['trojki']['liga_3par']}%), 3P% {b['trojki']['3p_pct']}; "
             f"противниците 3PAr {b['trojki']['3par_dozvoluva']}%.",
             f"- **Скокови:** OREB% {b['skokovi']['oreb_pct']}, DREB% {b['skokovi']['dreb_pct']}.",
             f"- **Тренер (box score):** {b['trener'] or '—'}",
             "", "### DvP — што пушта / затвора (на 36 мин, наспроти лигата)", "",
             "| Позиција | поени | ast | скок | тројки |", "|---|---|---|---|---|"]
    for pos, d in b["dvp"].items():
        lines.append(f"| {pos} | {d['poeni_pct']:+.0f}% | {d['ast_pct']:+.0f}% | {d['skok_pct']:+.0f}% | {d['trojki_pct']:+.0f}% |")
    lines += ["", "### Кој ја носи топката", "", "| Играч | мин | USG% | AST% | поени | ast |", "|---|---|---|---|---|---|"]
    for n in b["nositeli"]:
        lines.append(f"| {n['igrac']} | {n['minuti']} | {n['usg_pct']} | {n['ast_pct']} | {n['poeni']} | {n['ast']} |")
    lines += ["", "### Ротација", "", "| Играч | мин | натпревари | стартер % |", "|---|---|---|---|"]
    for r in b["rotacija"]:
        lines.append(f"| {r['igrac']} | {r['minuti']} | {r['natprevari']} | {r['starter_pct']} |")
    lines += ["", f"Стартна петорка (најчесто, посл. 5): {', '.join(b['startna_petorka_posledni_5']) or '—'}", "",
              "## Опис (T9 — одобрено од Томе)", ""]
    lines.append(yaml.safe_dump(opis, allow_unicode=True, sort_keys=False) if opis else
                 "*Нема одобрен опис. Ќе дојде од агентите (T9) по инсталација на FreeLLMAPI, со извор + цитат, по одобрение.*")
    return "\n".join(lines)


def write_profile(folder: Path, league_mk: str, team: str, brojki: dict, today: date) -> list[str]:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "izvori").mkdir(exist_ok=True)
    yml = folder / "profil.yaml"
    old = yaml.safe_load(yml.read_text(encoding="utf-8")) if yml.exists() else {}
    changes = detect_changes((old or {}).get("brojki"), brojki)
    data = {"tim": team, "liga": league_mk, "azurirano": today.isoformat(), "brojki": brojki,
            "opis": (old or {}).get("opis")}                       # OPIS is never overwritten here
    yml.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (folder / "profil.md").write_text(render_md(league_mk, team, brojki, data["opis"]), encoding="utf-8")
    log = folder / "promeni.md"
    if not log.exists():
        log.write_text(f"# Промени — {team}\n\n| Датум | Промена | Извор |\n|---|---|---|\n", encoding="utf-8")
    if changes:
        with open(log, "a", encoding="utf-8") as f:
            for c in changes:
                f.write(f"| {today.isoformat()} | {c} | box score (автоматски) |\n")
    return changes


def build_league_profiles(team_logs: pd.DataFrame, player_logs: pd.DataFrame, roster: pd.DataFrame,
                          league: str, league_mk: str, root: Path, today: date,
                          params: Optional[ModelParams] = None) -> dict[str, list[str]]:
    games = game_frame(team_logs, league)
    before = (pd.Timestamp(today) + pd.Timedelta(days=1)).date().isoformat()
    ratings = compute_ratings(games, params or ModelParams(), hca_prior=0.0)
    table = team_table(games)
    styles = style_table(table, games, player_logs, roster)
    archive = root.parents[1] / "archive" / "izbori.db"
    out = {}
    for team in sorted(set(games["team"])):
        b = team_numbers(team, games, player_logs, roster, ratings, games, before)
        if b:
            out[team] = write_profile(root / league / team, league_mk, team, b, today)
            write_dossier(root / league / team, league, league_mk, team, b, table, games, player_logs, today, archive,
                          coach_official=official_coach(root.parents[1], league, team) or b.get("trener"),
                          stil=style_tags(team, styles, b.get("dvp")))
    write_index(root, league, league_mk, today)
    return out


def official_coach(project: Path, league: str, team: str) -> Optional[str]:
    """NBA head coach from the cached nba_api CommonTeamRoster (research/testset_build); EL/EC = box score."""
    if league != "nba":
        return None
    import json
    for path in sorted((project / "data" / "raw" / "nba" / "coaches").glob(f"*_{team}.json"), reverse=True):
        rs = next((r for r in json.loads(path.read_text(encoding="utf-8"))["resultSets"] if r["name"] == "Coaches"), None)
        rows = [dict(zip(rs["headers"], r)) for r in rs["rowSet"]] if rs else []
        head = [r for r in rows if r.get("COACH_TYPE") == "Head Coach"]
        if head:
            return f"{head[0]['FIRST_NAME']} {head[0]['LAST_NAME']}"
    return None
