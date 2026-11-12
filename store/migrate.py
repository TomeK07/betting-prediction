"""One-off migration of the hand-kept CSVs (data/archive/*.csv) into the SQLite archive.

Cleaning rules: BUILD_PROMPT §6. Nothing is lost: each original row is stored verbatim in
`legacy_json`, notes (BELESKA) go to the `beleski` table, duplicates stay as status DUPLIKAT,
manual corrections come from config/arhiva_korekcii.yaml, and every repair is reported.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from store.legacy_csv import LegacyRow, read_legacy_csv

log = logging.getLogger(__name__)

SEASON_YEAR = 2026            # legacy dates are "27.10 20:15" without a year (season 2026-27)
LEGACY_MODEL_VERSION = "rachno-v0"

LEAGUES = {"nba": "nba", "euroleague": "euroleague", "evroliga": "euroleague", "evrokup": "eurocup"}
SIDES = {"over": "nad", "poveke": "nad", "nad": "nad", "under": "pod", "pomalku": "pod", "pod": "pod"}
PLAYER_MARKETS = {
    "poeni": "poeni", "asistencii": "asistencii", "skokovi": "skokovi", "trojki": "trojki",
    "pra": "pra", "poeni_skok": "poeni+skokovi", "poeni+skok": "poeni+skokovi",
    "skokovi+asistencii": "skokovi+asistencii",
}
TEAM_MARKETS = {"vkupno": "vkupno", "total": "vkupno", "tim. vkupno": "tim_vkupno",
                "hendikep": "hendikep", "1x2": "1x2", "dvoen sans": "dvoen_sans"}
# group / umbrella rows that only ever describe rejected ideas
GROUP_MARKETS = {"", "site pazari", "props", "timski"}
STATUSES = {"DA", "NE", "VOID", "ODBIEN", "POVLECEN", "CEKA"}
COUNTED = {"DA", "NE", "VOID", "CEKA"}   # statuses that are real picks for the duplicate check

# A price is "unverified" when the note says it was orientational, unchecked, or copied from a
# comparison site / US line instead of the bookmaker actually used (decision 29.10: not in ROI).
UNVERIFIED_PRICE_RE = re.compile(r"orient|neprover|^\s*evrotip|espn linija", re.I)  # "Mozzart (evrotip 9.5)" = Mozzart price


def kvota_verified(note: Optional[str]) -> bool:
    return not UNVERIFIED_PRICE_RE.search(note or "")


DATUM_RE = re.compile(r"^\s*(\d{1,2})\.(\d{1,2})\s*~?(\d{1,2}:\d{2})?\s*$")
SIGNED_SIDE_RE = re.compile(r"^(.*?)\s*([+-]\d+(?:\.\d+)?)$")


@dataclass
class Migrated:
    picks: list[dict] = field(default_factory=list)
    notes: list[dict] = field(default_factory=list)
    repairs: list[str] = field(default_factory=list)
    corrections: list[str] = field(default_factory=list)


def _num(text: str) -> Optional[float]:
    text = (text or "").strip().replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def parse_datum(text: str) -> tuple[str, Optional[str]]:
    m = DATUM_RE.match(text or "")
    if not m:
        raise ValueError(f"непознат датум „{text}“")
    day, month, time = m.groups()
    return f"{SEASON_YEAR}-{int(month):02d}-{int(day):02d}", time


def _market(raw: str) -> tuple[str, bool]:
    """'poeni (Mozzart kupeno mbpi)' -> ('poeni', kupeno=True)."""
    key = raw.strip().lower()
    kupeno = "kupeno" in key
    if kupeno:
        key = key.split("(")[0].strip()
    if key in PLAYER_MARKETS:
        return PLAYER_MARKETS[key], kupeno
    if key in TEAM_MARKETS:
        return TEAM_MARKETS[key], kupeno
    if key in GROUP_MARKETS:
        return "grupa", kupeno
    raise ValueError(f"непознат пазар „{raw}“")


def _side_and_line(strana: str, linija: str, market: str) -> tuple[Optional[str], Optional[float], Optional[str]]:
    """Returns (side, line, team_alias). For handicaps the side is the team alias, line signed."""
    s = strana.strip()
    line = _num(linija)
    if s.lower() in SIDES:
        return SIDES[s.lower()], line, None
    if market in ("1x2", "dvoen_sans") or s.upper() in {"1", "2", "X", "1X", "X2", "12"}:
        return s.upper(), line, None
    if not s:
        return None, line, None
    m = SIGNED_SIDE_RE.match(s)
    if m:                                   # "Fener +2.5" -> alias Fener, line +2.5
        return None, float(m.group(2)), m.group(1).strip()
    return None, line, s                    # "PAO" with linija "-13.5"


def migrate_rows(rows: list[LegacyRow]) -> Migrated:
    out = Migrated()
    for r in rows:
        v = r.values
        src = f"{r.source}:{r.line}"
        for rep in r.repairs:
            out.repairs.append(f"{src}: {rep}")
        datum, vreme = parse_datum(v["datum"])
        liga = LEAGUES.get(v["liga"].strip().lower())
        if liga is None:
            raise ValueError(f"{src}: непозната лига „{v['liga']}“")

        status = v["pogodok"].strip().upper()
        rezultat = v["rezultat"].strip()
        if status == "BELESKA":
            out.notes.append({"datum": datum, "vreme": vreme, "liga": liga, "naslov": v["natprevar_ili_igrac"],
                              "tekst": v["zabeleska"], "izvor": src})
            continue
        if status == "" and rezultat.upper() == "CEKA":
            status, rezultat = "CEKA", ""
            out.repairs.append(f"{src}: „CEKA“ беше во колоната rezultat → status CEKA")
        if status not in STATUSES:
            raise ValueError(f"{src}: непознат статус „{v['pogodok']}“")

        market, kupeno = _market(v["pazar"])
        if market == "grupa" and status != "ODBIEN":
            raise ValueError(f"{src}: групен ред со статус {status} — очекував ODBIEN")
        side, line, alias = _side_and_line(v["strana"], v["linija"], market)
        if market == "hendikep" and alias:
            side = alias                    # resolved to a team code by `grade`
        if v["linija"].strip() and line is None:
            out.repairs.append(f"{src}: линија „{v['linija']}“ не е број → само во legacy_json")

        name = v["natprevar_ili_igrac"].strip()
        is_player = market in PLAYER_MARKETS.values()
        p_model = _num(v["p_model"])
        p_pazar = _num(v["p_pazar"])
        out.picks.append({
            "datum": datum, "vreme": vreme, "liga": liga,
            "natprevar": None if is_player else name,
            "tim": v["tim"].strip() or None,
            "igrac": name if is_player else None,
            "pazar": market, "linija": line, "strana": side,
            "kvota": _num(v["kvota"]),
            "kvota_proverena": (None if _num(v["kvota"]) is None else int(kvota_verified(v["zabeleska"]))),
            "uplateno": None,               # unknown for the legacy archive
            "uplateno_iznos": None,
            "p_model": p_model / 100 if p_model is not None else None,
            "p_pazar": p_pazar / 100 if p_pazar is not None else None,
            "status": status,
            "rezultat_tekst": rezultat or None,
            "minuti": v["minuti"].strip() or None,
            "vlezovi_json": json.dumps({"kupeno": True}) if kupeno else None,
            "zabeleska": v["zabeleska"].strip() or None,
            "verzija_model": LEGACY_MODEL_VERSION,
            "izvor": src,
            "legacy_json": json.dumps(v, ensure_ascii=False),
            "duplikat_od": None,
        })
    return out


def apply_corrections(migrated: Migrated, corrections: dict) -> None:
    """Explicit, reviewed corrections (config/arhiva_korekcii.yaml), keyed by '<file>:<line>'."""
    by_src = {p["izvor"]: p for p in migrated.picks}
    for src, corr in (corrections or {}).items():
        pick = by_src.get(src)
        if pick is None:
            raise ValueError(f"корекција за непостоечки запис {src}")
        before = {k: pick.get(k) for k in corr["set"]}
        pick.update(corr["set"])
        pick["zabeleska"] = "; ".join(filter(None, [pick["zabeleska"], f"[КОРЕКЦИЈА 29.10: {corr['prichina']}]"]))
        migrated.corrections.append(f"{src}: {before} → {corr['set']} ({corr['prichina']})")


def _is_kupeno(p: dict) -> bool:
    return bool(p.get("vlezovi_json") and json.loads(p["vlezovi_json"]).get("kupeno"))


def _norm_name(text: Optional[str]) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower().replace(" @ ", " - ")).strip()


def pick_key(p: dict) -> tuple:
    """BUILD_PROMPT §6 unique key."""
    return (p["datum"], p["natprevar"], p["igrac"], p["pazar"], p["linija"], p["strana"])


def mark_duplicates(picks: list[dict]) -> list[dict]:
    """Same pick recorded more than once → count once (decision 29.10). Returns the marked records.

    1. Exact unique key: a real pick beats a rejected one (rejected first, accepted later);
       otherwise the FIRST record wins (p_model is locked when first written).
    2. Same date + match/player + market + side, different line (another bookmaker or a later
       re-write, e.g. Paris–Valencia 186.5/187.5): first counted record wins.
       A bought alternative line ("kupeno") is a SEPARATE pick, never a duplicate.
    """
    marked = []

    def mark(dup: dict, keep: dict, why: str):
        dup["status"] = "DUPLIKAT"
        dup["duplikat_od"] = keep["izvor"]
        dup["zabeleska"] = "; ".join(filter(None, [dup["zabeleska"], f"[DUPLIKAT на {keep['izvor']}: {why}]"]))
        marked.append(dup)

    groups: dict[tuple, list[dict]] = {}
    for p in picks:
        groups.setdefault(pick_key(p), []).append(p)
    for group in groups.values():
        if len(group) > 1:
            real = [p for p in group if p["status"] != "ODBIEN"]
            keep = (real or group)[0]
            for p in group:
                if p is not keep:
                    mark(p, keep, "ист клуч")

    loose: dict[tuple, list[dict]] = {}
    for p in picks:
        if p["status"] in COUNTED and p["pazar"] != "grupa" and not _is_kupeno(p):
            loose.setdefault((p["datum"], _norm_name(p["igrac"] or p["natprevar"]), p["pazar"], p["strana"]),
                             []).append(p)
    for group in loose.values():
        for p in group[1:]:
            mark(p, group[0], f"ист избор, друга линија ({group[0]['linija']} / {p['linija']})")
    return marked


def load_corrections(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get("korekcii") or {}


def load_legacy(folder: Path, corrections: Optional[dict] = None) -> Migrated:
    rows: list[LegacyRow] = []
    for path in sorted(folder.glob("*.csv")):
        rows += read_legacy_csv(path)
    migrated = migrate_rows(rows)
    apply_corrections(migrated, corrections or {})
    mark_duplicates(migrated.picks)
    return migrated


def write_to_db(con, migrated: Migrated) -> int:
    cols = list(migrated.picks[0].keys())
    for p in migrated.picks:
        con.execute(f"INSERT INTO izbori ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                    [p[c] for c in cols])
    for n in migrated.notes:
        con.execute("INSERT INTO beleski (datum, vreme, liga, naslov, tekst, izvor) VALUES (?,?,?,?,?,?)",
                    [n["datum"], n["vreme"], n["liga"], n["naslov"], n["tekst"], n["izvor"]])
    for c in migrated.corrections:
        con.execute("INSERT INTO log_izmeni (akcija, detali) VALUES ('KOREKCIJA_PRI_MIGRACIJA', ?)", [c])
    con.execute("INSERT INTO log_izmeni (akcija, detali) VALUES ('MIGRACIJA', ?)",
                [json.dumps({"izbori": len(migrated.picks), "beleski": len(migrated.notes),
                             "duplikati": [p["izvor"] for p in migrated.picks if p["status"] == "DUPLIKAT"],
                             "popravki": migrated.repairs}, ensure_ascii=False)])
    con.commit()
    return len(migrated.picks)
