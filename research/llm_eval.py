"""Quality control of the free LLMs (LLM_ISTRAZUVACI §6): every model × every test example → archive/llm_log.db.

Scoring is against the OFFICIAL answer stored in tests/llm/*.jsonl (built by research/testset_build.py):
- T1: for each expected player — the model must report him with the official status and a real quote.
- T9: the head coach it names must be the official one; every key player it names must be on the official roster.
  Other T9 fields (style, defence, analysts) cannot be checked officially → only their quotes are checked.
- An invented quote (not in the text) counts in `izmisleni` for every field.
Thresholds: T1 ≥ 95% (§6), invented quotes < 5% (§6), answered ≥ 90% of examples;
T9 ≥ 90% — PROPOSED (not in §6 yet, same as T2), waits for Tome.
"""
from __future__ import annotations

import json
import re
import sqlite3
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Optional

from core.config import ROOT
from research.llm_client import LLMClient
from research.tasks import t1_povredi, t9_profil
from research.testset_build import OUT_DIR, norm, surname
from research.validate import has_forbidden, quote_in_text, validate_t1

LOG_DB = ROOT / "archive" / "llm_log.db"
THRESHOLDS = {"T1": 0.95, "T9": 0.90, "T10": 0.90}      # T9 and T10 = PROPOSED (not in §6), wait for Tome
MAX_INVENTED = 0.05
MIN_ANSWERED = 0.90
NEAR = [{"OUT", "DOUBTFUL"}, {"AVAILABLE", "PROBABLE", "RETURN"}]
# names in the FreeLLMAPI catalogue that are routers, aliases or safety/vision models, not extractors
NOT_EXTRACTORS = ("auto", "fusion", "claude-", "kilo-", "free-router", "-guard", "safety", "safeguard",
                  "robotics", "moondream", "gemini-flash", "gemini-flash-lite", "allam")

SCHEMA = """CREATE TABLE IF NOT EXISTS llm_test (
  id INTEGER PRIMARY KEY AUTOINCREMENT, vreme TEXT, model TEXT, provajder TEXT, zadaca TEXT, primer_id TEXT,
  prompt_verzija TEXT, sekundi REAL, od_kesh INTEGER, odgovoril INTEGER, greska TEXT,
  stavki INTEGER, izmisleni INTEGER, tochni INTEGER, vkupno INTEGER, blisku INTEGER, propushteni INTEGER, detali TEXT)"""


def load_examples(task: str, folder: Path = OUT_DIR) -> list[dict]:
    with open(folder / f"{task}.jsonl", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ---------- scoring (pure) ----------

def score_t1(answer: dict, ex: dict) -> dict:
    items = validate_t1(answer, ex["tekst"], ex["izvor_url"], None, datetime.now())
    invented = sum(1 for it in items if "цитатот" in it.get("reason", ""))
    correct = near = 0
    detail = []
    for e in ex["ocekuvano"]:
        hit = next((it for it in items if it.get("igrac") and surname(it["igrac"]) == surname(e["igrac"])), None)
        got = hit["status"] if hit else None
        ok = bool(hit and hit["ok"] and got == e["status"])
        close = bool(hit and hit["ok"] and not ok and any({got, e["status"]} <= g for g in NEAR))
        correct += ok
        near += close
        detail.append({"igrac": e["igrac"], "ocekuvano": e["status"], "dobieno": got,
                       "citat_ok": bool(hit and hit["ok"]), "tochno": ok})
    return {"stavki": len(items), "izmisleni": invented, "tochni": correct, "vkupno": len(ex["ocekuvano"]),
            "blisku": near, "propushteni": sum(1 for d in detail if d["dobieno"] is None), "detali": detail}


def _quotes(obj) -> list[str]:
    if isinstance(obj, dict):
        return ([obj["citat"]] if isinstance(obj.get("citat"), str) else []) + [q for v in obj.values() for q in _quotes(v)]
    if isinstance(obj, list):
        return [q for x in obj for q in _quotes(x)]
    return []


def score_t9(answer: dict, ex: dict) -> dict:
    text, exp = ex["tekst"], ex["ocekuvano"]
    if not isinstance(answer, dict) or has_forbidden(answer):
        return {"stavki": 1, "izmisleni": 0, "tochni": 0, "vkupno": 1, "blisku": 0, "propushteni": 0,
                "detali": [{"pole": "*", "zabeleska": "забрането поле или не е објект — целиот одговор отфрлен"}]}
    quotes = _quotes(answer)
    invented = sum(1 for q in quotes if not quote_in_text(q, text))
    detail, correct, total, missed = [], 0, 0, 0
    tr = answer.get("trener") if isinstance(answer.get("trener"), dict) else {}
    name = (tr or {}).get("ime")
    if name:
        ok = bool(exp["trener"]) and surname(name) == surname(exp["trener"]) and quote_in_text(tr.get("citat", ""), text)
        total += 1
        correct += ok
        detail.append({"pole": "trener", "dobieno": name, "oficijalno": exp["trener"], "tochno": ok})
    elif exp["trener"] and surname(exp["trener"]) in norm(text).split():
        missed += 1                          # the official coach is named in the text but the model left him out
    roster = {surname(p) for p in exp["sostav"]}
    for p in answer.get("klucni_igraci") or []:
        if not isinstance(p, dict) or not p.get("igrac"):
            continue
        ok = surname(p["igrac"]) in roster and quote_in_text(p.get("citat", ""), text)
        total += 1
        correct += ok
        detail.append({"pole": "igrac", "dobieno": p["igrac"], "vo_sostav": surname(p["igrac"]) in roster, "tochno": ok})
    gm = answer.get("gm") if isinstance(answer.get("gm"), dict) else None      # t9-v2: a player is never the GM
    if gm and gm.get("ime"):
        ok = surname(gm["ime"]) not in roster and quote_in_text(gm.get("citat", ""), text)
        total += 1
        correct += ok
        detail.append({"pole": "gm", "dobieno": gm["ime"], "tochno": ok})
    return {"stavki": len(quotes), "izmisleni": invented, "tochni": correct, "vkupno": total, "blisku": 0,
            "propushteni": missed, "detali": detail}


# T10 effect of an injury item, read from its own words (sto + vlijanie) → compared with the official status group
EFFECT = [("QUESTIONABLE", r"\b(questionable|game-time|day-to-day|uncertain|da valutare|incert[oa])\b"),
          ("OUT", r"\b(out|miss(es|ed)?|ruled out|sidelined|not play|won't play|will not play|unavailable|absent\w*|"
                  r"doubtful|not (be )?on the roster|surgery)\b"),
          ("AVAILABLE", r"\b(available|return(s|ed|ing)?|cleared|will play|active|probable|on the roster|full roster)\b")]
EFFECT_GROUP = {"OUT": "OUT", "DOUBTFUL": "OUT", "QUESTIONABLE": "QUESTIONABLE", "AVAILABLE": "AVAILABLE",
                "PROBABLE": "AVAILABLE", "RETURN": "AVAILABLE"}


def effect_of(text: str) -> Optional[str]:
    for group, rx in EFFECT:
        if re.search(rx, text or "", re.I):
            return group
    return None


def score_t10(answer: dict, ex: dict) -> dict:
    """Per expected player: the right type (official), the right player, and — for injuries — the right effect.
    tochni = all three; blisku = type + player right but the effect missing / wrong."""
    from research.tasks import t10_vesti
    text = ex["tekst"]
    if not isinstance(answer, dict) or has_forbidden(answer):
        return {"stavki": 1, "izmisleni": 0, "tochni": 0, "vkupno": len(ex["ocekuvano"]), "blisku": 0, "propushteni": 0,
                "detali": [{"zabeleska": "забрането поле или не е објект"}]}
    items = [it for it in answer.get("items") or [] if isinstance(it, dict)]
    invented = sum(1 for it in items if not it.get("citat") or not quote_in_text(it["citat"], text))
    correct = near = missed = 0
    detail = []
    for e in ex["ocekuvano"]:
        mine = [it for it in items if it.get("igrac") and surname(it["igrac"]) == surname(e["igrac"])
                and it.get("citat") and quote_in_text(it["citat"], text)]
        if not mine:
            missed += 1
            detail.append({"igrac": e["igrac"], "ocekuvano": e["tipovi"], "dobieno": None})
            continue
        types = [t10_vesti.TYPES.get(str(it.get("tip", "")).upper()) for it in mine]
        type_ok = any(t in e["tipovi"] for t in types)
        eff = None
        if e.get("vlijanie"):
            got = [effect_of(f"{it.get('sto', '')} {it.get('vlijanie', '')}") for it in mine
                   if t10_vesti.TYPES.get(str(it.get("tip", "")).upper()) in e["tipovi"]]
            eff = EFFECT_GROUP.get(e["vlijanie"]) in got
        ok = type_ok and (eff is None or eff)
        correct += ok
        near += bool(type_ok and eff is False)
        detail.append({"igrac": e["igrac"], "ocekuvano": e["tipovi"], "dobieno": types, "vlijanie_ok": eff, "tochno": ok})
    return {"stavki": len(items), "izmisleni": invented, "tochni": correct, "vkupno": len(ex["ocekuvano"]),
            "blisku": near, "propushteni": missed, "detali": detail}


# ---------- running ----------

def _prompt(task: str, ex: dict) -> tuple[str, str, str]:
    if task == "T1":
        user = f"SOURCE_URL: {ex['izvor_url']}   PUBLISHED: {ex['objaveno']}\nTEXT:\n{ex['tekst']}"
        return t1_povredi.SYSTEM_PROMPT, user, t1_povredi.PROMPT_VERSION
    if task == "T10":
        from research.tasks import t10_vesti
        from research.t10_news import USE_EXCERPT, excerpt, excerpt_keys
        text = excerpt(ex["tekst"], excerpt_keys(ex["liga"], ex["tim"])) if USE_EXCERPT else ex["tekst"]
        user = t10_vesti.user_text(ex["tim"], ex["liga"], ex["izvor_url"], ex["objaveno"], text)
        return t10_vesti.SYSTEM_PROMPT, user, t10_vesti.EXCERPT_VERSION if text != ex["tekst"] else t10_vesti.PROMPT_VERSION
    user = t9_profil.user_text(ex["tim"], ex["liga"], ex["izvor_url"], ex["objaveno"], ex["tekst"])
    return t9_profil.SYSTEM_PROMPT, user, t9_profil.PROMPT_VERSION


def candidate_models(base_url: Optional[str] = None, api_key: Optional[str] = None) -> list[str]:
    """Models FreeLLMAPI marks `available` (the 4 configured providers), without routers/aliases/safety models."""
    import os
    import urllib.request
    LLMClient("probe", base_url, api_key)            # loads .env, raises LLMDisabled if not configured
    req = urllib.request.Request(os.environ["LLM_BASE_URL"].rstrip("/") + "/models",
                                 headers={"Authorization": f"Bearer {os.environ['LLM_API_KEY']}"})
    data = json.load(urllib.request.urlopen(req, timeout=20))["data"]
    return [m["id"] for m in data if m.get("available")
            and not any(m["id"] == x or m["id"].startswith(x) or x in m["id"] for x in NOT_EXTRACTORS)]


def run_model(model: str, tasks: list[str], examples: dict, db: Path, lock: threading.Lock,
              skip_providers: tuple = ("kilo",), pause_s: float = 6.0) -> None:
    """One model, examples in order; `pause_s` between real calls keeps it under the free per-minute limits."""
    client = LLMClient(model, timeout_s=90, retries=3)
    con = sqlite3.connect(db)
    for task in tasks:
        for ex in examples[task]:
            system, user, version = _prompt(task, ex)
            meta, err, sc = {"provajder": None, "sekundi": None, "od_kesh": False}, None, None
            try:
                ans, meta = client.ask_json_meta(system, user, version)
                if not meta.get("od_kesh"):
                    time.sleep(pause_s)
                if meta.get("provajder") and meta["provajder"].split("/")[0] in skip_providers:
                    err = f"провајдер {meta['provajder']} е исклучен"
                else:
                    sc = {"T1": score_t1, "T9": score_t9, "T10": score_t10}[task](ans, ex)
            except Exception as exc:                  # provider down / limit / invalid JSON → "no answer"
                err = str(exc)[:300]
            sc = sc or {"stavki": 0, "izmisleni": 0, "tochni": 0, "vkupno": 0, "blisku": 0, "propushteni": 0, "detali": []}
            with lock:
                con.execute("INSERT INTO llm_test (vreme, model, provajder, zadaca, primer_id, prompt_verzija, sekundi, "
                            "od_kesh, odgovoril, greska, stavki, izmisleni, tochni, vkupno, blisku, propushteni, detali) "
                            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            (datetime.now().isoformat(timespec="seconds"), model, meta.get("provajder"), task, ex["id"],
                             version, meta.get("sekundi"), int(bool(meta.get("od_kesh"))), int(err is None), err,
                             sc["stavki"], sc["izmisleni"], sc["tochni"], sc["vkupno"], sc["blisku"], sc["propushteni"],
                             json.dumps(sc["detali"], ensure_ascii=False)))
                con.commit()
            if err and ("No candidate model" in err or "not found" in err.lower()):
                break                                 # model not served at all → do not burn the rest
    con.close()


def run_eval(models: list[str], tasks: list[str], db: Path = LOG_DB, run_tag: Optional[str] = None) -> str:
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db)
    con.execute(SCHEMA)
    con.commit()
    con.close()
    examples = {t: load_examples(t) for t in tasks}
    start = datetime.now().isoformat(timespec="seconds")
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=min(8, len(models))) as pool:
        for fut in [pool.submit(run_model, m, tasks, examples, db, lock) for m in models]:
            fut.result()
    return summary(db, since=start)


def summary(db: Path = LOG_DB, since: Optional[str] = None) -> str:
    con = sqlite3.connect(db)
    q = "SELECT model, zadaca, provajder, sekundi, odgovoril, stavki, izmisleni, tochni, vkupno, blisku, propushteni FROM llm_test"
    rows = con.execute(q + (" WHERE vreme >= ?" if since else ""), (since,) if since else ()).fetchall()
    con.close()
    agg: dict = {}
    for m, t, prov, sec, ans, st, inv, ok, tot, near, miss in rows:
        a = agg.setdefault((m, t), {"n": 0, "ans": 0, "st": 0, "inv": 0, "ok": 0, "tot": 0, "near": 0, "miss": 0,
                                     "sec": [], "prov": set()})
        a["n"] += 1
        a["ans"] += ans
        a["st"] += st
        a["inv"] += inv
        a["ok"] += ok
        a["tot"] += tot
        a["near"] += near
        a["miss"] += miss
        if sec:
            a["sec"].append(sec)
        if prov:
            a["prov"].add(prov.split("/")[0])
    out = ["| Модел | Задача | Провајдер | одговори | точност | (блиску) | измислени цитати | пропуштени | сек (медијана) | Поминува |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    passed = {}
    for (m, t), a in sorted(agg.items(), key=lambda kv: (kv[0][1], -(kv[1]["ok"] / kv[1]["tot"] if kv[1]["tot"] else 0))):
        acc = a["ok"] / a["tot"] if a["tot"] else 0.0
        inv = a["inv"] / a["st"] if a["st"] else 0.0
        answered = a["ans"] / a["n"] if a["n"] else 0.0
        ok = acc >= THRESHOLDS[t] and inv < MAX_INVENTED and answered >= MIN_ANSWERED
        if ok:
            passed.setdefault(t, []).append(m)
        out.append(f"| {m} | {t} | {', '.join(sorted(a['prov'])) or '—'} | {a['ans']}/{a['n']} | {acc * 100:.1f}% "
                   f"({a['ok']}/{a['tot']}) | {a['near']} | {inv * 100:.1f}% ({a['inv']}/{a['st']}) | {a['miss']} | "
                   f"{statistics.median(a['sec']) if a['sec'] else 0:.1f} | {'✅' if ok else '❌'} |")
    out += ["", f"Прагови: T1 ≥ {THRESHOLDS['T1']:.0%}, T9 ≥ {THRESHOLDS['T9']:.0%} и T10 ≥ {THRESHOLDS['T10']:.0%} (предлог), измислени цитати "
            f"< {MAX_INVENTED:.0%}, одговорени ≥ {MIN_ANSWERED:.0%} од примерите.",
            "Поминуваат: " + "; ".join(f"{t}: {', '.join(v)}" for t, v in sorted(passed.items())) if passed else "Поминуваат: никој"]
    return "\n".join(out)
