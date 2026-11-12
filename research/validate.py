"""Validation of LLM output (LLM_ISTRAZUVACI §1, §3). Everything an LLM returns is data to be checked."""
from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timedelta
from typing import Optional

T1_STATUSES = {"OUT", "DOUBTFUL", "QUESTIONABLE", "PROBABLE", "AVAILABLE", "RETURN"}
# §3.1: an answer carrying any of these is rejected as a whole (no probabilities, projections, picks)
FORBIDDEN_KEYS = {"verojatnost", "probability", "proekcija", "projection", "preporaka", "recommendation",
                  "izbor", "pick", "tiket", "ticket", "kvota_predlog", "odds_suggestion", "confidence"}
MAX_AGE = timedelta(hours=24)


def _norm_ws(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    return re.sub(r"\s+", " ", text).strip().lower()


def quote_in_text(quote: str, text: str) -> bool:
    """§1: the quote must literally exist in the downloaded text (whitespace/case-insensitive)."""
    q = _norm_ws(quote)
    return len(q) >= 8 and q in _norm_ws(text)


def has_forbidden(obj) -> bool:
    if isinstance(obj, dict):
        return any(k.lower() in FORBIDDEN_KEYS or has_forbidden(v) for k, v in obj.items())
    if isinstance(obj, list):
        return any(has_forbidden(x) for x in obj)
    return False


def validate_t1(answer: dict, text: str, source_url: str, published: Optional[datetime],
                now: datetime) -> list[dict]:
    """Each item gets `ok` and `reason`. Rejects the whole answer if it contains forbidden fields."""
    if has_forbidden(answer):
        return [{"ok": False, "reason": "забрането поле (веројатност/избор/…) — целиот одговор отфрлен"}]
    out = []
    stale = published is not None and now - published > MAX_AGE
    for it in answer.get("items", []) if isinstance(answer, dict) else []:
        status = str(it.get("status", "")).upper()
        problems = []
        if not it.get("igrac") or not it.get("tim"):
            problems.append("недостасува играч/тим")
        if status not in T1_STATUSES:
            problems.append(f"непознат статус „{status}“")
        if not it.get("citat") or not quote_in_text(it["citat"], text):
            problems.append("цитатот го нема во текстот (измислено?)")
        out.append({**it, "status": status, "izvor_url": source_url,
                    "objaveno": published.isoformat() if published else None,
                    "stara": bool(stale), "ok": not problems, "reason": "; ".join(problems)})
    return out


def agree(a: list[dict], b: list[dict]) -> list[dict]:
    """§1.5: two independent models; only items both report with the same player+team+status are PROVERENO."""
    key = lambda r: (_norm_ws(r.get("igrac", "")), _norm_ws(r.get("tim", "")), r.get("status"))
    kb = {key(r) for r in b if r.get("ok")}
    return [{**r, "proverka": "PROVERENO" if key(r) in kb else "NEPROVERENO"} for r in a if r.get("ok")]
