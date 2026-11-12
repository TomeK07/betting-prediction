"""T10 test set (Tome 31.10): 50 news texts whose correct classification comes from OFFICIAL sources only.

  ПОВРЕДА  25 — the T1 examples (ESPN note / EuroLeague club statement) + the official status (NBA injury report PDF,
               EuroLeague report + box score) → expected: type ПОВРЕДА or СОСТАВ, the player, the effect (OUT / ?/ plays)
  ДОГОВОР / ТРЕЈД 25 — the official NBA Player Movement log (stats.nba.com): Signing / Waive / ContractConverted /
               AwardOnWaivers → ДОГОВОР, Trade → ТРЕЈД. The TEXT is a cached news article that names the player next
               to a transaction word within 5 days of the official date; when there is none, the official line itself.
Nothing in an expected answer comes from an AI. Output: tests/llm/T10.jsonl.
"""
from __future__ import annotations

import glob
import html
import json
import random
import re
from datetime import date, datetime, timedelta

from core.config import Settings
from research.testset_build import OUT_DIR, Fetcher, norm, plain

MOVEMENT = "https://stats.nba.com/js/data/playermovement/NBA_Player_Movement.json"
MOVE_TYPES = {"Trade": "ТРЕЈД", "Signing": "ДОГОВОР", "Waive": "ДОГОВОР", "ContractConverted": "ДОГОВОР",
              "AwardOnWaivers": "ДОГОВОР"}
MOVE_WORDS = {"ТРЕЈД": r"\b(trade[ds]?|acquir\w+|deal(?:t)?|sen[dt])\b",
              "ДОГОВОР": r"\b(sign\w*|waiv\w*|releas\w*|two-way|contract|claim\w*|convert\w*|option)\b"}
INJURY_TYPES = ["ПОВРЕДА", "СОСТАВ"]
N_INJURY, N_MOVES = 25, 25


def _articles(raw) -> list[dict]:
    """Every cached news text we already hold (Eurohoops searches, ESPN stories): {date, url, title, text}."""
    out, seen = [], set()
    for fn in glob.glob(str(raw / "vesti" / "eurohoops" / "search_*.json")):
        for p in json.loads(open(fn, encoding="utf-8").read() or "[]"):
            if p["id"] in seen or "/en/" not in p["link"]:
                continue
            seen.add(p["id"])
            out.append({"date": p["date"][:10], "url": p["link"], "title": html.unescape(p["title"]["rendered"]),
                        "text": plain(p["content"]["rendered"])})
    for fn in glob.glob(str(raw / "vesti" / "espn" / "story_*.json")):
        h = (json.loads(open(fn, encoding="utf-8").read()).get("headlines") or [{}])[0]
        if h.get("story"):
            link = ((h.get("links") or {}).get("web") or {}).get("href") or fn
            out.append({"date": (h.get("published") or "")[:10], "url": link, "title": h.get("headline", ""),
                        "text": plain(h["story"])})
    return out


def _window(text: str, name: str, chars: int = 2500) -> str:
    i = norm(text).find(norm(name))
    if i < 0 or len(text) <= chars:
        return text[:chars]
    start = max(0, i - chars // 3)
    return text[start:start + chars]


def build_t10(settings: Settings, today: date, seed: int = 31) -> dict:
    rnd = random.Random(seed)
    t1 = [json.loads(line) for line in open(OUT_DIR / "T1.jsonl", encoding="utf-8") if line.strip()]
    inj = []
    for ex in t1:
        inj.append({"id": ex["id"].replace("T1", "T10"), "zadaca": "T10", "liga": ex["liga"],
                    "tim": ex["ocekuvano"][0].get("tim"), "tekst": ex["tekst"], "izvor_url": ex["izvor_url"],
                    "objaveno": ex["objaveno"],
                    "ocekuvano": [{"igrac": e["igrac"], "tipovi": INJURY_TYPES, "vlijanie": e["status"]} for e in ex["ocekuvano"]],
                    "vistina": ex["vistina"]})
    nba = [x for x in inj if x["liga"] == "nba"]
    el = [x for x in inj if x["liga"] != "nba"]
    rnd.shuffle(nba)
    inj = el[:10] + nba[:N_INJURY - min(10, len(el))]

    f, raw = Fetcher(settings), settings.paths.raw
    body = f.get(MOVEMENT, raw / "nba" / "player_movement" / f"NBA_Player_Movement_{today}.json")
    rows = json.loads(body)["NBA_Player_Movement"]["rows"] if body else []
    rows = [r for r in rows if r["TRANSACTION_DATE"][:10] >= "2026-07-01" and r["Transaction_Type"] in MOVE_TYPES
            and r.get("PLAYER_SLUG")]                       # trades of picks / cash have no player
    arts = _articles(raw)
    paired, official = [], []
    for r in rows:
        tip = MOVE_TYPES[r["Transaction_Type"]]
        name = " ".join(w.capitalize() for w in r["PLAYER_SLUG"].split("-"))
        d0 = datetime.fromisoformat(r["TRANSACTION_DATE"][:10]).date()
        ex = {"zadaca": "T10", "liga": "nba", "tim": r["TEAM_SLUG"],
              "ocekuvano": [{"igrac": name, "tipovi": [tip], "vlijanie": None}],
              "vistina": f"NBA Player Movement {r['TRANSACTION_DATE'][:10]}: {r['TRANSACTION_DESCRIPTION']}"}
        hit = None
        for a in arts:
            if not a["date"] or not (d0 <= datetime.fromisoformat(a["date"]).date() <= d0 + timedelta(days=5)):
                continue
            t = norm(a["text"])
            i = t.find(norm(name))
            near = t[max(0, i - 300):i + 300 + len(name)]
            # the article must name the transaction AND the official team next to the player (else it is another move)
            if i >= 0 and re.search(MOVE_WORDS[tip], near) and norm(r["TEAM_SLUG"].replace("-", " ")) in near:
                hit = a
                break
        if hit:
            paired.append({**ex, "tekst": _window(hit["text"], name), "izvor_url": hit["url"], "objaveno": hit["date"]})
        else:
            official.append({**ex, "tekst": r["TRANSACTION_DESCRIPTION"], "izvor_url": "https://www.nba.com/players/transactions",
                             "objaveno": r["TRANSACTION_DATE"][:10]})
    trades = [x for x in official if x["ocekuvano"][0]["tipovi"] == ["ТРЕЈД"]]
    rnd.shuffle(official)
    moves = paired[:N_MOVES]
    moves += trades[:max(0, min(5, N_MOVES - len(moves)))]
    moves += [x for x in official if x not in moves][:N_MOVES - len(moves)]
    out = inj + moves
    from research.t10_news import team_names
    names = {lg: team_names(lg) for lg in ("euroleague", "eurocup", "nba")}
    for i, ex in enumerate(out, 1):
        ex["id"] = f"T10-{i:03d}"
        t = ex.get("tim") or ""
        if t in names.get(ex["liga"], {}):                           # EL code → name
            t = names[ex["liga"]][t]["ime"]
        elif ex["liga"] == "nba" and t.islower():                     # NBA slug ('lakers') → 'Los Angeles Lakers'
            t = next((v["ime"] for v in names["nba"].values() if t in norm(v["ime"]).replace(" ", "")), t.title())
        ex["tim"] = t
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "T10.jsonl", "w", encoding="utf-8") as fh:
        for ex in out:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return {"vkupno": len(out), "povredi": len(inj), "dogovori_so_napis": min(len(paired), N_MOVES),
            "dogovori_oficijalen_tekst": len(moves) - min(len(paired), N_MOVES),
            "trejdovi": sum(x["ocekuvano"][0]["tipovi"] == ["ТРЕЈД"] for x in moves)}
