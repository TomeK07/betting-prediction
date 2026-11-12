"""Parse the official NBA injury report PDF (BUILD_PROMPT §2.4) into rows.

The PDF is a table without ruling lines: we read words with their x position and assign them to the
column whose header starts left of them. Reasons often wrap onto the line above/below the player
line — those fragments are attached to the nearest player row on the same page.
Team / date / time / matchup cells are only printed on the first row of their block → carried forward.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

from model.logs_common import clean_suffix

COLUMNS = ["game_date", "game_time", "matchup", "team", "player", "status", "reason"]
HEADER_WORDS = {"Date": "game_date", "Time": "game_time", "Matchup": "matchup", "Team": "team",
                "Player": "player", "Current": "status", "Reason": "reason"}
STATUSES = {"Out", "Questionable", "Doubtful", "Probable", "Available"}
NOT_SUBMITTED = "NOT YET SUBMITTED"
PAGE_FOOTER = re.compile(r"^Page\s*\d+\s*of\s*\d+$")


def _columns_from_header(words: list[dict]) -> Optional[dict]:
    """x start of each column, from the header line ('Game Date Game Time Matchup Team Player Name …')."""
    starts = {}
    for i, w in enumerate(words):
        if w["text"] == "Date" and i and words[i - 1]["text"] == "Game":
            starts["game_date"] = words[i - 1]["x0"]
        elif w["text"] == "Time" and i and words[i - 1]["text"] == "Game":
            starts["game_time"] = words[i - 1]["x0"]
        elif w["text"] in ("Matchup", "Team", "Player", "Current", "Reason") and HEADER_WORDS[w["text"]] not in starts:
            starts[HEADER_WORDS[w["text"]]] = w["x0"]
    return starts if len(starts) == 7 else None


def _column_of(x: float, starts: dict) -> str:
    col = "game_date"
    for name, x0 in sorted(starts.items(), key=lambda kv: kv[1]):
        if x >= x0 - 2:
            col = name
    return col


def parse_words(pages: list[list[dict]]) -> pd.DataFrame:
    """`pages` = per page, the pdfplumber words (text, x0, top). Separated from IO for testing."""
    rows, carry = [], {"game_date": None, "game_time": None, "matchup": None, "team": None}
    starts = None
    for words in pages:
        starts = _columns_from_header(words) or starts
        if starts is None:
            continue
        header_top = min((w["top"] for w in words if w["text"] == "Reason"), default=-1)
        by_top: dict[float, list[dict]] = {}
        for w in words:
            if w["top"] <= header_top + 1:
                continue
            by_top.setdefault(round(w["top"], 0), []).append(w)
        lines: dict[float, dict[str, list[str]]] = {}
        for top, ws in by_top.items():
            text = " ".join(x["text"] for x in sorted(ws, key=lambda x: x["x0"]))
            if PAGE_FOOTER.match(text) or text.startswith("Injury Report"):
                continue                                  # page footer / repeated title
            for w in ws:
                lines.setdefault(top, {}).setdefault(_column_of(w["x0"], starts), []).append(w["text"])
        page_rows, fragments = [], []
        for top in sorted(lines):
            cells = {k: " ".join(v) for k, v in lines[top].items()}
            if set(cells) == {"reason"}:
                fragments.append((top, cells["reason"]))
                continue
            for k in carry:
                if cells.get(k):
                    carry[k] = cells[k]
                    if k == "game_date":
                        carry.update(game_time=None, matchup=None, team=None)
                    if k == "matchup":
                        carry["team"] = None
            reason = cells.get("reason", "")
            if not cells.get("player"):
                if NOT_SUBMITTED in reason.replace("  ", " "):
                    page_rows.append({**carry, "top": top, "player": None, "status": "NOT_SUBMITTED", "reason": ""})
                continue
            page_rows.append({**carry, "top": top, "player": cells["player"],
                              "status": cells.get("status", ""), "reason": reason})
        for top, text in fragments:                     # attach wrapped reason parts to the nearest player row
            cand = [r for r in page_rows if r["player"]]
            if cand:
                near = min(cand, key=lambda r: abs(r["top"] - top))
                near.setdefault("_frag", []).append((top, text))
        for r in page_rows:
            parts = [(r["top"], r["reason"])] + r.pop("_frag", [])
            r["reason"] = " ".join(t for _, t in sorted(parts) if t).strip()
            r.pop("top")
            rows.append(r)
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["game_date"] = pd.to_datetime(df["game_date"], format="%m/%d/%Y", errors="coerce").dt.date.astype(str)
    df["reason"] = df["reason"].str.replace(r"\s*-\s*", " - ", regex=True)
    return df


def parse_pdf(path: Path) -> pd.DataFrame:
    import pdfplumber
    with pdfplumber.open(path) as pdf:
        pages = [p.extract_words(keep_blank_chars=False, x_tolerance=1.5) for p in pdf.pages]
        title = pdf.pages[0].extract_text().splitlines()[0] if pdf.pages else ""
    df = parse_words(pages)
    m = re.search(r"(\d{2}/\d{2}/\d{2}\s+\d{1,2}:\d{2}\s*[AP]M)", title)
    df.attrs["report_time"] = m.group(1) if m else None
    return df


def split_name(report_name: str) -> tuple[str, str]:
    """'Porter Jr., Michael' -> ('Porter Jr.', 'Michael'); 'Cardwell, Dylan' -> ('Cardwell', 'Dylan')."""
    if "," in report_name:                       # 'Udeh, Jr., Ernest' (2.11) → ('Udeh Jr.', 'Ernest')
        *last, first = [x.strip() for x in report_name.split(",")]
        return clean_suffix(" ".join(last)), first
    parts = report_name.split()
    return (parts[-1], " ".join(parts[:-1])) if len(parts) > 1 else (report_name, "")
