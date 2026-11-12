"""Read the hand-written legacy CSVs (data/archive/*.csv) and repair known structural damage.

The source files are never modified (archive). Every repair is reported so it can be audited.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

LEGACY_COLUMNS = ["datum", "liga", "natprevar_ili_igrac", "tim", "pazar", "linija", "strana", "kvota",
                  "p_model", "p_pazar", "rezultat", "pogodok", "minuti", "zabeleska"]
N = len(LEGACY_COLUMNS)

# a record's datum glued to the end of the previous record's note: "...Valencia tim.28.10 19:00"
GLUED_DATUM = re.compile(r"(\d{1,2}\.\d{1,2}(?: ~?\d{1,2}:\d{2})?)$")


@dataclass
class LegacyRow:
    source: str
    line: int
    values: dict
    repairs: list[str] = field(default_factory=list)


def read_legacy_csv(path: Path) -> list[LegacyRow]:
    with open(path, encoding="utf-8", newline="") as fh:
        raw = list(csv.reader(fh))
    header, body = raw[0], raw[1:]
    if header != LEGACY_COLUMNS:
        raise ValueError(f"{path.name}: неочекувано заглавје {header}")
    out: list[LegacyRow] = []
    for i, fields in enumerate(body, start=2):
        if not fields or all(not f.strip() for f in fields):
            continue
        if len(fields) == N:
            out.append(LegacyRow(path.name, i, dict(zip(LEGACY_COLUMNS, fields))))
        elif len(fields) == 2 * N - 1:
            out.extend(_split_glued(path.name, i, fields))
        elif len(fields) > N:
            # unquoted comma(s) inside the free-text note: re-join the overflow
            note = ",".join(fields[N - 1:])
            out.append(LegacyRow(path.name, i, dict(zip(LEGACY_COLUMNS, fields[:N - 1] + [note])),
                                 [f"запирка во белешка: {len(fields)} полиња споени во 14"]))
        else:
            raise ValueError(f"{path.name}:{i}: {len(fields)} полиња — не знам да поправам")
    return out


def _split_glued(source: str, line: int, fields: list[str]) -> list[LegacyRow]:
    """27 fields = record A (14) + record B (13) whose datum is stuck to A's note (missing newline)."""
    note_a = fields[N - 1]
    m = GLUED_DATUM.search(note_a)
    if not m:
        raise ValueError(f"{source}:{line}: залепени записи, но не најдов датум на крајот од белешката")
    a = fields[:N - 1] + [note_a[:m.start()]]
    b = [m.group(1)] + fields[N:]
    msg = f"два записи во еден ред (недостасува нов ред); датум „{m.group(1)}“ одвоен од белешката"
    return [LegacyRow(source, line, dict(zip(LEGACY_COLUMNS, a)), [msg]),
            LegacyRow(source, line, dict(zip(LEGACY_COLUMNS, b)), [msg])]
