"""Probability calibration helpers.

Until isotonic calibration has >= 50 picks per league+market (PROCES §3 step 7), a league can get
a temporary shrink toward 50%:  p' = 0.5 + k * (p − 0.5),  k in [0, 1] from config/pravila.yaml.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from core.config import ROOT

RULES_PATH = ROOT / "config" / "pravila.yaml"


def shrink_toward_half(p: float, k: float) -> float:
    """k=1 leaves p unchanged, k=0 turns every probability into 50%."""
    return 0.5 + k * (p - 0.5)


def best_shrink(p: np.ndarray, hit: np.ndarray) -> float:
    """Least-squares (Brier-optimal) k for p' = 0.5 + k(p − 0.5). Not clamped — callers report it raw."""
    x = np.asarray(p, float) - 0.5
    y = np.asarray(hit, float) - 0.5
    denom = float((x * x).sum())
    return float((x * y).sum() / denom) if denom else 1.0


def load_rules(path: Path = RULES_PATH) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get("pravila") or {}


def league_shrink(league: str, rules: Optional[dict] = None) -> Optional[float]:
    """Active (СЛЕДИ or ПОТВРДЕН) shrink factor for a league, if any."""
    rules = load_rules() if rules is None else rules
    for rule in rules.values():
        if rule.get("tip") == "stegnuvanje_kon_50" and rule.get("liga") == league \
                and rule.get("status") in ("СЛЕДИ", "ПОТВРДЕН"):
            return float(rule["k"])
    return None
