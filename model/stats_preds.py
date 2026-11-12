"""Статистички предикции за речиси сите активни играчи — БЕЗ квоти (anti-anchoring).

Статистичкиот модел не гледа квоти. Anchoring е забранет.
μ и P(НАД)/P(ПОД) се пресметуваат само од box-score проекции + дисперзија.
Линиите (Mozzart/evrotip) се споредуваат дури во picks (Фаза 7), не овде.

Покривање (Томе 12.11.2026):
- вклучи: EWMA минути ≥ 16 (NBA) / ≥ 14 (ЕЛ/ЕК), не OUT/DOUBTFUL, очекувани мин ≥ 12
- за секој: поени (НАД/ПОД) + еден од ast/reb (поголема дисперзија = помал NB r)
- < 5 натпревари оваа сезона → НИЗОК_CONFIDENCE (сепак се прикажува)
- < 12 очекувани мин или OUT → ODBIENI (не во главниот извештај)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
import pandas as pd

from model.picks import p_count_over
from model.player_model import dispersion_for

# EWMA floor for "active rotation" coverage (Томе 12.11)
EWMA_MIN = {"nba": 16.0, "euroleague": 14.0, "eurocup": 14.0}
SHOW_MIN = 12.0          # expected minutes below this → ODBIENI
LOW_SAMPLE_GAMES = 5
STAT_MK = {"pts": "поени", "ast": "асистенции", "reb": "скокови"}


@dataclass(frozen=True)
class StatPred:
    game_code: str
    team: str
    opp: str
    label: str
    player_code: str
    min_ewm: float
    min_proj: float
    n_games: int
    low_confidence: bool
    pazar: str              # pts / ast / reb
    pazar_mk: str
    mu: float
    line: float             # synthetic x.5 around μ (not a book line)
    p_nad: float
    p_pod: float


def synthetic_line(mu: float) -> float:
    """Classic prop line near the mean: floor(μ) + 0.5 (never a bookmaker line)."""
    if mu <= 0:
        return 0.5
    return float(math.floor(mu) + 0.5)


def season_games(logs: pd.DataFrame, player_code: str, before: str) -> int:
    """Games with minutes > 0 this season before `before` (YYYY-MM-DD)."""
    p = logs[(logs["player_code"].astype(str) == str(player_code))
             & (logs["game_date"] < before)
             & (logs["minutes"].fillna(0) > 0)]
    if "dnp" in p.columns:
        p = p[~p["dnp"].fillna(False).astype(bool)]
    return int(p["game_code"].nunique()) if len(p) else 0


def choose_secondary(mu_ast: float, mu_reb: float, disp_ast, disp_reb) -> str:
    """Pick assists vs rebounds: higher dispersion = smaller NB size r at that μ."""
    ra = float(dispersion_for([mu_ast], disp_ast)[0])
    rr = float(dispersion_for([mu_reb], disp_reb)[0])
    return "ast" if ra <= rr else "reb"


def build_stat_preds(league: str, day_iso: str, players: pd.DataFrame, logs: pd.DataFrame,
                     dispersion: dict) -> tuple[list[StatPred], list[dict]]:
    """From player_projections rows (already without OUT) → coverage preds + ODBIENI notes.

    Does NOT accept or read any odds/lines frame.
    """
    # Статистичкиот модел не гледа квоти. Anchoring е забранет.
    ewma_floor = EWMA_MIN[league]
    shown: list[StatPred] = []
    odbieni: list[dict] = []
    if players is None or players.empty:
        return shown, odbieni

    for row in players.itertuples():
        code = str(row.player_code)
        min_ewm = float(getattr(row, "min_ewm", 0) or 0)
        min_proj = float(getattr(row, "min_bez", 0) or 0)
        label = getattr(row, "label", code)
        team, opp, gc = row.team, row.opp, str(row.game_code)

        if min_proj < SHOW_MIN or min_ewm < ewma_floor:
            odbieni.append({"label": label, "team": team, "opp": opp, "min_ewm": round(min_ewm, 1),
                            "min_proj": round(min_proj, 1),
                            "prichina": ("OUT/надвор од ротација" if getattr(row, "out", False)
                                         else f"мин < праг (EWMA {min_ewm:.1f} / проекција {min_proj:.1f})")})
            continue

        n = season_games(logs, code, day_iso)
        low = n < LOW_SAMPLE_GAMES
        mu_pts = float(row.mu_pts)
        mu_ast = float(row.mu_ast)
        mu_reb = float(row.mu_reb)
        sec = choose_secondary(mu_ast, mu_reb, dispersion.get("ast", 8.0), dispersion.get("reb", 8.0))
        markets = [("pts", mu_pts, dispersion.get("pts", 5.0)),
                   (sec, mu_ast if sec == "ast" else mu_reb,
                    dispersion.get(sec, 8.0))]

        for col, mu, r_spec in markets:
            line = synthetic_line(mu)
            p_nad = p_count_over(mu, r_spec, line)
            shown.append(StatPred(
                game_code=gc, team=team, opp=opp, label=label, player_code=code,
                min_ewm=min_ewm, min_proj=min_proj, n_games=n, low_confidence=low,
                pazar=col, pazar_mk=STAT_MK[col], mu=mu, line=line,
                p_nad=p_nad, p_pod=1.0 - p_nad))
    return shown, odbieni


def format_stat_section(preds: list[StatPred], odbieni: list[dict], league_mk: str) -> list[str]:
    """Markdown block for picks report."""
    out = ["## Сите активни играчи — статистички предикции", "",
           f"*Само модел (μ → синтетичка линија floor(μ)+0.5 → P). **Без Mozzart/evrotip.** "
           f"Прагови: EWMA ≥ {EWMA_MIN.get('nba', 16):.0f} NBA / {EWMA_MIN.get('euroleague', 14):.0f} ЕЛ·ЕК; "
           f"проекција ≥ {SHOW_MIN:.0f} мин. НИЗОК_CONFIDENCE = < {LOW_SAMPLE_GAMES} натпревари оваа сезона.*", ""]
    if not preds:
        out += ["Нема играчи што ги минуваат праговите.", ""]
        return out

    head = ("| Играч | Тим | Мин (EWMA/прој.) | Пазар | μ | Лин. (синт.) | P НАД | P ПОД | Забелешка |\n"
            "|---|---|---|---|---|---|---|---|---|")
    by_game: dict[str, list[StatPred]] = {}
    for p in preds:
        by_game.setdefault(f"{p.team} vs {p.opp}", []).append(p)

    for title, rows in by_game.items():
        out += [f"### {title}", "", head]
        # group by player: pts then secondary
        rows = sorted(rows, key=lambda r: (-r.min_proj, r.label, 0 if r.pazar == "pts" else 1))
        for r in rows:
            flag = "НИЗОК_CONFIDENCE" if r.low_confidence else "—"
            out.append(
                f"| {r.label} | {r.team} | {r.min_ewm:.1f}/{r.min_proj:.1f} | {r.pazar_mk} | "
                f"{r.mu:.1f} | {r.line:g} | {r.p_nad * 100:.0f}% | {r.p_pod * 100:.0f}% | {flag} |")
        out.append("")

    if odbieni:
        out += ["### ODBIENI (не во главниот преглед)", "",
                "| Играч | Тим | EWMA | Прој. мин | Причина |",
                "|---|---|---|---|---|"]
        for o in sorted(odbieni, key=lambda x: (x["team"], -x["min_ewm"]))[:80]:
            out.append(f"| {o['label']} | {o['team']} | {o['min_ewm']} | {o['min_proj']} | {o['prichina']} |")
        if len(odbieni) > 80:
            out.append(f"| … | | | | (+{len(odbieni) - 80} повеќе) |")
        out.append("")
    out.append(f"*Лига: {league_mk}. Овие P не се EV наспроти кладилница — споредбата со квоти е само во делот Избори.*")
    out.append("")
    return out
