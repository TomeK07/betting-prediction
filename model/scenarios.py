"""Scenario flags for BOTH sides (decision 29.10: always look for НАД and ПОД).

Pure functions on already computed projections. They do not change probabilities — they explain them
and let the report show the best НАД and the best ПОД of every game with a reason.
Thresholds come from PROCES §4 / BUILD_PROMPT §5 and are candidates until a backtest confirms them.

Not available from free data: individual defenders ("елитен чувар") — DvP of the position is the proxy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

BLOWOUT_MARGIN = {"nba": 12.0, "euroleague": 10.0, "eurocup": 10.0}   # PROCES §4
DVP_STRONG_PCT = 8.0          # opponent allows ≥ 8% less (or more) than the league at that position
FOULS_PER36_RISK = 4.0        # BUILD_PROMPT §5 rule 10
FOUL_OUT = {"nba": 6, "euroleague": 5, "eurocup": 5}
PACE_SLOW = 2.0               # possessions below league for BOTH teams = "ѕид"


@dataclass
class Scenario:
    side: str                 # "pod" | "nad"
    code: str
    text_mk: str


@dataclass
class PlayerContext:
    league: str
    position: Optional[str]           # G / F / C
    starter: bool
    team_is_favorite: bool
    expected_margin: float            # projected |home − away|
    dvp_pct: Optional[float]          # opponent at this position and stat, % vs league
    team_pace_vs_lg: float            # projected team pace − league pace
    opp_pace_vs_lg: float
    team_drtg_vs_lg: float            # negative = better defence than the league
    opp_drtg_vs_lg: float
    fouls_per36: Optional[float]
    bench_teammate_gain: float        # minutes a same-position teammate is projected to gain (≥ 3 = flag)
    returning_teammate: bool          # a rotation teammate returns → he covered those minutes
    new_teammate_absence: bool        # a rotation teammate is newly out (minutes/usage up)
    first_scorer_of_depleted: bool    # first remaining scorer while a star is out (rule 4: never ПОД)


def player_scenarios(c: PlayerContext) -> list[Scenario]:
    out: list[Scenario] = []
    # ---- ПОД (minus) ----
    if c.dvp_pct is not None and c.dvp_pct <= -DVP_STRONG_PCT:
        out.append(Scenario("pod", "DVP_SILNA", f"противникот затвора позиција {c.position or '?'} ({c.dvp_pct:+.0f}% наспроти лигата)"))
    if c.team_pace_vs_lg <= -PACE_SLOW and c.opp_pace_vs_lg <= -PACE_SLOW and c.team_drtg_vs_lg < 0 and c.opp_drtg_vs_lg < 0:
        out.append(Scenario("pod", "ZID", "двата тима бавни и одбранбени („ѕид“) → помалку поседи"))
    if c.starter and c.team_is_favorite and c.expected_margin >= BLOWOUT_MARGIN[c.league]:
        out.append(Scenario("pod", "BLOWOUT_FAVORIT",
                            f"очекувана голема разлика ({c.expected_margin:.0f}) → стартерите на фаворитот помалку минути"))
    if c.bench_teammate_gain >= 3:
        out.append(Scenario("pod", "KLUPA_ZEMA", f"соиграч од иста позиција добива +{c.bench_teammate_gain:.0f} мин"))
    if c.returning_teammate:
        out.append(Scenario("pod", "SOIGRAC_SE_VRAKJA", "соиграч се враќа → ги враќа минутите што ги покриваше (ПОД наспроти „вжештена“ линија)"))
    if c.fouls_per36 is not None and (c.fouls_per36 >= FOULS_PER36_RISK or (c.position == "C" and FOUL_OUT[c.league] == 5 and c.fouls_per36 >= 3.5)):
        out.append(Scenario("pod", "FAULI", f"ризик од фаули ({c.fouls_per36:.1f} на 36 мин; излегува на {FOUL_OUT[c.league]})"))
    # ---- НАД (plus) ----
    if c.dvp_pct is not None and c.dvp_pct >= DVP_STRONG_PCT:
        out.append(Scenario("nad", "DVP_SLABA", f"противникот пушта на позиција {c.position or '?'} ({c.dvp_pct:+.0f}% наспроти лигата)"))
    if c.team_pace_vs_lg >= PACE_SLOW and c.opp_pace_vs_lg >= PACE_SLOW:
        out.append(Scenario("nad", "BRZO", "двата тима брзи → повеќе поседи"))
    if c.new_teammate_absence:
        out.append(Scenario("nad", "NOVO_OTSUSTVO", "ново отсуство на ротациски соиграч → повеќе минути/топки"))
    if c.first_scorer_of_depleted:
        out.append(Scenario("nad", "PRV_STRELEC", "прв стрелец на осакатен тим (правило 4: никогаш ПОД)"))
    if not c.team_is_favorite and c.expected_margin >= BLOWOUT_MARGIN[c.league] and not c.starter:
        out.append(Scenario("nad", "BLOWOUT_KLUPA", "голема разлика → клупата на губитникот добива повеќе минути"))
    return out


def blocked_sides(scenarios: list[Scenario]) -> set[str]:
    """Hard rules from experience: a ПОД is never allowed on the first scorer of a depleted team (rule 4)."""
    return {"pod"} if any(s.code == "PRV_STRELEC" for s in scenarios) else set()


def team_scenarios(league: str, home_pace_vs_lg: float, away_pace_vs_lg: float, home_drtg_vs_lg: float,
                   away_drtg_vs_lg: float, expected_margin: float) -> list[Scenario]:
    out = []
    if home_pace_vs_lg <= -PACE_SLOW and away_pace_vs_lg <= -PACE_SLOW and home_drtg_vs_lg < 0 and away_drtg_vs_lg < 0:
        out.append(Scenario("pod", "ZID", "ѕид vs ѕид: двата тима бавни и одбранбени → тотал ПОД"))
    if home_pace_vs_lg >= PACE_SLOW and away_pace_vs_lg >= PACE_SLOW and home_drtg_vs_lg > 0 and away_drtg_vs_lg > 0:
        out.append(Scenario("nad", "SUTAUT", "шутаут: двата тима брзи и слаба одбрана → тотал НАД"))
    if expected_margin >= BLOWOUT_MARGIN[league]:
        out.append(Scenario("pod", "BLOWOUT", f"очекувана голема разлика ({expected_margin:.0f}) → ѕвездите седат во 4. четвртина"))
    return out
