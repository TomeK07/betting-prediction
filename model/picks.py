"""Phase 7 — from projections to graded candidates (pure functions, no network, no archive).

Order (PROCES §3, anti-anchoring): the model's projection exists BEFORE any line is read. Only then:
P(over/under) at the bookmaker's line → calibration → de-vig market probability → EV → named filters.
Always BOTH sides (decision 29.10). Nothing is staked automatically (paper mode by default).
"""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats as st

from model.calibration import shrink_toward_half
from model.lines import devig_two_way
from model.player_model import dispersion_for, p_over

MIN_EV = 0.04                      # PROCES §1
ALARM_PP = 0.15                    # |p − p_market| > 15 pp → ПРОВЕРИ_ПОДАТОЦИ
ALARM_EV = 0.15                    # EV > +15% → ПРОВЕРИ_ПОДАТОЦИ (Tome 29.10: too good = usually our error)
MIN_MINUTES = {"nba": 24.0, "euroleague": 20.0, "eurocup": 20.0}   # rule 13
ELEC_OVER_CAP = 0.65               # backtest 29.10: EL/EC points НАД above 65% is overconfident
NBA_TEAM_MARKETS = {"hendikep", "vkupno", "tim_vkupno", "1x2"}   # Phase 4 backtest: no edge → paper only (Tome 30.10)
NBA_PTS_OVER_MIN_EV = 0.08         # Tome 30.10: NBA points НАД only with EV ≥ 8%
LOW_LINE_MAX = {"trojki": 1.5, "skokovi": 2.5, "asistencii": 2.5}   # Tome 30.10: low lines out of the ticket
PAPER_ONLY = "ЗНАК: САМО_ХАРТИЈА"
PLAYER_PROPS_PAPER = True          # Tome 31.10: archive 743 props — model log loss 0.754 vs market 0.691 → paper only


@dataclass
class Candidate:
    liga: str
    game_code: str
    natprevar: str
    tim: Optional[str]
    protivnik: Optional[str]
    igrac: Optional[str]
    player_code: Optional[str]
    pazar: str
    strana: str
    linija: float
    mu: float
    p_model: float
    p_kal: float = 0.0
    kvota: Optional[float] = None
    kladilnica: Optional[str] = None
    p_pazar: Optional[float] = None
    ev: Optional[float] = None
    kvoti: dict = field(default_factory=dict)        # {book: (odds this side, odds other side)} at this line
    nat_ids: dict = field(default_factory=dict)      # {book: bookmaker match id} — to find the closing price
    pravila: list = field(default_factory=list)      # rules that rejected / flagged it
    scenarija: list = field(default_factory=list)
    min_proj: Optional[float] = None
    zavisi: bool = False

    @property
    def odbien(self) -> bool:
        return any(not r.startswith("ЗНАК:") for r in self.pravila)

    @property
    def samo_hartija(self) -> bool:
        return any(r.startswith(PAPER_ONLY) for r in self.pravila)


# ---------- probabilities ----------

def p_normal_over(mu: float, sigma: float, line: float) -> float:
    return float(1 - st.norm.cdf((line - mu) / sigma))


def p_handicap_cover(mu_margin: float, sigma: float, team_line: float) -> float:
    """P(team margin + line > 0); team_line is from the team's own perspective (−6.5 = favourite)."""
    return float(1 - st.norm.cdf((-team_line - mu_margin) / sigma))


def p_count_over(mu: float, r, line: float) -> float:
    """r: a number or the per-mean spec from settings.yaml ({"r_po_mu": [...]}, low-line backtest 29.10)."""
    m = np.array([mu])
    return float(p_over(m, np.array([line]), dispersion_for(m, r))[0])


# ---------- market ----------

def price(c: Candidate, book: str, k_side: float, k_other: Optional[float], nat_id: Optional[str] = None) -> None:
    """Attach the book's price; keep the best EV across books. nat_id = the book's own match id (for CLV)."""
    c.kvoti[book] = (k_side, k_other)
    if nat_id:
        c.nat_ids[book] = str(nat_id)
    ev = c.p_kal * k_side - 1
    if c.ev is None or ev > c.ev:
        c.ev, c.kvota, c.kladilnica = ev, k_side, book
        c.p_pazar = devig_two_way(k_side, k_other) if k_other else None


def kelly_fraction(p: float, k: float, fraction: float = 0.25, cap: float = 0.02) -> float:
    """¼ Kelly, capped at 2% of the bankroll (PROCES §5)."""
    if k <= 1:
        return 0.0
    f = (p * k - 1) / (k - 1)
    return max(0.0, min(cap, fraction * f))


# ---------- filters ----------

def calibrate(c: Candidate, league_shrink: Optional[float]) -> None:
    """PROCES §3 step 7 — before the market is read. Temporary: EL shrink toward 50% (k from pravila.yaml)."""
    c.p_kal = shrink_toward_half(c.p_model, league_shrink) if league_shrink is not None else c.p_model


def apply_filters(c: Candidate, league: str, blocked_pod: bool) -> None:
    if c.ev is None:
        c.pravila.append("ЛИНИЈА_НЕ_ПОСТОИ (правило 16)")
        return
    if c.ev < MIN_EV:
        c.pravila.append(f"EV_ПОД_4% ({c.ev * 100:+.1f}%)")
    if c.p_pazar is not None and abs(c.p_kal - c.p_pazar) > ALARM_PP:
        c.pravila.append(f"ПРОВЕРИ_ПОДАТОЦИ (модел {c.p_kal * 100:.0f}% vs пазар {c.p_pazar * 100:.0f}%)")
    if c.ev > ALARM_EV:
        c.pravila.append(f"ПРОВЕРИ_ПОДАТОЦИ (EV {c.ev * 100:+.1f}% > +15% — премногу добро)")
    if c.igrac:
        if c.min_proj is not None and c.min_proj < MIN_MINUTES[league]:
            c.pravila.append(f"МИНУТИ < {MIN_MINUTES[league]:.0f} (правило 13: {c.min_proj:.1f})")
        if c.strana == "pod" and blocked_pod:
            c.pravila.append("ПРВ_СТРЕЛЕЦ_ОСАКАТЕН — никогаш ПОД (правило 4)")
        if c.pazar == "poeni" and c.strana == "nad" and c.mu < 1.2 * c.linija:
            c.pravila.append("ПОЕНИ_НАД: проекција < линија + 20% (правило 6)")
        if c.pazar == "poeni" and c.strana == "nad" and league in ("euroleague", "eurocup") and c.p_kal > ELEC_OVER_CAP:
            c.pravila.append("ЕЛ/ЕК ПОЕНИ НАД > 65% — моделот претерува (backtest 29.10)")
        if c.pazar == "poeni" and c.strana == "nad" and league == "nba" and c.ev < NBA_PTS_OVER_MIN_EV:
            c.pravila.append(f"NBA ПОЕНИ НАД: EV {c.ev * 100:+.1f}% < 8% (Томе 30.10)")
        if c.pazar in LOW_LINE_MAX and c.linija <= LOW_LINE_MAX[c.pazar]:
            c.pravila.append(f"ПРОВЕРИ_ПОДАТОЦИ (ниска линија {c.linija:g} — не во тикет, Томе 30.10)")
        if PLAYER_PROPS_PAPER:
            c.pravila.append(f"{PAPER_ONLY} (играчки пазар — моделот засега полош од пазарот, Томе 31.10)")
    if league == "nba" and not c.igrac and c.pazar in NBA_TEAM_MARKETS:
        c.pravila.append(f"{PAPER_ONLY} (NBA главен пазар — нема предност, backtest Фаза 4)")
    if c.zavisi:
        c.pravila.append("ЗНАК: ЗАВИСИ_ОД_СОСТАВ")


# ---------- ticket ----------

def build_ticket(cands: list[Candidate], league_counts: Optional[dict] = None, max_picks: int = 3) -> list[Candidate]:
    """2–3 picks, different games, no opposite directions from the same team (rule 9), max 1 points НАД
    (rule 6), no ЗАВИСИ_ОД_СОСТАВ / ПРОВЕРИ_ПОДАТОЦИ, EuroCup max 2 per night (rule 14)."""
    ok = [c for c in cands if not c.odbien and not c.zavisi and not c.samo_hartija]
    ok.sort(key=lambda c: c.ev, reverse=True)
    ticket, games, team_dir, over_pts, ec = [], set(), {}, 0, 0
    for c in ok:
        if c.game_code in games:
            continue
        if c.tim and team_dir.get(c.tim) and team_dir[c.tim] != c.strana:
            continue
        if c.pazar == "poeni" and c.strana == "nad" and over_pts >= 1:
            continue
        if c.liga == "eurocup" and ec >= 2:
            continue
        ticket.append(c)
        games.add(c.game_code)
        if c.tim:
            team_dir[c.tim] = c.strana
        over_pts += int(c.pazar == "poeni" and c.strana == "nad")
        ec += int(c.liga == "eurocup")
        if len(ticket) >= max_picks:
            break
    return ticket if len(ticket) >= 2 else []


MAX_TICKETS_PER_PLAYER = 2          # Tome 31.10 ("ефект White"): risk management, not a model change
TICKET_SIZES = [("мал", 2, 3), ("среден", 4, 5), ("голем", 7, 8)]


def propose_tickets(cands: list[Candidate], sizes=TICKET_SIZES,
                    max_per_player: int = MAX_TICKETS_PER_PLAYER) -> list[tuple[str, list[Candidate], str]]:
    """Proposal tickets (paper): per ticket max 1 pick per game and the build_ticket rules; across tickets one
    player on at most `max_per_player` tickets. A size that cannot be filled is returned empty with the reason
    (never padded with rejected picks)."""
    ok = sorted([c for c in cands if not c.odbien and not c.zavisi and not c.samo_hartija], key=lambda c: c.ev, reverse=True)
    used: dict[str, int] = {}
    out = []
    for name, lo, hi in sizes:
        ticket, games, team_dir, over_pts, ec = [], set(), {}, 0, 0
        for c in ok:
            who = c.player_code or c.igrac or f"{c.game_code}:{c.pazar}:{c.strana}"
            if used.get(who, 0) >= max_per_player or c.game_code in games:
                continue
            if c.tim and team_dir.get(c.tim) and team_dir[c.tim] != c.strana:
                continue
            if c.pazar == "poeni" and c.strana == "nad" and over_pts >= 1:
                continue
            if c.liga == "eurocup" and ec >= 2:
                continue
            ticket.append(c)
            games.add(c.game_code)
            if c.tim:
                team_dir[c.tim] = c.strana
            over_pts += int(c.pazar == "poeni" and c.strana == "nad")
            ec += int(c.liga == "eurocup")
            if len(ticket) >= hi:
                break
        if len(ticket) < lo:
            n_games = len({c.game_code for c in ok})
            out.append((name, [], f"не може {lo}–{hi}: само {len(ticket)} избори од различни натпревари "
                                   f"({n_games} натпревари со избор што ги поминува правилата)"))
            continue
        for c in ticket:
            who = c.player_code or c.igrac or f"{c.game_code}:{c.pazar}:{c.strana}"
            used[who] = used.get(who, 0) + 1
        out.append((name, ticket, ""))
    return out


# ---------- bookmaker player names → our players (rule 2: never guess) ----------

def _norm(t: str) -> str:
    t = unicodedata.normalize("NFKD", str(t)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]", " ", t)


def split_book_name(name: str) -> tuple[str, str]:
    """'Aar.Wiggins' → ('aar', 'wiggins'); 'C.J.McCollum' → ('c', 'mccollum'); 'R. Jokubaitis' → ('r', 'jokubaitis')."""
    tokens = [x for x in _norm(name.replace(".", " ")).split() if x not in ("jr", "sr", "ii", "iii", "iv")]
    if len(tokens) == 1:
        return "", tokens[0]
    return tokens[0], tokens[-1]


def match_book_player(name: str, players: list[dict]) -> Optional[dict]:
    """players: [{player_code, first, last, team, ...}] of the two teams in that game."""
    first, last = split_book_name(name)
    hits = []
    for p in players:
        lasts = _norm(p["last"]).split()
        if last not in lasts and _norm(p["last"]).replace(" ", "") != last:
            continue
        if first and not _norm(p["first"]).replace(" ", "").startswith(first.replace(" ", "")):
            continue
        hits.append(p)
    return hits[0] if len(hits) == 1 else None
