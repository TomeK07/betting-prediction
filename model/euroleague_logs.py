"""Turn raw EuroLeague/EuroCup feeds into tidy game logs, always sorted by date.

Rule 1 (error 27.10): logs are ordered by game date, never by round or feed order.
Rule 2 (error 16.10): players are identified by official code; the human label is
surname + initial + team, never surname alone.
"""
from __future__ import annotations

import logging
import re

import pandas as pd

from model.logs_common import (add_shot_totals, last_n_games, local_datetime, make_label,  # noqa: F401
                               parse_minutes, percent_columns_to_numbers, sort_logs)

log = logging.getLogger(__name__)

REGULATION_TEAM_MINUTES = 200.0  # 5 players x 40 min
OT_TEAM_MINUTES = 25.0           # 5 players x 5 min

# box score field -> our column
BOX_FIELDS = {
    "Points": "pts",
    "FieldGoalsMade2": "fgm2",
    "FieldGoalsAttempted2": "fga2",
    "FieldGoalsMade3": "fgm3",
    "FieldGoalsAttempted3": "fga3",
    "FreeThrowsMade": "ftm",
    "FreeThrowsAttempted": "fta",
    "OffensiveRebounds": "oreb",
    "DefensiveRebounds": "dreb",
    "TotalRebounds": "reb",
    "Assistances": "ast",
    "Steals": "stl",
    "Turnovers": "tov",
    "BlocksFavour": "blk",
    "BlocksAgainst": "blk_against",
    "FoulsCommited": "pf",
    "FoulsReceived": "fouls_drawn",
    "Valuation": "pir",
}


def normalize_player_code(raw: str) -> str:
    """Box score uses 'P011157   ', stats/roster use '011157'."""
    code = str(raw).strip()
    return code[1:] if code[:1].upper() == "P" else code


def _split_name(name: str) -> tuple[str, str]:
    """'WEILER-BABB, NICK' -> ('Weiler-Babb', 'Nick')."""
    if "," in name:
        last, first = name.split(",", 1)
    else:
        parts = name.split()
        last, first = (parts[-1], " ".join(parts[:-1])) if len(parts) > 1 else (name, "")
    return _roman(last.strip().title()), _roman(first.strip().title())


def _roman(text: str) -> str:
    """title() turns 'III' into 'Iii' — restore generational suffixes."""
    return re.sub(r"\b(Iii|Ii|Iv)\b", lambda m: m.group(0).upper(), text)


def player_label(name: str, team: str) -> str:
    """Human label: surname + initial + team, e.g. 'Motley J. (RED)'."""
    last, first = _split_name(name)
    return make_label(last, first, team)


def schedule_frame(games: list[dict], tz: str) -> pd.DataFrame:
    rows = []
    for g in games:
        local = local_datetime(g["date"], tz)
        rows.append({
            "game_code": int(g["code"]),
            "season": g["season"]["code"],
            "round": g["round"]["round"],
            "phase": g.get("phaseType", {}).get("code"),
            "date_utc": g["date"],
            "game_date": local.date().isoformat(),
            "date_local": local.date().isoformat(),
            "time_local": local.strftime("%H:%M"),
            "status": g["status"],
            "home": g["home"]["code"],
            "away": g["away"]["code"],
            "home_name": g["home"]["name"],
            "away_name": g["away"]["name"],
            # scores only mean something once the game is final
            "home_score": g["home"]["score"] if g["status"] == "result" else None,
            "away_score": g["away"]["score"] if g["status"] == "result" else None,
        })
    df = pd.DataFrame(rows)
    return df.sort_values(["date_utc", "game_code"]).reset_index(drop=True)


def _stat_values(src: dict, prefix: str = "") -> dict:
    row = {f"{prefix}{col}": src.get(field) for field, col in BOX_FIELDS.items()}
    add_shot_totals(row, prefix)
    return row


def parse_boxscore(box: dict, game: dict) -> tuple[list[dict], list[dict]]:
    """One finished game -> (2 team rows, N player rows). `game` is a schedule_frame row."""
    teams = {}
    for side in box["Stats"]:
        codes = {p["Team"].strip() for p in side["PlayersStats"]}
        if len(codes) != 1:
            raise ValueError(f"Натпревар {game['game_code']}: повеќе тим-кодови во една страна {codes}")
        teams[codes.pop()] = side
    expected = {game["home"], game["away"]}
    if set(teams) != expected:
        raise ValueError(f"Натпревар {game['game_code']}: тимови во box score {set(teams)} ≠ распоред {expected}")

    base = {k: game[k] for k in ("season", "game_code", "round", "phase", "date_utc", "game_date", "date_local")}
    team_rows, player_rows = [], []
    for code, side in teams.items():
        opp_code = game["away"] if code == game["home"] else game["home"]
        opp = teams[opp_code]
        home = code == game["home"]
        team_minutes = parse_minutes(side["totr"].get("Minutes"))
        row = {**base, "team": code, "opp": opp_code, "home": home,
               "team_minutes": team_minutes,
               "overtimes": (round((team_minutes - REGULATION_TEAM_MINUTES) / OT_TEAM_MINUTES)
                             if team_minutes else None),
               **_stat_values(side["totr"]),
               # team (non-player) rebounds/turnovers. Verified 29.10.2026 on 160 games: totr
               # oreb/tov ALREADY include these — do not add them again in the possession formula.
               "team_oreb": side["tmr"].get("OffensiveRebounds"),
               "team_dreb": side["tmr"].get("DefensiveRebounds"),
               "team_tov": side["tmr"].get("Turnovers"),
               **_stat_values(opp["totr"], "opp_"),
               "opp_team_oreb": opp["tmr"].get("OffensiveRebounds"),
               "opp_team_dreb": opp["tmr"].get("DefensiveRebounds"),
               "opp_team_tov": opp["tmr"].get("Turnovers"),
               "coach": side.get("Coach")}
        score_col = "home_score" if home else "away_score"
        if game.get(score_col) is not None and row["pts"] != game[score_col]:
            log.warning("Натпревар %s %s: поени во box score %s ≠ распоред %s",
                        game["game_code"], code, row["pts"], game[score_col])
        team_rows.append(row)

        for p in side["PlayersStats"]:
            raw_min = p.get("Minutes")
            name = p["Player"].strip()
            player_rows.append({
                **base, "team": code, "opp": opp_code, "home": home,
                "player_code": normalize_player_code(p["Player_ID"]),
                "player": name,
                "label": player_label(name, code),
                "dorsal": str(p.get("Dorsal", "")).strip(),
                "starter": bool(p.get("IsStarter")),
                "dnp": str(raw_min).strip().upper() == "DNP",
                "absence_reason": "DNP" if str(raw_min).strip().upper() == "DNP" else None,
                "minutes": parse_minutes(raw_min),
                **_stat_values(p),
                "plus_minus": p.get("Plusminus"),
            })
    return team_rows, player_rows


def build_logs(schedule: pd.DataFrame, boxscores: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    team_rows, player_rows = [], []
    by_code = schedule.set_index("game_code")
    for code, box in boxscores.items():
        game = {"game_code": code, **by_code.loc[code].to_dict()}
        t, p = parse_boxscore(box, game)
        team_rows += t
        player_rows += p
    return sort_logs(pd.DataFrame(team_rows)), sort_logs(pd.DataFrame(player_rows))


def roster_frame(people: list[dict]) -> pd.DataFrame:
    """Players only (type 'J'); coaches/staff dropped. Position: Guard/Forward/Center."""
    rows = []
    for item in people:
        if item.get("type") != "J":
            continue
        person = item["person"]
        club = item.get("club") or {}
        rows.append({
            "player_code": normalize_player_code(person["code"]),
            "player": person["name"],
            "label": player_label(person["name"], club.get("code", "?")),
            "team": club.get("code"),
            "position": item.get("positionName") or None,
            "height_cm": person.get("height") or None,
            "dorsal": item.get("dorsal"),
            "active": item.get("active"),
            "start_date": item.get("startDate"),
            "end_date": item.get("endDate"),
        })
    return pd.DataFrame(rows).sort_values(["team", "player"]).reset_index(drop=True)


def stat_table_frame(payload: dict, scope: str) -> pd.DataFrame:
    """Flatten a v3 statistics table (season per-game averages)."""
    rows = []
    for item in payload.get(scope, []):
        row = {k: v for k, v in item.items() if k not in ("player", "team")}
        if scope == "players":
            row = {"player_code": normalize_player_code(item["player"]["code"]),
                   "player": item["player"]["name"],
                   "team": item["player"]["team"]["code"],
                   "label": player_label(item["player"]["name"], item["player"]["team"]["code"]),
                   **row}
        else:
            row = {"team": item["team"]["code"], "team_name": item["team"]["name"], **row}
        rows.append(row)
    return percent_columns_to_numbers(pd.DataFrame(rows))
