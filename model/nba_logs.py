"""Turn raw stats.nba.com payloads (via nba_api) into the common, date-sorted log schema.

Dates: `game_date` is the official US Eastern date (what NBA and Mozzart call the game day);
`date_local` is the Skopje date, usually one day later for evening US games.
"""
from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd

from model.logs_common import (add_shot_totals, clean_suffix, last_n_games, local_datetime, make_label,  # noqa: F401
                               parse_minutes, sort_logs)

log = logging.getLogger(__name__)

REGULATION_TEAM_MINUTES = 240.0  # 5 players x 48 min
OT_TEAM_MINUTES = 25.0           # 5 players x 5 min

GAME_TYPES = {"001": "PRE", "002": "RS", "003": "ASG", "004": "PO", "005": "PI", "006": "CUP_FINAL"}
FINAL_STATUS = 3

# BoxScoreTraditionalV3 statistics field -> our column
BOX_FIELDS = {
    "points": "pts",
    "fieldGoalsMade": "fgm",
    "fieldGoalsAttempted": "fga",
    "threePointersMade": "fgm3",
    "threePointersAttempted": "fga3",
    "freeThrowsMade": "ftm",
    "freeThrowsAttempted": "fta",
    "reboundsOffensive": "oreb",
    "reboundsDefensive": "dreb",
    "reboundsTotal": "reb",
    "assists": "ast",
    "steals": "stl",
    "turnovers": "tov",
    "blocks": "blk",
    "foulsPersonal": "pf",
}


def result_set_frame(payload: dict, index: int = 0) -> pd.DataFrame:
    """stats.nba.com 'resultSets' table -> DataFrame."""
    sets = payload.get("resultSets") or payload.get("resultSet")
    rs = sets[index] if isinstance(sets, list) else sets
    return pd.DataFrame(rs["rowSet"], columns=rs["headers"])


def schedule_frame(payload: dict, tz: str, game_types: Iterable[str]) -> pd.DataFrame:
    keep = set(game_types)
    rows = []
    for day in payload["leagueSchedule"]["gameDates"]:
        for g in day["games"]:
            prefix = g["gameId"][:3]
            if prefix not in keep:
                continue
            local = local_datetime(g["gameDateTimeUTC"], tz)
            final = g["gameStatus"] == FINAL_STATUS
            rows.append({
                "game_code": g["gameId"],
                "season": g["gameId"][3:5],
                "round": None,
                "phase": GAME_TYPES.get(prefix, prefix),
                "label": g.get("gameLabel") or None,
                "date_utc": g["gameDateTimeUTC"],
                "game_date": g["gameDateEst"][:10],
                "date_local": local.date().isoformat(),
                "time_local": local.strftime("%H:%M"),
                "status": "result" if final else ("live" if g["gameStatus"] == 2 else "scheduled"),
                "postponed": g.get("postponedStatus") not in (None, "N"),
                "home": g["homeTeam"]["teamTricode"],
                "away": g["awayTeam"]["teamTricode"],
                "home_id": g["homeTeam"]["teamId"],
                "away_id": g["awayTeam"]["teamId"],
                "home_name": f'{g["homeTeam"]["teamCity"]} {g["homeTeam"]["teamName"]}',
                "away_name": f'{g["awayTeam"]["teamCity"]} {g["awayTeam"]["teamName"]}',
                "home_score": g["homeTeam"]["score"] if final else None,
                "away_score": g["awayTeam"]["score"] if final else None,
            })
    df = pd.DataFrame(rows)
    return df.sort_values(["date_utc", "game_code"]).reset_index(drop=True)


def team_id_map(schedule: pd.DataFrame) -> dict[int, str]:
    pairs = list(zip(schedule["home_id"], schedule["home"])) + list(zip(schedule["away_id"], schedule["away"]))
    return {int(i): code for i, code in pairs}


def _stat_values(stats: dict, prefix: str = "") -> dict:
    row = {f"{prefix}{col}": stats.get(field) for field, col in BOX_FIELDS.items()}
    add_shot_totals(row, prefix)
    return row


def parse_boxscore(payload: dict, game: dict) -> tuple[list[dict], list[dict]]:
    """BoxScoreTraditionalV3 -> (2 team rows, N player rows incl. DNP/inactive with reason)."""
    box = payload["boxScoreTraditional"]
    sides = {box["homeTeam"]["teamTricode"]: box["homeTeam"], box["awayTeam"]["teamTricode"]: box["awayTeam"]}
    expected = {game["home"], game["away"]}
    if set(sides) != expected:
        raise ValueError(f"Натпревар {game['game_code']}: тимови во box score {set(sides)} ≠ распоред {expected}")

    base = {k: game[k] for k in ("season", "game_code", "round", "phase", "date_utc", "game_date", "date_local")}
    team_rows, player_rows = [], []
    for code, side in sides.items():
        home = code == game["home"]
        opp_code = game["away"] if home else game["home"]
        opp = sides[opp_code]
        team_minutes = parse_minutes(side["statistics"].get("minutes"))
        row = {**base, "team": code, "opp": opp_code, "home": home,
               "team_minutes": team_minutes,
               "overtimes": (round((team_minutes - REGULATION_TEAM_MINUTES) / OT_TEAM_MINUTES)
                             if team_minutes else None),
               **_stat_values(side["statistics"]),
               **_stat_values(opp["statistics"], "opp_")}
        score_col = "home_score" if home else "away_score"
        if game.get(score_col) is not None and row["pts"] != game[score_col]:
            log.warning("Натпревар %s %s: поени во box score %s ≠ распоред %s",
                        game["game_code"], code, row["pts"], game[score_col])
        team_rows.append(row)

        for p in side["players"]:
            st = p["statistics"]
            minutes = parse_minutes(st.get("minutes"))
            reason = (p.get("comment") or "").strip() or None
            played = minutes is not None and minutes > 0
            player_rows.append({
                **base, "team": code, "opp": opp_code, "home": home,
                "player_code": str(p["personId"]),
                "player": f'{p["firstName"]} {clean_suffix(p["familyName"])}'.strip(),
                "label": make_label(clean_suffix(p["familyName"]), p["firstName"], code),
                "dorsal": str(p.get("jerseyNum") or "").strip(),
                "starter": bool((p.get("position") or "").strip()),
                "start_position": (p.get("position") or "").strip() or None,
                "dnp": not played,
                # e.g. "DNP - Coach's Decision", "Inactive - Injury/Illness - Left Ankle; Sprain"
                "absence_reason": reason,
                "minutes": minutes if played else 0.0,
                **_stat_values(st),
                "plus_minus": st.get("plusMinusPoints"),
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


LEAGUE_GAME_LOG_FIELDS = {
    "PTS": "pts", "FGM": "fgm", "FGA": "fga", "FG3M": "fgm3", "FG3A": "fga3", "FTM": "ftm", "FTA": "fta",
    "OREB": "oreb", "DREB": "dreb", "REB": "reb", "AST": "ast", "STL": "stl", "TOV": "tov", "BLK": "blk", "PF": "pf",
}


def from_league_game_log(team_payload: dict, player_payload: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Past-season path: LeagueGameLog (team + player) -> (schedule, team_logs, player_logs).

    Limits (documented): no tip-off time (date_utc = game date 00:00), player minutes are whole
    numbers, no starters/DNP rows. Good for team-level backtests.
    """
    t = result_set_frame(team_payload)
    t["home"] = t["MATCHUP"].str.contains(" vs. ")
    t["date_utc"] = t["GAME_DATE"] + "T00:00:00Z"
    stats = lambda df, prefix="": {f"{prefix}{c}": df[k].values for k, c in LEAGUE_GAME_LOG_FIELDS.items()}
    base = pd.DataFrame({
        "season": t["SEASON_ID"].astype(str).str[-2:], "game_code": t["GAME_ID"], "round": None,
        "phase": t["GAME_ID"].str[:3].map(GAME_TYPES), "date_utc": t["date_utc"], "game_date": t["GAME_DATE"],
        "date_local": t["GAME_DATE"], "team": t["TEAM_ABBREVIATION"], "home": t["home"],
        "team_minutes": t["MIN"].astype(float), **stats(t)})
    base["overtimes"] = ((base["team_minutes"] - REGULATION_TEAM_MINUTES) / OT_TEAM_MINUTES).round().astype(int)
    opp = base[["game_code", "team"] + list(LEAGUE_GAME_LOG_FIELDS.values())].rename(
        columns={c: f"opp_{c}" for c in LEAGUE_GAME_LOG_FIELDS.values()} | {"team": "opp"})
    team_logs = base.merge(opp, on="game_code")
    team_logs = team_logs[team_logs["team"] != team_logs["opp"]].copy()
    for prefix in ("", "opp_"):
        team_logs[f"{prefix}fgm2"] = team_logs[f"{prefix}fgm"] - team_logs[f"{prefix}fgm3"]
        team_logs[f"{prefix}fga2"] = team_logs[f"{prefix}fga"] - team_logs[f"{prefix}fga3"]
    if (team_logs.groupby("game_code").size() != 2).any():
        raise ValueError("LeagueGameLog: натпревар без точно 2 тима")

    home_rows = team_logs[team_logs["home"]]
    schedule = pd.DataFrame({
        "game_code": home_rows["game_code"], "season": home_rows["season"], "round": None,
        "phase": home_rows["phase"], "label": None, "date_utc": home_rows["date_utc"],
        "game_date": home_rows["game_date"], "date_local": home_rows["game_date"], "time_local": None,
        "status": "result", "home": home_rows["team"], "away": home_rows["opp"],
        "home_name": home_rows["team"], "away_name": home_rows["opp"],
        "home_score": home_rows["pts"], "away_score": home_rows["opp_pts"],
    }).sort_values(["date_utc", "game_code"]).reset_index(drop=True)

    p = result_set_frame(player_payload)
    first = p["PLAYER_NAME"].str.split(" ").str[0]
    last = p["PLAYER_NAME"].str.split(" ", n=1).str[1].fillna("")
    games = team_logs[["game_code", "team", "opp", "home"]]
    player_logs = pd.DataFrame({
        "season": p["SEASON_ID"].astype(str).str[-2:], "game_code": p["GAME_ID"], "round": None,
        "phase": p["GAME_ID"].str[:3].map(GAME_TYPES), "date_utc": p["GAME_DATE"] + "T00:00:00Z",
        "game_date": p["GAME_DATE"], "date_local": p["GAME_DATE"], "team": p["TEAM_ABBREVIATION"],
        "player_code": p["PLAYER_ID"].astype(str), "player": p["PLAYER_NAME"],
        "label": [make_label(l, f, t) for l, f, t in zip(last, first, p["TEAM_ABBREVIATION"])],
        "starter": None, "dnp": p["MIN"].fillna(0) <= 0, "absence_reason": None,
        "minutes": p["MIN"].astype(float), **stats(p), "plus_minus": p["PLUS_MINUS"],
    }).merge(games, on=["game_code", "team"], how="left")
    player_logs["fgm2"] = player_logs["fgm"] - player_logs["fgm3"]
    player_logs["fga2"] = player_logs["fga"] - player_logs["fga3"]
    return schedule, sort_logs(team_logs), sort_logs(player_logs)


def roster_frame(payload: dict) -> pd.DataFrame:
    """PlayerIndex -> one row per player on a current roster. Position: G, F, C, G-F, F-C…"""
    df = result_set_frame(payload)
    out = pd.DataFrame({
        "player_code": df["PERSON_ID"].astype(str),
        "player": (df["PLAYER_FIRST_NAME"] + " " + df["PLAYER_LAST_NAME"].map(clean_suffix)).str.strip(),
        "label": [make_label(clean_suffix(l), f, t or "?") for l, f, t in
                  zip(df["PLAYER_LAST_NAME"], df["PLAYER_FIRST_NAME"], df["TEAM_ABBREVIATION"])],
        "team": df["TEAM_ABBREVIATION"],
        "position": df["POSITION"].replace("", None),
        "height": df["HEIGHT"],
        "dorsal": df["JERSEY_NUMBER"],
        "roster_status": df["ROSTER_STATUS"],
    })
    return out.sort_values(["team", "player"]).reset_index(drop=True)


def stat_table_frame(payload: dict, scope: str, team_codes: dict[int, str]) -> pd.DataFrame:
    """LeagueDash{Team,Player}Stats -> DataFrame with our `team` (tricode) and `player_code`."""
    df = result_set_frame(payload)
    df = df[[c for c in df.columns if not c.endswith("_RANK")]]
    if scope == "players":
        df.insert(0, "player_code", df["PLAYER_ID"].astype(str))
        df.insert(1, "team", df["TEAM_ABBREVIATION"])
    else:
        df.insert(0, "team", df["TEAM_ID"].map(lambda i: team_codes.get(int(i))))
    return df.reset_index(drop=True)
