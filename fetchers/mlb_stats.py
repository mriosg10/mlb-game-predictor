"""
MLB Stats API fetcher (unofficial, undocumented).

Covers: schedule, probable pitchers, IL transactions, player details,
final scores, and venue metadata.

All HTTP calls use retry_with_backoff (NFR-03).
The API is undocumented; endpoint availability is not guaranteed (CONSTRAINT).
Empty or malformed responses are handled gracefully without hard-failing.
"""

import logging
from datetime import date, datetime, timezone
from typing import Any

import requests

from config import MLB_API_BASE, HTTP_TIMEOUT
from utils.retry import retry_with_backoff

logger = logging.getLogger(__name__)

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "mlb-prediction-pipeline/1.0"})

_TRANSIENT_ERRORS = (requests.exceptions.RequestException,)


# ---------------------------------------------------------------------------
# Low-level HTTP helper
# ---------------------------------------------------------------------------

@retry_with_backoff(retries=3, backoff_base=2, exceptions=_TRANSIENT_ERRORS)
def _get(path: str, params: dict | None = None) -> Any:
    url = f"{MLB_API_BASE}{path}"
    resp = _SESSION.get(url, params=params, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

def get_schedule(game_date: date) -> list[dict]:
    """
    Return a list of game dicts for the given date.

    Each dict contains:
        game_id, home_team, away_team, venue_id, venue_name,
        game_datetime_utc, status
    Returns [] on empty schedule or API error.
    """
    date_str = game_date.strftime("%Y-%m-%d")
    try:
        data = _get(
            "/schedule",
            params={
                "sportId": 1,
                "date": date_str,
                "hydrate": "probablePitcher,linescore,team",
            },
        )
    except Exception as exc:
        logger.error("schedule fetch failed for %s: %s", date_str, exc)
        return []

    games = []
    for day in data.get("dates", []):
        for g in day.get("games", []):
            status = g.get("status", {}).get("abstractGameState", "")
            games.append({
                "game_id":            str(g["gamePk"]),
                "game_date":          date_str,
                "home_team":          g["teams"]["home"]["team"]["abbreviation"],
                "home_team_id":       g["teams"]["home"]["team"]["id"],
                "away_team":          g["teams"]["away"]["team"]["abbreviation"],
                "away_team_id":       g["teams"]["away"]["team"]["id"],
                "venue_id":           g.get("venue", {}).get("id"),
                "venue_name":         g.get("venue", {}).get("name", ""),
                "game_datetime_utc":  g.get("gameDate"),  # ISO-8601 UTC
                "status":             status,
                "home_probable_id":   (
                    g["teams"]["home"].get("probablePitcher", {}).get("id")
                ),
                "away_probable_id":   (
                    g["teams"]["away"].get("probablePitcher", {}).get("id")
                ),
            })
    logger.info("schedule: %d games on %s", len(games), date_str)
    return games


# ---------------------------------------------------------------------------
# Player details
# ---------------------------------------------------------------------------

@retry_with_backoff(retries=3, backoff_base=2, exceptions=_TRANSIENT_ERRORS)
def get_player(player_id: int) -> dict:
    """Return player details dict. Raises on failure."""
    data = _get(f"/people/{player_id}", params={"hydrate": "currentTeam"})
    p = data["people"][0]
    return {
        "id":          p["id"],
        "full_name":   p["fullName"],
        "pitch_hand":  p.get("pitchHand", {}).get("code", "R"),  # R/L/S
        "bat_side":    p.get("batSide",   {}).get("code", "R"),  # R/L/S
        "position":    p.get("primaryPosition", {}).get("abbreviation", ""),
        "team_id":     p.get("currentTeam", {}).get("id"),
    }


def get_players_bulk(player_ids: list[int]) -> dict[int, dict]:
    """Fetch multiple players; skips failures and returns partial results."""
    result = {}
    for pid in player_ids:
        try:
            result[pid] = get_player(pid)
        except Exception as exc:
            logger.warning("player %d fetch failed: %s", pid, exc)
    return result


# ---------------------------------------------------------------------------
# Roster / IL transactions
# ---------------------------------------------------------------------------

def get_il_transactions(team_id: int, game_date: date) -> list[dict]:
    """
    Return IL placements/activations for the given team within the last 14 days.
    Used to count unavailable starters and bullpen arms.
    """
    start = game_date.replace(day=max(1, game_date.day - 14))
    try:
        data = _get(
            "/transactions",
            params={
                "teamId": team_id,
                "startDate": start.strftime("%Y-%m-%d"),
                "endDate": game_date.strftime("%Y-%m-%d"),
            },
        )
    except Exception as exc:
        logger.warning("IL transactions fetch failed for team %d: %s", team_id, exc)
        return []

    txns = []
    for t in data.get("transactions", []):
        txns.append({
            "player_id":   t.get("person", {}).get("id"),
            "player_name": t.get("person", {}).get("fullName"),
            "type":        t.get("typeCode", ""),
            "date":        t.get("date", ""),
        })
    return txns


def count_il_players(team_id: int, game_date: date, position_filter: str | None = None) -> int:
    """
    Count players currently on IL (placed but not yet activated).
    position_filter: 'P' for pitchers, None for all.
    """
    txns = get_il_transactions(team_id, game_date)
    on_il: set[int] = set()
    for t in sorted(txns, key=lambda x: x["date"]):
        pid = t.get("player_id")
        if pid is None:
            continue
        type_code = t.get("type", "")
        # IL placement codes
        if type_code in ("IL10", "IL15", "IL60", "SUSP", "BRV"):
            on_il.add(pid)
        elif type_code in ("CRA", "REL"):  # activated / reinstated
            on_il.discard(pid)

    if position_filter is None or not on_il:
        return len(on_il)

    # Filter by position
    count = 0
    players = get_players_bulk(list(on_il))
    for player in players.values():
        pos = player.get("position", "")
        if position_filter == "P" and pos == "P":
            count += 1
        elif position_filter != "P" and pos != "P":
            count += 1
    return count


# ---------------------------------------------------------------------------
# Final scores
# ---------------------------------------------------------------------------

def get_final_scores(game_date: date) -> list[dict]:
    """
    Return final scores for all games on game_date.
    Only games with status == 'Final' are included.
    """
    date_str = game_date.strftime("%Y-%m-%d")
    try:
        data = _get(
            "/schedule",
            params={
                "sportId": 1,
                "date": date_str,
                "hydrate": "linescore,team",
            },
        )
    except Exception as exc:
        logger.error("final scores fetch failed for %s: %s", date_str, exc)
        return []

    scores = []
    for day in data.get("dates", []):
        for g in day.get("games", []):
            abstract_state = g.get("status", {}).get("abstractGameState", "")
            if abstract_state != "Final":
                continue
            if "linescore" not in g:
                logger.debug("game %s has no linescore, skipping", g.get("gamePk"))
                continue
            home_score = g["linescore"]["teams"]["home"].get("runs", 0)
            away_score = g["linescore"]["teams"]["away"].get("runs", 0)
            home_abbr = g["teams"]["home"]["team"].get("abbreviation") or g["teams"]["home"]["team"].get("teamCode", "UNK").upper()
            away_abbr = g["teams"]["away"]["team"].get("abbreviation") or g["teams"]["away"]["team"].get("teamCode", "UNK").upper()
            scores.append({
                "game_id":    str(g["gamePk"]),
                "game_date":  date_str,
                "home_team":  home_abbr,
                "away_team":  away_abbr,
                "home_score": int(home_score),
                "away_score": int(away_score),
                "winner":     home_abbr if home_score > away_score else away_abbr,
                "total_runs": int(home_score) + int(away_score),
            })
    logger.info("final scores: %d completed games on %s", len(scores), date_str)
    return scores


# ---------------------------------------------------------------------------
# Venue metadata
# ---------------------------------------------------------------------------

def get_venue_info(venue_id: int) -> dict:
    """Return venue name, lat, lon for a given venue ID."""
    try:
        data = _get(f"/venues/{venue_id}", params={"hydrate": "location"})
        v = data["venues"][0]
        loc = v.get("location", {})
        return {
            "venue_id":   venue_id,
            "venue_name": v.get("name", ""),
            "lat":        loc.get("defaultCoordinates", {}).get("latitude"),
            "lon":        loc.get("defaultCoordinates", {}).get("longitude"),
        }
    except Exception as exc:
        logger.warning("venue %d fetch failed: %s", venue_id, exc)
        return {}


# ---------------------------------------------------------------------------
# Roster for a team (used for handedness match calculation)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Team rolling batting stats (14-day OPS and RISP avg)
# ---------------------------------------------------------------------------

_rolling_batting_cache: dict[tuple, dict] = {}

def get_team_rolling_batting(team_id: int, end_date: date, days: int = 14) -> dict:
    """
    Return OPS and RISP batting average for a team over the last `days` days.
    Uses MLB Stats API byDateRange and statSplits endpoints.
    """
    from config import CURRENT_SEASON, LEAGUE_AVG
    cache_key = (team_id, end_date, days)
    if cache_key in _rolling_batting_cache:
        return _rolling_batting_cache[cache_key]

    start_date = end_date - __import__("datetime").timedelta(days=days)
    start_str  = start_date.strftime("%Y-%m-%d")
    end_str    = end_date.strftime("%Y-%m-%d")

    defaults = {
        "ops_14d":  LEAGUE_AVG["ops_14d"],
        "risp_14d": LEAGUE_AVG["risp_14d"],
    }

    try:
        # 14-day OPS via byDateRange
        data = _get(
            f"/teams/{team_id}/stats",
            params={
                "season":    CURRENT_SEASON,
                "group":     "hitting",
                "stats":     "byDateRange",
                "startDate": start_str,
                "endDate":   end_str,
            },
        )
        splits = data.get("stats", [{}])[0].get("splits", [])
        stat = splits[0].get("stat", {}) if splits else {}

        def _f(key: str, fallback: float) -> float:
            v = stat.get(key)
            try:
                return float(v) if v is not None else fallback
            except (TypeError, ValueError):
                return fallback

        obp = _f("obp", 0.320)
        slg = _f("slg", 0.400)
        ops_14d = round(obp + slg, 4)
    except Exception as exc:
        logger.debug("rolling OPS fetch failed for team %d: %s", team_id, exc)
        ops_14d = LEAGUE_AVG["ops_14d"]

    try:
        # RISP avg via statSplits
        risp_data = _get(
            f"/teams/{team_id}/stats",
            params={
                "season":   CURRENT_SEASON,
                "group":    "hitting",
                "stats":    "statSplits",
                "sitCodes": "RISP",
            },
        )
        risp_splits = risp_data.get("stats", [{}])[0].get("splits", [])
        risp_stat = risp_splits[0].get("stat", {}) if risp_splits else {}
        v = risp_stat.get("avg")
        risp_14d = float(v) if v else LEAGUE_AVG["risp_14d"]
    except Exception as exc:
        logger.debug("RISP fetch failed for team %d: %s", team_id, exc)
        risp_14d = LEAGUE_AVG["risp_14d"]

    result = {"ops_14d": ops_14d, "risp_14d": risp_14d}
    _rolling_batting_cache[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Team run differential (from accumulated results in DuckDB)
# ---------------------------------------------------------------------------

def get_team_run_diff(team_abbr: str, up_to_date: date) -> float:
    """
    Season-to-date run differential per game for team_abbr, computed from
    the local results table. Returns 0.0 if no games found.
    """
    from config import DB_PATH, LEAGUE_AVG
    try:
        import duckdb
        conn = duckdb.connect(DB_PATH, read_only=True)
        row = conn.execute("""
            SELECT
                SUM(CASE WHEN home_team = ? THEN home_score - away_score
                         WHEN away_team = ? THEN away_score - home_score
                         ELSE 0 END)                AS total_diff,
                COUNT(*)                            AS games
            FROM results
            WHERE game_date < ?
              AND (home_team = ? OR away_team = ?)
        """, [team_abbr, team_abbr, up_to_date.strftime("%Y-%m-%d"),
              team_abbr, team_abbr]).fetchone()
        conn.close()
        if row and row[1] and row[1] > 0:
            return round(float(row[0]) / float(row[1]), 3)
    except Exception as exc:
        logger.debug("run_diff DB query failed for %s: %s", team_abbr, exc)
    return float(LEAGUE_AVG["run_diff"])


# ---------------------------------------------------------------------------
# Bullpen IP last 3 days (from MLB Stats API box scores)
# ---------------------------------------------------------------------------

_bullpen_ip_cache: dict[tuple, float] = {}

def get_bullpen_ip_3d(team_id: int, end_date: date) -> float:
    """
    Sum of reliever innings pitched for team_id over the 3 days ending on
    end_date (exclusive). Uses the schedule box score hydration endpoint.
    """
    from config import CURRENT_SEASON, LEAGUE_AVG
    import datetime as dt
    cache_key = (team_id, end_date)
    if cache_key in _bullpen_ip_cache:
        return _bullpen_ip_cache[cache_key]

    start = end_date - dt.timedelta(days=3)
    total_ip = 0.0

    try:
        # Fetch schedule to get game PKs (no boxscore hydration on schedule)
        sched = _get(
            "/schedule",
            params={
                "sportId":   1,
                "teamId":    team_id,
                "startDate": start.strftime("%Y-%m-%d"),
                "endDate":   (end_date - dt.timedelta(days=1)).strftime("%Y-%m-%d"),
                "gameType":  "R",
            },
        )
        for day in sched.get("dates", []):
            for game in day.get("games", []):
                if game.get("status", {}).get("abstractGameState") != "Final":
                    continue
                game_pk = game["gamePk"]
                home_id = game["teams"]["home"]["team"]["id"]
                side = "home" if home_id == team_id else "away"

                # Fetch box score per game
                try:
                    box = _get(f"/game/{game_pk}/boxscore")
                    pitchers    = box.get("teams", {}).get(side, {}).get("pitchers", [])
                    all_players = box.get("teams", {}).get(side, {}).get("players", {})
                    for pid in pitchers:
                        p_stats = all_players.get(f"ID{pid}", {}).get(
                            "stats", {}).get("pitching", {})
                        if p_stats.get("gamesStarted", 0) == 0:
                            ip_str = str(p_stats.get("inningsPitched", "0.0"))
                            try:
                                whole, thirds = ip_str.split(".")
                                total_ip += int(whole) + int(thirds) / 3
                            except Exception:
                                pass
                except Exception:
                    pass
    except Exception as exc:
        logger.debug("bullpen IP 3d fetch failed for team %d: %s", team_id, exc)
        _bullpen_ip_cache[cache_key] = float(LEAGUE_AVG["bp_ip_3d"])
        return float(LEAGUE_AVG["bp_ip_3d"])

    result = round(total_ip, 2) if total_ip > 0 else float(LEAGUE_AVG["bp_ip_3d"])
    _bullpen_ip_cache[cache_key] = result
    return result


def get_active_roster(team_id: int) -> list[dict]:
    """Return list of active roster players with position and bat-side."""
    try:
        data = _get(f"/teams/{team_id}/roster", params={"rosterType": "active"})
        roster = []
        for entry in data.get("roster", []):
            roster.append({
                "player_id":   entry["person"]["id"],
                "full_name":   entry["person"]["fullName"],
                "position":    entry.get("position", {}).get("abbreviation", ""),
                "status":      entry.get("status", {}).get("code", ""),
            })
        return roster
    except Exception as exc:
        logger.warning("roster fetch failed for team %d: %s", team_id, exc)
        return []
