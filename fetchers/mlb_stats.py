"""
MLB Stats API fetcher (unofficial, undocumented).

Covers: schedule, probable pitchers, IL transactions, player details,
final scores, and venue metadata.

All HTTP calls use retry_with_backoff (NFR-03).
The API is undocumented; endpoint availability is not guaranteed (CONSTRAINT).
Empty or malformed responses are handled gracefully without hard-failing.
"""

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any

import numpy as np
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
    start = game_date - timedelta(days=14)
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


def get_postponed_game_ids(game_date: date) -> set[str]:
    """Return gamePk strings for any game on game_date that is Postponed or Cancelled."""
    date_str = game_date.strftime("%Y-%m-%d")
    try:
        data = _get("/schedule", params={"sportId": 1, "date": date_str})
    except Exception as exc:
        logger.warning("postponed game check failed for %s: %s", date_str, exc)
        return set()
    postponed = set()
    for day in data.get("dates", []):
        for g in day.get("games", []):
            state = g.get("status", {}).get("detailedState", "")
            if "Postponed" in state or "Cancelled" in state:
                postponed.add(str(g["gamePk"]))
    if postponed:
        logger.info("Postponed/cancelled games on %s: %s", date_str, ", ".join(sorted(postponed)))
    return postponed


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
# Pitcher recent form — last N starts ERA and WHIP
# ---------------------------------------------------------------------------

_recent_form_cache: dict[tuple, dict] = {}

def get_pitcher_recent_form(pitcher_id: int, game_date: date, n_starts: int = 3) -> dict:
    """
    Return ERA and WHIP over the pitcher's last `n_starts` starts prior to game_date.
    Falls back to league average if the game log is unavailable or too short.
    """
    from config import CURRENT_SEASON, LEAGUE_AVG
    cache_key = (pitcher_id, game_date, n_starts)
    if cache_key in _recent_form_cache:
        return _recent_form_cache[cache_key]

    defaults = {
        "era_l3":  LEAGUE_AVG["sp_era_l3"],
        "whip_l3": LEAGUE_AVG["sp_whip_l3"],
    }

    try:
        data = _get(
            f"/people/{pitcher_id}/stats",
            params={
                "stats":  "gameLog",
                "group":  "pitching",
                "season": CURRENT_SEASON,
            },
        )
        splits = data.get("stats", [{}])[0].get("splits", [])
        # Filter to starts before game_date, sorted oldest→newest
        starts = [
            s for s in splits
            if s.get("stat", {}).get("gamesStarted", 0) > 0
            and s.get("date", "") < game_date.strftime("%Y-%m-%d")
        ]
        if not starts:
            _recent_form_cache[cache_key] = defaults
            return defaults

        recent = starts[-n_starts:]  # last N starts

        total_er = sum(s["stat"].get("earnedRuns", 0) for s in recent)
        total_h  = sum(s["stat"].get("hits", 0) for s in recent)
        total_bb = sum(s["stat"].get("baseOnBalls", 0) for s in recent)

        # Parse innings pitched (e.g. "6.2" = 6 and 2/3 innings)
        total_ip = 0.0
        for s in recent:
            ip_str = str(s["stat"].get("inningsPitched", "0.0"))
            try:
                whole, thirds = ip_str.split(".")
                total_ip += int(whole) + int(thirds) / 3
            except Exception:
                pass

        if total_ip < 0.1:
            _recent_form_cache[cache_key] = defaults
            return defaults

        era_l3  = round((total_er / total_ip) * 9, 2)
        whip_l3 = round((total_h + total_bb) / total_ip, 3)

        result = {"era_l3": era_l3, "whip_l3": whip_l3}
        _recent_form_cache[cache_key] = result
        return result

    except Exception as exc:
        logger.debug("recent form fetch failed for pitcher %d: %s", pitcher_id, exc)
        _recent_form_cache[cache_key] = defaults
        return defaults


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

    risp_14d = LEAGUE_AVG["risp_14d"]
    try:
        # Attempt true 14-day RISP via byDateRange + sitCodes
        risp_data = _get(
            f"/teams/{team_id}/stats",
            params={
                "season":    CURRENT_SEASON,
                "group":     "hitting",
                "stats":     "byDateRange",
                "startDate": start_str,
                "endDate":   end_str,
                "sitCodes":  "RISP",
            },
        )
        risp_splits = risp_data.get("stats", [{}])[0].get("splits", [])
        risp_stat = risp_splits[0].get("stat", {}) if risp_splits else {}
        v = risp_stat.get("avg")
        if v:
            risp_14d = float(v)
            logger.debug("RISP 14d from byDateRange for team %d: %.3f", team_id, risp_14d)
        else:
            raise ValueError("no avg in byDateRange+RISP response")
    except Exception:
        # Fall back to season-level RISP — still better than league average
        try:
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
            if v:
                risp_14d = float(v)
        except Exception as exc:
            logger.debug("RISP fetch failed for team %d: %s", team_id, exc)

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


def get_team_win_pct(team_abbr: str, up_to_date: date) -> float:
    """Season W-L win% for team_abbr in games before up_to_date, from DuckDB."""
    from config import DB_PATH
    try:
        import duckdb
        conn = duckdb.connect(DB_PATH, read_only=True)
        row = conn.execute("""
            SELECT
                SUM(CASE WHEN (home_team = ? AND home_score > away_score)
                              OR (away_team = ? AND away_score > home_score)
                         THEN 1 ELSE 0 END) AS wins,
                COUNT(*) AS games
            FROM results
            WHERE game_date < ?
              AND (home_team = ? OR away_team = ?)
        """, [team_abbr, team_abbr, up_to_date.strftime("%Y-%m-%d"),
              team_abbr, team_abbr]).fetchone()
        conn.close()
        if row and row[1] and row[1] > 0:
            return round(float(row[0] or 0) / float(row[1]), 4)
    except Exception as exc:
        logger.debug("win_pct DB query failed for %s: %s", team_abbr, exc)
    return 0.500


def get_team_back_to_back(team_abbr: str, game_date: date) -> int:
    """1 if team played yesterday (per DuckDB results), 0 otherwise."""
    from config import DB_PATH
    from datetime import timedelta
    yesterday = (game_date - timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        import duckdb
        conn = duckdb.connect(DB_PATH, read_only=True)
        row = conn.execute("""
            SELECT COUNT(*) FROM results
            WHERE game_date = ?
              AND (home_team = ? OR away_team = ?)
        """, [yesterday, team_abbr, team_abbr]).fetchone()
        conn.close()
        return 1 if (row and row[0] > 0) else 0
    except Exception as exc:
        logger.debug("back_to_back DB query failed for %s: %s", team_abbr, exc)
    return 0


def get_team_days_rest(team_abbr: str, game_date: date) -> int:
    """
    Days since team_abbr last played before game_date (capped at 7).
    Returns 1 for back-to-back, 2 for one day off, etc.  Defaults to 1.
    """
    from config import DB_PATH
    try:
        import duckdb
        conn = duckdb.connect(DB_PATH, read_only=True)
        row = conn.execute("""
            SELECT MAX(game_date) FROM results
            WHERE (home_team = ? OR away_team = ?)
              AND game_date < ?
        """, [team_abbr, team_abbr, game_date.strftime("%Y-%m-%d")]).fetchone()
        conn.close()
        if row and row[0]:
            last = row[0]
            if not isinstance(last, date):
                from datetime import datetime as _dt
                last = _dt.strptime(str(last), "%Y-%m-%d").date()
            return min(int((game_date - last).days), 7)
    except Exception as exc:
        logger.debug("team_days_rest DB query failed for %s: %s", team_abbr, exc)
    return 1


def get_series_game_num(team_abbr: str, opp_abbr: str, game_date: date) -> int:
    """
    Series game number (1-based) for today's game between team_abbr and opp_abbr.
    Counts consecutive prior days this matchup was played, going backward from yesterday.
    """
    from config import DB_PATH
    from datetime import timedelta
    try:
        import duckdb
        conn = duckdb.connect(DB_PATH, read_only=True)
        rows = conn.execute("""
            SELECT game_date FROM results
            WHERE ((home_team = ? AND away_team = ?)
                OR (home_team = ? AND away_team = ?))
              AND game_date < ?
            ORDER BY game_date DESC
            LIMIT 7
        """, [team_abbr, opp_abbr, opp_abbr, team_abbr,
              game_date.strftime("%Y-%m-%d")]).fetchall()
        conn.close()

        check = game_date - timedelta(days=1)
        count = 0
        for (rdate,) in rows:
            rd = rdate if isinstance(rdate, date) else datetime.strptime(str(rdate), "%Y-%m-%d").date()
            if rd == check:
                count += 1
                check -= timedelta(days=1)
            else:
                break
        return min(count + 1, 7)
    except Exception as exc:
        logger.debug("series_game DB query failed for %s vs %s: %s", team_abbr, opp_abbr, exc)
    return 1


# ---------------------------------------------------------------------------
# Win/loss streak (from DuckDB results)
# ---------------------------------------------------------------------------

def get_team_win_streak(team_abbr: str, before_date: date) -> int:
    """
    Consecutive win (+N) or loss (-N) streak for team_abbr before before_date.
    Caps at ±10. Returns 0 when no result history exists.
    """
    from config import DB_PATH
    try:
        import duckdb
        conn = duckdb.connect(DB_PATH, read_only=True)
        rows = conn.execute("""
            SELECT home_team, away_team, home_score, away_score
            FROM results
            WHERE (home_team = $1 OR away_team = $1)
              AND game_date < $2
            ORDER BY game_date DESC
            LIMIT 15
        """, [team_abbr, before_date.strftime("%Y-%m-%d")]).fetchall()
        conn.close()
    except Exception as exc:
        logger.debug("win_streak query failed for %s: %s", team_abbr, exc)
        return 0

    if not rows:
        return 0

    streak = 0
    first_won: bool | None = None
    for home, away, hs, as_ in rows:
        won = (hs > as_) if home == team_abbr else (as_ > hs)
        if first_won is None:
            first_won = won
        if won == first_won:
            streak += 1
        else:
            break

    direction = 1 if first_won else -1
    return int(max(-10, min(10, direction * streak)))


# ---------------------------------------------------------------------------
# Bullpen leverage index proxy (from MLB Stats API saves/holds/blown saves)
# ---------------------------------------------------------------------------

_bp_li_proxy_cache: dict[tuple, float] = {}

def get_team_bp_li_proxy(team_id: int, season: int) -> float:
    """
    Proxy for bullpen average leverage index.
    (SV + HLD) / (SV + HLD + BS) normalised so league-average ≈ 1.0.
    Falls back to LEAGUE_AVG when data is sparse.
    """
    from config import LEAGUE_AVG
    cache_key = (team_id, season)
    if cache_key in _bp_li_proxy_cache:
        return _bp_li_proxy_cache[cache_key]
    try:
        data = _get(f"/teams/{team_id}/stats",
                    params={"season": season, "group": "pitching", "stats": "season"})
        splits = data.get("stats", [{}])[0].get("splits", [])
        stat   = splits[0].get("stat", {}) if splits else {}
        sv  = int(stat.get("saves",       0) or 0)
        hld = int(stat.get("holds",       0) or 0)
        bs  = int(stat.get("blownSaves",  0) or 0)
        total = sv + hld + bs
        if total < 10:
            li = LEAGUE_AVG["bp_li"]
        else:
            ratio = (sv + hld) / total          # typical ~0.75 for average team
            li = round(float(np.clip((ratio / 0.75) * LEAGUE_AVG["bp_li"], 0.5, 2.0)), 3)
    except Exception as exc:
        logger.debug("bp_li_proxy failed for team %d: %s", team_id, exc)
        li = LEAGUE_AVG["bp_li"]
    _bp_li_proxy_cache[cache_key] = li
    return li


# ---------------------------------------------------------------------------
# Team lineup handedness split (from roster — used when no actual lineup)
# ---------------------------------------------------------------------------

_lineup_hand_cache: dict[tuple, dict] = {}

def get_team_lineup_hand_pct(team_id: int, season: int) -> dict:
    """
    Returns {"l_pct": float, "r_pct": float} — fraction of active-roster position
    players that bat left / right (switch hitters count as 0.5 each).
    Used as a fallback for hand_match_pct when no actual batting order is available.
    """
    cache_key = (team_id, season)
    if cache_key in _lineup_hand_cache:
        return _lineup_hand_cache[cache_key]
    defaults = {"l_pct": 0.45, "r_pct": 0.55}
    try:
        data = _get(f"/teams/{team_id}/roster",
                    params={"rosterType": "active", "season": season})
        l = r = s = 0
        for p in data.get("roster", []):
            if p.get("position", {}).get("abbreviation", "P") == "P":
                continue
            bats = p.get("person", {}).get("batSide", {}).get("code", "R")
            if bats == "L":
                l += 1
            elif bats == "S":
                s += 1
            else:
                r += 1
        total = l + r + s
        result = ({"l_pct": round((l + s * 0.5) / total, 3),
                   "r_pct": round((r + s * 0.5) / total, 3)}
                  if total > 0 else defaults)
    except Exception as exc:
        logger.debug("lineup_hand_pct failed for team %d: %s", team_id, exc)
        result = defaults
    _lineup_hand_cache[cache_key] = result
    return result


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


# ---------------------------------------------------------------------------
# Umpire assignments (pre-game) and career stats
# ---------------------------------------------------------------------------

_umpire_assign_cache: dict[str, dict] = {}   # date_str → {game_pk: {name, id}}
_umpire_stats_cache:  dict[str, dict] = {}   # umpire_name → stats dict


def get_umpire_assignments(game_date: date) -> dict:
    """
    Return {game_pk_str: {name, id}} for the HP umpire of each game on game_date.
    Umpires are typically assigned by 8 AM ET day-of.
    """
    date_str = game_date.strftime("%Y-%m-%d")
    if date_str in _umpire_assign_cache:
        return _umpire_assign_cache[date_str]

    result: dict = {}
    try:
        data = _get("/schedule", params={
            "sportId": 1,
            "date": date_str,
            "hydrate": "officials",
        })
        for day in data.get("dates", []):
            for g in day.get("games", []):
                pk = str(g["gamePk"])
                result[pk] = {"name": "TBD", "id": None}
                for official in g.get("officials", []):
                    if official.get("officialType") == "Home Plate":
                        person = official.get("official", {})
                        result[pk] = {
                            "name": person.get("fullName", "TBD"),
                            "id":   person.get("id"),
                        }
                        break
    except Exception as exc:
        logger.debug("umpire assignments failed for %s: %s", date_str, exc)

    _umpire_assign_cache[date_str] = result
    return result


_LEAGUE_AVG_IMPACT = 1.45   # umpscorecards total_run_impact_mean league mean
_LEAGUE_AVG_RUNS   = 9.0    # MLB avg total runs per game

def get_umpire_career_stats(umpire_name: str) -> dict:
    """
    Return {avg_runs, tendency} for an HP umpire.

    Priority:
      1. Local data/umpire_stats.json (computed from historical game totals — best source)
      2. umpscorecards.com API (total_run_impact_mean proxy — less accurate but available live)
    Returns {} on failure (caller treats missing as league average).
    """
    if not umpire_name or umpire_name == "TBD":
        return {}
    if umpire_name in _umpire_stats_cache:
        return _umpire_stats_cache[umpire_name]

    # 1. Local historical stats (built by build_training_data.py)
    try:
        import json as _json
        from pathlib import Path as _Path
        _stats_file = _Path(__file__).parent.parent / "data" / "umpire_stats.json"
        if _stats_file.exists():
            _local = _json.loads(_stats_file.read_text())
            if umpire_name in _local:
                avg_r = _local[umpire_name]
                result = {
                    "avg_runs": avg_r,
                    "tendency": ("Favors Over" if avg_r > 9.5
                                 else "Favors Under" if avg_r < 8.5 else "Neutral O/U"),
                }
                _umpire_stats_cache[umpire_name] = result
                return result
    except Exception:
        pass

    # 2. umpscorecards.com — response is {"rows": [...]} with "total_run_impact_mean"
    try:
        resp = requests.get(
            "https://umpscorecards.com/api/umpires/",
            timeout=8,
            headers={"User-Agent": "mlb-predictor/1.0"},
        )
        if resp.status_code == 200:
            rows = resp.json().get("rows", [])
            name_lower = umpire_name.lower()
            for ump in rows:
                api_name = (ump.get("umpire") or "").lower()
                if not api_name:
                    continue
                if name_lower in api_name or api_name in name_lower:
                    impact = float(ump.get("total_run_impact_mean") or _LEAGUE_AVG_IMPACT)
                    # Proxy: shift league avg runs by (impact - league_avg_impact)
                    avg_r = round(_LEAGUE_AVG_RUNS + (impact - _LEAGUE_AVG_IMPACT), 2)
                    result = {
                        "avg_runs": avg_r,
                        "tendency": ("Favors Over" if avg_r > 9.5
                                     else "Favors Under" if avg_r < 8.5 else "Neutral O/U"),
                    }
                    _umpire_stats_cache[umpire_name] = result
                    return result
    except Exception as exc:
        logger.debug("umpscorecards fetch failed for '%s': %s", umpire_name, exc)

    _umpire_stats_cache[umpire_name] = {}
    return {}


# ---------------------------------------------------------------------------
# Team recent W-L record (from local results table)
# ---------------------------------------------------------------------------

def get_team_recent_record(team_abbr: str, before_date: date, n: int = 10) -> dict:
    """
    Return {wins, losses, avg_runs_scored, avg_runs_allowed} over last n games
    before before_date, from the local DuckDB results table.
    """
    from config import DB_PATH
    try:
        import duckdb
        conn = duckdb.connect(DB_PATH, read_only=True)
        rows = conn.execute("""
            SELECT home_team, away_team, home_score, away_score
            FROM results
            WHERE (home_team = $1 OR away_team = $1)
              AND game_date < $2
            ORDER BY game_date DESC
            LIMIT $3
        """, [team_abbr, before_date.strftime("%Y-%m-%d"), n]).fetchall()
        conn.close()
    except Exception as exc:
        logger.debug("recent record query failed for %s: %s", team_abbr, exc)
        return {}

    if not rows:
        return {}

    wins = losses = rs = ra = 0
    for home, away, hs, as_ in rows:
        if home == team_abbr:
            rs += hs; ra += as_
            wins += (1 if hs > as_ else 0); losses += (1 if hs < as_ else 0)
        else:
            rs += as_; ra += hs
            wins += (1 if as_ > hs else 0); losses += (1 if as_ < hs else 0)

    g = len(rows)
    return {
        "wins":  wins,
        "losses": losses,
        "games": g,
        "avg_rs": round(rs / g, 1),
        "avg_ra": round(ra / g, 1),
    }


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
