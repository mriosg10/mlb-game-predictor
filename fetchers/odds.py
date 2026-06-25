"""
Fetch MLB over/under lines from The Odds API.
Endpoint: GET /v4/sports/baseball_mlb/odds?markets=totals&regions=us

Requires MLB_ODDS_API_KEY env var (already wired in config.py).
Returns {} if the key is missing or the call fails — callers treat missing
ou_line as None and fall back to the synthetic formula.
"""

import logging
from datetime import date, timedelta, timezone, datetime

import requests

from config import ODDS_API_KEY, ODDS_API_BASE, HTTP_TIMEOUT

logger = logging.getLogger(__name__)

# Full team name → 3-letter abbreviation used throughout the pipeline
_TEAM_ABBR: dict[str, str] = {
    "Arizona Diamondbacks":  "ARI",
    "Atlanta Braves":        "ATL",
    "Baltimore Orioles":     "BAL",
    "Boston Red Sox":        "BOS",
    "Chicago Cubs":          "CHC",
    "Chicago White Sox":     "CWS",
    "Cincinnati Reds":       "CIN",
    "Cleveland Guardians":   "CLE",
    "Colorado Rockies":      "COL",
    "Detroit Tigers":        "DET",
    "Houston Astros":        "HOU",
    "Kansas City Royals":    "KC",
    "Los Angeles Angels":    "LAA",
    "Los Angeles Dodgers":   "LAD",
    "Miami Marlins":         "MIA",
    "Milwaukee Brewers":     "MIL",
    "Minnesota Twins":       "MIN",
    "New York Mets":         "NYM",
    "New York Yankees":      "NYY",
    "Oakland Athletics":     "ATH",
    "Sacramento Athletics":  "ATH",
    "Athletics":             "ATH",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates":    "PIT",
    "San Diego Padres":      "SD",
    "San Francisco Giants":  "SF",
    "Seattle Mariners":      "SEA",
    "St. Louis Cardinals":   "STL",
    "Tampa Bay Rays":        "TB",
    "Texas Rangers":         "TEX",
    "Toronto Blue Jays":     "TOR",
    "Washington Nationals":  "WSH",
}

# Preferred bookmakers for consensus line (in priority order)
_PREFERRED_BOOKS = ["draftkings", "fanduel", "betmgm", "caesars", "pointsbet"]


def fetch_ou_lines(game_date: date) -> dict[str, float]:
    """
    Fetch consensus over/under lines for all MLB games on game_date.

    Returns a dict keyed by "AWAY@HOME" (3-letter team abbreviations)
    mapping to the consensus ou_line (average of available bookmakers).
    Returns {} on any failure so callers can degrade gracefully.
    """
    if not ODDS_API_KEY:
        logger.debug("MLB_ODDS_API_KEY not set — skipping odds fetch")
        return {}

    url = f"{ODDS_API_BASE}/sports/baseball_mlb/odds"
    # commenceTimeFrom/To: midnight ET (04:00 UTC) to 2 AM ET next day (06:00 UTC +1).
    # This captures all games starting on game_date in ET and avoids next-day games
    # bleeding into the response (which happens at Cycle B time ~17:30 UTC when
    # afternoon games have left the feed and next-day games appear instead).
    from_utc = f"{game_date.isoformat()}T04:00:00Z"
    to_utc   = f"{(game_date + timedelta(days=1)).isoformat()}T06:00:00Z"
    params = {
        "apiKey":             ODDS_API_KEY,
        "regions":            "us",
        "markets":            "totals",
        "oddsFormat":         "american",
        "dateFormat":         "iso",
        "commenceTimeFrom":   from_utc,
        "commenceTimeTo":     to_utc,
    }

    try:
        resp = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("Odds API fetch failed: %s", exc)
        return {}

    # The API returns events in the requested window; commence_time is UTC ISO.
    # target_dates kept as a safety net for any edge cases in the window.
    target_dates = {game_date.isoformat(), (game_date + timedelta(days=1)).isoformat()}

    result: dict[str, float] = {}
    remaining = resp.headers.get("x-requests-remaining", "?")
    logger.info("Odds API: %s requests remaining this month", remaining)

    for event in data:
        commence_utc = event.get("commence_time", "")
        event_date_utc = commence_utc[:10]  # "YYYY-MM-DD"
        if event_date_utc not in target_dates:
            continue

        away_full = event.get("away_team", "")
        home_full = event.get("home_team", "")
        away_abbr = _TEAM_ABBR.get(away_full)
        home_abbr = _TEAM_ABBR.get(home_full)

        if not away_abbr or not home_abbr:
            logger.warning("Odds API: unrecognised team name(s): '%s' / '%s' — add to _TEAM_ABBR",
                           away_full, home_full)
            continue

        key = f"{away_abbr}@{home_abbr}"

        # Collect totals lines across bookmakers
        lines: list[float] = []
        bookmakers = event.get("bookmakers", [])

        # Try preferred books first so consensus reflects sharp lines
        def _book_priority(bk: dict) -> int:
            k = bk.get("key", "")
            try:
                return _PREFERRED_BOOKS.index(k)
            except ValueError:
                return len(_PREFERRED_BOOKS)

        for bk in sorted(bookmakers, key=_book_priority):
            for market in bk.get("markets", []):
                if market.get("key") != "totals":
                    continue
                for outcome in market.get("outcomes", []):
                    if outcome.get("name") == "Over":
                        point = outcome.get("point")
                        if point is not None:
                            lines.append(float(point))
                        break  # one line per book is enough

        if lines:
            consensus = round(sum(lines) / len(lines), 1)
            result[key] = consensus
            logger.debug("Odds: %s → O/U %.1f (%d books)", key, consensus, len(lines))

    logger.info("Odds API: fetched O/U lines for %d games on %s", len(result), game_date)
    return result
