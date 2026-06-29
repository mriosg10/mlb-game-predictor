"""
Open-Meteo weather fetcher (free, no API key required).

Fetches hourly wind speed, wind direction, and temperature for a venue
coordinate at the scheduled first-pitch hour.

FR-09: fetch weather for outdoor parks.
FR-10: skip weather features for retractable-roof stadiums when roof confirmed closed.
"""

import logging
import math
from datetime import date, datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import requests

from config import OPEN_METEO_BASE, HTTP_TIMEOUT, RETRACTABLE_ROOF_VENUES, LEAGUE_AVG
from utils.retry import retry_with_backoff

logger = logging.getLogger(__name__)

_TRANSIENT = (requests.exceptions.RequestException,)

# Open-Meteo wind direction codes (degrees clockwise from North)
# We store the raw numeric degrees as wind_dir_deg for the model
# and a human-readable compass label as wind_dir.

_COMPASS = [
    (0,   "N"),  (22.5, "NNE"), (45,  "NE"),  (67.5, "ENE"),
    (90,  "E"),  (112.5,"ESE"), (135, "SE"),  (157.5,"SSE"),
    (180, "S"),  (202.5,"SSW"), (225, "SW"),  (247.5,"WSW"),
    (270, "W"),  (292.5,"WNW"), (315, "NW"),  (337.5,"NNW"),
    (360, "N"),
]


def _degrees_to_compass(deg: float) -> str:
    deg = deg % 360
    for threshold, label in reversed(_COMPASS):
        if deg >= threshold - 11.25:
            return label
    return "N"


_OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"


@retry_with_backoff(retries=3, backoff_base=2, exceptions=_TRANSIENT)
def _fetch_open_meteo(lat: float, lon: float, target_date: date) -> dict:
    # Past dates must use the archive endpoint; forecast endpoint only serves future data.
    base_url = _OPEN_METEO_ARCHIVE if target_date < date.today() else OPEN_METEO_BASE
    params = {
        "latitude":  lat,
        "longitude": lon,
        "hourly":    "temperature_2m,windspeed_10m,winddirection_10m",
        "temperature_unit": "fahrenheit",
        "windspeed_unit":   "mph",
        "timezone": "America/New_York",
        "start_date": target_date.strftime("%Y-%m-%d"),
        "end_date":   target_date.strftime("%Y-%m-%d"),
        # Note: do NOT include forecast_days when start_date/end_date are set —
        # they are mutually exclusive in the Open-Meteo API (causes HTTP 400).
    }
    resp = requests.get(base_url, params=params, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get_game_weather(
    lat: float,
    lon: float,
    game_datetime_utc: str | None,
    venue_name: str = "",
    roof_closed: bool = False,
) -> dict[str, Any]:
    """
    Return wind_speed (mph), wind_dir (compass), wind_dir_deg, temperature (°F)
    for the first-pitch hour of the game.

    Returns league-average fallbacks and sets _imputed=True when:
      - The venue has a closed retractable roof (FR-10)
      - The API call fails
      - Coordinates are missing
    """
    neutral = {
        "wind_speed":    LEAGUE_AVG["wind_speed"],
        "wind_dir":      "N",
        "wind_dir_deg":  LEAGUE_AVG["wind_dir_deg"],
        "temperature":   LEAGUE_AVG["temperature"],
        "_imputed":      True,
    }

    # Skip for retractable-roof stadiums with roof closed
    if roof_closed or venue_name in RETRACTABLE_ROOF_VENUES:
        logger.debug("Weather skipped for retractable-roof venue: %s", venue_name)
        return neutral

    if lat is None or lon is None:
        logger.warning("Missing coordinates for venue '%s'; skipping weather", venue_name)
        return neutral

    # Determine which hour to read (first-pitch local hour in ET)
    # Open-Meteo returns hourly data in America/New_York local time when
    # timezone="America/New_York" is set; we must convert UTC→ET the same way.
    first_pitch_hour = 19  # 7 PM ET default
    if game_datetime_utc:
        try:
            dt_utc = datetime.fromisoformat(game_datetime_utc.replace("Z", "+00:00"))
            first_pitch_hour = dt_utc.astimezone(ZoneInfo("America/New_York")).hour
        except Exception:
            pass

    target_date = date.today()
    if game_datetime_utc:
        try:
            target_date = datetime.fromisoformat(
                game_datetime_utc.replace("Z", "+00:00")
            ).date()
        except Exception:
            pass

    try:
        data = _fetch_open_meteo(lat, lon, target_date)
    except Exception as exc:
        logger.warning("Open-Meteo fetch failed for %s: %s", venue_name, exc)
        return neutral

    hourly = data.get("hourly", {})
    times        = hourly.get("time", [])
    temps        = hourly.get("temperature_2m", [])
    wind_speeds  = hourly.get("windspeed_10m", [])
    wind_dirs    = hourly.get("winddirection_10m", [])

    # Find the index matching first_pitch_hour
    idx = None
    for i, t in enumerate(times):
        try:
            hour = int(t.split("T")[1].split(":")[0])
            if hour == first_pitch_hour:
                idx = i
                break
        except Exception:
            continue

    if idx is None or idx >= len(temps):
        logger.warning(
            "Open-Meteo: no hourly data at hour %d for %s", first_pitch_hour, venue_name
        )
        return neutral

    try:
        wind_deg = float(wind_dirs[idx]) if idx < len(wind_dirs) else LEAGUE_AVG["wind_dir_deg"]
        return {
            "wind_speed":   float(wind_speeds[idx]) if idx < len(wind_speeds) else LEAGUE_AVG["wind_speed"],
            "wind_dir":     _degrees_to_compass(wind_deg),
            "wind_dir_deg": wind_deg,
            "temperature":  float(temps[idx]),
            "_imputed":     False,
        }
    except (TypeError, ValueError) as exc:
        logger.warning("Weather parse error for %s: %s", venue_name, exc)
        return neutral
