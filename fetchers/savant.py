"""
Baseball Savant / Statcast fetcher via pybaseball.

Provides pitcher-level Statcast aggregates (season-to-date) and
bullpen-level aggregates for the last N days.

Statcast data is updated post-game; Cycle A reflects previous-day stats (CONSTRAINT).
"""

import logging
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from config import CURRENT_SEASON, LEAGUE_AVG

logger = logging.getLogger(__name__)

# Statcast pitch types considered fastballs for spin-rate aggregation
_FASTBALL_TYPES = {"FF", "SI", "FC", "FS"}

# Shared cache for full-MLB Statcast windows — keyed by (start_str, end_str).
# Prevents downloading the same 50–80K-row dataset once per team per cycle.
_statcast_window_cache: dict[tuple, Any] = {}


def _safe_import_pybaseball():
    try:
        import pybaseball as pb
        pb.cache.enable()
        return pb
    except ImportError:
        logger.error("pybaseball not installed — Statcast features unavailable")
        return None


# ---------------------------------------------------------------------------
# Pitcher Statcast aggregates (season-to-date)
# ---------------------------------------------------------------------------

def get_pitcher_statcast(
    pitcher_id: int,
    season: int = CURRENT_SEASON,
) -> dict[str, float | None]:
    """
    Return season-to-date Statcast aggregates for a single pitcher.

    Fields:
        xera, barrel_pct, hard_hit_pct, avg_exit_velo, fastball_spin
    Returns league-average fallbacks on failure (individual fields only;
    the caller is responsible for tracking which fields are real vs imputed).
    """
    pb = _safe_import_pybaseball()
    defaults = {
        "xera":          LEAGUE_AVG["sp_xera"],
        "barrel_pct":    LEAGUE_AVG["sp_barrel"],
        "hard_hit_pct":  LEAGUE_AVG["sp_hh_pct"],
        "avg_exit_velo": LEAGUE_AVG["sp_exit_velo"],
        "fastball_spin": LEAGUE_AVG["sp_spin"],
        "_imputed":      True,
    }
    if pb is None:
        return defaults

    start_dt = f"{season}-03-20"
    end_dt = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        df = pb.statcast_pitcher(start_dt, end_dt, player_id=pitcher_id)
        if df is None or df.empty:
            logger.warning("statcast_pitcher returned empty for id=%d", pitcher_id)
            return defaults
    except Exception as exc:
        logger.warning("statcast_pitcher failed for id=%d: %s", pitcher_id, exc)
        return defaults

    result: dict[str, float | None] = {"_imputed": False}

    # xERA proxy: mean estimated_woba_using_speedangle (lower = better pitcher)
    if "estimated_woba_using_speedangle" in df.columns:
        vals = df["estimated_woba_using_speedangle"].dropna()
        result["xera"] = float(vals.mean()) if not vals.empty else LEAGUE_AVG["sp_xera"]
    else:
        result["xera"] = LEAGUE_AVG["sp_xera"]

    # Barrel rate
    if "barrel" in df.columns:
        bip = df[df["type"] == "X"]  # batted balls only
        if not bip.empty:
            result["barrel_pct"] = float(bip["barrel"].mean())
        else:
            result["barrel_pct"] = LEAGUE_AVG["sp_barrel"]
    else:
        result["barrel_pct"] = LEAGUE_AVG["sp_barrel"]

    # Hard-hit % (launch_speed >= 95)
    if "launch_speed" in df.columns:
        bip = df[df["type"] == "X"].dropna(subset=["launch_speed"])
        if not bip.empty:
            result["hard_hit_pct"] = float((bip["launch_speed"] >= 95).mean())
            result["avg_exit_velo"] = float(bip["launch_speed"].mean())
        else:
            result["hard_hit_pct"] = LEAGUE_AVG["sp_hh_pct"]
            result["avg_exit_velo"] = LEAGUE_AVG["sp_exit_velo"]
    else:
        result["hard_hit_pct"] = LEAGUE_AVG["sp_hh_pct"]
        result["avg_exit_velo"] = LEAGUE_AVG["sp_exit_velo"]

    # Fastball spin rate
    if "release_spin_rate" in df.columns and "pitch_type" in df.columns:
        fb = df[df["pitch_type"].isin(_FASTBALL_TYPES)]["release_spin_rate"].dropna()
        result["fastball_spin"] = float(fb.mean()) if not fb.empty else LEAGUE_AVG["sp_spin"]
    else:
        result["fastball_spin"] = LEAGUE_AVG["sp_spin"]

    return result


# ---------------------------------------------------------------------------
# Bullpen Statcast aggregates (last N days)
# ---------------------------------------------------------------------------

def get_bullpen_statcast(
    team_abbr: str,
    end_date: date,
    days: int = 7,
) -> dict[str, float]:
    """
    Return xERA aggregate for a team's bullpen over the last `days` days.

    Uses pybaseball.statcast_pitcher_exitvelo_barrels (season-level) as proxy.
    Falls back to league average on failure.
    """
    pb = _safe_import_pybaseball()
    defaults = {
        "bp_xera":    LEAGUE_AVG["bp_xera"],
        "_imputed": True,
    }
    if pb is None:
        return defaults

    start_date = end_date - timedelta(days=days)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    window_key = (start_str, end_str)

    if window_key in _statcast_window_cache:
        df = _statcast_window_cache[window_key]
    else:
        try:
            df = pb.statcast(start_dt=start_str, end_dt=end_str)
            if df is None or df.empty:
                _statcast_window_cache[window_key] = pd.DataFrame()
                return defaults
            _statcast_window_cache[window_key] = df
            logger.debug("Statcast window %s–%s cached (%d rows)", start_str, end_str, len(df))
        except Exception as exc:
            logger.warning("bullpen statcast fetch failed for %s: %s", team_abbr, exc)
            return defaults

    if df.empty:
        return defaults

    # In Statcast, "Top" = away team batting (home team pitching),
    # "Bot" = home team batting (away team pitching).
    # Filter to only rows where team_abbr is the pitching side.
    team_pitching = df[
        ((df["home_team"] == team_abbr) & (df["inning_topbot"] == "Top")) |
        ((df["away_team"] == team_abbr) & (df["inning_topbot"] == "Bot"))
    ]

    if "estimated_woba_using_speedangle" not in team_pitching.columns:
        return defaults

    vals = team_pitching["estimated_woba_using_speedangle"].dropna()
    xera_proxy = float(vals.mean()) if not vals.empty else LEAGUE_AVG["bp_xera"]
    return {"bp_xera": xera_proxy, "_imputed": False}
