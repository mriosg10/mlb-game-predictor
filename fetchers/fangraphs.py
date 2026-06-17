"""
Pitcher and team stat fetcher — primary and fallback sources.

Assumption A-01 (FanGraphs scraping permitted) has been found invalid:
FanGraphs now blocks the legacy API endpoint used by pybaseball.

Data sources used in order of preference:
  Pitcher quality  → Baseball Reference (pitching_stats_bref) for FIP/K%/BB%
                      Baseball Savant (statcast_pitcher_expected_stats) for xERA
                      Baseball Savant (statcast_pitcher_pitch_arsenal) for spin rate
  Team batting     → MLB Stats API /teams/{id}/stats  (OBP ≈ wOBA at team level)
  Park factors     → FanGraphs guts.aspx scrape (may 403)
                      → PARK_FACTORS_HARDCODED from config as static fallback

FanGraphs xFIP and SIERA cannot be sourced without their API; FIP is used
as a proxy (correlation > 0.90 at season level).
"""

import logging
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import requests

from config import CURRENT_SEASON, LEAGUE_AVG, PARK_FACTORS_HARDCODED, HTTP_TIMEOUT
from utils.retry import retry_with_backoff

logger = logging.getLogger(__name__)

_TRANSIENT = (requests.exceptions.RequestException,)

# FIP constant (league-average HR/FB adjusted)
_FIP_CONST = 3.10


def _safe_import_pybaseball():
    try:
        import pybaseball as pb
        pb.cache.enable()
        return pb
    except ImportError:
        logger.error("pybaseball not installed")
        return None


# ---------------------------------------------------------------------------
# Season-level stat tables (loaded once, cached in memory per season)
# ---------------------------------------------------------------------------

_bref_pitcher_cache: dict[int, pd.DataFrame] = {}
_savant_expected_cache: dict[int, pd.DataFrame] = {}
_spin_cache: dict[int, pd.DataFrame] = {}


def _load_bref_pitchers(season: int) -> pd.DataFrame:
    """Load Baseball Reference pitching stats for a season."""
    if season in _bref_pitcher_cache:
        return _bref_pitcher_cache[season]

    pb = _safe_import_pybaseball()
    if pb is None:
        return pd.DataFrame()

    try:
        df = pb.pitching_stats_bref(season)
    except Exception as exc:
        logger.warning("pitching_stats_bref(%d) failed: %s", season, exc)
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # Deduplicate: pitchers traded mid-season appear once per team + "TOT" total row.
    # Keep the "TOT" row when present, otherwise the row with the most IP.
    df["IP"] = pd.to_numeric(df["IP"], errors="coerce").fillna(0)
    df["mlbID"] = pd.to_numeric(df["mlbID"], errors="coerce")
    df = df.dropna(subset=["mlbID"])
    df["mlbID"] = df["mlbID"].astype(int)

    # Prefer Maj-NL/Maj-AL rows (MLB only, not minors)
    df = df[df["Lev"].str.startswith("Maj", na=False)]

    # For multi-team pitchers: keep row with highest IP (= season total)
    df = df.sort_values("IP", ascending=False).drop_duplicates(subset=["mlbID"], keep="first")

    # Pre-compute FIP, K%, BB%
    df["BF"]  = pd.to_numeric(df["BF"],  errors="coerce").fillna(1)
    df["SO"]  = pd.to_numeric(df["SO"],  errors="coerce").fillna(0)
    df["BB"]  = pd.to_numeric(df["BB"],  errors="coerce").fillna(0)
    df["HR"]  = pd.to_numeric(df["HR"],  errors="coerce").fillna(0)
    df["HBP"] = pd.to_numeric(df["HBP"], errors="coerce").fillna(0)

    df["calc_fip"]   = (13 * df["HR"] + 3 * (df["BB"] + df["HBP"]) - 2 * df["SO"]) / df["IP"].clip(lower=1) + _FIP_CONST
    df["calc_k_pct"] = df["SO"] / df["BF"].clip(lower=1)
    df["calc_bb_pct"]= df["BB"] / df["BF"].clip(lower=1)

    _bref_pitcher_cache[season] = df
    logger.debug("bref pitcher stats loaded for %d: %d rows", season, len(df))
    return df


def _load_savant_expected(season: int) -> pd.DataFrame:
    """Load Baseball Savant expected stats (xERA) for a season."""
    if season in _savant_expected_cache:
        return _savant_expected_cache[season]

    pb = _safe_import_pybaseball()
    if pb is None:
        return pd.DataFrame()

    try:
        df = pb.statcast_pitcher_expected_stats(season, minPA=20)
    except Exception as exc:
        logger.warning("statcast_pitcher_expected_stats(%d) failed: %s", season, exc)
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")
    df = df.dropna(subset=["player_id"])
    df["player_id"] = df["player_id"].astype(int)

    _savant_expected_cache[season] = df
    return df


def _load_spin_data(season: int) -> pd.DataFrame:
    """Load fastball spin rates from Baseball Savant pitch arsenal."""
    if season in _spin_cache:
        return _spin_cache[season]

    pb = _safe_import_pybaseball()
    if pb is None:
        return pd.DataFrame()

    try:
        df = pb.statcast_pitcher_pitch_arsenal(season, minP=50, arsenal_type="avg_spin")
    except Exception as exc:
        logger.warning("statcast_pitcher_pitch_arsenal(%d) failed: %s", season, exc)
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    df["pitcher"] = pd.to_numeric(df["pitcher"], errors="coerce")
    df = df.dropna(subset=["pitcher"])
    df["pitcher"] = df["pitcher"].astype(int)

    # Fastball spin: prefer FF, then SI, then FC
    spin_cols = [c for c in ["ff_avg_spin", "si_avg_spin", "fc_avg_spin"] if c in df.columns]
    if spin_cols:
        df["fastball_spin"] = df[spin_cols].bfill(axis=1).iloc[:, 0]
    else:
        df["fastball_spin"] = float("nan")

    _spin_cache[season] = df
    return df


# ---------------------------------------------------------------------------
# Public: pitcher features
# ---------------------------------------------------------------------------

_pitcher_cache: dict[tuple, dict] = {}


def get_pitcher_fangraphs(
    pitcher_id: int,
    pitcher_name: str,
    season: int = CURRENT_SEASON,
) -> dict[str, float]:
    """
    Return FIP, xFIP (≈FIP), SIERA (≈FIP), K%, BB% for a pitcher.

    Primary source: Baseball Reference (FIP computed) + Baseball Savant (xERA).
    FanGraphs xFIP/SIERA are proxied with FIP (r > 0.90 at season level).
    """
    cache_key = (pitcher_id, pitcher_name, season)
    if cache_key in _pitcher_cache:
        return _pitcher_cache[cache_key]

    defaults = {
        "fip":      LEAGUE_AVG["sp_fip"],
        "xfip":     LEAGUE_AVG["sp_xfip"],
        "siera":    LEAGUE_AVG["sp_siera"],
        "k_pct":    LEAGUE_AVG["sp_k_pct"],
        "bb_pct":   LEAGUE_AVG["sp_bb_pct"],
        "_imputed": True,
    }

    bref = _load_bref_pitchers(season)

    row = None
    if not bref.empty and pitcher_id:
        match = bref[bref["mlbID"] == pitcher_id]
        if not match.empty:
            row = match.iloc[0]

    if row is None and pitcher_name and not bref.empty and "Name" in bref.columns:
        name_clean = pitcher_name.strip().lower()
        name_match = bref[bref["Name"].str.strip().str.lower() == name_clean]
        if name_match.empty:
            last = name_clean.split()[-1]
            name_match = bref[bref["Name"].str.lower().str.contains(last, na=False)]
        if not name_match.empty:
            row = name_match.iloc[0]

    if row is None:
        logger.debug("bref: no row for pitcher_id=%s name=%s season=%d", pitcher_id, pitcher_name, season)
        return defaults

    fip = float(np.clip(row.get("calc_fip", _FIP_CONST), 0.5, 10.0))

    result = {
        "fip":      fip,
        "xfip":     fip,           # proxy
        "siera":    fip,           # proxy
        "k_pct":    float(np.clip(row.get("calc_k_pct",  LEAGUE_AVG["sp_k_pct"]),  0.0, 0.6)),
        "bb_pct":   float(np.clip(row.get("calc_bb_pct", LEAGUE_AVG["sp_bb_pct"]), 0.0, 0.3)),
        "_imputed": False,
    }

    _pitcher_cache[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Public: pitcher Statcast expected stats (xERA + spin)
# ---------------------------------------------------------------------------

_savant_pitcher_cache: dict[tuple, dict] = {}


def get_pitcher_savant_extras(
    pitcher_id: int,
    season: int = CURRENT_SEASON,
) -> dict[str, float]:
    """
    Return xERA and fastball spin rate from Baseball Savant.
    Both are indexed by MLBAM pitcher_id.
    """
    cache_key = (pitcher_id, season)
    if cache_key in _savant_pitcher_cache:
        return _savant_pitcher_cache[cache_key]

    defaults = {
        "xera":          LEAGUE_AVG["sp_xera"],
        "fastball_spin": LEAGUE_AVG["sp_spin"],
        "_imputed":      True,
    }
    if not pitcher_id:
        return defaults

    # xERA
    expected = _load_savant_expected(season)
    xera = LEAGUE_AVG["sp_xera"]
    if not expected.empty:
        match = expected[expected["player_id"] == pitcher_id]
        if not match.empty:
            raw = match.iloc[0].get("xera", None)
            if raw is not None:
                try:
                    xera = float(np.clip(raw, 0.5, 10.0))
                except (TypeError, ValueError):
                    pass

    # Spin rate
    spin_df = _load_spin_data(season)
    spin = LEAGUE_AVG["sp_spin"]
    if not spin_df.empty:
        match = spin_df[spin_df["pitcher"] == pitcher_id]
        if not match.empty:
            raw = match.iloc[0].get("fastball_spin", None)
            if raw is not None and not (isinstance(raw, float) and np.isnan(raw)):
                try:
                    spin = float(np.clip(raw, 1500, 3500))
                except (TypeError, ValueError):
                    pass

    result = {
        "xera":          xera,
        "fastball_spin": spin,
        "_imputed":      xera == LEAGUE_AVG["sp_xera"] and spin == LEAGUE_AVG["sp_spin"],
    }
    _savant_pitcher_cache[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Public: bullpen workload
# ---------------------------------------------------------------------------

_bullpen_cache: dict[tuple, Any] = {}


def get_bullpen_workload(
    team_abbr: str,
    end_date: date,
    season: int = CURRENT_SEASON,
    team_id: int | None = None,
) -> dict[str, float]:
    """
    Returns estimated bullpen IP last 3 days and average leverage index.
    FanGraphs leverage index is unavailable; IP proxy is derived from bref.
    """
    cache_key = ("bullpen", team_abbr, season)
    if cache_key in _bullpen_cache:
        return _bullpen_cache[cache_key]

    defaults = {
        "bp_ip_3d": LEAGUE_AVG["bp_ip_3d"],
        "bp_li":    LEAGUE_AVG["bp_li"],
        "_imputed": True,
    }

    bref = _load_bref_pitchers(season)
    if bref.empty:
        return defaults

    # Filter to relievers for this team (GS == 0 or very low)
    team_col = "Tm"
    if team_col not in bref.columns:
        return defaults

    # Map common abbreviation differences between MLB API and BR
    _BR_MAP = {
        "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
        "CHC": "CHC", "CWS": "CHW", "CIN": "CIN", "CLE": "CLE",
        "COL": "COL", "DET": "DET", "HOU": "HOU", "KC":  "KCR",
        "LAA": "LAA", "LAD": "LAD", "MIA": "MIA", "MIL": "MIL",
        "MIN": "MIN", "NYM": "NYM", "NYY": "NYY", "OAK": "OAK",
        "PHI": "PHI", "PIT": "PIT", "SD":  "SDP", "SF":  "SFG",
        "SEA": "SEA", "STL": "STL", "TB":  "TBR", "TEX": "TEX",
        "TOR": "TOR", "WSH": "WSH",
    }
    br_abbr = _BR_MAP.get(team_abbr.upper(), team_abbr.upper())
    team_df = bref[bref[team_col].str.upper() == br_abbr]

    if team_df.empty:
        _bullpen_cache[cache_key] = defaults
        return defaults

    relievers = team_df[pd.to_numeric(team_df.get("GS", 0), errors="coerce").fillna(0) == 0]
    total_ip = relievers["IP"].sum()
    games = pd.to_numeric(team_df.get("G", 162), errors="coerce").max() or 162

    bp_ip_per_game = total_ip / games if games > 0 else LEAGUE_AVG["bp_ip_3d"] / 3
    season_ip_3d = round(float(bp_ip_per_game * 3), 2)

    # Real last-3-days IP from box scores (overrides season average)
    if team_id is not None:
        from fetchers.mlb_stats import get_bullpen_ip_3d
        real_ip_3d = get_bullpen_ip_3d(team_id, end_date)
        bp_ip_3d = real_ip_3d if real_ip_3d > 0 else season_ip_3d
    else:
        bp_ip_3d = season_ip_3d

    result = {
        "bp_ip_3d": bp_ip_3d,
        "bp_li":    LEAGUE_AVG["bp_li"],   # leverage index unavailable without FanGraphs
        "_imputed": False,
    }
    _bullpen_cache[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Public: team batting (MLB Stats API — OBP ≈ wOBA at team level)
# ---------------------------------------------------------------------------

_team_batting_mlb_cache: dict[tuple, Any] = {}
_team_id_map: dict[str, int] = {}  # abbr -> teamId

_MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "mlb-prediction-pipeline/1.0"})


def _ensure_team_id_map(season: int = CURRENT_SEASON) -> None:
    if _team_id_map:
        return
    try:
        r = _SESSION.get(
            f"{_MLB_API_BASE}/teams",
            params={"sportId": 1, "season": season},
            timeout=HTTP_TIMEOUT,
        )
        r.raise_for_status()
        for t in r.json().get("teams", []):
            _team_id_map[t["abbreviation"]] = t["id"]
    except Exception as exc:
        logger.warning("team ID map fetch failed: %s", exc)


@retry_with_backoff(retries=3, backoff_base=2, exceptions=_TRANSIENT)
def _fetch_mlb_team_hitting(team_id: int, season: int) -> dict:
    r = _SESSION.get(
        f"{_MLB_API_BASE}/teams/{team_id}/stats",
        params={"season": season, "group": "hitting", "stats": "season"},
        timeout=HTTP_TIMEOUT,
    )
    r.raise_for_status()
    stats = r.json().get("stats", [{}])[0].get("splits", [{}])
    return stats[0].get("stat", {}) if stats else {}


def get_team_batting(
    team_abbr: str,
    season: int = CURRENT_SEASON,
    end_date: "date | None" = None,
) -> dict[str, float]:
    """
    Return team-level wOBA (≈OBP season), 14-day rolling OPS, and 14-day
    RISP batting average from the MLB Stats API.
    """
    from datetime import date as _date
    from fetchers.mlb_stats import get_team_rolling_batting

    if end_date is None:
        end_date = _date.today()

    cache_key = ("team_bat", team_abbr, season, end_date)
    if cache_key in _team_batting_mlb_cache:
        return _team_batting_mlb_cache[cache_key]

    defaults = {
        "lineup_woba": LEAGUE_AVG["lineup_woba"],
        "ops_14d":     LEAGUE_AVG["ops_14d"],
        "risp_14d":    LEAGUE_AVG["risp_14d"],
        "_imputed":    True,
    }

    _ensure_team_id_map(season)
    team_id = _team_id_map.get(team_abbr)
    if team_id is None:
        logger.warning("No MLB team ID for '%s'", team_abbr)
        return defaults

    # Season-level OBP for wOBA proxy
    try:
        stats = _fetch_mlb_team_hitting(team_id, season)
    except Exception as exc:
        logger.warning("MLB API team hitting failed for %s: %s", team_abbr, exc)
        stats = {}

    def _f(key: str, fallback: float) -> float:
        v = stats.get(key)
        try:
            return float(v) if v is not None else fallback
        except (TypeError, ValueError):
            return fallback

    lineup_woba = _f("obp", LEAGUE_AVG["lineup_woba"])

    # 14-day rolling OPS and RISP from date-filtered API
    rolling = get_team_rolling_batting(team_id, end_date, days=14)

    result = {
        "lineup_woba": lineup_woba,
        "ops_14d":     rolling["ops_14d"],
        "risp_14d":    rolling["risp_14d"],
        "_imputed":    False,
    }
    _team_batting_mlb_cache[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Public: park factors
# ---------------------------------------------------------------------------

_park_factor_cache: dict[int, dict[str, dict]] = {}
_FANGRAPHS_GUTS_URL = "https://www.fangraphs.com/guts.aspx?type=pf&teamid=0&season={year}"


@retry_with_backoff(retries=3, backoff_base=2, exceptions=_TRANSIENT + (Exception,))
def _scrape_park_factors(season: int) -> dict[str, dict]:
    from bs4 import BeautifulSoup
    url = _FANGRAPHS_GUTS_URL.format(year=season)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")
    table = soup.find("table", {"class": lambda c: c and "rgMasterTable" in c})
    if table is None:
        tables = soup.find_all("table")
        table = tables[0] if tables else None
    if table is None:
        raise ValueError("park factors table not found on FanGraphs guts.aspx")

    results: dict[str, dict] = {}
    for row in table.find_all("tr")[1:]:
        cells = row.find_all("td")
        if len(cells) < 5:
            continue
        try:
            team = cells[0].get_text(strip=True)
            runs_pf = float(cells[3].get_text(strip=True))
            hr_pf   = float(cells[4].get_text(strip=True))
            results[team] = {"runs": runs_pf, "hr": hr_pf}
        except (ValueError, IndexError):
            continue
    return results


def get_park_factors(
    team_abbr: str,
    season: int = CURRENT_SEASON,
) -> dict[str, float]:
    """
    Return park factors for runs and HR.
    Order of preference:
      1. FanGraphs live scrape (may be blocked)
      2. PARK_FACTORS_HARDCODED from config
    """
    # Try live scrape (cached per season)
    if season not in _park_factor_cache:
        try:
            _park_factor_cache[season] = _scrape_park_factors(season)
            logger.info("Park factors scraped for season %d", season)
        except Exception as exc:
            logger.warning(
                "FanGraphs park factor scrape failed (season %d): %s — using hardcoded values",
                season, exc,
            )
            _park_factor_cache[season] = {}

    scraped = _park_factor_cache.get(season, {})
    for key, val in scraped.items():
        if team_abbr.upper() in key.upper():
            return {"park_factor_runs": val["runs"], "park_factor_hr": val["hr"], "_imputed": False}

    # Hardcoded fallback
    hc = PARK_FACTORS_HARDCODED.get(team_abbr.upper())
    if hc:
        return {"park_factor_runs": hc["runs"], "park_factor_hr": hc["hr"], "_imputed": False}

    logger.debug("No park factor for '%s'; using neutral", team_abbr)
    return {
        "park_factor_runs": LEAGUE_AVG["park_factor_runs"],
        "park_factor_hr":   LEAGUE_AVG["park_factor_hr"],
        "_imputed":         True,
    }
