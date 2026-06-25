"""
Feature assembler — combines all data sources into the 70-column feature
vector required for XGBoost inference.

Each game produces one row keyed by (game_id, cycle).
Missing features are tracked; games exceeding the threshold (>30% missing)
are excluded from inference (AC-10).
"""

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from config import (
    CURRENT_SEASON,
    DOME_TEAMS,
    FEATURE_COLUMNS,
    LEAGUE_AVG,
    MISSING_FEATURE_THRESHOLD,
    RETRACTABLE_ROOF_VENUES,
    VENUE_COORDS,
)
from fetchers import fangraphs, mlb_stats, savant, weather as weather_fetcher
from fetchers.fangraphs import get_pitcher_savant_extras
from fetchers.mlb_stats import (
    get_team_run_diff, get_pitcher_recent_form,
    get_team_win_pct, get_team_back_to_back, get_series_game_num,
    get_team_win_streak, get_team_bp_li_proxy, get_team_lineup_hand_pct,
    get_team_days_rest,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BvP lookup table (Career wOBA of batters vs this pitcher)
# Loaded once from a CSV if present; otherwise falls back to league average.
# Populate with historical data from Baseball Reference (training phase).
# ---------------------------------------------------------------------------

_BVP_TABLE: dict[int, float] = {}  # {pitcher_mlbam_id: mean_career_bvp_woba}

def _load_bvp_table() -> None:
    csv_path = Path(__file__).parent.parent / "data" / "bvp_woba.csv"
    if not csv_path.exists():
        return
    import csv
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                _BVP_TABLE[int(row["pitcher_id"])] = float(row["bvp_woba"])
            except (KeyError, ValueError):
                continue
    logger.info("BvP table loaded: %d pitchers", len(_BVP_TABLE))


_load_bvp_table()


# ---------------------------------------------------------------------------
# Days-rest calculator
# ---------------------------------------------------------------------------

def _compute_days_rest(pitcher_id: int, game_date: date) -> int:
    """
    Estimate days rest from the MLB Stats API game log for the pitcher.
    Falls back to 5 (league average) if the log is unavailable.
    """
    try:
        data = mlb_stats._get(
            f"/people/{pitcher_id}/stats",
            params={
                "stats":   "gameLog",
                "group":   "pitching",
                "season":  CURRENT_SEASON,
                "hydrate": "team",
            },
        )
        splits = data.get("stats", [{}])[0].get("splits", [])
        # Sort by date descending; find the last game started
        starts = [
            s for s in splits
            if s.get("stat", {}).get("gamesStarted", 0) > 0
        ]
        if not starts:
            return LEAGUE_AVG["sp_days_rest"]

        last_date_str = starts[-1].get("date", "")
        if not last_date_str:
            return LEAGUE_AVG["sp_days_rest"]

        last_date = datetime.strptime(last_date_str, "%Y-%m-%d").date()
        rest = (game_date - last_date).days
        return max(1, rest)
    except Exception as exc:
        logger.debug("days_rest fetch failed for pitcher %d: %s", pitcher_id, exc)
        return int(LEAGUE_AVG["sp_days_rest"])


# ---------------------------------------------------------------------------
# Handedness match
# ---------------------------------------------------------------------------

def _compute_hand_match_pct(
    pitcher_hand: str,
    batting_order: list[dict],
    opp_team_id: int | None = None,
    season: int = CURRENT_SEASON,
) -> float:
    """
    Fraction of the opposing batting order that bats the same hand as the pitcher.
    When no batting order is available, falls back to team roster handedness split.
    """
    if batting_order:
        known = [b for b in batting_order if b.get("hand", "?") != "?"]
        if known:
            same = sum(1 for b in known if b["hand"] == pitcher_hand)
            return round(same / len(known), 4)

    if opp_team_id:
        hands = get_team_lineup_hand_pct(opp_team_id, season)
        return hands["l_pct"] if pitcher_hand == "L" else hands["r_pct"]

    return LEAGUE_AVG["sp_hand_match_pct"]


# ---------------------------------------------------------------------------
# Single-pitcher feature block
# ---------------------------------------------------------------------------

def _build_pitcher_features(
    prefix: str,
    pitcher_id: int | None,
    pitcher_name: str,
    pitcher_hand: str,
    opp_batting_order: list[dict],
    game_date: date,
    season: int = CURRENT_SEASON,
    opp_team_id: int | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """
    Returns (feature_dict, missing_feature_names).
    `prefix` is 'home_sp' or 'away_sp'.
    """
    feats: dict[str, Any] = {}
    missing: list[str] = []

    # --- Statcast contact-quality block (barrel%, exit velo, hard-hit%) ---
    if pitcher_id:
        sc = savant.get_pitcher_statcast(pitcher_id, season)
        fg = fangraphs.get_pitcher_fangraphs(pitcher_id, pitcher_name, season)
        sv = get_pitcher_savant_extras(pitcher_id, season)  # xERA, spin
    else:
        sc = {"_imputed": True}
        fg = {"_imputed": True}
        sv = {"_imputed": True}

    def _sc(key: str, fallback_key: str) -> float:
        val = sc.get(key)
        if val is None:
            missing.append(f"{prefix}_{fallback_key}")
            return LEAGUE_AVG[f"sp_{fallback_key}"]
        return float(val)

    def _fg(key: str, fallback_key: str) -> float:
        val = fg.get(key)
        if val is None:
            missing.append(f"{prefix}_{fallback_key}")
            return LEAGUE_AVG[f"sp_{fallback_key}"]
        return float(val)

    def _sv(key: str, fallback_key: str) -> float:
        val = sv.get(key)
        if val is None:
            missing.append(f"{prefix}_{fallback_key}")
            return LEAGUE_AVG[f"sp_{fallback_key}"]
        return float(val)

    feats[f"{prefix}_xera"]          = _sv("xera",          "xera")
    feats[f"{prefix}_fip"]           = _fg("fip",           "fip")
    feats[f"{prefix}_xfip"]          = _fg("xfip",          "xfip")
    feats[f"{prefix}_siera"]         = _fg("siera",         "siera")
    feats[f"{prefix}_k_pct"]         = _fg("k_pct",         "k_pct")
    feats[f"{prefix}_bb_pct"]        = _fg("bb_pct",        "bb_pct")
    feats[f"{prefix}_barrel"]        = _sc("barrel_pct",    "barrel")
    feats[f"{prefix}_hh_pct"]        = _sc("hard_hit_pct",  "hh_pct")
    feats[f"{prefix}_exit_velo"]     = _sc("avg_exit_velo", "exit_velo")
    feats[f"{prefix}_spin"]          = _sv("fastball_spin", "spin")

    # Days rest
    if pitcher_id:
        feats[f"{prefix}_days_rest"] = _compute_days_rest(pitcher_id, game_date)
    else:
        feats[f"{prefix}_days_rest"] = int(LEAGUE_AVG["sp_days_rest"])
        missing.append(f"{prefix}_days_rest")

    # Recent form — last 3 starts ERA and WHIP
    if pitcher_id:
        rf = get_pitcher_recent_form(pitcher_id, game_date, n_starts=3)
        feats[f"{prefix}_era_l3"]  = rf["era_l3"]
        feats[f"{prefix}_whip_l3"] = rf["whip_l3"]
    else:
        feats[f"{prefix}_era_l3"]  = LEAGUE_AVG["sp_era_l3"]
        feats[f"{prefix}_whip_l3"] = LEAGUE_AVG["sp_whip_l3"]
        missing.append(f"{prefix}_era_l3")
        missing.append(f"{prefix}_whip_l3")

    # xERA delta — regression-to-mean signal (positive = lucky, due for regression)
    feats[f"{prefix}_xera_delta"] = round(
        float(feats[f"{prefix}_xera"]) - float(feats[f"{prefix}_fip"]), 3
    )

    # Handedness match (pitcher vs opposing lineup)
    feats[f"{prefix}_hand_match_pct"] = _compute_hand_match_pct(
        pitcher_hand, opp_batting_order, opp_team_id=opp_team_id, season=season
    )

    # BvP wOBA
    bvp = _BVP_TABLE.get(pitcher_id, None) if pitcher_id else None
    if bvp is None:
        feats[f"{prefix}_bvp_woba"] = LEAGUE_AVG["sp_bvp_woba"]
        # Not counted as missing — expected to use fallback when no lookup data
    else:
        feats[f"{prefix}_bvp_woba"] = float(bvp)

    return feats, missing


# ---------------------------------------------------------------------------
# Single-team bullpen feature block
# ---------------------------------------------------------------------------

def _build_bullpen_features(
    prefix: str,
    team_abbr: str,
    team_id: int,
    game_date: date,
    season: int = CURRENT_SEASON,
) -> tuple[dict[str, Any], list[str]]:
    """
    Returns (feature_dict, missing_feature_names).
    `prefix` is 'home_bp' or 'away_bp'.
    """
    feats: dict[str, Any] = {}
    missing: list[str] = []

    # xERA from Statcast
    sc_bp = savant.get_bullpen_statcast(team_abbr, game_date, days=7)
    if sc_bp.get("_imputed"):
        missing.append(f"{prefix}_xera")
    feats[f"{prefix}_xera"] = sc_bp.get("bp_xera", LEAGUE_AVG["bp_xera"])

    # IP (real last-3-days via box scores) and leverage index
    fg_bp = fangraphs.get_bullpen_workload(team_abbr, game_date, season, team_id=team_id)
    if fg_bp.get("_imputed"):
        missing.append(f"{prefix}_ip_3d")
    feats[f"{prefix}_ip_3d"] = fg_bp.get("bp_ip_3d", LEAGUE_AVG["bp_ip_3d"])
    # LI proxy from MLB API saves/holds/blown saves (FanGraphs LI unavailable)
    feats[f"{prefix}_li"] = get_team_bp_li_proxy(team_id, season)

    # Relievers on IL
    try:
        feats[f"{prefix}_il_ct"] = mlb_stats.count_il_players(
            team_id, game_date, position_filter="P"
        )
    except Exception as exc:
        logger.warning("bullpen IL count failed for team %d: %s", team_id, exc)
        feats[f"{prefix}_il_ct"] = int(LEAGUE_AVG["bp_il_ct"])
        missing.append(f"{prefix}_il_ct")

    return feats, missing


# ---------------------------------------------------------------------------
# Single-team lineup / offense feature block
# ---------------------------------------------------------------------------

def _build_lineup_features(
    prefix: str,
    team_abbr: str,
    team_id: int,
    game_date: date,
    batting_order: list[dict],
    season: int = CURRENT_SEASON,
    opp_abbr: str = "",
) -> tuple[dict[str, Any], list[str]]:
    feats: dict[str, Any] = {}
    missing: list[str] = []

    fg_bat = fangraphs.get_team_batting(team_abbr, season, end_date=game_date)
    if fg_bat.get("_imputed"):
        missing += [f"{prefix}_lineup_woba", f"{prefix}_ops_14d", f"{prefix}_risp_14d"]

    feats[f"{prefix}_lineup_woba"] = fg_bat.get("lineup_woba", LEAGUE_AVG["lineup_woba"])
    feats[f"{prefix}_ops_14d"]     = fg_bat.get("ops_14d",     LEAGUE_AVG["ops_14d"])
    feats[f"{prefix}_risp_14d"]    = fg_bat.get("risp_14d",    LEAGUE_AVG["risp_14d"])

    # Starters on IL (non-pitchers)
    try:
        feats[f"{prefix}_starters_il"] = mlb_stats.count_il_players(
            team_id, game_date, position_filter=None
        )
    except Exception as exc:
        logger.warning("starters IL count failed for team %d: %s", team_id, exc)
        feats[f"{prefix}_starters_il"] = int(LEAGUE_AVG["starters_il"])
        missing.append(f"{prefix}_starters_il")

    # Season-to-date run differential per game from accumulated results
    feats[f"{prefix}_run_diff"] = get_team_run_diff(team_abbr, game_date)

    # Win%, back-to-back, series game#, win/loss streak — from DuckDB results table
    feats[f"{prefix}_win_pct"]      = get_team_win_pct(team_abbr, game_date)
    feats[f"{prefix}_back_to_back"] = get_team_back_to_back(team_abbr, game_date)
    feats[f"{prefix}_series_game"]  = (
        get_series_game_num(team_abbr, opp_abbr, game_date) if opp_abbr else int(LEAGUE_AVG["series_game"])
    )
    feats[f"{prefix}_win_streak"]      = get_team_win_streak(team_abbr, game_date)
    feats[f"{prefix}_team_days_rest"]  = get_team_days_rest(team_abbr, game_date)

    return feats, missing


# ---------------------------------------------------------------------------
# Park and weather feature block
# ---------------------------------------------------------------------------

def _build_park_weather_features(
    home_team: str,
    venue_name: str,
    game_datetime_utc: str | None,
    cycle: str,
) -> tuple[dict[str, Any], list[str]]:
    feats: dict[str, Any] = {}
    missing: list[str] = []

    pf = fangraphs.get_park_factors(home_team)
    if pf.get("_imputed"):
        missing += ["park_factor_runs", "park_factor_hr"]
    feats["park_factor_runs"] = pf.get("park_factor_runs", LEAGUE_AVG["park_factor_runs"])
    feats["park_factor_hr"]   = pf.get("park_factor_hr",   LEAGUE_AVG["park_factor_hr"])

    # Weather only in Cycle B (FR-09); Cycle A uses league averages
    if cycle == "B":
        venue_info = VENUE_COORDS.get(venue_name, {})
        lat = venue_info.get("lat")
        lon = venue_info.get("lon")
        roof_closed = venue_name in RETRACTABLE_ROOF_VENUES

        wx = weather_fetcher.get_game_weather(
            lat=lat,
            lon=lon,
            game_datetime_utc=game_datetime_utc,
            venue_name=venue_name,
            roof_closed=roof_closed,
        )
        if wx.get("_imputed"):
            missing += ["wind_speed", "wind_dir_deg", "temperature"]
    else:
        wx = {
            "wind_speed":   LEAGUE_AVG["wind_speed"],
            "wind_dir":     "N",
            "wind_dir_deg": LEAGUE_AVG["wind_dir_deg"],
            "temperature":  LEAGUE_AVG["temperature"],
            "_imputed":     True,
        }

    feats["wind_speed"]   = wx["wind_speed"]
    feats["wind_dir"]     = wx["wind_dir"]       # stored in features table (text)
    feats["wind_dir_deg"] = wx["wind_dir_deg"]   # used by model
    feats["temperature"]  = wx["temperature"]
    feats["is_dome"]      = 1 if home_team in DOME_TEAMS else 0

    return feats, missing


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def assemble_game_features(
    game: dict,
    cycle: str,
    home_lineup: list[dict] | None = None,
    away_lineup: list[dict] | None = None,
    home_pitcher_override: dict | None = None,
    away_pitcher_override: dict | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    """
    Build the full feature row for a single game.

    Returns:
        (feature_row, missing_columns) — feature_row is None when the game
        is excluded due to excessive missing features (AC-10).

    Parameters:
        game: a game dict from mlb_stats.get_schedule()
        cycle: 'A' or 'B'
        home_lineup / away_lineup: confirmed batting orders (Cycle B)
        home_pitcher_override / away_pitcher_override: dicts with
            {id, name, hand} for confirmed starters (Cycle B)
    """
    game_date_str = game["game_date"]
    game_date = datetime.strptime(game_date_str, "%Y-%m-%d").date()
    game_id = game["game_id"]

    home_team  = game["home_team"]
    home_team_id = game["home_team_id"]
    away_team  = game["away_team"]
    away_team_id = game["away_team_id"]
    venue_name = game.get("venue_name", "")
    game_datetime_utc = game.get("game_datetime_utc")

    all_missing: list[str] = []

    # --- Resolve starters ---
    home_sp_id   = (home_pitcher_override or {}).get("id") or game.get("home_probable_id")
    home_sp_name = (home_pitcher_override or {}).get("name", "")
    home_sp_hand = (home_pitcher_override or {}).get("hand", "R")

    away_sp_id   = (away_pitcher_override or {}).get("id") or game.get("away_probable_id")
    away_sp_name = (away_pitcher_override or {}).get("name", "")
    away_sp_hand = (away_pitcher_override or {}).get("hand", "R")

    # Enrich names/hands from MLB API if not provided
    if home_sp_id and not home_sp_name:
        try:
            p = mlb_stats.get_player(home_sp_id)
            home_sp_name = p.get("full_name", "")
            home_sp_hand = p.get("pitch_hand", "R")
        except Exception:
            pass

    if away_sp_id and not away_sp_name:
        try:
            p = mlb_stats.get_player(away_sp_id)
            away_sp_name = p.get("full_name", "")
            away_sp_hand = p.get("pitch_hand", "R")
        except Exception:
            pass

    # --- Build blocks ---
    # home SP faces the away lineup; away SP faces the home lineup
    home_sp_feats, home_sp_missing = _build_pitcher_features(
        "home_sp", home_sp_id, home_sp_name, home_sp_hand,
        away_lineup or [], game_date,
        opp_team_id=away_team_id,
    )
    all_missing.extend(home_sp_missing)

    away_sp_feats, away_sp_missing = _build_pitcher_features(
        "away_sp", away_sp_id, away_sp_name, away_sp_hand,
        home_lineup or [], game_date,
        opp_team_id=home_team_id,
    )
    all_missing.extend(away_sp_missing)

    home_bp_feats, home_bp_missing = _build_bullpen_features(
        "home_bp", home_team, home_team_id, game_date,
    )
    all_missing.extend(home_bp_missing)

    away_bp_feats, away_bp_missing = _build_bullpen_features(
        "away_bp", away_team, away_team_id, game_date,
    )
    all_missing.extend(away_bp_missing)

    home_lu_feats, home_lu_missing = _build_lineup_features(
        "home", home_team, home_team_id, game_date, home_lineup or [],
        opp_abbr=away_team,
    )
    all_missing.extend(home_lu_missing)

    away_lu_feats, away_lu_missing = _build_lineup_features(
        "away", away_team, away_team_id, game_date, away_lineup or [],
        opp_abbr=home_team,
    )
    all_missing.extend(away_lu_missing)

    park_wx_feats, park_wx_missing = _build_park_weather_features(
        home_team, venue_name, game_datetime_utc, cycle,
    )
    all_missing.extend(park_wx_missing)

    # --- Umpire run factor (career avg runs / 9.0 league avg) ---
    umpire_run_factor = LEAGUE_AVG["umpire_run_factor"]
    try:
        ump_assignments = mlb_stats.get_umpire_assignments(game_date)
        ump_info = ump_assignments.get(str(game_id), {})
        ump_name = ump_info.get("name", "TBD")
        if ump_name and ump_name != "TBD":
            ump_stats = mlb_stats.get_umpire_career_stats(ump_name)
            avg_runs = ump_stats.get("avg_runs")
            if avg_runs:
                umpire_run_factor = round(float(avg_runs) / 9.0, 3)
    except Exception as exc:
        logger.debug("umpire feature failed for game %s: %s", game_id, exc)

    # --- Combine ---
    row: dict[str, Any] = {
        "game_id":    game_id,
        "game_date":  game_date_str,
        "cycle":      cycle,
        "home_team":  home_team,
        "away_team":  away_team,
    }
    for block in (
        home_sp_feats, away_sp_feats,
        home_bp_feats, away_bp_feats,
        home_lu_feats, away_lu_feats,
        park_wx_feats,
    ):
        row.update(block)
    row["umpire_run_factor"] = umpire_run_factor
    # Market line default — overwritten with real sportsbook line at inference time
    row["ou_line"] = LEAGUE_AVG["ou_line"]

    # --- Missing-feature gate (AC-10 / Section 5.4) ---
    total_features = len(FEATURE_COLUMNS)
    missing_count = len(set(all_missing))
    missing_pct = missing_count / total_features

    if missing_pct > MISSING_FEATURE_THRESHOLD:
        logger.warning(
            "game %s excluded: %.0f%% features missing (%d/%d): %s",
            game_id, missing_pct * 100, missing_count, total_features,
            ", ".join(sorted(set(all_missing))),
        )
        return None, list(set(all_missing))

    if all_missing:
        logger.debug(
            "game %s: %d imputed features: %s",
            game_id, len(all_missing), ", ".join(sorted(set(all_missing))),
        )

    return row, list(set(all_missing))
