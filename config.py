"""
Central configuration for the MLB Game Prediction System.
All path defaults can be overridden via environment variables.
"""

import datetime
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DB_PATH = os.environ.get("MLB_DB_PATH", str(BASE_DIR / "mlb_predictions.duckdb"))
MODEL_DIR = Path(os.environ.get("MLB_MODEL_DIR", str(BASE_DIR / "models")))
LOG_DIR = Path(os.environ.get("MLB_LOG_DIR", str(BASE_DIR / "logs")))

WIN_MODEL_PATH        = str(MODEL_DIR / "xgb_win_prob.json")
TOTAL_MODEL_PATH      = str(MODEL_DIR / "xgb_run_total.json")
OU_MODEL_PATH         = str(MODEL_DIR / "xgb_ou_prob.json")
CALIBRATOR_PATH       = str(MODEL_DIR / "win_prob_calibrator.pkl")

_version_path = MODEL_DIR / "version.txt"
MODEL_VERSION = (
    _version_path.read_text().strip()
    if _version_path.exists()
    else os.environ.get("MLB_MODEL_VERSION", "v1.0")
)

# ---------------------------------------------------------------------------
# External API endpoints
# ---------------------------------------------------------------------------
MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"
ROTOWIRE_LINEUPS_URL = "https://www.rotowire.com/baseball/daily-lineups.php"

# Optional – moneyline odds (set MLB_ODDS_API_KEY env var to enable)
ODDS_API_KEY = os.environ.get("MLB_ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# ---------------------------------------------------------------------------
# Email notifications  (utils/notifier.py)
# Set MLB_SMTP_USER and MLB_SMTP_PASS to enable; leave blank to disable.
# ---------------------------------------------------------------------------
SMTP_HOST     = "smtp-mail.outlook.com"
SMTP_PORT     = 587
SMTP_USER     = os.environ.get("MLB_SMTP_USER", "")
SMTP_PASS     = os.environ.get("MLB_SMTP_PASS", "")
NOTIFY_TO     = os.environ.get("MLB_NOTIFY_TO", "ryberin@hotmail.com")

# ---------------------------------------------------------------------------
# HTTP retry settings  (NFR-03: 3 retries, 2/4/8s backoff)
# ---------------------------------------------------------------------------
HTTP_RETRIES = 3
HTTP_BACKOFF_BASE = 2  # seconds; delay = backoff_base ** (attempt + 1)
HTTP_TIMEOUT = 30      # seconds per request

# Post-game actuals: more aggressive retry because games end late
POSTgame_RETRIES = 6
POSTAGE_BACKOFF_BASE = 20  # 20s / 40s / 80s / 160s / 320s / 640s

# RotoWire scrape: longer delay between retries (rate-limit avoidance)
ROTOWIRE_RETRIES = 3
ROTOWIRE_RETRY_DELAY = 30  # seconds, not exponential

# ---------------------------------------------------------------------------
# Feature completeness threshold  (Section 5.4 / AC-10)
# Games with >30% features missing are excluded from inference.
# ---------------------------------------------------------------------------
MISSING_FEATURE_THRESHOLD = 0.30

# ---------------------------------------------------------------------------
# Current season
# ---------------------------------------------------------------------------
CURRENT_SEASON = datetime.date.today().year

# ---------------------------------------------------------------------------
# Retractable-roof stadiums  (FR-10 / Appendix A)
# Weather features are skipped when the roof is confirmed closed.
# ---------------------------------------------------------------------------
RETRACTABLE_ROOF_VENUES = {
    "Minute Maid Park",       # HOU
    "Daikin Park",            # HOU (renamed 2026)
    "Globe Life Field",       # TEX
    "T-Mobile Park",          # SEA
    "American Family Field",  # MIL
    "Chase Field",            # ARI
    "loanDepot park",         # MIA
    "Rogers Centre",          # TOR
    "Tropicana Field",        # TB
}

# ---------------------------------------------------------------------------
# Venue reference table  (Appendix A)
# Populated from MLB Stats API /venues; kept here as a static fallback so
# the pipeline does not need an extra API call per game.
# ---------------------------------------------------------------------------
VENUE_COORDS: dict[str, dict] = {
    "Wrigley Field":              {"lat": 41.9484,  "lon": -87.6553,  "team": "CHC"},
    "Oracle Park":                {"lat": 37.7786,  "lon": -122.3893, "team": "SF"},
    "Coors Field":                {"lat": 39.7559,  "lon": -104.9942, "team": "COL"},
    "Petco Park":                 {"lat": 32.7073,  "lon": -117.1566, "team": "SD"},
    "Minute Maid Park":           {"lat": 29.7573,  "lon": -95.3555,  "team": "HOU"},
    "Daikin Park":                {"lat": 29.7573,  "lon": -95.3555,  "team": "HOU"},  # renamed 2026
    "Globe Life Field":           {"lat": 32.7474,  "lon": -97.0832,  "team": "TEX"},
    "T-Mobile Park":              {"lat": 47.5914,  "lon": -122.3325, "team": "SEA"},
    "Fenway Park":                {"lat": 42.3467,  "lon": -71.0972,  "team": "BOS"},
    "Yankee Stadium":             {"lat": 40.8296,  "lon": -73.9262,  "team": "NYY"},
    "Dodger Stadium":             {"lat": 34.0739,  "lon": -118.2400, "team": "LAD"},
    "UNIQLO Field at Dodger Stadium": {"lat": 34.0739, "lon": -118.2400, "team": "LAD"},  # renamed 2026
    "Angel Stadium":              {"lat": 33.8003,  "lon": -117.8827, "team": "LAA"},
    "Busch Stadium":              {"lat": 38.6226,  "lon": -90.1928,  "team": "STL"},
    "Truist Park":                {"lat": 33.8907,  "lon": -84.4677,  "team": "ATL"},
    "PNC Park":                   {"lat": 40.4469,  "lon": -80.0057,  "team": "PIT"},
    "Great American Ball Park":   {"lat": 39.0979,  "lon": -84.5082,  "team": "CIN"},
    "Progressive Field":          {"lat": 41.4962,  "lon": -81.6852,  "team": "CLE"},
    "Comerica Park":              {"lat": 42.3390,  "lon": -83.0485,  "team": "DET"},
    "Target Field":               {"lat": 44.9817,  "lon": -93.2778,  "team": "MIN"},
    "Guaranteed Rate Field":      {"lat": 41.8300,  "lon": -87.6339,  "team": "CWS"},
    "Rate Field":                 {"lat": 41.8300,  "lon": -87.6339,  "team": "CWS"},
    "Kauffman Stadium":           {"lat": 39.0517,  "lon": -94.4803,  "team": "KC"},
    "Oakland Coliseum":           {"lat": 37.7516,  "lon": -122.2005, "team": "OAK"},
    "Citizens Bank Park":         {"lat": 39.9061,  "lon": -75.1665,  "team": "PHI"},
    "Citi Field":                 {"lat": 40.7571,  "lon": -73.8458,  "team": "NYM"},
    "Nationals Park":             {"lat": 38.8730,  "lon": -77.0074,  "team": "WSH"},
    "loanDepot park":             {"lat": 25.7781,  "lon": -80.2197,  "team": "MIA"},
    "American Family Field":      {"lat": 43.0280,  "lon": -87.9712,  "team": "MIL"},
    "Chase Field":                {"lat": 33.4455,  "lon": -112.0667, "team": "ARI"},
    "Rogers Centre":              {"lat": 43.6414,  "lon": -79.3894,  "team": "TOR"},
    "Tropicana Field":            {"lat": 27.7682,  "lon": -82.6534,  "team": "TB"},
    "Camden Yards":               {"lat": 39.2838,  "lon": -76.6218,  "team": "BAL"},
    "Sutter Health Park":         {"lat": 38.5779,  "lon": -121.5005, "team": "ATH"},
}

# ---------------------------------------------------------------------------
# Feature column ordering
# Must exactly match the column order used when training the XGBoost models.
# wind_dir is stored as text in the features table; wind_dir_deg (numeric
# degrees 0-360) is the model input derived from it.
# ---------------------------------------------------------------------------
FEATURE_COLUMNS: list[str] = [
    # Home starting pitcher (16)
    "home_sp_xera",    "home_sp_fip",      "home_sp_xfip",       "home_sp_siera",
    "home_sp_k_pct",   "home_sp_bb_pct",   "home_sp_barrel",     "home_sp_hh_pct",
    "home_sp_exit_velo","home_sp_spin",    "home_sp_days_rest",
    "home_sp_hand_match_pct",              "home_sp_bvp_woba",
    "home_sp_era_l3",  "home_sp_whip_l3",  "home_sp_xera_delta",
    # Away starting pitcher (16)
    "away_sp_xera",    "away_sp_fip",      "away_sp_xfip",       "away_sp_siera",
    "away_sp_k_pct",   "away_sp_bb_pct",   "away_sp_barrel",     "away_sp_hh_pct",
    "away_sp_exit_velo","away_sp_spin",    "away_sp_days_rest",
    "away_sp_hand_match_pct",              "away_sp_bvp_woba",
    "away_sp_era_l3",  "away_sp_whip_l3",  "away_sp_xera_delta",
    # Home bullpen (4)
    "home_bp_xera",    "home_bp_ip_3d",    "home_bp_li",         "home_bp_il_ct",
    # Away bullpen (4)
    "away_bp_xera",    "away_bp_ip_3d",    "away_bp_li",         "away_bp_il_ct",
    # Home lineup/offense (10)
    "home_lineup_woba","home_ops_14d",     "home_risp_14d",
    "home_starters_il","home_run_diff",
    "home_win_pct",    "home_back_to_back","home_series_game",
    "home_win_streak", "home_team_days_rest",
    # Away lineup/offense (10)
    "away_lineup_woba","away_ops_14d",     "away_risp_14d",
    "away_starters_il","away_run_diff",
    "away_win_pct",    "away_back_to_back","away_series_game",
    "away_win_streak", "away_team_days_rest",
    # Park, weather and umpire (7 numeric)
    "park_factor_runs","park_factor_hr",
    "wind_speed",      "wind_dir_deg",     "temperature",
    "umpire_run_factor","is_dome",
    # Run-environment composites — derived at train/inference time (3)
    "sum_sp_era_l3",   "sum_ops_14d",      "avg_sp_k_pct",
]

OU_FEATURE_COLUMNS = FEATURE_COLUMNS + ["ou_line"]  # 54 features for the OU classifier

# ---------------------------------------------------------------------------
# League-average fallback values
# Used when a feature fetch fails and the game still has enough coverage
# to remain above the missing-feature threshold.
# ---------------------------------------------------------------------------
LEAGUE_AVG: dict[str, float] = {
    "sp_xera":          4.10,
    "sp_fip":           4.10,
    "sp_xfip":          4.10,
    "sp_siera":         4.10,
    "sp_k_pct":         0.225,
    "sp_bb_pct":        0.082,
    "sp_barrel":        0.080,
    "sp_hh_pct":        0.370,
    "sp_exit_velo":    88.5,
    "sp_spin":       2300.0,
    "sp_days_rest":     5,
    "sp_hand_match_pct":0.50,
    "sp_bvp_woba":      0.320,
    "sp_era_l3":        4.20,
    "sp_whip_l3":       1.28,
    "bp_xera":          4.20,
    "bp_ip_3d":         9.0,
    "bp_li":            1.00,
    "bp_il_ct":         1,
    "lineup_woba":      0.320,
    "ops_14d":          0.720,
    "risp_14d":         0.255,
    "starters_il":      1,
    "run_diff":         0.0,
    "win_pct":          0.500,
    "back_to_back":     0,
    "series_game":      2,
    "win_streak":       0,
    "team_days_rest":   1,
    "sp_xera_delta":    0.0,
    "is_dome":          0,
    "umpire_run_factor":1.0,
    "sum_sp_era_l3":    8.40,   # 2 × league-avg 4.20
    "sum_ops_14d":      1.44,   # 2 × league-avg 0.720
    "avg_sp_k_pct":     0.225,
    "park_factor_runs":100.0,
    "park_factor_hr":  100.0,
    "wind_speed":        5.0,
    "wind_dir_deg":      0.0,
    "temperature":      72.0,
}

# ---------------------------------------------------------------------------
# OU model training threshold
# Minimum games with a real ou_line before training XGBoost OU classifier.
# Logistic fallback is used below this threshold.
# ---------------------------------------------------------------------------
OU_MIN_GAMES = 200

# Teams that play in dome or retractable-roof stadiums (weather-neutral).
# Used to populate the is_dome feature in both training data and inference.
DOME_TEAMS: set[str] = {"HOU", "TEX", "SEA", "MIL", "ARI", "MIA", "TOR", "TB"}

# ---------------------------------------------------------------------------
# Park factors — 3-year average (runs, HR), base 100
# Sourced from FanGraphs historical data; updated when FanGraphs is reachable.
# Used as a static fallback when the live scrape fails (Assumption A-01).
# ---------------------------------------------------------------------------
PARK_FACTORS_HARDCODED: dict[str, dict[str, float]] = {
    "COL": {"runs": 111.0, "hr": 120.0},  # Coors Field
    "CIN": {"runs": 105.0, "hr": 109.0},
    "PHI": {"runs": 102.0, "hr": 105.0},
    "BOS": {"runs": 104.0, "hr": 102.0},
    "BAL": {"runs": 104.0, "hr": 108.0},
    "CWS": {"runs": 104.0, "hr": 109.0},
    "ARI": {"runs": 104.0, "hr": 105.0},
    "TEX": {"runs": 103.0, "hr": 104.0},
    "TOR": {"runs": 103.0, "hr": 106.0},
    "CHC": {"runs": 102.0, "hr": 104.0},
    "NYY": {"runs": 102.0, "hr": 112.0},
    "MIL": {"runs": 101.0, "hr": 103.0},
    "ATL": {"runs": 101.0, "hr": 101.0},
    "WSH": {"runs": 101.0, "hr": 102.0},
    "HOU": {"runs":  99.0, "hr": 101.0},
    "LAA": {"runs": 100.0, "hr": 100.0},
    "KC":  {"runs":  99.0, "hr":  98.0},
    "STL": {"runs":  98.0, "hr":  96.0},
    "LAD": {"runs":  98.0, "hr":  99.0},
    "DET": {"runs":  98.0, "hr":  96.0},
    "PIT": {"runs":  98.0, "hr":  97.0},
    "CLE": {"runs":  97.0, "hr":  96.0},
    "MIN": {"runs":  97.0, "hr":  98.0},
    "NYM": {"runs":  97.0, "hr":  94.0},
    "SEA": {"runs":  97.0, "hr":  94.0},
    "TB":  {"runs":  96.0, "hr":  93.0},
    "MIA": {"runs":  96.0, "hr":  89.0},
    "OAK": {"runs":  95.0, "hr":  88.0},
    "ATH": {"runs":  96.0, "hr":  90.0},  # Sacramento (Sutter Health Park, est. 2025)
    "SF":  {"runs":  93.0, "hr":  89.0},
    "SD":  {"runs":  93.0, "hr":  89.0},
}
