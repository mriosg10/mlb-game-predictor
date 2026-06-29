"""
Historical training data builder — v2 (FanGraphs-free).

Known limitations:
  ou_line is absent from historical data (The Odds API wasn't scraped historically).
  build_feature_row() omits it; FEATURE_COLUMNS fills it with LEAGUE_AVG["ou_line"]
  during training.  The live pipeline writes a real sportsbook line at Cycle B time,
  so there is a train/serve skew for ou_line in the early weeks of each season.

Data sources (all confirmed working as of 2026):
  Pitcher quality: Baseball Reference (pitching_stats_bref) for FIP/K%/BB%
                   Baseball Savant (statcast_pitcher_expected_stats) for xERA
                   Baseball Savant (statcast_pitcher_exitvelo_barrels) for barrel%/exit_velo
                   Baseball Savant (statcast_pitcher_pitch_arsenal) for fastball spin
  Team batting:    MLB Stats API /teams/{id}/stats?group=hitting
  Park factors:    PARK_FACTORS_HARDCODED from config (FanGraphs 403)
  Game results:    MLB Stats API /schedule with linescore hydration

Proxy notes:
  - xFIP = FIP (r ≈ 0.94 at season level; SIERA same)
  - wOBA ≈ OBP (calibrated to match at league level)
  - Leverage index = league average (FanGraphs-only metric, unavailable)

Usage:
    python scripts/build_training_data.py [--seasons 2022,2023,2024]
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import requests

from config import DOME_TEAMS, FEATURE_COLUMNS, LEAGUE_AVG, PARK_FACTORS_HARDCODED, VENUE_COORDS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR  = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

MLB_BASE    = "https://statsapi.mlb.com/api/v1"
HTTP_TIMEOUT = 30
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "mlb-training-data-builder/2.0"})

_FIP_CONST = 3.10


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cp(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"

def _load(key: str):
    p = _cp(key)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)

def _save(key: str, data) -> None:
    with open(_cp(key), "w") as f:
        json.dump(data, f)

def _mlb_get(path: str, params: dict = None, retries: int = 3) -> dict:
    url = f"{MLB_BASE}{path}"
    for attempt in range(retries):
        try:
            r = _SESSION.get(url, params=params, timeout=HTTP_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** (attempt + 1)
            logger.warning("MLB API retry %d: %s (%ds)", attempt + 1, e, wait)
            time.sleep(wait)

def _pb():
    import pybaseball as pb
    pb.cache.enable()
    return pb


# ---------------------------------------------------------------------------
# 1. Season schedule + results
# ---------------------------------------------------------------------------

def fetch_season_schedule(season: int) -> list[dict]:
    cached = _load(f"schedule_{season}")
    if cached:
        logger.info("Schedule cache hit for %d (%d games)", season, len(cached))
        return cached

    logger.info("Fetching %d schedule from MLB Stats API...", season)
    games = []
    month_ranges = [
        (f"{season}-03-25", f"{season}-04-30"),
        (f"{season}-05-01", f"{season}-05-31"),
        (f"{season}-06-01", f"{season}-06-30"),
        (f"{season}-07-01", f"{season}-07-31"),
        (f"{season}-08-01", f"{season}-08-31"),
        (f"{season}-09-01", f"{season}-09-30"),
        (f"{season}-10-01", f"{season}-10-10"),
    ]
    for start, end in month_ranges:
        try:
            data = _mlb_get("/schedule", params={
                "sportId": 1, "startDate": start, "endDate": end,
                "gameType": "R", "hydrate": "probablePitcher,linescore,team",
            })
        except Exception as e:
            logger.warning("schedule %s-%s failed: %s", start, end, e)
            continue
        for day in data.get("dates", []):
            for g in day.get("games", []):
                if g.get("status", {}).get("abstractGameState") != "Final":
                    continue
                ls = g.get("linescore", {}).get("teams", {})
                home_r = ls.get("home", {}).get("runs")
                away_r = ls.get("away", {}).get("runs")
                if home_r is None or away_r is None:
                    continue
                _gdt = g.get("gameDate", "")
                games.append({
                    "game_id":            str(g["gamePk"]),
                    "game_date":          _gdt[:10],
                    "game_time_utc_hour": int(_gdt[11:13]) if len(_gdt) > 13 else 23,
                    "season":             season,
                    "home_team":    g["teams"]["home"]["team"]["abbreviation"],
                    "away_team":    g["teams"]["away"]["team"]["abbreviation"],
                    "home_team_id": g["teams"]["home"]["team"]["id"],
                    "away_team_id": g["teams"]["away"]["team"]["id"],
                    "home_sp_id":   g["teams"]["home"].get("probablePitcher", {}).get("id"),
                    "away_sp_id":   g["teams"]["away"].get("probablePitcher", {}).get("id"),
                    "home_score":   int(home_r),
                    "away_score":   int(away_r),
                })
        time.sleep(0.3)

    logger.info("  -> %d completed games for %d", len(games), season)
    _save(f"schedule_{season}", games)
    return games


# ---------------------------------------------------------------------------
# 2. Pitcher stats (bref + Statcast — no FanGraphs)
# ---------------------------------------------------------------------------

def fetch_pitcher_tables(season: int) -> dict:
    """Returns {mlbam_id: feature_dict} keyed by MLBAM ID."""
    cached = _load(f"pitcher_merged_{season}")
    if cached:
        logger.info("Pitcher stats cache hit for %d", season)
        return {int(k): v for k, v in cached.items()}

    pb = _pb()

    # --- Baseball Reference (FIP, K%, BB%) ---
    logger.info("Fetching pitching_stats_bref(%d)...", season)
    bref: dict[int, dict] = {}
    try:
        df = pb.pitching_stats_bref(season)
        df["IP"]  = pd.to_numeric(df["IP"],  errors="coerce").fillna(0)
        df["mlbID"] = pd.to_numeric(df["mlbID"], errors="coerce")
        df = df.dropna(subset=["mlbID"])
        df["mlbID"] = df["mlbID"].astype(int)
        df = df[df["Lev"].str.startswith("Maj", na=False)]
        df = df.sort_values("IP", ascending=False).drop_duplicates(subset=["mlbID"], keep="first")
        for col in ["BF","SO","BB","HR","HBP"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        df["fip"]    = (13*df["HR"] + 3*(df["BB"]+df["HBP"]) - 2*df["SO"]) / df["IP"].clip(1) + _FIP_CONST
        df["k_pct"]  = df["SO"] / df["BF"].clip(1)
        df["bb_pct"] = df["BB"] / df["BF"].clip(1)
        for _, row in df.iterrows():
            bref[int(row["mlbID"])] = {
                "fip":    float(np.clip(row["fip"],    0.5, 10.0)),
                "k_pct":  float(np.clip(row["k_pct"],  0.0,  0.6)),
                "bb_pct": float(np.clip(row["bb_pct"], 0.0,  0.3)),
            }
        logger.info("  bref: %d pitchers", len(bref))
    except Exception as e:
        logger.warning("pitching_stats_bref(%d) failed: %s", season, e)

    # --- Statcast expected stats (xERA) ---
    logger.info("Fetching statcast_pitcher_expected_stats(%d)...", season)
    xera_map: dict[int, float] = {}
    try:
        df = pb.statcast_pitcher_expected_stats(season, minPA=20)
        df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")
        df = df.dropna(subset=["player_id"])
        df["player_id"] = df["player_id"].astype(int)
        for _, row in df.iterrows():
            val = row.get("xera")
            if val is not None:
                try:
                    xera_map[int(row["player_id"])] = float(np.clip(val, 0.5, 10.0))
                except (TypeError, ValueError):
                    pass
        logger.info("  xERA: %d pitchers", len(xera_map))
    except Exception as e:
        logger.warning("statcast_pitcher_expected_stats(%d) failed: %s", season, e)

    # --- Statcast exit velo / barrel / hard-hit ---
    logger.info("Fetching statcast_pitcher_exitvelo_barrels(%d)...", season)
    contact_map: dict[int, dict] = {}
    try:
        df = pb.statcast_pitcher_exitvelo_barrels(season, minBBE=20)
        df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")
        df = df.dropna(subset=["player_id"])
        df["player_id"] = df["player_id"].astype(int)
        for _, row in df.iterrows():
            def _g(col, fallback):
                v = row.get(col)
                try: return float(v) if v is not None else fallback
                except: return fallback

            brl = _g("brl_percent", LEAGUE_AVG["sp_barrel"] * 100) / 100.0
            hh  = _g("ev95percent", LEAGUE_AVG["sp_hh_pct"] * 100) / 100.0
            ev  = _g("avg_hit_speed", LEAGUE_AVG["sp_exit_velo"])
            if brl > 1.0: brl /= 100.0
            if hh  > 1.0: hh  /= 100.0
            contact_map[int(row["player_id"])] = {
                "barrel_pct": brl, "hh_pct": hh, "avg_exit_velo": ev
            }
        logger.info("  contact: %d pitchers", len(contact_map))
    except Exception as e:
        logger.warning("statcast_pitcher_exitvelo_barrels(%d) failed: %s", season, e)

    # --- Statcast fastball spin ---
    logger.info("Fetching statcast_pitcher_pitch_arsenal(%d, avg_spin)...", season)
    spin_map: dict[int, float] = {}
    try:
        df = pb.statcast_pitcher_pitch_arsenal(season, minP=50, arsenal_type="avg_spin")
        df["pitcher"] = pd.to_numeric(df["pitcher"], errors="coerce")
        df = df.dropna(subset=["pitcher"])
        df["pitcher"] = df["pitcher"].astype(int)
        spin_cols = [c for c in ["ff_avg_spin","si_avg_spin","fc_avg_spin"] if c in df.columns]
        if spin_cols:
            df["best_spin"] = df[spin_cols].bfill(axis=1).iloc[:, 0]
            for _, row in df.iterrows():
                v = row.get("best_spin")
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    spin_map[int(row["pitcher"])] = float(np.clip(v, 1500, 3500))
        logger.info("  spin: %d pitchers", len(spin_map))
    except Exception as e:
        logger.warning("statcast_pitcher_pitch_arsenal(%d) failed: %s", season, e)

    # --- Merge all tables by MLBAM ID ---
    all_ids = set(bref) | set(xera_map) | set(contact_map) | set(spin_map)
    result: dict[int, dict] = {}
    for pid in all_ids:
        b = bref.get(pid, {})
        c = contact_map.get(pid, {})
        fip = b.get("fip", LEAGUE_AVG["sp_fip"])
        result[pid] = {
            "xera":          xera_map.get(pid, LEAGUE_AVG["sp_xera"]),
            "fip":           fip,
            "xfip":          fip,   # proxy
            "siera":         fip,   # proxy
            "k_pct":         b.get("k_pct",  LEAGUE_AVG["sp_k_pct"]),
            "bb_pct":        b.get("bb_pct", LEAGUE_AVG["sp_bb_pct"]),
            "barrel_pct":    c.get("barrel_pct",    LEAGUE_AVG["sp_barrel"]),
            "hh_pct":        c.get("hh_pct",        LEAGUE_AVG["sp_hh_pct"]),
            "avg_exit_velo": c.get("avg_exit_velo", LEAGUE_AVG["sp_exit_velo"]),
            "fastball_spin": spin_map.get(pid, LEAGUE_AVG["sp_spin"]),
        }

    logger.info("  -> %d pitchers merged for season %d", len(result), season)
    _save(f"pitcher_merged_{season}", {str(k): v for k, v in result.items()})
    return result


# ---------------------------------------------------------------------------
# 3. Team batting stats (MLB Stats API)
# ---------------------------------------------------------------------------

def fetch_team_batting(season: int) -> dict[str, dict]:
    cached = _load(f"team_batting_mlb_{season}")
    if cached:
        logger.info("Team batting cache hit for %d", season)
        return cached

    logger.info("Fetching team batting from MLB Stats API for %d...", season)

    # Get all team IDs
    try:
        data = _mlb_get("/teams", params={"sportId": 1, "season": season})
        teams = {t["abbreviation"]: t["id"] for t in data.get("teams", [])}
    except Exception as e:
        logger.warning("team list failed: %s", e)
        return {}

    result: dict[str, dict] = {}
    for abbr, tid in teams.items():
        try:
            data = _mlb_get(f"/teams/{tid}/stats",
                            params={"season": season, "group": "hitting", "stats": "season"})
            stats = data.get("stats", [{}])[0].get("splits", [{}])
            s = stats[0].get("stat", {}) if stats else {}
            def _f(k, fb):
                v = s.get(k)
                try: return float(v) if v is not None else fb
                except: return fb
            obp = _f("obp", 0.320)
            slg = _f("slg", 0.400)
            result[abbr] = {
                "woba": obp,
                "ops":  round(obp + slg, 4),
                "avg":  _f("avg", 0.250),
            }
            time.sleep(0.05)
        except Exception as e:
            logger.debug("team batting failed for %s: %s", abbr, e)

    logger.info("  -> %d teams for %d", len(result), season)
    _save(f"team_batting_mlb_{season}", result)
    return result


# ---------------------------------------------------------------------------
# 4. Pitcher game logs (for rolling ERA/WHIP per start)
# ---------------------------------------------------------------------------

def _parse_ip(ip_str) -> float:
    """Convert MLB API inningsPitched string ('5.2') to decimal innings."""
    try:
        s = str(ip_str or "0").strip()
        if "." in s:
            whole, thirds = s.split(".", 1)
            return int(whole) + int(thirds[:1]) / 3
        return float(s)
    except Exception:
        return 0.0


def fetch_pitcher_game_logs(season: int, pitcher_ids: list) -> dict:
    """Returns {pitcher_id: sorted list of start entries} for all pitchers in season."""
    cache_key = f"pitcher_gamelogs_{season}"
    cached = _load(cache_key)
    if cached:
        logger.info("Pitcher game-log cache hit for %d (%d pitchers)", season, len(cached))
        return {int(k): v for k, v in cached.items()}

    unique_ids = [pid for pid in set(pitcher_ids) if pid]
    logger.info("Fetching game logs for %d pitchers in %d ...", len(unique_ids), season)
    result: dict[int, list] = {}

    for i, pid in enumerate(unique_ids):
        if i % 100 == 0 and i > 0:
            logger.info("  game logs: %d/%d ...", i, len(unique_ids))
        try:
            data = _mlb_get(f"/people/{pid}/stats", params={
                "stats": "gameLog", "group": "pitching", "season": season,
            })
            starts = []
            for grp in data.get("stats", []):
                for split in grp.get("splits", []):
                    st = split.get("stat", {})
                    if int(st.get("gamesStarted", 0)) < 1:
                        continue
                    ip = _parse_ip(st.get("inningsPitched", 0))
                    starts.append({
                        "date": split.get("date", ""),
                        "er":   int(st.get("earnedRuns", 0)),
                        "h":    int(st.get("hits", 0)),
                        "bb":   int(st.get("baseOnBalls", 0)),
                        "ip":   ip,
                    })
            starts.sort(key=lambda x: x["date"])
            result[pid] = starts
        except Exception as e:
            logger.debug("game log failed for pitcher %d: %s", pid, e)
            result[pid] = []
        time.sleep(0.08)

    _save(cache_key, {str(k): v for k, v in result.items()})
    logger.info("  -> %d pitchers cached for %d", len(result), season)
    return result


def _era_whip_l3(game_logs: dict, pitcher_id, game_date: str, n: int = 3) -> tuple:
    """Compute ERA and WHIP over last n starts before game_date."""
    if not pitcher_id or pitcher_id not in game_logs:
        return LEAGUE_AVG["sp_era_l3"], LEAGUE_AVG["sp_whip_l3"]
    prev = [s for s in game_logs[int(pitcher_id)] if s["date"] < game_date]
    recent = prev[-n:]
    if not recent:
        return LEAGUE_AVG["sp_era_l3"], LEAGUE_AVG["sp_whip_l3"]
    total_ip = sum(s["ip"] for s in recent)
    if total_ip < 0.1:
        return LEAGUE_AVG["sp_era_l3"], LEAGUE_AVG["sp_whip_l3"]
    era  = float(np.clip((sum(s["er"] for s in recent) / total_ip) * 9, 0.0, 15.0))
    whip = float(np.clip((sum(s["h"] for s in recent) + sum(s["bb"] for s in recent)) / total_ip, 0.0, 4.0))
    return era, whip


# ---------------------------------------------------------------------------
# 4b. Days rest — computed from existing game logs
# ---------------------------------------------------------------------------

def _days_rest(game_logs: dict, pitcher_id, game_date_str: str) -> float:
    """Actual days since pitcher's last start before game_date (clamped 1-30)."""
    if not pitcher_id:
        return float(LEAGUE_AVG["sp_days_rest"])
    logs = game_logs.get(int(pitcher_id), [])
    prev = [s["date"] for s in logs if s["date"] < game_date_str]
    if not prev:
        return float(LEAGUE_AVG["sp_days_rest"])
    delta = (datetime.strptime(game_date_str, "%Y-%m-%d") -
             datetime.strptime(max(prev), "%Y-%m-%d")).days
    return float(min(max(delta, 1), 30))


# ---------------------------------------------------------------------------
# 4c. Season run differential — computed from schedule results
# ---------------------------------------------------------------------------

def compute_run_diffs(games: list) -> dict:
    """
    For each (team, game_date), returns cumulative season run differential
    BEFORE that date.  Doubleheader games on the same date both see the
    pre-day snapshot (snapshot taken before the first game of each day).
    """
    sorted_games = sorted(games, key=lambda g: (g["game_date"], g["game_id"]))
    running: dict[str, float] = {}
    result: dict[tuple, float] = {}
    for game in sorted_games:
        home, away = game["home_team"], game["away_team"]
        date = game["game_date"]
        if (home, date) not in result:
            result[(home, date)] = running.get(home, 0.0)
        if (away, date) not in result:
            result[(away, date)] = running.get(away, 0.0)
        hs, as_ = game["home_score"], game["away_score"]
        running[home] = running.get(home, 0.0) + (hs - as_)
        running[away] = running.get(away, 0.0) + (as_ - hs)
    return result


# ---------------------------------------------------------------------------
# 4d. Historical weather — Open-Meteo archive API
# ---------------------------------------------------------------------------

def fetch_weather_historical(season: int, games: list) -> dict:
    """
    Fetches hourly temp + wind for every game via the Open-Meteo historical
    archive.  Batches by home team (one season-long request per venue, cached).
    Returns {game_id: {temperature, wind_speed, wind_dir_deg}}.
    """
    team_coords: dict[str, dict] = {
        info["team"]: {"lat": info["lat"], "lon": info["lon"]}
        for info in VENUE_COORDS.values()
    }

    from collections import defaultdict
    games_by_team: dict[str, list] = defaultdict(list)
    for g in games:
        games_by_team[g["home_team"]].append(g)

    ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
    result: dict[str, dict] = {}

    for team, team_games in sorted(games_by_team.items()):
        coords = team_coords.get(team)
        if not coords:
            logger.debug("No venue coords for team %s — weather skipped", team)
            continue

        cache_key = f"weather_hist_{season}_{team}"
        cached = _load(cache_key)

        if not cached:
            logger.info("  weather: %s %d ...", team, season)
            try:
                resp = _SESSION.get(ARCHIVE_URL, params={
                    "latitude":         coords["lat"],
                    "longitude":        coords["lon"],
                    "start_date":       f"{season}-03-01",
                    "end_date":         f"{season}-10-31",
                    "hourly":           "temperature_2m,windspeed_10m,winddirection_10m",
                    "timezone":         "UTC",
                    "temperature_unit": "fahrenheit",
                    "windspeed_unit":   "mph",
                }, timeout=30)
                resp.raise_for_status()
                h = resp.json().get("hourly", {})
                cached = {
                    "time":       h.get("time", []),
                    "temp":       h.get("temperature_2m", []),
                    "wind_speed": h.get("windspeed_10m", []),
                    "wind_dir":   h.get("winddirection_10m", []),
                }
                _save(cache_key, cached)
                time.sleep(0.3)
            except Exception as exc:
                logger.warning("  weather failed for %s %d: %s", team, season, exc)
                cached = None

        if cached and cached.get("time"):
            t2i = {t: i for i, t in enumerate(cached["time"])}
            for game in team_games:
                hour = game.get("game_time_utc_hour", 23)
                key = f"{game['game_date']}T{hour:02d}:00"
                idx = t2i.get(key)
                if idx is not None:
                    def _safe(lst, i, fb):
                        v = lst[i] if i < len(lst) else None
                        return float(v) if v is not None else fb
                    result[game["game_id"]] = {
                        "temperature":  _safe(cached["temp"],       idx, LEAGUE_AVG["temperature"]),
                        "wind_speed":   _safe(cached["wind_speed"], idx, LEAGUE_AVG["wind_speed"]),
                        "wind_dir_deg": _safe(cached["wind_dir"],   idx, LEAGUE_AVG["wind_dir_deg"]),
                    }

    logger.info("Weather resolved for %d/%d games in season %d",
                len(result), len(games), season)
    return result


# ---------------------------------------------------------------------------
# 4e. Player name lookup
# ---------------------------------------------------------------------------

_name_cache: dict[int, str] = {}

def resolve_name(mlbam_id: int) -> str | None:
    if mlbam_id in _name_cache:
        return _name_cache[mlbam_id]
    try:
        d = _mlb_get(f"/people/{mlbam_id}")
        name = d["people"][0]["fullName"]
        _name_cache[mlbam_id] = name
        return name
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 4f. Rolling team batting stats (ops_14d / risp_14d)
# ---------------------------------------------------------------------------

def fetch_team_batting_gamelogs(season: int) -> dict:
    """
    Fetches per-game batting counting stats for all 30 teams.
    Returns {team_abbr: [{date, h, ab, bb, hbp, tb, sf}]} sorted by date.
    ~30 API calls per season, cached.
    """
    cache_key = f"team_batting_gamelogs_{season}"
    cached = _load(cache_key)
    if cached:
        logger.info("Team batting game-log cache hit for %d", season)
        return cached

    all_teams_data = _mlb_get("/teams", params={"sportId": 1, "season": season})
    teams = [
        (t["abbreviation"], t["id"])
        for t in all_teams_data.get("teams", [])
        if "abbreviation" in t and "id" in t
    ]

    logger.info("Fetching team batting game logs for %d (%d teams)...", season, len(teams))
    result: dict = {}

    for abbr, tid in teams:
        try:
            data = _mlb_get(f"/teams/{tid}/stats",
                            params={"stats": "gameLog", "group": "hitting", "season": season})
            splits = data.get("stats", [{}])[0].get("splits", [])
            games = []
            for s in splits:
                st = s.get("stat", {})
                def _i(k):
                    v = st.get(k, 0)
                    try: return int(v)
                    except: return 0
                games.append({
                    "date": s.get("date", ""),
                    "h":   _i("hits"),
                    "ab":  _i("atBats"),
                    "bb":  _i("baseOnBalls"),
                    "hbp": _i("hitByPitch"),
                    "tb":  _i("totalBases"),
                    "sf":  _i("sacFlies"),
                })
            games.sort(key=lambda x: x["date"])
            result[abbr] = games
            time.sleep(0.1)
        except Exception as exc:
            logger.warning("Team batting game log failed for %s: %s", abbr, exc)
            result[abbr] = []

    _save(cache_key, result)
    logger.info("  -> %d teams batting logs cached for %d", len(result), season)
    return result


def compute_rolling_ops(team_batting_logs: dict, window: int = 14) -> dict:
    """
    For each (team, game_date), computes OPS over the prior `window` days
    (exclusive of game_date itself — no look-ahead).
    OBP = (H+BB+HBP) / (AB+BB+HBP+SF)   SLG = TB / AB
    Falls back to league average when fewer than 3 games exist in the window.
    Returns {(team_abbr, date_str): ops_float}.
    """
    result: dict = {}
    for team, games in team_batting_logs.items():
        for i, game in enumerate(games):
            gd      = datetime.strptime(game["date"], "%Y-%m-%d")
            cutoff  = (gd - timedelta(days=window)).strftime("%Y-%m-%d")
            h = ab = bb = hbp = tb = sf = n_games = 0
            for prev in games[:i]:
                if prev["date"] >= cutoff:  # prior `window` days, exclusive of today
                    h    += prev["h"];   ab  += prev["ab"]
                    bb   += prev["bb"];  hbp += prev["hbp"]
                    tb   += prev["tb"];  sf  += prev["sf"]
                    n_games += 1
            if ab < 10 or n_games < 3:
                result[(team, game["date"])] = LEAGUE_AVG["ops_14d"]
            else:
                obp_d = ab + bb + hbp + sf
                obp   = (h + bb + hbp) / obp_d if obp_d > 0 else 0.320
                slg   = tb / ab
                result[(team, game["date"])] = round(float(np.clip(obp + slg, 0.3, 1.5)), 4)
    return result


def fetch_team_risp_season(season: int) -> dict:
    """
    Fetches season RISP batting average per team via statSplits (sitCodes=risp).
    Matches the live pipeline's RISP source exactly.
    Returns {team_abbr: risp_avg_float}.  ~30 API calls per season, cached.
    """
    cache_key = f"team_risp_{season}"
    cached = _load(cache_key)
    if cached:
        logger.info("RISP cache hit for %d", season)
        return cached

    all_teams_data = _mlb_get("/teams", params={"sportId": 1, "season": season})
    teams = [
        (t["abbreviation"], t["id"])
        for t in all_teams_data.get("teams", [])
        if "abbreviation" in t and "id" in t
    ]

    logger.info("Fetching RISP stats for %d teams in %d...", len(teams), season)
    result: dict = {}

    for abbr, tid in teams:
        try:
            data = _mlb_get(f"/teams/{tid}/stats", params={
                "stats": "statSplits", "sitCodes": "risp",
                "group": "hitting", "season": season,
            })
            splits = data.get("stats", [{}])[0].get("splits", [])
            stat   = splits[0].get("stat", {}) if splits else {}
            v      = stat.get("avg")
            result[abbr] = float(v) if v else LEAGUE_AVG["risp_14d"]
            time.sleep(0.05)
        except Exception as exc:
            logger.debug("RISP fetch failed for %s: %s", abbr, exc)
            result[abbr] = LEAGUE_AVG["risp_14d"]

    _save(cache_key, result)
    logger.info("  -> %d teams RISP cached for %d", len(result), season)
    return result


# ---------------------------------------------------------------------------
# 4g-extra. Win/loss streak per (team, date)
# ---------------------------------------------------------------------------

def compute_win_streaks(games: list) -> dict:
    """
    For each (team_abbr, game_date) compute the consecutive win/loss streak
    BEFORE that game.  Positive = win streak, negative = loss streak, 0 = none.
    Returns {(team_abbr, date_str): streak_int}.
    """
    games_sorted = sorted(games, key=lambda g: g["game_date"])
    team_results: dict[str, list] = {}  # team → [(date_str, won)]
    for g in games_sorted:
        hs, as_ = g["home_score"], g["away_score"]
        home_won = hs > as_
        d = g["game_date"]
        for team, won in [(g["home_team"], home_won), (g["away_team"], not home_won)]:
            team_results.setdefault(team, []).append((d, won))

    result = {}
    for team, results_list in team_results.items():
        for i, (d, _) in enumerate(results_list):
            prior = results_list[:i]
            if not prior:
                result[(team, d)] = 0
                continue
            last_won = prior[-1][1]
            streak = 0
            for _, won in reversed(prior):
                if won == last_won:
                    streak += 1
                else:
                    break
            direction = 1 if last_won else -1
            result[(team, d)] = int(max(-10, min(10, direction * streak)))
    return result


# ---------------------------------------------------------------------------
# 4g-extra-1b. Team days of rest per (team, date)
# ---------------------------------------------------------------------------

def compute_team_days_rest(games: list) -> dict:
    """
    For each (team_abbr, game_date) compute days since team's previous game.
    Returns {(team_abbr, date_str): days_int}, capped at 7.
    """
    games_sorted = sorted(games, key=lambda g: g["game_date"])
    last_played: dict[str, str] = {}
    result: dict = {}

    for g in games_sorted:
        date = g["game_date"]
        for team in (g["home_team"], g["away_team"]):
            if (team, date) in result:
                continue  # doubleheader: already recorded
            prev = last_played.get(team)
            if prev:
                days = (datetime.strptime(date, "%Y-%m-%d") -
                        datetime.strptime(prev, "%Y-%m-%d")).days
                result[(team, date)] = min(days, 7)
            else:
                result[(team, date)] = int(LEAGUE_AVG["team_days_rest"])

        # Update last-played after snapshot
        last_played[g["home_team"]] = date
        last_played[g["away_team"]] = date

    return result


# ---------------------------------------------------------------------------
# 4g-extra-2. Team roster handedness (for hand_match_pct fallback)
# ---------------------------------------------------------------------------

def fetch_team_roster_hands(season: int, team_id_map: dict) -> dict:
    """
    Returns {team_abbr: {"l_pct": float, "r_pct": float}} based on active roster.
    Switch hitters count as 0.5.  ~30 API calls per season, cached.
    """
    cache_key = f"team_roster_hands_{season}"
    cached = _load(cache_key)
    if cached:
        logger.info("Roster handedness cache hit for %d", season)
        return cached

    result = {}
    for tid, abbr in team_id_map.items():
        try:
            data = _mlb_get(f"/teams/{tid}/roster",
                            params={"rosterType": "active", "season": season})
            l = r = s = 0
            for p in data.get("roster", []):
                if p.get("position", {}).get("abbreviation", "P") == "P":
                    continue
                bats = p.get("person", {}).get("batSide", {}).get("code", "R")
                if bats == "L":   l += 1
                elif bats == "S": s += 1
                else:             r += 1
            total = l + r + s
            result[abbr] = ({"l_pct": round((l + s * 0.5) / total, 3),
                              "r_pct": round((r + s * 0.5) / total, 3)}
                            if total > 0 else {"l_pct": 0.45, "r_pct": 0.55})
            time.sleep(0.1)
        except Exception as exc:
            logger.warning("Roster hands failed for %s: %s", abbr, exc)
            result[abbr] = {"l_pct": 0.45, "r_pct": 0.55}

    _save(cache_key, result)
    logger.info("  -> Roster handedness cached for %d teams in %d", len(result), season)
    return result


# ---------------------------------------------------------------------------
# 4g-extra-3. Team bullpen LI proxy (saves/holds/blown saves from MLB API)
# ---------------------------------------------------------------------------

def fetch_team_bp_li_season(season: int, team_id_map: dict) -> dict:
    """
    Returns {team_abbr: li_proxy_float} for the season.  ~30 API calls, cached.
    """
    cache_key = f"team_bp_li_{season}"
    cached = _load(cache_key)
    if cached:
        logger.info("BP LI proxy cache hit for %d", season)
        return cached

    result = {}
    for tid, abbr in team_id_map.items():
        try:
            data = _mlb_get(f"/teams/{tid}/stats",
                            params={"season": season, "group": "pitching", "stats": "season"})
            splits = data.get("stats", [{}])[0].get("splits", [])
            stat   = splits[0].get("stat", {}) if splits else {}
            sv  = int(stat.get("saves",      0) or 0)
            hld = int(stat.get("holds",      0) or 0)
            bs  = int(stat.get("blownSaves", 0) or 0)
            total = sv + hld + bs
            if total >= 10:
                ratio = (sv + hld) / total
                li = round(float(np.clip((ratio / 0.75) * LEAGUE_AVG["bp_li"], 0.5, 2.0)), 3)
            else:
                li = LEAGUE_AVG["bp_li"]
            result[abbr] = li
            time.sleep(0.1)
        except Exception as exc:
            logger.warning("BP LI proxy failed for %s: %s", abbr, exc)
            result[abbr] = LEAGUE_AVG["bp_li"]

    _save(cache_key, result)
    logger.info("  -> BP LI proxy cached for %d teams in %d", len(result), season)
    return result


# ---------------------------------------------------------------------------
# 4g-extra-4. Umpire assignments per game (HP ump from schedule API)
# ---------------------------------------------------------------------------

def fetch_season_umpires(season: int) -> dict:
    """
    Returns {game_id_str: ump_name} for all regular-season games.
    Makes 7 monthly schedule API calls with officials hydration.  Cached per season.
    """
    cache_key = f"umpires_{season}"
    cached = _load(cache_key)
    if cached:
        logger.info("Umpire cache hit for %d (%d games)", season, len(cached))
        return cached

    logger.info("Fetching umpire assignments for %d...", season)
    result = {}
    month_ranges = [
        (f"{season}-03-25", f"{season}-04-30"),
        (f"{season}-05-01", f"{season}-05-31"),
        (f"{season}-06-01", f"{season}-06-30"),
        (f"{season}-07-01", f"{season}-07-31"),
        (f"{season}-08-01", f"{season}-08-31"),
        (f"{season}-09-01", f"{season}-09-30"),
        (f"{season}-10-01", f"{season}-10-10"),
    ]
    for start, end in month_ranges:
        try:
            data = _mlb_get("/schedule", params={
                "sportId": 1, "startDate": start, "endDate": end,
                "gameType": "R", "hydrate": "officials",
            })
            for day in data.get("dates", []):
                for g in day.get("games", []):
                    pk = str(g["gamePk"])
                    ump_name = "TBD"
                    for official in g.get("officials", []):
                        if official.get("officialType") == "Home Plate":
                            ump_name = official.get("official", {}).get("fullName", "TBD")
                            break
                    result[pk] = ump_name
            time.sleep(0.3)
        except Exception as exc:
            logger.warning("Umpire fetch failed for %s-%s: %s", start, end, exc)

    _save(cache_key, result)
    logger.info("  -> Umpires fetched for %d games in %d", len(result), season)
    return result


# ---------------------------------------------------------------------------
# 4g. Bullpen workload — relief appearances for bp_ip_3d
# ---------------------------------------------------------------------------

def fetch_team_id_map(season: int) -> dict:
    """Returns {team_id: team_abbr} for mapping game-log team IDs to abbreviations."""
    cache_key = f"team_id_map_{season}"
    cached = _load(cache_key)
    if cached:
        return {int(k): v for k, v in cached.items()}
    data = _mlb_get("/teams", params={"sportId": 1, "season": season})
    result = {
        t["id"]: t["abbreviation"]
        for t in data.get("teams", [])
        if "id" in t and "abbreviation" in t
    }
    _save(cache_key, {str(k): v for k, v in result.items()})
    return result


def fetch_season_pitcher_ids(season: int) -> list:
    """Returns all pitcher MLBAM IDs who appeared in regular-season games."""
    cache_key = f"all_pitcher_ids_{season}"
    cached = _load(cache_key)
    if cached:
        logger.info("All pitcher IDs cache hit for %d (%d pitchers)", season, len(cached))
        return cached
    logger.info("Fetching all pitcher IDs for %d ...", season)
    data = _mlb_get("/sports/1/players", params={"season": season, "gameType": "R"})
    ids = [
        p["id"] for p in data.get("people", [])
        if p.get("primaryPosition", {}).get("type") == "Pitcher"
    ]
    logger.info("  -> %d pitchers found for %d", len(ids), season)
    _save(cache_key, ids)
    return ids


def fetch_pitcher_relief_logs(season: int, all_pitcher_ids: list, starter_ids: set) -> dict:
    """
    Fetches game logs for pitchers not already in starter_ids (relief
    specialists + spot starters who rarely start).  Filters each log to
    non-start appearances only (gamesStarted == 0, ip > 0).
    Returns {pitcher_id: [{date, ip, team_id}]}.
    """
    cache_key = f"pitcher_relief_logs_{season}"
    cached = _load(cache_key)
    if cached:
        logger.info("Relief log cache hit for %d (%d pitchers)", season, len(cached))
        return {int(k): v for k, v in cached.items()}

    relief_ids = [pid for pid in all_pitcher_ids if pid not in starter_ids]
    logger.info("Fetching relief logs for %d pitchers in %d ...", len(relief_ids), season)

    result: dict = {}
    for i, pid in enumerate(relief_ids):
        if i % 100 == 0 and i > 0:
            logger.info("  relief logs: %d/%d ...", i, len(relief_ids))
        try:
            data = _mlb_get(f"/people/{pid}/stats", params={
                "stats": "gameLog", "group": "pitching", "season": season,
            })
            appearances = []
            for grp in data.get("stats", []):
                for split in grp.get("splits", []):
                    st = split.get("stat", {})
                    if int(st.get("gamesStarted", 0)) >= 1:
                        continue  # skip starts
                    ip = _parse_ip(st.get("inningsPitched", 0))
                    if ip <= 0:
                        continue
                    appearances.append({
                        "date":    split.get("date", ""),
                        "ip":      ip,
                        "team_id": split.get("team", {}).get("id"),
                    })
            result[pid] = appearances
        except Exception as exc:
            logger.debug("Relief log failed for pitcher %d: %s", pid, exc)
            result[pid] = []
        time.sleep(0.08)

    _save(cache_key, {str(k): v for k, v in result.items()})
    logger.info("  -> %d relief pitchers cached for %d", len(result), season)
    return result


def compute_daily_bp_ip(relief_logs: dict, team_id_map: dict) -> dict:
    """
    Aggregates relief appearances into {(team_abbr, date): total_ip}.
    Each appearance's team is taken from the game log split (correct across trades).
    """
    from collections import defaultdict
    daily: dict = defaultdict(float)
    for appearances in relief_logs.values():
        for app in appearances:
            abbr = team_id_map.get(app.get("team_id"))
            if abbr:
                daily[(abbr, app["date"])] += app["ip"]
    return dict(daily)


def compute_team_bp_xera(relief_logs: dict, team_id_map: dict, pitcher_stats: dict) -> dict:
    """IP-weighted average xERA per team using each reliever's prev-season xERA."""
    from collections import defaultdict
    team_ip: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for pid, appearances in relief_logs.items():
        pid_int = int(pid) if not isinstance(pid, int) else pid
        for app in appearances:
            abbr = team_id_map.get(app.get("team_id"))
            if abbr:
                team_ip[abbr][pid_int] += app.get("ip", 0.0)
    result = {}
    for abbr, pid_ips in team_ip.items():
        total_ip = sum(pid_ips.values())
        if total_ip <= 0:
            result[abbr] = LEAGUE_AVG["bp_xera"]
            continue
        weighted = sum(
            pitcher_stats.get(pid, {}).get("xera", LEAGUE_AVG["bp_xera"]) * ip
            for pid, ip in pid_ips.items()
        )
        result[abbr] = float(np.clip(weighted / total_ip, 0.5, 7.0))
    return result


def get_bp_ip_3d(team: str, game_date_str: str, daily_bp_ip: dict) -> float:
    """Total bullpen IP for `team` in the 3 calendar days before game_date."""
    gd = datetime.strptime(game_date_str, "%Y-%m-%d")
    total = sum(
        daily_bp_ip.get((team, (gd - timedelta(days=d)).strftime("%Y-%m-%d")), 0.0)
        for d in range(1, 4)
    )
    return float(np.clip(total, 0.0, 30.0))


# ---------------------------------------------------------------------------
# 4h. IL transactions — starters_il / bp_il_ct
# ---------------------------------------------------------------------------

def fetch_team_il_transactions(season: int, all_pitcher_ids: set, team_id_map: dict) -> dict:
    """
    Fetches all IL transactions for all 30 teams for the full season.
    Marks each player as pitcher/non-pitcher using all_pitcher_ids.
    Returns {team_abbr: [{player_id, is_pitcher, type, date}]} sorted by date.
    ~30 API calls per season, cached.
    """
    cache_key = f"team_il_transactions_{season}"
    cached = _load(cache_key)
    if cached:
        logger.info("IL transactions cache hit for %d", season)
        return cached

    pitcher_set = set(all_pitcher_ids)
    result: dict = {}

    for team_id, abbr in team_id_map.items():
        try:
            data = _mlb_get("/transactions", params={
                "teamId":    team_id,
                "startDate": f"{season}-03-01",
                "endDate":   f"{season}-10-15",
            })
            txns = []
            for t in data.get("transactions", []):
                pid = t.get("person", {}).get("id")
                if pid is None:
                    continue
                txns.append({
                    "player_id":  int(pid),
                    "is_pitcher": int(pid) in pitcher_set,
                    "type":       t.get("typeCode", ""),
                    "date":       t.get("date", ""),
                })
            txns.sort(key=lambda x: x["date"])
            result[abbr] = txns
            time.sleep(0.1)
        except Exception as exc:
            logger.warning("IL transactions failed for team %d (%s) %d: %s", team_id, abbr, season, exc)
            result[abbr] = []

    _save(cache_key, result)
    logger.info("  -> IL transactions cached for %d teams in %d", len(result), season)
    return result


def get_il_counts(team_abbr: str, game_date_str: str, il_transactions: dict) -> tuple:
    """
    Returns (bp_il_ct, starters_il) by replaying IL transactions up to game_date.
    Mirrors the logic in fetchers/mlb_stats.py:count_il_players().
    """
    txns = il_transactions.get(team_abbr, [])
    on_il_pitchers:     set = set()
    on_il_non_pitchers: set = set()

    for t in txns:
        if t.get("date", "") >= game_date_str:
            break
        pid = t.get("player_id")
        if pid is None:
            continue
        type_code = t.get("type", "")
        if type_code in ("IL10", "IL15", "IL60", "SUSP", "BRV"):
            if t.get("is_pitcher"):
                on_il_pitchers.add(pid)
            else:
                on_il_non_pitchers.add(pid)
        elif type_code in ("REA", "ACT"):  # reinstated / activated
            on_il_pitchers.discard(pid)
            on_il_non_pitchers.discard(pid)

    return len(on_il_pitchers), len(on_il_non_pitchers)


# ---------------------------------------------------------------------------
# 4i. Schedule-based features — win%, back-to-back, series game#
# ---------------------------------------------------------------------------

def compute_schedule_features(games: list) -> dict:
    """
    For each (team, game_date), computes pre-game win%, back-to-back flag,
    and series game number from the season's already-fetched schedule results.
    Returns {(team_abbr, date_str): {"win_pct": float, "back_to_back": int, "series_game": int}}.
    Handles doubleheaders: both games on the same date see the same pre-day snapshot.
    """
    sorted_games = sorted(games, key=lambda g: (g["game_date"], g["game_id"]))

    wins:       dict[str, int] = {}
    losses:     dict[str, int] = {}
    last_date:  dict[str, str] = {}
    last_opp:   dict[str, str] = {}
    series_num: dict[str, int] = {}
    result:     dict[tuple, dict] = {}

    for game in sorted_games:
        home, away = game["home_team"], game["away_team"]
        date = game["game_date"]
        prev_date_str = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")

        for team, opp in [(home, away), (away, home)]:
            if (team, date) in result:
                continue  # doubleheader: pre-game snapshot already recorded

            w, l = wins.get(team, 0), losses.get(team, 0)
            wpct = w / (w + l) if (w + l) > 0 else 0.500

            b2b = 1 if last_date.get(team) == prev_date_str else 0

            if last_opp.get(team) == opp and last_date.get(team) == prev_date_str:
                series_num[team] = series_num.get(team, 0) + 1
            else:
                series_num[team] = 1

            result[(team, date)] = {
                "win_pct":      round(wpct, 4),
                "back_to_back": b2b,
                "series_game":  min(series_num[team], 7),
            }

        # Post-game: update running totals and last-played tracking
        hs, as_ = game["home_score"], game["away_score"]
        if hs > as_:
            wins[home]   = wins.get(home, 0) + 1
            losses[away] = losses.get(away, 0) + 1
        elif as_ > hs:
            wins[away]   = wins.get(away, 0) + 1
            losses[home] = losses.get(home, 0) + 1

        last_date[home] = date
        last_date[away] = date
        last_opp[home]  = away
        last_opp[away]  = home

    return result


# ---------------------------------------------------------------------------
# 5. Feature row builder
# ---------------------------------------------------------------------------

def _sp(pitcher_id: int | None, pitchers: dict) -> dict:
    if pitcher_id and pitcher_id in pitchers:
        return pitchers[pitcher_id]
    return {
        "xera": LEAGUE_AVG["sp_xera"], "fip": LEAGUE_AVG["sp_fip"],
        "xfip": LEAGUE_AVG["sp_xfip"], "siera": LEAGUE_AVG["sp_siera"],
        "k_pct": LEAGUE_AVG["sp_k_pct"], "bb_pct": LEAGUE_AVG["sp_bb_pct"],
        "barrel_pct": LEAGUE_AVG["sp_barrel"], "hh_pct": LEAGUE_AVG["sp_hh_pct"],
        "avg_exit_velo": LEAGUE_AVG["sp_exit_velo"], "fastball_spin": LEAGUE_AVG["sp_spin"],
    }

def _team(abbr: str, batting: dict) -> dict:
    s = batting.get(abbr, {})
    if not s:
        for k, v in batting.items():
            if k.upper() == abbr.upper():
                s = v
                break
    return {
        "woba": s.get("woba", LEAGUE_AVG["lineup_woba"]),
        "ops":  s.get("ops",  LEAGUE_AVG["ops_14d"]),
        "avg":  s.get("avg",  LEAGUE_AVG["risp_14d"]),
    }

def _pf(home_team: str) -> dict:
    hc = PARK_FACTORS_HARDCODED.get(home_team.upper(), {})
    return {
        "runs": hc.get("runs", LEAGUE_AVG["park_factor_runs"]),
        "hr":   hc.get("hr",   LEAGUE_AVG["park_factor_hr"]),
    }

def build_feature_row(game: dict, pitchers: dict, batting: dict, game_logs: dict,
                      run_diffs: dict = None, weather_map: dict = None,
                      bp_ip_3d_map: dict = None, rolling_ops_map: dict = None,
                      risp_map: dict = None, il_transactions: dict = None,
                      schedule_features: dict = None, win_streaks: dict = None,
                      roster_hands: dict = None, bp_li_map: dict = None,
                      umpire_map: dict = None, team_days_rest: dict = None,
                      team_bp_xera: dict = None) -> dict:
    home_sp = _sp(game.get("home_sp_id"), pitchers)
    away_sp = _sp(game.get("away_sp_id"), pitchers)
    home_b  = _team(game["home_team"], batting)
    away_b  = _team(game["away_team"], batting)
    pf      = _pf(game["home_team"])

    h_era_l3, h_whip_l3 = _era_whip_l3(game_logs, game.get("home_sp_id"), game["game_date"])
    a_era_l3, a_whip_l3 = _era_whip_l3(game_logs, game.get("away_sp_id"), game["game_date"])

    # Previously constant features — now populated when data is available
    date = game["game_date"]
    home_dr  = _days_rest(game_logs, game.get("home_sp_id"), date)
    away_dr  = _days_rest(game_logs, game.get("away_sp_id"), date)
    home_rd  = run_diffs.get((game["home_team"], date), LEAGUE_AVG["run_diff"]) if run_diffs else LEAGUE_AVG["run_diff"]
    away_rd  = run_diffs.get((game["away_team"], date), LEAGUE_AVG["run_diff"]) if run_diffs else LEAGUE_AVG["run_diff"]
    wx       = weather_map.get(game["game_id"], {}) if weather_map else {}
    home_bip  = get_bp_ip_3d(game["home_team"], date, bp_ip_3d_map) if bp_ip_3d_map else LEAGUE_AVG["bp_ip_3d"]
    away_bip  = get_bp_ip_3d(game["away_team"], date, bp_ip_3d_map) if bp_ip_3d_map else LEAGUE_AVG["bp_ip_3d"]
    home_ops  = rolling_ops_map.get((game["home_team"], date), home_b["ops"]) if rolling_ops_map else home_b["ops"]
    away_ops  = rolling_ops_map.get((game["away_team"], date), away_b["ops"]) if rolling_ops_map else away_b["ops"]
    home_risp = risp_map.get(game["home_team"], LEAGUE_AVG["risp_14d"]) if risp_map else home_b["avg"]
    away_risp = risp_map.get(game["away_team"], LEAGUE_AVG["risp_14d"]) if risp_map else away_b["avg"]

    if il_transactions:
        home_bp_il, home_st_il = get_il_counts(game["home_team"], date, il_transactions)
        away_bp_il, away_st_il = get_il_counts(game["away_team"], date, il_transactions)
    else:
        home_bp_il = away_bp_il = int(LEAGUE_AVG["bp_il_ct"])
        home_st_il = away_st_il = int(LEAGUE_AVG["starters_il"])

    sched = schedule_features or {}
    h_sched = sched.get((game["home_team"], date), {})
    a_sched = sched.get((game["away_team"], date), {})
    home_wpct   = h_sched.get("win_pct",      LEAGUE_AVG["win_pct"])
    home_b2b    = h_sched.get("back_to_back", int(LEAGUE_AVG["back_to_back"]))
    home_series = h_sched.get("series_game",  int(LEAGUE_AVG["series_game"]))
    away_wpct   = a_sched.get("win_pct",      LEAGUE_AVG["win_pct"])
    away_b2b    = a_sched.get("back_to_back", int(LEAGUE_AVG["back_to_back"]))
    away_series = a_sched.get("series_game",  int(LEAGUE_AVG["series_game"]))

    # Win/loss streaks
    streaks = win_streaks or {}
    home_streak = streaks.get((game["home_team"], date), int(LEAGUE_AVG["win_streak"]))
    away_streak = streaks.get((game["away_team"], date), int(LEAGUE_AVG["win_streak"]))

    # Team days of rest
    drest = team_days_rest or {}
    home_drest = drest.get((game["home_team"], date), int(LEAGUE_AVG["team_days_rest"]))
    away_drest = drest.get((game["away_team"], date), int(LEAGUE_AVG["team_days_rest"]))

    # Handedness match (opposing roster hand composition vs SP hand)
    hands = roster_hands or {}
    home_sp_data = pitchers.get(game.get("home_sp_id"), {})
    away_sp_data = pitchers.get(game.get("away_sp_id"), {})
    home_hand = home_sp_data.get("hand", "R")
    away_hand = away_sp_data.get("hand", "R")
    away_hands = hands.get(game["away_team"], {"l_pct": 0.45, "r_pct": 0.55})
    home_hands = hands.get(game["home_team"], {"l_pct": 0.45, "r_pct": 0.55})
    home_hand_match = away_hands["l_pct"] if home_hand == "L" else away_hands["r_pct"]
    away_hand_match = home_hands["l_pct"] if away_hand == "L" else home_hands["r_pct"]

    # Bullpen LI proxy
    bp_li = bp_li_map or {}
    home_bp_li_val = bp_li.get(game["home_team"], LEAGUE_AVG["bp_li"])
    away_bp_li_val = bp_li.get(game["away_team"], LEAGUE_AVG["bp_li"])

    # Umpire run factor (career avg runs / 9.0; in-memory cache in get_umpire_career_stats)
    from fetchers.mlb_stats import get_umpire_career_stats as _get_ump_stats
    _LEAGUE_AVG_RUNS = 9.0
    ump_name = (umpire_map or {}).get(game["game_id"], "TBD")
    _ump_stats = _get_ump_stats(ump_name) if ump_name and ump_name != "TBD" else {}
    _avg_r = _ump_stats.get("avg_runs")
    umpire_run_factor = round(float(_avg_r) / _LEAGUE_AVG_RUNS, 3) if _avg_r else LEAGUE_AVG["umpire_run_factor"]

    _bp_xera = team_bp_xera or {}
    def _fip_bp(team_abbr):
        return _bp_xera.get(team_abbr, LEAGUE_AVG["bp_xera"])

    return {
        "game_id":   game["game_id"],
        "game_date": game["game_date"],
        "cycle":     "A",
        "home_team": game["home_team"],
        "away_team": game["away_team"],
        # Home SP
        "home_sp_xera":          home_sp["xera"],
        "home_sp_fip":           home_sp["fip"],
        "home_sp_xfip":          home_sp["xfip"],
        "home_sp_siera":         home_sp["siera"],
        "home_sp_k_pct":         home_sp["k_pct"],
        "home_sp_bb_pct":        home_sp["bb_pct"],
        "home_sp_barrel":        home_sp["barrel_pct"],
        "home_sp_hh_pct":        home_sp["hh_pct"],
        "home_sp_exit_velo":     home_sp["avg_exit_velo"],
        "home_sp_spin":          home_sp["fastball_spin"],
        "home_sp_days_rest":     int(home_dr),
        "home_sp_hand_match_pct":home_hand_match,
        "home_sp_bvp_woba":      LEAGUE_AVG["sp_bvp_woba"],
        "home_sp_era_l3":        h_era_l3,
        "home_sp_whip_l3":       h_whip_l3,
        "home_sp_xera_delta":    round(float(home_sp["xera"]) - float(home_sp["fip"]), 3),
        # Away SP
        "away_sp_xera":          away_sp["xera"],
        "away_sp_fip":           away_sp["fip"],
        "away_sp_xfip":          away_sp["xfip"],
        "away_sp_siera":         away_sp["siera"],
        "away_sp_k_pct":         away_sp["k_pct"],
        "away_sp_bb_pct":        away_sp["bb_pct"],
        "away_sp_barrel":        away_sp["barrel_pct"],
        "away_sp_hh_pct":        away_sp["hh_pct"],
        "away_sp_exit_velo":     away_sp["avg_exit_velo"],
        "away_sp_spin":          away_sp["fastball_spin"],
        "away_sp_days_rest":     int(away_dr),
        "away_sp_hand_match_pct":away_hand_match,
        "away_sp_bvp_woba":      LEAGUE_AVG["sp_bvp_woba"],
        "away_sp_era_l3":        a_era_l3,
        "away_sp_whip_l3":       a_whip_l3,
        "away_sp_xera_delta":    round(float(away_sp["xera"]) - float(away_sp["fip"]), 3),
        # Home BP
        "home_bp_xera":   _fip_bp(game["home_team"]),
        "home_bp_ip_3d":  home_bip,
        "home_bp_li":     home_bp_li_val,
        "home_bp_il_ct":  home_bp_il,
        # Away BP
        "away_bp_xera":   _fip_bp(game["away_team"]),
        "away_bp_ip_3d":  away_bip,
        "away_bp_li":     away_bp_li_val,
        "away_bp_il_ct":  away_bp_il,
        # Home lineup
        "home_lineup_woba":   home_b["woba"],
        "home_ops_14d":       home_ops,
        "home_risp_14d":      home_risp,
        "home_starters_il":   home_st_il,
        "home_run_diff":      home_rd,
        "home_win_pct":           home_wpct,
        "home_back_to_back":      home_b2b,
        "home_series_game":       home_series,
        "home_win_streak":        home_streak,
        "home_team_days_rest":    home_drest,
        # Away lineup
        "away_lineup_woba":       away_b["woba"],
        "away_ops_14d":           away_ops,
        "away_risp_14d":          away_risp,
        "away_starters_il":       away_st_il,
        "away_run_diff":          away_rd,
        "away_win_pct":           away_wpct,
        "away_back_to_back":      away_b2b,
        "away_series_game":       away_series,
        "away_win_streak":        away_streak,
        "away_team_days_rest":    away_drest,
        # Park / weather / umpire
        "park_factor_runs":       pf["runs"],
        "park_factor_hr":         pf["hr"],
        "wind_speed":             wx.get("wind_speed",   LEAGUE_AVG["wind_speed"]),
        "wind_dir_deg":           wx.get("wind_dir_deg", LEAGUE_AVG["wind_dir_deg"]),
        "temperature":            wx.get("temperature",  LEAGUE_AVG["temperature"]),
        "umpire_run_factor":      umpire_run_factor,
        "is_dome":                1 if game["home_team"] in DOME_TEAMS else 0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build MLB historical training data (v2)")
    parser.add_argument("--seasons", default="2019,2021,2022,2023,2024,2025",
                        help="Comma-separated list of game seasons (default: 2019,2021,2022,2023,2024,2025)")
    args = parser.parse_args()
    seasons = [int(s.strip()) for s in args.seasons.split(",")]

    all_rows: list[dict] = []
    all_results: list[dict] = []

    for season in seasons:
        prev = season - 1
        logger.info("=== Season %d (pitcher stats from %d) ===", season, prev)

        pitchers = fetch_pitcher_tables(prev)
        batting  = fetch_team_batting(prev)
        games    = fetch_season_schedule(season)

        if not games:
            logger.warning("No games for %d; skipping", season)
            continue

        # Fetch game logs for rolling ERA/WHIP (current season)
        all_sp_ids = [g.get("home_sp_id") for g in games] + [g.get("away_sp_id") for g in games]
        all_sp_ids = [pid for pid in all_sp_ids if pid]
        game_logs = fetch_pitcher_game_logs(season, all_sp_ids)

        # NEW: bullpen workload — relief IP per team per day
        team_id_map     = fetch_team_id_map(season)
        all_pitcher_ids = fetch_season_pitcher_ids(season)
        starter_ids     = set(game_logs.keys())
        relief_logs     = fetch_pitcher_relief_logs(season, all_pitcher_ids, starter_ids)
        daily_bp_ip     = compute_daily_bp_ip(relief_logs, team_id_map)
        logger.info("Daily BP IP computed for %d team-date pairs", len(daily_bp_ip))
        team_bp_xera    = compute_team_bp_xera(relief_logs, team_id_map, pitchers)
        logger.info("Team BP xERA computed for %d teams", len(team_bp_xera))

        # NEW: season run differentials (computed from already-fetched schedule)
        run_diffs = compute_run_diffs(games)
        logger.info("Run diffs computed for %d team-date pairs", len(run_diffs))

        # NEW: rolling 14-day team batting stats (ops_14d, risp_14d)
        team_batting_logs = fetch_team_batting_gamelogs(season)
        rolling_ops_map   = compute_rolling_ops(team_batting_logs)
        risp_map          = fetch_team_risp_season(season)
        logger.info("Rolling OPS computed for %d team-date pairs", len(rolling_ops_map))

        # NEW: historical weather per game via Open-Meteo archive
        logger.info("Fetching historical weather for season %d...", season)
        weather_map = fetch_weather_historical(season, games)

        # NEW: IL transactions for starters_il / bp_il_ct
        il_transactions = fetch_team_il_transactions(
            season, set(all_pitcher_ids), team_id_map
        )
        logger.info("IL transactions loaded for %d teams in %d", len(il_transactions), season)

        # NEW: schedule-based features (win%, back-to-back, series game#)
        schedule_features = compute_schedule_features(games)
        logger.info("Schedule features computed for %d team-date pairs", len(schedule_features))

        # NEW: win/loss streak per (team, date)
        win_streaks = compute_win_streaks(games)
        logger.info("Win streaks computed for %d team-date pairs", len(win_streaks))

        # NEW: team days of rest per (team, date)
        team_days_rest_map = compute_team_days_rest(games)
        logger.info("Team days rest computed for %d team-date pairs", len(team_days_rest_map))

        # NEW: team roster handedness (for hand_match_pct)
        roster_hands = fetch_team_roster_hands(season, team_id_map)

        # NEW: bullpen LI proxy (saves/holds/blown saves)
        bp_li_map = fetch_team_bp_li_season(season, team_id_map)

        # NEW: umpire assignments
        umpire_map = fetch_season_umpires(season)
        logger.info("Umpires loaded for %d games in %d", len(umpire_map), season)

        # Batch-resolve names to warm name cache (avoids per-game API calls)
        ids_needed = {int(g[k]) for g in games for k in ("home_sp_id","away_sp_id") if g.get(k)}
        ids_missing = ids_needed - set(pitchers.keys())
        if ids_missing:
            logger.info("Resolving %d pitcher names not in stat table...", len(ids_missing))
            for pid in ids_missing:
                resolve_name(pid)

        for game in games:
            row = build_feature_row(game, pitchers, batting, game_logs,
                                    run_diffs=run_diffs, weather_map=weather_map,
                                    bp_ip_3d_map=daily_bp_ip,
                                    rolling_ops_map=rolling_ops_map,
                                    risp_map=risp_map,
                                    il_transactions=il_transactions,
                                    schedule_features=schedule_features,
                                    win_streaks=win_streaks,
                                    roster_hands=roster_hands,
                                    bp_li_map=bp_li_map,
                                    umpire_map=umpire_map,
                                    team_days_rest=team_days_rest_map,
                                    team_bp_xera=team_bp_xera)
            all_rows.append(row)
            hs, as_ = game["home_score"], game["away_score"]
            all_results.append({
                "game_id":    game["game_id"],
                "game_date":  game["game_date"],
                "home_team":  game["home_team"],
                "away_team":  game["away_team"],
                "home_score": hs,
                "away_score": as_,
                "winner":     game["home_team"] if hs > as_ else game["away_team"],
                "total_runs": hs + as_,
            })

        logger.info("Season %d: %d rows built", season, len(games))

    if not all_rows:
        logger.error("No data collected.")
        sys.exit(1)

    features_df = pd.DataFrame(all_rows)
    results_df  = pd.DataFrame(all_results)

    for c in FEATURE_COLUMNS:
        if c not in features_df.columns:
            logger.warning("Missing column %s — filling 0", c)
            features_df[c] = 0.0

    feat_path   = DATA_DIR / "historical_features.parquet"
    result_path = DATA_DIR / "historical_results.csv"
    features_df.to_parquet(feat_path, index=False)
    results_df.to_csv(result_path, index=False)

    logger.info("Saved %d feature rows  -> %s", len(features_df), feat_path)
    logger.info("Saved %d result rows   -> %s", len(results_df),  result_path)

    # Compute and save per-umpire avg runs from this dataset (used on next build + live pipeline)
    import json as _json
    all_ump_maps = {}
    for _s in seasons:
        all_ump_maps.update(_load(f"umpires_{_s}") or {})
    game_totals = {r["game_id"]: r["total_runs"] for r in all_results}
    ump_runs: dict[str, list] = {}
    for gid, uname in all_ump_maps.items():
        if uname and uname != "TBD" and gid in game_totals:
            ump_runs.setdefault(uname, []).append(game_totals[gid])
    ump_stats = {u: round(sum(runs)/len(runs), 2) for u, runs in ump_runs.items() if len(runs) >= 10}
    ump_stats_path = DATA_DIR / "umpire_stats.json"
    ump_stats_path.write_text(_json.dumps(ump_stats, indent=2))
    logger.info("Umpire stats saved: %d umpires -> %s", len(ump_stats), ump_stats_path)

    # Coverage report
    n = len(features_df)
    for col in FEATURE_COLUMNS:
        la_val = None
        for k, v in LEAGUE_AVG.items():
            if col.endswith(k):
                la_val = v
                break
        if la_val is not None:
            imputed = (features_df[col] == la_val).sum()
            if imputed / n > 0.8:
                logger.warning("  %s: %.0f%% at league-average (low signal)", col, imputed/n*100)

    logger.info("")
    logger.info("Next step:  python model/train.py")


if __name__ == "__main__":
    main()
