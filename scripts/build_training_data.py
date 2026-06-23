"""
Historical training data builder — v2 (FanGraphs-free).

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

from config import FEATURE_COLUMNS, LEAGUE_AVG, PARK_FACTORS_HARDCODED, VENUE_COORDS

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
    return json.load(open(p)) if p.exists() else None

def _save(key: str, data) -> None:
    json.dump(data, open(_cp(key), "w"))

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
        raw = float(ip_str or 0)
        whole = int(raw)
        thirds = round(raw - whole, 1)
        # .1 = 1/3, .2 = 2/3 — convert to decimal
        return whole + (thirds / 0.3) * (1 / 3)
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
                      run_diffs: dict = None, weather_map: dict = None) -> dict:
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

    def _fip_bp(sp_stats):
        """Bullpen xERA proxy: SP xFIP * 1.05 (bullpen ERA typically slightly higher)."""
        return min(sp_stats.get("xfip", LEAGUE_AVG["sp_xfip"]) * 1.05, 7.0)

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
        "home_sp_hand_match_pct":LEAGUE_AVG["sp_hand_match_pct"],
        "home_sp_bvp_woba":      LEAGUE_AVG["sp_bvp_woba"],
        "home_sp_era_l3":        h_era_l3,
        "home_sp_whip_l3":       h_whip_l3,
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
        "away_sp_hand_match_pct":LEAGUE_AVG["sp_hand_match_pct"],
        "away_sp_bvp_woba":      LEAGUE_AVG["sp_bvp_woba"],
        "away_sp_era_l3":        a_era_l3,
        "away_sp_whip_l3":       a_whip_l3,
        # Home BP
        "home_bp_xera":   _fip_bp(home_sp),
        "home_bp_ip_3d":  LEAGUE_AVG["bp_ip_3d"],
        "home_bp_li":     LEAGUE_AVG["bp_li"],
        "home_bp_il_ct":  int(LEAGUE_AVG["bp_il_ct"]),
        # Away BP
        "away_bp_xera":   _fip_bp(away_sp),
        "away_bp_ip_3d":  LEAGUE_AVG["bp_ip_3d"],
        "away_bp_li":     LEAGUE_AVG["bp_li"],
        "away_bp_il_ct":  int(LEAGUE_AVG["bp_il_ct"]),
        # Home lineup
        "home_lineup_woba": home_b["woba"],
        "home_ops_14d":     home_b["ops"],
        "home_risp_14d":    home_b["avg"],
        "home_starters_il": int(LEAGUE_AVG["starters_il"]),
        "home_run_diff":    home_rd,
        # Away lineup
        "away_lineup_woba": away_b["woba"],
        "away_ops_14d":     away_b["ops"],
        "away_risp_14d":    away_b["avg"],
        "away_starters_il": int(LEAGUE_AVG["starters_il"]),
        "away_run_diff":    away_rd,
        # Park / weather
        "park_factor_runs": pf["runs"],
        "park_factor_hr":   pf["hr"],
        "wind_speed":       wx.get("wind_speed",   LEAGUE_AVG["wind_speed"]),
        "wind_dir_deg":     wx.get("wind_dir_deg", LEAGUE_AVG["wind_dir_deg"]),
        "temperature":      wx.get("temperature",  LEAGUE_AVG["temperature"]),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build MLB historical training data (v2)")
    parser.add_argument("--seasons", default="2022,2023,2024",
                        help="Comma-separated list of game seasons (default: 2022,2023,2024)")
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

        # NEW: season run differentials (computed from already-fetched schedule)
        run_diffs = compute_run_diffs(games)
        logger.info("Run diffs computed for %d team-date pairs", len(run_diffs))

        # NEW: historical weather per game via Open-Meteo archive
        logger.info("Fetching historical weather for season %d...", season)
        weather_map = fetch_weather_historical(season, games)

        # Batch-resolve names to warm name cache (avoids per-game API calls)
        ids_needed = {int(g[k]) for g in games for k in ("home_sp_id","away_sp_id") if g.get(k)}
        ids_missing = ids_needed - set(pitchers.keys())
        if ids_missing:
            logger.info("Resolving %d pitcher names not in stat table...", len(ids_missing))
            for pid in ids_missing:
                resolve_name(pid)

        for game in games:
            row = build_feature_row(game, pitchers, batting, game_logs,
                                    run_diffs=run_diffs, weather_map=weather_map)
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
