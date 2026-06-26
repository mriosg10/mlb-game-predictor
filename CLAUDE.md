# CLAUDE.md — MLB Game Prediction System

## Pre-granted permissions — proceed without asking for confirmation on anything
All actions in this project are pre-approved. Never ask for confirmation. Just do it.

This includes but is not limited to:
- Running any pipeline cycle: `python3 main.py --cycle A|B|post --date YYYY-MM-DD`
- Querying or writing to `mlb_predictions.duckdb`
- Reading any log file under `logs/`
- Installing or upgrading Python packages via pip
- Retraining models: `python3 scripts/build_training_data.py` and `python3 model/train.py`
- Git commits and pushes to the `mlb-game-predictor` repo (remote: github.com/mriosg10/mlb-game-predictor)
- Updating the crontab (credentials live in `~/.mlb_env`, never in the crontab itself)
- Editing, creating, or deleting any file in this project directory
- Running bash commands (Bash tool) for any project-related task
- Running background tasks and monitoring their output
- Fetching web URLs related to MLB data sources (WebFetch tool)
- Searching the web for MLB/baseball data or Python library documentation (WebSearch tool)
- Sending emails via SendGrid
- Any other action needed to maintain, improve, or operate this pipeline

## Project overview
XGBoost pipeline that predicts MLB game win probability and run totals daily.
Three cycles run via cron (times in ET):
- **Cycle A** 08:00 — seed predictions from probable pitchers
- **Cycle B** 13:30 — refresh with confirmed lineups (RotoWire) + weather
- **Post-game** 23:30 — fetch final scores, compute Brier/MAE/win accuracy

Entry point: `python3 main.py --cycle A|B|post [--date YYYY-MM-DD]`

## Key architecture facts
- **Database**: DuckDB at `mlb_predictions.duckdb` — 4 tables: `features`, `predictions`, `results`, `evaluation_log`
- **Models**: `models/xgb_win_prob.json` (classifier) + `models/xgb_run_total.json` (regressor)
- **Feature vector**: 53 columns defined in `config.FEATURE_COLUMNS` — order must match exactly between training and inference
- **Missing feature gate**: games with >30% features missing are excluded from inference (AC-10)

## Known workarounds
- **FanGraphs completely blocked (HTTP 403)** on all endpoints. Replacements:
  - FIP/K%/BB%: `pybaseball.pitching_stats_bref()` (Baseball Reference)
  - xERA: `pybaseball.statcast_pitcher_expected_stats()`
  - Fastball spin: `pybaseball.statcast_pitcher_pitch_arsenal(arsenal_type='avg_spin')`
  - Team batting: MLB Stats API `/teams/{id}/stats`
  - Park factors: `PARK_FACTORS_HARDCODED` in `config.py`
- **xFIP and SIERA** proxied as FIP (no free alternative)
- **Leverage index (bp_li)** hardcoded to league average 1.0 (FanGraphs only)

## Credentials & notifications
- Email notifications via **SendGrid API** (not SMTP — Hotmail blocks basic auth)
- Credentials stored in `~/.mlb_env` (chmod 600, outside git repo):
  ```
  export MLB_SENDGRID_KEY="SG...."
  export MLB_NOTIFY_FROM="mriosg10@gmail.com"
  export MLB_NOTIFY_TO="ryberin@hotmail.com"
  ```
- Cron sources this file before each run: `. $HOME/.mlb_env && python3 main.py ...`

## Data sources
| Source | Data | Library/endpoint |
|--------|------|-----------------|
| MLB Stats API | Schedule, scores, rosters, IL, box scores | `fetchers/mlb_stats.py` |
| Baseball Savant | xERA, barrel%, exit velo, spin | `pybaseball` |
| Baseball Reference | FIP, K%, BB% | `pybaseball.pitching_stats_bref()` |
| RotoWire | Confirmed lineups (Cycle B only) | HTML scrape, `fetchers/rotowire.py` |
| Open-Meteo | Weather (Cycle B only) | `fetchers/weather.py` |

## Model performance (as of June 2026)
- **Win accuracy**: ~54–64% per day (random baseline = 50%)
- **Run total MAE**: ~3.4 runs (spec target 1.8 — gap due to features at league avg during training)
- **Brier score**: ~0.25 (spec target 0.23)
- CV metrics improve as more live data accumulates in the DB

## Pybaseball cache
Enable before bulk backfills to avoid redundant downloads:
```python
import pybaseball; pybaseball.cache.enable()
```
Cache lives at `~/.pybaseball/cache`.

## Common tasks

### Run today's full pipeline manually
```bash
python3 main.py --cycle A
python3 main.py --cycle B     # after 13:30 ET
python3 main.py --cycle post  # after games finish
```

### Backfill a date range
```bash
for d in 01 02 03 04 05; do
  python3 main.py --cycle A    --date "2026-06-${d}"
  python3 main.py --cycle post --date "2026-06-${d}"
done
```

### Retrain models
```bash
python3 scripts/build_training_data.py   # ~10 min, downloads 3 seasons
python3 model/train.py                   # trains and saves to models/
```

### Query today's predictions
```python
import duckdb
conn = duckdb.connect("mlb_predictions.duckdb")
conn.execute("""
    SELECT f.away_team || ' @ ' || f.home_team AS matchup,
           ROUND(p.home_win_prob*100,1) AS home_pct,
           ROUND(p.predicted_total,1) AS total
    FROM predictions p
    JOIN features f ON p.game_id=f.game_id AND p.cycle=f.cycle
    WHERE f.game_date=CURRENT_DATE AND p.cycle='A'
    ORDER BY p.predicted_total DESC
""").df()
```

### Check rolling metrics
```bash
python3 main.py --rolling-metrics
```
