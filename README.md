# MLB Game Prediction System

Automated daily pipeline that predicts MLB game outcomes (win probability + run total) using XGBoost models trained on 3 seasons of historical Statcast and Baseball Reference data.

---

## How It Works

The pipeline runs in three cycles each day:

| Cycle | Time (ET) | What it does |
|-------|-----------|--------------|
| **Cycle A** — Seed | 08:00 | Fetches probable pitchers, builds pitcher/bullpen/park features, runs inference. Seeds all predictions for the day. |
| **Cycle B** — Lock | 13:30 | Re-runs with confirmed lineups (RotoWire) and weather (Open-Meteo). More accurate than Cycle A. |
| **Post-game** | 23:30 | Fetches final scores, computes Brier score / win accuracy / MAE, stores results for model evaluation. |

An email notification is sent to the configured recipient after each cycle.

---

## Prerequisites

- Python 3.10+
- pip packages (install once):

```bash
pip install xgboost duckdb pybaseball requests beautifulsoup4 pandas numpy scikit-learn
```

---

## First-Time Setup

### 1. Initialize the database

```bash
cd "artifacts-20260616T193406Z"
python3 main.py --setup-db
```

### 2. Train the models

Requires historical data in `data/` (built automatically by the training data script):

```bash
python3 scripts/build_training_data.py   # ~5–10 min; downloads 3 seasons from pybaseball
python3 model/train.py                   # trains XGBoost models; saves to models/
```

Models are saved to `models/xgb_win_prob.json` and `models/xgb_run_total.json`.

### 3. Set up email notifications

Export your Outlook credentials before running (or add them to `~/.bashrc`):

```bash
export MLB_SMTP_USER="your@outlook.com"
export MLB_SMTP_PASS="your-password"
export MLB_NOTIFY_TO="ryberin@hotmail.com"   # already the default
```

If these variables are not set, the pipeline still runs — email is simply skipped.

### 4. Install cron jobs

```bash
bash scripts/cron_setup.sh
```

This installs 4 cron jobs (Cycle A, B, post-game, and a post-game retry at 00:30 ET).
Cron runs automatically on startup via systemd.

---

## Running Manually

```bash
# Today's predictions
python3 main.py --cycle A

# Confirmed lineups + weather (run after 13:30 ET)
python3 main.py --cycle B

# Fetch results and evaluate (run after games finish)
python3 main.py --cycle post

# Any past date
python3 main.py --cycle A    --date 2026-06-15
python3 main.py --cycle post --date 2026-06-15

# 14-day rolling metrics
python3 main.py --rolling-metrics
```

---

## Output

All outputs land in `mlb_predictions.duckdb` (DuckDB database):

| Table | Contents |
|-------|----------|
| `features` | 49-column feature vector per game per cycle |
| `predictions` | Home win probability + predicted run total |
| `results` | Actual final scores |
| `evaluation_log` | Daily Brier score, win accuracy, MAE |

Quick query to see today's predictions:

```bash
python3 - <<'EOF'
import duckdb
conn = duckdb.connect("mlb_predictions.duckdb")
print(conn.execute("""
    SELECT f.away_team || ' @ ' || f.home_team AS matchup,
           ROUND(p.home_win_prob*100,1) AS home_pct,
           ROUND(p.predicted_total,1) AS total
    FROM predictions p
    JOIN features f ON p.game_id=f.game_id AND p.cycle=f.cycle
    WHERE f.game_date=CURRENT_DATE AND p.cycle='A'
    ORDER BY p.predicted_total DESC
""").df().to_string(index=False))
EOF
```

---

## Project Structure

```
artifacts-20260616T193406Z/
├── main.py                    # CLI entry point
├── config.py                  # All settings and constants
├── database.py                # DuckDB schema + upsert helpers
│
├── pipeline/
│   ├── cycle_a.py             # Cycle A runner
│   ├── cycle_b.py             # Cycle B runner
│   ├── post_game.py           # Post-game actuals + evaluation
│   └── evaluation.py          # Brier / MAE / rolling metrics
│
├── features/
│   └── assembler.py           # Builds the 49-feature vector per game
│
├── fetchers/
│   ├── mlb_stats.py           # MLB Stats API (schedule, scores, roster)
│   ├── fangraphs.py           # Pitcher stats via Baseball Reference + Savant
│   ├── savant.py              # Baseball Savant Statcast data
│   ├── rotowire.py            # Confirmed lineup scraper
│   └── weather.py             # Open-Meteo weather API
│
├── model/
│   ├── train.py               # Offline training script
│   └── inference.py           # Batch inference using saved XGBoost models
│
├── utils/
│   ├── retry.py               # Exponential backoff decorator
│   └── notifier.py            # Outlook email notifications
│
├── models/
│   ├── xgb_win_prob.json      # Trained binary classifier
│   └── xgb_run_total.json     # Trained regressor
│
├── data/
│   ├── historical_features.parquet
│   └── historical_results.csv
│
├── scripts/
│   ├── build_training_data.py # Collects 3 seasons of training data
│   └── cron_setup.sh          # Installs cron jobs
│
└── logs/
    ├── cycle_a.log
    ├── cycle_b.log
    ├── post_game.log
    └── post_game_retry.log
```

---

## Model Performance (June 2026, 201 games)

| Metric | Value | Target |
|--------|-------|--------|
| Win accuracy | 53.8% | > 50% |
| Avg Brier score | 0.252 | < 0.23 |
| Avg run total MAE | 3.43 | < 1.8 |

The Brier and MAE targets are not yet met because live-only features (lineup hotness, bullpen fatigue, days rest) default to league average during training. Accuracy improves as more live cycle data accumulates.

---

## Data Sources

| Source | Data | Access |
|--------|------|--------|
| MLB Stats API | Schedule, scores, rosters, IL | Free, no auth |
| Baseball Savant (pybaseball) | xERA, barrel%, exit velocity, spin | Free, no auth |
| Baseball Reference (pybaseball) | FIP, K%, BB% | Free, no auth |
| RotoWire | Confirmed batting lineups | Free HTML scrape |
| Open-Meteo | Game-time temperature, wind | Free, no auth |

> FanGraphs is blocked (HTTP 403). FIP proxies FIP/xFIP/SIERA; OBP proxies wOBA at team level.
