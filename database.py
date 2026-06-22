"""
DuckDB schema management and all write/read helpers.

A single file is used for the whole system: mlb_predictions.duckdb
All timestamps are stored in UTC (NFR-05).
"""

import logging
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

import duckdb

from config import DB_PATH

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL_FEATURES = """
CREATE TABLE IF NOT EXISTS features (
  game_id               TEXT    NOT NULL,
  game_date             DATE    NOT NULL,
  cycle                 TEXT    NOT NULL CHECK (cycle IN ('A','B')),
  home_team             TEXT,
  away_team             TEXT,
  -- Home starting pitcher (13)
  home_sp_xera          FLOAT,
  home_sp_fip           FLOAT,
  home_sp_xfip          FLOAT,
  home_sp_siera         FLOAT,
  home_sp_k_pct         FLOAT,
  home_sp_bb_pct        FLOAT,
  home_sp_barrel        FLOAT,
  home_sp_hh_pct        FLOAT,
  home_sp_exit_velo     FLOAT,
  home_sp_spin          FLOAT,
  home_sp_days_rest     INT,
  home_sp_hand_match_pct FLOAT,
  home_sp_bvp_woba      FLOAT,
  home_sp_era_l3        FLOAT,
  home_sp_whip_l3       FLOAT,
  -- Away starting pitcher (13)
  away_sp_xera          FLOAT,
  away_sp_fip           FLOAT,
  away_sp_xfip          FLOAT,
  away_sp_siera         FLOAT,
  away_sp_k_pct         FLOAT,
  away_sp_bb_pct        FLOAT,
  away_sp_barrel        FLOAT,
  away_sp_hh_pct        FLOAT,
  away_sp_exit_velo     FLOAT,
  away_sp_spin          FLOAT,
  away_sp_days_rest     INT,
  away_sp_hand_match_pct FLOAT,
  away_sp_bvp_woba      FLOAT,
  away_sp_era_l3        FLOAT,
  away_sp_whip_l3       FLOAT,
  -- Home bullpen (4)
  home_bp_xera          FLOAT,
  home_bp_ip_3d         FLOAT,
  home_bp_li            FLOAT,
  home_bp_il_ct         INT,
  -- Away bullpen (4)
  away_bp_xera          FLOAT,
  away_bp_ip_3d         FLOAT,
  away_bp_li            FLOAT,
  away_bp_il_ct         INT,
  -- Home lineup / offense (5)
  home_lineup_woba      FLOAT,
  home_ops_14d          FLOAT,
  home_risp_14d         FLOAT,
  home_starters_il      INT,
  home_run_diff         FLOAT,
  -- Away lineup / offense (5)
  away_lineup_woba      FLOAT,
  away_ops_14d          FLOAT,
  away_risp_14d         FLOAT,
  away_starters_il      INT,
  away_run_diff         FLOAT,
  -- Park and weather (text wind_dir + numeric degrees)
  park_factor_runs      FLOAT,
  park_factor_hr        FLOAT,
  wind_speed            FLOAT,
  wind_dir              TEXT,
  wind_dir_deg          FLOAT,
  temperature           FLOAT,
  created_at            TIMESTAMP DEFAULT now(),
  PRIMARY KEY (game_id, cycle)
)
"""

_DDL_PREDICTIONS = """
CREATE TABLE IF NOT EXISTS predictions (
  prediction_id    TEXT  PRIMARY KEY,
  game_id          TEXT  NOT NULL,
  cycle            TEXT  NOT NULL CHECK (cycle IN ('A','B')),
  home_win_prob    FLOAT,
  away_win_prob    FLOAT,
  predicted_total  FLOAT,
  ou_prob          FLOAT,
  ou_line          FLOAT,
  model_version    TEXT,
  created_at       TIMESTAMP DEFAULT now()
)
"""

_DDL_RESULTS = """
CREATE TABLE IF NOT EXISTS results (
  game_id          TEXT  PRIMARY KEY,
  game_date        DATE,
  home_team        TEXT,
  away_team        TEXT,
  home_score       INT,
  away_score       INT,
  winner           TEXT,
  total_runs       INT,
  created_at       TIMESTAMP DEFAULT now()
)
"""

_DDL_EVALUATION_LOG = """
CREATE TABLE IF NOT EXISTS evaluation_log (
  log_date         DATE  NOT NULL,
  cycle            TEXT  NOT NULL,
  run_status       TEXT  NOT NULL CHECK (run_status IN ('SUCCESS','FAILED','PARTIAL')),
  games_evaluated  INT,
  brier_score      FLOAT,
  win_accuracy     FLOAT,
  total_mae        FLOAT,
  failure_reason   TEXT,
  created_at       TIMESTAMP DEFAULT now(),
  PRIMARY KEY (log_date, cycle)
)
"""


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

@contextmanager
def get_conn():
    conn = duckdb.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    except Exception as original_exc:
        try:
            conn.rollback()
        except Exception as rb_exc:
            # DuckDB autocommit mode raises this when rollback is called with
            # no active transaction — safe to ignore, re-raise the original.
            logger.debug("rollback skipped (autocommit): %s", rb_exc)
        raise original_exc
    finally:
        conn.close()


_FEATURES_MIGRATIONS = [
    "ALTER TABLE features ADD COLUMN IF NOT EXISTS home_sp_era_l3  FLOAT",
    "ALTER TABLE features ADD COLUMN IF NOT EXISTS home_sp_whip_l3 FLOAT",
    "ALTER TABLE features ADD COLUMN IF NOT EXISTS away_sp_era_l3  FLOAT",
    "ALTER TABLE features ADD COLUMN IF NOT EXISTS away_sp_whip_l3 FLOAT",
]

_PREDICTIONS_MIGRATIONS = [
    "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS ou_prob FLOAT",
    "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS ou_line FLOAT",
]


def init_db() -> None:
    """Create all tables if they do not exist, and run column migrations."""
    with get_conn() as conn:
        conn.execute(_DDL_FEATURES)
        conn.execute(_DDL_PREDICTIONS)
        conn.execute(_DDL_RESULTS)
        conn.execute(_DDL_EVALUATION_LOG)
        for stmt in _FEATURES_MIGRATIONS + _PREDICTIONS_MIGRATIONS:
            try:
                conn.execute(stmt)
            except Exception:
                pass
    logger.info("Database schema initialised at %s", DB_PATH)


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def upsert_features(row: dict[str, Any]) -> None:
    """Insert or replace a feature row (keyed on game_id + cycle)."""
    # Strip private metadata keys (prefixed with _) — they're not DB columns.
    row = {k: v for k, v in row.items() if not k.startswith("_")}
    row.setdefault("created_at", datetime.now(timezone.utc))
    cols = ", ".join(row.keys())
    placeholders = ", ".join(f"${i+1}" for i in range(len(row)))
    sql = (
        f"INSERT INTO features ({cols}) VALUES ({placeholders}) "
        f"ON CONFLICT (game_id, cycle) DO UPDATE SET "
        + ", ".join(f"{k} = excluded.{k}" for k in row if k not in ("game_id", "cycle"))
    )
    with get_conn() as conn:
        conn.execute(sql, list(row.values()))


def insert_prediction(
    game_id: str,
    cycle: str,
    home_win_prob: float,
    predicted_total: float,
    model_version: str,
    ou_prob: float | None = None,
    ou_line: float | None = None,
) -> str:
    pred_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    # Delete any existing prediction for this game+cycle before inserting,
    # so re-running a cycle always reflects the latest model output.
    sql_delete = "DELETE FROM predictions WHERE game_id = $1 AND cycle = $2"
    sql_insert = """
        INSERT INTO predictions
          (prediction_id, game_id, cycle, home_win_prob, away_win_prob,
           predicted_total, ou_prob, ou_line, model_version, created_at)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
    """
    with get_conn() as conn:
        conn.execute(sql_delete, [game_id, cycle])
        conn.execute(sql_insert, [
            pred_id, game_id, cycle,
            home_win_prob, round(1.0 - home_win_prob, 6),
            predicted_total, ou_prob, ou_line, model_version, now,
        ])
    return pred_id


def upsert_result(row: dict[str, Any]) -> None:
    row.setdefault("created_at", datetime.now(timezone.utc))
    cols = ", ".join(row.keys())
    placeholders = ", ".join(f"${i+1}" for i in range(len(row)))
    non_pk = [k for k in row if k != "game_id"]
    sql = (
        f"INSERT INTO results ({cols}) VALUES ({placeholders}) "
        f"ON CONFLICT (game_id) DO UPDATE SET "
        + ", ".join(f"{k} = excluded.{k}" for k in non_pk)
    )
    with get_conn() as conn:
        conn.execute(sql, list(row.values()))


def upsert_eval_log(row: dict[str, Any]) -> None:
    row.setdefault("created_at", datetime.now(timezone.utc))
    cols = ", ".join(row.keys())
    placeholders = ", ".join(f"${i+1}" for i in range(len(row)))
    non_pk = [k for k in row if k not in ("log_date", "cycle")]
    sql = (
        f"INSERT INTO evaluation_log ({cols}) VALUES ({placeholders}) "
        f"ON CONFLICT (log_date, cycle) DO UPDATE SET "
        + ", ".join(f"{k} = excluded.{k}" for k in non_pk)
    )
    with get_conn() as conn:
        conn.execute(sql, list(row.values()))


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------

def get_predictions_for_date(game_date: str, cycle: str) -> list[dict]:
    sql = """
        SELECT p.*, f.home_team, f.away_team
        FROM predictions p
        JOIN features f ON p.game_id = f.game_id AND p.cycle = f.cycle
        WHERE f.game_date = $1 AND p.cycle = $2
    """
    with get_conn() as conn:
        result = conn.execute(sql, [game_date, cycle]).fetchdf()
    return result.to_dict(orient="records")


def get_results_for_date(game_date: str) -> list[dict]:
    sql = "SELECT * FROM results WHERE game_date = $1"
    with get_conn() as conn:
        result = conn.execute(sql, [game_date]).fetchdf()
    return result.to_dict(orient="records")


def get_rolling_predictions(days: int = 14) -> "pd.DataFrame":
    import pandas as pd
    sql = f"""
        SELECT
            p.game_id, p.cycle, p.home_win_prob, p.predicted_total,
            r.winner, r.total_runs,
            f.home_team, f.game_date
        FROM predictions p
        JOIN features f ON p.game_id = f.game_id AND p.cycle = f.cycle
        LEFT JOIN results r ON p.game_id = r.game_id
        WHERE f.game_date >= (CURRENT_DATE - INTERVAL '{days} days')
          AND r.total_runs IS NOT NULL
        ORDER BY f.game_date DESC
    """
    with get_conn() as conn:
        return conn.execute(sql).fetchdf()
