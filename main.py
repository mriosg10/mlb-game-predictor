"""
MLB Game Prediction System — entry point.

Usage:
    python main.py --cycle A              # Cycle A (seed)
    python main.py --cycle B              # Cycle B (lock)
    python main.py --cycle post           # Post-game actuals
    python main.py --cycle A --date 2026-06-15   # Run for a specific date
    python main.py --setup-db            # Initialise database only
    python main.py --rolling-metrics     # Print 14-day rolling metrics
"""

import argparse
import json
import logging
import sys
from datetime import date

from utils.logging_setup import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MLB Game Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cycle",
        choices=["A", "B", "post"],
        help="Pipeline cycle to run",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Game date in YYYY-MM-DD format (default: today)",
    )
    parser.add_argument(
        "--setup-db",
        action="store_true",
        help="Initialise database schema and exit",
    )
    parser.add_argument(
        "--rolling-metrics",
        action="store_true",
        help="Print 14-day rolling evaluation metrics and exit",
    )
    return parser.parse_args()


def _maybe_train_ou_model() -> None:
    """
    Train the OU classifier automatically the first time enough ou_line games
    accumulate. Called after each successful post-game cycle.
    Skipped if the model artifact already exists (weekly retrain handles updates).
    """
    import os
    from config import BASE_DIR, DB_PATH, OU_MIN_GAMES, OU_MODEL_PATH

    if os.path.exists(OU_MODEL_PATH):
        return  # Already trained; weekly retrain handles subsequent updates

    import duckdb
    try:
        conn = duckdb.connect(DB_PATH, read_only=True)
        n_ou = conn.execute("""
            SELECT COUNT(*) FROM predictions p
            JOIN results r ON p.game_id = r.game_id
            WHERE p.ou_line IS NOT NULL AND p.cycle = 'B'
        """).fetchone()[0]
        conn.close()
    except Exception as exc:
        logger.warning("OU threshold check failed: %s", exc)
        return

    if n_ou < OU_MIN_GAMES:
        logger.info("OU model not ready yet: %d/%d games with ou_line captured",
                    n_ou, OU_MIN_GAMES)
        return

    logger.info("OU threshold reached (%d/%d) — training OU classifier...",
                n_ou, OU_MIN_GAMES)

    import pandas as pd
    from model.train import load_live_data, train_ou_model

    feat_path = BASE_DIR / "data" / "historical_features.parquet"
    res_path  = BASE_DIR / "data" / "historical_results.csv"
    if not feat_path.exists() or not res_path.exists():
        logger.warning("OU auto-train skipped — historical data files missing")
        return

    features_df = pd.read_parquet(feat_path)
    results_df  = pd.read_csv(res_path, dtype={"game_id": str})
    features_df["game_id"] = features_df["game_id"].astype(str)
    features_df = features_df.merge(
        results_df[["game_id", "total_runs"]], on="game_id", how="inner"
    )
    live_df = load_live_data()
    if not live_df.empty:
        features_df = pd.concat(
            [features_df, live_df[~live_df["game_id"].isin(features_df["game_id"])]],
            ignore_index=True,
        )

    ou_clf = train_ou_model(features_df)
    if ou_clf is None:
        logger.warning("OU auto-train returned no model")
        return

    ou_clf.get_booster().save_model(OU_MODEL_PATH)
    logger.info("OU model saved: %s (%d games)", OU_MODEL_PATH, n_ou)

    import model.inference as inf
    inf._ou_model = None  # force singleton reload on next prediction


def main() -> int:
    args = parse_args()

    # Always initialise the DB first (idempotent CREATE IF NOT EXISTS)
    import database as db
    db.init_db()

    if args.setup_db:
        logger.info("Database initialised successfully.")
        return 0

    if args.rolling_metrics:
        from pipeline.evaluation import compute_rolling_metrics
        metrics = compute_rolling_metrics(days=14)
        print(json.dumps(metrics, indent=2, default=str))
        return 0

    if args.cycle is None:
        logger.error("--cycle is required unless --setup-db or --rolling-metrics is given")
        return 1

    # Parse date
    if args.date:
        try:
            game_date = date.fromisoformat(args.date)
        except ValueError:
            logger.error("Invalid date format: %s (expected YYYY-MM-DD)", args.date)
            return 1
    else:
        game_date = date.today()

    # Dispatch to the appropriate pipeline cycle
    if args.cycle == "A":
        from pipeline.cycle_a import run
    elif args.cycle == "B":
        from pipeline.cycle_b import run
    else:  # post
        from pipeline.post_game import run

    result = run(game_date=game_date)

    # After post-game, check whether the OU model should be trained for the first time
    if args.cycle == "post" and result.get("run_status") in ("SUCCESS", "PARTIAL"):
        _maybe_train_ou_model()

    # Print human-readable summary
    status = result.get("run_status", "UNKNOWN")
    n = result.get("games_evaluated", 0)
    reason = result.get("failure_reason", "")
    logger.info(
        "Cycle %s | %s | %d games | %s",
        args.cycle, status, n, reason or "—",
    )

    # Exit code: 0 = SUCCESS or PARTIAL, 1 = FAILED
    return 0 if status in ("SUCCESS", "PARTIAL") else 1


if __name__ == "__main__":
    sys.exit(main())
