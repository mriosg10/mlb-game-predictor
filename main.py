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
