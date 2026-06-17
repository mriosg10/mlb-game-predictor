"""
One-time database initialisation script.
Run before the first pipeline execution.

Usage:
    python scripts/setup_db.py
"""

import sys
from pathlib import Path

# Allow running from the scripts/ subdirectory
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logging_setup import setup_logging

setup_logging()

import database as db
from config import DB_PATH, LOG_DIR, MODEL_DIR

import logging
logger = logging.getLogger(__name__)

def main():
    # Create runtime directories
    for d in [LOG_DIR, MODEL_DIR, Path(DB_PATH).parent]:
        d.mkdir(parents=True, exist_ok=True)

    db.init_db()
    logger.info("Database ready at: %s", DB_PATH)
    logger.info("Model directory:   %s", MODEL_DIR)
    logger.info("Log directory:     %s", LOG_DIR)
    logger.info("")
    logger.info("IMPORTANT (Assumption A-02):")
    logger.info("  Trained model artifacts are required before running inference.")
    logger.info("  Place them at:")
    logger.info("    %s/xgb_win_prob.json", MODEL_DIR)
    logger.info("    %s/xgb_run_total.json", MODEL_DIR)
    logger.info("  Or run:  python model/train.py")
    logger.info("  (requires data/historical_features.parquet + data/historical_results.csv)")


if __name__ == "__main__":
    main()
