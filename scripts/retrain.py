"""
Weekly retrain script — merges historical parquet + live DuckDB data.

Cron: 0 5 * * 1  (Monday 5 AM UTC / 1 AM ET)

Usage:
    python scripts/retrain.py [--full-refresh] [--cv-folds N]

    --full-refresh   Also re-run build_training_data.py to pull fresh
                     historical data from Baseball Reference / Statcast.
                     Do this monthly; the default weekly run skips it.
    --cv-folds N     Number of TimeSeriesSplit folds (default: 5).
"""

import argparse
import logging
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import duckdb
import numpy as np

from config import (
    BASE_DIR, DB_PATH, MODEL_DIR,
    OU_MODEL_PATH, TOTAL_MODEL_PATH, WIN_MODEL_PATH,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger(__name__)

DATA_DIR = BASE_DIR / "data"
OU_MIN_GAMES = 100  # minimum games with a real ou_line before training OU model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _backup_models(today: str) -> list[str]:
    backed_up = []
    for path_str in [WIN_MODEL_PATH, TOTAL_MODEL_PATH, OU_MODEL_PATH]:
        p = Path(path_str)
        if p.exists():
            backup = p.with_name(f"{p.stem}_{today}{p.suffix}")
            shutil.copy2(p, backup)
            logger.info("Backed up: %s → %s", p.name, backup.name)
            backed_up.append(backup.name)
    return backed_up


def _write_version(version: str) -> None:
    version_path = MODEL_DIR / "version.txt"
    version_path.write_text(version)
    logger.info("Model version set to %s", version)


def _build_ou_csv() -> int:
    """
    Export (game_id, ou_line, total_runs) from the live DB into
    data/historical_ou.csv so train_ou_model() can find it.
    Returns the number of games written.
    """
    try:
        conn = duckdb.connect(DB_PATH, read_only=True)
        df = conn.execute("""
            SELECT p.game_id, p.ou_line, r.total_runs
            FROM predictions p
            JOIN results r ON p.game_id = r.game_id
            WHERE p.ou_line IS NOT NULL
              AND p.cycle = 'B'
            UNION ALL
            SELECT p.game_id, p.ou_line, r.total_runs
            FROM predictions p
            JOIN results r ON p.game_id = r.game_id
            WHERE p.ou_line IS NOT NULL
              AND p.cycle = 'A'
              AND p.game_id NOT IN (
                  SELECT game_id FROM predictions WHERE cycle = 'B' AND ou_line IS NOT NULL
              )
        """).fetchdf()
        conn.close()
        if df.empty:
            return 0
        out = DATA_DIR / "historical_ou.csv"
        df.to_csv(out, index=False)
        logger.info("OU training CSV: %d games → %s", len(df), out)
        return len(df)
    except Exception as exc:
        logger.warning("Could not build OU CSV: %s", exc)
        return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Weekly MLB model retrain")
    parser.add_argument("--full-refresh", action="store_true",
                        help="Re-run build_training_data.py before training")
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()

    today = date.today().strftime("%Y%m%d")
    model_version = f"v1.{today}"
    logger.info("=== MLB retrain starting — %s ===", today)

    # ------------------------------------------------------------------
    # Step 1: Optional historical data refresh (monthly)
    # ------------------------------------------------------------------
    if args.full_refresh:
        logger.info("Running full historical data refresh (~10 min)...")
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "build_training_data.py")],
            cwd=ROOT,
        )
        if result.returncode != 0:
            logger.error("build_training_data.py failed — aborting retrain")
            sys.exit(1)

    # ------------------------------------------------------------------
    # Step 2: Backup existing models
    # ------------------------------------------------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    backed_up = _backup_models(today)

    # ------------------------------------------------------------------
    # Step 3: Train win-prob + run-total models
    # ------------------------------------------------------------------
    from model.train import load_data, train_and_evaluate

    try:
        X, y_win, y_total, n_live = load_data(include_live=True)
    except FileNotFoundError as exc:
        logger.error("Training data missing: %s", exc)
        sys.exit(1)

    n_total = len(X)
    clf, reg, cv_metrics = train_and_evaluate(
        X, y_win, y_total, cv_folds=args.cv_folds, return_metrics=True
    )

    clf.get_booster().save_model(WIN_MODEL_PATH)
    reg.get_booster().save_model(TOTAL_MODEL_PATH)
    logger.info("Win + run-total models saved")

    # ------------------------------------------------------------------
    # Step 4: OU classifier (only when enough games with real lines)
    # ------------------------------------------------------------------
    ou_trained = False
    ou_brier   = None
    n_ou = _build_ou_csv()

    if n_ou >= OU_MIN_GAMES:
        from model.train import train_ou_model
        import pandas as pd

        feat_path = DATA_DIR / "historical_features.parquet"
        res_path  = DATA_DIR / "historical_results.csv"
        if feat_path.exists() and res_path.exists():
            features_df = pd.read_parquet(feat_path)
            results_df  = pd.read_csv(res_path, dtype={"game_id": str})
            features_df["game_id"] = features_df["game_id"].astype(str)
            features_df = features_df.merge(
                results_df[["game_id", "total_runs"]], on="game_id", how="inner"
            )
            ou_clf, ou_cv = train_ou_model(
                features_df, cv_folds=args.cv_folds, return_metrics=True
            )
            if ou_clf is not None:
                ou_clf.get_booster().save_model(OU_MODEL_PATH)
                ou_trained = True
                ou_brier   = ou_cv.get("brier_mean")
                logger.info("OU model saved (Brier=%.4f)", ou_brier or 0)
    else:
        logger.info(
            "OU model skipped — only %d games with ou_line (need %d)",
            n_ou, OU_MIN_GAMES,
        )

    # ------------------------------------------------------------------
    # Step 5: Write new model version
    # ------------------------------------------------------------------
    _write_version(model_version)

    # ------------------------------------------------------------------
    # Step 6: Reload inference singleton so next prediction uses new models
    # ------------------------------------------------------------------
    import model.inference as inf
    inf._win_model   = None
    inf._total_model = None
    inf._ou_model    = None
    logger.info("Inference model cache cleared — new models will load on next run")

    # ------------------------------------------------------------------
    # Step 7: Notify
    # ------------------------------------------------------------------
    from utils.notifier import notify_retrain
    notify_retrain({
        "retrain_date":  date.today().isoformat(),
        "model_version": model_version,
        "n_total":       n_total,
        "n_live":        n_live,
        "win_brier":     cv_metrics.get("brier_mean"),
        "win_brier_std": cv_metrics.get("brier_std"),
        "win_mae":       cv_metrics.get("mae_mean"),
        "win_mae_std":   cv_metrics.get("mae_std"),
        "ou_trained":    ou_trained,
        "ou_brier":      ou_brier,
        "backed_up":     backed_up,
    })

    logger.info("=== Retrain complete — %s ===", model_version)


if __name__ == "__main__":
    main()
