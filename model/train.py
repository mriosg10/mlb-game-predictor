"""
XGBoost model training script.

This script is NOT part of the daily pipeline. It is run offline once per
season (or when retraining is needed) to produce the model artifacts that
inference.py loads.

Assumption A-02 states that trained model artifacts exist prior to pipeline
deployment. If they do not, run this script first.

Inputs required (place in the data/ directory):
  data/historical_features.parquet  — feature rows for past games
                                       schema matches the features table
  data/historical_results.csv       — columns: game_id, winner, total_runs
                                       where winner is the home_team abbr when
                                       home team won, else away_team abbr

Usage:
    python model/train.py [--season YYYY] [--cv-folds N]

Outputs:
    models/xgb_win_prob.json
    models/xgb_run_total.json
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow running as  python model/train.py  from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from config import BASE_DIR, FEATURE_COLUMNS, MODEL_DIR, TOTAL_MODEL_PATH, WIN_MODEL_PATH

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Hyperparameters (validated against 2021-2023 seasons; adjust for new data)
# ---------------------------------------------------------------------------

WIN_PARAMS = {
    "objective":        "binary:logistic",
    "eval_metric":      "logloss",
    "n_estimators":     400,
    "max_depth":        4,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 10,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "seed":             42,
}

TOTAL_PARAMS = {
    "objective":        "reg:squarederror",
    "eval_metric":      "mae",
    "n_estimators":     400,
    "max_depth":        4,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 10,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "seed":             42,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    feat_path = BASE_DIR / "data" / "historical_features.parquet"
    res_path  = BASE_DIR / "data" / "historical_results.csv"

    if not feat_path.exists():
        raise FileNotFoundError(
            f"Historical features not found: {feat_path}\n"
            "Build this from Retrosheet / Baseball Reference game logs.\n"
            "Schema must match the features table (see database.py)."
        )
    if not res_path.exists():
        raise FileNotFoundError(
            f"Historical results not found: {res_path}\n"
            "Required columns: game_id, home_team, winner, total_runs"
        )

    features_df = pd.read_parquet(feat_path)
    results_df  = pd.read_csv(res_path, dtype={"game_id": str})

    features_df["game_id"] = features_df["game_id"].astype(str)

    df = features_df.merge(results_df[["game_id", "winner", "total_runs"]],
                           on="game_id", how="inner")

    # Target: 1 if home team won, 0 otherwise
    df["home_won"] = (df["winner"] == df["home_team"]).astype(int)

    X = df[FEATURE_COLUMNS].astype(float)
    y_win   = df["home_won"]
    y_total = df["total_runs"].astype(float)

    # Replace inf / NaN
    X = X.replace([float("inf"), float("-inf")], np.nan)
    for col in X.columns:
        X[col] = X[col].fillna(X[col].median())

    logger.info("Training data: %d games, %d features", len(df), len(FEATURE_COLUMNS))
    return X, y_win, y_total


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_evaluate(
    X: pd.DataFrame,
    y_win: pd.Series,
    y_total: pd.Series,
    cv_folds: int = 5,
) -> tuple[xgb.XGBClassifier, xgb.XGBRegressor]:

    tscv = TimeSeriesSplit(n_splits=cv_folds)

    brier_scores, mae_scores = [], []

    # Cross-validation pass for evaluation only
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_win_tr,   y_win_val   = y_win.iloc[train_idx],   y_win.iloc[val_idx]
        y_total_tr, y_total_val = y_total.iloc[train_idx], y_total.iloc[val_idx]

        clf = xgb.XGBClassifier(**WIN_PARAMS, use_label_encoder=False)
        clf.fit(X_tr, y_win_tr, eval_set=[(X_val, y_win_val)], verbose=False)

        reg = xgb.XGBRegressor(**TOTAL_PARAMS)
        reg.fit(X_tr, y_total_tr, eval_set=[(X_val, y_total_val)], verbose=False)

        p_win   = clf.predict_proba(X_val)[:, 1]
        p_total = reg.predict(X_val)

        brier = brier_score_loss(y_win_val, p_win)
        mae   = mean_absolute_error(y_total_val, p_total)
        brier_scores.append(brier)
        mae_scores.append(mae)
        logger.info("Fold %d: Brier=%.4f  MAE=%.4f", fold + 1, brier, mae)

    logger.info(
        "CV summary: Brier=%.4f±%.4f  MAE=%.4f±%.4f",
        np.mean(brier_scores), np.std(brier_scores),
        np.mean(mae_scores),   np.std(mae_scores),
    )

    # Check acceptance thresholds (A-04)
    if np.mean(brier_scores) >= 0.23:
        logger.warning(
            "CV Brier score %.4f does not meet the <0.23 threshold (AC-04). "
            "Consider adding features or adjusting hyperparameters before deployment.",
            np.mean(brier_scores),
        )
    if np.mean(mae_scores) >= 1.8:
        logger.warning(
            "CV MAE %.4f does not meet the <1.8 threshold (AC-05). "
            "Consider adding features or adjusting hyperparameters before deployment.",
            np.mean(mae_scores),
        )

    # Final fit on all data
    logger.info("Training final models on full dataset...")
    final_clf = xgb.XGBClassifier(**WIN_PARAMS, use_label_encoder=False)
    final_clf.fit(X, y_win, verbose=False)

    final_reg = xgb.XGBRegressor(**TOTAL_PARAMS)
    final_reg.fit(X, y_total, verbose=False)

    return final_clf, final_reg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train MLB XGBoost models")
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X, y_win, y_total = load_data()
    clf, reg = train_and_evaluate(X, y_win, y_total, cv_folds=args.cv_folds)

    # Save as XGBoost native JSON format (used by inference.py via Booster.load_model)
    clf.get_booster().save_model(WIN_MODEL_PATH)
    reg.get_booster().save_model(TOTAL_MODEL_PATH)

    logger.info("Models saved:\n  %s\n  %s", WIN_MODEL_PATH, TOTAL_MODEL_PATH)


if __name__ == "__main__":
    main()
