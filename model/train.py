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

from config import BASE_DIR, DB_PATH, FEATURE_COLUMNS, MODEL_DIR, OU_FEATURE_COLUMNS, OU_MODEL_PATH, TOTAL_MODEL_PATH, WIN_MODEL_PATH

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

OU_PARAMS = {
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

def load_live_data() -> pd.DataFrame:
    """
    Pull features + results from the live DuckDB database.
    Uses Cycle B rows (confirmed lineups + weather) where available,
    falling back to Cycle A for games without a B prediction.
    Returns a DataFrame with FEATURE_COLUMNS + home_team + winner + total_runs.
    """
    import duckdb
    feature_cols_sql = ", ".join(f"f.{c}" for c in FEATURE_COLUMNS)
    sql = f"""
        SELECT f.game_id, f.home_team, f.game_date,
               {feature_cols_sql},
               r.winner, r.total_runs
        FROM features f
        JOIN results r ON f.game_id = r.game_id
        WHERE f.cycle = 'B'
        UNION ALL
        SELECT f.game_id, f.home_team, f.game_date,
               {feature_cols_sql},
               r.winner, r.total_runs
        FROM features f
        JOIN results r ON f.game_id = r.game_id
        WHERE f.cycle = 'A'
          AND f.game_id NOT IN (SELECT game_id FROM features WHERE cycle = 'B')
    """
    try:
        conn = duckdb.connect(DB_PATH, read_only=True)
        df = conn.execute(sql).fetchdf()
        conn.close()
        df["game_id"] = df["game_id"].astype(str)
        logger.info("Live DB: %d games with results", len(df))
        return df
    except Exception as exc:
        logger.warning("Live data load failed: %s", exc)
        return pd.DataFrame()


def load_data(include_live: bool = False) -> tuple[pd.DataFrame, pd.Series, pd.Series, int]:
    """
    Load training data.  Returns (X, y_win, y_total, n_live).

    include_live=True merges the live DuckDB game records on top of the
    historical parquet, preferring live rows where game_ids overlap.
    """
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
    df["home_won"] = (df["winner"] == df["home_team"]).astype(int)

    n_live = 0
    if include_live:
        live_df = load_live_data()
        if not live_df.empty:
            live_df["home_won"] = (live_df["winner"] == live_df["home_team"]).astype(int)
            # Live rows take precedence over historical rows for the same game
            df = pd.concat(
                [df[~df["game_id"].isin(live_df["game_id"])], live_df],
                ignore_index=True,
            )
            n_live = len(live_df)

    X = df[FEATURE_COLUMNS].astype(float)
    y_win   = df["home_won"]
    y_total = df["total_runs"].astype(float)

    X = X.replace([float("inf"), float("-inf")], np.nan)
    for col in X.columns:
        X[col] = X[col].fillna(X[col].median())

    logger.info("Training data: %d games total (%d live), %d features",
                len(df), n_live, len(FEATURE_COLUMNS))
    return X, y_win, y_total, n_live


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_evaluate(
    X: pd.DataFrame,
    y_win: pd.Series,
    y_total: pd.Series,
    cv_folds: int = 5,
    return_metrics: bool = False,
):
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

    if return_metrics:
        cv_metrics = {
            "brier_mean": float(np.mean(brier_scores)),
            "brier_std":  float(np.std(brier_scores)),
            "mae_mean":   float(np.mean(mae_scores)),
            "mae_std":    float(np.std(mae_scores)),
        }
        return final_clf, final_reg, cv_metrics

    return final_clf, final_reg


# ---------------------------------------------------------------------------
# OU classifier training
# ---------------------------------------------------------------------------

def train_ou_model(
    features_df: pd.DataFrame,
    cv_folds: int = 5,
    return_metrics: bool = False,
):
    """
    Train a calibrated XGBoost classifier that predicts P(total > ou_line).

    Requires features_df to contain all FEATURE_COLUMNS plus:
      - 'ou_line'    (float): sportsbook over/under line for that game
      - 'total_runs' (int):   actual total runs scored

    Returns None and logs a warning when the required data is unavailable.
    """
    ou_path = BASE_DIR / "data" / "historical_ou.csv"
    def _no_model():
        return (None, {}) if return_metrics else None

    if not ou_path.exists():
        logger.warning(
            "OU model training skipped — %s not found.\n"
            "  Create this CSV with columns: game_id, ou_line\n"
            "  (one row per historical game where you have the sportsbook O/U line)",
            ou_path,
        )
        return _no_model()

    ou_df = pd.read_csv(ou_path, dtype={"game_id": str})
    if "ou_line" not in ou_df.columns:
        logger.warning("OU model training skipped — 'ou_line' column missing in %s", ou_path)
        return _no_model()

    df = features_df.merge(ou_df[["game_id", "ou_line"]], on="game_id", how="inner")
    if df.empty:
        logger.warning("OU model training skipped — no games matched after join with %s", ou_path)
        return _no_model()

    df = df.dropna(subset=["ou_line", "total_runs"])
    df["over"] = (df["total_runs"] > df["ou_line"]).astype(int)

    X = df[OU_FEATURE_COLUMNS].astype(float)
    X = X.replace([float("inf"), float("-inf")], np.nan)
    for col in X.columns:
        X[col] = X[col].fillna(X[col].median())
    y = df["over"]

    logger.info("OU training data: %d games", len(df))

    tscv = TimeSeriesSplit(n_splits=cv_folds)
    brier_scores = []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        clf = xgb.XGBClassifier(**OU_PARAMS, use_label_encoder=False)
        clf.fit(X.iloc[tr_idx], y.iloc[tr_idx], verbose=False)
        p = clf.predict_proba(X.iloc[val_idx])[:, 1]
        brier = brier_score_loss(y.iloc[val_idx], p)
        brier_scores.append(brier)
        logger.info("OU fold %d: Brier=%.4f", fold + 1, brier)

    logger.info("OU CV Brier=%.4f±%.4f", np.mean(brier_scores), np.std(brier_scores))

    final = xgb.XGBClassifier(**OU_PARAMS, use_label_encoder=False)
    final.fit(X, y, verbose=False)

    if return_metrics:
        cv_metrics = {
            "brier_mean": float(np.mean(brier_scores)),
            "brier_std":  float(np.std(brier_scores)),
        }
        return final, cv_metrics

    return final


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train MLB XGBoost models")
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X, y_win, y_total, n_live = load_data(include_live=True)
    clf, reg = train_and_evaluate(X, y_win, y_total, cv_folds=args.cv_folds)

    # Save as XGBoost native JSON format (used by inference.py via Booster.load_model)
    clf.get_booster().save_model(WIN_MODEL_PATH)
    reg.get_booster().save_model(TOTAL_MODEL_PATH)
    logger.info("Models saved:\n  %s\n  %s", WIN_MODEL_PATH, TOTAL_MODEL_PATH)

    # OU classifier — only trained when data/historical_ou.csv exists
    feat_path = BASE_DIR / "data" / "historical_features.parquet"
    res_path  = BASE_DIR / "data" / "historical_results.csv"
    if feat_path.exists() and res_path.exists():
        features_df = pd.read_parquet(feat_path)
        results_df  = pd.read_csv(res_path, dtype={"game_id": str})
        features_df["game_id"] = features_df["game_id"].astype(str)
        features_df = features_df.merge(
            results_df[["game_id", "total_runs"]], on="game_id", how="inner"
        )
        ou_clf = train_ou_model(features_df, cv_folds=args.cv_folds)
        if ou_clf is not None:
            ou_clf.get_booster().save_model(OU_MODEL_PATH)
            logger.info("OU model saved: %s", OU_MODEL_PATH)


if __name__ == "__main__":
    main()
