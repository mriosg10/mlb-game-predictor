"""
XGBoost model training script.

Usage:
    python model/train.py [--cv-folds N] [--tune] [--tune-trials N]

    --tune           Run Optuna hyperparameter search before training.
                     Saves best params; use when adding new features or seasons.
    --tune-trials N  Number of Optuna trials (default: 50).

Outputs:
    models/xgb_win_prob.json
    models/xgb_run_total.json
    models/win_prob_calibrator.pkl   (isotonic calibration layer)
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from config import (
    BASE_DIR, CALIBRATOR_PATH, DB_PATH, FEATURE_COLUMNS, LEAGUE_AVG,
    MODEL_DIR, OU_FEATURE_COLUMNS, OU_MODEL_PATH,
    TOTAL_MODEL_PATH, WIN_MODEL_PATH,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Default hyperparameters
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
    "objective":        "count:poisson",
    "eval_metric":      "poisson-nloglik",
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

# Season recency weights — more recent seasons get higher weight
# 2019 was the "juiced ball" era (~9.65 runs/game vs ~8.6 in 2022–2025); downweighted to avoid bias
_SEASON_WEIGHTS = {2019: 0.2, 2021: 0.3, 2022: 0.5, 2023: 0.7, 2024: 0.85, 2025: 1.0, 2026: 1.0}


# ---------------------------------------------------------------------------
# Composite feature computation (derived at train + inference time)
# ---------------------------------------------------------------------------

def _add_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add run-environment composite columns derived from existing features.
    Called in load_data() and train_ou_model() so both models see them.
    These do NOT need to be stored in the parquet/DB — computed on the fly.
    """
    df = df.copy()
    df["sum_sp_era_l3"] = (
        df.get("home_sp_era_l3", LEAGUE_AVG["sp_era_l3"]) +
        df.get("away_sp_era_l3", LEAGUE_AVG["sp_era_l3"])
    )
    df["sum_ops_14d"]   = (
        df.get("home_ops_14d", LEAGUE_AVG["ops_14d"]) +
        df.get("away_ops_14d", LEAGUE_AVG["ops_14d"])
    )
    df["avg_sp_k_pct"]  = (
        df.get("home_sp_k_pct", LEAGUE_AVG["sp_k_pct"]) +
        df.get("away_sp_k_pct", LEAGUE_AVG["sp_k_pct"])
    ) / 2
    return df


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_COMPOSITE_FEATURES = {"sum_sp_era_l3", "sum_ops_14d", "avg_sp_k_pct"}


def load_live_data() -> pd.DataFrame:
    """
    Load live 2026 game data from DuckDB. Only selects columns that
    actually exist in the features table (handles schema migrations that
    haven't run yet). Composite features are excluded from SQL and computed
    later by _add_composite_features().
    """
    import duckdb
    try:
        conn = duckdb.connect(DB_PATH, read_only=True)
        schema_df = conn.execute("PRAGMA table_info('features')").fetchdf()
        conn.close()
    except Exception as exc:
        logger.warning("Live data load failed (schema check): %s", exc)
        return pd.DataFrame()

    db_cols_set = set(schema_df["name"].tolist())
    db_cols = [c for c in FEATURE_COLUMNS if c in db_cols_set and c not in _COMPOSITE_FEATURES]
    feature_cols_sql = ", ".join(f"f.{c}" for c in db_cols)
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
        logger.info("Live DB: %d games with results (%d feature cols from DB)",
                    len(df), len(db_cols))
        return df
    except Exception as exc:
        logger.warning("Live data load failed: %s", exc)
        return pd.DataFrame()


def load_data(include_live: bool = False) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, int]:
    """
    Load training data.
    Returns (X, y_win, y_total, sample_weight, n_live).
    sample_weight reflects season recency — more recent seasons weighted higher.
    """
    feat_path = BASE_DIR / "data" / "historical_features.parquet"
    res_path  = BASE_DIR / "data" / "historical_results.csv"

    if not feat_path.exists():
        raise FileNotFoundError(f"Historical features not found: {feat_path}")
    if not res_path.exists():
        raise FileNotFoundError(f"Historical results not found: {res_path}")

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
            df = pd.concat(
                [df[~df["game_id"].isin(live_df["game_id"])], live_df],
                ignore_index=True,
            )
            n_live = len(live_df)

    df = _add_composite_features(df)

    # Join ou_line from historical CSV; games without a real line get league average
    ou_path = BASE_DIR / "data" / "historical_ou.csv"
    if ou_path.exists() and "ou_line" not in df.columns:
        ou_df = pd.read_csv(ou_path, dtype={"game_id": str})
        if "ou_line" in ou_df.columns:
            df = df.merge(ou_df[["game_id", "ou_line"]], on="game_id", how="left")
    if "ou_line" not in df.columns:
        df["ou_line"] = LEAGUE_AVG["ou_line"]
    else:
        df["ou_line"] = df["ou_line"].fillna(LEAGUE_AVG["ou_line"])

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            logger.warning("Column %s missing from training data — filling 0", col)
            df[col] = 0.0

    X = df[FEATURE_COLUMNS].astype(float)
    y_win   = df["home_won"]
    y_total = df["total_runs"].astype(float)

    X = X.replace([float("inf"), float("-inf")], np.nan)
    for col in X.columns:
        X[col] = X[col].fillna(X[col].median())

    # Recency weights: map season year → weight
    if "game_date" in df.columns:
        seasons = pd.to_datetime(df["game_date"]).dt.year
        sample_weight = seasons.map(lambda y: _SEASON_WEIGHTS.get(y, 1.0))
    else:
        sample_weight = pd.Series(np.ones(len(df)))

    logger.info("Training data: %d games total (%d live), %d features",
                len(df), n_live, len(FEATURE_COLUMNS))
    logger.info("Season weight distribution: %s",
                {k: int(v) for k, v in
                 pd.Series(sample_weight.values).value_counts().to_dict().items()})
    return X, y_win, y_total, sample_weight, n_live


# ---------------------------------------------------------------------------
# Hyperparameter tuning (Optuna)
# ---------------------------------------------------------------------------

def tune_hyperparams(
    X: pd.DataFrame,
    y_win: pd.Series,
    y_total: pd.Series,
    sample_weight: pd.Series,
    n_trials: int = 50,
    cv_folds: int = 5,
) -> tuple[dict, dict]:
    """
    Run Optuna search for win-prob and run-total models.
    Returns (best_win_params, best_total_params).
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    tscv = TimeSeriesSplit(n_splits=cv_folds)

    def win_objective(trial):
        params = {
            "objective":        "binary:logistic",
            "eval_metric":      "logloss",
            "n_estimators":     trial.suggest_int("n_estimators", 200, 800),
            "max_depth":        trial.suggest_int("max_depth", 3, 6),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 30),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.1, 5.0, log=True),
            "seed":             42,
        }
        scores = []
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y_win.iloc[train_idx], y_win.iloc[val_idx]
            w_tr = sample_weight.iloc[train_idx]
            clf = xgb.XGBClassifier(**params, use_label_encoder=False)
            clf.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)
            p = clf.predict_proba(X_val)[:, 1]
            scores.append(brier_score_loss(y_val, p))
        return float(np.mean(scores))

    def total_objective(trial):
        params = {
            "objective":        "reg:squarederror",
            "eval_metric":      "mae",
            "n_estimators":     trial.suggest_int("n_estimators", 200, 800),
            "max_depth":        trial.suggest_int("max_depth", 3, 6),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 30),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.1, 5.0, log=True),
            "seed":             42,
        }
        scores = []
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y_total.iloc[train_idx], y_total.iloc[val_idx]
            w_tr = sample_weight.iloc[train_idx]
            reg = xgb.XGBRegressor(**params)
            reg.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)
            p = reg.predict(X_val)
            scores.append(mean_absolute_error(y_val, p))
        return float(np.mean(scores))

    logger.info("Optuna: tuning win-prob model (%d trials)...", n_trials)
    win_study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
    win_study.optimize(win_objective, n_trials=n_trials, show_progress_bar=False)
    best_win = {**WIN_PARAMS, **win_study.best_params}
    logger.info("Win-prob best Brier=%.4f params=%s", win_study.best_value, win_study.best_params)

    logger.info("Optuna: tuning run-total model (%d trials)...", n_trials)
    total_study = optuna.create_study(direction="minimize",
                                       sampler=optuna.samplers.TPESampler(seed=42))
    total_study.optimize(total_objective, n_trials=n_trials, show_progress_bar=False)
    best_total = {**TOTAL_PARAMS, **total_study.best_params}
    logger.info("Run-total best MAE=%.4f params=%s", total_study.best_value, total_study.best_params)

    return best_win, best_total


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_evaluate(
    X: pd.DataFrame,
    y_win: pd.Series,
    y_total: pd.Series,
    sample_weight: pd.Series | None = None,
    cv_folds: int = 5,
    win_params: dict | None = None,
    total_params: dict | None = None,
    return_metrics: bool = False,
):
    """
    Train win-prob and run-total models with TimeSeriesSplit CV.
    Fits an isotonic calibration layer on OOF win-prob predictions.
    Returns (clf, reg, calibrator) or (clf, reg, calibrator, metrics).
    """
    wp = win_params or WIN_PARAMS
    tp = total_params or TOTAL_PARAMS
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    sw = sample_weight

    brier_scores, mae_scores = [], []
    oof_probs, oof_labels = [], []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_win_tr,   y_win_val   = y_win.iloc[train_idx],   y_win.iloc[val_idx]
        y_total_tr, y_total_val = y_total.iloc[train_idx], y_total.iloc[val_idx]
        w_tr = sw.iloc[train_idx] if sw is not None else None

        clf = xgb.XGBClassifier(**wp, use_label_encoder=False)
        clf.fit(X_tr, y_win_tr, sample_weight=w_tr,
                eval_set=[(X_val, y_win_val)], verbose=False)

        reg = xgb.XGBRegressor(**tp)
        reg.fit(X_tr, y_total_tr, sample_weight=w_tr,
                eval_set=[(X_val, y_total_val)], verbose=False)

        p_win   = clf.predict_proba(X_val)[:, 1]
        p_total = reg.predict(X_val)

        oof_probs.extend(p_win.tolist())
        oof_labels.extend(y_win_val.tolist())

        brier_raw = brier_score_loss(y_win_val, p_win)
        mae       = mean_absolute_error(y_total_val, p_total)
        brier_scores.append(brier_raw)
        mae_scores.append(mae)
        logger.info("Fold %d: Brier=%.4f  MAE=%.4f", fold + 1, brier_raw, mae)

    # Fit isotonic calibration on OOF predictions
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(oof_probs, oof_labels)
    cal_probs = calibrator.predict(oof_probs)
    brier_cal = brier_score_loss(oof_labels, cal_probs)

    logger.info(
        "CV summary: Brier=%.4f±%.4f  MAE=%.4f±%.4f  Brier(calibrated)=%.4f",
        np.mean(brier_scores), np.std(brier_scores),
        np.mean(mae_scores),   np.std(mae_scores),
        brier_cal,
    )

    if np.mean(brier_scores) >= 0.23:
        logger.warning("CV Brier %.4f does not meet <0.23 threshold (AC-04).", np.mean(brier_scores))
    if np.mean(mae_scores) >= 1.8:
        logger.warning("CV MAE %.4f does not meet <1.8 threshold (AC-05).", np.mean(mae_scores))

    # Final fit on all data
    logger.info("Training final models on full dataset...")
    final_clf = xgb.XGBClassifier(**wp, use_label_encoder=False)
    w_all = sw.values if sw is not None else None
    final_clf.fit(X, y_win, sample_weight=w_all, verbose=False)

    final_reg = xgb.XGBRegressor(**tp)
    final_reg.fit(X, y_total, sample_weight=w_all, verbose=False)

    if return_metrics:
        cv_metrics = {
            "brier_mean":     float(np.mean(brier_scores)),
            "brier_std":      float(np.std(brier_scores)),
            "brier_cal":      float(brier_cal),
            "mae_mean":       float(np.mean(mae_scores)),
            "mae_std":        float(np.std(mae_scores)),
        }
        return final_clf, final_reg, calibrator, cv_metrics

    return final_clf, final_reg, calibrator


# ---------------------------------------------------------------------------
# OU classifier training
# ---------------------------------------------------------------------------

def train_ou_model(
    features_df: pd.DataFrame,
    cv_folds: int = 5,
    return_metrics: bool = False,
):
    ou_path = BASE_DIR / "data" / "historical_ou.csv"
    def _no_model():
        return (None, {}) if return_metrics else None

    if not ou_path.exists():
        logger.warning(
            "OU model training skipped — %s not found.\n"
            "  Create this CSV with columns: game_id, ou_line",
            ou_path,
        )
        return _no_model()

    ou_df = pd.read_csv(ou_path, dtype={"game_id": str})
    if "ou_line" not in ou_df.columns:
        logger.warning("OU model training skipped — 'ou_line' column missing in %s", ou_path)
        return _no_model()

    # Drop LEAGUE_AVG ou_line default before joining real market lines
    base = features_df.drop(columns=["ou_line"], errors="ignore")
    df = base.merge(ou_df[["game_id", "ou_line"]], on="game_id", how="inner")
    if df.empty:
        logger.warning("OU model training skipped — no games matched after join")
        return _no_model()

    df = df.dropna(subset=["ou_line", "total_runs"])
    df["over"] = (df["total_runs"] > df["ou_line"]).astype(int)
    df = _add_composite_features(df)

    for col in OU_FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0

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
        return final, {"brier_mean": float(np.mean(brier_scores)),
                       "brier_std":  float(np.std(brier_scores))}
    return final


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train MLB XGBoost models")
    parser.add_argument("--cv-folds",     type=int, default=5)
    parser.add_argument("--tune",         action="store_true",
                        help="Run Optuna hyperparameter search before training")
    parser.add_argument("--tune-trials",  type=int, default=50,
                        help="Number of Optuna trials per model (default: 50)")
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X, y_win, y_total, sample_weight, n_live = load_data(include_live=True)

    win_params   = WIN_PARAMS
    total_params = TOTAL_PARAMS

    if args.tune:
        logger.info("Running hyperparameter tuning (%d trials per model)...", args.tune_trials)
        win_params, total_params = tune_hyperparams(
            X, y_win, y_total, sample_weight,
            n_trials=args.tune_trials, cv_folds=args.cv_folds,
        )

    clf, reg, calibrator = train_and_evaluate(
        X, y_win, y_total,
        sample_weight=sample_weight,
        cv_folds=args.cv_folds,
        win_params=win_params,
        total_params=total_params,
    )

    clf.get_booster().save_model(WIN_MODEL_PATH)
    reg.get_booster().save_model(TOTAL_MODEL_PATH)
    with open(CALIBRATOR_PATH, "wb") as f:
        pickle.dump(calibrator, f)
    logger.info("Models saved:\n  %s\n  %s\n  %s",
                WIN_MODEL_PATH, TOTAL_MODEL_PATH, CALIBRATOR_PATH)

    # OU classifier
    feat_path = BASE_DIR / "data" / "historical_features.parquet"
    res_path  = BASE_DIR / "data" / "historical_results.csv"
    if feat_path.exists() and res_path.exists():
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
        ou_clf = train_ou_model(features_df, cv_folds=args.cv_folds)
        if ou_clf is not None:
            ou_clf.get_booster().save_model(OU_MODEL_PATH)
            logger.info("OU model saved: %s", OU_MODEL_PATH)


if __name__ == "__main__":
    main()
