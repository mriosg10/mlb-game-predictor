"""
XGBoost inference module.

Assumption A-02: trained model artifacts exist at WIN_MODEL_PATH and
TOTAL_MODEL_PATH before this module is called. If the files are absent,
ModelNotFoundError is raised; the pipeline writes a FAILED status and exits.

Two separate models are used:
  xgb_win_prob.json   — binary classifier, outputs P(home team wins)
  xgb_run_total.json  — regressor, outputs expected total runs
"""

import logging
import os
import pickle
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from config import (
    CALIBRATOR_PATH, FEATURE_COLUMNS, LEAGUE_AVG, MODEL_VERSION,
    OU_FEATURE_COLUMNS, OU_MODEL_PATH, TOTAL_MODEL_PATH, WIN_MODEL_PATH,
)

logger = logging.getLogger(__name__)


class ModelNotFoundError(FileNotFoundError):
    """Raised when a required model artifact file is missing (Assumption A-02)."""


# ---------------------------------------------------------------------------
# Model loading (singleton pattern — load once, reuse across games)
# ---------------------------------------------------------------------------

_win_model:   xgb.Booster | None = None
_total_model: xgb.Booster | None = None
_ou_model:    xgb.Booster | None = None
_calibrator:  Any = None  # IsotonicRegression, or None if artifact absent


def _add_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror of train._add_composite_features — must stay in sync."""
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


def _load_calibrator() -> Any:
    """Load isotonic calibrator if artifact exists; returns None otherwise."""
    global _calibrator
    if _calibrator is not None:
        return _calibrator
    if not os.path.exists(CALIBRATOR_PATH):
        return None
    with open(CALIBRATOR_PATH, "rb") as f:
        _calibrator = pickle.load(f)
    logger.info("Calibrator loaded: %s", CALIBRATOR_PATH)
    return _calibrator


def load_ou_model() -> xgb.Booster | None:
    """
    Load the OU classifier if the artifact exists. Returns None otherwise,
    which triggers the logistic fallback in predict_ou_prob().
    """
    global _ou_model
    import os
    if _ou_model is not None:
        return _ou_model
    if not os.path.exists(OU_MODEL_PATH):
        return None
    _ou_model = xgb.Booster()
    _ou_model.load_model(OU_MODEL_PATH)
    logger.info("OU-prob model loaded: %s", OU_MODEL_PATH)
    return _ou_model


def predict_ou_prob(
    feature_row: dict[str, Any],
    predicted_total: float,
    ou_line: float | None,
) -> float | None:
    """
    Return P(actual_total > ou_line).
    - Uses XGBoost OU classifier when models/xgb_ou_prob.json exists.
    - Falls back to a logistic function of (predicted_total - ou_line).
    - Returns None when ou_line is None (no market line available).
    """
    if ou_line is None:
        return None

    diff = predicted_total - ou_line
    model = load_ou_model()

    if model is not None:
        df = pd.DataFrame([{**feature_row, "ou_line": ou_line}])
        for col in OU_FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = 0.0
        X = df[OU_FEATURE_COLUMNS].astype(float)
        X = X.replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
        dmat = xgb.DMatrix(X, feature_names=OU_FEATURE_COLUMNS)
        prob = float(model.predict(dmat)[0])
    else:
        # Logistic calibration: ±2 runs from the line → ~80% confidence
        prob = float(1.0 / (1.0 + np.exp(-diff * 0.7)))

    return float(np.clip(prob, 0.05, 0.95))


def load_models() -> tuple[xgb.Booster, xgb.Booster]:
    """
    Load both model artifacts from disk.
    Raises ModelNotFoundError if either file is missing.
    """
    global _win_model, _total_model

    import os
    for path, label in [(WIN_MODEL_PATH, "win-prob"), (TOTAL_MODEL_PATH, "run-total")]:
        if not os.path.exists(path):
            raise ModelNotFoundError(
                f"Model artifact not found: {path}\n"
                f"  ('{label}' model)\n"
                "  Resolution: train and save the model, or obtain a pre-trained artifact.\n"
                "  See model/train.py for training instructions (Assumption A-02)."
            )

    if _win_model is None:
        _win_model = xgb.Booster()
        _win_model.load_model(WIN_MODEL_PATH)
        logger.info("Win-prob model loaded: %s", WIN_MODEL_PATH)

    if _total_model is None:
        _total_model = xgb.Booster()
        _total_model.load_model(TOTAL_MODEL_PATH)
        logger.info("Run-total model loaded: %s", TOTAL_MODEL_PATH)

    return _win_model, _total_model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_game(
    feature_row: dict[str, Any],
    win_model: xgb.Booster,
    total_model: xgb.Booster,
) -> tuple[float, float]:
    """
    Run inference for a single game.

    Returns:
        (home_win_prob, predicted_total)
        home_win_prob is clipped to [0.01, 0.99].
        predicted_total is clipped to [0.0, 30.0].
    """
    # Select and order feature columns; fill any unexpected NaN with column median
    df = pd.DataFrame([feature_row])
    df = _add_composite_features(df)

    # Ensure all required columns exist; fill missing with 0 (should not happen
    # if assembler correctly gated on missing-feature threshold)
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            logger.warning("predict_game: missing column %s — filling with 0", col)
            df[col] = 0.0

    X = df[FEATURE_COLUMNS].astype(float)

    # Replace any remaining NaN / inf with 0
    X = X.replace([float("inf"), float("-inf")], 0.0).fillna(0.0)

    dmatrix = xgb.DMatrix(X, feature_names=FEATURE_COLUMNS)

    home_win_prob = float(win_model.predict(dmatrix)[0])
    predicted_total = float(total_model.predict(dmatrix)[0])

    cal = _load_calibrator()
    if cal is not None:
        home_win_prob = float(cal.predict([home_win_prob])[0])

    home_win_prob   = float(np.clip(home_win_prob,   0.01, 0.99))
    predicted_total = float(np.clip(predicted_total,  0.0, 30.0))

    return home_win_prob, predicted_total


def predict_batch(
    feature_rows: list[dict[str, Any]],
    win_model: xgb.Booster,
    total_model: xgb.Booster,
    ou_lines: dict[str, float | None] | None = None,
) -> list[dict[str, Any]]:
    """
    Run inference for multiple games in one DMatrix call (faster than per-game).
    Returns a list of {game_id, home_win_prob, predicted_total, ou_prob, ou_line}.

    ou_lines: optional dict mapping game_id (str) → sportsbook O/U line.
              Missing or None values result in ou_prob=None.
    """
    if not feature_rows:
        return []

    df = pd.DataFrame(feature_rows)
    df = _add_composite_features(df)

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0

    X = df[FEATURE_COLUMNS].astype(float)
    X = X.replace([float("inf"), float("-inf")], 0.0).fillna(0.0)

    dmatrix = xgb.DMatrix(X, feature_names=FEATURE_COLUMNS)

    win_probs   = win_model.predict(dmatrix)
    total_preds = total_model.predict(dmatrix)

    cal = _load_calibrator()
    if cal is not None:
        win_probs = cal.predict(win_probs)

    results = []
    for i, row in enumerate(feature_rows):
        game_id = row["game_id"]
        pred_total = float(np.clip(total_preds[i], 0.0, 30.0))
        line = (ou_lines or {}).get(str(game_id))
        ou_prob = predict_ou_prob(row, pred_total, line)
        results.append({
            "game_id":         game_id,
            "home_win_prob":   float(np.clip(win_probs[i], 0.01, 0.99)),
            "predicted_total": pred_total,
            "ou_prob":         ou_prob,
            "ou_line":         line,
            "model_version":   MODEL_VERSION,
        })
    return results
