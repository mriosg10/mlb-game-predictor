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
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from config import FEATURE_COLUMNS, MODEL_VERSION, TOTAL_MODEL_PATH, WIN_MODEL_PATH

logger = logging.getLogger(__name__)


class ModelNotFoundError(FileNotFoundError):
    """Raised when a required model artifact file is missing (Assumption A-02)."""


# ---------------------------------------------------------------------------
# Model loading (singleton pattern — load once, reuse across games)
# ---------------------------------------------------------------------------

_win_model:   xgb.Booster | None = None
_total_model: xgb.Booster | None = None


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

    home_win_prob   = float(np.clip(home_win_prob,   0.01, 0.99))
    predicted_total = float(np.clip(predicted_total,  0.0, 30.0))

    return home_win_prob, predicted_total


def predict_batch(
    feature_rows: list[dict[str, Any]],
    win_model: xgb.Booster,
    total_model: xgb.Booster,
) -> list[dict[str, float]]:
    """
    Run inference for multiple games in one DMatrix call (faster than per-game).
    Returns a list of {game_id, home_win_prob, predicted_total}.
    """
    if not feature_rows:
        return []

    df = pd.DataFrame(feature_rows)

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0

    X = df[FEATURE_COLUMNS].astype(float)
    X = X.replace([float("inf"), float("-inf")], 0.0).fillna(0.0)

    dmatrix = xgb.DMatrix(X, feature_names=FEATURE_COLUMNS)

    win_probs     = win_model.predict(dmatrix)
    total_preds   = total_model.predict(dmatrix)

    results = []
    for i, row in enumerate(feature_rows):
        results.append({
            "game_id":        row["game_id"],
            "home_win_prob":  float(np.clip(win_probs[i],   0.01, 0.99)),
            "predicted_total":float(np.clip(total_preds[i],  0.0, 30.0)),
            "model_version":  MODEL_VERSION,
        })
    return results
