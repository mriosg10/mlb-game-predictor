"""
Daily evaluation metrics computation.
Computes Brier score, win accuracy, and run-total MAE after post-game actuals
are written. Results are persisted to evaluation_log (FR-07).
"""

import logging
from datetime import date, datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

import database as db

logger = logging.getLogger(__name__)


def compute_daily_metrics(
    game_date: date,
    cycle: str,
) -> dict[str, Any]:
    """
    Compute metrics for predictions made on game_date under the given cycle.

    Returns a dict ready for upsert_eval_log().
    Returns partial results when some games are missing actuals.
    """
    preds = db.get_predictions_for_date(game_date.strftime("%Y-%m-%d"), cycle)
    results = db.get_results_for_date(game_date.strftime("%Y-%m-%d"))

    results_map = {r["game_id"]: r for r in results}

    matched: list[dict] = []
    for p in preds:
        r = results_map.get(p["game_id"])
        if r is None:
            continue
        if r.get("total_runs") is None:
            continue
        matched.append({
            "home_win_prob":   p["home_win_prob"],
            "home_team":       p.get("home_team", ""),
            "winner":          r["winner"],
            "total_runs":      r["total_runs"],
            "predicted_total": p["predicted_total"],
        })

    n = len(matched)
    status = "SUCCESS" if n > 0 else "PARTIAL"

    if n == 0:
        logger.warning("No matched game results found for %s cycle=%s", game_date, cycle)
        return {
            "log_date":       game_date,
            "cycle":          cycle,
            "run_status":     "PARTIAL",
            "games_evaluated": 0,
            "brier_score":    None,
            "win_accuracy":   None,
            "total_mae":      None,
            "failure_reason": "No completed game results available",
        }

    df = pd.DataFrame(matched)

    # Brier score: B = mean((p_home_win - actual_home_win)^2)
    actual_home_win = (df["winner"] == df["home_team"]).astype(float).values
    pred_home_win   = df["home_win_prob"].astype(float).values
    brier = float(np.mean((pred_home_win - actual_home_win) ** 2))

    # Win accuracy: predicted winner correct
    pred_winner_correct = (
        ((pred_home_win >= 0.5) & (actual_home_win == 1.0)) |
        ((pred_home_win < 0.5)  & (actual_home_win == 0.0))
    )
    win_acc = float(pred_winner_correct.mean())

    # Run total MAE
    mae = float(np.mean(np.abs(df["predicted_total"].values - df["total_runs"].values)))

    logger.info(
        "Metrics for %s cycle=%s: n=%d  Brier=%.4f  WinAcc=%.3f  MAE=%.3f",
        game_date, cycle, n, brier, win_acc, mae,
    )

    return {
        "log_date":        game_date,
        "cycle":           cycle,
        "run_status":      status,
        "games_evaluated": n,
        "brier_score":     round(brier, 6),
        "win_accuracy":    round(win_acc, 6),
        "total_mae":       round(mae, 6),
        "failure_reason":  None,
    }


def compute_rolling_metrics(days: int = 14) -> dict[str, Any]:
    """
    Compute rolling metrics over the last `days` days for Cycle B (AC-04, AC-05).
    Used by the post-game run to assess production readiness.
    """
    df = db.get_rolling_predictions(days=days)
    if df.empty:
        logger.warning("No rolling data available for %d-day evaluation", days)
        return {}

    # Only Cycle B for AC-04/05/06
    df_b = df[df["cycle"] == "B"].copy()
    df_a = df[df["cycle"] == "A"].copy()

    metrics: dict[str, Any] = {"window_days": days}

    for label, subset in [("B", df_b), ("A", df_a)]:
        if subset.empty:
            continue
        actual_hw = (subset["winner"] == subset["home_team"]).astype(float).values
        pred_hw   = subset["home_win_prob"].astype(float).values
        brier = float(np.mean((pred_hw - actual_hw) ** 2))
        win_acc = float(
            ((pred_hw >= 0.5) == (actual_hw == 1.0)).mean()
        )
        mae = float(
            np.mean(np.abs(subset["predicted_total"].values - subset["total_runs"].values))
        )
        metrics[f"cycle_{label}_brier"]      = round(brier, 6)
        metrics[f"cycle_{label}_win_accuracy"]= round(win_acc, 6)
        metrics[f"cycle_{label}_mae"]        = round(mae, 6)
        metrics[f"cycle_{label}_n"]          = len(subset)

    # AC-06: Cycle B uplift over Cycle A
    if "cycle_B_brier" in metrics and "cycle_A_brier" in metrics:
        brier_uplift = (metrics["cycle_A_brier"] - metrics["cycle_B_brier"]) / metrics["cycle_A_brier"]
        mae_uplift   = (metrics["cycle_A_mae"]   - metrics["cycle_B_mae"])   / metrics["cycle_A_mae"]
        metrics["brier_uplift_pct"] = round(brier_uplift * 100, 2)
        metrics["mae_uplift_pct"]   = round(mae_uplift   * 100, 2)
        if brier_uplift < 0.05:
            logger.warning("AC-06 not met: Cycle B Brier uplift only %.1f%% (need ≥5%%)", brier_uplift * 100)
        if mae_uplift < 0.05:
            logger.warning("AC-06 not met: Cycle B MAE uplift only %.1f%% (need ≥5%%)", mae_uplift * 100)

    logger.info("Rolling %d-day metrics: %s", days, metrics)
    return metrics
