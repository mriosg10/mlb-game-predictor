"""
Post-game pipeline (cron: 30 23 * * *, ET).

Steps:
  1. Fetch final scores from MLB Stats API
  2. Write results to DB
  3. Compute daily metrics (Brier, win accuracy, MAE)
  4. Write evaluation_log for both Cycle A and Cycle B

FR-06: write final scores within 30 minutes of last game ending.
FR-07: compute and persist daily metrics.
AC-03: final scores written for 100% of completed games.
FR-11: eval_log always written.

On partial results (some games still in progress), marks run_status=PARTIAL
and queues a retry (max 6 over 2 hours — handled by cron retry logic in
scripts/cron_setup.sh).
"""

import logging
import time
from datetime import date, datetime, timezone
from typing import Any

import database as db
from fetchers.mlb_stats import get_final_scores, get_postponed_game_ids
from pipeline.evaluation import compute_daily_metrics
from utils.notifier import notify_post_game

logger = logging.getLogger(__name__)


def run(game_date: date | None = None) -> dict[str, Any]:
    """
    Execute the post-game actuals pipeline.
    Guaranteed to write to evaluation_log (FR-11).
    """
    if game_date is None:
        game_date = date.today()

    date_str = game_date.strftime("%Y-%m-%d")
    start_ts = time.monotonic()
    logger.info("=== Post-game starting for %s ===", date_str)

    # Check for a prior successful run BEFORE any DB writes so the finally
    # block doesn't detect the current run's own upsert as a "prior success".
    prior_success = False
    try:
        with db.get_conn() as conn:
            row = conn.execute(
                "SELECT run_status FROM evaluation_log "
                "WHERE log_date = ? AND cycle = 'post' "
                "ORDER BY created_at DESC LIMIT 1",
                [game_date],
            ).fetchone()
            prior_success = row is not None and row[0] == "SUCCESS"
    except Exception:
        pass

    status: dict[str, Any] = {
        "log_date":        game_date,
        "cycle":           "post",
        "run_status":      "FAILED",
        "games_evaluated": 0,
        "brier_score":     None,
        "win_accuracy":    None,
        "total_mae":       None,
        "failure_reason":  None,
    }

    try:
        # ------------------------------------------------------------------
        # Step 1 & 2: Fetch final scores and write to results table
        # ------------------------------------------------------------------
        scores = get_final_scores(game_date)

        if not scores:
            status["failure_reason"] = "No final scores returned from MLB API"
            status["run_status"] = "PARTIAL"
            logger.warning("Post-game: no final scores for %s", date_str)
            db.upsert_eval_log(status)
            return status

        written = 0
        errors = []
        for score in scores:
            try:
                db.upsert_result(score)
                written += 1
            except Exception as exc:
                logger.error("DB write failed for result %s: %s", score["game_id"], exc)
                errors.append(f"{score['game_id']}: {exc}")

        logger.info("Post-game: wrote %d/%d results", written, len(scores))

        # Check that every game_id in predictions also has a result (AC-03)
        preds_b = db.get_predictions_for_date(date_str, "B")
        pred_game_ids   = {p["game_id"] for p in preds_b}
        result_game_ids = {s["game_id"] for s in scores}
        postponed_ids   = get_postponed_game_ids(game_date)

        # Exclude postponed/cancelled games — they will never produce a result
        missing_results = pred_game_ids - result_game_ids - postponed_ids

        if postponed_ids & pred_game_ids:
            logger.info(
                "Post-game: %d predicted game(s) postponed/cancelled (excluded from AC-03): %s",
                len(postponed_ids & pred_game_ids),
                ", ".join(sorted(postponed_ids & pred_game_ids)),
            )

        if missing_results:
            logger.warning(
                "AC-03: %d predicted games have no result yet: %s",
                len(missing_results), ", ".join(sorted(missing_results)),
            )
            status["failure_reason"] = (
                f"Missing results for {len(missing_results)} games: "
                + ", ".join(sorted(missing_results))
            )
            status["run_status"] = "PARTIAL"
            status["games_evaluated"] = written
            db.upsert_eval_log(status)
            return status

        # ------------------------------------------------------------------
        # Step 3: Compute and write daily metrics for Cycle B (primary)
        # ------------------------------------------------------------------
        try:
            metrics_b = compute_daily_metrics(game_date, "B")
            db.upsert_eval_log(metrics_b)
        except Exception as exc:
            logger.error("Metrics computation failed for Cycle B: %s", exc)
            errors.append(f"metrics_B: {exc}")

        # Also compute Cycle A metrics for AC-06 (uplift comparison)
        try:
            metrics_a = compute_daily_metrics(game_date, "A")
            db.upsert_eval_log(metrics_a)
        except Exception as exc:
            logger.warning("Metrics computation failed for Cycle A: %s", exc)

        elapsed = time.monotonic() - start_ts
        run_status = "SUCCESS" if not errors else "PARTIAL"

        logger.info(
            "Post-game done: %d results written, elapsed=%.1fs  [%s]",
            written, elapsed, run_status,
        )

        status.update({
            "run_status":      run_status,
            "games_evaluated": written,
            "failure_reason":  "; ".join(errors) if errors else None,
        })
        # Update post-game log record
        db.upsert_eval_log(status)

    except Exception as exc:
        logger.exception("Post-game unexpected failure: %s", exc)
        status["failure_reason"] = f"Unexpected error: {exc}"
        db.upsert_eval_log(status)

    finally:
        db.upsert_eval_log(status)

        if not prior_success:
            notify_post_game(game_date, status)
        else:
            logger.info(
                "Post-game retry: prior run already succeeded — skipping duplicate email"
            )

    return status
