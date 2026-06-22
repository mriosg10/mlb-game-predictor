"""
Cycle A — Seed pipeline (cron: 0 8 * * *, ET).

Steps (sequential; a failure halts the cycle):
  1. fetch_schedule
  2. fetch_probable_pitchers  (already hydrated in schedule)
  3. assemble_features        (Statcast + FanGraphs + park factors; no weather yet)
  4. run_inference
  5. write_predictions
  6. write_eval_log

AC-09: full cycle must complete in ≤ 8 minutes on single-core hardware.
AC-01: ≥ 95% of that day's games must be covered.
FR-11: eval_log record written on every run (success or failure).
"""

import logging
import time
from datetime import date, datetime, timezone
from typing import Any

import database as db
from features.assembler import assemble_game_features
from fetchers.mlb_stats import get_schedule
from fetchers.odds import fetch_ou_lines
from model.inference import ModelNotFoundError, load_models, predict_batch
from utils.notifier import notify_cycle_a

logger = logging.getLogger(__name__)


def run(game_date: date | None = None) -> dict[str, Any]:
    """
    Execute Cycle A. Returns a status dict.
    Guaranteed to write to evaluation_log regardless of outcome (FR-11).
    """
    if game_date is None:
        game_date = date.today()

    cycle = "A"
    date_str = game_date.strftime("%Y-%m-%d")
    start_ts = time.monotonic()
    logger.info("=== Cycle A starting for %s ===", date_str)

    status = {
        "log_date":        game_date,
        "cycle":           cycle,
        "run_status":      "FAILED",
        "games_evaluated": 0,
        "brier_score":     None,
        "win_accuracy":    None,
        "total_mae":       None,
        "failure_reason":  None,
    }

    try:
        # ------------------------------------------------------------------
        # Step 1: Load model artifacts (fail-fast before network calls)
        # ------------------------------------------------------------------
        try:
            win_model, total_model = load_models()
        except ModelNotFoundError as exc:
            status["failure_reason"] = str(exc)
            logger.error("Cycle A aborted — model not found: %s", exc)
            db.upsert_eval_log(status)
            return status

        # ------------------------------------------------------------------
        # Step 2: Fetch schedule + O/U lines
        # ------------------------------------------------------------------
        games = get_schedule(game_date)
        if not games:
            status["failure_reason"] = f"No games scheduled on {date_str}"
            status["run_status"] = "PARTIAL"
            logger.warning("Cycle A: no games found for %s", date_str)
            db.upsert_eval_log(status)
            return status

        logger.info("Cycle A: %d games to process", len(games))

        # Fetch O/U lines; returns {} if API key not set or call fails
        raw_ou = fetch_ou_lines(game_date)
        game_ou_lines: dict[str, float | None] = {
            str(g["game_id"]): raw_ou.get(f"{g['away_team']}@{g['home_team']}")
            for g in games
        }
        covered = sum(1 for v in game_ou_lines.values() if v is not None)
        logger.info("Cycle A: O/U lines fetched for %d/%d games", covered, len(games))

        # ------------------------------------------------------------------
        # Step 3: Assemble features + run inference
        # ------------------------------------------------------------------
        feature_rows: list[dict] = []
        excluded_games: list[str] = []
        excluded_reasons: list[str] = []

        for game in games:
            try:
                row, missing = assemble_game_features(game, cycle=cycle)
            except Exception as exc:
                logger.error("Feature assembly failed for game %s: %s", game["game_id"], exc)
                excluded_games.append(game["game_id"])
                excluded_reasons.append(f"{game['game_id']}: assembly error: {exc}")
                continue

            if row is None:
                # Excluded due to >30% missing features (AC-10)
                excluded_games.append(game["game_id"])
                excluded_reasons.append(
                    f"{game['game_id']}: >30% missing: {', '.join(missing)}"
                )
                continue

            try:
                db.upsert_features(row)
                feature_rows.append(row)
            except Exception as exc:
                logger.error("DB write failed for game %s features: %s", game["game_id"], exc)
                excluded_games.append(game["game_id"])
                excluded_reasons.append(f"{game['game_id']}: DB write error: {exc}")

        if not feature_rows:
            status["failure_reason"] = (
                f"All {len(games)} games excluded or errored. "
                + "; ".join(excluded_reasons[:5])
            )
            logger.error("Cycle A: no games reached inference stage")
            db.upsert_eval_log(status)
            return status

        # ------------------------------------------------------------------
        # Step 4: Batch inference
        # ------------------------------------------------------------------
        predictions = predict_batch(feature_rows, win_model, total_model, ou_lines=game_ou_lines)

        # ------------------------------------------------------------------
        # Step 5: Write predictions
        # ------------------------------------------------------------------
        written = 0
        for pred in predictions:
            try:
                db.insert_prediction(
                    game_id=pred["game_id"],
                    cycle=cycle,
                    home_win_prob=pred["home_win_prob"],
                    predicted_total=pred["predicted_total"],
                    model_version=pred["model_version"],
                    ou_prob=pred.get("ou_prob"),
                    ou_line=pred.get("ou_line"),
                )
                written += 1
                logger.debug(
                    "  %s: home_win=%.3f  total=%.1f",
                    pred["game_id"], pred["home_win_prob"], pred["predicted_total"],
                )
            except Exception as exc:
                logger.error("DB insert failed for prediction %s: %s", pred["game_id"], exc)

        elapsed = time.monotonic() - start_ts
        coverage = written / len(games)

        # AC-01: ≥ 95% coverage required for SUCCESS
        if coverage >= 0.95:
            run_status = "SUCCESS"
        elif written > 0:
            run_status = "PARTIAL"
        else:
            run_status = "FAILED"

        failure_reason = None
        if excluded_games:
            failure_reason = (
                f"Excluded games ({len(excluded_games)}): "
                + "; ".join(excluded_reasons[:10])
            )

        logger.info(
            "Cycle A done: %d/%d games, coverage=%.1f%%, elapsed=%.1fs  [%s]",
            written, len(games), coverage * 100, elapsed, run_status,
        )
        if elapsed > 480:  # 8 minutes = 480s (AC-09)
            logger.warning("Cycle A exceeded 8-minute runtime target: %.1fs", elapsed)

        status.update({
            "run_status":      run_status,
            "games_evaluated": written,
            "failure_reason":  failure_reason,
        })

    except Exception as exc:
        logger.exception("Cycle A unexpected failure: %s", exc)
        status["failure_reason"] = f"Unexpected error: {exc}"

    finally:
        # FR-11: always write log record
        db.upsert_eval_log(status)
        notify_cycle_a(game_date, status)

    return status
