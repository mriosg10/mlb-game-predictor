"""
Cycle B — Lock pipeline (cron: 30 13 * * *, ET).

Runs ~2.5 hours before the earliest first pitch (NFR-02).

Steps (sequential):
  1. fetch_schedule          (re-fetch to get current game status)
  2. fetch_confirmed_lineups (RotoWire scrape)
  3. reconcile_il_scratches  (FR-08: cross-reference IL log)
  4. assemble_features       (with confirmed lineups + weather)
  5. run_inference
  6. write_predictions
  7. write_eval_log

Fallback (CONSTRAINT): if RotoWire scrape fails, fall back to MLB Stats API
probable pitchers and mark prediction as Cycle-A-quality.

AC-02: confirmed lineups for ≥ 90% of scheduled games.
FR-11: eval_log record always written.
"""

import logging
import time
from datetime import date, datetime, timezone
from typing import Any

import database as db
from features.assembler import assemble_game_features
from fetchers.mlb_stats import get_player, get_schedule, get_il_transactions
from fetchers.rotowire import ConfirmedLineup, GameLineups, get_confirmed_lineups
from model.inference import ModelNotFoundError, load_models, predict_batch
from utils.notifier import notify_cycle_b

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IL scratch reconciliation (FR-08)
# ---------------------------------------------------------------------------

def _reconcile_il_scratches(
    batting_order: list[dict],
    team_id: int,
    game_date: date,
) -> list[dict]:
    """
    Remove any player in the batting order who appears in that day's IL
    transactions (same-day scratch not yet reflected in RotoWire).

    Returns the filtered batting order. Players removed are logged.
    """
    try:
        txns = get_il_transactions(team_id, game_date)
        # Only today's placements
        today_str = game_date.strftime("%Y-%m-%d")
        scratched_names = set()
        for t in txns:
            if t.get("date", "") == today_str:
                type_code = t.get("type", "")
                if type_code in ("IL10", "IL15", "IL60"):
                    name = t.get("player_name", "")
                    if name:
                        scratched_names.add(name.lower())
    except Exception as exc:
        logger.warning("IL scratch reconciliation failed for team %d: %s", team_id, exc)
        return batting_order

    if not scratched_names:
        return batting_order

    cleaned = [
        b for b in batting_order
        if b.get("name", "").lower() not in scratched_names
    ]
    removed = len(batting_order) - len(cleaned)
    if removed:
        logger.info(
            "IL reconcile: removed %d same-day scratch(es) from team %d lineup",
            removed, team_id,
        )
    return cleaned


# ---------------------------------------------------------------------------
# Lineup matching: map RotoWire abbreviation to schedule game
# ---------------------------------------------------------------------------

def _match_lineup_to_game(
    game: dict,
    lineups: dict[str, GameLineups],
) -> GameLineups | None:
    """
    Find the GameLineups for a given game by matching team abbreviations.
    RotoWire abbrs sometimes differ from MLB API abbrs; try both orders.
    """
    home = game["home_team"]
    away = game["away_team"]

    # Primary key
    key = f"{away}@{home}"
    if key in lineups:
        return lineups[key]

    # Case-insensitive fallback
    for k, gl in lineups.items():
        a, h = k.split("@") if "@" in k else (k, "")
        if a.upper() == away.upper() and h.upper() == home.upper():
            return gl

    return None


# ---------------------------------------------------------------------------
# Main Cycle B runner
# ---------------------------------------------------------------------------

def run(game_date: date | None = None) -> dict[str, Any]:
    """
    Execute Cycle B. Guaranteed to write to evaluation_log (FR-11).
    """
    if game_date is None:
        game_date = date.today()

    cycle = "B"
    date_str = game_date.strftime("%Y-%m-%d")
    start_ts = time.monotonic()
    logger.info("=== Cycle B starting for %s ===", date_str)

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
        # Step 1: Load models
        # ------------------------------------------------------------------
        try:
            win_model, total_model = load_models()
        except ModelNotFoundError as exc:
            status["failure_reason"] = str(exc)
            logger.error("Cycle B aborted — model not found: %s", exc)
            db.upsert_eval_log(status)
            return status

        # ------------------------------------------------------------------
        # Step 2: Fetch schedule
        # ------------------------------------------------------------------
        games = get_schedule(game_date)
        if not games:
            status["failure_reason"] = f"No games scheduled on {date_str}"
            status["run_status"] = "PARTIAL"
            db.upsert_eval_log(status)
            return status

        logger.info("Cycle B: %d games to process", len(games))

        # ------------------------------------------------------------------
        # Step 3: Fetch confirmed lineups from RotoWire
        # ------------------------------------------------------------------
        rotowire_lineups = get_confirmed_lineups(game_date)
        rotowire_available = len(rotowire_lineups) > 0
        confirmed_lineup_count = 0

        if not rotowire_available:
            logger.warning(
                "RotoWire scrape returned 0 lineups — "
                "falling back to MLB API probable pitchers (Cycle-A-quality predictions)"
            )

        # ------------------------------------------------------------------
        # Step 4: Per-game assembly
        # ------------------------------------------------------------------
        feature_rows: list[dict] = []
        excluded_games: list[str] = []
        excluded_reasons: list[str] = []

        for game in games:
            game_id = game["game_id"]

            # Resolve pitcher overrides from RotoWire
            home_pitcher_override = None
            away_pitcher_override = None
            home_lineup_batters: list[dict] = []
            away_lineup_batters: list[dict] = []
            used_confirmed_lineup = False

            gl = _match_lineup_to_game(game, rotowire_lineups)
            if gl is not None:
                confirmed_lineup_count += 1
                used_confirmed_lineup = True

                # Home pitcher
                if gl.home.pitcher_name and gl.home.pitcher_name != "TBD":
                    home_pitcher_override = {
                        "id":   None,   # RotoWire doesn't provide MLBAM IDs
                        "name": gl.home.pitcher_name,
                        "hand": gl.home.pitcher_hand,
                    }
                # Away pitcher
                if gl.away.pitcher_name and gl.away.pitcher_name != "TBD":
                    away_pitcher_override = {
                        "id":   None,
                        "name": gl.away.pitcher_name,
                        "hand": gl.away.pitcher_hand,
                    }

                home_lineup_batters = gl.home.batting_order
                away_lineup_batters = gl.away.batting_order

                # IL scratch reconciliation (FR-08)
                home_lineup_batters = _reconcile_il_scratches(
                    home_lineup_batters, game["home_team_id"], game_date
                )
                away_lineup_batters = _reconcile_il_scratches(
                    away_lineup_batters, game["away_team_id"], game_date
                )

            try:
                row, missing = assemble_game_features(
                    game,
                    cycle=cycle,
                    home_lineup=home_lineup_batters,
                    away_lineup=away_lineup_batters,
                    home_pitcher_override=home_pitcher_override,
                    away_pitcher_override=away_pitcher_override,
                )
            except Exception as exc:
                logger.error("Feature assembly failed for game %s: %s", game_id, exc)
                excluded_games.append(game_id)
                excluded_reasons.append(f"{game_id}: assembly error: {exc}")
                continue

            if row is None:
                excluded_games.append(game_id)
                excluded_reasons.append(
                    f"{game_id}: >30% missing: {', '.join(missing)}"
                )
                continue

            # Tag whether this prediction used confirmed lineup data
            row["_used_confirmed_lineup"] = used_confirmed_lineup

            try:
                db.upsert_features(row)
                feature_rows.append(row)
            except Exception as exc:
                logger.error("DB write failed for game %s features: %s", game_id, exc)
                excluded_games.append(game_id)
                excluded_reasons.append(f"{game_id}: DB write error: {exc}")

        # AC-02: ≥ 90% confirmed lineup coverage
        if games:
            confirmed_coverage = confirmed_lineup_count / len(games)
            if confirmed_coverage < 0.90:
                logger.warning(
                    "AC-02 not met: confirmed lineup coverage %.1f%% < 90%% (%d/%d)",
                    confirmed_coverage * 100, confirmed_lineup_count, len(games),
                )

        if not feature_rows:
            status["failure_reason"] = "All games excluded or errored"
            db.upsert_eval_log(status)
            return status

        # ------------------------------------------------------------------
        # Step 5: Batch inference + write predictions
        # ------------------------------------------------------------------
        predictions = predict_batch(feature_rows, win_model, total_model)

        written = 0
        for pred in predictions:
            try:
                db.insert_prediction(
                    game_id=pred["game_id"],
                    cycle=cycle,
                    home_win_prob=pred["home_win_prob"],
                    predicted_total=pred["predicted_total"],
                    model_version=pred["model_version"],
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

        run_status = "SUCCESS" if coverage >= 0.90 else ("PARTIAL" if written > 0 else "FAILED")

        failure_reason = None
        if excluded_games:
            failure_reason = (
                f"Excluded ({len(excluded_games)}): " + "; ".join(excluded_reasons[:10])
            )
        if not rotowire_available:
            note = "RotoWire unavailable; used probable pitchers (Cycle-A-quality)"
            failure_reason = (failure_reason + " | " + note) if failure_reason else note
            run_status = "PARTIAL" if run_status == "SUCCESS" else run_status

        logger.info(
            "Cycle B done: %d/%d games, confirmed_lineups=%d/%d, elapsed=%.1fs  [%s]",
            written, len(games),
            confirmed_lineup_count, len(games),
            elapsed, run_status,
        )

        status.update({
            "run_status":      run_status,
            "games_evaluated": written,
            "failure_reason":  failure_reason,
        })

    except Exception as exc:
        logger.exception("Cycle B unexpected failure: %s", exc)
        status["failure_reason"] = f"Unexpected error: {exc}"

    finally:
        db.upsert_eval_log(status)
        notify_cycle_b(game_date, status)

    return status
