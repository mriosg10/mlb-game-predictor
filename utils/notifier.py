"""
Email notifications via SendGrid API (https://sendgrid.com — free tier: 100/day).

Required environment variables:
  MLB_SENDGRID_KEY   — SendGrid API key (starts with "SG.")
  MLB_NOTIFY_FROM    — verified sender address in your SendGrid account
  MLB_NOTIFY_TO      — recipient address (default: ryberin@hotmail.com)

If MLB_SENDGRID_KEY is not set, all send calls are no-ops.

Setup (one-time):
  1. Create a free account at https://sendgrid.com
  2. Go to Settings → API Keys → Create API Key (Mail Send permission)
  3. Go to Settings → Sender Authentication → Single Sender Verification
     and verify the address you'll send FROM.
  4. Export:
       export MLB_SENDGRID_KEY="SG.xxxx"
       export MLB_NOTIFY_FROM="your-verified@email.com"
"""

import json
import logging
import os
from datetime import date

import duckdb
import requests

from config import DB_PATH

logger = logging.getLogger(__name__)

_SENDGRID_URL = "https://api.sendgrid.com/v3/mail/send"
_DEFAULT_TO   = "ryberin@hotmail.com"


def _send(subject: str, body_html: str) -> None:
    api_key  = os.environ.get("MLB_SENDGRID_KEY", "").strip()
    from_addr = os.environ.get("MLB_NOTIFY_FROM", "").strip()
    to_addr  = os.environ.get("MLB_NOTIFY_TO", _DEFAULT_TO).strip()

    if not api_key:
        logger.debug("Notifier: MLB_SENDGRID_KEY not set — skipping email")
        return
    if not from_addr:
        logger.warning("Notifier: MLB_NOTIFY_FROM not set — skipping email")
        return

    payload = {
        "personalizations": [{"to": [{"email": to_addr}]}],
        "from": {"email": from_addr},
        "subject": subject,
        "content": [{"type": "text/html", "value": body_html}],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(_SENDGRID_URL, headers=headers,
                             data=json.dumps(payload), timeout=15)
        if resp.status_code == 202:
            logger.info("Notifier: email sent to %s — %s", to_addr, subject)
        else:
            logger.warning("Notifier: SendGrid returned %d: %s",
                           resp.status_code, resp.text[:200])
    except Exception as exc:
        logger.warning("Notifier: failed to send email: %s", exc)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _status_badge(status: str) -> str:
    color = {"SUCCESS": "#2e7d32", "PARTIAL": "#e65100", "FAILED": "#c62828"}.get(status, "#555")
    return (f'<span style="background:{color};color:#fff;padding:2px 8px;'
            f'border-radius:3px;font-size:12px">{status}</span>')


def _html_wrap(title: str, body: str) -> str:
    return f"""
<html><body style="font-family:Arial,sans-serif;font-size:14px;color:#222;max-width:700px">
<h2 style="color:#1a237e">&#9918; MLB Predictor &#8212; {title}</h2>
{body}
<hr style="margin-top:24px">
<p style="font-size:11px;color:#888">MLB Game Prediction System · automated notification</p>
</body></html>"""


def _predictions_table(date_str: str, cycle: str) -> str:
    try:
        conn = duckdb.connect(DB_PATH, read_only=True)
        rows = conn.execute("""
            SELECT
                f.away_team || ' @ ' || f.home_team          AS matchup,
                CASE WHEN p.home_win_prob >= 0.5
                     THEN f.home_team ELSE f.away_team END    AS pick,
                ROUND(GREATEST(p.home_win_prob,
                               p.away_win_prob) * 100, 1)    AS confidence,
                ROUND(p.predicted_total, 1)                   AS total
            FROM predictions p
            JOIN features f ON p.game_id = f.game_id AND p.cycle = f.cycle
            WHERE f.game_date = ? AND p.cycle = ?
            ORDER BY p.predicted_total DESC
        """, [date_str, cycle]).fetchall()
        conn.close()
    except Exception as exc:
        logger.warning("Notifier: DB query failed: %s", exc)
        rows = []

    if not rows:
        return "<p>No predictions available.</p>"

    rows_html = "".join(
        f"<tr style='background:{'#f5f5f5' if i % 2 else '#fff'}'>"
        f"<td style='padding:6px 10px'>{r[0]}</td>"
        f"<td style='padding:6px 10px;font-weight:bold'>{r[1]}</td>"
        f"<td style='padding:6px 10px'>{r[2]}%</td>"
        f"<td style='padding:6px 10px'>{r[3]}</td>"
        f"</tr>"
        for i, r in enumerate(rows)
    )
    return f"""
<table border="0" cellspacing="0" cellpadding="0"
       style="border-collapse:collapse;width:100%;border:1px solid #ddd">
  <thead style="background:#1a237e;color:#fff">
    <tr>
      <th style="padding:8px 10px;text-align:left">Matchup</th>
      <th style="padding:8px 10px;text-align:left">Pick</th>
      <th style="padding:8px 10px;text-align:left">Confidence</th>
      <th style="padding:8px 10px;text-align:left">Pred Total</th>
    </tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table>"""


# ---------------------------------------------------------------------------
# Per-cycle email senders
# ---------------------------------------------------------------------------

def notify_cycle_a(game_date: date, status: dict) -> None:
    date_str   = game_date.strftime("%Y-%m-%d")
    run_status = status.get("run_status", "UNKNOWN")
    n_games    = status.get("games_evaluated", 0)
    failure    = status.get("failure_reason")

    failure_html = (f'<p style="color:#c62828"><b>Note:</b> {failure}</p>'
                    if failure else "")
    body = f"""
<p>{_status_badge(run_status)} &nbsp; <b>{n_games} games</b> · {date_str}</p>
{failure_html}
<h3 style="margin-bottom:6px">Today's Seed Predictions</h3>
{_predictions_table(date_str, 'A')}
"""
    _send(f"MLB Cycle A — {date_str} ({n_games} games, {run_status})",
          _html_wrap(f"Cycle A · {date_str}", body))


def notify_cycle_b(game_date: date, status: dict) -> None:
    date_str   = game_date.strftime("%Y-%m-%d")
    run_status = status.get("run_status", "UNKNOWN")
    n_games    = status.get("games_evaluated", 0)
    failure    = status.get("failure_reason")

    failure_html = (f'<p style="color:#c62828"><b>Note:</b> {failure}</p>'
                    if failure else "")
    body = f"""
<p>{_status_badge(run_status)} &nbsp; <b>{n_games} games</b> · {date_str}</p>
{failure_html}
<h3 style="margin-bottom:6px">Lock Predictions (confirmed lineups + weather)</h3>
{_predictions_table(date_str, 'B')}
"""
    _send(f"MLB Cycle B — {date_str} ({n_games} games, {run_status})",
          _html_wrap(f"Cycle B · {date_str}", body))


def notify_post_game(game_date: date, status: dict) -> None:
    date_str   = game_date.strftime("%Y-%m-%d")
    run_status = status.get("run_status", "UNKNOWN")
    n_games    = status.get("games_evaluated", 0)
    failure    = status.get("failure_reason")

    # Results vs predictions
    rows = []
    metrics: dict = {}
    try:
        conn = duckdb.connect(DB_PATH, read_only=True)
        rows = conn.execute("""
            SELECT
                f.away_team || ' @ ' || f.home_team          AS matchup,
                CASE WHEN p.home_win_prob >= 0.5
                     THEN f.home_team ELSE f.away_team END    AS pick,
                r.away_score || '-' || r.home_score           AS score,
                r.winner                                      AS actual,
                CASE WHEN (p.home_win_prob >= 0.5
                           AND r.winner = f.home_team)
                       OR (p.home_win_prob < 0.5
                           AND r.winner = f.away_team)
                     THEN '&#10003;' ELSE '&#10007;' END      AS correct,
                ROUND(p.predicted_total, 1)                   AS pred_total,
                r.total_runs                                  AS actual_total
            FROM predictions p
            JOIN features f ON p.game_id = f.game_id AND p.cycle = f.cycle
            JOIN results  r ON p.game_id = r.game_id
            WHERE f.game_date = ? AND p.cycle = 'A'
            ORDER BY f.game_date
        """, [date_str]).fetchall()

        row = conn.execute("""
            SELECT brier_score, win_accuracy, total_mae, games_evaluated
            FROM evaluation_log
            WHERE log_date = ? AND cycle = 'A'
            ORDER BY created_at DESC LIMIT 1
        """, [date_str]).fetchone()
        if row:
            metrics = {"brier": row[0], "win_acc": row[1],
                       "mae": row[2], "n": row[3]}
        conn.close()
    except Exception as exc:
        logger.warning("Notifier: DB query failed: %s", exc)

    correct = sum(1 for r in rows if "10003" in r[4])
    total_g = len(rows)

    rows_html = "".join(
        f"<tr style='background:{'#f5f5f5' if i % 2 else '#fff'}'>"
        f"<td style='padding:6px 10px'>{r[0]}</td>"
        f"<td style='padding:6px 10px'>{r[1]}</td>"
        f"<td style='padding:6px 10px'>{r[2]}</td>"
        f"<td style='padding:6px 10px;font-weight:bold'>{r[3]}</td>"
        f"<td style='padding:6px 10px;font-size:16px'>{r[4]}</td>"
        f"<td style='padding:6px 10px'>{r[5]} &rarr; {r[6]}</td>"
        f"</tr>"
        for i, r in enumerate(rows)
    )

    results_table = f"""
<table border="0" cellspacing="0" cellpadding="0"
       style="border-collapse:collapse;width:100%;border:1px solid #ddd">
  <thead style="background:#1a237e;color:#fff">
    <tr>
      <th style="padding:8px 10px;text-align:left">Matchup</th>
      <th style="padding:8px 10px;text-align:left">Pick</th>
      <th style="padding:8px 10px;text-align:left">Score</th>
      <th style="padding:8px 10px;text-align:left">Winner</th>
      <th style="padding:8px 10px;text-align:left">Result</th>
      <th style="padding:8px 10px;text-align:left">Total (pred&rarr;actual)</th>
    </tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table>""" if rows else "<p>No matched results found.</p>"

    metrics_html = ""
    if metrics:
        win_pct = round((metrics["win_acc"] or 0) * 100, 1)
        metrics_html = f"""
<table style="margin-bottom:16px;border-collapse:collapse">
  <tr><td style="padding:6px 16px 6px 0"><b>Win accuracy</b></td>
      <td>{correct}/{total_g} ({win_pct}%)</td></tr>
  <tr><td style="padding:6px 16px 6px 0"><b>Run total MAE</b></td>
      <td>{round(metrics['mae'] or 0, 2)} runs</td></tr>
  <tr><td style="padding:6px 16px 6px 0"><b>Brier score</b></td>
      <td>{round(metrics['brier'] or 0, 4)}</td></tr>
</table>"""

    failure_html = (f'<p style="color:#c62828"><b>Note:</b> {failure}</p>'
                    if failure else "")
    body = f"""
<p>{_status_badge(run_status)} &nbsp; <b>{n_games} results written</b> · {date_str}</p>
{failure_html}
<h3 style="margin-bottom:6px">Model Performance</h3>
{metrics_html}
<h3 style="margin-bottom:6px">Results vs Predictions</h3>
{results_table}
"""
    _send(f"MLB Post-game — {date_str} ({correct}/{total_g} correct)",
          _html_wrap(f"Post-game Results · {date_str}", body))
