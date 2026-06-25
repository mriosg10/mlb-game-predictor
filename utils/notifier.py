"""
Email notifications via SendGrid API (https://sendgrid.com — free tier: 100/day).

Required environment variables:
  MLB_SENDGRID_KEY   — SendGrid API key (starts with "SG.")
  MLB_NOTIFY_FROM    — verified sender address in your SendGrid account
  MLB_NOTIFY_TO      — recipient address (default: ryberin@hotmail.com)

If MLB_SENDGRID_KEY is not set, all send calls are no-ops.
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
    api_key   = os.environ.get("MLB_SENDGRID_KEY", "").strip()
    from_addr = os.environ.get("MLB_NOTIFY_FROM", "").strip()
    to_addr   = os.environ.get("MLB_NOTIFY_TO", _DEFAULT_TO).strip()

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
<html><body style="font-family:Arial,sans-serif;font-size:14px;color:#222;max-width:720px;margin:0 auto">
<h2 style="color:#1a237e;margin-bottom:4px">&#9918; MLB Predictor &#8212; {title}</h2>
{body}
<hr style="margin-top:24px">
<p style="font-size:11px;color:#888">MLB Game Prediction System &middot; automated notification</p>
</body></html>"""


def _td(val, bold=False, align="left", color=None, bg=None, pad="5px 10px"):
    style = f"padding:{pad};text-align:{align};"
    if bold:
        style += "font-weight:bold;"
    if color:
        style += f"color:{color};"
    if bg:
        style += f"background:{bg};"
    inner = f"<b>{val}</b>" if bold else str(val)
    return f'<td style="{style}">{inner}</td>'


def _label_row(text: str) -> str:
    return (f'<tr><td colspan="10" style="padding:3px 10px 2px;'
            f'font-size:11px;font-weight:bold;color:#555;'
            f'text-transform:uppercase;letter-spacing:0.5px;'
            f'background:#eeeeee">{text}</td></tr>')


# ---------------------------------------------------------------------------
# Summary predictions table (quick-scan header)
# ---------------------------------------------------------------------------

def _ou_display(
    ou_prob: float | None,
    ou_line: float | None,
    predicted_total: float,
) -> tuple:
    """
    Return (direction, line_str, conf_pct) for over/under display.
    Uses real sportsbook line and calibrated probability when available;
    falls back to synthetic formula (fixed 9.0 line) otherwise.
    """
    if ou_line is not None and ou_prob is not None:
        direction = "Over" if ou_prob >= 0.5 else "Under"
        conf = round((ou_prob if ou_prob >= 0.5 else 1.0 - ou_prob) * 100, 1)
        return direction, f"{ou_line:.1f}", conf
    # Synthetic fallback when odds API is unavailable
    diff = predicted_total - 9.0
    conf = min(50 + abs(diff) / 3.0 * 50, 99)
    return ("Over" if diff >= 0 else "Under"), "9.0*", round(conf, 1)


def _predictions_table(date_str: str, cycle: str) -> str:
    try:
        conn = duckdb.connect(DB_PATH, read_only=True)
        rows = conn.execute("""
            SELECT
                f.away_team || ' @ ' || f.home_team          AS matchup,
                CASE WHEN p.home_win_prob >= 0.5
                     THEN f.home_team ELSE f.away_team END    AS pick,
                ROUND(GREATEST(p.home_win_prob,
                               p.away_win_prob) * 100, 1)    AS win_conf,
                ROUND(p.predicted_total, 1)                   AS total,
                p.home_win_prob,
                p.ou_prob,
                p.ou_line
            FROM predictions p
            JOIN features f ON p.game_id = f.game_id AND p.cycle = f.cycle
            WHERE f.game_date = ? AND p.cycle = ?
            ORDER BY GREATEST(p.home_win_prob, p.away_win_prob) DESC
        """, [date_str, cycle]).fetchall()
        conn.close()
    except Exception as exc:
        logger.warning("Notifier: DB query failed: %s", exc)
        rows = []

    if not rows:
        return "<p>No predictions available.</p>"

    has_real_lines = any(r[6] is not None for r in rows)
    rows_html = ""
    for i, r in enumerate(rows):
        matchup, pick, win_conf, total, home_prob, ou_prob, ou_line = r
        ou_dir, line_str, ou_conf = _ou_display(ou_prob, ou_line, total)
        bg = "#f5f5f5" if i % 2 else "#fff"
        win_color = "#2e7d32" if win_conf >= 65 else ("#e65100" if win_conf >= 55 else "#555")
        ou_color  = "#1565c0" if ou_dir == "Over" else "#6a1b9a"
        rows_html += (
            f"<tr style='background:{bg}'>"
            f"<td style='padding:6px 10px'>{matchup}</td>"
            f"<td style='padding:6px 10px;font-weight:bold'>{pick}</td>"
            f"<td style='padding:6px 10px;color:{win_color};font-weight:bold'>{win_conf}%</td>"
            f"<td style='padding:6px 10px;color:{ou_color};font-weight:bold'>"
            f"{ou_dir} &nbsp;<span style='font-weight:normal;font-size:12px'>pred {total} / line {line_str}</span></td>"
            f"<td style='padding:6px 10px;color:{ou_color}'>{ou_conf}%</td>"
            f"</tr>"
        )

    footnote = (
        '<p style="font-size:11px;color:#888;margin-top:4px">O/U line from The Odds API (consensus). '
        'O/U % = calibrated model probability.</p>'
        if has_real_lines else
        '<p style="font-size:11px;color:#888;margin-top:4px">* No odds line available — O/U uses '
        'synthetic formula (9.0 fixed reference). Set MLB_ODDS_API_KEY to enable real lines.</p>'
    )

    return f"""
<table border="0" cellspacing="0" cellpadding="0"
       style="border-collapse:collapse;width:100%;border:1px solid #ddd">
  <thead style="background:#1a237e;color:#fff">
    <tr>
      <th style="padding:8px 10px;text-align:left">Matchup</th>
      <th style="padding:8px 10px;text-align:left">Result Pick</th>
      <th style="padding:8px 10px;text-align:left">Win %</th>
      <th style="padding:8px 10px;text-align:left">O/U (pred / line)</th>
      <th style="padding:8px 10px;text-align:left">O/U %</th>
    </tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table>
{footnote}"""


# ---------------------------------------------------------------------------
# Per-game brief cards
# ---------------------------------------------------------------------------

def _trend_badge(era_l3: float, xera: float) -> tuple:
    """Return (label, bg_color, text_color) trend indicator."""
    if era_l3 is None or xera is None or xera == 0:
        return "— Steady", "#f5f5f5", "#555"
    if era_l3 < xera * 0.85:
        return "&#8593; Hot", "#e8f5e9", "#2e7d32"
    if era_l3 > xera * 1.20:
        return "&#8595; Cooling", "#ffebee", "#c62828"
    return "&#8594; Steady", "#fff9c4", "#795548"


def _pitcher_row(prefix: str, team: str, name: str, hand: str,
                 xera, fip, k_pct, bb_pct, era_l3, whip_l3, bg: str) -> str:
    xera_s   = f"{xera:.2f}"  if xera  else "—"
    fip_s    = f"{fip:.2f}"   if fip   else "—"
    k_s      = f"{k_pct*100:.1f}%" if k_pct  else "—"
    bb_s     = f"{bb_pct*100:.1f}%" if bb_pct else "—"
    era_l3_s = f"{era_l3:.2f}"  if era_l3  else "—"
    whip_l3_s= f"{whip_l3:.3f}" if whip_l3 else "—"

    trend_label, trend_bg, trend_fg = _trend_badge(era_l3, xera)

    return f"""
<tr style="background:{bg}">
  <td style="padding:5px 10px;white-space:nowrap"><b>{team}</b></td>
  <td style="padding:5px 10px;white-space:nowrap">{name} ({hand})</td>
  <td style="padding:5px 10px;color:#333;white-space:nowrap">xERA {xera_s} | FIP {fip_s} | K% {k_s} | BB% {bb_s}</td>
  <td style="padding:5px 10px;white-space:nowrap">L3: ERA {era_l3_s} | WHIP {whip_l3_s}</td>
  <td style="padding:4px 8px;white-space:nowrap">
    <span style="background:{trend_bg};color:{trend_fg};padding:2px 7px;border-radius:3px;font-size:12px">{trend_label}</span>
  </td>
</tr>"""


def _build_game_brief_cards(date_str: str) -> str:
    """Build detailed HTML brief cards for every game on date_str."""
    from datetime import date as date_type
    from fetchers.mlb_stats import (
        get_schedule, get_player, get_umpire_assignments,
        get_umpire_career_stats, get_team_recent_record,
    )

    game_date = date_type.fromisoformat(date_str)

    # --- Schedule (pitcher IDs + game PKs) ---
    try:
        games = get_schedule(game_date)
    except Exception as exc:
        logger.warning("Notifier brief: schedule fetch failed: %s", exc)
        return ""

    if not games:
        return ""

    schedule_idx = {g["game_id"]: g for g in games}

    # --- Feature + prediction rows from DB ---
    try:
        conn = duckdb.connect(DB_PATH, read_only=True)
        feat_rows = conn.execute("""
            SELECT
                f.game_id, f.home_team, f.away_team,
                COALESCE(f.home_sp_xera,  0), COALESCE(f.home_sp_fip, 0),
                COALESCE(f.home_sp_k_pct, 0), COALESCE(f.home_sp_bb_pct, 0),
                COALESCE(f.home_sp_era_l3, 0), COALESCE(f.home_sp_whip_l3, 0),
                COALESCE(f.away_sp_xera,  0), COALESCE(f.away_sp_fip, 0),
                COALESCE(f.away_sp_k_pct, 0), COALESCE(f.away_sp_bb_pct, 0),
                COALESCE(f.away_sp_era_l3, 0), COALESCE(f.away_sp_whip_l3, 0),
                COALESCE(f.home_ops_14d, 0), COALESCE(f.home_risp_14d, 0),
                COALESCE(f.home_run_diff, 0),
                COALESCE(f.away_ops_14d, 0), COALESCE(f.away_risp_14d, 0),
                COALESCE(f.away_run_diff, 0),
                p.home_win_prob, p.predicted_total,
                p.ou_prob, p.ou_line
            FROM features f
            JOIN predictions p ON f.game_id = p.game_id AND f.cycle = p.cycle
            WHERE f.game_date = ? AND f.cycle = 'A'
            ORDER BY GREATEST(p.home_win_prob, p.away_win_prob) DESC
        """, [date_str]).fetchall()
        conn.close()
    except Exception as exc:
        logger.warning("Notifier brief: DB query failed: %s", exc)
        return ""

    if not feat_rows:
        return ""

    # --- Umpire assignments (single API call for all games) ---
    try:
        umpires = get_umpire_assignments(game_date)
    except Exception:
        umpires = {}

    # --- Pitcher name cache (bulk fetch) ---
    pitcher_ids = set()
    for row in feat_rows:
        g = schedule_idx.get(str(row[0]), {})
        if g.get("home_probable_id"):
            pitcher_ids.add(int(g["home_probable_id"]))
        if g.get("away_probable_id"):
            pitcher_ids.add(int(g["away_probable_id"]))

    pitcher_info: dict = {}
    for pid in pitcher_ids:
        try:
            p = get_player(pid)
            pitcher_info[pid] = {"name": p["full_name"], "hand": p["pitch_hand"]}
        except Exception:
            pitcher_info[pid] = {"name": f"#{pid}", "hand": "?"}

    # --- Build cards ---
    cards = []
    for row in feat_rows:
        (game_id, home_team, away_team,
         h_xera, h_fip, h_k, h_bb, h_era_l3, h_whip_l3,
         a_xera, a_fip, a_k, a_bb, a_era_l3, a_whip_l3,
         h_ops, h_risp, h_rdiff,
         a_ops, a_risp, a_rdiff,
         home_win_prob, pred_total,
         ou_prob, ou_line) = row

        pick       = home_team if home_win_prob >= 0.5 else away_team
        conf       = max(home_win_prob, 1 - home_win_prob) * 100
        conf_color = "#2e7d32" if conf >= 65 else ("#e65100" if conf >= 55 else "#555")
        ou_dir, line_str, ou_conf = _ou_display(ou_prob, ou_line, pred_total)

        # Pitcher info
        sched = schedule_idx.get(str(game_id), {})
        h_sp_id = sched.get("home_probable_id")
        a_sp_id = sched.get("away_probable_id")
        h_sp    = pitcher_info.get(int(h_sp_id), {"name": "TBD", "hand": "?"}) if h_sp_id else {"name": "TBD", "hand": "?"}
        a_sp    = pitcher_info.get(int(a_sp_id), {"name": "TBD", "hand": "?"}) if a_sp_id else {"name": "TBD", "hand": "?"}

        # Umpire
        ump       = umpires.get(str(game_id), {"name": "TBD"})
        ump_name  = ump.get("name", "TBD")
        ump_stats = {}
        if ump_name and ump_name != "TBD":
            try:
                ump_stats = get_umpire_career_stats(ump_name)
            except Exception:
                pass

        if ump_stats:
            ump_detail = (f"Avg R/G: {ump_stats.get('avg_runs','?')} &nbsp;|&nbsp; "
                          f"K%: {ump_stats.get('k_rate_pct','?')}% &nbsp;|&nbsp; "
                          f"{ump_stats.get('tendency','')}")
        else:
            ump_detail = ""

        # Team records
        try:
            h_rec = get_team_recent_record(home_team, game_date, n=10)
            a_rec = get_team_recent_record(away_team, game_date, n=10)
        except Exception:
            h_rec = a_rec = {}

        def _rec_str(rec: dict) -> str:
            if not rec:
                return "— (no data)"
            return f"{rec['wins']}-{rec['losses']} (L{rec['games']})"

        def _rdiff_str(v: float) -> str:
            return (f"+{v:.1f}" if v > 0 else f"{v:.1f}") if v else "0.0"

        # Build HTML card
        card = f"""
<table border="0" cellspacing="0" cellpadding="0" width="100%"
       style="border-collapse:collapse;border:1px solid #ddd;border-radius:4px;margin-bottom:16px;overflow:hidden">
  <!-- Header -->
  <tr style="background:#1a237e;color:#fff">
    <td style="padding:9px 13px;font-size:15px;font-weight:bold">{away_team} @ {home_team}</td>
    <td style="padding:9px 13px;text-align:right;white-space:nowrap">
      PICK: <b style="color:#fff176">{pick}</b>
      &nbsp;<span style="color:{conf_color};background:#fff;border-radius:3px;padding:1px 7px;font-size:13px">{conf:.1f}%</span>
      &nbsp;&nbsp;|&nbsp;&nbsp;
      <b style="color:#fff176">{ou_dir}</b> {pred_total:.1f} / {line_str}
      &nbsp;<span style="background:#fff;border-radius:3px;padding:1px 7px;font-size:13px;color:#555">{ou_conf:.0f}%</span>
    </td>
  </tr>
  {_label_row("Starting Pitchers")}
  {_pitcher_row("home", home_team, h_sp["name"], h_sp["hand"],
                h_xera, h_fip, h_k, h_bb, h_era_l3, h_whip_l3, "#fff")}
  {_pitcher_row("away", away_team, a_sp["name"], a_sp["hand"],
                a_xera, a_fip, a_k, a_bb, a_era_l3, a_whip_l3, "#fafafa")}
  <!-- Umpire -->
  <tr style="background:#fffde7">
    <td colspan="10" style="padding:5px 13px">
      <span style="font-size:11px;font-weight:bold;color:#555;text-transform:uppercase">HP Umpire &nbsp;</span>
      <b>{ump_name}</b>
      {"&nbsp;&nbsp;<span style='color:#555;font-size:13px'>" + ump_detail + "</span>" if ump_detail else ""}
    </td>
  </tr>
  {_label_row("Team Trends · Last 10 Games")}
  <!-- Away team trends -->
  <tr style="background:#fff">
    <td colspan="10" style="padding:5px 13px">
      <b>{away_team}</b> &nbsp;{_rec_str(a_rec)}
      &nbsp;&nbsp;|&nbsp; OPS: <b>{a_ops:.3f}</b>
      &nbsp;|&nbsp; RISP: <b>{a_risp:.3f}</b>
      &nbsp;|&nbsp; R/G: {a_rec.get("avg_rs","—")} scored / {a_rec.get("avg_ra","—")} allowed
      &nbsp;|&nbsp; Run Diff/G: <b>{_rdiff_str(a_rdiff)}</b>
    </td>
  </tr>
  <!-- Home team trends -->
  <tr style="background:#f5f5f5">
    <td colspan="10" style="padding:5px 13px">
      <b>{home_team}</b> &nbsp;{_rec_str(h_rec)}
      &nbsp;&nbsp;|&nbsp; OPS: <b>{h_ops:.3f}</b>
      &nbsp;|&nbsp; RISP: <b>{h_risp:.3f}</b>
      &nbsp;|&nbsp; R/G: {h_rec.get("avg_rs","—")} scored / {h_rec.get("avg_ra","—")} allowed
      &nbsp;|&nbsp; Run Diff/G: <b>{_rdiff_str(h_rdiff)}</b>
    </td>
  </tr>
</table>"""
        cards.append(card)

    return "\n".join(cards)


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

    # Build detailed briefs only when predictions were made
    briefs_html = ""
    if run_status in ("SUCCESS", "PARTIAL") and n_games > 0:
        try:
            briefs_html = _build_game_brief_cards(date_str)
        except Exception as exc:
            logger.warning("Notifier: brief build failed: %s", exc)

    body = f"""
<p>{_status_badge(run_status)} &nbsp; <b>{n_games} games</b> &middot; {date_str}</p>
{failure_html}
<h3 style="margin-bottom:6px">Quick Summary</h3>
{_predictions_table(date_str, 'A')}
{"<h3 style='margin:18px 0 10px'>Game Briefs</h3>" + briefs_html if briefs_html else ""}
"""
    _send(f"MLB Cycle A — {date_str} ({n_games} games, {run_status})",
          _html_wrap(f"Cycle A &middot; {date_str}", body))


def notify_cycle_b(game_date: date, status: dict) -> None:
    date_str   = game_date.strftime("%Y-%m-%d")
    run_status = status.get("run_status", "UNKNOWN")
    n_games    = status.get("games_evaluated", 0)
    failure    = status.get("failure_reason")

    failure_html = (f'<p style="color:#c62828"><b>Note:</b> {failure}</p>'
                    if failure else "")

    briefs_html = ""
    if run_status in ("SUCCESS", "PARTIAL") and n_games > 0:
        try:
            briefs_html = _build_game_brief_cards(date_str)
        except Exception as exc:
            logger.warning("Notifier: brief build failed: %s", exc)

    body = f"""
<p>{_status_badge(run_status)} &nbsp; <b>{n_games} games</b> &middot; {date_str}</p>
{failure_html}
<h3 style="margin-bottom:6px">Lock Predictions (confirmed lineups + weather)</h3>
{_predictions_table(date_str, 'B')}
{"<h3 style='margin:18px 0 10px'>Game Briefs</h3>" + briefs_html if briefs_html else ""}
"""
    _send(f"MLB Cycle B — {date_str} ({n_games} games, {run_status})",
          _html_wrap(f"Cycle B &middot; {date_str}", body))


def notify_post_game(game_date: date, status: dict) -> None:
    date_str   = game_date.strftime("%Y-%m-%d")
    run_status = status.get("run_status", "UNKNOWN")
    n_games    = status.get("games_evaluated", 0)
    failure    = status.get("failure_reason")

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
                r.total_runs                                  AS actual_total,
                p.ou_prob,
                p.ou_line
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

    # Compute MAE directly from rows (don't rely on eval_log which may be None)
    mae_vals = [abs((r[5] or 0) - (r[6] or 0)) for r in rows if r[5] is not None and r[6] is not None]
    mae_computed = round(sum(mae_vals) / len(mae_vals), 2) if mae_vals else None

    # O/U accuracy: only count games where we had a real sportsbook line
    ou_correct = 0
    ou_total   = 0
    for r in rows:
        ou_prob, ou_line, actual_total = r[7], r[8], r[6]
        if ou_prob is not None and ou_line is not None and actual_total is not None:
            ou_total += 1
            predicted_over = ou_prob >= 0.5
            actual_over    = actual_total > ou_line
            if predicted_over == actual_over:
                ou_correct += 1

    def _ou_result_cell(ou_prob, ou_line, actual_total) -> str:
        if ou_prob is None or ou_line is None or actual_total is None:
            return "<span style='color:#aaa'>—</span>"
        predicted_over = ou_prob >= 0.5
        actual_over    = actual_total > ou_line
        direction      = "Over" if predicted_over else "Under"
        correct_call   = predicted_over == actual_over
        icon  = "&#10003;" if correct_call else "&#10007;"
        color = "#2e7d32"  if correct_call else "#c62828"
        return (f'<span style="color:{color};font-weight:bold">{direction} {icon}</span>'
                f'<span style="color:#888;font-size:11px"> ({actual_total} vs {ou_line:.1f})</span>')

    rows_html = "".join(
        f"<tr style='background:{'#f5f5f5' if i % 2 else '#fff'}'>"
        f"<td style='padding:6px 10px'>{r[0]}</td>"
        f"<td style='padding:6px 10px'>{r[1]}</td>"
        f"<td style='padding:6px 10px'>{r[2]}</td>"
        f"<td style='padding:6px 10px;font-weight:bold'>{r[3]}</td>"
        f"<td style='padding:6px 10px;font-size:16px'>{r[4]}</td>"
        f"<td style='padding:6px 10px'>{r[5]} &rarr; {r[6]}</td>"
        f"<td style='padding:6px 10px'>{_ou_result_cell(r[7], r[8], r[6])}</td>"
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
      <th style="padding:8px 10px;text-align:left">O/U</th>
    </tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table>""" if rows else "<p>No matched results found.</p>"

    win_pct = round(correct / total_g * 100, 1) if total_g else 0
    ou_pct  = round(ou_correct / ou_total * 100, 1) if ou_total else None

    metrics_html = f"""
<table style="margin-bottom:16px;border-collapse:collapse">
  <tr><td style="padding:6px 16px 6px 0"><b>Win accuracy</b></td>
      <td>{correct}/{total_g} ({win_pct}%)</td></tr>
  <tr><td style="padding:6px 16px 6px 0"><b>O/U accuracy</b></td>
      <td>{"—" if ou_pct is None else f"{ou_correct}/{ou_total} ({ou_pct}%)"}</td></tr>
  <tr><td style="padding:6px 16px 6px 0"><b>Run total MAE</b></td>
      <td>{"—" if mae_computed is None else f"{mae_computed} runs"}</td></tr>
  <tr><td style="padding:6px 16px 6px 0"><b>Brier score</b></td>
      <td>{round(metrics["brier"] or 0, 4) if metrics.get("brier") is not None else "—"}</td></tr>
</table>"""

    failure_html = (f'<p style="color:#c62828"><b>Note:</b> {failure}</p>'
                    if failure else "")
    body = f"""
<p>{_status_badge(run_status)} &nbsp; <b>{n_games} results written</b> &middot; {date_str}</p>
{failure_html}
<h3 style="margin-bottom:6px">Model Performance</h3>
{metrics_html}
<h3 style="margin-bottom:6px">Results vs Predictions</h3>
{results_table}
"""
    _send(f"MLB Post-game — {date_str} ({correct}/{total_g} correct)",
          _html_wrap(f"Post-game Results &middot; {date_str}", body))


def notify_retrain(metrics: dict) -> None:
    """
    Send a retrain summary email.

    metrics keys:
      retrain_date  str         e.g. "2026-06-23"
      model_version str         e.g. "v1.20260623"
      n_total       int         total games in training set
      n_live        int         live DB games included
      win_brier     float       CV Brier score (win model)
      win_brier_std float
      win_mae       float       CV MAE (run total model)
      win_mae_std   float
      ou_trained    bool        whether OU model was retrained
      ou_brier      float|None
      backed_up     list[str]   model filenames that were backed up
    """
    date_str      = metrics.get("retrain_date", date.today().isoformat())
    model_version = metrics.get("model_version", "—")
    n_total       = metrics.get("n_total", 0)
    n_live        = metrics.get("n_live",  0)
    win_brier     = metrics.get("win_brier")
    win_brier_std = metrics.get("win_brier_std")
    mae           = metrics.get("win_mae")
    mae_std       = metrics.get("win_mae_std")
    ou_trained    = metrics.get("ou_trained", False)
    ou_brier      = metrics.get("ou_brier")
    backed_up     = metrics.get("backed_up", [])

    def _fmt(v, std=None, decimals=4):
        if v is None:
            return "—"
        s = f"{v:.{decimals}f}"
        if std is not None:
            s += f" ± {std:.{decimals}f}"
        return s

    backed_html = (
        "<ul style='margin:4px 0;padding-left:18px'>"
        + "".join(f"<li style='font-size:12px;color:#555'>{b}</li>" for b in backed_up)
        + "</ul>"
    ) if backed_up else "<p style='color:#888;font-size:12px'>none</p>"

    ou_row = (
        f"<tr><td style='padding:6px 16px 6px 0'><b>O/U model Brier</b></td>"
        f"<td>{_fmt(ou_brier)}</td></tr>"
    ) if ou_trained else (
        "<tr><td style='padding:6px 16px 6px 0'><b>O/U model</b></td>"
        "<td style='color:#888'>skipped (need ≥ 200 games with ou_line)</td></tr>"
    )

    body = f"""
<p>&#128296; Model retrained on <b>{date_str}</b> &nbsp;&middot;&nbsp;
   version <code>{model_version}</code></p>
<h3 style="margin-bottom:6px">Training Set</h3>
<table style="margin-bottom:16px;border-collapse:collapse">
  <tr><td style="padding:6px 16px 6px 0"><b>Total games</b></td>
      <td>{n_total:,}</td></tr>
  <tr><td style="padding:6px 16px 6px 0"><b>Live DB games</b></td>
      <td>{n_live:,} &nbsp;<span style="color:#888;font-size:12px">(Cycle B + A fallback)</span></td></tr>
  <tr><td style="padding:6px 16px 6px 0"><b>Historical games</b></td>
      <td>{n_total - n_live:,}</td></tr>
</table>
<h3 style="margin-bottom:6px">CV Metrics (TimeSeriesSplit, 5 folds)</h3>
<table style="margin-bottom:16px;border-collapse:collapse">
  <tr><td style="padding:6px 16px 6px 0"><b>Win model Brier</b></td>
      <td>{_fmt(win_brier, win_brier_std)} &nbsp;<span style="color:#888;font-size:12px">(target &lt; 0.23)</span></td></tr>
  <tr><td style="padding:6px 16px 6px 0"><b>Run total MAE</b></td>
      <td>{_fmt(mae, mae_std, decimals=3)} runs &nbsp;<span style="color:#888;font-size:12px">(target &lt; 1.8)</span></td></tr>
  {ou_row}
</table>
<h3 style="margin-bottom:6px">Backed-up Models</h3>
{backed_html}
"""
    _send(f"MLB Retrain — {date_str} ({model_version})",
          _html_wrap(f"Model Retrain &middot; {date_str}", body))
