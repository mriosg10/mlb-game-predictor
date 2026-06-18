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

_OU_LINE = 9.0  # default over/under reference line when no odds are available


def _ou_label(predicted_total: float) -> tuple:
    """Return (direction, confidence_pct) for over/under prediction."""
    diff = predicted_total - _OU_LINE
    # Confidence scales with distance from line; ±3 runs → ~100%
    conf = min(50 + abs(diff) / 3.0 * 50, 99)
    direction = "Over" if diff >= 0 else "Under"
    return direction, round(conf, 1)


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
                p.home_win_prob
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

    rows_html = ""
    for i, r in enumerate(rows):
        matchup, pick, win_conf, total, home_prob = r
        ou_dir, ou_conf = _ou_label(total)
        bg = "#f5f5f5" if i % 2 else "#fff"
        win_color = "#2e7d32" if win_conf >= 65 else ("#e65100" if win_conf >= 55 else "#555")
        ou_color  = "#1565c0" if ou_dir == "Over" else "#6a1b9a"
        rows_html += (
            f"<tr style='background:{bg}'>"
            f"<td style='padding:6px 10px'>{matchup}</td>"
            f"<td style='padding:6px 10px;font-weight:bold'>{pick}</td>"
            f"<td style='padding:6px 10px;color:{win_color};font-weight:bold'>{win_conf}%</td>"
            f"<td style='padding:6px 10px;color:{ou_color};font-weight:bold'>{ou_dir} ({total})</td>"
            f"<td style='padding:6px 10px;color:{ou_color}'>{ou_conf}%</td>"
            f"</tr>"
        )

    return f"""
<table border="0" cellspacing="0" cellpadding="0"
       style="border-collapse:collapse;width:100%;border:1px solid #ddd">
  <thead style="background:#1a237e;color:#fff">
    <tr>
      <th style="padding:8px 10px;text-align:left">Matchup</th>
      <th style="padding:8px 10px;text-align:left">Result Pick</th>
      <th style="padding:8px 10px;text-align:left">Win %</th>
      <th style="padding:8px 10px;text-align:left">O/U (pred total)</th>
      <th style="padding:8px 10px;text-align:left">O/U %</th>
    </tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table>
<p style="font-size:11px;color:#888;margin-top:4px">O/U reference line: {_OU_LINE} runs. O/U % = model confidence based on distance from line.</p>"""


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
                p.home_win_prob, p.predicted_total
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
         home_win_prob, pred_total) = row

        pick       = home_team if home_win_prob >= 0.5 else away_team
        conf       = max(home_win_prob, 1 - home_win_prob) * 100
        conf_color = "#2e7d32" if conf >= 65 else ("#e65100" if conf >= 55 else "#555")
        ou_dir, ou_conf = _ou_label(pred_total)

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
      <b style="color:#fff176">{ou_dir}</b> {pred_total:.1f}
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
<p>{_status_badge(run_status)} &nbsp; <b>{n_games} results written</b> &middot; {date_str}</p>
{failure_html}
<h3 style="margin-bottom:6px">Model Performance</h3>
{metrics_html}
<h3 style="margin-bottom:6px">Results vs Predictions</h3>
{results_table}
"""
    _send(f"MLB Post-game — {date_str} ({correct}/{total_g} correct)",
          _html_wrap(f"Post-game Results &middot; {date_str}", body))
