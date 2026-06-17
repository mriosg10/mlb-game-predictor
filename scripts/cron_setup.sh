#!/usr/bin/env bash
# ============================================================================
# MLB Prediction Pipeline — cron installation script
#
# Cron schedules are specified in Eastern Time.
# To convert to the server's local timezone, adjust accordingly.
# All jobs set PYTHONUNBUFFERED=1 so log output is flushed immediately.
#
# Schedule (Section 5.1):
#   Cycle A — Seed:    08:00 ET daily
#   Cycle B — Lock:    13:30 ET daily  (~2.5h before earliest first pitch)
#   Post-game:         23:30 ET daily
#
# Post-game actuals retry: if the 23:30 run returns PARTIAL (some games
# still in progress), a retry job runs at 00:30 the next day.
#
# Usage:
#   chmod +x scripts/cron_setup.sh
#   ./scripts/cron_setup.sh
#
# WARNING: This script APPENDS to your crontab.
# Review with:  crontab -l
# Remove with:  crontab -r  (or edit manually with crontab -e)
# ============================================================================

set -euo pipefail

# Detect the directory containing this script and set PIPELINE_DIR to its parent
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="${PYTHON:-python3}"
LOG_DIR="${PIPELINE_DIR}/logs"

mkdir -p "$LOG_DIR"

# Verify python + main.py are present
if ! command -v "$PYTHON" &>/dev/null; then
    echo "ERROR: Python interpreter not found at '$PYTHON'"
    echo "Set the PYTHON environment variable to the correct interpreter path."
    exit 1
fi

if [ ! -f "$PIPELINE_DIR/main.py" ]; then
    echo "ERROR: $PIPELINE_DIR/main.py not found"
    exit 1
fi

RUN_CMD="cd \"$PIPELINE_DIR\" && PYTHONUNBUFFERED=1 \"$PYTHON\" main.py"

# Build crontab entries
# Note: cron runs in UTC by default on many Linux systems.
# Adjust hours for UTC offset:
#   EDT (Apr–Nov): ET = UTC - 4  => ET 08:00 = UTC 12:00
#   EST (Nov–Mar): ET = UTC - 5  => ET 08:00 = UTC 13:00
# The offsets below use EDT. Update for EST if the season extends into November.

CYCLE_A_CRON="0 12 * * *"        # 08:00 ET (EDT) = 12:00 UTC
CYCLE_B_CRON="30 17 * * *"       # 13:30 ET (EDT) = 17:30 UTC
POST_GAME_CRON="30 3 * * *"      # 23:30 ET (EDT) = 03:30 UTC next day

# Post-game retry: 00:30 ET (+1 day) = 04:30 UTC
POST_RETRY_CRON="30 4 * * *"

CRON_A="${CYCLE_A_CRON}    ${RUN_CMD} --cycle A >> \"${LOG_DIR}/cycle_a.log\" 2>&1"
CRON_B="${CYCLE_B_CRON}    ${RUN_CMD} --cycle B >> \"${LOG_DIR}/cycle_b.log\" 2>&1"
CRON_POST="${POST_GAME_CRON}    ${RUN_CMD} --cycle post >> \"${LOG_DIR}/post_game.log\" 2>&1"
CRON_RETRY="${POST_RETRY_CRON}  ${RUN_CMD} --cycle post >> \"${LOG_DIR}/post_game_retry.log\" 2>&1"

echo "Installing cron jobs for pipeline at: $PIPELINE_DIR"
echo ""
echo "  Cycle A  (seed):       $CYCLE_A_CRON"
echo "  Cycle B  (lock):       $CYCLE_B_CRON"
echo "  Post-game actuals:     $POST_GAME_CRON"
echo "  Post-game retry:       $POST_RETRY_CRON"
echo ""

# Append to existing crontab (preserves other entries)
(crontab -l 2>/dev/null || true; echo "# MLB Prediction Pipeline — installed by cron_setup.sh") | crontab -
(crontab -l 2>/dev/null; echo "${CRON_A}") | crontab -
(crontab -l 2>/dev/null; echo "${CRON_B}") | crontab -
(crontab -l 2>/dev/null; echo "${CRON_POST}") | crontab -
(crontab -l 2>/dev/null; echo "${CRON_RETRY}") | crontab -

echo "Cron jobs installed. Verify with: crontab -l"
echo ""
echo "NOTE: Cron times above are UTC (EDT offset). Adjust for EST (Nov-Mar) by adding 1 hour."
