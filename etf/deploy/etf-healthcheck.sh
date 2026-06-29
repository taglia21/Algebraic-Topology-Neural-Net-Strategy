#!/usr/bin/env bash
# =============================================================================
# etf-healthcheck.sh — periodic liveness probe for the ETF stack.
#
# Run by etf-healthcheck.timer (default every 10 min). It verifies that BOTH
# the IB Gateway and the ETF engine are 'active'. Anything else (failed,
# inactive, or stuck auto-restarting/'activating') raises an alert via
# etf-alert.sh. This catches the two silent failure modes that previously went
# unnoticed for days:
#   1. the engine crash-looping (never reaches 'active'), and
#   2. the Gateway dropping its session so nothing can trade.
#
# Exit 0 = healthy, 1 = a problem was detected (and alerted).
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALERT="$SCRIPT_DIR/etf-alert.sh"

UNITS=("${ETF_HEALTHCHECK_UNITS:-ibc-gateway.service etf-engine.service}")
# shellcheck disable=SC2206
UNITS=(${UNITS[*]})

# Optional operator maintenance lock: when present, suppress liveness paging.
# Use this before intentional service stops/manual runs to avoid alert storms.
MAINT_FILE="${ETF_HEALTHCHECK_MAINTENANCE_FILE:-/tmp/etf-healthcheck.maintenance}"
if [[ -f "$MAINT_FILE" ]]; then
    exit 0
fi

problems=()
for unit in "${UNITS[@]}"; do
    state="$(systemctl is-active "$unit" 2>/dev/null || true)"
    if [[ "$state" != "active" ]]; then
        problems+=("$unit=$state")
    fi
done

if (( ${#problems[@]} > 0 )); then
    "$ALERT" "Health check FAILED — not trading. Unhealthy units: ${problems[*]}. Investigate with: systemctl status ${UNITS[*]}"
    exit 1
fi

exit 0
