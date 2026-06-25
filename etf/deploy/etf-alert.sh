#!/usr/bin/env bash
# =============================================================================
# etf-alert.sh — operational alert sink for the ETF engine.
#
# Always: appends the message to .etf_telemetry/ALERTS.log and to the system
#         journal (tag etf-alert), so an operator can ALWAYS find it locally.
# If ETF_ALERT_WEBHOOK is set: POSTs the message as JSON {"text": "..."} to that
#         URL. This payload shape is accepted by Slack, Discord (via /slack),
#         healthchecks.io, ntfy, and most generic webhook receivers, so you can
#         get a phone push with zero code changes — just set the env var.
#
# Usage:  etf-alert.sh "human readable message"
# =============================================================================
set -uo pipefail

MSG="${*:-ETF engine alert (no message provided)}"
TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
ALERT_DIR="${ETF_ALERT_DIR:-/root/etf-engine/.etf_telemetry}"

mkdir -p "$ALERT_DIR" 2>/dev/null || true
echo "[$TS] $MSG" | tee -a "$ALERT_DIR/ALERTS.log" 2>/dev/null || echo "[$TS] $MSG"
logger -t etf-alert -- "$MSG" 2>/dev/null || true

if [[ -n "${ETF_ALERT_WEBHOOK:-}" ]]; then
    # Build a JSON payload safely (escape backslashes and double quotes).
    esc=${MSG//\\/\\\\}
    esc=${esc//\"/\\\"}
    payload="{\"text\":\"[ETF $TS] $esc\"}"
    if curl -fsS -m 10 -H 'Content-Type: application/json' \
            -X POST -d "$payload" "$ETF_ALERT_WEBHOOK" >/dev/null 2>&1; then
        echo "[$TS] alert webhook delivered" >> "$ALERT_DIR/ALERTS.log" 2>/dev/null || true
    else
        echo "[$TS] alert webhook FAILED to deliver" >> "$ALERT_DIR/ALERTS.log" 2>/dev/null || true
    fi
fi

exit 0
