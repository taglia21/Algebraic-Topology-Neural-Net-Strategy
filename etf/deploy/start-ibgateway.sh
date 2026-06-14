#!/usr/bin/env bash
# ============================================================================
# start-ibgateway.sh — headless IB Gateway launcher (Xvfb + IBC)
# ============================================================================
# Run by the ibc-gateway.service systemd unit. It:
#   1. Renders the IBC config template -> a root-only runtime config, injecting
#      credentials from the environment (the systemd EnvironmentFile).
#   2. Starts a virtual X display (Xvfb) so the Java GUI app can run headless.
#   3. Launches IB Gateway under IBC (auto-login, dialog handling, daily restart).
#
# Credentials NEVER touch git: they come from /root/etf-engine/etf-engine.env.
# ============================================================================
set -euo pipefail

# --- Paths & versions (override via the env file if needed) ------------------
IBC_DIR="${IBC_DIR:-/opt/ibc}"
TWS_SETTINGS_DIR="${TWS_SETTINGS_DIR:-/root/Jts}"
IBGW_DIR="${IBGW_DIR:-/root/Jts/ibgateway}"
TWS_MAJOR_VRSN="${TWS_MAJOR_VRSN:-1030}"
XVFB_DISPLAY="${XVFB_DISPLAY:-:1}"
XVFB_RES="${XVFB_RES:-1024x768x24}"

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="$DEPLOY_DIR/ibc-config.ini.template"
RUNTIME_CONFIG="${IBC_INI:-/opt/ibc/config.runtime.ini}"

# --- Validate required environment ------------------------------------------
: "${IBKR_USERNAME:?Set IBKR_USERNAME in etf-engine.env}"
: "${IBKR_PASSWORD:?Set IBKR_PASSWORD in etf-engine.env}"
IBKR_TRADING_MODE="${IBKR_TRADING_MODE:-paper}"
IBKR_PORT="${IBKR_PORT:-4002}"
IBC_RESTART_TIME="${IBC_RESTART_TIME:-07:00 AM}"

if [ "$IBKR_TRADING_MODE" = "live" ]; then
    echo "[start-ibgateway] REFUSING to auto-start in LIVE mode from this script." >&2
    echo "                  Live bring-up is a deliberate, supervised action." >&2
    exit 3
fi

# --- 1. Render the runtime IBC config (root-only) ---------------------------
echo "[start-ibgateway] Rendering IBC config -> $RUNTIME_CONFIG"
mkdir -p "$(dirname "$RUNTIME_CONFIG")"
sed \
    -e "s|__IBKR_USERNAME__|${IBKR_USERNAME}|g" \
    -e "s|__IBKR_PASSWORD__|${IBKR_PASSWORD}|g" \
    -e "s|__IBKR_TRADING_MODE__|${IBKR_TRADING_MODE}|g" \
    -e "s|__IBKR_PORT__|${IBKR_PORT}|g" \
    -e "s|__IBC_RESTART_TIME__|${IBC_RESTART_TIME}|g" \
    "$TEMPLATE" > "$RUNTIME_CONFIG"
chmod 600 "$RUNTIME_CONFIG"

# --- 2. Start a virtual display ---------------------------------------------
# Kill any stale Xvfb on this display, then start a fresh one.
pkill -f "Xvfb ${XVFB_DISPLAY}" 2>/dev/null || true
echo "[start-ibgateway] Starting Xvfb on ${XVFB_DISPLAY} (${XVFB_RES})"
Xvfb "${XVFB_DISPLAY}" -screen 0 "${XVFB_RES}" -nolisten tcp &
XVFB_PID=$!
export DISPLAY="${XVFB_DISPLAY}"

# Ensure Xvfb is cleaned up when this script exits.
cleanup() {
    echo "[start-ibgateway] Shutting down; stopping Xvfb (${XVFB_PID})."
    kill "${XVFB_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Give Xvfb a moment to come up.
sleep 2

# --- 3. Ensure IBC can find the Gateway install -----------------------------
# IBC looks for the Gateway jars at  $TWS_PATH/$TWS_MAJOR_VRSN/jars.  The
# installer lays the Gateway down directly in $IBGW_DIR (jars at $IBGW_DIR/jars),
# so expose it under the version-named path IBC expects via a sibling symlink.
# This works for ANY installed version and needs no reinstall.
GW_PARENT="$(dirname "$IBGW_DIR")"            # e.g. /root/Jts
GW_VERSIONED="${GW_PARENT}/${TWS_MAJOR_VRSN}" # e.g. /root/Jts/1045
if [ -d "${IBGW_DIR}/jars" ] && [ ! -e "${GW_VERSIONED}/jars" ]; then
    echo "[start-ibgateway] Linking ${GW_VERSIONED} -> ${IBGW_DIR} (so IBC finds jars)"
    ln -sfn "${IBGW_DIR}" "${GW_VERSIONED}"
fi

# --- 4. Launch IB Gateway under IBC -----------------------------------------
echo "[start-ibgateway] Launching IB Gateway (mode=${IBKR_TRADING_MODE}, port=${IBKR_PORT})"
export TWS_MAJOR_VRSN
export IBC_INI="$RUNTIME_CONFIG"
export TWS_SETTINGS_PATH="$TWS_SETTINGS_DIR"
export IBC_PATH="$IBC_DIR"

# IBC's gateway start script. --gateway selects Gateway (not full TWS).
# --tws-path is the PARENT that contains the version-named folder above.
exec "${IBC_DIR}/scripts/ibcstart.sh" "${TWS_MAJOR_VRSN}" \
    --gateway \
    "--mode=${IBKR_TRADING_MODE}" \
    "--ibc-ini=${RUNTIME_CONFIG}" \
    "--ibc-path=${IBC_DIR}" \
    "--tws-path=${GW_PARENT}" \
    "--tws-settings-path=${TWS_SETTINGS_DIR}"
