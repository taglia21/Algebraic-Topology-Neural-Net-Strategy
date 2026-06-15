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
# Use Python for substitution (NOT sed): passwords frequently contain sed-special
# characters (& / | \ newlines). With sed those silently corrupt the rendered
# password, so Gateway login hangs forever at "Setting password" with no error.
# Python does a literal string replace and reads secrets straight from the
# environment, so nothing sensitive ever lands in argv (visible via `ps`).
echo "[start-ibgateway] Rendering IBC config -> $RUNTIME_CONFIG"
mkdir -p "$(dirname "$RUNTIME_CONFIG")"
export IBKR_USERNAME IBKR_PASSWORD IBKR_TRADING_MODE IBKR_PORT IBC_RESTART_TIME
python3 - "$TEMPLATE" "$RUNTIME_CONFIG" <<'PYEOF'
import os, sys
template_path, out_path = sys.argv[1], sys.argv[2]
with open(template_path, "r") as f:
    content = f.read()
replacements = {
    "__IBKR_USERNAME__": os.environ["IBKR_USERNAME"],
    "__IBKR_PASSWORD__": os.environ["IBKR_PASSWORD"],
    "__IBKR_TRADING_MODE__": os.environ.get("IBKR_TRADING_MODE", "paper"),
    "__IBKR_PORT__": os.environ.get("IBKR_PORT", "4002"),
    "__IBC_RESTART_TIME__": os.environ.get("IBC_RESTART_TIME", "07:00 AM"),
}
for token, value in replacements.items():
    content = content.replace(token, value)
with open(out_path, "w") as f:
    f.write(content)
PYEOF
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
# IBC (for Gateway) expects the install under  <tws-path>/ibgateway/<version>/
# and, finding jars there, uses ibgateway.vmoptions from the SAME folder. The
# IBKR installer laid the files down flat in <tws-path>/ibgateway/ (jars and
# ibgateway.vmoptions directly inside), so expose that flat install under the
# version-named subfolder IBC expects. Linking at ibgateway/<version> (NOT
# <tws-path>/<version>) is what makes IBC pick ibgateway.vmoptions instead of
# tws.vmoptions. Works for any version; no reinstall needed.
GW_PARENT="$(dirname "$IBGW_DIR")"            # e.g. /root/Jts  (IBC --tws-path)
GW_VERSIONED="${IBGW_DIR}/${TWS_MAJOR_VRSN}"  # e.g. /root/Jts/ibgateway/1045
if [ -d "${IBGW_DIR}/jars" ] && [ ! -e "${GW_VERSIONED}/jars" ]; then
    echo "[start-ibgateway] Linking ${GW_VERSIONED} -> ${IBGW_DIR} (so IBC finds jars + vmoptions)"
    ln -sfn "${IBGW_DIR}" "${GW_VERSIONED}"
fi
# Clean up a stale link from an earlier layout attempt (<tws-path>/<version>),
# which made IBC fall back to the TWS path and look for tws.vmoptions.
STALE_VERSIONED="${GW_PARENT}/${TWS_MAJOR_VRSN}"
if [ -L "${STALE_VERSIONED}" ]; then
    echo "[start-ibgateway] Removing stale link ${STALE_VERSIONED}"
    rm -f "${STALE_VERSIONED}"
fi

# --- 4. Launch IB Gateway under IBC -----------------------------------------
echo "[start-ibgateway] Launching IB Gateway (mode=${IBKR_TRADING_MODE}, port=${IBKR_PORT})"
export TWS_MAJOR_VRSN
export IBC_INI="$RUNTIME_CONFIG"
export TWS_SETTINGS_PATH="$TWS_SETTINGS_DIR"
export IBC_PATH="$IBC_DIR"

# IBC's gateway start script. --gateway selects Gateway (not full TWS).
# --tws-path is the PARENT that contains the ibgateway/<version> folder above.
exec "${IBC_DIR}/scripts/ibcstart.sh" "${TWS_MAJOR_VRSN}" \
    --gateway \
    "--mode=${IBKR_TRADING_MODE}" \
    "--ibc-ini=${RUNTIME_CONFIG}" \
    "--ibc-path=${IBC_DIR}" \
    "--tws-path=${GW_PARENT}" \
    "--tws-settings-path=${TWS_SETTINGS_DIR}"
