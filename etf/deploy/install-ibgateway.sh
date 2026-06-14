#!/usr/bin/env bash
# ============================================================================
# install-ibgateway.sh — install IB Gateway + IBC + Xvfb on a headless droplet
# ============================================================================
# Idempotent: safe to re-run (skips downloads that already exist). Installs:
#   1. Xvfb + a JRE-friendly minimal desktop stack (for the Java GUI app).
#   2. IB Gateway (the IBKR-published standalone installer, bundles its own JRE).
#   3. IBC (IbcAlpha/IBC) — the headless login/automation controller.
# After this, configure etf-engine.env and start the ibc-gateway service.
# ============================================================================
set -euo pipefail

IBC_VERSION="${IBC_VERSION:-3.20.0}"
IBC_DIR="${IBC_DIR:-/opt/ibc}"
IBGW_CHANNEL="${IBGW_CHANNEL:-stable}"   # 'stable' or 'latest'
DOWNLOAD_DIR="${DOWNLOAD_DIR:-/root/ibkr-downloads}"

echo "=== IB Gateway + IBC headless install ==="
mkdir -p "$DOWNLOAD_DIR"

# --- 1. System deps ---------------------------------------------------------
echo "[1/4] Installing system packages (Xvfb, fonts, unzip)..."
apt-get update -qq
# Xvfb = virtual framebuffer; libxtst/libxrender/fonts keep the JRE GUI happy.
apt-get install -y -qq \
    xvfb unzip curl ca-certificates \
    libxtst6 libxrender1 libxi6 fonts-dejavu

# --- 2. IB Gateway (standalone, bundles a JRE) ------------------------------
echo "[2/4] Installing IB Gateway (${IBGW_CHANNEL})..."
GW_INSTALLER="$DOWNLOAD_DIR/ibgateway-${IBGW_CHANNEL}-standalone-linux-x64.sh"
if [ ! -f "$GW_INSTALLER" ]; then
    curl -fsSL \
        "https://download2.interactivebrokers.com/installers/ibgateway/${IBGW_CHANNEL}-standalone/ibgateway-${IBGW_CHANNEL}-standalone-linux-x64.sh" \
        -o "$GW_INSTALLER"
fi
chmod +x "$GW_INSTALLER"
# -q = unattended; -dir installs to the default /root/Jts/ibgateway/<ver>.
# The installer is interactive by default; feed it the defaults non-interactively.
if [ ! -d "/root/Jts/ibgateway" ]; then
    echo "n" | "$GW_INSTALLER" -q -dir /root/Jts/ibgateway || \
        "$GW_INSTALLER" -q || {
            echo "  IB Gateway silent install failed; run it interactively once:" >&2
            echo "    $GW_INSTALLER" >&2
            exit 2
        }
fi
echo "  IB Gateway installed under /root/Jts/ibgateway."

# --- 3. IBC -----------------------------------------------------------------
echo "[3/4] Installing IBC ${IBC_VERSION}..."
IBC_ZIP="$DOWNLOAD_DIR/IBCLinux-${IBC_VERSION}.zip"
if [ ! -f "$IBC_ZIP" ]; then
    curl -fsSL \
        "https://github.com/IbcAlpha/IBC/releases/download/${IBC_VERSION}/IBCLinux-${IBC_VERSION}.zip" \
        -o "$IBC_ZIP"
fi
mkdir -p "$IBC_DIR"
unzip -o -q "$IBC_ZIP" -d "$IBC_DIR"
chmod +x "$IBC_DIR"/*.sh "$IBC_DIR"/scripts/*.sh 2>/dev/null || true
echo "  IBC installed under $IBC_DIR."

# --- 4. Done ----------------------------------------------------------------
echo "[4/4] IB Gateway + IBC install complete."
echo ""
echo "Detected Gateway version directory:"
ls -1 /root/Jts/ibgateway 2>/dev/null | sed 's/^/    /' || echo "    (none — check install)"
echo ""
echo "Next:"
echo "  1. Edit /root/etf-engine/etf-engine.env — set IBKR_USERNAME / IBKR_PASSWORD"
echo "     and confirm TWS_MAJOR_VRSN matches the version directory above"
echo "     (e.g. 10.30.x -> TWS_MAJOR_VRSN=1030)."
echo "  2. systemctl start ibc-gateway   &&   journalctl -u ibc-gateway -f"
echo "  3. Once Gateway is up: cd /root/etf-engine && python3 -m etf.main --mode preflight"
