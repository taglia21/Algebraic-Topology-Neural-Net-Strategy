#!/usr/bin/env bash
# ETF Engine — Droplet Setup Script (headless 24/7 paper trading)
# Run once on a fresh Ubuntu 22.04/24.04 droplet to configure BOTH services:
#   - ibc-gateway : headless IB Gateway (Xvfb + IBC auto-login + daily restart)
#   - etf-engine  : the market-hours-aware ETF trading runner
#
# Usage: bash setup-droplet.sh
set -euo pipefail

APP_DIR="/root/etf-engine"
REPO_URL="https://github.com/taglia21/Algebraic-Topology-Neural-Net-Strategy.git"

echo "=== ETF Engine — Droplet Setup (headless paper trading) ==="

# 1. System deps
echo "[1/9] Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq python3 python3-pip git tzdata

# 2. Timezone -> America/New_York (so the daily Gateway restart & market-hours
#    logic line up with the US session the strategy trades).
echo "[2/9] Setting timezone to America/New_York..."
timedatectl set-timezone America/New_York 2>/dev/null || \
    ln -sf /usr/share/zoneinfo/America/New_York /etc/localtime || true

# 3. Clone or pull repo
echo "[3/9] Setting up repository..."
if [ -d "$APP_DIR/.git" ]; then
    cd "$APP_DIR" && git pull --ff-only origin main
else
    git clone "$REPO_URL" "$APP_DIR"
fi
cd "$APP_DIR"

# 4. Install Python deps
echo "[4/9] Installing Python dependencies..."
pip3 install --quiet -r requirements.txt

# 5. Create telemetry/state directory
echo "[5/9] Creating state directory..."
mkdir -p "$APP_DIR/.etf_telemetry" "$APP_DIR/.etf_cache"

# 6. Install environment file (do not overwrite an existing one) and lock it down
echo "[6/9] Installing environment file..."
if [ ! -f "$APP_DIR/etf-engine.env" ]; then
    cp etf/deploy/etf-engine.env.example "$APP_DIR/etf-engine.env"
    echo "    -> created $APP_DIR/etf-engine.env (EDIT THIS before starting)."
else
    echo "    -> $APP_DIR/etf-engine.env already exists; left untouched."
fi
# Credentials live here -> root-only.
chmod 600 "$APP_DIR/etf-engine.env"
chmod +x "$APP_DIR/etf/deploy/"*.sh || true

# 7. Install IB Gateway + IBC (headless)
echo "[7/9] Installing IB Gateway + IBC (this downloads ~500MB)..."
bash "$APP_DIR/etf/deploy/install-ibgateway.sh"

# 8. Install systemd services
echo "[8/9] Installing systemd services..."
cp etf/deploy/ibc-gateway.service /etc/systemd/system/ibc-gateway.service
cp etf/deploy/etf-engine.service  /etc/systemd/system/etf-engine.service
# The ETF engine should only start once Gateway is up; express the dependency.
mkdir -p /etc/systemd/system/etf-engine.service.d
cat > /etc/systemd/system/etf-engine.service.d/10-after-gateway.conf <<'EOF'
[Unit]
After=ibc-gateway.service
Wants=ibc-gateway.service
EOF
systemctl daemon-reload
systemctl enable ibc-gateway etf-engine

# 9. Done
echo "[9/9] Setup complete."
echo ""
echo "Next steps:"
echo "  1. Edit $APP_DIR/etf-engine.env:"
echo "       IBKR_USERNAME / IBKR_PASSWORD  (your paper login)"
echo "       IBKR_ACCOUNT=DU...             (paper account id)"
echo "       TWS_MAJOR_VRSN=...             (match the printed Gateway version)"
echo "       IBKR_TRADING_MODE=paper        (keep paper)"
echo "  2. Start IB Gateway and wait ~30s for auto-login:"
echo "       systemctl start ibc-gateway"
echo "       journalctl -u ibc-gateway -f      # look for 'Login has completed'"
echo "  3. Validate connectivity WITHOUT trading:"
echo "       cd $APP_DIR && python3 -m etf.main --mode preflight"
echo "       -> every check should PASS (esp. 'IBKR connection')."
echo "  4. Start the engine (paper):"
echo "       systemctl start etf-engine"
echo "       journalctl -u etf-engine -f"
echo ""
echo "  See etf/deploy/RUNBOOK.md for day-to-day operations and incident steps."
echo ""

