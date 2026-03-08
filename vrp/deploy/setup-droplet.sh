#!/usr/bin/env bash
# VRP Engine — Droplet Setup Script
# Run once on a fresh Ubuntu 22.04 droplet to configure the service.
#
# Usage: bash setup-droplet.sh
set -euo pipefail

APP_DIR="/root/vrp-engine"
REPO_URL="https://github.com/taglia21/Algebraic-Topology-Neural-Net-Strategy.git"

echo "=== VRP Engine — Droplet Setup ==="

# 1. System deps
echo "[1/6] Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq python3 python3-pip git

# 2. Clone or pull repo
echo "[2/6] Setting up repository..."
if [ -d "$APP_DIR/.git" ]; then
    cd "$APP_DIR" && git pull --ff-only origin main
else
    git clone "$REPO_URL" "$APP_DIR"
fi
cd "$APP_DIR"

# 3. Install Python deps
echo "[3/6] Installing Python dependencies..."
pip3 install --quiet ib_async

# 4. Create state directory
echo "[4/6] Creating state directory..."
mkdir -p "$APP_DIR/state"

# 5. Install systemd service
echo "[5/6] Installing systemd service..."
cp vrp/deploy/vrp-engine.service /etc/systemd/system/vrp-engine.service
systemctl daemon-reload
systemctl enable vrp-engine

# 6. Done
echo "[6/6] Setup complete."
echo ""
echo "Next steps:"
echo "  1. Install IBKR Gateway on this machine (or on a LAN host)"
echo "  2. Edit /etc/systemd/system/vrp-engine.service to set:"
echo "       IBKR_HOST=127.0.0.1"
echo "       IBKR_PORT=4002  (paper) or 4001 (live)"
echo "       IBKR_ACCOUNT=U22452226"
echo "  3. Start the engine:"
echo "       systemctl start vrp-engine"
echo "       journalctl -u vrp-engine -f"
echo ""
