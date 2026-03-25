#!/usr/bin/env bash
# =============================================================================
# ATNN v2 — DigitalOcean Droplet Setup Script
# =============================================================================
# Run once on a fresh Ubuntu 22.04 droplet to install all dependencies,
# configure IB Gateway, and set up the trading bot as a system service.
#
# Usage:
#   ssh root@134.209.40.95 'bash -s' < deploy/setup-droplet.sh
#   — OR —
#   scp this file to the droplet and run: sudo bash setup-droplet.sh
# =============================================================================

set -euo pipefail

REPO_URL="https://github.com/taglia21/Algebraic-Topology-Neural-Net-Strategy.git"
BRANCH="v2-overhaul"
INSTALL_DIR="/opt/atnn"
LOG_DIR="/var/log/atnn"

echo "=============================================="
echo "  ATNN v2 — Droplet Setup"
echo "  Target: $(hostname) ($(uname -m))"
echo "=============================================="

# --- 1. System updates + Docker ---
echo "[1/7] Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq \
    git curl wget unzip \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    docker.io docker-compose-plugin \
    supervisor \
    htop tmux jq

# Enable Docker
systemctl enable docker
systemctl start docker

# --- 2. Clone / update repo ---
echo "[2/7] Setting up repository..."
if [ -d "$INSTALL_DIR" ]; then
    echo "  Updating existing installation..."
    cd "$INSTALL_DIR"
    git fetch origin
    git checkout "$BRANCH"
    git reset --hard "origin/$BRANCH"
else
    echo "  Fresh clone..."
    git clone --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

# --- 3. Python virtual environment ---
echo "[3/7] Setting up Python environment..."
if [ ! -d "$INSTALL_DIR/venv" ]; then
    python3.11 -m venv "$INSTALL_DIR/venv"
fi
source "$INSTALL_DIR/venv/bin/activate"
pip install --upgrade pip -q
pip install -r "$INSTALL_DIR/requirements.txt" -q

# --- 4. Create directories ---
echo "[4/7] Creating data directories..."
mkdir -p "$INSTALL_DIR/data"
mkdir -p "$INSTALL_DIR/models"
mkdir -p "$INSTALL_DIR/logs"
mkdir -p "$LOG_DIR"

# --- 5. Set up .env for IB Gateway ---
echo "[5/7] Configuring IB Gateway credentials..."
if [ ! -f "$INSTALL_DIR/.env" ]; then
    echo "  *** IMPORTANT: You need to set IBKR credentials ***"
    echo "  Edit $INSTALL_DIR/.env with your TWS_USERID and TWS_PASSWORD"
    cp "$INSTALL_DIR/.env.example" "$INSTALL_DIR/.env"
else
    echo "  .env already exists, keeping current credentials"
fi

# --- 6. Create systemd service ---
echo "[6/7] Installing systemd service..."
cat > /etc/systemd/system/atnn-bot.service << 'SERVICEEOF'
[Unit]
Description=ATNN v2 Trading Bot + IB Gateway
Requires=docker.service
After=docker.service network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/opt/atnn
EnvironmentFile=/opt/atnn/.env
ExecStartPre=/usr/bin/docker compose down --remove-orphans
ExecStart=/usr/bin/docker compose up --build
ExecStop=/usr/bin/docker compose down
Restart=always
RestartSec=30
StandardOutput=append:/var/log/atnn/bot.log
StandardError=append:/var/log/atnn/bot-error.log

# Safety: limit resources
MemoryMax=3G
CPUQuota=80%

[Install]
WantedBy=multi-user.target
SERVICEEOF

systemctl daemon-reload
systemctl enable atnn-bot.service

# --- 7. Log rotation ---
echo "[7/7] Setting up log rotation..."
cat > /etc/logrotate.d/atnn << 'LOGEOF'
/var/log/atnn/*.log {
    daily
    rotate 30
    compress
    missingok
    notifempty
    copytruncate
}
LOGEOF

echo ""
echo "=============================================="
echo "  Setup complete!"
echo "=============================================="
echo ""
echo "  Next steps:"
echo "  1. Edit credentials:  nano /opt/atnn/.env"
echo "     Set TWS_USERID and TWS_PASSWORD"
echo ""
echo "  2. Start the bot:     systemctl start atnn-bot"
echo "  3. Check status:      systemctl status atnn-bot"
echo "  4. View logs:         journalctl -u atnn-bot -f"
echo "  5. View trade logs:   tail -f /var/log/atnn/bot.log"
echo ""
echo "  The bot uses config/live.yaml (equities ENABLED, options DORMANT)"
echo "  Kill switch: 5% daily loss flattens, 15% max DD halts"
echo "=============================================="
