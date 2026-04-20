#!/bin/bash
# Minimal bootstrap - installs and clones only
set -e

echo "[1/5] apt update..."
apt-get update -qq

echo "[2/5] Installing packages..."
apt-get install -y -qq git docker.io docker-compose-plugin python3-venv python3-dev python3-pip curl 2>&1 | tail -3

echo "[3/5] Starting Docker..."
systemctl enable docker
systemctl start docker

echo "[4/5] Cloning repo..."
rm -rf /opt/atnn
git clone -b v2-overhaul https://github.com/taglia21/Algebraic-Topology-Neural-Net-Strategy.git /opt/atnn
cd /opt/atnn
echo "Commit: $(git log --oneline -1)"

echo "[5/5] Python setup..."
python3 -m venv /opt/atnn/venv
. /opt/atnn/venv/bin/activate
pip install -U pip -q
pip install -r /opt/atnn/requirements.txt -q 2>&1 | tail -5

mkdir -p /opt/atnn/data /opt/atnn/models /opt/atnn/logs /var/log/atnn
cp -n /opt/atnn/.env.example /opt/atnn/.env 2>/dev/null || true

# Systemd service
cat > /etc/systemd/system/atnn-bot.service << 'SVC'
[Unit]
Description=ATNN v2 Bot
After=docker.service
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
[Install]
WantedBy=multi-user.target
SVC

systemctl daemon-reload
systemctl enable atnn-bot

# Stop old service
systemctl stop trading-bot 2>/dev/null || true
systemctl disable trading-bot 2>/dev/null || true

echo ""
echo "=== DONE ==="
echo "Commit: $(cd /opt/atnn && git log --oneline -1)"
echo "Config: $(ls /opt/atnn/config/live.yaml 2>&1)"
echo "Docker: $(docker --version)"
echo "Service: $(systemctl is-enabled atnn-bot)"
