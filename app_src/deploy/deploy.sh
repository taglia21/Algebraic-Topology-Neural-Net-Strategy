#!/usr/bin/env bash
# =============================================================================
# ATNN v2 — Quick Deploy to DigitalOcean Droplet
# =============================================================================
# Pulls latest code on the droplet and restarts the bot.
#
# Usage:
#   bash deploy/deploy.sh                    # Deploy with default settings
#   bash deploy/deploy.sh --dry-run          # Preview without executing
#   DROPLET_IP=x.x.x.x bash deploy/deploy.sh  # Custom droplet IP
# =============================================================================

set -euo pipefail

DROPLET_IP="${DROPLET_IP:-134.209.40.95}"
DROPLET_USER="${DROPLET_USER:-root}"
INSTALL_DIR="/opt/atnn"
BRANCH="v2-overhaul"
DRY_RUN="${1:-}"

echo "=============================================="
echo "  ATNN v2 — Deploying to $DROPLET_IP"
echo "=============================================="

if [ "$DRY_RUN" = "--dry-run" ]; then
    echo "  [DRY RUN] Commands will be printed, not executed."
    echo ""
    echo "  ssh $DROPLET_USER@$DROPLET_IP <<'EOF'"
    echo "    cd $INSTALL_DIR"
    echo "    git fetch origin && git reset --hard origin/$BRANCH"
    echo "    source venv/bin/activate"
    echo "    pip install -r requirements.txt -q"
    echo "    systemctl restart atnn-bot"
    echo "    sleep 5"
    echo "    systemctl status atnn-bot --no-pager"
    echo "  EOF"
    exit 0
fi

echo "[1/4] Pulling latest code on droplet..."
ssh "$DROPLET_USER@$DROPLET_IP" << EOF
    cd $INSTALL_DIR
    git fetch origin
    git reset --hard origin/$BRANCH
    echo "  Commit: \$(git log --oneline -1)"
EOF

echo "[2/4] Updating Python dependencies..."
ssh "$DROPLET_USER@$DROPLET_IP" << EOF
    cd $INSTALL_DIR
    source venv/bin/activate
    pip install -r requirements.txt -q 2>&1 | tail -3
EOF

echo "[3/4] Restarting atnn-bot service..."
ssh "$DROPLET_USER@$DROPLET_IP" << EOF
    systemctl restart atnn-bot
    sleep 5
EOF

echo "[4/4] Verifying..."
ssh "$DROPLET_USER@$DROPLET_IP" << EOF
    echo "--- Service Status ---"
    systemctl status atnn-bot --no-pager -l | head -20
    echo ""
    echo "--- Docker Containers ---"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || echo "  (Docker not running yet)"
    echo ""
    echo "--- Recent Logs ---"
    journalctl -u atnn-bot --no-pager -n 15 2>/dev/null || echo "  (No logs yet)"
EOF

echo ""
echo "=============================================="
echo "  Deployment complete!"
echo "  Monitor: ssh $DROPLET_USER@$DROPLET_IP journalctl -u atnn-bot -f"
echo "=============================================="
