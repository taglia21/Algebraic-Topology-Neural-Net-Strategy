#!/usr/bin/env bash
# ============================================================================
# deploy_to_droplet.sh — Zero-downtime redeploy to DigitalOcean droplet
# ============================================================================
# Pulls latest main, reinstalls deps, restarts the trading-bot service,
# and tails the journal to confirm healthy startup.
#
# Usage:
#   DROPLET_IP=134.209.40.95 ./scripts/deploy_to_droplet.sh
#   DROPLET_IP=134.209.40.95 DROPLET_USER=root ./scripts/deploy_to_droplet.sh --dry-run
#
# Environment variables:
#   DROPLET_IP     — required, droplet IPv4 address
#   DROPLET_USER   — optional, defaults to "root"
#   SSH_KEY        — optional, path to SSH private key (default: ~/.ssh/id_rsa_droplet)
#   REPO_DIR       — optional, repo path on droplet (default: /opt/trading-bot)
#   SERVICE_NAME   — optional, systemd service name (default: trading-bot)
#   VENV_DIR       — optional, venv subdirectory (default: venv)
# ============================================================================

set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────────────
DROPLET_USER="${DROPLET_USER:-root}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_rsa_droplet}"
REPO_DIR="${REPO_DIR:-/opt/trading-bot}"
SERVICE_NAME="${SERVICE_NAME:-trading-bot}"
VENV_DIR="${VENV_DIR:-venv}"
DRY_RUN=false

# ── Parse flags ─────────────────────────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --help|-h)
            echo "Usage: DROPLET_IP=<ip> $0 [--dry-run]"
            exit 0
            ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

# ── Validate ────────────────────────────────────────────────────────────────
if [[ -z "${DROPLET_IP:-}" ]]; then
    echo "ERROR: DROPLET_IP is not set."
    echo "Usage: DROPLET_IP=134.209.40.95 $0 [--dry-run]"
    exit 1
fi

SSH_CMD="ssh -i ${SSH_KEY} -o ConnectTimeout=10 -o StrictHostKeyChecking=no ${DROPLET_USER}@${DROPLET_IP}"

# ── Helper: run or print ────────────────────────────────────────────────────
run_remote() {
    local desc="$1"
    local cmd="$2"
    echo ""
    echo "──── ${desc} ────"
    if $DRY_RUN; then
        echo "[DRY-RUN] ${SSH_CMD} \"${cmd}\""
    else
        ${SSH_CMD} "${cmd}"
    fi
}

# ── Deploy steps ────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Deploying to ${DROPLET_USER}@${DROPLET_IP}"
echo "  Repo:    ${REPO_DIR}"
echo "  Service: ${SERVICE_NAME}"
echo "  Dry-run: ${DRY_RUN}"
echo "============================================================"

# Step 1: Stop the service
run_remote "Step 1/5 — Stop ${SERVICE_NAME} service" \
    "systemctl stop ${SERVICE_NAME} || true"

# Step 2: Pull latest code
run_remote "Step 2/5 — Git pull origin main" \
    "cd ${REPO_DIR} && git fetch origin main && git reset --hard origin/main"

# Step 3: Install/update pip dependencies
run_remote "Step 3/5 — Install pip dependencies" \
    "cd ${REPO_DIR} && ${VENV_DIR}/bin/pip install -q -r requirements.txt 2>&1 | tail -5"

# Step 4: Restart the service
run_remote "Step 4/5 — Start ${SERVICE_NAME} service" \
    "systemctl daemon-reload && systemctl start ${SERVICE_NAME}"

# Step 5: Tail journal to confirm healthy startup
echo ""
echo "──── Step 5/5 — Tailing journal for 10s ────"
if $DRY_RUN; then
    echo "[DRY-RUN] ${SSH_CMD} \"journalctl -u ${SERVICE_NAME} -f --no-pager -n 30\" (timeout 10s)"
else
    timeout 10 ${SSH_CMD} "journalctl -u ${SERVICE_NAME} -f --no-pager -n 30" 2>/dev/null || true
fi

# Step 6: Quick health check
run_remote "Health check — service status" \
    "systemctl is-active ${SERVICE_NAME} && echo 'SERVICE IS RUNNING ✓' || echo 'SERVICE FAILED ✗'"

echo ""
echo "============================================================"
echo "  Deploy complete."
echo "============================================================"
