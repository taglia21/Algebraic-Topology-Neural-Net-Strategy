#!/usr/bin/env bash
# ===========================================================================
# deploy_ibkr.sh — Automated IBKR Trading Bot Deployment
# ===========================================================================
# Usage:
#   chmod +x scripts/deploy_ibkr.sh
#   ./scripts/deploy_ibkr.sh
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "================================================="
echo "  IBKR Trading Bot Deployment"
echo "  $(date)"
echo "================================================="

# ---------------------------------------------------------------------------
# 1. Validate .env has required IBKR credentials
# ---------------------------------------------------------------------------
ENV_FILE="$PROJECT_ROOT/.env"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: .env file not found at $ENV_FILE"
    echo "Copy deploy/.env.template to .env and fill in credentials."
    exit 1
fi

REQUIRED_VARS=("TWS_USERID" "TWS_PASSWORD" "IBKR_ACCOUNT")
MISSING=()

for var in "${REQUIRED_VARS[@]}"; do
    if ! grep -q "^${var}=" "$ENV_FILE" 2>/dev/null; then
        MISSING+=("$var")
    fi
    # Also check the var is not empty
    val=$(grep "^${var}=" "$ENV_FILE" 2>/dev/null | head -1 | cut -d'=' -f2-)
    if [[ -z "$val" ]]; then
        MISSING+=("${var} (empty)")
    fi
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "ERROR: Missing or empty required env vars in .env:"
    for m in "${MISSING[@]}"; do
        echo "  - $m"
    done
    exit 1
fi

echo "✅ .env validated — all required IBKR credentials present"

# ---------------------------------------------------------------------------
# 2. Git pull latest
# ---------------------------------------------------------------------------
echo ""
echo "--- Pulling latest from origin/main ---"
cd "$PROJECT_ROOT"
git pull origin main || echo "⚠️  git pull failed (offline or no remote?), continuing..."

# ---------------------------------------------------------------------------
# 3. Ensure directories exist
# ---------------------------------------------------------------------------
mkdir -p "$PROJECT_ROOT/logs" "$PROJECT_ROOT/state" "$PROJECT_ROOT/models"

# ---------------------------------------------------------------------------
# 4. Build & deploy with Docker Compose
# ---------------------------------------------------------------------------
COMPOSE_FILE="$PROJECT_ROOT/deploy/docker-compose.ibkr.yml"

if [[ ! -f "$COMPOSE_FILE" ]]; then
    echo "ERROR: Docker Compose file not found at $COMPOSE_FILE"
    exit 1
fi

echo ""
echo "--- Pulling base images ---"
docker compose -f "$COMPOSE_FILE" pull || true

echo ""
echo "--- Building and starting services ---"
docker compose -f "$COMPOSE_FILE" up -d --build

echo ""
echo "--- Waiting 10s for services to start ---"
sleep 10

# ---------------------------------------------------------------------------
# 5. Show logs
# ---------------------------------------------------------------------------
echo ""
echo "--- Recent trading-bot logs ---"
docker compose -f "$COMPOSE_FILE" logs --tail=50 trading-bot

# ---------------------------------------------------------------------------
# 6. Health check
# ---------------------------------------------------------------------------
echo ""
echo "--- Health check ---"
if curl -sf http://localhost:8080/health > /dev/null 2>&1; then
    echo "✅ Health endpoint responsive"
    curl -s http://localhost:8080/health | python3 -m json.tool 2>/dev/null || true
else
    echo "⚠️  Health endpoint not yet responding (may still be starting)"
fi

echo ""
echo "================================================="
echo "  Deployment complete!"
echo "  Monitor:  docker compose -f deploy/docker-compose.ibkr.yml logs -f"
echo "  Stop:     docker compose -f deploy/docker-compose.ibkr.yml down"
echo "================================================="
