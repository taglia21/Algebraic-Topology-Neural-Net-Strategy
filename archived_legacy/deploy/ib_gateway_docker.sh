#!/usr/bin/env bash
# ============================================================================
# IB Gateway Docker Deployment
# ============================================================================
# Pulls and runs gnzsnz/ib-gateway:stable, reads credentials from .env,
# exposes API ports 4001 (live) and 4002 (paper), and installs a watchdog
# cron job that restarts the container if it becomes unhealthy.
#
# Usage:
#   chmod +x deploy/ib_gateway_docker.sh
#   ./deploy/ib_gateway_docker.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="${PROJECT_ROOT}/.env"
CONTAINER_NAME="ib-gateway"
IMAGE="gnzsnz/ib-gateway:stable"

# ------------------------------------------------------------------
# 1. Load .env
# ------------------------------------------------------------------
if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: $ENV_FILE not found. Copy deploy/.env.example → .env and fill in values."
    exit 1
fi

set -a
# shellcheck source=/dev/null
source "$ENV_FILE"
set +a

: "${TWS_USERID:?TWS_USERID not set in .env}"
: "${TWS_PASSWORD:?TWS_PASSWORD not set in .env}"

echo "=== IB Gateway Docker Deployment ==="
echo "  Image      : $IMAGE"
echo "  Container  : $CONTAINER_NAME"
echo "  TWS User   : $TWS_USERID"
echo ""

# ------------------------------------------------------------------
# 2. Pull latest image
# ------------------------------------------------------------------
echo "→ Pulling $IMAGE …"
docker pull "$IMAGE"

# ------------------------------------------------------------------
# 3. Stop existing container (if any)
# ------------------------------------------------------------------
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "→ Removing existing container '$CONTAINER_NAME' …"
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
fi

# ------------------------------------------------------------------
# 4. Run container
# ------------------------------------------------------------------
echo "→ Starting IB Gateway container …"
docker run -d \
    --name "$CONTAINER_NAME" \
    --restart unless-stopped \
    -p 4001:4001 \
    -p 4002:4002 \
    -e TWS_USERID="$TWS_USERID" \
    -e TWS_PASSWORD="$TWS_PASSWORD" \
    -e TRADING_MODE="${TRADING_MODE:-paper}" \
    -e IBC_INI__auto_logon="yes" \
    -e IBC_INI__ReadOnlyApi="no" \
    -e IBC_INI__AcceptIncomingConnectionAction="accept" \
    -e IBC_INI__ExistingSessionDetectedAction="primary" \
    -e IBC_INI__AcceptNonBrokerageAccountWarning="yes" \
    "$IMAGE"

echo "→ Container '$CONTAINER_NAME' started."

# ------------------------------------------------------------------
# 5. Install watchdog cron (every 5 minutes)
# ------------------------------------------------------------------
WATCHDOG_SCRIPT="/usr/local/bin/ib_gateway_watchdog.sh"

cat > /tmp/ib_gateway_watchdog.sh << 'WATCHDOG'
#!/usr/bin/env bash
# IB Gateway watchdog — restarts container if unhealthy or stopped
CONTAINER="ib-gateway"
STATUS=$(docker inspect --format='{{.State.Health.Status}}' "$CONTAINER" 2>/dev/null || echo "missing")
if [[ "$STATUS" != "healthy" && "$STATUS" != "" ]]; then
    logger -t ib-watchdog "IB Gateway status=$STATUS — restarting"
    docker restart "$CONTAINER" 2>/dev/null || docker start "$CONTAINER" 2>/dev/null
fi
WATCHDOG

# Only install cron if running as root / with sudo
if [[ $EUID -eq 0 ]] || command -v sudo &>/dev/null; then
    SUDO=""
    [[ $EUID -ne 0 ]] && SUDO="sudo"
    $SUDO cp /tmp/ib_gateway_watchdog.sh "$WATCHDOG_SCRIPT"
    $SUDO chmod +x "$WATCHDOG_SCRIPT"

    # Add cron entry if not present
    CRON_LINE="*/5 * * * * $WATCHDOG_SCRIPT"
    ( crontab -l 2>/dev/null | grep -v "ib_gateway_watchdog" ; echo "$CRON_LINE" ) | crontab -
    echo "→ Watchdog cron installed (every 5 min)."
else
    echo "⚠  Skipped watchdog cron (no root/sudo). Install manually:"
    echo "   */5 * * * * $WATCHDOG_SCRIPT"
fi

rm -f /tmp/ib_gateway_watchdog.sh

# ------------------------------------------------------------------
# 6. Connection test instructions
# ------------------------------------------------------------------
echo ""
echo "============================================"
echo "  IB Gateway is starting up …"
echo "  Allow 30-60 seconds for IBC auto-login."
echo ""
echo "  Connection test (paper):"
echo "    python -c \""
echo "from ib_insync import IB"
echo "ib = IB()"
echo "ib.connect('127.0.0.1', 4002, clientId=99)"
echo "print('Connected:', ib.isConnected())"
echo "print('Account:', ib.managedAccounts())"
echo "ib.disconnect()"
echo "\""
echo ""
echo "  Connection test (live):"
echo "    Replace port 4002 with 4001."
echo ""
echo "  Logs:"
echo "    docker logs -f $CONTAINER_NAME"
echo "============================================"
