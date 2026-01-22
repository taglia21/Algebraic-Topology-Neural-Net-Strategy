#!/bin/bash
#===============================================================================
# V2.1 Emergency Rollback Script
#===============================================================================
# Instantly reverts to V1.3-only mode. Target: <60 seconds execution.
#
# Usage: ./scripts/emergency_rollback.sh [--reason "description"]
#===============================================================================

set -euo pipefail

DROPLET_IP="134.209.40.95"
SSH_KEY="$HOME/.ssh/id_rsa_droplet"
SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=no -o ConnectTimeout=10"
REMOTE_DIR="/opt/Algebraic-Topology-Neural-Net-Strategy"
REASON="${2:-Manual rollback}"

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

log() { echo -e "[$(date '+%H:%M:%S')] $*"; }
remote() { ssh $SSH_OPTS root@${DROPLET_IP} "$@"; }

log "${RED}🚨 EMERGENCY ROLLBACK INITIATED${NC}"
START=$(date +%s)

# Disable V2.1
remote "sed -i 's/V21_ENABLED=true/V21_ENABLED=false/' $REMOTE_DIR/.env 2>/dev/null || true"
remote "sed -i 's/\"V21_ENABLED\": true/\"V21_ENABLED\": false/' $REMOTE_DIR/config/v21_config.json 2>/dev/null || true"

# Log incident
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
remote "echo '$TIMESTAMP ROLLBACK: $REASON' >> $REMOTE_DIR/logs/v21_incidents.log"

# Send Discord alert
WEBHOOK="${DISCORD_WEBHOOK_URL:-}"
if [[ -n "$WEBHOOK" ]]; then
    curl -s -H "Content-Type: application/json" \
        -d "{\"embeds\":[{\"title\":\"🚨 V2.1 ROLLBACK\",\"description\":\"Reverted to V1.3-only mode\\nReason: $REASON\\nTime: $TIMESTAMP\",\"color\":15158332}]}" \
        "$WEBHOOK" > /dev/null 2>&1 || true
fi

END=$(date +%s)
DURATION=$((END - START))

log "${GREEN}✓ Rollback complete in ${DURATION}s${NC}"
log "V1.3 is now the only active system"
log "Incident logged to: $REMOTE_DIR/logs/v21_incidents.log"
