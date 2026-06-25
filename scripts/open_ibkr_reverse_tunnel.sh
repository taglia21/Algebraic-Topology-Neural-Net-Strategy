#!/usr/bin/env bash
set -euo pipefail

# Create a reverse SSH tunnel so a cloud-hosted bot can reach a local
# IBKR TWS/Gateway paper API endpoint.
#
# Run THIS script on the machine where TWS/Gateway is running.
#
# Usage:
#   scripts/open_ibkr_reverse_tunnel.sh <ssh_target> [remote_port] [local_ibkr_port]
#
# Example:
#   scripts/open_ibkr_reverse_tunnel.sh ubuntu@your-droplet 4002 7497
#
# Effect:
#   droplet 127.0.0.1:4002 -> your-local 127.0.0.1:7497

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <ssh_target> [remote_port] [local_ibkr_port]"
  echo "Example: $0 ubuntu@your-droplet 4002 7497"
  exit 2
fi

SSH_TARGET="$1"
REMOTE_PORT="${2:-4002}"
LOCAL_IBKR_PORT="${3:-7497}"

echo "[ibkr-tunnel] Opening reverse tunnel: ${SSH_TARGET} 127.0.0.1:${REMOTE_PORT} -> local 127.0.0.1:${LOCAL_IBKR_PORT}"

exec ssh -NT \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -R "127.0.0.1:${REMOTE_PORT}:127.0.0.1:${LOCAL_IBKR_PORT}" \
  "${SSH_TARGET}"
