#!/usr/bin/env bash
set -euo pipefail

# Real IBKR paper launcher (no simulated fallback).
# Run this on the same machine/network as TWS/IB Gateway.

HOST="${IBKR_HOST:-127.0.0.1}"
PORT="${IBKR_PORT:-}"
CLIENT_ID="${ETF_IBKR_CLIENT_ID:-7}"

ALLOW_GATE_BYPASS="${ETF_ALLOW_GATE_BYPASS:-1}"
FORCE_NOW="${ETF_FORCE_NOW:-0}"
ANYTIME="${ETF_ANYTIME:-0}"
ONCE="${ETF_ONCE:-0}"

if [[ -z "${PORT}" ]]; then
  echo "[etf-paper] IBKR_PORT not set; probing common paper ports on ${HOST} (7497 then 4002)..."
  for candidate in 7497 4002; do
    if python - <<PY >/dev/null 2>&1
import socket
s = socket.socket(); s.settimeout(1.5)
try:
    s.connect(("${HOST}", ${candidate}))
except Exception:
    raise SystemExit(1)
finally:
    s.close()
PY
    then
      PORT="${candidate}"
      break
    fi
  done
fi

if [[ -z "${PORT}" ]]; then
  echo "[etf-paper] ERROR: could not auto-detect a reachable IBKR paper port on ${HOST}."
  echo "[etf-paper] Set IBKR_PORT explicitly (7497 for TWS paper, 4002 for IB Gateway paper)."
  exit 2
fi

export IBKR_HOST="${HOST}"
export IBKR_PORT="${PORT}"
export ETF_IBKR_CLIENT_ID="${CLIENT_ID}"

echo "[etf-paper] Target IBKR endpoint: ${IBKR_HOST}:${IBKR_PORT} (client_id=${ETF_IBKR_CLIENT_ID})"

python - <<'PY'
import os, socket, sys
host = os.environ.get("IBKR_HOST", "127.0.0.1")
port = int(os.environ.get("IBKR_PORT", "7497"))
s = socket.socket()
s.settimeout(2.0)
try:
    s.connect((host, port))
except Exception as exc:
    print(f"[etf-paper] ERROR: IBKR API unreachable at {host}:{port}: {exc}")
    print("[etf-paper] Fix TWS/IB Gateway API + host/port before starting paper run.")
    sys.exit(2)
finally:
    s.close()
print(f"[etf-paper] Connectivity OK: {host}:{port}")
PY

echo "[etf-paper] Running preflight..."
python -m etf.main --mode preflight

cmd=(python -m etf.main --mode run --execute)

if [[ "${ALLOW_GATE_BYPASS}" == "1" ]]; then
  cmd+=(--allow-gate-bypass)
fi
if [[ "${FORCE_NOW}" == "1" ]]; then
  cmd+=(--force)
fi
if [[ "${ANYTIME}" == "1" ]]; then
  cmd+=(--anytime)
fi
if [[ "${ONCE}" == "1" ]]; then
  cmd+=(--once)
fi

echo "[etf-paper] Starting run loop (REAL IBKR paper path)..."
echo "[etf-paper] Command: ${cmd[*]}"
exec "${cmd[@]}"
