#!/usr/bin/env bash
# ============================================================================
# gw-debug.sh — capture the headless IB Gateway screen for troubleshooting.
# ============================================================================
# The Gateway GUI runs on a virtual display (Xvfb :1) that nobody can see. When
# login hangs, this grabs a PNG of that hidden screen so we can tell whether
# it's a 2FA prompt, a wrong-password dialog, a version notice, or a freeze.
#
# Usage (on the droplet, as root):
#     bash /root/etf-engine/etf/deploy/gw-debug.sh
#
# Then, on your normal computer, open:
#     http://<DROPLET_IP>:8080/gw-screen.png
# Take a screenshot of that image and share it. Press Ctrl+C here when done.
# ============================================================================
set -uo pipefail

DISPLAY_NUM="${XVFB_DISPLAY:-:1}"
OUT="/root/gw-screen.png"

# Ensure the screenshot tool is present.
if ! command -v import >/dev/null 2>&1; then
    echo "[gw-debug] Installing screenshot tool (imagemagick)..."
    apt-get update -qq >/dev/null 2>&1 || true
    apt-get install -y -qq imagemagick >/dev/null 2>&1 || true
fi

# Capture the virtual display's root window.
if DISPLAY="$DISPLAY_NUM" import -window root "$OUT" 2>/dev/null; then
    echo "[gw-debug] Screenshot saved: $OUT"
else
    echo "[gw-debug] Screenshot FAILED — is ibc-gateway running? Try:"
    echo "           systemctl status ibc-gateway --no-pager"
    exit 1
fi

IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo ""
echo "[gw-debug] Now open this in a browser on your normal computer:"
echo "           http://${IP:-<DROPLET_IP>}:8080/gw-screen.png"
echo "[gw-debug] Press Ctrl+C to stop when you're done viewing."
echo ""

cd /root && exec python3 -m http.server 8080 --bind 0.0.0.0
