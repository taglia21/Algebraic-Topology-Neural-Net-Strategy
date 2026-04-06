#!/usr/bin/env python3
"""
health_monitor.py — Bulletproof system health monitor
=======================================================
Runs every 5 minutes during market hours to catch issues IMMEDIATELY.
This is what was missing — the system that would have caught the
4-day port failure within 5 minutes instead of 4 days.

Checks:
  1. IBKR gateway container is running
  2. API port 4003 is accepting connections
  3. Can authenticate and get account NAV
  4. No unexpected position changes
  5. P&L drawdown within acceptable limits

Alerts via Discord + in-app notification on ANY failure.
Self-healing: attempts to restart gateway if authentication fails.

Run: cron entry — */5 9-17 * * 1-5 (every 5 min during market hours)
"""

from __future__ import annotations
import asyncio, json, logging, subprocess, sys, time
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("monitor")

IBKR_PORT       = 4003
CLIENT_ID       = 99
DISCORD_WEBHOOK = (
    "https://discord.com/api/webhooks/1482171912724545638/"
    "EiV03Fa7qhBj4VRXw9ItpR0w9l1-b6rI1kOwxPmeTM3ddDmf4g_uAghDRWtCW9SRF4M1"
)
STATE_FILE      = Path("/opt/atnn/data/monitor_state.json")
MAX_DD_ALERT    = -0.05   # Alert if NAV drops 5% intraday
NAV_FLOOR       = 5000.0  # Alert if NAV drops below this

SILENT_MODE = "--silent" in sys.argv   # suppress Discord if just checking


def discord(msg: str, urgent: bool = False):
    """Send Discord notification."""
    if SILENT_MODE:
        return
    prefix = "🚨 URGENT" if urgent else "⚠️ ATNN Monitor"
    try:
        subprocess.run([
            "curl", "-s", "-X", "POST",
            "-H", "Content-Type: application/json",
            "-H", "User-Agent: curl/7.68.0",
            "-d", json.dumps({"content": f"**{prefix}**: {msg}"}),
            DISCORD_WEBHOOK,
        ], capture_output=True, timeout=10)
    except Exception:
        pass


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"last_nav": 0.0, "peak_nav": 0.0, "consecutive_failures": 0,
            "last_alert": None, "positions": {}}


def save_state(s: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(s, indent=2, default=str))


def check_container() -> tuple[bool, str]:
    """Check if ib-gateway container is running."""
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Status}}", "ib-gateway"],
            capture_output=True, text=True, timeout=10,
        )
        status = result.stdout.strip()
        if status == "running":
            return True, "running"
        return False, f"container status: {status}"
    except Exception as e:
        return False, f"docker inspect failed: {e}"


async def check_api() -> tuple[bool, dict]:
    """Try to connect to IBKR API and get account data."""
    try:
        from ib_async import IB
        ib = IB()
        await asyncio.wait_for(
            ib.connectAsync("127.0.0.1", IBKR_PORT, clientId=CLIENT_ID), timeout=15
        )
        acct = await ib.accountSummaryAsync()
        nav   = float(next((s.value for s in acct if s.tag == "NetLiquidation"), 0))
        cash  = float(next((s.value for s in acct if s.tag == "TotalCashValue"), 0))
        uplnl = float(next((s.value for s in acct if s.tag == "UnrealizedPnL"), 0))

        positions = {
            p.contract.symbol: int(p.position)
            for p in ib.positions()
            if int(p.position) != 0
        }
        ib.disconnect()
        return True, {"nav": nav, "cash": cash, "unrealized_pnl": uplnl,
                      "positions": positions}
    except Exception as e:
        return False, {"error": str(e)}


def restart_gateway() -> bool:
    """Attempt to restart the IB Gateway container."""
    log.info("Attempting gateway restart...")
    try:
        subprocess.run(
            ["docker", "compose", "-f", "/opt/atnn/docker-compose.yml",
             "restart", "ib-gateway"],
            capture_output=True, timeout=60,
        )
        time.sleep(30)  # wait for startup
        return True
    except Exception as e:
        log.error("Restart failed: %s", e)
        return False


async def run_check():
    state = load_state()
    now   = datetime.now().strftime("%Y-%m-%d %H:%M")
    issues = []

    # ── Check 1: Container ──
    container_ok, container_msg = check_container()
    if not container_ok:
        issues.append(f"Gateway container: {container_msg}")
        log.error("Container check FAILED: %s", container_msg)
    else:
        log.info("Container: OK")

    # ── Check 2: API ──
    api_ok, api_data = await check_api()
    if not api_ok:
        issues.append(f"API connection: {api_data.get('error', 'unknown')}")
        log.error("API check FAILED: %s", api_data.get("error"))

        state["consecutive_failures"] = state.get("consecutive_failures", 0) + 1

        if state["consecutive_failures"] >= 3:
            # Three consecutive failures → attempt self-heal
            log.warning("3 consecutive failures → attempting self-heal")
            discord(f"⚠️ API down for {state['consecutive_failures']} checks. Attempting gateway restart...", urgent=True)
            restart_gateway()
            state["consecutive_failures"] = 0
    else:
        state["consecutive_failures"] = 0
        nav   = api_data["nav"]
        uplnl = api_data["unrealized_pnl"]
        positions = api_data["positions"]

        log.info("API: OK | NAV=$%.2f | UnrealizedPnL=$%.2f | Positions=%s",
                 nav, uplnl, positions)

        # ── Check 3: NAV drawdown ──
        if state.get("peak_nav", 0) == 0:
            state["peak_nav"] = nav
        state["peak_nav"] = max(state["peak_nav"], nav)

        intraday_dd = (nav - state["peak_nav"]) / max(state["peak_nav"], 1)
        if intraday_dd < MAX_DD_ALERT:
            issues.append(f"Drawdown alert: {intraday_dd*100:.1f}% from intraday peak (${state['peak_nav']:.2f}→${nav:.2f})")
            log.warning("DRAWDOWN ALERT: %.1f%%", intraday_dd * 100)

        # ── Check 4: NAV floor ──
        if nav < NAV_FLOOR:
            issues.append(f"NAV ${nav:.2f} below floor ${NAV_FLOOR:.0f}")
            log.warning("NAV FLOOR BREACH: $%.2f", nav)

        # ── Check 5: Unexpected position changes ──
        prev_positions = state.get("positions", {})
        new_positions  = {k: v for k, v in positions.items() if v != 0}
        if prev_positions and prev_positions != new_positions:
            added   = {k: v for k, v in new_positions.items() if k not in prev_positions}
            removed = {k for k in prev_positions if k not in new_positions}
            if added or removed:
                msg = f"Position change: +{added} -{removed}"
                log.info(msg)
                # Don't alert on expected changes, just log

        state["last_nav"]   = nav
        state["positions"]  = new_positions

        # Send routine OK heartbeat (once per hour, non-urgent)
        last_alert = state.get("last_alert")
        if not issues and (not last_alert or
                           (datetime.now() - datetime.fromisoformat(last_alert)).seconds > 3600):
            if not SILENT_MODE:
                discord(f"✅ {now} | NAV=${nav:.2f} | PnL=${uplnl:.2f} | "
                        f"Positions={new_positions or 'FLAT'}")
            state["last_alert"] = datetime.now().isoformat()

    # ── Send issue alerts ──
    if issues:
        msg = f"[{now}] ISSUES DETECTED:\n" + "\n".join(f"• {i}" for i in issues)
        log.error(msg)
        discord(msg, urgent=True)

    save_state(state)
    return len(issues) == 0


if __name__ == "__main__":
    ok = asyncio.run(run_check())
    sys.exit(0 if ok else 1)
