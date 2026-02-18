#!/usr/bin/env python3
"""
run_bot.py — SINGLE ENTRY POINT for the trading system
========================================================

This is the ONE script you should run.  It:

  1. Kills any other trading bot processes on this machine.
  2. Acquires the trading-system file lock so no other bot can start.
  3. Pre-flight checks: verifies Alpaca connection, prints account info.
  4. Delegates to `run_v28_production.py` (async equity + options engine).

Usage:
    python run_bot.py                   # default = paper mode
    python run_bot.py --mode live       # live mode (use with caution!)

Why?  The previous setup let continuous_trader, smart_trader, auto_trader,
aggressive_trader, and paper_trading_runner all run simultaneously on the
same Alpaca account, issuing conflicting orders and causing losses.
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("run_bot")

# ── Names of scripts that must NOT be running concurrently ───────────────

COMPETITOR_SCRIPTS = [
    "continuous_trader.py",
    "smart_trader.py",
    "auto_trader.py",
    "aggressive_trader.py",
    "paper_trading_runner.py",
    "paper_trading_bot.py",
    "live_trader.py",
    "continuous_tradier.py",
    "production_launcher.py",
]


def kill_competing_bots() -> int:
    """Find and terminate any competing trading bot processes.  Returns count killed."""
    killed = 0
    try:
        import psutil
    except ImportError:
        # Fallback: use `pkill -f`
        for script in COMPETITOR_SCRIPTS:
            ret = subprocess.call(
                ["pkill", "-f", script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if ret == 0:
                killed += 1
                logger.info(f"  ☠  Killed process running {script}")
        return killed

    my_pid = os.getpid()
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = " ".join(proc.info["cmdline"] or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if proc.info["pid"] == my_pid:
            continue
        for script in COMPETITOR_SCRIPTS:
            if script in cmdline:
                logger.info(f"  ☠  Killing PID {proc.info['pid']} ({script})")
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
                killed += 1
                break
    return killed


def preflight_check() -> bool:
    """Verify Alpaca API connection and print account snapshot."""
    api_key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
    api_secret = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
    if not api_key or not api_secret:
        logger.error("❌ No Alpaca API keys found in environment.  Set ALPACA_API_KEY / ALPACA_SECRET_KEY.")
        return False

    try:
        import alpaca_trade_api as tradeapi
        api = tradeapi.REST(api_key, api_secret, "https://paper-api.alpaca.markets", api_version="v2")
        acct = api.get_account()
        equity = float(acct.equity)
        cash = float(acct.cash)
        buying_power = float(acct.buying_power)
        positions = api.list_positions()
        logger.info("─" * 60)
        logger.info(f"  Account:       {acct.account_number}")
        logger.info(f"  Equity:        ${equity:,.2f}")
        logger.info(f"  Cash:          ${cash:,.2f}")
        logger.info(f"  Buying power:  ${buying_power:,.2f}")
        logger.info(f"  Positions:     {len(positions)}")
        logger.info("─" * 60)
        return True
    except Exception as exc:
        logger.error(f"❌ Alpaca pre-flight failed: {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Unified trading bot launcher")
    parser.add_argument("--mode", choices=["paper", "live"], default="paper")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  🤖  UNIFIED TRADING BOT LAUNCHER")
    logger.info(f"  Mode: {args.mode.upper()}")
    logger.info("=" * 60)

    # Step 1: Kill competing bots
    logger.info("\n[1/4] Killing competing bot processes…")
    killed = kill_competing_bots()
    if killed:
        logger.info(f"  Terminated {killed} competing process(es).")
    else:
        logger.info("  No competing bots found.")

    # Step 2: Acquire exclusive lock
    logger.info("\n[2/4] Acquiring trading lock…")
    try:
        from src.risk.process_lock import acquire_trading_lock
        if not acquire_trading_lock("run_bot"):
            logger.error("❌ Could not acquire trading lock. Another bot may be running.")
            sys.exit(1)
        logger.info("  Lock acquired.")
    except ImportError:
        logger.warning("  ⚠️ process_lock not available — skipping.")

    # Step 3: Pre-flight checks
    logger.info("\n[3/4] Running pre-flight checks…")
    if not preflight_check():
        logger.error("Pre-flight failed.  Aborting.")
        sys.exit(1)

    # Step 4: Launch v28 production engine
    logger.info(f"\n[4/4] Launching V28 production engine (mode={args.mode})…\n")

    # Forward to run_v28_production.py so it inherits our env / lock
    v28_script = PROJECT_ROOT / "run_v28_production.py"
    if not v28_script.exists():
        logger.error(f"❌ {v28_script} not found!")
        sys.exit(1)

    try:
        os.execv(
            sys.executable,
            [sys.executable, str(v28_script), f"--mode={args.mode}"],
        )
    except Exception as exc:
        logger.error(f"Failed to exec v28 engine: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
