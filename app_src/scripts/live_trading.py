#!/usr/bin/env python3
"""
live_trading.py — ATNN Live Trading Engine (Evidence-Based)
============================================================
Two-strategy system:

1. Dual Momentum GEM: monthly allocation between SPY / EFA / IEF
   - Primary capital allocation based on 12-month relative + absolute momentum
   - Rebalances on last trading day of each month

2. SPY IBS Mean Reversion: tactical overlay
   - Enters SPY when it's extremely oversold (IBS < 0.30 AND price dips)
   - Exits on next-day recovery above previous high
   - Overrides GEM allocation when signal fires

Execution: runs daily at 9:35 AM ET (after 2FA cron at 9:25 AM)
           checks signals using end-of-day data from previous session
           places market orders at open if signal changed

Run: python3 /opt/atnn/scripts/live_trading.py
"""

from __future__ import annotations
import asyncio, json, logging, sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import yfinance as yf

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("live")

IBKR_HOST   = "127.0.0.1"
IBKR_PORT   = 4003
CLIENT_ID   = 20
ACCOUNT     = "U22452226"
STATE_FILE  = Path("/opt/atnn/data/live_state.json")

GEM_TICKERS = ["SPY", "EFA", "IEF"]
TBILL_YIELD = 0.04  # approximate annual T-bill rate

# ─── State ────────────────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"gem_holding": None, "ibs_active": False, "last_gem_check": None,
            "day": 0}

def save_state(s: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(s, indent=2, default=str))

# ─── Market Data ──────────────────────────────────────────────────────────────

def fetch_prices() -> dict[str, pd.DataFrame]:
    data = yf.download(GEM_TICKERS, period="400d", interval="1d",
                       group_by="ticker", auto_adjust=True,
                       threads=True, progress=False)
    out = {}
    if isinstance(data.columns, pd.MultiIndex):
        for t in GEM_TICKERS:
            try:
                df = data[t].dropna(how="all")
                if len(df) > 50:
                    out[t] = df
            except KeyError:
                pass
    return out

# ─── Strategy Signals ─────────────────────────────────────────────────────────

def gem_target(prices: dict) -> str:
    """Returns 'SPY', 'EFA', or 'IEF' based on 12-month momentum."""
    closes = {t: prices[t]["Close"] for t in GEM_TICKERS if t in prices}
    if "SPY" not in closes:
        return "IEF"

    def ret12(s: pd.Series) -> float:
        if len(s) < 252:
            return 0.0
        return float(s.iloc[-1] / s.iloc[-252] - 1)

    spy_r = ret12(closes["SPY"])
    efa_r = ret12(closes.get("EFA", pd.Series()))
    tbill = TBILL_YIELD

    spy_abs = spy_r > tbill
    efa_abs = efa_r > tbill

    if not spy_abs and not efa_abs:
        return "IEF"
    if spy_r >= efa_r:
        return "SPY" if spy_abs else "IEF"
    return "EFA" if efa_abs else "IEF"


def ibs_signal(spy_df: pd.DataFrame) -> tuple[bool, float, float]:
    """Returns (should_be_long, ibs_value, entry_threshold)."""
    if len(spy_df) < 30:
        return False, 0.5, 0.0

    h, l, c = spy_df["High"], spy_df["Low"], spy_df["Close"]

    # IBS of most recent closed session
    ibs = float((c.iloc[-1] - l.iloc[-1]) / max(h.iloc[-1] - l.iloc[-1], 0.001))

    # Entry threshold: 10-day high – 2.5 × 25-day avg range
    avg_rng   = (h - l).rolling(25).mean().iloc[-1]
    roll_high = h.rolling(10).max().iloc[-1]
    threshold = float(roll_high - 2.5 * avg_rng)

    entry = (c.iloc[-1] < threshold) and (ibs < 0.30)

    # Exit: yesterday's close > previous session's high
    if len(spy_df) >= 2:
        prev_high  = float(h.iloc[-2])
        last_close = float(c.iloc[-1])
        exit_now   = last_close > prev_high
    else:
        exit_now = False

    return entry, ibs, threshold, exit_now


def is_month_end() -> bool:
    """True if today is the last trading day of the month."""
    today = datetime.now().date()
    nxt   = today + timedelta(days=1)
    while nxt.weekday() >= 5:  # skip weekends
        nxt += timedelta(days=1)
    return nxt.month != today.month

# ─── IBKR Execution ───────────────────────────────────────────────────────────

async def get_nav_and_positions(ib) -> tuple[float, dict]:
    acct = await ib.accountSummaryAsync()
    nav  = float(next((s.value for s in acct if s.tag == "NetLiquidation"), 0))
    pos  = {p.contract.symbol: int(p.position) for p in ib.positions()
            if int(p.position) != 0}
    return nav, pos


async def execute_order(ib, symbol: str, action: str, qty: int,
                        dry_run: bool = False) -> bool:
    if qty <= 0:
        return True
    from ib_async import Stock, MarketOrder

    contract = Stock(symbol, "SMART", "USD")
    order    = MarketOrder(action, qty)
    log.info("[%s] %s %d %s", "DRY" if dry_run else "LIVE", action, qty, symbol)
    if dry_run:
        return True

    trade = ib.placeOrder(contract, order)
    for _ in range(60):
        await asyncio.sleep(0.5)
        if trade.isDone():
            filled = trade.orderStatus.filled
            price  = trade.orderStatus.avgFillPrice
            log.info("Filled: %d @ $%.2f", int(filled), price)
            return trade.orderStatus.status == "Filled"
    log.warning("Order timeout for %s", symbol)
    return False

# ─── Main Cycle ───────────────────────────────────────────────────────────────

async def run(dry_run: bool = False):
    state = load_state()
    state["day"] = state.get("day", 0) + 1
    log.info("=== Day %d  %s ===", state["day"], datetime.now().strftime("%Y-%m-%d %H:%M"))

    # ── 1. Prices ──
    log.info("Fetching prices...")
    prices = fetch_prices()
    spy_df = prices.get("SPY")
    if spy_df is None:
        log.error("SPY data unavailable. Aborting.")
        save_state(state)
        return

    spy_close  = float(spy_df["Close"].iloc[-1])
    spy_prev_h = float(spy_df["High"].iloc[-2]) if len(spy_df) >= 2 else 0

    # ── 2. Signals ──
    gem_tgt = gem_target(prices)
    entry, ibs_val, ibs_thr, exit_now = ibs_signal(spy_df)

    log.info("GEM target:    %s", gem_tgt)
    log.info("IBS value:     %.3f (entry <0.30, thr $%.2f)", ibs_val, ibs_thr)
    log.info("IBS signal:    %s", "ENTRY" if entry else ("EXIT" if exit_now else "hold"))
    log.info("IBS active:    %s", state.get("ibs_active", False))

    # ── 3. Connect ──
    from ib_async import IB
    ib = IB()
    try:
        await asyncio.wait_for(
            ib.connectAsync(IBKR_HOST, IBKR_PORT, clientId=CLIENT_ID), timeout=20
        )
    except Exception as e:
        log.error("IBKR connect failed: %s", e)
        save_state(state)
        return

    nav, positions = await get_nav_and_positions(ib)
    log.info("NAV: $%.2f | Positions: %s", nav, positions)

    # ── 4. IBS overrides GEM allocation ──
    ibs_was_active = state.get("ibs_active", False)

    if entry and not ibs_was_active:
        # New IBS entry: go fully into SPY regardless of GEM
        log.info("IBS ENTRY: Close ($%.2f) < Threshold ($%.2f) AND IBS %.3f < 0.30",
                 spy_close, ibs_thr, ibs_val)
        target_ticker = "SPY"
        target_source = "IBS"
        state["ibs_active"] = True

    elif ibs_was_active and exit_now:
        # IBS exit: SPY recovered above prev high
        log.info("IBS EXIT: SPY $%.2f > prev high $%.2f", spy_close, spy_prev_h)
        state["ibs_active"] = False
        target_ticker = gem_tgt  # fall back to GEM after exit
        target_source = "GEM (post-IBS)"

    elif ibs_was_active and not exit_now:
        # Still in IBS trade — hold SPY
        target_ticker = "SPY"
        target_source = "IBS (holding)"

    else:
        # Normal GEM operation — check if monthly rebalance needed
        if is_month_end() or state.get("gem_holding") != gem_tgt:
            log.info("GEM rebalance: %s → %s",
                     state.get("gem_holding", "none"), gem_tgt)
            target_ticker = gem_tgt
            target_source = "GEM (rebalance)"
            state["gem_holding"] = gem_tgt
            state["last_gem_check"] = str(datetime.now().date())
        else:
            log.info("GEM: no change (%s), holding", gem_tgt)
            target_ticker = None  # no action needed
            target_source = "GEM (no change)"

    # ── 5. Execute ──
    if target_ticker:
        # Close everything not matching target
        for sym, qty in list(positions.items()):
            if sym != target_ticker and qty != 0:
                action = "SELL" if qty > 0 else "BUY"
                await execute_order(ib, sym, action, abs(qty), dry_run=dry_run)

        # Open target position if not already held
        target_qty_held = positions.get(target_ticker, 0)
        if target_qty_held <= 0:
            # Get target price
            tgt_close = float(prices[target_ticker]["Close"].iloc[-1]) if target_ticker in prices else 0
            if tgt_close > 0:
                # Use 95% of NAV (leave buffer for commissions)
                alloc_qty = int((nav * 0.95) / tgt_close)
                if alloc_qty > 0:
                    log.info("Opening %s: %d shares @ ~$%.2f (source: %s)",
                             target_ticker, alloc_qty, tgt_close, target_source)
                    await execute_order(ib, target_ticker, "BUY", alloc_qty,
                                        dry_run=dry_run)
                    state["gem_holding"] = target_ticker
        else:
            log.info("Already holding %d %s — no action", target_qty_held, target_ticker)

    # ── 6. Summary ──
    nav2, pos2 = await get_nav_and_positions(ib)
    log.info("Post-cycle: NAV=$%.2f | Positions: %s", nav2, pos2)

    # Send Discord notification
    try:
        import subprocess
        holding_str = ", ".join(f"{s}:{q}" for s, q in pos2.items()) or "FLAT"
        msg = (f"ATNN Day {state['day']}: NAV=${nav2:.2f} | "
               f"Holding: {holding_str} | "
               f"GEM={gem_tgt} IBS={'ACTIVE' if state.get('ibs_active') else 'flat'} "
               f"(IBS={ibs_val:.3f})")
        subprocess.run([
            "curl", "-s", "-X", "POST",
            "-H", "Content-Type: application/json",
            "-H", "User-Agent: curl/7.68.0",
            "-d", json.dumps({"content": msg}),
            "https://discord.com/api/webhooks/1482171912724545638/"
            "EiV03Fa7qhBj4VRXw9ItpR0w9l1-b6rI1kOwxPmeTM3ddDmf4g_uAghDRWtCW9SRF4M1"
        ], capture_output=True)
    except Exception:
        pass

    ib.disconnect()
    save_state(state)
    log.info("Done. State saved.")


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    if dry:
        log.info("DRY RUN MODE — no real orders")
    asyncio.run(run(dry_run=dry))
