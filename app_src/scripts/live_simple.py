#!/usr/bin/env python3
"""
live_simple.py — Simple Trend + IBS System
============================================
The lesson: complexity killed returns. This is the stripped-down version.

RULES:
  1. Be long when SPY is in an uptrend (above 50d MA AND 200d MA)
  2. Enter on IBS oversold signals (IBS < 0.20) for best timing
  3. Enter on trend confirmation if flat for too long in a clear uptrend
  4. Trail stop 80 points below highest close since entry
  5. Exit if SPY closes below 50d MA (trend break)
  6. Circuit breaker: 2% daily loss = flatten

That's it. No TDA. No TCN. No composite scorer. No intraday scalping.
Just trend + mean-reversion entry timing + trailing stop.

Runs twice daily:
  --morning (9:35 AM): Trail stop, check trend break, check exits
  --close   (3:50 PM): IBS entry check
"""

from __future__ import annotations
import asyncio, json, logging, os, sys, tempfile, subprocess
from datetime import datetime, date, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path('/opt/atnn/app_src')))

import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("simple")

IBKR_HOST  = "127.0.0.1"
IBKR_PORT  = 4003
CLIENT_ID  = 40

TRAIL_PTS       = 80       # trail stop 80 SPX points below highest close
IBS_ENTRY       = 0.20     # IBS < 0.20 for entry
MAX_FLAT_DAYS   = 10       # if flat for 10 days in uptrend, enter anyway
DAILY_LOSS_LIMIT = 0.02    # 2% circuit breaker

STATE_FILE = Path("/opt/atnn/data/simple_state.json")
DISCORD = "https://discord.com/api/webhooks/1482171912724545638/EiV03Fa7qhBj4VRXw9ItpR0w9l1-b6rI1kOwxPmeTM3ddDmf4g_uAghDRWtCW9SRF4M1"


def load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"position": 0, "entry_price": 0.0, "entry_date": "",
            "highest_close": 0.0, "stop_price": 0.0,
            "flat_days": 0, "nav_at_open": 0.0}


def save_state(s):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=STATE_FILE.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(s, f, indent=2, default=str)
        os.replace(tmp, STATE_FILE)
    except Exception:
        try:
            os.unlink(tmp)
        except Exception:
            pass
        raise


def notify(msg):
    try:
        subprocess.run([
            "curl", "-s", "-X", "POST",
            "-H", "Content-Type: application/json",
            "-H", "User-Agent: curl/7.68.0",
            "-d", json.dumps({"content": f"**ATNN**: {msg}"}),
            DISCORD,
        ], capture_output=True, timeout=10)
    except Exception:
        pass


def tick(price):
    return round(price * 4) / 4


def front_month():
    today = date.today()
    def third_friday(y, m):
        first = date(y, m, 1)
        return first + timedelta(days=(4 - first.weekday()) % 7) + timedelta(weeks=2)
    for y in [today.year, today.year + 1]:
        for m in [3, 6, 9, 12]:
            if today <= third_friday(y, m) - timedelta(days=8):
                return f"{y}{m:02d}"
    return f"{today.year + 1}03"


def get_market_data():
    spy = yf.download("SPY", period="250d", interval="1d",
                      auto_adjust=True, progress=False)
    close = spy["Close"].squeeze()
    high = spy["High"].squeeze()
    low = spy["Low"].squeeze()

    ma50 = float(close.rolling(50).mean().iloc[-1])
    ma200 = float(close.rolling(200).mean().iloc[-1])
    last_close = float(close.iloc[-1])
    ibs = float((close.iloc[-1] - low.iloc[-1]) / max(high.iloc[-1] - low.iloc[-1], 0.001))

    avg_rng = float((high - low).rolling(25).mean().iloc[-1])
    roll_hi = float(high.rolling(10).max().iloc[-1])
    ibs_threshold = roll_hi - 2.5 * avg_rng

    uptrend = last_close > ma50 and last_close > ma200
    ibs_entry = ibs < IBS_ENTRY and last_close < ibs_threshold and uptrend

    return {
        "close": last_close,
        "ma50": ma50,
        "ma200": ma200,
        "ibs": ibs,
        "ibs_entry": ibs_entry,
        "uptrend": uptrend,
        "prev_high": float(high.iloc[-2]) if len(high) >= 2 else 0,
    }


# ─── Morning: manage position ────────────────────────────────────────────────

async def morning(dry_run=False):
    state = load_state()
    log.info("=== MORNING ===")

    mkt = get_market_data()
    log.info("SPY: $%.2f | 50d: $%.2f | 200d: $%.2f | uptrend: %s | IBS: %.3f",
             mkt["close"], mkt["ma50"], mkt["ma200"], mkt["uptrend"], mkt["ibs"])

    from ib_async import IB, Future, MarketOrder, StopOrder
    ib = IB()
    try:
        await asyncio.wait_for(ib.connectAsync(IBKR_HOST, IBKR_PORT, clientId=CLIENT_ID), 20)
    except Exception as e:
        log.error("Connect failed: %s", e)
        save_state(state)
        return

    try:
        acct = await ib.accountSummaryAsync()
        nav = float(next((s.value for s in acct if s.tag == "NetLiquidation"), 0))
        upnl = float(next((s.value for s in acct if s.tag == "UnrealizedPnL"), 0))

        # Reconcile: check broker position
        broker_pos = sum(int(p.position) for p in ib.positions()
                        if p.contract.symbol == "MES" and int(p.position) != 0)
        if broker_pos != state.get("position", 0):
            log.warning("RECONCILE: broker=%d state=%d -> using broker", broker_pos, state["position"])
            state["position"] = broker_pos

        log.info("NAV: $%.2f | Unrealized: $%.2f | MES: %d", nav, upnl, state["position"])

        # ── If we have a position ──
        if state["position"] > 0:
            spx_close = mkt["close"] * 10  # SPY to SPX proxy

            # Update highest close for trailing stop
            if spx_close > state.get("highest_close", 0):
                state["highest_close"] = spx_close
                log.info("New high: %.2f", spx_close)

            # Trail the stop
            new_stop = tick(state["highest_close"] - TRAIL_PTS)
            old_stop = state.get("stop_price", 0)

            if new_stop > old_stop:
                log.info("Trailing stop: %.2f -> %.2f (+%.2f pts locked in)",
                         old_stop, new_stop, new_stop - old_stop)

                # Update stop on IBKR
                if not dry_run:
                    contract = Future(symbol="MES", lastTradeDateOrContractMonth=front_month(),
                                      exchange="CME", currency="USD", multiplier="5")
                    qualified = await ib.qualifyContractsAsync(contract)
                    if qualified:
                        # Cancel old stop, place new one
                        for trade in ib.trades():
                            if (not trade.isDone() and trade.order.orderType == "STP"
                                    and trade.contract.symbol == "MES"):
                                ib.cancelOrder(trade.order)
                        await asyncio.sleep(1)

                        stop = StopOrder("SELL", state["position"], new_stop)
                        ib.placeOrder(qualified[0], stop)
                        log.info("New stop placed on IBKR: %.2f", new_stop)

                state["stop_price"] = new_stop
                notify(f"Trailing stop: {new_stop:.2f} (locked {new_stop - old_stop:.0f} pts)")

            # Check trend break: SPY below 50d MA
            if not mkt["uptrend"]:
                log.info("TREND BREAK: SPY $%.2f below 50d MA $%.2f", mkt["close"], mkt["ma50"])
                if not dry_run:
                    contract = Future(symbol="MES", lastTradeDateOrContractMonth=front_month(),
                                      exchange="CME", currency="USD", multiplier="5")
                    qualified = await ib.qualifyContractsAsync(contract)
                    if qualified:
                        # Cancel stops first
                        for trade in ib.trades():
                            if not trade.isDone() and trade.contract.symbol == "MES":
                                ib.cancelOrder(trade.order)
                        await asyncio.sleep(1)
                        order = MarketOrder("SELL", state["position"])
                        trade = ib.placeOrder(qualified[0], order)
                        for _ in range(60):
                            await asyncio.sleep(0.5)
                            if trade.isDone():
                                fill = trade.orderStatus.avgFillPrice
                                entry_spx = state["entry_price"] * 10
                                pnl = (fill - entry_spx) * state["position"] * 5
                                log.info("TREND EXIT: sold @ %.2f, P&L=$%.2f", fill, pnl)
                                notify(f"TREND EXIT: sold @ {fill:.2f} P&L=${pnl:.2f}")
                                break
                state["position"] = 0
                state["flat_days"] = 0

            # Circuit breaker
            if state.get("nav_at_open", 0) > 0:
                daily_pnl_pct = (nav - state["nav_at_open"]) / state["nav_at_open"]
                if daily_pnl_pct < -DAILY_LOSS_LIMIT:
                    log.error("CIRCUIT BREAKER: daily P&L %.2f%%", daily_pnl_pct * 100)

        # ── If flat ──
        else:
            if mkt["uptrend"]:
                state["flat_days"] = state.get("flat_days", 0) + 1
                log.info("Flat in uptrend: %d days (enter after %d or on IBS)",
                         state["flat_days"], MAX_FLAT_DAYS)

                # If flat too long in a clear uptrend, enter
                if state["flat_days"] >= MAX_FLAT_DAYS:
                    log.info("TREND ENTRY: flat %d days in uptrend, entering", state["flat_days"])
                    if not dry_run:
                        contract = Future(symbol="MES", lastTradeDateOrContractMonth=front_month(),
                                          exchange="CME", currency="USD", multiplier="5")
                        qualified = await ib.qualifyContractsAsync(contract)
                        if qualified:
                            order = MarketOrder("BUY", 1)
                            trade = ib.placeOrder(qualified[0], order)
                            for _ in range(60):
                                await asyncio.sleep(0.5)
                                if trade.isDone():
                                    fill = trade.orderStatus.avgFillPrice
                                    state["position"] = 1
                                    state["entry_price"] = mkt["close"]
                                    state["entry_date"] = str(date.today())
                                    state["highest_close"] = fill
                                    state["stop_price"] = tick(fill - TRAIL_PTS)

                                    stop = StopOrder("SELL", 1, state["stop_price"])
                                    ib.placeOrder(qualified[0], stop)
                                    log.info("ENTERED: BUY 1 MES @ %.2f, stop %.2f",
                                             fill, state["stop_price"])
                                    notify(f"TREND ENTRY: BUY 1 MES @ {fill:.2f} stop {state['stop_price']:.2f}")
                                    break
                        state["flat_days"] = 0
            else:
                state["flat_days"] = 0
                log.info("No uptrend. Staying flat.")

        state["nav_at_open"] = nav

        msg = (f"Morning: NAV=${nav:.2f} | MES={state['position']} | "
               f"uptrend={mkt['uptrend']} | IBS={mkt['ibs']:.3f}")
        log.info(msg)
        notify(msg)

        ib.disconnect()
    except Exception as e:
        log.error("Morning error: %s", e, exc_info=True)
        try:
            ib.disconnect()
        except Exception:
            pass

    save_state(state)


# ─── Close: IBS entry ────────────────────────────────────────────────────────

async def close_cycle(dry_run=False):
    state = load_state()
    log.info("=== CLOSE ===")

    if state.get("position", 0) > 0:
        log.info("Already holding. No IBS entry needed.")
        save_state(state)
        return

    mkt = get_market_data()
    log.info("IBS check: ibs=%.3f entry=%s uptrend=%s", mkt["ibs"], mkt["ibs_entry"], mkt["uptrend"])

    if not mkt["ibs_entry"]:
        log.info("No IBS signal at close.")
        save_state(state)
        return

    log.info("IBS ENTRY SIGNAL: IBS=%.3f in uptrend", mkt["ibs"])

    from ib_async import IB, Future, MarketOrder, StopOrder
    ib = IB()
    try:
        await asyncio.wait_for(ib.connectAsync(IBKR_HOST, IBKR_PORT, clientId=CLIENT_ID + 1), 20)

        contract = Future(symbol="MES", lastTradeDateOrContractMonth=front_month(),
                          exchange="CME", currency="USD", multiplier="5")
        qualified = await ib.qualifyContractsAsync(contract)
        if not qualified:
            ib.disconnect()
            return
        contract = qualified[0]

        if dry_run:
            log.info("[DRY] Would BUY 1 MES on IBS signal")
        else:
            order = MarketOrder("BUY", 1)
            trade = ib.placeOrder(contract, order)
            for _ in range(60):
                await asyncio.sleep(0.5)
                if trade.isDone():
                    fill = trade.orderStatus.avgFillPrice
                    state["position"] = 1
                    state["entry_price"] = mkt["close"]
                    state["entry_date"] = str(date.today())
                    state["highest_close"] = fill
                    state["stop_price"] = tick(fill - TRAIL_PTS)
                    state["flat_days"] = 0

                    stop = StopOrder("SELL", 1, state["stop_price"])
                    ib.placeOrder(contract, stop)
                    log.info("IBS ENTRY: BUY 1 MES @ %.2f, stop %.2f", fill, state["stop_price"])
                    notify(f"IBS ENTRY: BUY 1 MES @ {fill:.2f} stop {state['stop_price']:.2f} IBS={mkt['ibs']:.3f}")
                    break

        ib.disconnect()
    except Exception as e:
        log.error("Close error: %s", e, exc_info=True)
        try:
            ib.disconnect()
        except Exception:
            pass

    save_state(state)


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    if "--close" in sys.argv:
        asyncio.run(close_cycle(dry_run=dry))
    else:
        asyncio.run(morning(dry_run=dry))
