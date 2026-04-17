#!/usr/bin/env python3
"""
live_multi.py — Multi-instrument trend following + mean reversion
=================================================================
Trades micro futures (MES, MNQ, M2K) using simple, proven edges.

Strategy:
  1. TREND FILTER: Long only when proxy ETF > 50d MA AND > 200d MA
  2. IBS ENTRY: Enter on IBS < 0.20 in confirmed uptrend (mean reversion)
  3. PULLBACK ENTRY: Enter on RSI(14) < 40 in confirmed uptrend
  4. ATR TRAILING STOP: 1.5x 14-day ATR from highest close (adaptive)
  5. TREND BREAK EXIT: Proxy close < 50d MA → exit immediately

Risk management:
  - ATR-based stops (adaptive to volatility)
  - Max 10% NAV risk per trade
  - Max 2 concurrent positions
  - 2% daily circuit breaker (aggregate realized losses)
  - Margin check before new entries

Runs twice daily via cron:
  --morning (9:35 AM ET): Trail stops, check exits, reconcile
  --close   (3:50 PM ET): Check for new entries

Usage:
  python3 live_multi.py --morning [--dry-run]
  python3 live_multi.py --close [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import calendar
import json
import logging
import os
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

# ═══════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════

os.makedirs("/opt/atnn/logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("/opt/atnn/logs/multi.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("multi")


# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

IBKR_HOST = "127.0.0.1"
IBKR_PORT = 4003          # CRITICAL: must be 4003, NOT 4001
CLIENT_ID = 50             # Dedicated client ID for this script

STATE_FILE = Path("/opt/atnn/data/multi_state.json")

DISCORD_WEBHOOK = (
    "https://discord.com/api/webhooks/1482171912724545638/"
    "EiV03Fa7qhBj4VRXw9ItpR0w9l1-b6rI1kOwxPmeTM3ddDmf4g_uAghDRWtCW9SRF4M1"
)

# ── Instrument definitions ──
# proxy: ETF for daily data (yfinance)
# scale: approximate multiplier from ETF price to futures price
# multiplier: USD per index point
# tick_size: minimum price increment
# margin_est: approximate initial margin requirement
INSTRUMENTS = {
    "MES": {
        "proxy": "SPY",
        "scale": 10,
        "multiplier": 5,
        "tick_size": 0.25,
        "exchange": "CME",
        "expiry": "202606",
        "margin_est": 1500,
    },
    "MNQ": {
        "proxy": "QQQ",
        "scale": 40,
        "multiplier": 2,
        "tick_size": 0.25,
        "exchange": "CME",
        "expiry": "202606",
        "margin_est": 2100,
    },
    "M2K": {
        "proxy": "IWM",
        "scale": 10,
        "multiplier": 5,
        "tick_size": 0.10,
        "exchange": "CME",
        "expiry": "202606",
        "margin_est": 800,
    },
}

# ── Risk parameters ──
ATR_PERIOD = 14
ATR_MULTIPLIER = 1.5       # Trail stop = 1.5x ATR from highest close
MAX_RISK_PCT = 0.10         # Max 10% of NAV risk per trade
MAX_POSITIONS = 2           # Max concurrent positions
MAX_MARGIN_PCT = 0.65       # Don't use more than 65% of NAV for margin
CIRCUIT_BREAKER_PCT = 0.02  # 2% daily loss limit (no new entries)

# ── Signal parameters ──
IBS_THRESHOLD = 0.20        # IBS below this = oversold entry
RSI_THRESHOLD = 40          # RSI below this = pullback entry
MA_FAST = 50                # 50-day moving average
MA_SLOW = 200               # 200-day moving average

# ── Safety ──
ROLL_WARNING_DAYS = 14      # Warn when contract expiry is near
HISTORY_DAYS = 300          # Days of proxy data to fetch


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

def tick_round(price: float, tick_size: float) -> float:
    """Round price DOWN to nearest valid tick (conservative for stops)."""
    return round(int(price / tick_size) * tick_size, 6)


def discord_msg(msg: str, dry_run: bool = False):
    """Send message to Discord webhook."""
    if dry_run:
        log.info("[DRY-RUN] Discord: %s", msg[:200])
        return
    try:
        subprocess.run(
            [
                "curl", "-s", "-X", "POST",
                "-H", "Content-Type: application/json",
                "-H", "User-Agent: curl/7.68.0",
                "-d", json.dumps({"content": msg[:2000]}),
                DISCORD_WEBHOOK,
            ],
            capture_output=True, timeout=10,
        )
    except Exception as e:
        log.error("Discord send failed: %s", e)


def load_state() -> dict:
    """Load state from JSON file."""
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception as e:
            log.error("Failed to load state: %s", e)
    return {
        "positions": {},
        "daily_pnl": 0.0,
        "last_date": "",
        "nav_at_open": 0.0,
        "trades_today": 0,
        "circuit_broken": False,
    }


def save_state(state: dict):
    """Save state atomically (write to temp, then rename)."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    tmp.rename(STATE_FILE)
    log.info("State saved.")


# ═══════════════════════════════════════════════════════════════════
# MARKET DATA & INDICATORS
# ═══════════════════════════════════════════════════════════════════

def fetch_market_data(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Fetch daily OHLCV from yfinance for proxy ETFs."""
    data = {}
    for sym in symbols:
        try:
            df = yf.download(sym, period=f"{HISTORY_DAYS}d", progress=False)
            if df.empty:
                log.error("%s: yfinance returned empty dataframe", sym)
                continue
            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) < MA_SLOW + 10:
                log.warning("%s: only %d bars (need %d+) — skipping",
                            sym, len(df), MA_SLOW + 10)
                continue
            data[sym] = df
            log.info("%s: %d bars, last close=$%.2f (%s)",
                     sym, len(df), float(df["Close"].iloc[-1]),
                     str(df.index[-1].date()))
        except Exception as e:
            log.error("Failed to fetch %s: %s", sym, e)
    return data


def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                period: int = ATR_PERIOD) -> float:
    """Compute Average True Range."""
    # True Range needs previous close, so we start from index 1
    prev_close = close[:-1]
    curr_high = high[1:]
    curr_low = low[1:]

    tr = np.maximum(
        curr_high - curr_low,
        np.maximum(
            np.abs(curr_high - prev_close),
            np.abs(curr_low - prev_close),
        ),
    )
    # Simple moving average of last `period` true ranges
    if len(tr) < period:
        return float(np.mean(tr)) if len(tr) > 0 else 0.0
    return float(np.mean(tr[-period:]))


def compute_rsi(close: np.ndarray, period: int = 14) -> float:
    """Compute RSI using Wilder's smoothing (EWM)."""
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    if len(gains) < period:
        return 50.0  # Neutral if insufficient data

    avg_gain = float(pd.Series(gains).ewm(span=period, min_periods=period).mean().iloc[-1])
    avg_loss = float(pd.Series(losses).ewm(span=period, min_periods=period).mean().iloc[-1])

    if avg_loss < 1e-10:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def compute_indicators(df: pd.DataFrame) -> dict:
    """Compute all indicators from proxy ETF daily data."""
    close = df["Close"].astype(float).values
    high = df["High"].astype(float).values
    low = df["Low"].astype(float).values

    last_close = float(close[-1])

    # Moving averages (last value of rolling window)
    ma_fast = float(np.mean(close[-MA_FAST:])) if len(close) >= MA_FAST else last_close
    ma_slow = float(np.mean(close[-MA_SLOW:])) if len(close) >= MA_SLOW else last_close

    # ATR
    atr = compute_atr(high, low, close)

    # IBS (Internal Bar Strength) of the latest bar
    h, l, c = float(high[-1]), float(low[-1]), last_close
    ibs = (c - l) / (h - l) if (h - l) > 1e-6 else 0.5

    # RSI(14)
    rsi = compute_rsi(close)

    uptrend = (last_close > ma_fast) and (last_close > ma_slow)

    return {
        "close": last_close,
        "ma_fast": ma_fast,
        "ma_slow": ma_slow,
        "atr": atr,
        "ibs": ibs,
        "rsi": rsi,
        "uptrend": uptrend,
    }


# ═══════════════════════════════════════════════════════════════════
# IBKR CONNECTION & TRADING
# ═══════════════════════════════════════════════════════════════════

async def connect_ibkr():
    """Connect to IBKR gateway on port 4003."""
    from ib_async import IB
    ib = IB()
    await asyncio.wait_for(
        ib.connectAsync(IBKR_HOST, IBKR_PORT, clientId=CLIENT_ID),
        timeout=15,
    )
    log.info("Connected to IBKR (port %d, clientId %d)", IBKR_PORT, CLIENT_ID)
    return ib


async def get_account_info(ib) -> dict:
    """Get NAV, cash, unrealized P&L, margin used."""
    acct = await ib.accountSummaryAsync()

    def val(tag: str) -> float:
        return float(next((s.value for s in acct if s.tag == tag), 0))

    return {
        "nav": val("NetLiquidation"),
        "cash": val("TotalCashValue"),
        "unrealized_pnl": val("UnrealizedPnL"),
        "margin_used": val("InitMarginReq"),
    }


async def qualify_contract(ib, symbol: str, cfg: dict):
    """Qualify a futures contract. Returns contract or None."""
    from ib_async import Future
    contract = Future(
        symbol=symbol,
        lastTradeDateOrContractMonth=cfg["expiry"],
        exchange=cfg["exchange"],
        currency="USD",
    )
    try:
        qualified = await ib.qualifyContractsAsync(contract)
        if qualified:
            log.info("Qualified %s (conId=%d)", symbol, qualified[0].conId)
            return qualified[0]
    except Exception as e:
        log.warning("Failed to qualify %s: %s", symbol, e)
    log.warning("Could not qualify %s — will skip this instrument", symbol)
    return None


async def get_broker_positions(ib) -> dict[str, int]:
    """Get all non-zero positions from broker."""
    positions = {}
    for p in ib.positions():
        qty = int(p.position)
        if qty != 0:
            positions[p.contract.symbol] = qty
    return positions


async def cancel_stops_for(ib, symbol: str):
    """Cancel all open stop orders for a given symbol (any client ID)."""
    cancelled = 0
    # Use reqAllOpenOrders to find stops from ANY client ID
    all_trades = await ib.reqAllOpenOrdersAsync()
    for trade in all_trades:
        if (trade.contract.symbol == symbol
                and trade.order.orderType == "STP"
                and not trade.isDone()):
            ib.cancelOrder(trade.order)
            cancelled += 1
            log.info("Cancelling stop for %s (orderId=%d, price=%.2f, clientId=%d)",
                     symbol, trade.order.orderId, trade.order.auxPrice,
                     trade.order.clientId)
            await asyncio.sleep(0.5)
    if cancelled:
        log.info("Cancelled %d stop order(s) for %s", cancelled, symbol)


async def place_stop_order(ib, contract, qty: int, stop_price: float,
                           tick_size: float, dry_run: bool = False) -> float:
    """Place a stop order. Returns the rounded stop price."""
    from ib_async import StopOrder
    rounded = tick_round(stop_price, tick_size)
    action = "SELL" if qty > 0 else "BUY"

    if dry_run:
        log.info("[DRY-RUN] Would place %s STOP %d @ %.2f", action, abs(qty), rounded)
        return rounded

    order = StopOrder(action, abs(qty), rounded)
    trade = ib.placeOrder(contract, order)
    await asyncio.sleep(2)
    log.info("Stop order: %s %d %s @ %.2f — status=%s",
             action, abs(qty), contract.symbol, rounded, trade.orderStatus.status)
    return rounded


async def place_market_order(ib, contract, action: str, qty: int,
                             dry_run: bool = False) -> Optional[float]:
    """Place a market order. Returns fill price or None."""
    from ib_async import MarketOrder

    if dry_run:
        log.info("[DRY-RUN] Would place %s MKT %d %s", action, qty, contract.symbol)
        return None

    order = MarketOrder(action, qty)
    trade = ib.placeOrder(contract, order)

    # Wait for fill (up to 30 seconds)
    for _ in range(30):
        await asyncio.sleep(1)
        if trade.orderStatus.status == "Filled":
            fill = trade.orderStatus.avgFillPrice
            log.info("FILLED: %s %d %s @ %.2f", action, qty, contract.symbol, fill)
            return fill

    status = trade.orderStatus.status
    log.warning("Order not filled after 30s: %s %d %s — status=%s",
                action, qty, contract.symbol, status)
    return None


# ═══════════════════════════════════════════════════════════════════
# MORNING CYCLE — Position Management
# ═══════════════════════════════════════════════════════════════════

async def morning_cycle(ib, state: dict, market_data: dict,
                        contracts: dict, dry_run: bool = False) -> list[str]:
    """
    Manage existing positions:
      1. Reconcile with broker
      2. Update trailing stops (ATR-based)
      3. Check for trend breaks → exit
      4. Update circuit breaker
    """
    actions = []
    account = await get_account_info(ib)
    nav = account["nav"]
    today = str(date.today())

    # ── Reset daily tracking on new day ──
    if state.get("last_date", "") != today:
        state["daily_pnl"] = 0.0
        state["nav_at_open"] = nav
        state["trades_today"] = 0
        state["circuit_broken"] = False
        state["last_date"] = today
        log.info("New trading day — NAV at open: $%.2f", nav)

    # ── Get broker positions and portfolio market prices ──
    broker_pos = await get_broker_positions(ib)
    log.info("Broker positions: %s", broker_pos if broker_pos else "FLAT")

    # Get live market prices from portfolio (more accurate than proxy*scale)
    portfolio_prices = {}
    for item in ib.portfolio():
        sym_p = item.contract.symbol
        if sym_p in INSTRUMENTS and item.marketPrice > 0:
            portfolio_prices[sym_p] = float(item.marketPrice)
    if portfolio_prices:
        log.info("Portfolio market prices: %s", portfolio_prices)

    positions = state.setdefault("positions", {})

    # ── Reconcile: broker has position we don't know about ──
    for sym, broker_qty in broker_pos.items():
        if sym not in INSTRUMENTS:
            continue
        state_qty = positions.get(sym, {}).get("qty", 0)
        if state_qty == 0 and broker_qty != 0:
            log.warning("%s: UNEXPECTED position on broker (%d) — adopting into state",
                        sym, broker_qty)
            cfg = INSTRUMENTS[sym]
            proxy = cfg["proxy"]
            if proxy in market_data:
                ind = compute_indicators(market_data[proxy])
                fut_price = ind["close"] * cfg["scale"]
                atr_fut = ind["atr"] * cfg["scale"]
                stop = tick_round(fut_price - ATR_MULTIPLIER * atr_fut, cfg["tick_size"])
                positions[sym] = {
                    "qty": broker_qty,
                    "entry_price": ind["close"],
                    "entry_futures": fut_price,
                    "entry_date": today,
                    "highest_close": fut_price,
                    "stop_price": stop,
                    "signal": "reconciled",
                }
                actions.append(f"{sym}: Adopted unexpected position ({broker_qty})")

    # ── Manage each known position ──
    to_remove = []
    for sym, pos in list(positions.items()):
        if pos.get("qty", 0) == 0:
            continue

        cfg = INSTRUMENTS.get(sym)
        if not cfg:
            continue

        proxy = cfg["proxy"]
        if proxy not in market_data:
            log.warning("%s: no market data for %s — skipping management", sym, proxy)
            continue

        ind = compute_indicators(market_data[proxy])
        broker_qty = broker_pos.get(sym, 0)
        state_qty = pos["qty"]

        # ── Reconcile: state says position but broker disagrees ──
        if broker_qty != state_qty:
            log.warning("%s: MISMATCH — broker=%d, state=%d", sym, broker_qty, state_qty)
            if broker_qty == 0:
                # Stop was hit or position was closed externally
                entry_fut = pos.get("entry_futures", 0)
                approx_exit = ind["close"] * cfg["scale"]
                pnl = (approx_exit - entry_fut) * cfg["multiplier"] * state_qty
                log.warning("%s: Broker is flat — stop likely hit. Est P&L: $%.0f", sym, pnl)
                state["daily_pnl"] = state.get("daily_pnl", 0.0) + pnl
                actions.append(f"{sym}: STOPPED OUT (broker flat). Est P&L: ${pnl:.0f}")
                to_remove.append(sym)
                continue
            else:
                # Trust broker quantity
                pos["qty"] = broker_qty
                log.warning("%s: Updated state to match broker (%d)", sym, broker_qty)

        # ── Compute ATR-based trailing stop ──
        atr_futures = ind["atr"] * cfg["scale"]
        trail_distance = ATR_MULTIPLIER * atr_futures

        # Sanity check: trail must be at least 0.5x ATR
        min_trail = 0.5 * atr_futures
        trail_distance = max(trail_distance, min_trail)

        # Update highest close (use max of proxy estimate AND actual market price)
        current_fut_close = ind["close"] * cfg["scale"]
        market_price = portfolio_prices.get(sym, current_fut_close)
        best_current = max(current_fut_close, market_price)
        prev_highest = pos.get("highest_close", best_current)
        new_highest = max(prev_highest, best_current)
        pos["highest_close"] = new_highest

        new_stop = tick_round(new_highest - trail_distance, cfg["tick_size"])
        old_stop = pos.get("stop_price", 0)

        # Trail can only move UP for long positions
        if new_stop > old_stop:
            pos["stop_price"] = new_stop
            log.info("%s: TRAIL ↑ %.2f → %.2f (highest=%.2f, 1.5xATR=%.1f pts)",
                     sym, old_stop, new_stop, new_highest, trail_distance)

            # Update stop order on IBKR
            contract = contracts.get(sym)
            if contract:
                await cancel_stops_for(ib, sym)
                await place_stop_order(ib, contract, pos["qty"], new_stop,
                                       cfg["tick_size"], dry_run)
                actions.append(f"{sym}: Trail → {new_stop:.2f}")
        else:
            log.info("%s: Stop unchanged at %.2f (highest=%.2f, new_calc=%.2f)",
                     sym, old_stop, new_highest, new_stop)

        # ── Check trend break: close below 50d MA → exit ──
        if ind["close"] < ind["ma_fast"]:
            log.warning("%s: TREND BREAK — $%.2f < 50d MA $%.2f",
                        sym, ind["close"], ind["ma_fast"])
            contract = contracts.get(sym)
            if contract:
                await cancel_stops_for(ib, sym)
                fill = await place_market_order(ib, contract, "SELL",
                                                abs(pos["qty"]), dry_run)
                if fill or dry_run:
                    entry_fut = pos.get("entry_futures", 0)
                    exit_price = fill if fill else current_fut_close
                    pnl = (exit_price - entry_fut) * cfg["multiplier"] * pos["qty"]
                    state["daily_pnl"] = state.get("daily_pnl", 0.0) + pnl
                    actions.append(f"{sym}: TREND BREAK EXIT @ {exit_price:.2f} "
                                   f"(P&L: ${pnl:.0f})")
                to_remove.append(sym)

    # ── Remove closed positions ──
    for sym in to_remove:
        positions.pop(sym, None)

    # ── Circuit breaker ──
    nav_open = state.get("nav_at_open", nav)
    if nav_open > 0:
        dd_pct = state.get("daily_pnl", 0.0) / nav_open
        if dd_pct < -CIRCUIT_BREAKER_PCT:
            state["circuit_broken"] = True
            log.warning("⚠️ CIRCUIT BREAKER TRIGGERED: $%.0f (%.1f%%)",
                        state["daily_pnl"], dd_pct * 100)
            actions.append(f"⚠️ CIRCUIT BREAKER: ${state['daily_pnl']:.0f} "
                           f"({dd_pct * 100:.1f}%)")
        else:
            state["circuit_broken"] = False

    active = sum(1 for p in positions.values() if p.get("qty", 0) != 0)
    log.info("Morning done — NAV=$%.2f, active=%d, daily_pnl=$%.2f",
             nav, active, state.get("daily_pnl", 0.0))

    return actions


# ═══════════════════════════════════════════════════════════════════
# CLOSE CYCLE — Entry Signals
# ═══════════════════════════════════════════════════════════════════

async def close_cycle(ib, state: dict, market_data: dict,
                      contracts: dict, dry_run: bool = False) -> list[str]:
    """
    Check for new entry signals at ~3:50 PM ET:
      1. Check circuit breaker
      2. For each uninvested instrument:
         a. Verify uptrend (close > 50d + 200d MA)
         b. Check IBS < 0.20 OR RSI(14) < 40
         c. Verify risk and margin limits
         d. Enter position + place stop
    """
    actions = []
    account = await get_account_info(ib)
    nav = account["nav"]
    margin_used = account["margin_used"]
    today = str(date.today())

    # ── Reset daily tracking if needed ──
    if state.get("last_date", "") != today:
        state["daily_pnl"] = 0.0
        state["nav_at_open"] = nav
        state["trades_today"] = 0
        state["circuit_broken"] = False
        state["last_date"] = today

    # ── Circuit breaker check ──
    if state.get("circuit_broken", False):
        log.info("Circuit breaker active — no new entries today")
        actions.append("Circuit breaker active — skipping all entries")
        return actions

    positions = state.setdefault("positions", {})
    active_count = sum(1 for p in positions.values() if p.get("qty", 0) != 0)

    if active_count >= MAX_POSITIONS:
        log.info("Max positions (%d/%d) reached — no new entries", active_count, MAX_POSITIONS)
        actions.append(f"Max positions ({active_count}/{MAX_POSITIONS}) — no entries")
        return actions

    # ── Calculate existing portfolio risk ──
    existing_risk = 0.0
    for sym, pos in positions.items():
        if pos.get("qty", 0) == 0:
            continue
        cfg = INSTRUMENTS.get(sym, {})
        entry = pos.get("entry_futures", 0)
        stop = pos.get("stop_price", 0)
        if entry > 0 and stop > 0:
            existing_risk += (entry - stop) * cfg.get("multiplier", 5) * pos["qty"]

    log.info("Portfolio: NAV=$%.2f, active=%d, existing_risk=$%.0f, margin=$%.0f",
             nav, active_count, existing_risk, margin_used)

    # ── Scan each instrument for entry ──
    for sym, cfg in INSTRUMENTS.items():
        # Already holding this instrument
        if sym in positions and positions[sym].get("qty", 0) != 0:
            log.info("%s: Already holding — skip", sym)
            continue

        # Max positions reached (could have been filled by prior iteration)
        if active_count >= MAX_POSITIONS:
            break

        # No qualified contract
        contract = contracts.get(sym)
        if not contract:
            log.info("%s: No qualified contract — skip", sym)
            continue

        # No market data
        proxy = cfg["proxy"]
        if proxy not in market_data:
            log.warning("%s: No data for %s — skip", sym, proxy)
            continue

        ind = compute_indicators(market_data[proxy])

        log.info("%s (%s): close=$%.2f, 50dMA=$%.2f, 200dMA=$%.2f, "
                 "uptrend=%s, IBS=%.3f, RSI=%.1f, ATR=$%.2f",
                 sym, proxy, ind["close"], ind["ma_fast"], ind["ma_slow"],
                 ind["uptrend"], ind["ibs"], ind["rsi"], ind["atr"])

        # ── FILTER 1: Uptrend required ──
        if not ind["uptrend"]:
            log.info("%s: NOT in uptrend — skip", sym)
            actions.append(f"{sym}: No uptrend ({proxy} ${ind['close']:.2f} "
                           f"vs 50d ${ind['ma_fast']:.2f})")
            continue

        # ── FILTER 2: Entry signal required ──
        ibs_signal = ind["ibs"] < IBS_THRESHOLD
        rsi_signal = ind["rsi"] < RSI_THRESHOLD

        if not ibs_signal and not rsi_signal:
            log.info("%s: No signal (IBS=%.3f, RSI=%.1f) — skip",
                     sym, ind["ibs"], ind["rsi"])
            actions.append(f"{sym}: Uptrend but no signal "
                           f"(IBS={ind['ibs']:.2f}, RSI={ind['rsi']:.0f})")
            continue

        signal_type = "IBS" if ibs_signal else "RSI_PULLBACK"
        if ibs_signal and rsi_signal:
            signal_type = "IBS+RSI"
        log.info("%s: ENTRY SIGNAL → %s", sym, signal_type)

        # ── FILTER 3: Risk check ──
        atr_futures = ind["atr"] * cfg["scale"]
        trail_distance = ATR_MULTIPLIER * atr_futures
        risk_per_contract = trail_distance * cfg["multiplier"]
        max_risk = nav * MAX_RISK_PCT

        if risk_per_contract > max_risk:
            log.warning("%s: Risk $%.0f exceeds max $%.0f (%.0f%% NAV) — skip",
                        sym, risk_per_contract, max_risk, MAX_RISK_PCT * 100)
            actions.append(f"{sym}: {signal_type} signal but risk too high "
                           f"(${risk_per_contract:.0f} > ${max_risk:.0f})")
            continue

        # ── FILTER 4: Margin check ──
        margin_after = margin_used + cfg["margin_est"]
        if margin_after > nav * MAX_MARGIN_PCT:
            log.warning("%s: Margin $%.0f would exceed %.0f%% NAV — skip",
                        sym, margin_after, MAX_MARGIN_PCT * 100)
            actions.append(f"{sym}: {signal_type} signal but insufficient margin")
            continue

        # ── EXECUTE ENTRY ──
        futures_close = ind["close"] * cfg["scale"]
        stop_price = tick_round(futures_close - trail_distance, cfg["tick_size"])

        log.info("%s: ENTERING LONG 1 — est_price=%.2f, stop=%.2f, "
                 "trail=%.1fpts, risk=$%.0f",
                 sym, futures_close, stop_price, trail_distance, risk_per_contract)

        fill = await place_market_order(ib, contract, "BUY", 1, dry_run)

        if fill or dry_run:
            entry_futures = fill if fill else futures_close
            # Recalculate stop from actual fill price (more accurate than proxy)
            actual_stop_price = tick_round(entry_futures - trail_distance,
                                           cfg["tick_size"])
            actual_stop = await place_stop_order(
                ib, contract, 1, actual_stop_price, cfg["tick_size"], dry_run
            )

            positions[sym] = {
                "qty": 1,
                "entry_price": ind["close"],
                "entry_futures": entry_futures,
                "entry_date": today,
                "highest_close": entry_futures,
                "stop_price": actual_stop,
                "signal": signal_type,
            }

            active_count += 1
            margin_used += cfg["margin_est"]
            state["trades_today"] = state.get("trades_today", 0) + 1

            actions.append(
                f"{sym}: LONG 1 @ {entry_futures:.2f} | "
                f"Stop: {actual_stop:.2f} | {signal_type} | "
                f"Risk: ${risk_per_contract:.0f}"
            )
            log.info("%s: Entry complete. Active positions: %d", sym, active_count)
        else:
            log.error("%s: Market order failed — no fill received", sym)
            actions.append(f"{sym}: ENTRY FAILED — no fill")

    if not actions:
        actions.append("No signals or entries today")

    return actions


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

async def main():
    parser = argparse.ArgumentParser(description="Multi-instrument trend system")
    parser.add_argument("--morning", action="store_true", help="Run morning cycle")
    parser.add_argument("--close", action="store_true", help="Run close cycle")
    parser.add_argument("--dry-run", action="store_true", help="No real orders")
    args = parser.parse_args()

    if not args.morning and not args.close:
        log.error("Must specify --morning or --close")
        sys.exit(1)

    mode = "MORNING" if args.morning else "CLOSE"
    log.info("=" * 60)
    log.info("live_multi.py — %s%s", mode, " [DRY-RUN]" if args.dry_run else "")
    log.info("=" * 60)

    # ── Load state ──
    state = load_state()

    # ── Roll warning ──
    for sym, cfg in INSTRUMENTS.items():
        try:
            year = int(cfg["expiry"][:4])
            month = int(cfg["expiry"][4:6])
            cal = calendar.monthcalendar(year, month)
            fridays = [w[calendar.FRIDAY] for w in cal if w[calendar.FRIDAY] != 0]
            third_friday = date(year, month, fridays[2])
            days_left = (third_friday - date.today()).days
            if 0 < days_left <= ROLL_WARNING_DAYS:
                log.warning("⚠️ %s expires in %d days (%s) — ROLL NEEDED SOON",
                            sym, days_left, third_friday)
        except Exception:
            pass

    # ── Fetch market data ──
    proxies = list(set(cfg["proxy"] for cfg in INSTRUMENTS.values()))
    log.info("Fetching data for proxies: %s", proxies)
    market_data = fetch_market_data(proxies)

    if not market_data:
        msg = "🚨 live_multi.py: No market data — aborting"
        log.error(msg)
        discord_msg(msg, args.dry_run)
        sys.exit(1)

    # ── Connect to IBKR ──
    ib = None
    try:
        ib = await connect_ibkr()

        # ── Qualify all contracts ──
        contracts = {}
        for sym, cfg in INSTRUMENTS.items():
            c = await qualify_contract(ib, sym, cfg)
            if c:
                contracts[sym] = c
        log.info("Qualified contracts: %s", list(contracts.keys()))

        if not contracts:
            msg = "🚨 No contracts qualified — aborting"
            log.error(msg)
            discord_msg(msg, args.dry_run)
            sys.exit(1)

        # ── Run the appropriate cycle ──
        if args.morning:
            actions = await morning_cycle(ib, state, market_data, contracts, args.dry_run)
        else:
            actions = await close_cycle(ib, state, market_data, contracts, args.dry_run)

        # ── Save state ──
        save_state(state)

        # ── Build and send summary ──
        account = await get_account_info(ib)
        active_pos = {s: p for s, p in state.get("positions", {}).items()
                      if p.get("qty", 0) != 0}

        icon = "☀️" if args.morning else "🌙"
        lines = [
            f"**{icon} {mode} — "
            f"{'[DRY-RUN] ' if args.dry_run else ''}"
            f"{datetime.now().strftime('%Y-%m-%d %H:%M')}**",
            f"NAV: ${account['nav']:,.2f} | "
            f"UnrlzPnL: ${account['unrealized_pnl']:,.2f} | "
            f"DayPnL: ${state.get('daily_pnl', 0):.2f}",
        ]

        if active_pos:
            for s, p in active_pos.items():
                ef = p.get('entry_futures', 0)
                sp = p.get('stop_price', 0)
                sig = p.get('signal', '?')
                lines.append(f"  {s}: {p['qty']}x @ {ef:.2f} "
                             f"(stop {sp:.2f}, sig={sig})")
        else:
            lines.append("  Positions: FLAT")

        for a in actions:
            lines.append(f"• {a}")

        summary = "\n".join(lines)
        log.info("Summary:\n%s", summary)
        discord_msg(summary, args.dry_run)

    except Exception as e:
        log.error("FATAL: %s", e, exc_info=True)
        discord_msg(f"🚨 live_multi.py FATAL: {e}", args.dry_run)
        sys.exit(1)
    finally:
        if ib and ib.isConnected():
            ib.disconnect()
            log.info("Disconnected from IBKR")


if __name__ == "__main__":
    asyncio.run(main())
