#!/usr/bin/env python3
"""
live_momentum.py — Validated momentum rotation strategy for live IBKR trading.

This is the live trading runner for the backtest-validated momentum rotation
strategy (Sharpe 1.03, +144% over 3.2 years).

Logic:
1. Connect to IBKR via the existing gateway
2. Every morning at 10:00 AM ET, fetch latest prices via yfinance
3. Compute momentum rankings + regime filter
4. Every 10 trading days, rebalance: close positions not in top 3, open new ones
5. Monitor stops daily: 6% stop loss, 3.5% trailing stop
6. Go flat in bear regime (SPY < 200-day SMA)
7. Long-only, max 3 positions, 30% allocation each

Run: python scripts/live_momentum.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("live_momentum")

# ─── Config ──────────────────────────────────────────────────────────────

SYMBOLS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "META", "GOOGL", "TSLA",
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK",
    "AMD", "NFLX", "CRM", "AVGO", "INTC",
    "JPM", "BAC", "GS", "XOM", "CVX",
    "GLD", "JNJ", "UNH", "PFE", "MRK",
    "PLTR", "SOFI",
]

MAX_POSITIONS = 3
POSITION_PCT = 0.30
REBALANCE_DAYS = 10
MOM_FAST = 5
MOM_SLOW = 20
SMA_FILTER = 50
REGIME_SMA = 200
MIN_SCORE = 0.02
STOP_LOSS = 0.06
TRAILING_STOP = 0.035
MAX_HOLD = 20

IBKR_HOST = "127.0.0.1"
IBKR_PORT = 4003
IBKR_CLIENT_ID = 2  # Use different client ID from gateway monitoring
IBKR_ACCOUNT = "U22452226"

STATE_FILE = Path("/opt/atnn/data/momentum_state.json")


# ─── State Management ────────────────────────────────────────────────────

def load_state() -> dict:
    """Load persisted strategy state."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "day_counter": 0,
        "last_rebal_day": 0,
        "positions": {},  # sym: {qty, cost, entry_day, peak}
    }


def save_state(state: dict):
    """Persist strategy state."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ─── Market Data ─────────────────────────────────────────────────────────

def get_price_history(symbols: list, lookback_days: int = 300) -> pd.DataFrame:
    """Fetch daily close prices from yfinance."""
    logger.info("Fetching %d-day price history for %d symbols...", lookback_days, len(symbols))
    
    data = yf.download(
        symbols, period=f"{lookback_days}d", interval="1d",
        group_by="ticker", auto_adjust=True,
        threads=True, progress=False,
    )
    
    close = pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        for sym in symbols:
            try:
                close[sym] = data[sym]["Close"].dropna()
            except KeyError:
                pass
    
    close = close.ffill().dropna(how="all")
    logger.info("Got %d days x %d symbols", len(close), close.shape[1])
    return close


# ─── Strategy Logic ──────────────────────────────────────────────────────

def compute_regime(spy_prices: pd.Series) -> str:
    """BULL / NEUTRAL / BEAR based on SMA crossovers."""
    if len(spy_prices) < REGIME_SMA + 5:
        return "NEUTRAL"
    
    sma200 = spy_prices.rolling(REGIME_SMA).mean().iloc[-1]
    sma50 = spy_prices.rolling(SMA_FILTER).mean().iloc[-1]
    current = spy_prices.iloc[-1]
    
    if pd.isna(sma200) or pd.isna(sma50):
        return "NEUTRAL"
    
    if current > sma50 and sma50 > sma200:
        return "BULL"
    elif current < sma50 and sma50 < sma200:
        return "BEAR"
    return "NEUTRAL"


def rank_universe(close: pd.DataFrame) -> list:
    """Rank by momentum, filter by trend. Returns [(sym, score), ...]."""
    results = []
    
    for sym in close.columns:
        p = close[sym].dropna()
        if len(p) < SMA_FILTER + 5:
            continue
        if p.iloc[-MOM_FAST] <= 0 or p.iloc[-MOM_SLOW] <= 0:
            continue
        
        ret_fast = p.iloc[-1] / p.iloc[-MOM_FAST] - 1
        ret_slow = p.iloc[-1] / p.iloc[-MOM_SLOW] - 1
        
        # Both timeframes agree
        if np.sign(ret_fast) != np.sign(ret_slow):
            continue
        
        score = 0.5 * ret_fast + 0.5 * ret_slow
        
        # Above SMA filter
        sma = p.rolling(SMA_FILTER).mean().iloc[-1]
        if pd.isna(sma) or p.iloc[-1] <= sma:
            continue
        
        if score > MIN_SCORE:
            results.append((sym, float(score)))
    
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def check_exits(state: dict, current_prices: dict) -> list:
    """Check which positions should be exited.
    
    Returns list of (symbol, reason) tuples.
    """
    exits = []
    
    for sym, pos in state["positions"].items():
        if sym not in current_prices:
            continue
        
        price = current_prices[sym]
        entry = pos["cost"]
        peak = pos.get("peak", entry)
        age = state["day_counter"] - pos["entry_day"]
        
        ret = (price - entry) / entry
        
        # Update peak
        if price > peak:
            pos["peak"] = price
            peak = price
        
        trail_ret = (price - peak) / peak
        
        # Stop loss
        if ret < -STOP_LOSS:
            exits.append((sym, f"STOP_LOSS ({ret*100:.1f}%)"))
            continue
        
        # Trailing stop (if in profit and past initial hold)
        if age >= 3 and ret > 0.01 and trail_ret < -TRAILING_STOP:
            exits.append((sym, f"TRAILING_STOP ({trail_ret*100:.1f}% from peak)"))
            continue
        
        # Max hold
        if age >= MAX_HOLD:
            exits.append((sym, f"MAX_HOLD ({age} days)"))
            continue
    
    return exits


# ─── IBKR Execution ─────────────────────────────────────────────────────

async def connect_ibkr():
    """Connect to IBKR Gateway."""
    from ib_async import IB
    
    ib = IB()
    await ib.connectAsync(IBKR_HOST, IBKR_PORT, clientId=IBKR_CLIENT_ID)
    logger.info("Connected to IBKR: %s", ib.managedAccounts())
    return ib


async def get_account_nav(ib) -> float:
    """Get current account NAV."""
    summary = await ib.accountSummaryAsync()
    for item in summary:
        if item.tag == "NetLiquidation":
            return float(item.value)
    return 0.0


async def get_current_positions(ib) -> dict:
    """Get current IBKR positions. Returns {symbol: qty}."""
    positions = {}
    for pos in ib.positions():
        sym = pos.contract.symbol
        positions[sym] = int(pos.position)
    return positions


async def place_market_order(ib, symbol: str, qty: int, action: str):
    """Place a market order. action = 'BUY' or 'SELL'."""
    from ib_async import Stock, MarketOrder
    
    contract = Stock(symbol, "SMART", "USD")
    order = MarketOrder(action, abs(qty))
    
    trade = ib.placeOrder(contract, order)
    logger.info("Order placed: %s %d %s", action, abs(qty), symbol)
    
    # Wait for fill (up to 30 seconds)
    for _ in range(60):
        await asyncio.sleep(0.5)
        if trade.isDone():
            if trade.orderStatus.status == "Filled":
                logger.info("  Filled: %s %d %s @ $%.2f",
                           action, abs(qty), symbol,
                           trade.orderStatus.avgFillPrice)
                return True
            else:
                logger.warning("  Order status: %s", trade.orderStatus.status)
                return False
    
    logger.warning("  Order timeout for %s %d %s", action, abs(qty), symbol)
    return False


# ─── Main Loop ───────────────────────────────────────────────────────────

async def run_cycle():
    """Execute one trading cycle."""
    state = load_state()
    state["day_counter"] += 1
    day = state["day_counter"]
    
    logger.info("="*60)
    logger.info("Day %d — Starting momentum rotation cycle", day)
    logger.info("="*60)
    
    # 1. Get price data
    close = get_price_history(SYMBOLS, lookback_days=300)
    if len(close) < REGIME_SMA + 5:
        logger.error("Insufficient price data. Skipping cycle.")
        save_state(state)
        return
    
    # Current prices
    current_prices = close.iloc[-1].dropna().to_dict()
    
    # 2. Check regime
    regime = compute_regime(close["SPY"]) if "SPY" in close.columns else "NEUTRAL"
    logger.info("Regime: %s", regime)
    
    # 3. Connect to IBKR
    try:
        ib = await connect_ibkr()
    except Exception as e:
        logger.error("IBKR connection failed: %s", e)
        save_state(state)
        return
    
    try:
        nav = await get_account_nav(ib)
        ibkr_positions = await get_current_positions(ib)
        logger.info("NAV: $%.2f | IBKR positions: %s", nav, ibkr_positions)
        
        # 4. Check exits
        exits = check_exits(state, current_prices)
        for sym, reason in exits:
            logger.info("EXIT: %s — %s", sym, reason)
            if sym in ibkr_positions and ibkr_positions[sym] > 0:
                await place_market_order(ib, sym, ibkr_positions[sym], "SELL")
            if sym in state["positions"]:
                del state["positions"][sym]
        
        # 5. Bear regime → close everything
        if regime == "BEAR":
            logger.info("Bear regime — closing all positions")
            for sym in list(state["positions"].keys()):
                if sym in ibkr_positions and ibkr_positions[sym] > 0:
                    await place_market_order(ib, sym, ibkr_positions[sym], "SELL")
                del state["positions"][sym]
            save_state(state)
            ib.disconnect()
            return
        
        # 6. Rebalance check
        should_rebal = (day - state["last_rebal_day"]) >= REBALANCE_DAYS
        
        if should_rebal:
            logger.info("REBALANCE — day %d (last: %d)", day, state["last_rebal_day"])
            
            # Rank universe
            rankings = rank_universe(close)
            top_picks = rankings[:MAX_POSITIONS]
            target_syms = {sym for sym, _ in top_picks}
            
            logger.info("Top picks: %s", [(s, f"{sc:.4f}") for s, sc in top_picks])
            
            # Close positions not in top picks
            for sym in list(state["positions"].keys()):
                if sym not in target_syms:
                    logger.info("Closing %s (no longer top-ranked)", sym)
                    if sym in ibkr_positions and ibkr_positions[sym] > 0:
                        await place_market_order(ib, sym, ibkr_positions[sym], "SELL")
                    del state["positions"][sym]
            
            # Open new positions
            regime_scale = 1.0 if regime == "BULL" else 0.7
            for sym, score in top_picks:
                if sym in state["positions"]:
                    continue  # Already holding
                if sym not in current_prices or current_prices[sym] <= 0:
                    continue
                
                alloc = nav * POSITION_PCT * regime_scale
                qty = int(alloc / current_prices[sym])
                if qty <= 0:
                    continue
                
                logger.info("Opening %s: %d shares @ $%.2f (score=%.4f)",
                           sym, qty, current_prices[sym], score)
                
                success = await place_market_order(ib, sym, qty, "BUY")
                if success:
                    state["positions"][sym] = {
                        "qty": qty,
                        "cost": current_prices[sym],
                        "entry_day": day,
                        "peak": current_prices[sym],
                    }
            
            state["last_rebal_day"] = day
        else:
            logger.info("No rebalance (next in %d days)",
                        REBALANCE_DAYS - (day - state["last_rebal_day"]))
        
        # Update peaks
        for sym, pos in state["positions"].items():
            if sym in current_prices:
                if current_prices[sym] > pos.get("peak", 0):
                    pos["peak"] = current_prices[sym]
        
        # Summary
        total_value = sum(
            pos["qty"] * current_prices.get(sym, pos["cost"])
            for sym, pos in state["positions"].items()
        )
        logger.info("Positions: %d | Invested: $%.2f | Cash: ~$%.2f",
                    len(state["positions"]), total_value, nav - total_value)
        
        ib.disconnect()
    except Exception as e:
        logger.error("Cycle error: %s", e, exc_info=True)
        try:
            ib.disconnect()
        except:
            pass
    
    save_state(state)
    logger.info("Cycle complete. State saved.")


def main():
    """Entry point — run one cycle."""
    asyncio.run(run_cycle())


if __name__ == "__main__":
    main()
