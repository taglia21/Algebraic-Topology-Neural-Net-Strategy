"""
vrp/main.py
===========
VRP Alpha Engine — Main Orchestrator.

Single entry point for all operating modes:
  python -m vrp.main --mode backtest --start 2020-01-01 --end 2025-12-31
  python -m vrp.main --mode paper
  python -m vrp.main --mode live

Production features:
- Market hours awareness (only trades 9:45 AM - 3:45 PM ET)
- IBKR reconnection with exponential backoff
- State persistence (positions survive restarts)
- Position reconciliation on startup (syncs with IBKR)
- Proper order fill monitoring
- Graceful shutdown with state save
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from vrp.config import Config, get_config
from vrp.strategy import VRPStrategy, TradeAction, SpreadPosition, SpreadLeg
from vrp.risk import RiskManager
from vrp.utils import setup_logger

logger = logging.getLogger(__name__)

# State file location
STATE_DIR = Path(os.environ.get("VRP_STATE_DIR", "state"))
STATE_FILE = STATE_DIR / "vrp_state.json"
TRADE_LOG = STATE_DIR / "vrp_trades.jsonl"

# Market hours (Eastern Time)
# SPX options trade 9:30-4:15 ET but we buffer 15 min on each side
MARKET_OPEN_HOUR, MARKET_OPEN_MIN = 9, 45
MARKET_CLOSE_HOUR, MARKET_CLOSE_MIN = 15, 45

# Cycle interval
CYCLE_SECONDS = 300  # 5 minutes between cycles
OVERNIGHT_SECONDS = 60  # check every minute outside hours for market open


# ---------------------------------------------------------------------------
# Timezone helpers
# ---------------------------------------------------------------------------

def _now_et() -> datetime:
    """Get current time in US/Eastern."""
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("America/New_York"))


def is_market_hours() -> bool:
    """Check if we're within tradable market hours (9:45 AM - 3:45 PM ET, weekdays)."""
    now = _now_et()
    if now.weekday() >= 5:  # Saturday or Sunday
        return False
    market_open = now.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MIN, second=0)
    market_close = now.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MIN, second=0)
    return market_open <= now <= market_close


def seconds_until_market_open() -> int:
    """Seconds until next market open (for sleep scheduling)."""
    now = _now_et()
    # Find next weekday
    target = now.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MIN, second=0, microsecond=0)
    if now >= target or now.weekday() >= 5:
        # Move to next day
        target += timedelta(days=1)
    while target.weekday() >= 5:
        target += timedelta(days=1)
    diff = (target - now).total_seconds()
    return max(60, int(diff))


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def save_state(strategy: VRPStrategy, equity: float, hwm: float) -> None:
    """Save strategy state to disk for crash recovery."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "equity": equity,
        "high_water_mark": hwm,
        "next_id": strategy._next_id,
        "positions": [],
    }
    for pos in strategy.positions:
        state["positions"].append({
            "id": pos.id,
            "short_strike": pos.short_leg.strike,
            "long_strike": pos.long_leg.strike,
            "expiry": pos.short_leg.expiry.isoformat(),
            "entry_date": pos.entry_date.isoformat(),
            "entry_credit": pos.entry_credit,
            "quantity": pos.quantity,
            "current_value": pos.current_value,
            "spx_at_entry": pos.spx_at_entry,
            "vix_at_entry": pos.vix_at_entry,
            "status": pos.status,
            "close_date": pos.close_date.isoformat() if pos.close_date else None,
            "close_pnl": pos.close_pnl,
            "close_reason": pos.close_reason,
        })
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
    logger.debug(f"State saved: {len(strategy.positions)} positions")


def load_state(strategy: VRPStrategy) -> Tuple[float, float]:
    """Load strategy state from disk. Returns (equity, high_water_mark)."""
    if not STATE_FILE.exists():
        logger.info("No saved state found — starting fresh")
        return 0.0, 0.0

    try:
        with open(STATE_FILE) as f:
            state = json.load(f)

        strategy._next_id = state.get("next_id", 0)
        strategy.positions = []

        for p in state.get("positions", []):
            expiry = date.fromisoformat(p["expiry"])
            pos = SpreadPosition(
                id=p["id"],
                short_leg=SpreadLeg(
                    strike=p["short_strike"],
                    expiry=expiry,
                    side="sell",
                    premium=0,
                ),
                long_leg=SpreadLeg(
                    strike=p["long_strike"],
                    expiry=expiry,
                    side="buy",
                    premium=0,
                ),
                entry_date=date.fromisoformat(p["entry_date"]),
                entry_credit=p["entry_credit"],
                quantity=p["quantity"],
                current_value=p.get("current_value", 0),
                spx_at_entry=p.get("spx_at_entry", 0),
                vix_at_entry=p.get("vix_at_entry", 0),
                status=p.get("status", "open"),
                close_date=date.fromisoformat(p["close_date"]) if p.get("close_date") else None,
                close_pnl=p.get("close_pnl", 0),
                close_reason=p.get("close_reason", ""),
            )
            strategy.positions.append(pos)

        equity = state.get("equity", 0)
        hwm = state.get("high_water_mark", equity)
        n_open = len([p for p in strategy.positions if p.status == "open"])
        logger.info(
            f"Loaded state: {n_open} open / {len(strategy.positions)} total positions, "
            f"equity=${equity:,.0f}, HWM=${hwm:,.0f}"
        )
        return equity, hwm

    except Exception as e:
        logger.error(f"Failed to load state: {e}")
        return 0.0, 0.0


def log_trade(pos: SpreadPosition, action: str) -> None:
    """Append a trade to the JSONL trade log."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "id": pos.id,
        "short_strike": pos.short_leg.strike,
        "long_strike": pos.long_leg.strike,
        "expiry": pos.short_leg.expiry.isoformat(),
        "entry_credit": pos.entry_credit,
        "quantity": pos.quantity,
        "close_pnl": pos.close_pnl,
        "close_reason": pos.close_reason,
        "spx_at_entry": pos.spx_at_entry,
        "vix_at_entry": pos.vix_at_entry,
    }
    with open(TRADE_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# IBKR connection with reconnection
# ---------------------------------------------------------------------------

async def connect_with_retry(broker, max_retries: int = 10) -> bool:
    """Connect to IBKR with exponential backoff."""
    for attempt in range(max_retries):
        try:
            connected = await broker.connect()
            if connected:
                logger.info(f"Connected to IBKR (attempt {attempt + 1})")
                return True
        except Exception as e:
            logger.warning(f"Connection attempt {attempt + 1} failed: {e}")

        wait = min(30 * (2 ** attempt), 300)  # 30s, 60s, 120s, 240s, 300s cap
        logger.info(f"Retrying in {wait}s...")
        await asyncio.sleep(wait)

    logger.error(f"Failed to connect after {max_retries} attempts")
    return False


# ---------------------------------------------------------------------------
# Position reconciliation
# ---------------------------------------------------------------------------

async def reconcile_positions(
    broker,
    strategy: VRPStrategy,
) -> None:
    """Reconcile strategy positions with IBKR actual positions.

    On startup, we need to make sure our in-memory positions match
    what IBKR actually has. If IBKR shows positions we don't know about,
    log a warning. If we think we have positions that IBKR doesn't, clean up.
    """
    try:
        ibkr_positions = await broker.get_positions()
        if not ibkr_positions:
            logger.info("No IBKR positions found — strategy state is authoritative")
            return

        # Build a set of IBKR option positions
        ibkr_puts = {}
        for p in ibkr_positions:
            if p.get("secType") == "OPT" and p.get("right") == "P":
                key = (p["strike"], p.get("expiry", ""))
                ibkr_puts[key] = p.get("quantity", 0)

        # Check strategy positions against IBKR
        for pos in strategy.open_positions:
            short_key = (pos.short_leg.strike, pos.short_leg.expiry.strftime("%Y%m%d"))
            long_key = (pos.long_leg.strike, pos.long_leg.expiry.strftime("%Y%m%d"))

            has_short = short_key in ibkr_puts
            has_long = long_key in ibkr_puts

            if has_short and has_long:
                logger.info(f"Position {pos.id} confirmed in IBKR")
            elif not has_short and not has_long:
                logger.warning(
                    f"Position {pos.id} NOT found in IBKR — marking as closed "
                    f"(may have been closed manually or expired)"
                )
                pos.status = "closed"
                pos.close_reason = "reconciliation_missing"
                pos.close_date = date.today()
            else:
                logger.warning(
                    f"Position {pos.id} partially found in IBKR "
                    f"(short={'yes' if has_short else 'no'}, long={'yes' if has_long else 'no'})"
                )

        n_open = len(strategy.open_positions)
        logger.info(f"Reconciliation complete: {n_open} open positions after sync")

    except Exception as e:
        logger.error(f"Position reconciliation failed: {e}")
        logger.info("Continuing with saved state as authoritative")


# ---------------------------------------------------------------------------
# Live/Paper Trading Loop
# ---------------------------------------------------------------------------

async def run_live(config: Config) -> None:
    """Run the live or paper trading loop.

    Production-grade loop with:
    - Market hours awareness
    - IBKR reconnection
    - State persistence
    - Position reconciliation
    - Order fill monitoring
    - Graceful shutdown
    """
    from vrp.broker import IBKRBroker, SpreadOrder

    mode = config.mode.upper()
    print(f"\n{'='*60}")
    print(f"  VRP ALPHA ENGINE — {mode} TRADING")
    print(f"{'='*60}")
    print(f"  IBKR: {config.ibkr.host}:{config.ibkr.port}")
    print(f"  Account: {config.ibkr.account or 'auto'}")
    print(f"  State: {STATE_FILE}")
    print(f"{'='*60}\n")

    broker = IBKRBroker(config.ibkr)
    strategy = VRPStrategy(config)
    risk_mgr = RiskManager(config.risk)

    # Load saved state
    saved_equity, saved_hwm = load_state(strategy)

    # Connect to IBKR
    connected = await connect_with_retry(broker)
    if not connected:
        print("FATAL: Failed to connect to IBKR after all retries. Exiting.")
        return

    # Get initial account state
    account = await broker.get_account_summary()
    if account:
        equity = account.equity
        if saved_hwm > 0:
            # Use the higher of saved HWM and current equity
            risk_mgr._high_water_mark = max(saved_hwm, equity)
        else:
            risk_mgr._high_water_mark = equity
        risk_mgr._day_start_equity = equity
        logger.info(f"Account equity: ${equity:,.2f}")
    else:
        equity = saved_equity or config.backtest.initial_capital
        logger.warning(f"Could not get account data — using ${equity:,.0f}")

    # Reconcile positions with IBKR
    await reconcile_positions(broker, strategy)

    # Save state after reconciliation
    save_state(strategy, equity, risk_mgr._high_water_mark)

    # Shutdown handler
    shutdown_requested = False

    def handle_shutdown(signum, frame):
        nonlocal shutdown_requested
        shutdown_requested = True
        logger.info(f"Shutdown signal received ({signum})")

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    cycle = 0

    try:
        while not shutdown_requested:
            cycle += 1

            # ---- Market hours check ----
            if not is_market_hours():
                now = _now_et()
                if cycle == 1 or now.minute == 0:  # log once per hour
                    print(
                        f"  [{now.strftime('%H:%M ET')}] Market closed — "
                        f"next check in {seconds_until_market_open() // 3600:.0f}h"
                    )
                await asyncio.sleep(OVERNIGHT_SECONDS)
                continue

            # ---- Ensure IBKR connection ----
            if not broker.is_connected:
                logger.warning("IBKR disconnected — reconnecting...")
                connected = await connect_with_retry(broker, max_retries=5)
                if not connected:
                    logger.error("Reconnection failed — sleeping 5 min")
                    await asyncio.sleep(300)
                    continue

            try:
                # ---- Get market data ----
                spx_price = await broker.get_spx_price()
                vix = await broker.get_vix()

                if spx_price is None or vix is None:
                    logger.warning("Missing market data, retrying in 60s")
                    await asyncio.sleep(60)
                    continue

                # ---- Get account state ----
                account = await broker.get_account_summary()
                if account:
                    equity = account.equity
                else:
                    logger.warning("Failed to get account data — using last known equity")

                # ---- Update risk state ----
                greeks = strategy.portfolio_greeks
                risk_state = risk_mgr.update(
                    equity=equity,
                    positions=strategy.open_positions,
                    portfolio_greeks=greeks,
                    as_of=date.today(),
                )

                if not risk_state.is_trading_allowed:
                    logger.warning(f"Trading halted: {risk_state.halt_reason}")
                    print(
                        f"  [cycle {cycle}] HALTED: {risk_state.halt_reason} | "
                        f"equity=${equity:,.0f}"
                    )
                    save_state(strategy, equity, risk_mgr._high_water_mark)
                    await asyncio.sleep(CYCLE_SECONDS)
                    continue

                # ---- Mark positions to market and evaluate exits ----
                iv = vix / 100.0
                actions = strategy.evaluate_positions(
                    spx_price, vix, iv, as_of=date.today(),
                    risk_free_rate=config.backtest.risk_free_rate,
                )

                for pos, action in actions:
                    if action in (
                        TradeAction.CLOSE_PROFIT,
                        TradeAction.CLOSE_STOP,
                        TradeAction.CLOSE_EXPIRY,
                    ):
                        # Calculate limit price for closing
                        # current_value is cost to close per contract in dollars
                        close_limit = pos.current_value / 100.0  # convert to per-share for IBKR

                        order_id = await broker.close_spread(
                            short_strike=pos.short_leg.strike,
                            long_strike=pos.long_leg.strike,
                            expiry=pos.short_leg.expiry,
                            quantity=pos.quantity,
                            limit_price=close_limit if close_limit > 0.05 else None,
                        )
                        if order_id:
                            strategy.close_position(pos, action.value, as_of=date.today())
                            log_trade(pos, f"close_{action.value}")
                            logger.info(
                                f"Closed {pos.id}: {action.value} | "
                                f"P&L ${pos.close_pnl:+,.0f} | order {order_id}"
                            )

                    elif action == TradeAction.ROLL:
                        # Close existing
                        close_limit = pos.current_value / 100.0
                        order_id = await broker.close_spread(
                            short_strike=pos.short_leg.strike,
                            long_strike=pos.long_leg.strike,
                            expiry=pos.short_leg.expiry,
                            quantity=pos.quantity,
                            limit_price=close_limit if close_limit > 0.05 else None,
                        )
                        if order_id:
                            strategy.close_position(pos, "roll", as_of=date.today())
                            log_trade(pos, "roll_close")

                            # Open new position with further expiry
                            new_pos = strategy.construct_spread(
                                spx_price=spx_price,
                                vix=vix,
                                account_equity=equity,
                                as_of=date.today(),
                                risk_free_rate=config.backtest.risk_free_rate,
                            )
                            if new_pos:
                                # Place entry order for roll
                                entry_order = SpreadOrder(
                                    short_strike=new_pos.short_leg.strike,
                                    long_strike=new_pos.long_leg.strike,
                                    expiry=new_pos.short_leg.expiry,
                                    quantity=new_pos.quantity,
                                    limit_price=new_pos.entry_credit / 100.0,
                                )
                                entry_id = await broker.place_spread(entry_order)
                                if entry_id:
                                    log_trade(new_pos, "roll_open")
                                    logger.info(f"Roll: closed {pos.id} → opened {new_pos.id}")
                                else:
                                    # Roll open failed — remove the position
                                    strategy.positions.remove(new_pos)

                # ---- Check for new entries ----
                if strategy.should_open_new_trade(spx_price, vix, as_of=date.today()):
                    # Try to get real option chain for better strike selection
                    available_strikes = None
                    try:
                        from vrp.utils import next_monthly_expiry
                        target_expiry = next_monthly_expiry(date.today())
                        chain = await broker.get_option_chain(
                            target_expiry,
                            strike_range=(spx_price * 0.85, spx_price * 0.99),
                        )
                        if chain:
                            available_strikes = [q.strike for q in chain]
                            logger.info(
                                f"Got {len(chain)} real strikes for {target_expiry}"
                            )
                    except Exception as e:
                        logger.debug(f"Option chain request failed, using BS: {e}")

                    new_pos = strategy.construct_spread(
                        spx_price=spx_price,
                        vix=vix,
                        account_equity=equity,
                        as_of=date.today(),
                        risk_free_rate=config.backtest.risk_free_rate,
                        available_strikes=available_strikes,
                    )

                    if new_pos:
                        # Check risk approval
                        allowed, reason = risk_mgr.can_open_trade(
                            risk_state,
                            proposed_risk=new_pos.total_max_risk,
                            proposed_delta=greeks.get("delta", 0),
                            proposed_vega=greeks.get("vega", 0),
                        )

                        if allowed:
                            # entry_credit is in dollars (e.g. $180)
                            # IBKR limit price is per-share (e.g. $1.80)
                            entry_order = SpreadOrder(
                                short_strike=new_pos.short_leg.strike,
                                long_strike=new_pos.long_leg.strike,
                                expiry=new_pos.short_leg.expiry,
                                quantity=new_pos.quantity,
                                limit_price=new_pos.entry_credit / 100.0,
                            )
                            order_id = await broker.place_spread(entry_order)
                            if order_id:
                                log_trade(new_pos, "open")
                                logger.info(
                                    f"Opened {new_pos.id}: "
                                    f"sell {new_pos.short_leg.strike}P / "
                                    f"buy {new_pos.long_leg.strike}P "
                                    f"x{new_pos.quantity} @ "
                                    f"${new_pos.entry_credit:.0f} credit | "
                                    f"order {order_id}"
                                )
                            else:
                                logger.warning(f"Order placement failed for {new_pos.id}")
                                strategy.positions.remove(new_pos)
                        else:
                            logger.info(f"Trade rejected by risk: {reason}")
                            strategy.positions.remove(new_pos)

                # ---- Save state after every cycle ----
                save_state(strategy, equity, risk_mgr._high_water_mark)

                # ---- Print status ----
                now = _now_et()
                n_open = len(strategy.open_positions)
                unrealized = sum(p.current_pnl for p in strategy.open_positions)
                total_risk = sum(p.total_max_risk for p in strategy.open_positions)
                print(
                    f"  [{now.strftime('%H:%M')}] "
                    f"equity=${equity:>10,.0f} | "
                    f"open={n_open} | "
                    f"risk=${total_risk:>8,.0f} | "
                    f"unreal=${unrealized:>+8,.0f} | "
                    f"DD={risk_state.drawdown:>+6.1%} | "
                    f"SPX={spx_price:.0f} VIX={vix:.1f}"
                )

            except Exception as e:
                logger.error(f"Cycle {cycle} error: {e}", exc_info=True)
                # Save state even on error
                try:
                    save_state(strategy, equity, risk_mgr._high_water_mark)
                except:
                    pass

            await asyncio.sleep(CYCLE_SECONDS)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt")
    finally:
        # Always save state on exit
        try:
            save_state(strategy, equity, risk_mgr._high_water_mark)
            print(f"\nState saved to {STATE_FILE}")
        except:
            pass
        try:
            await broker.disconnect()
        except:
            pass
        print("Disconnected. Goodbye.")


# ---------------------------------------------------------------------------
# Backtest wrapper
# ---------------------------------------------------------------------------

def run_backtest(
    start: str,
    end: str,
    capital: float,
    output: str,
    verbose: bool = True,
) -> None:
    """Run the backtest and save results."""
    from vrp.backtest import VRPBacktester

    config = get_config()
    config.backtest.initial_capital = capital

    bt = VRPBacktester(config)
    metrics = bt.run(start=start, end=end, verbose=verbose)
    bt.save_results(output)

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="VRP Alpha Engine — Systematic SPX Options Trading",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["backtest", "paper", "live"],
        default="backtest", help="Operating mode",
    )
    parser.add_argument("--start", default="2020-01-01", help="Backtest start date")
    parser.add_argument("--end", default="2025-12-31", help="Backtest end date")
    parser.add_argument("--capital", type=float, default=10_000, help="Initial capital")
    parser.add_argument("--output", default="vrp_backtest_results.json", help="Output file")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress")

    args = parser.parse_args()

    setup_logger("vrp", args.log_level)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.mode == "backtest":
        run_backtest(
            start=args.start,
            end=args.end,
            capital=args.capital,
            output=args.output,
            verbose=not args.quiet,
        )
    else:
        config = get_config()
        config.mode = args.mode
        asyncio.run(run_live(config))


if __name__ == "__main__":
    main()
