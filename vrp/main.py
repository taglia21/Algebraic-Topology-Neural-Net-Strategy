"""
vrp/main.py
===========
VRP Alpha Engine — Main Orchestrator.

Single entry point for all operating modes:
  python -m vrp.main --mode backtest --start 2020-01-01 --end 2025-12-31
  python -m vrp.main --mode paper
  python -m vrp.main --mode live

In backtest mode, runs the full historical simulation.
In paper/live mode, connects to IBKR and runs the trading loop.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from datetime import date, datetime, timedelta
from typing import Optional

from vrp.config import Config, get_config
from vrp.strategy import VRPStrategy, TradeAction
from vrp.risk import RiskManager
from vrp.utils import setup_logger

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Live/Paper Trading Loop
# ---------------------------------------------------------------------------

async def run_live(config: Config) -> None:
    """Run the live or paper trading loop.

    Connects to IBKR, then enters a loop:
    1. Check market hours
    2. Get current SPX price and VIX
    3. Mark positions to market
    4. Evaluate exits (profit/stop/time)
    5. Check for new entry signals
    6. Execute any trades
    7. Sleep until next cycle
    """
    from vrp.broker import IBKRBroker
    from vrp.strategy import VRPStrategy

    mode = config.mode.upper()
    print(f"\n{'='*60}")
    print(f"  VRP ALPHA ENGINE — {mode} TRADING")
    print(f"{'='*60}")
    print(f"  IBKR: {config.ibkr.host}:{config.ibkr.port}")
    print(f"  Account: {config.ibkr.account or 'auto'}")
    print(f"{'='*60}\n")

    broker = IBKRBroker(config.ibkr)
    strategy = VRPStrategy(config)
    risk_mgr = RiskManager(config.risk)

    # Connect to IBKR
    connected = await broker.connect()
    if not connected:
        print("ERROR: Failed to connect to IBKR. Exiting.")
        return

    cycle = 0

    try:
        while True:
            cycle += 1
            logger.info(f"--- Cycle {cycle} ---")

            try:
                # Get market data
                spx_price = await broker.get_spx_price()
                vix = await broker.get_vix()

                if spx_price is None or vix is None:
                    logger.warning("Missing market data, retrying in 60s")
                    await asyncio.sleep(60)
                    continue

                # Get account state
                account = await broker.get_account_summary()
                if account is None:
                    logger.warning("Failed to get account summary")
                    await asyncio.sleep(60)
                    continue

                equity = account.equity

                # Update risk state
                greeks = strategy.portfolio_greeks
                risk_state = risk_mgr.update(
                    equity=equity,
                    positions=strategy.open_positions,
                    portfolio_greeks=greeks,
                )

                if not risk_state.is_trading_allowed:
                    logger.warning(f"Trading halted: {risk_state.halt_reason}")
                    print(
                        f"  [cycle {cycle}] HALTED: {risk_state.halt_reason} | "
                        f"equity=${equity:,.0f}"
                    )
                    await asyncio.sleep(300)
                    continue

                # Mark positions to market and evaluate exits
                iv = vix / 100.0
                actions = strategy.evaluate_positions(spx_price, vix, iv)

                for pos, action in actions:
                    if action in (
                        TradeAction.CLOSE_PROFIT,
                        TradeAction.CLOSE_STOP,
                        TradeAction.CLOSE_EXPIRY,
                    ):
                        # Place close order via broker
                        order_id = await broker.close_spread(
                            short_strike=pos.short_leg.strike,
                            long_strike=pos.long_leg.strike,
                            expiry=pos.short_leg.expiry,
                            quantity=pos.quantity,
                        )
                        if order_id:
                            strategy.close_position(pos, action.value)
                            logger.info(f"Close order {order_id} for {pos.id}")

                # Check for new entries
                if strategy.should_open_new_trade(spx_price, vix):
                    new_pos = strategy.construct_spread(
                        spx_price=spx_price,
                        vix=vix,
                        account_equity=equity,
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
                            from vrp.broker import SpreadOrder
                            order = SpreadOrder(
                                short_strike=new_pos.short_leg.strike,
                                long_strike=new_pos.long_leg.strike,
                                expiry=new_pos.short_leg.expiry,
                                quantity=new_pos.quantity,
                                limit_price=new_pos.entry_credit / 100.0,  # per-share
                            )
                            order_id = await broker.place_spread(order)
                            if order_id:
                                logger.info(f"Entry order {order_id} for {new_pos.id}")
                        else:
                            logger.info(f"Trade rejected by risk: {reason}")
                            # Remove the constructed position
                            strategy.positions.remove(new_pos)

                # Print status
                n_open = len(strategy.open_positions)
                unrealized = sum(p.current_pnl for p in strategy.open_positions)
                print(
                    f"  [cycle {cycle}] equity=${equity:>10,.0f} | "
                    f"open={n_open} | "
                    f"unrealized=${unrealized:>+8,.0f} | "
                    f"DD={risk_state.drawdown:>+6.1%} | "
                    f"SPX={spx_price:.0f} VIX={vix:.1f}"
                )

            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.error(f"Cycle {cycle} error: {e}", exc_info=True)

            # Sleep between cycles (5 minutes)
            await asyncio.sleep(300)

    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        await broker.disconnect()
        print("Disconnected from IBKR.")


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
        datefmt="%H:%M:%S",
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
