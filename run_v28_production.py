#!/usr/bin/env python3
"""
V28 Production Trading System - Unified Runner
================================================
Runs BOTH the equity engine and options engine concurrently via asyncio.
Entry point for the systemd service (deploy/v28_trading_bot.service).

Usage:
    python run_v28_production.py --mode=live
    python run_v28_production.py --mode=paper
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

log_file = LOG_DIR / f"v28_production_{datetime.now():%Y%m%d_%H%M%S}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_file)),
    ],
)
logger = logging.getLogger("v28_production")

# ---------------------------------------------------------------------------
# Market hours helpers  (ET)
# ---------------------------------------------------------------------------

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore

ET = ZoneInfo("America/New_York")


def _now_et() -> datetime:
    return datetime.now(ET)


def market_is_open() -> bool:
    """True if current time is within the trading window (9:45 AM – 3:45 PM ET, weekdays)."""
    now = _now_et()
    if now.weekday() >= 5:  # Saturday / Sunday
        return False
    t = now.time()
    from datetime import time as dt_time
    return dt_time(9, 45) <= t <= dt_time(15, 45)


def in_premarket_window() -> bool:
    """True between 9:00 AM and 9:45 AM ET – used for pre-market analysis."""
    now = _now_et()
    if now.weekday() >= 5:
        return False
    t = now.time()
    from datetime import time as dt_time
    return dt_time(9, 0) <= t < dt_time(9, 45)


def seconds_until_premarket() -> float:
    """Seconds until the next 9:00 AM ET."""
    now = _now_et()
    target = now.replace(hour=9, minute=0, second=0, microsecond=0)
    if now.time() >= target.time():
        target += timedelta(days=1)
    # skip weekends
    while target.weekday() >= 5:
        target += timedelta(days=1)
    return (target - now).total_seconds()


# ---------------------------------------------------------------------------
# Engine wrappers
# ---------------------------------------------------------------------------

class EquityEngine:
    """Async wrapper around EnhancedTradingEngine."""

    def __init__(self, mode: str):
        self.mode = mode
        self.engine = None
        self.logger = logging.getLogger("equity_engine")

    async def initialize(self):
        from src.enhanced_trading_engine import EnhancedTradingEngine, EngineConfig
        self.engine = EnhancedTradingEngine(EngineConfig())
        # SignalAggregator, CAPM, GARCH, ML are all wired inside EnhancedTradingEngine.__init__
        self.logger.info(f"Equity engine initialized (mode={self.mode})")

    async def run_cycle(self, symbols: list[str]):
        """Run a single equity trading cycle."""
        if self.engine is None:
            return

        for symbol in symbols:
            try:
                decision = self.engine.analyze(symbol)
                if decision and decision.is_tradeable:
                    self.logger.info(
                        f"EQUITY SIGNAL: {symbol} → {decision.signal.name} "
                        f"(confidence={decision.confidence:.2f}, "
                        f"combined={decision.combined_score:.2f})"
                    )
                    if self.mode == "live":
                        # Execute via Alpaca
                        await self._execute_equity_trade(decision)
                    else:
                        self.logger.info(f"[PAPER] Would trade {symbol}: {decision.signal.name}")
            except Exception as exc:
                self.logger.error(f"Equity cycle error for {symbol}: {exc}", exc_info=True)

    async def _execute_equity_trade(self, decision):
        """Place equity order via Alpaca REST."""
        try:
            from src.trading.alpaca_client import AlpacaClient, OrderSide
            client = AlpacaClient()
            side = OrderSide.BUY if decision.signal.name in ("STRONG_BUY", "BUY") else OrderSide.SELL
            result = client.submit_order(
                symbol=decision.symbol,
                qty=max(1, int(decision.recommended_quantity)),
                side=side,
            )
            self.logger.info(f"Equity order submitted: {result}")
        except Exception as exc:
            self.logger.error(f"Equity execution failed: {exc}", exc_info=True)


class OptionsEngine:
    """Async wrapper around the autonomous options engine."""

    def __init__(self, mode: str):
        self.mode = mode
        self.engine = None
        self.logger = logging.getLogger("options_engine")

    async def initialize(self):
        paper = self.mode == "paper"
        try:
            from src.options.autonomous_engine import AutonomousTradingEngine
            portfolio_value = float(os.getenv("PORTFOLIO_VALUE", "100000"))
            self.engine = AutonomousTradingEngine(
                portfolio_value=portfolio_value,
                paper=paper or True,  # always paper until explicitly live
            )
            self.logger.info(f"Options engine initialized (paper={paper}, portfolio=${portfolio_value:,.0f})")
        except Exception as exc:
            self.logger.error(f"Options engine init failed: {exc}", exc_info=True)

    async def run_forever(self):
        """Delegate to the autonomous engine's own run_forever."""
        if self.engine is None:
            self.logger.error("Options engine not initialized - skipping")
            return
        try:
            await self.engine.run_forever()
        except asyncio.CancelledError:
            self.logger.info("Options engine cancelled")
        except Exception as exc:
            self.logger.error(f"Options engine fatal: {exc}", exc_info=True)


# ---------------------------------------------------------------------------
# Pre-market analysis
# ---------------------------------------------------------------------------

async def run_premarket_analysis():
    """Run pre-market analysis: GARCH vol forecast, CAPM screening, regime detection."""
    logger.info("=== PRE-MARKET ANALYSIS ===")
    try:
        from src.quant_models.garch import GARCHModel
        garch = GARCHModel()
        for sym in ["SPY", "QQQ", "IWM"]:
            try:
                forecast = garch.fit_and_forecast(sym, horizon=5)
                logger.info(f"GARCH vol forecast {sym}: {forecast}")
            except Exception as e:
                logger.warning(f"GARCH forecast failed for {sym}: {e}")
    except ImportError:
        logger.warning("GARCH model not available for pre-market")

    try:
        from src.quant_models.capm import CAPMModel
        capm = CAPMModel()
        logger.info("CAPM screening in pre-market...")
    except ImportError:
        logger.warning("CAPM model not available for pre-market")

    logger.info("Pre-market analysis complete")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

EQUITY_UNIVERSE = [
    "SPY", "QQQ", "IWM", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "JPM", "V", "UNH", "XOM", "JNJ",
]

EQUITY_CYCLE_INTERVAL = 300  # 5 minutes
OPTIONS_CYCLE_INTERVAL = 300  # 5 minutes


async def equity_loop(engine: EquityEngine, stop_event: asyncio.Event):
    """Run equity engine in a loop during market hours."""
    while not stop_event.is_set():
        if market_is_open():
            logger.info("--- Equity Cycle Start ---")
            await engine.run_cycle(EQUITY_UNIVERSE)
            logger.info("--- Equity Cycle End ---")
        else:
            logger.debug("Market closed – equity engine sleeping")

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=EQUITY_CYCLE_INTERVAL)
            break  # stop_event set
        except asyncio.TimeoutError:
            pass  # normal timeout – next cycle


async def main(mode: str):
    logger.info("=" * 70)
    logger.info(f"  V28 PRODUCTION TRADING SYSTEM — mode={mode}")
    logger.info(f"  PID={os.getpid()}  Python={sys.version.split()[0]}")
    logger.info(f"  Time (ET): {_now_et():%Y-%m-%d %H:%M:%S %Z}")
    logger.info("=" * 70)

    stop_event = asyncio.Event()

    # Graceful shutdown on SIGTERM / SIGINT
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda s=sig: _handle_signal(s, stop_event))

    # Initialize engines
    equity = EquityEngine(mode)
    options = OptionsEngine(mode)

    await equity.initialize()
    await options.initialize()

    # Pre-market analysis if in window
    if in_premarket_window():
        await run_premarket_analysis()
    elif not market_is_open():
        wait_secs = seconds_until_premarket()
        if wait_secs < 14400:  # less than 4 hours
            logger.info(f"Waiting {wait_secs/60:.0f} min until pre-market window...")
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=wait_secs)
                logger.info("Shutdown requested during wait")
                return
            except asyncio.TimeoutError:
                await run_premarket_analysis()
        else:
            logger.info(f"Next pre-market in {wait_secs/3600:.1f}h — running analysis now for testing")
            await run_premarket_analysis()

    # Launch both engines concurrently
    tasks = [
        asyncio.create_task(equity_loop(equity, stop_event), name="equity"),
        asyncio.create_task(options.run_forever(), name="options"),
    ]

    logger.info("Both engines running. Ctrl+C or SIGTERM to stop.")

    # Wait until stop or any task finishes
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
    for t in done:
        if t.exception():
            logger.error(f"Task {t.get_name()} failed: {t.exception()}", exc_info=t.exception())

    # Cancel remaining
    stop_event.set()
    for t in pending:
        t.cancel()
    if pending:
        await asyncio.wait(pending, timeout=10)

    logger.info("V28 Production System shutdown complete.")


def _handle_signal(sig, stop_event: asyncio.Event):
    logger.info(f"Received signal {sig.name} — initiating graceful shutdown")
    stop_event.set()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V28 Production Trading System")
    parser.add_argument("--mode", choices=["live", "paper"], default="paper",
                        help="Trading mode (default: paper)")
    args = parser.parse_args()

    try:
        asyncio.run(main(args.mode))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as exc:
        logger.critical(f"Fatal error: {exc}", exc_info=True)
        sys.exit(1)
