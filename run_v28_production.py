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
# Safety modules — circuit breaker, regime filter, sector caps, process lock
# ---------------------------------------------------------------------------

try:
    from src.risk.trading_gate import check_trading_allowed
    HAS_TRADING_GATE = True
except ImportError:
    HAS_TRADING_GATE = False

try:
    from src.risk.regime_filter import is_bullish_regime, get_position_scale
    HAS_REGIME_FILTER = True
except ImportError:
    HAS_REGIME_FILTER = False

try:
    from src.risk.sector_caps import sector_allows_trade
    HAS_SECTOR_CAPS = True
except ImportError:
    HAS_SECTOR_CAPS = False

try:
    from src.risk.process_lock import acquire_trading_lock, release_trading_lock
    HAS_PROCESS_LOCK = True
except ImportError:
    HAS_PROCESS_LOCK = False

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

# NYSE holidays (2025-2027) — market closed, no trading
_NYSE_HOLIDAYS = {
    # 2025
    (2025, 1, 1), (2025, 1, 20), (2025, 2, 17), (2025, 4, 18),
    (2025, 5, 26), (2025, 6, 19), (2025, 7, 4), (2025, 9, 1),
    (2025, 11, 27), (2025, 12, 25),
    # 2026
    (2026, 1, 1), (2026, 1, 19), (2026, 2, 16), (2026, 4, 3),
    (2026, 5, 25), (2026, 6, 19), (2026, 7, 3), (2026, 9, 7),
    (2026, 11, 26), (2026, 12, 25),
    # 2027
    (2027, 1, 1), (2027, 1, 18), (2027, 2, 15), (2027, 3, 26),
    (2027, 5, 31), (2027, 6, 18), (2027, 7, 5), (2027, 9, 6),
    (2027, 11, 25), (2027, 12, 24),
}


def _is_nyse_holiday(dt: datetime) -> bool:
    return (dt.year, dt.month, dt.day) in _NYSE_HOLIDAYS


def _now_et() -> datetime:
    return datetime.now(ET)


def market_is_open() -> bool:
    """True if current time is within the trading window (9:45 AM – 3:45 PM ET, weekdays, non-holidays)."""
    now = _now_et()
    if now.weekday() >= 5:  # Saturday / Sunday
        return False
    if _is_nyse_holiday(now):
        return False
    t = now.time()
    from datetime import time as dt_time
    return dt_time(9, 45) <= t <= dt_time(15, 45)


def in_premarket_window() -> bool:
    """True between 9:00 AM and 9:45 AM ET – used for pre-market analysis."""
    now = _now_et()
    if now.weekday() >= 5:
        return False
    if _is_nyse_holiday(now):
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
        self.client = None  # reusable AlpacaClient (issue #14)
        self.logger = logging.getLogger("equity_engine")

    async def initialize(self):
        from src.enhanced_trading_engine import EnhancedTradingEngine, EngineConfig
        self.engine = EnhancedTradingEngine(EngineConfig())
        # Pre-create the AlpacaClient once instead of per-trade (issue #14)
        if self.mode == "live":
            from src.trading.alpaca_client import AlpacaClient
            self.client = AlpacaClient()
        self.logger.info(f"Equity engine initialized (mode={self.mode})")

    async def run_cycle(self, symbols: list[str]):
        """Run a single equity trading cycle."""
        if self.engine is None:
            return

        # Circuit breaker gate (issue #3)
        if HAS_TRADING_GATE:
            allowed, reason = check_trading_allowed()
            if not allowed:
                self.logger.warning(f"⚠️ CIRCUIT BREAKER: {reason} — skipping equity cycle")
                return

        # Regime filter: skip new longs in bear markets (issue #3)
        regime_scale = 1.0
        skip_buys = False
        if HAS_REGIME_FILTER:
            if not is_bullish_regime():
                skip_buys = True
                self.logger.info("📉 Bear regime — skipping new long entries")
            else:
                regime_scale = get_position_scale()

        # Build current position map for sector cap checks
        pos_values: dict[str, float] = {}
        equity = 100_000.0
        if self.client:
            try:
                acct = self.client.get_account()
                equity = acct.equity
                for p in self.client.get_positions():
                    pos_values[p.symbol] = abs(p.market_value)
            except Exception:
                pass

        # ── Position monitoring: check existing equity positions for SL/TP ──
        if self.client:
            await self._monitor_equity_positions(equity)

        for symbol in symbols:
            try:
                decision = self.engine.analyze(symbol)
                if decision and decision.is_tradeable:
                    is_buy = decision.signal.name in ("STRONG_BUY", "BUY")

                    # Skip buys in bear regime
                    if is_buy and skip_buys:
                        self.logger.info(f"[REGIME] Skipping BUY {symbol} — bear market")
                        continue

                    # Sector cap check (issue #3)
                    if is_buy and HAS_SECTOR_CAPS:
                        cost = decision.recommended_quantity * decision.entry_price
                        allowed, cap_reason = sector_allows_trade(symbol, cost, pos_values, equity)
                        if not allowed:
                            self.logger.info(f"🚫 Sector cap: {cap_reason}")
                            continue

                    self.logger.info(
                        f"EQUITY SIGNAL: {symbol} → {decision.signal.name} "
                        f"(confidence={decision.confidence:.2f}, "
                        f"combined={decision.combined_score:.2f})"
                    )
                    if self.mode == "live":
                        await self._execute_equity_trade(decision, regime_scale)
                    else:
                        self.logger.info(f"[PAPER] Would trade {symbol}: {decision.signal.name}")
            except Exception as exc:
                self.logger.error(f"Equity cycle error for {symbol}: {exc}", exc_info=True)

    # ── Position monitoring thresholds ──
    EQUITY_STOP_LOSS_PCT = -0.05    # -5% unrealized P&L → close
    EQUITY_TAKE_PROFIT_PCT = 0.10   # +10% unrealized P&L → close

    async def _monitor_equity_positions(self, equity: float):
        """Monitor existing equity positions and close on SL/TP thresholds."""
        try:
            positions = self.client.get_positions()
        except Exception as exc:
            self.logger.error(f"Failed to fetch positions for monitoring: {exc}")
            return

        for pos in positions:
            # Skip option positions (handled by OptionsEngine)
            if len(pos.symbol) > 6 or any(ch.isdigit() for ch in pos.symbol[:4]):
                continue

            try:
                unrealized_pnl = float(pos.unrealized_pl)
                cost_basis = abs(float(pos.cost_basis))
                if cost_basis <= 0:
                    continue

                pnl_pct = unrealized_pnl / cost_basis

                if pnl_pct <= self.EQUITY_STOP_LOSS_PCT:
                    self.logger.warning(
                        f"🛑 EQUITY STOP-LOSS: {pos.symbol} "
                        f"P&L ${unrealized_pnl:+,.2f} ({pnl_pct:+.1%}) — closing"
                    )
                    self.client.close_position(pos.symbol)
                elif pnl_pct >= self.EQUITY_TAKE_PROFIT_PCT:
                    self.logger.info(
                        f"🎯 EQUITY TAKE-PROFIT: {pos.symbol} "
                        f"P&L ${unrealized_pnl:+,.2f} ({pnl_pct:+.1%}) — closing"
                    )
                    self.client.close_position(pos.symbol)
                else:
                    self.logger.debug(
                        f"  Equity holding {pos.symbol}: {float(pos.qty):.0f} sh, "
                        f"P&L ${unrealized_pnl:+,.2f} ({pnl_pct:+.1%})"
                    )
            except Exception as exc:
                self.logger.error(f"Error monitoring {pos.symbol}: {exc}")

    async def _execute_equity_trade(self, decision, regime_scale: float = 1.0):
        """Place equity order via Alpaca REST — limit order with bracket stop/TP."""
        if self.client is None:
            return
        try:
            from src.trading.alpaca_client import OrderSide, OrderType

            side = OrderSide.BUY if decision.signal.name in ("STRONG_BUY", "BUY") else OrderSide.SELL
            qty = max(1, int(decision.recommended_quantity * regime_scale))

            # Get current quote for limit price (issue #2)
            quote = self.client.get_latest_quote(decision.symbol)
            if side == OrderSide.BUY:
                limit_price = round(quote["ask"] * 1.001, 2)  # slightly above ask
            else:
                limit_price = round(quote["bid"] * 0.999, 2)  # slightly below bid

            if limit_price <= 0:
                self.logger.warning(f"Bad quote for {decision.symbol} — skipping")
                return

            # Place bracket order with stop-loss and take-profit (issue #11)
            if side == OrderSide.BUY and decision.stop_loss and decision.take_profits:
                stop_price = round(decision.stop_loss, 2)
                tp_price = round(decision.take_profits[0] if decision.take_profits else limit_price * 1.04, 2)
                order_data = {
                    "symbol": decision.symbol,
                    "qty": str(qty),
                    "side": "buy",
                    "type": "limit",
                    "time_in_force": "day",
                    "order_class": "bracket",
                    "limit_price": str(limit_price),
                    "stop_loss": {"stop_price": str(stop_price)},
                    "take_profit": {"limit_price": str(tp_price)},
                }
                data = self.client._request("POST", "/v2/orders", data=order_data)
                self.logger.info(
                    f"Bracket order submitted: {decision.symbol} {qty}sh "
                    f"limit=${limit_price} SL=${stop_price} TP=${tp_price} → {data.get('id', '?')}"
                )
            else:
                # Simple limit order for sells or if no stop/TP available
                result = self.client.submit_order(
                    symbol=decision.symbol,
                    qty=qty,
                    side=side,
                    order_type=OrderType.LIMIT,
                    limit_price=limit_price,
                )
                self.logger.info(f"Limit order submitted: {result}")
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
                paper=paper,  # respect --mode flag
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
    # Broad Market ETFs
    "SPY", "QQQ", "IWM", "DIA",
    # Technology (~20%)
    "AAPL", "MSFT", "GOOGL", "NVDA", "AMD", "CRM", "ADBE", "ORCL",
    # Consumer / Communication
    "AMZN", "META", "TSLA", "NFLX", "DIS",
    # Financials
    "JPM", "V", "GS", "MA", "BAC",
    # Healthcare
    "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK",
    # Energy
    "XOM", "CVX", "COP", "SLB",
    # Industrials
    "CAT", "HON", "UPS", "GE", "RTX", "DE",
    # Consumer Staples
    "PG", "KO", "PEP", "COST", "WMT",
    # Utilities / REITs
    "NEE", "SO", "AMT",
    # Materials
    "LIN", "FCX", "NEM",
    # Semiconductors (separate from broad tech)
    "AVGO", "QCOM",
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

    # Acquire process lock (issue #3)
    if HAS_PROCESS_LOCK:
        if not acquire_trading_lock("v28_production"):
            logger.error("❌ Could not acquire trading lock. Another bot may be running.")
            sys.exit(1)
        logger.info("✅ Trading lock acquired")

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
