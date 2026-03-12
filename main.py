"""
ATNN Quant Powerhouse — Main Orchestrator
==========================================
Single entry point for all operating modes: backtest, paper, live.

Usage
-----
    python main.py --mode backtest --start 2022-01-01 --end 2025-12-31
    python main.py --mode backtest --start 2022-01-01 --end 2025-12-31 --ml
    python main.py --mode paper
    python main.py --mode live   # requires IBKR TWS/Gateway running

Environment Variables
---------------------
    IBKR_HOST             — IBKR TWS/Gateway host (default: 127.0.0.1)
    IBKR_PORT             — IBKR TWS/Gateway port (default: 7497)
    IBKR_CLIENT_ID        — IBKR client ID (default: 1)
    IBKR_ACCOUNT          — IBKR account ID (e.g. U22452226)
    SYSTEM_MODE           — backtest | paper | live
    LOG_LEVEL             — DEBUG | INFO | WARNING | ERROR
    PORTFOLIO_VALUE       — Initial portfolio value in USD

Architecture
------------
The SAME :class:`SystemOrchestrator` runs in all modes.  Components differ:

    Mode       Data source          Broker
    ---------  -------------------  ------------------
    backtest   DataManager (hist)   SimulatedBroker
    paper      DataManager (live)   SimulatedBroker
    live       DataManager (live)   IBKRBroker (TODO)

All signal generation, regime detection, risk management, and ML code
is identical across modes — this is the core design guarantee.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from typing import List, Optional

from core.config import get_config
from core.logger import get_trade_logger
from backtest.backtester import Backtester
from backtest.metrics import BacktestResult, PerformanceMetrics

logger = logging.getLogger(__name__)

# Default symbol universe for standalone runs — sourced from config
from core.config import _DEFAULT_SYMBOLS


# ---------------------------------------------------------------------------
# SystemOrchestrator
# ---------------------------------------------------------------------------

class SystemOrchestrator:
    """Ties all ATNN Quant Powerhouse components into a single flow.

    Data → Features → Regime → Signals → ML → Risk → Execution

    The same orchestrator runs in all modes (backtest, paper, live).
    Only the data source and broker implementation change.

    Parameters
    ----------
    mode:
        Operating mode: ``"backtest"``, ``"paper"``, or ``"live"``.
    """

    def __init__(self, mode: str = "backtest") -> None:
        self.mode   = mode.lower()
        self.config = get_config()
        self._log   = get_trade_logger()

        # Override mode in config so all components see the right mode
        self.config.system.mode = self.mode
        self.config.validate()

        logger.info(f"SystemOrchestrator initialised in {self.mode!r} mode.")

    # ------------------------------------------------------------------
    # Backtest
    # ------------------------------------------------------------------

    def run_backtest(
        self,
        start: str,
        end: str,
        symbols: Optional[List[str]] = None,
        initial_capital: float = 100_000.0,
        use_ml: bool = False,
    ) -> BacktestResult:
        """Run a full backtest and print a summary to the terminal.

        Parameters
        ----------
        start:
            ISO-8601 start date (e.g. ``"2022-01-01"``).
        end:
            ISO-8601 end date.
        symbols:
            Trading universe.  Defaults to the built-in top-18 S&P 500 names.
        initial_capital:
            Starting portfolio cash in USD.
        use_ml:
            Whether to enable the ML meta-learner pipeline.

        Returns
        -------
        BacktestResult
        """
        symbols = symbols or _DEFAULT_SYMBOLS
        print("\n" + "=" * 62)
        print("  ATNN QUANT POWERHOUSE — BACKTEST")
        print("=" * 62)
        print(f"  Period   : {start} → {end}")
        print(f"  Universe : {len(symbols)} symbols")
        print(f"  Capital  : ${initial_capital:,.0f}")
        print(f"  ML       : {'enabled' if use_ml else 'disabled'}")
        print("=" * 62 + "\n")

        bt = Backtester(
            config=self.config,
            initial_cash=initial_capital,
            verbose=True,
        )

        t0 = time.time()
        result = bt.run(
            symbols=symbols,
            start_date=start,
            end_date=end,
            use_ml=use_ml,
        )
        elapsed = time.time() - t0

        self.print_summary(result)
        print(f"\n  Runtime: {elapsed:.1f}s\n")
        return result

    # ------------------------------------------------------------------
    # Live / Paper
    # ------------------------------------------------------------------

    def run_live(self) -> None:
        """Run the live or paper trading loop with full production safety.

        Operates in a tight loop during US market hours, executing one
        trading cycle per iteration:

            1. Wait for market open (market hours awareness).
            2. Fetch latest bars from the data provider.
            3. Detect current regime.
            4. Generate signals.
            5. Kill switch / circuit breaker pre-trade check.
            6. Apply risk checks.
            7. Submit orders to the broker.
            8. Reconcile positions against broker.
            9. Sleep until next cycle.

        In paper mode the broker is a :class:`SimulatedBroker`.
        In live mode the broker is :class:`IBKRBroker` (TODO), routing real
        orders to IBKR TWS/Gateway.
        """
        from data.data_manager import DataManager
        from data.cache import DataCache
        from equities.execution import ExecutionManager, SimulatedBroker
        from equities.signal_generator import SignalGenerator
        from core.regime_detector import RegimeDetector
        from core.risk_manager import RiskManager
        from core.market_hours import MarketCalendar
        from core.kill_switch import KillSwitch, CircuitBreakerConfig
        from core.reconciliation import Reconciler

        print(f"\n[{self.mode.upper()} MODE] Starting trading loop ...")

        cfg = self.config
        data_manager = DataManager(mode=self.mode)
        data_cache = DataCache(max_entries=5000)
        regime_detector = RegimeDetector()
        risk_manager = RiskManager(cfg.risk, self._log)
        market_cal = MarketCalendar()

        # --- Select broker based on mode ---
        # TODO: implement IBKRBroker in broker/ package for live trading
        if self.mode == "live" and cfg.ibkr.is_configured():
            raise NotImplementedError(
                "IBKRBroker not yet implemented. Use --mode paper for now."
            )
        else:
            broker = SimulatedBroker(
                initial_cash=cfg.system.initial_portfolio_value,
                slippage_bps=cfg.backtest.slippage_bps,
                commission_per_share=cfg.backtest.commission_per_share,
                trade_logger=self._log,
            )
            logger.info("Using SimulatedBroker.")

        # --- Kill switch + circuit breaker ---
        # Use actual broker equity (not config default) so we don't carry
        # a stale $100K peak from a paper-reset that never happened.
        _broker_equity = getattr(broker, 'equity', None) or cfg.system.initial_portfolio_value
        if hasattr(broker, 'get_account'):
            try:
                _acct = broker.get_account()
                _broker_equity = float(getattr(_acct, 'equity', _broker_equity))
            except Exception:
                pass
        kill_switch = KillSwitch(
            config=CircuitBreakerConfig(
                max_drawdown_pct=-0.99,  # disabled — only daily loss halts trading
                max_daily_loss_pct=-0.08,  # halt if we lose 8% in a single day
            ),
            initial_equity=_broker_equity,
        )

        # --- Reconciler ---
        reconciler = Reconciler(broker=broker, mode="soft")

        # TODO: re-implement strategies in v2 (options-first, equities dormant)
        signal_gen = SignalGenerator(
            strategies=[],
            trade_logger=self._log,
        )
        exec_mgr = ExecutionManager(
            broker=broker,
            risk_manager=risk_manager,
            trade_logger=self._log,
        )

        symbols = cfg.data.symbols
        cycle = 0
        _last_trading_date = None  # tracks current trading day for daily reset

        while True:
            cycle += 1

            # --- Market hours gate ---
            if not market_cal.is_market_open():
                if cycle == 1:
                    next_open = market_cal.next_open()
                    print(
                        f"  Market closed. Next open: "
                        f"{next_open.strftime('%Y-%m-%d %H:%M ET')}",
                        flush=True,
                    )
                time.sleep(60)
                continue

            logger.info(f"Live cycle {cycle}")

            try:
                # --- Daily reset at market open ---
                from datetime import date as _date_cls
                _today = _date_cls.today()
                if _last_trading_date != _today:
                    # New trading day — reset kill switch with current equity
                    _broker_eq_now = _broker_equity
                    if hasattr(broker, 'get_account'):
                        try:
                            _acct_now = broker.get_account()
                            _broker_eq_now = float(getattr(_acct_now, 'equity', _broker_eq_now))
                        except Exception:
                            pass
                    # Reset broker SOD equity for daily P&L tracking
                    if hasattr(broker, 'reset_daily'):
                        broker.reset_daily()
                    kill_switch.reset_daily(_broker_eq_now)
                    logger.info(f"Daily reset: SOD equity=${_broker_eq_now:,.2f}")
                    _last_trading_date = _today

                # --- Kill switch check ---
                if not kill_switch.is_trading_allowed():
                    # Check if cooldown has expired
                    kill_switch.check_cooldown_expired()
                if not kill_switch.is_trading_allowed():
                    logger.warning(f"Trading blocked: {kill_switch.block_reason}")
                    time.sleep(60)
                    continue

                # Fetch latest bars
                bars = data_manager.get_latest_bars(symbols, limit=cfg.data.history_days)
                if bars is None or len(bars) == 0:
                    logger.warning("No bar data available; skipping cycle.")
                    time.sleep(60)
                    continue

                # Extract SPY history for regime detection
                spy_data = None
                try:
                    sym_level = bars.index.get_level_values("symbol")
                    spy_mask  = sym_level == "SPY"
                    spy_df    = bars.loc[spy_mask].copy()
                    spy_df.index = spy_df.index.get_level_values("datetime")
                    spy_df.columns = [c.lower() for c in spy_df.columns]
                    spy_df = spy_df[~spy_df.index.duplicated(keep="last")]
                    spy_data = spy_df
                except Exception as exc:
                    logger.warning(f"SPY extraction failed: {exc}")

                # Regime detection
                regime_state = Backtester._default_regime_state()
                if spy_data is not None and len(spy_data) >= 60:
                    try:
                        if not regime_detector.is_fitted:
                            regime_detector.fit(spy_data)
                        regime_state = regime_detector.predict(spy_data)
                        logger.info(
                            f"Regime: {regime_state.regime.value} "
                            f"(confidence={regime_state.confidence:.1%})"
                        )
                    except Exception as exc:
                        logger.warning(f"RegimeDetector failed: {exc}")

                # Build price pivot
                close_col = next(
                    (c for c in bars.columns if c.lower() == "close"), None
                )
                if close_col is None:
                    logger.warning("No 'close' column; skipping signal generation.")
                    time.sleep(60)
                    continue

                price_data = bars[close_col].unstack(level="symbol")
                trade_syms = [s for s in symbols if s in price_data.columns]
                price_data = price_data[trade_syms] if trade_syms else price_data

                # Extract volume data for mean reversion volume spike filter
                volume_data = None
                vol_col = next(
                    (c for c in bars.columns if c.lower() == "volume"), None
                )
                if vol_col is not None:
                    try:
                        volume_data = bars[vol_col].unstack(level="symbol")
                        if trade_syms:
                            vol_cols = [s for s in trade_syms if s in volume_data.columns]
                            volume_data = volume_data[vol_cols] if vol_cols else volume_data
                    except Exception:
                        volume_data = None

                # Latest prices
                current_prices = price_data.iloc[-1].to_dict() if len(price_data) > 0 else {}
                if hasattr(broker, 'update_prices'):
                    broker.update_prices(current_prices)

                # --- Pre-trade circuit breaker check ---
                portfolio = broker.get_portfolio_state()
                if not kill_switch.pre_order_check(portfolio):
                    logger.warning(f"Circuit breaker tripped: {kill_switch.block_reason}")
                    time.sleep(60)
                    continue

                # Generate signals
                if len(price_data) >= 20:
                    signals = signal_gen.generate_all_signals(price_data, regime_state, volume_data=volume_data)
                    if signals:
                        orders = exec_mgr.process_signals(signals, current_prices)
                        logger.info(
                            f"Cycle {cycle}: {len(signals)} signals → "
                            f"{len(orders)} orders submitted."
                        )

                # Refresh portfolio after trades
                portfolio = broker.get_portfolio_state()

                # --- Position reconciliation (every 10 cycles) ---
                if cycle % 10 == 0:
                    try:
                        internal_positions = (
                            broker.get_positions()
                            if hasattr(broker, 'get_positions')
                            else {}
                        )
                        recon_report = reconciler.reconcile(internal_positions)
                        if recon_report.has_discrepancies:
                            logger.warning(recon_report.summary())
                    except Exception as exc:
                        logger.warning(f"Reconciliation failed: {exc}")

                # --- Cache maintenance (every 20 cycles ≈ 100 min) ---
                if cycle % 20 == 0:
                    purged = data_cache.purge_expired()
                    if purged > 0:
                        logger.debug(f"Cache: purged {purged} stale entries.")

                # Print portfolio snapshot
                mins_left = market_cal.minutes_until_close()
                print(
                    f"  [cycle {cycle}] equity={portfolio.equity:,.2f} | "
                    f"cash={portfolio.cash:,.2f} | "
                    f"positions={len(portfolio.positions)} | "
                    f"regime={regime_state.regime.value} | "
                    f"close_in={mins_left:.0f}m",
                    flush=True,
                )

            except KeyboardInterrupt:
                print("\n[LIVE] Interrupted by user. Shutting down.")
                break
            except Exception as exc:
                logger.error(f"Live cycle {cycle} failed: {exc}", exc_info=True)

            # Sleep between cycles (5 minutes in paper/live mode)
            time.sleep(900)  # 15 min — reduced overtrading

    # ------------------------------------------------------------------
    # Summary printer
    # ------------------------------------------------------------------

    def print_summary(self, result: BacktestResult) -> None:
        """Print a clean backtest summary to the terminal.

        Parameters
        ----------
        result:
            Completed :class:`BacktestResult`.
        """
        report = PerformanceMetrics.generate_report(
            metrics=result.metrics,
            equity_curve=result.equity_curve,
            benchmark=None,  # included in metrics dict already
        )
        print(report)

        # Extra summary line
        m = result.metrics
        n_trades = m.get("total_trades", 0)
        win_rate = m.get("win_rate", float("nan"))
        sharpe   = m.get("sharpe_ratio", float("nan"))
        max_dd   = m.get("max_drawdown", float("nan"))

        import math
        def _s(v, pct=False):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return "N/A"
            return f"{v:.1%}" if pct else f"{v:.2f}"

        print(
            f"\n  Quick summary: "
            f"trades={n_trades} | "
            f"win={_s(win_rate, pct=True)} | "
            f"Sharpe={_s(sharpe)} | "
            f"MaxDD={_s(max_dd, pct=True)}\n"
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="ATNN Quant Powerhouse — Quantitative Trading System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["backtest", "paper", "live"],
        default="backtest",
        help="Operating mode.",
    )
    parser.add_argument(
        "--start",
        default="2022-01-01",
        help="Backtest start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end",
        default="2025-12-31",
        help="Backtest end date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100_000.0,
        help="Initial portfolio capital in USD.",
    )
    parser.add_argument(
        "--ml",
        action="store_true",
        default=False,
        help="Enable ML meta-learner pipeline.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Space-separated list of ticker symbols to trade.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity.",
    )
    return parser.parse_args()


def _configure_logging(level: str) -> None:
    """Configure root logger with a clean format.

    Parameters
    ----------
    level:
        Log level string (``"DEBUG"``, ``"INFO"``, etc.).
    """
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


if __name__ == "__main__":
    args = _parse_args()
    _configure_logging(args.log_level)

    orchestrator = SystemOrchestrator(mode=args.mode)

    if args.mode == "backtest":
        orchestrator.run_backtest(
            start=args.start,
            end=args.end,
            symbols=args.symbols,
            initial_capital=args.capital,
            use_ml=args.ml,
        )
    else:
        orchestrator.run_live()
