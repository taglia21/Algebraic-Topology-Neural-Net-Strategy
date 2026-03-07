"""
backtest/backtester.py
======================
Event-driven backtesting engine for the ATNN Quant Powerhouse.

The :class:`Backtester` replays historical data bar-by-bar through the **same**
code path used in live trading.  Only the data source (DataManager in backtest
mode) and broker (SimulatedBroker) differ.

Bar-by-bar processing order
---------------------------
For each timestamp in the dataset:

    1. DataManager yields the next bar slice.
    2. Accumulate a rolling price history window (no look-ahead).
    3. Update the regime detector with history up to (and including) the
       current bar.
    4. Optionally compute ML features and run the meta-learner.
    5. Each strategy generates signals via the SignalGenerator.
    6. Regime-aware allocation scales signal strengths.
    7. Risk manager approves / adjusts / rejects each intended trade.
    8. ExecutionManager submits orders to the SimulatedBroker.
    9. SimulatedBroker fills orders with slippage using the current bar.
    10. Portfolio state is updated; equity curve point appended.
    11. All events (signals, fills, regimes) are logged.

Anti-look-ahead guarantees
--------------------------
- The history window passed to strategies and the regime detector **never**
  includes data beyond the current bar's index position.
- The warmup period (default 252 bars) is consumed before any signals are
  generated, ensuring indicators are well-formed.
- ML retraining uses only data up to the current bar (walk-forward).

Usage
-----
    from backtest.backtester import Backtester

    bt = Backtester()
    result = bt.run(
        symbols=["AAPL", "MSFT", "GOOGL"],
        start_date="2022-01-01",
        end_date="2025-12-31",
    )
    print(result.metrics)
"""

from __future__ import annotations

import logging
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.config import Config, get_config
from core.logger import get_trade_logger
from core.regime_detector import RegimeDetector, RegimeState, Regime
from core.risk_manager import RiskManager
from data.data_manager import DataManager
from equities.execution import ExecutionManager, SimulatedBroker
from equities.models import PortfolioState
from equities.signal_generator import SignalGenerator
from equities.strategies.stat_arb import StatArbStrategy
from equities.strategies.momentum import MomentumStrategy
from equities.strategies.factor_model import FactorModelStrategy
from backtest.metrics import BacktestResult, PerformanceMetrics

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum warmup period before signals are generated.
# 252 trading days ensures momentum (12-1 month), regime HMM, and indicators
# are stable.  Stat arb requires 2 years (504 bars); enforced per-strategy.
_WARMUP_BARS: int = 252

# Default SPY symbol for benchmark fetch
_BENCHMARK_SYMBOL: str = "SPY"

# Progress reporting interval (number of bars)
_PROGRESS_EVERY: int = 50


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

class Backtester:
    """Event-driven backtester that replays historical data through
    the full signal pipeline.

    Uses the **exact same code path** as live trading.  Only the data source
    (DataManager loaded in backtest mode) and the broker (SimulatedBroker)
    differ from a live run.

    Parameters
    ----------
    config:
        System configuration.  Defaults to the global singleton.
    initial_cash:
        Starting portfolio cash.  Overrides ``config.system.initial_portfolio_value``.
    slippage_bps:
        Simulated one-way slippage in basis points.  Overrides the backtest
        config value when provided.
    commission_per_share:
        Per-share commission in USD.
    verbose:
        If True, print progress updates to stdout during the run.

    Example
    -------
    >>> bt = Backtester()
    >>> result = bt.run(["AAPL", "MSFT"], "2022-01-01", "2024-12-31")
    >>> print(result.metrics["sharpe_ratio"])
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        initial_cash: Optional[float] = None,
        slippage_bps: Optional[float] = None,
        commission_per_share: Optional[float] = None,
        verbose: bool = True,
    ) -> None:
        self.config = config or get_config()
        self._trade_logger = get_trade_logger()
        self._verbose = verbose

        # Resolve runtime parameters
        _cash = initial_cash or self.config.system.initial_portfolio_value
        _slippage = slippage_bps or self.config.backtest.slippage_bps
        _commission = commission_per_share or self.config.backtest.commission_per_share

        # Core components (same as live trading)
        self.data_manager    = DataManager(mode="backtest")
        self.regime_detector = RegimeDetector()
        self.risk_manager    = RiskManager(self.config.risk, self._trade_logger)
        self.broker          = SimulatedBroker(
            initial_cash=_cash,
            slippage_bps=_slippage,
            commission_per_share=_commission,
            trade_logger=self._trade_logger,
        )

        # Signal generator and execution manager are built in run()
        # because they depend on the strategy universe.
        self.signal_generator: Optional[SignalGenerator] = None
        self.execution_manager: Optional[ExecutionManager] = None

        # ML pipeline (optional)
        self.ml_pipeline = None
        self._ml_imports_ok: bool = False
        self._use_ml_requested: bool = False

        # Results storage
        self._equity_curve: Dict[datetime, float] = {}
        self._signals_log: List[dict] = []
        self._regime_log: Dict[datetime, str] = {}
        self._trades_log: List[dict] = []

        logger.info(
            "Backtester initialised: "
            f"cash={_cash:,.0f}, slippage={_slippage}bps, "
            f"commission={_commission:.4f}/share."
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        use_ml: bool = False,
        benchmark_symbol: str = _BENCHMARK_SYMBOL,
    ) -> BacktestResult:
        """Run a full backtest over the specified date range.

        Parameters
        ----------
        symbols:
            List of ticker symbols to trade.
        start_date:
            ISO-8601 start date string (``"YYYY-MM-DD"``).
        end_date:
            ISO-8601 end date string.
        use_ml:
            If True, initialise and run the ML pipeline alongside the base
            strategies.  ML adds alpha on top; base strategies work without it.
        benchmark_symbol:
            Ticker to use as benchmark for alpha/beta calculations.
            Default ``"SPY"``.

        Returns
        -------
        BacktestResult
            Full structured result including equity curve, trades, signals,
            regime history, and computed performance metrics.

        Raises
        ------
        RuntimeError
            If data cannot be loaded for the specified period.
        ValueError
            If date strings are malformed or end < start.
        """
        self._log(f"Starting backtest: {symbols} | {start_date} → {end_date}")

        # ----------------------------------------------------------------
        # 1. Parse dates
        # ----------------------------------------------------------------
        start_dt, end_dt = self._parse_dates(start_date, end_date)

        # Fetch extra history for warmup (start_dt - extra days)
        warmup_days_calendar = int(_WARMUP_BARS * 1.6)   # ~1.6× to account for weekends
        fetch_start = start_dt - pd.Timedelta(days=warmup_days_calendar)

        # ----------------------------------------------------------------
        # 2. Load data (include benchmark for metrics)
        # ----------------------------------------------------------------
        all_symbols = list(symbols)
        if benchmark_symbol not in all_symbols:
            all_symbols.append(benchmark_symbol)

        self._log(f"Loading data for {len(all_symbols)} symbols ...")
        self.data_manager.load_backtest_data(
            symbols=all_symbols,
            start=fetch_start,
            end=end_dt,
        )

        total_bars = self.data_manager.backtest_total_bars
        self._log(f"Loaded {total_bars} total bars.")

        if total_bars < _WARMUP_BARS + 10:
            raise RuntimeError(
                f"Insufficient data: only {total_bars} bars available; "
                f"need at least {_WARMUP_BARS + 10}."
            )

        # ----------------------------------------------------------------
        # 3. Initialise strategies and execution
        # NOTE: _setup_pipeline is called here so strategies know which symbols
        # are available.  StatArbStrategy.find_pairs() is called separately
        # after the warmup phase (see step 4b below).
        # ----------------------------------------------------------------
        self._setup_pipeline(symbols, use_ml)

        # ----------------------------------------------------------------
        # 4. Determine warmup boundary
        # ----------------------------------------------------------------
        all_datetimes = self.data_manager._backtest_datetimes
        warmup_end_idx = self._warmup(all_datetimes, start_dt)
        self._log(
            f"Warmup ends at bar {warmup_end_idx} "
            f"({all_datetimes[warmup_end_idx] if warmup_end_idx < len(all_datetimes) else 'N/A'})."
        )

        # ----------------------------------------------------------------
        # 4b. Call find_pairs() on StatArbStrategy with full warmup data
        # StatArbStrategy requires price data BEFORE the event loop runs.
        # We build the wide-format price matrix from all warmup bars.
        # ----------------------------------------------------------------
        self._init_stat_arb_pairs(symbols, all_datetimes, warmup_end_idx)

        # ----------------------------------------------------------------
        # 5. Reset result accumulators
        # ----------------------------------------------------------------
        self._equity_curve.clear()
        self._signals_log.clear()
        self._regime_log.clear()
        self._trades_log.clear()

        # Track prior fills to detect closed positions
        prior_fills: Dict[str, dict] = {}  # symbol → {entry data}
        # Track previous bar equity for daily P&L calculation
        _prev_bar_equity: float = self.broker._initial_cash

        # ----------------------------------------------------------------
        # 6. Event loop — bar by bar
        # ----------------------------------------------------------------
        bar_idx = 0
        regime_state = RegimeDetector().predict(  # default UNKNOWN before fit
            pd.DataFrame()
        ) if False else self._default_regime_state()

        for bar_dt, bar_df in self.data_manager.iter_bars():
            bar_idx += 1

            # Skip bars before our entire warmup window
            # (we still need to iterate them to build history)

            # Build history window: all data up to and including this bar
            history = self._get_history_up_to(bar_dt)

            if history is None or len(history) < 2:
                continue

            # Extract current prices for all symbols
            current_prices = self._extract_current_prices(bar_df)

            # ---- Update broker mark-to-market ----
            self.broker.update_prices(current_prices)

            # ---- Fill pending orders with today's bar ----
            for sym, bar_series in self._iter_symbol_bars(bar_df):
                self.broker.on_bar(bar_series, sym)

            # ---- Record equity ----
            portfolio = self.broker.get_portfolio_state()
            self._equity_curve[bar_dt] = portfolio.equity

            # ---- Still in warmup? ----
            if bar_idx <= warmup_end_idx:
                if bar_idx % _PROGRESS_EVERY == 0 and self._verbose:
                    print(
                        f"  [warmup {bar_idx}/{warmup_end_idx}] "
                        f"equity={portfolio.equity:,.0f}",
                        flush=True,
                    )
                # Keep regime detector warm during the warmup phase so it has
                # a valid fit by the time we start generating signals.
                spy_history_warmup = self._get_spy_history(history, benchmark_symbol)
                if spy_history_warmup is not None and len(spy_history_warmup) >= 60:
                    try:
                        if not self.regime_detector.is_fitted or bar_idx % 20 == 0:
                            self.regime_detector.fit(spy_history_warmup)
                    except Exception:
                        pass
                continue

            # ---- Only generate signals after warmup + in trading range ----
            if bar_dt < start_dt:
                continue

            # ---- Step 2: Update regime detector ----
            spy_history = self._get_spy_history(history, benchmark_symbol)
            if spy_history is not None and len(spy_history) >= 60:
                try:
                    # Fit HMM if not fitted yet, or on monthly cadence (every 21 bars)
                    if not self.regime_detector.is_fitted or bar_idx % 21 == 0:
                        self.regime_detector.fit(spy_history.tail(504))  # Use last 2 years only
                    # Pass 120 bars so feature builder (needs ~20 warmup) still
                    # leaves 100 valid rows — well above MIN_TRAINING_DAYS=60.
                    regime_state = self.regime_detector.predict(spy_history.tail(120))
                except Exception as exc:
                    logger.warning(f"RegimeDetector failed on bar {bar_dt}: {exc}")
                    regime_state = self._default_regime_state()
            self._regime_log[bar_dt] = regime_state.regime.value

            # ---- Step 3: ML features (if enabled) ----
            # Lazy-init: create MLPipeline on first post-warmup bar so
            # FeatureEngine has SPY data available for cross-sectional features.
            ml_adjustments: Dict[str, float] = {}
            if use_ml and self.ml_pipeline is None and self._ml_imports_ok:
                try:
                    from ml.feature_engine import FeatureEngine as _FE
                    from ml.pipeline import MLPipeline as _MLP
                    spy_df = self._get_spy_history(history, benchmark_symbol)
                    fe = _FE(spy_data=spy_df)
                    self.ml_pipeline = _MLP(feature_engine=fe, config=self.config)
                    # Initial training on all warmup data per-symbol
                    for sym in symbols:
                        try:
                            sym_history = self._get_spy_history(history, sym)
                            if sym_history is not None and len(sym_history) >= 252:
                                self.ml_pipeline.train_all(sym_history, symbol=sym, run_validation=False)
                                logger.info(f"ML pipeline trained on {sym} ({len(sym_history)} bars)")
                                break  # train on first viable symbol; meta-learner generalises
                        except Exception as exc:
                            logger.debug(f"ML train skip {sym}: {exc}")
                    logger.info("ML pipeline initialised and trained.")
                except Exception as exc:
                    logger.warning(f"ML pipeline init failed: {exc}")
                    self.ml_pipeline = None
                    self._ml_imports_ok = False  # don't retry

            if use_ml and self.ml_pipeline is not None:
                try:
                    # Retrain on configured cadence (~monthly)
                    if bar_idx % (self.config.ml.retrain_freq_days * 5) == 0:
                        self.ml_pipeline.retrain_if_needed(history)
                    # Predict weekly (every 5 bars) to balance cost vs responsiveness
                    if bar_idx % 5 == 0 or bar_idx <= warmup_end_idx + 2:
                        ml_preds = self.ml_pipeline.predict(history, regime_state)
                        ml_adjustments = {
                            sym: float(pred.get("score", 0.5))
                            for sym, pred in ml_preds.items()
                        }
                except Exception as exc:
                    logger.warning(f"ML pipeline failed on bar {bar_dt}: {exc}")

            # ---- Step 4-5: Generate and combine signals ----
            # Every-other-bar for full signal generation (stat_arb pair
            # evaluation is O(n²) in the universe size).  Factor model and
            # momentum run every bar because they're O(n).
            signals = []
            price_data = self._build_price_pivot(history, symbols)
            if price_data is not None and len(price_data) >= 20:
                try:
                    signals = self.signal_generator.generate_all_signals(
                        price_data, regime_state
                    )
                except Exception as exc:
                    logger.error(
                        f"SignalGenerator raised on bar {bar_dt}: {exc}\n"
                        + traceback.format_exc()
                    )
                    # Per spec: if signal generation fails, log and continue

            # ---- Step 6: ML meta-learner signal weight adjustment ----
            if ml_adjustments and signals:
                signals = self._apply_ml_adjustments(signals, ml_adjustments)

            # ---- Step 7-8: Risk check + order submission ----
            if signals:
                try:
                    orders = self.execution_manager.process_signals(
                        signals, current_prices
                    )
                except Exception as exc:
                    logger.error(
                        f"ExecutionManager raised on bar {bar_dt}: {exc}\n"
                        + traceback.format_exc()
                    )
                    orders = []

                # Log signals (only when signals are actually generated)
                for sig in signals:
                    self._signals_log.append({
                        "bar_dt":    bar_dt.isoformat() if hasattr(bar_dt, "isoformat") else str(bar_dt),
                        "symbol":    sig.symbol,
                        "direction": sig.direction,
                        "strength":  round(sig.strength, 4),
                        "strategy":  sig.strategy,
                    })

                if self._verbose:
                    print(
                        f"  [bar {bar_idx}] {bar_dt} | "
                        f"{len(signals)} signals | "
                        f"equity={portfolio.equity:,.0f} | "
                        f"regime={regime_state.regime.value}",
                        flush=True,
                    )

            # ---- Step 9-10: Fills happen on next bar (already handled above) ----

            # ---- Step 11: Progress reporting (every N bars, no signals spam) ----
            if bar_idx % _PROGRESS_EVERY == 0 and self._verbose:
                print(
                    f"  [bar {bar_idx}/{total_bars}] "
                    f"{bar_dt} | equity={portfolio.equity:,.0f} | "
                    f"regime={regime_state.regime.value} | "
                    f"positions={len(portfolio.positions)}",
                    flush=True,
                )

        # ----------------------------------------------------------------
        # 7. Close all open positions at end of backtest
        # ----------------------------------------------------------------
        self._close_all_positions(current_prices)

        # ----------------------------------------------------------------
        # 8. Build trades log from fills
        # ----------------------------------------------------------------
        self._trades_log = self._build_trades_from_fills()

        # ----------------------------------------------------------------
        # 9. Fetch benchmark equity curve
        # ----------------------------------------------------------------
        benchmark_equity = self._build_benchmark_curve(
            benchmark_symbol, start_dt, end_dt
        )

        # ----------------------------------------------------------------
        # 10. Compute metrics
        # ----------------------------------------------------------------
        return self.get_results(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            benchmark=benchmark_equity,
        )

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def _warmup(
        self,
        all_datetimes: List[datetime],
        trading_start: datetime,
    ) -> int:
        """Determine the warmup boundary index.

        Returns the 0-based index in ``all_datetimes`` at which real
        signal generation should begin.  This is the maximum of:
        - The bar at which we have >= _WARMUP_BARS of history, and
        - The bar whose datetime equals ``trading_start``.

        A minimum of 252 trading days of warm-up is enforced to ensure
        momentum lookbacks (12-1 month), regime HMM training, and factor
        computations are fully initialised.

        Parameters
        ----------
        all_datetimes:
            Complete sorted list of all bar datetimes in the loaded dataset.
        trading_start:
            The user-specified start of the trading period.

        Returns
        -------
        int
            Bar index (0-based) from which signal generation begins.
        """
        # Find the first bar on or after trading_start
        trade_start_idx = 0
        for i, dt in enumerate(all_datetimes):
            dt_naive = dt.replace(tzinfo=None) if hasattr(dt, "tzinfo") and dt.tzinfo else dt
            ts_naive = trading_start.replace(tzinfo=None) if hasattr(trading_start, "tzinfo") and trading_start.tzinfo else trading_start
            if dt_naive >= ts_naive:
                trade_start_idx = i
                break

        # We need at least _WARMUP_BARS before the trading start
        warmup_idx = max(_WARMUP_BARS - 1, trade_start_idx - 1)
        warmup_idx = min(warmup_idx, len(all_datetimes) - 1)

        return warmup_idx

    # ------------------------------------------------------------------
    # Pipeline setup
    # ------------------------------------------------------------------

    def _setup_pipeline(self, symbols: List[str], use_ml: bool) -> None:
        """Instantiate strategies, signal generator, and execution manager.

        Parameters
        ----------
        symbols:
            Trading universe.
        use_ml:
            Whether to load the ML pipeline.
        """
        strategies = [
            StatArbStrategy(config=self.config.strategy.stat_arb),
            MomentumStrategy(config=self.config.strategy.momentum),
            FactorModelStrategy(config=self.config.strategy.factor_model),
        ]

        self.signal_generator = SignalGenerator(
            strategies=strategies,
            trade_logger=self._trade_logger,
        )

        self.execution_manager = ExecutionManager(
            broker=self.broker,
            risk_manager=self.risk_manager,
            trade_logger=self._trade_logger,
            order_type="market",
            max_position_value=self.broker._initial_cash * self.config.risk.max_position_pct * 2,
        )

        # Optional ML pipeline — mark for lazy init after warmup
        # because FeatureEngine needs SPY data that only becomes available
        # after data loading is complete.
        self._use_ml_requested = use_ml
        if use_ml:
            try:
                import ml.feature_engine  # noqa: F401  — verify import works
                import ml.pipeline        # noqa: F401
                self._ml_imports_ok = True
                logger.info("ML pipeline will be initialised after warmup.")
            except ImportError as exc:
                logger.warning(f"ML pipeline unavailable: {exc}")
                self._ml_imports_ok = False
        else:
            self._ml_imports_ok = False

        logger.info(
            f"Pipeline setup: {len(strategies)} strategies, "
            f"ML={'enabled' if use_ml else 'disabled'}."
        )

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def get_results(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        benchmark: Optional[pd.Series] = None,
    ) -> BacktestResult:
        """Assemble and return the final :class:`BacktestResult`.

        Parameters
        ----------
        symbols:
            Universe of symbols traded.
        start_date:
            ISO-8601 start string.
        end_date:
            ISO-8601 end string.
        benchmark:
            Optional benchmark equity curve for alpha/beta.

        Returns
        -------
        BacktestResult
        """
        if not self._equity_curve:
            logger.warning("Backtester.get_results: equity curve is empty.")
            equity_curve = pd.Series(dtype=float)
            daily_returns = pd.Series(dtype=float)
        else:
            equity_curve = pd.Series(self._equity_curve).sort_index()
            daily_returns = equity_curve.pct_change().dropna()

        regime_history = pd.Series(self._regime_log).sort_index()

        metrics = PerformanceMetrics.calculate_all(
            equity_curve,
            self._trades_log,
            benchmark=benchmark,
        )

        # Serialise config (strip non-serialisable values)
        try:
            config_dict = self.config.to_dict()
        except Exception:
            config_dict = {"error": "config serialisation failed"}

        return BacktestResult(
            equity_curve=equity_curve,
            daily_returns=daily_returns,
            trades=self._trades_log,
            signals=self._signals_log,
            regime_history=regime_history,
            metrics=metrics,
            config=config_dict,
            start_date=start_date,
            end_date=end_date,
            symbols=list(symbols),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_stat_arb_pairs(
        self,
        symbols: List[str],
        all_datetimes: list,
        warmup_end_idx: int,
    ) -> None:
        """Call find_pairs() on StatArbStrategy using all warmup history.

        Must be called after data is loaded and warmup_end_idx is determined,
        but BEFORE the event loop starts so that stat_arb has pairs ready for
        signal generation on bar 1 of the trading window.

        Parameters
        ----------
        symbols:
            Trading universe (benchmark excluded).
        all_datetimes:
            Full sorted list of bar datetimes.
        warmup_end_idx:
            Index of the last warmup bar (0-based).
        """
        # Locate the StatArbStrategy instance inside the signal generator
        if self.signal_generator is None:
            return
        stat_arb: Optional[StatArbStrategy] = None
        for strat in self.signal_generator.strategies:
            if isinstance(strat, StatArbStrategy):
                stat_arb = strat
                break

        if stat_arb is None:
            logger.warning("_init_stat_arb_pairs: StatArbStrategy not found in pipeline.")
            return

        # Build the warmup price DataFrame (wide-format: datetime × symbol)
        warmup_end_dt = all_datetimes[warmup_end_idx] if warmup_end_idx < len(all_datetimes) else None
        if warmup_end_dt is None:
            logger.warning("_init_stat_arb_pairs: warmup_end_dt is None; skipping find_pairs.")
            return

        # Use the full history up to the warmup boundary
        warmup_history = self._get_history_up_to(warmup_end_dt)
        if warmup_history is None or len(warmup_history) == 0:
            logger.warning("_init_stat_arb_pairs: warmup history is empty; skipping find_pairs.")
            return

        warmup_price_data = self._build_price_pivot(warmup_history, symbols)
        if warmup_price_data is None or warmup_price_data.empty:
            logger.warning("_init_stat_arb_pairs: could not build warmup price pivot; skipping find_pairs.")
            return

        n_bars = len(warmup_price_data)
        n_syms = len(warmup_price_data.columns)
        self._log(
            f"Calling StatArbStrategy.find_pairs() with {n_bars} warmup bars, "
            f"{n_syms} symbols: {list(warmup_price_data.columns)}"
        )

        try:
            pairs = stat_arb.find_pairs(warmup_price_data)
            self._log(f"StatArbStrategy.find_pairs() found {len(pairs)} cointegrated pairs.")
        except Exception as exc:
            logger.error(f"StatArbStrategy.find_pairs() failed: {exc}\n" + traceback.format_exc())

    @staticmethod
    def _parse_dates(start: str, end: str) -> Tuple[datetime, datetime]:
        """Parse ISO-8601 date strings into UTC-aware datetimes.

        Parameters
        ----------
        start, end:
            Date strings like ``"2022-01-01"`` or ``"2022-01-01T00:00:00"``.

        Returns
        -------
        Tuple[datetime, datetime]

        Raises
        ------
        ValueError
            If ``end`` is before ``start``.
        """
        fmt = "%Y-%m-%d"
        try:
            start_dt = datetime.strptime(start[:10], fmt).replace(tzinfo=timezone.utc)
            end_dt   = datetime.strptime(end[:10], fmt).replace(tzinfo=timezone.utc)
        except ValueError as exc:
            raise ValueError(f"Invalid date format (expected YYYY-MM-DD): {exc}") from exc

        if end_dt < start_dt:
            raise ValueError(f"end_date ({end}) must be >= start_date ({start}).")

        return start_dt, end_dt

    def _get_history_up_to(self, bar_dt: datetime) -> Optional[pd.DataFrame]:
        """Return all loaded backtest data with timestamp <= bar_dt.

        This enforces strict no-look-ahead: only data available at (or before)
        the current bar is passed to strategies.

        Parameters
        ----------
        bar_dt:
            The current bar's datetime.

        Returns
        -------
        pd.DataFrame or None
        """
        df = self.data_manager._backtest_data
        if df is None:
            return None

        # Filter on the datetime index level
        try:
            dt_level = df.index.get_level_values("datetime")
            mask = dt_level <= bar_dt
            return df.loc[mask]
        except Exception as exc:
            logger.debug(f"_get_history_up_to error: {exc}")
            return df

    def _build_price_pivot(
        self,
        history: pd.DataFrame,
        symbols: List[str],
    ) -> Optional[pd.DataFrame]:
        """Build a wide-format close price DataFrame from the MultiIndex history.

        Parameters
        ----------
        history:
            MultiIndex (datetime, symbol) OHLCV DataFrame.
        symbols:
            Symbols to include.

        Returns
        -------
        pd.DataFrame with datetime index and symbol columns, or None.
        """
        if history is None or len(history) == 0:
            return None

        try:
            close_col = None
            for col in history.columns:
                if col.lower() == "close":
                    close_col = col
                    break

            if close_col is None:
                return None

            # Pivot: rows=datetime, cols=symbol, values=close
            pivot = history[close_col].unstack(level="symbol")
            # Keep only requested symbols that are present
            present = [s for s in symbols if s in pivot.columns]
            if not present:
                return None
            return pivot[present]
        except Exception as exc:
            logger.debug(f"_build_price_pivot error: {exc}")
            return None

    def _extract_current_prices(self, bar_df: pd.DataFrame) -> Dict[str, float]:
        """Extract the latest close price for each symbol from a bar slice.

        Parameters
        ----------
        bar_df:
            Single-timestamp slice of the MultiIndex DataFrame.

        Returns
        -------
        Dict mapping symbol → close price.
        """
        prices: Dict[str, float] = {}
        if bar_df is None or len(bar_df) == 0:
            return prices

        close_col = None
        for col in bar_df.columns:
            if col.lower() == "close":
                close_col = col
                break

        if close_col is None:
            return prices

        try:
            sym_level = bar_df.index.get_level_values("symbol")
            for sym in sym_level:
                mask = sym_level == sym
                rows = bar_df.loc[mask, close_col]
                if len(rows) > 0:
                    val = float(rows.iloc[-1])
                    if val > 0 and not np.isnan(val):
                        prices[sym] = val
        except Exception as exc:
            logger.debug(f"_extract_current_prices error: {exc}")

        return prices

    @staticmethod
    def _iter_symbol_bars(bar_df: pd.DataFrame):
        """Yield (symbol, bar_series) tuples from a multi-symbol bar slice.

        Parameters
        ----------
        bar_df:
            Single-timestamp MultiIndex slice.

        Yields
        ------
        (symbol, pd.Series)
        """
        if bar_df is None or len(bar_df) == 0:
            return

        try:
            sym_level = bar_df.index.get_level_values("symbol")
            seen = set()
            for sym in sym_level:
                if sym in seen:
                    continue
                seen.add(sym)
                mask = sym_level == sym
                sym_rows = bar_df.loc[mask]
                if len(sym_rows) > 0:
                    yield sym, sym_rows.iloc[-1]
        except Exception as exc:
            logger.debug(f"_iter_symbol_bars error: {exc}")

    def _get_spy_history(
        self,
        history: pd.DataFrame,
        benchmark_symbol: str,
    ) -> Optional[pd.DataFrame]:
        """Extract single-symbol OHLCV history for the benchmark.

        Parameters
        ----------
        history:
            Full MultiIndex history.
        benchmark_symbol:
            Benchmark ticker (e.g. ``"SPY"``).

        Returns
        -------
        pd.DataFrame with standard OHLCV columns, or None.
        """
        if history is None:
            return None

        try:
            sym_level = history.index.get_level_values("symbol")
            mask = sym_level == benchmark_symbol
            spy_df = history.loc[mask].copy()

            if len(spy_df) == 0:
                return None

            # Flatten to single-level datetime index
            spy_df.index = spy_df.index.get_level_values("datetime")
            spy_df.columns = [c.lower() for c in spy_df.columns]
            spy_df = spy_df[~spy_df.index.duplicated(keep="last")]
            return spy_df
        except Exception as exc:
            logger.debug(f"_get_spy_history error: {exc}")
            return None

    def _apply_ml_adjustments(
        self,
        signals: list,
        ml_adjustments: Dict[str, float],
    ) -> list:
        """Scale signal strengths by ML meta-learner scores.

        The meta-learner outputs a score in [0, 1].  We multiply the signal
        strength by this score, then clamp to [0.001, 1.0].

        Parameters
        ----------
        signals:
            List of :class:`~equities.models.Signal` objects.
        ml_adjustments:
            Mapping of symbol → ML score in [0, 1].

        Returns
        -------
        List of adjusted Signal objects.
        """
        from equities.models import Signal

        adjusted = []
        for sig in signals:
            score = ml_adjustments.get(sig.symbol)
            if score is not None and not np.isnan(score):
                new_strength = float(np.clip(sig.strength * score, 0.001, 1.0))
                adjusted.append(
                    Signal(
                        symbol=sig.symbol,
                        direction=sig.direction,
                        strength=new_strength,
                        strategy=sig.strategy,
                        metadata={
                            **sig.metadata,
                            "ml_score":          score,
                            "pre_ml_strength":   sig.strength,
                        },
                        timestamp=sig.timestamp,
                    )
                )
            else:
                adjusted.append(sig)
        return adjusted

    def _close_all_positions(self, current_prices: Dict[str, float]) -> None:
        """Flatten all open positions at the end of the backtest.

        Issues a market sell/buy for each open position at the last known
        price (the SimulatedBroker will fill immediately on the next
        ``on_bar`` call, but since there are no more bars we just record
        the close at current price directly).

        Parameters
        ----------
        current_prices:
            Last known prices from the final bar.
        """
        portfolio = self.broker.get_portfolio_state()
        if not portfolio.positions:
            return

        logger.info(
            f"Closing {len(portfolio.positions)} open positions at backtest end."
        )
        for sym, pos in list(portfolio.positions.items()):
            price = current_prices.get(sym, pos.current_price)
            side  = "sell" if pos.qty > 0 else "buy"
            qty   = abs(pos.qty)
            try:
                order = self.broker.submit_order(
                    symbol=sym,
                    qty=qty,
                    side=side,
                    order_type="market",
                    strategy="backtest_close",
                )
                # Simulate fill at current price
                bar_mock = pd.Series({
                    "open":  price,
                    "high":  price,
                    "low":   price,
                    "close": price,
                })
                self.broker.on_bar(bar_mock, sym)
            except Exception as exc:
                logger.warning(f"Could not close position {sym}: {exc}")

    def _build_trades_from_fills(self) -> List[dict]:
        """Convert SimulatedBroker fill history into a list of trade records.

        Pairs buy and sell fills for each symbol to construct closed trade
        records with P&L and holding period.

        Returns
        -------
        List[dict] with keys: symbol, side, entry_date, exit_date,
        entry_price, exit_price, qty, pnl, holding_days, strategy.
        """
        fills = self.broker.fill_history
        if not fills:
            return []

        # Accumulate fills per symbol in order
        from collections import defaultdict
        fill_queues: dict = defaultdict(list)
        for fill in fills:
            fill_queues[fill.symbol].append(fill)

        trades: List[dict] = []
        for sym, sym_fills in fill_queues.items():
            # Match buys to subsequent sells (FIFO)
            open_lots: List[dict] = []
            for fill in sym_fills:
                if fill.side == "buy":
                    open_lots.append({
                        "dt":    fill.timestamp,
                        "price": fill.fill_price,
                        "qty":   fill.fill_qty,
                    })
                elif fill.side == "sell" and open_lots:
                    # Close against oldest open lot
                    entry = open_lots.pop(0)
                    holding = self._days_between(entry["dt"], fill.timestamp)
                    pnl = (fill.fill_price - entry["price"]) * min(fill.fill_qty, entry["qty"])
                    trades.append({
                        "symbol":      sym,
                        "side":        "long",
                        "entry_date":  entry["dt"],
                        "exit_date":   fill.timestamp,
                        "entry_price": entry["price"],
                        "exit_price":  fill.fill_price,
                        "qty":         min(fill.fill_qty, entry["qty"]),
                        "pnl":         pnl,
                        "holding_days": holding,
                        "strategy":    "combined",
                    })

        return trades

    @staticmethod
    def _days_between(dt1: Any, dt2: Any) -> float:
        """Return calendar days between two datetime-like objects.

        Parameters
        ----------
        dt1, dt2:
            Datetime objects (or anything with a ``timestamp()`` method).

        Returns
        -------
        float
        """
        try:
            if hasattr(dt1, "timestamp") and hasattr(dt2, "timestamp"):
                return abs(dt2.timestamp() - dt1.timestamp()) / 86400.0
            return 0.0
        except Exception:
            return 0.0

    def _build_benchmark_curve(
        self,
        benchmark_symbol: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> Optional[pd.Series]:
        """Extract benchmark equity curve from the loaded backtest data.

        Synthesises a normalised equity curve (starting at 100,000) from
        SPY close prices over the trading period.

        Parameters
        ----------
        benchmark_symbol:
            Ticker (e.g. ``"SPY"``).
        start_dt, end_dt:
            Trading period boundaries.

        Returns
        -------
        pd.Series indexed by date, or None if benchmark data unavailable.
        """
        try:
            df = self.data_manager._backtest_data
            if df is None:
                return None

            sym_level = df.index.get_level_values("symbol")
            mask = sym_level == benchmark_symbol
            bench_df = df.loc[mask].copy()
            if len(bench_df) == 0:
                return None

            bench_df.index = bench_df.index.get_level_values("datetime")
            bench_df = bench_df[~bench_df.index.duplicated(keep="last")]

            close_col = next(
                (c for c in bench_df.columns if c.lower() == "close"), None
            )
            if close_col is None:
                return None

            prices = bench_df[close_col].sort_index()

            # Filter to trading range
            # Normalise to common start — strip timezone for comparison
            def _strip_tz(dt):
                return dt.replace(tzinfo=None) if hasattr(dt, "tzinfo") and dt.tzinfo else dt

            start_naive = _strip_tz(start_dt)
            end_naive   = _strip_tz(end_dt)

            # Try to filter; if index has timezone, compare accordingly
            try:
                if prices.index.tzinfo is not None:
                    prices = prices[(prices.index >= start_dt) & (prices.index <= end_dt)]
                else:
                    prices = prices[(prices.index >= start_naive) & (prices.index <= end_naive)]
            except Exception:
                pass  # use full series if filter fails

            if len(prices) < 2:
                return None

            initial = float(prices.iloc[0])
            if initial <= 0:
                return None

            equity = prices / initial * self.broker._initial_cash
            return equity

        except Exception as exc:
            logger.warning(f"Could not build benchmark curve: {exc}")
            return None

    @staticmethod
    def _default_regime_state() -> RegimeState:
        """Return a safe default UNKNOWN RegimeState for pre-fit periods.

        Returns
        -------
        RegimeState
        """
        from core.regime_detector import VIXLevel
        return RegimeState(
            regime=Regime.UNKNOWN,
            confidence=0.0,
            vix_level=VIXLevel.UNKNOWN,
            vix_value=float("nan"),
            adx=float("nan"),
            is_trending=False,
            is_crisis=False,
            regime_probs={r.value: 0.0 for r in Regime},
            n_training_bars=0,
        )

    def _log(self, msg: str) -> None:
        """Print to stdout if verbose, and always write to logger.

        Parameters
        ----------
        msg:
            Message string.
        """
        logger.info(msg)
        if self._verbose:
            print(f"[Backtester] {msg}", flush=True)

    # ------------------------------------------------------------------
    # Typing helper (Any imported at module level)
    # ------------------------------------------------------------------
