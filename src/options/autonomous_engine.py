"""
Autonomous Options Trading Engine
==================================

Main orchestrator for fully autonomous options trading.

6-Step Trading Loop (60-second cycle):
1. SCAN: Generate signals from all strategies
2. FILTER: Remove invalid/duplicate signals
3. SIZE: Calculate position size using Kelly Criterion
4. EXECUTE: Place orders via Alpaca API
5. MANAGE: Monitor positions, trigger stops/targets
6. CHECK: Verify portfolio risk within limits

Features:
- Multi-strategy signal generation
- Kelly Criterion position sizing
- Automated trade execution
- Real-time position management
- Portfolio risk monitoring
- Graceful shutdown and state persistence
"""

import asyncio
import argparse
import logging
import logging.handlers
from datetime import datetime, time
from typing import Dict, List, Optional
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

# ---------------------------------------------------------------------------
# Rotating file logger (5 MB, 3 backups) — Phase 4
# ---------------------------------------------------------------------------
_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)

_options_log_file = _LOG_DIR / "options_engine.log"
_rotating_handler = logging.handlers.RotatingFileHandler(
    str(_options_log_file), maxBytes=5 * 1024 * 1024, backupCount=3,
)
_rotating_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s")
)
logging.getLogger("src.options").addHandler(_rotating_handler)
logging.getLogger("src.options").setLevel(logging.INFO)

from .config import RISK_CONFIG, MONITORING_CONFIG
from .universe import get_universe
from .signal_generator import SignalGenerator, Signal, SignalType
from .position_sizer import MedallionPositionSizer, PositionSize, calculate_max_loss_per_contract
from .trade_executor import AlpacaOptionsExecutor, OrderSide, ExecutionResult
from .iv_data_manager import IVDataManager
from .contract_resolver import OptionContractResolver, ResolvedContract, ResolvedSpread, ResolvedIronCondor
from .earnings_gate import should_block_for_earnings
from .greeks_monitor import PortfolioGreeksMonitor
from .vix_regime import VIXRegimeOverlay

# ==== NEW ENHANCED MODULES ====
from .regime_detector import RegimeDetector, MarketRegime
from .correlation_manager import CorrelationManager, Position as CorrPosition
from .weight_optimizer import DynamicWeightOptimizer
from .volatility_surface import VolatilitySurfaceEngine
from .cointegration_engine import CointegrationEngine

# ==== PHASE 6: EXIT MANAGEMENT, GEX, DAILY P&L ====
from .exit_manager import ExitManager, ExitAction, ExitReason
from .gex_analyzer import GammaExposureAnalyzer, GEXProfile
from src.metrics.daily_performance import DailyPerformanceLogger

# ==== PHASE 7: OCC UTILS, DELTA HEDGING, SMART PRICING ====
from .occ_utils import parse_occ_symbol, compute_option_delta, smart_limit_price

# ==== PHASE 4: WIRED ORPHANED MODULES ====
try:
    from .manifold_regime_detector import ManifoldRegimeDetector
    MANIFOLD_AVAILABLE = True
except ImportError:
    ManifoldRegimeDetector = None
    MANIFOLD_AVAILABLE = False

try:
    from src.ml.adaptive_ensemble import AdaptiveEnsemble
    ML_AVAILABLE = True
except ImportError:
    AdaptiveEnsemble = None
    ML_AVAILABLE = False

try:
    from src.ml.continuous_learner import ContinuousLearner
    CONTINUOUS_LEARNER_AVAILABLE = True
except ImportError:
    ContinuousLearner = None
    CONTINUOUS_LEARNER_AVAILABLE = False

try:
    from src.quant_models.garch import GARCHModel
    GARCH_AVAILABLE = True
except ImportError:
    GARCHModel = None
    GARCH_AVAILABLE = False

try:
    from src.quant_models.monte_carlo_pricer import MonteCarloPricer
    from src.quant_models.heston_model import HestonModel
    ADVANCED_PRICING_AVAILABLE = True
except ImportError:
    MonteCarloPricer = None
    HestonModel = None
    ADVANCED_PRICING_AVAILABLE = False

try:
    from src.signal_aggregator import SignalAggregator
    AGGREGATOR_AVAILABLE = True
except ImportError:
    SignalAggregator = None
    AGGREGATOR_AVAILABLE = False

try:
    from src.optimization.bayesian_tuner import BayesianTuner
    BAYESIAN_AVAILABLE = True
except ImportError:
    BayesianTuner = None
    BAYESIAN_AVAILABLE = False


# ============================================================================
# MARKET HOURS
# ============================================================================

# NYSE holidays (2025-2027)
_NYSE_HOLIDAYS = {
    (2025, 1, 1), (2025, 1, 20), (2025, 2, 17), (2025, 4, 18),
    (2025, 5, 26), (2025, 6, 19), (2025, 7, 4), (2025, 9, 1),
    (2025, 11, 27), (2025, 12, 25),
    (2026, 1, 1), (2026, 1, 19), (2026, 2, 16), (2026, 4, 3),
    (2026, 5, 25), (2026, 6, 19), (2026, 7, 3), (2026, 9, 7),
    (2026, 11, 26), (2026, 12, 25),
    (2027, 1, 1), (2027, 1, 18), (2027, 2, 15), (2027, 3, 26),
    (2027, 5, 31), (2027, 6, 18), (2027, 7, 5), (2027, 9, 6),
    (2027, 11, 25), (2027, 12, 24),
}


def market_is_open() -> bool:
    """
    Check if market is currently open.
    
    Returns:
        True if open, False otherwise
    """
    now_et_dt = datetime.now(ZoneInfo("America/New_York"))
    now = now_et_dt.time()
    
    market_open = time(9, 30)
    market_close = time(16, 0)
    
    is_weekday = now_et_dt.weekday() < 5
    is_holiday = (now_et_dt.year, now_et_dt.month, now_et_dt.day) in _NYSE_HOLIDAYS
    
    return is_weekday and not is_holiday and market_open <= now <= market_close


def safe_entry_window() -> bool:
    """
    Check if we're in the safe entry window (avoid first/last 15 min).
    
    Returns:
        True if safe to enter, False otherwise
    """
    now = datetime.now(ZoneInfo("America/New_York")).time()
    
    safe_open = time(9, 45)  # 15 min after open
    safe_close = time(15, 45)  # 15 min before close
    
    return safe_open <= now <= safe_close


# ============================================================================
# OPTIONS RISK CONTROLS (2026-02-20 emergency fix)
# ============================================================================

MAX_DAILY_OPTIONS_SPEND = 500       # Max $500/day total spend on options
MAX_OPTIONS_POSITIONS = 5            # Max 5 open option positions at once
MAX_DAILY_OPTIONS_TRADES = 10        # Max 10 option trades per day
DAILY_LOSS_STOP = -1000              # Stop ALL trading if daily P/L < -$1,000
MAX_STRIKE_OTM_PCT = 0.05            # Max 5% OTM for strike selection
MIN_OPTION_DELTA = 0.15              # Min |delta| for tradable options

# Phase 4: Daily P&L circuit breaker (realized + unrealized)
DAILY_PNL_CIRCUIT_BREAKER = -500.0   # Halt if day P&L < -$500


# ============================================================================
# AUTONOMOUS TRADING ENGINE
# ============================================================================

class AutonomousTradingEngine:
    """Main autonomous options trading engine.

    Runs continuously during market hours, executing a 6-step loop
    every ``signal_scan_interval_seconds`` (default 300 s):

    1. **SCAN** — Generate signals from all strategies.
    2. **FILTER** — Remove invalid / duplicate / gated signals.
    3. **SIZE** — Calculate position size (fixed-fractional).
    4. **EXECUTE** — Resolve OCC contracts and submit to Alpaca.
    5. **MANAGE** — Monitor positions, trigger stops / targets.
    6. **CHECK** — Verify portfolio risk within limits.

    Attributes:
        portfolio_value: Current portfolio equity (refreshed each cycle).
        current_positions: In-memory list of open option positions.
        paper: Whether the engine is using paper trading.
        stats: Cumulative run-time statistics dict.
    """
    
    def __init__(
        self,
        portfolio_value: float,
        paper: bool = True,
        state_file: str = "",
    ):
        """Initialise the engine and all sub-components.

        Args:
            portfolio_value: Starting portfolio value in USD.
            paper: If ``True`` (default), use Alpaca paper trading.
            state_file: Path to the JSON state-persistence file.
                Falls back to ``state/trading_state.json``.
        """
        # Deterministic state path (issue #15)
        if not state_file:
            import pathlib
            _project_root = pathlib.Path(__file__).resolve().parent.parent.parent
            _state_dir = _project_root / "state"
            _state_dir.mkdir(exist_ok=True)
            state_file = str(_state_dir / "trading_state.json")
        # get_config() in options/config.py requires a key and returns a single value.
        # The engine expects a dict-like config, so we merge the relevant config dicts.
        self.config = {**RISK_CONFIG, **MONITORING_CONFIG}
        self.logger = logging.getLogger(__name__)
        
        # Portfolio state
        self.portfolio_value = portfolio_value
        self.current_positions = []
        self.portfolio_delta = 0.0
        self.paper = paper
        self.state_file = state_file
        self._stop_event = asyncio.Event()

        # --- Daily risk-control counters (reset each trading day) ---
        self._daily_options_spent = 0.0       # $ spent on options today
        self._daily_options_trades = 0        # number of option trades today
        self._daily_tracking_date = None      # date these counters apply to
        self._day_start_portfolio = portfolio_value  # portfolio value at day start
        self._executed_occ_symbols: set = set()      # OCC symbols already bought today

        # Phase 4: Daily P&L circuit breaker state
        self._pnl_circuit_breaker_tripped = False

        # Initialize components
        self.signal_generator = SignalGenerator()
        self.position_sizer = MedallionPositionSizer()
        self.trade_executor = AlpacaOptionsExecutor(paper=paper)
        self.iv_data_manager = IVDataManager()  # NEW: IV data management
        
        # Contract resolver — bridges signals to real OCC symbols with live pricing
        self.contract_resolver = OptionContractResolver(
            trading_client=self.trade_executor.trading_client,
            data_client=self.trade_executor.data_client,
        )
        
        # ==== ENHANCED MODULES ====
        self.regime_detector = RegimeDetector()
        self.correlation_manager = CorrelationManager()
        self.weight_optimizer = DynamicWeightOptimizer(
            strategies=["iv_rank", "theta_decay", "mean_reversion", "delta_hedging"],
            regime_detector=self.regime_detector
        )
        self.vol_surface_engine = VolatilitySurfaceEngine()
        self.cointegration_engine = CointegrationEngine()

        # ==== PHASE 4: WIRED ORPHANED MODULES ====
        self.manifold_detector = None
        if MANIFOLD_AVAILABLE:
            try:
                self.manifold_detector = ManifoldRegimeDetector()
                self.logger.info("✓ ManifoldRegimeDetector loaded")
            except Exception as e:
                self.logger.warning(f"ManifoldRegimeDetector init failed: {e}")

        self.adaptive_ml = None
        if ML_AVAILABLE:
            try:
                self.adaptive_ml = AdaptiveEnsemble()
                self.logger.info("✓ AdaptiveEnsemble (self-training ML) loaded for options")
            except Exception as e:
                self.logger.warning(f"AdaptiveEnsemble init failed: {e}")

        self.continuous_learner = None
        if CONTINUOUS_LEARNER_AVAILABLE:
            try:
                self.continuous_learner = ContinuousLearner()
                self.logger.info("✓ ContinuousLearner loaded")
            except Exception as e:
                self.logger.warning(f"ContinuousLearner init failed: {e}")

        self.garch_model = None
        if GARCH_AVAILABLE:
            try:
                self.garch_model = GARCHModel()
                self.logger.info("✓ GARCH model loaded")
            except Exception as e:
                self.logger.warning(f"GARCH init failed: {e}")

        self.mc_pricer = None
        self.heston_model = None
        if ADVANCED_PRICING_AVAILABLE:
            try:
                self.mc_pricer = MonteCarloPricer(n_paths=50000)
                self.heston_model = HestonModel()
                self.logger.info("✓ Monte Carlo + Heston pricing loaded")
            except Exception as e:
                self.logger.warning(f"Advanced pricing init failed: {e}")

        self.signal_aggregator = None
        if AGGREGATOR_AVAILABLE:
            try:
                self.signal_aggregator = SignalAggregator(min_confidence=0.4)
                self.signal_aggregator.initialize()
                self.logger.info("✓ SignalAggregator loaded")
            except Exception as e:
                self.logger.warning(f"SignalAggregator init failed: {e}")

        self.bayesian_tuner = None
        if BAYESIAN_AVAILABLE:
            try:
                self.bayesian_tuner = BayesianTuner()
                self.logger.info("✓ BayesianTuner loaded")
            except Exception as e:
                self.logger.warning(f"BayesianTuner init failed: {e}")

        # ==== PHASE 3: Greeks monitor + VIX overlay ====
        self.greeks_monitor = PortfolioGreeksMonitor()
        self.vix_overlay = VIXRegimeOverlay()

        # ==== PHASE 6: Exit Manager, GEX, Daily P&L ====
        exit_config = {
            "profit_target_pct": RISK_CONFIG.get("exit_profit_target_pct", 0.50),
            "stop_loss_multiplier": RISK_CONFIG.get("exit_stop_loss_multiplier", 2.0),
            "dte_exit_threshold": RISK_CONFIG.get("exit_dte_threshold", 7),
            "trailing_stop_activate_pct": RISK_CONFIG.get("exit_trailing_stop_activate", 0.30),
            "trailing_stop_trail_pct": RISK_CONFIG.get("exit_trailing_stop_trail", 0.50),
            "time_accel_dte_pct": RISK_CONFIG.get("exit_time_accel_dte_pct", 0.50),
            "time_accel_profit_pct": RISK_CONFIG.get("exit_time_accel_profit_pct", 0.25),
            "use_mleg_close": RISK_CONFIG.get("exit_use_mleg_close", True),
        }
        self.exit_manager = ExitManager(
            trading_client=self.trade_executor.trading_client,
            data_client=self.trade_executor.data_client,
            config=exit_config,
        )

        self.gex_analyzer = GammaExposureAnalyzer(
            data_client=self.trade_executor.data_client,
            sticky_strike_threshold=RISK_CONFIG.get("gex_sticky_strike_threshold", 0.30),
            avoidance_radius_pct=RISK_CONFIG.get("gex_avoidance_radius_pct", 0.005),
            cache_ttl_minutes=RISK_CONFIG.get("gex_cache_ttl_minutes", 15),
        )
        self.gex_enabled = RISK_CONFIG.get("gex_enabled", True)

        self.daily_perf_logger = DailyPerformanceLogger(
            initial_equity=portfolio_value,
        )
        self.logger.info(
            "✓ Phase 6: ExitManager, GEX Analyzer, DailyPerformanceLogger loaded"
        )

        # Backfill IV data on startup
        self._backfill_iv_data()
        
        # Current market regime
        self.current_regime: Optional[MarketRegime] = None
        self.regime_fitted = False
        
        # Statistics
        self.stats = {
            "cycles_run": 0,
            "signals_generated": 0,
            "trades_executed": 0,
            "trades_failed": 0,
            "positions_closed": 0,
            "total_pnl": 0.0,
            "start_time": datetime.now().isoformat(),
        }
        
        # Load previous state if exists
        self._load_state()
        
        self.logger.info(f"Initialized autonomous engine (paper={paper}, portfolio=${portfolio_value:,.0f})")
        self.logger.info("✓ Enhanced modules loaded: RegimeDetector, CorrelationManager, WeightOptimizer, VolSurface, Cointegration")
        self.logger.info(f"✓ Phase4 modules: Manifold={self.manifold_detector is not None}, "
                        f"ML={self.adaptive_ml is not None}, GARCH={self.garch_model is not None}, "
                        f"Heston={self.heston_model is not None}, Aggregator={self.signal_aggregator is not None}")

    # ------------------------------------------------------------------ #
    # PHASE 3: Sync in-memory positions from Alpaca
    # ------------------------------------------------------------------ #

    def _sync_positions_from_alpaca(self) -> None:
        """Replace ``self.current_positions`` with actual Alpaca state.

        Called at the START of every ``_trading_cycle`` so that in-memory
        state matches reality even after a restart, crash, or manual
        intervention on the Alpaca dashboard.
        """
        try:
            alpaca_opts = self._get_alpaca_option_positions()
        except Exception as exc:
            self.logger.warning(f"Position sync failed: {exc}")
            return

        synced: list = []
        for occ, data in alpaca_opts.items():
            # Phase 7: Use centralized OCC parser instead of inline char loop
            parsed = parse_occ_symbol(occ)
            if parsed is not None:
                underlying = parsed['underlying']
            else:
                # Fallback: extract letters before first digit
                underlying = ""
                for ch in occ:
                    if ch.isdigit():
                        break
                    underlying += ch

            synced.append({
                "symbol": underlying.upper(),
                "occ_symbol": occ,
                "qty": data.get("qty", 0),
                "cost_basis": data.get("cost_basis", 0),
                "unrealized_pl": data.get("unrealized_pl", 0),
                "entry_time": datetime.now().isoformat(),   # approx if unknown
                "signal": None,   # no original signal on restart
                "position_size": None,
                "execution": None,
            })

        prev_count = len(self.current_positions)
        self.current_positions = synced

        # Phase 7 (Improvement 2): Register all synced positions with ExitManager
        # so that positions survive restart with proper exit management
        try:
            self.exit_manager.sync_from_alpaca_state(alpaca_opts)
            self.logger.info(
                f"ExitManager reconciled: {len(self.exit_manager.positions)} tracked positions"
            )
        except Exception as exc:
            self.logger.warning(f"ExitManager sync on startup failed: {exc}")

        if len(synced) != prev_count:
            self.logger.info(
                f"Position sync: {prev_count} -> {len(synced)} option positions"
            )

    def request_shutdown(self) -> None:
        """Request a graceful shutdown of the engine.

        Sets the internal stop-event so that ``run_forever`` exits
        after the current cycle finishes.
        """
        self._stop_event.set()

    async def _sleep_or_stop(self, seconds: float) -> None:
        if seconds <= 0:
            return
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            return

    async def run_forever(self) -> None:
        """Run the trading loop continuously until shutdown is requested.

        The loop respects market hours and implements an error-cooldown
        mechanism: after ``MAX_CONSECUTIVE_ERRORS`` (5) failures in a
        row the engine pauses for 15 minutes before retrying.
        """
        self.logger.info("🚀 AUTONOMOUS TRADING ENGINE STARTED")
        consecutive_errors = 0
        MAX_CONSECUTIVE_ERRORS = 5
        ERROR_COOLDOWN_SECONDS = 900  # 15 minutes

        try:
            while not self._stop_event.is_set():
                # Check if market is open
                if not market_is_open():
                    self.logger.info("Market closed, waiting...")
                    await self._sleep_or_stop(60)
                    continue

                # Run trading cycle with per-cycle error handling
                try:
                    await self._trading_cycle()
                    consecutive_errors = 0  # reset on success
                except asyncio.CancelledError:
                    raise  # propagate cancellation
                except Exception as cycle_err:
                    consecutive_errors += 1
                    self.logger.error(
                        f"Trading cycle error ({consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}): "
                        f"{cycle_err}",
                        exc_info=True,
                    )
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        self.logger.critical(
                            f"{MAX_CONSECUTIVE_ERRORS} consecutive errors — "
                            f"entering {ERROR_COOLDOWN_SECONDS}s cooldown"
                        )
                        await self._sleep_or_stop(ERROR_COOLDOWN_SECONDS)
                        consecutive_errors = 0  # reset after cooldown

                # Save state
                self._save_state()

                # Sleep between cycles
                cycle_sleep = self.config["signal_scan_interval_seconds"]
                self.logger.info(f"Cycle complete, sleeping {cycle_sleep}s")
                await self._sleep_or_stop(cycle_sleep)

        except asyncio.CancelledError:
            self.logger.info("Shutdown task cancelled")
            raise
        except KeyboardInterrupt:
            self.logger.info("Shutdown signal received")
        except Exception as e:
            self.logger.error(f"Fatal error in main loop: {e}", exc_info=True)
        finally:
            await self._shutdown()
    
    async def run(self):
        """Alias for :meth:`run_forever`."""
        await self.run_forever()
    
    async def _trading_cycle(self):
        """Execute one complete 6-step trading cycle.

        Steps:
            0a. Refresh portfolio value from Alpaca.
            0b. Refresh portfolio delta.
            0c. Regime detection & weight optimisation.
            0d. Greeks monitor check.
            0e. VIX regime overlay.
            1. SCAN for signals.
            2. FILTER invalid signals.
            3. SIZE positions.
            4. EXECUTE trades (gated by Greeks, VIX, safe window).
            5. MANAGE existing positions.
            6. CHECK risk limits.
        """
        # --- Daily risk-control reset ---
        self._reset_daily_counters_if_needed()

        # --- PHASE 3: sync in-memory positions from Alpaca ---
        self._sync_positions_from_alpaca()

        self.stats["cycles_run"] += 1
        cycle_num = self.stats["cycles_run"]
        
        self.logger.info(f"{'='*60}")
        self.logger.info(f"CYCLE #{cycle_num} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(
            f"  Options budget: ${self._daily_options_spent:.0f}"
            f"/${MAX_DAILY_OPTIONS_SPEND} | "
            f"Trades: {self._daily_options_trades}/{MAX_DAILY_OPTIONS_TRADES}"
        )
        self.logger.info(f"{'='*60}")
        
        # STEP 0a: Refresh portfolio value from Alpaca (issue #5)
        try:
            acct = self.trade_executor.trading_client.get_account()
            new_equity = float(acct.equity)
            if new_equity > 0:
                self.portfolio_value = new_equity
                self.logger.info(f"Portfolio value refreshed: ${self.portfolio_value:,.2f}")
        except Exception as e:
            self.logger.warning(f"Could not refresh portfolio value: {e}")

        # STEP 0a-bis: Daily loss stop check (halt ALL trading)
        if self._daily_loss_exceeded():
            self.logger.critical("Cycle aborted — daily loss stop active.")
            return

        # STEP 0a-ter: Phase 4 daily P&L circuit breaker (-$500)
        if not self._check_daily_pnl_circuit_breaker():
            self.logger.critical("Cycle aborted — daily P&L circuit breaker active.")
            return
        
        # STEP 0b: Refresh portfolio delta from Alpaca positions (issue #10)
        # Phase 7: Use Black-Scholes delta via occ_utils instead of hardcoded ±50
        try:
            alpaca_positions = self.trade_executor.trading_client.get_all_positions()
            total_delta = 0.0
            # Fetch underlying prices for delta calculation (cache per underlying)
            _underlying_prices: dict[str, float] = {}
            for ap in alpaca_positions:
                sym = ap.symbol or ""
                is_option = len(sym) > 6 and any(c.isdigit() for c in sym[:6])
                if not is_option:
                    # Equity position (e.g. SPY shares from delta hedging)
                    # Each share = +1 delta (long) or -1 delta (short)
                    equity_qty = float(ap.qty) if ap.qty else 0
                    total_delta += equity_qty  # shares are 1 delta each
                    continue

                qty = float(ap.qty) if ap.qty else 0
                parsed = parse_occ_symbol(sym)
                if parsed is None:
                    # Fallback: crude ±50 delta per contract
                    is_put = "P" in sym[6:8] or "P" in sym[-9:-8]
                    total_delta += qty * (-50 if is_put else 50)
                    continue

                underlying = parsed['underlying']
                # Get underlying price (cache across positions)
                if underlying not in _underlying_prices:
                    try:
                        ul_price = float(
                            getattr(
                                self.trade_executor.trading_client.get_open_position(underlying),
                                'current_price', 0
                            ) or 0
                        )
                    except Exception:
                        ul_price = 0.0
                    # Fallback: try yfinance for underlying price
                    if ul_price <= 0:
                        try:
                            import yfinance as yf
                            tick = yf.Ticker(underlying)
                            ul_price = tick.info.get('regularMarketPrice', 0) or tick.fast_info.get('lastPrice', 0) or 0
                        except Exception:
                            ul_price = 0.0
                    _underlying_prices[underlying] = ul_price

                ul_price = _underlying_prices[underlying]
                if ul_price > 0:
                    per_share_delta = compute_option_delta(sym, ul_price)
                    # qty from Alpaca is signed (+ = long, - = short)
                    # per_share_delta is per-share, multiply by 100 for per-contract
                    total_delta += qty * per_share_delta * 100
                else:
                    # No underlying price — use crude fallback
                    is_put = parsed['option_type'] == 'P'
                    total_delta += qty * (-50 if is_put else 50)

            self.portfolio_delta = total_delta
            self.logger.info(f"Portfolio delta (BS-computed): {total_delta:.1f}")
        except Exception as e:
            self.logger.warning(f"Could not refresh portfolio delta: {e}")
        
        # STEP 0c (NEW): REGIME DETECTION & WEIGHT OPTIMIZATION
        await self._update_regime_and_weights()

        # STEP 0d (PHASE 3): Portfolio Greeks monitor
        try:
            greeks = self.greeks_monitor.get_portfolio_greeks(
                self.trade_executor.trading_client
            )
            self.greeks_monitor.log_greeks(greeks)
            greeks_ok, violations = self.greeks_monitor.is_within_limits(greeks)
            if not greeks_ok:
                for v in violations:
                    self.logger.warning(f"Greeks violation: {v.message}")
                self.logger.warning("Greeks limits breached — skipping new entries this cycle")
        except Exception as e:
            self.logger.warning(f"Greeks monitor error: {e}")
            greeks_ok = True  # fail-open so the bot doesn't freeze

        # STEP 0e (PHASE 3): VIX regime overlay
        vix_snap = self.vix_overlay.get_snapshot()
        if vix_snap.multiplier <= 0:
            self.logger.critical(
                f"VIX CRISIS ({vix_snap.level:.1f}) — halting ALL new entries"
            )
        
        # STEP 1: SCAN - Generate signals
        signals = await self._scan_for_signals()
        self.logger.info(f"Step 1 (SCAN): Generated {len(signals)} signals")
        
        # STEP 2: FILTER - Remove invalid signals
        valid_signals = await self._filter_signals(signals)
        self.logger.info(f"Step 2 (FILTER): {len(valid_signals)} valid signals")
        
        # STEP 3: SIZE - Calculate position sizes
        sized_signals = await self._size_positions(valid_signals)
        self.logger.info(f"Step 3 (SIZE): {len(sized_signals)} positions sized")
        
        # STEP 4: EXECUTE - Place orders
        if safe_entry_window() and greeks_ok and vix_snap.multiplier > 0:
            executions = await self._execute_trades(sized_signals, vix_multiplier=vix_snap.multiplier)
            self.logger.info(f"Step 4 (EXECUTE): {len(executions)} orders submitted")
        else:
            reason = []
            if not safe_entry_window():
                reason.append("outside safe window")
            if not greeks_ok:
                reason.append("Greeks limits breached")
            if vix_snap.multiplier <= 0:
                reason.append(f"VIX crisis ({vix_snap.level:.1f})")
            self.logger.info(f"Step 4 (EXECUTE): Skipped — {', '.join(reason)}")
        
        # STEP 5: MANAGE - Monitor positions (PHASE 6: ExitManager-driven)
        await self._manage_positions()
        self.logger.info(f"Step 5 (MANAGE): {len(self.current_positions)} positions monitored")

        # STEP 5b (PHASE 6): Run ExitManager checks on all tracked positions
        await self._run_exit_manager()
        
        # STEP 6: CHECK - Verify risk limits
        risk_ok = await self._check_risk_limits()
        self.logger.info(f"Step 6 (CHECK): Risk limits {'✓ OK' if risk_ok else '✗ EXCEEDED'}")

        # STEP 7 (PHASE 6): Daily P&L logging
        self._log_daily_performance()
        
        # Log cycle summary
        self._log_cycle_summary()
    
    async def _scan_for_signals(self) -> List[Signal]:
        """Step 1: Generate signals from all strategies.

        Returns:
            List of raw :class:`Signal` objects from the signal generator.
        """
        symbols = get_universe()
        
        signals = await self.signal_generator.generate_all_signals(
            symbols=symbols,
            portfolio_delta=self.portfolio_delta,
        )
        
        self.stats["signals_generated"] += len(signals)
        
        return signals
    
    async def _filter_signals(self, signals: List[Signal]) -> List[Signal]:
        """Step 2: Filter signals to remove invalid / duplicate ones.

        Applies concentration-risk check, confidence floor, IV rank
        enforcement, earnings gate, underlying concentration limit,
        and optional aggregator / Heston boosting.

        Args:
            signals: Raw signals from :meth:`_scan_for_signals`.

        Returns:
            Filtered list of actionable signals.
        """
        # First, check concentration risk
        concentration_ok = await self._check_concentration_risk()
        if not concentration_ok:
            self.logger.warning("Concentration risk too high - blocking new positions")
            return []
        
        valid_signals = []
        
        for signal in signals:
            # Skip HOLD signals
            if signal.signal_type == SignalType.HOLD:
                continue
            
            # Skip low confidence - require meaningful signal quality
            min_confidence = 0.40 if self.paper else 0.50

            # PHASE 5: Boost confidence using SignalAggregator on the underlying
            if self.signal_aggregator is not None:
                try:
                    agg = self.signal_aggregator.aggregate(signal.symbol, min_confidence=0.3)
                    # Align: if aggregator agrees with signal direction, boost confidence
                    if signal.signal_type == SignalType.BUY and agg.signal > 0.2:
                        signal.confidence = min(signal.confidence * 1.25, 0.99)
                        self.logger.debug(f"Boosted {signal.symbol} confidence via aggregator (bullish agreement)")
                    elif signal.signal_type == SignalType.SELL and agg.signal < -0.2:
                        signal.confidence = min(signal.confidence * 1.25, 0.99)
                        self.logger.debug(f"Boosted {signal.symbol} confidence via aggregator (bearish agreement)")
                    elif abs(agg.signal) > 0.3 and (
                        (signal.signal_type == SignalType.BUY and agg.signal < -0.3)
                        or (signal.signal_type == SignalType.SELL and agg.signal > 0.3)
                    ):
                        # Aggregator strongly disagrees — dampen confidence
                        signal.confidence *= 0.7
                        self.logger.debug(f"Dampened {signal.symbol} confidence via aggregator (disagreement)")
                except Exception as e:
                    self.logger.debug(f"Aggregator boost failed for {signal.symbol}: {e}")

            # PHASE 5: Heston/MC pricing comparison — if model price deviates
            # from market mid-price significantly, boost/reject signal
            if self.heston_model is not None and signal.expected_premium:
                try:
                    from src.quant_models.heston_model import HestonParams
                    # Use stored GARCH vol as v0
                    v0 = getattr(self, '_last_garch_vol', 0.04)
                    params = HestonParams(v0=v0, kappa=2.0, theta=v0, xi=0.3, rho=-0.7)
                    current_price = signal.current_price or 100
                    strike = signal.strike_put or signal.strike_call or current_price
                    T = (signal.dte or 30) / 365.0
                    heston_result = self.heston_model.price_call(
                        current_price, strike, T, params
                    )
                    model_price = heston_result.price
                    market_price = signal.expected_premium
                    if market_price > 0 and model_price > 0:
                        price_ratio = model_price / market_price
                        if price_ratio > 1.15:  # Model says option is underpriced
                            signal.confidence = min(signal.confidence * 1.15, 0.99)
                            self.logger.info(f"Heston: {signal.symbol} underpriced by {(price_ratio-1)*100:.0f}%")
                        elif price_ratio < 0.85:  # Model says option is overpriced
                            signal.confidence *= 0.85
                            self.logger.info(f"Heston: {signal.symbol} overpriced by {(1-price_ratio)*100:.0f}%")
                except Exception as e:
                    self.logger.debug(f"Heston pricing comparison failed: {e}")
            
            if signal.confidence < min_confidence:
                self.logger.debug(f"Skipping low confidence signal: {signal.symbol} ({signal.confidence:.1%})")
                # PHASE 5: Log every signal, even rejected ones
                await self._log_signal_with_reasoning(signal, execution_result=None)
                continue
            
            # Skip if already have position in this symbol
            if self._has_position(signal.symbol):
                self.logger.debug(f"Skipping {signal.symbol} - already have position")
                continue
            
            # Check max positions limit
            max_positions = self.config["max_positions"]
            if len(self.current_positions) >= max_positions:
                self.logger.warning(f"Max positions ({max_positions}) reached, skipping new signals")
                break

            # --- PHASE 2: IV RANK ENFORCEMENT ---
            iv = signal.iv_rank
            if iv is not None:
                if signal.signal_type == SignalType.SELL and iv < 50:
                    self.logger.info(
                        f"Skipping SELL {signal.symbol}: IV rank {iv:.0f} < 50 floor"
                    )
                    continue
                if signal.signal_type == SignalType.BUY and iv > 30:
                    self.logger.info(
                        f"Skipping BUY {signal.symbol}: IV rank {iv:.0f} > 30 ceiling"
                    )
                    continue

            # --- PHASE 2: EARNINGS GATE ---
            sig_dte = signal.dte or 30
            if should_block_for_earnings(signal.symbol, sig_dte):
                self.logger.info(
                    f"Skipping {signal.symbol}: earnings within {sig_dte}-day DTE window"
                )
                continue

            # --- PHASE 3: UNDERLYING CONCENTRATION LIMIT ---
            max_per_underlying = self.config.get("max_positions_per_underlying", 2)
            count_this_sym = sum(
                1 for p in self.current_positions
                if (
                    isinstance(p, dict)
                    and (
                        p.get("symbol") == signal.symbol
                        or getattr(p.get("signal"), "symbol", None) == signal.symbol
                    )
                )
            )
            if count_this_sym >= max_per_underlying:
                self.logger.info(
                    f"Skipping {signal.symbol}: already {count_this_sym} positions "
                    f"(max {max_per_underlying} per underlying)"
                )
                continue
            
            valid_signals.append(signal)
        
        return valid_signals
    
    async def _size_positions(self, signals: List[Signal]) -> List[tuple]:
        """Step 3: Calculate position sizes.

        Uses fixed-fractional sizing (1 % risk per trade) with
        constraints from the :class:`MedallionPositionSizer`.

        Args:
            signals: Validated signals from :meth:`_filter_signals`.

        Returns:
            List of ``(signal, position_size)`` tuples.
        """
        sized_signals = []
        
        for signal in signals:
            # Estimate max loss per contract using actual signal data
            strike_width = 5.0  # Default $5 wide spreads
            # Use actual expected premium from signal if available, otherwise conservative default
            premium_estimate = signal.expected_premium if signal.expected_premium and signal.expected_premium > 0 else 0.0
            max_loss = calculate_max_loss_per_contract(
                strategy=signal.strategy,
                strike_width=strike_width,
                premium_received=premium_estimate,
            )
            
            # Calculate position size
            position_size = self.position_sizer.calculate_position_size(
                portfolio_value=self.portfolio_value,
                max_loss_per_contract=max_loss,
                signal_confidence=signal.confidence,
                probability_of_profit=signal.probability_of_profit,
                iv_rank=signal.iv_rank,
                current_portfolio_delta=self.portfolio_delta,
                position_delta_per_contract=signal.delta or 0.0,
            )
            
            # Validate
            if self.position_sizer.validate_position_size(position_size, self.portfolio_value):
                sized_signals.append((signal, position_size))
                self.logger.info(
                    f"Sized {signal.symbol}: {position_size.contracts} contracts "
                    f"(risk: {position_size.risk_percent:.2%})"
                )
            else:
                self.logger.warning(f"Invalid position size for {signal.symbol}, skipping")
        
        return sized_signals
    
    async def _execute_trades(self, sized_signals: List[tuple], *, vix_multiplier: float = 1.0) -> List[ExecutionResult]:
        """Step 4: Resolve signals to real contracts and execute via Alpaca API.

        For each (signal, position_size) pair:
        1. Resolve the abstract signal to real OCC contract(s) with live pricing.
        2. Use the resolved mid-price as the limit price (not hardcoded).
        3. Pass the real OCC symbol to trade_executor (not "{symbol}_CALL_100").
        4. If resolution fails, log a warning and skip (never crash).
        
        ENHANCED: Alpaca-sourced position guards, duplicate prevention,
                  portfolio heat check, budget cap, daily loss stop.
        """
        executions: List[ExecutionResult] = []

        # ── ALPACA-SOURCED STATE (survives restarts) ──
        alpaca_opts = self._get_alpaca_option_positions()
        alpaca_option_count = len(alpaca_opts)
        held_occ_symbols: set = set(alpaca_opts.keys())
        total_options_exposure = sum(abs(p["cost_basis"]) for p in alpaca_opts.values())

        self.logger.info(
            f"Alpaca options state: {alpaca_option_count} positions, "
            f"${total_options_exposure:,.0f} exposure, "
            f"held={list(held_occ_symbols)[:5]}{'...' if len(held_occ_symbols) > 5 else ''}"
        )

        # ── RISK GATE: Alpaca position count ──
        if alpaca_option_count >= MAX_OPTIONS_POSITIONS:
            self.logger.warning(
                f"⚠️ Alpaca option position limit reached: "
                f"{alpaca_option_count}/{MAX_OPTIONS_POSITIONS} — blocking new entries"
            )
            return executions

        # ── RISK GATE: portfolio heat ──
        max_heat = self.config.get("max_portfolio_heat", 0.08)
        if self.portfolio_value > 0 and total_options_exposure > self.portfolio_value * max_heat:
            self.logger.warning(
                f"⚠️ Portfolio heat exceeded: options exposure "
                f"${total_options_exposure:,.0f} > "
                f"{max_heat:.0%} of ${self.portfolio_value:,.0f} "
                f"(${self.portfolio_value * max_heat:,.0f}) — blocking new entries"
            )
            return executions

        # Pre-check buying power to avoid wasting API calls on orders that will fail
        available_bp = 0.0
        try:
            acct = self.trade_executor.trading_client.get_account()
            available_bp = float(getattr(acct, 'options_buying_power', 0) or
                                getattr(acct, 'buying_power', 0) or 0)
            self.logger.info(f"Options buying power: ${available_bp:,.2f}")
        except Exception as e:
            self.logger.warning(f"Could not check buying power: {e}")
            available_bp = float('inf')  # Proceed if check fails

        cycle_num = self.stats.get("cycles_run", 0)

        for signal, position_size in sized_signals:
            try:
                # --- RISK GATE: daily loss stop ---
                if self._daily_loss_exceeded():
                    self.logger.critical("Daily loss stop — aborting remaining executions.")
                    break

                # --- RISK GATE: daily trade count ---
                if not self._options_trade_count_allows():
                    break

                # --- RISK GATE: Alpaca position count (re-check as we go) ---
                if alpaca_option_count >= MAX_OPTIONS_POSITIONS:
                    self.logger.warning(
                        f"⚠️ Alpaca option position limit reached mid-loop: "
                        f"{alpaca_option_count}/{MAX_OPTIONS_POSITIONS}"
                    )
                    break

                # --- RISK GATE: daily budget ---
                estimated_cost = getattr(position_size, 'max_risk', 0) or 500.0
                if not self._options_budget_allows(estimated_cost):
                    break

                # Skip if estimated cost exceeds buying power (avoid 40310000 errors)
                if estimated_cost > available_bp:
                    self.logger.warning(
                        f"⚠️ Skipping {signal.symbol} {signal.strategy}: "
                        f"estimated cost ${estimated_cost:,.0f} > BP ${available_bp:,.0f}"
                    )
                    self.stats["trades_failed"] += 1
                    continue

                result = await self._resolve_and_execute(
                    signal, position_size,
                    held_occ_symbols=held_occ_symbols,
                    cycle_num=cycle_num,
                    vix_multiplier=vix_multiplier,
                )
                if result is not None:
                    # --- Track daily counters on successful submission ---
                    occ = getattr(signal, "occ_symbol", None) or ""
                    if result.success:
                        self._daily_options_trades += 1
                        self._daily_options_spent += estimated_cost
                        alpaca_option_count += 1  # local counter for this loop
                        if occ:
                            self._executed_occ_symbols.add(occ)
                            held_occ_symbols.add(occ)
                    executions.append(result)
                    # PHASE 5: Log signal with execution result + Discord
                    await self._log_signal_with_reasoning(signal, execution_result=result)
                    await self._send_discord_notification(
                        f"🔔 Trade Executed: {signal.symbol} {signal.strategy} "
                        f"conf={signal.confidence:.0%} status={result.status}"
                    )
                    # PHASE 4: ContinuousLearner post-trade recording
                    if self.continuous_learner is not None:
                        try:
                            from src.ml.continuous_learner import TradeResult
                            trade_result = TradeResult(
                                timestamp=datetime.now(),
                                ticker=signal.symbol,
                                signal_direction="long" if signal.signal_type == SignalType.BUY else "short",
                                signal_confidence=signal.confidence,
                                predicted_return=signal.probability_of_profit * 0.02,
                                actual_return=0.0,  # Will be updated on position close
                                is_hit=True,  # Placeholder, updated on close
                                features={
                                    "iv_rank": signal.iv_rank or 0.0,
                                    "confidence": signal.confidence,
                                    "strategy": signal.strategy,
                                },
                                regime=self.current_regime.value if self.current_regime else "unknown",
                            )
                            self.continuous_learner.record_trade(trade_result)
                        except Exception as e:
                            self.logger.debug(f"ContinuousLearner recording failed: {e}")
            except Exception as e:
                self.logger.error(
                    f"Execution error for {signal.symbol} ({signal.strategy}): {e}",
                    exc_info=True,
                )
                self.stats["trades_failed"] += 1

        return executions

    async def _resolve_and_execute(
        self, signal: Signal, position_size,
        held_occ_symbols: set = None,
        cycle_num: int = 0,
        vix_multiplier: float = 1.0,
    ) -> Optional[ExecutionResult]:
        """Resolve a single signal to real contracts, then execute.

        Args:
            signal: Trading signal
            position_size: Calculated position size
            held_occ_symbols: Set of OCC symbols currently held on Alpaca
                              (Alpaca-sourced, survives restarts)
            cycle_num: Current cycle number for idempotent order IDs
            vix_multiplier: VIX regime size multiplier (0.0 = halt)

        Returns:
            ExecutionResult on success/failure, or None if resolution fails.
        """
        if held_occ_symbols is None:
            held_occ_symbols = set()
        target_dte = signal.dte or 30

        # --- PHASE 3: Apply VIX regime multiplier to contracts ---
        if vix_multiplier < 1.0 and hasattr(position_size, 'contracts'):
            import math
            original = position_size.contracts
            adjusted = max(1, math.floor(original * vix_multiplier))
            if adjusted != original:
                self.logger.info(
                    f"VIX multiplier {vix_multiplier:.1f}x: "
                    f"{signal.symbol} contracts {original} -> {adjusted}"
                )
                position_size.contracts = adjusted

        # ---------------------------------------------------------------- #
        # CREDIT SPREAD / PUT SPREAD
        # ---------------------------------------------------------------- #
        if signal.strategy in ("credit_spread", "put_spread", "call_spread"):
            resolved = await self.contract_resolver.resolve_spread(
                symbol=signal.symbol,
                spread_type=signal.strategy,
                target_dte=target_dte,
            )
            if resolved is None:
                self.logger.warning(
                    f"Contract resolution failed for {signal.symbol} "
                    f"{signal.strategy} ~{target_dte}DTE — skipping trade"
                )
                return None

            # Populate signal with resolved data
            signal.occ_symbol = resolved.short_leg.occ_symbol
            signal.expiration_date = resolved.short_leg.expiration

            # --- BID-ASK SPREAD QUALITY FILTER (Phase 2, Change #5) ---
            max_ba_ratio = RISK_CONFIG.get("max_bid_ask_spread_pct", 0.10)
            for leg in (resolved.short_leg, resolved.long_leg):
                if leg.mid_price > 0:
                    ba_ratio = (leg.ask - leg.bid) / leg.mid_price
                    if ba_ratio > max_ba_ratio:
                        self.logger.warning(
                            f"Bid-ask too wide on {leg.occ_symbol}: "
                            f"{ba_ratio:.1%} > {max_ba_ratio:.0%} — skipping"
                        )
                        return None

            # Alpaca-sourced duplicate prevention
            if resolved.short_leg.occ_symbol in held_occ_symbols:
                self.logger.warning(
                    f"⚠️ Duplicate blocked (Alpaca): already hold {resolved.short_leg.occ_symbol}"
                )
                return None

            # --- PHASE 6: GEX FILTER ---
            if self.gex_enabled:
                try:
                    gex_profile = await self.gex_analyzer.compute_gex_profile(
                        signal.symbol, target_dte=target_dte
                    )
                    short_strike = resolved.short_leg.strike if hasattr(resolved.short_leg, 'strike') else 0
                    if short_strike > 0:
                        gex_filter = self.gex_analyzer.filter_signal(
                            gex_profile, short_strike, signal.strategy
                        )
                        if not gex_filter.is_safe:
                            self.logger.warning(
                                f"GEX BLOCKED: {signal.symbol} {signal.strategy} — {gex_filter.reason}"
                            )
                            return None
                        if gex_filter.recommended_action == "reduce_size" and hasattr(position_size, 'contracts'):
                            import math
                            reduce_factor = RISK_CONFIG.get("gex_negative_size_reduction", 0.50)
                            original = position_size.contracts
                            position_size.contracts = max(1, math.floor(original * reduce_factor))
                            if position_size.contracts != original:
                                self.logger.info(
                                    f"GEX reduce: {signal.symbol} contracts {original} -> {position_size.contracts}"
                                )
                except Exception as e:
                    self.logger.debug(f"GEX filter failed for {signal.symbol}: {e}")

            self.logger.info(
                f"Executing spread {signal.symbol}: "
                f"short={resolved.short_leg.occ_symbol} (${resolved.short_leg.mid_price:.2f}) "
                f"long={resolved.long_leg.occ_symbol} (${resolved.long_leg.mid_price:.2f}) "
                f"net_credit=${resolved.net_credit:.2f}"
            )

            result = await self.trade_executor.submit_spread_order(
                long_symbol=resolved.long_leg.occ_symbol,
                short_symbol=resolved.short_leg.occ_symbol,
                quantity=position_size.contracts,
                net_credit=resolved.net_credit if resolved.net_credit > 0 else None,
                net_debit=abs(resolved.net_credit) if resolved.net_credit <= 0 else None,
                client_order_id=self._make_client_order_id(
                    resolved.short_leg.occ_symbol, cycle_num
                ),
            )

        # ---------------------------------------------------------------- #
        # IRON CONDOR
        # ---------------------------------------------------------------- #
        elif signal.strategy == "iron_condor":
            resolved = await self.contract_resolver.resolve_iron_condor(
                symbol=signal.symbol,
                target_dte=target_dte,
            )
            if resolved is None:
                self.logger.warning(
                    f"Contract resolution failed for {signal.symbol} "
                    f"iron_condor ~{target_dte}DTE — skipping trade"
                )
                return None

            signal.occ_symbol = resolved.put_spread.short_leg.occ_symbol
            signal.expiration_date = resolved.put_spread.short_leg.expiration

            # --- BID-ASK SPREAD QUALITY FILTER (Phase 2, Change #5) ---
            max_ba_ratio = RISK_CONFIG.get("max_bid_ask_spread_pct", 0.10)
            for leg in (
                resolved.put_spread.short_leg, resolved.put_spread.long_leg,
                resolved.call_spread.short_leg, resolved.call_spread.long_leg,
            ):
                if leg.mid_price > 0:
                    ba_ratio = (leg.ask - leg.bid) / leg.mid_price
                    if ba_ratio > max_ba_ratio:
                        self.logger.warning(
                            f"Bid-ask too wide on {leg.occ_symbol}: "
                            f"{ba_ratio:.1%} > {max_ba_ratio:.0%} — skipping IC"
                        )
                        return None

            # Alpaca-sourced duplicate prevention
            if resolved.put_spread.short_leg.occ_symbol in held_occ_symbols:
                self.logger.warning(
                    f"⚠️ Duplicate blocked (Alpaca): already hold {resolved.put_spread.short_leg.occ_symbol}"
                )
                return None

            # --- PHASE 6: GEX FILTER (Iron Condor) ---
            if self.gex_enabled:
                try:
                    gex_profile = await self.gex_analyzer.compute_gex_profile(
                        signal.symbol, target_dte=target_dte
                    )
                    for spread_leg in (resolved.put_spread.short_leg, resolved.call_spread.short_leg):
                        short_strike = spread_leg.strike if hasattr(spread_leg, 'strike') else 0
                        if short_strike > 0:
                            gex_filter = self.gex_analyzer.filter_signal(
                                gex_profile, short_strike, "iron_condor"
                            )
                            if not gex_filter.is_safe:
                                self.logger.warning(
                                    f"GEX BLOCKED IC leg: {signal.symbol} strike={short_strike} — {gex_filter.reason}"
                                )
                                return None
                except Exception as e:
                    self.logger.debug(f"GEX filter failed for IC {signal.symbol}: {e}")

            self.logger.info(
                f"Executing iron condor {signal.symbol}: "
                f"put_spread=[{resolved.put_spread.short_leg.occ_symbol}/"
                f"{resolved.put_spread.long_leg.occ_symbol}] "
                f"call_spread=[{resolved.call_spread.short_leg.occ_symbol}/"
                f"{resolved.call_spread.long_leg.occ_symbol}] "
                f"total_credit=${resolved.total_credit:.2f}"
            )

            result = await self.trade_executor.submit_iron_condor(
                underlying=signal.symbol,
                put_long_occ=resolved.put_spread.long_leg.occ_symbol,
                put_short_occ=resolved.put_spread.short_leg.occ_symbol,
                call_short_occ=resolved.call_spread.short_leg.occ_symbol,
                call_long_occ=resolved.call_spread.long_leg.occ_symbol,
                quantity=position_size.contracts,
                net_credit=resolved.total_credit,
            )

        # ---------------------------------------------------------------- #
        # SINGLE LEG (default: calls/puts, straddles, etc.)
        # ---------------------------------------------------------------- #
        else:
            option_type = "call" if signal.signal_type == SignalType.BUY else "put"
            resolved = await self.contract_resolver.resolve_single_leg(
                symbol=signal.symbol,
                option_type=option_type,
                target_dte=target_dte,
            )
            if resolved is None:
                self.logger.warning(
                    f"Contract resolution failed for {signal.symbol} "
                    f"{option_type} ~{target_dte}DTE — skipping trade"
                )
                return None

            signal.occ_symbol = resolved.occ_symbol
            signal.expiration_date = resolved.expiration

            # --- BID-ASK SPREAD QUALITY FILTER (Phase 2, Change #5) ---
            max_ba_ratio = RISK_CONFIG.get("max_bid_ask_spread_pct", 0.10)
            if resolved.mid_price > 0:
                ba_ratio = (resolved.ask - resolved.bid) / resolved.mid_price
                if ba_ratio > max_ba_ratio:
                    self.logger.warning(
                        f"Bid-ask too wide on {resolved.occ_symbol}: "
                        f"{ba_ratio:.1%} > {max_ba_ratio:.0%} — skipping"
                    )
                    return None

            # Alpaca-sourced duplicate prevention
            if resolved.occ_symbol in held_occ_symbols:
                self.logger.warning(
                    f"⚠️ Duplicate blocked (Alpaca): already hold {resolved.occ_symbol}"
                )
                return None

            self.logger.info(
                f"Executing single leg {signal.symbol}: {resolved.occ_symbol} "
                f"strike={resolved.strike} exp={resolved.expiration} "
                f"bid={resolved.bid:.2f} ask={resolved.ask:.2f} "
                f"limit={resolved.mid_price:.2f}"
            )

            # Phase 7: Smart limit price — lean toward bid for buys, ask for sells
            order_side = OrderSide.BUY if signal.signal_type == SignalType.BUY else OrderSide.SELL
            limit = smart_limit_price(
                bid=resolved.bid,
                ask=resolved.ask,
                side=order_side.value,
            )

            result = await self.trade_executor.submit_single_leg_order(
                option_symbol=resolved.occ_symbol,
                side=order_side,
                quantity=position_size.contracts,
                limit_price=limit,
                client_order_id=self._make_client_order_id(
                    resolved.occ_symbol, cycle_num
                ),
            )

        # ---------------------------------------------------------------- #
        # POST-EXECUTION BOOKKEEPING
        # ---------------------------------------------------------------- #
        if result.success:
            self.stats["trades_executed"] += 1
            self.logger.info(
                f"✓ Trade executed: {signal.symbol} ({signal.strategy}) "
                f"— Order {result.order_id}"
            )
            self.current_positions.append({
                "signal": signal,
                "position_size": position_size,
                "execution": result,
                "entry_time": datetime.now().isoformat(),
            })

            # PHASE 6: Register position with ExitManager for systematic exits
            self._register_with_exit_manager(signal, position_size, result)
        else:
            self.stats["trades_failed"] += 1
            self.logger.error(
                f"✗ Trade failed: {signal.symbol} ({signal.strategy}) "
                f"— {result.error_message}"
            )

        return result
    
    async def _manage_positions(self):
        """Step 5: Monitor open positions using real Alpaca P&L data.

        Checks each position for:
            * Stop-loss trigger (unrealized loss > ``stop_loss_pct``).
            * Take-profit trigger (unrealized gain > ``target_profit_pct``).
            * Trailing stop (activate at +20 %, trail 40 % of peak).
            * Time-based profit acceleration (close at +25 % after 50 % DTE).
            * DTE management (close positions < 7 DTE).
        """
        if not self.current_positions:
            return
        
        positions_to_close = []
        target_profit = RISK_CONFIG.get("target_profit_pct", 0.50)
        stop_loss = RISK_CONFIG.get("stop_loss_pct", 0.75)
        trailing_stop_activate = 0.20  # Activate trailing stop at +20% gain
        trailing_stop_trail = 0.40     # Give back at most 40% of peak
        dte_close_threshold = 7  # Close positions with < 7 DTE regardless of P&L

        # Initialize high-water marks dict if not present
        if not hasattr(self, '_options_hwm'):
            self._options_hwm: dict[str, float] = {}
        
        # Get real positions from Alpaca
        try:
            alpaca_positions = self.trade_executor.trading_client.get_all_positions()
            alpaca_map = {}
            for ap in alpaca_positions:
                alpaca_map[ap.symbol] = {
                    "unrealized_pl": float(ap.unrealized_pl) if ap.unrealized_pl else 0.0,
                    "unrealized_plpc": float(ap.unrealized_plpc) if ap.unrealized_plpc else 0.0,
                    "market_value": float(ap.market_value) if ap.market_value else 0.0,
                    "cost_basis": float(ap.cost_basis) if ap.cost_basis else 0.0,
                }
        except Exception as e:
            self.logger.warning(f"Failed to get Alpaca positions: {e}")
            alpaca_map = {}
        
        for position in self.current_positions:
            try:
                symbol = None
                if isinstance(position, dict):
                    signal = position.get("signal")
                    symbol = getattr(signal, "symbol", None) or position.get("symbol")
                    execution = position.get("execution")
                    entry_credit = position.get("entry_credit", 0)
                    max_loss = position.get("max_loss", 0)
                elif hasattr(position, "symbol"):
                    symbol = position.symbol
                    entry_credit = getattr(position, "entry_credit", 0)
                    max_loss = getattr(position, "max_loss", 0)
                
                if not symbol:
                    continue
                
                # Check against Alpaca data — Phase 7: use parse_occ_symbol
                # instead of substring matching (fixes "A" matching "AAPL" etc.)
                unrealized_pnl = 0.0
                unrealized_pnl_pct = 0.0
                
                # Look for matching option positions using OCC parser
                for occ_sym, data in alpaca_map.items():
                    parsed_occ = parse_occ_symbol(occ_sym)
                    if parsed_occ is not None:
                        occ_underlying = parsed_occ['underlying']
                    else:
                        occ_underlying = ''
                        for ch in occ_sym:
                            if ch.isdigit():
                                break
                            occ_underlying += ch
                    if occ_underlying.upper() == symbol.upper():
                        unrealized_pnl += data["unrealized_pl"]
                
                # Calculate P&L percentage using real cost_basis from Alpaca
                total_cost_basis = 0.0
                for occ_sym, data in alpaca_map.items():
                    parsed_occ = parse_occ_symbol(occ_sym)
                    if parsed_occ is not None:
                        occ_underlying = parsed_occ['underlying']
                    else:
                        occ_underlying = ''
                        for ch in occ_sym:
                            if ch.isdigit():
                                break
                            occ_underlying += ch
                    if occ_underlying.upper() == symbol.upper():
                        total_cost_basis += abs(data.get("cost_basis", 0.0))
                
                if total_cost_basis > 0:
                    unrealized_pnl_pct = unrealized_pnl / total_cost_basis
                elif max_loss > 0:
                    unrealized_pnl_pct = unrealized_pnl / max_loss
                # If we have no cost basis and no max_loss, pnl_pct stays 0
                # which means stops won't trigger — this is intentionally safe
                
                close_reason = None

                # Update high-water mark for trailing stop
                prev_hwm = self._options_hwm.get(symbol, 0.0)
                if unrealized_pnl_pct > prev_hwm:
                    self._options_hwm[symbol] = unrealized_pnl_pct
                    prev_hwm = unrealized_pnl_pct
                
                # Stop-loss check: loss exceeds threshold
                if unrealized_pnl < 0 and abs(unrealized_pnl_pct) > stop_loss:
                    close_reason = "STOP_LOSS"
                    self.logger.warning(
                        f"STOP LOSS triggered for {symbol}: "
                        f"P&L: ${unrealized_pnl:+,.2f} ({unrealized_pnl_pct:+.1%})"
                    )
                
                # Trailing stop: activated after gaining trailing_stop_activate,
                # triggers if we give back trailing_stop_trail fraction of peak
                elif (prev_hwm >= trailing_stop_activate and
                      unrealized_pnl_pct < prev_hwm * (1 - trailing_stop_trail)):
                    trail_floor = prev_hwm * (1 - trailing_stop_trail)
                    close_reason = "TRAILING_STOP"
                    self.logger.info(
                        f"TRAILING STOP for {symbol}: peak={prev_hwm:+.1%}, "
                        f"now={unrealized_pnl_pct:+.1%}, floor={trail_floor:+.1%}"
                    )
                
                # Take-profit check: profit exceeds target
                elif unrealized_pnl > 0 and unrealized_pnl_pct > target_profit:
                    close_reason = "TAKE_PROFIT"
                    self.logger.info(
                        f"PROFIT TARGET hit for {symbol}: "
                        f"P&L: ${unrealized_pnl:+,.2f} ({unrealized_pnl_pct:+.1%})"
                    )
                
                # --- PHASE 2: TIME-BASED PROFIT ACCELERATION ---
                # If held > 50% of DTE and at +25% profit, close early
                if close_reason is None and unrealized_pnl_pct >= 0.25:
                    entry_time_str = None
                    sig_dte = None
                    if isinstance(position, dict):
                        entry_time_str = position.get("entry_time")
                        sig = position.get("signal")
                        if sig:
                            sig_dte = getattr(sig, "dte", None)
                    if entry_time_str and sig_dte and sig_dte > 0:
                        try:
                            entry_dt = datetime.fromisoformat(entry_time_str)
                            held_days = (datetime.now() - entry_dt).days
                            half_dte = sig_dte / 2.0
                            if held_days >= half_dte:
                                close_reason = "TIME_ACCEL_PROFIT"
                                self.logger.info(
                                    f"TIME-ACCEL profit for {symbol}: "
                                    f"+{unrealized_pnl_pct:.0%} after {held_days}d "
                                    f"(>{half_dte:.0f}d = 50% of {sig_dte}DTE)"
                                )
                        except (ValueError, TypeError):
                            pass

                # DTE-based time exit: close positions nearing expiration
                if close_reason is None:
                    from datetime import date as _date_cls
                    today = _date_cls.today()
                    for occ_sym in alpaca_map:
                        parsed_occ = parse_occ_symbol(occ_sym)
                        if parsed_occ is not None:
                            occ_underlying = parsed_occ['underlying']
                        else:
                            occ_underlying = ''
                            for ch in occ_sym:
                                if ch.isdigit():
                                    break
                                occ_underlying += ch
                        if occ_underlying.upper() != (symbol or '').upper():
                            continue
                        exp_date = parsed_occ['expiry_date'] if parsed_occ else self._parse_occ_expiration(occ_sym)
                        if exp_date is not None:
                            dte = (exp_date - today).days
                            if dte < dte_close_threshold:
                                close_reason = "DTE_EXIT"
                                self.logger.warning(
                                    f"DTE EXIT for {symbol} ({occ_sym}): "
                                    f"only {dte} DTE remaining (threshold={dte_close_threshold})"
                                )
                                break
                
                if close_reason:
                    positions_to_close.append((position, close_reason, unrealized_pnl))
                    # Clean up HWM on close
                    self._options_hwm.pop(symbol, None)
                else:
                    self.logger.info(
                        f"  Position {symbol}: P&L ${unrealized_pnl:+,.2f} ({unrealized_pnl_pct:+.1%}), "
                        f"HWM={prev_hwm:+.1%}"
                    )
            except Exception as e:
                self.logger.warning(f"Error checking position: {e}")
        
        # Close triggered positions — ACTUALLY CLOSE ON ALPACA (issue #4)
        for position, reason, pnl in positions_to_close:
            symbol = None
            if isinstance(position, dict):
                signal = position.get("signal")
                symbol = getattr(signal, "symbol", None) or position.get("symbol")
            elif hasattr(position, "symbol"):
                symbol = position.symbol
            
            self.logger.info(f"Closing position: {symbol} ({reason}) P&L=${pnl:+,.2f}")
            
            # Send closing orders to Alpaca for all option legs matching this underlying
            try:
                alpaca_positions = self.trade_executor.trading_client.get_all_positions()
                for ap in alpaca_positions:
                    parsed_occ_close = parse_occ_symbol(ap.symbol)
                    if parsed_occ_close is not None:
                        occ_underlying = parsed_occ_close['underlying']
                    else:
                        occ_underlying = ''
                        for ch in ap.symbol:
                            if ch.isdigit():
                                break
                            occ_underlying += ch
                    if occ_underlying.upper() == (symbol or '').upper():
                        self.logger.info(f"  Closing Alpaca position: {ap.symbol}")
                        try:
                            self.trade_executor.trading_client.close_position(ap.symbol)
                        except Exception as close_err:
                            self.logger.error(f"  Failed to close {ap.symbol}: {close_err}")
            except Exception as e:
                self.logger.error(f"Failed to close positions on Alpaca for {symbol}: {e}")
            
            self.current_positions.remove(position)
            self.stats["positions_closed"] += 1
            self.stats["total_pnl"] += pnl
    
    async def _check_risk_limits(self) -> bool:
        """Step 6: Verify portfolio risk is within configured limits.

        Also triggers automatic delta hedging when thresholds are breached
        (Phase 7 — Bug 4 fix).

        Returns:
            ``True`` if all limits are satisfied.
        """
        # Phase 7: Automatic delta hedging with SPY shares
        await self._auto_delta_hedge()

        # Check portfolio delta
        max_delta = self.config["max_portfolio_delta"]
        if abs(self.portfolio_delta) > max_delta:
            self.logger.warning(f"Portfolio delta {self.portfolio_delta:.2f} exceeds max {max_delta}")
            return False
        
        # Check max positions
        max_positions = self.config["max_positions"]
        if len(self.current_positions) > max_positions:
            self.logger.warning(f"Position count {len(self.current_positions)} exceeds max {max_positions}")
            return False
        
        return True

    # ================================================================== #
    # PHASE 7: AUTOMATIC DELTA HEDGING
    # ================================================================== #

    async def _auto_delta_hedge(self) -> None:
        """Automatically hedge portfolio delta by trading SPY shares.

        When |portfolio_delta| exceeds the configured threshold (default 150),
        submit a market order for SPY shares to bring delta closer to neutral.

        Safety:
        - Only hedges once per cycle (no rapid-fire).
        - Caps hedge size at 200 shares per cycle.
        - Uses market orders for reliability.
        - Logs every hedge action.
        """
        DELTA_HEDGE_THRESHOLD = self.config.get("auto_delta_hedge_threshold", 150.0)
        MAX_HEDGE_SHARES = 200  # cap hedge size per cycle

        if abs(self.portfolio_delta) <= DELTA_HEDGE_THRESHOLD:
            return  # within tolerance

        # Determine hedge direction and size
        # If delta is +300, we need to sell ~300 shares of SPY to neutralize
        hedge_shares = -int(round(self.portfolio_delta))  # negate to offset
        hedge_shares = max(-MAX_HEDGE_SHARES, min(MAX_HEDGE_SHARES, hedge_shares))

        if hedge_shares == 0:
            return

        side_str = "sell" if hedge_shares < 0 else "buy"
        abs_shares = abs(hedge_shares)

        self.logger.info(
            f"🔀 AUTO DELTA HEDGE: portfolio delta={self.portfolio_delta:+.1f} "
            f"exceeds ±{DELTA_HEDGE_THRESHOLD:.0f} → "
            f"{side_str.upper()} {abs_shares} shares SPY"
        )

        try:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide as AlpacaOrderSide, TimeInForce

            order_req = MarketOrderRequest(
                symbol="SPY",
                qty=abs_shares,
                side=AlpacaOrderSide.SELL if hedge_shares < 0 else AlpacaOrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
            order = self.trade_executor.trading_client.submit_order(order_req)
            self.logger.info(
                f"✓ Delta hedge order submitted: {side_str} {abs_shares} SPY "
                f"order_id={order.id}"
            )
            await self._send_discord_notification(
                f"🔀 Delta Hedge: {side_str.upper()} {abs_shares} SPY "
                f"(portfolio delta was {self.portfolio_delta:+.1f})"
            )
        except Exception as exc:
            self.logger.error(f"Delta hedge order failed: {exc}")

    def _has_position(self, symbol: str) -> bool:
        """Check if we have a position in symbol."""
        for pos in self.current_positions:
            # Handle dict positions (normal case)
            if isinstance(pos, dict):
                if pos.get("symbol") == symbol:
                    return True
                signal = pos.get("signal")
                if signal is not None and getattr(signal, "symbol", None) == symbol:
                    return True
            # Handle string positions (legacy)
            elif isinstance(pos, str) and pos == symbol:
                return True
        return False
    
    # ------------------------------------------------------------------ #
    # ALPACA-SOURCED POSITION HELPERS (2026-02-21 — survive restarts)
    # ------------------------------------------------------------------ #

    def _get_alpaca_option_positions(self) -> Dict[str, dict]:
        """Return a dict of OCC symbol -> position data from Alpaca.

        Only includes option positions (symbol length > 6, contains digits).
        This is the SINGLE SOURCE OF TRUTH for position state — never rely
        on self.current_positions which drifts on service restart.

        Returns:
            {occ_symbol: {"qty": float, "cost_basis": float,
                          "market_value": float, "unrealized_pl": float}} 
        """
        result: Dict[str, dict] = {}
        try:
            alpaca_positions = self.trade_executor.trading_client.get_all_positions()
            for ap in alpaca_positions:
                sym = ap.symbol or ""
                # OCC option symbols are >6 chars and contain digits in first 6
                if len(sym) <= 6:
                    continue
                has_digit = any(c.isdigit() for c in sym[:8])
                if not has_digit:
                    continue
                result[sym] = {
                    "qty": float(ap.qty) if ap.qty else 0,
                    "cost_basis": float(ap.cost_basis) if ap.cost_basis else 0.0,
                    "market_value": float(ap.market_value) if ap.market_value else 0.0,
                    "unrealized_pl": float(ap.unrealized_pl) if ap.unrealized_pl else 0.0,
                    "unrealized_plpc": float(ap.unrealized_plpc) if ap.unrealized_plpc else 0.0,
                    "symbol": sym,
                }
        except Exception as e:
            self.logger.error(f"Failed to query Alpaca positions: {e}")
        return result

    @staticmethod
    def _parse_occ_expiration(occ_symbol: str):
        """Extract expiration date from OCC symbol.

        OCC format: AAPL260320P00230000
                     ^^^^------  underlying
                         ^^^^^^  YYMMDD
        Returns datetime.date or None on failure.
        """
        from datetime import date as _date
        try:
            # Find where digits start (end of underlying ticker)
            idx = 0
            for ch in occ_symbol:
                if ch.isdigit():
                    break
                idx += 1
            date_str = occ_symbol[idx:idx + 6]  # YYMMDD
            if len(date_str) < 6:
                return None
            yy = int(date_str[0:2])
            mm = int(date_str[2:4])
            dd = int(date_str[4:6])
            return _date(2000 + yy, mm, dd)
        except (ValueError, IndexError):
            return None

    @staticmethod
    def _make_client_order_id(occ_symbol: str, cycle_num: int) -> str:
        """Generate a deterministic client_order_id for idempotent orders.

        Format: {occ_symbol}_{date}_{cycle} — Alpaca rejects duplicate
        client_order_ids, so retries within the same cycle are no-ops.
        Truncated to 48 chars (Alpaca limit).
        """
        from datetime import date as _date
        day_str = _date.today().isoformat()
        raw = f"{occ_symbol}_{day_str}_{cycle_num}"
        return raw[:48]

    # ------------------------------------------------------------------ #
    # DAILY RISK-CONTROL HELPERS  (2026-02-20 emergency fix)
    # ------------------------------------------------------------------ #

    def _reset_daily_counters_if_needed(self):
        """Reset daily counters at the start of each new trading day.

        Resets spend, trade count, OCC set, and the P&L circuit breaker flag.
        """
        today = datetime.now(ZoneInfo("America/New_York")).date()
        if self._daily_tracking_date != today:
            self.logger.info(
                f"New trading day {today} — resetting daily options counters "
                f"(prev spent=${self._daily_options_spent:.0f}, "
                f"trades={self._daily_options_trades})"
            )
            self._daily_options_spent = 0.0
            self._daily_options_trades = 0
            self._daily_tracking_date = today
            self._day_start_portfolio = self.portfolio_value
            self._executed_occ_symbols.clear()
            self._pnl_circuit_breaker_tripped = False

    def _daily_loss_exceeded(self) -> bool:
        """Return True if daily P/L has breached the DAILY_LOSS_STOP."""
        daily_pnl = self.portfolio_value - self._day_start_portfolio
        if daily_pnl < DAILY_LOSS_STOP:
            self.logger.critical(
                f"⛔ DAILY LOSS STOP triggered: P/L=${daily_pnl:,.0f} "
                f"(limit={DAILY_LOSS_STOP}). All trading halted for today."
            )
            return True
        return False

    # ------------------------------------------------------------------ #
    # PHASE 4: DAILY P&L CIRCUIT BREAKER  (-$500 realized + unrealized)
    # ------------------------------------------------------------------ #

    def _check_daily_pnl_circuit_breaker(self) -> bool:
        """Check if the daily P&L circuit breaker should trip.

        Computes realized + unrealized P&L since market open.  If the
        combined number drops below ``DAILY_PNL_CIRCUIT_BREAKER`` (-$500),
        all new entries are halted for the remainder of the session.
        The flag resets automatically at the next market open via
        ``_reset_daily_counters_if_needed``.

        Returns:
            True if trading is allowed (circuit breaker NOT tripped).
            False if trading should stop.
        """
        if self._pnl_circuit_breaker_tripped:
            return False  # already tripped earlier today

        # Compute unrealized P&L from Alpaca positions
        unrealized_pnl = 0.0
        try:
            positions = self.trade_executor.trading_client.get_all_positions()
            for pos in positions:
                sym = pos.symbol or ""
                if len(sym) > 6:  # option position
                    unrealized_pnl += float(pos.unrealized_pl or 0)
        except Exception as e:
            self.logger.warning(f"Circuit breaker: cannot fetch positions: {e}")
            return True  # fail-open

        # Realized component = portfolio change minus unrealized portion
        total_day_pnl = (self.portfolio_value - self._day_start_portfolio)

        if total_day_pnl < DAILY_PNL_CIRCUIT_BREAKER:
            self._pnl_circuit_breaker_tripped = True
            self.logger.critical(
                f"🚨 DAILY P&L CIRCUIT BREAKER TRIPPED: "
                f"day P&L=${total_day_pnl:,.2f} (unrealized=${unrealized_pnl:,.2f}) "
                f"< threshold=${DAILY_PNL_CIRCUIT_BREAKER:,.0f}. "
                f"All new entries halted until next market open."
            )
            return False

        return True

    def _options_budget_allows(self, estimated_cost: float) -> bool:
        """Check whether we can afford *estimated_cost* within today's budget."""
        if self._daily_options_spent + estimated_cost > MAX_DAILY_OPTIONS_SPEND:
            self.logger.warning(
                f"⚠️ Daily options budget exhausted: "
                f"spent=${self._daily_options_spent:.0f} + "
                f"new=${estimated_cost:.0f} > "
                f"limit=${MAX_DAILY_OPTIONS_SPEND}"
            )
            return False
        return True

    def _options_trade_count_allows(self) -> bool:
        """Check whether we haven't exceeded MAX_DAILY_OPTIONS_TRADES."""
        if self._daily_options_trades >= MAX_DAILY_OPTIONS_TRADES:
            self.logger.warning(
                f"⚠️ Daily options trade limit reached: "
                f"{self._daily_options_trades}/{MAX_DAILY_OPTIONS_TRADES}"
            )
            return False
        return True

    def _options_position_count_allows(self) -> bool:
        """Check open options positions < MAX_OPTIONS_POSITIONS."""
        # Count option positions from Alpaca (OCC symbols are > 6 chars)
        option_count = sum(
            1 for p in self.current_positions
            if isinstance(p, dict)
            and len(str((p.get("signal") and getattr(p["signal"], "occ_symbol", "")) or "")) > 6
        )
        if option_count >= MAX_OPTIONS_POSITIONS:
            self.logger.warning(
                f"⚠️ Max options positions reached: "
                f"{option_count}/{MAX_OPTIONS_POSITIONS}"
            )
            return False
        return True

    def _is_duplicate_contract(self, occ_symbol: str) -> bool:
        """Return True if we already traded or hold *occ_symbol* today."""
        if not occ_symbol:
            return False
        if occ_symbol in self._executed_occ_symbols:
            self.logger.warning(
                f"⚠️ Duplicate contract blocked: {occ_symbol} already traded today"
            )
            return True
        # Also check current Alpaca positions
        for pos in self.current_positions:
            if isinstance(pos, dict):
                sig = pos.get("signal")
                if sig and getattr(sig, "occ_symbol", None) == occ_symbol:
                    self.logger.warning(
                        f"⚠️ Duplicate contract blocked: already holding {occ_symbol}"
                    )
                    return True
        return False

    def _log_cycle_summary(self):
        """Log summary of current cycle."""
        self.logger.info(f"Portfolio Value: ${self.portfolio_value:,.0f}")
        self.logger.info(f"Open Positions: {len(self.current_positions)}")
        self.logger.info(f"Portfolio Delta: {self.portfolio_delta:.2f}")
        self.logger.info(f"Total Trades: {self.stats['trades_executed']}")
        self.logger.info(f"Total P&L: ${self.stats['total_pnl']:,.0f}")
        # Phase 6: Exit manager summary
        exit_summary = self.exit_manager.get_summary()
        self.logger.info(
            f"ExitManager: {exit_summary['open_positions']} tracked, "
            f"open_pnl=${exit_summary['open_pnl']:+,.2f}, "
            f"exits={exit_summary['stats']['total_exits']}"
        )

    # ================================================================== #
    # PHASE 6: EXIT MANAGER INTEGRATION
    # ================================================================== #

    def _register_with_exit_manager(self, signal, position_size, result):
        """Register a newly executed trade with the ExitManager.

        Maps the signal/execution to the correct ExitManager registration
        method based on strategy type.
        """
        try:
            strategy = signal.strategy
            underlying = signal.symbol
            premium_estimate = signal.expected_premium if signal.expected_premium and signal.expected_premium > 0 else 0.50
            strike_width = 5.0  # Default width

            if strategy in ("credit_spread", "put_spread", "call_spread"):
                # Spread — 2 legs
                legs = result.legs if result.legs else []
                short_occ = ""
                long_occ = ""
                for leg in legs:
                    if hasattr(leg, 'side'):
                        if leg.side == OrderSide.SELL or str(leg.side).lower() == "sell":
                            short_occ = leg.symbol
                        else:
                            long_occ = leg.symbol

                if short_occ and long_occ:
                    net_credit = premium_estimate
                    max_profit = net_credit * 100 * position_size.contracts
                    max_loss = (strike_width - net_credit) * 100 * position_size.contracts
                    self.exit_manager.register_spread(
                        underlying=underlying,
                        short_occ=short_occ,
                        long_occ=long_occ,
                        qty=position_size.contracts,
                        net_credit=net_credit,
                        max_profit=max_profit,
                        max_loss=max_loss,
                        strategy=strategy,
                    )
                    self.logger.info(f"Registered spread with ExitManager: {underlying} ({strategy})")

            elif strategy == "iron_condor":
                # Iron condor — 4 legs
                legs = result.legs if result.legs else []
                put_long = put_short = call_short = call_long = ""
                for leg in legs:
                    sym = leg.symbol
                    side_str = str(getattr(leg, 'side', '')).lower()
                    is_put = 'P' in sym[6:8] if len(sym) > 8 else False
                    if is_put and "buy" in side_str:
                        put_long = sym
                    elif is_put and "sell" in side_str:
                        put_short = sym
                    elif not is_put and "sell" in side_str:
                        call_short = sym
                    elif not is_put and "buy" in side_str:
                        call_long = sym

                if put_long and put_short and call_short and call_long:
                    net_credit = premium_estimate
                    max_profit = net_credit * 100 * position_size.contracts
                    max_loss = (strike_width - net_credit) * 100 * position_size.contracts
                    self.exit_manager.register_iron_condor(
                        underlying=underlying,
                        put_long_occ=put_long,
                        put_short_occ=put_short,
                        call_short_occ=call_short,
                        call_long_occ=call_long,
                        qty=position_size.contracts,
                        net_credit=net_credit,
                        max_profit=max_profit,
                        max_loss=max_loss,
                    )
                    self.logger.info(f"Registered iron condor with ExitManager: {underlying}")

            else:
                # Single leg
                occ_sym = getattr(signal, 'occ_symbol', '') or ''
                side = "buy" if signal.signal_type == SignalType.BUY else "sell"
                entry_price = premium_estimate
                max_profit = entry_price * 100 * position_size.contracts
                max_loss = entry_price * 100 * position_size.contracts
                if occ_sym:
                    self.exit_manager.register_single_leg(
                        underlying=underlying,
                        occ_symbol=occ_sym,
                        side=side,
                        qty=position_size.contracts,
                        entry_price=entry_price,
                        max_profit=max_profit,
                        max_loss=max_loss,
                        strategy=strategy,
                    )
                    self.logger.info(f"Registered single leg with ExitManager: {underlying} ({strategy})")

        except Exception as e:
            self.logger.warning(f"Failed to register with ExitManager: {e}")

    async def _run_exit_manager(self):
        """Run the ExitManager to check all tracked positions for exits.

        Syncs orphaned Alpaca positions, checks exit triggers, and
        executes closing orders (MLEG where possible).
        """
        try:
            # Sync any orphaned Alpaca positions not yet tracked
            alpaca_opts = self._get_alpaca_option_positions()
            self.exit_manager.sync_from_alpaca_state(alpaca_opts)

            # Check all positions for exit triggers
            exit_actions = await self.exit_manager.check_all_positions()

            if exit_actions:
                self.logger.info(
                    f"ExitManager: {len(exit_actions)} exit(s) triggered"
                )

            for action in exit_actions:
                self.logger.info(
                    f"  EXIT: {action.underlying} ({action.strategy}) "
                    f"reason={action.reason.value} P&L=${action.current_pnl:+,.2f}"
                )
                success = await self.exit_manager.execute_exit(action)
                if success:
                    self.stats["positions_closed"] += 1
                    self.stats["total_pnl"] += action.current_pnl
                    # Also remove from current_positions
                    self.current_positions = [
                        p for p in self.current_positions
                        if not self._position_matches_exit(p, action)
                    ]
                    await self._send_discord_notification(
                        f"🔒 Position Closed: {action.underlying} ({action.strategy}) "
                        f"reason={action.reason.value} P&L=${action.current_pnl:+,.2f} "
                        f"{action.details}"
                    )
                else:
                    self.logger.error(
                        f"Failed to execute exit for {action.underlying}"
                    )

        except Exception as e:
            self.logger.error(f"ExitManager check failed: {e}", exc_info=True)

    def _position_matches_exit(self, position, action: ExitAction) -> bool:
        """Check if an in-memory position matches an exit action."""
        symbol = None
        if isinstance(position, dict):
            signal = position.get("signal")
            symbol = getattr(signal, "symbol", None) or position.get("symbol")
        elif hasattr(position, "symbol"):
            symbol = position.symbol
        return (symbol or "").upper() == action.underlying.upper()

    def _log_daily_performance(self):
        """Log daily performance metrics (Phase 6).

        Calls DailyPerformanceLogger.log_daily() once per day (idempotent).
        Computes realized P&L from ExitManager + unrealized from Alpaca.
        """
        try:
            # Daily P&L components
            realized_pnl = self.exit_manager.stats.get("total_realized_pnl", 0.0)
            unrealized_pnl = sum(
                p.current_pnl for p in self.exit_manager.positions.values()
            )
            total_daily_pnl = self.portfolio_value - self._day_start_portfolio

            n_trades = self.stats.get("trades_executed", 0)
            n_positions = len(self.current_positions) + len(self.exit_manager.positions)

            snap = self.daily_perf_logger.log_daily(
                equity=self.portfolio_value,
                daily_pnl=total_daily_pnl,
                n_positions=n_positions,
                n_trades=n_trades,
                turnover_pct=0.0,
            )

            if snap is not None:
                self.logger.info(
                    f"Daily P&L: ${total_daily_pnl:+,.2f} "
                    f"(realized=${realized_pnl:+,.2f}, unrealized=${unrealized_pnl:+,.2f}) "
                    f"equity=${self.portfolio_value:,.2f}"
                )
        except Exception as e:
            self.logger.debug(f"Daily performance logging failed: {e}")

    @staticmethod
    def _to_json_native(v):
        """Convert a value to a JSON-native type preserving numeric precision."""
        if v is None or isinstance(v, (bool, int, float, str)):
            return v
        if isinstance(v, (list, tuple)):
            return [AutonomousTradingEngine._to_json_native(i) for i in v]
        if isinstance(v, dict):
            return {str(k): AutonomousTradingEngine._to_json_native(val) for k, val in v.items()}
        if isinstance(v, datetime):
            return v.isoformat()
        if hasattr(v, 'value'):  # Enum
            return v.value
        if hasattr(v, '__dict__'):
            return {k: AutonomousTradingEngine._to_json_native(val) for k, val in vars(v).items()}
        return str(v)

    def _save_state(self):
        """Save engine state to file with proper serialization (issue #16)."""
        # Serialize positions to plain dicts with JSON-native types only
        serializable_positions = []
        for pos in self.current_positions:
            if isinstance(pos, dict):
                serializable_positions.append(self._to_json_native(pos))
            elif hasattr(pos, '__dict__'):
                serializable_positions.append(self._to_json_native(vars(pos)))
            else:
                serializable_positions.append(str(pos))

        state = {
            "portfolio_value": float(self.portfolio_value),
            "portfolio_delta": float(self.portfolio_delta),
            "current_positions": serializable_positions,
            "stats": self._to_json_native(self.stats),
            "last_update": datetime.now().isoformat(),
            "exit_manager": self.exit_manager.save_state(),
        }
        
        try:
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
    
    def _load_state(self):
        """Load engine state from file."""
        if not os.path.exists(self.state_file):
            return
        
        try:
            with open(self.state_file, "r") as f:
                state = json.load(f)
            
            self.portfolio_value = state.get("portfolio_value", self.portfolio_value)
            self.portfolio_delta = state.get("portfolio_delta", 0.0)
            self.current_positions = state.get("current_positions", [])
            self.stats = state.get("stats", self.stats)

            # Sanity: delta must be consistent with positions.  If no
            # positions are tracked the delta should be zero.
            if not self.current_positions and self.portfolio_delta != 0.0:
                self.logger.warning(
                    f"Stale delta={self.portfolio_delta:.1f} with 0 positions — resetting to 0"
                )
                self.portfolio_delta = 0.0
            
            self.logger.info(f"Loaded state from {self.state_file}")

            # Phase 6: Restore ExitManager state
            exit_state = state.get("exit_manager")
            if exit_state:
                try:
                    self.exit_manager.load_state(exit_state)
                    self.logger.info(
                        f"Restored ExitManager: {len(self.exit_manager.positions)} tracked positions"
                    )
                except Exception as e:
                    self.logger.warning(f"ExitManager state restore failed: {e}")
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
    
    def _backfill_iv_data(self):
        """
        Backfill historical IV data on startup to enable IV rank calculations.
        
        Fixes: "Insufficient data for IV rank (need 20 days)" errors
        """
        try:
            self.logger.info("🔄 Checking IV data cache on startup...")
            
            # Get current IV data stats
            stats = self.iv_data_manager.get_stats()
            symbols_cached = stats.get('symbols', 0)
            total_records = stats.get('total_records', 0)
            
            self.logger.info(
                f"Current IV cache: {total_records} records across {symbols_cached} symbols"
            )
            
            # Get trading universe
            universe = get_universe()
            
            # Backfill for each symbol if needed
            for symbol in universe:
                # Check if we have sufficient data
                iv_rank = self.iv_data_manager.get_iv_rank(symbol, lookback_days=252)
                
                if iv_rank is None:
                    self.logger.info(f"Backfilling IV data for {symbol}...")
                    records = self.iv_data_manager.backfill_historical_iv(symbol, days=252)
                    
                    if records > 0:
                        self.logger.info(f"✓ {symbol}: Added {records} days of IV history")
                    else:
                        self.logger.warning(f"✗ {symbol}: Backfill failed, using synthetic...")
                        records = self.iv_data_manager.backfill_synthetic_data(symbol, days=252)
                        self.logger.info(f"✓ {symbol}: Added {records} days of synthetic IV")
                else:
                    self.logger.info(f"✓ {symbol}: IV rank = {iv_rank:.1f}% (data OK)")
            
            # Log final stats
            stats = self.iv_data_manager.get_stats()
            self.logger.info(
                f"✅ IV backfill complete: {stats['total_records']} records, "
                f"{stats['symbols']} symbols"
            )
            
        except Exception as e:
            self.logger.error(f"IV backfill failed (non-fatal): {e}")
    
    # ========================================================================
    # ENHANCED METHODS (NEW)
    # ========================================================================
    
    async def _update_regime_and_weights(self):
        """
        Update market regime detection and rebalance strategy weights.
        
        ENHANCED: Now uses ManifoldRegimeDetector alongside HMM detector.
        Also runs GARCH vol forecast to adjust risk parameters.
        """
        # Fit regime detector on first run
        if not self.regime_fitted:
            try:
                self.logger.info("Fitting regime detector for first time...")
                await self.regime_detector.fit()
                self.regime_fitted = True
                self.logger.info("✓ Regime detector fitted")
            except Exception as e:
                self.logger.error(f"Failed to fit regime detector: {e}")
                return
        
        # Detect current regime (HMM)
        try:
            regime_state = await self.regime_detector.detect_current_regime()
            old_regime = self.current_regime
            self.current_regime = regime_state.current_regime
            
            self.logger.info(
                f"HMM Regime: {self.current_regime.value} "
                f"(confidence: {regime_state.confidence:.1%})"
            )

            # ManifoldRegimeDetector cross-validation — requires price data + vol
            if self.manifold_detector is not None:
                try:
                    import yfinance as yf
                    spy_data = yf.download("SPY", period="1y", interval="1d", progress=False)
                    if not spy_data.empty and len(spy_data) > 30:
                        prices = spy_data["Close"].values.flatten()
                        log_rets = np.diff(np.log(prices[-21:]))
                        realized_vol = float(np.std(log_rets) * np.sqrt(252))
                        # Implied vol from VIX
                        implied_vol = realized_vol * 1.2
                        try:
                            vix_data = yf.download("^VIX", period="5d", interval="1d", progress=False)
                            if not vix_data.empty:
                                implied_vol = float(vix_data["Close"].values.flatten()[-1]) / 100.0
                        except Exception:
                            pass
                        manifold_result = self.manifold_detector.detect_regime(
                            prices, realized_vol, implied_vol
                        )
                        manifold_regime = getattr(manifold_result, 'regime', 'unknown')
                        manifold_conf = getattr(manifold_result, 'confidence', 0.0)
                        self.logger.info(
                            f"Manifold Regime: {manifold_regime} (confidence: {manifold_conf:.1%})"
                        )
                        # Store for use in signal scoring
                        self._last_manifold_state = manifold_result
                except Exception as e:
                    self.logger.debug(f"Manifold regime detection failed: {e}")

            # GARCH vol overlay
            if self.garch_model is not None:
                try:
                    garch_forecast = self.garch_model.fit_and_forecast("SPY", horizon=5)
                    self._last_garch_vol = garch_forecast.current_vol ** 2  # variance for Heston v0
                    self.logger.info(
                        f"GARCH Vol: current={garch_forecast.current_vol:.1%}, "
                        f"5d_forecast={garch_forecast.forecast_vols[-1]:.1%}, "
                        f"persistence={garch_forecast.params.persistence:.4f}"
                    )
                    # Adjust position sizing risk if vol is elevated
                    if garch_forecast.current_vol > 0.30:
                        self.logger.warning("Elevated vol detected — tightening risk limits")
                except Exception as e:
                    self.logger.debug(f"GARCH update failed: {e}")
            
            # Rebalance weights if regime changed
            if old_regime != self.current_regime or self.stats["cycles_run"] % 20 == 0:
                self.logger.info("Rebalancing strategy weights...")
                new_weights = await self.weight_optimizer.rebalance(
                    regime=self.current_regime,
                    force=(old_regime != self.current_regime)
                )
                self.logger.info(f"Updated strategy weights: {new_weights}")
        
        except Exception as e:
            self.logger.error(f"Regime update failed: {e}")
    
    async def _check_concentration_risk(self) -> bool:
        """
        Check for portfolio concentration risk.
        
        Returns:
            True if safe to proceed, False if concentration limits exceeded
        """
        if len(self.current_positions) == 0:
            return True
        
        try:
            # Convert positions to CorrelationManager format
            corr_positions = []
            for pos in self.current_positions:
                signal_obj = None
                if isinstance(pos, dict):
                    signal_obj = pos.get("signal")
                elif isinstance(pos, str):
                    signal_obj = pos

                if not signal_obj:
                    continue

                symbol = None
                strategy_type = "unknown"
                delta = 0.0

                if isinstance(signal_obj, Signal):
                    symbol = signal_obj.symbol
                    strategy_type = signal_obj.strategy
                    delta = signal_obj.delta or 0.0
                elif isinstance(signal_obj, dict):
                    symbol = signal_obj.get("symbol")
                    strategy_type = signal_obj.get("strategy", strategy_type)
                    delta = float(signal_obj.get("delta", 0.0) or 0.0)
                elif isinstance(signal_obj, str):
                    symbol = signal_obj

                if not symbol:
                    continue

                corr_positions.append(CorrPosition(
                    symbol=str(symbol),
                    quantity=1,
                    entry_price=1.0,
                    current_price=1.0,
                    strategy_type=str(strategy_type),
                    delta=delta,
                    gamma=0.0,
                    theta=0.0,
                    vega=0.0,
                    notional_value=1000.0,  # Simplified
                    sector="Unknown",
                ))
            
            if len(corr_positions) == 0:
                return True
            
            # Build correlation matrix
            corr_matrix = await self.correlation_manager.build_correlation_matrix(corr_positions)
            
            # Check for alerts
            alerts = self.correlation_manager.detect_concentration_risk(
                positions=corr_positions,
                portfolio_value=self.portfolio_value,
                correlation_matrix=corr_matrix,
            )
            
            # Log alerts
            critical_alerts = [a for a in alerts if a.severity == "critical"]
            if critical_alerts:
                for alert in critical_alerts:
                    self.logger.warning(f"⚠ CRITICAL: {alert.message}")
                return False
            
            if alerts:
                for alert in alerts[:3]:  # Show top 3
                    self.logger.warning(f"⚠ {alert.severity.upper()}: {alert.message}")
            
            return True
        
        except Exception as e:
            self.logger.error(f"Concentration check failed: {e}")
            return True  # Allow trading to proceed on error
    
    async def _get_vol_surface_signals(self, symbols: List[str]) -> List[Signal]:
        """
        Generate additional signals from volatility surface analysis.
        
        Args:
            symbols: Symbols to analyze
        
        Returns:
            List of vol-based signals
        """
        vol_signals = []
        
        # Only analyze a few symbols per cycle to avoid slowdown
        for symbol in symbols[:2]:
            try:
                # Build surface
                surface = await self.vol_surface_engine.build_iv_surface(symbol)
                
                # Detect anomalies
                anomalies = await self.vol_surface_engine.detect_anomalies(surface)
                
                # Generate arb signals
                arb_signals = await self.vol_surface_engine.generate_arb_signals(
                    anomalies, surface
                )
                
                # Convert to Signal format (simplified)
                for arb in arb_signals[:1]:  # Max 1 per symbol
                    vol_signals.append(Signal(
                        symbol=symbol,
                        signal_type=SignalType.BUY if "buy" in arb.signal_type else SignalType.SELL,
                        signal_source="vol_surface",
                        strategy="vol_arb",
                        confidence=arb.confidence,
                        timestamp=datetime.now(),
                        reason=arb.reasoning,
                    ))
            
            except Exception as e:
                self.logger.debug(f"Vol surface analysis failed for {symbol}: {e}")
                continue
        
        return vol_signals
    
    async def _get_cointegration_signals(self, symbols: List[str]) -> List[Signal]:
        """
        Generate pairs trading signals from cointegration analysis.
        
        Args:
            symbols: Symbols to test for pairs
        
        Returns:
            List of pairs signals
        """
        # Only scan for pairs periodically (every 50 cycles)
        if self.stats["cycles_run"] % 50 != 1:
            return []
        
        try:
            self.logger.info("Scanning for cointegrated pairs...")
            pairs = await self.cointegration_engine.find_cointegrated_pairs(
                symbols=symbols[:10],  # Limit to avoid slowdown
                max_pairs=5,
            )
            
            if pairs:
                self.logger.info(f"Found {len(pairs)} cointegrated pairs")
        
        except Exception as e:
            self.logger.error(f"Cointegration scan failed: {e}")
        
        return []  # Could convert pairs signals to Signal format

    # ========================================================================
    # PHASE 5: REGIME-BASED OPTIONS STRATEGY SELECTION
    # ========================================================================

    def _get_regime_strategies(self) -> List[str]:
        """
        Return the preferred options strategies based on current regime.

        Covered calls, cash-secured puts, vertical spreads, iron condors
        are selected depending on the detected market regime.
        """
        if self.current_regime is None:
            return ["credit_spread", "iron_condor"]

        regime_str = str(self.current_regime.value).upper()

        if "BULL_LOW" in regime_str:
            return ["cash_secured_put", "covered_call", "call_spread"]
        elif "BULL_HIGH" in regime_str:
            return ["iron_condor", "credit_spread", "covered_call"]
        elif "BEAR_LOW" in regime_str:
            return ["put_spread", "covered_call"]
        elif "BEAR_HIGH" in regime_str:
            return ["iron_condor", "put_spread"]
        else:
            return ["iron_condor", "credit_spread"]

    async def _send_discord_notification(self, message: str) -> None:
        """Send a trade notification to Discord webhook (if configured)."""
        webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
        if not webhook_url:
            return
        try:
            import aiohttp
            payload = {"content": message}
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload, timeout=aiohttp.ClientTimeout(total=10)):
                    pass
        except Exception as e:
            self.logger.debug(f"Discord notification failed: {e}")

    async def _log_signal_with_reasoning(self, signal, execution_result=None) -> None:
        """Log every signal with full reasoning for audit trail."""
        reasoning = (
            f"📊 **Signal: {signal.symbol}** ({signal.strategy})\n"
            f"  Type: {signal.signal_type.value} | Confidence: {signal.confidence:.1%}\n"
            f"  IV Rank: {getattr(signal, 'iv_rank', 'N/A')} | "
            f"Regime: {self.current_regime.value if self.current_regime else 'unknown'}\n"
        )
        if execution_result:
            if execution_result.success:
                reasoning += f"  ✅ EXECUTED: Order {execution_result.order_id}\n"
            else:
                reasoning += f"  ❌ FAILED: {execution_result.error_message}\n"

        self.logger.info(reasoning)
        await self._send_discord_notification(reasoning)

    async def _shutdown(self):
        """Graceful shutdown."""
        self.logger.info("Shutting down autonomous engine...")
        
        # Save final state
        self._save_state()
        
        # Log final stats
        self.logger.info("="*60)
        self.logger.info("FINAL STATISTICS")
        self.logger.info("="*60)
        for key, value in self.stats.items():
            self.logger.info(f"{key}: {value}")
        
        self.logger.info("Shutdown complete")


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous options trading engine")
    parser.add_argument(
        "--portfolio-value",
        type=float,
        default=100000,
        help="Starting portfolio value in dollars (default: 100000)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    async def _runner() -> None:
        engine = AutonomousTradingEngine(portfolio_value=args.portfolio_value)

        loop = asyncio.get_running_loop()
        try:
            import signal

            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.add_signal_handler(sig, engine.request_shutdown)
                except NotImplementedError:
                    signal.signal(sig, lambda *_: engine.request_shutdown())
        except Exception:
            # If signal wiring fails for any reason, the engine can still be stopped with Ctrl+C.
            pass

        await engine.run_forever()

    try:
        asyncio.run(_runner())
    except ValueError as e:
        logging.getLogger(__name__).error(str(e))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
