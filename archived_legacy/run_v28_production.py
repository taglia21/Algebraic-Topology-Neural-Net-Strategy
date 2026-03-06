#!/usr/bin/env python3
"""
V28 Production Trading System - Unified Runner
================================================
Runs BOTH the equity engine and options engine concurrently via asyncio.
Entry point for the systemd service (deploy/v28_trading_bot.service).

All trading decisions are gated through:
  - RiskGuardian (bracket stops, regime sizing, drawdown circuit breakers)
  - StrategyEngine (BB+RSI mean reversion, stat-arb pairs, momentum)
  - Anti-churn (15% daily turnover cap, 6-bar minimum hold)
  - Universe filter (BANNED_SYMBOLS, freefall, death-cross)
  - Correlation check (max 0.7 pairwise correlation)
  - Volume confirmation (1.5x 20-period average)

Usage:
    python run_v28_production.py --mode=live
    python run_v28_production.py --mode=paper
"""

import argparse
import asyncio
import logging
import logging.handlers
import os
import signal
import sys
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any

import numpy as np

# ===========================================================================
# OPTIONS ENGINE — Re-enabled after Phases 1-3 hardening (2026-02-21)
# Guards: Alpaca-sourced position limits, idempotent orders, fixed fractional
# sizing, reduced universe (8 symbols), earnings gate, IV rank enforcement,
# bid-ask filter, DTE exit, portfolio heat cap, Greeks monitor, VIX overlay,
# underlying concentration limit, error resilience w/ 15-min cooldown.
# ===========================================================================
OPTIONS_ENGINE_ENABLED = True

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Core improved modules (Phases 1-4 of the trading fix)
# ---------------------------------------------------------------------------

from risk_guardian import RiskGuardian
from strategy_engine import StrategyEngine, EngineConfig, SignalDirection, StrategyType
from config.universe import BANNED_SYMBOLS, MAX_BETA

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

try:
    from pair_finder import PairFinder, PairFinderConfig
    HAS_PAIR_FINDER = True
except ImportError:
    HAS_PAIR_FINDER = False

try:
    from src.factor_monitor import FactorMonitor
    HAS_FACTOR_MONITOR = True
except ImportError:
    HAS_FACTOR_MONITOR = False

try:
    from portfolio_allocator import PortfolioAllocator
    HAS_PORTFOLIO_ALLOCATOR = True
except ImportError:
    HAS_PORTFOLIO_ALLOCATOR = False

# Prometheus metrics
try:
    from src.metrics import (
        MetricsServer,
        update_portfolio_metrics,
        record_order,
        record_signal,
        record_filter_block,
        CYCLE_DURATION,
    )
    HAS_METRICS = True
except ImportError:
    HAS_METRICS = False

# ---------------------------------------------------------------------------
# Phase B: Signal quality + ML + execution enhancements
# ---------------------------------------------------------------------------

try:
    from src.signal_aggregator import SignalAggregator
    HAS_SIGNAL_AGGREGATOR = True
except ImportError:
    HAS_SIGNAL_AGGREGATOR = False

try:
    from src.nn_predictor import NeuralNetPredictor
    HAS_NN_PREDICTOR = True
except ImportError:
    HAS_NN_PREDICTOR = False

try:
    from src.smart_execution import SmartExecutor, SmartExecConfig
    HAS_SMART_EXEC = True
except ImportError:
    HAS_SMART_EXEC = False

try:
    from src.signal_filters import SignalFilter
    HAS_SIGNAL_FILTER = True
except ImportError:
    HAS_SIGNAL_FILTER = False

try:
    from src.tda_features import TDAFeatureGenerator
    HAS_TDA = True
except ImportError:
    HAS_TDA = False

# ---------------------------------------------------------------------------
# Phase C: Kalman spread tracking, transaction costs, retraining
# ---------------------------------------------------------------------------

try:
    from pair_finder import KalmanSpreadTracker
    HAS_KALMAN_TRACKER = True
except ImportError:
    HAS_KALMAN_TRACKER = False

try:
    from src.risk.transaction_costs import TransactionCostModel
    HAS_TCA = True
except ImportError:
    HAS_TCA = False

try:
    from src.ml.retraining_scheduler import RetrainingScheduler, RetrainingConfig
    HAS_RETRAIN = True
except ImportError:
    HAS_RETRAIN = False

try:
    import pandas as _pd
except ImportError:
    _pd = None

try:
    import tensorflow as _tf
except ImportError:
    _tf = None

try:
    from src.metrics.daily_performance import DailyPerformanceLogger
    HAS_DAILY_PERF = True
except ImportError:
    HAS_DAILY_PERF = False

# ---------------------------------------------------------------------------
# Phase D: IBKR broker, ML self-training, alpha signals, Sharpe optimizer, health monitor
# ---------------------------------------------------------------------------

try:
    from src.brokers.ibkr_client import IBKRBrokerClient
    HAS_IBKR = True
except ImportError:
    HAS_IBKR = False

try:
    from src.ml.online_learner import OnlineLearner, OnlineLearnerConfig, TradeOutcome
    HAS_ONLINE_LEARNER = True
except ImportError:
    HAS_ONLINE_LEARNER = False

try:
    from src.options.signal_engine import SignalEngine as AlphaSignalEngine
    HAS_SIGNAL_ENGINE = True
except ImportError:
    HAS_SIGNAL_ENGINE = False

try:
    from src.optimization.sharpe_optimizer import SharpeOptimizer, SharpeOptimizerConfig
    HAS_SHARPE_OPT = True
except ImportError:
    HAS_SHARPE_OPT = False

try:
    from src.monitoring.health_monitor import HealthMonitor, HealthMonitorConfig
    HAS_HEALTH_MONITOR = True
except ImportError:
    HAS_HEALTH_MONITOR = False

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

# Phase 4: Rotating file logger shared with options engine
_rotating_handler = logging.handlers.RotatingFileHandler(
    str(LOG_DIR / "options_engine.log"), maxBytes=5 * 1024 * 1024, backupCount=3,
)
_rotating_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s")
)
logging.getLogger().addHandler(_rotating_handler)

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
    """
    Async equity engine — delegates ALL risk/sizing to RiskGuardian,
    signals to EnhancedTradingEngine + StrategyEngine, and gates every
    order through anti-churn, universe filter, correlation, and volume checks.
    """

    # ── Position monitoring thresholds (fallback — RiskGuardian is primary) ──
    EQUITY_STOP_LOSS_PCT = -0.05
    EQUITY_TAKE_PROFIT_PCT = 0.10
    TRAILING_STOP_ACTIVATE_PCT = 0.04
    TRAILING_STOP_TRAIL_PCT = 0.40

    # ── Anti-churn constants ──
    MAX_DAILY_TURNOVER_PCT = 0.15   # 15% of equity per day
    MIN_HOLD_BARS = 6               # minimum scan cycles before soft exit

    def __init__(self, mode: str, broker: str = "alpaca"):
        self.mode = mode
        self.broker_type = broker      # 'alpaca' or 'ibkr'
        self.engine = None             # EnhancedTradingEngine (signal source)
        self.client = None             # AlpacaClient (legacy)
        self.ibkr_client = None        # IBKRBrokerClient (new)
        self.risk_guardian = None       # RiskGuardian (safety layer)
        self.strategy_engine = None    # StrategyEngine (improved MR/pairs/momentum)
        self.pair_finder = None        # PairFinder (stat-arb)
        self.factor_monitor = None     # FactorMonitor (factor exposure)

        # Phase B: Signal quality + ML + execution
        self.signal_aggregator = None
        self.nn_predictor = None
        self.smart_executor = None
        self.signal_filter = None
        self.tda_engine = None
        self._nn_trained = False

        # Phase C: Kalman + TCA + retraining
        self.tca_model = None
        self.retrain_scheduler = None

        # Phase D: IBKR + ML self-training + alpha signals + Sharpe optimizer
        self.online_learner = None
        self.alpha_signal_engine = None
        self.sharpe_optimizer = None
        self.health_monitor = None

        # Phase 6: Daily performance logger
        self.daily_perf = None

        self.logger = logging.getLogger("equity_engine")
        self._high_water_marks: dict[str, float] = {}

        # Anti-churn state
        self._daily_turnover_used: float = 0.0
        self._position_entry_bar: dict[str, int] = {}   # symbol → scan bar
        self._bar_count: int = 0
        self._last_trading_date: Optional[date] = None

        # Regime state
        self._regime_size_scale: float = 0.70
        self._max_positions_regime: int = 6

    async def initialize(self):
        # 1. EnhancedTradingEngine for signal analysis
        from src.enhanced_trading_engine import EnhancedTradingEngine, EngineConfig as ETEConfig
        config = ETEConfig()
        self.engine = EnhancedTradingEngine(config)

        # 2. Broker client — IBKR (preferred) or Alpaca (legacy)
        init_equity = 100_000.0
        if self.broker_type == "ibkr" and HAS_IBKR:
            ibkr_host = os.getenv("IBKR_HOST", "127.0.0.1")
            ibkr_port = int(os.getenv("IBKR_PORT", "4002"))
            ibkr_account = os.getenv("IBKR_ACCOUNT", "U22452226")
            self.ibkr_client = IBKRBrokerClient(
                host=ibkr_host, port=ibkr_port, account=ibkr_account,
                paper=(self.mode == "paper"),
            )
            try:
                self.ibkr_client.connect()
                acct = self.ibkr_client.get_account()
                init_equity = acct.portfolio_value
                self.logger.info(f"IBKR connected — equity ${init_equity:,.0f}")
            except Exception as e:
                self.logger.warning(f"IBKR connect failed: {e} — will retry")
        else:
            from src.trading.alpaca_client import AlpacaClient
            self.client = AlpacaClient()
            try:
                acct = self.client.get_account()
                init_equity = acct.equity
            except Exception as e:
                self.logger.warning(f"Alpaca account fetch failed: {e}")

        # 3. RiskGuardian — bracket stops, drawdown circuit breaker, regime sizing
        self.risk_guardian = RiskGuardian(
            initial_equity=init_equity,
            max_drawdown_pct=0.15,
            daily_loss_limit_pct=0.03,
            hard_stop_pct=0.08,
            consecutive_loss_limit=3,
            max_positions=10,
            max_sector_positions=3,
            max_correlation=0.70,
        )

        # 4. StrategyEngine — improved MR/pairs/momentum with wider params
        self.strategy_engine = StrategyEngine(EngineConfig())

        # 5. PairFinder (optional)
        if HAS_PAIR_FINDER:
            try:
                self.pair_finder = PairFinder(PairFinderConfig())
                self.logger.info("PairFinder loaded for stat-arb signals")
            except Exception as e:
                self.logger.warning(f"PairFinder init failed: {e}")

        # 6. FactorMonitor (optional)
        if HAS_FACTOR_MONITOR:
            try:
                self.factor_monitor = FactorMonitor()
                self.logger.info("FactorMonitor loaded for factor exposure tracking")
            except Exception as e:
                self.logger.warning(f"FactorMonitor init failed: {e}")

        # 7. Signal Aggregator (Phase B) — ensemble signal quality
        if HAS_SIGNAL_AGGREGATOR:
            try:
                self.signal_aggregator = SignalAggregator(min_confidence=0.55, min_models=2)
                self.signal_aggregator.initialize()
                self.logger.info("SignalAggregator loaded (ensemble signal quality)")
            except Exception as e:
                self.logger.warning(f"SignalAggregator init failed: {e}")

        # 8. NN Predictor (Phase B) — LSTM ML confidence filter
        if HAS_NN_PREDICTOR:
            try:
                self.nn_predictor = NeuralNetPredictor(sequence_length=20, n_features=6)
                self.nn_predictor.compile_model()
                # Try to load pre-trained weights (bootstrap path first, then results/)
                import glob
                _bootstrap_path = PROJECT_ROOT / "models" / "nn_predictor.weights.h5"
                weight_files = glob.glob(str(PROJECT_ROOT / "results" / "*weights*.h5"))
                _weight_path = None
                if _bootstrap_path.exists():
                    _weight_path = str(_bootstrap_path)
                elif weight_files:
                    _weight_path = weight_files[0]

                if _weight_path:
                    try:
                        self.nn_predictor.load_checkpoint(_weight_path)
                        self._nn_trained = True
                        self.logger.info(f"NeuralNetPredictor loaded with weights: {_weight_path}")
                    except Exception as e:
                        self.logger.warning(f"NeuralNetPredictor weight load failed: {e}")
                        self.logger.info("NeuralNetPredictor loaded (untrained — skipping ML gate)")
                else:
                    self.logger.info("NeuralNetPredictor loaded (no weights found — skipping ML gate)")
            except Exception as e:
                self.logger.warning(f"NeuralNetPredictor init failed: {e}")

        # 9. Smart Execution (Phase B) — TWAP/VWAP order splitting
        if HAS_SMART_EXEC:
            try:
                def _sync_submit(symbol, qty, side, limit_price):
                    """Sync bridge: SmartExecutor → AlpacaClient."""
                    from src.trading.alpaca_client import OrderSide, OrderType
                    _side = OrderSide.BUY if side == "buy" else OrderSide.SELL
                    return self.client.submit_order(
                        symbol=symbol, qty=qty, side=_side,
                        order_type=OrderType.LIMIT, limit_price=limit_price,
                    )
                self.smart_executor = SmartExecutor(
                    submit_fn=_sync_submit,
                    config=SmartExecConfig(default_duration_sec=120.0, max_slices=5),
                )
                self.logger.info("SmartExecutor loaded (TWAP/VWAP order splitting)")
            except Exception as e:
                self.logger.warning(f"SmartExecutor init failed: {e}")

        # 10. Signal Filter (Phase B) — RSI + volatility pre-filter
        if HAS_SIGNAL_FILTER:
            try:
                self.signal_filter = SignalFilter(rsi_period=14, vol_threshold=0.30)
                self.logger.info("SignalFilter loaded (RSI + volatility filter)")
            except Exception as e:
                self.logger.warning(f"SignalFilter init failed: {e}")

        # 11. TDA Feature Generator (Phase B) — Topological Data Analysis
        if HAS_TDA:
            try:
                self.tda_engine = TDAFeatureGenerator(window=30, embedding_dim=3, feature_mode='v1.3')
                self.logger.info("TDAFeatureGenerator loaded (topological features)")
            except Exception as e:
                self.logger.warning(f"TDAFeatureGenerator init failed: {e}")

        # 12. Transaction Cost Model (Phase C)
        if HAS_TCA:
            try:
                self.tca_model = TransactionCostModel()
                self.logger.info("TransactionCostModel loaded (pre-trade cost gate)")
            except Exception as e:
                self.logger.warning(f"TransactionCostModel init failed: {e}")

        # 13. Retraining Scheduler (Phase C)
        if HAS_RETRAIN:
            try:
                self.retrain_scheduler = RetrainingScheduler(
                    config=RetrainingConfig(
                        accuracy_floor=0.52,
                        sharpe_floor=0.3,
                        check_interval_hours=24,
                        cooldown_hours=72,
                    ),
                )
                self.logger.info("RetrainingScheduler loaded (performance watchdog)")
            except Exception as e:
                self.logger.warning(f"RetrainingScheduler init failed: {e}")

        # 14. Daily Performance Logger
        if HAS_DAILY_PERF:
            try:
                self.daily_perf = DailyPerformanceLogger(
                    log_dir=str(LOG_DIR),
                    initial_equity=float(init_equity),
                )
                self.logger.info("DailyPerformanceLogger loaded")
            except Exception as e:
                self.logger.warning(f"DailyPerformanceLogger init failed: {e}")

        # 15. Online Learner (Phase D — self-training ML)
        if HAS_ONLINE_LEARNER:
            try:
                self.online_learner = OnlineLearner()
                self.online_learner.load_latest_checkpoint()
                self.logger.info("OnlineLearner loaded (self-training ML)")
            except Exception as e:
                self.logger.warning(f"OnlineLearner init failed: {e}")

        # 16. Alpha Signal Engine (Phase D)
        if HAS_SIGNAL_ENGINE:
            try:
                self.alpha_signal_engine = AlphaSignalEngine()
                self.logger.info("AlphaSignalEngine loaded (VIX/GEX/IV/PC signals)")
            except Exception as e:
                self.logger.warning(f"AlphaSignalEngine init failed: {e}")

        # 17. Sharpe Optimizer (Phase D)
        if HAS_SHARPE_OPT:
            try:
                self.sharpe_optimizer = SharpeOptimizer()
                self.sharpe_optimizer.update_daily_equity(init_equity)
                self.logger.info("SharpeOptimizer loaded (dynamic sizing + DD breaker)")
            except Exception as e:
                self.logger.warning(f"SharpeOptimizer init failed: {e}")

        # 18. Health Monitor (Phase D)
        if HAS_HEALTH_MONITOR:
            try:
                hm_config = HealthMonitorConfig(
                    ibkr_host=os.getenv("IBKR_HOST", "127.0.0.1"),
                    ibkr_port=int(os.getenv("IBKR_PORT", "4002")),
                    discord_webhook=os.getenv("DISCORD_WEBHOOK_MARCUS", ""),
                )
                self.health_monitor = HealthMonitor(config=hm_config)
                if self.ibkr_client:
                    self.health_monitor.set_ibkr_client(self.ibkr_client)
                self.health_monitor.start()
                self.logger.info("HealthMonitor started (60s interval)")
            except Exception as e:
                self.logger.warning(f"HealthMonitor init failed: {e}")

        _phase_b = []
        if self.signal_aggregator: _phase_b.append("SignalAgg")
        if self.nn_predictor: _phase_b.append("NNPredictor")
        if self.smart_executor: _phase_b.append("SmartExec")
        if self.signal_filter: _phase_b.append("SigFilter")
        if self.tda_engine: _phase_b.append("TDA")
        if self.tca_model: _phase_b.append("TCA")
        if self.retrain_scheduler: _phase_b.append("Retrain")
        if self.daily_perf: _phase_b.append("DailyPerf")

        _phase_d = []
        if self.ibkr_client: _phase_d.append("IBKR")
        if self.online_learner: _phase_d.append("OnlineLearner")
        if self.alpha_signal_engine: _phase_d.append("AlphaSignals")
        if self.sharpe_optimizer: _phase_d.append("SharpeOpt")
        if self.health_monitor: _phase_d.append("HealthMon")

        self.logger.info(
            f"Equity engine initialized (mode={self.mode}, broker={self.broker_type}) with "
            f"RiskGuardian + StrategyEngine + anti-churn + universe filter"
            + (f" + Phase B: {', '.join(_phase_b)}" if _phase_b else "")
            + (f" + Phase D: {', '.join(_phase_d)}" if _phase_d else "")
        )

    # ── Anti-churn helpers ──────────────────────────────────────────

    def _turnover_allows_trade(self, proposed_cost: float, equity: float) -> bool:
        """Return True if adding proposed_cost stays within daily turnover cap."""
        if equity <= 0:
            return False
        return (self._daily_turnover_used + proposed_cost) / equity <= self.MAX_DAILY_TURNOVER_PCT

    def _record_fill(self, symbol: str, cost: float, equity: float):
        """Record a fill for turnover tracking."""
        self._daily_turnover_used += cost
        self._position_entry_bar[symbol] = self._bar_count
        pct = self._daily_turnover_used / equity * 100 if equity > 0 else 0
        self.logger.info(f"Turnover: +${cost:,.0f} → {pct:.1f}% of equity used today")

    def _min_hold_allows_exit(self, symbol: str) -> bool:
        """Return True if position has been held long enough for soft exit."""
        entry_bar = self._position_entry_bar.get(symbol)
        if entry_bar is None:
            return True  # unknown entry → allow exit
        return (self._bar_count - entry_bar) >= self.MIN_HOLD_BARS

    # ── Universe / quality filters ──────────────────────────────────

    def _passes_universe_filter(self, symbol: str, prices: Optional[np.ndarray] = None) -> bool:
        """Check banned list, freefall, and death-cross filters."""
        if symbol in BANNED_SYMBOLS:
            self.logger.debug(f"Blocked {symbol}: in BANNED_SYMBOLS")
            return False

        if prices is not None and len(prices) >= 6:
            five_bar_ret = (prices[-1] / prices[-6]) - 1.0
            if five_bar_ret < -0.08:
                self.logger.info(f"Blocked {symbol}: freefall {five_bar_ret:+.1%} in 5 bars")
                return False

        if prices is not None and len(prices) >= 200:
            sma50 = float(np.mean(prices[-50:]))
            sma200 = float(np.mean(prices[-200:]))
            if sma50 < sma200:
                self.logger.debug(f"Blocked {symbol}: death cross SMA50 < SMA200")
                return False

        return True

    def _passes_volume_check(self, volumes: Optional[np.ndarray]) -> bool:
        """Liquidity gate: avg daily volume > 500K and today's volume > 30% avg.

        Original 1.5x filter blocked ALL symbols because we check mid-day
        (partial volume) against full-day averages.  Replaced with a
        liquidity floor + dead-day filter.
        """
        if volumes is None or len(volumes) < 21:
            return True  # insufficient data → allow (don't block on missing data)
        avg_20 = float(np.mean(volumes[-21:-1]))
        if avg_20 < 500_000:
            return False  # illiquid stock
        if avg_20 <= 0:
            return True
        # Block abnormally dead days (< 30% of average, accounts for mid-day check)
        return float(volumes[-1]) >= 0.3 * avg_20

    def _passes_correlation_check(self, symbol: str) -> bool:
        """Check correlation vs. existing holdings using RiskGuardian."""
        if self.risk_guardian is None:
            return True
        try:
            existing = [s for s in self._position_entry_bar.keys()]
            if not existing:
                return True
            allowed, max_corr, reason = self.risk_guardian.correlation_checker.check_correlation(
                symbol, existing
            )
            if not allowed:
                self.logger.info(f"Correlation block: {symbol} — {reason}")
            return allowed
        except Exception as e:
            self.logger.debug(f"Correlation check error for {symbol}: {e}")
            return True  # allow on error

    # ── Regime detection ────────────────────────────────────────────

    def _update_regime(self):
        """Set regime-gated position sizing from RiskGuardian/regime filter."""
        regime_label = "neutral"
        if HAS_REGIME_FILTER:
            try:
                if is_bullish_regime():
                    regime_label = "bull"
                else:
                    regime_label = "bear"
            except Exception as e:
                self.logger.warning(f"Regime detection failed: {e}")

        if regime_label == "bull":
            self._regime_size_scale = 1.0
            self._max_positions_regime = 10
        elif regime_label == "neutral":
            self._regime_size_scale = 0.70
            self._max_positions_regime = 6
        else:
            self._regime_size_scale = 0.40
            self._max_positions_regime = 4

        self.logger.info(
            f"Regime: {regime_label} → scale={self._regime_size_scale:.0%}, "
            f"max_pos={self._max_positions_regime}"
        )

    async def run_cycle(self, symbols: list[str]):
        """Run a single equity trading cycle with ALL protections."""
        if self.engine is None:
            return

        import time as _time
        _cycle_start = _time.monotonic()

        self._bar_count += 1

        # Reset daily turnover at start of new trading day
        today = date.today()
        if self._last_trading_date != today:
            self._daily_turnover_used = 0.0
            self._last_trading_date = today

        # Circuit breaker gate
        if HAS_TRADING_GATE:
            allowed, reason = check_trading_allowed()
            if not allowed:
                self.logger.warning(f"⚠️ CIRCUIT BREAKER: {reason}")
                return

        # Regime detection → sets _regime_size_scale and _max_positions_regime
        self._update_regime()
        skip_buys = (self._regime_size_scale <= 0.40)

        # Get account state
        equity = 100_000.0
        pos_values: dict[str, float] = {}
        existing_symbols: list[str] = []

        # IBKR path
        if self.ibkr_client:
            try:
                acct = self.ibkr_client.get_account()
                equity = acct.portfolio_value
                for p in self.ibkr_client.get_positions():
                    if len(p.symbol) <= 6:
                        pos_values[p.symbol] = abs(p.market_value)
                        existing_symbols.append(p.symbol)
            except Exception as e:
                self.logger.warning(f"IBKR position fetch failed: {e}")
        # Alpaca fallback
        elif self.client:
            try:
                acct = self.client.get_account()
                equity = acct.equity
                for p in self.client.get_positions():
                    if len(p.symbol) <= 6 and not any(ch.isdigit() for ch in p.symbol[:4]):
                        pos_values[p.symbol] = abs(p.market_value)
                        existing_symbols.append(p.symbol)
            except Exception as e:
                self.logger.warning(f"Position fetch failed, using stale data: {e}")

        # Phase D: Sharpe optimizer — update daily equity + drawdown check
        if self.sharpe_optimizer:
            self.sharpe_optimizer.update_daily_equity(equity)
            dd_ok, dd_msg = self.sharpe_optimizer.check_drawdown(equity)
            if not dd_ok:
                self.logger.warning(f"🔴 SHARPE DD HALT: {dd_msg}")
                return

        # Phase D: Online learner circuit breaker
        if self.online_learner:
            allowed, cb_reason = self.online_learner.is_trading_allowed()
            if not allowed:
                self.logger.warning(f"🔴 ML CIRCUIT BREAKER: {cb_reason}")
                return

        # Update RiskGuardian with current equity
        if self.risk_guardian:
            guardian_state = self.risk_guardian.update(equity)
            if guardian_state.should_liquidate:
                self.logger.error(f"🚨 EMERGENCY LIQUIDATION: {guardian_state.halt_reasons}")
                return
            if guardian_state.should_halt:
                self.logger.warning(f"🔴 Guardian HALT: {guardian_state.halt_reasons}")
                await self._monitor_equity_positions(equity)
                return

        # Update FactorMonitor — check factor exposures
        factor_tilt_warning = ""
        if self.factor_monitor and existing_symbols:
            try:
                weights = {}
                total_val = sum(pos_values.values()) or 1.0
                for sym, val in pos_values.items():
                    weights[sym] = val / total_val
                exposure = self.factor_monitor.get_factor_exposures(existing_symbols, weights)
                largest_factor, largest_beta = self.factor_monitor.get_largest_tilt()
                if abs(largest_beta) > 0.30:
                    factor_tilt_warning = f"{largest_factor}={largest_beta:+.2f}"
                    self.logger.info(f"⚠️ Factor tilt: {factor_tilt_warning}")
            except Exception as e:
                self.logger.debug(f"FactorMonitor error: {e}")

        # Monitor existing positions (SL/TP/trailing)
        if self.client:
            await self._monitor_equity_positions(equity)

        # Skip new entries if bearish or at position limit
        n_positions = len(existing_symbols)
        if skip_buys:
            self.logger.info("📉 Bear regime — skipping new entries")
            return
        if n_positions >= self._max_positions_regime:
            self.logger.info(
                f"At position cap ({n_positions}/{self._max_positions_regime})"
            )
            return

        # Analyze each symbol
        _filter_counts = {"universe": 0, "volume": 0, "correlation": 0,
                          "no_signal": 0, "not_buy": 0, "scanned": 0}
        for symbol in symbols:
            if n_positions >= self._max_positions_regime:
                break
            if symbol in existing_symbols:
                continue
            _filter_counts["scanned"] += 1

            try:
                # Universe filter (banned, freefall, death-cross)
                price_data = self._fetch_price_array(symbol)
                if not self._passes_universe_filter(symbol, price_data):
                    _filter_counts["universe"] += 1
                    if HAS_METRICS:
                        record_filter_block("universe")
                    continue

                # Volume confirmation
                vol_data = self._fetch_volume_array(symbol)
                if not self._passes_volume_check(vol_data):
                    _filter_counts["volume"] += 1
                    self.logger.debug(f"Skipping {symbol}: low volume")
                    if HAS_METRICS:
                        record_filter_block("volume")
                    continue

                # Correlation check vs existing holdings
                if not self._passes_correlation_check(symbol):
                    _filter_counts["correlation"] += 1
                    self.logger.info(f"Skipping {symbol}: too correlated with existing holdings")
                    if HAS_METRICS:
                        record_filter_block("correlation")
                    continue

                # Get signal from EnhancedTradingEngine
                decision = self.engine.analyze(symbol)
                if not decision or not decision.is_tradeable:
                    _filter_counts["no_signal"] += 1
                    continue

                is_buy = decision.signal.name in ("STRONG_BUY", "BUY")
                if not is_buy:
                    _filter_counts["not_buy"] += 1
                    continue  # only longs for now

                # ── Phase B: Signal quality enhancements ──

                # B1: TDA topological turbulence check
                if self.tda_engine and price_data is not None and len(price_data) >= 30:
                    try:
                        tda_feats = self.tda_engine.compute_persistence_features(price_data[-60:])
                        turbulence = np.sqrt(
                            tda_feats.get('persistence_l0', 0) ** 2
                            + tda_feats.get('persistence_l1', 0) ** 2
                        )
                        if turbulence > 2.0:
                            decision.confidence *= 0.80
                            self.logger.debug(f"TDA turbulence {symbol}: {turbulence:.2f} → conf reduced")
                    except Exception as e:
                        self.logger.debug(f"TDA error for {symbol}: {e}")

                # B2: Signal Aggregator — ensemble confirmation
                if self.signal_aggregator:
                    try:
                        agg = self.signal_aggregator.aggregate(symbol)
                        if agg and not agg.is_actionable:
                            self.logger.debug(
                                f"SignalAggregator rejected {symbol} "
                                f"(sig={agg.signal:.2f}, conf={agg.confidence:.2f})"
                            )
                            _filter_counts["no_signal"] += 1
                            continue
                        if agg and agg.confidence > 0:
                            decision.confidence = min(
                                1.0, (decision.confidence + agg.confidence) / 2
                            )
                    except Exception as e:
                        self.logger.debug(f"SignalAggregator error {symbol}: {e}")

                # B3: Signal Filter — RSI + volatility gate
                if self.signal_filter and price_data is not None and len(price_data) >= 21:
                    try:
                        _price_df = _pd.DataFrame({"close": price_data})
                        filt = self.signal_filter.filter_signal("buy", _price_df)
                        if filt.get("filtered", False):
                            self.logger.info(
                                f"SignalFilter blocked {symbol}: "
                                f"{filt.get('filter_reason', '?')}"
                            )
                            _filter_counts["no_signal"] += 1
                            continue
                    except Exception as e:
                        self.logger.debug(f"SignalFilter error {symbol}: {e}")

                # B4: NN Predictor — ML confidence gate (only with trained weights)
                if self.nn_predictor and self._nn_trained and price_data is not None and len(price_data) >= 21:
                    try:
                        _rets = np.diff(price_data[-21:]) / np.maximum(price_data[-21:-1], 1e-8)
                        _seq = _rets.reshape(1, 20, 1)
                        _seq = np.pad(_seq, ((0, 0), (0, 0), (0, 5)))  # pad to 6 features
                        _pred = float(self.nn_predictor(
                            _tf.constant(_seq, dtype=_tf.float32)
                        ).numpy().flatten()[0])
                        if _pred < 0.42:
                            self.logger.info(
                                f"NNPredictor LOW conf {symbol}: {_pred:.2f} — skip"
                            )
                            _filter_counts["no_signal"] += 1
                            continue
                        decision.confidence = min(
                            1.0, decision.confidence * 0.65 + _pred * 0.35
                        )
                    except Exception as e:
                        self.logger.debug(f"NNPredictor error {symbol}: {e}")

                # Sector cap check
                cost = decision.recommended_quantity * decision.entry_price
                if HAS_SECTOR_CAPS:
                    allowed, cap_reason = sector_allows_trade(symbol, cost, pos_values, equity)
                    if not allowed:
                        self.logger.info(f"🚫 Sector cap: {cap_reason}")
                        continue

                # RiskGuardian position sizing with regime scale
                if self.risk_guardian:
                    atr_pct = decision.metadata.get('atr', decision.entry_price * 0.02) / max(decision.entry_price, 1)
                    safe_size = self.risk_guardian.compute_safe_position_size(
                        base_pct=decision.recommended_position_value / max(equity, 1),
                        atr_pct=atr_pct,
                        confidence=decision.confidence,
                        regime_scale=self._regime_size_scale,
                    )
                    cost = equity * safe_size
                    decision.recommended_quantity = max(1, int(cost / decision.entry_price))

                proposed_cost = decision.recommended_quantity * decision.entry_price

                # Anti-churn: turnover gate
                if not self._turnover_allows_trade(proposed_cost, equity):
                    self.logger.info(f"Turnover cap reached — skipping {symbol}")
                    continue

                # Max single position: 8% of portfolio
                if proposed_cost > equity * 0.08:
                    decision.recommended_quantity = max(1, int(equity * 0.08 / decision.entry_price))
                    proposed_cost = decision.recommended_quantity * decision.entry_price

                # Phase C: Transaction Cost gate
                if self.tca_model:
                    try:
                        _vol = None
                        if price_data is not None and len(price_data) >= 22:
                            _rets = np.diff(price_data[-22:]) / np.maximum(price_data[-22:-1], 1e-8)
                            _vol = float(np.std(_rets))
                        _adv = None
                        if vol_data is not None and len(vol_data) >= 20:
                            _adv = float(np.mean(vol_data[-20:]))
                        tca_est = self.tca_model.estimate_cost(
                            symbol=symbol,
                            qty=decision.recommended_quantity,
                            price=decision.entry_price,
                            side="buy",
                            adv=_adv,
                            volatility=_vol,
                        )
                        allowed, tca_reason = self.tca_model.should_trade(tca_est)
                        if not allowed:
                            self.logger.info(f"TCA blocked {symbol}: {tca_reason}")
                            continue
                    except Exception as e:
                        self.logger.debug(f"TCA error {symbol}: {e}")

                self.logger.info(
                    f"EQUITY SIGNAL: {symbol} → {decision.signal.name} "
                    f"(conf={decision.confidence:.2f}, combined={decision.combined_score:.2f}, "
                    f"qty={decision.recommended_quantity}, cost=${proposed_cost:,.0f})"
                )
                if HAS_METRICS:
                    record_signal(decision.signal.name)

                await self._execute_equity_trade(decision, self._regime_size_scale)

                # Record fill for anti-churn tracking
                self._record_fill(symbol, proposed_cost, equity)
                n_positions += 1

            except Exception as exc:
                self.logger.error(f"Equity cycle error for {symbol}: {exc}", exc_info=True)

        # Log cycle summary
        self.logger.info(
            f"Scan: {_filter_counts['scanned']} symbols | "
            f"universe={_filter_counts['universe']} vol={_filter_counts['volume']} "
            f"corr={_filter_counts['correlation']} no_sig={_filter_counts['no_signal']} "
            f"not_buy={_filter_counts['not_buy']} | positions={n_positions}/{self._max_positions_regime}"
        )

        # ── End-of-cycle metrics update ──
        if HAS_METRICS:
            _cycle_elapsed = _time.monotonic() - _cycle_start
            CYCLE_DURATION.observe(_cycle_elapsed)
            turnover_pct = (self._daily_turnover_used / equity * 100) if equity > 0 else 0
            dd_pct = 0.0
            if self.risk_guardian:
                try:
                    dd_pct = self.risk_guardian.get_current_drawdown() * 100
                except Exception as e:
                    self.logger.debug(f"Drawdown calc error: {e}")
            update_portfolio_metrics(
                equity=equity,
                n_positions=n_positions,
                turnover_pct=turnover_pct,
                regime_scale=self._regime_size_scale,
                max_dd_pct=dd_pct,
            )

        # ── Phase C: Retraining check (once per cycle) ──
        if self.retrain_scheduler:
            try:
                result = self.retrain_scheduler.check_and_retrain()
                if result.get("triggered"):
                    self.logger.info(
                        f"🔄 Retrain result: success={result.get('success')}, "
                        f"reason={result.get('reason')}"
                    )
            except Exception as e:
                self.logger.debug(f"RetrainingScheduler check error: {e}")

        # ── Daily performance logging (once per cycle, deduped per day) ──
        if self.daily_perf:
            try:
                _turnover = (self._daily_turnover_used / equity * 100) if equity > 0 else 0
                self.daily_perf.log_daily(
                    equity=equity,
                    daily_pnl=0.0,  # computed inside logger from equity delta
                    n_positions=n_positions,
                    n_trades=_filter_counts.get("scanned", 0) - _filter_counts.get("no_signal", 0),
                    turnover_pct=_turnover,
                )
            except Exception as e:
                self.logger.debug(f"DailyPerformanceLogger error: {e}")

    def _fetch_price_array(self, symbol: str) -> Optional[np.ndarray]:
        """Fetch recent close prices as numpy array."""
        try:
            import yfinance as yf
            data = yf.download(symbol, period="1y", interval="1d", progress=False)
            if data is not None and len(data) >= 50:
                return data["Close"].values.flatten().astype(float)
        except Exception as e:
            self.logger.debug(f"Close array fetch failed for symbol: {e}")
        return None

    def _fetch_volume_array(self, symbol: str) -> Optional[np.ndarray]:
        """Fetch recent volume as numpy array."""
        try:
            import yfinance as yf
            data = yf.download(symbol, period="2mo", interval="1d", progress=False)
            if data is not None and len(data) >= 21:
                return data["Volume"].values.flatten().astype(float)
        except Exception as e:
            self.logger.debug(f"Volume array fetch failed: {e}")
        return None

    async def _monitor_equity_positions(self, equity: float):
        """Monitor existing equity positions — SL, TP, trailing stop, and min-hold."""
        try:
            positions = self.client.get_positions()
        except Exception as exc:
            self.logger.error(f"Failed to fetch positions for monitoring: {exc}")
            return

        active_symbols = set()
        for pos in positions:
            # Skip option positions (handled by OptionsEngine)
            if len(pos.symbol) > 6 or any(ch.isdigit() for ch in pos.symbol[:4]):
                continue

            active_symbols.add(pos.symbol)
            try:
                unrealized_pnl = float(pos.unrealized_pl)
                cost_basis = abs(float(pos.cost_basis))
                if cost_basis <= 0:
                    cost_basis = abs(float(pos.qty) * float(pos.avg_entry_price))
                if cost_basis <= 0:
                    continue

                pnl_pct = unrealized_pnl / cost_basis

                # Update high-water mark for trailing stop
                prev_hwm = self._high_water_marks.get(pos.symbol, 0.0)
                if pnl_pct > prev_hwm:
                    self._high_water_marks[pos.symbol] = pnl_pct
                    prev_hwm = pnl_pct

                # 1. Hard stop loss — ALWAYS fires regardless of hold period
                if pnl_pct <= self.EQUITY_STOP_LOSS_PCT:
                    self.logger.warning(
                        f"🛑 EQUITY STOP-LOSS: {pos.symbol} "
                        f"P&L ${unrealized_pnl:+,.2f} ({pnl_pct:+.1%}) — closing"
                    )
                    self.client.close_position(pos.symbol)
                    self._position_entry_bar.pop(pos.symbol, None)
                    if self.risk_guardian:
                        self.risk_guardian.record_trade_result(pnl_pct > 0)
                        self.risk_guardian.record_trade_for_kelly(pnl_pct)

                # ── Soft exits below require min-hold check ──
                elif not self._min_hold_allows_exit(pos.symbol):
                    self.logger.debug(
                        f"  {pos.symbol}: P&L {pnl_pct:+.1%} but min-hold not met "
                        f"({self._bar_count - self._position_entry_bar.get(pos.symbol, 0)}"
                        f"/{self.MIN_HOLD_BARS} bars)"
                    )

                # 2. Trailing stop: reached +4% and gave back 40% of peak
                elif (prev_hwm >= self.TRAILING_STOP_ACTIVATE_PCT and
                      pnl_pct < prev_hwm * (1 - self.TRAILING_STOP_TRAIL_PCT)):
                    trail_floor = prev_hwm * (1 - self.TRAILING_STOP_TRAIL_PCT)
                    self.logger.info(
                        f"📉 TRAILING STOP: {pos.symbol} peak={prev_hwm:+.1%}, "
                        f"now={pnl_pct:+.1%}, floor={trail_floor:+.1%} — closing"
                    )
                    self.client.close_position(pos.symbol)
                    self._position_entry_bar.pop(pos.symbol, None)
                    if self.risk_guardian:
                        self.risk_guardian.record_trade_result(pnl_pct > 0)
                        self.risk_guardian.record_trade_for_kelly(pnl_pct)

                # 3. Hard take profit
                elif pnl_pct >= self.EQUITY_TAKE_PROFIT_PCT:
                    self.logger.info(
                        f"🎯 EQUITY TAKE-PROFIT: {pos.symbol} "
                        f"P&L ${unrealized_pnl:+,.2f} ({pnl_pct:+.1%}) — closing"
                    )
                    self.client.close_position(pos.symbol)
                    self._position_entry_bar.pop(pos.symbol, None)
                    if self.risk_guardian:
                        self.risk_guardian.record_trade_result(pnl_pct > 0)
                        self.risk_guardian.record_trade_for_kelly(pnl_pct)

                else:
                    self.logger.debug(
                        f"  Equity holding {pos.symbol}: {float(pos.qty):.0f} sh, "
                        f"P&L ${unrealized_pnl:+,.2f} ({pnl_pct:+.1%}), "
                        f"HWM={prev_hwm:+.1%}"
                    )
            except Exception as exc:
                self.logger.error(f"Error monitoring {pos.symbol}: {exc}")

        # Clean up high-water marks for closed positions
        for sym in list(self._high_water_marks):
            if sym not in active_symbols:
                del self._high_water_marks[sym]

    async def _execute_equity_trade(self, decision, regime_scale: float = 1.0):
        """Place equity order via Alpaca REST — LIMIT order with bracket stop/TP.
        
        Sizing is already done in run_cycle via RiskGuardian, so qty is final.
        All orders are LIMIT — MARKET orders are NEVER used.
        """
        if self.client is None:
            return
        try:
            from src.trading.alpaca_client import OrderSide, OrderType

            side = OrderSide.BUY if decision.signal.name in ("STRONG_BUY", "BUY") else OrderSide.SELL
            qty = max(1, int(decision.recommended_quantity))

            # Get current quote for limit price
            quote = self.client.get_latest_quote(decision.symbol)
            if side == OrderSide.BUY:
                limit_price = round(quote["ask"] * 1.001, 2)  # slightly above ask
            else:
                limit_price = round(quote["bid"] * 0.999, 2)  # slightly below bid

            if limit_price <= 0:
                self.logger.warning(f"Bad quote for {decision.symbol} — skipping")
                return

            # Place bracket order with stop-loss and take-profit for buys
            if side == OrderSide.BUY and decision.stop_loss and decision.take_profits:
                stop_price = round(decision.stop_loss, 2)
                tp_price = round(
                    decision.take_profits[0] if decision.take_profits else limit_price * 1.04,
                    2,
                )
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
                    f"✅ Bracket order: {decision.symbol} {qty}sh "
                    f"limit=${limit_price} SL=${stop_price} TP=${tp_price} "
                    f"→ {data.get('id', '?')}"
                )
                if HAS_METRICS:
                    record_order("buy", "submitted")
            else:
                # Simple limit order for sells or if no stop/TP available
                # Phase B: use SmartExecutor for TWAP splitting on larger orders
                if self.smart_executor and qty >= 50:
                    try:
                        plan = self.smart_executor.plan_execution(
                            symbol=decision.symbol,
                            qty=qty,
                            side="buy" if side == OrderSide.BUY else "sell",
                            ref_price=limit_price,
                            strategy="twap",
                        )
                        report = self.smart_executor.execute_all_slices(
                            plan, current_price=limit_price
                        )
                        self.logger.info(
                            f"✅ SmartExec TWAP: {decision.symbol} {qty}sh "
                            f"in {report.slices_filled} slices, "
                            f"slippage={report.slippage_bps:+.1f}bps"
                        )
                        if HAS_METRICS:
                            record_order(
                                side.value if hasattr(side, 'value') else str(side),
                                "submitted",
                            )
                    except Exception as se:
                        self.logger.warning(f"SmartExec failed, falling back: {se}")
                        result = self.client.submit_order(
                            symbol=decision.symbol,
                            qty=qty,
                            side=side,
                            order_type=OrderType.LIMIT,
                            limit_price=limit_price,
                        )
                        self.logger.info(f"✅ Limit order (fallback): {result}")
                        if HAS_METRICS:
                            record_order(side.value if hasattr(side, 'value') else str(side), "submitted")
                else:
                    result = self.client.submit_order(
                        symbol=decision.symbol,
                        qty=qty,
                        side=side,
                        order_type=OrderType.LIMIT,
                        limit_price=limit_price,
                    )
                    self.logger.info(f"✅ Limit order: {result}")
                    if HAS_METRICS:
                        record_order(side.value if hasattr(side, 'value') else str(side), "submitted")
        except Exception as exc:
            self.logger.error(f"Equity execution failed: {exc}", exc_info=True)


class OptionsEngine:
    """Async wrapper around the autonomous options engine.

    Attributes:
        mode: Trading mode ('paper' or 'live').
        engine: The underlying AutonomousTradingEngine instance.
    """

    def __init__(self, mode: str):
        self.mode = mode
        self.engine = None
        self.logger = logging.getLogger("options_engine")

    async def initialize(self):
        if not OPTIONS_ENGINE_ENABLED:
            self.logger.warning("Options engine disabled — skipping initialization")
            return
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
        if not OPTIONS_ENGINE_ENABLED:
            self.logger.warning(
                "⛔ Options engine DISABLED (OPTIONS_ENGINE_ENABLED=False) — "
                "emergency stop to prevent further losses. "
                "Re-enable only after risk controls are verified."
            )
            return
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
    s for s in [
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
    ] if s not in BANNED_SYMBOLS
]

EQUITY_CYCLE_INTERVAL = 300  # 5 minutes
OPTIONS_CYCLE_INTERVAL = 1800  # 30 minutes (was 300s — caused excessive trading)


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


# ---------------------------------------------------------------------------
# Phase 4: Health-check HTTP endpoint (port 8080)
# ---------------------------------------------------------------------------

_STARTUP_TIME = time.monotonic()
_HEALTH_STATE: Dict[str, Any] = {
    "positions_count": 0,
    "today_pnl": 0.0,
    "last_cycle_ts": None,
    "engine_status": "starting",
}


async def _health_handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    """Minimal HTTP/1.1 handler that returns JSON health payload."""
    try:
        # Read request line (we don't care about path — everything returns health)
        await asyncio.wait_for(reader.readline(), timeout=5.0)
        # Drain remaining headers
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=2.0)
            if line in (b"\r\n", b"\n", b""):
                break
    except Exception:
        pass

    import json as _json
    body = _json.dumps({
        "status": "ok",
        "uptime_seconds": round(time.monotonic() - _STARTUP_TIME, 1),
        "positions_count": _HEALTH_STATE["positions_count"],
        "today_pnl": _HEALTH_STATE["today_pnl"],
        "last_cycle_ts": _HEALTH_STATE["last_cycle_ts"],
        "engine_status": _HEALTH_STATE["engine_status"],
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })
    response = (
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Connection: close\r\n"
        "\r\n"
        f"{body}"
    )
    writer.write(response.encode())
    await writer.drain()
    writer.close()


async def _start_health_server(port: int = 8080):
    """Start a lightweight asyncio TCP server for health checks."""
    try:
        server = await asyncio.start_server(_health_handler, "0.0.0.0", port)
        logger.info(f"Health-check endpoint listening on :{port}")
        return server
    except Exception as exc:
        logger.warning(f"Could not start health-check server on :{port}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Phase 4: Graceful shutdown — cancel open orders, log final state, exit 0
# ---------------------------------------------------------------------------

async def _graceful_shutdown(
    equity: EquityEngine,
    options: OptionsEngine,
    stop_event: asyncio.Event,
):
    """Cancel all open orders on Alpaca and log final state."""
    logger.info("=== GRACEFUL SHUTDOWN: cancelling open orders ===")

    # Cancel orders via equity engine's Alpaca client
    if equity.client is not None:
        try:
            equity.client._request("DELETE", "/v2/orders")
            logger.info("All open equity orders cancelled")
        except Exception as exc:
            logger.warning(f"Failed to cancel equity orders: {exc}")

    # Cancel orders via options engine's Alpaca client
    if options.engine is not None:
        try:
            options.engine.trade_executor.trading_client.cancel_orders()
            logger.info("All open option orders cancelled")
        except Exception as exc:
            logger.warning(f"Failed to cancel option orders: {exc}")

        # Persist final state
        try:
            options.engine._save_state()
            logger.info("Options engine state saved")
        except Exception as exc:
            logger.warning(f"Failed to save options state: {exc}")

    logger.info("=== GRACEFUL SHUTDOWN COMPLETE ===")


async def main(mode: str, broker: str = "alpaca"):
    """Orchestrate equity + options engines with health check and graceful shutdown.

    Args:
        mode: 'paper' or 'live'.
        broker: 'ibkr' or 'alpaca'.
    """
    logger.info("=" * 70)
    logger.info(f"  V28 PRODUCTION TRADING SYSTEM — mode={mode}, broker={broker}")
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

    # Initialize engines
    equity = EquityEngine(mode, broker=broker)
    options = OptionsEngine(mode)

    await equity.initialize()
    await options.initialize()

    # Phase 4: Graceful shutdown on SIGTERM / SIGINT
    loop = asyncio.get_running_loop()

    def _handle_sig(sig):
        logger.info(f"Received signal {sig.name} — initiating graceful shutdown")
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _handle_sig, sig)

    # Phase 4: Start health-check HTTP server
    health_server = await _start_health_server(
        port=int(os.getenv("HEALTH_PORT", "8080"))
    )
    _HEALTH_STATE["engine_status"] = "running"

    # Start Prometheus metrics server
    if HAS_METRICS:
        metrics_port = int(os.getenv("METRICS_PORT", "9090"))
        metrics_server = MetricsServer(port=metrics_port)
        metrics_server.start()

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
                await _graceful_shutdown(equity, options, stop_event)
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
    _HEALTH_STATE["engine_status"] = "shutting_down"
    for t in pending:
        t.cancel()
    if pending:
        await asyncio.wait(pending, timeout=10)

    # Phase 4: graceful shutdown
    await _graceful_shutdown(equity, options, stop_event)

    # Phase D: save ML model on shutdown
    if equity.online_learner:
        try:
            equity.online_learner.save_checkpoint(tag="shutdown")
            logger.info("OnlineLearner checkpoint saved on shutdown")
        except Exception as e:
            logger.warning(f"OnlineLearner save failed: {e}")

    # Phase D: stop health monitor
    if equity.health_monitor:
        try:
            equity.health_monitor.stop()
        except Exception:
            pass

    # Phase D: disconnect IBKR
    if equity.ibkr_client:
        try:
            equity.ibkr_client.disconnect()
            logger.info("IBKR disconnected")
        except Exception:
            pass

    if health_server is not None:
        health_server.close()
        await health_server.wait_closed()

    logger.info("V28 Production System shutdown complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V28 Production Trading System")
    parser.add_argument("--mode", choices=["live", "paper"], default="paper",
                        help="Trading mode (default: paper)")
    parser.add_argument("--broker", choices=["ibkr", "alpaca"], default="alpaca",
                        help="Broker backend (default: alpaca)")
    parser.add_argument("--live", action="store_true",
                        help="Shortcut for --mode=live")
    args = parser.parse_args()

    # --live flag overrides --mode
    mode = "live" if args.live else args.mode
    # BROKER env var overrides CLI
    broker = os.getenv("BROKER", args.broker)

    # Grand Overhaul: while True restart loop with crash logging
    crash_log_path = PROJECT_ROOT / "logs" / "crash.log"
    crash_log_path.parent.mkdir(parents=True, exist_ok=True)

    while True:
        try:
            asyncio.run(main(mode, broker))
            break  # Clean exit
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            break
        except Exception as exc:
            logger.critical(f"Fatal crash: {exc}", exc_info=True)

            # Log crash to file
            try:
                import traceback
                with open(crash_log_path, "a") as f:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"CRASH: {datetime.now().isoformat()}\n")
                    f.write(f"Error: {exc}\n")
                    f.write(traceback.format_exc())
                    f.write(f"{'='*60}\n")
            except Exception:
                pass

            # Discord alert
            try:
                discord_url = os.getenv("DISCORD_WEBHOOK_MARCUS", "")
                if discord_url:
                    import requests
                    requests.post(discord_url, json={
                        "embeds": [{
                            "title": "🚨 V28 CRASH — Auto-restarting",
                            "description": f"```{str(exc)[:1500]}```",
                            "color": 0xFF0000,
                        }]
                    }, timeout=10)
            except Exception:
                pass

            logger.info("Restarting in 30 seconds...")
            time.sleep(30)
