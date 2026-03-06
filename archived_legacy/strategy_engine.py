#!/usr/bin/env python3
"""
Strategy Engine — Market-Neutral Statistical Arbitrage Signal Generator
========================================================================

Replaces the old momentum-chasing signal generation with THREE proven
strategies used by profitable quant funds:

  A) PAIRS TRADING (Mean Reversion via Cointegration)
     - 50% capital allocation
     - Market-neutral: long underperformer, short outperformer
     - 55-60% win rate with small, consistent gains

  B) MEAN REVERSION (Bollinger + RSI + Volume)
     - 30% capital allocation
     - Enter on extremes, exit at mean
     - Max 5-day hold, tight stops

  C) MOMENTUM WITH REGIME FILTER (trend-following, only when A+B idle)
     - 20% capital allocation
     - 200-day SMA regime filter + ADX confirmation
     - Pullback entries to 20-day EMA
     - 2x ATR trailing stop

Core principle: be MARKET NEUTRAL. Profit from relative mispricings,
not directional bets.

Usage:
    from strategy_engine import StrategyEngine, EngineConfig

    engine = StrategyEngine(EngineConfig())
    signals = engine.get_signals(price_data, volume_data, equity, positions)
"""

import os
import sys
import json
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import numpy as np
import pandas as pd

from pair_finder import (
    PairFinder,
    PairFinderConfig,
    CointegrationResult,
    KalmanSpreadTracker,
    SECTOR_UNIVERSE,
    ALL_SYMBOLS,
)

# ---------------------------------------------------------------------------
# Optional advanced modules (graceful fallback if unavailable)
# ---------------------------------------------------------------------------
try:
    from src.regime_detector import HMMRegimeDetector, Regime
    _HMM_AVAILABLE = True
except ImportError:
    _HMM_AVAILABLE = False

try:
    from src.regime_detector import LiveRegimeDetector, RegimeAdjustments, LiveRegime
    _LIVE_REGIME_AVAILABLE = True
except ImportError:
    _LIVE_REGIME_AVAILABLE = False

try:
    from src.quant_models.garch import GARCHModel
    _GARCH_AVAILABLE = True
except ImportError:
    _GARCH_AVAILABLE = False

try:
    from src.ml_ensemble_stacker import MLEnsembleStacker
    _STACKER_AVAILABLE = True
except ImportError:
    _STACKER_AVAILABLE = False

try:
    from src.order_flow_analyzer import OrderFlowAnalyzer, RealTimeFlowTracker
    _FLOW_AVAILABLE = True
except ImportError:
    _FLOW_AVAILABLE = False

try:
    from src.adaptive_parameters import AdaptiveParameterTuner, TradeRecord
    _ADAPTIVE_AVAILABLE = True
except ImportError:
    _ADAPTIVE_AVAILABLE = False

try:
    from src.sentiment_alpha import SentimentSignalProcessor, SentimentSignal
    _SENTIMENT_AVAILABLE = True
except ImportError:
    _SENTIMENT_AVAILABLE = False

logger = logging.getLogger("strategy_engine")


# ============================================================================
# ENUMS & DATA CLASSES
# ============================================================================

class StrategyType(Enum):
    PAIRS = "pairs_trading"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum_regime"
    VWAP_REVERSION = "vwap_reversion"


class SignalDirection(Enum):
    LONG = "long"
    SHORT = "short"
    CLOSE = "close"


@dataclass
class TradeSignal:
    """
    A single trade signal produced by the strategy engine.

    Every signal includes all the information needed for execution:
    entry, exit, stop, size, and the reasoning behind it.
    """
    symbol: str
    direction: SignalDirection
    strategy: StrategyType
    confidence: float               # [0, 1] — how confident the strategy is

    # Sizing
    position_size_pct: float        # % of equity to allocate
    shares: int = 0                 # Computed by portfolio_allocator

    # Prices
    entry_price: float = 0.0        # Expected entry price
    stop_price: float = 0.0         # Hard stop loss
    target_price: float = 0.0       # Take-profit target

    # Strategy-specific metadata
    strategy_source: str = ""       # Human-readable reason
    z_score: float = 0.0            # For pairs/MR strategies
    hedge_ratio: float = 0.0        # For pairs trading
    pair_symbol: str = ""           # The other leg of a pair
    half_life: float = 0.0          # Mean reversion half-life
    atr: float = 0.0                # ATR at signal time
    rsi: float = 0.0                # RSI at signal time
    adx: float = 0.0                # ADX at signal time

    # Advanced module enrichment
    regime: str = ""                # HMM regime (e.g. "trending_bull")
    regime_confidence: float = 0.0  # HMM posterior probability for current regime
    garch_vol: float = 0.0         # GARCH 1-day annualized vol forecast
    flow_score: float = 0.0        # Institutional smart money score [-1, 1]
    ml_alpha: float = 0.0          # ML ensemble stacker alpha score [0, 1]

    # Tracking
    timestamp: str = ""
    max_hold_days: int = 0          # 0 = no limit
    pair_id: str = ""               # Links pair legs together

    def to_dict(self) -> dict:
        d = asdict(self)
        d["direction"] = self.direction.value
        d["strategy"] = self.strategy.value
        return d


# ============================================================================
# TECHNICAL INDICATOR CALCULATIONS
# ============================================================================

def compute_rsi(prices: np.ndarray, period: int = 14) -> float:
    """
    Relative Strength Index.
    RSI < 30 = oversold (mean-reversion buy signal)
    RSI > 70 = overbought (mean-reversion sell signal)
    """
    if len(prices) < period + 1:
        return 50.0  # Neutral default

    deltas = np.diff(prices[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)

    if avg_loss < 1e-10:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def compute_atr(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> float:
    """
    Average True Range — measures volatility.
    Used for stop placement: stops at N x ATR from entry.
    """
    if len(closes) < period + 1:
        # Fallback: simple range
        return float(np.mean(highs[-period:] - lows[-period:])) if len(highs) >= period else 0.0

    tr_list = []
    for i in range(1, min(period + 1, len(closes))):
        high_low = highs[-period + i - 1] - lows[-period + i - 1]
        high_close = abs(highs[-period + i - 1] - closes[-period + i - 2])
        low_close = abs(lows[-period + i - 1] - closes[-period + i - 2])
        tr_list.append(max(high_low, high_close, low_close))

    return float(np.mean(tr_list)) if tr_list else 0.0


def compute_adx(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> float:
    """
    Average Directional Index — measures trend strength.
    ADX > 25 = trending market (use momentum strategy)
    ADX < 20 = ranging market (use mean reversion)
    """
    n = len(closes)
    if n < period * 2:
        return 0.0

    # +DM and -DM
    plus_dm = np.zeros(n - 1)
    minus_dm = np.zeros(n - 1)
    tr = np.zeros(n - 1)

    for i in range(1, n):
        high_diff = highs[i] - highs[i - 1]
        low_diff = lows[i - 1] - lows[i]

        plus_dm[i - 1] = high_diff if (high_diff > low_diff and high_diff > 0) else 0
        minus_dm[i - 1] = low_diff if (low_diff > high_diff and low_diff > 0) else 0

        tr[i - 1] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )

    # Smoothed averages (Wilder's)
    def wilder_smooth(arr, p):
        result = np.zeros(len(arr))
        result[p - 1] = np.sum(arr[:p])
        for i in range(p, len(arr)):
            result[i] = result[i - 1] - result[i - 1] / p + arr[i]
        return result

    atr_smooth = wilder_smooth(tr, period)
    plus_di_smooth = wilder_smooth(plus_dm, period)
    minus_di_smooth = wilder_smooth(minus_dm, period)

    # +DI and -DI
    valid = atr_smooth > 0
    plus_di = np.where(valid, 100 * plus_di_smooth / atr_smooth, 0)
    minus_di = np.where(valid, 100 * minus_di_smooth / atr_smooth, 0)

    # DX
    di_sum = plus_di + minus_di
    di_diff = np.abs(plus_di - minus_di)
    dx = np.where(di_sum > 0, 100 * di_diff / di_sum, 0)

    # ADX = smoothed DX
    adx_vals = wilder_smooth(dx, period)

    # Return latest ADX
    return float(adx_vals[-1]) if len(adx_vals) > 0 else 0.0


def compute_bollinger_bands(
    closes: np.ndarray,
    period: int = 20,
    num_std: float = 2.0,
) -> Tuple[float, float, float]:
    """
    Bollinger Bands.
    Returns (upper_band, middle_band, lower_band).

    Price below lower band = oversold (potential buy for mean reversion)
    Price above upper band = overbought (potential sell for mean reversion)
    """
    if len(closes) < period:
        mid = float(np.mean(closes))
        std = float(np.std(closes))
        return mid + num_std * std, mid, mid - num_std * std

    recent = closes[-period:]
    mid = float(np.mean(recent))
    std = float(np.std(recent))

    return mid + num_std * std, mid, mid - num_std * std


def compute_ema(prices: np.ndarray, period: int) -> float:
    """Exponential Moving Average — last value."""
    if len(prices) < period:
        return float(np.mean(prices))

    multiplier = 2.0 / (period + 1)
    ema = float(prices[0])
    for price in prices[1:]:
        ema = (float(price) - ema) * multiplier + ema
    return ema


def compute_sma(prices: np.ndarray, period: int) -> float:
    """Simple Moving Average — last value."""
    if len(prices) < period:
        return float(np.mean(prices))
    return float(np.mean(prices[-period:]))


def compute_volume_ratio(volumes: np.ndarray, period: int = 20) -> float:
    """
    Current volume / 20-day average volume.
    > 1.5 = volume spike (confirms signal).
    """
    if len(volumes) < period + 1:
        return 1.0
    avg_vol = float(np.mean(volumes[-period - 1:-1]))  # Excludes current bar
    current_vol = float(volumes[-1])
    if avg_vol < 1:
        return 1.0
    return current_vol / avg_vol


# ============================================================================
# ENGINE CONFIGURATION
# ============================================================================

@dataclass
class EngineConfig:
    """Configuration for the strategy engine."""

    # --- Strategy allocation (must sum to 1.0) ---
    pairs_allocation: float = 0.50      # 50% to pairs trading
    mr_allocation: float = 0.30         # 30% to mean reversion
    momentum_allocation: float = 0.20   # 20% to momentum (only when A+B idle)

    # --- Dynamic Strategy Allocation ---
    use_dynamic_allocation: bool = True
    dynamic_alloc_min_trades: int = 10  # Min trades per strategy before dynamic kicks in
    dynamic_alloc_floor: float = 0.10   # 10% minimum per strategy
    dynamic_alloc_lookback: int = 30    # Last 30 trades for rolling Sharpe

    # --- Drawdown-Responsive Position Sizing ---
    drawdown_scale_threshold: float = 0.05   # Start scaling at 5% drawdown
    drawdown_half_threshold: float = 0.10    # Halve sizes at 10% drawdown
    drawdown_halt_threshold: float = 0.15    # Stop new positions at 15% drawdown

    # --- STRATEGY A: Pairs Trading ---
    pairs_entry_z: float = 2.0          # Enter when |z-score| > 2.0 (2 std from mean)
    pairs_exit_z: float = 0.5           # Exit when |z-score| < 0.5 (close to mean)
    pairs_stop_z: float = 4.0           # Stop when |z-score| > 4.0 (relationship breaking)
    pairs_lookback: int = 60            # 60-day rolling window for z-score & hedge ratio
    pairs_rebalance_days: int = 30      # Recalculate pairs monthly
    pairs_max_positions: int = 5        # Max 5 pair positions simultaneously
    pairs_position_pct: float = 0.04    # 4% per leg (8% per pair) — conservative

    # --- STRATEGY B: Mean Reversion (Bollinger + RSI) ---
    mr_bb_period: int = 20              # 20-period Bollinger Bands (standard)
    mr_bb_std: float = 2.5              # 2.5 standard deviations (wider entry -> fewer false MR)
    mr_rsi_period: int = 14             # 14-period RSI (standard)
    mr_rsi_oversold: float = 25.0       # RSI < 25 = oversold (stricter filter)
    mr_rsi_overbought: float = 70.0     # RSI > 70 = overbought (sell)
    mr_volume_spike: float = 1.5        # Volume must be 1.5x 20-day avg (confirms capitulation)
    mr_atr_stop_mult: float = 2.5       # Stop at 2.5x ATR from entry (wider for MR)
    mr_max_hold_days: int = 8           # Max 8 day hold (give MR more room)
    mr_target_sma: bool = True          # Target = 20-day SMA (reversion target)
    mr_max_positions: int = 5           # Max 5 MR positions

    # --- STRATEGY C: Momentum with Regime Filter ---
    mom_sma_period: int = 200           # 200-day SMA regime filter (institutional standard)
    mom_ema_period: int = 20            # 20-day EMA for pullback entry
    mom_adx_threshold: float = 25.0     # ADX > 25 = confirmed trend (Wilder's threshold)
    mom_atr_trail_mult: float = 2.0     # 2x ATR trailing stop
    mom_scale_in: bool = True           # Scale in: 50% initial, 50% on confirmation
    mom_max_positions: int = 3          # Max 3 momentum positions

    # --- Global risk controls ---
    max_position_pct: float = 0.05      # 5% max per individual position
    max_sector_pct: float = 0.15        # 15% max per sector
    max_net_beta: float = 0.20          # Keep net beta between -0.2 and +0.2
    min_confidence: float = 0.50        # Minimum confidence to generate a signal

    # --- Phase 5b: VWAP intraday mean reversion ---
    vwap_enabled: bool = True               # Enable VWAP reversion scanning
    vwap_lookback: int = 20                 # 20-bar VWAP window
    vwap_entry_std: float = 1.5             # Enter at 1.5 std devs from VWAP
    vwap_rsi_oversold: float = 30.0         # RSI confirmation for longs
    vwap_rsi_overbought: float = 70.0       # RSI confirmation for shorts
    vwap_stop_mult: float = 2.0             # Stop at 2x the entry distance
    vwap_max_positions: int = 3             # Max simultaneous VWAP positions
    vwap_max_hold_days: int = 3             # Short hold — intraday/overnight

    # --- Data ---
    min_bars_required: int = 250        # Need at least 250 bars for 200-SMA

    # --- Advanced modules (graceful fallback when unavailable) ---
    use_hmm_regime: bool = True         # Use HMM regime instead of 200-SMA
    use_live_regime: bool = True        # Use 3-state LiveRegimeDetector for strategy weighting
    use_garch_stops: bool = True        # Use GARCH vol for dynamic stops
    use_ml_stacker: bool = True         # Score signals through ML ensemble
    use_order_flow: bool = True         # Confirm signals with institutional flow
    use_adaptive_params: bool = True    # Self-tune thresholds from P&L
    use_sentiment: bool = True          # Process sentiment with decay for confidence boost
    hmm_high_vol_scale: float = 0.5     # Scale confidence by this in HIGH_VOL regime
    flow_reject_threshold: float = -0.3 # Reject signals with flow score below this


# ============================================================================
# STRATEGY ENGINE
# ============================================================================

class StrategyEngine:
    """
    Market-neutral strategy engine combining three proven approaches.

    Usage flow:
      1. engine = StrategyEngine(config)
      2. signals = engine.get_signals(price_data, volume_data, equity, positions)
      3. Pass signals to portfolio_allocator for sizing
      4. Pass sized signals to risk_guardian for validation
      5. Execute via unified_trader
    """

    def __init__(self, config: EngineConfig = None):
        self.cfg = config or EngineConfig()

        # Pair finder (handles cointegration discovery)
        pair_cfg = PairFinderConfig(
            entry_z=self.cfg.pairs_entry_z,
            exit_z=self.cfg.pairs_exit_z,
            stop_z=self.cfg.pairs_stop_z,
            z_score_lookback=self.cfg.pairs_lookback,
        )
        self.pair_finder = PairFinder(pair_cfg)

        # State
        self._pairs: List[CointegrationResult] = []
        self._last_pair_refresh: Optional[datetime] = None
        self._active_pair_positions: Dict[str, dict] = {}  # pair_id -> info
        self._strategy_stats: Dict[str, dict] = {
            "pairs_trading": {"wins": 0, "losses": 0, "total_pnl": 0.0},
            "mean_reversion": {"wins": 0, "losses": 0, "total_pnl": 0.0},
            "momentum_regime": {"wins": 0, "losses": 0, "total_pnl": 0.0},
            "vwap_reversion": {"wins": 0, "losses": 0, "total_pnl": 0.0},
        }

        # --- Drawdown tracking ---
        self._peak_equity: float = 0.0
        self._current_drawdown_pct: float = 0.0

        # --- Dynamic allocation: per-strategy PnL history ---
        self._trade_pnls: Dict[str, List[float]] = {
            "pairs_trading": [],
            "mean_reversion": [],
            "momentum_regime": [],
            "vwap_reversion": [],
        }

        # --- Kalman spread tracker cache (per pair_id) ---
        self._kalman_trackers: Dict[str, 'KalmanSpreadTracker'] = {}
        self._kalman_data_lengths: Dict[str, int] = {}  # track data size at bootstrap

        # --- Advanced module instances (None when unavailable) ---
        self._hmm: Optional[Any] = None
        self._garch: Optional[Any] = None
        self._stacker: Optional[Any] = None
        self._flow_analyzer: Optional[Any] = None
        self._rt_flow_tracker: Optional[Any] = None
        self._adaptive_tuner: Optional[Any] = None
        self._live_regime: Optional[Any] = None
        self._current_regime: str = "unknown"
        self._regime_probs: Optional[np.ndarray] = None
        self._regime_adjustments: Optional[Any] = None

        if self.cfg.use_live_regime and _LIVE_REGIME_AVAILABLE:
            try:
                self._live_regime = LiveRegimeDetector(lookback_bars=252, refit_days=7)
                logger.info("LiveRegimeDetector (3-state) initialized")
            except Exception as e:
                logger.warning(f"LiveRegimeDetector init failed: {e}")

        if self.cfg.use_hmm_regime and _HMM_AVAILABLE:
            try:
                self._hmm = HMMRegimeDetector(n_states=4)
                logger.info("HMM regime detector initialized")
            except Exception as e:
                logger.warning(f"HMM init failed: {e}")

        if self.cfg.use_garch_stops and _GARCH_AVAILABLE:
            try:
                self._garch = GARCHModel(lookback_days=504)
                logger.info("GARCH volatility model initialized")
            except Exception as e:
                logger.warning(f"GARCH init failed: {e}")

        if self.cfg.use_ml_stacker and _STACKER_AVAILABLE:
            try:
                self._stacker = MLEnsembleStacker()
                logger.info("ML ensemble stacker initialized")
            except Exception as e:
                logger.warning(f"ML stacker init failed: {e}")

        if self.cfg.use_order_flow and _FLOW_AVAILABLE:
            try:
                self._flow_analyzer = OrderFlowAnalyzer()
                self._rt_flow_tracker = RealTimeFlowTracker(window_seconds=60)
                logger.info("Order flow analyzer + real-time tracker initialized")
            except Exception as e:
                logger.warning(f"Order flow init failed: {e}")

        if self.cfg.use_adaptive_params and _ADAPTIVE_AVAILABLE:
            try:
                self._adaptive_tuner = AdaptiveParameterTuner()
                logger.info("Adaptive parameter tuner initialized")
            except Exception as e:
                logger.warning(f"Adaptive tuner init failed: {e}")

        self._sentiment_processor: Optional[Any] = None
        if self.cfg.use_sentiment and _SENTIMENT_AVAILABLE:
            try:
                self._sentiment_processor = SentimentSignalProcessor(
                    fast_half_life_hours=6.0,
                    slow_half_life_hours=24.0,
                )
                logger.info("Sentiment signal processor initialized (6h/24h decay)")
            except Exception as e:
                logger.warning(f"Sentiment processor init failed: {e}")

    # ------------------------------------------------------------------
    # Advanced module helpers
    # ------------------------------------------------------------------

    def _detect_regime(self, price_data: pd.DataFrame) -> Tuple[str, float]:
        """
        Detect market regime via HMM (or 200-SMA fallback).

        Returns (regime_name, confidence) e.g. ("trending_bull", 0.72).
        """
        if self._hmm is None:
            return "unknown", 0.0

        try:
            # Build feature matrix from a representative symbol (SPY or first col)
            ref_sym = "SPY" if "SPY" in price_data.columns else price_data.columns[0]
            prices = price_data[ref_sym].dropna().values.astype(float)
            if len(prices) < 60:
                return "unknown", 0.0

            # Daily log returns
            returns = np.diff(np.log(prices))

            # 20-day realised vol (annualized)
            vol_window = 20
            volatility = np.array([
                np.std(returns[max(0, i - vol_window):i]) * np.sqrt(252)
                if i >= vol_window else np.std(returns[:max(1, i)]) * np.sqrt(252)
                for i in range(1, len(returns) + 1)
            ])

            # Momentum sign (10-day cumulative return)
            mom = np.array([
                np.sign(np.sum(returns[max(0, i - 10):i]))
                for i in range(1, len(returns) + 1)
            ])

            # Fit if not yet fitted
            if not self._hmm.is_fitted:
                self._hmm.fit(returns, volatility, mom)

            state_idx, probs = self._hmm.predict(returns, volatility, mom)
            regime = self._hmm.state_to_regime(state_idx)
            confidence = float(probs[state_idx])

            regime_name = regime.value if hasattr(regime, 'value') else str(regime)
            self._current_regime = regime_name
            self._regime_probs = probs

            logger.info(f"HMM regime: {regime_name} (conf={confidence:.2f})")
            return regime_name, confidence

        except Exception as e:
            logger.warning(f"HMM regime detection failed: {e}")
            return "unknown", 0.0

    def _get_garch_vol(self, prices: np.ndarray) -> float:
        """
        Get GARCH 1-day ahead annualized vol forecast.
        Returns 0.0 if unavailable (caller falls back to ATR).
        """
        if self._garch is None or len(prices) < 60:
            return 0.0

        try:
            returns = np.diff(np.log(prices))
            params, _, sigma2 = self._garch.fit(returns)
            eps = returns - np.mean(returns)
            forecasts = self._garch.forecast(params, eps[-1], sigma2[-1], horizon=1)
            return forecasts[0] if forecasts else 0.0
        except Exception as e:
            logger.debug(f"GARCH forecast failed: {e}")
            return 0.0

    def _get_flow_signal(self, symbol: str, ohlcv_data: Optional[Dict[str, pd.DataFrame]]) -> Tuple[float, str]:
        """
        Get institutional flow score for a symbol.
        Returns (smart_money_score, bias_string).
        Score is in [-1, 1]: positive = accumulation, negative = distribution.

        Uses real-time microstructure data when available, falls back to bar-based analysis.
        """
        if self._flow_analyzer is None or ohlcv_data is None:
            return 0.0, "neutral"

        # 1. Try real-time microstructure signal first
        if self._rt_flow_tracker is not None and self._rt_flow_tracker.is_connected:
            try:
                micro = self._rt_flow_tracker.get_microstructure_signal(symbol)
                if micro.confidence >= 0.2:
                    logger.debug(
                        f"RT flow {symbol}: imb={micro.imbalance_score:+.3f} "
                        f"conf={micro.confidence:.2f} dir={micro.suggested_direction}"
                    )
                    return micro.imbalance_score, micro.suggested_direction
            except Exception as e:
                logger.debug(f"RT flow failed for {symbol}: {e}")

        try:
            if symbol not in ohlcv_data:
                return 0.0, "neutral"
            df = ohlcv_data[symbol]
            # Convert DataFrame rows to list of dicts for OrderFlowAnalyzer
            bars = []
            for _, row in df.tail(60).iterrows():
                bars.append({
                    'open': float(row.get('open', row.get('Open', 0))),
                    'high': float(row.get('high', row.get('High', 0))),
                    'low': float(row.get('low', row.get('Low', 0))),
                    'close': float(row.get('close', row.get('Close', 0))),
                    'volume': float(row.get('volume', row.get('Volume', 0))),
                })
            if len(bars) < 20:
                return 0.0, "neutral"

            flow = self._flow_analyzer.analyze(symbol, bars)
            return flow.smart_money_score, flow.trade_bias
        except Exception as e:
            logger.debug(f"Flow analysis failed for {symbol}: {e}")
            return 0.0, "neutral"

    def _apply_regime_scaling(self, signal: TradeSignal, regime: str, regime_conf: float) -> TradeSignal:
        """
        Scale signal confidence and sizing based on current regime.

        If LiveRegimeDetector is active, uses its position_scale & stop_multiplier.
        Otherwise falls back to the original hardcoded rules.
        """
        signal.regime = regime
        signal.regime_confidence = regime_conf

        # --- Live regime path (preferred) ---
        if self._regime_adjustments is not None:
            adj = self._regime_adjustments
            signal.confidence *= adj.position_scale
            signal.position_size_pct *= adj.position_scale

            # Tighter or looser stops via stop_multiplier
            if signal.stop_price > 0 and signal.atr > 0:
                if signal.direction == SignalDirection.LONG:
                    gap = signal.entry_price - signal.stop_price
                    signal.stop_price = signal.entry_price - gap * adj.stop_multiplier
                elif signal.direction == SignalDirection.SHORT:
                    gap = signal.stop_price - signal.entry_price
                    signal.stop_price = signal.entry_price + gap * adj.stop_multiplier
                signal.stop_price = round(signal.stop_price, 2)
            return signal

        # --- Fallback: original hardcoded logic ---
        if regime == "high_volatility":
            signal.confidence *= self.cfg.hmm_high_vol_scale
            signal.position_size_pct *= self.cfg.hmm_high_vol_scale
        elif regime == "mean_reverting" and signal.strategy == StrategyType.MEAN_REVERSION:
            signal.confidence = min(0.95, signal.confidence * 1.15)
        elif regime in ("trending_bull", "trending_bear") and signal.strategy == StrategyType.MOMENTUM:
            signal.confidence = min(0.95, signal.confidence * 1.10)
        elif regime == "mean_reverting" and signal.strategy == StrategyType.MOMENTUM:
            signal.confidence *= 0.7  # Momentum in ranging market = bad

        return signal

    def _enrich_with_flow(self, signal: TradeSignal, ohlcv_data: Optional[Dict[str, pd.DataFrame]]) -> Optional[TradeSignal]:
        """
        Enrich signal with order flow data. Returns None if flow rejects the signal.
        """
        flow_score, flow_bias = self._get_flow_signal(signal.symbol, ohlcv_data)
        signal.flow_score = flow_score

        # Reject if institutional flow strongly opposes the signal
        if signal.direction == SignalDirection.LONG and flow_score < self.cfg.flow_reject_threshold:
            logger.info(f"Flow REJECTS {signal.symbol} LONG (score={flow_score:.2f})")
            return None
        if signal.direction == SignalDirection.SHORT and flow_score > -self.cfg.flow_reject_threshold:
            # For shorts, positive flow (accumulation) is opposing
            if flow_score > abs(self.cfg.flow_reject_threshold):
                logger.info(f"Flow REJECTS {signal.symbol} SHORT (score={flow_score:.2f})")
                return None

        # Boost confidence if flow agrees
        if signal.direction == SignalDirection.LONG and flow_score > 0.3:
            signal.confidence = min(0.95, signal.confidence * (1 + flow_score * 0.2))
        elif signal.direction == SignalDirection.SHORT and flow_score < -0.3:
            signal.confidence = min(0.95, signal.confidence * (1 + abs(flow_score) * 0.2))

        return signal

    # ------------------------------------------------------------------
    # Real-Time Flow Connection
    # ------------------------------------------------------------------

    def connect_realtime_flow(self, api_client: Optional[object] = None, symbols: Optional[list] = None):
        """
        Connect the real-time flow tracker to live trade stream.

        Args:
            api_client: Alpaca API client with streaming support.
            symbols: List of symbols to subscribe to.
        """
        if self._rt_flow_tracker is not None:
            self._rt_flow_tracker.connect_realtime(api_client, symbols)
            logger.info(f"Real-time flow tracker connected for {len(symbols or [])} symbols")
        else:
            logger.warning("Real-time flow tracker not available")

    def feed_trade_tick(self, symbol: str, price: float, size: float):
        """Feed a single trade tick to the real-time flow tracker (manual mode)."""
        if self._rt_flow_tracker is not None:
            self._rt_flow_tracker.on_trade(symbol, price, size)

    # ------------------------------------------------------------------
    # Drawdown-Responsive Position Sizing
    # ------------------------------------------------------------------

    def _update_drawdown_state(self, equity: float) -> float:
        """
        Update peak equity tracker and compute current drawdown percentage.

        Called at the start of every signal generation cycle to track
        the portfolio's high-water mark and current drawdown.

        Returns the current drawdown as a positive fraction
        (0.0 = no drawdown, 0.15 = 15% from peak).
        """
        if equity <= 0:
            return self._current_drawdown_pct

        if equity > self._peak_equity:
            self._peak_equity = equity

        if self._peak_equity > 0:
            self._current_drawdown_pct = (self._peak_equity - equity) / self._peak_equity
        else:
            self._current_drawdown_pct = 0.0

        return self._current_drawdown_pct

    def _get_drawdown_scale_factor(self, drawdown_pct: float) -> float:
        """
        Compute position size scaling factor based on current drawdown.

        Implements a tiered drawdown response:
          - drawdown < 5%:   1.0  (full size, no adjustment)
          - drawdown 5-10%:  linear scale from 1.0 → 0.5
          - drawdown 10-15%: 0.5  (halved positions)
          - drawdown > 15%:  0.0  (circuit breaker — no new positions)

        This prevents the death spiral of adding risk into a drawdown.

        Returns a multiplier in [0.0, 1.0].
        """
        if drawdown_pct >= self.cfg.drawdown_halt_threshold:
            return 0.0  # Circuit breaker: no new positions

        if drawdown_pct >= self.cfg.drawdown_half_threshold:
            return 0.5  # Halve all positions

        if drawdown_pct >= self.cfg.drawdown_scale_threshold:
            # Linear interpolation from 1.0 at scale_threshold to 0.5 at half_threshold
            range_width = self.cfg.drawdown_half_threshold - self.cfg.drawdown_scale_threshold
            if range_width > 0:
                progress = (drawdown_pct - self.cfg.drawdown_scale_threshold) / range_width
                return 1.0 - 0.5 * progress
            return 0.5

        return 1.0  # No drawdown scaling needed

    @property
    def current_drawdown_pct(self) -> float:
        """Current drawdown as a percentage (0.0 to 1.0)."""
        return self._current_drawdown_pct

    @property
    def peak_equity(self) -> float:
        """High-water mark equity."""
        return self._peak_equity

    # ------------------------------------------------------------------
    # Dynamic Strategy Allocation
    # ------------------------------------------------------------------

    def _compute_dynamic_weights(self) -> Tuple[float, float, float]:
        """
        Compute dynamic strategy allocation weights based on realized rolling Sharpe.

        Uses the last N trade PnLs per strategy to estimate a trade-level Sharpe
        ratio, then reweights capital allocation proportional to Sharpe.
        Each strategy gets at least `dynamic_alloc_floor` (default 10%).

        When insufficient trade history exists, falls back to static config weights.

        Returns (w_pairs, w_mr, w_mom) that sum to 1.0.
        """
        if not self.cfg.use_dynamic_allocation:
            return self.cfg.pairs_allocation, self.cfg.mr_allocation, self.cfg.momentum_allocation

        strategies = ["pairs_trading", "mean_reversion", "momentum_regime"]
        sharpes: Dict[str, float] = {}
        any_has_data = False

        for strat in strategies:
            pnls = self._trade_pnls.get(strat, [])
            recent = pnls[-self.cfg.dynamic_alloc_lookback:]

            if len(recent) < self.cfg.dynamic_alloc_min_trades:
                sharpes[strat] = 0.0  # Not enough data
                continue

            any_has_data = True
            arr = np.array(recent, dtype=float)
            std = float(np.std(arr))

            if std < 1e-10:
                # Perfect consistency: assign very high Sharpe
                sharpes[strat] = 10.0 if float(np.mean(arr)) > 0 else 0.0
            else:
                sharpes[strat] = max(0.0, float(np.mean(arr)) / std)

        if not any_has_data:
            # No strategy has enough trade history → use static defaults
            return self.cfg.pairs_allocation, self.cfg.mr_allocation, self.cfg.momentum_allocation

        total_sharpe = sum(sharpes.values())
        floor = self.cfg.dynamic_alloc_floor

        if total_sharpe <= 0:
            # All strategies have zero/negative Sharpe → equal weight
            n = len(strategies)
            equal = 1.0 / n
            logger.info(f"Dynamic allocation: all Sharpes <= 0, using equal {equal:.0%}")
            return equal, equal, equal

        # Weight proportional to Sharpe, with floor + iterative normalization
        raw_weights = {}
        for strat in strategies:
            raw_weights[strat] = max(floor, sharpes[strat] / total_sharpe)

        # Iterative normalization: normalize, re-clamp floors, repeat
        for _ in range(3):
            total_w = sum(raw_weights.values())
            if total_w > 0:
                raw_weights = {k: v / total_w for k, v in raw_weights.items()}
            # Re-enforce floor after normalization
            for strat in strategies:
                if raw_weights[strat] < floor:
                    raw_weights[strat] = floor

        w_pairs = raw_weights["pairs_trading"]
        w_mr = raw_weights["mean_reversion"]
        w_mom = raw_weights["momentum_regime"]

        logger.info(
            f"Dynamic allocation: pairs={w_pairs:.0%} mr={w_mr:.0%} mom={w_mom:.0%} "
            f"(sharpes: {', '.join(f'{k}={v:.2f}' for k, v in sharpes.items())})"
        )

        return w_pairs, w_mr, w_mom

    def get_dynamic_weights(self) -> Dict[str, float]:
        """Get current dynamic strategy allocation weights (for monitoring)."""
        w_p, w_m, w_mom = self._compute_dynamic_weights()
        return {"pairs_trading": w_p, "mean_reversion": w_m, "momentum_regime": w_mom}

    # ------------------------------------------------------------------
    # Kalman Spread Tracker Cache (per-pair streaming z-score)
    # ------------------------------------------------------------------

    def _compute_kalman_z_score(
        self,
        pair_id: str,
        pair: CointegrationResult,
        price_a: np.ndarray,
        price_b: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Compute z-score using cached KalmanSpreadTracker for streaming efficiency.

        Maintains persistent Kalman filter state across trading cycles to
        avoid full re-fit each time. Falls back to pair_finder.compute_pair_z_score
        on any error.

        Returns (z_score, hedge_ratio).
        """
        n = min(len(price_a), len(price_b))
        if n < 30:
            return 0.0, pair.hedge_ratio

        try:
            log_a = np.log(np.asarray(price_a[-n:], dtype=float))
            log_b = np.log(np.asarray(price_b[-n:], dtype=float))
        except (ValueError, RuntimeWarning):
            return self.pair_finder.compute_pair_z_score(pair, price_a, price_b)

        tracker = self._kalman_trackers.get(pair_id)

        # Re-bootstrap if tracker is missing or data has grown significantly
        last_n = self._kalman_data_lengths.get(pair_id, 0)
        needs_bootstrap = (
            tracker is None
            or tracker._n_obs < 30
            or (n - last_n) > 10  # Significant new data since last bootstrap
        )

        if needs_bootstrap:
            try:
                tracker = KalmanSpreadTracker(
                    entry_z=self.cfg.pairs_entry_z,
                    exit_z=self.cfg.pairs_exit_z,
                    stop_z=self.cfg.pairs_stop_z,
                )
                # Fit on all but last observation, then stream-update with last
                lookback = min(n - 1, self.cfg.pairs_lookback)
                tracker.fit(log_a[-(lookback + 1):-1], log_b[-(lookback + 1):-1])
                self._kalman_trackers[pair_id] = tracker
                self._kalman_data_lengths[pair_id] = n
            except Exception as e:
                logger.debug(f"Kalman bootstrap failed for {pair_id}: {e}")
                return self.pair_finder.compute_pair_z_score(pair, price_a, price_b)

        try:
            result = tracker.update(float(log_a[-1]), float(log_b[-1]))
            z_score = result.get('z_score', 0.0)
            hedge_ratio = result.get('hedge_ratio', pair.hedge_ratio)

            if not np.isfinite(z_score):
                z_score = 0.0
            if not np.isfinite(hedge_ratio):
                hedge_ratio = pair.hedge_ratio

            logger.debug(
                f"Kalman {pair_id}: z={z_score:+.2f} β={hedge_ratio:.3f} "
                f"HL={result.get('half_life', 'N/A')} n_obs={result.get('n_obs', 0)}"
            )
            return z_score, hedge_ratio
        except Exception as e:
            logger.debug(f"Kalman update failed for {pair_id}: {e}")
            self._kalman_trackers.pop(pair_id, None)
            return self.pair_finder.compute_pair_z_score(pair, price_a, price_b)

    # ------------------------------------------------------------------
    # Main entry point — get all signals
    # ------------------------------------------------------------------

    def get_signals(
        self,
        price_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame] = None,
        equity: float = 100_000.0,
        current_positions: Optional[Dict[str, Any]] = None,
        ohlcv_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> List[TradeSignal]:
        """
        Generate trade signals from all three strategies.

        Parameters
        ----------
        price_data : pd.DataFrame
            Close prices. Columns = symbols, rows = dates.
        volume_data : pd.DataFrame, optional
            Daily volumes. Same shape as price_data.
        equity : float
            Current portfolio equity.
        current_positions : dict, optional
            Currently held positions {symbol: {qty, entry_price, ...}}.
        ohlcv_data : dict, optional
            Dict of {symbol: DataFrame with OHLCV columns} for ATR/ADX/etc.
            If None, we derive what we can from price_data.

        Returns
        -------
        List[TradeSignal]
            All signals across strategies, ready for sizing & validation.
        """
        if current_positions is None:
            current_positions = {}

        all_signals: List[TradeSignal] = []
        timestamp = datetime.now().isoformat()

        # --- 1. Live regime detection (3-state, preferred) ---
        self._regime_adjustments = None
        regime_adj: Optional[Any] = None

        if self._live_regime is not None:
            try:
                ref_sym = "SPY" if "SPY" in price_data.columns else price_data.columns[0]
                spy_prices = price_data[ref_sym].dropna().values.astype(float)
                regime_adj = self._live_regime.predict_regime(spy_prices)
                self._regime_adjustments = regime_adj
                regime = regime_adj.regime.value          # "bull" / "neutral" / "bear"
                regime_conf = regime_adj.confidence
                self._current_regime = regime
                logger.info(f"LiveRegime: {regime_adj.describe()}")
            except Exception as e:
                logger.warning(f"LiveRegimeDetector failed, falling back to HMM: {e}")
                regime_adj = None

        # Fallback to legacy 4-state HMM if live detector unavailable
        if regime_adj is None:
            regime, regime_conf = self._detect_regime(price_data)

        # --- Get adaptive parameter adjustments ---
        adjustments = None
        if self._adaptive_tuner is not None:
            try:
                adjustments = self._adaptive_tuner.get_adjustments(regime)
                if adjustments.skip_next_n_signals > 0:
                    logger.info(f"Adaptive cooldown: skipping {adjustments.skip_next_n_signals} signals")
                    return []
                logger.debug(f"Adaptive adjustments: {adjustments.describe()}")
            except Exception as e:
                logger.debug(f"Adaptive params failed: {e}")

        # Refresh cointegrated pairs if needed
        self._refresh_pairs_if_needed(price_data, volume_data)

        # --- Drawdown tracking ---
        drawdown_pct = self._update_drawdown_state(equity)
        drawdown_scale = self._get_drawdown_scale_factor(drawdown_pct)

        if drawdown_pct >= self.cfg.drawdown_scale_threshold:
            logger.info(
                f"Drawdown: {drawdown_pct:.1%} (peak=${self._peak_equity:,.0f} → "
                f"curr=${equity:,.0f}) scale={drawdown_scale:.2f}"
            )

        # --- Strategy scanning (weights: dynamic allocation + regime blend) ---
        # Start with dynamic allocation based on realized Sharpe per strategy
        w_pairs, w_mr, w_mom = self._compute_dynamic_weights()

        # If regime adjustment available, blend 50/50 with dynamic weights
        if regime_adj is not None:
            r_pairs = regime_adj.strategy_weights.get("pairs", w_pairs)
            r_mr = regime_adj.strategy_weights.get("mr", w_mr)
            r_mom = regime_adj.strategy_weights.get("momentum", w_mom)
            w_pairs = 0.5 * w_pairs + 0.5 * r_pairs
            w_mr = 0.5 * w_mr + 0.5 * r_mr
            w_mom = 0.5 * w_mom + 0.5 * r_mom
            logger.info(
                f"Blended allocations (dynamic+regime): pairs={w_pairs:.0%} "
                f"mr={w_mr:.0%} mom={w_mom:.0%}"
            )

        # --- Strategy A: Pairs Trading ---
        pairs_signals = self._scan_pairs(price_data, equity, current_positions, timestamp)
        # Scale position sizes by regime weight ratio
        for sig in pairs_signals:
            sig.position_size_pct *= (w_pairs / self.cfg.pairs_allocation)
        all_signals.extend(pairs_signals)
        logger.info(f"Pairs trading: {len(pairs_signals)} signals")

        # --- Strategy B: Mean Reversion ---
        mr_signals = self._scan_mean_reversion(
            price_data, volume_data, ohlcv_data, equity, current_positions, timestamp
        )
        for sig in mr_signals:
            sig.position_size_pct *= (w_mr / self.cfg.mr_allocation)
        all_signals.extend(mr_signals)
        logger.info(f"Mean reversion: {len(mr_signals)} signals")

        # --- Strategy C: Momentum ---
        # In BULL regime (high momentum weight), always scan momentum.
        # Otherwise, momentum is a fallback when stat-arb has few signals.
        run_momentum = (w_mom >= 0.30) or (len(pairs_signals) + len(mr_signals) < 2)
        if run_momentum:
            mom_signals = self._scan_momentum(
                price_data, volume_data, ohlcv_data, equity, current_positions, timestamp
            )
            for sig in mom_signals:
                sig.position_size_pct *= (w_mom / self.cfg.momentum_allocation)
            all_signals.extend(mom_signals)
            logger.info(f"Momentum: {len(mom_signals)} signals (w_mom={w_mom:.0%})")
        else:
            mom_signals = []
            logger.info("Momentum skipped — stat-arb strategies have signals")

        # --- Strategy D: VWAP Intraday Mean Reversion (Phase 5b) ---
        vwap_signals: List[TradeSignal] = []
        if self.cfg.vwap_enabled:
            vwap_signals = self._scan_vwap_reversion(
                price_data, volume_data, ohlcv_data, equity, current_positions, timestamp
            )
            all_signals.extend(vwap_signals)
            logger.info(f"VWAP reversion: {len(vwap_signals)} signals")

        # --- Post-processing: apply regime scaling ---
        if regime != "unknown":
            all_signals = [
                self._apply_regime_scaling(s, regime, regime_conf)
                for s in all_signals
            ]

        # --- Post-processing: order flow confirmation ---
        if self._flow_analyzer is not None:
            filtered = []
            for sig in all_signals:
                enriched = self._enrich_with_flow(sig, ohlcv_data)
                if enriched is not None:
                    filtered.append(enriched)
                # CLOSE signals always pass
                elif sig.direction == SignalDirection.CLOSE:
                    filtered.append(sig)
            rejected = len(all_signals) - len(filtered)
            if rejected > 0:
                logger.info(f"Order flow rejected {rejected} signals")
            all_signals = filtered

        # --- Post-processing: adaptive parameter adjustments ---
        if adjustments is not None:
            for sig in all_signals:
                sig.position_size_pct *= adjustments.position_size_mult
                # Clamp to max
                sig.position_size_pct = min(sig.position_size_pct, self.cfg.max_position_pct)
                # Adjust stops via ATR multiplier
                if sig.stop_price > 0 and sig.atr > 0:
                    if sig.direction == SignalDirection.LONG:
                        sig.stop_price = sig.entry_price - sig.atr * self.cfg.mr_atr_stop_mult * adjustments.atr_stop_mult
                    elif sig.direction == SignalDirection.SHORT:
                        sig.stop_price = sig.entry_price + sig.atr * self.cfg.mr_atr_stop_mult * adjustments.atr_stop_mult
                    sig.stop_price = round(sig.stop_price, 2)

        # --- Post-processing: ML ensemble scoring (if fitted) ---
        if self._stacker is not None:
            try:
                if hasattr(self._stacker, '_is_fitted') and self._stacker._is_fitted:
                    for sig in all_signals:
                        features = np.array([[
                            sig.confidence, sig.z_score, sig.rsi, sig.adx,
                            sig.atr, sig.flow_score, sig.regime_confidence,
                        ]])
                        result = self._stacker.predict_single(features)
                        sig.ml_alpha = result.alpha_score
                        # Blend: 60% original confidence + 40% ML alpha
                        sig.confidence = 0.6 * sig.confidence + 0.4 * result.alpha_score
            except Exception as e:
                logger.debug(f"ML stacker scoring failed: {e}")

        # --- Post-processing: sentiment signal boost ---
        if self._sentiment_processor is not None:
            try:
                syms_in_signals = set(s.symbol for s in all_signals if s.direction != SignalDirection.CLOSE)
                if syms_in_signals:
                    sent_batch = self._sentiment_processor.process_batch(list(syms_in_signals))
                    for sig in all_signals:
                        if sig.direction == SignalDirection.CLOSE:
                            continue
                        sent = sent_batch.get(sig.symbol)
                        if sent is None:
                            continue
                        # Boost or penalize confidence based on agreement
                        if sig.direction == SignalDirection.LONG and sent.blended_score > 0:
                            sig.confidence = min(0.95, sig.confidence + sent.confidence_boost)
                        elif sig.direction == SignalDirection.SHORT and sent.blended_score < 0:
                            sig.confidence = min(0.95, sig.confidence + sent.confidence_boost)
                        elif sent.contrarian_flag:
                            # Contrarian: sentiment extreme opposes → reduce
                            sig.confidence *= 0.85
                        logger.debug(f"Sentiment {sig.symbol}: {sent.describe()}")
            except Exception as e:
                logger.debug(f"Sentiment processing failed: {e}")

        # Filter below minimum confidence
        all_signals = [s for s in all_signals if s.confidence >= self.cfg.min_confidence
                       or s.direction == SignalDirection.CLOSE]

        # --- Drawdown-responsive position sizing ---
        if drawdown_scale <= 0:
            # Circuit breaker: block ALL new entries, only allow exits
            entry_count = len([s for s in all_signals if s.direction != SignalDirection.CLOSE])
            all_signals = [s for s in all_signals if s.direction == SignalDirection.CLOSE]
            logger.warning(
                f"DRAWDOWN HALT: {drawdown_pct:.1%} exceeds "
                f"{self.cfg.drawdown_halt_threshold:.0%} threshold — "
                f"blocked {entry_count} new entries"
            )
        elif drawdown_scale < 1.0:
            for sig in all_signals:
                if sig.direction != SignalDirection.CLOSE:
                    sig.position_size_pct *= drawdown_scale
            logger.info(
                f"Drawdown scaling applied: {drawdown_scale:.2f}x to "
                f"{len([s for s in all_signals if s.direction != SignalDirection.CLOSE])} entries"
            )

        # Sort by confidence (highest first)
        all_signals.sort(key=lambda s: s.confidence, reverse=True)

        logger.info(
            f"Strategy engine: {len(all_signals)} total signals "
            f"(pairs={len(pairs_signals)}, mr={len(mr_signals)}, "
            f"mom={len(all_signals) - len(pairs_signals) - len(mr_signals)}) "
            f"regime={regime}"
        )

        return all_signals

    # ------------------------------------------------------------------
    # Strategy A: Pairs Trading
    # ------------------------------------------------------------------

    def _refresh_pairs_if_needed(
        self,
        price_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame],
    ):
        """Refresh cointegrated pairs monthly (or on first call)."""
        now = datetime.now()
        if (self._last_pair_refresh is None or
                (now - self._last_pair_refresh).days >= self.cfg.pairs_rebalance_days):
            logger.info("Refreshing cointegrated pairs...")
            self._pairs = self.pair_finder.find_pairs(
                price_data, volume_data, force_refresh=True
            )
            self._last_pair_refresh = now
            logger.info(f"Found {len(self._pairs)} cointegrated pairs")

    def _scan_pairs(
        self,
        price_data: pd.DataFrame,
        equity: float,
        positions: Dict[str, Any],
        timestamp: str,
    ) -> List[TradeSignal]:
        """
        Scan cointegrated pairs for entry/exit signals.

        Entry logic:
          - z-score > +2.0  ->  SHORT A, LONG B  (A outperformed, expect reversion)
          - z-score < -2.0  ->  LONG A, SHORT B   (B outperformed, expect reversion)

        Exit logic:
          - |z-score| < 0.5  ->  close both legs (spread reverted to mean)

        Stop logic:
          - |z-score| > 4.0  ->  close both legs (cointegration may be breaking)
        """
        signals: List[TradeSignal] = []
        active_pair_count = sum(
            1 for s in positions
            if any(s in [p.sym_a, p.sym_b] for p in self._pairs)
        )

        for pair in self._pairs:
            if pair.sym_a not in price_data.columns or pair.sym_b not in price_data.columns:
                continue

            pa = price_data[pair.sym_a].dropna().values.astype(float)
            pb = price_data[pair.sym_b].dropna().values.astype(float)

            if len(pa) < self.cfg.pairs_lookback or len(pb) < self.cfg.pairs_lookback:
                continue

            pair_id = f"{pair.sym_a}_{pair.sym_b}"

            # Compute z-score via cached Kalman spread tracker (streaming, adaptive)
            z_score, hedge_ratio = self._compute_kalman_z_score(pair_id, pair, pa, pb)

            current_price_a = float(pa[-1])
            current_price_b = float(pb[-1])

            # --- EXIT signals for active pairs ---
            a_held = pair.sym_a in positions
            b_held = pair.sym_b in positions

            if a_held or b_held:
                # Exit if spread reverted to mean
                if abs(z_score) < self.cfg.pairs_exit_z:
                    if a_held:
                        signals.append(TradeSignal(
                            symbol=pair.sym_a,
                            direction=SignalDirection.CLOSE,
                            strategy=StrategyType.PAIRS,
                            confidence=0.9,
                            position_size_pct=0,
                            entry_price=current_price_a,
                            strategy_source=f"Pair exit: {pair_id} z={z_score:+.2f} reverted to mean",
                            z_score=z_score,
                            hedge_ratio=hedge_ratio,
                            pair_symbol=pair.sym_b,
                            pair_id=pair_id,
                            timestamp=timestamp,
                        ))
                    if b_held:
                        signals.append(TradeSignal(
                            symbol=pair.sym_b,
                            direction=SignalDirection.CLOSE,
                            strategy=StrategyType.PAIRS,
                            confidence=0.9,
                            position_size_pct=0,
                            entry_price=current_price_b,
                            strategy_source=f"Pair exit: {pair_id} z={z_score:+.2f} reverted to mean",
                            z_score=z_score,
                            hedge_ratio=hedge_ratio,
                            pair_symbol=pair.sym_a,
                            pair_id=pair_id,
                            timestamp=timestamp,
                        ))
                    continue

                # Stop if spread diverging too much (cointegration breaking)
                if abs(z_score) > self.cfg.pairs_stop_z:
                    if a_held:
                        signals.append(TradeSignal(
                            symbol=pair.sym_a,
                            direction=SignalDirection.CLOSE,
                            strategy=StrategyType.PAIRS,
                            confidence=0.95,
                            position_size_pct=0,
                            entry_price=current_price_a,
                            strategy_source=f"Pair STOP: {pair_id} z={z_score:+.2f} > {self.cfg.pairs_stop_z} (breakdown)",
                            z_score=z_score,
                            pair_symbol=pair.sym_b,
                            pair_id=pair_id,
                            timestamp=timestamp,
                        ))
                    if b_held:
                        signals.append(TradeSignal(
                            symbol=pair.sym_b,
                            direction=SignalDirection.CLOSE,
                            strategy=StrategyType.PAIRS,
                            confidence=0.95,
                            position_size_pct=0,
                            entry_price=current_price_b,
                            strategy_source=f"Pair STOP: {pair_id} z={z_score:+.2f} (breakdown)",
                            z_score=z_score,
                            pair_symbol=pair.sym_a,
                            pair_id=pair_id,
                            timestamp=timestamp,
                        ))
                    continue

            # --- ENTRY signals for new pairs ---
            if active_pair_count >= self.cfg.pairs_max_positions:
                continue  # At capacity

            if a_held or b_held:
                continue  # Already in this pair

            # Confidence based on how far from entry threshold
            # z=2.0 -> conf=0.50, z=3.0 -> conf=0.75, z=4.0 -> stop
            confidence = min(0.95, 0.50 + (abs(z_score) - self.cfg.pairs_entry_z) * 0.25)
            confidence = max(self.cfg.min_confidence, confidence)

            if z_score > self.cfg.pairs_entry_z:
                # A is overpriced relative to B -> SHORT A, LONG B
                signals.append(TradeSignal(
                    symbol=pair.sym_a,
                    direction=SignalDirection.SHORT,
                    strategy=StrategyType.PAIRS,
                    confidence=confidence,
                    position_size_pct=self.cfg.pairs_position_pct,
                    entry_price=current_price_a,
                    stop_price=0,  # Stops managed by z-score, not price
                    target_price=0,
                    strategy_source=(
                        f"Pair SHORT: {pair_id} z={z_score:+.2f} "
                        f"HL={pair.half_life:.0f}d beta={hedge_ratio:.3f}"
                    ),
                    z_score=z_score,
                    hedge_ratio=hedge_ratio,
                    pair_symbol=pair.sym_b,
                    half_life=pair.half_life,
                    pair_id=pair_id,
                    timestamp=timestamp,
                ))
                signals.append(TradeSignal(
                    symbol=pair.sym_b,
                    direction=SignalDirection.LONG,
                    strategy=StrategyType.PAIRS,
                    confidence=confidence,
                    position_size_pct=self.cfg.pairs_position_pct,
                    entry_price=current_price_b,
                    stop_price=0,
                    target_price=0,
                    strategy_source=(
                        f"Pair LONG: {pair_id} z={z_score:+.2f} "
                        f"HL={pair.half_life:.0f}d beta={hedge_ratio:.3f}"
                    ),
                    z_score=z_score,
                    hedge_ratio=hedge_ratio,
                    pair_symbol=pair.sym_a,
                    half_life=pair.half_life,
                    pair_id=pair_id,
                    timestamp=timestamp,
                ))
                active_pair_count += 1

            elif z_score < -self.cfg.pairs_entry_z:
                # B is overpriced relative to A -> LONG A, SHORT B
                signals.append(TradeSignal(
                    symbol=pair.sym_a,
                    direction=SignalDirection.LONG,
                    strategy=StrategyType.PAIRS,
                    confidence=confidence,
                    position_size_pct=self.cfg.pairs_position_pct,
                    entry_price=current_price_a,
                    stop_price=0,
                    target_price=0,
                    strategy_source=(
                        f"Pair LONG: {pair_id} z={z_score:+.2f} "
                        f"HL={pair.half_life:.0f}d beta={hedge_ratio:.3f}"
                    ),
                    z_score=z_score,
                    hedge_ratio=hedge_ratio,
                    pair_symbol=pair.sym_b,
                    half_life=pair.half_life,
                    pair_id=pair_id,
                    timestamp=timestamp,
                ))
                signals.append(TradeSignal(
                    symbol=pair.sym_b,
                    direction=SignalDirection.SHORT,
                    strategy=StrategyType.PAIRS,
                    confidence=confidence,
                    position_size_pct=self.cfg.pairs_position_pct,
                    entry_price=current_price_b,
                    stop_price=0,
                    target_price=0,
                    strategy_source=(
                        f"Pair SHORT: {pair_id} z={z_score:+.2f} "
                        f"HL={pair.half_life:.0f}d beta={hedge_ratio:.3f}"
                    ),
                    z_score=z_score,
                    hedge_ratio=hedge_ratio,
                    pair_symbol=pair.sym_a,
                    half_life=pair.half_life,
                    pair_id=pair_id,
                    timestamp=timestamp,
                ))
                active_pair_count += 1

        return signals

    # ------------------------------------------------------------------
    # Strategy B: Mean Reversion (Bollinger + RSI + Volume)
    # ------------------------------------------------------------------

    def _scan_mean_reversion(
        self,
        price_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame],
        ohlcv_data: Optional[Dict[str, pd.DataFrame]],
        equity: float,
        positions: Dict[str, Any],
        timestamp: str,
    ) -> List[TradeSignal]:
        """
        Mean reversion for liquid large-caps.

        LONG when:
          - Price below lower Bollinger Band (2.0 std, 20-period)
          - RSI(14) < 30 (oversold)
          - Volume > 1.5x 20-day average (capitulation spike)

        SHORT when:
          - Price above upper Bollinger Band
          - RSI(14) > 70 (overbought)
          - Volume spike

        Target: 20-day SMA (mean)
        Stop: 1.5x ATR(14) from entry
        Max hold: 5 days
        """
        signals: List[TradeSignal] = []
        mr_count = sum(
            1 for s, info in positions.items()
            if isinstance(info, dict) and info.get("strategy") == "mean_reversion"
        )

        # Only scan symbols we have enough data for
        for sym in price_data.columns:
            if sym in positions:
                continue
            if mr_count >= self.cfg.mr_max_positions:
                break

            prices = price_data[sym].dropna().values.astype(float)
            if len(prices) < self.cfg.mr_bb_period + 5:
                continue

            current_price = float(prices[-1])
            if current_price < 10:
                continue  # Skip penny stocks

            # Phase 3: banned symbol check
            try:
                from config.universe import BANNED_SYMBOLS
                if sym in BANNED_SYMBOLS:
                    continue
            except ImportError:
                pass

            # Phase 3: freefall filter — skip if down >8% in last 5 bars
            if len(prices) >= 6:
                five_bar_ret = (prices[-1] / prices[-6]) - 1.0
                if five_bar_ret < -0.08:
                    continue

            # Phase 3: trend filter — skip LONG if SMA50 < SMA200 (downtrend)
            if len(prices) >= 200:
                sma50 = float(np.mean(prices[-50:]))
                sma200 = float(np.mean(prices[-200:]))
                if sma50 < sma200:
                    continue  # death cross — skip MR longs in downtrend

            # Bollinger Bands
            upper_bb, middle_bb, lower_bb = compute_bollinger_bands(
                prices, self.cfg.mr_bb_period, self.cfg.mr_bb_std
            )

            # RSI
            rsi = compute_rsi(prices, self.cfg.mr_rsi_period)

            # Volume check
            vol_ratio = 1.0
            if volume_data is not None and sym in volume_data.columns:
                vols = volume_data[sym].dropna().values.astype(float)
                if len(vols) > 20:
                    vol_ratio = compute_volume_ratio(vols, 20)

            # ATR for stop placement
            atr = 0.0
            if ohlcv_data and sym in ohlcv_data:
                df = ohlcv_data[sym]
                if all(c in df.columns for c in ['high', 'low', 'close']):
                    atr = compute_atr(
                        df['high'].values, df['low'].values, df['close'].values,
                        self.cfg.mr_rsi_period,
                    )
            if atr < 1e-6:
                # Fallback: estimate ATR from close prices
                if len(prices) >= 15:
                    daily_ranges = np.abs(np.diff(prices[-15:]))
                    atr = float(np.mean(daily_ranges))

            # === LONG SIGNAL: oversold ===
            if (current_price < lower_bb and
                    rsi < self.cfg.mr_rsi_oversold and
                    vol_ratio >= self.cfg.mr_volume_spike):

                # GARCH-enhanced stop: use max of ATR-based and GARCH-based
                atr_stop_dist = self.cfg.mr_atr_stop_mult * atr
                garch_vol = self._get_garch_vol(prices)
                if garch_vol > 0:
                    garch_stop_dist = current_price * garch_vol / np.sqrt(252) * 2.0
                    stop_dist = max(atr_stop_dist, garch_stop_dist)
                else:
                    stop_dist = atr_stop_dist
                stop_price = round(current_price - stop_dist, 2)
                # Target: 20-day SMA (the mean we're reverting to)
                target_price = round(middle_bb, 2)

                # Confidence: stronger when RSI is more extreme + price further from band
                bb_distance = (lower_bb - current_price) / (upper_bb - lower_bb) if (upper_bb - lower_bb) > 0 else 0
                rsi_extreme = (self.cfg.mr_rsi_oversold - rsi) / self.cfg.mr_rsi_oversold
                confidence = min(0.95, 0.55 + 0.2 * bb_distance + 0.2 * rsi_extreme)
                confidence = max(self.cfg.min_confidence, confidence)

                # Position size based on risk (stop distance)
                risk_per_share = current_price - stop_price
                if risk_per_share > 0:
                    # Risk 1% of equity per trade
                    risk_dollars = equity * 0.01
                    shares = int(risk_dollars / risk_per_share)
                    size_pct = min(
                        self.cfg.max_position_pct,
                        (shares * current_price) / equity if equity > 0 else 0,
                    )
                else:
                    size_pct = self.cfg.max_position_pct * 0.5

                signals.append(TradeSignal(
                    symbol=sym,
                    direction=SignalDirection.LONG,
                    strategy=StrategyType.MEAN_REVERSION,
                    confidence=confidence,
                    position_size_pct=size_pct,
                    entry_price=current_price,
                    stop_price=stop_price,
                    target_price=target_price,
                    strategy_source=(
                        f"MR LONG: price ${current_price:.2f} < BB_low ${lower_bb:.2f}, "
                        f"RSI={rsi:.0f}, vol={vol_ratio:.1f}x"
                    ),
                    z_score=(current_price - middle_bb) / (upper_bb - lower_bb) * 2 if (upper_bb - lower_bb) > 0 else 0,
                    atr=atr,
                    rsi=rsi,
                    timestamp=timestamp,
                    max_hold_days=self.cfg.mr_max_hold_days,
                ))
                mr_count += 1

            # === SHORT SIGNAL: overbought ===
            elif (current_price > upper_bb and
                    rsi > self.cfg.mr_rsi_overbought and
                    vol_ratio >= self.cfg.mr_volume_spike):

                # GARCH-enhanced stop for SHORT
                atr_stop_dist = self.cfg.mr_atr_stop_mult * atr
                garch_vol = self._get_garch_vol(prices)
                if garch_vol > 0:
                    garch_stop_dist = current_price * garch_vol / np.sqrt(252) * 2.0
                    stop_dist = max(atr_stop_dist, garch_stop_dist)
                else:
                    stop_dist = atr_stop_dist
                stop_price = round(current_price + stop_dist, 2)
                target_price = round(middle_bb, 2)

                bb_distance = (current_price - upper_bb) / (upper_bb - lower_bb) if (upper_bb - lower_bb) > 0 else 0
                rsi_extreme = (rsi - self.cfg.mr_rsi_overbought) / (100 - self.cfg.mr_rsi_overbought)
                confidence = min(0.95, 0.55 + 0.2 * bb_distance + 0.2 * rsi_extreme)
                confidence = max(self.cfg.min_confidence, confidence)

                risk_per_share = stop_price - current_price
                if risk_per_share > 0:
                    risk_dollars = equity * 0.01
                    shares = int(risk_dollars / risk_per_share)
                    size_pct = min(
                        self.cfg.max_position_pct,
                        (shares * current_price) / equity if equity > 0 else 0,
                    )
                else:
                    size_pct = self.cfg.max_position_pct * 0.5

                signals.append(TradeSignal(
                    symbol=sym,
                    direction=SignalDirection.SHORT,
                    strategy=StrategyType.MEAN_REVERSION,
                    confidence=confidence,
                    position_size_pct=size_pct,
                    entry_price=current_price,
                    stop_price=stop_price,
                    target_price=target_price,
                    strategy_source=(
                        f"MR SHORT: price ${current_price:.2f} > BB_high ${upper_bb:.2f}, "
                        f"RSI={rsi:.0f}, vol={vol_ratio:.1f}x"
                    ),
                    z_score=(current_price - middle_bb) / (upper_bb - lower_bb) * 2 if (upper_bb - lower_bb) > 0 else 0,
                    atr=atr,
                    rsi=rsi,
                    timestamp=timestamp,
                    max_hold_days=self.cfg.mr_max_hold_days,
                ))
                mr_count += 1

        return signals

    # ------------------------------------------------------------------
    # Strategy D: VWAP Intraday Mean Reversion  (Phase 5b)
    # ------------------------------------------------------------------

    def _scan_vwap_reversion(
        self,
        price_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame],
        ohlcv_data: Optional[Dict[str, pd.DataFrame]],
        equity: float,
        positions: Dict[str, Any],
        timestamp: str,
    ) -> List[TradeSignal]:
        """
        VWAP mean-reversion strategy.

        Uses Volume-Weighted Average Price as the "fair value" anchor:

        LONG when:
          • Price < VWAP − ``vwap_entry_std`` × std
          • RSI(14) < ``vwap_rsi_oversold``

        SHORT when:
          • Price > VWAP + ``vwap_entry_std`` × std
          • RSI(14) > ``vwap_rsi_overbought``

        Target: revert to VWAP
        Stop: 2× entry distance from VWAP (configurable)
        Max hold: 3 days (short-duration reversion)
        """
        signals: List[TradeSignal] = []
        vwap_count = sum(
            1 for s, info in positions.items()
            if isinstance(info, dict) and info.get("strategy") == "vwap_reversion"
        )

        for sym in price_data.columns:
            if sym in positions:
                continue
            if vwap_count >= self.cfg.vwap_max_positions:
                break

            prices = price_data[sym].dropna().values.astype(float)
            lookback = self.cfg.vwap_lookback
            if len(prices) < lookback + 5:
                continue

            current_price = float(prices[-1])
            if current_price < 10:
                continue

            # --- Compute VWAP ---
            window_prices = prices[-lookback:]
            if volume_data is not None and sym in volume_data.columns:
                vols = volume_data[sym].dropna().values.astype(float)
                if len(vols) >= lookback:
                    window_vols = vols[-lookback:]
                    total_vol = window_vols.sum()
                    if total_vol > 0:
                        vwap = float(np.sum(window_prices * window_vols) / total_vol)
                    else:
                        vwap = float(np.mean(window_prices))
                else:
                    vwap = float(np.mean(window_prices))
            else:
                # Fallback: equal-weight (simple mean)
                vwap = float(np.mean(window_prices))

            # Standard deviation of price around VWAP
            deviations = window_prices - vwap
            vwap_std = float(np.std(deviations))
            if vwap_std < 1e-6:
                continue

            distance_from_vwap = current_price - vwap
            z_vwap = distance_from_vwap / vwap_std

            # RSI confirmation
            rsi = compute_rsi(prices, 14)

            # ATR for sizing
            atr = 0.0
            if ohlcv_data and sym in ohlcv_data:
                df = ohlcv_data[sym]
                if all(c in df.columns for c in ['high', 'low', 'close']):
                    atr = compute_atr(df['high'].values, df['low'].values, df['close'].values, 14)
            if atr < 1e-6 and len(prices) >= 15:
                atr = float(np.mean(np.abs(np.diff(prices[-15:]))))

            entry_threshold = self.cfg.vwap_entry_std

            # === LONG: price significantly below VWAP ===
            if z_vwap < -entry_threshold and rsi < self.cfg.vwap_rsi_oversold:
                entry_dist = abs(distance_from_vwap)
                stop_price = round(current_price - self.cfg.vwap_stop_mult * entry_dist, 2)
                target_price = round(vwap, 2)

                # Confidence: stronger at more extreme deviations
                conf = min(0.95, 0.50 + 0.15 * (abs(z_vwap) - entry_threshold))
                conf = max(self.cfg.min_confidence, conf)

                # Risk-based sizing
                risk_per_share = current_price - stop_price
                if risk_per_share > 0:
                    risk_dollars = equity * 0.01
                    shares = int(risk_dollars / risk_per_share)
                    size_pct = min(self.cfg.max_position_pct,
                                   (shares * current_price) / equity if equity > 0 else 0)
                else:
                    size_pct = self.cfg.max_position_pct * 0.3

                signals.append(TradeSignal(
                    symbol=sym,
                    direction=SignalDirection.LONG,
                    strategy=StrategyType.VWAP_REVERSION,
                    confidence=conf,
                    position_size_pct=size_pct,
                    entry_price=current_price,
                    stop_price=stop_price,
                    target_price=target_price,
                    strategy_source=(
                        f"VWAP LONG: ${current_price:.2f} < VWAP ${vwap:.2f} "
                        f"(z={z_vwap:.2f}), RSI={rsi:.0f}"
                    ),
                    z_score=z_vwap,
                    atr=atr,
                    rsi=rsi,
                    timestamp=timestamp,
                    max_hold_days=self.cfg.vwap_max_hold_days,
                ))
                vwap_count += 1

            # === SHORT: price significantly above VWAP ===
            elif z_vwap > entry_threshold and rsi > self.cfg.vwap_rsi_overbought:
                entry_dist = abs(distance_from_vwap)
                stop_price = round(current_price + self.cfg.vwap_stop_mult * entry_dist, 2)
                target_price = round(vwap, 2)

                conf = min(0.95, 0.50 + 0.15 * (abs(z_vwap) - entry_threshold))
                conf = max(self.cfg.min_confidence, conf)

                risk_per_share = stop_price - current_price
                if risk_per_share > 0:
                    risk_dollars = equity * 0.01
                    shares = int(risk_dollars / risk_per_share)
                    size_pct = min(self.cfg.max_position_pct,
                                   (shares * current_price) / equity if equity > 0 else 0)
                else:
                    size_pct = self.cfg.max_position_pct * 0.3

                signals.append(TradeSignal(
                    symbol=sym,
                    direction=SignalDirection.SHORT,
                    strategy=StrategyType.VWAP_REVERSION,
                    confidence=conf,
                    position_size_pct=size_pct,
                    entry_price=current_price,
                    stop_price=stop_price,
                    target_price=target_price,
                    strategy_source=(
                        f"VWAP SHORT: ${current_price:.2f} > VWAP ${vwap:.2f} "
                        f"(z={z_vwap:.2f}), RSI={rsi:.0f}"
                    ),
                    z_score=z_vwap,
                    atr=atr,
                    rsi=rsi,
                    timestamp=timestamp,
                    max_hold_days=self.cfg.vwap_max_hold_days,
                ))
                vwap_count += 1

        return signals

    # ------------------------------------------------------------------
    # Strategy C: Momentum with Regime Filter
    # ------------------------------------------------------------------

    def _scan_momentum(
        self,
        price_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame],
        ohlcv_data: Optional[Dict[str, pd.DataFrame]],
        equity: float,
        positions: Dict[str, Any],
        timestamp: str,
    ) -> List[TradeSignal]:
        """
        Trend-following with strict regime filter.

        Rules:
          1. 200-day SMA is the regime filter:
             - Price above 200-SMA -> only LONG
             - Price below 200-SMA -> only SHORT
          2. Enter on pullback to 20-day EMA (buy the dip in a trend)
          3. ADX > 25 confirms the trend is real (not choppy)
          4. 2x ATR trailing stop
          5. Scale in: 50% initial, add 50% on confirmation

        This is ONLY activated when pairs + MR have no signals (fallback).
        """
        signals: List[TradeSignal] = []
        mom_count = sum(
            1 for s, info in positions.items()
            if isinstance(info, dict) and info.get("strategy") == "momentum_regime"
        )

        for sym in price_data.columns:
            if sym in positions:
                continue
            if mom_count >= self.cfg.mom_max_positions:
                break

            prices = price_data[sym].dropna().values.astype(float)
            if len(prices) < self.cfg.min_bars_required:
                continue

            current_price = float(prices[-1])
            if current_price < 10:
                continue

            # Phase 3: banned symbol check
            try:
                from config.universe import BANNED_SYMBOLS
                if sym in BANNED_SYMBOLS:
                    continue
            except ImportError:
                pass

            # Phase 3: freefall filter — skip if down >8% in last 5 bars
            if len(prices) >= 6:
                five_bar_ret = (prices[-1] / prices[-6]) - 1.0
                if five_bar_ret < -0.08:
                    continue

            # 200-day SMA — regime filter (enhanced by HMM when available)
            sma_200 = compute_sma(prices, self.cfg.mom_sma_period)

            # HMM-enhanced regime: if HMM detected regime, use it;
            # otherwise fall back to price-vs-SMA200.
            hmm_is_bullish = self._current_regime == "trending_bull"
            hmm_is_bearish = self._current_regime == "trending_bear"
            hmm_skip_momentum = self._current_regime in ("mean_reverting", "high_volatility")

            if self._hmm is not None and self._current_regime != "unknown":
                # HMM regime overrides SMA filter
                is_bullish = hmm_is_bullish
                is_bearish = hmm_is_bearish
                if hmm_skip_momentum:
                    continue  # Mean-reverting/high-vol → skip momentum
            else:
                # Fallback: original SMA regime filter
                is_bullish = current_price > sma_200
                is_bearish = current_price < sma_200

            # 20-day EMA — pullback level
            ema_20 = compute_ema(prices, self.cfg.mom_ema_period)

            # ADX — trend strength
            adx = 0.0
            if ohlcv_data and sym in ohlcv_data:
                df = ohlcv_data[sym]
                if all(c in df.columns for c in ['high', 'low', 'close']):
                    adx = compute_adx(
                        df['high'].values, df['low'].values, df['close'].values,
                    )
            elif len(prices) >= 30:
                # Rough ADX proxy from closes
                recent = prices[-30:]
                up_moves = np.sum(np.diff(recent) > 0)
                trend_ratio = up_moves / len(np.diff(recent))
                adx = abs(trend_ratio - 0.5) * 100  # Rough proxy

            # ATR for stops
            atr = 0.0
            if ohlcv_data and sym in ohlcv_data:
                df = ohlcv_data[sym]
                if all(c in df.columns for c in ['high', 'low', 'close']):
                    atr = compute_atr(
                        df['high'].values, df['low'].values, df['close'].values,
                    )
            if atr < 1e-6 and len(prices) >= 15:
                daily_ranges = np.abs(np.diff(prices[-15:]))
                atr = float(np.mean(daily_ranges))

            # === LONG: bullish regime, pulling back to 20-EMA, strong trend ===
            if (is_bullish and                           # Bullish regime (HMM or SMA)
                    adx > self.cfg.mom_adx_threshold and    # Confirmed trend
                    current_price <= ema_20 * 1.02 and      # At or below 20-EMA (pullback)
                    current_price >= ema_20 * 0.96):        # Not too far below (still in trend)

                # GARCH-enhanced stop: use max of ATR-based and GARCH-based distance
                atr_stop_dist = self.cfg.mom_atr_trail_mult * atr
                garch_vol = self._get_garch_vol(prices)
                if garch_vol > 0:
                    # 2-sigma daily move from GARCH
                    garch_stop_dist = current_price * garch_vol / np.sqrt(252) * 2.0
                    stop_dist = max(atr_stop_dist, garch_stop_dist)
                else:
                    stop_dist = atr_stop_dist
                stop_price = round(current_price - stop_dist, 2)

                # Target: project the trend forward
                trend_strength = (current_price - sma_200) / sma_200
                target_price = round(current_price * (1.0 + trend_strength), 2)

                # Scale-in: initial position is 50% of allocation
                initial_pct = self.cfg.max_position_pct * (0.5 if self.cfg.mom_scale_in else 1.0)

                # Confidence from ADX strength
                confidence = min(0.90, 0.50 + (adx - self.cfg.mom_adx_threshold) * 0.01)
                confidence = max(self.cfg.min_confidence, confidence)

                signals.append(TradeSignal(
                    symbol=sym,
                    direction=SignalDirection.LONG,
                    strategy=StrategyType.MOMENTUM,
                    confidence=confidence,
                    position_size_pct=initial_pct,
                    entry_price=current_price,
                    stop_price=stop_price,
                    target_price=target_price,
                    strategy_source=(
                        f"MOM LONG: pullback to EMA20 ${ema_20:.2f}, "
                        f"regime={'HMM:'+self._current_regime if self._hmm else 'SMA200:'+str(round(sma_200,2))}, "
                        f"ADX={adx:.0f}, trend={trend_strength:+.1%}"
                    ),
                    garch_vol=garch_vol if garch_vol > 0 else None,
                    atr=atr,
                    adx=adx,
                    timestamp=timestamp,
                ))
                mom_count += 1

            # === SHORT: bearish regime, bouncing to 20-EMA, strong downtrend ===
            elif (is_bearish and
                    adx > self.cfg.mom_adx_threshold and
                    current_price >= ema_20 * 0.98 and
                    current_price <= ema_20 * 1.04):

                # GARCH-enhanced stop for SHORT
                atr_stop_dist = self.cfg.mom_atr_trail_mult * atr
                garch_vol = self._get_garch_vol(prices)
                if garch_vol > 0:
                    garch_stop_dist = current_price * garch_vol / np.sqrt(252) * 2.0
                    stop_dist = max(atr_stop_dist, garch_stop_dist)
                else:
                    stop_dist = atr_stop_dist
                stop_price = round(current_price + stop_dist, 2)
                trend_strength = (sma_200 - current_price) / sma_200
                target_price = round(current_price * (1.0 - trend_strength), 2)

                initial_pct = self.cfg.max_position_pct * (0.5 if self.cfg.mom_scale_in else 1.0)
                confidence = min(0.90, 0.50 + (adx - self.cfg.mom_adx_threshold) * 0.01)
                confidence = max(self.cfg.min_confidence, confidence)

                signals.append(TradeSignal(
                    symbol=sym,
                    direction=SignalDirection.SHORT,
                    strategy=StrategyType.MOMENTUM,
                    confidence=confidence,
                    position_size_pct=initial_pct,
                    entry_price=current_price,
                    stop_price=stop_price,
                    target_price=target_price,
                    strategy_source=(
                        f"MOM SHORT: bounce to EMA20 ${ema_20:.2f}, "
                        f"regime={'HMM:'+self._current_regime if self._hmm else 'SMA200:'+str(round(sma_200,2))}, "
                        f"ADX={adx:.0f}, trend={trend_strength:+.1%}"
                    ),
                    garch_vol=garch_vol if garch_vol > 0 else None,
                    atr=atr,
                    adx=adx,
                    timestamp=timestamp,
                ))
                mom_count += 1

        return signals

    # ------------------------------------------------------------------
    # Strategy performance tracking
    # ------------------------------------------------------------------

    def record_trade_result(
        self,
        strategy: str,
        pnl: float,
        *,
        symbol: str = "",
        entry_time: Optional[datetime] = None,
        exit_time: Optional[datetime] = None,
        pnl_pct: float = 0.0,
        holding_bars: int = 0,
        regime: str = "",
        composite_score: float = 0.0,
        ml_confidence: float = 0.0,
        atr_pct: float = 0.0,
        stop_distance_pct: float = 0.0,
        exit_reason: str = "manual",
    ):
        """Record a completed trade for strategy performance tracking."""
        if strategy in self._strategy_stats:
            stats = self._strategy_stats[strategy]
            stats["total_pnl"] += pnl
            if pnl > 0:
                stats["wins"] += 1
            else:
                stats["losses"] += 1

        # Track PnL for dynamic allocation (rolling Sharpe estimation)
        if strategy in self._trade_pnls:
            self._trade_pnls[strategy].append(pnl)
            # Keep bounded to avoid unbounded memory growth
            max_history = self.cfg.dynamic_alloc_lookback * 3
            if len(self._trade_pnls[strategy]) > max_history:
                self._trade_pnls[strategy] = self._trade_pnls[strategy][-max_history:]

        # Feed adaptive parameter tuner (if available)
        if self._adaptive_tuner is not None and symbol:
            try:
                from src.adaptive_parameters import TradeRecord as _TR
                now = datetime.now()
                record = _TR(
                    symbol=symbol,
                    entry_time=entry_time or now,
                    exit_time=exit_time or now,
                    pnl_pct=pnl_pct if pnl_pct else (pnl / 1.0),
                    holding_bars=holding_bars,
                    regime=regime or self._current_regime,
                    composite_score=composite_score,
                    ml_confidence=ml_confidence,
                    atr_pct=atr_pct,
                    stop_distance_pct=stop_distance_pct,
                    exit_reason=exit_reason,
                )
                self._adaptive_tuner.record_trade(record)
            except Exception:
                pass  # adaptive tuner is best-effort

    def get_strategy_stats(self) -> Dict[str, dict]:
        """Get win rate and P&L stats per strategy."""
        result = {}
        for strategy, stats in self._strategy_stats.items():
            total = stats["wins"] + stats["losses"]
            result[strategy] = {
                "wins": stats["wins"],
                "losses": stats["losses"],
                "total": total,
                "win_rate": stats["wins"] / total if total > 0 else 0.0,
                "total_pnl": stats["total_pnl"],
            }
        return result

    def get_strategy_sharpe(self, strategy: str) -> float:
        """
        Estimate realized Sharpe for a strategy.
        Used by portfolio_allocator for dynamic reweighting.
        """
        stats = self._strategy_stats.get(strategy, {})
        total = stats.get("wins", 0) + stats.get("losses", 0)
        if total < 10:
            return 0.0  # Not enough data
        win_rate = stats["wins"] / total
        # Rough Sharpe proxy: (win_rate - 0.5) * sqrt(trades)
        return (win_rate - 0.5) * np.sqrt(total)


# ============================================================================
# MAIN — Standalone test
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    print("=" * 70)
    print("STRATEGY ENGINE — Market-Neutral Signal Generator")
    print("=" * 70)

    # Generate synthetic test data
    np.random.seed(42)
    n_days = 300
    dates = pd.date_range("2025-04-01", periods=n_days, freq="B")

    symbols = ["AAPL", "MSFT", "GOOGL", "META", "NVDA",
               "JPM", "GS", "MS", "BAC",
               "XOM", "CVX", "COP"]

    price_data = pd.DataFrame(index=dates)
    volume_data = pd.DataFrame(index=dates)

    for i, sym in enumerate(symbols):
        base = 100 + i * 20
        trend = np.cumsum(np.random.randn(n_days) * 0.5)
        sector_factor = np.cumsum(np.random.randn(n_days) * 0.3)
        price_data[sym] = base + trend + sector_factor + np.random.randn(n_days) * 0.5
        price_data[sym] = price_data[sym].clip(lower=10)
        volume_data[sym] = np.random.randint(500_000, 5_000_000, n_days)

    engine = StrategyEngine(EngineConfig())
    signals = engine.get_signals(price_data, volume_data, equity=100_000)

    print(f"\nGenerated {len(signals)} signals:\n")
    for sig in signals:
        print(
            f"  {sig.direction.value:>5} {sig.symbol:<6} "
            f"via {sig.strategy.value:<20} "
            f"conf={sig.confidence:.2f} size={sig.position_size_pct:.1%} "
            f"| {sig.strategy_source}"
        )
