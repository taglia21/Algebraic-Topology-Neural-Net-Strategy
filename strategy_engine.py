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
    SECTOR_UNIVERSE,
    ALL_SYMBOLS,
)

logger = logging.getLogger("strategy_engine")


# ============================================================================
# ENUMS & DATA CLASSES
# ============================================================================

class StrategyType(Enum):
    PAIRS = "pairs_trading"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum_regime"


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
    mr_bb_std: float = 2.0              # 2.0 standard deviations (captures 95% of moves)
    mr_rsi_period: int = 14             # 14-period RSI (standard)
    mr_rsi_oversold: float = 30.0       # RSI < 30 = oversold (buy)
    mr_rsi_overbought: float = 70.0     # RSI > 70 = overbought (sell)
    mr_volume_spike: float = 1.5        # Volume must be 1.5x 20-day avg (confirms capitulation)
    mr_atr_stop_mult: float = 1.5       # Stop at 1.5x ATR from entry (tight for MR)
    mr_max_hold_days: int = 5           # Max 5 day hold (MR is fast or it's wrong)
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

    # --- Data ---
    min_bars_required: int = 250        # Need at least 250 bars for 200-SMA


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
        }

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

        # Refresh cointegrated pairs if needed
        self._refresh_pairs_if_needed(price_data, volume_data)

        # --- Strategy A: Pairs Trading (50% allocation) ---
        pairs_signals = self._scan_pairs(price_data, equity, current_positions, timestamp)
        all_signals.extend(pairs_signals)
        logger.info(f"Pairs trading: {len(pairs_signals)} signals")

        # --- Strategy B: Mean Reversion (30% allocation) ---
        mr_signals = self._scan_mean_reversion(
            price_data, volume_data, ohlcv_data, equity, current_positions, timestamp
        )
        all_signals.extend(mr_signals)
        logger.info(f"Mean reversion: {len(mr_signals)} signals")

        # --- Strategy C: Momentum (20% allocation, only if A+B have few signals) ---
        # Momentum is a FALLBACK — only used when stat-arb has no signals
        if len(pairs_signals) + len(mr_signals) < 2:
            mom_signals = self._scan_momentum(
                price_data, volume_data, ohlcv_data, equity, current_positions, timestamp
            )
            all_signals.extend(mom_signals)
            logger.info(f"Momentum (fallback): {len(mom_signals)} signals")
        else:
            logger.info("Momentum skipped — stat-arb strategies have signals")

        # Sort by confidence (highest first)
        all_signals.sort(key=lambda s: s.confidence, reverse=True)

        logger.info(
            f"Strategy engine: {len(all_signals)} total signals "
            f"(pairs={len(pairs_signals)}, mr={len(mr_signals)}, "
            f"mom={len(all_signals) - len(pairs_signals) - len(mr_signals)})"
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

            # Compute current z-score and rolling hedge ratio
            z_score, hedge_ratio = self.pair_finder.compute_pair_z_score(pair, pa, pb)
            pair_id = f"{pair.sym_a}_{pair.sym_b}"

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

                # Stop: 1.5x ATR below entry
                stop_price = round(current_price - self.cfg.mr_atr_stop_mult * atr, 2)
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

                stop_price = round(current_price + self.cfg.mr_atr_stop_mult * atr, 2)
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

            # 200-day SMA — regime filter
            sma_200 = compute_sma(prices, self.cfg.mom_sma_period)

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

            # === LONG: price above 200-SMA, pulling back to 20-EMA, strong trend ===
            if (current_price > sma_200 and             # Bullish regime
                    adx > self.cfg.mom_adx_threshold and    # Confirmed trend
                    current_price <= ema_20 * 1.02 and      # At or below 20-EMA (pullback)
                    current_price >= ema_20 * 0.96):        # Not too far below (still in trend)

                # Trailing stop at 2x ATR below entry
                stop_price = round(current_price - self.cfg.mom_atr_trail_mult * atr, 2)

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
                        f"above SMA200 ${sma_200:.2f}, ADX={adx:.0f}, "
                        f"trend={trend_strength:+.1%}"
                    ),
                    atr=atr,
                    adx=adx,
                    timestamp=timestamp,
                ))
                mom_count += 1

            # === SHORT: price below 200-SMA, bouncing to 20-EMA, strong downtrend ===
            elif (current_price < sma_200 and
                    adx > self.cfg.mom_adx_threshold and
                    current_price >= ema_20 * 0.98 and
                    current_price <= ema_20 * 1.04):

                stop_price = round(current_price + self.cfg.mom_atr_trail_mult * atr, 2)
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
                        f"below SMA200 ${sma_200:.2f}, ADX={adx:.0f}, "
                        f"trend={trend_strength:+.1%}"
                    ),
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
    ):
        """Record a completed trade for strategy performance tracking."""
        if strategy in self._strategy_stats:
            stats = self._strategy_stats[strategy]
            stats["total_pnl"] += pnl
            if pnl > 0:
                stats["wins"] += 1
            else:
                stats["losses"] += 1

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
