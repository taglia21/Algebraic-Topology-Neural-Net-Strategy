"""
equities/strategies/mean_reversion.py
=======================================
Mean Reversion strategy for the ATNN trading system.

Overview
--------
This module implements a z-score-based mean reversion strategy that identifies
oversold (buy) and overbought (sell) stocks based on deviations from their
rolling mean, confirmed by RSI and scaled by realized volatility.

Pipeline
--------
1. Compute rolling z-score of close prices over a configurable lookback window.
2. Filter entries by RSI:
   - BUY when z-score < -entry_z AND RSI < 30 (oversold)
   - SELL when z-score >  entry_z AND RSI > 70 (overbought)
3. Exit when z-score crosses back toward mean (|z| < exit_z).
4. Hard stop at |z-score| >= 3.0.
5. Volume spike filter: require today's volume >= 1.2x the 20-day average.
6. Scale signal strength by z-score magnitude and inverse realized volatility.
7. Blocked entirely in CRISIS regime.

References
----------
- Avellaneda & Lee (2010), Quantitative Finance
- Jegadeesh (1990), "Evidence of Predictable Behavior of Security Returns"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from core.config import get_config
from core.logger import TradeLogger, get_trade_logger
from core.regime_detector import Regime, RegimeState
from equities.models import Signal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_HISTORY_BARS: int = 65         # Minimum bars to compute indicators
_DEFAULT_LOOKBACK: int = 60         # Rolling window for z-score
_ENTRY_Z: float = 1.2               # Z-score threshold for entry (was 1.5, lowered for more signals)
_EXIT_Z: float = 0.5                # Z-score threshold for exit
_HARD_STOP_Z: float = 3.0           # Hard stop z-score
_RSI_PERIOD: int = 14               # RSI computation period
_RSI_OVERSOLD: float = 30.0         # RSI oversold threshold
_RSI_OVERBOUGHT: float = 70.0       # RSI overbought threshold
_VOLUME_SPIKE_THRESHOLD: float = 1.2  # Volume spike multiplier
_RV_WINDOW: int = 20                # Realized volatility window


@dataclass
class MeanReversionConfig:
    """Configuration for the mean reversion strategy."""
    lookback: int = _DEFAULT_LOOKBACK
    entry_z: float = _ENTRY_Z
    exit_z: float = _EXIT_Z
    hard_stop_z: float = _HARD_STOP_Z
    rsi_period: int = _RSI_PERIOD
    rsi_oversold: float = _RSI_OVERSOLD
    rsi_overbought: float = _RSI_OVERBOUGHT
    rv_window: int = _RV_WINDOW


class MeanReversionStrategy:
    """Z-score and RSI-based mean reversion strategy.

    Parameters
    ----------
    config:
        Strategy configuration. Uses global config defaults if not provided.

    Attributes
    ----------
    STRATEGY_NAME : str
        Identifier used by the SignalGenerator for allocation lookups.
    """

    STRATEGY_NAME = "mean_reversion"

    def __init__(self, config=None) -> None:
        if config is not None:
            self._cfg = config
        else:
            self._cfg = MeanReversionConfig()
        self._open_positions: Set[str] = set()
        self._bar_count: int = 0

    # ------------------------------------------------------------------
    # RSI computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_rsi(series: pd.Series, period: int = 14) -> float:
        """Compute the latest RSI value for a price series."""
        if len(series) < period + 1:
            return 50.0  # neutral default

        delta = series.diff()
        gains = delta.clip(lower=0)
        losses = (-delta).clip(lower=0)

        avg_gain = gains.rolling(window=period, min_periods=period).mean()
        avg_loss = losses.rolling(window=period, min_periods=period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        latest = rsi.iloc[-1]
        return float(latest) if not np.isnan(latest) else 50.0

    # ------------------------------------------------------------------
    # Realized volatility
    # ------------------------------------------------------------------

    @staticmethod
    def _realized_volatility(series: pd.Series, window: int = 20) -> float:
        """Annualized realized volatility from daily returns."""
        if len(series) < window + 1:
            return 0.3  # moderate default

        log_ret = np.log(series / series.shift(1)).dropna()
        if len(log_ret) < window:
            return 0.3

        rv = float(log_ret.iloc[-window:].std() * np.sqrt(252))
        return max(rv, 0.01)  # floor to avoid division by zero

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        price_data: pd.DataFrame,
        regime_state: RegimeState,
        volume_data: Optional[pd.DataFrame] = None,
    ) -> List[Signal]:
        """Generate mean reversion entry and exit signals for one bar.

        Parameters
        ----------
        price_data:
            Wide-format close price DataFrame with a sorted
            DatetimeIndex and one column per symbol.
            Must contain at least ``_MIN_HISTORY_BARS`` (65) rows.
        regime_state:
            Current market regime from RegimeDetector.
        volume_data:
            Optional wide-format volume DataFrame matching the shape and
            columns of ``price_data``.  When provided, volume spike
            filtering is applied.  When ``None``, the volume filter is
            skipped (all stocks pass volume gate).

        Returns
        -------
        List[Signal]
        """
        self._bar_count += 1

        # Crisis regime: no signals
        if regime_state.is_crisis:
            logger.info("MeanReversionStrategy: blocked — CRISIS regime.")
            return []

        if len(price_data) < _MIN_HISTORY_BARS:
            logger.warning(
                f"MeanReversion: insufficient history ({len(price_data)} < {_MIN_HISTORY_BARS})."
            )
            return []

        if volume_data is None:
            logger.warning("MeanReversion: no volume data supplied — volume filter bypassed")

        # Exclude benchmark columns
        _EXCLUDE = {"SPY", "QQQ", "IWM"}
        trade_cols = [c for c in price_data.columns if c not in _EXCLUDE]

        # Volume spike check
        volume_ok: Dict[str, bool] = {}
        if volume_data is not None:
            for sym in trade_cols:
                if sym not in volume_data.columns:
                    volume_ok[sym] = True  # pass if data unavailable
                    continue
                vol_series = volume_data[sym].dropna()
                if len(vol_series) < 21:
                    volume_ok[sym] = True
                    continue
                avg_vol = vol_series.iloc[-21:-1].mean()
                today_vol = vol_series.iloc[-1]
                if avg_vol <= 0:
                    volume_ok[sym] = True
                    continue
                volume_ok[sym] = (today_vol / avg_vol) >= _VOLUME_SPIKE_THRESHOLD
        else:
            # No volume data — everyone passes
            for sym in trade_cols:
                volume_ok[sym] = True

        signals: List[Signal] = []
        exits = 0
        entries = 0

        for sym in trade_cols:
            series = price_data[sym].dropna()
            if len(series) < _MIN_HISTORY_BARS:
                continue

            # Rolling z-score
            lookback = self._cfg.lookback
            rolling_mean = series.rolling(window=lookback).mean()
            rolling_std = series.rolling(window=lookback).std()

            if rolling_std.iloc[-1] == 0 or np.isnan(rolling_std.iloc[-1]):
                continue

            z = float((series.iloc[-1] - rolling_mean.iloc[-1]) / rolling_std.iloc[-1])

            # RSI
            rsi = self._compute_rsi(series, self._cfg.rsi_period)

            # Realized volatility
            rv = self._realized_volatility(series, self._cfg.rv_window)

            # --- EXIT signals for open positions ---
            if sym in self._open_positions:
                # Exit: z-score crossed back toward mean
                if abs(z) < self._cfg.exit_z:
                    signals.append(Signal(
                        symbol=sym,
                        direction="close",
                        strength=1.0,
                        strategy=self.STRATEGY_NAME,
                        metadata={
                            "z": round(z, 4),
                            "rsi": round(rsi, 2),
                            "rv": round(rv, 4),
                            "regime": regime_state.regime.value,
                            "action": "mean_reversion_exit",
                        },
                    ))
                    self._open_positions.discard(sym)
                    exits += 1
                    continue

                # Hard stop
                if abs(z) >= self._cfg.hard_stop_z:
                    signals.append(Signal(
                        symbol=sym,
                        direction="close",
                        strength=1.0,
                        strategy=self.STRATEGY_NAME,
                        metadata={
                            "z": round(z, 4),
                            "rsi": round(rsi, 2),
                            "rv": round(rv, 4),
                            "regime": regime_state.regime.value,
                            "action": "hard_stop",
                        },
                    ))
                    self._open_positions.discard(sym)
                    exits += 1
                    continue

            # --- ENTRY signals ---
            # Strength scales with z-score magnitude, inversely with vol
            # Higher z → stronger signal; higher vol → weaker (more noise)
            raw_strength = max(0.0, (abs(z) - 1.0)) / (1.0 + rv)
            strength = min(float(raw_strength), 1.0)

            # BUY: oversold
            if z < -self._cfg.entry_z and rsi < self._cfg.rsi_oversold:
                if not volume_ok.get(sym, True):
                    continue  # skip low-volume

                signals.append(Signal(
                    symbol=sym,
                    direction="long",
                    strength=max(strength, 0.1),  # floor at 0.1
                    strategy=self.STRATEGY_NAME,
                    metadata={
                        "z": round(z, 4),
                        "rsi": round(rsi, 2),
                        "rv": round(rv, 4),
                        "regime": regime_state.regime.value,
                    },
                ))
                self._open_positions.add(sym)
                entries += 1

            # SELL: overbought
            elif z > self._cfg.entry_z and rsi > self._cfg.rsi_overbought:
                if not volume_ok.get(sym, True):
                    continue

                signals.append(Signal(
                    symbol=sym,
                    direction="short",
                    strength=max(strength, 0.1),  # floor at 0.1
                    strategy=self.STRATEGY_NAME,
                    metadata={
                        "z": round(z, 4),
                        "rsi": round(rsi, 2),
                        "rv": round(rv, 4),
                        "regime": regime_state.regime.value,
                    },
                ))
                self._open_positions.add(sym)
                entries += 1

        logger.info(
            f"MeanReversionStrategy.generate_signals: emitted {len(signals)} signals "
            f"({exits} exits, {entries} entries) | "
            f"regime={regime_state.regime.value} | "
            f"open_positions={len(self._open_positions)} | "
            f"bar={self._bar_count}"
        )

        return signals
