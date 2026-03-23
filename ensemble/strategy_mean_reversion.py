"""
ensemble/strategy_mean_reversion.py
====================================
Mean-Reversion Strategy (ORIA alpha sleeve #2).

Detects short-term oversold/overbought conditions using Bollinger Band
Z-scores and RSI, generating contrarian signals.

This complements the TDA (topological mean-reversion on graph diffusion)
by operating on pure price-level technicals rather than cross-asset
topology, providing a weakly correlated alpha source.

Features:
- Bollinger Band Z-score (20d SMA, 2σ)
- RSI confirmation (14d)
- Volume exhaustion detection
- Regime-aware: suppressed during CRASH (trend following dominates)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_REGIME_DISCOUNT = {"NORMAL": 1.0, "STRESSED": 0.60, "CRASH": 0.25}


def _compute_rsi(prices: pd.Series, window: int = 14) -> float:
    """Compute RSI for the last value."""
    if len(prices) < window + 1:
        return 50.0
    deltas = prices.diff().dropna()
    recent = deltas.tail(window)
    gains = recent.clip(lower=0).mean()
    losses = (-recent.clip(upper=0)).mean()
    if losses < 1e-10:
        return 100.0
    rs = gains / losses
    return 100.0 - (100.0 / (1.0 + rs))


class MeanReversionStrategy:
    """Bollinger Band + RSI mean-reversion alpha sleeve.

    Parameters
    ----------
    bb_window : int
        Bollinger Band SMA window (default 20).
    bb_std : float
        Number of standard deviations for bands (default 2.0).
    rsi_window : int
        RSI calculation window (default 14).
    rsi_oversold : float
        RSI threshold for oversold (default 30).
    rsi_overbought : float
        RSI threshold for overbought (default 70).
    """

    def __init__(
        self,
        bb_window: int = 20,
        bb_std: float = 2.0,
        rsi_window: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
    ):
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.rsi_window = rsi_window
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def generate_signals(
        self,
        price_df: pd.DataFrame,
        volume_df: Optional[pd.DataFrame] = None,
        regime: str = "NORMAL",
    ) -> pd.DataFrame:
        """Generate mean-reversion signals.

        Parameters
        ----------
        price_df : pd.DataFrame
            Price data.
        volume_df : pd.DataFrame, optional
            Volume data.
        regime : str
            Current regime.

        Returns
        -------
        pd.DataFrame
            Columns: ticker, direction, strength, regime, timestamp.
        """
        if price_df is None or price_df.empty or len(price_df) < self.bb_window + 5:
            return pd.DataFrame(columns=["ticker", "direction", "strength", "regime", "timestamp"])

        records = []
        discount = _REGIME_DISCOUNT.get(regime, 1.0)
        timestamp = price_df.index[-1] if hasattr(price_df.index, '__len__') else None

        for col in price_df.columns:
            prices = price_df[col].dropna()
            if len(prices) < self.bb_window + 1:
                continue

            # Bollinger Band Z-score
            sma = prices.tail(self.bb_window).mean()
            std = prices.tail(self.bb_window).std()
            current = prices.iloc[-1]

            if std < 1e-8:
                continue

            bb_z = (current - sma) / std

            # RSI
            rsi = _compute_rsi(prices, self.rsi_window)

            # Volume exhaustion: is volume declining during the move?
            vol_exhaustion = False
            if volume_df is not None and col in volume_df.columns:
                vol = volume_df[col].dropna()
                if len(vol) >= 10:
                    vol_recent = vol.tail(3).mean()
                    vol_prior = vol.tail(10).head(7).mean()
                    # Declining volume during an extreme move suggests exhaustion
                    if vol_prior > 0 and vol_recent < vol_prior * 0.7:
                        vol_exhaustion = True

            # Signal logic
            direction = "NEUTRAL"
            raw_strength = 0.0

            # Oversold: price below lower BB + RSI confirms
            if bb_z < -self.bb_std and rsi < self.rsi_oversold:
                direction = "LONG"
                # Strength: how far below the band
                raw_strength = min(1.0, (abs(bb_z) - self.bb_std) / self.bb_std)
                # Volume exhaustion bonus
                if vol_exhaustion:
                    raw_strength = min(1.0, raw_strength * 1.2)

            # Overbought: price above upper BB + RSI confirms
            elif bb_z > self.bb_std and rsi > self.rsi_overbought:
                direction = "SHORT"
                raw_strength = min(1.0, (abs(bb_z) - self.bb_std) / self.bb_std)
                if vol_exhaustion:
                    raw_strength = min(1.0, raw_strength * 1.2)

            # Moderate signals (less strict — BB extreme without RSI)
            elif bb_z < -(self.bb_std * 1.5):
                direction = "LONG"
                raw_strength = min(0.5, (abs(bb_z) - self.bb_std * 1.5) / self.bb_std)

            elif bb_z > (self.bb_std * 1.5):
                direction = "SHORT"
                raw_strength = min(0.5, (abs(bb_z) - self.bb_std * 1.5) / self.bb_std)

            if direction != "NEUTRAL" and raw_strength > 0.05:
                records.append({
                    "ticker": col,
                    "direction": direction,
                    "strength": round(raw_strength * discount, 6),
                    "regime": regime,
                    "timestamp": timestamp,
                })

        result = pd.DataFrame(records) if records else pd.DataFrame(
            columns=["ticker", "direction", "strength", "regime", "timestamp"]
        )

        logger.info(
            "MeanReversionStrategy: %d signals (%d LONG, %d SHORT) | regime=%s",
            len(result),
            len(result[result["direction"] == "LONG"]) if not result.empty else 0,
            len(result[result["direction"] == "SHORT"]) if not result.empty else 0,
            regime,
        )
        return result
