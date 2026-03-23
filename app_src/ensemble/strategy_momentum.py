"""
ensemble/strategy_momentum.py
=============================
Cross-Sectional Momentum Strategy (ORIA alpha sleeve #1).

Ranks stocks by recent momentum and generates LONG signals for top
performers and SHORT signals for bottom performers.

This provides an independent alpha source orthogonal to the TDA diffusion
strategy, which is fundamentally a mean-reversion signal.

Features:
- Dual-timeframe momentum (fast 5d + slow 20d)
- Volume-weighted momentum confirmation
- Regime-aware strength scaling
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_REGIME_DISCOUNT = {"NORMAL": 1.0, "STRESSED": 0.70, "CRASH": 0.40}


class MomentumStrategy:
    """Cross-sectional momentum alpha sleeve.

    Parameters
    ----------
    fast_window : int
        Fast momentum lookback in days (default 5).
    slow_window : int
        Slow momentum lookback in days (default 20).
    top_n : int
        Number of top/bottom stocks to signal (default 5).
    volume_confirm : bool
        Require above-average volume for confirmation (default True).
    """

    def __init__(
        self,
        fast_window: int = 5,
        slow_window: int = 20,
        top_n: int = 5,
        volume_confirm: bool = True,
    ):
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.top_n = top_n
        self.volume_confirm = volume_confirm

    def generate_signals(
        self,
        price_df: pd.DataFrame,
        volume_df: Optional[pd.DataFrame] = None,
        regime: str = "NORMAL",
    ) -> pd.DataFrame:
        """Generate momentum signals from price data.

        Parameters
        ----------
        price_df : pd.DataFrame
            Price data, columns = tickers, rows = dates.
        volume_df : pd.DataFrame, optional
            Volume data, same structure as price_df.
        regime : str
            Current market regime.

        Returns
        -------
        pd.DataFrame
            Columns: ticker, direction, strength, regime, timestamp.
        """
        if price_df is None or price_df.empty or len(price_df) < self.slow_window + 5:
            return pd.DataFrame(columns=["ticker", "direction", "strength", "regime", "timestamp"])

        records = []
        discount = _REGIME_DISCOUNT.get(regime, 1.0)

        # Compute fast and slow momentum for each ticker
        momentum_scores = {}
        for col in price_df.columns:
            prices = price_df[col].dropna()
            if len(prices) < self.slow_window + 1:
                continue

            # Fast momentum: 5-day return
            fast_ret = (prices.iloc[-1] / prices.iloc[-self.fast_window] - 1) if prices.iloc[-self.fast_window] > 0 else 0
            # Slow momentum: 20-day return
            slow_ret = (prices.iloc[-1] / prices.iloc[-self.slow_window] - 1) if prices.iloc[-self.slow_window] > 0 else 0

            # Combined score: weighted blend
            score = 0.6 * fast_ret + 0.4 * slow_ret

            # Volume confirmation
            vol_confirmed = True
            if self.volume_confirm and volume_df is not None and col in volume_df.columns:
                vol = volume_df[col].dropna()
                if len(vol) >= 20:
                    avg_vol = vol.tail(20).mean()
                    recent_vol = vol.tail(5).mean()
                    vol_confirmed = recent_vol > avg_vol * 0.8  # At least 80% of average

            momentum_scores[col] = {
                "score": score,
                "fast_ret": fast_ret,
                "slow_ret": slow_ret,
                "vol_confirmed": vol_confirmed,
            }

        if not momentum_scores:
            return pd.DataFrame(columns=["ticker", "direction", "strength", "regime", "timestamp"])

        # Rank by score
        sorted_tickers = sorted(momentum_scores.keys(), key=lambda t: momentum_scores[t]["score"], reverse=True)

        # Cross-sectional stats for normalization
        scores = [momentum_scores[t]["score"] for t in sorted_tickers]
        score_std = np.std(scores) if len(scores) > 2 else 0.01
        score_mean = np.mean(scores)

        timestamp = price_df.index[-1] if hasattr(price_df.index, '__len__') else None

        # Top N → LONG, Bottom N → SHORT
        n = min(self.top_n, len(sorted_tickers) // 3)  # Don't signal more than 1/3

        for i, ticker in enumerate(sorted_tickers):
            info = momentum_scores[ticker]

            # z-score relative to cross-section
            z = (info["score"] - score_mean) / score_std if score_std > 0.001 else 0

            if i < n and info["score"] > 0 and info["vol_confirmed"]:
                # Top momentum → LONG
                raw_strength = min(1.0, abs(z) / 3.0)
                direction = "LONG"
            elif i >= len(sorted_tickers) - n and info["score"] < 0 and info["vol_confirmed"]:
                # Bottom momentum → SHORT
                raw_strength = min(1.0, abs(z) / 3.0)
                direction = "SHORT"
            else:
                direction = "NEUTRAL"
                raw_strength = 0.0

            if direction != "NEUTRAL":
                records.append({
                    "ticker": ticker,
                    "direction": direction,
                    "strength": round(raw_strength * discount, 6),
                    "regime": regime,
                    "timestamp": timestamp,
                })

        result = pd.DataFrame(records) if records else pd.DataFrame(
            columns=["ticker", "direction", "strength", "regime", "timestamp"]
        )

        logger.info(
            "MomentumStrategy: %d signals (%d LONG, %d SHORT) | regime=%s",
            len(result),
            len(result[result["direction"] == "LONG"]) if not result.empty else 0,
            len(result[result["direction"] == "SHORT"]) if not result.empty else 0,
            regime,
        )
        return result
