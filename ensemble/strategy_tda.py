"""
ensemble/strategy_tda.py
========================
TDA Diffusion Arbitrage Strategy.

Stocks whose price deviates significantly from their topological consensus
(diffusion residual) are considered mispriced:

- Large positive residual → OVERPRICED vs neighbours → SHORT
- Large negative residual → UNDERPRICED vs neighbours → LONG
- Within threshold → NEUTRAL
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Regime discount factors
_REGIME_DISCOUNT = {
    "NORMAL": 1.0,
    "STRESSED": 0.75,
    "CRASH": 0.50,
}


class TDADiffusionStrategy:
    """Generate trading signals from TDA diffusion residuals.

    Parameters
    ----------
    residual_threshold : float
        Number of standard deviations for signal generation (default 1.5).
        Residuals beyond ±threshold produce LONG/SHORT signals.
    """

    def __init__(self, residual_threshold: float = 1.5) -> None:
        self.residual_threshold = residual_threshold
        self._last_signals: Optional[pd.DataFrame] = None

    def generate_signals(
        self,
        tda_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate per-stock trading signals from TDA features.

        Expects a DataFrame with columns including per-stock diffusion
        residuals (z-scored).  The ``tda.laplacian_diffusion.LaplacianDiffusion
        .generate_signals`` method produces exactly this format: one column per
        ticker with z-scored residual values.

        Optionally, ``tda_features`` may contain a ``'regime'`` column (values
        ``'NORMAL'``, ``'STRESSED'``, ``'CRASH'`` or numeric 0/1/2) to gate
        signal strength.

        Parameters
        ----------
        tda_features : pd.DataFrame
            Per-stock diffusion residual z-scores.  Each column is a ticker,
            each row is a date.  May include a ``'regime'`` column.

        Returns
        -------
        pd.DataFrame
            Columns: ticker, direction, strength, regime, timestamp.
            One row per (date, ticker) combination.
        """
        # Separate regime column if present
        regime_col = None
        _REGIME_NUM_MAP = {0: "NORMAL", 1: "STRESSED", 2: "CRASH"}

        if "regime" in tda_features.columns:
            regime_col = tda_features["regime"]
            residual_df = tda_features.drop(columns=["regime"])
        else:
            residual_df = tda_features

        records = []

        for date in residual_df.index:
            row = residual_df.loc[date]

            # Determine regime for this date
            if regime_col is not None:
                raw_regime = regime_col.loc[date]
                if isinstance(raw_regime, (int, float, np.integer, np.floating)):
                    regime = _REGIME_NUM_MAP.get(int(raw_regime), "NORMAL")
                else:
                    regime = str(raw_regime)
            else:
                regime = "NORMAL"

            discount = _REGIME_DISCOUNT.get(regime, 1.0)

            # Compute threshold from the cross-sectional std of this row
            row_values = row.values.astype(float)
            row_std = np.nanstd(row_values)
            if row_std == 0:
                row_std = 1.0

            threshold = self.residual_threshold

            for ticker in residual_df.columns:
                z_score = float(row[ticker])

                if np.isnan(z_score):
                    continue

                # Direction based on residual sign
                # Positive residual = outperformed neighbours = overpriced → SHORT
                # Negative residual = underperformed neighbours = underpriced → LONG
                if z_score > threshold:
                    direction = "SHORT"
                elif z_score < -threshold:
                    direction = "LONG"
                else:
                    direction = "NEUTRAL"

                # Strength: how far beyond threshold, scaled to 0-1
                abs_z = abs(z_score)
                if direction == "NEUTRAL":
                    raw_strength = 0.0
                else:
                    # Scale: at threshold → 0, at 2*threshold → ~0.5, at 3*threshold → ~0.67
                    raw_strength = min(1.0, (abs_z - threshold) / threshold) if threshold > 0 else min(1.0, abs_z)

                # Apply regime discount
                strength = raw_strength * discount

                records.append({
                    "ticker": ticker,
                    "direction": direction,
                    "strength": round(strength, 6),
                    "regime": regime,
                    "timestamp": date,
                })

        result = pd.DataFrame(records)
        if result.empty:
            result = pd.DataFrame(
                columns=["ticker", "direction", "strength", "regime", "timestamp"]
            )
        self._last_signals = result
        return result

    def get_top_signals(self, n: int = 10) -> pd.DataFrame:
        """Return the top N strongest signals by absolute strength.

        Parameters
        ----------
        n : int
            Number of top signals to return (default 10).

        Returns
        -------
        pd.DataFrame
            Top signals sorted by strength descending.
        """
        if self._last_signals is None or self._last_signals.empty:
            return pd.DataFrame(
                columns=["ticker", "direction", "strength", "regime", "timestamp"]
            )

        active = self._last_signals[self._last_signals["direction"] != "NEUTRAL"]
        return (
            active.sort_values("strength", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )
