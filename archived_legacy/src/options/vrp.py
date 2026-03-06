"""
Phase P — Volatility Risk Premium.

Item 11: VolatilityRiskPremium — VRP = realized vol - IV, sell premium > 2 vol pts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VRPSignal:
    """Volatility Risk Premium signal."""
    vrp: float = 0.0           # realized_vol - implied_vol (annualized)
    realized_vol: float = 0.0
    implied_vol: float = 0.0
    signal: str = "hold"       # 'sell_premium', 'buy_premium', 'hold'
    strength: float = 0.0      # |VRP| normalized
    z_score: float = 0.0       # VRP z-score vs history


class VolatilityRiskPremium:
    """Volatility Risk Premium (VRP) strategy.

    VRP = Realized Vol - Implied Vol

    When VRP > sell_threshold (vol points):
      → IV is cheap relative to realized → buy premium (vol is underpriced)

    When VRP < -sell_threshold:
      → IV is expensive relative to realized → sell premium (collect VRP)

    Typical: sell premium when VRP < -2 vol points (IV > RV by 2+ pts).
    """

    def __init__(
        self,
        sell_threshold: float = 2.0,
        buy_threshold: float = 2.0,
        realized_window: int = 21,
        annualization_factor: float = 252.0,
    ):
        """
        Args:
            sell_threshold: VRP points to trigger sell premium (IV overpriced).
            buy_threshold: VRP points to trigger buy premium (IV underpriced).
            realized_window: Days for realized vol calculation.
            annualization_factor: Annualization factor (252 for daily).
        """
        self.sell_threshold = sell_threshold
        self.buy_threshold = buy_threshold
        self.realized_window = realized_window
        self.annualization_factor = annualization_factor
        self._history: List[float] = []

    def realized_vol(self, returns: np.ndarray, window: Optional[int] = None) -> float:
        """Compute annualized realized volatility.

        Args:
            returns: Array of daily log returns.
            window: Lookback window (default: self.realized_window).

        Returns:
            Annualized realized volatility (e.g., 0.20 = 20%).
        """
        w = window or self.realized_window
        returns = np.asarray(returns, dtype=np.float64)
        if len(returns) < w:
            w = len(returns)
        if w < 2:
            return 0.0

        recent = returns[-w:]
        daily_vol = float(np.std(recent, ddof=1))
        return daily_vol * np.sqrt(self.annualization_factor)

    def compute_vrp(
        self,
        returns: np.ndarray,
        implied_vol: float,
    ) -> VRPSignal:
        """Compute VRP signal.

        Args:
            returns: Array of daily log returns.
            implied_vol: Current implied volatility (annualized, e.g., 0.20).

        Returns:
            VRPSignal with direction and strength.
        """
        rv = self.realized_vol(returns)
        # VRP = RV - IV; negative VRP means IV > RV (premium to sell)
        vrp = (rv - implied_vol) * 100  # Convert to vol points

        self._history.append(vrp)

        # Z-score relative to history
        z_score = 0.0
        if len(self._history) > 10:
            hist = np.array(self._history)
            z_score = (vrp - np.mean(hist)) / max(np.std(hist), 1e-6)

        # Signal generation
        if vrp < -self.sell_threshold:
            signal = "sell_premium"
            strength = abs(vrp) / max(self.sell_threshold, 1e-6)
        elif vrp > self.buy_threshold:
            signal = "buy_premium"
            strength = abs(vrp) / max(self.buy_threshold, 1e-6)
        else:
            signal = "hold"
            strength = 0.0

        vrp_signal = VRPSignal(
            vrp=vrp,
            realized_vol=rv,
            implied_vol=implied_vol,
            signal=signal,
            strength=min(strength, 3.0),  # cap at 3x
            z_score=z_score,
        )

        logger.info(
            "VRP: %.2f pts (RV=%.1f%%, IV=%.1f%%) → %s",
            vrp, rv * 100, implied_vol * 100, signal,
        )
        return vrp_signal

    @property
    def history(self) -> List[float]:
        return self._history
