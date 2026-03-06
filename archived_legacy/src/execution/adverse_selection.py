"""
Phase N — Adverse Selection Filter.

Item 5: AdverseSelectionFilter — PIN model, PIN > 0.25 → widen spread 2x, reduce qty 50%.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PINEstimate:
    """Probability of Informed Trading (PIN) estimate."""
    pin: float = 0.0
    alpha: float = 0.0  # probability of information event
    delta: float = 0.0  # probability of bad news (given event)
    mu: float = 0.0     # informed arrival rate
    eps_b: float = 0.0  # uninformed buy arrival rate
    eps_s: float = 0.0  # uninformed sell arrival rate
    n_days: int = 0


@dataclass
class AdverseSelectionAdjustment:
    """Adjustments to apply based on adverse selection risk."""
    spread_multiplier: float = 1.0
    size_multiplier: float = 1.0
    is_toxic: bool = False
    pin_estimate: float = 0.0
    reason: str = ""


class AdverseSelectionFilter:
    """Detect adverse selection using the PIN model.

    PIN (Probability of Informed Trading) from Easley, Kiefer, O'Hara, Paperman (1996).
    PIN = alpha * mu / (alpha * mu + eps_b + eps_s)

    When PIN > threshold:
      - Widen spread by 2x
      - Reduce quantity by 50%

    Uses EM-style estimation on buy/sell trade counts.
    """

    def __init__(
        self,
        pin_threshold: float = 0.25,
        spread_multiplier_toxic: float = 2.0,
        size_multiplier_toxic: float = 0.5,
        min_days: int = 10,
    ):
        self.pin_threshold = pin_threshold
        self.spread_multiplier_toxic = spread_multiplier_toxic
        self.size_multiplier_toxic = size_multiplier_toxic
        self.min_days = min_days
        self._last_estimate: Optional[PINEstimate] = None

    def estimate_pin(
        self,
        buy_counts: np.ndarray,
        sell_counts: np.ndarray,
    ) -> PINEstimate:
        """Estimate PIN from daily buy/sell trade counts using MLE.

        Uses a simplified closed-form PIN estimator.

        Args:
            buy_counts: Array of daily buy-initiated trade counts.
            sell_counts: Array of daily sell-initiated trade counts.

        Returns:
            PINEstimate with model parameters.
        """
        buy_counts = np.asarray(buy_counts, dtype=np.float64)
        sell_counts = np.asarray(sell_counts, dtype=np.float64)

        n_days = len(buy_counts)
        if n_days < self.min_days:
            self._last_estimate = PINEstimate(n_days=n_days)
            return self._last_estimate

        # Simplified PIN: use order imbalance approach
        # Higher absolute imbalance days → higher information events
        total = buy_counts + sell_counts
        imbalance = np.abs(buy_counts - sell_counts)

        avg_total = np.mean(total)
        avg_imbalance = np.mean(imbalance)

        # Estimate uninformed rates as minimum observed activity
        eps_b = float(np.mean(buy_counts)) * 0.7
        eps_s = float(np.mean(sell_counts)) * 0.7

        # Estimate informed rate from excess imbalance
        mu = float(avg_imbalance)

        # Estimate alpha from proportion of high-imbalance days
        median_imbalance = np.median(imbalance)
        alpha = float(np.mean(imbalance > median_imbalance * 1.5))
        alpha = max(min(alpha, 0.99), 0.01)

        # Estimate delta (prob bad news) from sell-dominant days
        sell_dominant = np.sum(sell_counts > buy_counts)
        delta = float(sell_dominant / max(n_days, 1))

        # PIN = alpha * mu / (alpha * mu + eps_b + eps_s)
        numerator = alpha * mu
        denominator = numerator + eps_b + eps_s
        pin = numerator / max(denominator, 1e-12)
        pin = max(min(pin, 1.0), 0.0)

        estimate = PINEstimate(
            pin=pin,
            alpha=alpha,
            delta=delta,
            mu=mu,
            eps_b=eps_b,
            eps_s=eps_s,
            n_days=n_days,
        )
        self._last_estimate = estimate
        logger.info(
            "PIN estimate: %.3f (alpha=%.2f, mu=%.1f, eps_b=%.1f, eps_s=%.1f)",
            pin, alpha, mu, eps_b, eps_s,
        )
        return estimate

    def check_toxicity(
        self,
        buy_counts: np.ndarray,
        sell_counts: np.ndarray,
    ) -> AdverseSelectionAdjustment:
        """Check for adverse selection and return spread/size adjustments.

        Args:
            buy_counts: Array of daily buy-initiated trade counts.
            sell_counts: Array of daily sell-initiated trade counts.

        Returns:
            AdverseSelectionAdjustment with spread and size multipliers.
        """
        estimate = self.estimate_pin(buy_counts, sell_counts)
        is_toxic = estimate.pin > self.pin_threshold

        if is_toxic:
            adj = AdverseSelectionAdjustment(
                spread_multiplier=self.spread_multiplier_toxic,
                size_multiplier=self.size_multiplier_toxic,
                is_toxic=True,
                pin_estimate=estimate.pin,
                reason=f"PIN={estimate.pin:.3f} > {self.pin_threshold} threshold",
            )
            logger.warning(
                "ADVERSE SELECTION: PIN=%.3f > %.3f — widening spread %.1fx, reducing size %.1fx",
                estimate.pin, self.pin_threshold,
                self.spread_multiplier_toxic, self.size_multiplier_toxic,
            )
        else:
            adj = AdverseSelectionAdjustment(
                spread_multiplier=1.0,
                size_multiplier=1.0,
                is_toxic=False,
                pin_estimate=estimate.pin,
                reason=f"PIN={estimate.pin:.3f} ≤ {self.pin_threshold} — normal conditions",
            )

        return adj

    @property
    def last_estimate(self) -> Optional[PINEstimate]:
        return self._last_estimate
