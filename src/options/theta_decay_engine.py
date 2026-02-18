"""
Theta Decay Engine
====================

Models time-decay dynamics for options positions and provides
optimal DTE (days to expiration) recommendations based on the
current implied-volatility regime and trend direction.

Key Concepts:
- Theta decay accelerates as options approach expiration
- Optimal entry window balances premium captured vs gamma risk
- IV regime and trend direction shift the recommended DTE
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================

class IVRegime(Enum):
    """Implied-volatility regime classification."""
    LOW = "low"           # IV Rank <30 – cheap premium
    NORMAL = "normal"     # IV Rank 30-70
    HIGH = "high"         # IV Rank 70-90 – rich premium
    EXTREME = "extreme"   # IV Rank >90 – crisis / event-driven


class TrendDirection(Enum):
    """Underlying trend classification."""
    STRONG_UP = "strong_up"
    UP = "up"
    NEUTRAL = "neutral"
    DOWN = "down"
    STRONG_DOWN = "strong_down"


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class DTERecommendation:
    """Optimal DTE recommendation with supporting data."""
    entry_dte_min: int
    entry_dte_max: int
    exit_dte: int             # Target DTE to close position
    theta_efficiency: float   # 0-1 score for theta capture potential
    gamma_risk: float         # 0-1 score for gamma exposure risk
    strategy_note: str        # Human-readable rationale


@dataclass
class ThetaProfile:
    """Theta-decay profile snapshot for a given DTE."""
    dte: int
    daily_theta_pct: float    # Approximate daily theta as % of premium
    theta_accel: float        # Rate of theta acceleration
    gamma_theta_ratio: float  # Gamma/Theta – lower is better for sellers


# ============================================================================
# ENGINE
# ============================================================================

class ThetaDecayEngine:
    """
    Compute optimal entry/exit DTE windows for premium-selling strategies.

    Methodology:
    - Base DTE range comes from the strategy type (spreads vs naked vs
      calendar).
    - IV regime shifts the range: higher IV → can afford shorter DTE
      (more premium per day) while lower IV → go further out for more
      time value.
    - Trend direction applies a tilt: strong trends against the
      position warrant wider DTE buffers.
    """

    # Base DTE ranges by strategy type
    _BASE_RANGES = {
        "spreads":   {"min": 25, "max": 45, "exit": 10},
        "naked":     {"min": 30, "max": 60, "exit": 14},
        "calendar":  {"min": 30, "max": 50, "exit": 7},
        "iron_condor": {"min": 25, "max": 45, "exit": 10},
        "straddle":  {"min": 21, "max": 42, "exit": 7},
    }

    # IV regime DTE adjustments (days added/subtracted to min/max)
    _IV_ADJUSTMENTS = {
        IVRegime.LOW:     {"min": +5,  "max": +10, "efficiency": 0.4},
        IVRegime.NORMAL:  {"min": 0,   "max": 0,   "efficiency": 0.6},
        IVRegime.HIGH:    {"min": -5,  "max": -5,  "efficiency": 0.8},
        IVRegime.EXTREME: {"min": -10, "max": -10, "efficiency": 0.95},
    }

    # Trend adjustments – adverse trend widens buffer
    _TREND_SHIFT = {
        TrendDirection.STRONG_UP:   -3,
        TrendDirection.UP:          -1,
        TrendDirection.NEUTRAL:      0,
        TrendDirection.DOWN:        +2,
        TrendDirection.STRONG_DOWN: +5,
    }

    def __init__(self) -> None:
        logger.info("ThetaDecayEngine initialised")

    # ─── public API ───────────────────────────────────────────────

    def calculate_optimal_dte(
        self,
        iv_rank: float,
        trend: TrendDirection = TrendDirection.NEUTRAL,
        volatility_regime: Optional[IVRegime] = None,
        strategy_type: str = "spreads",
    ) -> DTERecommendation:
        """
        Return an optimal DTE window for the given market context.

        Args:
            iv_rank: Current IV Rank (0-100).
            trend: Underlying trend direction.
            volatility_regime: Explicit IV regime override; if *None*,
                               derived from *iv_rank*.
            strategy_type: One of the keys in ``_BASE_RANGES``.

        Returns:
            DTERecommendation with entry_dte_min/max, exit_dte,
            theta-efficiency, gamma-risk, and a strategy note.
        """
        # Derive regime from iv_rank if not provided
        if volatility_regime is None:
            volatility_regime = self._classify_regime(iv_rank)

        base = self._BASE_RANGES.get(strategy_type, self._BASE_RANGES["spreads"])
        iv_adj = self._IV_ADJUSTMENTS[volatility_regime]
        trend_shift = self._TREND_SHIFT.get(trend, 0)

        entry_min = max(7, base["min"] + iv_adj["min"] + trend_shift)
        entry_max = max(entry_min + 5, base["max"] + iv_adj["max"] + trend_shift)
        exit_dte = max(3, base["exit"])

        # Theta efficiency: combination of IV premium level and sweet-spot
        theta_eff = iv_adj["efficiency"]

        # Gamma risk: shorter DTE → higher gamma risk
        mid_dte = (entry_min + entry_max) / 2
        gamma_risk = max(0.0, min(1.0, 1.0 - (mid_dte - 7) / 53))

        note = (
            f"{strategy_type} | regime={volatility_regime.value} "
            f"trend={trend.value} → DTE {entry_min}-{entry_max}, "
            f"close at {exit_dte}"
        )
        logger.debug(note)

        return DTERecommendation(
            entry_dte_min=entry_min,
            entry_dte_max=entry_max,
            exit_dte=exit_dte,
            theta_efficiency=theta_eff,
            gamma_risk=gamma_risk,
            strategy_note=note,
        )

    def get_theta_profile(self, dte: int, iv: float = 0.25) -> ThetaProfile:
        """
        Estimate theta-decay characteristics at a given DTE.

        Uses simplified Black-Scholes theta approximation:
            theta ∝ 1/sqrt(T)
        where T is time to expiration in years.
        """
        t_years = max(dte, 1) / 365.0
        sqrt_t = math.sqrt(t_years)

        # Normalised daily theta as % of premium
        daily_theta_pct = iv / (2.0 * sqrt_t * 365.0) * 100.0

        # Theta acceleration (second derivative proxy)
        t_next = max(dte - 1, 1) / 365.0
        sqrt_t_next = math.sqrt(t_next)
        next_theta_pct = iv / (2.0 * sqrt_t_next * 365.0) * 100.0
        theta_accel = next_theta_pct - daily_theta_pct

        # Gamma/Theta ratio (lower is better for sellers)
        gamma_theta_ratio = sqrt_t * 10.0  # simplified proxy

        return ThetaProfile(
            dte=dte,
            daily_theta_pct=round(daily_theta_pct, 4),
            theta_accel=round(theta_accel, 4),
            gamma_theta_ratio=round(gamma_theta_ratio, 4),
        )

    # ─── helpers ──────────────────────────────────────────────────

    @staticmethod
    def _classify_regime(iv_rank: float) -> IVRegime:
        """Bucket IV rank into a regime."""
        if iv_rank >= 90:
            return IVRegime.EXTREME
        if iv_rank >= 70:
            return IVRegime.HIGH
        if iv_rank <= 30:
            return IVRegime.LOW
        return IVRegime.NORMAL
