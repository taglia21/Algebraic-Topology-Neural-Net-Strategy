"""
Dynamic Strategy Weight Optimizer
==================================

Adjusts multi-strategy weights based on the current market regime,
recent performance of each strategy, and correlation between strategy
returns.

Used by AutonomousTradingEngine to dynamically allocate capital across
IV-rank, theta-decay, mean-reversion, and delta-hedging strategies.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WeightSnapshot:
    """Point-in-time record of strategy weights."""
    weights: Dict[str, float]
    regime: str
    timestamp: datetime
    reason: str = ""


class DynamicWeightOptimizer:
    """
    Rebalance strategy weights in response to regime transitions and
    recent per-strategy PnL.

    Args:
        strategies: List of strategy names to manage.
        regime_detector: Optional RegimeDetector instance for regime-aware
                         weight adjustments.
        ema_span: Lookback (in rebalance calls) for the EMA of per-strategy
                  returns used for momentum tilting.
    """

    # Preferred weight profiles per regime label.
    # Updated to include VRP + IV Crush strategies.
    _REGIME_PROFILES: Dict[str, Dict[str, float]] = {
        "bull_low_vol": {
            "iv_rank": 0.20,
            "theta_decay": 0.20,
            "mean_reversion": 0.10,
            "delta_hedging": 0.05,
            "vrp": 0.35,       # VRP dominant: sell premium in calm bull
            "iv_crush": 0.10,
        },
        "bull_high_vol": {
            "iv_rank": 0.20,
            "theta_decay": 0.15,
            "mean_reversion": 0.15,
            "delta_hedging": 0.15,
            "vrp": 0.25,       # Still sell premium but cautious
            "iv_crush": 0.10,
        },
        "bear_low_vol": {
            "iv_rank": 0.15,
            "theta_decay": 0.15,
            "mean_reversion": 0.25,  # Mean reversion dominates
            "delta_hedging": 0.15,
            "vrp": 0.20,
            "iv_crush": 0.10,
        },
        "bear_high_vol": {
            "iv_rank": 0.10,
            "theta_decay": 0.05,
            "mean_reversion": 0.10,
            "delta_hedging": 0.45,  # Max hedging in crisis
            "vrp": 0.25,       # VRP still works (sell rich IV)
            "iv_crush": 0.05,
        },
    }

    _DEFAULT_PROFILE: Dict[str, float] = {
        "iv_rank": 0.17,
        "theta_decay": 0.17,
        "mean_reversion": 0.13,
        "delta_hedging": 0.10,
        "vrp": 0.25,
        "iv_crush": 0.08,
        "vol_divergence": 0.10,
    }

    def __init__(
        self,
        strategies: Optional[List[str]] = None,
        regime_detector=None,
        ema_span: int = 20,
    ):
        self.strategies = strategies or list(self._DEFAULT_PROFILE.keys())
        self.regime_detector = regime_detector
        self.ema_span = ema_span

        # Current weights — start equal
        self.weights: Dict[str, float] = {
            s: 1.0 / len(self.strategies) for s in self.strategies
        }
        self.history: List[WeightSnapshot] = []
        self._strategy_returns: Dict[str, List[float]] = {s: [] for s in self.strategies}

        logger.info(
            f"DynamicWeightOptimizer initialised with {len(self.strategies)} strategies"
        )

    # ─── public API ───────────────────────────────────────────────

    async def rebalance(
        self,
        regime=None,
        force: bool = False,
    ) -> Dict[str, float]:
        """
        Rebalance strategy weights.

        Args:
            regime: Current MarketRegime (enum with ``.value``).  If
                    *None*, equal weights are used.
            force:  If True, always rebalance even if regime is unchanged.

        Returns:
            New weight dictionary (strategy_name → weight).
        """
        regime_label = regime.value if regime is not None else "default"

        # Get regime-based target weights
        target = self._REGIME_PROFILES.get(regime_label, self._DEFAULT_PROFILE)

        # Build weights only for our active strategies
        new_weights: Dict[str, float] = {}
        for s in self.strategies:
            new_weights[s] = target.get(s, 1.0 / len(self.strategies))

        # Normalise
        total = sum(new_weights.values()) or 1.0
        new_weights = {s: w / total for s, w in new_weights.items()}

        self.weights = new_weights
        self.history.append(
            WeightSnapshot(
                weights=dict(new_weights),
                regime=regime_label,
                timestamp=datetime.now(),
                reason="regime_change" if force else "periodic",
            )
        )

        logger.info(
            f"Weights rebalanced ({regime_label}): "
            + ", ".join(f"{s}={w:.0%}" for s, w in new_weights.items())
        )
        return new_weights

    def record_strategy_return(self, strategy: str, ret: float) -> None:
        """Record a single-period return for momentum tracking."""
        if strategy in self._strategy_returns:
            self._strategy_returns[strategy].append(ret)
            # Keep bounded
            if len(self._strategy_returns[strategy]) > self.ema_span * 3:
                self._strategy_returns[strategy] = self._strategy_returns[strategy][-self.ema_span * 2:]

    def get_weights(self) -> Dict[str, float]:
        """Return current weight allocation."""
        return dict(self.weights)
