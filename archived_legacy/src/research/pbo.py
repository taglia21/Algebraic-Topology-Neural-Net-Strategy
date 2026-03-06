"""
Phase S — Probability of Backtest Overfitting (PBO).

Item 20: PBO calculator — CSCV method, N=16 sub-periods, PBO > 0.5 alert.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PBOResult:
    """Probability of Backtest Overfitting result."""
    pbo: float = 0.0                    # Probability of overfitting [0, 1]
    is_overfit: bool = False             # PBO > threshold
    n_combinations: int = 0
    n_subperiods: int = 0
    logit_distribution: np.ndarray = field(default_factory=lambda: np.array([]))
    rank_logits: List[float] = field(default_factory=list)
    performance_degradation: float = 0.0  # avg in-sample vs out-of-sample


class PBOCalculator:
    """Probability of Backtest Overfitting (PBO) using CSCV.

    Based on Bailey, Borwein, Lopez de Prado, Zhu (2017).

    Combinatorially Symmetric Cross-Validation (CSCV):
      1. Split returns into N sub-periods.
      2. For all C(N, N/2) combinations, split into IS/OOS.
      3. Find best strategy in IS, check its OOS rank.
      4. PBO = proportion where IS-best has OOS rank below median.

    Alert when PBO > 0.5.
    """

    def __init__(
        self,
        n_subperiods: int = 16,
        pbo_threshold: float = 0.5,
    ):
        if n_subperiods % 2 != 0:
            n_subperiods += 1  # Must be even
        self.n_subperiods = n_subperiods
        self.pbo_threshold = pbo_threshold

    def compute(
        self,
        strategy_returns: np.ndarray,
    ) -> PBOResult:
        """Compute PBO from a matrix of strategy returns.

        Args:
            strategy_returns: (T, S) matrix where T = time periods,
                             S = number of strategy variants/trials.

        Returns:
            PBOResult with PBO estimate.
        """
        strategy_returns = np.asarray(strategy_returns, dtype=np.float64)
        T, S = strategy_returns.shape

        if S < 2:
            return PBOResult(pbo=0.0, n_subperiods=self.n_subperiods)

        # Split into N sub-periods
        n = min(self.n_subperiods, T)
        if n % 2 != 0:
            n -= 1
        if n < 4:
            return PBOResult(pbo=0.0, n_subperiods=n)

        # Compute sub-period Sharpe ratios for each strategy
        period_len = T // n
        sub_returns = np.zeros((n, S))
        for i in range(n):
            start = i * period_len
            end = start + period_len
            chunk = strategy_returns[start:end]
            # Use average return as performance measure (Sharpe proxy)
            sub_returns[i] = np.mean(chunk, axis=0)

        half = n // 2
        indices = list(range(n))

        # Limit combinations for computational feasibility
        all_combos = list(combinations(indices, half))
        max_combos = min(len(all_combos), 500)  # Cap at 500
        if len(all_combos) > max_combos:
            rng = np.random.RandomState(42)
            combo_indices = rng.choice(len(all_combos), max_combos, replace=False)
            selected_combos = [all_combos[i] for i in combo_indices]
        else:
            selected_combos = all_combos

        logits = []
        n_overfit = 0

        for is_indices in selected_combos:
            oos_indices = [i for i in indices if i not in is_indices]

            # In-sample performance
            is_perf = np.mean(sub_returns[list(is_indices), :], axis=0)
            # Out-of-sample performance
            oos_perf = np.mean(sub_returns[oos_indices, :], axis=0)

            # Find best IS strategy
            best_is = int(np.argmax(is_perf))

            # Rank of IS-best in OOS
            oos_rank = int(np.sum(oos_perf >= oos_perf[best_is]))
            relative_rank = oos_rank / S

            # Compute logit (log-loss of OOS rank)
            # logit = log(rank / (S - rank))
            rank_pct = max(min(relative_rank, 0.999), 0.001)
            logit = float(np.log(rank_pct / (1 - rank_pct)))
            logits.append(logit)

            # Count overfit: IS-best ranks below median in OOS
            if relative_rank > 0.5:
                n_overfit += 1

        # PBO = fraction of combinations where IS-best ranks below OOS median
        pbo = n_overfit / max(len(selected_combos), 1)

        # Performance degradation
        is_mean = float(np.mean(sub_returns[:half, :]))
        oos_mean = float(np.mean(sub_returns[half:, :]))
        degradation = is_mean - oos_mean

        result = PBOResult(
            pbo=pbo,
            is_overfit=pbo > self.pbo_threshold,
            n_combinations=len(selected_combos),
            n_subperiods=n,
            logit_distribution=np.array(logits),
            rank_logits=logits[:20],  # Store first 20 for inspection
            performance_degradation=degradation,
        )

        if result.is_overfit:
            logger.warning(
                "PBO ALERT: PBO=%.3f > %.3f — strategy is likely overfit!",
                pbo, self.pbo_threshold,
            )
        else:
            logger.info("PBO=%.3f (threshold=%.3f) — acceptable", pbo, self.pbo_threshold)

        return result
