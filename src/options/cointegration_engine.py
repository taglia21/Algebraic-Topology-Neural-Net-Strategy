"""
Cointegration Engine
=====================

Identifies cointegrated pairs of options-eligible underlyings for
pairs-trading strategies.  The engine runs periodic scans (triggered
by the AutonomousTradingEngine) and returns pairs that pass the
Engle-Granger cointegration test.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Try importing statsmodels for proper cointegration test
try:
    from statsmodels.tsa.stattools import coint as _sm_coint
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


@dataclass
class CointegratedPair:
    """A pair of symbols that pass the cointegration test."""
    symbol_a: str
    symbol_b: str
    p_value: float
    hedge_ratio: float        # β in  A = α + β·B + ε
    half_life: float           # Mean-reversion half-life (days)
    spread_zscore: float       # Current z-score of the spread
    timestamp: datetime


class CointegrationEngine:
    """
    Scan a universe of symbols for Engle–Granger cointegrated pairs.

    Usage (async, called from AutonomousTradingEngine):

        engine = CointegrationEngine()
        pairs = await engine.find_cointegrated_pairs(
            symbols=["SPY", "QQQ", "IWM", ...],
            max_pairs=5,
        )
    """

    def __init__(
        self,
        lookback_days: int = 120,
        p_value_threshold: float = 0.05,
        min_half_life: float = 2.0,
        max_half_life: float = 60.0,
    ):
        self.lookback_days = lookback_days
        self.p_value_threshold = p_value_threshold
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self._price_cache: Dict[str, np.ndarray] = {}

        if not HAS_STATSMODELS:
            logger.warning(
                "statsmodels not available — cointegration tests will use "
                "a correlation-based heuristic instead"
            )
        logger.info(
            f"CointegrationEngine initialised (lookback={lookback_days}d, "
            f"p<{p_value_threshold})"
        )

    # ─── public API ───────────────────────────────────────────────

    async def find_cointegrated_pairs(
        self,
        symbols: List[str],
        max_pairs: int = 5,
        price_data: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[CointegratedPair]:
        """
        Find cointegrated pairs among *symbols*.

        Args:
            symbols: Candidate ticker list (capped internally to avoid
                     O(n²) blowup).
            max_pairs: Maximum pairs to return.
            price_data: Pre-fetched close-price arrays keyed by symbol.
                        If *None*, the engine uses an internal stub that
                        returns empty results (real data should be injected).

        Returns:
            List of CointegratedPair ordered by p-value (best first).
        """
        if price_data:
            self._price_cache.update(price_data)

        results: List[CointegratedPair] = []
        n = min(len(symbols), 20)  # cap to avoid huge pairwise scan

        for i in range(n):
            for j in range(i + 1, n):
                pair = self._test_pair(symbols[i], symbols[j])
                if pair is not None:
                    results.append(pair)

        # Sort by p-value and return top N
        results.sort(key=lambda p: p.p_value)
        top = results[:max_pairs]

        if top:
            logger.info(
                f"Found {len(top)} cointegrated pairs (best p={top[0].p_value:.4f})"
            )
        return top

    # ─── internals ────────────────────────────────────────────────

    def _test_pair(self, sym_a: str, sym_b: str) -> Optional[CointegratedPair]:
        """Run Engle-Granger test on a single pair."""
        prices_a = self._price_cache.get(sym_a)
        prices_b = self._price_cache.get(sym_b)

        if prices_a is None or prices_b is None:
            return None

        # Align lengths
        min_len = min(len(prices_a), len(prices_b))
        if min_len < 30:
            return None
        pa = prices_a[-min_len:].astype(float)
        pb = prices_b[-min_len:].astype(float)

        # Cointegration test
        if HAS_STATSMODELS:
            try:
                _, p_value, _ = _sm_coint(pa, pb)
            except Exception:
                return None
        else:
            # Fallback: correlation-based heuristic (not a true coint test)
            corr = float(np.corrcoef(pa, pb)[0, 1])
            p_value = max(0.0, 1.0 - abs(corr))  # rough proxy

        if p_value > self.p_value_threshold:
            return None

        # Hedge ratio via OLS:  A = α + β·B
        try:
            beta = float(np.polyfit(pb, pa, 1)[0])
        except Exception:
            beta = 1.0

        # Spread and half-life
        spread = pa - beta * pb
        spread_mean = float(np.mean(spread))
        spread_std = float(np.std(spread)) or 1.0
        z_score = float((spread[-1] - spread_mean) / spread_std)

        # Ornstein-Uhlenbeck half-life estimate
        half_life = self._estimate_half_life(spread)

        if not (self.min_half_life <= half_life <= self.max_half_life):
            return None

        return CointegratedPair(
            symbol_a=sym_a,
            symbol_b=sym_b,
            p_value=round(p_value, 6),
            hedge_ratio=round(beta, 4),
            half_life=round(half_life, 1),
            spread_zscore=round(z_score, 3),
            timestamp=datetime.now(),
        )

    @staticmethod
    def _estimate_half_life(spread: np.ndarray) -> float:
        """Estimate mean-reversion half-life from an OU-process fit."""
        lag = spread[:-1]
        delta = np.diff(spread)
        if len(lag) < 10:
            return 999.0
        try:
            beta = float(np.polyfit(lag, delta, 1)[0])
            if beta >= 0:
                return 999.0  # Not mean-reverting
            return -np.log(2) / beta
        except Exception:
            return 999.0
