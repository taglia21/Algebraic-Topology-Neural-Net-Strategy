"""
core/dynamic_allocator.py
=========================
ORIA-inspired dynamic strategy allocator.

Replaces the static 50/50 TDA/NN weight split with a data-driven
allocation signal that shifts capital toward whichever strategy sleeve
is best suited to current market conditions.

From Joshua Aalampour's ORIA Part 2:
    s_t^{alloc} = f(v_ratio, disp, corr, breadth, trend, skp, ...)

The allocator computes a feature vector of market-state indicators and
uses them to weight strategies adaptively.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AllocationResult:
    """Result of dynamic allocation."""
    tda_weight: float
    nn_weight: float
    reasoning: str
    features: Dict[str, float]


def _safe_rolling(series: pd.Series, window: int, func: str = "std") -> float:
    """Compute a rolling stat safely, returning NaN on failure."""
    try:
        if len(series) < window:
            vals = series
        else:
            vals = series.tail(window)
        if func == "std":
            return float(vals.std())
        elif func == "mean":
            return float(vals.mean())
        elif func == "corr":
            return float(vals.corr(vals.shift(1)))
        return np.nan
    except Exception:
        return np.nan


class DynamicAllocator:
    """ORIA-style market-state-driven strategy allocator.

    Computes allocation signal from:
    - v_ratio: short-term vs long-term volatility ratio (regime proxy)
    - disp: cross-sectional return dispersion (stock-picking opportunity)
    - corr: average pairwise correlation (macro vs idiosyncratic)
    - breadth: market breadth (% of stocks above their 20d MA)
    - trend: market trend strength (momentum of SPY/benchmark)
    - skp: return skewness (tail risk proxy)

    TDA strategy benefits from: high dispersion, low correlation, trending markets
    NN strategy benefits from: mean-reverting regimes, high correlation, stable vol

    Parameters
    ----------
    base_tda_weight : float
        Baseline TDA allocation (default 0.50).
    base_nn_weight : float
        Baseline NN allocation (default 0.50).
    sensitivity : float
        How aggressively to shift from base (0-1, default 0.3).
    """

    def __init__(
        self,
        base_tda_weight: float = 0.50,
        base_nn_weight: float = 0.50,
        sensitivity: float = 0.30,
    ):
        self.base_tda = base_tda_weight
        self.base_nn = base_nn_weight
        self.sensitivity = sensitivity
        self._last_features: Dict[str, float] = {}

    def compute_market_features(
        self,
        price_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        benchmark: str = "SPY",
    ) -> Dict[str, float]:
        """Compute the ORIA-style allocation feature vector.

        Parameters
        ----------
        price_df : pd.DataFrame
            Price data, columns = symbols.
        returns_df : pd.DataFrame
            Return data, columns = symbols.
        benchmark : str
            Benchmark symbol for trend calculation.

        Returns
        -------
        dict
            Feature vector with keys: v_ratio, disp, corr, breadth, trend, skp.
        """
        features = {}

        # 1. Volatility ratio: short-term (5d) vs long-term (20d) vol
        if benchmark in returns_df.columns:
            bm_ret = returns_df[benchmark].dropna()
            vol_short = _safe_rolling(bm_ret, 5, "std")
            vol_long = _safe_rolling(bm_ret, 20, "std")
            features["v_ratio"] = (
                vol_short / vol_long if vol_long > 1e-8 else 1.0
            )
        else:
            features["v_ratio"] = 1.0

        # 2. Cross-sectional dispersion: std of returns across all stocks
        try:
            latest_returns = returns_df.iloc[-1].dropna()
            features["disp"] = float(latest_returns.std()) if len(latest_returns) > 2 else 0.0
        except Exception:
            features["disp"] = 0.0

        # 3. Average pairwise correlation (rolling 20d)
        try:
            recent = returns_df.tail(20).dropna(axis=1, how="any")
            if recent.shape[1] > 2:
                corr_matrix = recent.corr()
                # Average off-diagonal correlation
                n = corr_matrix.shape[0]
                mask = ~np.eye(n, dtype=bool)
                features["corr"] = float(corr_matrix.values[mask].mean())
            else:
                features["corr"] = 0.5
        except Exception:
            features["corr"] = 0.5

        # 4. Market breadth: % of stocks above their 20d SMA
        try:
            if price_df is not None and not price_df.empty:
                above_sma = 0
                total = 0
                for col in price_df.columns:
                    series = price_df[col].dropna()
                    if len(series) >= 20:
                        sma20 = series.tail(20).mean()
                        current = series.iloc[-1]
                        if current > sma20:
                            above_sma += 1
                        total += 1
                features["breadth"] = above_sma / total if total > 0 else 0.5
            else:
                features["breadth"] = 0.5
        except Exception:
            features["breadth"] = 0.5

        # 5. Trend strength: benchmark 20d momentum
        try:
            if benchmark in price_df.columns:
                bm_prices = price_df[benchmark].dropna()
                if len(bm_prices) >= 20:
                    current = bm_prices.iloc[-1]
                    past = bm_prices.iloc[-20]
                    features["trend"] = (current - past) / past if past > 0 else 0.0
                else:
                    features["trend"] = 0.0
            else:
                features["trend"] = 0.0
        except Exception:
            features["trend"] = 0.0

        # 6. Return skewness (tail risk proxy, 20d rolling)
        try:
            if benchmark in returns_df.columns:
                bm_ret = returns_df[benchmark].dropna().tail(20)
                features["skp"] = float(bm_ret.skew()) if len(bm_ret) >= 10 else 0.0
            else:
                features["skp"] = 0.0
        except Exception:
            features["skp"] = 0.0

        self._last_features = features
        return features

    def allocate(
        self,
        price_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        regime: str = "NORMAL",
        benchmark: str = "SPY",
    ) -> AllocationResult:
        """Compute dynamic strategy allocation weights.

        Parameters
        ----------
        price_df : pd.DataFrame
            Price data.
        returns_df : pd.DataFrame
            Return data.
        regime : str
            Current regime from TDA detector.
        benchmark : str
            Benchmark symbol.

        Returns
        -------
        AllocationResult
            Weighted allocation with reasoning.
        """
        features = self.compute_market_features(price_df, returns_df, benchmark)

        # Score each feature for TDA vs NN preference
        # Positive score → favor TDA; Negative → favor NN
        score = 0.0
        reasons = []

        # High dispersion → TDA (topological signals thrive on cross-section spread)
        if features["disp"] > 0.015:
            score += 0.15
            reasons.append(f"disp={features['disp']:.4f} high→TDA")
        elif features["disp"] < 0.005:
            score -= 0.10
            reasons.append(f"disp={features['disp']:.4f} low→NN")

        # Low correlation → TDA (idiosyncratic alpha, topology shines)
        if features["corr"] < 0.3:
            score += 0.15
            reasons.append(f"corr={features['corr']:.2f} low→TDA")
        elif features["corr"] > 0.6:
            score -= 0.15
            reasons.append(f"corr={features['corr']:.2f} high→NN")

        # High v_ratio (vol expansion) → reduce both, but NN more resilient
        if features["v_ratio"] > 1.5:
            score -= 0.10
            reasons.append(f"v_ratio={features['v_ratio']:.2f} vol spike→NN")
        elif features["v_ratio"] < 0.7:
            score += 0.05
            reasons.append(f"v_ratio={features['v_ratio']:.2f} calm→TDA")

        # Strong trend → TDA (momentum/topology signals align)
        if abs(features["trend"]) > 0.03:
            score += 0.10
            reasons.append(f"trend={features['trend']:.3f} strong→TDA")
        elif abs(features["trend"]) < 0.005:
            score -= 0.05
            reasons.append(f"trend={features['trend']:.3f} flat→NN")

        # Breadth: extreme breadth divergence → TDA opportunity
        if features["breadth"] < 0.3 or features["breadth"] > 0.8:
            score += 0.05
            reasons.append(f"breadth={features['breadth']:.2f} extreme→TDA")

        # Negative skew → cautious, favor NN (more conservative)
        if features["skp"] < -0.5:
            score -= 0.10
            reasons.append(f"skp={features['skp']:.2f} neg tail→NN")

        # Regime adjustment
        if regime == "STRESSED":
            score -= 0.10
            reasons.append("regime=STRESSED→NN")
        elif regime == "CRASH":
            score -= 0.20
            reasons.append("regime=CRASH→NN")

        # Convert score to weights (clamp and scale)
        tda_shift = np.clip(score * self.sensitivity, -0.25, 0.25)
        tda_w = np.clip(self.base_tda + tda_shift, 0.20, 0.80)
        nn_w = 1.0 - tda_w

        reasoning = f"Dynamic: TDA={tda_w:.2f} NN={nn_w:.2f} [{'; '.join(reasons[:3])}]"

        logger.info(
            "DynamicAllocator: TDA=%.2f NN=%.2f | v_ratio=%.2f disp=%.4f corr=%.2f "
            "breadth=%.2f trend=%.3f skp=%.2f",
            tda_w, nn_w, features["v_ratio"], features["disp"],
            features["corr"], features["breadth"], features["trend"], features["skp"],
        )

        return AllocationResult(
            tda_weight=round(tda_w, 4),
            nn_weight=round(nn_w, 4),
            reasoning=reasoning,
            features=features,
        )
