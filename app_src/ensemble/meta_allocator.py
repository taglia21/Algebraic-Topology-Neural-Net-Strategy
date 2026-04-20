"""
ensemble/meta_allocator.py
==========================
Meta-classifier for dynamic capital allocation between TDA and NN strategies.

Decides how much weight to give each strategy based on rolling performance
metrics, regime state, and strategy agreement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default regime-based allocation (before meta-classifier is trained)
_DEFAULT_REGIME_WEIGHTS: Dict[str, Dict[str, float]] = {
    "CRASH": {"tda_weight": 0.70, "nn_weight": 0.30},
    "NORMAL": {"tda_weight": 0.50, "nn_weight": 0.50},
    "STRESSED": {"tda_weight": 0.40, "nn_weight": 0.60},
}


@dataclass
class AllocationResult:
    """Result from the meta-allocator."""

    tda_weight: float
    nn_weight: float
    reasoning: str


class MetaAllocator:
    """Dynamically allocate capital between TDA and NN strategies.

    Parameters
    ----------
    sharpe_lookbacks : list[int]
        Rolling windows for Sharpe ratio calculation (default [63, 126, 252]).
    min_history : int
        Minimum number of observations before the meta-classifier can be
        trained (default 252).  Until then, uses regime-based defaults.
    model_type : str
        Type of meta-classifier: ``'logistic'`` or ``'lightgbm'``
        (default ``'logistic'``).
    """

    def __init__(
        self,
        sharpe_lookbacks: Optional[List[int]] = None,
        min_history: int = 252,
        model_type: str = "logistic",
    ) -> None:
        self.sharpe_lookbacks = sharpe_lookbacks or [63, 126, 252]
        self.min_history = min_history
        self.model_type = model_type
        self._classifier: Optional[Any] = None
        self._is_trained = False

        # Rolling history for strategy returns
        self._tda_returns: List[float] = []
        self._nn_returns: List[float] = []

    @property
    def is_trained(self) -> bool:
        """Whether the meta-classifier has been trained."""
        return self._is_trained

    def update_history(
        self,
        tda_return: float,
        nn_return: float,
    ) -> None:
        """Append a single period's strategy returns to history.

        Parameters
        ----------
        tda_return : float
            TDA strategy return for this period.
        nn_return : float
            NN strategy return for this period.
        """
        self._tda_returns.append(tda_return)
        self._nn_returns.append(nn_return)

    def _compute_meta_features(self, regime: str) -> Dict[str, float]:
        """Compute features for the meta-classifier.

        Parameters
        ----------
        regime : str
            Current market regime.

        Returns
        -------
        dict
            Feature dict for the meta-classifier.
        """
        tda = np.array(self._tda_returns)
        nn = np.array(self._nn_returns)
        n = len(tda)

        features: Dict[str, float] = {}

        # Rolling Sharpe for each lookback
        for lb in self.sharpe_lookbacks:
            window = min(lb, n)
            if window < 2:
                features[f"tda_sharpe_{lb}"] = 0.0
                features[f"nn_sharpe_{lb}"] = 0.0
            else:
                tda_w = tda[-window:]
                nn_w = nn[-window:]
                tda_std = tda_w.std()
                nn_std = nn_w.std()
                features[f"tda_sharpe_{lb}"] = (
                    float(tda_w.mean() / tda_std * np.sqrt(252))
                    if tda_std > 0
                    else 0.0
                )
                features[f"nn_sharpe_{lb}"] = (
                    float(nn_w.mean() / nn_std * np.sqrt(252))
                    if nn_std > 0
                    else 0.0
                )

        # Rolling hit rate (fraction of positive returns)
        lookback = min(63, n)
        if lookback > 0:
            features["tda_hit_rate"] = float((tda[-lookback:] > 0).mean())
            features["nn_hit_rate"] = float((nn[-lookback:] > 0).mean())
        else:
            features["tda_hit_rate"] = 0.5
            features["nn_hit_rate"] = 0.5

        # Strategy agreement: fraction of days both had same sign
        if lookback > 0:
            tda_sign = np.sign(tda[-lookback:])
            nn_sign = np.sign(nn[-lookback:])
            features["agreement_rate"] = float((tda_sign == nn_sign).mean())
        else:
            features["agreement_rate"] = 0.5

        # Recent max drawdown
        for label, arr in [("tda", tda), ("nn", nn)]:
            window = min(63, n)
            if window > 0:
                cum = np.cumsum(arr[-window:])
                peak = np.maximum.accumulate(cum)
                dd = cum - peak
                features[f"{label}_max_dd"] = float(dd.min()) if len(dd) > 0 else 0.0
            else:
                features[f"{label}_max_dd"] = 0.0

        # Regime encoding
        regime_map = {"NORMAL": 0, "STRESSED": 1, "CRASH": 2}
        features["regime"] = float(regime_map.get(regime, 0))

        return features

    def train(
        self,
        tda_returns: np.ndarray,
        nn_returns: np.ndarray,
        regimes: List[str],
    ) -> None:
        """Train the meta-classifier on historical strategy returns.

        Parameters
        ----------
        tda_returns : np.ndarray
            Historical daily returns from TDA strategy.
        nn_returns : np.ndarray
            Historical daily returns from NN strategy.
        regimes : list[str]
            Regime label for each day.
        """
        self._tda_returns = list(tda_returns)
        self._nn_returns = list(nn_returns)

        n = len(tda_returns)
        if n < self.min_history:
            logger.warning(
                "Insufficient history (%d < %d) — skipping training",
                n,
                self.min_history,
            )
            return

        # Build feature matrix and labels
        # Label: 1 if TDA outperformed NN over next 5 days, else 0
        X_rows = []
        y = []
        forward = 5

        for i in range(self.min_history, n - forward):
            self._tda_returns = list(tda_returns[: i + 1])
            self._nn_returns = list(nn_returns[: i + 1])
            feat = self._compute_meta_features(regimes[i])
            X_rows.append(feat)

            tda_fwd = tda_returns[i + 1 : i + 1 + forward].sum()
            nn_fwd = nn_returns[i + 1 : i + 1 + forward].sum()
            y.append(1 if tda_fwd > nn_fwd else 0)

        if len(X_rows) < 50:
            logger.warning("Too few training samples (%d) — skipping", len(X_rows))
            # Restore full history
            self._tda_returns = list(tda_returns)
            self._nn_returns = list(nn_returns)
            return

        X = pd.DataFrame(X_rows).values
        y_arr = np.array(y)

        if self.model_type == "lightgbm":
            try:
                import lightgbm as lgb

                self._classifier = lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.05,
                    verbose=-1,
                )
                self._classifier.fit(X, y_arr)
                self._is_trained = True
            except ImportError:
                logger.warning("lightgbm not available — falling back to logistic")
                self.model_type = "logistic"

        if self.model_type == "logistic":
            from sklearn.linear_model import LogisticRegression

            self._classifier = LogisticRegression(max_iter=1000)
            self._classifier.fit(X, y_arr)
            self._is_trained = True

        # Restore full history
        self._tda_returns = list(tda_returns)
        self._nn_returns = list(nn_returns)
        logger.info("Meta-classifier trained on %d samples", len(X_rows))

    def allocate(
        self,
        tda_signals: pd.DataFrame,
        nn_signals: pd.DataFrame,
        market_state: Dict[str, Any],
    ) -> AllocationResult:
        """Decide allocation weights for TDA vs NN strategies.

        Parameters
        ----------
        tda_signals : pd.DataFrame
            Current TDA strategy signals.
        nn_signals : pd.DataFrame
            Current NN strategy signals.
        market_state : dict
            Must contain ``'regime'`` key (str).

        Returns
        -------
        AllocationResult
            Contains tda_weight, nn_weight (sum to 1), and reasoning.
        """
        regime = market_state.get("regime", "NORMAL")

        # Default mode: regime-based allocation
        if not self._is_trained:
            weights = _DEFAULT_REGIME_WEIGHTS.get(
                regime, _DEFAULT_REGIME_WEIGHTS["NORMAL"]
            )
            return AllocationResult(
                tda_weight=weights["tda_weight"],
                nn_weight=weights["nn_weight"],
                reasoning=f"Default regime-based allocation ({regime})",
            )

        # Trained mode: use meta-classifier
        features = self._compute_meta_features(regime)
        X = np.array([list(features.values())])

        try:
            proba = self._classifier.predict_proba(X)[0]
            # proba[1] = probability that TDA outperforms
            tda_prob = float(proba[1]) if len(proba) > 1 else 0.5
        except Exception as exc:
            logger.warning("Meta-classifier prediction failed: %s", exc)
            tda_prob = 0.5

        # Convert probability to weights (clamped to [0.2, 0.8])
        tda_weight = max(0.2, min(0.8, tda_prob))
        nn_weight = 1.0 - tda_weight

        return AllocationResult(
            tda_weight=round(tda_weight, 4),
            nn_weight=round(nn_weight, 4),
            reasoning=(
                f"Meta-classifier: TDA prob={tda_prob:.3f}, "
                f"regime={regime}, "
                f"weights TDA={tda_weight:.2f}/NN={nn_weight:.2f}"
            ),
        )
