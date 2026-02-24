"""
Walk-Forward Validator (Phase K, Item 17)
==========================================

Expanding-window walk-forward validation with 60-day train / 20-day
test splits.  Computes out-of-sample Sharpe, max drawdown, and hit
rate per fold.  Writes results to ``logs/walk_forward_results.json``.
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["WalkForwardValidator", "WFVConfig", "FoldResult"]


@dataclass
class WFVConfig:
    """Walk-forward validation configuration."""
    train_days: int = 60
    test_days: int = 20
    min_train_samples: int = 30
    output_path: str = "logs/walk_forward_results.json"


@dataclass
class FoldResult:
    """Result from one walk-forward fold."""
    fold_index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    oos_sharpe: float
    oos_max_drawdown: float
    oos_hit_rate: float
    oos_return: float
    n_train: int
    n_test: int


class WalkForwardValidator:
    """Expanding-window walk-forward validator.

    Parameters
    ----------
    config : WFVConfig or None
    """

    def __init__(self, config: Optional[WFVConfig] = None):
        self.config = config or WFVConfig()
        self._results: List[FoldResult] = []

    def validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_predict_fn: Callable,
        returns: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Run walk-forward validation.

        Parameters
        ----------
        X : ndarray (n_samples, n_features)
        y : ndarray (n_samples,) binary labels
        train_predict_fn : callable
            ``fn(X_train, y_train, X_test) -> predictions``
        returns : ndarray or None
            Actual returns per bar (for Sharpe/drawdown); defaults to y.

        Returns
        -------
        dict with aggregate metrics and per-fold results.
        """
        n = len(X)
        if returns is None:
            returns = y.astype(float)

        train_days = self.config.train_days
        test_days = self.config.test_days
        self._results = []
        fold_idx = 0

        start = 0
        while start + train_days + test_days <= n:
            train_end = start + train_days
            test_end = min(train_end + test_days, n)

            X_train = X[start:train_end]
            y_train = y[start:train_end]
            X_test = X[train_end:test_end]
            y_test = y[train_end:test_end]
            r_test = returns[train_end:test_end]

            if len(X_train) < self.config.min_train_samples:
                start += test_days
                continue

            # Train and predict
            try:
                preds = train_predict_fn(X_train, y_train, X_test)
            except Exception as exc:
                logger.warning("Fold %d failed: %s", fold_idx, exc)
                start += test_days
                continue

            # Compute OOS metrics
            oos_sharpe = self._compute_sharpe(r_test, preds)
            oos_mdd = self._compute_max_drawdown(r_test)
            oos_hit = self._compute_hit_rate(y_test, preds)
            oos_ret = float(np.sum(r_test))

            fold = FoldResult(
                fold_index=fold_idx,
                train_start=start, train_end=train_end,
                test_start=train_end, test_end=test_end,
                oos_sharpe=oos_sharpe, oos_max_drawdown=oos_mdd,
                oos_hit_rate=oos_hit, oos_return=oos_ret,
                n_train=len(X_train), n_test=len(X_test),
            )
            self._results.append(fold)
            fold_idx += 1

            # Expanding window: move test window forward
            start += test_days

        summary = self._summarize()
        self._save_results(summary)
        return summary

    def _compute_sharpe(self, returns, predictions) -> float:
        """Sharpe ratio of strategy returns (sign of prediction × return)."""
        if len(returns) == 0:
            return 0.0
        pred_sign = np.where(np.array(predictions) >= 0.5, 1, -1)
        strat_returns = np.array(returns) * pred_sign
        mu = np.mean(strat_returns)
        sigma = np.std(strat_returns)
        return float(mu / sigma * np.sqrt(252)) if sigma > 0 else 0.0

    def _compute_max_drawdown(self, returns) -> float:
        """Max drawdown from cumulative returns."""
        cum = np.cumsum(returns)
        peak = np.maximum.accumulate(cum)
        dd = peak - cum
        return float(np.max(dd)) if len(dd) > 0 else 0.0

    def _compute_hit_rate(self, y_true, y_pred) -> float:
        """Fraction of correct predictions."""
        y_pred_binary = (np.array(y_pred) >= 0.5).astype(int)
        return float(np.mean(y_pred_binary == np.array(y_true)))

    def _summarize(self) -> Dict[str, Any]:
        """Aggregate metrics across all folds."""
        if not self._results:
            return {"folds": 0, "avg_oos_sharpe": 0.0}

        return {
            "folds": len(self._results),
            "avg_oos_sharpe": float(np.mean([f.oos_sharpe for f in self._results])),
            "avg_oos_mdd": float(np.mean([f.oos_max_drawdown for f in self._results])),
            "avg_oos_hit_rate": float(np.mean([f.oos_hit_rate for f in self._results])),
            "avg_oos_return": float(np.mean([f.oos_return for f in self._results])),
            "per_fold": [asdict(f) for f in self._results],
            "timestamp": datetime.now().isoformat(),
        }

    def _save_results(self, summary: Dict) -> None:
        """Write results to JSON file."""
        try:
            path = Path(self.config.output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(summary, f, indent=2)
            logger.info("Walk-forward results saved to %s", path)
        except Exception as exc:
            logger.error("Failed to save WF results: %s", exc)

    @property
    def results(self) -> List[FoldResult]:
        return list(self._results)
