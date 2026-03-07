"""
ml/meta_labeler.py
==================
Meta-Labeling and ML-Driven Bet Sizing (AFML Ch. 3.6 + Ch. 10).

The meta-labeling framework splits prediction into two stages:

    **Stage 1 — Primary Model (Direction)**
    The base model or strategy decides BUY vs SELL.  This can be any
    model or rule-based system (momentum, mean-reversion, etc.).

    **Stage 2 — Secondary Model (Sizing / Gating)**
    A meta-labeler decides:
    - Whether to take the trade (p > threshold → trade, else skip)
    - How large the position should be (bet sizing from predicted probability)

Why Meta-Labeling?
------------------
1. Separates direction from sizing → each model can specialise.
2. Reduces false positives: primary model is aggressive, meta-model filters.
3. Enables Kelly-criterion-style bet sizing from calibrated probabilities.
4. Dramatically improves Sharpe: only take trades where the secondary
   model predicts the primary model will be correct.

Bet Sizing (AFML Ch. 10)
-------------------------
Given a predicted probability p from the meta-labeler:
    bet_size = 2 * p - 1   (maps [0.5, 1.0] → [0.0, 1.0])
    bet_size = max(0, bet_size) * max_size

This is the "average active bet" approach: we scale position size by the
meta-model's confidence that the primary signal is correct.

Usage
-----
    from ml.meta_labeler import MetaLabeler

    ml = MetaLabeler()
    ml.train(features, primary_predictions, actual_outcomes)
    result = ml.predict(features_new, primary_signal_direction)
    # result → {"probability": 0.72, "bet_size": 0.44, "take_trade": True}

References
----------
- Lopez de Prado (2018), AFML, Ch. 3.6 — Meta-Labeling
- Lopez de Prado (2018), AFML, Ch. 10 — Bet Sizing
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


class MetaLabeler:
    """Secondary model that decides trade/no-trade and bet size.

    Parameters
    ----------
    max_bet_size : float
        Maximum position scale factor (1.0 = full position).
    min_probability : float
        Minimum predicted probability to take a trade.
    n_estimators : int
        Number of boosting trees for the meta-model.
    max_depth : int
        Maximum tree depth.
    """

    def __init__(
        self,
        max_bet_size: float = 1.0,
        min_probability: float = 0.55,
        n_estimators: int = 200,
        max_depth: int = 3,
    ) -> None:
        self.max_bet_size = max_bet_size
        self.min_probability = min_probability

        self._model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.05,
            subsample=0.8,
            max_features=0.7,
            random_state=42,
        )
        self._calibrator: Optional[IsotonicRegression] = None
        self._is_fitted: bool = False
        self._feature_names: List[str] = []

    # ------------------------------------------------------------------
    # Label creation
    # ------------------------------------------------------------------

    @staticmethod
    def create_meta_labels(
        primary_direction: pd.Series,
        forward_returns: pd.Series,
    ) -> pd.Series:
        """Create binary meta-labels: 1 if the primary model was correct.

        Parameters
        ----------
        primary_direction : pd.Series
            Direction from the primary model (+1 for long, -1 for short).
        forward_returns : pd.Series
            Actual forward returns over the trade horizon.

        Returns
        -------
        pd.Series
            Binary labels: 1 = primary was correct, 0 = incorrect.
        """
        common = primary_direction.index.intersection(forward_returns.index)
        direction = primary_direction.loc[common]
        returns = forward_returns.loc[common]

        # Primary was correct if direction * return > 0
        correct = (direction * returns > 0).astype(int)
        return correct

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        features: pd.DataFrame,
        meta_labels: pd.Series,
        sample_weight: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """Train the meta-labeling model with time-series cross-validation.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix aligned with meta_labels.
        meta_labels : pd.Series
            Binary labels (1 = primary model correct, 0 = incorrect).
        sample_weight : pd.Series, optional
            Sample weights (e.g., from CUSUM event importance).

        Returns
        -------
        dict
            Training report with CV scores.
        """
        # Align
        common = features.index.intersection(meta_labels.dropna().index)
        X = features.loc[common].fillna(0.0)
        y = meta_labels.loc[common].astype(int)

        if len(X) < 100:
            logger.warning(
                f"MetaLabeler.train: only {len(X)} samples; need >= 100."
            )
            return {"fitted": False, "reason": "insufficient samples"}

        self._feature_names = list(X.columns)

        # Time-series CV for OOS calibration
        tscv = TimeSeriesSplit(n_splits=5)
        oos_preds = np.full(len(y), np.nan)
        cv_scores = []

        for train_idx, test_idx in tscv.split(X):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

            sw = None
            if sample_weight is not None:
                sw_aligned = sample_weight.reindex(X_tr.index).fillna(1.0)
                sw = sw_aligned.values

            self._model.fit(X_tr.values, y_tr.values, sample_weight=sw)
            preds = self._model.predict_proba(X_te.values)[:, 1]
            oos_preds[test_idx] = preds

            accuracy = ((preds > 0.5) == y_te.values).mean()
            cv_scores.append(accuracy)

        # Fit calibrator on OOS predictions
        valid_mask = ~np.isnan(oos_preds)
        if valid_mask.sum() >= 20:
            self._calibrator = IsotonicRegression(
                out_of_bounds="clip", increasing=True
            )
            self._calibrator.fit(oos_preds[valid_mask], y.values[valid_mask])

        # Fit final model on all data
        sw_full = None
        if sample_weight is not None:
            sw_full = sample_weight.reindex(X.index).fillna(1.0).values

        self._model.fit(X.values, y.values, sample_weight=sw_full)
        self._is_fitted = True

        mean_cv = float(np.mean(cv_scores))
        logger.info(
            f"MetaLabeler trained: {len(X)} samples, "
            f"CV accuracy={mean_cv:.3f}, features={len(self._feature_names)}"
        )

        return {
            "fitted": True,
            "n_samples": len(X),
            "cv_accuracy": mean_cv,
            "cv_scores": cv_scores,
            "n_features": len(self._feature_names),
        }

    # ------------------------------------------------------------------
    # Prediction + Bet Sizing
    # ------------------------------------------------------------------

    def predict(
        self,
        features: pd.DataFrame,
        primary_direction: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """Predict trade probability and compute bet size.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix (single row or multi-row).
        primary_direction : pd.Series, optional
            +1/-1 direction from the primary model.

        Returns
        -------
        dict with keys:
            probability : float or Series
                Calibrated probability that the primary model is correct.
            bet_size : float or Series
                Position scale factor in [0, max_bet_size].
            take_trade : bool or Series
                Whether to take the trade (probability >= min_probability).
        """
        if not self._is_fitted:
            logger.warning("MetaLabeler.predict: model not fitted.")
            return {"probability": 0.5, "bet_size": 0.0, "take_trade": False}

        X = features.reindex(
            columns=self._feature_names, fill_value=0.0
        ).fillna(0.0)

        raw_prob = self._model.predict_proba(X.values)[:, 1]

        # Calibrate
        if self._calibrator is not None:
            prob = self._calibrator.predict(raw_prob)
        else:
            prob = raw_prob

        prob = np.clip(prob, 0.0, 1.0)

        # Bet sizing: average active bet (AFML Ch. 10)
        # Maps [0.5, 1.0] → [0.0, max_bet_size]
        bet_size = np.maximum(0.0, 2.0 * prob - 1.0) * self.max_bet_size

        # Trade/no-trade gating
        take_trade = prob >= self.min_probability

        if len(prob) == 1:
            return {
                "probability": float(prob[0]),
                "bet_size": float(bet_size[0]),
                "take_trade": bool(take_trade[0]),
            }

        return {
            "probability": pd.Series(prob, index=features.index),
            "bet_size": pd.Series(bet_size, index=features.index),
            "take_trade": pd.Series(take_trade, index=features.index),
        }

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def feature_importance(self) -> Dict[str, float]:
        """Return feature importance from the meta-model."""
        if not self._is_fitted:
            return {}
        imp = self._model.feature_importances_
        return dict(zip(self._feature_names, imp))
