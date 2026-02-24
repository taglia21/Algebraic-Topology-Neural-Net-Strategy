"""
ML Ensemble Stacker (Phase K, Item 16)
========================================

Combine SGDClassifier (existing), RandomForestClassifier, and
LogisticRegression into a meta-learner (LogisticRegression on
out-of-fold predictions).  Retrain weekly.

Expose ``ensemble_predict_proba()`` returning blended probability.
"""

import logging
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["EnsembleStacker", "EnsembleConfig"]

try:
    from sklearn.linear_model import SGDClassifier, LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class EnsembleConfig:
    """Configuration for the ensemble stacker."""
    rf_n_estimators: int = 100
    rf_max_depth: int = 5
    sgd_alpha: float = 1e-4
    meta_C: float = 1.0
    min_samples_for_train: int = 50
    retrain_interval_days: int = 7

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class EnsembleStacker:
    """Three-model stacked ensemble with LogisticRegression meta-learner.

    Base models:
      1. SGDClassifier (incremental-capable)
      2. RandomForestClassifier (n=100, depth=5)
      3. LogisticRegression (L2)

    Meta-learner: LogisticRegression trained on out-of-fold predictions.

    Parameters
    ----------
    config : EnsembleConfig or None
    """

    def __init__(self, config: Optional[EnsembleConfig] = None):
        self.config = config or EnsembleConfig()
        self._fitted = False
        self._last_train_time: Optional[datetime] = None
        self._base_models: List = []
        self._meta_model = None
        self._n_features: int = 0

        if SKLEARN_AVAILABLE:
            self._init_models()

    def _init_models(self):
        """Initialize base learners and meta-learner."""
        self._base_models = [
            SGDClassifier(
                loss="log_loss", penalty="l2",
                alpha=self.config.sgd_alpha, warm_start=True,
                random_state=42,
            ),
            RandomForestClassifier(
                n_estimators=self.config.rf_n_estimators,
                max_depth=self.config.rf_max_depth,
                random_state=42, n_jobs=-1,
            ),
            LogisticRegression(
                C=self.config.meta_C, max_iter=500, random_state=42,
            ),
        ]
        self._meta_model = LogisticRegression(
            C=1.0, max_iter=500, random_state=42,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train all base models and meta-learner.

        Uses 3-fold cross_val_predict to generate OOF predictions
        for meta-learner training.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,) with binary labels {0, 1}

        Returns
        -------
        dict with training metrics.
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn not available — ensemble disabled")
            return {"error": "sklearn not installed"}

        if len(X) < self.config.min_samples_for_train:
            return {"error": f"need {self.config.min_samples_for_train} samples, got {len(X)}"}

        self._n_features = X.shape[1]
        n_classes = len(np.unique(y))
        if n_classes < 2:
            return {"error": "need at least 2 classes"}

        # Generate OOF predictions from each base model
        oof = np.zeros((len(X), len(self._base_models)))
        cv = min(3, len(X) // 10) if len(X) >= 30 else 2

        for i, model in enumerate(self._base_models):
            try:
                preds = cross_val_predict(
                    model, X, y, cv=cv, method="predict_proba",
                )
                oof[:, i] = preds[:, 1]  # probability of class 1
            except Exception as exc:
                logger.warning("OOF for model %d failed: %s", i, exc)
                oof[:, i] = 0.5

        # Fit base models on full data
        for model in self._base_models:
            try:
                model.fit(X, y)
            except Exception as exc:
                logger.warning("Base model fit failed: %s", exc)

        # Fit meta-learner on OOF predictions
        try:
            self._meta_model.fit(oof, y)
        except Exception as exc:
            logger.warning("Meta-learner fit failed: %s", exc)

        self._fitted = True
        self._last_train_time = datetime.now()

        logger.info("Ensemble trained: %d samples, %d features, %d base models",
                     len(X), X.shape[1], len(self._base_models))

        return {
            "n_samples": len(X),
            "n_features": X.shape[1],
            "n_base_models": len(self._base_models),
            "fitted": True,
        }

    def ensemble_predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities using the stacked ensemble.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,) with probability of class 1.
        """
        if not self._fitted or not SKLEARN_AVAILABLE:
            return np.full(len(X), 0.5)

        # Get base model predictions
        base_preds = np.zeros((len(X), len(self._base_models)))
        for i, model in enumerate(self._base_models):
            try:
                proba = model.predict_proba(X)
                base_preds[:, i] = proba[:, 1]
            except Exception:
                base_preds[:, i] = 0.5

        # Meta-learner prediction
        try:
            meta_proba = self._meta_model.predict_proba(base_preds)
            return meta_proba[:, 1]
        except Exception:
            return np.mean(base_preds, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class labels."""
        proba = self.ensemble_predict_proba(X)
        return (proba >= 0.5).astype(int)

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def needs_retrain(self) -> bool:
        """Check if retraining is due (weekly schedule)."""
        if self._last_train_time is None:
            return True
        days = (datetime.now() - self._last_train_time).days
        return days >= self.config.retrain_interval_days
