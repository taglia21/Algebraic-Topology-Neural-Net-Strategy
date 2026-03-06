"""
ml/models/gradient_boost.py
============================
LightGBM model wrapper for multi-horizon return prediction.

The model predicts:
  - direction (classification): +1 if forward return > 0, else 0
  - magnitude (regression): actual forward return (optional)

Three instances are typically created — one per horizon (1d, 5d, 20d).

Usage
-----
    from ml.models.gradient_boost import GradientBoostModel

    model = GradientBoostModel(horizon=5)
    labels = model.prepare_labels(price_data)
    model.train(features, labels, val_features, val_labels)
    probs = model.predict(new_features)
    imp = model.get_feature_importance()
    model.save("models/lgbm/horizon_5.pkl")
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default LightGBM hyperparameters (mirror core.config.LightGBMParams)
# ---------------------------------------------------------------------------
_DEFAULT_PARAMS: Dict = {
    "objective":         "binary",
    "metric":            "binary_logloss",
    "max_depth":         6,
    "num_leaves":        31,
    "learning_rate":     0.05,
    "min_child_samples": 50,
    "n_estimators":      300,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        0.1,
    "random_state":      42,
    "n_jobs":            -1,
    "verbose":           -1,
}


class GradientBoostModel:
    """LightGBM model for return prediction at multiple horizons.

    Supports both classification (direction) and regression (magnitude).
    Classification mode is the default and is used by the pipeline for
    generating trade signals.

    Parameters
    ----------
    horizon:
        Forward-return horizon in trading days (typically 1, 5, or 20).
    params:
        LightGBM hyperparameters.  Merged with ``_DEFAULT_PARAMS``; user
        values take precedence.
    mode:
        ``"classification"`` (default) predicts direction (+1 / 0).
        ``"regression"`` predicts the actual forward return.
    threshold:
        Minimum predicted probability (classification) or predicted return
        (regression) to emit a positive signal.  Exposed for strategy use.
    """

    def __init__(
        self,
        horizon: int = 5,
        params: Optional[Dict] = None,
        mode: str = "classification",
        threshold: float = 0.5,
    ) -> None:
        if mode not in ("classification", "regression"):
            raise ValueError(f"mode must be 'classification' or 'regression'; got {mode!r}")

        self.horizon = horizon
        self.mode = mode
        self.threshold = threshold

        # Merge with defaults
        merged = dict(_DEFAULT_PARAMS)
        if params:
            merged.update(params)

        # Adjust objective / metric for regression
        if mode == "regression":
            merged["objective"] = "regression"
            merged["metric"]    = "rmse"

        self._params = merged
        self.model = None          # fitted LightGBM estimator
        self.feature_names: List[str] = []
        self.feature_importances: Dict[str, float] = {}
        self._is_fitted: bool = False
        self._train_score: Optional[float] = None
        self._val_score:   Optional[float] = None

    # ------------------------------------------------------------------
    # Label preparation
    # ------------------------------------------------------------------

    def prepare_labels(
        self,
        price_data: pd.DataFrame,
        symbol: Optional[str] = None,
    ) -> pd.Series:
        """Create forward-return labels for the prediction horizon.

        Parameters
        ----------
        price_data:
            DataFrame with at minimum a ``close`` column.
        symbol:
            Optional ticker string used for column lookup when *price_data*
            has a MultiIndex or multiple symbol columns.

        Returns
        -------
        pd.Series
            For classification: binary {0, 1} (1 = positive forward return).
            For regression: actual ``horizon``-day forward log return.
            Index matches ``price_data``; the last ``horizon`` rows are NaN
            (no future available).
        """
        df = price_data.copy()
        df.columns = [c.lower() for c in df.columns]

        if "close" not in df.columns:
            raise KeyError(
                f"price_data must contain a 'close' column; found {list(df.columns)}"
            )

        close = df["close"].astype(float)

        # Forward return: log(close_{t+h} / close_t)
        forward_return = np.log(
            close.shift(-self.horizon) / close.replace(0.0, np.nan)
        )

        if self.mode == "classification":
            labels = (forward_return > 0).astype(float)
            labels[forward_return.isna()] = np.nan
        else:
            labels = forward_return

        labels.name = f"label_h{self.horizon}"
        return labels

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        val_features: Optional[pd.DataFrame] = None,
        val_labels: Optional[pd.Series] = None,
    ) -> "GradientBoostModel":
        """Train the LightGBM model.

        NaN rows (in features OR labels) are automatically dropped.  Early
        stopping is applied when validation data is provided.

        Parameters
        ----------
        features:
            Training feature matrix.
        labels:
            Target labels aligned with *features*.
        val_features:
            Optional hold-out feature matrix for early stopping.
        val_labels:
            Optional hold-out labels for early stopping.

        Returns
        -------
        self
            Returns the fitted model for chaining.

        Raises
        ------
        ImportError
            If lightgbm is not installed.
        ValueError
            If insufficient training samples remain after NaN removal.
        """
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError(
                "lightgbm is required. Install with: pip install lightgbm"
            ) from exc

        # --- Align and drop NaN ---
        X_train, y_train = self._clean_data(features, labels)

        min_samples = self._params.get("min_child_samples", 50)
        if len(X_train) < max(min_samples * 2, 100):
            raise ValueError(
                f"GradientBoostModel.train: only {len(X_train)} clean samples "
                f"for horizon={self.horizon}. Need at least "
                f"{max(min_samples * 2, 100)}."
            )

        self.feature_names = list(X_train.columns)

        # --- Build estimator params (strip sklearn-extra keys) ---
        estimator_params = {
            k: v for k, v in self._params.items()
            if k not in ("metric",)   # LGBMClassifier takes metric via fit kwargs
        }

        if self.mode == "classification":
            from lightgbm import LGBMClassifier
            estimator = LGBMClassifier(**estimator_params)
        else:
            from lightgbm import LGBMRegressor
            estimator = LGBMRegressor(**estimator_params)

        # --- Fit ---
        fit_kwargs: Dict = {}
        if val_features is not None and val_labels is not None:
            X_val, y_val = self._clean_data(val_features, val_labels)
            if len(X_val) >= min_samples:
                # Re-index validation features to match training feature names
                X_val = X_val.reindex(columns=self.feature_names, fill_value=0.0)
                fit_kwargs["eval_set"] = [(X_val.values, y_val.values)]
                fit_kwargs["callbacks"] = [
                    lgb.early_stopping(stopping_rounds=30, verbose=False),
                    lgb.log_evaluation(period=-1),
                ]

        logger.info(
            f"GradientBoostModel.train: horizon={self.horizon}, "
            f"mode={self.mode}, n_train={len(X_train)}, "
            f"n_features={len(self.feature_names)}"
        )

        estimator.fit(X_train.values, y_train.values, **fit_kwargs)

        self.model = estimator
        self._is_fitted = True

        # --- Store feature importances ---
        importances = estimator.feature_importances_
        self.feature_importances = dict(
            zip(self.feature_names, importances.tolist())
        )

        # --- Compute in-sample score ---
        if self.mode == "classification":
            from sklearn.metrics import roc_auc_score
            train_preds = estimator.predict_proba(X_train.values)[:, 1]
            try:
                self._train_score = float(roc_auc_score(y_train.values, train_preds))
            except Exception:
                self._train_score = None
        else:
            from sklearn.metrics import r2_score
            train_preds = estimator.predict(X_train.values)
            self._train_score = float(r2_score(y_train.values, train_preds))

        is_score_str = f"{self._train_score:.4f}" if self._train_score is not None else "N/A"
        logger.info(
            f"GradientBoostModel.train: done. "
            f"IS score={is_score_str}. "
            f"n_iterations={estimator.n_estimators_}"
        )

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Return prediction scores.

        For classification: probability of positive return in [0, 1].
        For regression: predicted forward return.

        Parameters
        ----------
        features:
            Feature DataFrame.  Unknown columns are ignored; missing columns
            are filled with 0.

        Returns
        -------
        np.ndarray
            1-D array of predictions, length = len(features).

        Raises
        ------
        RuntimeError
            If called before :meth:`train`.
        """
        if not self._is_fitted or self.model is None:
            raise RuntimeError(
                "GradientBoostModel has not been trained yet. Call train() first."
            )

        X = features.reindex(columns=self.feature_names, fill_value=0.0)
        # Fill any remaining NaN with column mean (or 0 as fallback)
        X = X.fillna(0.0)

        if self.mode == "classification":
            return self.model.predict_proba(X.values)[:, 1]
        else:
            return self.model.predict(X.values)

    def predict_single(self, features: pd.DataFrame) -> float:
        """Convenience wrapper: predict and return the last row score.

        Useful for live / incremental inference.

        Parameters
        ----------
        features:
            Feature row(s) — typically the latest available bar.

        Returns
        -------
        float
            Prediction score for the most recent row.
        """
        preds = self.predict(features)
        return float(preds[-1]) if len(preds) > 0 else np.nan

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_feature_importance(self, importance_type: str = "split") -> pd.Series:
        """Return feature importance sorted descending.

        Parameters
        ----------
        importance_type:
            ``"split"`` (number of times used in a split) or
            ``"gain"`` (total gain from splits).

        Returns
        -------
        pd.Series
            Feature importances, indexed by feature name, sorted descending.

        Raises
        ------
        RuntimeError
            If called before :meth:`train`.
        """
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model not fitted.")

        try:
            importances = self.model.booster_.feature_importance(
                importance_type=importance_type
            )
        except AttributeError:
            importances = self.model.feature_importances_

        s = pd.Series(
            importances,
            index=self.feature_names,
            name=f"importance_{importance_type}",
        )
        return s.sort_values(ascending=False)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist the fitted model and metadata to disk.

        Parameters
        ----------
        path:
            File path for the saved pickle.  Parent directories are created
            if absent.

        Raises
        ------
        RuntimeError
            If called before the model is fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted GradientBoostModel.")

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model":               self.model,
            "horizon":             self.horizon,
            "mode":                self.mode,
            "threshold":           self.threshold,
            "feature_names":       self.feature_names,
            "feature_importances": self.feature_importances,
            "params":              self._params,
            "train_score":         self._train_score,
            "val_score":           self._val_score,
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)

        logger.info(f"GradientBoostModel (horizon={self.horizon}) saved to {path}")

    def load(self, path: str) -> "GradientBoostModel":
        """Load a previously saved model from disk.

        Parameters
        ----------
        path:
            Path to the pickle file created by :meth:`save`.

        Returns
        -------
        self
            Populates this instance in-place and returns it for chaining.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"No saved GradientBoostModel at {path!r}")

        with open(path, "rb") as fh:
            payload = pickle.load(fh)

        self.model               = payload["model"]
        self.horizon             = payload["horizon"]
        self.mode                = payload["mode"]
        self.threshold           = payload["threshold"]
        self.feature_names       = payload["feature_names"]
        self.feature_importances = payload["feature_importances"]
        self._params             = payload["params"]
        self._train_score        = payload.get("train_score")
        self._val_score          = payload.get("val_score")
        self._is_fitted          = True

        logger.info(f"GradientBoostModel loaded from {path}")
        return self

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """True if the model has been successfully trained."""
        return self._is_fitted

    def summary(self) -> Dict:
        """Return a dict of model metadata for logging / auditing.

        Returns
        -------
        dict
            Keys: horizon, mode, n_features, n_iterations, train_score,
            val_score, top_features.
        """
        top = (
            sorted(self.feature_importances.items(), key=lambda x: x[1], reverse=True)[:10]
            if self.feature_importances
            else []
        )
        return {
            "horizon":       self.horizon,
            "mode":          self.mode,
            "n_features":    len(self.feature_names),
            "n_iterations":  getattr(self.model, "n_estimators_", None),
            "train_score":   self._train_score,
            "val_score":     self._val_score,
            "top_features":  top,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_data(
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare features and labels for training.

        Strategy:
          1. Align on common index.
          2. Drop rows where the label is NaN.
          3. Fill per-column NaN in features with the column median
             (forward-fill first to handle isolated NaN streaks).
          4. Drop any rows that still have all-NaN features after imputation.

        This preserves early rows whose long-lookback features are NaN but
        whose shorter-lookback features are valid, which maximises training
        sample size.

        Parameters
        ----------
        features, labels:
            Aligned feature matrix and target series.

        Returns
        -------
        (X_clean, y_clean)
            Cleaned and aligned DataFrames/Series.
        """
        # Align on common index
        common_idx = features.index.intersection(labels.index)
        X = features.loc[common_idx].copy()
        y = labels.loc[common_idx].copy()

        # Drop rows where the label is NaN
        valid_label = y.notna()
        X = X.loc[valid_label]
        y = y.loc[valid_label]

        # Forward-fill to propagate last valid observation
        X = X.ffill()

        # Fill remaining NaN with column median
        col_medians = X.median()
        X = X.fillna(col_medians)

        # Final safety: drop rows that are still entirely NaN
        not_all_nan = ~X.isna().all(axis=1)
        X = X.loc[not_all_nan]
        y = y.loc[not_all_nan]

        # Fill any residual NaN with 0 (should be rare at this point)
        X = X.fillna(0.0)

        return X, y
