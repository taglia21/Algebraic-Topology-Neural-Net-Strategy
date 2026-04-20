"""
ml/models/gradient_boost.py
============================
LightGBM model wrapper for multi-horizon return prediction.

The model predicts:
  - direction (classification): +1 if forward return > 0, else 0
  - magnitude (regression): actual forward return (optional)

Three instances are typically created — one per horizon (1d, 5d, 20d).

Includes state-of-the-art improvements from Lopez de Prado's
"Advances in Financial Machine Learning" (AFML):
  - Triple-Barrier Labeling (AFML Ch. 3)
  - Sample Uniqueness Weighting (AFML Ch. 4)
  - Improved early stopping and regularization
  - Optional monotone constraints for directional features

Usage
-----
    from ml.models.gradient_boost import GradientBoostModel

    model = GradientBoostModel(horizon=5)
    labels = model.prepare_labels(price_data)                     # triple-barrier by default
    labels = model.prepare_labels(price_data, labeling_method="binary")  # legacy
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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default LightGBM hyperparameters (mirror core.config.LightGBMParams)
# ---------------------------------------------------------------------------
_DEFAULT_PARAMS: Dict = {
    "objective":          "binary",
    "metric":             "binary_logloss",
    "max_depth":          6,
    "num_leaves":         31,
    "learning_rate":      0.05,
    "min_child_samples":  50,
    "n_estimators":       300,
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "reg_alpha":          0.1,
    "reg_lambda":         0.1,
    "random_state":       42,
    "n_jobs":             -1,
    "verbose":            -1,
    "feature_pre_filter": False,   # do not drop features before training (AFML improvement)
}

# ---------------------------------------------------------------------------
# Monotone constraint registry
# Maps substrings of feature names to LightGBM monotone_constraint values:
#   +1 = monotone increasing, -1 = monotone decreasing, 0 = unconstrained
#
# Economic rationale:
#   rsi_*     → mean-reversion: higher RSI → lower future return  (-1)
#   obv_*     → trend-following: rising OBV → higher future return (+1)
#   macd_*    → trend-following: rising MACD → higher future return (+1)
#   bb_pct_*  → mean-reversion: higher % position in band → lower return (-1)
# ---------------------------------------------------------------------------
_MONOTONE_FEATURE_RULES: Dict[str, int] = {
    "rsi":    -1,
    "bb_pct": -1,
    "macd":   +1,
    "obv":    +1,
}


def _build_monotone_constraints(feature_names: List[str]) -> Optional[List[int]]:
    """Build a monotone constraints list aligned to *feature_names*.

    Returns ``None`` when no known monotone features are detected (avoids
    passing an all-zero constraint list to LightGBM unnecessarily).

    Parameters
    ----------
    feature_names:
        Ordered list of feature column names as passed to LightGBM.

    Returns
    -------
    list of int or None
        Per-feature constraint values (+1, -1, or 0), or None if no
        constraints apply.
    """
    constraints = [0] * len(feature_names)
    any_constrained = False
    for i, name in enumerate(feature_names):
        name_lower = name.lower()
        for pattern, direction in _MONOTONE_FEATURE_RULES.items():
            if pattern in name_lower:
                constraints[i] = direction
                any_constrained = True
                break
    return constraints if any_constrained else None


class GradientBoostModel:
    """LightGBM model for return prediction at multiple horizons.

    Supports both classification (direction) and regression (magnitude).
    Classification mode is the default and is used by the pipeline for
    generating trade signals.

    Incorporates AFML improvements:
    - Triple-Barrier Labeling for financially meaningful label construction
    - Sample Uniqueness Weighting to de-weight overlapping label windows
    - Enhanced regularization and early stopping
    - Optional monotone constraints for known directional features

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
        labeling_method: str = "triple_barrier",
        pt_sl: Tuple[float, float] = (1.5, 1.5),
        vol_lookback: int = 20,
    ) -> pd.Series:
        """Create forward-return labels for the prediction horizon.

        Supports two labeling methods:

        **triple_barrier** (default, AFML Ch. 3):
            For each bar *t*, sets three barriers:
            - *Vertical barrier*: ``horizon`` days ahead.
            - *Upper (profit-taking) barrier*: +``pt_sl[0]`` × daily_vol above close_t.
            - *Lower (stop-loss) barrier*: -``pt_sl[1]`` × daily_vol below close_t.
            The label is determined by which barrier is touched first:
            +1 (upper hit first) → bullish, -1 (lower hit first) → bearish,
            0 (vertical/timeout) → no signal.
            For classification mode, the mapping is {+1: 1, 0: 0, -1: 0}.

        **binary** (legacy):
            Labels are 1 if the ``horizon``-day forward log return > 0, else 0.
            Regression mode always returns the raw forward log return.

        Parameters
        ----------
        price_data:
            DataFrame with at minimum a ``close`` column.
        symbol:
            Optional ticker string (reserved for MultiIndex compatibility).
        labeling_method:
            ``"triple_barrier"`` (default) or ``"binary"`` for backward compat.
        pt_sl:
            (profit_take_multiplier, stop_loss_multiplier) applied to daily_vol.
            Symmetric at (1.5, 1.5) by default.
        vol_lookback:
            Rolling window (days) for computing daily_vol as std of log returns.
            Default 20.

        Returns
        -------
        pd.Series
            For classification + triple_barrier: {0, 1} (1 = upper barrier hit).
            For classification + binary: {0, 1} (1 = positive forward return).
            For regression: actual ``horizon``-day forward log return.
            Index matches ``price_data``; trailing rows without a full look-ahead
            window are NaN.
        """
        df = price_data.copy()
        df.columns = [c.lower() for c in df.columns]

        if "close" not in df.columns:
            raise KeyError(
                f"price_data must contain a 'close' column; found {list(df.columns)}"
            )

        close = df["close"].astype(float)

        # Regression mode always uses raw forward log return
        if self.mode == "regression":
            forward_return = np.log(
                close.shift(-self.horizon) / close.replace(0.0, np.nan)
            )
            forward_return.name = f"label_h{self.horizon}"
            return forward_return

        # ----------------------------------------------------------------
        # Binary labeling (backward-compatible legacy method)
        # ----------------------------------------------------------------
        if labeling_method == "binary":
            forward_return = np.log(
                close.shift(-self.horizon) / close.replace(0.0, np.nan)
            )
            labels = (forward_return > 0).astype(float)
            labels[forward_return.isna()] = np.nan
            labels.name = f"label_h{self.horizon}"
            return labels

        # ----------------------------------------------------------------
        # Triple-Barrier Labeling (AFML Ch. 3)
        # ----------------------------------------------------------------
        if labeling_method != "triple_barrier":
            raise ValueError(
                f"labeling_method must be 'triple_barrier' or 'binary'; got {labeling_method!r}"
            )

        log_returns = np.log(close / close.shift(1))

        # Daily volatility: rolling std of log returns (annualised window)
        daily_vol = log_returns.rolling(window=vol_lookback, min_periods=vol_lookback // 2).std()

        labels = self._apply_triple_barrier(close, daily_vol, pt_sl)

        # Classification mapping: +1 → 1 (bullish), 0 and -1 → 0 (neutral/bearish)
        mapped = (labels == 1).astype(float)
        mapped[labels.isna()] = np.nan
        mapped.name = f"label_h{self.horizon}"
        return mapped

    def _apply_triple_barrier(
        self,
        close: pd.Series,
        daily_vol: pd.Series,
        pt_sl: Tuple[float, float],
    ) -> pd.Series:
        """Vectorised triple-barrier label assignment.

        For each bar *t*, evaluates a window of length ``horizon`` days.
        Within that window, determines which barrier is touched first:

        - Upper barrier: cumulative log return ≥ +``pt_sl[0]`` × daily_vol[t]
        - Lower barrier: cumulative log return ≤ -``pt_sl[1]`` × daily_vol[t]
        - Vertical barrier: end of window (neither price barrier touched)

        Uses numpy array slicing for speed; falls back to NaN for bars
        where the full ``horizon``-day window is unavailable.

        Parameters
        ----------
        close:
            Closing price series (float, DatetimeIndex or RangeIndex).
        daily_vol:
            Per-bar daily volatility estimate (same index as *close*).
        pt_sl:
            (profit_multiplier, stop_multiplier) relative to daily_vol.

        Returns
        -------
        pd.Series
            Raw triple-barrier labels: +1 (upper hit), -1 (lower hit), 0 (timeout).
            NaN for the last ``horizon`` bars and wherever vol is NaN.
        """
        n = len(close)
        close_arr = close.values.astype(np.float64)
        vol_arr   = daily_vol.values.astype(np.float64)

        # Pre-compute log-price array for fast cumulative-return slicing
        with np.errstate(divide="ignore", invalid="ignore"):
            log_close = np.where(close_arr > 0, np.log(close_arr), np.nan)

        raw_labels = np.full(n, np.nan, dtype=np.float64)

        # Only label bars where we have both vol and a full forward window
        valid_end = n - self.horizon  # exclusive upper bound
        for t in range(valid_end):
            if np.isnan(vol_arr[t]) or vol_arr[t] == 0.0:
                continue

            upper_thresh =  pt_sl[0] * vol_arr[t]
            lower_thresh = -pt_sl[1] * vol_arr[t]

            # Cumulative log returns over the look-ahead window [t+1, t+horizon]
            window_log_close = log_close[t + 1 : t + self.horizon + 1]
            if np.any(np.isnan(window_log_close)):
                continue

            cum_ret = window_log_close - log_close[t]

            # Which step first crosses a barrier?
            upper_cross = cum_ret >= upper_thresh
            lower_cross = cum_ret <= lower_thresh

            first_upper = int(np.argmax(upper_cross)) if upper_cross.any() else self.horizon
            first_lower = int(np.argmax(lower_cross)) if lower_cross.any() else self.horizon

            if not upper_cross.any() and not lower_cross.any():
                # Vertical barrier: timeout
                raw_labels[t] = 0.0
            elif first_upper <= first_lower:
                raw_labels[t] = 1.0
            else:
                raw_labels[t] = -1.0

        return pd.Series(raw_labels, index=close.index, name="triple_barrier_label")

    # ------------------------------------------------------------------
    # Sample uniqueness weighting (AFML Ch. 4)
    # ------------------------------------------------------------------

    def _compute_sample_weights(
        self,
        labels: pd.Series,
        price_data_index: Optional[pd.Index] = None,
    ) -> np.ndarray:
        """Compute sample uniqueness weights for the training set.

        Each sample *t* spans a return window from *t* to *t + horizon*.
        Two samples are "overlapping" if their return windows share any bars.
        The weight for each sample is inversely proportional to the average
        number of simultaneously open return windows (average uniqueness).

        Weights are normalised to sum to the number of samples so that the
        total effective training mass is preserved.

        Parameters
        ----------
        labels:
            Label series whose index is used to identify the sample bar
            positions.  Labels must be pre-cleaned (no NaN).
        price_data_index:
            Optional original price index (unused; reserved for path-based
            uniqueness variants).  When ``None``, positional integer
            counting is used.

        Returns
        -------
        np.ndarray
            1-D float array of sample weights, length = len(labels).
        """
        n = len(labels)
        if n == 0:
            return np.ones(0, dtype=np.float64)

        horizon = self.horizon

        # Build a concurrency matrix: for each bar b in [0, n + horizon),
        # count how many label windows [t, t + horizon] cover bar b.
        # We work in positional integers aligned to the label index.
        # concurrency[t] = average number of labels whose window overlaps
        # with label t's window.

        # For label at position t, window covers bars t..t+horizon-1 (horizon bars).
        # Overlap between windows of t1 and t2 exists iff:
        #   max(t1, t2) < min(t1+horizon, t2+horizon)  →  |t1 - t2| < horizon
        # So average concurrency for sample t = number of other labels j such that
        # |t - j| < horizon, divided by horizon, plus 1 for itself.

        positions = np.arange(n, dtype=np.float64)

        # Efficient vectorised concurrency computation using cumsum trick.
        # Build indicator array of window starts/ends over extended timeline.
        total_bars = n + horizon
        counts = np.zeros(total_bars, dtype=np.float64)
        for t in range(n):
            counts[t : t + horizon] += 1.0

        # Average uniqueness per sample = 1 / mean(counts over its own window)
        avg_uniqueness = np.empty(n, dtype=np.float64)
        for t in range(n):
            window_counts = counts[t : t + horizon]
            # Avoid division by zero (shouldn't happen but guard anyway)
            mean_count = window_counts.mean()
            avg_uniqueness[t] = 1.0 / mean_count if mean_count > 0 else 1.0

        # Normalise so weights sum to n (preserves effective sample count)
        total = avg_uniqueness.sum()
        if total > 0:
            weights = avg_uniqueness * (n / total)
        else:
            weights = np.ones(n, dtype=np.float64)

        return weights

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
        stopping is applied when validation data is provided (50 rounds).

        Sample uniqueness weights (AFML Ch. 4) are computed and passed to
        LightGBM's ``fit()`` to down-weight clustered/overlapping labels.

        Monotone constraints are injected automatically when feature names
        match known directional patterns (e.g., rsi_14, macd_signal).

        Dynamic ``min_data_in_leaf`` is set to ``max(20, n_train // 100)``
        for data-adaptive regularization.

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

        # --- Build estimator params (strip sklearn-incompatible keys) ---
        estimator_params = {
            k: v for k, v in self._params.items()
            if k not in ("metric",)
        }

        # Data-adaptive leaf regularization (AFML improvement)
        # Prevents overfitting on large/small datasets alike
        dynamic_min_leaf = max(20, len(X_train) // 100)
        estimator_params["min_data_in_leaf"] = dynamic_min_leaf

        # Monotone constraints (optional; only when features match known patterns)
        monotone = _build_monotone_constraints(self.feature_names)
        if monotone is not None:
            estimator_params["monotone_constraints"] = monotone
            logger.info(
                f"GradientBoostModel.train: applying monotone constraints on "
                f"{sum(c != 0 for c in monotone)} feature(s)"
            )

        if self.mode == "classification":
            from lightgbm import LGBMClassifier
            estimator = LGBMClassifier(**estimator_params)
        else:
            from lightgbm import LGBMRegressor
            estimator = LGBMRegressor(**estimator_params)

        # --- Sample uniqueness weights (AFML Ch. 4) ---
        sample_weights = self._compute_sample_weights(y_train)

        # --- Fit ---
        fit_kwargs: Dict = {"sample_weight": sample_weights}

        if val_features is not None and val_labels is not None:
            X_val, y_val = self._clean_data(val_features, val_labels)
            if len(X_val) >= min_samples:
                X_val = X_val.reindex(columns=self.feature_names, fill_value=0.0)
                fit_kwargs["eval_set"] = [(X_val.values, y_val.values)]
                fit_kwargs["callbacks"] = [
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=-1),
                ]

        logger.info(
            f"GradientBoostModel.train: horizon={self.horizon}, "
            f"mode={self.mode}, n_train={len(X_train)}, "
            f"n_features={len(self.feature_names)}, "
            f"min_data_in_leaf={dynamic_min_leaf}"
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
            "horizon":      self.horizon,
            "mode":         self.mode,
            "n_features":   len(self.feature_names),
            "n_iterations": getattr(self.model, "n_estimators_", None),
            "train_score":  self._train_score,
            "val_score":    self._val_score,
            "top_features": top,
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
