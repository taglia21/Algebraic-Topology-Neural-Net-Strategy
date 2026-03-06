"""
ml/pipeline.py
==============
ML training orchestrator for the ATNN quantitative trading system.

Coordinates: feature engineering → model training (3 horizons) →
walk-forward validation → meta-learner (Ridge ensemble) → live prediction.

Drift detection triggers automatic retraining when statistical properties
of the feature distribution shift materially.

Usage
-----
    from ml.feature_engine import FeatureEngine
    from ml.pipeline import MLPipeline
    from core.config import get_config

    cfg    = get_config()
    engine = FeatureEngine(spy_data=spy_df)
    pipe   = MLPipeline(feature_engine=engine, config=cfg)

    report = pipe.train_all(price_data)
    preds  = pipe.predict(recent_price_data, regime_state)
    # preds → {"AAPL": {"score": 0.63, "confidence": 0.71, "horizon": 5}, ...}
"""

from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from core.config import get_config, Config
from core.regime_detector import Regime
from ml.feature_engine import FeatureEngine
from ml.models.gradient_boost import GradientBoostModel
from ml.validation import validate_model, walk_forward_validate

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Mapping from Regime enum → numeric encoding for the meta-learner
_REGIME_ENCODING: Dict[str, float] = {
    Regime.BULL.value:     1.0,
    Regime.SIDEWAYS.value: 0.0,
    Regime.BEAR.value:    -1.0,
    Regime.UNKNOWN.value:  0.0,
}

# Population Stability Index threshold for drift detection
_PSI_THRESHOLD: float = 0.2

# Minimum bars required to run prediction
_MIN_PREDICT_BARS: int = 201


# ===========================================================================
# Drift detector (Population Stability Index)
# ===========================================================================

def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Compute the Population Stability Index between two distributions.

    PSI < 0.10 → no change
    PSI 0.10–0.25 → moderate shift
    PSI > 0.25 → significant shift (retrain recommended)

    Parameters
    ----------
    expected:
        Reference distribution (e.g., training set feature values).
    actual:
        Current distribution (e.g., live window feature values).
    bins:
        Number of equal-width histogram bins.

    Returns
    -------
    float
        PSI value.
    """
    expected = expected[np.isfinite(expected)]
    actual   = actual[np.isfinite(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Use training-set edges
    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())
    if min_val == max_val:
        return 0.0

    edges = np.linspace(min_val, max_val, bins + 1)

    exp_counts, _ = np.histogram(expected, bins=edges)
    act_counts, _ = np.histogram(actual,   bins=edges)

    # Smooth to avoid log(0)
    exp_pct = (exp_counts + 0.5) / (len(expected) + 0.5 * bins)
    act_pct = (act_counts + 0.5) / (len(actual)   + 0.5 * bins)

    psi = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
    return psi


# ===========================================================================
# MLPipeline
# ===========================================================================

class MLPipeline:
    """Orchestrates the full ML workflow: features → training → validation → prediction.

    The pipeline trains three LightGBM base models at horizons 1d, 5d, and
    20d, then combines their scores with the current market regime via a
    Ridge meta-learner to produce a single composite signal per symbol.

    Parameters
    ----------
    feature_engine:
        A fitted (or configurable) :class:`~ml.feature_engine.FeatureEngine`
        instance.
    config:
        System-level configuration (:class:`~core.config.Config`).  If None,
        the singleton is loaded via :func:`~core.config.get_config`.
    model_dir:
        Directory for persisting trained models.  Defaults to
        ``config.ml.model_dir``.
    """

    def __init__(
        self,
        feature_engine: FeatureEngine,
        config: Optional[Config] = None,
        model_dir: Optional[str] = None,
    ) -> None:
        self.feature_engine: FeatureEngine = feature_engine
        self.config: Config = config or get_config()

        self.model_dir: str = model_dir or self.config.ml.model_dir
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)

        # Base models — one per prediction horizon
        self.models: Dict[int, GradientBoostModel] = {}

        # Meta-learner: Ridge regression over [score_1d, score_5d, score_20d, regime]
        self.meta_learner: Optional[Ridge] = None
        self._meta_scaler: StandardScaler  = StandardScaler()
        self._meta_fitted: bool = False

        # Validation reports
        self._validation_reports: Dict[int, Dict] = {}
        self._meta_validation_report: Dict = {}

        # Drift monitoring: store training feature statistics
        self._train_feature_stats: Dict[str, Tuple[np.ndarray, int]] = {}

        # Timestamps
        self._last_train_time: Optional[float] = None
        self._train_data_hash: Optional[str]   = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_all(
        self,
        price_data: pd.DataFrame,
        symbol: Optional[str] = None,
        run_validation: bool = True,
    ) -> Dict:
        """Train all base models (1d, 5d, 20d horizons) plus the meta-learner.

        Parameters
        ----------
        price_data:
            OHLCV DataFrame for a single symbol, datetime-indexed.
        symbol:
            Optional ticker for logging.
        run_validation:
            If True, run walk-forward validation for each base model and
            the meta-learner (recommended for production).

        Returns
        -------
        dict
            ``{horizon: validation_report}`` for each base model, plus
            ``"meta"`` key for the meta-learner report.  Also includes
            ``"summary"`` with pass/fail status.

        Raises
        ------
        ValueError
            If price_data has insufficient history.
        """
        ml_cfg = self.config.ml
        horizons: List[int] = ml_cfg.horizons  # [1, 5, 20]

        logger.info(
            f"MLPipeline.train_all: symbol={symbol}, "
            f"horizons={horizons}, bars={len(price_data)}"
        )

        # --- Compute features ---
        logger.info("MLPipeline.train_all: computing features ...")
        features = self.feature_engine.compute_features(price_data, symbol=symbol)

        # Store training stats for drift detection
        for col in features.columns:
            vals = features[col].dropna().values
            if len(vals) > 0:
                self._train_feature_stats[col] = (vals, len(vals))

        reports: Dict = {}

        # --- Train base models ---
        base_model_scores: Dict[int, pd.Series] = {}  # for meta-learner training

        for horizon in horizons:
            logger.info(f"MLPipeline.train_all: training horizon={horizon}d ...")

            model = GradientBoostModel(
                horizon = horizon,
                params  = {
                    "max_depth":         ml_cfg.model_params.max_depth,
                    "num_leaves":        ml_cfg.model_params.num_leaves,
                    "learning_rate":     ml_cfg.model_params.learning_rate,
                    "min_child_samples": ml_cfg.model_params.min_child_samples,
                    "n_estimators":      ml_cfg.model_params.n_estimators,
                    "subsample":         ml_cfg.model_params.subsample,
                    "colsample_bytree":  ml_cfg.model_params.colsample_bytree,
                    "reg_alpha":         ml_cfg.model_params.reg_alpha,
                    "reg_lambda":        ml_cfg.model_params.reg_lambda,
                    "random_state":      ml_cfg.model_params.random_state,
                },
                mode = "classification",
            )

            labels = model.prepare_labels(price_data)

            # --- Walk-forward split for final model training ---
            train_end = len(features) - ml_cfg.train_window_days // 10  # ~10% holdout
            if train_end < 100:
                train_end = len(features)

            X_train = features.iloc[:train_end]
            y_train = labels.iloc[:train_end]
            X_val   = features.iloc[train_end:]
            y_val   = labels.iloc[train_end:]

            model.train(X_train, y_train, X_val, y_val)
            self.models[horizon] = model

            # Save to disk
            model_path = os.path.join(self.model_dir, f"lgbm_h{horizon}.pkl")
            model.save(model_path)

            # Collect IS scores for meta-learner training data
            try:
                all_preds = model.predict(features)
                base_model_scores[horizon] = pd.Series(
                    all_preds, index=features.index, name=f"score_{horizon}d"
                )
            except Exception as exc:
                logger.warning(f"Could not collect IS scores for horizon {horizon}: {exc}")

            # --- Validation ---
            if run_validation:
                try:
                    def _factory(h=horizon, p=model._params) -> GradientBoostModel:
                        return GradientBoostModel(horizon=h, params=p, mode="classification")

                    val_report = validate_model(
                        model_factory = _factory,
                        features      = features,
                        labels        = labels,
                        config        = {
                            "train_window":      ml_cfg.train_window_days,
                            "test_window":       21,
                            "step":              21,
                            "min_windows":       6,  # relaxed for training runs
                            "cpcv_n_groups":     self.config.backtest.cpcv_groups,
                            "cpcv_purge_window": self.config.backtest.cpcv_purge_days,
                        },
                    )
                    self._validation_reports[horizon] = val_report
                    reports[horizon] = val_report
                    logger.info(
                        f"MLPipeline.train_all: horizon={horizon}d validation "
                        f"recommendation={val_report['recommendation']}"
                    )
                except Exception as exc:
                    logger.warning(
                        f"MLPipeline.train_all: validation failed for horizon {horizon}: {exc}"
                    )
                    reports[horizon] = {"error": str(exc), "pass": False}

        # --- Train meta-learner ---
        logger.info("MLPipeline.train_all: training meta-learner ...")
        try:
            meta_report = self._train_meta_learner(
                price_data        = price_data,
                base_model_scores = base_model_scores,
                run_validation    = run_validation,
            )
            reports["meta"] = meta_report
        except Exception as exc:
            logger.warning(f"MLPipeline.train_all: meta-learner training failed: {exc}")
            reports["meta"] = {"error": str(exc)}

        self._last_train_time = time.time()

        # --- Summary ---
        pass_count = sum(
            1 for h in horizons
            if reports.get(h, {}).get("overall_pass", False)
        )
        reports["summary"] = {
            "horizons_trained":       horizons,
            "base_models_passing":    pass_count,
            "total_base_models":      len(horizons),
            "meta_learner_fitted":    self._meta_fitted,
        }

        logger.info(
            f"MLPipeline.train_all: complete. "
            f"{pass_count}/{len(horizons)} base models passed validation."
        )

        return reports

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        price_data: pd.DataFrame,
        regime_state: Any,
        symbol: Optional[str] = None,
    ) -> Dict:
        """Generate predictions from all models + meta-learner ensemble.

        Parameters
        ----------
        price_data:
            Recent OHLCV data for the symbol.
        regime_state:
            Current :class:`~core.regime_detector.RegimeState` or
            :class:`~core.regime_detector.Regime` enum.
        symbol:
            Optional ticker label for the returned dict key.

        Returns
        -------
        dict
            ``{symbol: {"score": float, "confidence": float, "horizon": int,
                         "scores_by_horizon": dict, "meta_score": float}}``

        Notes
        -----
        Returns an empty dict if no models are fitted or price data is
        insufficient.
        """
        if not self.models:
            logger.warning("MLPipeline.predict: no models fitted.")
            return {}

        if len(price_data) < _MIN_PREDICT_BARS:
            logger.warning(
                f"MLPipeline.predict: only {len(price_data)} bars; "
                f"need {_MIN_PREDICT_BARS}."
            )
            return {}

        # Compute features
        try:
            features = self.feature_engine.compute_features(price_data, symbol=symbol)
        except Exception as exc:
            logger.warning(f"MLPipeline.predict: feature computation failed: {exc}")
            return {}

        if features.empty:
            return {}

        # Use the last complete row
        last_row = features.dropna(how="all").iloc[[-1]]
        if last_row.empty:
            return {}

        # --- Base model predictions ---
        horizon_scores: Dict[int, float] = {}
        for horizon, model in self.models.items():
            try:
                score = model.predict_single(last_row)
                horizon_scores[horizon] = score
            except Exception as exc:
                logger.warning(
                    f"MLPipeline.predict: horizon {horizon} prediction failed: {exc}"
                )
                horizon_scores[horizon] = float("nan")

        valid_scores = [s for s in horizon_scores.values() if not math.isnan(s)]
        if not valid_scores:
            return {}

        # Regime encoding
        regime_val = self._encode_regime(regime_state)

        # --- Meta-learner prediction ---
        meta_score: Optional[float] = None
        if self._meta_fitted and self.meta_learner is not None:
            try:
                meta_input = self._build_meta_input(horizon_scores, regime_val)
                meta_scaled = self._meta_scaler.transform(meta_input)
                meta_score  = float(self.meta_learner.predict(meta_scaled)[0])
            except Exception as exc:
                logger.debug(f"Meta-learner prediction failed: {exc}")

        # --- Composite signal ---
        # Use meta-learner if available; otherwise average base model scores
        if meta_score is not None:
            final_score = meta_score
        else:
            final_score = float(np.nanmean(valid_scores))

        # Confidence = fraction of base models that agree on direction
        directions = [int(s > 0.5) for s in valid_scores]
        mode_dir   = int(np.round(np.mean(directions)))
        agreement  = sum(1 for d in directions if d == mode_dir) / len(directions)

        key = symbol or "default"
        return {
            key: {
                "score":             final_score,
                "confidence":        agreement,
                "horizon":           5,  # primary horizon
                "scores_by_horizon": horizon_scores,
                "meta_score":        meta_score,
                "regime":            regime_val,
            }
        }

    # ------------------------------------------------------------------
    # Drift detection and retraining
    # ------------------------------------------------------------------

    def retrain_if_needed(
        self,
        price_data: pd.DataFrame,
        symbol: Optional[str] = None,
        force: bool = False,
    ) -> bool:
        """Check if feature drift is detected; retrain if needed.

        Uses Population Stability Index (PSI) on current feature values
        vs. training-set statistics.  Retrains if PSI > threshold for
        more than 20% of features.

        Parameters
        ----------
        price_data:
            Recent OHLCV data.
        symbol:
            Optional ticker.
        force:
            Force retrain regardless of drift check.

        Returns
        -------
        bool
            True if the model was retrained.
        """
        ml_cfg = self.config.ml

        # Check scheduled retrain frequency
        if not force and self._last_train_time is not None:
            elapsed_days = (time.time() - self._last_train_time) / 86400
            if elapsed_days < ml_cfg.retrain_freq_days:
                return False

        # Check drift if we have training stats
        if not force and self._train_feature_stats:
            try:
                features = self.feature_engine.compute_features(price_data, symbol=symbol)
                n_drifted = 0
                n_total   = 0
                for col, (train_vals, _) in self._train_feature_stats.items():
                    if col not in features.columns:
                        continue
                    curr_vals = features[col].dropna().values
                    if len(curr_vals) < 20:
                        continue
                    psi_val = _psi(train_vals, curr_vals)
                    n_total += 1
                    if psi_val > _PSI_THRESHOLD:
                        n_drifted += 1

                drift_pct = n_drifted / n_total if n_total > 0 else 0.0
                if drift_pct < 0.20:
                    logger.info(
                        f"MLPipeline.retrain_if_needed: drift_pct={drift_pct:.2%} "
                        f"< 20%, no retrain needed."
                    )
                    return False
                else:
                    logger.info(
                        f"MLPipeline.retrain_if_needed: drift_pct={drift_pct:.2%} "
                        f">= 20%, triggering retrain."
                    )
            except Exception as exc:
                logger.warning(f"Drift check failed: {exc}; proceeding with retrain.")

        # Retrain
        try:
            self.train_all(price_data, symbol=symbol, run_validation=False)
            return True
        except Exception as exc:
            logger.error(f"MLPipeline.retrain_if_needed: retrain failed: {exc}")
            return False

    # ------------------------------------------------------------------
    # Validation report
    # ------------------------------------------------------------------

    def get_validation_report(self) -> Dict:
        """Return the latest validation metrics for all models.

        Returns
        -------
        dict
            ``{horizon: report_dict}`` for each base model, plus
            ``"meta"`` key for the meta-learner report.
        """
        report = dict(self._validation_reports)
        if self._meta_validation_report:
            report["meta"] = self._meta_validation_report
        return report

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------

    def save_all(self, directory: Optional[str] = None) -> None:
        """Save all fitted models to *directory*.

        Parameters
        ----------
        directory:
            Target directory.  Defaults to ``self.model_dir``.
        """
        save_dir = directory or self.model_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        for horizon, model in self.models.items():
            path = os.path.join(save_dir, f"lgbm_h{horizon}.pkl")
            model.save(path)

        if self._meta_fitted and self.meta_learner is not None:
            import pickle
            meta_payload = {
                "meta_learner": self.meta_learner,
                "meta_scaler":  self._meta_scaler,
                "horizons":     list(self.models.keys()),
            }
            meta_path = os.path.join(save_dir, "meta_learner.pkl")
            with open(meta_path, "wb") as fh:
                import pickle as pkl
                pkl.dump(meta_payload, fh)
            logger.info(f"Meta-learner saved to {meta_path}")

    def load_all(self, directory: Optional[str] = None) -> None:
        """Load all previously saved models from *directory*.

        Parameters
        ----------
        directory:
            Source directory.  Defaults to ``self.model_dir``.

        Raises
        ------
        FileNotFoundError
            If no model files are found in the directory.
        """
        load_dir = directory or self.model_dir
        loaded = 0

        for horizon in self.config.ml.horizons:
            path = os.path.join(load_dir, f"lgbm_h{horizon}.pkl")
            if Path(path).exists():
                model = GradientBoostModel(horizon=horizon)
                model.load(path)
                self.models[horizon] = model
                loaded += 1
            else:
                logger.warning(f"MLPipeline.load_all: model file not found at {path}")

        meta_path = os.path.join(load_dir, "meta_learner.pkl")
        if Path(meta_path).exists():
            import pickle as pkl
            with open(meta_path, "rb") as fh:
                meta_payload = pkl.load(fh)
            self.meta_learner    = meta_payload["meta_learner"]
            self._meta_scaler    = meta_payload["meta_scaler"]
            self._meta_fitted    = True
            logger.info(f"Meta-learner loaded from {meta_path}")

        if loaded == 0:
            raise FileNotFoundError(
                f"No model files found in {load_dir!r}. "
                "Call train_all() first."
            )
        logger.info(f"MLPipeline.load_all: loaded {loaded} base models from {load_dir}.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_meta_learner(
        self,
        price_data: pd.DataFrame,
        base_model_scores: Dict[int, pd.Series],
        run_validation: bool = True,
    ) -> Dict:
        """Train a Ridge regression meta-learner combining base model scores
        and regime state.

        The meta-learner has at most 4 inputs (score_1d, score_5d, score_20d,
        regime) to prevent overfitting (spec: 3–5 features only).

        Parameters
        ----------
        price_data:
            Raw price data (used to construct pseudo regime labels).
        base_model_scores:
            Dict of {horizon: pd.Series of IS scores}.
        run_validation:
            Run a walk-forward validation for the meta-learner if True.

        Returns
        -------
        dict
            Meta-learner training report.
        """
        if not base_model_scores:
            logger.warning("_train_meta_learner: no base model scores available.")
            return {"error": "no base model scores", "fitted": False}

        # Align scores to a common index
        score_frames: List[pd.Series] = []
        for horizon in sorted(base_model_scores.keys()):
            s = base_model_scores[horizon].rename(f"score_{horizon}d")
            score_frames.append(s)

        meta_X = pd.concat(score_frames, axis=1).dropna()

        # Add regime column (use a static neutral 0 for IS training;
        # in production, the live regime is passed to predict())
        meta_X["regime"] = 0.0

        # Use the 5-day forward return as the meta-learner target
        primary_model = self.models.get(5) or next(iter(self.models.values()))
        meta_y = primary_model.prepare_labels(price_data, symbol=None)
        meta_y = meta_y.reindex(meta_X.index).dropna()
        meta_X = meta_X.reindex(meta_y.index).dropna()

        if len(meta_X) < 50:
            logger.warning(
                f"_train_meta_learner: only {len(meta_X)} aligned samples; skipping."
            )
            return {"error": "insufficient aligned samples", "fitted": False}

        # Scale and fit Ridge
        X_scaled = self._meta_scaler.fit_transform(meta_X.values)
        self.meta_learner = Ridge(alpha=1.0, fit_intercept=True)
        self.meta_learner.fit(X_scaled, meta_y.values)
        self._meta_fitted = True

        # Walk-forward validation for meta-learner
        meta_report: Dict = {"fitted": True}
        if run_validation and len(meta_X) >= 100:
            try:
                meta_scaler_copy = StandardScaler()

                def _meta_factory() -> _MetaLearnerWrapper:
                    return _MetaLearnerWrapper(
                        scaler=meta_scaler_copy,
                        alpha=1.0,
                    )

                wfv_meta = walk_forward_validate(
                    model_factory = _meta_factory,
                    features      = meta_X,
                    labels        = meta_y,
                    train_window  = min(200, len(meta_X) // 3),
                    test_window   = 21,
                    step          = 21,
                    min_windows   = 4,
                    verbose       = False,
                )
                meta_report["walk_forward"] = wfv_meta
                meta_report["pass"] = wfv_meta.get("pass", False)
                self._meta_validation_report = meta_report
            except Exception as exc:
                logger.warning(f"Meta-learner validation failed: {exc}")
                meta_report["validation_error"] = str(exc)

        logger.info(
            f"_train_meta_learner: fitted Ridge on {len(meta_X)} samples "
            f"with {meta_X.shape[1]} features. Coefs={self.meta_learner.coef_}"
        )
        return meta_report

    @staticmethod
    def _encode_regime(regime_state: Any) -> float:
        """Encode a regime state to a numeric value.

        Parameters
        ----------
        regime_state:
            :class:`~core.regime_detector.RegimeState`,
            :class:`~core.regime_detector.Regime`, or str.

        Returns
        -------
        float
            Numeric regime code: BULL=1.0, SIDEWAYS=0.0, BEAR=−1.0.
        """
        if regime_state is None:
            return 0.0
        if isinstance(regime_state, str):
            return _REGIME_ENCODING.get(regime_state.upper(), 0.0)
        if hasattr(regime_state, "regime"):
            # RegimeState dataclass
            regime_str = str(regime_state.regime.value)
            return _REGIME_ENCODING.get(regime_str, 0.0)
        if hasattr(regime_state, "value"):
            # Regime enum
            return _REGIME_ENCODING.get(str(regime_state.value), 0.0)
        return 0.0

    @staticmethod
    def _build_meta_input(
        horizon_scores: Dict[int, float],
        regime_val: float,
    ) -> np.ndarray:
        """Build the (1, n_features) input array for the meta-learner.

        Parameters
        ----------
        horizon_scores:
            Dict of {horizon: score}.
        regime_val:
            Numeric regime encoding.

        Returns
        -------
        np.ndarray
            Shape (1, n_features) ready for Ridge prediction.
        """
        row = [
            horizon_scores.get(1,  0.5),
            horizon_scores.get(5,  0.5),
            horizon_scores.get(20, 0.5),
            regime_val,
        ]
        return np.array(row, dtype=float).reshape(1, -1)


# ===========================================================================
# Helper: Ridge wrapper that conforms to the model_factory interface
# ===========================================================================

class _MetaLearnerWrapper:
    """Thin Ridge regression wrapper compatible with the model_factory API.

    Used internally by the meta-learner walk-forward validation.

    Parameters
    ----------
    scaler:
        StandardScaler instance (shared across train/predict for this wrapper).
    alpha:
        Ridge regularisation strength.
    """

    def __init__(self, scaler: StandardScaler, alpha: float = 1.0) -> None:
        self._scaler = scaler
        self._alpha  = alpha
        self._model: Optional[Ridge] = None
        self._feature_names: List[str] = []

    def train(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        val_features: Optional[pd.DataFrame] = None,
        val_labels: Optional[pd.Series] = None,
    ) -> "_MetaLearnerWrapper":
        """Fit the Ridge model on *features* and *labels*.

        Parameters
        ----------
        features:
            Training feature matrix.
        labels:
            Target series.
        val_features, val_labels:
            Ignored (Ridge does not support early stopping).

        Returns
        -------
        self
        """
        self._feature_names = list(features.columns)
        X = self._scaler.fit_transform(features.values)
        self._model = Ridge(alpha=self._alpha, fit_intercept=True)
        self._model.fit(X, labels.values)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Return Ridge regression predictions.

        Parameters
        ----------
        features:
            Feature matrix to predict on.

        Returns
        -------
        np.ndarray
            Predicted values.

        Raises
        ------
        RuntimeError
            If called before :meth:`train`.
        """
        if self._model is None:
            raise RuntimeError("_MetaLearnerWrapper not fitted.")
        X = self._scaler.transform(
            features.reindex(columns=self._feature_names, fill_value=0.0).fillna(0.0).values
        )
        return self._model.predict(X)

    def get_feature_importance(self) -> pd.Series:
        """Return absolute Ridge coefficients as a feature importance proxy.

        Returns
        -------
        pd.Series
            Absolute coefficient values, indexed by feature name, sorted
            descending.

        Raises
        ------
        RuntimeError
            If called before :meth:`train`.
        """
        if self._model is None:
            raise RuntimeError("_MetaLearnerWrapper not fitted.")
        return pd.Series(
            np.abs(self._model.coef_),
            index=self._feature_names,
        ).sort_values(ascending=False)
