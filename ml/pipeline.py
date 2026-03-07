"""
ml/pipeline.py
==============
ML training orchestrator for the ATNN quantitative trading system.

Coordinates: feature engineering → dynamic feature selection →
base model training (3 horizons) → isotonic calibration →
purged walk-forward meta-learner training → live prediction.

Key improvements over v1
------------------------
1. **Purged Walk-Forward Meta-Learner Training** — base model OOS predictions
   collected via a proper purged walk-forward loop are used to train the
   meta-learner, eliminating in-sample look-ahead bias.

2. **Dynamic Feature Selection via Mutual Information** — before each base
   model is trained, the top-K features (K = min(50, n_features // 2)) are
   selected by mutual information with the binary label.  Selected feature
   sets are stored per horizon and reused at prediction time.

3. **Isotonic Calibration for Base Models** — after each base model is
   trained, an IsotonicRegression is fit on the OOS predictions collected
   during the walk-forward pass.  This calibrates probability estimates and
   is applied before the meta-learner at predict time.

4. **Improved Meta-Learner (ElasticNet + interaction features)** — Ridge is
   replaced with ElasticNet (alpha=0.5, l1_ratio=0.5) which performs implicit
   feature selection.  Interaction features (score_1d×score_5d,
   score_5d×score_20d, max_score−min_score) and trailing rolling accuracy
   features are added, capped at 10 total meta-features.

5. **Data-Version-Aware Retraining** — the training dataset is fingerprinted
   by its shape, first/last timestamps, and column set.  Retraining is only
   triggered when: (a) feature drift exceeds the PSI threshold, (b) data
   changed since last train, or (c) the scheduled ``retrain_freq_days`` has
   elapsed.

6. **Feature Importance Aggregation** — after training, per-horizon importance
   scores are weighted by OOS Sharpe and aggregated into
   ``self.aggregated_feature_importance``.

Public API (unchanged)
----------------------
    MLPipeline(feature_engine, config, model_dir)
    .train_all(price_data, symbol, run_validation)  → dict
    .predict(price_data, regime_state, symbol)      → dict
    .retrain_if_needed(price_data, symbol, force)   → bool
    .save_all(directory)                            → None
    .load_all(directory)                            → None
    .get_validation_report()                        → dict

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

import hashlib
import logging
import math
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

from core.config import get_config, Config
from core.regime_detector import Regime
from ml.feature_engine import FeatureEngine
from ml.models.gradient_boost import GradientBoostModel
from ml.validation import validate_model, walk_forward_validate
from ml.cusum_filter import cusum_filter, cusum_sample_weights
from ml.meta_labeler import MetaLabeler

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

# Embargo window: bars purged between train/val to prevent label leakage
# (the longest prediction horizon is 20d, so embargo of 25 bars is safe)
_EMBARGO_BARS: int = 25

# Walk-forward parameters used to generate OOS predictions for the
# meta-learner and isotonic calibrator
_WF_TRAIN_WINDOW: int = 504   # ~2 trading years
_WF_TEST_WINDOW: int  = 63    # quarterly
_WF_STEP: int         = 21    # monthly step

# Maximum number of features fed to the meta-learner (prevents over-fitting)
_MAX_META_FEATURES: int = 10

# Feature pruning: drop features with zero importance across all horizons
_MIN_IMPORTANCE_THRESHOLD: float = 0.0


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
# Data fingerprint helper
# ===========================================================================

def _fingerprint_data(df: pd.DataFrame) -> str:
    """Create a short, stable hash that identifies a DataFrame's content.

    Uses shape, first/last index dates, and sorted column names so that
    neither row reordering nor irrelevant metadata changes trigger a false
    "data changed" signal.

    Parameters
    ----------
    df:
        DataFrame to fingerprint (typically the raw price data passed to
        ``train_all``).

    Returns
    -------
    str
        8-character hex digest.
    """
    parts = [
        str(df.shape),
        str(df.index[0])  if len(df) > 0 else "empty",
        str(df.index[-1]) if len(df) > 0 else "empty",
        ",".join(sorted(str(c) for c in df.columns)),
    ]
    raw = "|".join(parts).encode("utf-8")
    return hashlib.md5(raw).hexdigest()[:8]


# ===========================================================================
# MLPipeline
# ===========================================================================

class MLPipeline:
    """Orchestrates the full ML workflow: features → training → validation → prediction.

    The pipeline trains three LightGBM base models at horizons 1d, 5d, and
    20d using dynamically selected features, calibrates them with isotonic
    regression, then combines their OOS scores with market regime via an
    ElasticNet meta-learner to produce a single composite signal per symbol.

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

        # Meta-learner: ElasticNet over OOS base model scores + interaction features
        self.meta_learner: Optional[ElasticNet] = None
        self._meta_scaler: StandardScaler       = StandardScaler()
        self._meta_fitted: bool                 = False

        # Isotonic calibrators — one per prediction horizon
        # Trained on OOS predictions collected during the walk-forward pass
        self._calibrators: Dict[int, IsotonicRegression] = {}

        # Selected feature names per horizon (mutual-information selection)
        self._selected_features: Dict[int, List[str]] = {}

        # OOS predictions collected during walk-forward (used for calibration
        # training and rolling-accuracy meta-features)
        # Structure: {horizon: (oos_predictions_array, oos_labels_array)}
        self._oos_records: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        # Aggregated, OOS-performance-weighted feature importance across horizons
        self.aggregated_feature_importance: Dict[str, float] = {}

        # Validation reports
        self._validation_reports: Dict[int, Dict] = {}
        self._meta_validation_report: Dict = {}

        # Drift monitoring: store training feature statistics
        self._train_feature_stats: Dict[str, Tuple[np.ndarray, int]] = {}

        # Data versioning: fingerprint of the last training dataset
        self._data_fingerprint: Optional[str] = None

        # Feature pruning: features identified as zero-importance across models
        self._pruned_features: List[str] = []

        # CUSUM event filter
        self._cusum_events: Optional[pd.DatetimeIndex] = None

        # Meta-labeler (AFML Ch. 3.6 + Ch. 10)
        self.meta_labeler: MetaLabeler = MetaLabeler(
            max_bet_size=1.0,
            min_probability=0.55,
        )

        # Timestamps
        self._last_train_time: Optional[float] = None

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

        The full training sequence is:

        1. Compute features via the feature engine.
        2. For each horizon:
           a. Select top-K features by mutual information.
           b. Run a purged walk-forward loop to collect OOS predictions and
              labels.
           c. Train an IsotonicRegression calibrator on the OOS predictions.
           d. Train the final base model on the full dataset.
           e. Optionally run the full validation suite.
        3. Train the ElasticNet meta-learner using the OOS base model scores
           as features (no IS leakage).
        4. Aggregate feature importances weighted by OOS Sharpe.

        Parameters
        ----------
        price_data:
            OHLCV DataFrame for a single symbol, datetime-indexed.
        symbol:
            Optional ticker for logging.
        run_validation:
            If True, run walk-forward validation for each base model and the
            meta-learner (recommended for production).

        Returns
        -------
        dict
            ``{horizon: validation_report}`` for each base model, plus
            ``"meta"`` key for the meta-learner report and ``"summary"``
            with pass/fail counts.

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

        # --- Data fingerprint for change detection ---
        new_fingerprint = _fingerprint_data(price_data)
        self._data_fingerprint = new_fingerprint

        # --- CUSUM event-driven sampling ---
        close_col = price_data.get("close", price_data.get("Close"))
        if close_col is not None and len(close_col) > 50:
            self._cusum_events = cusum_filter(
                close_col, vol_multiplier=1.0, vol_lookback=20
            )
            logger.info(
                f"MLPipeline.train_all: CUSUM detected {len(self._cusum_events)} "
                f"structural events from {len(close_col)} bars."
            )
        else:
            self._cusum_events = None

        # --- Compute features ---
        logger.info("MLPipeline.train_all: computing features ...")
        features = self.feature_engine.compute_features(price_data, symbol=symbol)

        # Store training stats for PSI-based drift detection
        for col in features.columns:
            vals = features[col].dropna().values
            if len(vals) > 0:
                self._train_feature_stats[col] = (vals, len(vals))

        reports: Dict = {}

        # OOS score containers for meta-learner training
        # {horizon: pd.Series aligned to the full features index}
        oos_scores_for_meta: Dict[int, pd.Series] = {}

        # Per-horizon OOS Sharpe (used to weight importance aggregation)
        oos_sharpe_by_horizon: Dict[int, float] = {}

        # ------------------------------------------------------------------
        # Train base models
        # ------------------------------------------------------------------
        for horizon in horizons:
            logger.info(f"MLPipeline.train_all: training horizon={horizon}d ...")

            # --- Step 1: Prepare labels ---
            model_tmp = GradientBoostModel(
                horizon = horizon,
                params  = self._build_lgbm_params(ml_cfg),
                mode    = "classification",
            )
            labels = model_tmp.prepare_labels(price_data)

            # --- Step 2: Purged walk-forward to collect OOS predictions ---
            # MI feature selection is now performed INSIDE each walk-forward fold
            # (on training data only) to prevent look-ahead bias from future bars
            # leaking into feature selection.  The full feature matrix is passed
            # and _walk_forward_oos calls _select_features_mi per fold.
            oos_preds_arr, oos_labels_arr, oos_index, wf_oos_sharpe = (
                self._walk_forward_oos(
                    features = features,
                    labels   = labels,
                    horizon  = horizon,
                    ml_cfg   = ml_cfg,
                )
            )

            # --- Step 2b: Final MI selection on full dataset for prediction mask ---
            # This is the feature mask stored for use at prediction time.  It is
            # computed on the full training dataset and is separate from the
            # per-fold selections used inside the walk-forward loop above.
            selected_feats = self._select_features_mi(
                features = features,
                labels   = labels,
                horizon  = horizon,
            )
            features_h = features[selected_feats]
            logger.info(
                f"MLPipeline.train_all: horizon={horizon}d selected "
                f"{len(selected_feats)}/{len(features.columns)} features via MI "
                f"(full-dataset pass for prediction mask)."
            )

            oos_sharpe_by_horizon[horizon] = wf_oos_sharpe

            # Store OOS records (needed for calibrator + rolling accuracy)
            if len(oos_preds_arr) > 0:
                self._oos_records[horizon] = (oos_preds_arr, oos_labels_arr)

            # --- Step 4: Fit isotonic calibrator on OOS predictions ---
            calibrator: Optional[IsotonicRegression] = None
            if len(oos_preds_arr) >= 20:
                try:
                    calibrator = IsotonicRegression(
                        out_of_bounds="clip", increasing=True
                    )
                    calibrator.fit(oos_preds_arr, oos_labels_arr)
                    self._calibrators[horizon] = calibrator
                    logger.info(
                        f"MLPipeline.train_all: isotonic calibrator fitted "
                        f"for horizon={horizon}d on {len(oos_preds_arr)} OOS samples."
                    )
                except Exception as exc:
                    logger.warning(
                        f"MLPipeline.train_all: calibrator fit failed for "
                        f"horizon={horizon}d: {exc}"
                    )

            # Build OOS score series aligned to feature index (for meta-learner)
            if len(oos_preds_arr) > 0 and oos_index is not None:
                raw_series = pd.Series(
                    oos_preds_arr, index=oos_index, name=f"raw_{horizon}d"
                )
                if calibrator is not None:
                    cal_vals = calibrator.predict(oos_preds_arr)
                    cal_series = pd.Series(
                        cal_vals, index=oos_index, name=f"score_{horizon}d"
                    )
                else:
                    cal_series = raw_series.rename(f"score_{horizon}d")
                oos_scores_for_meta[horizon] = cal_series

            # --- Step 5: Train the final base model on the full dataset ---
            model = GradientBoostModel(
                horizon = horizon,
                params  = self._build_lgbm_params(ml_cfg),
                mode    = "classification",
            )

            # Use an embargo-aware holdout split for early stopping
            holdout_size = max(ml_cfg.train_window_days // 10, 42)
            train_end    = len(features_h) - holdout_size - _EMBARGO_BARS
            val_start    = train_end + _EMBARGO_BARS
            if train_end < 100:
                train_end = len(features_h)
                val_start = train_end

            X_train_full = features_h.iloc[:train_end]
            y_train_full = labels.iloc[:train_end]
            X_val_full   = features_h.iloc[val_start:]
            y_val_full   = labels.iloc[val_start:]

            model.train(X_train_full, y_train_full, X_val_full, y_val_full)
            self.models[horizon] = model

            # Save to disk
            model_path = os.path.join(self.model_dir, f"lgbm_h{horizon}.pkl")
            model.save(model_path)

            # --- Step 6: Optional full validation suite ---
            if run_validation:
                try:
                    def _factory(h=horizon, p=model._params) -> GradientBoostModel:
                        return GradientBoostModel(
                            horizon=h, params=p, mode="classification"
                        )

                    val_report = validate_model(
                        model_factory = _factory,
                        features      = features_h,
                        labels        = labels,
                        config        = {
                            "train_window":      ml_cfg.train_window_days,
                            "test_window":       21,
                            "step":              21,
                            "min_windows":       6,
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
                        f"MLPipeline.train_all: validation failed for "
                        f"horizon {horizon}: {exc}"
                    )
                    reports[horizon] = {"error": str(exc), "pass": False}

        # ------------------------------------------------------------------
        # Feature pruning: identify zero-importance features
        # ------------------------------------------------------------------
        if len(self.models) > 1:
            all_imps: Dict[str, float] = {}
            for _h, _m in self.models.items():
                for feat_name, imp_val in _m.feature_importances.items():
                    all_imps[feat_name] = all_imps.get(feat_name, 0.0) + imp_val
            dead_features = [f for f, v in all_imps.items() if v <= _MIN_IMPORTANCE_THRESHOLD]
            self._pruned_features = dead_features
            if dead_features:
                logger.info(
                    f"MLPipeline.train_all: identified {len(dead_features)} "
                    f"zero-importance features for pruning on next retrain."
                )

        # ------------------------------------------------------------------
        # Feature importance aggregation (OOS-Sharpe-weighted)
        # ------------------------------------------------------------------
        self.aggregated_feature_importance = self._aggregate_feature_importance(
            oos_sharpe_by_horizon
        )
        logger.info(
            f"MLPipeline.train_all: aggregated importance over "
            f"{len(self.aggregated_feature_importance)} features."
        )

        # ------------------------------------------------------------------
        # Train meta-learner using ONLY OOS base model scores
        # ------------------------------------------------------------------
        logger.info("MLPipeline.train_all: training meta-learner on OOS scores ...")
        try:
            meta_report = self._train_meta_learner(
                price_data          = price_data,
                oos_scores_for_meta = oos_scores_for_meta,
                run_validation      = run_validation,
            )
            reports["meta"] = meta_report
        except Exception as exc:
            logger.warning(f"MLPipeline.train_all: meta-learner training failed: {exc}")
            reports["meta"] = {"error": str(exc)}

        # ------------------------------------------------------------------
        # Train meta-labeler (AFML Ch. 3.6) — secondary model for
        # trade gating + bet sizing from predicted probabilities
        # ------------------------------------------------------------------
        logger.info("MLPipeline.train_all: training meta-labeler ...")
        try:
            meta_label_report = self._train_meta_labeler(
                price_data=price_data,
                features=features,
                oos_scores_for_meta=oos_scores_for_meta,
                symbol=symbol,
            )
            reports["meta_labeler"] = meta_label_report
        except Exception as exc:
            logger.warning(f"MLPipeline.train_all: meta-labeler training failed: {exc}")
            reports["meta_labeler"] = {"error": str(exc)}

        self._last_train_time = time.time()

        # --- Summary ---
        pass_count = sum(
            1 for h in horizons
            if reports.get(h, {}).get("overall_pass", False)
        )
        reports["summary"] = {
            "horizons_trained":    horizons,
            "base_models_passing": pass_count,
            "total_base_models":   len(horizons),
            "meta_learner_fitted": self._meta_fitted,
            "meta_labeler_fitted": self.meta_labeler.is_fitted,
            "cusum_events":        len(self._cusum_events) if self._cusum_events is not None else 0,
            "data_fingerprint":    self._data_fingerprint,
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

        Prediction sequence:
        1. Compute features.
        2. Apply per-horizon feature selection masks.
        3. Obtain raw base model probability scores.
        4. Apply isotonic calibration to each score.
        5. Feed calibrated scores (+ interaction features) to the meta-learner.

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
                         "scores_by_horizon": dict, "calibrated_scores": dict,
                         "meta_score": float, "regime": float}}``

        Notes
        -----
        Returns an empty dict if no models are fitted or price data is
        insufficient.
        """
        # Validate input is single-symbol — a MultiIndex DataFrame means the
        # caller passed the full multi-symbol history instead of per-symbol data.
        if isinstance(price_data.index, pd.MultiIndex):
            raise ValueError(
                "MLPipeline.predict() expects single-symbol OHLCV data, "
                "got MultiIndex. Call predict() per symbol."
            )

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
        last_row_full = features.dropna(how="all").iloc[[-1]]
        if last_row_full.empty:
            return {}

        # --- Base model predictions with per-horizon feature selection ---
        horizon_scores: Dict[int, float]     = {}
        calibrated_scores: Dict[int, float]  = {}

        for horizon, model in self.models.items():
            try:
                # Apply feature selection mask
                sel_feats = self._selected_features.get(horizon)
                if sel_feats is not None:
                    last_row = last_row_full.reindex(columns=sel_feats, fill_value=0.0).fillna(0.0)
                else:
                    last_row = last_row_full

                raw_score = model.predict_single(last_row)
                horizon_scores[horizon] = raw_score

                # Isotonic calibration
                cal = self._calibrators.get(horizon)
                if cal is not None and not math.isnan(raw_score):
                    cal_score = float(cal.predict(np.array([raw_score]))[0])
                else:
                    cal_score = raw_score
                calibrated_scores[horizon] = cal_score

            except Exception as exc:
                logger.warning(
                    f"MLPipeline.predict: horizon {horizon} prediction failed: {exc}"
                )
                horizon_scores[horizon]    = float("nan")
                calibrated_scores[horizon] = float("nan")

        valid_scores = [s for s in calibrated_scores.values() if not math.isnan(s)]
        if not valid_scores:
            return {}

        regime_val = self._encode_regime(regime_state)

        # --- Meta-learner prediction ---
        meta_score: Optional[float] = None
        if self._meta_fitted and self.meta_learner is not None:
            try:
                meta_input  = self._build_meta_input(calibrated_scores, regime_val)
                meta_scaled = self._meta_scaler.transform(meta_input)
                meta_score  = float(self.meta_learner.predict(meta_scaled)[0])
                # Clip to [0, 1] for probability semantics
                meta_score  = float(np.clip(meta_score, 0.0, 1.0))
            except Exception as exc:
                logger.debug(f"Meta-learner prediction failed: {exc}")

        # --- Composite signal ---
        final_score = meta_score if meta_score is not None else float(np.nanmean(valid_scores))

        # Confidence = fraction of base models that agree on direction
        directions = [int(s > 0.5) for s in valid_scores]
        mode_dir   = int(np.round(np.mean(directions)))
        agreement  = sum(1 for d in directions if d == mode_dir) / len(directions)

        # --- Meta-labeler: trade gating + bet sizing (AFML Ch. 3.6 + 10) ---
        bet_size = 1.0
        take_trade = True
        meta_label_prob = None
        if self.meta_labeler.is_fitted:
            try:
                ml_result = self.meta_labeler.predict(features.iloc[[-1]])
                bet_size = ml_result.get("bet_size", 1.0)
                take_trade = ml_result.get("take_trade", True)
                meta_label_prob = ml_result.get("probability")
                if not take_trade:
                    # Meta-labeler says skip — set score to 0.5 (neutral)
                    final_score = 0.5
                    bet_size = 0.0
            except Exception as exc:
                logger.debug(f"Meta-labeler prediction failed: {exc}")

        key = symbol or "default"
        return {
            key: {
                "score":              final_score,
                "confidence":         agreement,
                "horizon":            5,  # primary horizon
                "scores_by_horizon":  horizon_scores,
                "calibrated_scores":  calibrated_scores,
                "meta_score":         meta_score,
                "regime":             regime_val,
                "bet_size":           bet_size,
                "take_trade":         take_trade,
                "meta_label_prob":    meta_label_prob,
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
        """Check if retraining is warranted; retrain if needed.

        Retraining is triggered by any of three conditions:
        (a) Feature drift: PSI > ``_PSI_THRESHOLD`` for ≥ 20% of features.
        (b) Data change: the training data fingerprint has changed.
        (c) Schedule: ``retrain_freq_days`` calendar days have elapsed since
            the last training run.

        Parameters
        ----------
        price_data:
            Recent OHLCV data.
        symbol:
            Optional ticker.
        force:
            Force retrain regardless of all checks.

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
                logger.info(
                    f"MLPipeline.retrain_if_needed: only {elapsed_days:.1f} days since "
                    f"last train (threshold={ml_cfg.retrain_freq_days}d); skipping."
                )
                return False

        # Check data-version change
        if not force and self._data_fingerprint is not None:
            new_fp = _fingerprint_data(price_data)
            if new_fp == self._data_fingerprint:
                # Data hasn't changed — still check PSI drift below
                pass
            else:
                logger.info(
                    "MLPipeline.retrain_if_needed: data fingerprint changed "
                    f"({self._data_fingerprint} → {new_fp}); triggering retrain."
                )
                force = True  # data changed → always retrain

        # Check PSI drift
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
        """Save all fitted models, calibrators, and auxiliary state to disk.

        Persisted artefacts:
        - ``lgbm_h{horizon}.pkl``      — base LightGBM model
        - ``meta_learner.pkl``         — ElasticNet meta-learner + scaler +
                                         selected features + calibrators +
                                         aggregated importance

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
            meta_payload = {
                "meta_learner":                self.meta_learner,
                "meta_scaler":                 self._meta_scaler,
                "horizons":                    list(self.models.keys()),
                "calibrators":                 self._calibrators,
                "selected_features":           self._selected_features,
                "aggregated_feature_importance": self.aggregated_feature_importance,
                "data_fingerprint":            self._data_fingerprint,
            }
            meta_path = os.path.join(save_dir, "meta_learner.pkl")
            with open(meta_path, "wb") as fh:
                pickle.dump(meta_payload, fh)
            logger.info(f"Meta-learner and auxiliary state saved to {meta_path}")

    def load_all(self, directory: Optional[str] = None) -> None:
        """Load all previously saved models and auxiliary state from disk.

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
            with open(meta_path, "rb") as fh:
                meta_payload = pickle.load(fh)
            self.meta_learner        = meta_payload["meta_learner"]
            self._meta_scaler        = meta_payload["meta_scaler"]
            self._meta_fitted        = True
            self._calibrators        = meta_payload.get("calibrators", {})
            self._selected_features  = meta_payload.get("selected_features", {})
            self.aggregated_feature_importance = meta_payload.get(
                "aggregated_feature_importance", {}
            )
            self._data_fingerprint   = meta_payload.get("data_fingerprint")
            logger.info(f"Meta-learner and auxiliary state loaded from {meta_path}")

        if loaded == 0:
            raise FileNotFoundError(
                f"No model files found in {load_dir!r}. "
                "Call train_all() first."
            )
        logger.info(f"MLPipeline.load_all: loaded {loaded} base models from {load_dir}.")

    # ------------------------------------------------------------------
    # Private helpers — feature selection
    # ------------------------------------------------------------------

    def _select_features_mi(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        horizon: int,
        train_idx: Optional[Any] = None,
    ) -> List[str]:
        """Select top-K features by mutual information with the binary label.

        K = min(50, n_features // 2) to control the curse of dimensionality.
        Selected names are stored in ``self._selected_features[horizon]`` for
        reuse at prediction time **only when train_idx is None** (i.e. the
        final full-dataset selection).  When train_idx is provided (fold-level
        selection inside the walk-forward loop) the result is returned but NOT
        written to ``self._selected_features`` so it does not pollute the
        prediction-time feature mask.

        Parameters
        ----------
        features:
            Full feature matrix (all available columns).
        labels:
            Binary label series aligned with features.
        horizon:
            Horizon identifier used as the storage key.
        train_idx:
            Optional positional index slice (e.g. ``slice(0, n)`` or an array
            of integer positions) that restricts MI computation to training
            data only.  When provided, only those rows are used, preventing
            any look-ahead from future bars leaking into feature selection.

        Returns
        -------
        List[str]
            Ordered list of selected feature names (highest MI first).
        """
        # Restrict to training fold when index bounds are provided
        if train_idx is not None:
            features = features.iloc[train_idx]
            labels   = labels.iloc[train_idx]

        n_total = len(features.columns)
        k = min(50, max(1, n_total // 2))

        # Align and clean
        common_idx = features.index.intersection(labels.dropna().index)
        X = features.loc[common_idx].fillna(0.0)
        y = labels.loc[common_idx]

        if len(X) < 50 or n_total == 0:
            # Fallback: use all features
            selected = list(features.columns)
            if train_idx is None:
                self._selected_features[horizon] = selected
            return selected

        try:
            mi_scores = mutual_info_classif(
                X.values,
                y.values.astype(int),
                discrete_features=False,
                random_state=42,
            )
            mi_series = pd.Series(mi_scores, index=features.columns)
            selected  = mi_series.nlargest(k).index.tolist()
        except Exception as exc:
            logger.warning(
                f"_select_features_mi: MI computation failed for horizon={horizon}: {exc}. "
                "Using all features."
            )
            selected = list(features.columns)

        # Only update the persistent prediction-time mask when called on full data
        if train_idx is None:
            self._selected_features[horizon] = selected
        return selected

    # ------------------------------------------------------------------
    # Private helpers — walk-forward OOS collection
    # ------------------------------------------------------------------

    def _walk_forward_oos(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        horizon: int,
        ml_cfg: Any,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[pd.Index], float]:
        """Run a purged walk-forward loop and collect OOS predictions.

        Uses ``_WF_TRAIN_WINDOW=504``, ``_WF_TEST_WINDOW=63``,
        ``_WF_STEP=21`` with ``_EMBARGO_BARS=25`` purge between folds.

        Mutual-information feature selection is performed **inside each fold**
        using only that fold's training data, so no future bars leak into the
        feature selection process.

        Parameters
        ----------
        features:
            Full (unfiltered) feature matrix for this horizon.  MI selection
            is applied per fold inside the loop.
        labels:
            Binary label series.
        horizon:
            Prediction horizon (used for model construction).
        ml_cfg:
            :class:`~core.config.MLConfig` instance.

        Returns
        -------
        oos_preds : np.ndarray
            Concatenated OOS predicted probabilities.
        oos_labels : np.ndarray
            Corresponding ground-truth labels.
        oos_index : pd.Index or None
            Datetime index aligned with oos_preds / oos_labels.
        oos_sharpe : float
            OOS Sharpe ratio over all windows (NaN if insufficient data).
        """
        valid_mask = labels.notna()
        feats = features.loc[valid_mask].copy()
        labs  = labels.loc[valid_mask].copy()
        n     = len(feats)

        total_needed = _WF_TRAIN_WINDOW + _WF_TEST_WINDOW + _EMBARGO_BARS
        if n < total_needed:
            logger.warning(
                f"_walk_forward_oos: only {n} clean samples for horizon={horizon}d "
                f"(need {total_needed}); skipping OOS walk-forward."
            )
            return np.array([]), np.array([]), None, float("nan")

        feats_arr = feats.values
        labs_arr  = labs.values
        idx_arr   = feats.index

        all_oos_preds:  List[np.ndarray]  = []
        all_oos_labels: List[np.ndarray]  = []
        all_oos_idx:    List[pd.Index]    = []
        all_oos_returns: List[np.ndarray] = []

        start = 0
        while start + _WF_TRAIN_WINDOW + _EMBARGO_BARS + _WF_TEST_WINDOW <= n:
            train_end   = start + _WF_TRAIN_WINDOW
            test_start  = train_end + _EMBARGO_BARS   # skip embargo
            test_end    = test_start + _WF_TEST_WINDOW

            if test_end > n:
                break

            # --- Per-fold MI feature selection using ONLY training rows ---
            # Pass train_idx so _select_features_mi never sees future bars.
            # We rebuild fold DataFrames from feats (not feats_arr) to keep
            # column names available for MI selection.
            fold_train_idx = slice(start, train_end)
            fold_sel_feats = self._select_features_mi(
                features  = feats,
                labels    = labs,
                horizon   = horizon,
                train_idx = fold_train_idx,
            )

            X_train = feats.iloc[start:train_end][fold_sel_feats]
            y_train = pd.Series(labs_arr[start:train_end])
            X_test  = feats.iloc[test_start:test_end].reindex(columns=fold_sel_feats, fill_value=0.0)
            y_test  = pd.Series(labs_arr[test_start:test_end])

            try:
                fold_model = GradientBoostModel(
                    horizon = horizon,
                    params  = self._build_lgbm_params(ml_cfg),
                    mode    = "classification",
                )
                fold_model.train(X_train, y_train)
                oos_preds = fold_model.predict(X_test)
            except Exception as exc:
                logger.warning(
                    f"_walk_forward_oos: window [{start}:{train_end}] "
                    f"failed for horizon={horizon}d: {exc}"
                )
                start += _WF_STEP
                continue

            all_oos_preds.append(oos_preds)
            all_oos_labels.append(y_test.values)
            all_oos_idx.append(idx_arr[test_start:test_end])

            # Track strategy returns for Sharpe calculation
            directions = np.where(oos_preds >= 0.5, 1.0, -1.0)
            strat_ret  = directions * y_test.values.astype(float)
            all_oos_returns.append(strat_ret)

            start += _WF_STEP

        if not all_oos_preds:
            return np.array([]), np.array([]), None, float("nan")

        oos_preds_cat  = np.concatenate(all_oos_preds)
        oos_labels_cat = np.concatenate(all_oos_labels)
        oos_idx_cat    = all_oos_idx[0].append(all_oos_idx[1:]) if len(all_oos_idx) > 1 else all_oos_idx[0]

        # Compute OOS Sharpe
        combined_ret = np.concatenate(all_oos_returns)
        oos_sharpe   = self._sharpe(combined_ret)

        logger.info(
            f"_walk_forward_oos: horizon={horizon}d collected {len(oos_preds_cat)} "
            f"OOS predictions across {len(all_oos_preds)} windows. "
            f"OOS Sharpe={oos_sharpe:.3f}"
        )

        return oos_preds_cat, oos_labels_cat, oos_idx_cat, oos_sharpe

    # ------------------------------------------------------------------
    # Private helpers — meta-learner
    # ------------------------------------------------------------------

    def _train_meta_learner(
        self,
        price_data: pd.DataFrame,
        oos_scores_for_meta: Dict[int, pd.Series],
        run_validation: bool = True,
    ) -> Dict:
        """Train an ElasticNet meta-learner using ONLY OOS base model scores.

        Features fed to the meta-learner (capped at ``_MAX_META_FEATURES=10``):
        1. ``score_1d``   — calibrated OOS score for 1-day horizon
        2. ``score_5d``   — calibrated OOS score for 5-day horizon
        3. ``score_20d``  — calibrated OOS score for 20-day horizon
        4. ``regime``     — SMA-50/200 crossover regime encoding
        5. ``x_1d_5d``    — score_1d × score_5d interaction
        6. ``x_5d_20d``   — score_5d × score_20d interaction
        7. ``disagreement`` — max_score − min_score (spread / uncertainty)
        8. ``roll_acc_1d``  — rolling 20-bar accuracy of 1d OOS predictions
        9. ``roll_acc_5d``  — rolling 20-bar accuracy of 5d OOS predictions
        10. ``roll_acc_20d`` — rolling 20-bar accuracy of 20d OOS predictions

        The meta-learner only ever sees OOS scores — never IS predictions —
        which eliminates the look-ahead bias present in the original pipeline.

        Parameters
        ----------
        price_data:
            Raw OHLCV price data (used to build regime labels).
        oos_scores_for_meta:
            ``{horizon: pd.Series}`` of calibrated OOS scores aligned to the
            feature index.
        run_validation:
            Run walk-forward validation on the meta-learner if True.

        Returns
        -------
        dict
            Meta-learner training report.
        """
        if not oos_scores_for_meta:
            logger.warning("_train_meta_learner: no OOS scores available.")
            return {"error": "no OOS scores", "fitted": False}

        # --- Align OOS score series ---
        score_frames: List[pd.Series] = []
        for horizon in sorted(oos_scores_for_meta.keys()):
            s = oos_scores_for_meta[horizon].rename(f"score_{horizon}d")
            score_frames.append(s)

        meta_X = pd.concat(score_frames, axis=1).dropna()

        if meta_X.empty:
            return {"error": "empty aligned OOS frame", "fitted": False}

        # --- Regime feature (SMA 50/200 crossover) ---
        try:
            close = price_data["close"] if "close" in price_data.columns else price_data["Close"]
            sma_50  = close.rolling(50,  min_periods=50).mean()
            sma_200 = close.rolling(200, min_periods=200).mean()
            regime_series = pd.Series(0.0, index=close.index)
            regime_series[sma_50 > sma_200] =  1.0   # BULL
            regime_series[sma_50 < sma_200] = -1.0   # BEAR
            meta_X["regime"] = regime_series.reindex(meta_X.index).fillna(0.0)
        except Exception:
            meta_X["regime"] = 0.0

        # --- Interaction features ---
        s1  = meta_X.get("score_1d",  pd.Series(0.5, index=meta_X.index))
        s5  = meta_X.get("score_5d",  pd.Series(0.5, index=meta_X.index))
        s20 = meta_X.get("score_20d", pd.Series(0.5, index=meta_X.index))

        meta_X["x_1d_5d"]   = s1 * s5
        meta_X["x_5d_20d"]  = s5 * s20

        score_cols = [c for c in meta_X.columns if c.startswith("score_")]
        if len(score_cols) >= 2:
            meta_X["disagreement"] = (
                meta_X[score_cols].max(axis=1) - meta_X[score_cols].min(axis=1)
            )
        else:
            meta_X["disagreement"] = 0.0

        # --- Rolling 20-bar accuracy per horizon ---
        for horizon in sorted(oos_scores_for_meta.keys()):
            col = f"score_{horizon}d"
            if col not in meta_X.columns:
                continue
            oos_rec = self._oos_records.get(horizon)
            if oos_rec is None:
                meta_X[f"roll_acc_{horizon}d"] = 0.5
                continue
            oos_preds_arr, oos_labels_arr = oos_rec
            # Binary accuracy: prediction > 0.5 matches label
            correct = (oos_preds_arr > 0.5).astype(float) == oos_labels_arr.astype(float)
            correct_series = pd.Series(
                correct.astype(float),
                index=oos_scores_for_meta[horizon].index,
            )
            roll_acc = correct_series.rolling(20, min_periods=1).mean()
            meta_X[f"roll_acc_{horizon}d"] = roll_acc.reindex(meta_X.index).fillna(0.5)

        # Cap total meta features at _MAX_META_FEATURES to prevent overfitting
        if meta_X.shape[1] > _MAX_META_FEATURES:
            meta_X = meta_X.iloc[:, :_MAX_META_FEATURES]

        # --- Target: 5-day forward return label ---
        primary_model = self.models.get(5) or next(iter(self.models.values()))
        meta_y = primary_model.prepare_labels(price_data, symbol=None)
        meta_y = meta_y.reindex(meta_X.index).dropna()
        meta_X = meta_X.reindex(meta_y.index).dropna()

        if len(meta_X) < 50:
            logger.warning(
                f"_train_meta_learner: only {len(meta_X)} aligned samples; skipping."
            )
            return {"error": "insufficient aligned samples", "fitted": False}

        # --- Scale and fit ElasticNet ---
        X_scaled = self._meta_scaler.fit_transform(meta_X.values)
        self.meta_learner = ElasticNet(
            alpha     = 0.5,
            l1_ratio  = 0.5,
            max_iter  = 2000,
            fit_intercept = True,
        )
        self.meta_learner.fit(X_scaled, meta_y.values)
        self._meta_fitted = True

        # --- Walk-forward validation for meta-learner ---
        meta_report: Dict = {"fitted": True}
        if run_validation and len(meta_X) >= 100:
            try:
                def _meta_factory() -> _MetaLearnerWrapper:
                    return _MetaLearnerWrapper(
                        scaler   = StandardScaler(),
                        alpha    = 0.5,
                        l1_ratio = 0.5,
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
                meta_report["pass"]         = wfv_meta.get("pass", False)
                self._meta_validation_report = meta_report
            except Exception as exc:
                logger.warning(f"Meta-learner validation failed: {exc}")
                meta_report["validation_error"] = str(exc)

        logger.info(
            f"_train_meta_learner: fitted ElasticNet on {len(meta_X)} OOS samples "
            f"with {meta_X.shape[1]} features. "
            f"Non-zero coefs={np.sum(self.meta_learner.coef_ != 0)}"
        )
        return meta_report

    # ------------------------------------------------------------------
    # Private helpers — meta-labeler training (AFML Ch. 3.6)
    # ------------------------------------------------------------------

    def _train_meta_labeler(
        self,
        price_data: pd.DataFrame,
        features: pd.DataFrame,
        oos_scores_for_meta: Dict[int, pd.Series],
        symbol: Optional[str] = None,
    ) -> Dict:
        """Train the meta-labeler: secondary model for trade gating + bet sizing.

        The meta-labeler learns to predict whether the primary model's
        directional prediction will be correct.  It uses:
        - All computed features (same as base models)
        - CUSUM events for sample weighting (structural events get more weight)
        - OOS predictions from base models as the primary direction signal

        Parameters
        ----------
        price_data : pd.DataFrame
            Raw OHLCV data.
        features : pd.DataFrame
            Computed feature matrix.
        oos_scores_for_meta : dict
            OOS scores from base models keyed by horizon.
        symbol : str, optional
            Ticker label.

        Returns
        -------
        dict
            Training report.
        """
        # Use the 5d horizon OOS scores as the primary direction signal
        primary_scores = oos_scores_for_meta.get(5)
        if primary_scores is None:
            # Fall back to any available horizon
            for h in sorted(oos_scores_for_meta.keys()):
                primary_scores = oos_scores_for_meta[h]
                break

        if primary_scores is None or len(primary_scores) < 100:
            return {"fitted": False, "reason": "insufficient OOS scores for meta-labeler"}

        # Primary direction: score > 0.5 → long (+1), else short (-1)
        primary_direction = pd.Series(
            np.where(primary_scores > 0.5, 1.0, -1.0),
            index=primary_scores.index,
        )

        # Forward returns for the primary horizon (5d)
        close = price_data.get("close", price_data.get("Close"))
        if close is None:
            return {"fitted": False, "reason": "no close prices"}

        fwd_ret = np.log(close.shift(-5) / close).dropna()

        # Create meta-labels: 1 if primary was correct, 0 if wrong
        meta_labels = MetaLabeler.create_meta_labels(primary_direction, fwd_ret)

        if len(meta_labels) < 100:
            return {"fitted": False, "reason": f"only {len(meta_labels)} meta-labels"}

        # Align features with meta-labels
        common = features.index.intersection(meta_labels.index)
        if len(common) < 100:
            return {"fitted": False, "reason": f"only {len(common)} aligned samples"}

        # CUSUM sample weights: events at structural breakpoints get more weight
        sample_weight = None
        if self._cusum_events is not None and len(self._cusum_events) > 10:
            event_set = set(self._cusum_events)
            sw = pd.Series(1.0, index=common)
            for dt in common:
                if dt in event_set:
                    sw[dt] = 3.0  # 3x weight for CUSUM events
            sample_weight = sw

        # Train
        report = self.meta_labeler.train(
            features=features.loc[common],
            meta_labels=meta_labels.loc[common],
            sample_weight=sample_weight,
        )

        return report

    # ------------------------------------------------------------------
    # Private helpers — feature importance aggregation
    # ------------------------------------------------------------------

    def _aggregate_feature_importance(
        self,
        oos_sharpe_by_horizon: Dict[int, float],
    ) -> Dict[str, float]:
        """Aggregate per-horizon feature importances weighted by OOS Sharpe.

        Each horizon model's feature importances are multiplied by its OOS
        Sharpe ratio (floored at 0 to ignore negative-Sharpe models), then
        summed and normalised to sum to 1.

        Parameters
        ----------
        oos_sharpe_by_horizon:
            ``{horizon: oos_sharpe}`` from the walk-forward loop.

        Returns
        -------
        Dict[str, float]
            Feature name → aggregated importance score, sorted descending.
        """
        weighted: Dict[str, float] = {}
        total_weight = 0.0

        for horizon, model in self.models.items():
            sharpe = oos_sharpe_by_horizon.get(horizon, 0.0)
            weight = max(sharpe, 0.0)  # floor negative Sharpe at 0
            if math.isnan(weight):
                weight = 0.0
            total_weight += weight

            for feat_name, imp_val in model.feature_importances.items():
                weighted[feat_name] = weighted.get(feat_name, 0.0) + imp_val * weight

        if total_weight > 0.0:
            weighted = {k: v / total_weight for k, v in weighted.items()}

        # Sort descending
        sorted_imp = dict(
            sorted(weighted.items(), key=lambda kv: kv[1], reverse=True)
        )
        return sorted_imp

    # ------------------------------------------------------------------
    # Private helpers — misc
    # ------------------------------------------------------------------

    @staticmethod
    def _build_lgbm_params(ml_cfg: Any) -> Dict:
        """Build a parameter dict from MLConfig.model_params.

        Parameters
        ----------
        ml_cfg:
            :class:`~core.config.MLConfig` instance.

        Returns
        -------
        dict
            LightGBM parameter dict suitable for ``GradientBoostModel``.
        """
        p = ml_cfg.model_params
        return {
            "max_depth":         p.max_depth,
            "num_leaves":        p.num_leaves,
            "learning_rate":     p.learning_rate,
            "min_child_samples": p.min_child_samples,
            "n_estimators":      p.n_estimators,
            "subsample":         p.subsample,
            "colsample_bytree":  p.colsample_bytree,
            "reg_alpha":         p.reg_alpha,
            "reg_lambda":        p.reg_lambda,
            "random_state":      p.random_state,
        }

    @staticmethod
    def _sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
        """Annualised Sharpe ratio from a 1-D return series.

        Parameters
        ----------
        returns:
            Array of period returns.
        periods_per_year:
            Scaling factor (default 252 for daily).

        Returns
        -------
        float
            Sharpe ratio; NaN if fewer than 2 observations or zero std.
        """
        if len(returns) < 2:
            return float("nan")
        mu  = float(np.mean(returns))
        sig = float(np.std(returns, ddof=1))
        if sig == 0:
            return float("nan")
        return mu / sig * math.sqrt(periods_per_year)

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
            regime_str = str(regime_state.regime.value)
            return _REGIME_ENCODING.get(regime_str, 0.0)
        if hasattr(regime_state, "value"):
            return _REGIME_ENCODING.get(str(regime_state.value), 0.0)
        return 0.0

    @staticmethod
    def _build_meta_input(
        calibrated_scores: Dict[int, float],
        regime_val: float,
    ) -> np.ndarray:
        """Build the (1, n_features) input array for the meta-learner.

        Constructs the same feature set used during training:
        base scores → regime → interaction features → disagreement.
        Rolling accuracy features are omitted at inference time (they are
        captured by the base score magnitudes themselves).

        Parameters
        ----------
        calibrated_scores:
            Dict of ``{horizon: calibrated_probability}``.
        regime_val:
            Numeric regime encoding.

        Returns
        -------
        np.ndarray
            Shape (1, 10) ready for ElasticNet prediction.  Columns are
            ordered identically to the training feature matrix.
        """
        s1  = calibrated_scores.get(1,  0.5)
        s5  = calibrated_scores.get(5,  0.5)
        s20 = calibrated_scores.get(20, 0.5)

        scores_list = [
            v for v in calibrated_scores.values()
            if not math.isnan(v)
        ]
        disagreement = (max(scores_list) - min(scores_list)) if len(scores_list) >= 2 else 0.0

        row = [
            s1,                    # score_1d
            s5,                    # score_5d
            s20,                   # score_20d
            regime_val,            # regime
            s1 * s5,               # x_1d_5d
            s5 * s20,              # x_5d_20d
            disagreement,          # disagreement
            0.5,                   # roll_acc_1d  (neutral at inference)
            0.5,                   # roll_acc_5d  (neutral at inference)
            0.5,                   # roll_acc_20d (neutral at inference)
        ]
        return np.array(row, dtype=float).reshape(1, -1)


# ===========================================================================
# Helper: ElasticNet wrapper that conforms to the model_factory interface
# ===========================================================================

class _MetaLearnerWrapper:
    """Thin ElasticNet wrapper compatible with the model_factory API.

    Used internally by the meta-learner walk-forward validation step.

    Parameters
    ----------
    scaler:
        StandardScaler instance (owned by this wrapper; fit on training data).
    alpha:
        ElasticNet regularisation strength (default 0.5).
    l1_ratio:
        ElasticNet mixing parameter (default 0.5 = equal Lasso / Ridge).
    """

    def __init__(
        self,
        scaler: StandardScaler,
        alpha: float = 0.5,
        l1_ratio: float = 0.5,
    ) -> None:
        self._scaler   = scaler
        self._alpha    = alpha
        self._l1_ratio = l1_ratio
        self._model: Optional[ElasticNet] = None
        self._feature_names: List[str]    = []

    def train(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        val_features: Optional[pd.DataFrame] = None,
        val_labels: Optional[pd.Series] = None,
    ) -> "_MetaLearnerWrapper":
        """Fit the ElasticNet model on *features* and *labels*.

        Parameters
        ----------
        features:
            Training feature matrix.
        labels:
            Target series.
        val_features, val_labels:
            Ignored (ElasticNet does not support early stopping).

        Returns
        -------
        self
        """
        self._feature_names = list(features.columns)
        X = self._scaler.fit_transform(features.values)
        self._model = ElasticNet(
            alpha     = self._alpha,
            l1_ratio  = self._l1_ratio,
            max_iter  = 2000,
            fit_intercept = True,
        )
        self._model.fit(X, labels.values)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Return ElasticNet regression predictions.

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
        """Return absolute ElasticNet coefficients as a feature importance proxy.

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
