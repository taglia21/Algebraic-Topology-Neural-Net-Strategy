"""
ML Ensemble Stacker (TIER 4 — Maximize Alpha)
===============================================

Advanced ML stacking ensemble combining gradient boosters with a
ridge meta-learner for alpha prediction.

Components:
1. **Base models** — XGBoost, LightGBM, CatBoost (diverse learners)
2. **Meta-learner** — Ridge regression on out-of-fold predictions
3. **Walk-forward CV** — Expanding-window cross-validation
4. **SHAP importance** — Global & local feature explanations
5. **Probability calibration** — Isotonic regression for reliable scores
6. **predict_alpha()** — Returns calibrated confidence score [0, 1]

Usage:
    from src.ml_ensemble_stacker import MLEnsembleStacker, StackerConfig

    stacker = MLEnsembleStacker()
    stacker.fit(X_train, y_train)
    alpha = stacker.predict_alpha(X_new)   # 0-1 confidence
    importance = stacker.feature_importance()
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)

# Optional heavy imports — degrade gracefully
try:
    import xgboost as xgb
    XGB_OK = True
except ImportError:
    XGB_OK = False

try:
    import lightgbm as lgb
    LGB_OK = True
except ImportError:
    LGB_OK = False

try:
    import catboost as cb
    CB_OK = True
except ImportError:
    CB_OK = False

try:
    from sklearn.linear_model import Ridge
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, mean_squared_error
    SK_OK = True
except ImportError:
    SK_OK = False

try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class StackerConfig:
    """Configuration for the stacking ensemble."""
    # Walk-forward CV
    n_splits: int = 5
    min_train_size: int = 252          # ~1 year of daily bars

    # XGBoost defaults
    xgb_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "verbosity": 0,
        "n_jobs": -1,
    })

    # LightGBM defaults
    lgb_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "n_jobs": -1,
        "force_col_wise": True,
    })

    # CatBoost defaults
    cb_params: Dict[str, Any] = field(default_factory=lambda: {
        "iterations": 300,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3.0,
        "loss_function": "Logloss",
        "verbose": 0,
        "thread_count": -1,
    })

    # Meta-learner
    meta_alpha: float = 1.0            # Ridge regularisation
    use_calibration: bool = True       # Isotonic calibration
    use_shap: bool = True              # Compute SHAP values

    # Feature guardrails
    max_features: int = 200
    clip_predictions: bool = True


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class StackerResult:
    """Prediction result from the stacker."""
    alpha_score: float = 0.5           # calibrated [0, 1]
    raw_score: float = 0.5             # pre-calibration
    base_scores: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0            # agreement among bases
    regime_flag: str = ""


@dataclass
class FeatureImportanceEntry:
    """Feature importance record."""
    feature: str = ""
    importance: float = 0.0
    shap_mean: float = 0.0
    rank: int = 0


# =============================================================================
# ML ENSEMBLE STACKER
# =============================================================================

class MLEnsembleStacker:
    """
    Three-model stacking ensemble with walk-forward validation.

    Workflow:
        1. fit(X, y)  — trains base models via walk-forward OOF predictions,
                         then trains Ridge meta-learner + isotonic calibrator.
        2. predict_alpha(X) — returns calibrated confidence ∈ [0, 1].
        3. feature_importance() — SHAP-based feature ranking.
    """

    def __init__(self, config: Optional[StackerConfig] = None):
        self.config = config or StackerConfig()
        self._base_models: Dict[str, Any] = {}
        self._meta_model: Any = None
        self._calibrator: Any = None
        self._scaler: Any = None
        self._feature_names: List[str] = []
        self._is_fitted: bool = False
        self._oof_auc: float = 0.0
        self._shap_values: Optional[np.ndarray] = None
        logger.info("MLEnsembleStacker initialised (splits=%d)", self.config.n_splits)

    # ── Fit pipeline ─────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: Optional[List[str]] = None) -> "MLEnsembleStacker":
        """
        Train the stacking ensemble with walk-forward cross-validation.

        Parameters
        ----------
        X : array-like (n_samples, n_features)
        y : array-like (n_samples,) binary labels {0, 1}
        feature_names : optional list of feature names

        Returns self for chaining.
        """
        if not SK_OK:
            raise RuntimeError("scikit-learn is required for MLEnsembleStacker")

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        n, d = X.shape
        self._feature_names = feature_names or [f"f{i}" for i in range(d)]

        logger.info("Fitting stacker: n=%d, d=%d", n, d)

        # ---- Walk-forward OOF predictions ----
        tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
        oof_preds = {name: np.full(n, np.nan) for name in self._model_names()}
        models_last: Dict[str, Any] = {}

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            if len(train_idx) < self.config.min_train_size:
                continue
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va = X[val_idx]

            for name, model in self._make_base_models().items():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_tr, y_tr)
                preds = self._predict_proba(model, X_va)
                oof_preds[name][val_idx] = preds
                models_last[name] = model

        # ---- Retrain on full data ----
        for name, model in self._make_base_models().items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X, y)
            self._base_models[name] = model

        # ---- Meta-learner (Ridge on OOF predictions) ----
        valid_mask = ~np.isnan(oof_preds[self._model_names()[0]])
        if valid_mask.sum() < 20:
            logger.warning("Too few OOF samples (%d); using simple average", valid_mask.sum())
            self._meta_model = None
        else:
            meta_X = np.column_stack([oof_preds[n][valid_mask] for n in self._model_names()])
            meta_y = y[valid_mask]
            self._scaler = StandardScaler()
            meta_X_scaled = self._scaler.fit_transform(meta_X)
            self._meta_model = Ridge(alpha=self.config.meta_alpha)
            self._meta_model.fit(meta_X_scaled, meta_y)

            # OOF AUC
            try:
                raw = self._meta_model.predict(meta_X_scaled)
                self._oof_auc = float(roc_auc_score(meta_y, raw))
            except Exception:
                self._oof_auc = 0.5

            # ---- Isotonic calibration ----
            if self.config.use_calibration:
                raw_oof = self._meta_model.predict(meta_X_scaled)
                self._calibrator = IsotonicRegression(
                    y_min=0.0, y_max=1.0, out_of_bounds="clip",
                )
                self._calibrator.fit(raw_oof, meta_y)

        # ---- SHAP (on XGBoost as representative) ----
        if self.config.use_shap and SHAP_OK and "xgb" in self._base_models:
            try:
                explainer = shap.TreeExplainer(self._base_models["xgb"])
                sample = X[:min(500, n)]
                self._shap_values = explainer.shap_values(sample)
            except Exception as e:
                logger.warning("SHAP computation failed: %s", e)

        self._is_fitted = True
        logger.info("Stacker fitted — OOF AUC=%.4f", self._oof_auc)
        return self

    # ── Prediction ───────────────────────────────────────────────────────

    def predict_alpha(self, X: np.ndarray) -> np.ndarray:
        """
        Predict calibrated alpha score ∈ [0, 1].

        Parameters
        ----------
        X : array-like (n_samples, n_features)

        Returns
        -------
        scores : np.ndarray of shape (n_samples,)
        """
        X = np.asarray(X, dtype=np.float32)
        base_preds = {}
        for name, model in self._base_models.items():
            base_preds[name] = self._predict_proba(model, X)

        if self._meta_model is not None and self._scaler is not None:
            meta_X = np.column_stack([base_preds[n] for n in self._model_names()])
            meta_X_scaled = self._scaler.transform(meta_X)
            raw = self._meta_model.predict(meta_X_scaled)
        else:
            # Simple average fallback
            raw = np.mean(list(base_preds.values()), axis=0)

        if self._calibrator is not None:
            scores = self._calibrator.predict(raw)
        else:
            scores = np.clip(raw, 0, 1)

        if self.config.clip_predictions:
            scores = np.clip(scores, 0, 1)

        return scores.astype(np.float64)

    def predict_single(self, X: np.ndarray) -> StackerResult:
        """Predict a single sample with detailed breakdown."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        base_scores = {}
        for name, model in self._base_models.items():
            base_scores[name] = float(self._predict_proba(model, X)[0])

        alpha = float(self.predict_alpha(X)[0])
        raw = float(np.mean(list(base_scores.values())))
        preds = list(base_scores.values())
        confidence = 1.0 - float(np.std(preds)) * 2 if len(preds) > 1 else 0.5

        return StackerResult(
            alpha_score=alpha,
            raw_score=raw,
            base_scores=base_scores,
            confidence=max(0.0, min(1.0, confidence)),
        )

    # ── Feature importance ───────────────────────────────────────────────

    def feature_importance(self, top_n: int = 20) -> List[FeatureImportanceEntry]:
        """Return ranked feature importance (SHAP-based if available)."""
        entries: List[FeatureImportanceEntry] = []

        if self._shap_values is not None and SHAP_OK:
            sv = self._shap_values
            if isinstance(sv, list):
                sv = sv[1] if len(sv) > 1 else sv[0]
            mean_abs = np.mean(np.abs(sv), axis=0)
            for i, importance in enumerate(mean_abs):
                name = self._feature_names[i] if i < len(self._feature_names) else f"f{i}"
                entries.append(FeatureImportanceEntry(
                    feature=name, importance=float(importance),
                    shap_mean=float(importance),
                ))
        elif "xgb" in self._base_models and hasattr(self._base_models["xgb"], "feature_importances_"):
            imp = self._base_models["xgb"].feature_importances_
            for i, importance in enumerate(imp):
                name = self._feature_names[i] if i < len(self._feature_names) else f"f{i}"
                entries.append(FeatureImportanceEntry(feature=name, importance=float(importance)))

        entries.sort(key=lambda e: e.importance, reverse=True)
        for rank, e in enumerate(entries, 1):
            e.rank = rank
        return entries[:top_n]

    # ── Metrics & serialisation ──────────────────────────────────────────

    @property
    def oof_auc(self) -> float:
        return self._oof_auc

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def meta_weights(self) -> Optional[np.ndarray]:
        if self._meta_model is not None and hasattr(self._meta_model, "coef_"):
            return self._meta_model.coef_
        return None

    def summary(self) -> Dict[str, Any]:
        """Return a summary dict."""
        return {
            "is_fitted": self._is_fitted,
            "oof_auc": self._oof_auc,
            "base_models": list(self._base_models.keys()),
            "n_features": len(self._feature_names),
            "meta_weights": self.meta_weights.tolist() if self.meta_weights is not None else None,
            "has_calibrator": self._calibrator is not None,
            "has_shap": self._shap_values is not None,
        }

    # ── Internals ────────────────────────────────────────────────────────

    def _model_names(self) -> List[str]:
        names = []
        if XGB_OK:
            names.append("xgb")
        if LGB_OK:
            names.append("lgb")
        if CB_OK:
            names.append("cb")
        if not names:
            raise RuntimeError("No gradient boosting library available")
        return names

    def _make_base_models(self) -> Dict[str, Any]:
        models: Dict[str, Any] = {}
        if XGB_OK:
            models["xgb"] = xgb.XGBClassifier(**self.config.xgb_params)
        if LGB_OK:
            models["lgb"] = lgb.LGBMClassifier(**self.config.lgb_params)
        if CB_OK:
            models["cb"] = cb.CatBoostClassifier(**self.config.cb_params)
        return models

    @staticmethod
    def _predict_proba(model: Any, X: np.ndarray) -> np.ndarray:
        """Extract probability of positive class."""
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            if proba.ndim == 2:
                return proba[:, 1]
            return proba
        return model.predict(X)


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.random.seed(42)

    n, d = 1000, 20
    X = np.random.randn(n, d).astype(np.float32)
    w = np.random.randn(d)
    y = (X @ w + np.random.randn(n) * 0.5 > 0).astype(np.float32)

    stacker = MLEnsembleStacker(StackerConfig(n_splits=3, min_train_size=100))
    stacker.fit(X, y)

    scores = stacker.predict_alpha(X[:10])
    print(f"Scores: {scores}")
    print(f"OOF AUC: {stacker.oof_auc:.4f}")
    print(f"Summary: {stacker.summary()}")

    importance = stacker.feature_importance(top_n=5)
    for e in importance:
        print(f"  #{e.rank} {e.feature}: {e.importance:.4f}")
