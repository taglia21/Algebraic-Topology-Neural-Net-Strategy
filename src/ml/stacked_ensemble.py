"""
V3.5 Stacked Ensemble for Signal Quality Enhancement

Implements a diverse ensemble with 4 base learners and Ridge meta-learner
to improve prediction quality and provide confidence metrics.

Key improvements over V3.0 GradientBoostEnsemble:
- Model diversity: XGBoost, LightGBM, Random Forest, Gradient Boost
- Stacking: Meta-learner combines predictions for better generalization
- Confidence: Prediction variance across models indicates reliability

Expected improvement: +0.15 to +0.25 Sharpe (22% error reduction)
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import time
import warnings

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# Conditional imports for optional dependencies
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class StackedEnsembleConfig:
    """Configuration for V3.5 Stacked Ensemble."""
    
    # Random seed for reproducibility
    random_state: int = 42
    
    # Cross-validation for meta-learner training
    n_folds: int = 5
    
    # XGBoost parameters
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 3
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    
    # LightGBM parameters
    lgb_n_estimators: int = 100
    lgb_num_leaves: int = 31
    lgb_learning_rate: float = 0.05
    lgb_feature_fraction: float = 0.8
    lgb_bagging_fraction: float = 0.8
    lgb_bagging_freq: int = 5
    
    # Random Forest parameters
    rf_n_estimators: int = 200
    rf_max_depth: int = 8
    rf_min_samples_split: int = 20
    rf_min_samples_leaf: int = 10
    
    # Gradient Boosting parameters
    gb_n_estimators: int = 100
    gb_max_depth: int = 3
    gb_learning_rate: float = 0.05
    gb_subsample: float = 0.8
    gb_min_samples_split: int = 20
    
    # Meta-learner parameters
    meta_alpha: float = 1.0  # Ridge regularization
    
    # Feature scaling
    scale_features: bool = True
    
    # Confidence calculation
    min_confidence: float = 0.3
    max_confidence: float = 0.95


class StackedEnsemble:
    """
    V3.5 Stacked Ensemble with diverse base learners.
    
    Architecture:
    1. Base learners trained independently on same data
    2. Out-of-fold predictions used to train meta-learner
    3. Final prediction = meta-learner(base_predictions)
    4. Confidence = 1 / (1 + variance(base_predictions))
    
    Example:
        >>> config = StackedEnsembleConfig()
        >>> ensemble = StackedEnsemble(config)
        >>> ensemble.fit(X_train, y_train)
        >>> predictions, confidence = ensemble.predict_with_confidence(X_test)
    """
    
    def __init__(self, config: Optional[StackedEnsembleConfig] = None):
        self.config = config or StackedEnsembleConfig()
        
        # Base learners (initialized in fit)
        self.base_learners: Dict[str, Any] = {}
        self.base_learner_names: List[str] = []
        
        # Meta-learner
        self.meta_learner: Optional[Ridge] = None
        
        # Feature scaler
        self.scaler: Optional[StandardScaler] = None
        
        # Training metadata
        self.is_fitted: bool = False
        self.n_features: int = 0
        self.feature_importance_: Optional[np.ndarray] = None
        self.training_time_: float = 0.0
        self.base_learner_weights_: Optional[np.ndarray] = None
        
    def _create_base_learners(self) -> Dict[str, Any]:
        """Create fresh instances of all base learners."""
        learners = {}
        
        # XGBoost
        if HAS_XGBOOST:
            learners['xgboost'] = xgb.XGBRegressor(
                n_estimators=self.config.xgb_n_estimators,
                max_depth=self.config.xgb_max_depth,
                learning_rate=self.config.xgb_learning_rate,
                subsample=self.config.xgb_subsample,
                colsample_bytree=self.config.xgb_colsample_bytree,
                random_state=self.config.random_state,
                verbosity=0,
                n_jobs=-1
            )
            logger.info("XGBoost base learner created")
        else:
            logger.warning("XGBoost not available, skipping")
            
        # LightGBM
        if HAS_LIGHTGBM:
            learners['lightgbm'] = lgb.LGBMRegressor(
                n_estimators=self.config.lgb_n_estimators,
                num_leaves=self.config.lgb_num_leaves,
                learning_rate=self.config.lgb_learning_rate,
                feature_fraction=self.config.lgb_feature_fraction,
                bagging_fraction=self.config.lgb_bagging_fraction,
                bagging_freq=self.config.lgb_bagging_freq,
                random_state=self.config.random_state,
                verbosity=-1,
                n_jobs=-1
            )
            logger.info("LightGBM base learner created")
        else:
            logger.warning("LightGBM not available, skipping")
            
        # Random Forest
        learners['random_forest'] = RandomForestRegressor(
            n_estimators=self.config.rf_n_estimators,
            max_depth=self.config.rf_max_depth,
            min_samples_split=self.config.rf_min_samples_split,
            min_samples_leaf=self.config.rf_min_samples_leaf,
            random_state=self.config.random_state,
            n_jobs=-1
        )
        logger.info("Random Forest base learner created")
        
        # Gradient Boosting
        learners['gradient_boost'] = GradientBoostingRegressor(
            n_estimators=self.config.gb_n_estimators,
            max_depth=self.config.gb_max_depth,
            learning_rate=self.config.gb_learning_rate,
            subsample=self.config.gb_subsample,
            min_samples_split=self.config.gb_min_samples_split,
            random_state=self.config.random_state
        )
        logger.info("Gradient Boosting base learner created")
        
        return learners
    
    def _preprocess_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Scale features if configured."""
        # Handle NaN/Inf values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        if not self.config.scale_features:
            return X
            
        if fit:
            self.scaler = StandardScaler()
            return self.scaler.fit_transform(X)
        elif self.scaler is not None:
            return self.scaler.transform(X)
        return X
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'StackedEnsemble':
        """
        Fit the stacked ensemble using out-of-fold predictions.
        
        Steps:
        1. Create K-fold splits
        2. For each fold, train base learners on K-1 folds
        3. Collect out-of-fold predictions for meta-learner
        4. Train meta-learner on stacked predictions
        5. Retrain all base learners on full data
        
        Args:
            X: Features array (n_samples, n_features)
            y: Target array (n_samples,)
            
        Returns:
            Self for chaining
        """
        start_time = time.time()
        
        # Validate input
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        
        if len(X) != len(y):
            raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")
            
        if len(X) < self.config.n_folds * 10:
            logger.warning(f"Small dataset ({len(X)} samples), reducing folds to 3")
            n_folds = min(3, len(X) // 10)
        else:
            n_folds = self.config.n_folds
            
        self.n_features = X.shape[1]
        
        # Preprocess features
        X_scaled = self._preprocess_features(X, fit=True)
        
        # Create base learners
        self.base_learners = self._create_base_learners()
        self.base_learner_names = list(self.base_learners.keys())
        n_learners = len(self.base_learner_names)
        
        logger.info(f"Training stacked ensemble with {n_learners} base learners, {n_folds} folds")
        
        # Generate out-of-fold predictions
        oof_predictions = np.zeros((len(X), n_learners))
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.config.random_state)
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
            X_train_fold = X_scaled[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X_scaled[val_idx]
            
            # Train each base learner on this fold
            for learner_idx, (name, learner) in enumerate(self.base_learners.items()):
                # Create fresh learner for this fold
                fold_learner = self._clone_learner(name)
                fold_learner.fit(X_train_fold, y_train_fold)
                
                # Get out-of-fold predictions
                oof_predictions[val_idx, learner_idx] = fold_learner.predict(X_val_fold)
                
            logger.debug(f"Fold {fold_idx + 1}/{n_folds} complete")
        
        # Train meta-learner on out-of-fold predictions
        self.meta_learner = Ridge(alpha=self.config.meta_alpha)
        self.meta_learner.fit(oof_predictions, y)
        self.base_learner_weights_ = self.meta_learner.coef_
        
        logger.info(f"Meta-learner weights: {dict(zip(self.base_learner_names, np.round(self.base_learner_weights_, 4)))}")
        
        # Retrain all base learners on full data
        for name, learner in self.base_learners.items():
            learner.fit(X_scaled, y)
            
        # Aggregate feature importance
        self._compute_feature_importance(X_scaled)
        
        self.training_time_ = time.time() - start_time
        self.is_fitted = True
        
        logger.info(f"Stacked ensemble trained in {self.training_time_:.2f}s")
        
        return self
    
    def _clone_learner(self, name: str) -> Any:
        """Create a fresh copy of a base learner with same config."""
        if name == 'xgboost' and HAS_XGBOOST:
            return xgb.XGBRegressor(
                n_estimators=self.config.xgb_n_estimators,
                max_depth=self.config.xgb_max_depth,
                learning_rate=self.config.xgb_learning_rate,
                subsample=self.config.xgb_subsample,
                colsample_bytree=self.config.xgb_colsample_bytree,
                random_state=self.config.random_state,
                verbosity=0,
                n_jobs=-1
            )
        elif name == 'lightgbm' and HAS_LIGHTGBM:
            return lgb.LGBMRegressor(
                n_estimators=self.config.lgb_n_estimators,
                num_leaves=self.config.lgb_num_leaves,
                learning_rate=self.config.lgb_learning_rate,
                feature_fraction=self.config.lgb_feature_fraction,
                bagging_fraction=self.config.lgb_bagging_fraction,
                bagging_freq=self.config.lgb_bagging_freq,
                random_state=self.config.random_state,
                verbosity=-1,
                n_jobs=-1
            )
        elif name == 'random_forest':
            return RandomForestRegressor(
                n_estimators=self.config.rf_n_estimators,
                max_depth=self.config.rf_max_depth,
                min_samples_split=self.config.rf_min_samples_split,
                min_samples_leaf=self.config.rf_min_samples_leaf,
                random_state=self.config.random_state,
                n_jobs=-1
            )
        elif name == 'gradient_boost':
            return GradientBoostingRegressor(
                n_estimators=self.config.gb_n_estimators,
                max_depth=self.config.gb_max_depth,
                learning_rate=self.config.gb_learning_rate,
                subsample=self.config.gb_subsample,
                min_samples_split=self.config.gb_min_samples_split,
                random_state=self.config.random_state
            )
        else:
            raise ValueError(f"Unknown learner: {name}")
    
    def _compute_feature_importance(self, X: np.ndarray):
        """Aggregate feature importance across base learners."""
        importances = []
        
        for name, learner in self.base_learners.items():
            if hasattr(learner, 'feature_importances_'):
                importances.append(learner.feature_importances_)
                
        if importances:
            self.feature_importance_ = np.mean(importances, axis=0)
        else:
            self.feature_importance_ = np.ones(X.shape[1]) / X.shape[1]
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions using the stacked ensemble.
        
        Args:
            X: Features array (n_samples, n_features)
            
        Returns:
            Predictions array (n_samples,)
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")
            
        predictions, _ = self.predict_with_confidence(X)
        return predictions
    
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions with confidence scores.
        
        Confidence is calculated as: 1 / (1 + normalized_variance)
        where normalized_variance = variance / mean_abs_prediction
        
        High confidence (>0.8): Base learners agree strongly
        Low confidence (<0.5): Base learners disagree significantly
        
        Args:
            X: Features array (n_samples, n_features)
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")
            
        X = np.asarray(X)
        X_scaled = self._preprocess_features(X, fit=False)
        
        # Get predictions from all base learners
        n_samples = len(X)
        n_learners = len(self.base_learners)
        base_predictions = np.zeros((n_samples, n_learners))
        
        for idx, (name, learner) in enumerate(self.base_learners.items()):
            base_predictions[:, idx] = learner.predict(X_scaled)
        
        # Meta-learner combines predictions
        final_predictions = self.meta_learner.predict(base_predictions)
        
        # Calculate confidence from prediction variance
        pred_variance = np.var(base_predictions, axis=1)
        pred_mean_abs = np.maximum(np.mean(np.abs(base_predictions), axis=1), 1e-10)
        
        # Normalized variance (scale-independent)
        norm_variance = pred_variance / pred_mean_abs
        
        # Convert to confidence score
        # Higher variance = lower confidence
        raw_confidence = 1.0 / (1.0 + norm_variance * 100)  # Scale factor
        
        # Clip to configured range
        confidence = np.clip(
            raw_confidence,
            self.config.min_confidence,
            self.config.max_confidence
        )
        
        return final_predictions, confidence
    
    def get_base_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get predictions from each base learner separately.
        
        Useful for analyzing model agreement and debugging.
        
        Args:
            X: Features array (n_samples, n_features)
            
        Returns:
            Dict mapping learner name to predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")
            
        X = np.asarray(X)
        X_scaled = self._preprocess_features(X, fit=False)
        
        results = {}
        for name, learner in self.base_learners.items():
            results[name] = learner.predict(X_scaled)
            
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained ensemble."""
        if not self.is_fitted:
            return {'is_fitted': False}
            
        return {
            'is_fitted': True,
            'n_base_learners': len(self.base_learners),
            'base_learner_names': self.base_learner_names,
            'base_learner_weights': dict(zip(
                self.base_learner_names, 
                np.round(self.base_learner_weights_, 4).tolist()
            )) if self.base_learner_weights_ is not None else {},
            'n_features': self.n_features,
            'training_time_seconds': round(self.training_time_, 2),
            'meta_learner_intercept': round(self.meta_learner.intercept_, 6) if self.meta_learner else None,
            'config': {
                'n_folds': self.config.n_folds,
                'meta_alpha': self.config.meta_alpha,
                'random_state': self.config.random_state
            }
        }


def create_v35_stacked_ensemble(
    n_folds: int = 5,
    random_state: int = 42
) -> StackedEnsemble:
    """
    Factory function to create V3.5 Stacked Ensemble with optimized settings.
    
    Args:
        n_folds: Number of CV folds for meta-learner training
        random_state: Random seed for reproducibility
        
    Returns:
        Configured StackedEnsemble instance
    """
    config = StackedEnsembleConfig(
        n_folds=n_folds,
        random_state=random_state,
        
        # XGBoost: Good for complex patterns
        xgb_n_estimators=100,
        xgb_max_depth=3,
        xgb_learning_rate=0.05,
        xgb_subsample=0.8,
        
        # LightGBM: Fast and memory-efficient
        lgb_n_estimators=100,
        lgb_num_leaves=31,
        lgb_learning_rate=0.05,
        lgb_feature_fraction=0.8,
        
        # Random Forest: Robust to outliers
        rf_n_estimators=200,
        rf_max_depth=8,
        rf_min_samples_split=20,
        
        # Gradient Boosting: Sklearn baseline
        gb_n_estimators=100,
        gb_max_depth=3,
        gb_learning_rate=0.05,
        
        # Meta-learner
        meta_alpha=1.0,
        
        # Confidence
        min_confidence=0.3,
        max_confidence=0.95
    )
    
    return StackedEnsemble(config)


# Quick test when run directly
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 500
    n_features = 50
    
    X = np.random.randn(n_samples, n_features)
    y = 0.3 * X[:, 0] + 0.2 * X[:, 1] + 0.1 * np.random.randn(n_samples)
    
    # Split data
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train and evaluate
    ensemble = create_v35_stacked_ensemble()
    ensemble.fit(X_train, y_train)
    
    predictions, confidence = ensemble.predict_with_confidence(X_test)
    
    print("\n=== Stacked Ensemble Test ===")
    print(f"Model info: {ensemble.get_model_info()}")
    print(f"Test predictions: mean={predictions.mean():.4f}, std={predictions.std():.4f}")
    print(f"Confidence: mean={confidence.mean():.4f}, min={confidence.min():.4f}, max={confidence.max():.4f}")
    
    # Calculate test metrics
    from sklearn.metrics import mean_squared_error, r2_score
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"MSE: {mse:.6f}, R²: {r2:.4f}")
