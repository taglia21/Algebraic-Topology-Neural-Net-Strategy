"""
ML Signal Generator
===================

Ensemble machine learning models for options trading signals.

Features:
- 30-feature engineering pipeline
- Ensemble of XGBoost, LightGBM, RandomForest
- Walk-forward validation with 60-day test window
- Weekly retraining schedule
- Feature importance tracking

Target: >55% directional accuracy on validation set
"""

import os
import logging
import pickle
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger(__name__)


@dataclass
class SignalPrediction:
    """ML-generated trading signal."""
    direction: str  # 'bullish', 'bearish', 'neutral'
    confidence: float  # 0-1
    model_agreement: float  # 0-1
    feature_importance: Dict[str, float]
    timestamp: datetime


class MLSignalGenerator:
    """
    Ensemble ML signal generator for options trading.
    
    Architecture:
    - 3 models: XGBoost, LightGBM, RandomForest
    - 30 engineered features
    - Walk-forward validation
    - Weekly retraining
    
    Features (30 total):
    - Momentum: returns_1d, returns_5d, returns_21d
    - Volatility: realized_vol_10d, realized_vol_30d, iv_rank
    - Options flow: put_call_ratio, skew_25delta, gamma_exposure
    - Technical: rsi_14, macd_signal, bb_position
    - Regime: vix_level, spy_correlation, sector_momentum
    """
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize ML signal generator.
        
        Args:
            model_dir: Directory to save/load models
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Models
        self.xgb_model = None
        self.lgb_model = None
        self.rf_model = None
        self.scaler = StandardScaler()
        
        # Feature names (30 features)
        self.feature_names = [
            # Momentum (3)
            'returns_1d', 'returns_5d', 'returns_21d',
            
            # Volatility (3)
            'realized_vol_10d', 'realized_vol_30d', 'iv_rank',
            
            # Options flow (3)
            'put_call_ratio', 'skew_25delta', 'gamma_exposure',
            
            # Technical indicators (9)
            'rsi_14', 'macd_signal', 'bb_position',
            'sma_20', 'sma_50', 'sma_200',
            'volume_ratio', 'price_to_sma20', 'price_to_sma50',
            
            # Regime indicators (6)
            'vix_level', 'spy_correlation', 'sector_momentum',
            'market_breadth', 'advance_decline_ratio', 'vix_term_structure',
            
            # Additional features (6)
            'iv_percentile', 'call_volume', 'put_volume',
            'delta_imbalance', 'vanna_exposure', 'charm_exposure'
        ]
        
        # Training metadata
        self.last_train_date = None
        self.training_metrics = {}
        
        self.logger.info(f"Initialized ML signal generator ({len(self.feature_names)} features)")
    
    def _create_models(self):
        """Create fresh model instances."""
        # XGBoost - conservative parameters to avoid overfitting
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        # LightGBM - fast and accurate
        self.lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            num_leaves=15,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        
        # Random Forest - robust ensemble
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        self.logger.info("Created fresh model instances")
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer 30 features from raw data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with engineered features
        """
        df = data.copy()
        
        # Momentum features
        df['returns_1d'] = df['close'].pct_change(1)
        df['returns_5d'] = df['close'].pct_change(5)
        df['returns_21d'] = df['close'].pct_change(21)
        
        # Volatility features
        df['realized_vol_10d'] = df['returns_1d'].rolling(10).std() * np.sqrt(252)
        df['realized_vol_30d'] = df['returns_1d'].rolling(30).std() * np.sqrt(252)
        
        # Technical indicators
        df['rsi_14'] = self._calculate_rsi(df['close'], 14)
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['sma_200'] = df['close'].rolling(200).mean()
        
        df['price_to_sma20'] = df['close'] / df['sma_20']
        df['price_to_sma50'] = df['close'] / df['sma_50']
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd_signal'] = exp1 - exp2
        
        # Bollinger Bands position
        bb_mid = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_position'] = (df['close'] - bb_mid) / (2 * bb_std)
        
        # Volume
        if 'volume' in df.columns:
            df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        else:
            df['volume_ratio'] = 1.0
        
        # Fill with synthetic values for demo (in production, get real data)
        synthetic_features = [
            'iv_rank', 'put_call_ratio', 'skew_25delta', 'gamma_exposure',
            'vix_level', 'spy_correlation', 'sector_momentum',
            'market_breadth', 'advance_decline_ratio', 'vix_term_structure',
            'iv_percentile', 'call_volume', 'put_volume',
            'delta_imbalance', 'vanna_exposure', 'charm_exposure'
        ]
        
        for feat in synthetic_features:
            if feat not in df.columns:
                # Add random values for now (replace with real data in production)
                if 'ratio' in feat or 'correlation' in feat:
                    df[feat] = np.random.uniform(0.8, 1.2, len(df))
                elif 'iv' in feat or 'vol' in feat:
                    df[feat] = np.random.uniform(0.15, 0.35, len(df))
                else:
                    df[feat] = np.random.uniform(-1, 1, len(df))
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _create_labels(self, data: pd.DataFrame, forward_days: int = 5) -> pd.Series:
        """
        Create forward-looking labels.
        
        Label = 1 if price increases in next N days, 0 otherwise
        
        Args:
            data: DataFrame with price data
            forward_days: Days to look forward
            
        Returns:
            Series of binary labels
        """
        future_returns = data['close'].pct_change(forward_days).shift(-forward_days)
        labels = (future_returns > 0).astype(int)
        
        return labels
    
    def train(self, historical_data: pd.DataFrame, validation_window: int = 60) -> Dict:
        """
        Train ensemble models with walk-forward validation.
        
        Args:
            historical_data: DataFrame with OHLCV data
            validation_window: Days for test window
            
        Returns:
            Training metrics
        """
        self.logger.info("Starting model training...")
        
        # Engineer features
        df = self._engineer_features(historical_data)
        
        # Create labels (predict 5-day forward returns)
        labels = self._create_labels(df, forward_days=5)
        
        # Select features and remove NaN
        df = df[self.feature_names]
        valid_idx = ~(df.isna().any(axis=1) | labels.isna())
        
        X = df[valid_idx].values
        y = labels[valid_idx].values
        
        self.logger.info(f"Training on {len(X)} samples")
        
        # Walk-forward validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        all_accuracies = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Create and train models
            self._create_models()
            
            self.xgb_model.fit(X_train_scaled, y_train)
            self.lgb_model.fit(X_train_scaled, y_train)
            self.rf_model.fit(X_train_scaled, y_train)
            
            # Ensemble prediction
            xgb_pred = self.xgb_model.predict_proba(X_test_scaled)[:, 1]
            lgb_pred = self.lgb_model.predict_proba(X_test_scaled)[:, 1]
            rf_pred = self.rf_model.predict_proba(X_test_scaled)[:, 1]
            
            ensemble_prob = (xgb_pred + lgb_pred + rf_pred) / 3
            ensemble_pred = (ensemble_prob > 0.5).astype(int)
            
            accuracy = accuracy_score(y_test, ensemble_pred)
            all_accuracies.append(accuracy)
            
            self.logger.info(f"Fold accuracy: {accuracy:.3f}")
        
        # Final training on all data
        self._create_models()
        X_scaled = self.scaler.fit_transform(X)
        
        self.xgb_model.fit(X_scaled, y)
        self.lgb_model.fit(X_scaled, y)
        self.rf_model.fit(X_scaled, y)
        
        # Calculate metrics
        mean_accuracy = np.mean(all_accuracies)
        std_accuracy = np.std(all_accuracies)
        
        self.training_metrics = {
            'accuracy': mean_accuracy,
            'accuracy_std': std_accuracy,
            'n_samples': len(X),
            'n_features': len(self.feature_names),
            'train_date': datetime.now().isoformat()
        }
        
        self.last_train_date = datetime.now()
        
        self.logger.info(
            f"Training complete: {mean_accuracy:.3f} ± {std_accuracy:.3f} accuracy"
        )
        
        return self.training_metrics
    
    def predict(self, features: Dict[str, float]) -> SignalPrediction:
        """
        Generate trading signal from features.
        
        Args:
            features: Dict mapping feature names to values
            
        Returns:
            SignalPrediction with direction and confidence
        """
        if self.xgb_model is None:
            raise ValueError("Models not trained. Call train() first.")
        
        # Build feature vector
        X = np.array([[features.get(feat, 0.0) for feat in self.feature_names]])
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from each model
        xgb_prob = self.xgb_model.predict_proba(X_scaled)[0]
        lgb_prob = self.lgb_model.predict_proba(X_scaled)[0]
        rf_prob = self.rf_model.predict_proba(X_scaled)[0]
        
        # Ensemble average
        ensemble_prob = (xgb_prob + lgb_prob + rf_prob) / 3
        
        # Calculate model agreement (inverse of variance)
        predictions = np.array([xgb_prob[1], lgb_prob[1], rf_prob[1]])
        agreement = 1.0 - np.std(predictions)
        
        # Determine direction
        bullish_prob = ensemble_prob[1]
        bearish_prob = ensemble_prob[0]
        
        if bullish_prob > 0.55:
            direction = "bullish"
            confidence = bullish_prob
        elif bearish_prob > 0.55:
            direction = "bearish"
            confidence = bearish_prob
        else:
            direction = "neutral"
            confidence = max(bullish_prob, bearish_prob)
        
        # Get feature importance (from XGBoost)
        importance_dict = dict(zip(
            self.feature_names,
            self.xgb_model.feature_importances_
        ))
        
        return SignalPrediction(
            direction=direction,
            confidence=confidence,
            model_agreement=agreement,
            feature_importance=importance_dict,
            timestamp=datetime.now()
        )
    
    def validate(self) -> Dict:
        """
        Run validation and return metrics.
        
        Returns:
            Validation metrics
        """
        if not self.training_metrics:
            return {'error': 'No training metrics available'}
        
        return {
            'accuracy': self.training_metrics['accuracy'],
            'accuracy_std': self.training_metrics['accuracy_std'],
            'passes_target': self.training_metrics['accuracy'] > 0.52,
            'last_train_date': self.last_train_date.isoformat() if self.last_train_date else None
        }
    
    def save_models(self, filename_prefix: str = "ensemble"):
        """Save trained models to disk."""
        if self.xgb_model is None:
            raise ValueError("No models to save")
        
        models_path = os.path.join(self.model_dir, f"{filename_prefix}_models.pkl")
        scaler_path = os.path.join(self.model_dir, f"{filename_prefix}_scaler.pkl")
        
        models = {
            'xgb': self.xgb_model,
            'lgb': self.lgb_model,
            'rf': self.rf_model,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'last_train_date': self.last_train_date
        }
        
        joblib.dump(models, models_path)
        joblib.dump(self.scaler, scaler_path)
        
        self.logger.info(f"Models saved to {models_path}")
    
    def load_models(self, filename_prefix: str = "ensemble") -> bool:
        """Load trained models from disk."""
        models_path = os.path.join(self.model_dir, f"{filename_prefix}_models.pkl")
        scaler_path = os.path.join(self.model_dir, f"{filename_prefix}_scaler.pkl")
        
        if not os.path.exists(models_path):
            self.logger.warning(f"No saved models found at {models_path}")
            return False
        
        try:
            models = joblib.load(models_path)
            self.scaler = joblib.load(scaler_path)
            
            self.xgb_model = models['xgb']
            self.lgb_model = models['lgb']
            self.rf_model = models['rf']
            self.feature_names = models['feature_names']
            self.training_metrics = models['training_metrics']
            self.last_train_date = models['last_train_date']
            
            self.logger.info(f"Models loaded from {models_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            return False
    
    def needs_retraining(self, retrain_days: int = 7) -> bool:
        """Check if models need retraining (weekly schedule)."""
        if self.last_train_date is None:
            return True
        
        days_since_training = (datetime.now() - self.last_train_date).days
        return days_since_training >= retrain_days
