#!/usr/bin/env python3
"""
V39 Deep ML Engine - Advanced Machine Learning Trading Module

Multi-model ensemble using XGBoost, LightGBM, and Neural Networks.
Comprehensive feature engineering with technical indicators.

Author: V39 Trading System
Date: 2026-01-26
"""

from dotenv import load_dotenv
load_dotenv()

import os
import sys
import logging
import argparse
import pickle
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, classification_report, confusion_matrix
)
from sklearn.ensemble import VotingClassifier

# ML libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not installed. Run: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    warnings.warn("LightGBM not installed. Run: pip install lightgbm")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MLConfig:
    """Configuration for ML engine."""
    # Feature parameters
    rsi_periods: List[int] = field(default_factory=lambda: [7, 14, 21])
    macd_params: List[Tuple[int, int, int]] = field(
        default_factory=lambda: [(12, 26, 9), (5, 13, 8)]
    )
    momentum_periods: List[int] = field(default_factory=lambda: [5, 10, 20])
    volatility_periods: List[int] = field(default_factory=lambda: [10, 20, 30])
    sma_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    ema_periods: List[int] = field(default_factory=lambda: [9, 21, 50])
    
    # Target parameters
    forward_periods: int = 5  # Days ahead to predict
    target_threshold: float = 0.02  # 2% threshold for classification
    
    # Model parameters
    test_size: float = 0.2
    n_splits: int = 5
    random_state: int = 42
    
    # Neural network params
    nn_hidden_layers: Tuple[int, ...] = (128, 64, 32)
    nn_max_iter: int = 500
    nn_alpha: float = 0.001
    
    # XGBoost params
    xgb_n_estimators: int = 200
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    
    # LightGBM params
    lgb_n_estimators: int = 200
    lgb_max_depth: int = 6
    lgb_learning_rate: float = 0.1
    
    # Paths
    model_dir: str = "models"


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

class FeatureEngineer:
    """Generate technical features for ML models."""
    
    def __init__(self, config: Optional[MLConfig] = None):
        self.config = config or MLConfig()
    
    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.inf)
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(prices: pd.Series, fast: int = 12, slow: int = 26, 
             signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD, Signal, and Histogram."""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def momentum(prices: pd.Series, period: int = 10) -> pd.Series:
        """Calculate price momentum (rate of change)."""
        return prices.pct_change(periods=period)
    
    @staticmethod
    def volatility(prices: pd.Series, period: int = 20) -> pd.Series:
        """Calculate rolling volatility (std of returns)."""
        returns = prices.pct_change()
        return returns.rolling(window=period).std()
    
    @staticmethod
    def sma(prices: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average."""
        return prices.rolling(window=period).mean()
    
    @staticmethod
    def ema(prices: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average."""
        return prices.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def bollinger_bands(prices: pd.Series, period: int = 20, 
                        std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, 
            period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                   k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=d_period).mean()
        return k, d
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all features from OHLCV data."""
        features = pd.DataFrame(index=df.index)
        
        close = df['close']
        high = df.get('high', close)
        low = df.get('low', close)
        volume = df.get('volume', pd.Series(1, index=df.index))
        
        # RSI features
        for period in self.config.rsi_periods:
            features[f'rsi_{period}'] = self.rsi(close, period)
        
        # MACD features
        for fast, slow, signal in self.config.macd_params:
            macd_line, signal_line, hist = self.macd(close, fast, slow, signal)
            features[f'macd_{fast}_{slow}'] = macd_line
            features[f'macd_signal_{fast}_{slow}'] = signal_line
            features[f'macd_hist_{fast}_{slow}'] = hist
        
        # Momentum features
        for period in self.config.momentum_periods:
            features[f'momentum_{period}'] = self.momentum(close, period)
        
        # Volatility features
        for period in self.config.volatility_periods:
            features[f'volatility_{period}'] = self.volatility(close, period)
        
        # SMA features and price ratios
        for period in self.config.sma_periods:
            sma = self.sma(close, period)
            features[f'sma_{period}'] = sma
            features[f'price_to_sma_{period}'] = close / sma
        
        # EMA features
        for period in self.config.ema_periods:
            ema = self.ema(close, period)
            features[f'ema_{period}'] = ema
            features[f'price_to_ema_{period}'] = close / ema
        
        # Bollinger Bands
        bb_upper, bb_mid, bb_lower = self.bollinger_bands(close, 20)
        features['bb_upper'] = bb_upper
        features['bb_lower'] = bb_lower
        features['bb_width'] = (bb_upper - bb_lower) / bb_mid
        features['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)
        
        # ATR
        features['atr_14'] = self.atr(high, low, close, 14)
        features['atr_pct'] = features['atr_14'] / close
        
        # Stochastic
        stoch_k, stoch_d = self.stochastic(high, low, close)
        features['stoch_k'] = stoch_k
        features['stoch_d'] = stoch_d
        
        # Volume features
        features['volume_sma_20'] = volume.rolling(window=20).mean()
        features['volume_ratio'] = volume / features['volume_sma_20']
        
        # Price patterns
        features['returns_1d'] = close.pct_change(1)
        features['returns_5d'] = close.pct_change(5)
        features['high_low_range'] = (high - low) / close
        features['close_position'] = (close - low) / (high - low)
        
        # Trend features
        features['trend_5_20'] = self.sma(close, 5) / self.sma(close, 20)
        features['trend_10_50'] = self.sma(close, 10) / self.sma(close, 50)
        
        return features
    
    def create_target(self, df: pd.DataFrame, forward_periods: int = 5,
                      threshold: float = 0.02) -> pd.Series:
        """Create classification target (1: up, 0: neutral, -1: down)."""
        future_returns = df['close'].pct_change(forward_periods).shift(-forward_periods)
        
        target = pd.Series(0, index=df.index)
        target[future_returns > threshold] = 1
        target[future_returns < -threshold] = -1
        
        return target


# =============================================================================
# ML MODELS
# =============================================================================

class DeepMLEngine:
    """Multi-model ML engine for trading predictions."""
    
    def __init__(self, config: Optional[MLConfig] = None):
        self.config = config or MLConfig()
        self.feature_engineer = FeatureEngineer(config)
        self.scaler = RobustScaler()
        self.models: Dict[str, Any] = {}
        self.ensemble = None
        self.feature_names: List[str] = []
        self.is_trained = False
        
        # Create model directory
        Path(self.config.model_dir).mkdir(exist_ok=True)
    
    def _build_xgboost(self) -> Optional[xgb.XGBClassifier]:
        """Build XGBoost classifier."""
        if not XGBOOST_AVAILABLE:
            return None
        
        return xgb.XGBClassifier(
            n_estimators=self.config.xgb_n_estimators,
            max_depth=self.config.xgb_max_depth,
            learning_rate=self.config.xgb_learning_rate,
            objective='multi:softmax',
            num_class=3,
            random_state=self.config.random_state,
            n_jobs=-1,
            verbosity=0
        )
    
    def _build_lightgbm(self) -> Optional[lgb.LGBMClassifier]:
        """Build LightGBM classifier."""
        if not LIGHTGBM_AVAILABLE:
            return None
        
        return lgb.LGBMClassifier(
            n_estimators=self.config.lgb_n_estimators,
            max_depth=self.config.lgb_max_depth,
            learning_rate=self.config.lgb_learning_rate,
            objective='multiclass',
            num_class=3,
            random_state=self.config.random_state,
            n_jobs=-1,
            verbose=-1
        )
    
    def _build_neural_network(self) -> MLPClassifier:
        """Build Neural Network classifier."""
        return MLPClassifier(
            hidden_layer_sizes=self.config.nn_hidden_layers,
            activation='relu',
            solver='adam',
            alpha=self.config.nn_alpha,
            max_iter=self.config.nn_max_iter,
            random_state=self.config.random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            verbose=False
        )
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target from raw data."""
        # Generate features
        features = self.feature_engineer.generate_features(df)
        
        # Create target
        target = self.feature_engineer.create_target(
            df, self.config.forward_periods, self.config.target_threshold
        )
        
        # Align and drop NaN
        combined = pd.concat([features, target.rename('target')], axis=1)
        combined = combined.dropna()
        
        X = combined.drop('target', axis=1)
        y = combined['target']
        
        # Map target to 0, 1, 2 for classifiers
        y = y.map({-1: 0, 0: 1, 1: 2})
        
        self.feature_names = list(X.columns)
        return X, y
    
    def train(self, df: pd.DataFrame, verbose: bool = True) -> Dict[str, float]:
        """Train all models on the provided data."""
        logger.info("Preparing training data...")
        X, y = self.prepare_data(df)
        
        if len(X) < 100:
            raise ValueError(f"Insufficient data: {len(X)} samples (need at least 100)")
        
        # Split data (time-series aware)
        split_idx = int(len(X) * (1 - self.config.test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        # Train XGBoost
        if XGBOOST_AVAILABLE:
            logger.info("Training XGBoost...")
            self.models['xgboost'] = self._build_xgboost()
            self.models['xgboost'].fit(X_train_scaled, y_train)
            xgb_pred = self.models['xgboost'].predict(X_test_scaled)
            results['xgboost_accuracy'] = accuracy_score(y_test, xgb_pred)
            if verbose:
                logger.info(f"XGBoost Accuracy: {results['xgboost_accuracy']:.4f}")
        
        # Train LightGBM
        if LIGHTGBM_AVAILABLE:
            logger.info("Training LightGBM...")
            self.models['lightgbm'] = self._build_lightgbm()
            self.models['lightgbm'].fit(X_train_scaled, y_train)
            lgb_pred = self.models['lightgbm'].predict(X_test_scaled)
            results['lightgbm_accuracy'] = accuracy_score(y_test, lgb_pred)
            if verbose:
                logger.info(f"LightGBM Accuracy: {results['lightgbm_accuracy']:.4f}")
        
        # Train Neural Network
        logger.info("Training Neural Network...")
        self.models['neural_net'] = self._build_neural_network()
        self.models['neural_net'].fit(X_train_scaled, y_train)
        nn_pred = self.models['neural_net'].predict(X_test_scaled)
        results['neural_net_accuracy'] = accuracy_score(y_test, nn_pred)
        if verbose:
            logger.info(f"Neural Network Accuracy: {results['neural_net_accuracy']:.4f}")
        
        # Build ensemble
        self._build_ensemble()
        if self.ensemble:
            logger.info("Training Ensemble...")
            self.ensemble.fit(X_train_scaled, y_train)
            ensemble_pred = self.ensemble.predict(X_test_scaled)
            results['ensemble_accuracy'] = accuracy_score(y_test, ensemble_pred)
            if verbose:
                logger.info(f"Ensemble Accuracy: {results['ensemble_accuracy']:.4f}")
                print("\nClassification Report (Ensemble):")
                print(classification_report(y_test, ensemble_pred, 
                      target_names=['DOWN', 'NEUTRAL', 'UP']))
        
        self.is_trained = True
        return results
    
    def _build_ensemble(self):
        """Build voting ensemble from trained models."""
        estimators = []
        
        if 'xgboost' in self.models:
            estimators.append(('xgb', self.models['xgboost']))
        if 'lightgbm' in self.models:
            estimators.append(('lgb', self.models['lightgbm']))
        if 'neural_net' in self.models:
            estimators.append(('nn', self.models['neural_net']))
        
        if len(estimators) >= 2:
            self.ensemble = VotingClassifier(
                estimators=estimators,
                voting='soft'
            )
    
    def predict(self, df: pd.DataFrame, use_ensemble: bool = True) -> pd.DataFrame:
        """Generate predictions for new data."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Generate features
        features = self.feature_engineer.generate_features(df)
        features = features.dropna()
        
        if len(features) == 0:
            return pd.DataFrame()
        
        # Scale
        X_scaled = self.scaler.transform(features[self.feature_names])
        
        predictions = pd.DataFrame(index=features.index)
        
        # Individual model predictions
        for name, model in self.models.items():
            pred = model.predict(X_scaled)
            proba = model.predict_proba(X_scaled)
            predictions[f'{name}_pred'] = pred
            predictions[f'{name}_conf'] = np.max(proba, axis=1)
        
        # Ensemble prediction
        if use_ensemble and self.ensemble:
            predictions['ensemble_pred'] = self.ensemble.predict(X_scaled)
            predictions['ensemble_conf'] = np.max(
                self.ensemble.predict_proba(X_scaled), axis=1
            )
        
        # Map predictions back to signals
        signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        pred_col = 'ensemble_pred' if use_ensemble and self.ensemble else 'neural_net_pred'
        predictions['signal'] = predictions[pred_col].map(signal_map)
        
        return predictions
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from tree-based models."""
        importance_data = []
        
        if 'xgboost' in self.models:
            xgb_imp = self.models['xgboost'].feature_importances_
            for name, imp in zip(self.feature_names, xgb_imp):
                importance_data.append({'feature': name, 'xgboost': imp})
        
        if 'lightgbm' in self.models:
            lgb_imp = self.models['lightgbm'].feature_importances_
            for i, imp in enumerate(lgb_imp):
                if i < len(importance_data):
                    importance_data[i]['lightgbm'] = imp
        
        df = pd.DataFrame(importance_data)
        if 'xgboost' in df.columns and 'lightgbm' in df.columns:
            df['avg_importance'] = (df['xgboost'] + df['lightgbm']) / 2
            df = df.sort_values('avg_importance', ascending=False)
        
        return df
    
    def save(self, filepath: Optional[str] = None):
        """Save trained models to disk."""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        filepath = filepath or os.path.join(self.config.model_dir, 'v39_ml_engine.pkl')
        
        state = {
            'models': self.models,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'config': self.config,
            'ensemble': self.ensemble
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: Optional[str] = None):
        """Load trained models from disk."""
        filepath = filepath or os.path.join(self.config.model_dir, 'v39_ml_engine.pkl')
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.models = state['models']
        self.scaler = state['scaler']
        self.feature_names = state['feature_names']
        self.config = state['config']
        self.ensemble = state.get('ensemble')
        self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")
    
    def cross_validate(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """Perform time-series cross-validation."""
        X, y = self.prepare_data(df)
        X_scaled = self.scaler.fit_transform(X)
        
        tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
        results = {}
        
        for name, model in [
            ('xgboost', self._build_xgboost()),
            ('lightgbm', self._build_lightgbm()),
            ('neural_net', self._build_neural_network())
        ]:
            if model is None:
                continue
            
            scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='accuracy')
            results[name] = scores.tolist()
            logger.info(f"{name}: CV Mean={np.mean(scores):.4f}, Std={np.std(scores):.4f}")
        
        return results


# =============================================================================
# DEMO & CLI
# =============================================================================

def generate_demo_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    
    dates = pd.date_range(end=datetime.now(), periods=n_samples, freq='D')
    
    # Random walk with trend
    returns = np.random.normal(0.0005, 0.02, n_samples)
    close = 100 * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': close * (1 + np.random.uniform(-0.01, 0.01, n_samples)),
        'high': close * (1 + np.random.uniform(0, 0.02, n_samples)),
        'low': close * (1 - np.random.uniform(0, 0.02, n_samples)),
        'close': close,
        'volume': np.random.uniform(1e6, 1e7, n_samples)
    })
    df.set_index('timestamp', inplace=True)
    
    return df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="V39 Deep ML Engine")
    parser.add_argument("--train", action="store_true", help="Train models on demo data")
    parser.add_argument("--predict", action="store_true", help="Run prediction on demo data")
    parser.add_argument("--cv", action="store_true", help="Run cross-validation")
    parser.add_argument("--importance", action="store_true", help="Show feature importance")
    parser.add_argument("--save", type=str, help="Save model to path")
    parser.add_argument("--load", type=str, help="Load model from path")
    
    args = parser.parse_args()
    
    engine = DeepMLEngine()
    
    if args.load:
        engine.load(args.load)
        logger.info("Model loaded successfully")
    
    if args.train or args.cv or (not any([args.train, args.predict, args.cv, args.importance])):
        logger.info("Generating demo data...")
        df = generate_demo_data(1000)
        
        if args.cv:
            logger.info("Running cross-validation...")
            results = engine.cross_validate(df)
            print("\nCross-Validation Results:")
            for model, scores in results.items():
                print(f"  {model}: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
        else:
            logger.info("Training models...")
            results = engine.train(df, verbose=True)
            print("\nTraining Results:")
            for metric, value in results.items():
                print(f"  {metric}: {value:.4f}")
    
    if args.predict:
        if not engine.is_trained:
            df = generate_demo_data(1000)
            engine.train(df, verbose=False)
        
        test_df = generate_demo_data(100)
        predictions = engine.predict(test_df)
        
        print("\nLatest Predictions:")
        print(predictions.tail(10)[['signal', 'ensemble_conf']].to_string())
    
    if args.importance:
        if not engine.is_trained:
            df = generate_demo_data(1000)
            engine.train(df, verbose=False)
        
        importance = engine.get_feature_importance()
        print("\nTop 15 Features:")
        print(importance.head(15).to_string())
    
    if args.save:
        engine.save(args.save)


if __name__ == "__main__":
    main()
