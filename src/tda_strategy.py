#!/usr/bin/env python3
"""
TDA Strategy Module
===================
Wapper for TDA (Topological Data Analysis) based market analysis.
Integrates with neural networks for signal generation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
import torch
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TDAResult:
    """Result from TDA analysis."""
    persistence_score: float
    betti_0: int  # Connected components
    betti_1: int  # Loops/cycles
    regime: str
    trend_strength: float
    volatility_regime: str


class TDAStrategy:
    """
    TDA-based trading strategy that uses topological features
    to identify market regimes and generate signals.
    """
    
    def __init__(self, lookback_window: int = 60, prediction_horizon: int = 5):
        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.device = torch.device('cpu')
        self._load_model()
        logger.info(f"TDA Strategy initialized (lookback={lookback_window}, horizon={prediction_horizon})")
    
    def _load_model(self):
        """Load trained neural network model if available."""
        model_path = 'models/tda_nn_model.pt'
        if os.path.exists(model_path):
            try:
                from train_model import TDANeuralNet
                self.model = TDANeuralNet(input_size=10, hidden_size=64, output_size=1)
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
                logger.info("Loaded trained TDA-NN model")
            except Exception as e:
                logger.warning(f"Could not load model: {e}")
                self.model = None
        else:
            logger.info("No trained model found - using rule-based signals")
    
    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run TDA analysis on price data."""
        if len(df) < self.lookback_window:
            logger.warning(f"Insufficient data: {len(df)} < {self.lookback_window}")
            return self._default_result()
        
        # Get recent data
        close = df['Close'].values[-self.lookback_window:]
        
        # Calculate returns and volatility
        returns = np.diff(close) / close[:-1]
        volatility = np.std(returns) * np.sqrt(252)
        
        # Simple TDA-inspired metrics (simplified from full persistent homology)
        # Persistence score based on trend consistency
        trend = (close[-1] - close[0]) / close[0]
        trend_consistency = self._calculate_trend_consistency(close)
        persistence_score = 0.5 + trend_consistency * 0.3 + np.clip(trend, -0.2, 0.2)
        
        # Betti numbers approximation
        # Betti-0: Number of distinct price levels (clusters)
        price_range = close.max() - close.min()
        n_levels = int(np.ceil(price_range / (close.std() * 0.5))) if close.std() > 0 else 1
        betti_0 = max(1, min(n_levels, 5))
        
        # Betti-1: Number of cycles (reversals)
        reversals = np.sum(np.diff(np.sign(returns)) != 0)
        betti_1 = min(reversals // 3, 5)  # Scale down
        
        # Determine regime
        if trend > 0.05 and trend_consistency > 0.3:
            regime = 'bullish'
        elif trend < -0.05 and trend_consistency > 0.3:
            regime = 'bearish'
        else:
            regime = 'neutral'
        
        # Volatility regime
        if volatility > 0.3:
            vol_regime = 'high'
        elif volatility < 0.15:
            vol_regime = 'low'
        else:
            vol_regime = 'normal'
        
        return {
            'persistence_score': float(np.clip(persistence_score, 0, 1)),
            'betti_0': betti_0,
            'betti_1': betti_1,
            'regime': regime,
            'trend_strength': float(abs(trend)),
            'volatility_regime': vol_regime,
            'volatility': float(volatility),
            'recent_return': float(trend)
        }

    
    def _calculate_trend_consistency(self, prices: np.ndarray) -> float:
        """Calculate how consistent the trend is."""
        if len(prices) < 2:
            return 0.0
        
        returns = np.diff(prices) / prices[:-1]
        positive_days = np.sum(returns > 0)
        negative_days = np.sum(returns < 0)
        total = len(returns)
        
        if total == 0:
            return 0.0
        
        # Consistency is deviation from 50/50
        consistency = abs(positive_days - negative_days) / total
        return float(consistency)
    
    def _default_result(self) -> Dict[str, Any]:
        """Return default TDA result when analysis fails."""
        return {
            'persistence_score': 0.5,
            'betti_0': 1,
            'betti_1': 0,
            'regime': 'neutral',
            'trend_strength': 0.0,
            'volatility_regime': 'normal',
            'volatility': 0.2,
            'recent_return': 0.0
        }
    
    def predict(self, df: pd.DataFrame) -> float:
        """Generate prediction using NN model or rule-based fallback."""
        if self.model is not None:
            try:
                features = self._extract_features(df)
                with torch.no_grad():
                    tensor = torch.FloatTensor(features).unsqueeze(0)
                    prediction = self.model(tensor).item()
                return float(np.clip(prediction, -1, 1))
            except Exception as e:
                logger.warning(f"Model prediction failed: {e}")
        
        # Rule-based fallback
        analysis = self.analyze(df)
        regime = analysis.get('regime', 'neutral')
        trend_strength = analysis.get('trend_strength', 0.0)
        
        if regime == 'bullish':
            return min(0.7, 0.3 + trend_strength)
        elif regime == 'bearish':
            return max(-0.7, -0.3 - trend_strength)
        else:
            return 0.0
    
    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features for neural network."""
        if len(df) < self.lookback_window:
            return np.zeros(10)
        
        close = df['Close'].values[-self.lookback_window:]
        returns = np.diff(close) / close[:-1]
        
        features = [
            np.mean(returns),           # Mean return
            np.std(returns),            # Volatility
            np.percentile(returns, 25), # 25th percentile
            np.percentile(returns, 75), # 75th percentile
            (close[-1] - close[0]) / close[0],  # Period return
            np.max(close) / np.min(close) - 1,  # Range
            np.corrcoef(np.arange(len(close)), close)[0,1] if len(close) > 1 else 0,  # Trend
            np.sum(returns > 0) / len(returns),  # Up ratio
            (close[-1] - np.mean(close)) / np.std(close) if np.std(close) > 0 else 0,  # Z-score
            np.mean(np.abs(returns))  # Mean abs return
        ]
        
        return np.array(features, dtype=np.float32)


def demo_tda_strategy():
    """Demo the TDA strategy."""
    import yfinance as yf
    
    print("=" * 60)
    print("TDA STRATEGY - DEMO")
    print("=" * 60)
    
    strategy = TDAStrategy(lookback_window=60)
    
    # Test with SPY
    print("\n[Fetching SPY data...]")
    spy = yf.Ticker('SPY')
    df = spy.history(period='3mo', interval='1d')
    
    if not df.empty:
        print(f"Data points: {len(df)}")
        
        # Run analysis
        print("\n[TDA Analysis]")
        analysis = strategy.analyze(df)
        for key, value in analysis.items():
            print(f"  {key}: {value}")
        
        # Get prediction
        print("\n[NN Prediction]")
        prediction = strategy.predict(df)
        print(f"  Signal: {prediction:.4f}")
        if prediction > 0.2:
            print("  Direction: BULLISH")
        elif prediction < -0.2:
            print("  Direction: BEARISH")
        else:
            print("  Direction: NEUTRAL")
    
    print("\n" + "=" * 60)
    print("TDA STRATEGY READY")
    print("=" * 60)
    
    return strategy


if __name__ == "__main__":
    demo_tda_strategy()

