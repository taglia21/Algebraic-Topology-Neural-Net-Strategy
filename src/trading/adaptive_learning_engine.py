"""
Adaptive Learning Engine for TDA Trading Bot
=============================================

This module implements 4 key features for truly adaptive trading:

1. NEURAL NETWORK INTEGRATION
   - LSTM-based direction prediction for each stock
   - Uses TDA features + price data for predictions
   - Combines NN confidence with factor scores

2. REINFORCEMENT LEARNING FOR POSITION SIZING
   - Q-learning for optimal position sizing
   - Learns from realized P&L
   - Adjusts position sizes based on stock-specific performance

3. ONLINE LEARNING FOR FACTOR WEIGHTS
   - Adapts factor weights daily based on what's working
   - Exponential moving average of factor contributions to returns
   - Self-correcting model

4. RISK PARITY ALLOCATION
   - Allocates based on inverse volatility
   - Each position contributes equal risk
   - Dynamic rebalancing as volatilities change

Author: Renaissance-Style Trading Bot
"""

import os
import sys
import json
import logging
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import pandas as pd

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logger = logging.getLogger(__name__)

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class StockPrediction:
    """Neural network prediction for a stock."""
    ticker: str
    direction_prob: float  # Probability of going UP (0-1)
    confidence: float      # Model confidence (distance from 0.5)
    predicted_return: float  # Expected return
    
@dataclass
class PositionSizeDecision:
    """RL-based position sizing decision."""
    ticker: str
    base_weight: float     # From factor model
    rl_multiplier: float   # RL adjustment (0.5 to 2.0)
    final_weight: float    # base_weight * rl_multiplier
    q_value: float         # Expected future reward

@dataclass
class FactorPerformance:
    """Tracks performance of each factor."""
    momentum_contribution: float = 0.0
    vol_adjusted_contribution: float = 0.0
    relative_strength_contribution: float = 0.0
    liquidity_contribution: float = 0.0
    nn_contribution: float = 0.0


# =============================================================================
# 1. NEURAL NETWORK INTEGRATION
# =============================================================================

class NeuralNetworkPredictor:
    """
    LSTM-based stock direction predictor.
    
    Uses the existing nn_predictor.py infrastructure but adapted for
    real-time stock-level predictions.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the NN predictor."""
        self.model = None
        self.preprocessor = None
        self.model_path = model_path or "models/adaptive_nn.weights.h5"
        self.is_trained = False
        self.feature_cache: Dict[str, np.ndarray] = {}
        
        # Try to import TensorFlow (may not be available in production)
        try:
            import tensorflow as tf
            from tensorflow import keras
            self.tf = tf
            self.keras = keras
            self.tf_available = True
        except ImportError:
            logger.warning("TensorFlow not available - NN predictions disabled")
            self.tf_available = False
    
    def _build_model(self, n_features: int = 10, sequence_length: int = 20):
        """Build LSTM model architecture."""
        if not self.tf_available:
            return None
        
        model = self.keras.Sequential([
            self.keras.layers.LSTM(64, return_sequences=True, 
                                   input_shape=(sequence_length, n_features)),
            self.keras.layers.Dropout(0.2),
            self.keras.layers.LSTM(32, return_sequences=False),
            self.keras.layers.Dropout(0.2),
            self.keras.layers.Dense(16, activation='relu'),
            self.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=self.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_features(self, price_data: pd.DataFrame, 
                         sequence_length: int = 20) -> Optional[np.ndarray]:
        """
        Prepare feature sequences for a single stock.
        
        Features:
        1. Log returns
        2. High-Low range (normalized)
        3. Volume change
        4. RSI (14-period)
        5. MACD signal
        6. Bollinger Band position
        7-10. TDA-inspired features (momentum at different scales)
        """
        if len(price_data) < sequence_length + 50:
            return None
        
        try:
            close = price_data['Close'].values.astype(float)
            high = price_data['High'].values.astype(float)
            low = price_data['Low'].values.astype(float)
            volume = price_data['Volume'].values.astype(float)
            
            # 1. Log returns
            log_returns = np.diff(np.log(close + 1e-10))
            log_returns = np.concatenate([[0], log_returns])
            
            # 2. High-Low range normalized
            hl_range = (high - low) / (close + 1e-10)
            
            # 3. Volume change
            vol_ma = pd.Series(volume).rolling(20).mean().values
            vol_change = volume / (vol_ma + 1e-10) - 1
            vol_change = np.nan_to_num(vol_change, 0)
            
            # 4. RSI (14-period)
            rsi = self._compute_rsi(close, 14)
            rsi_norm = (rsi - 50) / 50  # Normalize to [-1, 1]
            
            # 5. MACD signal
            macd = self._compute_macd(close)
            macd_norm = macd / (np.std(macd) + 1e-10)  # Z-score
            
            # 6. Bollinger Band position
            bb_pos = self._compute_bb_position(close, 20)
            
            # 7-10. Multi-scale momentum (TDA-inspired)
            mom_5 = pd.Series(close).pct_change(5).values
            mom_10 = pd.Series(close).pct_change(10).values
            mom_20 = pd.Series(close).pct_change(20).values
            mom_50 = pd.Series(close).pct_change(50).values
            
            # Stack features
            features = np.column_stack([
                log_returns,
                hl_range,
                np.nan_to_num(vol_change, 0),
                np.nan_to_num(rsi_norm, 0),
                np.nan_to_num(macd_norm, 0),
                np.nan_to_num(bb_pos, 0),
                np.nan_to_num(mom_5, 0),
                np.nan_to_num(mom_10, 0),
                np.nan_to_num(mom_20, 0),
                np.nan_to_num(mom_50, 0),
            ])
            
            # Z-score normalize
            means = np.nanmean(features, axis=0)
            stds = np.nanstd(features, axis=0) + 1e-10
            features = (features - means) / stds
            
            # Take last sequence_length rows
            if len(features) >= sequence_length:
                return features[-sequence_length:].reshape(1, sequence_length, 10)
            
        except Exception as e:
            logger.warning(f"Feature preparation failed: {e}")
        
        return None
    
    def _compute_rsi(self, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Compute RSI."""
        delta = np.diff(close)
        delta = np.concatenate([[0], delta])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = pd.Series(gain).rolling(period).mean().values
        avg_loss = pd.Series(loss).rolling(period).mean().values
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return np.nan_to_num(rsi, 50)
    
    def _compute_macd(self, close: np.ndarray) -> np.ndarray:
        """Compute MACD line."""
        ema12 = pd.Series(close).ewm(span=12).mean().values
        ema26 = pd.Series(close).ewm(span=26).mean().values
        return ema12 - ema26
    
    def _compute_bb_position(self, close: np.ndarray, period: int = 20) -> np.ndarray:
        """Compute position within Bollinger Bands (-1 to 1)."""
        sma = pd.Series(close).rolling(period).mean().values
        std = pd.Series(close).rolling(period).std().values
        
        upper = sma + 2 * std
        lower = sma - 2 * std
        
        # Position: -1 at lower, 0 at middle, 1 at upper
        position = (close - sma) / (2 * std + 1e-10)
        return np.clip(position, -1, 1)
    
    def predict(self, price_data: Dict[str, pd.DataFrame]) -> List[StockPrediction]:
        """
        Generate predictions for all stocks.
        
        Returns list of StockPrediction ordered by confidence.
        """
        predictions = []
        
        for ticker, df in price_data.items():
            features = self.prepare_features(df)
            
            if features is None:
                continue
            
            if self.tf_available and self.model is not None and self.is_trained:
                # Use neural network prediction
                try:
                    prob = float(self.model.predict(features, verbose=0)[0, 0])
                except:
                    prob = 0.5
            else:
                # Fallback: use simple momentum-based prediction
                close = df['Close'].values
                if len(close) >= 20:
                    mom = close[-1] / close[-20] - 1
                    # Convert momentum to probability
                    prob = 1 / (1 + np.exp(-mom * 10))  # Sigmoid of momentum
                else:
                    prob = 0.5
            
            # Confidence is distance from 0.5
            confidence = abs(prob - 0.5) * 2  # 0 to 1
            
            # Expected return: bullish if prob > 0.5, bearish otherwise
            # Scale by confidence
            expected_return = (prob - 0.5) * 0.02 * confidence  # Max ~1% expected
            
            predictions.append(StockPrediction(
                ticker=ticker,
                direction_prob=prob,
                confidence=confidence,
                predicted_return=expected_return
            ))
        
        # Sort by confidence (most confident first)
        predictions.sort(key=lambda x: x.confidence, reverse=True)
        
        return predictions
    
    def train_on_history(self, price_data: Dict[str, pd.DataFrame], 
                         epochs: int = 5, min_samples: int = 500,
                         max_stocks: int = 200, samples_per_stock: int = 50):
        """
        Train the neural network on historical data (OPTIMIZED).
        
        Uses sampling to limit training time:
        - max_stocks: Maximum number of stocks to train on
        - samples_per_stock: Maximum samples per stock
        
        Uses the last day's return as labels.
        """
        if not self.tf_available:
            logger.warning("TensorFlow not available - cannot train")
            return
        
        X_all = []
        y_all = []
        
        # OPTIMIZATION: Sample a subset of stocks for faster training
        tickers = list(price_data.keys())
        if len(tickers) > max_stocks:
            # Randomly sample stocks for diversity
            import random
            tickers = random.sample(tickers, max_stocks)
            logger.info(f"Training on random sample of {max_stocks} stocks (from {len(price_data)})")
        
        for ticker in tickers:
            df = price_data[ticker]
            if len(df) < 100:
                continue
            
            close = df['Close'].values
            
            # OPTIMIZATION: Sample positions instead of all historical data
            valid_positions = list(range(50, len(df) - 1))
            if len(valid_positions) > samples_per_stock:
                import random
                valid_positions = random.sample(valid_positions, samples_per_stock)
            
            # Create training samples from sampled positions
            for i in valid_positions:
                subset = df.iloc[:i+1]
                features = self.prepare_features(subset)
                
                if features is not None:
                    X_all.append(features[0])
                    # Label: 1 if next day is up, 0 if down
                    label = 1 if close[i+1] > close[i] else 0
                    y_all.append(label)
        
        if len(X_all) < min_samples:
            logger.warning(f"Insufficient training data: {len(X_all)} samples")
            return
        
        X = np.array(X_all)
        y = np.array(y_all)
        
        # Build and train model
        if self.model is None:
            self.model = self._build_model(n_features=10)
        
        if self.model is not None:
            logger.info(f"Training NN on {len(X)} samples...")
            self.model.fit(X, y, epochs=epochs, batch_size=32, 
                          validation_split=0.2, verbose=0)
            self.is_trained = True
            logger.info("NN training complete")
            
            # Save model
            try:
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                self.model.save_weights(self.model_path)
            except Exception as e:
                logger.warning(f"Could not save model: {e}")


# =============================================================================
# 2. REINFORCEMENT LEARNING FOR POSITION SIZING
# =============================================================================

class PositionSizingRL:
    """
    Q-Learning based position sizing.
    
    States: Discretized (momentum, volatility, past_performance)
    Actions: Position size multiplier (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
    Rewards: Realized P&L from the position
    """
    
    ACTIONS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]  # Position multipliers
    
    def __init__(self, learning_rate: float = 0.1, discount: float = 0.95,
                 epsilon: float = 0.1, state_file: str = "models/rl_q_table.pkl"):
        """Initialize RL agent."""
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon  # Exploration rate
        self.state_file = state_file
        
        # Q-table: state -> action -> value
        self.q_table: Dict[str, Dict[float, float]] = {}
        
        # Track positions for learning
        self.active_positions: Dict[str, Tuple[str, float, float]] = {}  # ticker -> (state, action, entry_price)
        
        # Performance tracking per stock
        self.stock_performance: Dict[str, List[float]] = {}  # ticker -> list of returns
        
        self._load_state()
    
    def _discretize_state(self, ticker: str, momentum: float, volatility: float) -> str:
        """Convert continuous features to discrete state."""
        # Discretize momentum: very_negative, negative, neutral, positive, very_positive
        if momentum < -0.1:
            mom_state = "vn"
        elif momentum < -0.02:
            mom_state = "n"
        elif momentum < 0.02:
            mom_state = "z"
        elif momentum < 0.1:
            mom_state = "p"
        else:
            mom_state = "vp"
        
        # Discretize volatility: low, medium, high
        if volatility < 0.02:
            vol_state = "l"
        elif volatility < 0.04:
            vol_state = "m"
        else:
            vol_state = "h"
        
        # Past performance for this stock
        past_perf = self.stock_performance.get(ticker, [])
        if len(past_perf) >= 3:
            avg_perf = np.mean(past_perf[-5:])
            if avg_perf > 0.02:
                perf_state = "w"  # Winner
            elif avg_perf < -0.02:
                perf_state = "l"  # Loser
            else:
                perf_state = "n"  # Neutral
        else:
            perf_state = "u"  # Unknown
        
        return f"{ticker[:4]}_{mom_state}_{vol_state}_{perf_state}"
    
    def get_position_multiplier(self, ticker: str, momentum: float, 
                                volatility: float, base_weight: float) -> PositionSizeDecision:
        """
        Get RL-based position size multiplier.
        
        Uses epsilon-greedy exploration.
        """
        state = self._discretize_state(ticker, momentum, volatility)
        
        # Initialize Q-values for new state
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.ACTIONS}
        
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.ACTIONS)
        else:
            # Choose best action
            action = max(self.q_table[state], key=self.q_table[state].get)
        
        q_value = self.q_table[state][action]
        
        return PositionSizeDecision(
            ticker=ticker,
            base_weight=base_weight,
            rl_multiplier=action,
            final_weight=base_weight * action,
            q_value=q_value
        )
    
    def record_entry(self, ticker: str, state: str, action: float, entry_price: float):
        """Record position entry for later learning."""
        self.active_positions[ticker] = (state, action, entry_price)
    
    def learn_from_exit(self, ticker: str, exit_price: float):
        """Update Q-value based on realized return."""
        if ticker not in self.active_positions:
            return
        
        state, action, entry_price = self.active_positions[ticker]
        
        # Compute reward (return)
        reward = (exit_price - entry_price) / entry_price
        
        # Update stock performance history
        if ticker not in self.stock_performance:
            self.stock_performance[ticker] = []
        self.stock_performance[ticker].append(reward)
        if len(self.stock_performance[ticker]) > 20:
            self.stock_performance[ticker] = self.stock_performance[ticker][-20:]
        
        # Q-learning update
        if state in self.q_table and action in self.q_table[state]:
            old_q = self.q_table[state][action]
            # Simple Q-update (no next state in this formulation)
            new_q = old_q + self.lr * (reward - old_q)
            self.q_table[state][action] = new_q
        
        del self.active_positions[ticker]
        self._save_state()
    
    def _save_state(self):
        """Save Q-table to disk."""
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            with open(self.state_file, 'wb') as f:
                pickle.dump({
                    'q_table': self.q_table,
                    'stock_performance': self.stock_performance
                }, f)
        except Exception as e:
            logger.warning(f"Could not save RL state: {e}")
    
    def _load_state(self):
        """Load Q-table from disk."""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'rb') as f:
                    data = pickle.load(f)
                    self.q_table = data.get('q_table', {})
                    self.stock_performance = data.get('stock_performance', {})
                logger.info(f"Loaded RL state with {len(self.q_table)} states")
        except Exception as e:
            logger.warning(f"Could not load RL state: {e}")


# =============================================================================
# 3. ONLINE LEARNING FOR FACTOR WEIGHTS
# =============================================================================

class OnlineFactorLearner:
    """
    Adaptive factor weight learning.
    
    Tracks which factors are contributing to returns and adjusts
    weights using exponential moving average.
    """
    
    def __init__(self, initial_weights: Optional[Dict[str, float]] = None,
                 learning_rate: float = 0.05, 
                 state_file: str = "models/factor_weights.json"):
        """Initialize with default or custom weights."""
        self.state_file = state_file
        self.lr = learning_rate
        
        # Default weights
        self.weights = initial_weights or {
            'momentum': 0.30,
            'vol_adjusted': 0.20,
            'relative_strength': 0.15,
            'liquidity': 0.10,
            'nn_prediction': 0.25,  # NEW: neural network factor
        }
        
        # Track factor contributions
        self.factor_returns: Dict[str, List[float]] = {k: [] for k in self.weights}
        
        # Load saved state
        self._load_state()
    
    def get_weights(self) -> Dict[str, float]:
        """Get current factor weights."""
        # Ensure weights sum to 1
        total = sum(self.weights.values())
        return {k: v/total for k, v in self.weights.items()}
    
    def update_weights(self, realized_returns: Dict[str, float], 
                       factor_scores: Dict[str, Dict[str, float]]):
        """
        Update factor weights based on realized returns.
        
        Args:
            realized_returns: Dict of ticker -> realized return
            factor_scores: Dict of ticker -> {factor: score}
        """
        # Compute correlation of each factor with returns
        factor_contributions = {k: 0.0 for k in self.weights}
        count = 0
        
        for ticker, ret in realized_returns.items():
            if ticker not in factor_scores:
                continue
            
            scores = factor_scores[ticker]
            for factor in self.weights:
                if factor in scores:
                    # Contribution = factor_score * return (positive if aligned)
                    contribution = scores[factor] * ret
                    factor_contributions[factor] += contribution
            count += 1
        
        if count == 0:
            return
        
        # Average contributions
        for factor in factor_contributions:
            factor_contributions[factor] /= count
        
        # Update weights using EMA
        for factor in self.weights:
            contribution = factor_contributions.get(factor, 0)
            
            # Track history
            self.factor_returns[factor].append(contribution)
            if len(self.factor_returns[factor]) > 50:
                self.factor_returns[factor] = self.factor_returns[factor][-50:]
            
            # If factor contributed positively, increase weight
            # If negatively, decrease weight
            adjustment = self.lr * contribution
            self.weights[factor] = max(0.05, self.weights[factor] + adjustment)
        
        # Normalize to sum to 1
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
        
        self._save_state()
        logger.info(f"Updated factor weights: {self.weights}")
    
    def get_factor_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics about factor performance."""
        stats = {}
        for factor, returns in self.factor_returns.items():
            if returns:
                stats[factor] = {
                    'weight': self.weights[factor],
                    'avg_contribution': np.mean(returns),
                    'recent_contribution': np.mean(returns[-10:]) if len(returns) >= 10 else np.mean(returns),
                    'volatility': np.std(returns),
                }
            else:
                stats[factor] = {
                    'weight': self.weights[factor],
                    'avg_contribution': 0.0,
                    'recent_contribution': 0.0,
                    'volatility': 0.0,
                }
        return stats
    
    def _save_state(self):
        """Save weights to disk."""
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump({
                    'weights': self.weights,
                    'factor_returns': self.factor_returns,
                    'updated_at': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save factor weights: {e}")
    
    def _load_state(self):
        """Load weights from disk."""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.weights = data.get('weights', self.weights)
                    self.factor_returns = data.get('factor_returns', self.factor_returns)
                logger.info(f"Loaded factor weights: {self.weights}")
        except Exception as e:
            logger.warning(f"Could not load factor weights: {e}")


# =============================================================================
# 4. RISK PARITY ALLOCATION
# =============================================================================

class RiskParityAllocator:
    """
    Risk Parity portfolio allocation.
    
    Each position contributes equal risk (volatility) to the portfolio.
    """
    
    def __init__(self, lookback: int = 60, min_weight: float = 0.005, 
                 max_weight: float = 0.05):
        """Initialize allocator."""
        self.lookback = lookback
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.volatilities: Dict[str, float] = {}
    
    def compute_volatilities(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Compute annualized volatility for each stock."""
        volatilities = {}
        
        for ticker, df in price_data.items():
            if len(df) < self.lookback:
                continue
            
            try:
                close = df['Close'].iloc[-self.lookback:]
                returns = close.pct_change().dropna()
                vol = returns.std() * np.sqrt(252)  # Annualized
                volatilities[ticker] = max(0.01, vol)  # Floor at 1%
            except:
                volatilities[ticker] = 0.25  # Default 25% vol
        
        self.volatilities = volatilities
        return volatilities
    
    def allocate(self, selected_stocks: List[str], 
                 total_allocation: float = 0.60) -> Dict[str, float]:
        """
        Compute risk parity weights.
        
        Each stock's weight is inversely proportional to its volatility.
        """
        if not selected_stocks:
            return {}
        
        # Get volatilities
        vols = {t: self.volatilities.get(t, 0.25) for t in selected_stocks}
        
        # Inverse volatility weights
        inv_vols = {t: 1/v for t, v in vols.items()}
        total_inv_vol = sum(inv_vols.values())
        
        # Risk parity weights
        weights = {}
        for ticker in selected_stocks:
            weight = (inv_vols[ticker] / total_inv_vol) * total_allocation
            # Apply min/max constraints
            weight = max(self.min_weight, min(self.max_weight, weight))
            weights[ticker] = weight
        
        # Normalize to total_allocation
        current_total = sum(weights.values())
        if current_total > 0:
            weights = {k: v * total_allocation / current_total for k, v in weights.items()}
        
        return weights
    
    def get_risk_contribution(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate risk contribution of each position.
        
        Risk contribution = weight * volatility
        """
        contributions = {}
        for ticker, weight in weights.items():
            vol = self.volatilities.get(ticker, 0.25)
            contributions[ticker] = weight * vol
        
        # Normalize to percentages
        total = sum(contributions.values())
        if total > 0:
            contributions = {k: v/total for k, v in contributions.items()}
        
        return contributions


# =============================================================================
# MASTER ADAPTIVE ENGINE
# =============================================================================

class AdaptiveLearningEngine:
    """
    Master engine that combines all 4 adaptive learning components.
    
    This replaces the static MultiFactorEngine with a learning system.
    """
    
    def __init__(self, models_dir: str = "models"):
        """Initialize all components."""
        os.makedirs(models_dir, exist_ok=True)
        
        # Initialize components
        self.nn_predictor = NeuralNetworkPredictor(
            model_path=os.path.join(models_dir, "adaptive_nn.weights.h5")
        )
        
        self.position_rl = PositionSizingRL(
            state_file=os.path.join(models_dir, "rl_q_table.pkl")
        )
        
        self.factor_learner = OnlineFactorLearner(
            state_file=os.path.join(models_dir, "factor_weights.json")
        )
        
        self.risk_parity = RiskParityAllocator()
        
        # Track previous positions for learning
        self.prev_positions: Dict[str, float] = {}  # ticker -> entry_price
        self.prev_factor_scores: Dict[str, Dict[str, float]] = {}
        
        logger.info("AdaptiveLearningEngine initialized")
        logger.info(f"  - NN Predictor: {'enabled' if self.nn_predictor.tf_available else 'disabled'}")
        logger.info(f"  - RL Position Sizing: {len(self.position_rl.q_table)} learned states")
        logger.info(f"  - Factor Weights: {self.factor_learner.get_weights()}")
    
    def train_models(self, price_data: Dict[str, pd.DataFrame]):
        """Train/update all learning models."""
        # Train NN on historical data (if not already trained)
        if not self.nn_predictor.is_trained:
            logger.info("Training neural network...")
            self.nn_predictor.train_on_history(price_data, epochs=10)
        
        # Compute volatilities for risk parity
        self.risk_parity.compute_volatilities(price_data)
    
    def learn_from_trades(self, current_prices: Dict[str, float],
                          current_positions: Dict[str, float]):
        """
        Learn from realized trades.
        
        Call this before computing new portfolio to update learning.
        """
        realized_returns = {}
        
        # Find closed positions
        for ticker, entry_price in list(self.prev_positions.items()):
            if ticker not in current_positions:
                # Position was closed
                exit_price = current_prices.get(ticker, entry_price)
                ret = (exit_price - entry_price) / entry_price
                realized_returns[ticker] = ret
                
                # Update RL
                self.position_rl.learn_from_exit(ticker, exit_price)
        
        # Update factor weights if we have returns and scores
        if realized_returns and self.prev_factor_scores:
            self.factor_learner.update_weights(realized_returns, self.prev_factor_scores)
        
        # Update position tracking
        self.prev_positions = {t: p for t, p in current_prices.items() 
                              if t in current_positions}
    
    def compute_adaptive_scores(
        self,
        price_data: Dict[str, pd.DataFrame],
        spy_data: pd.DataFrame,
        n_stocks: int = 100
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        """
        Compute adaptive stock scores combining all components.
        
        Returns:
            Tuple of (target_weights, factor_scores_for_learning)
        """
        # 1. Get NN predictions
        nn_predictions = self.nn_predictor.predict(price_data)
        nn_scores = {p.ticker: p.direction_prob for p in nn_predictions}
        nn_confidence = {p.ticker: p.confidence for p in nn_predictions}
        
        # 2. Compute traditional factors
        traditional_scores = self._compute_traditional_factors(price_data, spy_data)
        
        # 3. Get current factor weights (adaptive)
        weights = self.factor_learner.get_weights()
        
        # 4. Combine scores
        combined_scores = {}
        factor_scores_for_learning = {}
        
        for ticker in set(traditional_scores.keys()) & set(nn_scores.keys()):
            trad = traditional_scores[ticker]
            
            # Normalize NN prediction to [-1, 1] range (like other factors)
            nn_factor = (nn_scores[ticker] - 0.5) * 2
            
            # Weighted combination
            score = (
                weights.get('momentum', 0.3) * trad['momentum'] +
                weights.get('vol_adjusted', 0.2) * trad['vol_adjusted'] +
                weights.get('relative_strength', 0.15) * trad['relative_strength'] +
                weights.get('liquidity', 0.1) * trad['liquidity'] +
                weights.get('nn_prediction', 0.25) * nn_factor
            )
            
            # Boost by NN confidence
            confidence_boost = 1 + (nn_confidence.get(ticker, 0) * 0.5)
            score *= confidence_boost
            
            combined_scores[ticker] = score
            factor_scores_for_learning[ticker] = {
                'momentum': trad['momentum'],
                'vol_adjusted': trad['vol_adjusted'],
                'relative_strength': trad['relative_strength'],
                'liquidity': trad['liquidity'],
                'nn_prediction': nn_factor
            }
        
        # Save for learning
        self.prev_factor_scores = factor_scores_for_learning
        
        # 5. Select top stocks
        sorted_stocks = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        selected = [t for t, s in sorted_stocks[:n_stocks * 2] if s > 0][:n_stocks]
        
        if not selected:
            return {}, {}
        
        # 6. Apply risk parity allocation
        self.risk_parity.compute_volatilities(price_data)
        base_weights = self.risk_parity.allocate(selected, total_allocation=0.58)
        
        # 7. Apply RL position sizing adjustments
        final_weights = {}
        for ticker, base_weight in base_weights.items():
            # Get momentum and volatility for state
            trad = traditional_scores.get(ticker, {})
            momentum = trad.get('momentum', 0)
            volatility = self.risk_parity.volatilities.get(ticker, 0.25)
            
            # Get RL decision
            decision = self.position_rl.get_position_multiplier(
                ticker, momentum, volatility, base_weight
            )
            
            final_weights[ticker] = decision.final_weight
            
            # Record for learning
            state = self.position_rl._discretize_state(ticker, momentum, volatility)
            current_price = price_data[ticker]['Close'].iloc[-1] if ticker in price_data else 0
            self.position_rl.record_entry(ticker, state, decision.rl_multiplier, current_price)
        
        # Normalize weights
        total = sum(final_weights.values())
        if total > 0.58:
            final_weights = {k: v * 0.58 / total for k, v in final_weights.items()}
        
        return final_weights, factor_scores_for_learning
    
    def _compute_traditional_factors(
        self, 
        price_data: Dict[str, pd.DataFrame],
        spy_data: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """Compute traditional factor scores (z-scored)."""
        raw_scores = {}
        
        for ticker, df in price_data.items():
            if len(df) < 252:
                continue
            
            try:
                close = df['Close']
                
                # Momentum (12m skip 1m)
                if len(close) >= 252:
                    momentum = close.iloc[-21] / close.iloc[-252] - 1
                else:
                    momentum = close.iloc[-21] / close.iloc[0] - 1
                
                # Vol-adjusted return
                returns = close.pct_change().dropna()
                vol = returns.iloc[-60:].std() * np.sqrt(252)
                annual_ret = close.iloc[-1] / close.iloc[-252] - 1 if len(close) >= 252 else momentum
                vol_adj = annual_ret / vol if vol > 0 else 0
                
                # Relative strength
                spy_close = spy_data['Close']
                if len(spy_close) >= 252:
                    stock_ret = close.iloc[-1] / close.iloc[-252] - 1
                    spy_ret = spy_close.iloc[-1] / spy_close.iloc[-252] - 1
                    rel_strength = stock_ret - spy_ret
                else:
                    rel_strength = 0
                
                # Liquidity
                if 'Volume' in df.columns:
                    dollar_vol = (df['Close'] * df['Volume']).iloc[-20:].mean()
                    liquidity = np.log10(max(dollar_vol, 1))
                else:
                    liquidity = 8
                
                raw_scores[ticker] = {
                    'momentum': momentum,
                    'vol_adjusted': vol_adj,
                    'relative_strength': rel_strength,
                    'liquidity': liquidity
                }
            except:
                continue
        
        if not raw_scores:
            return {}
        
        # Z-score normalize
        df = pd.DataFrame(raw_scores).T
        for col in df.columns:
            if df[col].std() > 0:
                df[col] = (df[col] - df[col].mean()) / df[col].std()
            else:
                df[col] = 0
        
        return df.to_dict('index')
    
    def get_status(self) -> Dict:
        """Get current status of all learning components."""
        return {
            'nn_trained': self.nn_predictor.is_trained,
            'nn_available': self.nn_predictor.tf_available,
            'rl_states': len(self.position_rl.q_table),
            'rl_epsilon': self.position_rl.epsilon,
            'factor_weights': self.factor_learner.get_weights(),
            'factor_stats': self.factor_learner.get_factor_stats(),
            'tracked_positions': len(self.prev_positions),
        }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create engine
    engine = AdaptiveLearningEngine(models_dir="test_models")
    
    print("AdaptiveLearningEngine Status:")
    status = engine.get_status()
    for k, v in status.items():
        print(f"  {k}: {v}")
    
    print("\n✅ Adaptive Learning Engine initialized successfully!")
