#!/usr/bin/env python3
"""
Adaptive Learning Engine - Continuous Self-Improvement System
============================================================
A comprehensive ML system that enables the trading bot to:
1. Learn incrementally from every trade (Online Learning)
2. Track performance and adjust strategies (Feedback Loop)
3. Detect model drift and auto-retrain (Drift Detection)
4. Evolve strategies using genetic algorithms (Strategy Evolution)
5. Use reinforcement learning for optimal decisions (RL Rewards)

Author: Team of Rivals Trading System
Version: 2.0
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import deque
import threading
import time
import logging
from pathlib import Path

# ML Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TradeRecord:
    """Record of a completed trade for learning"""
    trade_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    entry_price: float
    exit_price: float
    quantity: int
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_percent: float
    hold_duration_minutes: int
    signal_strength: float
    predicted_direction: int  # -1, 0, 1
    actual_direction: int  # -1, 0, 1
    features: Dict[str, float] = field(default_factory=dict)
    market_regime: str = 'unknown'
    was_profitable: bool = False
    
    def __post_init__(self):
        self.was_profitable = self.pnl > 0
        if self.exit_price > self.entry_price:
            self.actual_direction = 1
        elif self.exit_price < self.entry_price:
            self.actual_direction = -1
        else:
            self.actual_direction = 0

@dataclass
class PerformanceMetrics:
    """Real-time performance tracking"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    prediction_accuracy: float = 0.0
    model_confidence: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass  
class ModelState:
    """Tracks model health and drift"""
    accuracy_history: List[float] = field(default_factory=list)
    loss_history: List[float] = field(default_factory=list)
    feature_means: Dict[str, float] = field(default_factory=dict)
    feature_stds: Dict[str, float] = field(default_factory=dict)
    last_retrain: datetime = field(default_factory=datetime.now)
    retrain_count: int = 0
    drift_detected: bool = False
    drift_score: float = 0.0


# ============================================================================
# ONLINE LEARNING MODEL - Learns from every trade incrementally
# ============================================================================

class OnlineLearningModel:
    """
    Incremental learning model that updates with each new trade.
    Uses Passive-Aggressive algorithm for online classification.
    """
    
    def __init__(self, n_features: int = 20):
        self.n_features = n_features
        self.scaler = StandardScaler()
        self.scaler_fitted = False
        
        # Primary online classifier (Passive-Aggressive)
        self.classifier = PassiveAggressiveClassifier(
            C=0.1,  # Regularization
            max_iter=1,
            warm_start=True,
            random_state=42
            ,
        )
        
        # Backup SGD classifier for comparison
        self.sgd_classifier = SGDClassifier(
            loss='log_loss',
            penalty='l2',
            alpha=0.0001,
            warm_start=True,
            random_state=42
            ,
        )
        
        # Experience buffer for mini-batch updates
        self.experience_buffer = deque(maxlen=1000)
        self.batch_size = 32
        
        # Track performance
        self.predictions_made = 0
        self.correct_predictions = 0
        self.recent_accuracy = deque(maxlen=100)
        
        # Classes: -1 (bearish), 0 (neutral), 1 (bullish)
        self.classes = np.array([-1, 0, 1])
        self.is_fitted = False
        
    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Incrementally update the model with new data.
        This is called after EVERY trade.
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        if isinstance(y, (int, float)):
            y = np.array([y])
            
        # Update scaler incrementally
        if not self.scaler_fitted:
            self.scaler.fit(X)
            self.scaler_fitted = True
        else:
            # Incremental scaling update
            self.scaler.partial_fit(X)
        
        X_scaled = self.scaler.transform(X)
        
        # Fit both classifiers
        self.classifier.partial_fit(X_scaled, y, classes=self.classes)
        self.sgd_classifier.partial_fit(X_scaled, y, classes=self.classes)
        self.is_fitted = True
        
        # Store in experience buffer
        for i in range(len(X)):
            self.experience_buffer.append((X[i], y[i]))
        
        return {'status': 'updated', 'buffer_size': len(self.experience_buffer)}
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make prediction and return confidence scores"""
        if not self.is_fitted:
            return np.array([0]), np.array([0.33, 0.34, 0.33])
            
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from both models
        pred1 = self.classifier.predict(X_scaled)
        
        # Get decision function for confidence
        decision = self.classifier.decision_function(X_scaled)
        
        # Convert decision to probability-like confidence
        confidence = np.abs(decision).max(axis=1) if len(decision.shape) > 1 else np.abs(decision)
        confidence = np.clip(confidence / 2, 0, 1)  # Normalize to 0-1
        
        return pred1, confidence
    
    def update_accuracy(self, predicted: int, actual: int):
        """Track prediction accuracy in real-time"""
        self.predictions_made += 1
        correct = 1 if predicted == actual else 0
        self.correct_predictions += correct
        self.recent_accuracy.append(correct)
        
    def get_recent_accuracy(self) -> float:
        """Get rolling accuracy over recent predictions"""
        if len(self.recent_accuracy) == 0:
            return 0.5
        return sum(self.recent_accuracy) / len(self.recent_accuracy)
    
    def experience_replay(self, n_samples: int = 100):
        """
        Replay past experiences to reinforce learning.
        Called periodically to prevent catastrophic forgetting.
        """
        if len(self.experience_buffer) < n_samples:
            return
            
        # Sample random experiences
        indices = np.random.choice(len(self.experience_buffer), n_samples, replace=False)
        X_batch = np.array([self.experience_buffer[i][0] for i in indices])
        y_batch = np.array([self.experience_buffer[i][1] for i in indices])
        
        # Retrain on batch
        X_scaled = self.scaler.transform(X_batch)
        self.classifier.partial_fit(X_scaled, y_batch)
        
        logger.info(f"Experience replay completed with {n_samples} samples")


# ============================================================================
# REINFORCEMENT LEARNING REWARD SYSTEM
# ============================================================================

class ReinforcementLearningReward:
    """
    Calculates rewards for trading actions to guide model improvement.
    Uses a sophisticated reward function that considers:
    - P&L (profit/loss)
    - Risk-adjusted returns
    - Holding period efficiency
    - Prediction accuracy
    """
    
    def __init__(self):
        self.gamma = 0.99  # Discount factor
        self.risk_penalty = 0.1  # Penalty for high-risk trades
        self.reward_history = deque(maxlen=1000)
        self.cumulative_reward = 0.0
        
    def calculate_reward(self, trade: TradeRecord) -> float:
        """
        Calculate reward for a completed trade.
        Positive reward for profitable trades, negative for losses.
        """
        # Base reward: P&L percentage
        pnl_reward = trade.pnl_percent * 100  # Scale up for learning
        
        # Prediction accuracy bonus
        if trade.predicted_direction == trade.actual_direction:
            accuracy_bonus = 0.5
        elif trade.predicted_direction * trade.actual_direction > 0:  # Same sign
            accuracy_bonus = 0.2
        else:
            accuracy_bonus = -0.3  # Wrong direction penalty
        
        # Holding period efficiency (penalize too long or too short)
        optimal_hold_minutes = 60  # 1 hour optimal
        hold_efficiency = 1 - abs(trade.hold_duration_minutes - optimal_hold_minutes) / 1000
        hold_efficiency = max(0, hold_efficiency)
        
        # Risk-adjusted component (penalize high volatility trades)
        volatility_penalty = -self.risk_penalty * trade.features.get('volatility', 0)
        
        # Signal strength reward (reward confident correct predictions)
        if trade.was_profitable:
            signal_reward = trade.signal_strength * 0.5
        else:
            signal_reward = -trade.signal_strength * 0.3  # Penalize confident wrong trades
        
        # Combine all components
        total_reward = (
            pnl_reward * 0.5 +           # 50% weight on P&L
            accuracy_bonus * 0.2 +        # 20% weight on prediction accuracy
            hold_efficiency * 0.1 +       # 10% weight on holding efficiency
            volatility_penalty * 0.1 +    # 10% weight on risk
            signal_reward * 0.1           # 10% weight on signal confidence
        )
        
        self.reward_history.append(total_reward)
        self.cumulative_reward += total_reward
        
        return total_reward
    
    def get_average_reward(self, n_recent: int = 100) -> float:
        """Get average reward over recent trades"""
        if len(self.reward_history) == 0:
            return 0.0
        recent = list(self.reward_history)[-n_recent:]
        return sum(recent) / len(recent)
    
    def get_reward_trend(self) -> str:
        """Determine if rewards are improving, declining, or stable"""
        if len(self.reward_history) < 20:
            return 'insufficient_data'
        
        recent_50 = list(self.reward_history)[-50:]
        first_half = sum(recent_50[:25]) / 25
        second_half = sum(recent_50[25:]) / 25
        
        diff = second_half - first_half
        if diff > 0.1:
            return 'improving'
        elif diff < -0.1:
            return 'declining'
        else:
            return 'stable'


# ============================================================================
# MODEL DRIFT DETECTION
# ============================================================================

class DriftDetector:
    """
    Detects when the model's performance degrades or when
    the underlying data distribution changes (concept drift).
    """
    
    def __init__(self, window_size: int = 100, drift_threshold: float = 0.15):
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        
        # Performance tracking
        self.accuracy_window = deque(maxlen=window_size)
        self.pnl_window = deque(maxlen=window_size)
        
        # Feature distribution tracking
        self.feature_history = deque(maxlen=window_size * 2)
        self.baseline_features = None
        
        # Drift detection state
        self.drift_detected = False
        self.drift_score = 0.0
        self.drift_type = None  # 'accuracy', 'distribution', or 'performance'
        self.last_check = datetime.now()
        
    def add_observation(self, features: np.ndarray, prediction_correct: bool, pnl: float):
        """Add a new observation for drift monitoring"""
        self.accuracy_window.append(1 if prediction_correct else 0)
        self.pnl_window.append(pnl)
        self.feature_history.append(features)
        
    def check_accuracy_drift(self) -> Tuple[bool, float]:
        """Check if prediction accuracy has dropped significantly"""
        if len(self.accuracy_window) < self.window_size:
            return False, 0.0
            
        recent = list(self.accuracy_window)
        first_half = sum(recent[:len(recent)//2]) / (len(recent)//2)
        second_half = sum(recent[len(recent)//2:]) / (len(recent) - len(recent)//2)
        
        drift_score = first_half - second_half  # Positive means degradation
        drift_detected = drift_score > self.drift_threshold
        
        return drift_detected, drift_score
    
    def check_distribution_drift(self) -> Tuple[bool, float]:
        """
        Check if feature distributions have shifted using
        Kolmogorov-Smirnov test.
        """
        if len(self.feature_history) < self.window_size * 2:
            return False, 0.0
            
        features = np.array(list(self.feature_history))
        n = len(features)
        
        # Compare first half vs second half
        first_half = features[:n//2]
        second_half = features[n//2:]
        
        # KS test for each feature
        max_drift = 0.0
        for i in range(min(features.shape[1], 10)):  # Check first 10 features
            try:
                statistic, _ = stats.ks_2samp(first_half[:, i], second_half[:, i])
                max_drift = max(max_drift, statistic)
            except:
                pass
        
        drift_detected = max_drift > 0.3  # KS statistic threshold
        return drift_detected, max_drift
    
    def check_performance_drift(self) -> Tuple[bool, float]:
        """Check if P&L performance has degraded"""
        if len(self.pnl_window) < self.window_size:
            return False, 0.0
            
        recent = list(self.pnl_window)
        first_half = sum(recent[:len(recent)//2])
        second_half = sum(recent[len(recent)//2:])
        
        # Significant drop in cumulative P&L
        drift_score = (first_half - second_half) / (abs(first_half) + 1)
        drift_detected = drift_score > 0.3  # 30% degradation
        
        return drift_detected, drift_score
    
    def detect_drift(self) -> Dict[str, Any]:
        """Run all drift detection checks"""
        acc_drift, acc_score = self.check_accuracy_drift()
        dist_drift, dist_score = self.check_distribution_drift()
        perf_drift, perf_score = self.check_performance_drift()
        
        self.drift_detected = acc_drift or dist_drift or perf_drift
        self.drift_score = max(acc_score, dist_score, perf_score)
        
        if acc_drift:
            self.drift_type = 'accuracy'
        elif dist_drift:
            self.drift_type = 'distribution'
        elif perf_drift:
            self.drift_type = 'performance'
        else:
            self.drift_type = None
            
        self.last_check = datetime.now()
        
        return {
            'drift_detected': self.drift_detected,
            'drift_type': self.drift_type,
            'drift_score': self.drift_score,
            'accuracy_drift': acc_drift,
            'distribution_drift': dist_drift,
            'performance_drift': perf_drift,
            'recommendation': 'retrain' if self.drift_detected else 'continue'
        }


# ============================================================================
# STRATEGY EVOLUTION - Genetic Algorithm for Parameter Optimization
# ============================================================================

class StrategyEvolution:
    """
    Uses genetic algorithms to evolve and optimize trading strategy parameters.
    Continuously searches for better configurations based on performance.
    """
    
    def __init__(self, population_size: int = 20):
        self.population_size = population_size
        self.generation = 0
        self.best_fitness = float('-inf')
        self.best_params = None
        
        # Parameter ranges for evolution
        self.param_ranges = {
            'profit_target': (0.01, 0.05),      # 1% to 5%
            'stop_loss': (0.01, 0.05),          # 1% to 5%
            'signal_threshold': (0.2, 0.5),     # Signal strength threshold
            'position_size_pct': (0.02, 0.10),  # 2% to 10% of portfolio
            'momentum_weight': (0.3, 0.9),      # Momentum score weight
            'risk_weight': (0.1, 0.5),          # Risk score weight
            'lookback_short': (5, 20),          # Short-term lookback days
            'lookback_long': (20, 60),          # Long-term lookback days
        }
        
        # Current population
        self.population = self._initialize_population()
        self.fitness_scores = [0.0] * population_size
        
    def _initialize_population(self) -> List[Dict[str, float]]:
        """Create initial random population"""
        population = []
        for _ in range(self.population_size):
            individual = {}
            for param, (low, high) in self.param_ranges.items():
                if param.endswith('_short') or param.endswith('_long'):
                    individual[param] = int(np.random.uniform(low, high))
                else:
                    individual[param] = np.random.uniform(low, high)
            population.append(individual)
        return population
    
    def evaluate_fitness(self, params: Dict[str, float], trade_history: List[TradeRecord]) -> float:
        """
        Evaluate fitness of a parameter set based on historical performance.
        Higher fitness = better strategy.
        """
        if len(trade_history) < 10:
            return 0.0
            
        # Simulate performance with these parameters
        total_pnl = sum(t.pnl for t in trade_history)
        win_rate = sum(1 for t in trade_history if t.was_profitable) / len(trade_history)
        
        # Calculate Sharpe-like ratio
        returns = [t.pnl_percent for t in trade_history]
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0
        
        # Fitness combines multiple objectives
        fitness = (
            total_pnl * 0.3 +              # Total profit
            win_rate * 100 * 0.3 +         # Win rate
            sharpe * 10 * 0.2 +            # Risk-adjusted returns
            len(trade_history) * 0.1 * 0.2  # Number of trades (activity)
        )
        
        return fitness
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Create child by combining two parents"""
        child = {}
        for param in self.param_ranges.keys():
            # Random crossover point
            if np.random.random() < 0.5:
                child[param] = parent1[param]
            else:
                child[param] = parent2[param]
        return child
    
    def _mutate(self, individual: Dict, mutation_rate: float = 0.1) -> Dict:
        """Apply random mutations to individual"""
        mutated = individual.copy()
        for param, (low, high) in self.param_ranges.items():
            if np.random.random() < mutation_rate:
                # Gaussian mutation
                delta = (high - low) * np.random.normal(0, 0.1)
                new_value = mutated[param] + delta
                new_value = np.clip(new_value, low, high)
                if param.endswith('_short') or param.endswith('_long'):
                    mutated[param] = int(new_value)
                else:
                    mutated[param] = new_value
        return mutated
    
    def evolve(self, trade_history: List[TradeRecord]) -> Dict[str, float]:
        """
        Run one generation of evolution.
        Returns the best parameters found.
        """
        # Evaluate fitness for all individuals
        self.fitness_scores = [
            self.evaluate_fitness(ind, trade_history) 
            for ind in self.population
        ]
        
        # Track best
        best_idx = np.argmax(self.fitness_scores)
        if self.fitness_scores[best_idx] > self.best_fitness:
            self.best_fitness = self.fitness_scores[best_idx]
            self.best_params = self.population[best_idx].copy()
        
        # Selection - tournament selection
        new_population = []
        
        # Elitism - keep best 2
        sorted_indices = np.argsort(self.fitness_scores)[::-1]
        new_population.append(self.population[sorted_indices[0]].copy())
        new_population.append(self.population[sorted_indices[1]].copy())
        
        # Create rest through crossover and mutation
        while len(new_population) < self.population_size:
            # Tournament selection
            candidates = np.random.choice(len(self.population), 4, replace=False)
            fitness_candidates = [self.fitness_scores[i] for i in candidates]
            parent1 = self.population[candidates[np.argmax(fitness_candidates[:2])]]
            parent2 = self.population[candidates[np.argmax(fitness_candidates[2:])]]
            
            # Crossover
            child = self._crossover(parent1, parent2)
            
            # Mutate
            child = self._mutate(child)
            
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1
        
        logger.info(f"Generation {self.generation}: Best fitness = {self.best_fitness:.2f}")
        
        return self.best_params


# ============================================================================
# MAIN ADAPTIVE LEARNING ENGINE - Orchestrates all components
# ============================================================================

class AdaptiveLearningEngine:
    """
    Main orchestrator for the continuous learning system.
    Coordinates online learning, drift detection, RL rewards, and evolution.
    """
    
    def __init__(self, state_dir: str = 'state/learning'):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize all components
        self.online_model = OnlineLearningModel(n_features=20)
        self.reward_system = ReinforcementLearningReward()
        self.drift_detector = DriftDetector(window_size=100)
        self.strategy_evolution = StrategyEvolution(population_size=20)
        
        # Trade history for learning
        self.trade_history: List[TradeRecord] = []
        self.max_history = 10000
        
        # Performance tracking
        self.metrics = PerformanceMetrics()
        self.model_state = ModelState()
        
        # Learning configuration
        self.config = {
            'learn_after_every_trade': True,
            'experience_replay_interval': 50,  # Replay every N trades
            'drift_check_interval': 20,        # Check drift every N trades
            'evolution_interval': 100,         # Evolve every N trades
            'auto_retrain_on_drift': True,
            'min_trades_before_evolution': 50,
        }
        
        # Current best parameters
        self.active_params = {
            'profit_target': 0.015,
            'stop_loss': 0.02,
            'signal_threshold': 0.3,
            'position_size_pct': 0.05,
            'momentum_weight': 0.85,
            'risk_weight': 0.15,
            'lookback_short': 5,
            'lookback_long': 20,
        }
        
        # Load saved state if exists
        self._load_state()
        
        logger.info("Adaptive Learning Engine initialized")
        
    def record_trade(self, trade: TradeRecord) -> Dict[str, Any]:
        """
        Record a completed trade and trigger learning.
        This is the main entry point - called after EVERY trade.
        """
        # Add to history
        self.trade_history.append(trade)
        if len(self.trade_history) > self.max_history:
            self.trade_history = self.trade_history[-self.max_history:]
        
        # Update metrics
        self._update_metrics(trade)
        
        results = {
            'trade_recorded': True,
            'trade_id': trade.trade_id,
            'learning_updates': []
        }
        
        # 1. Calculate RL reward
        reward = self.reward_system.calculate_reward(trade)
        results['reward'] = reward
        results['learning_updates'].append('reward_calculated')
        
        # 2. Online learning update
        if self.config['learn_after_every_trade'] and trade.features:
            features = np.array(list(trade.features.values()))
            self.online_model.partial_fit(features, np.array([trade.actual_direction]))
            self.online_model.update_accuracy(trade.predicted_direction, trade.actual_direction)
            results['learning_updates'].append('online_model_updated')
        
        # 3. Update drift detector
        if trade.features:
            features = np.array(list(trade.features.values()))
            prediction_correct = trade.predicted_direction == trade.actual_direction
            self.drift_detector.add_observation(features, prediction_correct, trade.pnl)
        
        n_trades = len(self.trade_history)
        
        # 4. Experience replay (periodic)
        if n_trades % self.config['experience_replay_interval'] == 0:
            self.online_model.experience_replay(n_samples=100)
            results['learning_updates'].append('experience_replay')
        
        # 5. Check for drift (periodic)
        if n_trades % self.config['drift_check_interval'] == 0:
            drift_result = self.drift_detector.detect_drift()
            results['drift_check'] = drift_result
            
            if drift_result['drift_detected'] and self.config['auto_retrain_on_drift']:
                self._trigger_retrain(drift_result['drift_type'])
                results['learning_updates'].append('auto_retrain_triggered')
        
        # 6. Strategy evolution (periodic)
        if (n_trades % self.config['evolution_interval'] == 0 and 
            n_trades >= self.config['min_trades_before_evolution']):
            new_params = self.strategy_evolution.evolve(self.trade_history)
            if new_params:
                self._apply_evolved_params(new_params)
                results['learning_updates'].append('strategy_evolved')
                results['new_params'] = new_params
        
        # Save state periodically
        if n_trades % 50 == 0:
            self._save_state()
        
        return results
    
    def get_prediction(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Get prediction from the online learning model.
        Returns direction and confidence.
        """
        if not features:
            return {'direction': 0, 'confidence': 0.0, 'model_ready': False}
        
        feature_array = np.array(list(features.values()))
        direction, confidence = self.online_model.predict(feature_array)
        
        return {
            'direction': int(direction[0]),
            'confidence': float(confidence[0]),
            'model_ready': self.online_model.is_fitted,
            'recent_accuracy': self.online_model.get_recent_accuracy()
        }
    
    def _update_metrics(self, trade: TradeRecord):
        """Update performance metrics"""
        self.metrics.total_trades += 1
        self.metrics.total_pnl += trade.pnl
        
        if trade.was_profitable:
            self.metrics.winning_trades += 1
            wins = [t.pnl for t in self.trade_history if t.was_profitable]
            self.metrics.avg_win = sum(wins) / len(wins) if wins else 0
        else:
            self.metrics.losing_trades += 1
            losses = [t.pnl for t in self.trade_history if not t.was_profitable]
            self.metrics.avg_loss = sum(losses) / len(losses) if losses else 0
        
        self.metrics.win_rate = self.metrics.winning_trades / self.metrics.total_trades
        
        if self.metrics.avg_loss != 0:
            self.metrics.profit_factor = abs(self.metrics.avg_win / self.metrics.avg_loss)
        
        # Prediction accuracy
        correct = sum(1 for t in self.trade_history[-100:] 
                     if t.predicted_direction == t.actual_direction)
        self.metrics.prediction_accuracy = correct / min(100, len(self.trade_history))
        
        self.metrics.last_updated = datetime.now()
    
    def _trigger_retrain(self, drift_type: str):
        """Trigger model retraining due to drift"""
        logger.warning(f"Drift detected ({drift_type}), triggering retrain...")
        
        # Get recent training data
        recent_trades = self.trade_history[-500:]
        if len(recent_trades) < 50:
            logger.warning("Not enough trades for retraining")
            return
        
        # Extract features and labels
        X = np.array([list(t.features.values()) for t in recent_trades if t.features])
        y = np.array([t.actual_direction for t in recent_trades if t.features])
        
        if len(X) < 50:
            return
        
        # Reset and retrain online model
        self.online_model = OnlineLearningModel(n_features=X.shape[1])
        
        # Train in batches
        batch_size = 32
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            self.online_model.partial_fit(batch_X, batch_y)
        
        self.model_state.last_retrain = datetime.now()
        self.model_state.retrain_count += 1
        
        logger.info(f"Retrain complete. Total retrains: {self.model_state.retrain_count}")
    
    def _apply_evolved_params(self, new_params: Dict[str, float]):
        """Apply evolved parameters to active trading"""
        # Only apply if significantly better
        if self.strategy_evolution.best_fitness > 0:
            self.active_params.update(new_params)
            logger.info(f"Applied evolved parameters: {new_params}")
    
    def get_active_params(self) -> Dict[str, float]:
        """Get current active trading parameters"""
        return self.active_params.copy()
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the learning system"""
        return {
            'metrics': asdict(self.metrics),
            'model_state': {
                'is_fitted': self.online_model.is_fitted,
                'recent_accuracy': self.online_model.get_recent_accuracy(),
                'experience_buffer_size': len(self.online_model.experience_buffer),
                'last_retrain': self.model_state.last_retrain.isoformat(),
                'retrain_count': self.model_state.retrain_count,
            },
            'drift': {
                'detected': self.drift_detector.drift_detected,
                'type': self.drift_detector.drift_type,
                'score': self.drift_detector.drift_score,
            },
            'reward_system': {
                'cumulative_reward': self.reward_system.cumulative_reward,
                'average_reward': self.reward_system.get_average_reward(),
                'trend': self.reward_system.get_reward_trend(),
            },
            'evolution': {
                'generation': self.strategy_evolution.generation,
                'best_fitness': self.strategy_evolution.best_fitness,
            },
            'active_params': self.active_params,
            'total_trades_recorded': len(self.trade_history),
        }
    
    def _save_state(self):
        """Save learning state to disk"""
        state = {
            'trade_history': [asdict(t) for t in self.trade_history[-1000:]],
            'active_params': self.active_params,
            'model_state': asdict(self.model_state),
            'evolution_generation': self.strategy_evolution.generation,
            'evolution_best_fitness': self.strategy_evolution.best_fitness,
            'saved_at': datetime.now().isoformat(),
        }
        
        state_file = self.state_dir / 'learning_state.json'
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        # Save online model
        model_file = self.state_dir / 'online_model.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump({
                'classifier': self.online_model.classifier,
                'scaler': self.online_model.scaler,
                'is_fitted': self.online_model.is_fitted,
            }, f)
    
    def _load_state(self):
        """Load learning state from disk"""
        state_file = self.state_dir / 'learning_state.json'
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)
                self.active_params.update(state.get('active_params', {}))
                logger.info(f"Loaded learning state from {state_file}")
            except Exception as e:
                logger.warning(f"Could not load state: {e}")
        
        model_file = self.state_dir / 'online_model.pkl'
        if model_file.exists():
            try:
                with open(model_file, 'rb') as f:
                    data = pickle.load(f)
                self.online_model.classifier = data['classifier']
                self.online_model.scaler = data['scaler']
                self.online_model.is_fitted = data['is_fitted']
                logger.info("Loaded online model from disk")
            except Exception as e:
                logger.warning(f"Could not load model: {e}")


# ============================================================================
# SINGLETON INSTANCE FOR GLOBAL ACCESS
# ============================================================================

_learning_engine: Optional[AdaptiveLearningEngine] = None

def get_learning_engine() -> AdaptiveLearningEngine:
    """Get or create the global learning engine instance"""
    global _learning_engine
    if _learning_engine is None:
        _learning_engine = AdaptiveLearningEngine()
    return _learning_engine


if __name__ == '__main__':
    # Test the system
    engine = get_learning_engine()
    print("Adaptive Learning Engine Status:")
    print(json.dumps(engine.get_status(), indent=2, default=str))
