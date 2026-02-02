#!/usr/bin/env python3
"""
V36 Adaptive HMM Regime Detector
================================
Enhanced HMM regime detection with online learning and persistence filtering.

Features:
- 4-state HMM: BULL, BEAR, SIDEWAYS, CRISIS
- Online learning with rolling 252-day training window
- Regime persistence filter (3 consecutive signals required)
- Confidence scoring via state probabilities
- Model persistence with pickle

Usage:
    hmm = AdaptiveHMM()
    hmm.fit(historical_returns)
    regime = hmm.get_current_regime(latest_returns)
    probs = hmm.get_regime_probabilities()
"""

import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    warnings.warn("hmmlearn not installed. Install with: pip install hmmlearn")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('V36_HMM')


class MarketRegime(Enum):
    """Market regime states."""
    BULL = 0
    BEAR = 1
    SIDEWAYS = 2
    CRISIS = 3


@dataclass
class HMMConfig:
    """Configuration for adaptive HMM."""
    n_states: int = 4
    rolling_window: int = 252  # 1 year of trading days
    persistence_threshold: int = 3  # Consecutive signals required
    covariance_type: str = "full"
    n_iter: int = 100
    random_state: int = 42


@dataclass
class RegimeResult:
    """Result of regime detection."""
    regime: MarketRegime
    raw_state: int
    probabilities: Dict[str, float]
    confidence: float
    is_confirmed: bool  # Passed persistence filter
    consecutive_count: int
    timestamp: datetime = field(default_factory=datetime.now)


class AdaptiveHMM:
    """
    Adaptive Hidden Markov Model for market regime detection.
    
    Uses online learning with rolling window and persistence filtering
    to provide stable regime classifications.
    
    Args:
        config: HMM configuration parameters
    
    Example:
        hmm = AdaptiveHMM()
        hmm.fit(returns_series)
        result = hmm.get_current_regime(new_returns)
        print(f"Regime: {result.regime.name}, Confidence: {result.confidence:.2%}")
    """

    def __init__(self, config: Optional[HMMConfig] = None):
        if not HMM_AVAILABLE:
            raise ImportError("hmmlearn required. Install with: pip install hmmlearn")
        
        self.config = config or HMMConfig()
        self.model: Optional[GaussianHMM] = None
        self.returns_buffer: List[float] = []
        self._state_history: List[int] = []
        self._current_regime: MarketRegime = MarketRegime.SIDEWAYS
        self._consecutive_count: int = 0
        self._last_raw_state: int = -1
        self._state_mapping: Dict[int, MarketRegime] = {}
        self._is_fitted = False

    def _create_model(self) -> GaussianHMM:
        """Create a new HMM model."""
        return GaussianHMM(
            n_components=self.config.n_states,
            covariance_type=self.config.covariance_type,
            n_iter=self.config.n_iter,
            random_state=self.config.random_state
        )

    def _prepare_features(self, returns: np.ndarray) -> np.ndarray:
        """Prepare features for HMM from returns."""
        returns = np.asarray(returns).flatten()
        volatility = pd.Series(returns).rolling(10, min_periods=1).std().values
        features = np.column_stack([returns, volatility])
        return features[~np.isnan(features).any(axis=1)]

    def _map_states_to_regimes(self, features: np.ndarray) -> None:
        """Map HMM states to regime labels based on characteristics."""
        if self.model is None:
            return
        
        means = self.model.means_[:, 0]  # Return means
        covars = np.array([self.model.covars_[i][0, 0] for i in range(self.config.n_states)])
        
        # Sort states by mean return and volatility
        state_scores = []
        for i in range(self.config.n_states):
            state_scores.append((i, means[i], covars[i]))
        
        # Assign based on return/volatility characteristics
        sorted_by_return = sorted(state_scores, key=lambda x: x[1], reverse=True)
        sorted_by_vol = sorted(state_scores, key=lambda x: x[2], reverse=True)
        
        self._state_mapping = {}
        assigned = set()
        
        # Highest return, low vol = BULL
        for s, m, v in sorted_by_return:
            if s not in assigned and v < np.median(covars):
                self._state_mapping[s] = MarketRegime.BULL
                assigned.add(s)
                break
        
        # Highest vol = CRISIS
        for s, m, v in sorted_by_vol:
            if s not in assigned:
                self._state_mapping[s] = MarketRegime.CRISIS
                assigned.add(s)
                break
        
        # Lowest return = BEAR
        for s, m, v in reversed(sorted_by_return):
            if s not in assigned:
                self._state_mapping[s] = MarketRegime.BEAR
                assigned.add(s)
                break
        
        # Remaining = SIDEWAYS
        for i in range(self.config.n_states):
            if i not in assigned:
                self._state_mapping[i] = MarketRegime.SIDEWAYS

    def fit(self, returns: np.ndarray) -> 'AdaptiveHMM':
        """
        Fit HMM on historical returns.
        
        Args:
            returns: Array of daily returns
        
        Returns:
            self for method chaining
        """
        returns = np.asarray(returns).flatten()
        self.returns_buffer = list(returns[-self.config.rolling_window:])
        
        features = self._prepare_features(returns)
        if len(features) < 50:
            raise ValueError("Insufficient data for HMM fitting (need 50+ samples)")
        
        self.model = self._create_model()
        self.model.fit(features)
        self._map_states_to_regimes(features)
        self._is_fitted = True
        
        logger.info(f"HMM fitted on {len(features)} samples, {self.config.n_states} states")
        return self

    def retrain_with_new_data(self, new_returns: np.ndarray) -> None:
        """
        Incrementally update model with new data using rolling window.
        
        Args:
            new_returns: New daily returns to incorporate
        """
        new_returns = np.asarray(new_returns).flatten()
        self.returns_buffer.extend(new_returns)
        
        # Maintain rolling window
        if len(self.returns_buffer) > self.config.rolling_window:
            self.returns_buffer = self.returns_buffer[-self.config.rolling_window:]
        
        # Retrain on rolling window
        features = self._prepare_features(np.array(self.returns_buffer))
        if len(features) >= 50:
            self.model = self._create_model()
            self.model.fit(features)
            self._map_states_to_regimes(features)
            logger.debug(f"HMM retrained on {len(features)} samples")

    def _apply_persistence_filter(self, raw_state: int) -> Tuple[MarketRegime, bool]:
        """Apply persistence filter requiring consecutive signals."""
        if raw_state == self._last_raw_state:
            self._consecutive_count += 1
        else:
            self._consecutive_count = 1
            self._last_raw_state = raw_state
        
        proposed_regime = self._state_mapping.get(raw_state, MarketRegime.SIDEWAYS)
        
        # Require threshold consecutive signals to switch
        if self._consecutive_count >= self.config.persistence_threshold:
            if proposed_regime != self._current_regime:
                logger.info(f"Regime change: {self._current_regime.name} -> {proposed_regime.name}")
                self._current_regime = proposed_regime
            return self._current_regime, True
        
        return self._current_regime, False

    def get_current_regime(self, latest_returns: Optional[np.ndarray] = None) -> RegimeResult:
        """
        Get current regime with optional new data.
        
        Args:
            latest_returns: Optional new returns to update model
        
        Returns:
            RegimeResult with regime, probabilities, and confidence
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        if latest_returns is not None:
            self.retrain_with_new_data(latest_returns)
        
        features = self._prepare_features(np.array(self.returns_buffer))
        
        # Get state and probabilities
        raw_state = self.model.predict(features[-1:].reshape(1, -1))[0]
        state_probs = self.model.predict_proba(features[-1:].reshape(1, -1))[0]
        
        self._state_history.append(raw_state)
        
        # Apply persistence filter
        regime, is_confirmed = self._apply_persistence_filter(raw_state)
        
        # Build probability dict
        prob_dict = {}
        for state_idx, regime_enum in self._state_mapping.items():
            prob_dict[regime_enum.name] = float(state_probs[state_idx])
        
        confidence = max(state_probs)
        
        return RegimeResult(
            regime=regime,
            raw_state=raw_state,
            probabilities=prob_dict,
            confidence=confidence,
            is_confirmed=is_confirmed,
            consecutive_count=self._consecutive_count
        )

    def get_regime_probabilities(self) -> Dict[str, float]:
        """
        Get probability distribution over regimes.
        
        Returns:
            Dict mapping regime names to probabilities
        """
        if not self._is_fitted or len(self.returns_buffer) == 0:
            return {r.name: 0.25 for r in MarketRegime}
        
        features = self._prepare_features(np.array(self.returns_buffer))
        state_probs = self.model.predict_proba(features[-1:].reshape(1, -1))[0]
        
        return {
            self._state_mapping[i].name: float(state_probs[i])
            for i in range(self.config.n_states)
        }

    def save_model(self, path: str) -> None:
        """Save model to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'model': self.model,
            'config': self.config,
            'returns_buffer': self.returns_buffer,
            'state_mapping': self._state_mapping,
            'current_regime': self._current_regime,
            'consecutive_count': self._consecutive_count,
            'last_raw_state': self._last_raw_state,
            'is_fitted': self._is_fitted
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> 'AdaptiveHMM':
        """Load model from file."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.model = state['model']
        self.config = state['config']
        self.returns_buffer = state['returns_buffer']
        self._state_mapping = state['state_mapping']
        self._current_regime = state['current_regime']
        self._consecutive_count = state['consecutive_count']
        self._last_raw_state = state['last_raw_state']
        self._is_fitted = state['is_fitted']
        
        logger.info(f"Model loaded from {path}")
        return self


def main() -> None:
    """Example usage."""
    np.random.seed(42)
    
    # Simulate returns with regime changes
    bull = np.random.normal(0.001, 0.01, 100)
    bear = np.random.normal(-0.001, 0.02, 50)
    crisis = np.random.normal(-0.003, 0.04, 30)
    returns = np.concatenate([bull, bear, crisis, bull])
    
    hmm = AdaptiveHMM()
    hmm.fit(returns)
    
    result = hmm.get_current_regime()
    print(f"Current Regime: {result.regime.name}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Probabilities: {result.probabilities}")
    
    # Save and load
    hmm.save_model("cache/hmm_model.pkl")
    hmm2 = AdaptiveHMM().load_model("cache/hmm_model.pkl")
    print(f"Loaded regime: {hmm2.get_current_regime().regime.name}")


if __name__ == "__main__":
    main()
