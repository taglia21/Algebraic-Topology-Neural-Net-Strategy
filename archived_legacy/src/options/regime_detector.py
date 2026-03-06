"""
Market Regime Detection using Hidden Markov Model
==================================================

Renaissance Technologies-inspired regime detection system.
Uses HMM to identify 4 market regimes and dynamically adjust strategy weights.

Regimes:
1. BULL_LOW_VOL: Risk-on, sell premium aggressively
2. BULL_HIGH_VOL: Cautious long, reduced premium selling
3. BEAR_LOW_VOL: Mean reversion opportunities
4. BEAR_HIGH_VOL: Crisis mode, maximum hedging

Features:
- VIX level (normalized)
- VIX term structure slope
- SPY 20-day return
- Put/Call ratio
- Market breadth (advance/decline)
"""

import numpy as np
import pandas as pd
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
from hmmlearn import hmm
import logging
import yfinance as yf
from datetime import datetime, timedelta


# ============================================================================
# ENUMS AND DATA MODELS
# ============================================================================

class MarketRegime(Enum):
    """Market regime classification."""
    BULL_LOW_VOL = "bull_low_vol"      # Risk-on, sell premium
    BULL_HIGH_VOL = "bull_high_vol"    # Cautious long
    BEAR_LOW_VOL = "bear_low_vol"      # Mean reversion
    BEAR_HIGH_VOL = "bear_high_vol"    # Crisis, max hedging


@dataclass
class RegimeState:
    """Current regime state with metadata."""
    current_regime: MarketRegime
    confidence: float  # 0-1, probability of current regime
    probabilities: Dict[MarketRegime, float]  # All regime probabilities
    features: Dict[str, float]  # Current feature values
    timestamp: datetime


# ============================================================================
# REGIME DETECTOR
# ============================================================================

class RegimeDetector:
    """
    Hidden Markov Model regime detector.
    
    This is the foundation module - all other modules depend on knowing
    the current market regime to adjust their parameters.
    
    Features used:
    - VIX level (normalized 0-100)
    - VIX term structure slope (VIX/VIX3M ratio)
    - SPY 20-day return (%)
    - Put/Call volume ratio
    - Market breadth (advance/decline line)
    """
    
    # Strategy weights for each regime (must sum to 1.0)
    REGIME_WEIGHTS: Dict[MarketRegime, Dict[str, float]] = {
        MarketRegime.BULL_LOW_VOL: {
            "iv_rank": 0.40,         # High weight: sell premium
            "theta_decay": 0.35,     # High weight: collect decay
            "mean_reversion": 0.15,  # Low weight: trending market
            "delta_hedging": 0.10,   # Low weight: low risk
        },
        MarketRegime.BULL_HIGH_VOL: {
            "iv_rank": 0.35,         # Moderate: still sell premium
            "theta_decay": 0.25,     # Lower: more uncertainty
            "mean_reversion": 0.20,  # Higher: choppier action
            "delta_hedging": 0.20,   # Higher: need hedging
        },
        MarketRegime.BEAR_LOW_VOL: {
            "iv_rank": 0.25,         # Lower: less premium to sell
            "theta_decay": 0.30,     # Moderate: still works
            "mean_reversion": 0.30,  # High: mean reversion dominates
            "delta_hedging": 0.15,   # Moderate: some protection
        },
        MarketRegime.BEAR_HIGH_VOL: {
            "iv_rank": 0.15,         # Low: risky to sell premium
            "theta_decay": 0.10,     # Low: too much gamma risk
            "mean_reversion": 0.15,  # Low: trends can persist
            "delta_hedging": 0.60,   # High: protect the portfolio
        }
    }
    
    def __init__(self, n_regimes: int = 4, lookback_days: int = 252):
        """
        Initialize regime detector.
        
        Args:
            n_regimes: Number of HMM states (default 4)
            lookback_days: Historical data window (default 252 = 1 year)
        """
        self.n_regimes = n_regimes
        self.lookback_days = lookback_days
        self.model: Optional[hmm.GaussianHMM] = None
        self.is_fitted = False
        self.logger = logging.getLogger(__name__)
        
        # Mapping from HMM state index to MarketRegime
        self._regime_mapping: Dict[int, MarketRegime] = {}
        
        # Feature statistics for normalization
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_std: Optional[np.ndarray] = None
        
        self.logger.info(f"Initialized RegimeDetector (n_regimes={n_regimes}, lookback={lookback_days})")
    
    async def fit(self, market_data: Optional[pd.DataFrame] = None) -> None:
        """
        Fit HMM on historical market data.
        
        If market_data is None, fetches last year of data automatically.
        
        Args:
            market_data: Historical data with columns ['vix', 'spy_return', 'put_call', 'breadth', 'vix_slope']
        
        Raises:
            ValueError: If fitting fails
        """
        try:
            # Fetch data if not provided
            if market_data is None:
                self.logger.info("Fetching historical market data...")
                market_data = await self._fetch_historical_features()
            
            # Validate data
            if len(market_data) < 60:
                raise ValueError(f"Insufficient data: {len(market_data)} rows (need at least 60)")
            
            # Extract features
            features = market_data[['vix', 'spy_return', 'put_call', 'breadth', 'vix_slope']].values
            
            # Normalize features (zero mean, unit variance)
            self._feature_mean = np.mean(features, axis=0)
            self._feature_std = np.std(features, axis=0)
            features_normalized = (features - self._feature_mean) / (self._feature_std + 1e-8)
            
            # Initialize HMM model
            self.model = hmm.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="full",  # Allow correlations between features
                n_iter=100,
                random_state=42,
                init_params="stmc",  # Initialize startprob, transmat, means, covars
            )
            
            # Fit model
            self.logger.info(f"Fitting HMM on {len(features_normalized)} samples...")
            self.model.fit(features_normalized)
            
            # Map HMM states to MarketRegime enum
            self._map_states_to_regimes(features_normalized)
            
            self.is_fitted = True
            self.logger.info("✓ HMM fitting complete, regime mapping established")
            
            # Log regime characteristics
            self._log_regime_characteristics()
        
        except Exception as e:
            self.logger.error(f"Failed to fit HMM: {e}", exc_info=True)
            raise ValueError(f"HMM fitting failed: {e}")
    
    async def detect_current_regime(self) -> RegimeState:
        """
        Detect current market regime.
        
        Returns:
            RegimeState with current regime, confidence, and probabilities
        
        Raises:
            RuntimeError: If model not fitted
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        try:
            # Fetch current features
            current_features = await self._fetch_current_features()
            
            # Normalize using stored statistics
            features_normalized = (current_features - self._feature_mean) / (self._feature_std + 1e-8)
            features_normalized = features_normalized.reshape(1, -1)
            
            # Predict regime probabilities
            log_prob, state_sequence = self.model.decode(features_normalized, algorithm="viterbi")
            current_state = state_sequence[0]
            
            # Get regime probabilities using forward algorithm
            posteriors = self.model.predict_proba(features_normalized)[0]
            
            # Map to MarketRegime
            current_regime = self._regime_mapping[current_state]
            confidence = posteriors[current_state]
            
            # Build probability dict for all regimes
            probabilities = {
                self._regime_mapping[i]: posteriors[i]
                for i in range(self.n_regimes)
            }
            
            # Build feature dict
            feature_names = ['vix', 'spy_return', 'put_call', 'breadth', 'vix_slope']
            features_dict = {
                name: float(current_features[i])
                for i, name in enumerate(feature_names)
            }
            
            regime_state = RegimeState(
                current_regime=current_regime,
                confidence=confidence,
                probabilities=probabilities,
                features=features_dict,
                timestamp=datetime.now(),
            )
            
            self.logger.info(
                f"Detected regime: {current_regime.value} "
                f"(confidence: {confidence:.1%})"
            )
            
            return regime_state
        
        except Exception as e:
            self.logger.error(f"Regime detection failed: {e}", exc_info=True)
            # Return default regime on error
            return self._get_default_regime()
    
    def get_strategy_weights(self, regime: MarketRegime) -> Dict[str, float]:
        """
        Get strategy weights for given regime.
        
        Args:
            regime: Market regime
        
        Returns:
            Dict of strategy weights (sum to 1.0)
        """
        return self.REGIME_WEIGHTS[regime].copy()
    
    async def _fetch_current_features(self) -> np.ndarray:
        """
        Fetch current market features.
        
        Returns:
            Array of [vix, spy_return, put_call, breadth, vix_slope]
        """
        try:
            # Fetch VIX
            vix_ticker = yf.Ticker("^VIX")
            vix_data = vix_ticker.history(period="5d")
            vix_current = float(vix_data['Close'].iloc[-1])
            
            # Fetch VIX3M for term structure
            vix3m_ticker = yf.Ticker("^VIX3M")
            vix3m_data = vix3m_ticker.history(period="5d")
            vix3m_current = float(vix3m_data['Close'].iloc[-1])
            vix_slope = vix_current / (vix3m_current + 1e-8)
            
            # Fetch SPY for returns
            spy_ticker = yf.Ticker("SPY")
            spy_data = spy_ticker.history(period="1mo")
            spy_return = float(
                (spy_data['Close'].iloc[-1] / spy_data['Close'].iloc[-20] - 1) * 100
            )
            
            # Estimate put/call ratio (simplified: use VIX level as proxy)
            # In production, fetch actual CBOE put/call data
            put_call_ratio = vix_current / 20.0  # Normalize around 1.0
            
            # Estimate market breadth (simplified: use SPY momentum)
            # In production, fetch NYSE advance/decline data
            breadth = 1.0 if spy_return > 0 else -1.0
            
            features = np.array([
                vix_current,
                spy_return,
                put_call_ratio,
                breadth,
                vix_slope,
            ])
            
            return features
        
        except Exception as e:
            self.logger.error(f"Failed to fetch current features: {e}")
            # Return default neutral features
            return np.array([20.0, 0.0, 1.0, 0.0, 1.0])
    
    async def _fetch_historical_features(self) -> pd.DataFrame:
        """
        Fetch historical market data for fitting.
        
        Returns:
            DataFrame with features
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days + 30)
            
            self.logger.info(f"Fetching data from {start_date.date()} to {end_date.date()}")
            
            # Fetch VIX
            vix_ticker = yf.Ticker("^VIX")
            vix_data = vix_ticker.history(start=start_date, end=end_date)
            
            # Fetch VIX3M
            vix3m_ticker = yf.Ticker("^VIX3M")
            vix3m_data = vix3m_ticker.history(start=start_date, end=end_date)
            
            # Fetch SPY
            spy_ticker = yf.Ticker("SPY")
            spy_data = spy_ticker.history(start=start_date, end=end_date)
            
            # Build dataframe
            df = pd.DataFrame({
                'vix': vix_data['Close'],
                'vix3m': vix3m_data['Close'],
                'spy_close': spy_data['Close'],
            })
            
            # Calculate derived features
            df['vix_slope'] = df['vix'] / (df['vix3m'] + 1e-8)
            df['spy_return'] = df['spy_close'].pct_change(20) * 100  # 20-day return %
            df['put_call'] = df['vix'] / 20.0  # Proxy for put/call ratio
            df['breadth'] = np.where(df['spy_return'] > 0, 1.0, -1.0)
            
            # Drop NaN rows
            df = df.dropna()
            
            # Select final features
            df = df[['vix', 'spy_return', 'put_call', 'breadth', 'vix_slope']]
            
            self.logger.info(f"Fetched {len(df)} historical samples")
            
            return df
        
        except Exception as e:
            self.logger.error(f"Failed to fetch historical data: {e}", exc_info=True)
            raise
    
    def _map_states_to_regimes(self, features: np.ndarray) -> None:
        """
        Map HMM states to MarketRegime enum based on feature centroids.
        
        Logic:
        - State with highest VIX + negative returns = BEAR_HIGH_VOL
        - State with high VIX + positive returns = BULL_HIGH_VOL
        - State with low VIX + positive returns = BULL_LOW_VOL
        - State with low VIX + negative returns = BEAR_LOW_VOL
        
        Args:
            features: Normalized feature array used for fitting
        """
        # Predict states for all historical data
        states = self.model.predict(features)
        
        # Calculate centroids (mean features) for each state
        centroids = {}
        for state in range(self.n_regimes):
            state_mask = (states == state)
            if state_mask.sum() > 0:
                # Use unnormalized features for interpretation
                unnormalized = features[state_mask] * self._feature_std + self._feature_mean
                centroids[state] = {
                    'vix': np.mean(unnormalized[:, 0]),
                    'spy_return': np.mean(unnormalized[:, 1]),
                    'put_call': np.mean(unnormalized[:, 2]),
                    'breadth': np.mean(unnormalized[:, 3]),
                    'vix_slope': np.mean(unnormalized[:, 4]),
                }
        
        # Classify each state into a regime
        regime_mapping = {}
        
        for state, centroid in centroids.items():
            vix = centroid['vix']
            spy_ret = centroid['spy_return']
            
            # High volatility threshold
            high_vol = vix > 20.0
            
            # Bullish threshold
            bullish = spy_ret > 0.0
            
            # Assign regime
            if bullish and not high_vol:
                regime = MarketRegime.BULL_LOW_VOL
            elif bullish and high_vol:
                regime = MarketRegime.BULL_HIGH_VOL
            elif not bullish and not high_vol:
                regime = MarketRegime.BEAR_LOW_VOL
            else:  # not bullish and high_vol
                regime = MarketRegime.BEAR_HIGH_VOL
            
            regime_mapping[state] = regime
        
        # Handle case where not all 4 regimes are represented
        # Ensure all regimes exist in mapping
        used_regimes = set(regime_mapping.values())
        all_regimes = set(MarketRegime)
        missing_regimes = all_regimes - used_regimes
        
        if missing_regimes:
            self.logger.warning(f"Not all regimes detected in data: {missing_regimes}")
            # Assign missing regimes to nearest state
            for missing_regime in missing_regimes:
                # Find unused state (if any)
                unused_states = set(range(self.n_regimes)) - set(regime_mapping.keys())
                if unused_states:
                    regime_mapping[unused_states.pop()] = missing_regime
        
        self._regime_mapping = regime_mapping
        
        self.logger.info("Regime mapping established:")
        for state, regime in regime_mapping.items():
            self.logger.info(f"  State {state} -> {regime.value}")
    
    def _log_regime_characteristics(self) -> None:
        """Log characteristics of each detected regime."""
        self.logger.info("="*60)
        self.logger.info("REGIME CHARACTERISTICS")
        self.logger.info("="*60)
        
        for regime in MarketRegime:
            weights = self.REGIME_WEIGHTS[regime]
            self.logger.info(f"\n{regime.value.upper()}:")
            self.logger.info(f"  Strategy Weights:")
            for strategy, weight in weights.items():
                self.logger.info(f"    {strategy}: {weight:.1%}")
    
    def _get_default_regime(self) -> RegimeState:
        """
        Get default regime state (fallback on errors).
        
        Returns:
            RegimeState with BULL_LOW_VOL (neutral default)
        """
        return RegimeState(
            current_regime=MarketRegime.BULL_LOW_VOL,
            confidence=0.25,  # Low confidence
            probabilities={
                MarketRegime.BULL_LOW_VOL: 0.25,
                MarketRegime.BULL_HIGH_VOL: 0.25,
                MarketRegime.BEAR_LOW_VOL: 0.25,
                MarketRegime.BEAR_HIGH_VOL: 0.25,
            },
            features={
                'vix': 20.0,
                'spy_return': 0.0,
                'put_call': 1.0,
                'breadth': 0.0,
                'vix_slope': 1.0,
            },
            timestamp=datetime.now(),
        )


# ============================================================================
# TESTING HELPER
# ============================================================================

async def test_regime_detector():
    """Test the regime detector."""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    detector = RegimeDetector()
    
    # Fit on historical data
    print("\n" + "="*60)
    print("FITTING HMM MODEL")
    print("="*60)
    await detector.fit()
    
    # Detect current regime
    print("\n" + "="*60)
    print("DETECTING CURRENT REGIME")
    print("="*60)
    state = await detector.detect_current_regime()
    
    print(f"\nCurrent Regime: {state.current_regime.value}")
    print(f"Confidence: {state.confidence:.1%}")
    print(f"\nRegime Probabilities:")
    for regime, prob in state.probabilities.items():
        print(f"  {regime.value}: {prob:.1%}")
    
    print(f"\nCurrent Features:")
    for feature, value in state.features.items():
        print(f"  {feature}: {value:.2f}")
    
    print(f"\nStrategy Weights for {state.current_regime.value}:")
    weights = detector.get_strategy_weights(state.current_regime)
    for strategy, weight in weights.items():
        print(f"  {strategy}: {weight:.1%}")
    
    # Validate
    assert isinstance(state.current_regime, MarketRegime)
    assert 0.0 <= state.confidence <= 1.0
    assert abs(sum(state.probabilities.values()) - 1.0) < 0.01
    assert abs(sum(weights.values()) - 1.0) < 0.01
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_regime_detector())
