#!/usr/bin/env python3
"""
V28 Advanced Regime Detection System
=====================================
Multi-layer market regime detection combining HMM and GARCH models.

Features:
- 4-State HMM for market regime classification (bull/bear/sideways/crisis)
- GARCH(1,1) volatility regime detection
- Adaptive strategy switching based on regime
- Regime persistence and transition probabilities
- Ensemble regime voting for robustness

States:
- HMM States: LowVolTrend, HighVolTrend, LowVolMeanRevert, Crisis
- GARCH States: LowVol, NormalVol, HighVol, ExtremeVol
- Combined Regime: bull, bear, sideways, crisis
"""

import logging
import pickle
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats

# Conditional imports
try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    warnings.warn("hmmlearn not installed, HMM regime detection disabled")

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    warnings.warn("arch not installed, GARCH volatility detection disabled")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('V28_Regime')


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class MarketRegime(Enum):
    """Primary market regime classification."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    CRISIS = "crisis"


class VolatilityRegime(Enum):
    """Volatility state classification."""
    LOW = "low_vol"
    NORMAL = "normal_vol"
    HIGH = "high_vol"
    EXTREME = "extreme_vol"


class HMMState(Enum):
    """HMM state names."""
    LOW_VOL_TREND = 0
    HIGH_VOL_TREND = 1
    LOW_VOL_MEAN_REVERT = 2
    CRISIS = 3


@dataclass
class RegimeState:
    """Complete regime state information."""
    market_regime: MarketRegime
    volatility_regime: VolatilityRegime
    hmm_state: int
    hmm_state_name: str
    hmm_probability: float
    garch_volatility: float
    garch_forecast_1d: float
    garch_forecast_5d: float
    regime_confidence: float
    trend_strength: float
    momentum_score: float
    transition_probability: float  # Probability of regime change
    recommended_strategies: Dict[str, float]  # Strategy name -> allocation
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'market_regime': self.market_regime.value,
            'volatility_regime': self.volatility_regime.value,
            'hmm_state': self.hmm_state,
            'hmm_state_name': self.hmm_state_name,
            'hmm_probability': self.hmm_probability,
            'garch_volatility': self.garch_volatility,
            'garch_forecast_1d': self.garch_forecast_1d,
            'garch_forecast_5d': self.garch_forecast_5d,
            'regime_confidence': self.regime_confidence,
            'trend_strength': self.trend_strength,
            'momentum_score': self.momentum_score,
            'transition_probability': self.transition_probability,
            'recommended_strategies': self.recommended_strategies,
            'timestamp': self.timestamp.isoformat()
        }


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

class RegimeFeatureBuilder:
    """Build features for regime detection models."""
    
    def __init__(
        self,
        return_lookback: int = 10,
        vol_lookback: int = 20,
        trend_lookback: int = 50
    ):
        self.return_lookback = return_lookback
        self.vol_lookback = vol_lookback
        self.trend_lookback = trend_lookback
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive features for regime detection.
        
        Args:
            df: DataFrame with 'close', 'high', 'low', 'volume' columns
            
        Returns:
            DataFrame with feature columns
        """
        df = df.copy()
        
        # Ensure sorted by date
        if 'date' in df.columns:
            df = df.sort_values('date')
        
        close = df['close']
        high = df.get('high', close)
        low = df.get('low', close)
        volume = df.get('volume', pd.Series([1e8] * len(df), index=close.index))
        
        # Returns features
        df['returns_1d'] = close.pct_change()
        df['returns_5d'] = close.pct_change(5)
        df['returns_10d'] = close.pct_change(self.return_lookback)
        df['returns_20d'] = close.pct_change(20)
        
        # Volatility features
        df['realized_vol_10d'] = df['returns_1d'].rolling(10).std() * np.sqrt(252)
        df['realized_vol_20d'] = df['returns_1d'].rolling(self.vol_lookback).std() * np.sqrt(252)
        df['realized_vol_60d'] = df['returns_1d'].rolling(60).std() * np.sqrt(252)
        
        # Volatility ratio
        df['vol_ratio'] = df['realized_vol_10d'] / df['realized_vol_60d'].replace(0, np.nan)
        
        # Parkinson volatility (High-Low based)
        log_hl = np.log(high / low)
        df['parkinson_vol'] = np.sqrt((log_hl ** 2).rolling(20).mean() / (4 * np.log(2))) * np.sqrt(252)
        
        # Trend features
        df['sma_20'] = close.rolling(20).mean()
        df['sma_50'] = close.rolling(self.trend_lookback).mean()
        df['sma_200'] = close.rolling(200).mean()
        
        df['price_vs_sma20'] = (close - df['sma_20']) / df['sma_20']
        df['price_vs_sma50'] = (close - df['sma_50']) / df['sma_50']
        df['price_vs_sma200'] = (close - df['sma_200']) / df['sma_200']
        
        df['sma_trend'] = (df['sma_20'] > df['sma_50']).astype(float)
        
        # Momentum features
        df['rsi_14'] = self._calculate_rsi(close, 14)
        df['rsi_5'] = self._calculate_rsi(close, 5)
        
        # MACD
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Volume features
        df['volume_sma_20'] = volume.rolling(20).mean()
        df['volume_ratio'] = volume / df['volume_sma_20'].replace(0, np.nan)
        
        # ATR
        tr = pd.DataFrame({
            'hl': high - low,
            'hc': abs(high - close.shift()),
            'lc': abs(low - close.shift())
        }).max(axis=1)
        df['atr_14'] = tr.rolling(14).mean()
        df['atr_pct'] = df['atr_14'] / close
        
        # Drawdown
        rolling_max = close.rolling(252, min_periods=1).max()
        df['drawdown'] = (close - rolling_max) / rolling_max
        
        # Higher moments
        df['skewness_20d'] = df['returns_1d'].rolling(20).skew()
        df['kurtosis_20d'] = df['returns_1d'].rolling(20).kurt()
        
        return df
    
    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def get_hmm_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get feature matrix for HMM model.
        
        Returns:
            2D array of shape (n_samples, n_features)
        """
        feature_cols = [
            'returns_10d',
            'realized_vol_20d',
            'volume_ratio',
            'price_vs_sma50'
        ]
        
        # Calculate features if not present
        if 'returns_10d' not in df.columns:
            df = self.calculate_features(df)
        
        # Handle missing values
        features_df = df[feature_cols].dropna()
        
        # Standardize features
        features = features_df.values
        if len(features) > 0:
            features = (features - np.nanmean(features, axis=0)) / (np.nanstd(features, axis=0) + 1e-8)
        
        return features


# =============================================================================
# HMM REGIME DETECTOR
# =============================================================================

class HMMRegimeDetector:
    """
    4-State Hidden Markov Model for regime detection.
    
    States:
    - 0: LowVolTrend - Calm uptrend, momentum strategies work
    - 1: HighVolTrend - Volatile uptrend, breakout/trend strategies
    - 2: LowVolMeanRevert - Range-bound, mean reversion works
    - 3: Crisis - High volatility drawdown, defensive positioning
    """
    
    STATE_NAMES = {
        0: 'LowVolTrend',
        1: 'HighVolTrend',
        2: 'LowVolMeanRevert',
        3: 'Crisis'
    }
    
    STATE_STRATEGIES = {
        0: {'momentum': 0.7, 'trend_follow': 0.2, 'mean_reversion': 0.1},
        1: {'momentum': 0.4, 'trend_follow': 0.5, 'mean_reversion': 0.1},
        2: {'momentum': 0.1, 'mean_reversion': 0.7, 'stat_arb': 0.2},
        3: {'defensive': 0.8, 'vol_target': 0.2}
    }
    
    def __init__(
        self,
        n_states: int = 4,
        n_iter: int = 100,
        random_state: int = 42,
        model_path: str = 'state/hmm_regime_model.pkl'
    ):
        self.n_states = n_states
        self.n_iter = n_iter
        self.random_state = random_state
        self.model_path = Path(model_path)
        
        self.model: Optional[GaussianHMM] = None
        self.feature_builder = RegimeFeatureBuilder()
        self.is_fitted = False
        
        # Learned parameters
        self.state_means: Optional[np.ndarray] = None
        self.state_covars: Optional[np.ndarray] = None
        self.state_mapping: Dict[int, int] = {}  # Raw state -> semantic state
        
        # History for smoothing
        self.state_history: List[int] = []
        self.probability_history: List[np.ndarray] = []
    
    def fit(self, df: pd.DataFrame) -> 'HMMRegimeDetector':
        """
        Fit HMM on market data (typically SPY).
        
        Args:
            df: DataFrame with price data
        """
        if not HMM_AVAILABLE:
            logger.warning("HMM not available, using fallback regime detection")
            self.is_fitted = True
            return self
        
        logger.info("🔄 Fitting HMM regime detector...")
        
        # Build features
        df = self.feature_builder.calculate_features(df)
        X = self.feature_builder.get_hmm_features(df)
        
        if len(X) < 100:
            raise ValueError(f"Insufficient data for HMM training: {len(X)} samples")
        
        logger.info(f"   Training samples: {len(X)}")
        
        # Initialize and fit HMM
        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type='full',
            n_iter=self.n_iter,
            random_state=self.random_state
        )
        
        self.model.fit(X)
        
        # Store learned parameters
        self.state_means = self.model.means_
        self.state_covars = self.model.covars_
        
        # Map raw states to semantic states based on characteristics
        self._map_states_to_regimes()
        
        self.is_fitted = True
        logger.info(f"✅ HMM fitted with {self.n_states} states")
        
        return self
    
    def _map_states_to_regimes(self):
        """
        Map learned HMM states to semantic regime names.
        
        Uses state means to classify based on return/volatility quadrants.
        """
        if self.state_means is None:
            return
        
        # Index: [returns_10d, realized_vol_20d, volume_ratio, price_vs_sma50]
        returns_idx = 0
        vol_idx = 1
        
        # Get mean returns and volatility for each state
        state_chars = []
        for i in range(self.n_states):
            avg_return = self.state_means[i, returns_idx]
            avg_vol = self.state_means[i, vol_idx]
            state_chars.append((i, avg_return, avg_vol))
        
        # Sort by characteristics
        sorted_states = sorted(state_chars, key=lambda x: (-x[1], x[2]))
        
        # Assign semantic meanings
        self.state_mapping = {}
        if len(sorted_states) >= 4:
            # Best returns, low vol -> LowVolTrend
            self.state_mapping[sorted_states[0][0]] = 0
            # Good returns, high vol -> HighVolTrend
            self.state_mapping[sorted_states[1][0]] = 1
            # Mediocre returns, low vol -> LowVolMeanRevert
            self.state_mapping[sorted_states[2][0]] = 2
            # Worst returns (negative), high vol -> Crisis
            self.state_mapping[sorted_states[3][0]] = 3
        else:
            # Default 1:1 mapping
            self.state_mapping = {i: i for i in range(self.n_states)}
    
    def predict(self, df: pd.DataFrame) -> Tuple[int, float]:
        """
        Predict current regime state.
        
        Returns:
            Tuple of (state_id, probability)
        """
        if not self.is_fitted:
            return 2, 0.5  # Default to mean reversion
        
        if not HMM_AVAILABLE or self.model is None:
            return self._fallback_predict(df)
        
        # Build features
        df = self.feature_builder.calculate_features(df)
        X = self.feature_builder.get_hmm_features(df)
        
        if len(X) == 0:
            return 2, 0.5
        
        # Get state probabilities
        state_probs = self.model.predict_proba(X)
        
        # Get most likely current state
        raw_state = np.argmax(state_probs[-1])
        probability = state_probs[-1, raw_state]
        
        # Map to semantic state
        semantic_state = self.state_mapping.get(raw_state, raw_state)
        
        # Update history
        self.state_history.append(semantic_state)
        self.probability_history.append(state_probs[-1])
        
        # Keep limited history
        if len(self.state_history) > 252:
            self.state_history = self.state_history[-252:]
            self.probability_history = self.probability_history[-252:]
        
        return semantic_state, probability
    
    def _fallback_predict(self, df: pd.DataFrame) -> Tuple[int, float]:
        """Fallback regime prediction without HMM."""
        df = self.feature_builder.calculate_features(df)
        
        if len(df) == 0:
            return 2, 0.5
        
        last = df.iloc[-1]
        
        # Simple rule-based classification
        returns_10d = last.get('returns_10d', 0)
        vol_20d = last.get('realized_vol_20d', 0.15)
        trend = last.get('price_vs_sma50', 0)
        
        # Determine regime
        if returns_10d < -0.05 and vol_20d > 0.25:
            return 3, 0.7  # Crisis
        elif trend > 0.02 and vol_20d < 0.15:
            return 0, 0.7  # LowVolTrend
        elif trend > 0 and vol_20d >= 0.15:
            return 1, 0.6  # HighVolTrend
        else:
            return 2, 0.6  # LowVolMeanRevert
    
    def get_transition_probability(self) -> float:
        """Estimate probability of regime change."""
        if len(self.probability_history) < 5:
            return 0.1
        
        # Look at recent state stability
        recent_probs = self.probability_history[-5:]
        current_state = self.state_history[-1]
        
        # Average probability of staying in current state
        stay_probs = [p[current_state] for p in recent_probs]
        avg_stay = np.mean(stay_probs)
        
        return 1.0 - avg_stay
    
    def get_state_name(self, state: int) -> str:
        """Get human-readable state name."""
        return self.STATE_NAMES.get(state, f"Unknown_{state}")
    
    def get_strategy_allocation(self, state: int) -> Dict[str, float]:
        """Get recommended strategy allocation for state."""
        return self.STATE_STRATEGIES.get(state, {'defensive': 1.0})
    
    def save(self):
        """Save model to disk."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'model': self.model,
            'state_means': self.state_means,
            'state_covars': self.state_covars,
            'state_mapping': self.state_mapping,
            'is_fitted': self.is_fitted
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Saved HMM model to {self.model_path}")
    
    def load(self) -> bool:
        """Load model from disk."""
        if not self.model_path.exists():
            return False
        
        try:
            with open(self.model_path, 'rb') as f:
                state = pickle.load(f)
            
            self.model = state['model']
            self.state_means = state['state_means']
            self.state_covars = state['state_covars']
            self.state_mapping = state['state_mapping']
            self.is_fitted = state['is_fitted']
            
            logger.info(f"Loaded HMM model from {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load HMM model: {e}")
            return False


# =============================================================================
# GARCH VOLATILITY REGIME DETECTOR
# =============================================================================

class GARCHVolatilityDetector:
    """
    GARCH(1,1) based volatility regime detection.
    
    Volatility States:
    - LOW: Below 25th percentile of historical vol
    - NORMAL: 25th-75th percentile
    - HIGH: 75th-95th percentile
    - EXTREME: Above 95th percentile
    """
    
    VOL_PERCENTILES = {
        'low': 25,
        'normal': 75,
        'high': 95,
        'extreme': 100
    }
    
    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        model_path: str = 'state/garch_vol_model.pkl'
    ):
        self.p = p
        self.q = q
        self.model_path = Path(model_path)
        
        self.model: Optional[Any] = None
        self.is_fitted = False
        self.vol_history: List[float] = []
        
        # Historical percentiles for classification
        self.vol_percentile_25: float = 0.12
        self.vol_percentile_75: float = 0.20
        self.vol_percentile_95: float = 0.35
    
    def fit(self, returns: pd.Series) -> 'GARCHVolatilityDetector':
        """
        Fit GARCH model on returns data.
        
        Args:
            returns: Daily returns series (NOT percentage, decimal)
        """
        if not ARCH_AVAILABLE:
            logger.warning("ARCH not available, using fallback volatility detection")
            self.is_fitted = True
            self._set_percentiles_from_returns(returns)
            return self
        
        logger.info("🔄 Fitting GARCH(1,1) volatility model...")
        
        # Scale returns to percentage for numerical stability
        returns_pct = returns.dropna() * 100
        
        if len(returns_pct) < 100:
            raise ValueError(f"Insufficient data for GARCH: {len(returns_pct)} samples")
        
        # Fit GARCH model
        self.model = arch_model(
            returns_pct,
            mean='Zero',
            vol='GARCH',
            p=self.p,
            q=self.q
        )
        
        self.result = self.model.fit(disp='off')
        
        # Store historical volatility percentiles
        self._set_percentiles_from_returns(returns)
        
        self.is_fitted = True
        logger.info("✅ GARCH model fitted")
        
        return self
    
    def _set_percentiles_from_returns(self, returns: pd.Series):
        """Calculate volatility percentiles from returns."""
        realized_vol = returns.rolling(20).std() * np.sqrt(252)
        realized_vol = realized_vol.dropna()
        
        if len(realized_vol) > 0:
            self.vol_percentile_25 = np.percentile(realized_vol, 25)
            self.vol_percentile_75 = np.percentile(realized_vol, 75)
            self.vol_percentile_95 = np.percentile(realized_vol, 95)
    
    def predict(
        self,
        returns: pd.Series,
        horizon: int = 5
    ) -> Tuple[VolatilityRegime, float, float]:
        """
        Predict volatility regime and forecast.
        
        Args:
            returns: Recent returns series
            horizon: Forecast horizon in days
            
        Returns:
            Tuple of (regime, current_vol, forecast_vol)
        """
        if not self.is_fitted:
            return VolatilityRegime.NORMAL, 0.15, 0.15
        
        if not ARCH_AVAILABLE or self.model is None:
            return self._fallback_predict(returns)
        
        try:
            # Scale returns
            returns_pct = returns.dropna() * 100
            
            # Refit on recent data
            model = arch_model(
                returns_pct,
                mean='Zero',
                vol='GARCH',
                p=self.p,
                q=self.q
            )
            result = model.fit(disp='off', last_obs=len(returns_pct))
            
            # Get forecast
            forecast = result.forecast(horizon=horizon)
            
            # Current volatility (annualized)
            current_vol = np.sqrt(result.conditional_volatility.iloc[-1] ** 2 * 252) / 100
            
            # Forecast volatility
            forecast_variance = forecast.variance.values[-1, :]
            forecast_vol_1d = np.sqrt(forecast_variance[0] * 252) / 100
            forecast_vol_5d = np.sqrt(np.mean(forecast_variance) * 252) / 100
            
            # Classify regime
            regime = self._classify_volatility(current_vol)
            
            # Update history
            self.vol_history.append(current_vol)
            if len(self.vol_history) > 252:
                self.vol_history = self.vol_history[-252:]
            
            return regime, current_vol, forecast_vol_5d
            
        except Exception as e:
            logger.warning(f"GARCH prediction error: {e}")
            return self._fallback_predict(returns)
    
    def _fallback_predict(self, returns: pd.Series) -> Tuple[VolatilityRegime, float, float]:
        """Fallback volatility prediction."""
        if len(returns) < 20:
            return VolatilityRegime.NORMAL, 0.15, 0.15
        
        # Simple realized volatility
        current_vol = returns.iloc[-20:].std() * np.sqrt(252)
        
        # Simple exponential smoothing forecast
        ewm_vol = returns.ewm(span=20).std().iloc[-1] * np.sqrt(252)
        
        regime = self._classify_volatility(current_vol)
        
        return regime, current_vol, ewm_vol
    
    def _classify_volatility(self, vol: float) -> VolatilityRegime:
        """Classify volatility into regime."""
        if vol < self.vol_percentile_25:
            return VolatilityRegime.LOW
        elif vol < self.vol_percentile_75:
            return VolatilityRegime.NORMAL
        elif vol < self.vol_percentile_95:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.EXTREME
    
    def get_vol_forecast(self, returns: pd.Series, horizon: int = 5) -> np.ndarray:
        """Get volatility forecast path."""
        if not ARCH_AVAILABLE or self.model is None:
            # Simple persistence model
            current_vol = returns.iloc[-20:].std() * np.sqrt(252)
            return np.full(horizon, current_vol)
        
        try:
            returns_pct = returns.dropna() * 100
            model = arch_model(returns_pct, mean='Zero', vol='GARCH', p=1, q=1)
            result = model.fit(disp='off')
            forecast = result.forecast(horizon=horizon)
            
            vol_path = np.sqrt(forecast.variance.values[-1, :] * 252) / 100
            return vol_path
        except Exception:
            current_vol = returns.iloc[-20:].std() * np.sqrt(252)
            return np.full(horizon, current_vol)


# =============================================================================
# ADAPTIVE STRATEGY ROUTER
# =============================================================================

class AdaptiveStrategyRouter:
    """
    Route trading signals to appropriate strategies based on regime.
    
    Implements adaptive strategy switching with smooth transitions.
    """
    
    REGIME_STRATEGY_WEIGHTS = {
        MarketRegime.BULL: {
            'momentum': 0.5,
            'trend_follow': 0.3,
            'mean_reversion': 0.1,
            'defensive': 0.1
        },
        MarketRegime.BEAR: {
            'momentum': 0.1,
            'trend_follow': 0.2,
            'mean_reversion': 0.2,
            'defensive': 0.5
        },
        MarketRegime.SIDEWAYS: {
            'momentum': 0.1,
            'trend_follow': 0.1,
            'mean_reversion': 0.6,
            'defensive': 0.2
        },
        MarketRegime.CRISIS: {
            'momentum': 0.0,
            'trend_follow': 0.0,
            'mean_reversion': 0.1,
            'defensive': 0.9
        }
    }
    
    VOL_ADJUSTMENTS = {
        VolatilityRegime.LOW: {'scale': 1.2, 'size_mult': 1.3},
        VolatilityRegime.NORMAL: {'scale': 1.0, 'size_mult': 1.0},
        VolatilityRegime.HIGH: {'scale': 0.7, 'size_mult': 0.7},
        VolatilityRegime.EXTREME: {'scale': 0.3, 'size_mult': 0.3}
    }
    
    def __init__(self, transition_smoothing: float = 0.3):
        self.transition_smoothing = transition_smoothing
        self.current_weights: Dict[str, float] = {}
        self.weight_history: List[Dict[str, float]] = []
    
    def get_strategy_weights(
        self,
        regime: MarketRegime,
        volatility: VolatilityRegime,
        confidence: float = 1.0
    ) -> Dict[str, float]:
        """
        Get strategy weights for current regime.
        
        Uses exponential smoothing for regime transitions.
        """
        # Get target weights for regime
        target_weights = self.REGIME_STRATEGY_WEIGHTS.get(
            regime,
            {'defensive': 1.0}
        ).copy()
        
        # Apply volatility adjustments
        vol_adj = self.VOL_ADJUSTMENTS.get(volatility, {'scale': 1.0, 'size_mult': 1.0})
        
        # Scale non-defensive strategies
        for strategy in target_weights:
            if strategy != 'defensive':
                target_weights[strategy] *= vol_adj['scale']
        
        # Normalize
        total = sum(target_weights.values())
        if total > 0:
            target_weights = {k: v / total for k, v in target_weights.items()}
        
        # Apply confidence weighting
        if confidence < 0.6:
            # Low confidence -> more defensive
            defensive_boost = (0.6 - confidence) * 0.5
            target_weights['defensive'] = min(1.0, target_weights.get('defensive', 0) + defensive_boost)
            
            # Renormalize
            total = sum(target_weights.values())
            target_weights = {k: v / total for k, v in target_weights.items()}
        
        # Smooth transition from current weights
        if self.current_weights:
            smoothed = {}
            alpha = self.transition_smoothing
            
            all_strategies = set(target_weights.keys()) | set(self.current_weights.keys())
            for strategy in all_strategies:
                current = self.current_weights.get(strategy, 0)
                target = target_weights.get(strategy, 0)
                smoothed[strategy] = alpha * target + (1 - alpha) * current
            
            target_weights = smoothed
        
        # Update current weights
        self.current_weights = target_weights
        self.weight_history.append(target_weights.copy())
        
        if len(self.weight_history) > 252:
            self.weight_history = self.weight_history[-252:]
        
        return target_weights
    
    def get_position_size_multiplier(self, volatility: VolatilityRegime) -> float:
        """Get position size multiplier based on volatility regime."""
        return self.VOL_ADJUSTMENTS.get(volatility, {'size_mult': 1.0})['size_mult']


# =============================================================================
# COMBINED REGIME DETECTOR
# =============================================================================

class V28RegimeDetector:
    """
    Combined regime detection system.
    
    Integrates HMM and GARCH for robust regime classification.
    """
    
    def __init__(
        self,
        hmm_model_path: str = 'state/v28_hmm_model.pkl',
        garch_model_path: str = 'state/v28_garch_model.pkl'
    ):
        self.hmm_detector = HMMRegimeDetector(model_path=hmm_model_path)
        self.garch_detector = GARCHVolatilityDetector(model_path=garch_model_path)
        self.strategy_router = AdaptiveStrategyRouter()
        self.feature_builder = RegimeFeatureBuilder()
        
        self.is_fitted = False
        self.current_state: Optional[RegimeState] = None
        self.state_history: List[RegimeState] = []
    
    def fit(self, df: pd.DataFrame) -> 'V28RegimeDetector':
        """
        Fit all regime detection models.
        
        Args:
            df: DataFrame with OHLCV data (typically SPY)
        """
        logger.info("🔄 Fitting V28 regime detection system...")
        
        # Ensure we have required columns
        df = self.feature_builder.calculate_features(df)
        
        # Fit HMM
        self.hmm_detector.fit(df)
        
        # Fit GARCH
        returns = df['close'].pct_change().dropna()
        self.garch_detector.fit(returns)
        
        self.is_fitted = True
        logger.info("✅ V28 regime detection system fitted")
        
        return self
    
    def detect(self, df: pd.DataFrame) -> RegimeState:
        """
        Detect current market regime.
        
        Args:
            df: Recent OHLCV data
            
        Returns:
            RegimeState with full regime information
        """
        if not self.is_fitted:
            self.fit(df)
        
        # Calculate features
        df = self.feature_builder.calculate_features(df)
        
        # Get HMM prediction
        hmm_state, hmm_prob = self.hmm_detector.predict(df)
        
        # Get GARCH volatility prediction
        returns = df['close'].pct_change().dropna()
        vol_regime, current_vol, forecast_vol = self.garch_detector.predict(returns)
        
        # Map HMM state to market regime
        market_regime = self._hmm_to_market_regime(hmm_state, vol_regime)
        
        # Get strategy allocation
        strategy_weights = self.strategy_router.get_strategy_weights(
            market_regime, vol_regime, hmm_prob
        )
        
        # Calculate trend and momentum scores
        trend_strength = df['price_vs_sma50'].iloc[-1] if 'price_vs_sma50' in df.columns else 0
        momentum_score = (df['rsi_14'].iloc[-1] - 50) / 50 if 'rsi_14' in df.columns else 0
        
        # Create regime state
        state = RegimeState(
            market_regime=market_regime,
            volatility_regime=vol_regime,
            hmm_state=hmm_state,
            hmm_state_name=self.hmm_detector.get_state_name(hmm_state),
            hmm_probability=float(hmm_prob),
            garch_volatility=float(current_vol),
            garch_forecast_1d=float(forecast_vol),
            garch_forecast_5d=float(forecast_vol),
            regime_confidence=float(hmm_prob),
            trend_strength=float(trend_strength),
            momentum_score=float(momentum_score),
            transition_probability=self.hmm_detector.get_transition_probability(),
            recommended_strategies=strategy_weights
        )
        
        # Update history
        self.current_state = state
        self.state_history.append(state)
        
        if len(self.state_history) > 252:
            self.state_history = self.state_history[-252:]
        
        return state
    
    def _hmm_to_market_regime(
        self,
        hmm_state: int,
        vol_regime: VolatilityRegime
    ) -> MarketRegime:
        """Convert HMM state to market regime."""
        if hmm_state == 3:  # Crisis
            return MarketRegime.CRISIS
        elif hmm_state == 0:  # LowVolTrend
            return MarketRegime.BULL
        elif hmm_state == 1:  # HighVolTrend
            if vol_regime in [VolatilityRegime.HIGH, VolatilityRegime.EXTREME]:
                return MarketRegime.BEAR
            return MarketRegime.BULL
        else:  # LowVolMeanRevert
            return MarketRegime.SIDEWAYS
    
    def get_position_size_multiplier(self) -> float:
        """Get position size multiplier based on current regime."""
        if self.current_state is None:
            return 1.0
        return self.strategy_router.get_position_size_multiplier(
            self.current_state.volatility_regime
        )
    
    def save(self):
        """Save models to disk."""
        self.hmm_detector.save()
    
    def load(self) -> bool:
        """Load models from disk."""
        return self.hmm_detector.load()


# =============================================================================
# MAIN / DEMO
# =============================================================================

def demo():
    """Demo the regime detection system."""
    import yfinance as yf
    
    logger.info("📊 V28 Regime Detection Demo")
    logger.info("=" * 50)
    
    # Load SPY data
    try:
        spy = yf.download('SPY', period='2y', progress=False)
        spy = spy.reset_index()
        spy.columns = ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
    except Exception as e:
        logger.error(f"Failed to download data: {e}")
        # Create synthetic data
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=500, freq='D')
        close = 400 * np.cumprod(1 + np.random.randn(500) * 0.015)
        spy = pd.DataFrame({
            'date': dates,
            'open': close * 0.99,
            'high': close * 1.01,
            'low': close * 0.98,
            'close': close,
            'volume': np.random.randint(50000000, 100000000, 500)
        })
    
    # Initialize detector
    detector = V28RegimeDetector()
    
    # Fit on historical data
    detector.fit(spy)
    
    # Detect current regime
    current_state = detector.detect(spy)
    
    # Print results
    logger.info(f"\n📈 Current Market Regime:")
    logger.info(f"   Market Regime: {current_state.market_regime.value.upper()}")
    logger.info(f"   Volatility Regime: {current_state.volatility_regime.value}")
    logger.info(f"   HMM State: {current_state.hmm_state_name} (prob: {current_state.hmm_probability:.2%})")
    logger.info(f"   GARCH Vol: {current_state.garch_volatility:.2%} (forecast: {current_state.garch_forecast_5d:.2%})")
    logger.info(f"   Trend Strength: {current_state.trend_strength:.3f}")
    logger.info(f"   Momentum Score: {current_state.momentum_score:.3f}")
    logger.info(f"   Transition Probability: {current_state.transition_probability:.2%}")
    
    logger.info(f"\n📊 Recommended Strategy Allocation:")
    for strategy, weight in sorted(current_state.recommended_strategies.items(), key=lambda x: -x[1]):
        if weight > 0.01:
            logger.info(f"   {strategy}: {weight:.1%}")
    
    logger.info(f"\n💹 Position Size Multiplier: {detector.get_position_size_multiplier():.2f}")


if __name__ == '__main__':
    demo()
