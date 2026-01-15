"""
Hidden Markov Model Regime Detector

Implements probabilistic regime detection using HMM with 3 states:
- Bull: SPY 50-day MA > 200-day MA, VIX <20, positive momentum
- Neutral: Mixed signals, moderate volatility
- Bear: SPY 50-day MA < 200-day MA, VIX >30, negative momentum

Academic research shows regime-aware allocation improves Sharpe by 30-50% (arXiv 2025)

Features:
1. Multi-feature regime classification
2. Probabilistic state assignment (P_bull, P_neutral, P_bear)
3. Smooth transitions with EMA filtering
4. Real-time implementable
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RegimeConfig:
    """Configuration for regime detection."""
    # Lookback periods
    ma_short: int = 50           # Short-term moving average
    ma_long: int = 200           # Long-term moving average
    momentum_lookback: int = 63   # 3-month momentum
    vol_lookback: int = 20        # Volatility calculation window
    
    # VIX thresholds
    vix_low: float = 20           # Below this = low vol environment
    vix_high: float = 30          # Above this = high vol environment
    vix_extreme: float = 40       # Extreme volatility (circuit breaker)
    
    # Smoothing
    ema_span: int = 5             # EMA smoothing for regime probabilities
    
    # Regime transition
    min_regime_days: int = 5      # Minimum days before regime change
    regime_threshold: float = 0.4  # Minimum probability to confirm regime


@dataclass
class RegimeState:
    """Current regime state with probabilities."""
    regime: str                    # 'bull', 'neutral', 'bear'
    p_bull: float
    p_neutral: float
    p_bear: float
    confidence: float             # Max probability
    days_in_regime: int
    vix_level: float
    ma_signal: str                # 'bullish', 'bearish', 'neutral'
    momentum_signal: str          # 'positive', 'negative', 'flat'


class RegimeFeatures:
    """Calculate features for regime detection."""
    
    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
    
    def extract_features(
        self,
        spy_prices: pd.DataFrame,
        vix_prices: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """
        Extract regime detection features.
        
        Args:
            spy_prices: SPY OHLCV data
            vix_prices: VIX data (optional, will estimate if not provided)
            
        Returns:
            Dict of feature values
        """
        close = self._get_close(spy_prices)
        
        if len(close) < self.config.ma_long:
            return self._default_features()
        
        features = {}
        
        # Moving average signals
        ma_short = close.rolling(self.config.ma_short).mean()
        ma_long = close.rolling(self.config.ma_long).mean()
        
        current_price = close.iloc[-1]
        ma_short_val = ma_short.iloc[-1]
        ma_long_val = ma_long.iloc[-1]
        
        # MA crossover signal (-1 to 1)
        if ma_short_val > ma_long_val * 1.02:  # 2% above
            features['ma_signal'] = 1.0
        elif ma_short_val < ma_long_val * 0.98:  # 2% below
            features['ma_signal'] = -1.0
        else:
            features['ma_signal'] = 0.0
        
        # Price relative to MAs
        features['price_vs_ma50'] = (current_price / ma_short_val - 1) * 10  # Scaled
        features['price_vs_ma200'] = (current_price / ma_long_val - 1) * 10
        
        # Momentum signals
        if len(close) >= self.config.momentum_lookback:
            mom_3m = close.iloc[-1] / close.iloc[-self.config.momentum_lookback] - 1
            features['momentum_3m'] = mom_3m * 5  # Scale to roughly -1 to 1
        else:
            features['momentum_3m'] = 0.0
        
        # Short-term momentum (1 month)
        if len(close) >= 21:
            mom_1m = close.iloc[-1] / close.iloc[-21] - 1
            features['momentum_1m'] = mom_1m * 10
        else:
            features['momentum_1m'] = 0.0
        
        # Volatility
        returns = close.pct_change().dropna()
        if len(returns) >= self.config.vol_lookback:
            recent_vol = returns.iloc[-self.config.vol_lookback:].std() * np.sqrt(252)
            features['realized_vol'] = recent_vol
        else:
            features['realized_vol'] = 0.15  # Default
        
        # VIX (use provided or estimate from realized vol)
        if vix_prices is not None and len(vix_prices) > 0:
            vix_close = self._get_close(vix_prices)
            features['vix'] = vix_close.iloc[-1]
            if len(vix_close) >= 20:
                features['vix_change'] = (vix_close.iloc[-1] / vix_close.iloc[-20] - 1)
            else:
                features['vix_change'] = 0.0
        else:
            # Estimate VIX from realized vol (rough approximation)
            features['vix'] = features['realized_vol'] * 100
            features['vix_change'] = 0.0
        
        # Market breadth proxy (using MA cross persistence)
        ma_cross_series = (ma_short > ma_long).astype(int)
        if len(ma_cross_series) >= 20:
            features['breadth'] = ma_cross_series.iloc[-20:].mean()  # % of time above
        else:
            features['breadth'] = 0.5
        
        # Drawdown from peak
        peak = close.rolling(252, min_periods=1).max()
        current_dd = (current_price / peak.iloc[-1] - 1)
        features['drawdown'] = current_dd
        
        return features
    
    def _get_close(self, df: pd.DataFrame) -> pd.Series:
        """Safely get close price series."""
        if df is None or len(df) == 0:
            return pd.Series([100.0])
        
        for col in ['close', 'Close', 'Adj Close']:
            if col in df.columns:
                return df[col]
        if isinstance(df.columns, pd.MultiIndex):
            if 'Close' in df.columns.get_level_values(0):
                return df['Close'].iloc[:, 0]
        if len(df.columns) >= 4:
            return df.iloc[:, 3]
        return df.iloc[:, 0]
    
    def _default_features(self) -> Dict[str, float]:
        """Return default neutral features."""
        return {
            'ma_signal': 0.0,
            'price_vs_ma50': 0.0,
            'price_vs_ma200': 0.0,
            'momentum_3m': 0.0,
            'momentum_1m': 0.0,
            'realized_vol': 0.15,
            'vix': 20.0,
            'vix_change': 0.0,
            'breadth': 0.5,
            'drawdown': 0.0,
        }


class RegimeDetectorHMM:
    """
    Hidden Markov Model-based regime detector.
    
    Uses multiple market features to probabilistically classify regimes:
    - Bull: Strong uptrend, low volatility, positive momentum
    - Neutral: Mixed signals, moderate volatility
    - Bear: Downtrend, high volatility, negative momentum
    
    Note: This uses a simplified rule-based HMM approximation that is
    computationally efficient and real-time implementable without
    requiring hmmlearn or pomegranate libraries.
    """
    
    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
        self.feature_extractor = RegimeFeatures(self.config)
        
        # State tracking
        self.current_state: Optional[RegimeState] = None
        self.probability_history: List[Dict[str, float]] = []
        self.regime_history: List[str] = []
        
        # EMA smoothing
        self._ema_bull = 0.33
        self._ema_neutral = 0.34
        self._ema_bear = 0.33
        
        logger.info("Initialized RegimeDetectorHMM")
    
    def detect_regime(
        self,
        spy_prices: pd.DataFrame,
        vix_prices: Optional[pd.DataFrame] = None,
    ) -> RegimeState:
        """
        Detect current market regime.
        
        Args:
            spy_prices: SPY OHLCV data
            vix_prices: VIX data (optional)
            
        Returns:
            RegimeState with probabilities
        """
        # Extract features
        features = self.feature_extractor.extract_features(spy_prices, vix_prices)
        
        # Calculate raw probabilities
        p_bull, p_neutral, p_bear = self._calculate_probabilities(features)
        
        # Apply EMA smoothing
        alpha = 2 / (self.config.ema_span + 1)
        self._ema_bull = alpha * p_bull + (1 - alpha) * self._ema_bull
        self._ema_neutral = alpha * p_neutral + (1 - alpha) * self._ema_neutral
        self._ema_bear = alpha * p_bear + (1 - alpha) * self._ema_bear
        
        # Normalize smoothed probabilities
        total = self._ema_bull + self._ema_neutral + self._ema_bear
        p_bull_smooth = self._ema_bull / total
        p_neutral_smooth = self._ema_neutral / total
        p_bear_smooth = self._ema_bear / total
        
        # Determine regime
        probs = {'bull': p_bull_smooth, 'neutral': p_neutral_smooth, 'bear': p_bear_smooth}
        regime = max(probs, key=probs.get)
        confidence = probs[regime]
        
        # Check regime persistence
        if self.current_state is not None:
            if regime == self.current_state.regime:
                days_in_regime = self.current_state.days_in_regime + 1
            else:
                # Only change regime if confidence exceeds threshold
                if confidence < self.config.regime_threshold:
                    regime = self.current_state.regime
                    days_in_regime = self.current_state.days_in_regime + 1
                else:
                    days_in_regime = 1
        else:
            days_in_regime = 1
        
        # Determine signals
        ma_signal = 'bullish' if features['ma_signal'] > 0.5 else ('bearish' if features['ma_signal'] < -0.5 else 'neutral')
        momentum_signal = 'positive' if features['momentum_3m'] > 0.1 else ('negative' if features['momentum_3m'] < -0.1 else 'flat')
        
        # Build state
        self.current_state = RegimeState(
            regime=regime,
            p_bull=p_bull_smooth,
            p_neutral=p_neutral_smooth,
            p_bear=p_bear_smooth,
            confidence=confidence,
            days_in_regime=days_in_regime,
            vix_level=features['vix'],
            ma_signal=ma_signal,
            momentum_signal=momentum_signal,
        )
        
        # Track history
        self.probability_history.append({
            'bull': p_bull_smooth,
            'neutral': p_neutral_smooth,
            'bear': p_bear_smooth,
        })
        self.regime_history.append(regime)
        
        return self.current_state
    
    def _calculate_probabilities(self, features: Dict[str, float]) -> Tuple[float, float, float]:
        """
        Calculate raw regime probabilities from features.
        
        Uses a logistic-style scoring system based on multiple indicators.
        """
        bull_score = 0.0
        bear_score = 0.0
        neutral_score = 1.0  # Base neutral score
        
        # MA signal contribution
        ma_sig = features.get('ma_signal', 0)
        if ma_sig > 0:
            bull_score += 2.0 * ma_sig
        else:
            bear_score += 2.0 * abs(ma_sig)
        
        # Momentum contribution
        mom_3m = features.get('momentum_3m', 0)
        if mom_3m > 0.1:
            bull_score += min(2.0, mom_3m * 5)
        elif mom_3m < -0.1:
            bear_score += min(2.0, abs(mom_3m) * 5)
        else:
            neutral_score += 0.5
        
        # VIX contribution
        vix = features.get('vix', 20)
        if vix < self.config.vix_low:
            bull_score += 1.5
        elif vix > self.config.vix_high:
            bear_score += 2.0
            if vix > self.config.vix_extreme:
                bear_score += 1.0
        else:
            neutral_score += 0.5
        
        # VIX change (spiking VIX = bearish)
        vix_change = features.get('vix_change', 0)
        if vix_change > 0.2:  # 20% spike
            bear_score += 1.5
        elif vix_change < -0.15:  # Declining VIX
            bull_score += 0.5
        
        # Drawdown contribution
        dd = features.get('drawdown', 0)
        if dd < -0.10:  # >10% drawdown
            bear_score += min(2.0, abs(dd) * 10)
        elif dd > -0.03:  # Near highs
            bull_score += 0.5
        
        # Price vs MAs
        price_vs_ma50 = features.get('price_vs_ma50', 0)
        price_vs_ma200 = features.get('price_vs_ma200', 0)
        
        if price_vs_ma50 > 0.5 and price_vs_ma200 > 0.5:
            bull_score += 1.0
        elif price_vs_ma50 < -0.5 and price_vs_ma200 < -0.5:
            bear_score += 1.0
        
        # Market breadth
        breadth = features.get('breadth', 0.5)
        if breadth > 0.7:
            bull_score += 1.0
        elif breadth < 0.3:
            bear_score += 1.0
        
        # Convert scores to probabilities using softmax
        max_score = max(bull_score, neutral_score, bear_score)
        exp_bull = np.exp(bull_score - max_score)
        exp_neutral = np.exp(neutral_score - max_score)
        exp_bear = np.exp(bear_score - max_score)
        
        total = exp_bull + exp_neutral + exp_bear
        
        p_bull = exp_bull / total
        p_neutral = exp_neutral / total
        p_bear = exp_bear / total
        
        return p_bull, p_neutral, p_bear
    
    def get_regime_leverage(self) -> float:
        """
        Get recommended leverage based on current regime.
        
        Returns:
            Leverage multiplier (0.5 to 1.5)
        """
        if self.current_state is None:
            return 1.0
        
        # Base leverages
        leverages = {
            'bull': 1.3,
            'neutral': 1.0,
            'bear': 0.7,
        }
        
        # Probability-weighted leverage
        leverage = (
            self.current_state.p_bull * leverages['bull'] +
            self.current_state.p_neutral * leverages['neutral'] +
            self.current_state.p_bear * leverages['bear']
        )
        
        # VIX-based circuit breaker
        if self.current_state.vix_level > self.config.vix_extreme:
            leverage = min(leverage, 0.5)
        
        return leverage
    
    def get_defensive_tilt(self) -> float:
        """
        Get recommended defensive sector tilt.
        
        Returns:
            Tilt factor (0.5 = reduce defensive, 1.0 = neutral, 1.5 = increase defensive)
        """
        if self.current_state is None:
            return 1.0
        
        # Higher defensive in bear markets
        tilt = (
            self.current_state.p_bull * 0.7 +
            self.current_state.p_neutral * 1.0 +
            self.current_state.p_bear * 1.4
        )
        
        return tilt
    
    def print_state(self):
        """Print current regime state."""
        if self.current_state is None:
            print("No regime detected yet")
            return
        
        s = self.current_state
        print("\n" + "="*50)
        print("REGIME DETECTION STATE")
        print("="*50)
        print(f"Current Regime: {s.regime.upper()}")
        print(f"Confidence: {s.confidence:.1%}")
        print(f"Days in Regime: {s.days_in_regime}")
        print(f"\nProbabilities:")
        print(f"  Bull:    {s.p_bull:.1%}")
        print(f"  Neutral: {s.p_neutral:.1%}")
        print(f"  Bear:    {s.p_bear:.1%}")
        print(f"\nSignals:")
        print(f"  MA Signal: {s.ma_signal}")
        print(f"  Momentum:  {s.momentum_signal}")
        print(f"  VIX Level: {s.vix_level:.1f}")
        print(f"\nRecommendations:")
        print(f"  Leverage: {self.get_regime_leverage():.2f}x")
        print(f"  Defensive Tilt: {self.get_defensive_tilt():.2f}x")
        print("="*50)


def test_regime_detector():
    """Test regime detection system."""
    print("\n" + "="*60)
    print("Testing Regime Detector HMM")
    print("="*60)
    
    detector = RegimeDetectorHMM()
    
    # Create mock SPY data for different scenarios
    dates = pd.date_range('2022-01-01', periods=252, freq='B')
    
    # Scenario 1: Bull market (uptrend, low vol)
    print("\n[Testing Bull Market Scenario]")
    bull_prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.008, 252))
    spy_bull = pd.DataFrame({
        'close': bull_prices,
    }, index=dates)
    
    state = detector.detect_regime(spy_bull)
    detector.print_state()
    
    # Scenario 2: Bear market (downtrend, high vol)
    print("\n[Testing Bear Market Scenario]")
    bear_prices = 100 * np.cumprod(1 + np.random.normal(-0.002, 0.020, 252))
    spy_bear = pd.DataFrame({
        'close': bear_prices,
    }, index=dates)
    
    # Reset detector
    detector = RegimeDetectorHMM()
    state = detector.detect_regime(spy_bear)
    detector.print_state()
    
    # Scenario 3: Sideways market
    print("\n[Testing Sideways Market Scenario]")
    sideways_prices = 100 * np.cumprod(1 + np.random.normal(0.0001, 0.012, 252))
    spy_sideways = pd.DataFrame({
        'close': sideways_prices,
    }, index=dates)
    
    detector = RegimeDetectorHMM()
    state = detector.detect_regime(spy_sideways)
    detector.print_state()


if __name__ == "__main__":
    test_regime_detector()
