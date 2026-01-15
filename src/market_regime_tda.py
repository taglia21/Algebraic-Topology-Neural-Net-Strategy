"""Market Regime Detection using TDA Features.

Phase 5: Uses topological features to detect market regimes.
Combines Betti numbers, persistence metrics, and turbulence to classify:
- BULL: Low turbulence, stable topology, positive momentum
- BEAR: High fragmentation, unstable topology, negative momentum
- TRANSITION: Rapidly changing topology, uncertain direction
- CRISIS: Extreme turbulence, regime breakdown
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications."""
    BULL = "BULL"           # Strong risk-on, buy equities
    BEAR = "BEAR"           # Strong risk-off, reduce exposure
    TRANSITION = "TRANSITION"  # Uncertain, hedge or stay neutral
    CRISIS = "CRISIS"       # Extreme stress, capital preservation
    RECOVERY = "RECOVERY"   # Coming out of crisis, selective buying


@dataclass
class RegimeSignal:
    """Regime detection signal with confidence."""
    regime: MarketRegime
    confidence: float       # 0-100 confidence
    recommended_exposure: float  # 0-1 equity exposure
    risk_budget: float      # 0-1 risk budget
    description: str        # Human-readable description


@dataclass
class TDARegimeFeatures:
    """Features used for regime detection."""
    # Current topology
    betti_0: int
    betti_1: int
    fragmentation: float
    stability: float
    entropy: float
    
    # Trend in topology
    betti_0_trend: float    # 5-day slope
    persistence_trend: float
    
    # Market indicators
    turbulence: float
    cross_correlation: float  # Average pairwise correlation
    
    # Derived signals
    topology_score: float   # Combined topology health score


class MarketRegimeDetector:
    """
    Detects market regimes using TDA features.
    
    The detector uses multiple TDA signals:
    1. Betti numbers: β₀ high = fragmented, β₁ high = cyclic relationships
    2. Persistence: Long-lived features = stable structure
    3. Turbulence: Rate of topology change
    4. Cross-correlation: Market-wide correlation level
    
    Regime Classification Logic:
    - BULL: Low fragmentation, stable persistence, low turbulence
    - BEAR: High fragmentation, declining persistence, moderate turbulence
    - TRANSITION: Rapidly changing Betti numbers, unstable
    - CRISIS: Extreme turbulence, correlation breakdown
    - RECOVERY: Declining turbulence, rebuilding structure
    """
    
    def __init__(
        self,
        lookback: int = 20,
        turbulence_threshold: float = 60.0,
        crisis_threshold: float = 80.0,
        fragmentation_threshold: float = 0.5,  # Increased from 0.3 - more tolerant
    ):
        """
        Initialize regime detector.
        
        Args:
            lookback: Days of history for trend calculation
            turbulence_threshold: Turbulence level triggering RISK_OFF
            crisis_threshold: Turbulence level triggering CRISIS
            fragmentation_threshold: Fragmentation level for concern
        """
        self.lookback = lookback
        self.turbulence_threshold = turbulence_threshold
        self.crisis_threshold = crisis_threshold
        self.fragmentation_threshold = fragmentation_threshold
        
        # History for regime persistence
        self.regime_history: List[MarketRegime] = []
        self.feature_history: List[TDARegimeFeatures] = []
    
    def extract_regime_features(
        self,
        tda_features_list: List[Dict],
        market_returns: pd.Series = None,
    ) -> TDARegimeFeatures:
        """
        Extract regime-relevant features from TDA analysis.
        
        Args:
            tda_features_list: List of daily TDA features
            market_returns: Optional market returns for context
            
        Returns:
            TDARegimeFeatures for regime detection
        """
        if len(tda_features_list) == 0:
            return self._default_features()
        
        current = tda_features_list[-1]
        
        # Current topology metrics
        betti_0 = current.get('betti_0', 1)
        betti_1 = current.get('betti_1', 0)
        fragmentation = current.get('fragmentation', 0.0)
        stability = current.get('stability', 1.0)
        entropy = current.get('entropy_h0', 0.0) + current.get('entropy_h1', 0.0)
        
        # Trend calculations (5-day slope)
        if len(tda_features_list) >= 5:
            recent = tda_features_list[-5:]
            betti_0_values = [f.get('betti_0', 1) for f in recent]
            persistence_values = [
                f.get('total_persistence_h0', 0) + f.get('total_persistence_h1', 0) 
                for f in recent
            ]
            
            # Linear regression slope
            betti_0_trend = np.polyfit(range(5), betti_0_values, 1)[0]
            persistence_trend = np.polyfit(range(5), persistence_values, 1)[0]
        else:
            betti_0_trend = 0.0
            persistence_trend = 0.0
        
        # Turbulence
        turbulence = current.get('turbulence_index', 50.0)
        
        # Cross-correlation (from raw features if available)
        cross_correlation = current.get('avg_correlation', 0.3)
        
        # Topology health score
        # Higher is better: low fragmentation, high stability, low entropy
        topology_score = (
            (1 - fragmentation) * 30 +
            min(stability, 2.0) * 20 +
            (1 - min(entropy / 5, 1)) * 25 +
            (1 - min(turbulence / 100, 1)) * 25
        )
        
        return TDARegimeFeatures(
            betti_0=betti_0,
            betti_1=betti_1,
            fragmentation=fragmentation,
            stability=stability,
            entropy=entropy,
            betti_0_trend=betti_0_trend,
            persistence_trend=persistence_trend,
            turbulence=turbulence,
            cross_correlation=cross_correlation,
            topology_score=topology_score,
        )
    
    def _default_features(self) -> TDARegimeFeatures:
        """Return default neutral features."""
        return TDARegimeFeatures(
            betti_0=1,
            betti_1=0,
            fragmentation=0.2,
            stability=1.0,
            entropy=1.0,
            betti_0_trend=0.0,
            persistence_trend=0.0,
            turbulence=50.0,
            cross_correlation=0.3,
            topology_score=50.0,
        )
    
    def detect_regime(
        self,
        features: TDARegimeFeatures,
        market_returns_5d: float = None,
        market_returns_20d: float = None,
    ) -> RegimeSignal:
        """
        Detect current market regime from TDA features.
        
        Args:
            features: Current TDA regime features
            market_returns_5d: 5-day market return (optional)
            market_returns_20d: 20-day market return (optional)
            
        Returns:
            RegimeSignal with regime and recommendations
        """
        # Store for history
        self.feature_history.append(features)
        if len(self.feature_history) > 100:
            self.feature_history = self.feature_history[-100:]
        
        # ========================================
        # RULE-BASED REGIME CLASSIFICATION
        # ========================================
        
        # CRISIS Detection (highest priority)
        if features.turbulence > self.crisis_threshold:
            regime = MarketRegime.CRISIS
            confidence = min(100, features.turbulence)
            exposure = 0.1  # Minimal exposure
            risk_budget = 0.1
            description = f"CRISIS: Extreme turbulence ({features.turbulence:.0f}), topology breakdown"
        
        # BEAR Detection - more specific criteria
        elif (features.turbulence > self.turbulence_threshold and 
              features.fragmentation > self.fragmentation_threshold):
            # Both high turbulence AND high fragmentation required
            regime = MarketRegime.BEAR
            confidence = 60 + features.turbulence * 0.4
            exposure = 0.3  # Reduced exposure
            risk_budget = 0.3
            description = f"BEAR: High turbulence ({features.turbulence:.0f}), fragmented market"
        
        # TRANSITION Detection
        elif (abs(features.betti_0_trend) > 5 or  # Increased from 2
              features.persistence_trend < -2):    # More tolerant
            regime = MarketRegime.TRANSITION
            confidence = 50 + abs(features.betti_0_trend) * 10
            exposure = 0.5  # Neutral exposure
            risk_budget = 0.4
            description = f"TRANSITION: Rapidly changing topology (β₀ trend: {features.betti_0_trend:.1f})"
        
        # RECOVERY Detection
        elif (len(self.regime_history) >= 3 and 
              self.regime_history[-1] in [MarketRegime.CRISIS, MarketRegime.BEAR] and
              features.turbulence < self.turbulence_threshold - 10 and
              features.persistence_trend > 0):
            regime = MarketRegime.RECOVERY
            confidence = 60 + (self.turbulence_threshold - features.turbulence)
            exposure = 0.7  # Increasing exposure
            risk_budget = 0.6
            description = f"RECOVERY: Turbulence declining ({features.turbulence:.0f}), rebuilding structure"
        
        # BULL Detection (default healthy market) - more inclusive
        elif (features.turbulence < 50 and  # Increased from 40
              features.topology_score > 40):  # Lowered from 60
            regime = MarketRegime.BULL
            confidence = features.topology_score
            exposure = 1.0  # Full exposure
            risk_budget = 1.0
            description = f"BULL: Healthy topology (score: {features.topology_score:.0f}), low turbulence"
        
        # NEUTRAL (between regimes)
        else:
            regime = MarketRegime.TRANSITION
            confidence = 50
            exposure = 0.6
            risk_budget = 0.5
            description = f"NEUTRAL: Mixed signals (turbulence: {features.turbulence:.0f})"
        
        # Apply regime persistence (avoid whipsaws)
        regime = self._apply_regime_persistence(regime, confidence)
        
        # Store regime
        self.regime_history.append(regime)
        if len(self.regime_history) > 100:
            self.regime_history = self.regime_history[-100:]
        
        return RegimeSignal(
            regime=regime,
            confidence=min(100, max(0, confidence)),
            recommended_exposure=exposure,
            risk_budget=risk_budget,
            description=description,
        )
    
    def _apply_regime_persistence(
        self,
        new_regime: MarketRegime,
        confidence: float
    ) -> MarketRegime:
        """
        Apply regime persistence to avoid whipsaws.
        
        Only change regime if:
        1. High confidence (>70)
        2. Consistent signals (3+ days)
        """
        if len(self.regime_history) < 3:
            return new_regime
        
        current_regime = self.regime_history[-1]
        
        # Always allow transitions to CRISIS
        if new_regime == MarketRegime.CRISIS:
            return new_regime
        
        # If low confidence, stay in current regime
        if confidence < 60:
            return current_regime
        
        # Check for regime consistency
        recent_regimes = self.regime_history[-3:]
        if all(r == current_regime for r in recent_regimes):
            # Strong current regime - need high confidence to change
            if confidence < 70:
                return current_regime
        
        return new_regime
    
    def get_position_sizing(
        self,
        regime_signal: RegimeSignal,
        base_position: float = 1.0,
    ) -> Dict[str, float]:
        """
        Get position sizing recommendations based on regime.
        
        Args:
            regime_signal: Current regime signal
            base_position: Base position size (1.0 = 100%)
            
        Returns:
            Dict with position sizing recommendations
        """
        exposure = regime_signal.recommended_exposure
        risk_budget = regime_signal.risk_budget
        
        return {
            'equity_allocation': base_position * exposure,
            'cash_allocation': base_position * (1 - exposure),
            'max_position_size': 0.10 * risk_budget,  # Max 10% per position
            'max_sector_exposure': 0.25 * risk_budget,  # Max 25% per sector
            'stop_loss_multiplier': 1.0 + (1 - risk_budget),  # Tighter stops in low-risk budget
        }
    
    def get_regime_stats(self) -> Dict:
        """Get regime detection statistics."""
        if len(self.regime_history) == 0:
            return {}
        
        from collections import Counter
        regime_counts = Counter([r.value for r in self.regime_history])
        
        return {
            'total_days': len(self.regime_history),
            'regime_distribution': dict(regime_counts),
            'current_regime': self.regime_history[-1].value if self.regime_history else None,
            'regime_changes': sum(
                1 for i in range(1, len(self.regime_history))
                if self.regime_history[i] != self.regime_history[i-1]
            ),
        }


class TDARegimeBacktester:
    """
    Backtest regime-based allocation strategies.
    """
    
    def __init__(
        self,
        detector: MarketRegimeDetector = None,
    ):
        """Initialize backtester."""
        self.detector = detector or MarketRegimeDetector()
        self.results = []
    
    def backtest_allocation(
        self,
        tda_features_series: List[Dict],
        market_returns: pd.Series,
        dates: List[str],
    ) -> pd.DataFrame:
        """
        Backtest regime-based allocation.
        
        Args:
            tda_features_series: Daily TDA features
            market_returns: Daily market returns
            dates: Date labels
            
        Returns:
            DataFrame with backtest results
        """
        results = []
        equity = 1.0
        
        for i, (date, features_dict, ret) in enumerate(zip(dates, tda_features_series, market_returns)):
            # Get historical features up to this point
            history = tda_features_series[:i+1]
            
            # Extract regime features
            regime_features = self.detector.extract_regime_features(history)
            
            # Detect regime
            signal = self.detector.detect_regime(regime_features)
            
            # Position sizing
            sizing = self.detector.get_position_sizing(signal)
            exposure = sizing['equity_allocation']
            
            # Calculate return
            portfolio_return = exposure * ret
            equity *= (1 + portfolio_return)
            
            results.append({
                'date': date,
                'regime': signal.regime.value,
                'confidence': signal.confidence,
                'exposure': exposure,
                'market_return': ret,
                'portfolio_return': portfolio_return,
                'equity': equity,
            })
        
        return pd.DataFrame(results)


if __name__ == "__main__":
    print("Testing Market Regime TDA Detector...")
    print("=" * 60)
    
    # Create synthetic TDA features simulating different regimes
    detector = MarketRegimeDetector()
    
    # Test 1: Bull market features
    print("\nTest 1: Bull Market Features")
    bull_features = TDARegimeFeatures(
        betti_0=5,
        betti_1=2,
        fragmentation=0.1,
        stability=1.5,
        entropy=0.5,
        betti_0_trend=0.0,
        persistence_trend=0.1,
        turbulence=25.0,
        cross_correlation=0.4,
        topology_score=75.0,
    )
    signal = detector.detect_regime(bull_features)
    print(f"  Regime: {signal.regime.value}")
    print(f"  Confidence: {signal.confidence:.0f}%")
    print(f"  Exposure: {signal.recommended_exposure:.0%}")
    print(f"  Description: {signal.description}")
    
    # Test 2: Crisis features
    print("\nTest 2: Crisis Features")
    crisis_features = TDARegimeFeatures(
        betti_0=50,
        betti_1=20,
        fragmentation=0.6,
        stability=0.3,
        entropy=3.0,
        betti_0_trend=5.0,
        persistence_trend=-1.0,
        turbulence=90.0,
        cross_correlation=0.8,
        topology_score=20.0,
    )
    signal = detector.detect_regime(crisis_features)
    print(f"  Regime: {signal.regime.value}")
    print(f"  Confidence: {signal.confidence:.0f}%")
    print(f"  Exposure: {signal.recommended_exposure:.0%}")
    print(f"  Description: {signal.description}")
    
    # Test 3: Transition features
    print("\nTest 3: Transition Features")
    transition_features = TDARegimeFeatures(
        betti_0=15,
        betti_1=5,
        fragmentation=0.25,
        stability=0.8,
        entropy=1.5,
        betti_0_trend=3.0,  # Rapidly changing
        persistence_trend=-0.3,
        turbulence=50.0,
        cross_correlation=0.5,
        topology_score=55.0,
    )
    signal = detector.detect_regime(transition_features)
    print(f"  Regime: {signal.regime.value}")
    print(f"  Confidence: {signal.confidence:.0f}%")
    print(f"  Exposure: {signal.recommended_exposure:.0%}")
    print(f"  Description: {signal.description}")
    
    # Test regime stats
    print("\nRegime Detection Stats:")
    stats = detector.get_regime_stats()
    print(f"  Total signals: {stats['total_days']}")
    print(f"  Distribution: {stats['regime_distribution']}")
    
    print("\nMarket Regime TDA Detector tests complete!")
