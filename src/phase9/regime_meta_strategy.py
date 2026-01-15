"""Hierarchical Regime Meta-Strategy for Phase 9.

Implements a 3-layer regime detection system:
Layer 1: Macro HMM - Detects Bull/Bear/High-Vol/Low-Vol using VIX, breadth
Layer 2: TDA Regime Correlation - Market topology analysis 
Layer 3: Dynamic Factor Allocation - Regime-specific factor weights

Target: Optimal strategy selection based on market conditions
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import logging
from collections import deque

logger = logging.getLogger(__name__)


class MacroRegime(Enum):
    """Primary macro regime states from HMM Layer 1."""
    BULL_MOMENTUM = "bull_momentum"
    BEAR_DEFENSIVE = "bear_defensive"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRANSITION = "transition"


class TDARegime(Enum):
    """TDA-derived regime states from Layer 2."""
    RISK_ON = "risk_on"         # Low fragmentation, stable topology
    RISK_OFF = "risk_off"       # High fragmentation, unstable topology  
    REGIME_BREAK = "regime_break"  # Topological bifurcation detected
    CONSOLIDATION = "consolidation"  # Low persistence, ranging market


@dataclass
class RegimeMeta:
    """Complete regime metadata across all layers."""
    # Layer 1: Macro
    macro_regime: MacroRegime
    macro_confidence: float
    macro_transition_prob: float
    
    # Layer 2: TDA
    tda_regime: TDARegime
    tda_turbulence: float  # 0-100
    tda_fragmentation: float  # Beta_0 derived
    tda_cyclicity: float  # Beta_1 derived
    
    # Layer 3: Strategy parameters
    recommended_leverage: float
    factor_weights: Dict[str, float]
    position_scale: float
    stop_multiplier: float
    
    # Signals
    trade_allowed: bool
    strategy_mode: str  # 'momentum', 'mean_reversion', 'defensive', 'neutral'
    
    # Timestamps
    timestamp: str = ""
    days_in_regime: int = 0


@dataclass
class HMMState:
    """Hidden Markov Model state for regime detection."""
    # State probabilities (sum to 1)
    p_bull: float = 0.25
    p_bear: float = 0.25
    p_high_vol: float = 0.25
    p_low_vol: float = 0.25
    
    # Transition matrix (from -> to)
    transition_matrix: np.ndarray = field(default_factory=lambda: np.array([
        # Bull   Bear   HighV  LowV
        [0.85,  0.05,  0.05,  0.05],  # From Bull
        [0.05,  0.85,  0.08,  0.02],  # From Bear
        [0.10,  0.15,  0.70,  0.05],  # From HighVol
        [0.15,  0.03,  0.02,  0.80],  # From LowVol
    ]))
    
    # Emission probabilities (simplified)
    mean_returns: np.ndarray = field(default_factory=lambda: np.array([0.08, -0.05, 0.0, 0.05]))
    volatilities: np.ndarray = field(default_factory=lambda: np.array([0.15, 0.25, 0.35, 0.10]))


class MacroRegimeDetector:
    """Layer 1: HMM-based macro regime detection.
    
    Uses:
    - VIX levels and changes
    - Market breadth (% above MA50/200)
    - Sector rotation patterns
    - Momentum/trend strength
    """
    
    def __init__(
        self,
        lookback_days: int = 60,
        ema_span: int = 5,
        vix_low: float = 15,
        vix_mid: float = 20,
        vix_high: float = 30,
        vix_extreme: float = 40,
    ):
        self.lookback_days = lookback_days
        self.ema_span = ema_span
        self.vix_low = vix_low
        self.vix_mid = vix_mid
        self.vix_high = vix_high
        self.vix_extreme = vix_extreme
        
        # HMM state
        self.hmm_state = HMMState()
        self.regime_history = deque(maxlen=252)  # 1 year
        self.current_regime = MacroRegime.TRANSITION
        self.days_in_current_regime = 0
        
        # Feature cache
        self._feature_cache = {}
        
    def extract_features(
        self,
        spy_prices: pd.DataFrame,
        vix_data: Optional[pd.DataFrame] = None,
        sector_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Dict[str, float]:
        """Extract regime detection features from market data."""
        features = {}
        
        # Get close prices
        close = self._get_close(spy_prices)
        if len(close) < 200:
            return self._default_features()
        
        current_price = close.iloc[-1]
        
        # Moving averages
        ma_50 = close.rolling(50).mean()
        ma_200 = close.rolling(200).mean()
        
        # 1. Trend features
        features['price_vs_ma50'] = current_price / ma_50.iloc[-1] - 1
        features['price_vs_ma200'] = current_price / ma_200.iloc[-1] - 1
        features['ma_cross'] = 1.0 if ma_50.iloc[-1] > ma_200.iloc[-1] else -1.0
        
        # MA cross persistence
        ma_cross_series = (ma_50 > ma_200).astype(float).iloc[-60:]
        features['ma_cross_pct'] = ma_cross_series.mean() if len(ma_cross_series) > 0 else 0.5
        
        # 2. Momentum features (multi-horizon)
        for days, name in [(21, '1m'), (63, '3m'), (126, '6m')]:
            if len(close) >= days:
                mom = close.iloc[-1] / close.iloc[-days] - 1
                features[f'momentum_{name}'] = mom
            else:
                features[f'momentum_{name}'] = 0.0
        
        # Momentum acceleration
        if len(close) >= 42:
            mom_recent = close.iloc[-1] / close.iloc[-21] - 1
            mom_prior = close.iloc[-21] / close.iloc[-42] - 1
            features['momentum_accel'] = mom_recent - mom_prior
        else:
            features['momentum_accel'] = 0.0
        
        # 3. Volatility features
        returns = close.pct_change().dropna()
        if len(returns) >= 20:
            vol_20d = returns.iloc[-20:].std() * np.sqrt(252)
            vol_60d = returns.iloc[-60:].std() * np.sqrt(252) if len(returns) >= 60 else vol_20d
            features['realized_vol'] = vol_20d
            features['vol_ratio'] = vol_20d / vol_60d if vol_60d > 0 else 1.0
        else:
            features['realized_vol'] = 0.15
            features['vol_ratio'] = 1.0
        
        # 4. VIX features
        if vix_data is not None and len(vix_data) > 0:
            vix_close = self._get_close(vix_data)
            features['vix'] = vix_close.iloc[-1]
            if len(vix_close) >= 20:
                features['vix_ma20'] = vix_close.rolling(20).mean().iloc[-1]
                features['vix_pct_change'] = vix_close.iloc[-1] / vix_close.iloc[-5] - 1
            else:
                features['vix_ma20'] = features['vix']
                features['vix_pct_change'] = 0.0
        else:
            # Estimate VIX from realized vol
            features['vix'] = features['realized_vol'] * 100
            features['vix_ma20'] = features['vix']
            features['vix_pct_change'] = 0.0
        
        # 5. Drawdown features
        rolling_max = close.rolling(252, min_periods=1).max()
        drawdown = (close / rolling_max - 1).iloc[-1]
        features['drawdown'] = drawdown
        features['drawdown_5d'] = (close.iloc[-1] / close.iloc[-5:].max()) - 1 if len(close) >= 5 else 0
        
        # 6. Breadth proxy (using price position relative to MAs)
        pct_above_ma50 = (close.iloc[-60:] > ma_50.iloc[-60:]).mean() if len(close) >= 60 else 0.5
        pct_above_ma200 = (close.iloc[-60:] > ma_200.iloc[-60:]).mean() if len(close) >= 60 else 0.5
        features['breadth_ma50'] = pct_above_ma50
        features['breadth_ma200'] = pct_above_ma200
        
        # 7. Sector rotation (if available)
        if sector_data:
            features.update(self._compute_sector_rotation(sector_data))
        else:
            features['sector_dispersion'] = 0.15
            features['defensive_outperform'] = 0.0
        
        return features
    
    def _compute_sector_rotation(
        self,
        sector_data: Dict[str, pd.DataFrame],
    ) -> Dict[str, float]:
        """Compute sector rotation signals."""
        sector_features = {}
        
        # Define sector categories
        defensive = ['XLU', 'XLP', 'XLV']  # Utilities, Staples, Healthcare
        cyclical = ['XLY', 'XLI', 'XLF']   # Discretionary, Industrial, Financial
        growth = ['XLK', 'XLC']             # Tech, Communication
        
        returns_60d = {}
        for sector, df in sector_data.items():
            if df is not None and len(df) >= 60:
                close = self._get_close(df)
                ret = close.iloc[-1] / close.iloc[-60] - 1
                returns_60d[sector] = ret
        
        if len(returns_60d) < 3:
            return {'sector_dispersion': 0.15, 'defensive_outperform': 0.0}
        
        # Sector dispersion (cross-sectional volatility)
        all_returns = list(returns_60d.values())
        sector_features['sector_dispersion'] = np.std(all_returns)
        
        # Defensive vs cyclical performance
        def_ret = np.mean([returns_60d.get(s, 0) for s in defensive if s in returns_60d])
        cyc_ret = np.mean([returns_60d.get(s, 0) for s in cyclical if s in returns_60d])
        sector_features['defensive_outperform'] = def_ret - cyc_ret
        
        return sector_features
    
    def classify_regime(self, features: Dict[str, float]) -> Tuple[MacroRegime, float]:
        """Classify current macro regime using fuzzy logic + HMM priors."""
        
        # Compute regime scores
        scores = {
            MacroRegime.BULL_MOMENTUM: self._score_bull(features),
            MacroRegime.BEAR_DEFENSIVE: self._score_bear(features),
            MacroRegime.HIGH_VOLATILITY: self._score_high_vol(features),
            MacroRegime.LOW_VOLATILITY: self._score_low_vol(features),
        }
        
        # Apply HMM transition priors
        if self.current_regime != MacroRegime.TRANSITION:
            prior_idx = self._regime_to_idx(self.current_regime)
            for regime, score in scores.items():
                target_idx = self._regime_to_idx(regime)
                transition_prob = self.hmm_state.transition_matrix[prior_idx, target_idx]
                # Bayesian update: posterior ∝ prior × likelihood
                scores[regime] = score * transition_prob
        
        # Normalize to probabilities
        total = sum(scores.values()) + 1e-10
        probs = {k: v / total for k, v in scores.items()}
        
        # Update HMM state
        self.hmm_state.p_bull = probs[MacroRegime.BULL_MOMENTUM]
        self.hmm_state.p_bear = probs[MacroRegime.BEAR_DEFENSIVE]
        self.hmm_state.p_high_vol = probs[MacroRegime.HIGH_VOLATILITY]
        self.hmm_state.p_low_vol = probs[MacroRegime.LOW_VOLATILITY]
        
        # Select regime
        best_regime = max(probs.keys(), key=lambda k: probs[k])
        confidence = probs[best_regime]
        
        # Update regime tracking
        if best_regime == self.current_regime:
            self.days_in_current_regime += 1
        else:
            if confidence > 0.4:  # Threshold for regime change
                self.current_regime = best_regime
                self.days_in_current_regime = 1
        
        self.regime_history.append((self.current_regime, confidence))
        
        return self.current_regime, confidence
    
    def _score_bull(self, f: Dict[str, float]) -> float:
        """Score for Bull Momentum regime."""
        score = 0.0
        
        # Positive trend
        if f.get('ma_cross', 0) > 0:
            score += 0.25
        if f.get('price_vs_ma50', 0) > 0:
            score += 0.15
        if f.get('price_vs_ma200', 0) > 0:
            score += 0.15
        
        # Strong momentum
        mom_3m = f.get('momentum_3m', 0)
        if mom_3m > 0.05:
            score += min(0.25, mom_3m * 2)
        
        # Low VIX
        vix = f.get('vix', 20)
        if vix < self.vix_mid:
            score += 0.15 * (self.vix_mid - vix) / self.vix_mid
        
        # Small/no drawdown
        if f.get('drawdown', 0) > -0.05:
            score += 0.10
        
        return max(0, min(1, score))
    
    def _score_bear(self, f: Dict[str, float]) -> float:
        """Score for Bear Defensive regime."""
        score = 0.0
        
        # Negative trend
        if f.get('ma_cross', 0) < 0:
            score += 0.25
        if f.get('price_vs_ma200', 0) < 0:
            score += 0.20
        
        # Negative momentum
        mom_3m = f.get('momentum_3m', 0)
        if mom_3m < -0.05:
            score += min(0.25, abs(mom_3m) * 2)
        
        # VIX elevated
        vix = f.get('vix', 20)
        if vix > self.vix_high:
            score += 0.15
        
        # Defensive sectors outperforming
        if f.get('defensive_outperform', 0) > 0.02:
            score += 0.10
        
        # Significant drawdown
        if f.get('drawdown', 0) < -0.10:
            score += 0.15
        
        return max(0, min(1, score))
    
    def _score_high_vol(self, f: Dict[str, float]) -> float:
        """Score for High Volatility regime."""
        score = 0.0
        
        # Elevated VIX
        vix = f.get('vix', 20)
        if vix > self.vix_extreme:
            score += 0.40
        elif vix > self.vix_high:
            score += 0.25
        elif vix > self.vix_mid:
            score += 0.10
        
        # Vol ratio expanding
        if f.get('vol_ratio', 1) > 1.3:
            score += 0.20
        
        # VIX spiking
        if f.get('vix_pct_change', 0) > 0.15:
            score += 0.15
        
        # Sector dispersion high
        if f.get('sector_dispersion', 0) > 0.20:
            score += 0.10
        
        # Quick drawdown
        if f.get('drawdown_5d', 0) < -0.03:
            score += 0.15
        
        return max(0, min(1, score))
    
    def _score_low_vol(self, f: Dict[str, float]) -> float:
        """Score for Low Volatility regime."""
        score = 0.0
        
        # Low VIX
        vix = f.get('vix', 20)
        if vix < self.vix_low:
            score += 0.35
        elif vix < self.vix_mid:
            score += 0.15
        
        # Vol ratio contracting
        if f.get('vol_ratio', 1) < 0.8:
            score += 0.20
        
        # Stable VIX
        if abs(f.get('vix_pct_change', 0)) < 0.05:
            score += 0.15
        
        # Low drawdown, positive trend
        if f.get('drawdown', 0) > -0.03:
            score += 0.10
        if f.get('ma_cross', 0) > 0:
            score += 0.10
        
        # Low sector dispersion
        if f.get('sector_dispersion', 0) < 0.10:
            score += 0.10
        
        return max(0, min(1, score))
    
    def _regime_to_idx(self, regime: MacroRegime) -> int:
        """Convert regime to matrix index."""
        mapping = {
            MacroRegime.BULL_MOMENTUM: 0,
            MacroRegime.BEAR_DEFENSIVE: 1,
            MacroRegime.HIGH_VOLATILITY: 2,
            MacroRegime.LOW_VOLATILITY: 3,
            MacroRegime.TRANSITION: 0,
        }
        return mapping.get(regime, 0)
    
    def _get_close(self, df: pd.DataFrame) -> pd.Series:
        """Extract close price series."""
        if df is None or len(df) == 0:
            return pd.Series([100.0])
        for col in ['close', 'Close', 'Adj Close']:
            if col in df.columns:
                return df[col]
        if len(df.columns) >= 4:
            return df.iloc[:, 3]
        return df.iloc[:, 0]
    
    def _default_features(self) -> Dict[str, float]:
        """Return default neutral features."""
        return {
            'price_vs_ma50': 0.0, 'price_vs_ma200': 0.0, 'ma_cross': 0.0,
            'ma_cross_pct': 0.5, 'momentum_1m': 0.0, 'momentum_3m': 0.0,
            'momentum_6m': 0.0, 'momentum_accel': 0.0, 'realized_vol': 0.15,
            'vol_ratio': 1.0, 'vix': 20.0, 'vix_ma20': 20.0, 'vix_pct_change': 0.0,
            'drawdown': 0.0, 'drawdown_5d': 0.0, 'breadth_ma50': 0.5,
            'breadth_ma200': 0.5, 'sector_dispersion': 0.15, 'defensive_outperform': 0.0,
        }


class TDARegimeAnalyzer:
    """Layer 2: TDA-based regime correlation analysis.
    
    Uses topological features to detect:
    - Market fragmentation (β₀ - connected components)
    - Cyclical patterns (β₁ - holes/loops)
    - Regime transitions (persistence diagram changes)
    """
    
    def __init__(
        self,
        fragmentation_threshold_high: float = 0.7,
        fragmentation_threshold_low: float = 0.3,
        turbulence_threshold: float = 60.0,
        persistence_lookback: int = 20,
    ):
        self.fragmentation_high = fragmentation_threshold_high
        self.fragmentation_low = fragmentation_threshold_low
        self.turbulence_threshold = turbulence_threshold
        self.persistence_lookback = persistence_lookback
        
        self.regime_history = deque(maxlen=60)
        self.current_regime = TDARegime.CONSOLIDATION
    
    def analyze_tda_regime(
        self,
        tda_features: pd.DataFrame,
        lookback: int = 10,
    ) -> Tuple[TDARegime, Dict[str, float]]:
        """
        Classify TDA regime from persistence features.
        
        Args:
            tda_features: DataFrame with TDA columns (betti_0, betti_1, persistence, etc.)
            lookback: Days to analyze
            
        Returns:
            (TDARegime, metrics dict)
        """
        if tda_features is None or len(tda_features) < lookback:
            return TDARegime.CONSOLIDATION, {'turbulence': 50.0, 'fragmentation': 0.5, 'cyclicity': 0.5}
        
        recent = tda_features.iloc[-lookback:]
        
        # Extract core TDA metrics
        metrics = {}
        
        # Turbulence index
        if 'turbulence_index' in recent.columns:
            metrics['turbulence'] = recent['turbulence_index'].mean()
        elif 'turbulence' in recent.columns:
            metrics['turbulence'] = recent['turbulence'].mean()
        else:
            metrics['turbulence'] = 50.0
        
        # Fragmentation (β₀ derived)
        if 'betti_0' in recent.columns:
            b0 = recent['betti_0'].mean()
            metrics['fragmentation'] = min(1.0, b0 / 10.0)  # Normalize
        elif 'fragmentation' in recent.columns:
            metrics['fragmentation'] = recent['fragmentation'].mean()
        else:
            metrics['fragmentation'] = 0.5
        
        # Cyclicity (β₁ derived)
        if 'betti_1' in recent.columns:
            b1 = recent['betti_1'].mean()
            metrics['cyclicity'] = min(1.0, b1 / 5.0)
        elif 'cyclicity' in recent.columns:
            metrics['cyclicity'] = recent['cyclicity'].mean()
        else:
            metrics['cyclicity'] = 0.5
        
        # Persistence entropy (market complexity)
        if 'entropy_h0' in recent.columns:
            metrics['entropy'] = recent['entropy_h0'].mean()
        else:
            metrics['entropy'] = 0.5
        
        # Classify regime
        regime = self._classify_from_metrics(metrics)
        self.current_regime = regime
        self.regime_history.append((regime, metrics['turbulence']))
        
        return regime, metrics
    
    def _classify_from_metrics(self, metrics: Dict[str, float]) -> TDARegime:
        """Classify TDA regime from computed metrics."""
        turb = metrics.get('turbulence', 50)
        frag = metrics.get('fragmentation', 0.5)
        
        # High turbulence = Risk Off
        if turb > self.turbulence_threshold:
            return TDARegime.RISK_OFF
        
        # Low fragmentation + low turbulence = Risk On
        if frag < self.fragmentation_low and turb < self.turbulence_threshold * 0.6:
            return TDARegime.RISK_ON
        
        # High fragmentation = possible regime break
        if frag > self.fragmentation_high:
            return TDARegime.REGIME_BREAK
        
        return TDARegime.CONSOLIDATION
    
    def detect_topology_shift(
        self,
        tda_features: pd.DataFrame,
        window: int = 10,
    ) -> Tuple[bool, float]:
        """
        Detect topological bifurcation (regime break signal).
        
        Uses rate of change in Betti numbers and persistence.
        """
        if tda_features is None or len(tda_features) < window * 2:
            return False, 0.0
        
        recent = tda_features.iloc[-window:]
        prior = tda_features.iloc[-window*2:-window]
        
        # Compute change in key metrics
        shift_score = 0.0
        
        for col in ['betti_0', 'betti_1', 'turbulence_index', 'entropy_h0']:
            if col in tda_features.columns:
                recent_val = recent[col].mean()
                prior_val = prior[col].mean()
                if prior_val > 0:
                    change = abs(recent_val - prior_val) / prior_val
                    shift_score += min(0.25, change * 0.25)
        
        is_shift = shift_score > 0.5
        return is_shift, shift_score


class DynamicFactorAllocator:
    """Layer 3: Dynamic factor weight allocation based on regimes.
    
    Adjusts factor weights (Momentum, TDA, Value, Quality) based on:
    - Macro regime from Layer 1
    - TDA regime from Layer 2
    - Historical factor performance in each regime
    """
    
    def __init__(self):
        # Base weights (neutral)
        self.base_weights = {
            'momentum': 0.30,
            'tda': 0.25,
            'value': 0.20,
            'quality': 0.20,
            'reversal': 0.05,
        }
        
        # Regime-specific weight adjustments
        self.regime_weights = {
            # Macro regimes
            MacroRegime.BULL_MOMENTUM: {
                'momentum': 0.45, 'tda': 0.20, 'value': 0.10, 'quality': 0.15, 'reversal': 0.10
            },
            MacroRegime.BEAR_DEFENSIVE: {
                'momentum': 0.10, 'tda': 0.25, 'value': 0.25, 'quality': 0.35, 'reversal': 0.05
            },
            MacroRegime.HIGH_VOLATILITY: {
                'momentum': 0.15, 'tda': 0.35, 'value': 0.15, 'quality': 0.25, 'reversal': 0.10
            },
            MacroRegime.LOW_VOLATILITY: {
                'momentum': 0.35, 'tda': 0.20, 'value': 0.20, 'quality': 0.20, 'reversal': 0.05
            },
            MacroRegime.TRANSITION: {
                'momentum': 0.25, 'tda': 0.30, 'value': 0.20, 'quality': 0.20, 'reversal': 0.05
            },
        }
        
        # TDA overlay adjustments
        self.tda_overlays = {
            TDARegime.RISK_ON: {'tda': -0.05, 'momentum': 0.05},
            TDARegime.RISK_OFF: {'tda': 0.10, 'momentum': -0.10},
            TDARegime.REGIME_BREAK: {'tda': 0.15, 'quality': 0.05, 'momentum': -0.20},
            TDARegime.CONSOLIDATION: {'reversal': 0.05, 'momentum': -0.05},
        }
        
        # Strategy mode parameters - more aggressive for capturing upside
        self.strategy_params = {
            MacroRegime.BULL_MOMENTUM: {
                'mode': 'momentum',
                'leverage': 1.3,  # Increased for bull markets
                'position_scale': 1.4,  # More aggressive position sizing
                'stop_multiplier': 2.5,  # Wider stop to stay in trend
            },
            MacroRegime.BEAR_DEFENSIVE: {
                'mode': 'defensive',
                'leverage': 0.4,  # Reduced from 0.6
                'position_scale': 0.3,  # Reduced from 0.5
                'stop_multiplier': 1.0,  # Tighter stop
            },
            MacroRegime.HIGH_VOLATILITY: {
                'mode': 'neutral',
                'leverage': 0.5,  # Slightly higher to capture volatility moves
                'position_scale': 0.4,  # Reduced from 0.4
                'stop_multiplier': 1.0,
            },
            MacroRegime.LOW_VOLATILITY: {
                'mode': 'mean_reversion',
                'leverage': 1.1,  # Slightly higher
                'position_scale': 1.2,  # More positions
                'stop_multiplier': 2.0,
            },
            MacroRegime.TRANSITION: {
                'mode': 'neutral',
                'leverage': 0.8,  # Higher for transition
                'position_scale': 0.7,
                'stop_multiplier': 1.5,
            },
        }
    
    def get_allocation(
        self,
        macro_regime: MacroRegime,
        tda_regime: TDARegime,
        macro_confidence: float,
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Compute dynamic factor allocation.
        
        Returns:
            (factor_weights, strategy_params)
        """
        # Start with regime-specific weights
        weights = self.regime_weights.get(macro_regime, self.base_weights).copy()
        
        # Apply TDA overlay
        if tda_regime in self.tda_overlays:
            for factor, adjustment in self.tda_overlays[tda_regime].items():
                if factor in weights:
                    # Scale adjustment by macro confidence
                    weights[factor] += adjustment * (1 - macro_confidence) * 0.5
        
        # Normalize weights to sum to 1
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
        
        # Get strategy parameters
        params = self.strategy_params.get(macro_regime, self.strategy_params[MacroRegime.TRANSITION]).copy()
        
        # Adjust for TDA regime
        if tda_regime == TDARegime.RISK_OFF:
            params['leverage'] *= 0.8
            params['position_scale'] *= 0.7
        elif tda_regime == TDARegime.REGIME_BREAK:
            params['leverage'] *= 0.6
            params['position_scale'] *= 0.5
            params['mode'] = 'defensive'
        
        return weights, params


class HierarchicalRegimeStrategy:
    """
    Complete 3-layer hierarchical regime strategy.
    
    Combines:
    - Layer 1: Macro HMM regime detection
    - Layer 2: TDA topology analysis
    - Layer 3: Dynamic factor allocation
    """
    
    def __init__(
        self,
        macro_detector: Optional[MacroRegimeDetector] = None,
        tda_analyzer: Optional[TDARegimeAnalyzer] = None,
        factor_allocator: Optional[DynamicFactorAllocator] = None,
    ):
        self.macro_detector = macro_detector or MacroRegimeDetector()
        self.tda_analyzer = tda_analyzer or TDARegimeAnalyzer()
        self.factor_allocator = factor_allocator or DynamicFactorAllocator()
        
        self.current_meta = None
        self.meta_history = deque(maxlen=252)
    
    def analyze(
        self,
        spy_prices: pd.DataFrame,
        vix_data: Optional[pd.DataFrame] = None,
        sector_data: Optional[Dict[str, pd.DataFrame]] = None,
        tda_features: Optional[pd.DataFrame] = None,
        timestamp: str = "",
    ) -> RegimeMeta:
        """
        Run full hierarchical regime analysis.
        
        Returns:
            RegimeMeta with complete regime state and strategy parameters
        """
        # Layer 1: Macro regime
        macro_features = self.macro_detector.extract_features(spy_prices, vix_data, sector_data)
        macro_regime, macro_confidence = self.macro_detector.classify_regime(macro_features)
        
        # Transition probability (for risk management)
        transition_prob = 1 - macro_confidence
        
        # Layer 2: TDA regime
        tda_regime, tda_metrics = self.tda_analyzer.analyze_tda_regime(tda_features)
        
        # Detect topology shift
        is_shift, shift_score = self.tda_analyzer.detect_topology_shift(tda_features)
        if is_shift:
            tda_regime = TDARegime.REGIME_BREAK
        
        # Layer 3: Dynamic allocation
        factor_weights, strategy_params = self.factor_allocator.get_allocation(
            macro_regime, tda_regime, macro_confidence
        )
        
        # Determine if trading is allowed - more permissive for capturing upside
        trade_allowed = True
        # Only block in extreme conditions
        if macro_regime == MacroRegime.BEAR_DEFENSIVE and tda_regime == TDARegime.RISK_OFF and macro_confidence > 0.8:
            trade_allowed = False  # Maximum caution only when very confident it's a bear
        # Regime breaks are opportunities, don't block trading
        # if tda_regime == TDARegime.REGIME_BREAK and macro_confidence < 0.5:
        #     trade_allowed = False  # Regime uncertainty
        
        # Build meta object
        meta = RegimeMeta(
            # Layer 1
            macro_regime=macro_regime,
            macro_confidence=macro_confidence,
            macro_transition_prob=transition_prob,
            
            # Layer 2
            tda_regime=tda_regime,
            tda_turbulence=tda_metrics.get('turbulence', 50.0),
            tda_fragmentation=tda_metrics.get('fragmentation', 0.5),
            tda_cyclicity=tda_metrics.get('cyclicity', 0.5),
            
            # Layer 3
            recommended_leverage=strategy_params['leverage'],
            factor_weights=factor_weights,
            position_scale=strategy_params['position_scale'],
            stop_multiplier=strategy_params['stop_multiplier'],
            
            # Signals
            trade_allowed=trade_allowed,
            strategy_mode=strategy_params['mode'],
            
            # Timestamps
            timestamp=timestamp,
            days_in_regime=self.macro_detector.days_in_current_regime,
        )
        
        self.current_meta = meta
        self.meta_history.append(meta)
        
        return meta
    
    def get_regime_summary(self) -> Dict[str, Any]:
        """Get summary of current regime state."""
        if self.current_meta is None:
            return {'status': 'uninitialized'}
        
        m = self.current_meta
        return {
            'macro_regime': m.macro_regime.value,
            'macro_confidence': f"{m.macro_confidence:.1%}",
            'tda_regime': m.tda_regime.value,
            'turbulence': f"{m.tda_turbulence:.1f}",
            'strategy_mode': m.strategy_mode,
            'leverage': f"{m.recommended_leverage:.2f}x",
            'trade_allowed': m.trade_allowed,
            'days_in_regime': m.days_in_regime,
            'factor_weights': {k: f"{v:.0%}" for k, v in m.factor_weights.items()},
        }
