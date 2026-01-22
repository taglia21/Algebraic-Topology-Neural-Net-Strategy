"""
Hierarchical Regime Controller
==============================

V2.2 Meta-state controller for regime-aware policy selection.

Architecture:
- Level 0: Volatility regime detection (low/medium/high)
- Level 1: Trend regime detection (trending/flat/mean-reverting)  
- Level 2: Combined meta-state → sub-policy selection

CUSUM Change Detection:
- Detects regime transitions with > 2.5 sigma threshold
- Fast response to market structure changes
- Reduces whipsaws compared to raw volatility measures

Sub-Policies:
- Aggressive: For trending low-vol (larger positions, wider stops)
- Neutral: For mixed regimes (balanced sizing)
- Conservative: For volatile/uncertain (smaller positions, tighter risk)

Research Basis:
- Hierarchical RL improves multi-regime adaptation
- CUSUM detects structural breaks with controlled false positive rate
"""

import os
import json
import logging
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


# =============================================================================
# REGIME STATES
# =============================================================================

class VolatilityRegime(Enum):
    """Volatility level classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    
    
class TrendRegime(Enum):
    """Trend direction classification."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    FLAT = "flat"
    MEAN_REVERTING = "mean_reverting"


@dataclass
class RegimeState:
    """
    Combined meta-state from hierarchical regime detection.
    
    Attributes:
        volatility: Current volatility regime
        trend: Current trend regime
        confidence: Detection confidence [0, 1]
        transition_prob: Probability of regime transition
        cusum_statistic: CUSUM statistic for change detection
        timestamp: When state was computed
    """
    volatility: VolatilityRegime = VolatilityRegime.MEDIUM
    trend: TrendRegime = TrendRegime.FLAT
    confidence: float = 0.5
    transition_prob: float = 0.1
    cusum_statistic: float = 0.0
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
            
    def to_dict(self) -> Dict[str, Any]:
        return {
            "volatility": self.volatility.value,
            "trend": self.trend.value,
            "confidence": self.confidence,
            "transition_prob": self.transition_prob,
            "cusum_statistic": self.cusum_statistic,
            "timestamp": self.timestamp,
        }
    
    @property
    def meta_state(self) -> str:
        """Combine regimes into meta-state string."""
        return f"{self.volatility.value}_{self.trend.value}"
    
    @property
    def policy_name(self) -> str:
        """Map meta-state to sub-policy name."""
        # Aggressive: trending + low vol
        if self.volatility == VolatilityRegime.LOW and \
           self.trend in (TrendRegime.TRENDING_UP, TrendRegime.TRENDING_DOWN):
            return "aggressive"
            
        # Conservative: high vol or mean-reverting
        if self.volatility == VolatilityRegime.HIGH or \
           self.trend == TrendRegime.MEAN_REVERTING:
            return "conservative"
            
        # Neutral: everything else
        return "neutral"


# =============================================================================
# SUB-POLICIES
# =============================================================================

@dataclass
class SubPolicy:
    """
    Sub-policy configuration for regime-specific behavior.
    
    Controls position sizing, risk parameters, and trading thresholds
    based on current market regime.
    """
    name: str
    description: str
    
    # Position sizing
    max_position_pct: float = 0.03  # 3% max
    position_scale: float = 1.0     # Multiplier on base position
    
    # Risk parameters
    stop_loss_pct: float = 0.02     # 2% stop loss
    take_profit_pct: float = 0.06   # 6% take profit
    max_drawdown_pct: float = 0.15  # 15% max drawdown
    
    # Trading thresholds
    signal_threshold: float = 0.6   # Min signal strength to trade
    holding_period: int = 5         # Target holding period (days)
    
    # Execution
    use_limit_orders: bool = False
    slippage_tolerance: float = 0.001
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "max_position_pct": self.max_position_pct,
            "position_scale": self.position_scale,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "signal_threshold": self.signal_threshold,
            "holding_period": self.holding_period,
        }


# Default sub-policies
AGGRESSIVE_POLICY = SubPolicy(
    name="aggressive",
    description="Trending low-volatility regime: larger positions, wider stops",
    max_position_pct=0.035,
    position_scale=1.2,
    stop_loss_pct=0.025,
    take_profit_pct=0.08,
    max_drawdown_pct=0.18,
    signal_threshold=0.55,
    holding_period=7,
)

NEUTRAL_POLICY = SubPolicy(
    name="neutral",
    description="Mixed regime: balanced sizing and risk",
    max_position_pct=0.025,
    position_scale=1.0,
    stop_loss_pct=0.02,
    take_profit_pct=0.06,
    max_drawdown_pct=0.15,
    signal_threshold=0.6,
    holding_period=5,
)

CONSERVATIVE_POLICY = SubPolicy(
    name="conservative",
    description="High volatility or uncertain regime: reduced positions",
    max_position_pct=0.015,
    position_scale=0.6,
    stop_loss_pct=0.015,
    take_profit_pct=0.045,
    max_drawdown_pct=0.10,
    signal_threshold=0.7,
    holding_period=3,
    use_limit_orders=True,
)


# =============================================================================
# CUSUM CHANGE DETECTOR
# =============================================================================

class CUSUMDetector:
    """
    CUSUM (Cumulative Sum) change detection algorithm.
    
    Detects structural changes in time series with controlled
    false positive rate. Used for regime transition detection.
    
    Algorithm:
    - Maintains cumulative sum of deviations from target mean
    - Signals change when sum exceeds threshold (> 2.5 sigma)
    - Resets after detection to detect new changes
    
    Properties:
    - Fast detection of persistent changes
    - Robust to transient noise
    - Controllable sensitivity via threshold
    """
    
    def __init__(self, 
                 threshold: float = 2.5,  # Sigma threshold
                 drift: float = 0.5,      # Allowable drift before flagging
                 warmup_period: int = 50):
        """
        Initialize CUSUM detector.
        
        Args:
            threshold: Detection threshold in sigma units
            drift: Minimum shift to detect (in sigma)
            warmup_period: Samples before detection enabled
        """
        self.threshold = threshold
        self.drift = drift
        self.warmup_period = warmup_period
        
        self.reset()
        
    def reset(self):
        """Reset detector state."""
        self.cusum_pos = 0.0  # Upper CUSUM
        self.cusum_neg = 0.0  # Lower CUSUM
        self.mean = 0.0
        self.std = 1.0
        self.samples = 0
        self.history = deque(maxlen=500)
        self.change_detected = False
        self.last_change_sample = 0
        
    def update(self, value: float) -> bool:
        """
        Update CUSUM with new observation.
        
        Args:
            value: New observation
            
        Returns:
            True if change detected
        """
        self.history.append(value)
        self.samples += 1
        
        # Update running statistics
        if len(self.history) >= 2:
            self.mean = np.mean(self.history)
            self.std = max(np.std(self.history), 1e-8)
            
        # Standardize observation
        z = (value - self.mean) / self.std
        
        # Update CUSUM statistics
        self.cusum_pos = max(0, self.cusum_pos + z - self.drift)
        self.cusum_neg = min(0, self.cusum_neg + z + self.drift)
        
        # Check for change
        self.change_detected = False
        
        if self.samples > self.warmup_period:
            if self.cusum_pos > self.threshold:
                self.change_detected = True
                self.cusum_pos = 0
                self.last_change_sample = self.samples
                
            if -self.cusum_neg > self.threshold:
                self.change_detected = True
                self.cusum_neg = 0
                self.last_change_sample = self.samples
                
        return self.change_detected
    
    def get_statistic(self) -> float:
        """Get current CUSUM statistic (max of pos/neg)."""
        return max(self.cusum_pos, -self.cusum_neg)
    
    def get_state(self) -> Dict[str, Any]:
        """Get detector state for logging."""
        return {
            "cusum_pos": self.cusum_pos,
            "cusum_neg": self.cusum_neg,
            "mean": self.mean,
            "std": self.std,
            "samples": self.samples,
            "change_detected": self.change_detected,
            "last_change_sample": self.last_change_sample,
        }


# =============================================================================
# VOLATILITY REGIME DETECTOR
# =============================================================================

class VolatilityRegimeDetector:
    """
    Detects volatility regime from price/return data.
    
    Uses multiple timeframes and GARCH-like exponential smoothing
    for robust regime classification.
    """
    
    def __init__(self,
                 low_threshold: float = 0.12,   # Annualized vol < 12%
                 high_threshold: float = 0.25,  # Annualized vol > 25%
                 lookback_short: int = 20,
                 lookback_long: int = 60,
                 decay_factor: float = 0.94):
        """
        Initialize volatility detector.
        
        Args:
            low_threshold: Threshold for low vol regime (annualized)
            high_threshold: Threshold for high vol regime (annualized)
            lookback_short: Short-term lookback window
            lookback_long: Long-term lookback window
            decay_factor: Exponential decay for EWMA vol
        """
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.lookback_short = lookback_short
        self.lookback_long = lookback_long
        self.decay_factor = decay_factor
        
        self.returns_buffer = deque(maxlen=lookback_long * 2)
        self.ewma_vol = None
        self.cusum = CUSUMDetector(threshold=2.0)
        
    def update(self, returns: np.ndarray) -> VolatilityRegime:
        """
        Update with new returns and get regime.
        
        Args:
            returns: Array of daily returns
            
        Returns:
            Current volatility regime
        """
        # Add to buffer
        for r in returns:
            self.returns_buffer.append(r)
            
        if len(self.returns_buffer) < self.lookback_short:
            return VolatilityRegime.MEDIUM
            
        arr = np.array(self.returns_buffer)
        
        # Compute realized volatility (annualized)
        short_vol = np.std(arr[-self.lookback_short:]) * np.sqrt(252)
        long_vol = np.std(arr[-self.lookback_long:]) * np.sqrt(252) if len(arr) >= self.lookback_long else short_vol
        
        # EWMA volatility
        if self.ewma_vol is None:
            self.ewma_vol = short_vol
        else:
            self.ewma_vol = self.decay_factor * self.ewma_vol + \
                           (1 - self.decay_factor) * short_vol
                           
        # Use blend of short and EWMA
        vol = 0.5 * short_vol + 0.5 * self.ewma_vol
        
        # Update CUSUM for vol changes
        self.cusum.update(vol)
        
        # Classify regime
        if vol < self.low_threshold:
            return VolatilityRegime.LOW
        elif vol > self.high_threshold:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.MEDIUM
            
    def get_current_vol(self) -> float:
        """Get current volatility estimate."""
        if self.ewma_vol is not None:
            return self.ewma_vol
        return 0.15  # Default


# =============================================================================
# TREND REGIME DETECTOR
# =============================================================================

class TrendRegimeDetector:
    """
    Detects trend regime from price data.
    
    Uses multiple indicators:
    - Price vs moving averages
    - ADX-like directional measure
    - Hurst exponent for mean-reversion
    """
    
    def __init__(self,
                 trend_threshold: float = 0.02,  # 2% above/below MA
                 lookback_short: int = 10,
                 lookback_medium: int = 30,
                 lookback_long: int = 60,
                 hurst_lookback: int = 100):
        """
        Initialize trend detector.
        
        Args:
            trend_threshold: Min % deviation from MA to be trending
            lookback_short: Short MA period
            lookback_medium: Medium MA period
            lookback_long: Long MA period
            hurst_lookback: Lookback for Hurst estimation
        """
        self.trend_threshold = trend_threshold
        self.lookback_short = lookback_short
        self.lookback_medium = lookback_medium
        self.lookback_long = lookback_long
        self.hurst_lookback = hurst_lookback
        
        self.price_buffer = deque(maxlen=hurst_lookback * 2)
        self.cusum = CUSUMDetector(threshold=2.5)
        
    def _estimate_hurst(self, prices: np.ndarray) -> float:
        """
        Estimate Hurst exponent using R/S analysis.
        
        H < 0.5: Mean-reverting
        H = 0.5: Random walk
        H > 0.5: Trending
        """
        if len(prices) < 20:
            return 0.5
            
        n = len(prices)
        max_k = min(n // 2, 50)
        
        rs_list = []
        n_list = []
        
        for k in range(10, max_k):
            # Subseries of length k
            rs_k = []
            for i in range(0, n - k, k):
                subseries = prices[i:i+k]
                if len(subseries) < k:
                    continue
                    
                mean = np.mean(subseries)
                deviations = subseries - mean
                cumdev = np.cumsum(deviations)
                
                r = np.max(cumdev) - np.min(cumdev)
                s = np.std(subseries)
                
                if s > 0:
                    rs_k.append(r / s)
                    
            if rs_k:
                rs_list.append(np.mean(rs_k))
                n_list.append(k)
                
        if len(rs_list) < 5:
            return 0.5
            
        # Log-log regression
        log_n = np.log(n_list)
        log_rs = np.log(rs_list)
        
        try:
            coeffs = np.polyfit(log_n, log_rs, 1)
            hurst = coeffs[0]
            return np.clip(hurst, 0.0, 1.0)
        except:
            return 0.5
            
    def update(self, prices: np.ndarray) -> TrendRegime:
        """
        Update with new prices and get regime.
        
        Args:
            prices: Array of prices
            
        Returns:
            Current trend regime
        """
        # Add to buffer
        for p in prices:
            self.price_buffer.append(p)
            
        if len(self.price_buffer) < self.lookback_medium:
            return TrendRegime.FLAT
            
        arr = np.array(self.price_buffer)
        current_price = arr[-1]
        
        # Moving averages
        ma_short = np.mean(arr[-self.lookback_short:])
        ma_medium = np.mean(arr[-self.lookback_medium:])
        
        # Deviation from medium MA
        deviation = (current_price - ma_medium) / ma_medium
        
        # Check Hurst for mean-reversion
        hurst = self._estimate_hurst(arr[-self.hurst_lookback:])
        
        # Update CUSUM
        self.cusum.update(deviation)
        
        # Classify regime
        if hurst < 0.4:
            return TrendRegime.MEAN_REVERTING
        elif deviation > self.trend_threshold and ma_short > ma_medium:
            return TrendRegime.TRENDING_UP
        elif deviation < -self.trend_threshold and ma_short < ma_medium:
            return TrendRegime.TRENDING_DOWN
        else:
            return TrendRegime.FLAT
            
    def get_trend_strength(self) -> float:
        """Get current trend strength [0, 1]."""
        if len(self.price_buffer) < self.lookback_medium:
            return 0.0
            
        arr = np.array(self.price_buffer)
        ma_medium = np.mean(arr[-self.lookback_medium:])
        deviation = abs((arr[-1] - ma_medium) / ma_medium)
        
        return min(1.0, deviation / 0.05)  # Normalize to 5% as full strength


# =============================================================================
# HIERARCHICAL CONTROLLER
# =============================================================================

class HierarchicalController:
    """
    Hierarchical Regime Controller for adaptive trading.
    
    Two-level hierarchy:
    1. Regime Detection: Volatility + Trend classification
    2. Policy Selection: Map regime to sub-policy
    
    Features:
    - CUSUM-based regime transition detection
    - Smooth policy blending during transitions
    - Logging for analysis and debugging
    """
    
    def __init__(self,
                 policies: Optional[Dict[str, SubPolicy]] = None,
                 transition_cooldown: int = 5,
                 blend_window: int = 3,
                 log_dir: str = "logs"):
        """
        Initialize controller.
        
        Args:
            policies: Dict of sub-policies (default: aggressive/neutral/conservative)
            transition_cooldown: Min samples between transitions
            blend_window: Samples for blending policies during transition
            log_dir: Directory for logging
        """
        self.policies = policies or {
            "aggressive": AGGRESSIVE_POLICY,
            "neutral": NEUTRAL_POLICY,
            "conservative": CONSERVATIVE_POLICY,
        }
        
        self.transition_cooldown = transition_cooldown
        self.blend_window = blend_window
        
        # Regime detectors
        self.volatility_detector = VolatilityRegimeDetector()
        self.trend_detector = TrendRegimeDetector()
        self.cusum = CUSUMDetector(threshold=2.5)
        
        # State tracking
        self.current_state = RegimeState()
        self.previous_state = None
        self.state_history: List[RegimeState] = []
        self.samples_since_transition = 0
        
        # Logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.regime_log = self.log_dir / "regime_history.jsonl"
        
        logger.info("HierarchicalController initialized")
        
    def update(self, 
               returns: np.ndarray,
               prices: np.ndarray) -> RegimeState:
        """
        Update regime detection with new data.
        
        Args:
            returns: Recent daily returns
            prices: Recent prices
            
        Returns:
            Updated regime state
        """
        self.previous_state = self.current_state
        
        # Level 0: Volatility regime
        vol_regime = self.volatility_detector.update(returns)
        
        # Level 1: Trend regime
        trend_regime = self.trend_detector.update(prices)
        
        # Compute transition probability (based on CUSUM)
        vol = self.volatility_detector.get_current_vol()
        self.cusum.update(vol)
        cusum_stat = self.cusum.get_statistic()
        
        # Transition probability increases as CUSUM approaches threshold
        transition_prob = min(1.0, cusum_stat / self.cusum.threshold)
        
        # Confidence based on detector consistency
        vol_consistent = (self.previous_state.volatility == vol_regime if self.previous_state else True)
        trend_consistent = (self.previous_state.trend == trend_regime if self.previous_state else True)
        confidence = 0.5 + 0.25 * vol_consistent + 0.25 * trend_consistent
        
        # Create new state
        self.current_state = RegimeState(
            volatility=vol_regime,
            trend=trend_regime,
            confidence=confidence,
            transition_prob=transition_prob,
            cusum_statistic=cusum_stat,
        )
        
        # Check for regime change
        if self.previous_state and self.current_state.meta_state != self.previous_state.meta_state:
            if self.samples_since_transition >= self.transition_cooldown:
                logger.info(f"Regime transition: {self.previous_state.meta_state} → {self.current_state.meta_state}")
                self.samples_since_transition = 0
            else:
                # Ignore transition during cooldown
                self.current_state = self.previous_state
                
        self.samples_since_transition += 1
        self.state_history.append(self.current_state)
        
        # Log state
        self._log_state()
        
        return self.current_state
    
    def get_active_policy(self) -> SubPolicy:
        """Get currently active sub-policy."""
        policy_name = self.current_state.policy_name
        return self.policies.get(policy_name, self.policies["neutral"])
    
    def get_blended_parameters(self) -> Dict[str, float]:
        """
        Get blended policy parameters during transition.
        
        Smoothly interpolates between old and new policy
        during blend_window after transition.
        """
        policy = self.get_active_policy()
        
        if self.samples_since_transition >= self.blend_window or self.previous_state is None:
            # No blending needed
            return policy.to_dict()
            
        # Blend with previous policy
        prev_policy = self.policies.get(
            self.previous_state.policy_name,
            self.policies["neutral"]
        )
        
        # Linear interpolation weight
        alpha = self.samples_since_transition / self.blend_window
        
        blended = {}
        for key in ["max_position_pct", "position_scale", "stop_loss_pct",
                    "take_profit_pct", "max_drawdown_pct", "signal_threshold"]:
            old_val = getattr(prev_policy, key)
            new_val = getattr(policy, key)
            blended[key] = old_val * (1 - alpha) + new_val * alpha
            
        # Non-numeric params from new policy
        blended["name"] = f"{prev_policy.name}→{policy.name}"
        blended["holding_period"] = policy.holding_period
        
        return blended
    
    def get_position_scale(self) -> float:
        """Get current position size multiplier."""
        params = self.get_blended_parameters()
        return params.get("position_scale", 1.0)
    
    def get_regime_string(self) -> str:
        """Get simple regime string for SAC."""
        meta = self.current_state.meta_state
        
        # Map to simple categories for SAC
        if "high" in meta:
            return "volatile"
        elif "trending" in meta and "low" in meta:
            return "trending"
        else:
            return "flat"
            
    def should_reduce_exposure(self) -> bool:
        """Check if exposure should be reduced due to regime uncertainty."""
        return (
            self.current_state.transition_prob > 0.7 or
            self.current_state.volatility == VolatilityRegime.HIGH or
            self.current_state.confidence < 0.5
        )
    
    def _log_state(self):
        """Log current regime state."""
        try:
            with open(self.regime_log, 'a') as f:
                f.write(json.dumps(self.current_state.to_dict()) + "\n")
        except Exception as e:
            logger.debug(f"Failed to log regime state: {e}")
            
    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of current state for monitoring."""
        return {
            "meta_state": self.current_state.meta_state,
            "policy": self.current_state.policy_name,
            "confidence": self.current_state.confidence,
            "transition_prob": self.current_state.transition_prob,
            "cusum": self.current_state.cusum_statistic,
            "current_vol": self.volatility_detector.get_current_vol(),
            "trend_strength": self.trend_detector.get_trend_strength(),
            "samples_since_transition": self.samples_since_transition,
            "reduce_exposure": self.should_reduce_exposure(),
        }
    
    def get_regime_distribution(self, 
                                lookback: int = 252) -> Dict[str, float]:
        """
        Get distribution of regimes over lookback period.
        
        Args:
            lookback: Number of samples to analyze
            
        Returns:
            Dict of regime → frequency
        """
        if not self.state_history:
            return {}
            
        recent = self.state_history[-lookback:]
        
        dist = {}
        for state in recent:
            meta = state.meta_state
            dist[meta] = dist.get(meta, 0) + 1
            
        total = len(recent)
        return {k: v / total for k, v in dist.items()}
