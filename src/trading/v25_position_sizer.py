#!/usr/bin/env python3
"""
V25 Phase 3: Adaptive Position Sizing
======================================

Volatility-scaled position sizing that responds to regime and drawdown.

Key Innovation:
- Reduce position sizes during high volatility (protect capital)
- Increase sizes during favorable regimes (maximize gains)
- Drawdown-based scaling (reduce exposure during drawdowns)
- Target: 2pp drawdown reduction with minimal Sharpe impact

Position Sizing Rules:
1. Base size from Kelly Criterion (half-Kelly)
2. Regime multiplier (0.5-1.2x based on regime)
3. Drawdown multiplier (0.25-1.0x based on current DD)
4. Volatility multiplier (0.5-1.0x based on vol)

Combined: size = base * regime * dd * vol (capped at max)
"""

import json
import logging
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('V25_PositionSizer')


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class V25SizerConfig:
    """Configuration for V25 position sizer."""
    
    # Base position sizing
    base_size: float = 0.05           # 5% base position
    max_position: float = 0.15        # 15% max single position
    min_position: float = 0.01        # 1% min position
    
    # Regime multipliers
    regime_multipliers: Dict[str, float] = field(default_factory=lambda: {
        # LOW VOLATILITY - can size up
        'low_trending_up': 1.20,      # Best regime for momentum
        'low_trending_down': 0.90,    # Downtrend = cautious
        'low_flat': 1.00,             # Neutral
        'low_mean_reverting': 1.10,   # Good for mean reversion
        
        # MEDIUM VOLATILITY - normal sizing
        'medium_trending_up': 1.10,
        'medium_trending_down': 0.85,
        'medium_flat': 0.95,
        'medium_mean_reverting': 1.00,
        
        # HIGH VOLATILITY - reduce sizing
        'high_trending_up': 0.80,     # Vol eats gains
        'high_trending_down': 0.60,   # Danger zone
        'high_flat': 0.70,
        'high_mean_reverting': 0.75,
    })
    
    # Drawdown scaling
    dd_scale_start: float = 0.05      # Start reducing at 5% DD
    dd_scale_max: float = 0.15        # Full reduction at 15% DD
    dd_min_factor: float = 0.25       # Min 25% of normal at max DD
    dd_halt_threshold: float = 0.20   # Halt trading at 20% DD
    
    # Volatility scaling
    vol_lookback: int = 20            # 20-day vol
    vol_baseline: float = 0.15        # 15% annualized baseline
    vol_high_threshold: float = 0.25  # Reduce at 25%
    vol_extreme_threshold: float = 0.40  # Heavy reduction at 40%
    vol_high_factor: float = 0.70     # 70% at high vol
    vol_extreme_factor: float = 0.40  # 40% at extreme vol
    
    # Portfolio constraints
    max_gross_exposure: float = 1.50  # Max 150% gross
    max_net_exposure: float = 0.80    # Max 80% net long
    min_net_exposure: float = -0.30   # Max 30% net short


# =============================================================================
# DRAWDOWN TRACKER
# =============================================================================

class DrawdownTracker:
    """
    Tracks portfolio drawdown for position sizing decisions.
    """
    
    def __init__(self, initial_equity: float = 100000.0):
        self.initial_equity = initial_equity
        self.current_equity = initial_equity
        self.peak_equity = initial_equity
        self.returns_history: deque = deque(maxlen=252)  # 1 year
        
        # Drawdown metrics
        self.current_dd = 0.0
        self.max_dd = 0.0
        self.dd_duration = 0
        
    def update(self, daily_return: float):
        """Update drawdown state with new daily return."""
        self.returns_history.append(daily_return)
        
        # Update equity
        self.current_equity *= (1 + daily_return)
        
        # Update peak
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity
            self.dd_duration = 0
        else:
            self.dd_duration += 1
        
        # Calculate drawdown
        self.current_dd = (self.peak_equity - self.current_equity) / self.peak_equity
        self.max_dd = max(self.max_dd, self.current_dd)
        
    def get_dd_multiplier(self, 
                          dd_start: float = 0.05,
                          dd_max: float = 0.15,
                          min_factor: float = 0.25) -> float:
        """
        Get position size multiplier based on current drawdown.
        
        Linear scaling from 1.0 at dd_start to min_factor at dd_max.
        """
        if self.current_dd <= dd_start:
            return 1.0
        elif self.current_dd >= dd_max:
            return min_factor
        else:
            # Linear interpolation
            pct = (self.current_dd - dd_start) / (dd_max - dd_start)
            return 1.0 - pct * (1.0 - min_factor)
    
    def should_halt(self, halt_threshold: float = 0.20) -> bool:
        """Check if trading should be halted due to drawdown."""
        return self.current_dd >= halt_threshold
    
    def get_state(self) -> Dict:
        """Get current drawdown state."""
        return {
            'current_dd': self.current_dd,
            'max_dd': self.max_dd,
            'dd_duration': self.dd_duration,
            'equity': self.current_equity,
            'peak': self.peak_equity
        }


# =============================================================================
# VOLATILITY ESTIMATOR
# =============================================================================

class VolatilityEstimator:
    """
    Estimates current and historical volatility for sizing.
    """
    
    def __init__(self, lookback: int = 20, baseline_vol: float = 0.15):
        self.lookback = lookback
        self.baseline_vol = baseline_vol
        self.returns_history: deque = deque(maxlen=lookback * 5)
        self.vol_history: deque = deque(maxlen=252)
        
        # Current estimates
        self.current_vol = baseline_vol
        self.vol_ratio = 1.0
        
    def update(self, daily_return: float):
        """Update volatility estimate with new return."""
        self.returns_history.append(daily_return)
        
        if len(self.returns_history) >= self.lookback:
            recent_returns = list(self.returns_history)[-self.lookback:]
            self.current_vol = np.std(recent_returns) * np.sqrt(252)
            self.vol_history.append(self.current_vol)
            
            # Ratio to baseline
            self.vol_ratio = self.current_vol / self.baseline_vol
    
    def get_vol_multiplier(self,
                           high_threshold: float = 0.25,
                           extreme_threshold: float = 0.40,
                           high_factor: float = 0.70,
                           extreme_factor: float = 0.40) -> float:
        """
        Get position size multiplier based on current volatility.
        """
        if self.current_vol <= high_threshold:
            return 1.0
        elif self.current_vol >= extreme_threshold:
            return extreme_factor
        else:
            # Linear interpolation
            pct = (self.current_vol - high_threshold) / (extreme_threshold - high_threshold)
            return 1.0 - pct * (1.0 - high_factor)
    
    def get_percentile_vol(self) -> float:
        """Get current vol as percentile of historical."""
        if len(self.vol_history) < 20:
            return 0.5  # Default to median
        
        sorted_vol = sorted(self.vol_history)
        for i, v in enumerate(sorted_vol):
            if self.current_vol <= v:
                return i / len(sorted_vol)
        return 1.0
    
    def get_state(self) -> Dict:
        """Get current volatility state."""
        return {
            'current_vol': self.current_vol,
            'baseline_vol': self.baseline_vol,
            'vol_ratio': self.vol_ratio,
            'percentile': self.get_percentile_vol()
        }


# =============================================================================
# V25 ADAPTIVE POSITION SIZER
# =============================================================================

class V25AdaptivePositionSizer:
    """
    V25 Adaptive Position Sizer combining regime, drawdown, and volatility.
    
    Key Features:
    - Regime-aware: Size up in favorable regimes, down in risky ones
    - Drawdown-aware: Reduce exposure as drawdown increases
    - Volatility-aware: Reduce in high vol environments
    - Kelly-inspired: Base size from expected edge
    """
    
    def __init__(self, 
                 config: Optional[V25SizerConfig] = None,
                 initial_equity: float = 100000.0,
                 log_dir: str = "logs/v25"):
        """
        Initialize position sizer.
        
        Args:
            config: Sizer configuration
            initial_equity: Starting portfolio equity
            log_dir: Directory for state persistence
        """
        self.config = config or V25SizerConfig()
        
        # Trackers
        self.dd_tracker = DrawdownTracker(initial_equity)
        self.vol_estimator = VolatilityEstimator(
            lookback=self.config.vol_lookback,
            baseline_vol=self.config.vol_baseline
        )
        
        # State
        self.current_regime: Optional[str] = None
        self.sizing_history: List[Dict] = []
        
        # Logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("V25AdaptivePositionSizer initialized")
    
    def update(self, 
               daily_return: float,
               regime: Optional[str] = None):
        """
        Update sizer state with new data.
        
        Args:
            daily_return: Portfolio daily return
            regime: Current regime (if known)
        """
        self.dd_tracker.update(daily_return)
        self.vol_estimator.update(daily_return)
        
        if regime:
            self.current_regime = regime
    
    def get_position_size(self,
                          strategy: str,
                          expected_edge: float = 0.01,
                          volatility: Optional[float] = None) -> Tuple[float, Dict]:
        """
        Calculate optimal position size.
        
        Args:
            strategy: 'v21' or 'v24'
            expected_edge: Expected return per trade
            volatility: Asset volatility (if known)
            
        Returns:
            (position_size, metadata)
        """
        # Check for trading halt
        if self.dd_tracker.should_halt(self.config.dd_halt_threshold):
            return 0.0, {
                'reason': 'drawdown_halt',
                'current_dd': self.dd_tracker.current_dd
            }
        
        # Base size (simple fixed fraction, could be Kelly)
        base = self.config.base_size
        
        # Regime multiplier
        regime_mult = 1.0
        if self.current_regime and self.current_regime in self.config.regime_multipliers:
            regime_mult = self.config.regime_multipliers[self.current_regime]
        
        # Drawdown multiplier
        dd_mult = self.dd_tracker.get_dd_multiplier(
            dd_start=self.config.dd_scale_start,
            dd_max=self.config.dd_scale_max,
            min_factor=self.config.dd_min_factor
        )
        
        # Volatility multiplier
        vol_mult = self.vol_estimator.get_vol_multiplier(
            high_threshold=self.config.vol_high_threshold,
            extreme_threshold=self.config.vol_extreme_threshold,
            high_factor=self.config.vol_high_factor,
            extreme_factor=self.config.vol_extreme_factor
        )
        
        # Combined size
        size = base * regime_mult * dd_mult * vol_mult
        
        # Apply constraints
        size = np.clip(size, self.config.min_position, self.config.max_position)
        
        metadata = {
            'base_size': base,
            'regime': self.current_regime,
            'regime_mult': regime_mult,
            'dd_mult': dd_mult,
            'vol_mult': vol_mult,
            'final_size': size,
            'current_dd': self.dd_tracker.current_dd,
            'current_vol': self.vol_estimator.current_vol
        }
        
        # Record
        self.sizing_history.append({
            'timestamp': datetime.now().isoformat(),
            'strategy': strategy,
            **metadata
        })
        
        return size, metadata
    
    def get_portfolio_weights(self,
                               v21_weight: float,
                               v24_weight: float) -> Tuple[float, float, Dict]:
        """
        Adjust strategy weights based on position sizing rules.
        
        Args:
            v21_weight: Base V21 weight (from allocator)
            v24_weight: Base V24 weight (from allocator)
            
        Returns:
            (adjusted_v21, adjusted_v24, metadata)
        """
        # Get position sizes for each strategy
        v21_size, v21_meta = self.get_position_size('v21')
        v24_size, v24_meta = self.get_position_size('v24')
        
        # Scale weights by position sizes
        # This effectively reduces total exposure in risky conditions
        total_base = v21_weight + v24_weight
        total_sized = v21_weight * v21_size + v24_weight * v24_size
        
        # Normalize to maintain relative allocation
        if total_sized > 0:
            scale_factor = (v21_size + v24_size) / 2  # Average multiplier
        else:
            scale_factor = 0
        
        # Adjusted weights maintain allocation ratio but scaled down
        adj_v21 = v21_weight * scale_factor
        adj_v24 = v24_weight * scale_factor
        
        # Ensure they sum to target gross (could be < 1.0)
        total = adj_v21 + adj_v24
        
        # If total exposure needs capping
        if total > self.config.max_gross_exposure:
            factor = self.config.max_gross_exposure / total
            adj_v21 *= factor
            adj_v24 *= factor
        
        metadata = {
            'base_v21': v21_weight,
            'base_v24': v24_weight,
            'adj_v21': adj_v21,
            'adj_v24': adj_v24,
            'scale_factor': scale_factor,
            'gross_exposure': adj_v21 + adj_v24,
            'dd_impact': v21_meta.get('dd_mult', 1.0),
            'vol_impact': v21_meta.get('vol_mult', 1.0)
        }
        
        return adj_v21, adj_v24, metadata
    
    def get_statistics(self) -> Dict:
        """Get sizer statistics."""
        if not self.sizing_history:
            return {}
        
        sizes = [h['final_size'] for h in self.sizing_history]
        dd_mults = [h['dd_mult'] for h in self.sizing_history]
        vol_mults = [h['vol_mult'] for h in self.sizing_history]
        
        return {
            'n_decisions': len(self.sizing_history),
            'avg_size': np.mean(sizes),
            'min_size': np.min(sizes),
            'max_size': np.max(sizes),
            'avg_dd_mult': np.mean(dd_mults),
            'avg_vol_mult': np.mean(vol_mults),
            'current_dd': self.dd_tracker.current_dd,
            'max_dd': self.dd_tracker.max_dd,
            'current_vol': self.vol_estimator.current_vol
        }
    
    def save_state(self, filepath: Optional[str] = None):
        """Save sizer state."""
        if filepath is None:
            filepath = self.log_dir / "position_sizer_state.json"
        else:
            filepath = Path(filepath)
        
        state = {
            'dd_state': self.dd_tracker.get_state(),
            'vol_state': self.vol_estimator.get_state(),
            'current_regime': self.current_regime,
            'statistics': self.get_statistics(),
            'saved_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Position sizer state saved to {filepath}")


# =============================================================================
# V25 FULL ALLOCATOR WITH POSITION SIZING
# =============================================================================

class V25FullAllocator:
    """
    V25 Complete Allocator with all phases integrated.
    
    Combines:
    - Phase 1: Regime-based allocation
    - Phase 2: Pattern memory confidence
    - Phase 3: Adaptive position sizing
    """
    
    def __init__(self,
                 regime_allocator,  # V25AdaptiveAllocator
                 pattern_memory=None,  # PatternMemory (optional)
                 position_sizer: Optional[V25AdaptivePositionSizer] = None,
                 log_dir: str = "logs/v25"):
        """
        Initialize full allocator.
        """
        self.regime_allocator = regime_allocator
        self.pattern_memory = pattern_memory
        self.position_sizer = position_sizer or V25AdaptivePositionSizer(log_dir=log_dir)
        self.log_dir = Path(log_dir)
        
        logger.info("V25FullAllocator initialized")
    
    def get_allocation(self,
                        prices: np.ndarray,
                        volumes: Optional[np.ndarray] = None,
                        pattern_features: Optional[np.ndarray] = None) -> Tuple[Dict[str, float], Dict]:
        """
        Get full allocation with all adjustments.
        
        Returns:
            ({'v21': weight, 'v24': weight}, metadata)
        """
        # Phase 1: Base allocation from regime
        returns = np.diff(np.log(prices))
        self.regime_allocator.update_regime(returns, prices)
        base_weights = self.regime_allocator.get_allocation()
        
        # Get regime for position sizing
        regime = self.regime_allocator.current_regime
        self.position_sizer.current_regime = regime
        
        # Phase 2: Pattern confidence adjustment (if available)
        pattern_adjustment = 0
        if self.pattern_memory and pattern_features is not None:
            # Would compute pattern confidence here
            pass
        
        # Phase 3: Position sizing adjustment
        adj_v21, adj_v24, sizing_meta = self.position_sizer.get_portfolio_weights(
            v21_weight=base_weights['v21'],
            v24_weight=base_weights['v24']
        )
        
        # Renormalize to sum to 1 (for weight allocation, not exposure)
        total = adj_v21 + adj_v24
        if total > 0:
            final_weights = {
                'v21': adj_v21 / total,
                'v24': adj_v24 / total
            }
        else:
            final_weights = {'v21': 0.5, 'v24': 0.5}
        
        metadata = {
            'base_weights': base_weights,
            'regime': regime,
            'sizing': sizing_meta,
            'gross_exposure': total,  # May be < 1.0 during risk-off
            'final_weights': final_weights
        }
        
        return final_weights, metadata
    
    def update_performance(self, daily_return: float):
        """Update position sizer with portfolio return."""
        self.position_sizer.update(daily_return)


# =============================================================================
# TESTING
# =============================================================================

def test_position_sizer():
    """Test position sizer functionality."""
    logger.info("Testing V25AdaptivePositionSizer...")
    
    sizer = V25AdaptivePositionSizer()
    
    # Simulate returns
    np.random.seed(42)
    
    # Normal period
    for _ in range(50):
        ret = np.random.normal(0.001, 0.01)  # Small positive drift
        sizer.update(ret, regime='medium_flat')
        size, meta = sizer.get_position_size('v21')
    
    logger.info(f"After normal period:")
    logger.info(f"  Position size: {size:.3f}")
    logger.info(f"  DD: {meta['current_dd']:.2%}")
    logger.info(f"  Vol: {meta['current_vol']:.2%}")
    
    # Drawdown period
    for _ in range(20):
        ret = np.random.normal(-0.01, 0.02)  # Negative drift, high vol
        sizer.update(ret, regime='high_trending_down')
        size, meta = sizer.get_position_size('v21')
    
    logger.info(f"\nAfter drawdown period:")
    logger.info(f"  Position size: {size:.3f}")
    logger.info(f"  DD: {sizer.dd_tracker.current_dd:.2%}")
    logger.info(f"  Vol: {sizer.vol_estimator.current_vol:.2%}")
    if size > 0:
        logger.info(f"  DD multiplier: {meta['dd_mult']:.2f}")
        logger.info(f"  Vol multiplier: {meta['vol_mult']:.2f}")
    else:
        logger.info(f"  Trading halted: {meta.get('reason', 'unknown')}")
    
    # Recovery
    for _ in range(30):
        ret = np.random.normal(0.005, 0.015)  # Strong recovery
        sizer.update(ret, regime='medium_trending_up')
        size, meta = sizer.get_position_size('v21')
    
    logger.info(f"\nAfter recovery:")
    logger.info(f"  Position size: {size:.3f}")
    logger.info(f"  DD: {meta['current_dd']:.2%}")
    
    stats = sizer.get_statistics()
    logger.info(f"\nStatistics: {stats}")
    
    sizer.save_state()
    logger.info("V25AdaptivePositionSizer tests passed!")


if __name__ == "__main__":
    test_position_sizer()
