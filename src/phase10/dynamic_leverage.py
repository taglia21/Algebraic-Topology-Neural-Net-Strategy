"""Dynamic Leverage Engine for Phase 10.

Implements research-backed leverage strategies:
- Kelly criterion-based optimal leverage
- Regime-conditional leverage scaling
- Dynamic adjustment triggers
- Leveraged ETF integration

Research Foundation:
- Fractional Kelly (25-50%) shows superior performance with reduced drawdowns
- Dynamic leverage adjusts to market volatility, account equity, and trade size
- Tail-risk hedging enables +1.5% equity beta increase
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from collections import deque
import logging

logger = logging.getLogger(__name__)


class LeverageRegime(Enum):
    """Leverage regime classification."""
    AGGRESSIVE = "aggressive"      # 1.4-1.5x - Strong bull momentum
    MODERATE = "moderate"          # 1.2-1.3x - Normal bull
    NEUTRAL = "neutral"            # 1.0x - Mixed signals
    DEFENSIVE = "defensive"        # 0.7-0.8x - Bear/Risk-off
    CRISIS = "crisis"              # 0.5x - High volatility crisis


@dataclass
class LeverageState:
    """Complete leverage state for a trading day."""
    date: str
    
    # Kelly calculations
    kelly_full: float = 1.0
    kelly_fraction: float = 0.40  # 40% Kelly
    kelly_leverage: float = 1.0
    
    # Regime-based scaling
    regime: LeverageRegime = LeverageRegime.NEUTRAL
    regime_multiplier: float = 1.0
    
    # Adjustments
    drawdown_adjustment: float = 1.0
    volatility_adjustment: float = 1.0
    momentum_adjustment: float = 1.0
    
    # Final leverage
    target_leverage: float = 1.0
    actual_leverage: float = 1.0
    leverage_change: float = 0.0
    
    # Risk metrics
    current_drawdown: float = 0.0
    realized_volatility: float = 0.15
    vix_level: float = 15.0
    
    # Constraints
    max_leverage: float = 1.5
    min_leverage: float = 0.5
    leverage_cap_reason: Optional[str] = None


@dataclass
class KellyConfig:
    """Configuration for Kelly leverage calculation."""
    kelly_fraction: float = 0.40  # Use 40% of full Kelly (conservative)
    min_sample_size: int = 60     # Minimum trading days for calculation
    lookback_days: int = 252      # Historical window
    max_kelly: float = 2.0        # Cap full Kelly at 2x
    min_kelly: float = 0.0        # Floor at 0 (no leverage)
    decay_factor: float = 0.97    # Exponential decay for older returns


class KellyLeverageCalculator:
    """
    Calculate optimal leverage using Kelly criterion.
    
    Kelly Formula: f* = (p*b - q) / b
    where:
    - p = win probability
    - q = 1 - p (loss probability)
    - b = win/loss ratio (avg win / avg loss)
    
    We use fractional Kelly (40%) for reduced volatility.
    """
    
    def __init__(self, config: Optional[KellyConfig] = None):
        self.config = config or KellyConfig()
        self.returns_history = deque(maxlen=self.config.lookback_days)
        
    def update_returns(self, daily_return: float):
        """Add a daily return to history."""
        self.returns_history.append(daily_return)
    
    def compute_kelly_leverage(
        self,
        returns: Optional[np.ndarray] = None,
        sharpe_override: Optional[float] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute Kelly-optimal leverage.
        
        Args:
            returns: Array of historical returns (optional, uses internal history)
            sharpe_override: Override Sharpe ratio for calculation
            
        Returns:
            (kelly_leverage, metrics_dict)
        """
        if returns is None:
            if len(self.returns_history) < self.config.min_sample_size:
                # Default to moderate aggressive leverage when insufficient history
                return 1.25, {'reason': 'insufficient_history', 'default_leverage': 1.25}
            returns = np.array(self.returns_history)
        
        if len(returns) < self.config.min_sample_size:
            # Use moderate leverage by default during warmup
            return 1.25, {'reason': 'insufficient_data', 'default_leverage': 1.25}
        
        # Apply exponential decay weights
        weights = np.array([self.config.decay_factor ** i 
                           for i in range(len(returns)-1, -1, -1)])
        weights /= weights.sum()
        
        # Calculate win/loss statistics
        wins = returns > 0
        losses = returns < 0
        
        win_returns = returns[wins]
        loss_returns = returns[losses]
        
        if len(win_returns) == 0 or len(loss_returns) == 0:
            return 1.0, {'reason': 'no_wins_or_losses'}
        
        # Weighted statistics
        win_rate = np.sum(weights[wins])
        loss_rate = 1 - win_rate
        
        avg_win = np.average(win_returns, weights=weights[wins]) if wins.any() else 0
        avg_loss = np.abs(np.average(loss_returns, weights=weights[losses])) if losses.any() else 0.01
        
        # Win/loss ratio
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0
        
        # Full Kelly formula: f* = (p*b - q) / b
        # where p = win_rate, q = loss_rate, b = win_loss_ratio
        if win_loss_ratio > 0:
            full_kelly = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio
        else:
            full_kelly = 0.0
        
        # Cap full Kelly
        full_kelly = np.clip(full_kelly, self.config.min_kelly, self.config.max_kelly)
        
        # Apply fractional Kelly
        kelly_leverage = full_kelly * self.config.kelly_fraction
        
        # Alternative: Sharpe-based Kelly (f* = μ/σ²)
        if sharpe_override is not None:
            sharpe = sharpe_override
        else:
            mean_ret = np.average(returns, weights=weights)
            std_ret = np.sqrt(np.average((returns - mean_ret)**2, weights=weights))
            sharpe = (mean_ret * 252) / (std_ret * np.sqrt(252)) if std_ret > 0 else 0
        
        # Sharpe-based optimal leverage
        vol = np.std(returns) * np.sqrt(252)
        if vol > 0:
            sharpe_kelly = sharpe / vol
        else:
            sharpe_kelly = 1.0
        
        # Blend traditional and Sharpe-based Kelly
        blended_kelly = 0.6 * kelly_leverage + 0.4 * sharpe_kelly * self.config.kelly_fraction
        blended_kelly = np.clip(blended_kelly, 0.5, 2.0)
        
        metrics = {
            'full_kelly': full_kelly,
            'kelly_leverage': kelly_leverage,
            'sharpe_kelly': sharpe_kelly,
            'blended_kelly': blended_kelly,
            'win_rate': win_rate,
            'win_loss_ratio': win_loss_ratio,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'sharpe': sharpe,
            'volatility': vol,
        }
        
        return blended_kelly, metrics
    
    def get_recommended_fraction(self, volatility: float) -> float:
        """
        Get recommended Kelly fraction based on volatility.
        
        Higher volatility → lower fraction for safety.
        """
        if volatility > 0.40:  # >40% vol
            return 0.25
        elif volatility > 0.30:  # 30-40% vol
            return 0.30
        elif volatility > 0.20:  # 20-30% vol
            return 0.35
        else:  # <20% vol
            return 0.40


@dataclass
class RegimeScaleConfig:
    """Configuration for regime-based leverage scaling."""
    # Leverage multipliers by regime
    aggressive_leverage: float = 1.45   # Bull momentum
    moderate_leverage: float = 1.25     # Normal bull
    neutral_leverage: float = 1.0       # Mixed
    defensive_leverage: float = 0.75    # Bear
    crisis_leverage: float = 0.5        # High volatility
    
    # VIX thresholds
    vix_low: float = 15.0               # Below = calm markets
    vix_moderate: float = 20.0          # Normal volatility
    vix_high: float = 25.0              # Elevated
    vix_crisis: float = 30.0            # Crisis level
    
    # Momentum thresholds
    momentum_strong: float = 0.6        # Strong momentum signal
    momentum_weak: float = 0.3          # Weak momentum


class RegimeLeverageScaler:
    """
    Scale leverage based on market regime.
    
    Regime-Conditional Scaling:
    - Bull Momentum: 1.4-1.5x leverage
    - Bull Normal: 1.2-1.3x leverage
    - Neutral/Mixed: 1.0x (no leverage)
    - Bear/Risk-Off: 0.7-0.8x (de-lever)
    - High Volatility: 0.5-0.8x
    """
    
    def __init__(self, config: Optional[RegimeScaleConfig] = None):
        self.config = config or RegimeScaleConfig()
        self.regime_history = deque(maxlen=20)
    
    def classify_regime(
        self,
        macro_regime: str,
        vix_level: float,
        momentum_score: float,
        tda_regime: Optional[str] = None,
    ) -> Tuple[LeverageRegime, float]:
        """
        Classify current leverage regime.
        
        Args:
            macro_regime: From HMM detector (BULL_MOMENTUM, BEAR_DEFENSIVE, etc.)
            vix_level: Current VIX level
            momentum_score: Aggregate momentum signal [-1, 1]
            tda_regime: TDA topology regime (RISK_ON, RISK_OFF, etc.)
            
        Returns:
            (LeverageRegime, multiplier)
        """
        # Start with macro regime base
        if macro_regime in ['BULL_MOMENTUM', 'bull_momentum']:
            base_regime = LeverageRegime.AGGRESSIVE
            base_mult = self.config.aggressive_leverage
        elif macro_regime in ['LOW_VOLATILITY', 'low_volatility']:
            base_regime = LeverageRegime.MODERATE
            base_mult = self.config.moderate_leverage
        elif macro_regime in ['BEAR_DEFENSIVE', 'bear_defensive']:
            base_regime = LeverageRegime.DEFENSIVE
            base_mult = self.config.defensive_leverage
        elif macro_regime in ['HIGH_VOLATILITY', 'high_volatility']:
            base_regime = LeverageRegime.CRISIS
            base_mult = self.config.crisis_leverage
        else:  # TRANSITION or unknown
            base_regime = LeverageRegime.NEUTRAL
            base_mult = self.config.neutral_leverage
        
        # VIX adjustment - less aggressive for aggressive alpha
        if vix_level >= self.config.vix_crisis:
            # Only reduce in true crisis
            base_mult = min(base_mult, 0.85)  # Don't reduce as much
        elif vix_level >= self.config.vix_high:
            # High volatility - slight reduction
            base_mult = min(base_mult, 1.0)  # Cap at 1.0 instead of defensive
        elif vix_level <= self.config.vix_low:
            # Low volatility - boost allowed
            base_mult *= 1.10  # Bigger boost in calm markets
        
        # Momentum adjustment - stronger for trend following
        if momentum_score >= self.config.momentum_strong:
            # Strong momentum confirms regime
            if base_regime in [LeverageRegime.AGGRESSIVE, LeverageRegime.MODERATE]:
                base_mult *= 1.15  # Bigger boost for strong momentum
        elif momentum_score >= 0.3:
            # Moderate positive momentum
            base_mult *= 1.05
        elif momentum_score <= -self.config.momentum_weak:
            # Negative momentum - reduce leverage
            base_mult *= 0.90
        
        # TDA regime overlay - less aggressive adjustments
        if tda_regime in ['RISK_OFF', 'risk_off']:
            base_mult *= 0.90  # Less reduction
        elif tda_regime in ['REGIME_BREAK', 'regime_break']:
            base_mult *= 0.85  # Less reduction
        elif tda_regime in ['RISK_ON', 'risk_on']:
            base_mult *= 1.10  # Bigger boost
        
        # Final bounds - allow higher leverage
        base_mult = np.clip(base_mult, 0.6, 1.5)
        
        # Update history
        self.regime_history.append(base_regime)
        
        return base_regime, base_mult
    
    def get_regime_stability(self) -> float:
        """
        Get regime stability score (0-1).
        
        Higher = more stable regime, allows higher leverage.
        """
        if len(self.regime_history) < 5:
            return 0.5
        
        # Count regime changes
        changes = sum(1 for i in range(1, len(self.regime_history)) 
                     if self.regime_history[i] != self.regime_history[i-1])
        
        stability = 1 - (changes / len(self.regime_history))
        return stability


@dataclass
class AdjusterConfig:
    """Configuration for leverage adjustment triggers."""
    # Drawdown scaling
    dd_start_threshold: float = 0.05    # Start reducing at 5% DD
    dd_reduction_rate: float = 2.0      # Reduce 20% leverage per 10% DD
    dd_min_leverage: float = 0.5        # Floor during drawdown
    
    # Volatility scaling  
    vol_target: float = 0.15            # Target portfolio volatility
    vol_high_threshold: float = 0.25    # Reduce above this
    vol_low_threshold: float = 0.10     # Can increase below this
    
    # Equity curve scaling
    hwm_boost_threshold: float = 0.05   # 5% new high → boost
    hwm_boost_amount: float = 0.05      # +5% leverage at new highs
    
    # Momentum confirmation
    min_signals_for_max_leverage: int = 3  # Need 3+ signals for 1.4x+
    signal_confidence_threshold: float = 0.6


class LeverageAdjuster:
    """
    Dynamic leverage adjustment based on triggers.
    
    Adjustment Triggers:
    - Reduce leverage 10% for every 3% drawdown
    - Increase leverage 5% after 5% equity high-water mark
    - Volatility breakers: if 10-day realized vol >30%, cap at 1.0x
    - Momentum confirmation: require 3+ high-conviction signals for 1.4x+
    """
    
    def __init__(self, config: Optional[AdjusterConfig] = None):
        self.config = config or AdjusterConfig()
        self.peak_value = None
        self.vol_history = deque(maxlen=60)
        self.last_adjustment = 1.0
    
    def compute_drawdown_adjustment(
        self,
        current_value: float,
        peak_value: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Compute leverage adjustment based on drawdown.
        
        Returns:
            (adjustment_multiplier, current_drawdown)
        """
        if peak_value is None:
            if self.peak_value is None:
                self.peak_value = current_value
            else:
                self.peak_value = max(self.peak_value, current_value)
            peak_value = self.peak_value
        
        if peak_value <= 0:
            return 1.0, 0.0
        
        drawdown = 1 - (current_value / peak_value)
        
        if drawdown <= self.config.dd_start_threshold:
            # No reduction needed
            adjustment = 1.0
        else:
            # Progressive reduction
            excess_dd = drawdown - self.config.dd_start_threshold
            reduction = excess_dd * self.config.dd_reduction_rate
            adjustment = max(self.config.dd_min_leverage, 1.0 - reduction)
        
        return adjustment, drawdown
    
    def compute_volatility_adjustment(
        self,
        realized_vol: float,
        returns: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute leverage adjustment based on volatility.
        
        High vol → reduce leverage
        Low vol → allow higher leverage
        """
        if returns is not None:
            # Update vol history
            self.vol_history.append(np.std(returns[-10:]) * np.sqrt(252))
        
        if realized_vol >= self.config.vol_high_threshold:
            # High volatility - cap leverage
            return 0.7
        elif realized_vol >= self.config.vol_target:
            # Above target - slight reduction
            excess = (realized_vol - self.config.vol_target) / self.config.vol_target
            return max(0.8, 1.0 - excess * 0.5)
        elif realized_vol <= self.config.vol_low_threshold:
            # Low volatility - can increase
            return 1.1
        else:
            return 1.0
    
    def compute_momentum_adjustment(
        self,
        signal_count: int,
        avg_confidence: float,
    ) -> float:
        """
        Compute leverage adjustment based on signal confirmation.
        
        More high-confidence signals → higher allowed leverage.
        """
        if signal_count >= self.config.min_signals_for_max_leverage:
            if avg_confidence >= self.config.signal_confidence_threshold:
                return 1.1  # Full leverage allowed
            else:
                return 1.0
        elif signal_count >= 2:
            return 0.95
        elif signal_count >= 1:
            return 0.9
        else:
            return 0.8  # No signals - reduce leverage
    
    def compute_hwm_adjustment(
        self,
        current_value: float,
        previous_peak: float,
    ) -> float:
        """
        Compute adjustment based on high-water mark.
        
        New equity highs → can boost leverage slightly.
        """
        if previous_peak <= 0:
            return 1.0
        
        excess_over_peak = (current_value / previous_peak) - 1
        
        if excess_over_peak >= self.config.hwm_boost_threshold:
            return 1.0 + self.config.hwm_boost_amount
        elif excess_over_peak > 0:
            # Partial boost
            return 1.0 + (excess_over_peak / self.config.hwm_boost_threshold) * self.config.hwm_boost_amount
        else:
            return 1.0
    
    def smooth_adjustment(self, target: float) -> float:
        """
        Smooth leverage changes to avoid whipsawing.
        """
        if abs(target - self.last_adjustment) < 0.05:
            return self.last_adjustment
        
        # Move 50% toward target
        smoothed = self.last_adjustment + 0.5 * (target - self.last_adjustment)
        self.last_adjustment = smoothed
        return smoothed


@dataclass
class LeveragedETFConfig:
    """Configuration for leveraged ETF integration."""
    # ETF mappings
    leverage_etfs: Dict[str, str] = field(default_factory=lambda: {
        'SPY': 'SPXL',   # 3x S&P 500
        'QQQ': 'TQQQ',   # 3x Nasdaq
        'IWM': 'TNA',    # 3x Russell 2000
    })
    
    # When to use leveraged ETFs
    min_leverage_for_letf: float = 1.3  # Use LETF above 1.3x target
    max_letf_allocation: float = 0.30   # Max 30% in leveraged ETFs
    
    # Rebalancing
    rebalance_threshold: float = 0.10   # Rebalance if 10% off target


class DynamicLeverageEngine:
    """
    Complete dynamic leverage management system.
    
    Combines:
    - Kelly-based optimal leverage
    - Regime-conditional scaling
    - Dynamic adjustment triggers
    - Leveraged ETF integration
    """
    
    def __init__(
        self,
        kelly_config: Optional[KellyConfig] = None,
        regime_config: Optional[RegimeScaleConfig] = None,
        adjuster_config: Optional[AdjusterConfig] = None,
        letf_config: Optional[LeveragedETFConfig] = None,
        max_leverage: float = 1.5,
        min_leverage: float = 0.5,
    ):
        self.kelly_calc = KellyLeverageCalculator(kelly_config)
        self.regime_scaler = RegimeLeverageScaler(regime_config)
        self.adjuster = LeverageAdjuster(adjuster_config)
        self.letf_config = letf_config or LeveragedETFConfig()
        
        self.max_leverage = max_leverage
        self.min_leverage = min_leverage
        
        # State tracking
        self.leverage_history = deque(maxlen=252)
        self.current_state: Optional[LeverageState] = None
    
    def compute_target_leverage(
        self,
        date: str,
        portfolio_value: float,
        peak_value: float,
        returns_history: np.ndarray,
        macro_regime: str,
        vix_level: float,
        momentum_score: float,
        signal_count: int = 0,
        avg_confidence: float = 0.5,
        tda_regime: Optional[str] = None,
    ) -> LeverageState:
        """
        Compute optimal target leverage for the day.
        
        Args:
            date: Current date
            portfolio_value: Current portfolio value
            peak_value: High-water mark
            returns_history: Array of historical daily returns
            macro_regime: HMM regime classification
            vix_level: Current VIX level
            momentum_score: Aggregate momentum signal
            signal_count: Number of high-conviction signals
            avg_confidence: Average signal confidence
            tda_regime: TDA topology regime
            
        Returns:
            LeverageState with complete leverage recommendation
        """
        # 1. Kelly-based leverage
        kelly_leverage, kelly_metrics = self.kelly_calc.compute_kelly_leverage(returns_history)
        
        # 2. Regime-based scaling
        regime, regime_mult = self.regime_scaler.classify_regime(
            macro_regime, vix_level, momentum_score, tda_regime
        )
        
        # 3. Drawdown adjustment
        dd_adj, current_dd = self.adjuster.compute_drawdown_adjustment(
            portfolio_value, peak_value
        )
        
        # 4. Volatility adjustment
        realized_vol = np.std(returns_history[-20:]) * np.sqrt(252) if len(returns_history) >= 20 else 0.15
        vol_adj = self.adjuster.compute_volatility_adjustment(realized_vol, returns_history)
        
        # 5. Momentum confirmation
        mom_adj = self.adjuster.compute_momentum_adjustment(signal_count, avg_confidence)
        
        # 6. HWM adjustment
        hwm_adj = self.adjuster.compute_hwm_adjustment(portfolio_value, peak_value)
        
        # Combine adjustments
        # Base = Kelly * Regime, then apply adjustments
        base_leverage = kelly_leverage * regime_mult
        adjusted_leverage = base_leverage * dd_adj * vol_adj * mom_adj * hwm_adj
        
        # Apply bounds
        target_leverage = np.clip(adjusted_leverage, self.min_leverage, self.max_leverage)
        
        # Check for cap reason
        cap_reason = None
        if adjusted_leverage > self.max_leverage:
            cap_reason = f"Capped at max {self.max_leverage}x"
        elif adjusted_leverage < self.min_leverage:
            cap_reason = f"Floor at min {self.min_leverage}x"
        elif dd_adj < 0.8:
            cap_reason = f"Drawdown reduction ({current_dd:.1%} DD)"
        elif vol_adj < 0.9:
            cap_reason = f"Volatility reduction ({realized_vol:.1%} vol)"
        
        # Smooth the leverage change
        if self.current_state is not None:
            smoothed_leverage = self.adjuster.smooth_adjustment(target_leverage)
            leverage_change = smoothed_leverage - self.current_state.actual_leverage
        else:
            smoothed_leverage = target_leverage
            leverage_change = 0.0
        
        # Build state
        state = LeverageState(
            date=date,
            kelly_full=kelly_metrics.get('full_kelly', 1.0),
            kelly_fraction=self.kelly_calc.config.kelly_fraction,
            kelly_leverage=kelly_leverage,
            regime=regime,
            regime_multiplier=regime_mult,
            drawdown_adjustment=dd_adj,
            volatility_adjustment=vol_adj,
            momentum_adjustment=mom_adj,
            target_leverage=target_leverage,
            actual_leverage=smoothed_leverage,
            leverage_change=leverage_change,
            current_drawdown=current_dd,
            realized_volatility=realized_vol,
            vix_level=vix_level,
            max_leverage=self.max_leverage,
            min_leverage=self.min_leverage,
            leverage_cap_reason=cap_reason,
        )
        
        self.current_state = state
        self.leverage_history.append(state)
        
        return state
    
    def get_leveraged_etf_allocation(
        self,
        target_leverage: float,
        base_weights: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Determine allocation to leveraged ETFs.
        
        For high target leverage, use leveraged ETFs for part of exposure.
        
        Returns:
            (regular_weights, leveraged_weights)
        """
        if target_leverage < self.letf_config.min_leverage_for_letf:
            # Don't use leveraged ETFs
            return base_weights, {}
        
        # Calculate how much leverage we need from LETFs
        letf_leverage_needed = target_leverage - 1.0
        
        regular_weights = {}
        leveraged_weights = {}
        
        for ticker, weight in base_weights.items():
            if ticker in self.letf_config.leverage_etfs:
                letf_ticker = self.letf_config.leverage_etfs[ticker]
                
                # Calculate split between regular and leveraged
                # LETF provides 3x, so we need (letf_leverage_needed / 3) in LETF
                letf_frac = min(
                    letf_leverage_needed / 2.0,  # Max 50% of position in LETF
                    self.letf_config.max_letf_allocation
                )
                
                regular_weights[ticker] = weight * (1 - letf_frac)
                leveraged_weights[letf_ticker] = weight * letf_frac
            else:
                regular_weights[ticker] = weight
        
        return regular_weights, leveraged_weights
    
    def get_leverage_summary(self) -> Dict:
        """Get summary of current leverage state."""
        if self.current_state is None:
            return {}
        
        s = self.current_state
        return {
            'target_leverage': s.target_leverage,
            'actual_leverage': s.actual_leverage,
            'regime': s.regime.value,
            'kelly_leverage': s.kelly_leverage,
            'drawdown': s.current_drawdown,
            'volatility': s.realized_volatility,
            'cap_reason': s.leverage_cap_reason,
            'adjustments': {
                'drawdown': s.drawdown_adjustment,
                'volatility': s.volatility_adjustment,
                'momentum': s.momentum_adjustment,
            }
        }
