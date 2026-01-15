"""
Adaptive Risk Manager for Phase 12
===================================

Regime-aware risk management with:
- Daily loss limits
- Regime transition protection
- VIX-based circuit breakers
- Drawdown-adjusted position sizing
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .regime_classifier import RegimeState

logger = logging.getLogger(__name__)


class RiskState(Enum):
    """Portfolio risk states."""
    NORMAL = "normal"
    CAUTION = "caution"
    DEFENSIVE = "defensive"
    CRISIS = "crisis"


@dataclass
class RiskMetrics:
    """Current risk metrics snapshot."""
    portfolio_drawdown: float
    daily_pnl: float
    consecutive_losses: int
    vix_level: float
    regime: RegimeState
    risk_state: RiskState
    position_scale: float
    circuit_breaker_active: bool


@dataclass
class RiskConfig:
    """Risk management configuration."""
    # Daily loss limits
    daily_loss_limit: float = -0.03  # -3% triggers deleveraging
    severe_daily_loss: float = -0.05  # -5% triggers full exit
    
    # Consecutive loss handling
    max_consecutive_losses: int = 3
    loss_streak_scale: float = 0.50
    
    # Drawdown thresholds
    drawdown_caution: float = 0.05
    drawdown_defensive: float = 0.10
    drawdown_crisis: float = 0.15
    
    # VIX thresholds
    vix_caution: float = 25
    vix_defensive: float = 35
    vix_crisis: float = 45
    
    # Recovery parameters
    recovery_days_required: int = 2
    min_position_scale: float = 0.10


class AdaptiveRiskManager:
    """
    Manages portfolio risk adaptively based on regime and conditions.
    
    Key features:
    - Scales position size based on risk state
    - Implements circuit breakers for extreme conditions
    - Handles regime transition risk
    - Tracks consecutive losses
    """
    
    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()
        
        # State tracking
        self.risk_state = RiskState.NORMAL
        self.consecutive_losses = 0
        self.days_since_loss = 0
        self.circuit_breaker_active = False
        self.circuit_breaker_days = 0
        
        # Performance tracking
        self.daily_returns: List[float] = []
        self.equity_curve: List[float] = [1.0]
        self.peak_equity: float = 1.0
        self.current_drawdown: float = 0.0
        
        # Regime tracking for transition risk
        self.last_regime: Optional[RegimeState] = None
        self.regime_change_date: Optional[datetime] = None
    
    def update(
        self,
        daily_return: float,
        vix_level: float,
        regime: RegimeState,
        date: datetime = None,
    ) -> RiskMetrics:
        """
        Update risk state based on new market data.
        
        Args:
            daily_return: Portfolio return for the day
            vix_level: Current VIX level
            regime: Current market regime
            date: Current date
            
        Returns:
            RiskMetrics with current risk state
        """
        # Update equity curve
        new_equity = self.equity_curve[-1] * (1 + daily_return)
        self.equity_curve.append(new_equity)
        self.daily_returns.append(daily_return)
        
        # Update peak and drawdown
        if new_equity > self.peak_equity:
            self.peak_equity = new_equity
        self.current_drawdown = (self.peak_equity - new_equity) / self.peak_equity
        
        # Update consecutive losses
        if daily_return < -0.001:  # Small threshold for noise
            self.consecutive_losses += 1
            self.days_since_loss = 0
        else:
            self.days_since_loss += 1
            if self.days_since_loss >= self.config.recovery_days_required:
                self.consecutive_losses = max(0, self.consecutive_losses - 1)
        
        # Handle regime transition
        if self.last_regime is not None and regime != self.last_regime:
            self.regime_change_date = date
            logger.info(f"Regime change: {self.last_regime} -> {regime}")
        self.last_regime = regime
        
        # Check circuit breakers
        self._update_circuit_breaker(daily_return, vix_level)
        
        # Determine risk state
        self.risk_state = self._determine_risk_state(
            daily_return, vix_level, regime
        )
        
        # Calculate position scale
        position_scale = self._calculate_position_scale(
            vix_level, regime, date
        )
        
        return RiskMetrics(
            portfolio_drawdown=self.current_drawdown,
            daily_pnl=daily_return,
            consecutive_losses=self.consecutive_losses,
            vix_level=vix_level,
            regime=regime,
            risk_state=self.risk_state,
            position_scale=position_scale,
            circuit_breaker_active=self.circuit_breaker_active,
        )
    
    def _update_circuit_breaker(self, daily_return: float, vix_level: float):
        """Update circuit breaker state."""
        # Trigger conditions
        if (daily_return <= self.config.severe_daily_loss or 
            vix_level >= self.config.vix_crisis):
            self.circuit_breaker_active = True
            self.circuit_breaker_days = 0
            logger.warning(f"Circuit breaker TRIGGERED: ret={daily_return:.2%}, VIX={vix_level:.1f}")
        
        # Recovery conditions
        if self.circuit_breaker_active:
            self.circuit_breaker_days += 1
            if (self.circuit_breaker_days >= 2 and 
                vix_level < self.config.vix_defensive and
                daily_return > 0):
                self.circuit_breaker_active = False
                logger.info("Circuit breaker RELEASED")
    
    def _determine_risk_state(
        self,
        daily_return: float,
        vix_level: float,
        regime: RegimeState,
    ) -> RiskState:
        """Determine overall risk state."""
        # Check for crisis conditions
        crisis_conditions = [
            self.current_drawdown >= self.config.drawdown_crisis,
            vix_level >= self.config.vix_crisis,
            self.circuit_breaker_active,
            self.consecutive_losses >= self.config.max_consecutive_losses + 2,
        ]
        if sum(crisis_conditions) >= 1:
            return RiskState.CRISIS
        
        # Check for defensive conditions
        defensive_conditions = [
            self.current_drawdown >= self.config.drawdown_defensive,
            vix_level >= self.config.vix_defensive,
            self.consecutive_losses >= self.config.max_consecutive_losses,
            daily_return <= self.config.daily_loss_limit,
        ]
        if sum(defensive_conditions) >= 2:
            return RiskState.DEFENSIVE
        
        # Check for caution conditions
        caution_conditions = [
            self.current_drawdown >= self.config.drawdown_caution,
            vix_level >= self.config.vix_caution,
            self.consecutive_losses >= 2,
            regime == RegimeState.NEUTRAL,
        ]
        if sum(caution_conditions) >= 2:
            return RiskState.CAUTION
        
        return RiskState.NORMAL
    
    def _calculate_position_scale(
        self,
        vix_level: float,
        regime: RegimeState,
        date: datetime = None,
    ) -> float:
        """Calculate position size scaling factor."""
        scale = 1.0
        
        # Risk state scaling
        state_scales = {
            RiskState.NORMAL: 1.0,
            RiskState.CAUTION: 0.70,
            RiskState.DEFENSIVE: 0.40,
            RiskState.CRISIS: 0.15,
        }
        scale *= state_scales.get(self.risk_state, 1.0)
        
        # Consecutive loss penalty
        if self.consecutive_losses >= self.config.max_consecutive_losses:
            scale *= self.config.loss_streak_scale
        
        # VIX adjustment (additional to state)
        if vix_level > self.config.vix_caution:
            vix_penalty = min(0.30, (vix_level - self.config.vix_caution) / 50)
            scale *= (1 - vix_penalty)
        
        # Regime transition grace period
        if self.regime_change_date is not None and date is not None:
            days_since_change = (date - self.regime_change_date).days
            if days_since_change <= 2:
                scale *= 0.70  # Reduced size during transition
        
        # Ensure minimum
        scale = max(self.config.min_position_scale, scale)
        
        return scale
    
    def get_allocation_multiplier(self) -> float:
        """Get the current allocation multiplier based on risk state."""
        if self.circuit_breaker_active:
            return 0.10
        
        multipliers = {
            RiskState.NORMAL: 1.0,
            RiskState.CAUTION: 0.70,
            RiskState.DEFENSIVE: 0.40,
            RiskState.CRISIS: 0.15,
        }
        return multipliers.get(self.risk_state, 1.0)
    
    def should_exit_all(self) -> bool:
        """Check if we should exit all positions immediately."""
        return (
            self.circuit_breaker_active and 
            self.current_drawdown >= self.config.drawdown_crisis
        )
    
    def reset(self):
        """Reset risk manager state."""
        self.risk_state = RiskState.NORMAL
        self.consecutive_losses = 0
        self.days_since_loss = 0
        self.circuit_breaker_active = False
        self.circuit_breaker_days = 0
        self.daily_returns = []
        self.equity_curve = [1.0]
        self.peak_equity = 1.0
        self.current_drawdown = 0.0
        self.last_regime = None
        self.regime_change_date = None
