"""
Production Controls for Phase 13
=================================

Circuit breakers, position limits, and monitoring for live trading.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable

import numpy as np

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class Alert:
    """Trading alert."""
    level: AlertLevel
    message: str
    timestamp: datetime
    metric: str
    value: float
    threshold: float


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breakers."""
    # Daily loss limits
    daily_loss_warning: float = -0.02  # -2% warning
    daily_loss_reduce: float = -0.03  # -3% reduce leverage 50%
    daily_loss_exit: float = -0.05  # -5% exit all
    
    # Consecutive losses
    max_consecutive_losses: int = 3
    
    # Volatility limits
    vix_warning: float = 35
    vix_reduce: float = 45
    vix_exit: float = 55
    
    # Drawdown limits
    drawdown_warning: float = 0.10
    drawdown_reduce: float = 0.15
    drawdown_exit: float = 0.20
    
    # Recovery
    recovery_days_before_resume: int = 2


class CircuitBreaker:
    """
    Circuit breaker for emergency risk control.
    
    Triggers protective actions when thresholds are breached.
    """
    
    def __init__(self, config: CircuitBreakerConfig = None):
        self.config = config or CircuitBreakerConfig()
        
        # State
        self.is_triggered = False
        self.trigger_level = None  # 'warning', 'reduce', 'exit'
        self.trigger_time = None
        self.trigger_reason = ""
        
        # Tracking
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.peak_equity = 0.0
        self.current_equity = 0.0
        self.current_drawdown = 0.0
        
        # History
        self.alerts: List[Alert] = []
        self.trigger_history: List[Dict] = []
    
    def update(
        self,
        daily_return: float,
        vix_level: float,
        equity: float,
        timestamp: datetime = None,
    ) -> Dict:
        """
        Update circuit breaker state and check thresholds.
        
        Returns:
            Dict with action recommendations
        """
        timestamp = timestamp or datetime.now()
        actions = {
            'leverage_multiplier': 1.0,
            'exit_all': False,
            'pause_trading': False,
            'alerts': [],
        }
        
        # Update state
        self.daily_pnl = daily_return
        self.current_equity = equity
        
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        self.current_drawdown = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0
        
        # Track consecutive losses
        if daily_return < -0.001:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Check daily loss thresholds
        if daily_return <= self.config.daily_loss_exit:
            actions['exit_all'] = True
            actions['leverage_multiplier'] = 0.0
            self._trigger('exit', f"Daily loss {daily_return:.1%} exceeded exit threshold", timestamp)
            actions['alerts'].append(self._create_alert(
                AlertLevel.EMERGENCY, f"CIRCUIT BREAKER: Exit all - daily loss {daily_return:.1%}",
                'daily_loss', daily_return, self.config.daily_loss_exit, timestamp
            ))
        elif daily_return <= self.config.daily_loss_reduce:
            actions['leverage_multiplier'] = 0.50
            self._trigger('reduce', f"Daily loss {daily_return:.1%} triggered 50% reduction", timestamp)
            actions['alerts'].append(self._create_alert(
                AlertLevel.CRITICAL, f"Reduce leverage 50% - daily loss {daily_return:.1%}",
                'daily_loss', daily_return, self.config.daily_loss_reduce, timestamp
            ))
        elif daily_return <= self.config.daily_loss_warning:
            actions['alerts'].append(self._create_alert(
                AlertLevel.WARNING, f"Daily loss warning: {daily_return:.1%}",
                'daily_loss', daily_return, self.config.daily_loss_warning, timestamp
            ))
        
        # Check VIX thresholds
        if vix_level >= self.config.vix_exit:
            actions['exit_all'] = True
            actions['leverage_multiplier'] = 0.0
            self._trigger('exit', f"VIX {vix_level:.1f} exceeded exit threshold", timestamp)
            actions['alerts'].append(self._create_alert(
                AlertLevel.EMERGENCY, f"CIRCUIT BREAKER: Exit all - VIX {vix_level:.1f}",
                'vix', vix_level, self.config.vix_exit, timestamp
            ))
        elif vix_level >= self.config.vix_reduce:
            if actions['leverage_multiplier'] > 0.30:
                actions['leverage_multiplier'] = 0.30
            actions['alerts'].append(self._create_alert(
                AlertLevel.CRITICAL, f"Reduce leverage 70% - VIX {vix_level:.1f}",
                'vix', vix_level, self.config.vix_reduce, timestamp
            ))
        elif vix_level >= self.config.vix_warning:
            actions['alerts'].append(self._create_alert(
                AlertLevel.WARNING, f"VIX elevated: {vix_level:.1f}",
                'vix', vix_level, self.config.vix_warning, timestamp
            ))
        
        # Check drawdown thresholds
        if self.current_drawdown >= self.config.drawdown_exit:
            actions['exit_all'] = True
            actions['leverage_multiplier'] = 0.0
            self._trigger('exit', f"Drawdown {self.current_drawdown:.1%} exceeded exit threshold", timestamp)
            actions['alerts'].append(self._create_alert(
                AlertLevel.EMERGENCY, f"CIRCUIT BREAKER: Exit all - drawdown {self.current_drawdown:.1%}",
                'drawdown', self.current_drawdown, self.config.drawdown_exit, timestamp
            ))
        elif self.current_drawdown >= self.config.drawdown_reduce:
            if actions['leverage_multiplier'] > 0.40:
                actions['leverage_multiplier'] = 0.40
            actions['alerts'].append(self._create_alert(
                AlertLevel.CRITICAL, f"Reduce leverage 60% - drawdown {self.current_drawdown:.1%}",
                'drawdown', self.current_drawdown, self.config.drawdown_reduce, timestamp
            ))
        elif self.current_drawdown >= self.config.drawdown_warning:
            actions['alerts'].append(self._create_alert(
                AlertLevel.WARNING, f"Drawdown warning: {self.current_drawdown:.1%}",
                'drawdown', self.current_drawdown, self.config.drawdown_warning, timestamp
            ))
        
        # Check consecutive losses
        if self.consecutive_losses >= self.config.max_consecutive_losses:
            actions['pause_trading'] = True
            actions['alerts'].append(self._create_alert(
                AlertLevel.CRITICAL, f"Pause trading: {self.consecutive_losses} consecutive losses",
                'consecutive_losses', self.consecutive_losses, self.config.max_consecutive_losses, timestamp
            ))
        
        # Store alerts
        self.alerts.extend(actions['alerts'])
        
        return actions
    
    def _trigger(self, level: str, reason: str, timestamp: datetime):
        """Trigger circuit breaker."""
        self.is_triggered = True
        self.trigger_level = level
        self.trigger_time = timestamp
        self.trigger_reason = reason
        self.trigger_history.append({
            'level': level,
            'reason': reason,
            'timestamp': timestamp,
            'equity': self.current_equity,
            'drawdown': self.current_drawdown,
        })
        logger.warning(f"CIRCUIT BREAKER TRIGGERED ({level}): {reason}")
    
    def _create_alert(
        self,
        level: AlertLevel,
        message: str,
        metric: str,
        value: float,
        threshold: float,
        timestamp: datetime,
    ) -> Alert:
        """Create an alert."""
        return Alert(
            level=level,
            message=message,
            timestamp=timestamp,
            metric=metric,
            value=value,
            threshold=threshold,
        )
    
    def check_recovery(self, daily_return: float, days_since_trigger: int) -> bool:
        """Check if we can resume trading after a trigger."""
        if not self.is_triggered:
            return True
        
        if (days_since_trigger >= self.config.recovery_days_before_resume and
            daily_return > 0 and
            self.current_drawdown < self.config.drawdown_warning):
            self.is_triggered = False
            self.trigger_level = None
            logger.info("Circuit breaker released - recovery conditions met")
            return True
        
        return False
    
    def reset(self):
        """Reset circuit breaker state."""
        self.is_triggered = False
        self.trigger_level = None
        self.trigger_time = None
        self.trigger_reason = ""
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.peak_equity = 0.0
        self.current_equity = 0.0
        self.current_drawdown = 0.0


@dataclass 
class PositionLimitsConfig:
    """Configuration for position limits."""
    max_single_stock: float = 0.08  # 8% per stock
    max_leveraged_etf: float = 0.25  # 25% per leveraged ETF
    max_options: float = 0.30  # 30% total options
    max_single_option: float = 0.05  # 5% per option position
    max_sector: float = 0.35  # 35% per sector
    max_gross_leverage: float = 2.0  # 2x gross leverage
    max_net_leverage: float = 1.5  # 1.5x net leverage


class PositionLimits:
    """
    Position limit enforcement.
    """
    
    def __init__(self, config: PositionLimitsConfig = None):
        self.config = config or PositionLimitsConfig()
    
    def check_position(
        self,
        ticker: str,
        proposed_weight: float,
        position_type: str,  # 'stock', 'etf', 'leveraged_etf', 'option'
        current_positions: Dict[str, float],
        portfolio_value: float,
    ) -> Dict:
        """
        Check if a proposed position is within limits.
        
        Returns:
            Dict with 'allowed', 'adjusted_weight', 'violations'
        """
        violations = []
        adjusted_weight = proposed_weight
        
        # Check position-specific limits
        if position_type == 'stock':
            if proposed_weight > self.config.max_single_stock:
                violations.append(f"{ticker} weight {proposed_weight:.1%} exceeds stock limit {self.config.max_single_stock:.0%}")
                adjusted_weight = self.config.max_single_stock
        
        elif position_type == 'leveraged_etf':
            if proposed_weight > self.config.max_leveraged_etf:
                violations.append(f"{ticker} weight {proposed_weight:.1%} exceeds leveraged ETF limit {self.config.max_leveraged_etf:.0%}")
                adjusted_weight = self.config.max_leveraged_etf
        
        elif position_type == 'option':
            if proposed_weight > self.config.max_single_option:
                violations.append(f"{ticker} weight {proposed_weight:.1%} exceeds single option limit {self.config.max_single_option:.0%}")
                adjusted_weight = self.config.max_single_option
            
            # Check total options limit
            current_options = sum(w for t, w in current_positions.items() if 'OPT' in t)
            if current_options + proposed_weight > self.config.max_options:
                remaining = self.config.max_options - current_options
                if remaining > 0:
                    adjusted_weight = remaining
                    violations.append(f"Total options would exceed {self.config.max_options:.0%}, reduced to {remaining:.1%}")
                else:
                    adjusted_weight = 0
                    violations.append("Total options limit reached")
        
        return {
            'allowed': len(violations) == 0 or adjusted_weight > 0,
            'original_weight': proposed_weight,
            'adjusted_weight': adjusted_weight,
            'violations': violations,
        }
    
    def check_leverage(
        self,
        positions: Dict[str, float],
        position_leverages: Dict[str, float],  # ticker -> leverage factor
    ) -> Dict:
        """
        Check if portfolio leverage is within limits.
        """
        gross_exposure = 0.0
        net_exposure = 0.0
        
        for ticker, weight in positions.items():
            leverage = position_leverages.get(ticker, 1.0)
            gross_exposure += abs(weight * leverage)
            net_exposure += weight * leverage
        
        violations = []
        
        if gross_exposure > self.config.max_gross_leverage:
            violations.append(f"Gross leverage {gross_exposure:.2f}x exceeds limit {self.config.max_gross_leverage:.1f}x")
        
        if abs(net_exposure) > self.config.max_net_leverage:
            violations.append(f"Net leverage {net_exposure:.2f}x exceeds limit {self.config.max_net_leverage:.1f}x")
        
        return {
            'gross_leverage': gross_exposure,
            'net_leverage': net_exposure,
            'within_limits': len(violations) == 0,
            'violations': violations,
        }


@dataclass
class ExecutionMetrics:
    """Metrics for execution quality."""
    fills: int = 0
    partial_fills: int = 0
    rejections: int = 0
    total_slippage: float = 0.0
    avg_fill_time_ms: float = 0.0
    max_fill_time_ms: float = 0.0


class ProductionController:
    """
    Main production controller integrating all safety mechanisms.
    """
    
    def __init__(
        self,
        circuit_breaker: CircuitBreaker = None,
        position_limits: PositionLimits = None,
    ):
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.position_limits = position_limits or PositionLimits()
        
        # Execution tracking
        self.execution_metrics = ExecutionMetrics()
        
        # State
        self.is_trading_paused = False
        self.pause_reason = ""
        self.last_update = None
    
    def pre_trade_check(
        self,
        ticker: str,
        proposed_weight: float,
        position_type: str,
        current_positions: Dict[str, float],
        portfolio_value: float,
    ) -> Dict:
        """
        Pre-trade check before executing any order.
        """
        result = {
            'approved': True,
            'adjusted_weight': proposed_weight,
            'warnings': [],
            'errors': [],
        }
        
        # Check if trading is paused
        if self.is_trading_paused:
            result['approved'] = False
            result['errors'].append(f"Trading paused: {self.pause_reason}")
            return result
        
        # Check circuit breaker
        if self.circuit_breaker.is_triggered:
            if self.circuit_breaker.trigger_level == 'exit':
                result['approved'] = False
                result['errors'].append("Circuit breaker in EXIT mode - no new positions")
                return result
            elif self.circuit_breaker.trigger_level == 'reduce':
                proposed_weight *= 0.50
                result['warnings'].append("Circuit breaker REDUCE mode - position halved")
        
        # Check position limits
        limit_check = self.position_limits.check_position(
            ticker, proposed_weight, position_type, current_positions, portfolio_value
        )
        
        if limit_check['violations']:
            result['warnings'].extend(limit_check['violations'])
        
        result['adjusted_weight'] = limit_check['adjusted_weight']
        result['approved'] = limit_check['allowed']
        
        return result
    
    def update_execution_metrics(
        self,
        is_fill: bool,
        is_partial: bool,
        slippage: float,
        fill_time_ms: float,
    ):
        """Update execution quality metrics."""
        if is_fill:
            self.execution_metrics.fills += 1
        else:
            self.execution_metrics.rejections += 1
        
        if is_partial:
            self.execution_metrics.partial_fills += 1
        
        self.execution_metrics.total_slippage += slippage
        self.execution_metrics.max_fill_time_ms = max(
            self.execution_metrics.max_fill_time_ms, fill_time_ms
        )
        
        # Rolling average
        n = self.execution_metrics.fills
        if n > 0:
            self.execution_metrics.avg_fill_time_ms = (
                (self.execution_metrics.avg_fill_time_ms * (n - 1) + fill_time_ms) / n
            )
        
        # Alert if slippage is high
        avg_slippage = self.execution_metrics.total_slippage / n if n > 0 else 0
        if avg_slippage > 0.003:  # 0.3% average slippage
            logger.warning(f"High average slippage: {avg_slippage:.2%}")
    
    def daily_update(
        self,
        daily_return: float,
        vix_level: float,
        equity: float,
        timestamp: datetime = None,
    ) -> Dict:
        """
        Daily update with all production checks.
        """
        self.last_update = timestamp or datetime.now()
        
        # Update circuit breaker
        cb_actions = self.circuit_breaker.update(daily_return, vix_level, equity, timestamp)
        
        # Update trading pause state
        if cb_actions['pause_trading']:
            self.is_trading_paused = True
            self.pause_reason = "Consecutive losses exceeded"
        elif cb_actions['exit_all']:
            self.is_trading_paused = True
            self.pause_reason = "Circuit breaker exit triggered"
        
        return {
            'circuit_breaker_actions': cb_actions,
            'is_trading_paused': self.is_trading_paused,
            'execution_metrics': {
                'fills': self.execution_metrics.fills,
                'rejections': self.execution_metrics.rejections,
                'avg_slippage': self.execution_metrics.total_slippage / max(1, self.execution_metrics.fills),
            },
        }
    
    def get_status(self) -> Dict:
        """Get current production status."""
        return {
            'is_trading_paused': self.is_trading_paused,
            'pause_reason': self.pause_reason,
            'circuit_breaker': {
                'triggered': self.circuit_breaker.is_triggered,
                'level': self.circuit_breaker.trigger_level,
                'reason': self.circuit_breaker.trigger_reason,
                'drawdown': self.circuit_breaker.current_drawdown,
            },
            'execution': {
                'total_fills': self.execution_metrics.fills,
                'rejections': self.execution_metrics.rejections,
                'avg_slippage': self.execution_metrics.total_slippage / max(1, self.execution_metrics.fills),
                'avg_fill_time_ms': self.execution_metrics.avg_fill_time_ms,
            },
            'last_update': self.last_update,
        }
    
    def reset(self):
        """Reset all production state."""
        self.circuit_breaker.reset()
        self.execution_metrics = ExecutionMetrics()
        self.is_trading_paused = False
        self.pause_reason = ""
        self.last_update = None
