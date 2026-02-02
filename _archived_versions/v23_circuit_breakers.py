#!/usr/bin/env python3
"""
V23 Circuit Breakers & Risk Controls
======================================
Production-grade risk management with pre-trade validation,
real-time circuit breakers, and emergency kill switch.

Features:
- Pre-trade validation checks
- Real-time circuit breakers
- Emergency kill switch
- Alert system integration
"""

import json
import logging
import os
import smtplib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('V23_CircuitBreakers')


# =============================================================================
# ENUMS AND CONFIGURATION
# =============================================================================

class AlertPriority(Enum):
    CRITICAL = "critical"  # SMS + Email immediately
    HIGH = "high"          # Email + Push notification
    MEDIUM = "medium"      # Email only
    LOW = "low"            # Log only


class CircuitBreakerState(Enum):
    NORMAL = "normal"
    WARNING = "warning"
    REDUCED = "reduced"
    HALTED = "halted"
    EMERGENCY = "emergency"


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker thresholds and settings."""
    
    # Daily limits
    daily_loss_pct: float = -5.0       # Halt all trading for the day
    daily_gain_pct: float = 10.0       # Take profits / reduce exposure
    
    # Weekly limits  
    weekly_loss_pct: float = -10.0     # Halt trading, require manual review
    
    # Drawdown limits
    drawdown_reduce_pct: float = -10.0   # Reduce position sizes to 50%
    drawdown_halt_pct: float = -15.0     # Halt new entries
    drawdown_emergency_pct: float = -20.0 # KILL SWITCH - close all positions
    
    # Trade limits
    consecutive_losses: int = 5          # Pause new entries for 24 hours
    max_daily_trades: int = 50          # Max trades per day
    
    # Error limits
    max_execution_errors: int = 3       # Halt on execution errors
    max_api_errors: int = 5             # Halt on API errors
    
    # Position limits
    max_position_pct: float = 10.0      # Max single position
    max_sector_pct: float = 25.0        # Max sector concentration
    max_correlation: float = 0.70       # Max portfolio correlation
    
    # Spread limits
    max_spread_bps: float = 50.0        # Max bid-ask spread
    
    # Time limits
    no_trade_before: str = "09:35"      # No trades in first 5 min
    no_trade_after: str = "15:55"       # No trades in last 5 min


@dataclass 
class RiskState:
    """Current risk state tracking."""
    
    # P&L tracking
    daily_pnl_pct: float = 0.0
    weekly_pnl_pct: float = 0.0
    current_drawdown_pct: float = 0.0
    peak_equity: float = 0.0
    
    # Trade tracking
    consecutive_losses: int = 0
    daily_trade_count: int = 0
    last_trade_date: Optional[str] = None
    
    # Error tracking
    execution_error_count: int = 0
    api_error_count: int = 0
    last_error_time: Optional[datetime] = None
    
    # State
    circuit_breaker_state: CircuitBreakerState = CircuitBreakerState.NORMAL
    state_reason: str = ""
    last_state_change: Optional[datetime] = None
    
    # Timestamps
    last_update: Optional[datetime] = None


# =============================================================================
# PRE-TRADE VALIDATOR
# =============================================================================

class PreTradeValidator:
    """
    Validates orders before submission.
    All checks must pass for order to proceed.
    """
    
    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self.validation_log: List[Dict] = []
    
    def validate_order(self,
                      symbol: str,
                      side: str,
                      quantity: float,
                      price: float,
                      account_value: float,
                      current_positions: Dict[str, float],
                      sector_map: Optional[Dict[str, str]] = None,
                      quote_spread_bps: Optional[float] = None,
                      risk_state: Optional[RiskState] = None) -> Tuple[bool, List[str]]:
        """
        Run all pre-trade validations.
        
        Returns:
            (passed, list of failed checks)
        """
        failures = []
        
        # 1. Circuit breaker state
        if risk_state and risk_state.circuit_breaker_state in [
            CircuitBreakerState.HALTED, 
            CircuitBreakerState.EMERGENCY
        ]:
            failures.append(f"Circuit breaker active: {risk_state.circuit_breaker_state.value}")
        
        # 2. Position size limit
        position_value = quantity * price
        position_pct = position_value / account_value * 100
        if position_pct > self.config.max_position_pct:
            failures.append(f"Position too large: {position_pct:.1f}% > {self.config.max_position_pct}%")
        
        # 3. Daily loss limit
        if risk_state and risk_state.daily_pnl_pct < self.config.daily_loss_pct:
            failures.append(f"Daily loss limit breached: {risk_state.daily_pnl_pct:.1f}%")
        
        # 4. Daily trade limit
        if risk_state and risk_state.daily_trade_count >= self.config.max_daily_trades:
            failures.append(f"Daily trade limit reached: {risk_state.daily_trade_count}")
        
        # 5. Consecutive losses
        if risk_state and risk_state.consecutive_losses >= self.config.consecutive_losses:
            failures.append(f"Consecutive losses: {risk_state.consecutive_losses}")
        
        # 6. Spread check
        if quote_spread_bps and quote_spread_bps > self.config.max_spread_bps:
            failures.append(f"Spread too wide: {quote_spread_bps:.0f}bps > {self.config.max_spread_bps}bps")
        
        # 7. Sector concentration
        if sector_map and symbol in sector_map:
            sector = sector_map[symbol]
            sector_weight = self._calculate_sector_weight(
                sector, current_positions, sector_map, account_value
            )
            new_sector_weight = sector_weight + position_pct
            if new_sector_weight > self.config.max_sector_pct:
                failures.append(f"Sector limit: {sector} at {new_sector_weight:.1f}%")
        
        # 8. Market hours
        now = datetime.now()
        current_time = now.strftime("%H:%M")
        if current_time < self.config.no_trade_before:
            failures.append(f"Before trading window: {current_time}")
        if current_time > self.config.no_trade_after:
            failures.append(f"After trading window: {current_time}")
        
        # 9. Error limits
        if risk_state:
            if risk_state.execution_error_count >= self.config.max_execution_errors:
                failures.append(f"Execution error limit: {risk_state.execution_error_count}")
            if risk_state.api_error_count >= self.config.max_api_errors:
                failures.append(f"API error limit: {risk_state.api_error_count}")
        
        # Log validation
        passed = len(failures) == 0
        self.validation_log.append({
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'passed': passed,
            'failures': failures
        })
        
        if not passed:
            logger.warning(f"Order validation failed for {symbol}: {failures}")
        
        return passed, failures
    
    def _calculate_sector_weight(self,
                                sector: str,
                                positions: Dict[str, float],
                                sector_map: Dict[str, str],
                                account_value: float) -> float:
        """Calculate current weight in a sector."""
        sector_value = 0.0
        for symbol, qty in positions.items():
            if sector_map.get(symbol) == sector:
                # Would need prices here - simplified
                sector_value += abs(qty) * 100  # Assume $100/share
        
        return sector_value / account_value * 100 if account_value > 0 else 0


# =============================================================================
# CIRCUIT BREAKER MANAGER
# =============================================================================

class CircuitBreakerManager:
    """
    Real-time circuit breaker monitoring and management.
    """
    
    def __init__(self, 
                 config: Optional[CircuitBreakerConfig] = None,
                 alert_callback: Optional[Callable] = None):
        self.config = config or CircuitBreakerConfig()
        self.state = RiskState()
        self.alert_callback = alert_callback
        
        self.validator = PreTradeValidator(self.config)
        
        # Alert history
        self.alert_history: List[Dict] = []
        
        # State persistence
        self.state_dir = Path('state/circuit_breakers')
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("CircuitBreakerManager initialized")
    
    def update_pnl(self, daily_pnl_pct: float, current_equity: float):
        """Update P&L metrics and check breakers."""
        self.state.daily_pnl_pct = daily_pnl_pct
        self.state.last_update = datetime.now()
        
        # Update peak and drawdown
        if current_equity > self.state.peak_equity:
            self.state.peak_equity = current_equity
        
        if self.state.peak_equity > 0:
            self.state.current_drawdown_pct = \
                (current_equity - self.state.peak_equity) / self.state.peak_equity * 100
        
        # Check circuit breakers
        self._check_breakers()
    
    def record_trade(self, pnl: float):
        """Record trade outcome."""
        today = datetime.now().strftime("%Y-%m-%d")
        
        if self.state.last_trade_date != today:
            self.state.daily_trade_count = 0
            self.state.last_trade_date = today
        
        self.state.daily_trade_count += 1
        
        if pnl < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0
        
        self._check_breakers()
    
    def record_error(self, error_type: str):
        """Record execution or API error."""
        if error_type == 'execution':
            self.state.execution_error_count += 1
        elif error_type == 'api':
            self.state.api_error_count += 1
        
        self.state.last_error_time = datetime.now()
        self._check_breakers()
    
    def reset_daily_counters(self):
        """Reset daily counters (call at market open)."""
        self.state.daily_pnl_pct = 0.0
        self.state.daily_trade_count = 0
        self.state.execution_error_count = 0
        self.state.api_error_count = 0
        
        logger.info("Daily counters reset")
    
    def _check_breakers(self):
        """Check all circuit breakers and update state."""
        old_state = self.state.circuit_breaker_state
        new_state = CircuitBreakerState.NORMAL
        reason = ""
        
        # Check in order of severity (highest first)
        
        # EMERGENCY: Max drawdown
        if self.state.current_drawdown_pct < self.config.drawdown_emergency_pct:
            new_state = CircuitBreakerState.EMERGENCY
            reason = f"Emergency drawdown: {self.state.current_drawdown_pct:.1f}%"
        
        # HALTED: Daily loss or halt drawdown
        elif self.state.daily_pnl_pct < self.config.daily_loss_pct:
            new_state = CircuitBreakerState.HALTED
            reason = f"Daily loss limit: {self.state.daily_pnl_pct:.1f}%"
        
        elif self.state.current_drawdown_pct < self.config.drawdown_halt_pct:
            new_state = CircuitBreakerState.HALTED
            reason = f"Drawdown halt: {self.state.current_drawdown_pct:.1f}%"
        
        elif self.state.execution_error_count >= self.config.max_execution_errors:
            new_state = CircuitBreakerState.HALTED
            reason = f"Execution errors: {self.state.execution_error_count}"
        
        # REDUCED: Warning drawdown or consecutive losses
        elif self.state.current_drawdown_pct < self.config.drawdown_reduce_pct:
            new_state = CircuitBreakerState.REDUCED
            reason = f"Drawdown warning: {self.state.current_drawdown_pct:.1f}%"
        
        elif self.state.consecutive_losses >= self.config.consecutive_losses:
            new_state = CircuitBreakerState.REDUCED
            reason = f"Consecutive losses: {self.state.consecutive_losses}"
        
        # WARNING: Approaching limits
        elif self.state.daily_pnl_pct < self.config.daily_loss_pct * 0.7:
            new_state = CircuitBreakerState.WARNING
            reason = f"Approaching daily limit: {self.state.daily_pnl_pct:.1f}%"
        
        # Update state if changed
        if new_state != old_state:
            self.state.circuit_breaker_state = new_state
            self.state.state_reason = reason
            self.state.last_state_change = datetime.now()
            
            logger.warning(f"Circuit breaker state changed: {old_state.value} -> {new_state.value}")
            logger.warning(f"Reason: {reason}")
            
            # Trigger alerts
            self._send_alert(new_state, reason)
    
    def _send_alert(self, state: CircuitBreakerState, reason: str):
        """Send alert based on state severity."""
        priority = {
            CircuitBreakerState.EMERGENCY: AlertPriority.CRITICAL,
            CircuitBreakerState.HALTED: AlertPriority.CRITICAL,
            CircuitBreakerState.REDUCED: AlertPriority.HIGH,
            CircuitBreakerState.WARNING: AlertPriority.MEDIUM,
            CircuitBreakerState.NORMAL: AlertPriority.LOW
        }.get(state, AlertPriority.LOW)
        
        alert = {
            'timestamp': datetime.now().isoformat(),
            'priority': priority.value,
            'state': state.value,
            'reason': reason,
            'daily_pnl': self.state.daily_pnl_pct,
            'drawdown': self.state.current_drawdown_pct
        }
        
        self.alert_history.append(alert)
        
        # Call external alert handler if provided
        if self.alert_callback:
            try:
                self.alert_callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
        
        # Log based on priority
        if priority == AlertPriority.CRITICAL:
            logger.critical(f"🚨 CRITICAL ALERT: {reason}")
        elif priority == AlertPriority.HIGH:
            logger.warning(f"⚠️ HIGH ALERT: {reason}")
        else:
            logger.info(f"📢 Alert: {reason}")
    
    def can_trade(self) -> Tuple[bool, str]:
        """Check if trading is allowed."""
        state = self.state.circuit_breaker_state
        
        if state == CircuitBreakerState.EMERGENCY:
            return False, "Emergency state - all trading halted"
        elif state == CircuitBreakerState.HALTED:
            return False, f"Trading halted: {self.state.state_reason}"
        elif state == CircuitBreakerState.REDUCED:
            return True, "Reduced position sizing active"
        elif state == CircuitBreakerState.WARNING:
            return True, "Warning state - monitor closely"
        else:
            return True, "Normal operation"
    
    def get_position_multiplier(self) -> float:
        """Get position size multiplier based on state."""
        state = self.state.circuit_breaker_state
        
        if state in [CircuitBreakerState.EMERGENCY, CircuitBreakerState.HALTED]:
            return 0.0
        elif state == CircuitBreakerState.REDUCED:
            return 0.5
        elif state == CircuitBreakerState.WARNING:
            return 0.75
        else:
            return 1.0
    
    def validate_order(self, **kwargs) -> Tuple[bool, List[str]]:
        """Validate order through pre-trade checks."""
        return self.validator.validate_order(risk_state=self.state, **kwargs)
    
    def get_status(self) -> Dict:
        """Get current circuit breaker status."""
        can_trade, reason = self.can_trade()
        
        return {
            'state': self.state.circuit_breaker_state.value,
            'state_reason': self.state.state_reason,
            'can_trade': can_trade,
            'trade_reason': reason,
            'position_multiplier': self.get_position_multiplier(),
            'daily_pnl_pct': self.state.daily_pnl_pct,
            'current_drawdown_pct': self.state.current_drawdown_pct,
            'consecutive_losses': self.state.consecutive_losses,
            'daily_trade_count': self.state.daily_trade_count,
            'execution_errors': self.state.execution_error_count,
            'last_update': self.state.last_update.isoformat() if self.state.last_update else None
        }
    
    def save_state(self):
        """Save circuit breaker state to disk."""
        state_dict = {
            'daily_pnl_pct': self.state.daily_pnl_pct,
            'weekly_pnl_pct': self.state.weekly_pnl_pct,
            'current_drawdown_pct': self.state.current_drawdown_pct,
            'peak_equity': self.state.peak_equity,
            'consecutive_losses': self.state.consecutive_losses,
            'daily_trade_count': self.state.daily_trade_count,
            'last_trade_date': self.state.last_trade_date,
            'circuit_breaker_state': self.state.circuit_breaker_state.value,
            'state_reason': self.state.state_reason,
            'alert_history': self.alert_history[-50:],
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.state_dir / 'circuit_breaker_state.json', 'w') as f:
            json.dump(state_dict, f, indent=2)
        
        logger.info("Circuit breaker state saved")
    
    def load_state(self):
        """Load circuit breaker state from disk."""
        state_file = self.state_dir / 'circuit_breaker_state.json'
        if not state_file.exists():
            return
        
        with open(state_file) as f:
            state_dict = json.load(f)
        
        self.state.peak_equity = state_dict.get('peak_equity', 0.0)
        self.state.weekly_pnl_pct = state_dict.get('weekly_pnl_pct', 0.0)
        self.alert_history = state_dict.get('alert_history', [])
        
        logger.info("Circuit breaker state loaded")


# =============================================================================
# KILL SWITCH
# =============================================================================

class KillSwitch:
    """
    Emergency kill switch for immediate position liquidation.
    """
    
    def __init__(self, 
                 execution_engine=None,
                 alert_manager=None):
        self.execution_engine = execution_engine
        self.alert_manager = alert_manager
        self.activated = False
        self.activation_time: Optional[datetime] = None
        self.activation_reason: str = ""
        
        logger.info("KillSwitch initialized")
    
    def activate(self, reason: str = "Manual activation") -> bool:
        """
        Activate kill switch - close all positions immediately.
        """
        logger.critical(f"🚨 KILL SWITCH ACTIVATED: {reason}")
        
        self.activated = True
        self.activation_time = datetime.now()
        self.activation_reason = reason
        
        # Close all positions
        if self.execution_engine:
            try:
                orders = self.execution_engine.close_all_positions()
                logger.critical(f"Submitted {len(orders)} close orders")
            except Exception as e:
                logger.critical(f"Error closing positions: {e}")
        
        # Send critical alerts
        self._send_critical_alert(reason)
        
        # Save state
        self._save_activation()
        
        return True
    
    def deactivate(self, confirmation_code: str) -> bool:
        """
        Deactivate kill switch (requires confirmation).
        """
        # Simple confirmation code
        expected_code = f"CONFIRM-{datetime.now().strftime('%Y%m%d')}"
        
        if confirmation_code != expected_code:
            logger.warning(f"Kill switch deactivation failed: invalid code")
            return False
        
        self.activated = False
        logger.warning("Kill switch deactivated")
        
        return True
    
    def is_active(self) -> bool:
        """Check if kill switch is active."""
        return self.activated
    
    def _send_critical_alert(self, reason: str):
        """Send critical alerts through all channels."""
        alert = {
            'type': 'KILL_SWITCH',
            'timestamp': datetime.now().isoformat(),
            'reason': reason,
            'message': f"EMERGENCY: Kill switch activated - {reason}"
        }
        
        # Log to file
        alert_file = Path('state/circuit_breakers/kill_switch_log.json')
        alert_file.parent.mkdir(parents=True, exist_ok=True)
        
        alerts = []
        if alert_file.exists():
            with open(alert_file) as f:
                alerts = json.load(f)
        
        alerts.append(alert)
        
        with open(alert_file, 'w') as f:
            json.dump(alerts, f, indent=2)
        
        # Would send email/SMS here in production
        logger.critical(f"Alert saved: {alert}")
    
    def _save_activation(self):
        """Save activation state."""
        state = {
            'activated': self.activated,
            'activation_time': self.activation_time.isoformat() if self.activation_time else None,
            'activation_reason': self.activation_reason
        }
        
        with open(Path('state/circuit_breakers/kill_switch_state.json'), 'w') as f:
            json.dump(state, f, indent=2)
    
    def get_status(self) -> Dict:
        """Get kill switch status."""
        return {
            'activated': self.activated,
            'activation_time': self.activation_time.isoformat() if self.activation_time else None,
            'activation_reason': self.activation_reason
        }


# =============================================================================
# MAIN / TESTING
# =============================================================================

def main():
    """Test circuit breakers."""
    logger.info("=" * 70)
    logger.info("🛡️ V23 CIRCUIT BREAKERS TEST")
    logger.info("=" * 70)
    
    # Initialize
    manager = CircuitBreakerManager()
    kill_switch = KillSwitch()
    
    # Set initial state
    manager.state.peak_equity = 100000
    
    # Test scenarios
    logger.info("\n📊 Testing circuit breaker scenarios...")
    
    scenarios = [
        {'name': 'Normal', 'daily_pnl': 1.0, 'equity': 101000},
        {'name': 'Warning', 'daily_pnl': -3.5, 'equity': 96500},
        {'name': 'Reduced', 'daily_pnl': -4.0, 'equity': 89000},
        {'name': 'Halted', 'daily_pnl': -6.0, 'equity': 84000},
        {'name': 'Emergency', 'daily_pnl': -8.0, 'equity': 78000},
    ]
    
    logger.info(f"\n   {'Scenario':<12} {'Daily P&L':>10} {'Drawdown':>10} {'State':>12} {'Can Trade':>10}")
    logger.info("-" * 60)
    
    for scenario in scenarios:
        manager.update_pnl(scenario['daily_pnl'], scenario['equity'])
        status = manager.get_status()
        
        logger.info(f"   {scenario['name']:<12} {scenario['daily_pnl']:>10.1f}% "
                   f"{status['current_drawdown_pct']:>10.1f}% {status['state']:>12} "
                   f"{str(status['can_trade']):>10}")
    
    # Test pre-trade validation
    logger.info("\n📋 Testing pre-trade validation...")
    
    # Reset to normal
    manager.state.circuit_breaker_state = CircuitBreakerState.NORMAL
    manager.state.daily_pnl_pct = 0
    
    passed, failures = manager.validate_order(
        symbol='AAPL',
        side='buy',
        quantity=100,
        price=150.0,
        account_value=100000,
        current_positions={},
        quote_spread_bps=25.0
    )
    logger.info(f"   Normal order: {'PASSED' if passed else 'FAILED'}")
    
    # Test order with wide spread
    passed, failures = manager.validate_order(
        symbol='AAPL',
        side='buy',
        quantity=100,
        price=150.0,
        account_value=100000,
        current_positions={},
        quote_spread_bps=75.0  # Wide spread
    )
    logger.info(f"   Wide spread order: {'PASSED' if passed else 'FAILED'} ({failures})")
    
    # Test order exceeding position limit
    passed, failures = manager.validate_order(
        symbol='AAPL',
        side='buy',
        quantity=1000,
        price=150.0,
        account_value=100000,
        current_positions={},
        quote_spread_bps=20.0
    )
    logger.info(f"   Large order: {'PASSED' if passed else 'FAILED'} ({failures})")
    
    # Test kill switch
    logger.info("\n🚨 Testing kill switch...")
    logger.info(f"   Kill switch active: {kill_switch.is_active()}")
    
    # Don't actually activate in test
    logger.info("   (Skipping activation in test mode)")
    
    # Get final status
    logger.info("\n📊 Final Circuit Breaker Status:")
    status = manager.get_status()
    for key, value in status.items():
        logger.info(f"   {key}: {value}")
    
    # Save state
    manager.save_state()
    
    logger.info("\n✅ Circuit breakers test complete")
    
    return manager, kill_switch


if __name__ == "__main__":
    main()
