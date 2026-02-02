"""Integrated Risk Manager with Circuit Breakers.

This addresses Issue #2: Risk Management Gaps
"""

import logging
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    HALT = "halt"


@dataclass
class CircuitBreaker:
    """Circuit breaker to halt trading on extreme conditions."""
    name: str
    threshold: float
    current_value: float = 0.0
    triggered: bool = False
    triggered_at: Optional[datetime] = None
    

class IntegratedRiskManager:
    """Production-ready risk manager with circuit breakers."""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_size = 0.10  # 10% per position
        self.max_portfolio_risk = 0.02  # 2% max risk per day
        self.max_drawdown = 0.15  # 15% max drawdown before halt
        self.max_daily_loss = 0.05  # 5% max daily loss
        
        # Circuit breakers
        self.circuit_breakers: Dict[str, CircuitBreaker] = {
            'daily_loss': CircuitBreaker('Daily Loss Limit', self.max_daily_loss),
            'max_drawdown': CircuitBreaker('Maximum Drawdown', self.max_drawdown),
            'position_concentration': CircuitBreaker('Position Concentration', self.max_position_size),
        }
        
        # Tracking
        self.daily_pnl = 0.0
        self.peak_capital = initial_capital
        self.day_start_capital = initial_capital
        self.positions_value: Dict[str, float] = {}
        self.halted = False
        self.halt_reason = None
        
        logger.info(f"Risk Manager initialized with ${initial_capital:,.2f}")
    
    def check_order_risk(self, symbol: str, quantity: int, price: float, side: str) -> tuple[bool, str]:
        """Check if order passes risk checks."""
        
        # Check if trading is halted
        if self.halted:
            return False, f"Trading halted: {self.halt_reason}"
        
        # Calculate order value
        order_value = quantity * price
        
        # Check position size limit
        position_pct = order_value / self.current_capital
        if position_pct > self.max_position_size:
            msg = f"Position size {position_pct:.1%} exceeds limit {self.max_position_size:.1%}"
            logger.warning(msg)
            return False, msg
        
        # Check if circuit breakers would be triggered
        for name, breaker in self.circuit_breakers.items():
            if breaker.triggered:
                return False, f"Circuit breaker '{name}' is triggered"
        
        return True, "Order approved"
    
    def update_position(self, symbol: str, value: float):
        """Update position value."""
        self.positions_value[symbol] = value
        self._check_circuit_breakers()
    
    def update_pnl(self, pnl: float):
        """Update P&L and check circuit breakers."""
        self.daily_pnl = pnl
        self.current_capital = self.initial_capital + pnl
        
        # Update peak capital for drawdown calculation
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        self._check_circuit_breakers()
    
    def _check_circuit_breakers(self):
        """Check all circuit breakers."""
        
        # Daily loss check
        daily_loss_pct = abs(self.daily_pnl / self.day_start_capital) if self.day_start_capital > 0 else 0
        if self.daily_pnl < 0 and daily_loss_pct > self.max_daily_loss:
            self._trigger_circuit_breaker('daily_loss', daily_loss_pct)
        
        # Drawdown check
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital if self.peak_capital > 0 else 0
        if drawdown > self.max_drawdown:
            self._trigger_circuit_breaker('max_drawdown', drawdown)
        
        # Position concentration check
        if self.current_capital > 0:
            for symbol, value in self.positions_value.items():
                concentration = value / self.current_capital
                if concentration > self.max_position_size * 1.5:  # 50% buffer
                    self._trigger_circuit_breaker('position_concentration', concentration)
    
    def _trigger_circuit_breaker(self, name: str, value: float):
        """Trigger a circuit breaker and halt trading."""
        breaker = self.circuit_breakers[name]
        if not breaker.triggered:
            breaker.triggered = True
            breaker.current_value = value
            breaker.triggered_at = datetime.now()
            
            self.halted = True
            self.halt_reason = f"{name}: {value:.2%} exceeds {breaker.threshold:.2%}"
            
            logger.critical(f"🚨 CIRCUIT BREAKER TRIGGERED: {self.halt_reason}")
            logger.critical("🛑 ALL TRADING HALTED")
    
    def reset_daily(self):
        """Reset daily metrics (call at market open)."""
        self.daily_pnl = 0.0
        self.day_start_capital = self.current_capital
        
        # Reset daily loss circuit breaker
        self.circuit_breakers['daily_loss'].triggered = False
        self.circuit_breakers['daily_loss'].current_value = 0.0
        
        logger.info("Daily risk metrics reset")
    
    def get_risk_status(self) -> Dict:
        """Get current risk status."""
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital if self.peak_capital > 0 else 0
        daily_loss_pct = abs(self.daily_pnl / self.day_start_capital) if self.day_start_capital > 0 and self.daily_pnl < 0 else 0
        
        # Determine risk level
        risk_level = RiskLevel.NORMAL
        if self.halted:
            risk_level = RiskLevel.HALT
        elif drawdown > self.max_drawdown * 0.75 or daily_loss_pct > self.max_daily_loss * 0.75:
            risk_level = RiskLevel.CRITICAL
        elif drawdown > self.max_drawdown * 0.5 or daily_loss_pct > self.max_daily_loss * 0.5:
            risk_level = RiskLevel.WARNING
        
        return {
            'risk_level': risk_level.value,
            'halted': self.halted,
            'halt_reason': self.halt_reason,
            'current_capital': self.current_capital,
            'peak_capital': self.peak_capital,
            'drawdown': drawdown,
            'daily_pnl': self.daily_pnl,
            'daily_loss_pct': daily_loss_pct,
            'circuit_breakers': {
                name: {
                    'triggered': cb.triggered,
                    'current': cb.current_value,
                    'threshold': cb.threshold
                }
                for name, cb in self.circuit_breakers.items()
            }
        }


if __name__ == "__main__":
    # Test the risk manager
    rm = IntegratedRiskManager(initial_capital=100000.0)
    
    # Test normal order
    approved, msg = rm.check_order_risk("AAPL", 50, 150.0, "buy")
    print(f"Order 1: {approved} - {msg}")
    
    # Test oversized order
    approved, msg = rm.check_order_risk("TSLA", 1000, 200.0, "buy")
    print(f"Order 2: {approved} - {msg}")
    
    # Simulate loss and trigger circuit breaker
    rm.update_pnl(-6000)  # 6% loss
    status = rm.get_risk_status()
    print(f"\nRisk Status: {status['risk_level']}")
    print(f"Halted: {status['halted']}")
    if status['halted']:
        print(f"Reason: {status['halt_reason']}")
