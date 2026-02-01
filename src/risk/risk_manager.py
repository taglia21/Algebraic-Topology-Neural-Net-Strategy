#!/usr/bin/env python3
"""
Production Risk Management System
=================================
Institutional-grade risk controls for live trading.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import json
import os

class RiskLevel(Enum):
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"
    HALTED = "halted"

@dataclass
class RiskLimits:
    """Configurable risk limits"""
    max_position_size: float = 0.02  # 2% max per position
    max_portfolio_risk: float = 0.10  # 10% max total exposure
    max_daily_loss: float = 0.02  # 2% daily loss limit
    max_weekly_loss: float = 0.05  # 5% weekly loss limit
    max_drawdown: float = 0.10  # 10% max drawdown before halt
    max_trades_per_day: int = 20  # Max trades per day
    max_correlation: float = 0.7  # Max correlation between positions
    min_confidence: float = 0.55  # Min signal confidence to trade
    volatility_scalar: float = 1.0  # Reduce size in high vol

@dataclass
class PortfolioState:
    """Current portfolio state tracking"""
    capital: float = 100000
    peak_capital: float = 100000
    positions: Dict = field(default_factory=dict)
    daily_pnl: float = 0
    weekly_pnl: float = 0
    daily_trades: int = 0
    last_reset: datetime = field(default_factory=datetime.now)
    trade_history: List = field(default_factory=list)

class CircuitBreaker:
    """Emergency stop system"""
    def __init__(self):
        self.is_halted = False
        self.halt_reason = ""
        self.halt_time = None
        self.cool_down_hours = 24
    
    def halt(self, reason: str):
        self.is_halted = True
        self.halt_reason = reason
        self.halt_time = datetime.now()
        print(f"🛑 CIRCUIT BREAKER TRIGGERED: {reason}")
    
    def check_cool_down(self) -> bool:
        if not self.is_halted:
            return True
        if self.halt_time and datetime.now() - self.halt_time > timedelta(hours=self.cool_down_hours):
            self.reset()
            return True
        return False
    
    def reset(self):
        self.is_halted = False
        self.halt_reason = ""
        self.halt_time = None
        print("✅ Circuit breaker reset")
    
    def manual_override(self, password: str) -> bool:
        if password == "OVERRIDE_2026":
            self.reset()
            return True
        return False

class RiskManager:
    """Production Risk Management System"""
    
    def __init__(self, limits: RiskLimits = None, initial_capital: float = 100000):
        self.limits = limits or RiskLimits()
        self.state = PortfolioState(capital=initial_capital, peak_capital=initial_capital)
        self.circuit_breaker = CircuitBreaker()
        self.risk_level = RiskLevel.NORMAL
        self.alerts = []
        self.log_file = "logs/risk_events.json"
        os.makedirs("logs", exist_ok=True)
    
    def can_trade(self) -> tuple[bool, str]:
        """Check if trading is allowed"""
        # Circuit breaker check
        if self.circuit_breaker.is_halted:
            if not self.circuit_breaker.check_cool_down():
                return False, f"Circuit breaker active: {self.circuit_breaker.halt_reason}"
        
        # Daily trade limit
        if self.state.daily_trades >= self.limits.max_trades_per_day:
            return False, f"Daily trade limit reached: {self.state.daily_trades}"
        
        # Daily loss limit
        daily_loss_pct = self.state.daily_pnl / self.state.capital
        if daily_loss_pct <= -self.limits.max_daily_loss:
            self.circuit_breaker.halt(f"Daily loss limit: {daily_loss_pct*100:.2f}%")
            return False, "Daily loss limit exceeded"
        
        # Max drawdown check
        drawdown = (self.state.peak_capital - self.state.capital) / self.state.peak_capital
        if drawdown >= self.limits.max_drawdown:
            self.circuit_breaker.halt(f"Max drawdown: {drawdown*100:.2f}%")
            return False, "Max drawdown exceeded"
        
        return True, "OK"
    
    def calculate_position_size(self, signal_confidence: float, volatility: float, 
                                 current_price: float) -> float:
        """Calculate risk-adjusted position size"""
        # Base position size from limits
        base_size = self.limits.max_position_size
        
        # Adjust for confidence
        if signal_confidence < self.limits.min_confidence:
            return 0  # Don't trade low confidence signals
        confidence_mult = min(1.0, (signal_confidence - 0.5) * 4)  # 0.5->0, 0.75->1
        
        # Adjust for volatility (reduce size in high vol)
        vol_mult = 1.0 / (1 + volatility * 10 * self.limits.volatility_scalar)
        
        # Adjust for current risk level
        risk_mult = {
            RiskLevel.NORMAL: 1.0,
            RiskLevel.ELEVATED: 0.75,
            RiskLevel.HIGH: 0.5,
            RiskLevel.CRITICAL: 0.25,
            RiskLevel.HALTED: 0
        }[self.risk_level]
        
        # Adjust for portfolio exposure
        current_exposure = sum(abs(p.get('value', 0)) for p in self.state.positions.values())
        exposure_pct = current_exposure / self.state.capital
        if exposure_pct >= self.limits.max_portfolio_risk:
            return 0  # At max exposure
        remaining_room = self.limits.max_portfolio_risk - exposure_pct
        
        # Final size
        size = base_size * confidence_mult * vol_mult * risk_mult
        size = min(size, remaining_room)  # Don't exceed portfolio limit
        size = max(0, size)  # No negative sizes
        
        return size
    
    def validate_trade(self, symbol: str, direction: int, size: float, 
                       price: float, stop_loss: float) -> tuple[bool, str, float]:
        """Validate a proposed trade"""
        can_trade, reason = self.can_trade()
        if not can_trade:
            return False, reason, 0
        
        # Calculate risk
        risk_per_share = abs(price - stop_loss)
        position_value = size * self.state.capital
        shares = position_value / price
        max_loss = shares * risk_per_share
        max_loss_pct = max_loss / self.state.capital
        
        # Check if risk is acceptable
        if max_loss_pct > self.limits.max_position_size:
            adjusted_size = (self.limits.max_position_size * self.state.capital) / (shares * risk_per_share / size)
            adjusted_size = max(0, min(size, adjusted_size))
            return True, f"Size adjusted for risk: {size:.4f} -> {adjusted_size:.4f}", adjusted_size
        
        return True, "Trade validated", size
    
    def record_trade(self, symbol: str, direction: int, size: float, 
                     price: float, pnl: float = 0):
        """Record a trade execution"""
        self.state.daily_trades += 1
        self.state.daily_pnl += pnl
        self.state.weekly_pnl += pnl
        self.state.capital += pnl
        
        # Update peak capital
        if self.state.capital > self.state.peak_capital:
            self.state.peak_capital = self.state.capital
        
        # Log trade
        trade = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'direction': direction,
            'size': size,
            'price': price,
            'pnl': pnl
        }
        self.state.trade_history.append(trade)
        
        # Update risk level
        self._update_risk_level()
    
    def _update_risk_level(self):
        """Update current risk level based on portfolio state"""
        drawdown = (self.state.peak_capital - self.state.capital) / self.state.peak_capital
        daily_loss = -self.state.daily_pnl / self.state.capital if self.state.daily_pnl < 0 else 0
        
        if drawdown >= 0.08 or daily_loss >= 0.015:
            self.risk_level = RiskLevel.CRITICAL
        elif drawdown >= 0.05 or daily_loss >= 0.01:
            self.risk_level = RiskLevel.HIGH
        elif drawdown >= 0.03 or daily_loss >= 0.005:
            self.risk_level = RiskLevel.ELEVATED
        else:
            self.risk_level = RiskLevel.NORMAL
    
    def daily_reset(self):
        """Reset daily counters"""
        self.state.daily_pnl = 0
        self.state.daily_trades = 0
        self.state.last_reset = datetime.now()
        self._update_risk_level()
    
    def weekly_reset(self):
        """Reset weekly counters"""
        self.state.weekly_pnl = 0
    
    def get_status(self) -> dict:
        """Get current risk status"""
        drawdown = (self.state.peak_capital - self.state.capital) / self.state.peak_capital
        return {
            'can_trade': self.can_trade()[0],
            'risk_level': self.risk_level.value,
            'capital': self.state.capital,
            'peak_capital': self.state.peak_capital,
            'drawdown': drawdown,
            'daily_pnl': self.state.daily_pnl,
            'daily_pnl_pct': self.state.daily_pnl / self.state.capital,
            'daily_trades': self.state.daily_trades,
            'circuit_breaker': self.circuit_breaker.is_halted,
            'halt_reason': self.circuit_breaker.halt_reason
        }
    
    def save_state(self, path: str = "state/risk_state.json"):
        """Save risk manager state"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            'capital': self.state.capital,
            'peak_capital': self.state.peak_capital,
            'daily_pnl': self.state.daily_pnl,
            'weekly_pnl': self.state.weekly_pnl,
            'daily_trades': self.state.daily_trades,
            'risk_level': self.risk_level.value,
            'circuit_breaker_halted': self.circuit_breaker.is_halted,
            'halt_reason': self.circuit_breaker.halt_reason
        }
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, path: str = "state/risk_state.json"):
        """Load risk manager state"""
        if os.path.exists(path):
            with open(path, 'r') as f:
                state = json.load(f)
            self.state.capital = state.get('capital', self.state.capital)
            self.state.peak_capital = state.get('peak_capital', self.state.peak_capital)
            self.state.daily_pnl = state.get('daily_pnl', 0)
            self.state.weekly_pnl = state.get('weekly_pnl', 0)
            self.state.daily_trades = state.get('daily_trades', 0)
            self.risk_level = RiskLevel(state.get('risk_level', 'normal'))
            if state.get('circuit_breaker_halted'):
                self.circuit_breaker.halt(state.get('halt_reason', 'Restored from state'))
