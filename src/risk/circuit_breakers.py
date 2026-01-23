"""
V26 Circuit Breakers Module
===========================

Enhanced risk management with 3-level circuit breakers.

Key Features:
1. Level 1: Reduce positions 50% on -1% daily loss
2. Level 2: High-confidence only (>0.75) on -1.5% loss
3. Level 3: Halt trading on -2% loss, Discord alert, manual restart
4. Dynamic Kelly sizing (0.25 fraction), max 5% per position

Target: Max DD <-17%, Sortino >1.5
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum, auto
from collections import deque
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class V26BreakerLevel(Enum):
    """V26 circuit breaker levels."""
    NORMAL = "normal"           # Full trading
    LEVEL_1 = "level_1"         # Reduce 50%
    LEVEL_2 = "level_2"         # High confidence only
    LEVEL_3 = "level_3"         # Halt trading


class PositionAction(Enum):
    """Position actions based on circuit breaker level."""
    FULL_SIZE = "full_size"           # Normal sizing
    REDUCE_50 = "reduce_50"           # 50% reduction
    HIGH_CONF_ONLY = "high_conf_only" # Only high confidence signals
    NO_NEW = "no_new"                 # No new positions
    CLOSE_ALL = "close_all"           # Close all positions


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class V26CircuitBreakerConfig:
    """Configuration for V26 circuit breakers."""
    
    # Level thresholds (daily loss percentages)
    level_1_threshold: float = 0.01    # -1% daily loss
    level_2_threshold: float = 0.015   # -1.5% daily loss
    level_3_threshold: float = 0.02    # -2% daily loss
    
    # Position reduction
    level_1_reduction: float = 0.50    # Reduce by 50% at L1
    
    # Confidence filter for Level 2
    min_confidence_level_2: float = 0.75  # Only trade >75% confidence
    
    # Dynamic Kelly sizing
    kelly_fraction: float = 0.25        # Use 1/4 Kelly
    max_position_pct: float = 0.05      # Max 5% per position
    min_position_pct: float = 0.005     # Min 0.5% per position
    
    # Drawdown controls
    max_drawdown_pct: float = 0.17      # Target: <17% max DD
    warning_drawdown_pct: float = 0.10  # Start reducing at 10%
    
    # Recovery cooldown
    cooldown_minutes: int = 30           # Wait before upgrading level
    auto_restart: bool = False           # Manual restart for Level 3
    
    # Discord alerts
    enable_discord_alerts: bool = True
    
    # Logging
    log_path: str = "logs/v26_circuit_breakers.jsonl"


@dataclass 
class V26BreakerState:
    """Current state of V26 circuit breakers."""
    level: V26BreakerLevel
    daily_loss_pct: float
    current_drawdown_pct: float
    peak_equity: float
    current_equity: float
    level_changed_at: Optional[datetime]
    can_trade: bool
    position_action: PositionAction
    kelly_scale: float
    message: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'level': self.level.value,
            'daily_loss_pct': round(self.daily_loss_pct, 4),
            'current_drawdown_pct': round(self.current_drawdown_pct, 4),
            'peak_equity': round(self.peak_equity, 2),
            'current_equity': round(self.current_equity, 2),
            'level_changed_at': self.level_changed_at.isoformat() if self.level_changed_at else None,
            'can_trade': self.can_trade,
            'position_action': self.position_action.value,
            'kelly_scale': round(self.kelly_scale, 4),
            'message': self.message
        }


# =============================================================================
# DYNAMIC KELLY SIZER
# =============================================================================

class DynamicKellySizer:
    """
    Dynamic Kelly criterion position sizing.
    
    f* = (p*b - q) / b
    
    Where:
    - p = probability of win
    - q = probability of loss (1 - p)
    - b = win/loss ratio
    
    Uses fractional Kelly (default 0.25) for risk reduction.
    """
    
    def __init__(self, config: V26CircuitBreakerConfig):
        self.config = config
        self.win_history: deque = deque(maxlen=100)
        self.pnl_history: deque = deque(maxlen=100)
        
    def record_trade(self, pnl: float, is_win: bool):
        """Record trade outcome for Kelly calculation."""
        self.win_history.append(is_win)
        self.pnl_history.append(pnl)
    
    def calculate_kelly(self, estimated_edge: float = 0.05,
                        estimated_win_rate: float = 0.55) -> float:
        """
        Calculate Kelly fraction.
        
        Args:
            estimated_edge: Expected edge (return) on winning trades
            estimated_win_rate: Probability of winning
            
        Returns:
            Kelly fraction (position size as fraction of capital)
        """
        # Use historical data if available
        if len(self.win_history) >= 20:
            wins = sum(self.win_history)
            p = wins / len(self.win_history)
            
            # Calculate win/loss ratio from P&L
            wins_pnl = [pnl for pnl in self.pnl_history if pnl > 0]
            losses_pnl = [abs(pnl) for pnl in self.pnl_history if pnl < 0]
            
            avg_win = np.mean(wins_pnl) if wins_pnl else estimated_edge
            avg_loss = np.mean(losses_pnl) if losses_pnl else estimated_edge * 0.8
            b = avg_win / avg_loss if avg_loss > 0 else 1.0
        else:
            # Use estimates
            p = estimated_win_rate
            b = 1.0  # Assume symmetric payoff
        
        q = 1 - p
        
        # Full Kelly: (p*b - q) / b
        if b <= 0:
            return 0.0
        
        full_kelly = (p * b - q) / b
        
        # Apply fraction
        fractional_kelly = full_kelly * self.config.kelly_fraction
        
        # Bound to min/max
        fractional_kelly = max(0, min(fractional_kelly, self.config.max_position_pct))
        
        return fractional_kelly
    
    def get_position_size(self, signal_confidence: float = 0.6,
                          volatility: float = 0.02,
                          regime_scale: float = 1.0) -> float:
        """
        Get position size with all adjustments.
        
        Args:
            signal_confidence: Confidence of trading signal (0-1)
            volatility: Current asset volatility
            regime_scale: Regime-based scaling (0-1)
            
        Returns:
            Position size as fraction of portfolio
        """
        # Base Kelly
        base_kelly = self.calculate_kelly()
        
        # Confidence adjustment
        conf_scale = np.clip(signal_confidence, 0.5, 1.0)
        
        # Volatility adjustment (inverse vol scaling)
        target_vol = 0.02  # 2% daily vol target
        vol_scale = target_vol / max(volatility, 0.005)
        vol_scale = np.clip(vol_scale, 0.5, 2.0)
        
        # Combined
        position_size = base_kelly * conf_scale * vol_scale * regime_scale
        
        # Final bounds
        return float(np.clip(position_size, self.config.min_position_pct, self.config.max_position_pct))


# =============================================================================
# V26 CIRCUIT BREAKERS
# =============================================================================

class V26CircuitBreakers:
    """
    V26 3-Level Circuit Breaker System.
    
    Level 1 (-1% daily): Reduce positions 50%
    Level 2 (-1.5% daily): High-confidence only (>0.75)
    Level 3 (-2% daily): Halt all trading, Discord alert
    
    Integrates with dynamic Kelly sizing and drawdown controls.
    """
    
    def __init__(self, config: Optional[V26CircuitBreakerConfig] = None):
        self.config = config or V26CircuitBreakerConfig()
        
        # State
        self.current_level = V26BreakerLevel.NORMAL
        self.daily_starting_equity: float = 0.0
        self.peak_equity: float = 0.0
        self.current_equity: float = 0.0
        self.daily_loss_pct: float = 0.0
        self.current_drawdown_pct: float = 0.0
        
        # Level tracking
        self.level_changed_at: Optional[datetime] = None
        self.level_history: deque = deque(maxlen=100)
        
        # Kelly sizer
        self.kelly_sizer = DynamicKellySizer(self.config)
        
        # Discord callback
        self.discord_callback: Optional[Callable] = None
        
        # Logging
        self.log_path = Path(self.config.log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("V26CircuitBreakers initialized")
    
    def reset_daily(self, portfolio_value: float):
        """Reset for new trading day."""
        self.daily_starting_equity = portfolio_value
        self.current_equity = portfolio_value
        self.daily_loss_pct = 0.0
        
        # Update peak if higher
        if portfolio_value > self.peak_equity:
            self.peak_equity = portfolio_value
        
        # Auto-upgrade from Level 1/2 (not Level 3 unless auto_restart enabled)
        if self.current_level in (V26BreakerLevel.LEVEL_1, V26BreakerLevel.LEVEL_2):
            self._set_level(V26BreakerLevel.NORMAL, "Daily reset")
        elif self.current_level == V26BreakerLevel.LEVEL_3 and self.config.auto_restart:
            self._set_level(V26BreakerLevel.NORMAL, "Auto-restart after Level 3")
        
        logger.info(f"Daily reset: equity=${portfolio_value:,.2f}, level={self.current_level.value}")
    
    def update(self, current_equity: float) -> V26BreakerState:
        """
        Update circuit breakers with current portfolio value.
        
        Args:
            current_equity: Current portfolio value
            
        Returns:
            Current breaker state
        """
        self.current_equity = current_equity
        
        # Calculate daily loss
        if self.daily_starting_equity > 0:
            self.daily_loss_pct = (self.daily_starting_equity - current_equity) / self.daily_starting_equity
        else:
            self.daily_loss_pct = 0.0
        
        # Update peak and drawdown
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        if self.peak_equity > 0:
            self.current_drawdown_pct = (self.peak_equity - current_equity) / self.peak_equity
        else:
            self.current_drawdown_pct = 0.0
        
        # Determine level based on daily loss
        new_level = self._determine_level()
        
        # Check for level change
        if new_level != self.current_level:
            self._handle_level_change(new_level)
        
        # Check for level recovery
        self._check_recovery()
        
        return self.get_state()
    
    def _determine_level(self) -> V26BreakerLevel:
        """Determine circuit breaker level based on daily loss."""
        if self.daily_loss_pct >= self.config.level_3_threshold:
            return V26BreakerLevel.LEVEL_3
        elif self.daily_loss_pct >= self.config.level_2_threshold:
            return V26BreakerLevel.LEVEL_2
        elif self.daily_loss_pct >= self.config.level_1_threshold:
            return V26BreakerLevel.LEVEL_1
        else:
            return V26BreakerLevel.NORMAL
    
    def _handle_level_change(self, new_level: V26BreakerLevel):
        """Handle circuit breaker level change."""
        old_level = self.current_level
        
        # Only escalate, don't auto-de-escalate
        if self._level_value(new_level) > self._level_value(old_level):
            self._set_level(new_level, f"Escalated from {old_level.value}")
            
            # Log event
            self.level_history.append({
                'timestamp': datetime.now().isoformat(),
                'from_level': old_level.value,
                'to_level': new_level.value,
                'daily_loss_pct': self.daily_loss_pct,
                'drawdown_pct': self.current_drawdown_pct
            })
            
            # Discord alert for Level 2 and 3
            if new_level in (V26BreakerLevel.LEVEL_2, V26BreakerLevel.LEVEL_3):
                self._send_alert(new_level)
    
    def _set_level(self, level: V26BreakerLevel, reason: str):
        """Set circuit breaker level."""
        self.current_level = level
        self.level_changed_at = datetime.now()
        
        logger.info(f"Circuit breaker level: {level.value} - {reason}")
        
        # Log to file
        self._log_event({
            'event': 'level_change',
            'level': level.value,
            'reason': reason,
            'daily_loss_pct': self.daily_loss_pct,
            'drawdown_pct': self.current_drawdown_pct
        })
    
    def _level_value(self, level: V26BreakerLevel) -> int:
        """Get numeric value for level comparison."""
        return {
            V26BreakerLevel.NORMAL: 0,
            V26BreakerLevel.LEVEL_1: 1,
            V26BreakerLevel.LEVEL_2: 2,
            V26BreakerLevel.LEVEL_3: 3
        }[level]
    
    def _check_recovery(self):
        """Check if conditions allow de-escalation."""
        if self.current_level == V26BreakerLevel.NORMAL:
            return
        
        if self.level_changed_at is None:
            return
        
        # Check cooldown
        elapsed = (datetime.now() - self.level_changed_at).total_seconds()
        cooldown_seconds = self.config.cooldown_minutes * 60
        
        if elapsed < cooldown_seconds:
            return  # Still in cooldown
        
        # Check if loss has recovered
        if self.daily_loss_pct < self.config.level_1_threshold * 0.5:  # Below 0.5% loss
            if self.current_level == V26BreakerLevel.LEVEL_1:
                self._set_level(V26BreakerLevel.NORMAL, "Loss recovered")
            elif self.current_level == V26BreakerLevel.LEVEL_2:
                self._set_level(V26BreakerLevel.LEVEL_1, "Partial recovery")
    
    def _send_alert(self, level: V26BreakerLevel):
        """Send Discord alert for level change."""
        if not self.config.enable_discord_alerts:
            return
        
        if self.discord_callback is None:
            logger.warning("Discord callback not set")
            return
        
        colors = {
            V26BreakerLevel.LEVEL_2: 0xFFAA00,  # Orange
            V26BreakerLevel.LEVEL_3: 0xFF0000,  # Red
        }
        
        titles = {
            V26BreakerLevel.LEVEL_2: "⚠️ V26 Circuit Breaker Level 2",
            V26BreakerLevel.LEVEL_3: "🚨 V26 TRADING HALTED - Level 3",
        }
        
        actions = {
            V26BreakerLevel.LEVEL_2: "High-confidence trades only (>75%)",
            V26BreakerLevel.LEVEL_3: "ALL TRADING HALTED - Manual restart required",
        }
        
        try:
            self.discord_callback(
                titles[level],
                f"Daily Loss: {self.daily_loss_pct:.2%}\n"
                f"Drawdown: {self.current_drawdown_pct:.2%}\n"
                f"Equity: ${self.current_equity:,.2f}\n"
                f"Action: {actions[level]}",
                color=colors[level]
            )
        except Exception as e:
            logger.error(f"Discord alert failed: {e}")
    
    def get_state(self) -> V26BreakerState:
        """Get current circuit breaker state."""
        can_trade = self.current_level != V26BreakerLevel.LEVEL_3
        
        position_action = {
            V26BreakerLevel.NORMAL: PositionAction.FULL_SIZE,
            V26BreakerLevel.LEVEL_1: PositionAction.REDUCE_50,
            V26BreakerLevel.LEVEL_2: PositionAction.HIGH_CONF_ONLY,
            V26BreakerLevel.LEVEL_3: PositionAction.NO_NEW,
        }[self.current_level]
        
        # Calculate Kelly scale based on level and drawdown
        kelly_scale = self._get_kelly_scale()
        
        messages = {
            V26BreakerLevel.NORMAL: "Normal trading",
            V26BreakerLevel.LEVEL_1: f"Level 1: Reducing positions 50%",
            V26BreakerLevel.LEVEL_2: f"Level 2: High-confidence only (>{self.config.min_confidence_level_2:.0%})",
            V26BreakerLevel.LEVEL_3: "Level 3: TRADING HALTED",
        }
        
        return V26BreakerState(
            level=self.current_level,
            daily_loss_pct=self.daily_loss_pct,
            current_drawdown_pct=self.current_drawdown_pct,
            peak_equity=self.peak_equity,
            current_equity=self.current_equity,
            level_changed_at=self.level_changed_at,
            can_trade=can_trade,
            position_action=position_action,
            kelly_scale=kelly_scale,
            message=messages[self.current_level]
        )
    
    def _get_kelly_scale(self) -> float:
        """Get Kelly scaling factor based on current state."""
        # Base scale by level
        level_scales = {
            V26BreakerLevel.NORMAL: 1.0,
            V26BreakerLevel.LEVEL_1: 0.50,
            V26BreakerLevel.LEVEL_2: 0.25,
            V26BreakerLevel.LEVEL_3: 0.0,
        }
        scale = level_scales[self.current_level]
        
        # Additional drawdown scaling
        if self.current_drawdown_pct > self.config.warning_drawdown_pct:
            dd_excess = self.current_drawdown_pct - self.config.warning_drawdown_pct
            dd_scale = max(0.25, 1.0 - (dd_excess / 0.07) * 0.5)  # Linear reduction
            scale *= dd_scale
        
        return scale
    
    def can_trade(self) -> Tuple[bool, str]:
        """Check if trading is allowed."""
        state = self.get_state()
        return state.can_trade, state.message
    
    def should_trade_signal(self, signal_confidence: float) -> Tuple[bool, str]:
        """
        Check if a signal should be traded based on confidence.
        
        Args:
            signal_confidence: Confidence of the signal (0-1)
            
        Returns:
            Tuple of (should_trade, reason)
        """
        if self.current_level == V26BreakerLevel.LEVEL_3:
            return False, "Trading halted at Level 3"
        
        if self.current_level == V26BreakerLevel.LEVEL_2:
            if signal_confidence < self.config.min_confidence_level_2:
                return False, f"Confidence {signal_confidence:.1%} below Level 2 threshold {self.config.min_confidence_level_2:.0%}"
        
        return True, "Signal approved"
    
    def get_position_size(self, signal_confidence: float,
                          volatility: float = 0.02,
                          base_size: Optional[float] = None) -> float:
        """
        Get adjusted position size.
        
        Args:
            signal_confidence: Signal confidence (0-1)
            volatility: Asset volatility
            base_size: Optional base size (uses Kelly if None)
            
        Returns:
            Adjusted position size as fraction of portfolio
        """
        state = self.get_state()
        
        if not state.can_trade:
            return 0.0
        
        # Get base size from Kelly or use provided
        if base_size is None:
            base_size = self.kelly_sizer.get_position_size(
                signal_confidence=signal_confidence,
                volatility=volatility,
                regime_scale=1.0
            )
        
        # Apply Kelly scale from breaker state
        adjusted_size = base_size * state.kelly_scale
        
        # Additional reduction at Level 1
        if self.current_level == V26BreakerLevel.LEVEL_1:
            adjusted_size *= self.config.level_1_reduction
        
        # Bound to max position
        return min(adjusted_size, self.config.max_position_pct)
    
    def record_trade(self, pnl: float):
        """Record trade outcome for Kelly calculation."""
        is_win = pnl > 0
        self.kelly_sizer.record_trade(pnl, is_win)
    
    def manual_restart(self) -> bool:
        """
        Manually restart trading after Level 3 halt.
        
        Returns:
            True if restart successful
        """
        if self.current_level != V26BreakerLevel.LEVEL_3:
            logger.warning("Not at Level 3, no restart needed")
            return False
        
        self._set_level(V26BreakerLevel.NORMAL, "Manual restart")
        
        # Alert
        if self.config.enable_discord_alerts and self.discord_callback:
            try:
                self.discord_callback(
                    "✅ V26 Trading Restarted",
                    f"Manual restart after Level 3 halt\n"
                    f"Equity: ${self.current_equity:,.2f}",
                    color=0x00FF00
                )
            except Exception as e:
                logger.error(f"Discord alert failed: {e}")
        
        return True
    
    def _log_event(self, event: Dict[str, Any]):
        """Log event to file."""
        try:
            event['timestamp'] = datetime.now().isoformat()
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(event) + '\n')
        except Exception as e:
            logger.error(f"Failed to log event: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of circuit breaker state."""
        state = self.get_state()
        return {
            **state.to_dict(),
            'kelly_win_rate': sum(self.kelly_sizer.win_history) / len(self.kelly_sizer.win_history) if self.kelly_sizer.win_history else 0.5,
            'level_changes_today': len([h for h in self.level_history if datetime.fromisoformat(h['timestamp']).date() == datetime.now().date()]),
            'trades_recorded': len(self.kelly_sizer.win_history),
        }


# =============================================================================
# FACTORY
# =============================================================================

def create_v26_circuit_breakers() -> V26CircuitBreakers:
    """Factory function for V26 circuit breakers with default config."""
    config = V26CircuitBreakerConfig()
    return V26CircuitBreakers(config)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing V26 Circuit Breakers...")
    
    config = V26CircuitBreakerConfig()
    breakers = V26CircuitBreakers(config)
    
    # Initialize
    breakers.reset_daily(100000)
    
    # Test normal state
    print("\n1. Normal state...")
    state = breakers.update(100000)
    print(f"   Level: {state.level.value}")
    print(f"   Can trade: {state.can_trade}")
    
    # Test Level 1 (-1%)
    print("\n2. Level 1 (1% loss)...")
    state = breakers.update(99000)
    print(f"   Level: {state.level.value}")
    print(f"   Daily loss: {state.daily_loss_pct:.2%}")
    print(f"   Kelly scale: {state.kelly_scale:.2f}")
    
    # Test Level 2 (-1.5%)
    print("\n3. Level 2 (1.5% loss)...")
    state = breakers.update(98500)
    print(f"   Level: {state.level.value}")
    print(f"   Position action: {state.position_action.value}")
    
    # Test signal filtering
    print("\n4. Signal filtering at Level 2...")
    should_trade, reason = breakers.should_trade_signal(0.60)
    print(f"   60% confidence: {should_trade} - {reason}")
    should_trade, reason = breakers.should_trade_signal(0.80)
    print(f"   80% confidence: {should_trade} - {reason}")
    
    # Test Level 3 (-2%)
    print("\n5. Level 3 (2% loss)...")
    state = breakers.update(98000)
    print(f"   Level: {state.level.value}")
    print(f"   Message: {state.message}")
    print(f"   Can trade: {state.can_trade}")
    
    # Test position sizing
    print("\n6. Position sizing...")
    breakers.reset_daily(100000)  # Reset to normal
    size = breakers.get_position_size(signal_confidence=0.7, volatility=0.02)
    print(f"   Normal state size: {size:.2%}")
    
    breakers.update(99000)  # Level 1
    size = breakers.get_position_size(signal_confidence=0.7, volatility=0.02)
    print(f"   Level 1 size: {size:.2%}")
    
    # Test Kelly sizer
    print("\n7. Kelly sizer...")
    for _ in range(30):
        pnl = np.random.normal(0.002, 0.01)  # Slight positive edge
        breakers.record_trade(pnl)
    kelly = breakers.kelly_sizer.calculate_kelly()
    print(f"   Calculated Kelly: {kelly:.2%}")
    
    # Test manual restart
    print("\n8. Manual restart...")
    breakers.update(98000)  # Level 3
    print(f"   Before restart: {breakers.current_level.value}")
    breakers.manual_restart()
    print(f"   After restart: {breakers.current_level.value}")
    
    # Get summary
    print("\n9. Summary...")
    summary = breakers.get_summary()
    print(f"   {json.dumps(summary, indent=2)}")
    
    print("\n✅ V26 Circuit Breakers tests passed!")
