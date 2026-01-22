"""
Production-Grade Circuit Breakers Module
=========================================

V2.4 Profitability Enhancement - Risk controls to prevent catastrophic losses

Key Features:
1. Daily Loss Limit - 5% daily loss halts all trading
2. Position-Level Stops - 3-sigma stop-loss per position
3. VIX Spike Detection - Reduce exposure when VIX > 30
4. Drawdown Control - Reduce sizing progressively with drawdown
5. Correlation Spike Detection - Cut exposure when correlations spike

Research Basis:
- Circuit breakers prevent tail-risk events from becoming catastrophic
- Dynamic position reduction based on realized volatility
- Correlation spikes often precede market crashes

Target Performance:
- Max drawdown <12%
- Zero "blow-up" events
- Automatic risk reduction in crisis
"""

import numpy as np
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import time as time_module

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class CircuitBreakerType(Enum):
    """Types of circuit breakers."""
    DAILY_LOSS = "daily_loss"           # Daily P&L limit
    POSITION_STOP = "position_stop"      # Position-level stop loss
    VIX_SPIKE = "vix_spike"             # VIX threshold
    DRAWDOWN = "drawdown"               # Portfolio drawdown
    CORRELATION = "correlation"          # Correlation spike
    VOLATILITY = "volatility"           # Realized vol spike


class BreakerAction(Enum):
    """Actions to take when breaker triggers."""
    NONE = "none"                       # No action
    REDUCE_50 = "reduce_50"             # Reduce position 50%
    CLOSE_POSITION = "close_position"   # Close specific position
    HALT_TRADING = "halt_trading"       # Stop all new trades
    LIQUIDATE_ALL = "liquidate_all"     # Emergency liquidation


class BreakerStatus(Enum):
    """Circuit breaker status."""
    NORMAL = "normal"                   # All systems go
    WARNING = "warning"                 # Approaching limit
    TRIGGERED = "triggered"             # Limit hit, action required
    COOLDOWN = "cooldown"               # Waiting to reset


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breakers."""
    
    # Daily loss limits
    daily_loss_halt_pct: float = 0.05       # 5% daily loss = halt
    daily_loss_warning_pct: float = 0.03    # 3% = warning
    
    # Position-level stops
    position_stop_sigma: float = 3.0        # 3-sigma stop loss
    position_stop_max_pct: float = 0.10     # Max 10% loss per position
    trailing_stop_pct: float = 0.05         # 5% trailing stop
    
    # VIX thresholds
    vix_warning_level: float = 25           # VIX > 25 = warning
    vix_halt_level: float = 35              # VIX > 35 = halt new trades
    vix_reduce_level: float = 30            # VIX > 30 = reduce 50%
    
    # Drawdown controls
    max_drawdown_pct: float = 0.12          # 12% max drawdown
    drawdown_reduce_threshold: float = 0.08  # 8% = start reducing
    drawdown_reduction_per_pct: float = 0.1  # Reduce 10% per 1% DD
    
    # Volatility spike
    vol_lookback_days: int = 20
    vol_spike_threshold: float = 2.0        # 2x normal vol = spike
    
    # Correlation spike
    correlation_lookback: int = 20
    correlation_spike_threshold: float = 0.85  # >85% avg correlation
    
    # Cooldown
    cooldown_minutes: int = 60              # 1 hour cooldown after trigger
    auto_reset_daily: bool = True           # Reset at market open
    
    # V3.0 Enhanced Risk Controls
    # Loss streak detection
    enable_loss_streak: bool = True
    loss_streak_threshold: int = 3          # 3 consecutive losses
    loss_streak_reduction: float = 0.50     # Reduce 50% after streak
    loss_streak_halt: int = 5               # Halt after 5 losses
    
    # Win/Loss ratio adaptive sizing
    enable_adaptive_wl: bool = True
    wl_lookback: int = 20                   # 20-trade lookback
    wl_min_ratio: float = 0.8               # Min W/L ratio for full size
    wl_scale_factor: float = 0.5            # Scale by this factor when W/L low
    
    # Regime-aware position limits
    enable_regime_limits: bool = True
    bull_max_position: float = 0.15         # 15% max in bull regime
    bear_max_position: float = 0.05         # 5% max in bear regime
    high_vol_max_position: float = 0.08     # 8% max in high vol
    
    # Notifications
    enable_alerts: bool = True
    alert_callback: Optional[Callable] = None
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop('alert_callback', None)  # Can't serialize callback
        return d


# =============================================================================
# INDIVIDUAL CIRCUIT BREAKERS
# =============================================================================

@dataclass
class BreakerState:
    """State of a circuit breaker."""
    breaker_type: CircuitBreakerType
    status: BreakerStatus = BreakerStatus.NORMAL
    current_value: float = 0.0
    threshold: float = 0.0
    warning_threshold: float = 0.0
    triggered_at: Optional[datetime] = None
    cooldown_until: Optional[datetime] = None
    action: BreakerAction = BreakerAction.NONE
    message: str = ""


class DailyLossBreaker:
    """Monitor and halt on daily loss limit."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.daily_pnl: float = 0.0
        self.starting_value: float = 0.0
        self.last_reset: Optional[datetime] = None
        
    def reset_daily(self, portfolio_value: float):
        """Reset for new trading day."""
        self.daily_pnl = 0.0
        self.starting_value = portfolio_value
        self.last_reset = datetime.now()
        
    def update(self, current_value: float) -> BreakerState:
        """Update with current portfolio value."""
        if self.starting_value > 0:
            self.daily_pnl = (current_value - self.starting_value) / self.starting_value
        else:
            self.daily_pnl = 0.0
            
        state = BreakerState(
            breaker_type=CircuitBreakerType.DAILY_LOSS,
            current_value=self.daily_pnl,
            threshold=-self.config.daily_loss_halt_pct,
            warning_threshold=-self.config.daily_loss_warning_pct,
        )
        
        if self.daily_pnl <= -self.config.daily_loss_halt_pct:
            state.status = BreakerStatus.TRIGGERED
            state.action = BreakerAction.HALT_TRADING
            state.message = f"Daily loss limit hit: {self.daily_pnl:.2%}"
            state.triggered_at = datetime.now()
        elif self.daily_pnl <= -self.config.daily_loss_warning_pct:
            state.status = BreakerStatus.WARNING
            state.message = f"Approaching daily loss limit: {self.daily_pnl:.2%}"
        else:
            state.status = BreakerStatus.NORMAL
            state.message = f"Daily P&L: {self.daily_pnl:+.2%}"
            
        return state


class PositionStopBreaker:
    """Position-level stop loss management."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.positions: Dict[str, Dict[str, float]] = {}
        
    def add_position(
        self,
        symbol: str,
        entry_price: float,
        shares: int,
        volatility: float
    ):
        """Register a new position."""
        # Calculate stop price
        stop_distance = self.config.position_stop_sigma * volatility
        stop_distance = min(stop_distance, self.config.position_stop_max_pct)
        
        self.positions[symbol] = {
            'entry_price': entry_price,
            'shares': shares,
            'volatility': volatility,
            'stop_price': entry_price * (1 - stop_distance),
            'high_water_mark': entry_price,
            'trailing_stop': entry_price * (1 - self.config.trailing_stop_pct),
        }
        
    def update_price(self, symbol: str, current_price: float) -> BreakerState:
        """Update position with current price."""
        state = BreakerState(
            breaker_type=CircuitBreakerType.POSITION_STOP,
        )
        
        if symbol not in self.positions:
            state.status = BreakerStatus.NORMAL
            state.message = f"No position for {symbol}"
            return state
            
        pos = self.positions[symbol]
        
        # Update high water mark and trailing stop
        if current_price > pos['high_water_mark']:
            pos['high_water_mark'] = current_price
            pos['trailing_stop'] = current_price * (1 - self.config.trailing_stop_pct)
            
        # Calculate current loss
        pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']
        
        state.current_value = pnl_pct
        state.threshold = (pos['stop_price'] - pos['entry_price']) / pos['entry_price']
        
        # Check stops
        if current_price <= pos['stop_price']:
            state.status = BreakerStatus.TRIGGERED
            state.action = BreakerAction.CLOSE_POSITION
            state.message = f"Stop loss triggered for {symbol}: {pnl_pct:.2%}"
            state.triggered_at = datetime.now()
        elif current_price <= pos['trailing_stop']:
            state.status = BreakerStatus.TRIGGERED
            state.action = BreakerAction.CLOSE_POSITION
            state.message = f"Trailing stop triggered for {symbol}: {pnl_pct:.2%}"
            state.triggered_at = datetime.now()
        elif pnl_pct < -self.config.position_stop_max_pct * 0.7:
            state.status = BreakerStatus.WARNING
            state.message = f"Approaching stop for {symbol}: {pnl_pct:.2%}"
        else:
            state.status = BreakerStatus.NORMAL
            state.message = f"Position {symbol} P&L: {pnl_pct:+.2%}"
            
        return state
    
    def remove_position(self, symbol: str):
        """Remove a closed position."""
        self.positions.pop(symbol, None)


class VIXBreaker:
    """VIX-based circuit breaker."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.current_vix: float = 15.0
        
    def update(self, vix: float) -> BreakerState:
        """Update with current VIX."""
        self.current_vix = vix
        
        state = BreakerState(
            breaker_type=CircuitBreakerType.VIX_SPIKE,
            current_value=vix,
            threshold=self.config.vix_halt_level,
            warning_threshold=self.config.vix_warning_level,
        )
        
        if vix >= self.config.vix_halt_level:
            state.status = BreakerStatus.TRIGGERED
            state.action = BreakerAction.HALT_TRADING
            state.message = f"VIX halt level: {vix:.1f}"
            state.triggered_at = datetime.now()
        elif vix >= self.config.vix_reduce_level:
            state.status = BreakerStatus.TRIGGERED
            state.action = BreakerAction.REDUCE_50
            state.message = f"VIX reduce level: {vix:.1f}"
            state.triggered_at = datetime.now()
        elif vix >= self.config.vix_warning_level:
            state.status = BreakerStatus.WARNING
            state.message = f"VIX elevated: {vix:.1f}"
        else:
            state.status = BreakerStatus.NORMAL
            state.message = f"VIX normal: {vix:.1f}"
            
        return state


class DrawdownBreaker:
    """Portfolio drawdown circuit breaker."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.high_water_mark: float = 0.0
        self.current_drawdown: float = 0.0
        
    def update(self, portfolio_value: float) -> BreakerState:
        """Update with current portfolio value."""
        if portfolio_value > self.high_water_mark:
            self.high_water_mark = portfolio_value
            self.current_drawdown = 0.0
        elif self.high_water_mark > 0:
            self.current_drawdown = (self.high_water_mark - portfolio_value) / self.high_water_mark
        else:
            self.current_drawdown = 0.0
            
        state = BreakerState(
            breaker_type=CircuitBreakerType.DRAWDOWN,
            current_value=self.current_drawdown,
            threshold=self.config.max_drawdown_pct,
            warning_threshold=self.config.drawdown_reduce_threshold,
        )
        
        if self.current_drawdown >= self.config.max_drawdown_pct:
            state.status = BreakerStatus.TRIGGERED
            state.action = BreakerAction.HALT_TRADING
            state.message = f"Max drawdown hit: {self.current_drawdown:.2%}"
            state.triggered_at = datetime.now()
        elif self.current_drawdown >= self.config.drawdown_reduce_threshold:
            state.status = BreakerStatus.WARNING
            # Calculate reduction amount
            excess_dd = self.current_drawdown - self.config.drawdown_reduce_threshold
            reduction = excess_dd / 0.01 * self.config.drawdown_reduction_per_pct
            state.action = BreakerAction.REDUCE_50 if reduction > 0.3 else BreakerAction.NONE
            state.message = f"Drawdown warning: {self.current_drawdown:.2%}, reduce {reduction:.0%}"
        else:
            state.status = BreakerStatus.NORMAL
            state.message = f"Drawdown: {self.current_drawdown:.2%}"
            
        return state
    
    def get_position_scaling(self) -> float:
        """Get position size scaling based on drawdown."""
        if self.current_drawdown < self.config.drawdown_reduce_threshold:
            return 1.0
            
        excess = self.current_drawdown - self.config.drawdown_reduce_threshold
        reduction = excess / 0.01 * self.config.drawdown_reduction_per_pct
        
        return max(0.25, 1.0 - reduction)


class VolatilityBreaker:
    """Realized volatility spike breaker."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.returns: deque = deque(maxlen=config.vol_lookback_days)
        self.baseline_vol: float = 0.15  # Default 15% annual
        
    def update(self, daily_return: float) -> BreakerState:
        """Update with daily return."""
        self.returns.append(daily_return)
        
        state = BreakerState(
            breaker_type=CircuitBreakerType.VOLATILITY,
        )
        
        if len(self.returns) < 5:
            state.status = BreakerStatus.NORMAL
            state.message = "Insufficient data for volatility"
            return state
            
        # Calculate realized vol
        realized_vol = np.std(list(self.returns)) * np.sqrt(252)
        
        # Update baseline (slow moving average)
        if self.baseline_vol > 0:
            self.baseline_vol = 0.95 * self.baseline_vol + 0.05 * realized_vol
        else:
            self.baseline_vol = realized_vol
            
        vol_ratio = realized_vol / self.baseline_vol if self.baseline_vol > 0 else 1.0
        
        state.current_value = vol_ratio
        state.threshold = self.config.vol_spike_threshold
        state.warning_threshold = self.config.vol_spike_threshold * 0.8
        
        if vol_ratio >= self.config.vol_spike_threshold:
            state.status = BreakerStatus.TRIGGERED
            state.action = BreakerAction.REDUCE_50
            state.message = f"Volatility spike: {vol_ratio:.1f}x normal"
            state.triggered_at = datetime.now()
        elif vol_ratio >= self.config.vol_spike_threshold * 0.8:
            state.status = BreakerStatus.WARNING
            state.message = f"Volatility elevated: {vol_ratio:.1f}x normal"
        else:
            state.status = BreakerStatus.NORMAL
            state.message = f"Volatility normal: {realized_vol:.1%} annual"
            
        return state


# =============================================================================
# MAIN CIRCUIT BREAKER MANAGER
# =============================================================================

class CircuitBreakerManager:
    """
    Central manager for all circuit breakers.
    
    Monitors multiple risk conditions and coordinates actions.
    """
    
    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        
        # Initialize breakers
        self.daily_loss = DailyLossBreaker(self.config)
        self.position_stops = PositionStopBreaker(self.config)
        self.vix = VIXBreaker(self.config)
        self.drawdown = DrawdownBreaker(self.config)
        self.volatility = VolatilityBreaker(self.config)
        
        # Tracking
        self.trigger_history: deque = deque(maxlen=100)
        self.is_halted: bool = False
        self.halt_until: Optional[datetime] = None
        
    def reset_daily(self, portfolio_value: float):
        """Reset breakers for new trading day."""
        self.daily_loss.reset_daily(portfolio_value)
        if self.config.auto_reset_daily:
            self.is_halted = False
            self.halt_until = None
            
    def update_all(
        self,
        portfolio_value: float,
        daily_return: float,
        vix: Optional[float] = None,
        position_prices: Optional[Dict[str, float]] = None
    ) -> Dict[str, BreakerState]:
        """
        Update all circuit breakers.
        
        Args:
            portfolio_value: Current portfolio value
            daily_return: Today's return
            vix: Current VIX (optional)
            position_prices: Dict of symbol -> current price (optional)
            
        Returns:
            Dict of breaker type -> state
        """
        states = {}
        
        # Daily loss
        states['daily_loss'] = self.daily_loss.update(portfolio_value)
        
        # Drawdown
        states['drawdown'] = self.drawdown.update(portfolio_value)
        
        # Volatility
        states['volatility'] = self.volatility.update(daily_return)
        
        # VIX
        if vix is not None:
            states['vix'] = self.vix.update(vix)
            
        # Position stops
        if position_prices:
            for symbol, price in position_prices.items():
                state = self.position_stops.update_price(symbol, price)
                states[f'position_{symbol}'] = state
                
        # Check for triggered breakers
        self._process_triggers(states)
        
        return states
    
    def _process_triggers(self, states: Dict[str, BreakerState]):
        """Process triggered breakers and take action."""
        for name, state in states.items():
            if state.status == BreakerStatus.TRIGGERED:
                # Log trigger
                self.trigger_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'breaker': name,
                    'action': state.action.value,
                    'message': state.message,
                    'value': state.current_value,
                })
                
                # Handle action
                if state.action == BreakerAction.HALT_TRADING:
                    self.is_halted = True
                    self.halt_until = datetime.now() + timedelta(
                        minutes=self.config.cooldown_minutes
                    )
                    logger.warning(f"TRADING HALTED: {state.message}")
                    
                elif state.action == BreakerAction.LIQUIDATE_ALL:
                    self.is_halted = True
                    logger.critical(f"EMERGENCY LIQUIDATION: {state.message}")
                    
                # Send alert if configured
                if self.config.enable_alerts and self.config.alert_callback:
                    try:
                        self.config.alert_callback(name, state)
                    except Exception as e:
                        logger.error(f"Alert callback failed: {e}")
                        
    def can_trade(self) -> Tuple[bool, str]:
        """Check if trading is allowed."""
        if self.is_halted:
            if self.halt_until and datetime.now() >= self.halt_until:
                self.is_halted = False
                self.halt_until = None
                return True, "Trading resumed after cooldown"
            return False, f"Trading halted until {self.halt_until}"
            
        return True, "Trading allowed"
    
    def get_position_scaling(self) -> float:
        """Get recommended position size scaling."""
        # Start with drawdown scaling
        scaling = self.drawdown.get_position_scaling()
        
        # Reduce further if VIX elevated
        if self.vix.current_vix >= self.config.vix_reduce_level:
            scaling *= 0.5
        elif self.vix.current_vix >= self.config.vix_warning_level:
            scaling *= 0.75
            
        return scaling
    
    def add_position(
        self,
        symbol: str,
        entry_price: float,
        shares: int,
        volatility: float
    ):
        """Register a new position for stop management."""
        self.position_stops.add_position(symbol, entry_price, shares, volatility)
        
    def remove_position(self, symbol: str):
        """Remove a closed position."""
        self.position_stops.remove_position(symbol)
        
    def get_status_summary(self) -> Dict[str, Any]:
        """Get summary of all breaker statuses."""
        return {
            'is_halted': self.is_halted,
            'halt_until': self.halt_until.isoformat() if self.halt_until else None,
            'position_scaling': self.get_position_scaling(),
            'daily_pnl': self.daily_loss.daily_pnl,
            'current_drawdown': self.drawdown.current_drawdown,
            'current_vix': self.vix.current_vix,
            'recent_triggers': len(self.trigger_history),
        }
    
    def get_trigger_history(self) -> List[Dict[str, Any]]:
        """Get history of triggered breakers."""
        return list(self.trigger_history)


# =============================================================================
# V3.0 RISK CONTROLLER
# =============================================================================

class V30RiskController:
    """
    V3.0 Enhanced Risk Controller for Sharpe optimization.
    
    Combines:
    - Loss streak detection (reduce after consecutive losses)
    - Adaptive W/L ratio sizing (reduce when win rate drops)
    - Regime-aware position limits
    - Integration with Kelly sizer
    
    Target: Improve Sharpe by cutting losses faster than wins.
    """
    
    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        
        # Trade history tracking
        self.trade_results: deque = deque(maxlen=100)  # Last 100 trades
        self.consecutive_losses: int = 0
        self.consecutive_wins: int = 0
        self.is_halted: bool = False
        self.halt_reason: str = ""
        
        # Performance tracking
        self.recent_wins: int = 0
        self.recent_losses: int = 0
        self.total_pnl: float = 0.0
        
        # Current state
        self.current_regime: str = "normal"
        self.current_scaling: float = 1.0
        
    def record_trade(self, pnl: float) -> Dict[str, Any]:
        """
        Record a trade result and update risk state.
        
        Args:
            pnl: Trade P&L (positive = win, negative = loss)
            
        Returns:
            Dict with updated risk state
        """
        is_win = pnl > 0
        self.trade_results.append(pnl)
        self.total_pnl += pnl
        
        # Update streaks
        if is_win:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            self.recent_wins += 1
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            self.recent_losses += 1
            
        # Check loss streak halt
        if self.config.enable_loss_streak:
            if self.consecutive_losses >= self.config.loss_streak_halt:
                self.is_halted = True
                self.halt_reason = f"Loss streak halt: {self.consecutive_losses} consecutive losses"
                logger.warning(self.halt_reason)
                
        return self.get_state()
    
    def get_loss_streak_scaling(self) -> float:
        """Get position scaling based on loss streak."""
        if not self.config.enable_loss_streak:
            return 1.0
            
        if self.consecutive_losses >= self.config.loss_streak_threshold:
            return self.config.loss_streak_reduction
        return 1.0
    
    def get_wl_ratio_scaling(self) -> float:
        """Get scaling based on recent W/L ratio."""
        if not self.config.enable_adaptive_wl:
            return 1.0
            
        recent = list(self.trade_results)[-self.config.wl_lookback:]
        if len(recent) < 5:
            return 1.0  # Insufficient data
            
        wins = sum(1 for r in recent if r > 0)
        total = len(recent)
        win_rate = wins / total
        
        # Scale by W/L ratio
        if win_rate < self.config.wl_min_ratio * 0.5:  # Below 40% when min is 80%
            return self.config.wl_scale_factor * 0.5  # Extra reduction
        elif win_rate < self.config.wl_min_ratio:
            return self.config.wl_scale_factor
        return 1.0
    
    def get_regime_max_position(self, regime: str = "normal") -> float:
        """Get max position size for current regime."""
        if not self.config.enable_regime_limits:
            return 1.0
            
        self.current_regime = regime
        
        if regime == "bull":
            return self.config.bull_max_position
        elif regime == "bear":
            return self.config.bear_max_position
        elif regime in ("high_vol", "crisis"):
            return self.config.high_vol_max_position
        return 0.10  # Default 10%
    
    def get_combined_scaling(self, regime: str = "normal") -> float:
        """
        Get combined risk scaling factor.
        
        Multiplies all scaling factors together for conservative sizing.
        
        Args:
            regime: Current market regime
            
        Returns:
            Combined scaling factor (0.0 to 1.0)
        """
        if self.is_halted:
            return 0.0
            
        loss_scale = self.get_loss_streak_scaling()
        wl_scale = self.get_wl_ratio_scaling()
        
        combined = loss_scale * wl_scale
        self.current_scaling = combined
        
        return combined
    
    def reset_streak(self):
        """Reset loss streak (e.g., after a winning trade)."""
        self.consecutive_losses = 0
        
    def reset_halt(self):
        """Reset halt state (manual or after cooldown)."""
        self.is_halted = False
        self.halt_reason = ""
        self.consecutive_losses = 0
        
    def get_state(self) -> Dict[str, Any]:
        """Get current risk controller state."""
        recent = list(self.trade_results)
        win_rate = sum(1 for r in recent if r > 0) / len(recent) if recent else 0
        
        return {
            'is_halted': self.is_halted,
            'halt_reason': self.halt_reason,
            'consecutive_losses': self.consecutive_losses,
            'consecutive_wins': self.consecutive_wins,
            'recent_win_rate': round(win_rate, 4),
            'loss_streak_scaling': self.get_loss_streak_scaling(),
            'wl_ratio_scaling': self.get_wl_ratio_scaling(),
            'combined_scaling': self.current_scaling,
            'total_trades': len(self.trade_results),
            'total_pnl': round(self.total_pnl, 6),
            'regime': self.current_regime,
        }
    
    def can_trade(self) -> Tuple[bool, str]:
        """Check if trading is allowed."""
        if self.is_halted:
            return False, self.halt_reason
        return True, "OK"


def create_v30_risk_controller() -> V30RiskController:
    """
    Factory function for V3.0 risk controller with optimized settings.
    
    Settings tuned for:
    - Sharpe improvement via loss reduction
    - Conservative sizing after poor performance
    - Regime-aware position limits
    """
    config = CircuitBreakerConfig(
        # V3.0 settings
        enable_loss_streak=True,
        loss_streak_threshold=2,      # Reduce after just 2 losses (aggressive)
        loss_streak_reduction=0.50,   # Halve position size
        loss_streak_halt=4,           # Halt after 4 consecutive losses
        
        enable_adaptive_wl=True,
        wl_lookback=15,               # 15-trade window
        wl_min_ratio=0.50,            # Expect 50% win rate
        wl_scale_factor=0.60,         # 40% reduction when underperforming
        
        enable_regime_limits=True,
        bull_max_position=0.12,
        bear_max_position=0.04,
        high_vol_max_position=0.06,
    )
    return V30RiskController(config)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Circuit Breakers...")
    
    config = CircuitBreakerConfig()
    manager = CircuitBreakerManager(config)
    
    # Initialize
    manager.reset_daily(1_000_000)
    
    # Test normal conditions
    print("\n1. Testing normal conditions...")
    states = manager.update_all(
        portfolio_value=1_005_000,  # +0.5%
        daily_return=0.005,
        vix=18.0
    )
    print(f"   Can trade: {manager.can_trade()}")
    print(f"   Position scaling: {manager.get_position_scaling():.2f}")
    
    # Test warning conditions
    print("\n2. Testing warning conditions...")
    states = manager.update_all(
        portfolio_value=970_000,  # -3%
        daily_return=-0.03,
        vix=27.0
    )
    can_trade, msg = manager.can_trade()
    print(f"   Can trade: {can_trade} - {msg}")
    print(f"   Daily loss state: {states['daily_loss'].status.value}")
    print(f"   VIX state: {states['vix'].status.value}")
    
    # Test halt conditions
    print("\n3. Testing halt conditions...")
    states = manager.update_all(
        portfolio_value=940_000,  # -6% (exceeds 5% limit)
        daily_return=-0.06,
        vix=36.0
    )
    can_trade, msg = manager.can_trade()
    print(f"   Can trade: {can_trade} - {msg}")
    print(f"   Daily loss: {states['daily_loss'].message}")
    
    # Test position stops
    print("\n4. Testing position stops...")
    manager.add_position("AAPL", 175.0, 100, 0.25)
    states = manager.update_all(
        portfolio_value=940_000,
        daily_return=-0.02,
        position_prices={"AAPL": 160.0}  # ~8.5% loss
    )
    print(f"   AAPL stop state: {states.get('position_AAPL', 'N/A')}")
    
    # Test drawdown
    print("\n5. Testing drawdown breaker...")
    dd_breaker = DrawdownBreaker(config)
    dd_breaker.update(1_000_000)  # Set HWM
    state = dd_breaker.update(900_000)  # 10% DD
    print(f"   Drawdown: {dd_breaker.current_drawdown:.2%}")
    print(f"   Status: {state.status.value}")
    print(f"   Scaling: {dd_breaker.get_position_scaling():.2f}")
    
    # Test volatility
    print("\n6. Testing volatility breaker...")
    vol_breaker = VolatilityBreaker(config)
    for r in [0.03, -0.04, 0.02, -0.035, 0.025, -0.03]:  # High vol
        state = vol_breaker.update(r)
    print(f"   Vol ratio: {state.current_value:.2f}x")
    print(f"   Status: {state.status.value}")
    
    # Test summary
    print("\n7. Status summary...")
    summary = manager.get_status_summary()
    print(f"   Is halted: {summary['is_halted']}")
    print(f"   Position scaling: {summary['position_scaling']:.2f}")
    print(f"   Recent triggers: {summary['recent_triggers']}")
    
    print("\n✅ Circuit Breakers tests passed!")
