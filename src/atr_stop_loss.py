"""
ATR-Based Dynamic Stop-Loss System for Phase 7.

Implements multiple stop-loss mechanisms:
- ATR-based trailing stops
- Drawdown-based position reduction  
- Volatility-adjusted stops
- Time-based stops for stale positions

Stop levels tested: 1.5x, 2.0x, 2.5x ATR
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StopLossConfig:
    """Configuration for stop-loss system."""
    atr_multiplier: float = 2.0  # Stop at entry - 2x ATR
    atr_period: int = 14  # ATR calculation period
    trailing: bool = True  # Use trailing stops
    trail_activation_pct: float = 0.05  # Trail after 5% profit
    max_loss_pct: float = 0.15  # Maximum loss per position (hard stop)
    drawdown_reduction_trigger: float = 0.10  # Reduce at 10% drawdown
    drawdown_reduction_pct: float = 0.50  # Reduce position by 50%
    time_stop_days: int = 60  # Close if no profit after 60 days


@dataclass
class StopLossLevel:
    """Stop loss levels for a position."""
    ticker: str
    entry_price: float
    entry_date: datetime
    current_price: float
    
    # ATR-based stop
    atr_value: float
    atr_stop: float  # Initial ATR-based stop
    
    # Trailing stop
    highest_price: float = 0.0
    trailing_stop: float = 0.0
    trailing_active: bool = False
    
    # Current effective stop
    effective_stop: float = 0.0
    stop_type: str = "atr"  # 'atr', 'trailing', 'hard', 'time'
    
    # P&L tracking
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    days_held: int = 0


@dataclass  
class StopLossEvent:
    """Record of a stop-loss trigger."""
    ticker: str
    trigger_date: datetime
    trigger_type: str  # 'atr', 'trailing', 'drawdown', 'time', 'max_loss'
    entry_price: float
    exit_price: float
    stop_price: float
    pnl_pct: float
    days_held: int


class ATRCalculator:
    """Calculate Average True Range for volatility-based stops."""
    
    @staticmethod
    def calculate_atr(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14,
    ) -> float:
        """
        Calculate ATR (Average True Range).
        
        True Range = max(
            high - low,
            abs(high - previous_close),
            abs(low - previous_close)
        )
        
        ATR = rolling mean of True Range
        """
        if len(close) < period + 1:
            return (high[-1] - low[-1]) if len(high) > 0 else 0.0
        
        # True Range components
        hl = high[1:] - low[1:]
        hpc = np.abs(high[1:] - close[:-1])
        lpc = np.abs(low[1:] - close[:-1])
        
        true_range = np.maximum(hl, np.maximum(hpc, lpc))
        
        # ATR is smoothed average
        atr = np.mean(true_range[-period:])
        
        return atr
    
    @staticmethod
    def calculate_atr_from_df(
        df: pd.DataFrame,
        period: int = 14,
    ) -> float:
        """Calculate ATR from OHLCV DataFrame."""
        high = df['high'].values if 'high' in df.columns else df['High'].values
        low = df['low'].values if 'low' in df.columns else df['Low'].values
        close = df['close'].values if 'close' in df.columns else df['Close'].values
        
        return ATRCalculator.calculate_atr(high, low, close, period)


class DynamicStopLossManager:
    """
    Manages dynamic stop-losses for all positions.
    
    Features:
    - ATR-based initial stops
    - Trailing stops after profit threshold
    - Hard stops for max loss protection
    - Drawdown-based position reduction
    - Time-based exits
    """
    
    def __init__(self, config: StopLossConfig = None):
        """
        Initialize stop-loss manager.
        
        Args:
            config: Stop-loss configuration
        """
        self.config = config or StopLossConfig()
        
        self.positions: Dict[str, StopLossLevel] = {}
        self.stop_events: List[StopLossEvent] = []
        
        logger.info(f"StopLossManager initialized: {self.config.atr_multiplier}x ATR, "
                   f"trailing={self.config.trailing}")
    
    def add_position(
        self,
        ticker: str,
        entry_price: float,
        entry_date: datetime,
        atr: float,
    ) -> StopLossLevel:
        """
        Add a new position with initial stop-loss.
        
        Args:
            ticker: Stock ticker
            entry_price: Entry price
            entry_date: Entry date
            atr: Current ATR value
            
        Returns:
            StopLossLevel for this position
        """
        atr_stop = entry_price - (atr * self.config.atr_multiplier)
        hard_stop = entry_price * (1 - self.config.max_loss_pct)
        
        # Use the higher (more protective) stop
        initial_stop = max(atr_stop, hard_stop)
        
        level = StopLossLevel(
            ticker=ticker,
            entry_price=entry_price,
            entry_date=entry_date,
            current_price=entry_price,
            atr_value=atr,
            atr_stop=atr_stop,
            highest_price=entry_price,
            trailing_stop=0.0,
            trailing_active=False,
            effective_stop=initial_stop,
            stop_type="atr",
        )
        
        self.positions[ticker] = level
        logger.debug(f"Added position {ticker} @ ${entry_price:.2f}, "
                    f"ATR stop @ ${initial_stop:.2f}")
        
        return level
    
    def update_position(
        self,
        ticker: str,
        current_price: float,
        current_date: datetime,
        atr: float = None,
    ) -> Tuple[StopLossLevel, bool]:
        """
        Update position and check for stop triggers.
        
        Args:
            ticker: Stock ticker
            current_price: Current price
            current_date: Current date
            atr: Updated ATR (optional)
            
        Returns:
            Tuple of (updated StopLossLevel, stop_triggered: bool)
        """
        if ticker not in self.positions:
            logger.warning(f"Position {ticker} not found")
            return None, False
        
        level = self.positions[ticker]
        level.current_price = current_price
        
        if atr is not None:
            level.atr_value = atr
        
        # Update P&L
        level.unrealized_pnl = current_price - level.entry_price
        level.unrealized_pnl_pct = level.unrealized_pnl / level.entry_price
        level.days_held = (current_date - level.entry_date).days
        
        # Update highest price
        if current_price > level.highest_price:
            level.highest_price = current_price
        
        # Activate trailing stop if profit threshold reached
        if (self.config.trailing and 
            level.unrealized_pnl_pct >= self.config.trail_activation_pct and
            not level.trailing_active):
            level.trailing_active = True
            logger.debug(f"{ticker}: Trailing stop activated at {level.unrealized_pnl_pct:.1%} profit")
        
        # Update trailing stop
        if level.trailing_active:
            new_trail_stop = level.highest_price - (level.atr_value * self.config.atr_multiplier)
            if new_trail_stop > level.trailing_stop:
                level.trailing_stop = new_trail_stop
        
        # Determine effective stop
        stops = [level.atr_stop]
        if level.trailing_active:
            stops.append(level.trailing_stop)
        
        level.effective_stop = max(stops)
        level.stop_type = "trailing" if level.trailing_active else "atr"
        
        # Check for stop trigger
        stop_triggered = current_price <= level.effective_stop
        
        if stop_triggered:
            self._record_stop_event(level, current_date)
        
        # Check for time-based stop
        if (level.days_held >= self.config.time_stop_days and 
            level.unrealized_pnl_pct <= 0):
            level.stop_type = "time"
            stop_triggered = True
            self._record_stop_event(level, current_date)
        
        return level, stop_triggered
    
    def _record_stop_event(
        self,
        level: StopLossLevel,
        trigger_date: datetime,
    ) -> None:
        """Record a stop-loss event."""
        event = StopLossEvent(
            ticker=level.ticker,
            trigger_date=trigger_date,
            trigger_type=level.stop_type,
            entry_price=level.entry_price,
            exit_price=level.current_price,
            stop_price=level.effective_stop,
            pnl_pct=level.unrealized_pnl_pct,
            days_held=level.days_held,
        )
        self.stop_events.append(event)
        logger.info(f"Stop triggered: {level.ticker} {level.stop_type} @ ${level.current_price:.2f}, "
                   f"P&L: {level.unrealized_pnl_pct:.1%}")
    
    def remove_position(self, ticker: str) -> None:
        """Remove a position."""
        if ticker in self.positions:
            del self.positions[ticker]
    
    def check_drawdown_reduction(
        self,
        portfolio_value: float,
        peak_value: float,
    ) -> bool:
        """
        Check if portfolio drawdown triggers position reduction.
        
        Returns:
            True if reduction should be triggered
        """
        if peak_value <= 0:
            return False
        
        drawdown = (peak_value - portfolio_value) / peak_value
        return drawdown >= self.config.drawdown_reduction_trigger
    
    def get_stop_statistics(self) -> Dict[str, Any]:
        """Get statistics on stop-loss performance."""
        if not self.stop_events:
            return {
                'total_stops': 0,
                'stop_rate': 0,
                'avg_loss_at_stop': 0,
                'stops_by_type': {},
            }
        
        total = len(self.stop_events)
        
        # Average loss when stopped
        losses = [e.pnl_pct for e in self.stop_events]
        avg_loss = np.mean(losses)
        
        # Stops by type
        by_type = {}
        for e in self.stop_events:
            by_type[e.trigger_type] = by_type.get(e.trigger_type, 0) + 1
        
        # Average days held before stop
        avg_days = np.mean([e.days_held for e in self.stop_events])
        
        return {
            'total_stops': total,
            'avg_loss_at_stop': avg_loss,
            'avg_days_to_stop': avg_days,
            'stops_by_type': by_type,
        }
    
    def get_all_levels(self) -> List[StopLossLevel]:
        """Get all current stop-loss levels."""
        return list(self.positions.values())


def optimize_stop_multiplier(
    data: Dict[str, pd.DataFrame],
    multipliers: List[float] = [1.5, 2.0, 2.5, 3.0],
    backtest_fn=None,
) -> Dict[str, Any]:
    """
    Test different ATR multipliers to find optimal stop-loss.
    
    Args:
        data: OHLCV data dict
        multipliers: List of ATR multipliers to test
        backtest_fn: Function to run backtest with given multiplier
        
    Returns:
        Dict with results for each multiplier
    """
    results = {}
    
    for mult in multipliers:
        config = StopLossConfig(atr_multiplier=mult)
        
        if backtest_fn:
            metrics = backtest_fn(data, config)
        else:
            # Placeholder metrics
            metrics = {
                'sharpe': 0.0,
                'max_dd': 0.0,
                'return': 0.0,
                'stop_rate': 0.0,
            }
        
        results[mult] = {
            'multiplier': mult,
            **metrics,
        }
    
    return results


class DrawdownProtector:
    """
    Drawdown-based position management.
    
    Reduces exposure when drawdown exceeds thresholds.
    """
    
    def __init__(
        self,
        trigger_levels: List[Tuple[float, float]] = None,
    ):
        """
        Initialize drawdown protector.
        
        Args:
            trigger_levels: List of (drawdown_pct, exposure_reduction_pct) tuples
                           e.g., [(0.10, 0.25), (0.15, 0.50), (0.20, 0.75)]
        """
        self.trigger_levels = trigger_levels or [
            (0.10, 0.25),  # At 10% DD, reduce exposure 25%
            (0.15, 0.50),  # At 15% DD, reduce exposure 50%
            (0.20, 0.75),  # At 20% DD, reduce exposure 75%
        ]
        
        self.peak_value = 0.0
        self.current_reduction = 0.0
        
    def update(
        self,
        portfolio_value: float,
    ) -> Tuple[float, float]:
        """
        Update and get recommended exposure reduction.
        
        Args:
            portfolio_value: Current portfolio value
            
        Returns:
            Tuple of (drawdown_pct, recommended_reduction_pct)
        """
        # Update peak
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
        
        # Calculate drawdown
        if self.peak_value <= 0:
            return 0.0, 0.0
        
        drawdown = (self.peak_value - portfolio_value) / self.peak_value
        
        # Find applicable reduction level
        reduction = 0.0
        for dd_threshold, red_pct in sorted(self.trigger_levels):
            if drawdown >= dd_threshold:
                reduction = red_pct
        
        self.current_reduction = reduction
        
        return drawdown, reduction
    
    def get_target_exposure(self, base_exposure: float = 1.0) -> float:
        """
        Get target exposure after drawdown adjustment.
        
        Args:
            base_exposure: Base exposure level (1.0 = fully invested)
            
        Returns:
            Adjusted exposure level
        """
        return base_exposure * (1 - self.current_reduction)
    
    def reset(self) -> None:
        """Reset peak value (e.g., after new high)."""
        self.current_reduction = 0.0


if __name__ == "__main__":
    print("Testing ATR Stop-Loss System")
    print("="*50)
    
    # Test ATR calculation
    np.random.seed(42)
    n = 100
    
    # Generate synthetic price data
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    
    atr = ATRCalculator.calculate_atr(high, low, close, period=14)
    print(f"ATR(14): ${atr:.2f}")
    
    # Test stop-loss manager
    config = StopLossConfig(
        atr_multiplier=2.0,
        trailing=True,
        trail_activation_pct=0.05,
    )
    
    manager = DynamicStopLossManager(config)
    
    # Add position
    entry_date = datetime(2024, 1, 1)
    level = manager.add_position(
        ticker="AAPL",
        entry_price=180.00,
        entry_date=entry_date,
        atr=3.50,
    )
    
    print(f"\nPosition added: AAPL @ ${level.entry_price:.2f}")
    print(f"  ATR: ${level.atr_value:.2f}")
    print(f"  ATR Stop: ${level.atr_stop:.2f}")
    print(f"  Effective Stop: ${level.effective_stop:.2f}")
    
    # Simulate price movement
    prices = [182, 185, 190, 188, 192, 189, 185, 180]
    
    print("\nPrice simulation:")
    for i, price in enumerate(prices):
        date = entry_date + timedelta(days=i+1)
        level, stopped = manager.update_position("AAPL", price, date)
        
        status = "STOPPED" if stopped else "Active"
        trail = f"Trail @ ${level.trailing_stop:.2f}" if level.trailing_active else "No trail"
        
        print(f"  Day {i+1}: ${price:.2f} | P&L: {level.unrealized_pnl_pct:.1%} | {trail} | {status}")
        
        if stopped:
            break
    
    # Test drawdown protector
    print("\n" + "="*50)
    print("Testing Drawdown Protector")
    print("="*50)
    
    protector = DrawdownProtector()
    
    values = [100000, 105000, 103000, 95000, 90000, 85000, 88000]
    
    for val in values:
        dd, reduction = protector.update(val)
        exposure = protector.get_target_exposure(1.0)
        print(f"  Value: ${val:,} | DD: {dd:.1%} | Reduction: {reduction:.0%} | Target Exposure: {exposure:.0%}")
    
    print("\nATR Stop-Loss System tests complete!")
