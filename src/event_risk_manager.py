"""
Event-Driven Risk Manager
==========================

Protects against event risk and liquidity sweeps through multiplicative position sizing.
Does NOT block signals - instead adjusts position size based on risk factors.

Components:
- Economic calendar (FOMC, earnings blackouts)
- Volume anomaly detection
- Spread monitoring and liquidity sweep detection
- Time-of-day filters
- Circuit breakers for portfolio protection

Author: Trading System
Version: 1.0.0
"""

import logging
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Tuple, Deque
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Economic event types."""
    FOMC = "FOMC"
    CPI = "CPI"
    NFP = "NFP"
    EARNINGS = "Earnings"


@dataclass
class EventRiskConfig:
    """Configuration for event risk manager."""
    
    # Economic calendar settings
    fomc_blackout_days_before: int = 2
    fomc_blackout_days_after: int = 1
    high_impact_buffer_minutes: int = 30
    earnings_blackout_days: int = 2
    
    # Event multipliers (reduce position size during events)
    fomc_multiplier: float = 0.3  # 70% reduction
    high_impact_multiplier: float = 0.5  # 50% reduction
    earnings_multiplier: float = 0.7  # 30% reduction
    
    # Volume anomaly settings
    volume_lookback_days: int = 5
    volume_spike_threshold: float = 4.0  # 4x average
    volume_spike_multiplier: float = 0.6  # 40% reduction on spike
    
    # Spread monitoring
    spread_lookback_days: int = 20
    spread_warning_threshold: float = 2.0  # 2x baseline
    spread_critical_threshold: float = 4.0  # 4x baseline
    spread_warning_multiplier: float = 0.8  # 20% reduction
    spread_critical_multiplier: float = 0.4  # 60% reduction
    
    # Liquidity sweep detection
    liquidity_sweep_reversal_threshold: float = 0.015  # 1.5% reversal
    liquidity_sweep_multiplier: float = 0.5  # 50% reduction
    
    # Time-of-day filters
    market_open_buffer_minutes: int = 15
    market_close_buffer_minutes: int = 5
    lunch_start: time = field(default_factory=lambda: time(11, 30))
    lunch_end: time = field(default_factory=lambda: time(14, 0))
    
    opening_multiplier: float = 0.3  # 70% reduction (volatile)
    closing_multiplier: float = 0.3  # 70% reduction (volatile)
    lunch_multiplier: float = 0.7  # 30% reduction (low liquidity)
    prime_hours_multiplier: float = 1.0  # No reduction
    
    # Circuit breaker settings
    daily_loss_limit_pct: float = 0.02  # 2% daily loss -> halt
    consecutive_loss_limit: int = 3  # 3 losses -> pause
    pause_duration_minutes: int = 60
    weekly_drawdown_threshold_pct: float = 0.05  # 5% weekly drawdown
    weekly_drawdown_multiplier: float = 0.5  # 50% reduction
    
    # Circuit breaker multipliers
    circuit_breaker_halt_multiplier: float = 0.0  # Full halt
    circuit_breaker_pause_multiplier: float = 0.0  # Full pause


class EconomicCalendar:
    """
    Manages economic event calendar and blackout periods.
    Hardcoded 2026 FOMC dates (no external API calls).
    """
    
    def __init__(self, config: EventRiskConfig):
        self.config = config
        self.lock = threading.Lock()
        
        # 2026 FOMC meeting dates (8 scheduled meetings)
        self.fomc_dates_2026 = [
            datetime(2026, 1, 28),  # Jan 27-28
            datetime(2026, 3, 18),  # Mar 17-18
            datetime(2026, 5, 6),   # May 5-6
            datetime(2026, 6, 17),  # Jun 16-17
            datetime(2026, 7, 29),  # Jul 28-29
            datetime(2026, 9, 16),  # Sep 15-16
            datetime(2026, 11, 4),  # Nov 3-4
            datetime(2026, 12, 16), # Dec 15-16
        ]
        
        # Track earnings dates per symbol (to be populated dynamically)
        self.earnings_dates: Dict[str, datetime] = {}
        
        logger.info(f"EconomicCalendar initialized with {len(self.fomc_dates_2026)} FOMC dates")
    
    def set_earnings_date(self, symbol: str, earnings_date: datetime):
        """Set next earnings date for a symbol."""
        with self.lock:
            self.earnings_dates[symbol] = earnings_date
            logger.info(f"Earnings date set for {symbol}: {earnings_date.date()}")
    
    def is_fomc_blackout(self, check_date: datetime) -> bool:
        """Check if date falls within FOMC blackout period."""
        for fomc_date in self.fomc_dates_2026:
            days_until = (fomc_date - check_date).days
            
            # Before FOMC
            if 0 <= days_until <= self.config.fomc_blackout_days_before:
                return True
            
            # After FOMC
            if -self.config.fomc_blackout_days_after <= days_until < 0:
                return True
        
        return False
    
    def is_high_impact_event(self, check_datetime: datetime) -> bool:
        """Check if within buffer of high-impact event (FOMC decision time)."""
        buffer = timedelta(minutes=self.config.high_impact_buffer_minutes)
        
        for fomc_date in self.fomc_dates_2026:
            # FOMC typically announces at 2:00 PM ET
            event_time = datetime.combine(fomc_date.date(), time(14, 0))
            
            if abs(check_datetime - event_time) <= buffer:
                return True
        
        return False
    
    def is_earnings_blackout(self, symbol: str, check_date: datetime) -> bool:
        """Check if date falls within earnings blackout for symbol."""
        with self.lock:
            if symbol not in self.earnings_dates:
                return False
            
            earnings_date = self.earnings_dates[symbol]
            days_until = (earnings_date - check_date).days
            
            # Blackout period before earnings
            return 0 <= days_until <= self.config.earnings_blackout_days
    
    def get_event_multiplier(self, symbol: str, check_datetime: datetime) -> Tuple[float, str]:
        """
        Get position size multiplier based on economic events.
        
        Returns:
            (multiplier, reason)
        """
        # Check high-impact event first (most restrictive)
        if self.is_high_impact_event(check_datetime):
            return self.config.high_impact_multiplier, "High-impact event buffer"
        
        # Check FOMC blackout
        if self.is_fomc_blackout(check_datetime):
            return self.config.fomc_multiplier, "FOMC blackout period"
        
        # Check earnings blackout
        if self.is_earnings_blackout(symbol, check_datetime):
            return self.config.earnings_multiplier, f"{symbol} earnings blackout"
        
        return 1.0, "No event risk"


class VolumeAnomalyDetector:
    """
    Detects volume anomalies that may indicate liquidity issues.
    Tracks rolling average volume and flags spikes.
    """
    
    def __init__(self, config: EventRiskConfig):
        self.config = config
        self.lock = threading.Lock()
        
        # Store recent volume data per symbol
        self.volume_history: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=self.config.volume_lookback_days)
        )
        
        logger.info(f"VolumeAnomalyDetector initialized ({config.volume_lookback_days}d lookback)")
    
    def update_volume(self, symbol: str, volume: float):
        """Update volume history for symbol."""
        with self.lock:
            self.volume_history[symbol].append(volume)
    
    def get_average_volume(self, symbol: str) -> Optional[float]:
        """Calculate average volume for symbol."""
        with self.lock:
            if symbol not in self.volume_history or len(self.volume_history[symbol]) < 2:
                return None
            
            return np.mean(list(self.volume_history[symbol]))
    
    def is_volume_spike(self, symbol: str, current_volume: float) -> bool:
        """Check if current volume is anomalous spike."""
        avg_volume = self.get_average_volume(symbol)
        
        if avg_volume is None or avg_volume == 0:
            return False
        
        spike_ratio = current_volume / avg_volume
        return spike_ratio >= self.config.volume_spike_threshold
    
    def get_volume_multiplier(self, symbol: str, current_volume: float) -> Tuple[float, str]:
        """
        Get position size multiplier based on volume anomalies.
        
        Returns:
            (multiplier, reason)
        """
        # Check for spike BEFORE updating history (important!)
        if self.is_volume_spike(symbol, current_volume):
            avg = self.get_average_volume(symbol)
            ratio = current_volume / avg if avg else 0
            
            # Update history after checking
            self.update_volume(symbol, current_volume)
            
            return (
                self.config.volume_spike_multiplier,
                f"Volume spike detected ({ratio:.1f}x average)"
            )
        
        # Update history for non-spikes
        self.update_volume(symbol, current_volume)
        
        return 1.0, "Normal volume"


class SpreadMonitor:
    """
    Monitors bid-ask spreads and detects liquidity sweeps.
    Liquidity sweep: price exceeds recent swing high/low then reverses sharply.
    """
    
    def __init__(self, config: EventRiskConfig):
        self.config = config
        self.lock = threading.Lock()
        
        # Store spread history per symbol
        self.spread_history: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=self.config.spread_lookback_days)
        )
        
        # Store recent price swings for liquidity sweep detection
        self.price_swings: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {'high': 0.0, 'low': float('inf')}
        )
        
        logger.info(f"SpreadMonitor initialized ({config.spread_lookback_days}d lookback)")
    
    def update_spread(self, symbol: str, spread: float):
        """Update spread history for symbol."""
        with self.lock:
            self.spread_history[symbol].append(spread)
    
    def update_price_swing(self, symbol: str, high: float, low: float):
        """Update price swing range for liquidity sweep detection."""
        with self.lock:
            self.price_swings[symbol]['high'] = max(self.price_swings[symbol]['high'], high)
            self.price_swings[symbol]['low'] = min(self.price_swings[symbol]['low'], low)
    
    def get_average_spread(self, symbol: str) -> Optional[float]:
        """Calculate average spread for symbol."""
        with self.lock:
            if symbol not in self.spread_history or len(self.spread_history[symbol]) < 2:
                return None
            
            return np.mean(list(self.spread_history[symbol]))
    
    def detect_liquidity_sweep(self, symbol: str, current_price: float, 
                              price_5min_ago: Optional[float]) -> bool:
        """
        Detect liquidity sweep pattern.
        Price exceeds swing high/low then reverses sharply.
        """
        if price_5min_ago is None:
            return False
        
        with self.lock:
            swings = self.price_swings.get(symbol)
            if not swings:
                return False
            
            swing_high = swings['high']
            swing_low = swings['low']
            
            # Check if price exceeded swing and reversed
            exceeded_high = price_5min_ago > swing_high and current_price < swing_high
            exceeded_low = price_5min_ago < swing_low and current_price > swing_low
            
            if exceeded_high or exceeded_low:
                # Check reversal magnitude
                reversal = abs(current_price - price_5min_ago) / price_5min_ago
                if reversal >= self.config.liquidity_sweep_reversal_threshold:
                    logger.warning(f"Liquidity sweep detected for {symbol}: "
                                 f"reversal {reversal:.1%}")
                    return True
        
        return False
    
    def get_liquidity_multiplier(self, symbol: str, current_spread: float,
                                 current_price: float = 0.0,
                                 price_5min_ago: Optional[float] = None) -> Tuple[float, str]:
        """
        Get position size multiplier based on spread and liquidity.
        
        Returns:
            (multiplier, reason)
        """
        # Update history
        self.update_spread(symbol, current_spread)
        
        # Check for liquidity sweep
        if self.detect_liquidity_sweep(symbol, current_price, price_5min_ago):
            return (
                self.config.liquidity_sweep_multiplier,
                "Liquidity sweep detected"
            )
        
        # Check spread expansion (check critical BEFORE warning)
        avg_spread = self.get_average_spread(symbol)
        if avg_spread is None or avg_spread == 0:
            return 1.0, "No spread history"
        
        spread_ratio = current_spread / avg_spread
        
        # Check critical threshold first (more severe)
        if spread_ratio >= self.config.spread_critical_threshold:
            return (
                self.config.spread_critical_multiplier,
                f"Critical spread expansion ({spread_ratio:.1f}x)"
            )
        
        # Then check warning threshold
        elif spread_ratio >= self.config.spread_warning_threshold:
            return (
                self.config.spread_warning_multiplier,
                f"Spread warning ({spread_ratio:.1f}x)"
            )
        
        return 1.0, "Normal spread"


class TimeOfDayFilter:
    """
    Applies time-of-day filters to avoid volatile/illiquid periods.
    """
    
    def __init__(self, config: EventRiskConfig):
        self.config = config
        
        # Market hours (ET)
        self.market_open = time(9, 30)
        self.market_close = time(16, 0)
        
        logger.info("TimeOfDayFilter initialized")
    
    def get_time_multiplier(self, check_datetime: datetime) -> Tuple[float, str]:
        """
        Get position size multiplier based on time of day.
        
        Returns:
            (multiplier, reason)
        """
        current_time = check_datetime.time()
        
        # Opening volatility (first 15 minutes)
        opening_cutoff = (
            datetime.combine(datetime.today(), self.market_open) +
            timedelta(minutes=self.config.market_open_buffer_minutes)
        ).time()
        
        if self.market_open <= current_time < opening_cutoff:
            return self.config.opening_multiplier, "Opening volatility period"
        
        # Closing volatility (last 5 minutes)
        closing_cutoff = (
            datetime.combine(datetime.today(), self.market_close) -
            timedelta(minutes=self.config.market_close_buffer_minutes)
        ).time()
        
        if closing_cutoff <= current_time <= self.market_close:
            return self.config.closing_multiplier, "Closing volatility period"
        
        # Lunch hour (low liquidity)
        if self.config.lunch_start <= current_time < self.config.lunch_end:
            return self.config.lunch_multiplier, "Lunch hour"
        
        # Prime trading hours
        return self.config.prime_hours_multiplier, "Prime trading hours"


class CircuitBreaker:
    """
    Portfolio-level circuit breakers to prevent catastrophic losses.
    """
    
    def __init__(self, config: EventRiskConfig):
        self.config = config
        self.lock = threading.Lock()
        
        # Track daily/weekly performance
        self.daily_pnl: float = 0.0
        self.weekly_pnl: float = 0.0
        self.starting_equity_daily: float = 0.0
        self.starting_equity_weekly: float = 0.0
        
        # Track consecutive losses
        self.consecutive_losses: int = 0
        self.last_trade_loss: bool = False
        
        # Pause state
        self.is_paused: bool = False
        self.pause_until: Optional[datetime] = None
        
        # Halt state
        self.is_halted: bool = False
        
        logger.info("CircuitBreaker initialized")
    
    def set_starting_equity(self, equity: float, period: str = 'daily'):
        """Set starting equity for tracking."""
        with self.lock:
            if period == 'daily':
                self.starting_equity_daily = equity
                self.daily_pnl = 0.0
            elif period == 'weekly':
                self.starting_equity_weekly = equity
                self.weekly_pnl = 0.0
    
    def update_pnl(self, trade_pnl: float):
        """Update P&L and track consecutive losses."""
        with self.lock:
            self.daily_pnl += trade_pnl
            self.weekly_pnl += trade_pnl
            
            # Track consecutive losses
            if trade_pnl < 0:
                if self.last_trade_loss:
                    self.consecutive_losses += 1
                else:
                    self.consecutive_losses = 1
                self.last_trade_loss = True
            else:
                self.consecutive_losses = 0
                self.last_trade_loss = False
            
            # Check for pause condition
            if self.consecutive_losses >= self.config.consecutive_loss_limit:
                self.is_paused = True
                self.pause_until = datetime.now() + timedelta(
                    minutes=self.config.pause_duration_minutes
                )
                logger.error(
                    f"🚨 CIRCUIT BREAKER PAUSE: {self.consecutive_losses} consecutive losses. "
                    f"Paused until {self.pause_until}"
                )
    
    def check_daily_limit(self) -> bool:
        """Check if daily loss limit exceeded."""
        with self.lock:
            if self.starting_equity_daily == 0:
                return False
            
            loss_pct = self.daily_pnl / self.starting_equity_daily
            
            if loss_pct <= -self.config.daily_loss_limit_pct:
                self.is_halted = True
                logger.error(
                    f"🚨 CIRCUIT BREAKER HALT: Daily loss {loss_pct:.1%} exceeds limit "
                    f"{self.config.daily_loss_limit_pct:.1%}. TRADING HALTED."
                )
                return True
            
            return False
    
    def check_weekly_drawdown(self) -> bool:
        """Check if weekly drawdown threshold exceeded."""
        with self.lock:
            if self.starting_equity_weekly == 0:
                return False
            
            drawdown_pct = self.weekly_pnl / self.starting_equity_weekly
            
            return drawdown_pct <= -self.config.weekly_drawdown_threshold_pct
    
    def clear_pause(self):
        """Clear pause if duration elapsed."""
        with self.lock:
            if self.is_paused and self.pause_until:
                if datetime.now() >= self.pause_until:
                    self.is_paused = False
                    self.pause_until = None
                    self.consecutive_losses = 0
                    logger.info("Circuit breaker pause cleared - resuming trading")
    
    def get_circuit_multiplier(self, current_equity: float) -> Tuple[float, str]:
        """
        Get position size multiplier based on circuit breaker state.
        
        Returns:
            (multiplier, reason)
        """
        # Initialize equity if not set
        with self.lock:
            if self.starting_equity_daily == 0:
                self.set_starting_equity(current_equity, 'daily')
            if self.starting_equity_weekly == 0:
                self.set_starting_equity(current_equity, 'weekly')
        
        # Clear pause if expired
        self.clear_pause()
        
        # Check halt condition
        if self.is_halted or self.check_daily_limit():
            return self.config.circuit_breaker_halt_multiplier, "🚨 CIRCUIT BREAKER HALT"
        
        # Check pause condition
        if self.is_paused:
            return (
                self.config.circuit_breaker_pause_multiplier,
                f"Circuit breaker pause ({self.consecutive_losses} consecutive losses)"
            )
        
        # Check weekly drawdown
        if self.check_weekly_drawdown():
            drawdown = self.weekly_pnl / self.starting_equity_weekly
            return (
                self.config.weekly_drawdown_multiplier,
                f"Weekly drawdown {drawdown:.1%} - size reduced"
            )
        
        return 1.0, "No circuit breaker active"


class EventRiskManager:
    """
    Main orchestrator combining all event risk components.
    Returns multiplicative position size adjustment.
    """
    
    def __init__(self, config: Optional[EventRiskConfig] = None):
        """
        Initialize event risk manager.
        
        Args:
            config: Configuration (uses defaults if None)
        """
        self.config = config or EventRiskConfig()
        
        # Initialize components
        self.calendar = EconomicCalendar(self.config)
        self.volume_detector = VolumeAnomalyDetector(self.config)
        self.spread_monitor = SpreadMonitor(self.config)
        self.time_filter = TimeOfDayFilter(self.config)
        self.circuit_breaker = CircuitBreaker(self.config)
        
        self.lock = threading.Lock()
        
        logger.info("EventRiskManager initialized with all components")
    
    def calculate_position_multiplier(
        self,
        symbol: str,
        check_datetime: datetime,
        market_data: Optional[Dict] = None
    ) -> Tuple[float, List[str]]:
        """
        Calculate combined position size multiplier from all risk factors.
        
        Args:
            symbol: Stock symbol
            check_datetime: Time to check
            market_data: Optional dict with:
                - 'volume': current volume
                - 'spread': current bid-ask spread
                - 'price': current price
                - 'price_5min_ago': price 5 minutes ago
                - 'equity': current portfolio equity
        
        Returns:
            (combined_multiplier, reasons)
        """
        market_data = market_data or {}
        
        multipliers = []
        reasons = []
        
        # 1. Economic calendar
        mult, reason = self.calendar.get_event_multiplier(symbol, check_datetime)
        if mult < 1.0:
            multipliers.append(mult)
            reasons.append(reason)
        
        # 2. Volume anomaly
        if 'volume' in market_data:
            mult, reason = self.volume_detector.get_volume_multiplier(
                symbol, market_data['volume']
            )
            if mult < 1.0:
                multipliers.append(mult)
                reasons.append(reason)
        
        # 3. Spread/liquidity
        if 'spread' in market_data:
            mult, reason = self.spread_monitor.get_liquidity_multiplier(
                symbol,
                market_data['spread'],
                market_data.get('price', 0.0),
                market_data.get('price_5min_ago')
            )
            if mult < 1.0:
                multipliers.append(mult)
                reasons.append(reason)
        
        # 4. Time of day
        mult, reason = self.time_filter.get_time_multiplier(check_datetime)
        if mult < 1.0:
            multipliers.append(mult)
            reasons.append(reason)
        
        # 5. Circuit breaker
        if 'equity' in market_data:
            mult, reason = self.circuit_breaker.get_circuit_multiplier(
                market_data['equity']
            )
            if mult < 1.0:
                multipliers.append(mult)
                reasons.append(reason)
        
        # Calculate combined multiplier (product of all factors)
        combined_multiplier = np.prod(multipliers) if multipliers else 1.0
        
        # Log if position size is being reduced
        if combined_multiplier < 1.0:
            logger.info(
                f"Position multiplier for {symbol}: {combined_multiplier:.2f} "
                f"(Reasons: {', '.join(reasons)})"
            )
        
        return combined_multiplier, reasons
    
    def reset_daily(self, current_equity: float):
        """Reset daily tracking (call at market open)."""
        self.circuit_breaker.set_starting_equity(current_equity, 'daily')
        logger.info(f"Daily reset complete. Starting equity: ${current_equity:,.2f}")
    
    def reset_weekly(self, current_equity: float):
        """Reset weekly tracking (call on Monday)."""
        self.circuit_breaker.set_starting_equity(current_equity, 'weekly')
        logger.info(f"Weekly reset complete. Starting equity: ${current_equity:,.2f}")
    
    def update_trade(self, trade_pnl: float):
        """Update with trade result for circuit breaker tracking."""
        self.circuit_breaker.update_pnl(trade_pnl)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("="*70)
    print("EVENT RISK MANAGER - DEMONSTRATION")
    print("="*70)
    
    # Initialize
    config = EventRiskConfig()
    erm = EventRiskManager(config)
    
    # Test 1: Normal market conditions
    print("\n📊 Test 1: Normal market conditions")
    print("-"*70)
    now = datetime(2026, 3, 1, 10, 30)  # Sunday before FOMC
    market_data = {
        'volume': 1_000_000,
        'spread': 0.02,
        'price': 150.0,
        'equity': 100_000
    }
    
    mult, reasons = erm.calculate_position_multiplier('AAPL', now, market_data)
    print(f"Time: {now}")
    print(f"Position multiplier: {mult:.2f}")
    print(f"Reasons: {reasons if reasons else 'None - full size allowed'}")
    
    # Test 2: FOMC blackout
    print("\n📊 Test 2: FOMC blackout period")
    print("-"*70)
    fomc_date = datetime(2026, 3, 17, 10, 30)  # Day before FOMC
    mult, reasons = erm.calculate_position_multiplier('AAPL', fomc_date, market_data)
    print(f"Time: {fomc_date}")
    print(f"Position multiplier: {mult:.2f}")
    print(f"Reasons: {reasons}")
    
    # Test 3: Opening volatility
    print("\n📊 Test 3: Opening volatility")
    print("-"*70)
    opening = datetime(2026, 3, 1, 9, 35)  # 5 min after open
    mult, reasons = erm.calculate_position_multiplier('AAPL', opening, market_data)
    print(f"Time: {opening}")
    print(f"Position multiplier: {mult:.2f}")
    print(f"Reasons: {reasons}")
    
    # Test 4: Volume spike
    print("\n📊 Test 4: Volume spike")
    print("-"*70)
    # Build normal volume history
    for _ in range(5):
        erm.volume_detector.update_volume('TSLA', 1_000_000)
    
    spike_data = market_data.copy()
    spike_data['volume'] = 5_000_000  # 5x normal
    mult, reasons = erm.calculate_position_multiplier('TSLA', now, spike_data)
    print(f"Normal volume: 1M, Current: 5M")
    print(f"Position multiplier: {mult:.2f}")
    print(f"Reasons: {reasons}")
    
    # Test 5: Multiple factors
    print("\n📊 Test 5: Multiple risk factors (FOMC + opening + volume spike)")
    print("-"*70)
    mult, reasons = erm.calculate_position_multiplier('TSLA', fomc_date.replace(hour=9, minute=35), spike_data)
    print(f"Position multiplier: {mult:.2f}")
    print(f"Reasons: {reasons}")
    print(f"Calculation: {config.fomc_multiplier} × {config.opening_multiplier} × "
          f"{config.volume_spike_multiplier} = {mult:.2f}")
    
    print("\n" + "="*70)
    print("✅ Event Risk Manager operational!")
    print("="*70)
