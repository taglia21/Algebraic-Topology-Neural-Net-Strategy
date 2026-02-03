"""
Production-Ready Risk Management Module
========================================

Portfolio-level risk controls with thread-safe operations and comprehensive validation.

Features:
- ATR-based stop losses with input validation
- Trailing stop management
- Take-profit tier system with percentage validation
- Correlation-based position filtering with thread safety
- Position tracking with concurrent access protection
- Risk limit enforcement

Author: Trading System
Version: 1.0.0 (Production-Hardened)
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    """Risk management configuration."""
    # Stop loss parameters
    atr_stop_multiplier: float = 2.0  # Stop distance in ATR units
    min_stop_distance_pct: float = 0.01  # 1% minimum stop
    max_stop_distance_pct: float = 0.10  # 10% maximum stop
    
    # Trailing stop
    use_trailing_stop: bool = True
    trailing_stop_trigger_r: float = 1.0  # Activate after 1R profit
    trailing_stop_distance_r: float = 0.5  # Trail by 0.5R
    
    # Take profit levels (in R multiples)
    tp_levels: List[float] = field(default_factory=lambda: [1.5, 2.0, 3.0])
    tp_exit_percentages: List[float] = field(default_factory=lambda: [0.33, 0.33, 0.34])
    
    # Portfolio limits
    max_concurrent_positions: int = 5
    max_correlated_positions: int = 3
    correlation_threshold: float = 0.7
    correlation_lookback_days: int = 60
    
    # Cache settings
    correlation_cache_ttl_hours: float = 24.0
    max_cache_size: int = 100  # LRU cache limit
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate TP percentages sum to 1.0
        total = sum(self.tp_exit_percentages)
        if not (0.99 <= total <= 1.01):  # Allow 1% tolerance for rounding
            raise ValueError(f"TP exit percentages must sum to 1.0, got {total:.3f}")
        
        if len(self.tp_levels) != len(self.tp_exit_percentages):
            raise ValueError(f"TP levels ({len(self.tp_levels)}) and exit percentages "
                           f"({len(self.tp_exit_percentages)}) must match")
        
        # Validate stop loss ranges
        if self.min_stop_distance_pct <= 0:
            raise ValueError(f"min_stop_distance_pct must be > 0, got {self.min_stop_distance_pct}")
        
        if self.max_stop_distance_pct <= self.min_stop_distance_pct:
            raise ValueError(f"max_stop_distance_pct ({self.max_stop_distance_pct}) must be "
                           f"> min_stop_distance_pct ({self.min_stop_distance_pct})")


@dataclass
class Position:
    """Represents an active trading position."""
    symbol: str
    entry_price: float
    quantity: float
    entry_time: datetime
    stop_loss: float
    take_profits: List[float]
    is_long: bool = True
    trailing_stop_active: bool = False
    current_stop: Optional[float] = None
    highest_price: Optional[float] = None  # For long positions
    lowest_price: Optional[float] = None   # For short positions
    
    @property
    def position_value(self) -> float:
        """Calculate position value."""
        return self.entry_price * self.quantity
    
    @property
    def risk_amount(self) -> float:
        """Calculate risk amount (1R)."""
        return abs(self.entry_price - self.stop_loss) * self.quantity


class RiskManager:
    """
    Production-ready risk manager with thread-safe operations.
    
    Handles stop losses, take profits, correlation filtering, and position tracking
    with comprehensive input validation and error handling.
    """
    
    def __init__(self, config: Optional[RiskConfig] = None):
        """
        Initialize risk manager.
        
        Args:
            config: Risk configuration (uses defaults if None)
        """
        self.config = config or RiskConfig()
        
        # Position tracking (thread-safe)
        self.positions: Dict[str, Position] = {}
        self.positions_lock = threading.Lock()
        
        # Correlation cache (thread-safe with LRU)
        self.correlation_cache: Dict[Tuple[str, str], Tuple[float, datetime]] = {}
        self.cache_lock = threading.Lock()
        self.cache_ttl = timedelta(hours=self.config.correlation_cache_ttl_hours)
        
        logger.info(f"RiskManager initialized: max_positions={self.config.max_concurrent_positions}, "
                   f"correlation_threshold={self.config.correlation_threshold}")
    
    def calculate_stop_loss(self, entry_price: float, atr: float, 
                           is_long: bool = True) -> float:
        """
        Calculate ATR-based stop loss with comprehensive validation.
        
        Args:
            entry_price: Entry price for the position
            atr: Average True Range value
            is_long: True for long positions, False for short
            
        Returns:
            Stop loss price
            
        Raises:
            ValueError: If inputs are invalid
        """
        # Validate inputs
        if not np.isfinite(entry_price) or entry_price <= 0:
            raise ValueError(f"Invalid entry_price: {entry_price}")
        
        if not np.isfinite(atr) or atr < 0:
            raise ValueError(f"Invalid ATR: {atr}")
        
        # Calculate ATR-based stop distance
        stop_distance = atr * self.config.atr_stop_multiplier
        
        # Apply percentage bounds
        min_distance = entry_price * self.config.min_stop_distance_pct
        max_distance = entry_price * self.config.max_stop_distance_pct
        
        stop_distance = max(min_distance, min(stop_distance, max_distance))
        
        # Calculate stop price
        if is_long:
            stop_loss = entry_price - stop_distance
            
            # Validate: stop must be below entry and above zero
            if stop_loss >= entry_price:
                logger.error(f"Invalid long stop (above entry): entry={entry_price:.2f}, "
                           f"calculated_stop={stop_loss:.2f}")
                stop_loss = entry_price * 0.95  # Fallback: 5% stop
            
            if stop_loss <= 0:
                logger.error(f"Invalid long stop (below zero): {stop_loss:.2f}")
                stop_loss = entry_price * 0.95  # Fallback: 5% stop
        else:
            stop_loss = entry_price + stop_distance
            
            # Validate: stop must be above entry
            if stop_loss <= entry_price:
                logger.error(f"Invalid short stop (below entry): entry={entry_price:.2f}, "
                           f"calculated_stop={stop_loss:.2f}")
                stop_loss = entry_price * 1.05  # Fallback: 5% stop
        
        logger.debug(f"Stop loss calculated: entry=${entry_price:.2f}, stop=${stop_loss:.2f}, "
                    f"distance={abs(entry_price - stop_loss) / entry_price * 100:.2f}%")
        
        return stop_loss
    
    def calculate_take_profits(self, entry_price: float, stop_loss: float,
                               is_long: bool = True) -> List[float]:
        """
        Calculate take-profit levels based on risk multiples.
        
        Args:
            entry_price: Entry price for the position
            stop_loss: Stop loss price
            is_long: True for long positions, False for short
            
        Returns:
            List of take-profit prices
            
        Raises:
            ValueError: If inputs are invalid
        """
        # Validate inputs
        if not np.isfinite(entry_price) or entry_price <= 0:
            raise ValueError(f"Invalid entry_price: {entry_price}")
        
        if not np.isfinite(stop_loss) or stop_loss <= 0:
            raise ValueError(f"Invalid stop_loss: {stop_loss}")
        
        # Calculate 1R (risk amount)
        risk = abs(entry_price - stop_loss)
        
        # Validate risk
        if risk == 0:
            logger.error(f"Invalid take-profit calc: entry={entry_price:.2f} equals stop={stop_loss:.2f}")
            # Use default 2% risk
            risk = entry_price * 0.02
        
        take_profits = []
        for r_multiple in self.config.tp_levels:
            if is_long:
                tp_price = entry_price + (risk * r_multiple)
            else:
                tp_price = entry_price - (risk * r_multiple)
            
            # Validate TP is on correct side of entry
            if is_long and tp_price <= entry_price:
                logger.warning(f"Invalid long TP: {tp_price:.2f} <= entry {entry_price:.2f}, "
                             f"r_multiple={r_multiple:.1f}, risk={risk:.2f}")
                continue
            elif not is_long and tp_price >= entry_price:
                logger.warning(f"Invalid short TP: {tp_price:.2f} >= entry {entry_price:.2f}, "
                             f"r_multiple={r_multiple:.1f}, risk={risk:.2f}")
                continue
            
            take_profits.append(tp_price)
        
        # Ensure we have at least one valid TP
        if not take_profits:
            logger.error("No valid take-profits generated, using default 3% target")
            take_profits = [entry_price * 1.03] if is_long else [entry_price * 0.97]
        
        logger.debug(f"Take profits: {[f'${tp:.2f}' for tp in take_profits]}")
        return take_profits
    
    def update_trailing_stop(self, position: Position, current_price: float) -> Optional[float]:
        """
        Update trailing stop based on current price.
        
        Args:
            position: Position to update
            current_price: Current market price
            
        Returns:
            New stop price if updated, None otherwise
        """
        # Validate input
        if not np.isfinite(current_price) or current_price <= 0:
            logger.error(f"Invalid current_price for trailing stop: {current_price}")
            return None
        
        if not self.config.use_trailing_stop:
            return None
        
        # Calculate profit in R
        risk_amount = abs(position.entry_price - position.stop_loss)
        
        if risk_amount == 0:
            logger.warning(f"Cannot calculate trailing stop: risk_amount is zero for {position.symbol}")
            return None
        
        if position.is_long:
            # Update highest price
            if position.highest_price is None or current_price > position.highest_price:
                position.highest_price = current_price
            
            # Calculate profit in R
            profit = position.highest_price - position.entry_price
            profit_r = profit / risk_amount
            
            # Activate trailing stop after trigger threshold
            if profit_r >= self.config.trailing_stop_trigger_r:
                if not position.trailing_stop_active:
                    position.trailing_stop_active = True
                    logger.info(f"{position.symbol}: Trailing stop ACTIVATED at {profit_r:.1f}R profit")
                
                # Calculate new trailing stop
                trailing_distance = risk_amount * self.config.trailing_stop_distance_r
                new_stop = position.highest_price - trailing_distance
                
                # Only move stop up, never down
                current_stop = position.current_stop or position.stop_loss
                if new_stop > current_stop:
                    logger.info(f"{position.symbol}: Trailing stop moved: "
                              f"${current_stop:.2f} -> ${new_stop:.2f}")
                    return new_stop
        
        else:  # Short position
            # Update lowest price
            if position.lowest_price is None or current_price < position.lowest_price:
                position.lowest_price = current_price
            
            # Calculate profit in R
            profit = position.entry_price - position.lowest_price
            profit_r = profit / risk_amount
            
            # Activate trailing stop after trigger threshold
            if profit_r >= self.config.trailing_stop_trigger_r:
                if not position.trailing_stop_active:
                    position.trailing_stop_active = True
                    logger.info(f"{position.symbol}: Trailing stop ACTIVATED at {profit_r:.1f}R profit")
                
                # Calculate new trailing stop
                trailing_distance = risk_amount * self.config.trailing_stop_distance_r
                new_stop = position.lowest_price + trailing_distance
                
                # Only move stop down, never up
                current_stop = position.current_stop or position.stop_loss
                if new_stop < current_stop:
                    logger.info(f"{position.symbol}: Trailing stop moved: "
                              f"${current_stop:.2f} -> ${new_stop:.2f}")
                    return new_stop
        
        return None
    
    def check_correlation(self, symbol: str, existing_symbols: List[str],
                         price_data: Dict[str, pd.Series]) -> Tuple[bool, float]:
        """
        Check if new position correlates too highly with existing positions.
        Thread-safe implementation with LRU cache management.
        
        Args:
            symbol: Symbol to check
            existing_symbols: List of symbols in current portfolio
            price_data: Dictionary of price series for correlation calculation
            
        Returns:
            Tuple of (is_acceptable, max_correlation)
        """
        if not existing_symbols:
            return True, 0.0
        
        max_corr = 0.0
        
        for existing_symbol in existing_symbols:
            cache_key = tuple(sorted([symbol, existing_symbol]))
            
            # Thread-safe cache access
            with self.cache_lock:
                if cache_key in self.correlation_cache:
                    corr, timestamp = self.correlation_cache[cache_key]
                    if datetime.now() - timestamp < self.cache_ttl:
                        max_corr = max(max_corr, abs(corr))
                        continue
            
            # Calculate correlation (outside lock to minimize contention)
            corr = self._calculate_correlation(symbol, existing_symbol, price_data)
            
            if corr is None:
                continue
            
            # Thread-safe cache write with LRU management
            with self.cache_lock:
                self.correlation_cache[cache_key] = (corr, datetime.now())
                
                # LRU eviction if cache too large
                if len(self.correlation_cache) > self.config.max_cache_size:
                    # Remove oldest entry
                    oldest_key = min(self.correlation_cache.items(), 
                                   key=lambda x: x[1][1])[0]
                    del self.correlation_cache[oldest_key]
                    logger.debug(f"Cache evicted oldest entry: {oldest_key}")
            
            max_corr = max(max_corr, abs(corr))
        
        is_acceptable = max_corr < self.config.correlation_threshold
        
        if not is_acceptable:
            logger.warning(f"{symbol}: Correlation too high ({max_corr:.2f} >= "
                         f"{self.config.correlation_threshold})")
        
        return is_acceptable, max_corr
    
    def _calculate_correlation(self, symbol1: str, symbol2: str,
                              price_data: Dict[str, pd.Series]) -> Optional[float]:
        """
        Calculate correlation between two symbols with validation.
        
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            price_data: Dictionary of price series
            
        Returns:
            Correlation coefficient or None if insufficient data
        """
        if symbol1 not in price_data or symbol2 not in price_data:
            logger.warning(f"Missing price data for correlation: {symbol1}, {symbol2}")
            return None
        
        # Calculate returns
        returns1 = price_data[symbol1].pct_change()
        returns2 = price_data[symbol2].pct_change()
        
        # Align on common dates
        aligned = pd.DataFrame({'s1': returns1, 's2': returns2}).dropna()
        
        if len(aligned) < 30:  # Need at least 30 overlapping days
            logger.warning(f"Insufficient overlapping data for correlation: {len(aligned)} days")
            return None
        
        # Calculate correlation
        corr = aligned['s1'].corr(aligned['s2'])
        
        # Validate result
        if not np.isfinite(corr):
            logger.error(f"Invalid correlation calculated: {corr}")
            return None
        
        logger.debug(f"Correlation {symbol1} vs {symbol2}: {corr:.3f} ({len(aligned)} days)")
        return corr
    
    def add_position(self, symbol: str, entry_price: float, quantity: float,
                    stop_loss: float, take_profits: List[float],
                    is_long: bool = True) -> Position:
        """
        Add a new position to tracking (thread-safe).
        
        Args:
            symbol: Symbol being traded
            entry_price: Entry price
            quantity: Position size
            stop_loss: Stop loss price
            take_profits: List of take-profit prices
            is_long: True for long, False for short
            
        Returns:
            Position object
        """
        position = Position(
            symbol=symbol,
            entry_price=entry_price,
            quantity=quantity,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            take_profits=take_profits,
            is_long=is_long,
            current_stop=stop_loss,
            highest_price=entry_price if is_long else None,
            lowest_price=entry_price if not is_long else None
        )
        
        with self.positions_lock:
            self.positions[symbol] = position
            logger.info(f"Position added: {symbol} @ ${entry_price:.2f} x {quantity}")
        
        return position
    
    def remove_position(self, symbol: str, reason: str = "manual"):
        """
        Remove a position from tracking (thread-safe).
        
        Args:
            symbol: Symbol to remove
            reason: Reason for removal (for logging)
        """
        with self.positions_lock:
            if symbol in self.positions:
                del self.positions[symbol]
                logger.info(f"Position removed: {symbol} ({reason})")
            else:
                logger.warning(f"Attempted to remove non-existent position: {symbol}")
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position by symbol (thread-safe).
        
        Args:
            symbol: Symbol to retrieve
            
        Returns:
            Position or None if not found
        """
        with self.positions_lock:
            return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, Position]:
        """
        Get all positions (returns a copy for thread safety).
        
        Returns:
            Dictionary of all positions
        """
        with self.positions_lock:
            return self.positions.copy()
    
    def can_add_position(self) -> Tuple[bool, str]:
        """
        Check if a new position can be added based on limits.
        
        Returns:
            Tuple of (can_add, reason)
        """
        with self.positions_lock:
            if len(self.positions) >= self.config.max_concurrent_positions:
                return False, f"At max positions ({self.config.max_concurrent_positions})"
        
        return True, "OK"
    
    def clear_cache(self):
        """Clear correlation cache (thread-safe)."""
        with self.cache_lock:
            self.correlation_cache.clear()
            logger.info("Correlation cache cleared")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize risk manager
    risk_mgr = RiskManager()
    
    # Calculate stop loss
    entry = 100.0
    atr = 2.5
    stop = risk_mgr.calculate_stop_loss(entry, atr, is_long=True)
    print(f"Entry: ${entry:.2f}, ATR: ${atr:.2f}, Stop: ${stop:.2f}")
    
    # Calculate take profits
    tps = risk_mgr.calculate_take_profits(entry, stop, is_long=True)
    print(f"Take profits: {[f'${tp:.2f}' for tp in tps]}")
