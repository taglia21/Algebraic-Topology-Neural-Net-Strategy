"""
Delay Adapter
==============

Compensate for 15-minute delayed market data from Tradier.

Strategies:
1. Price movement estimation using ATR
2. Conservative entry buffers (1.5σ credit, 2.0σ debit)
3. Safe trading windows (avoid open/close volatility)
4. VIX-based entry restrictions
5. Greeks adjustment for stale data
"""

import logging
import numpy as np
from dataclasses import dataclass
from datetime import datetime, time
from typing import Dict, Optional, Tuple
from enum import Enum

from .utils.constants import (
    PRICE_DELAY_MINUTES,
    CREDIT_ENTRY_BUFFER_SIGMA,
    DEBIT_ENTRY_BUFFER_SIGMA,
    SAFE_TRADING_WINDOWS,
    MAX_VIX_FOR_ENTRY,
    MIN_DTE_FOR_DELAYED_DATA,
)

logger = logging.getLogger(__name__)


class MarketPeriod(Enum):
    """Current market period."""
    PRE_MARKET = "pre_market"
    OPEN_VOLATILITY = "open_volatility"
    SAFE_TRADING = "safe_trading"
    CLOSE_VOLATILITY = "close_volatility"
    AFTER_HOURS = "after_hours"
    CLOSED = "closed"


@dataclass
class DelayedPrice:
    """Price data with delay compensation."""
    quoted_price: float  # Actual quote from API
    estimated_current: float  # Estimated current price
    buffer_amount: float  # Safety buffer applied
    confidence: float  # Confidence in estimate (0-1)
    delay_minutes: int
    timestamp: datetime


@dataclass
class EntryAdjustment:
    """Recommended entry price adjustment for delayed data."""
    original_price: float
    adjusted_price: float
    adjustment_pct: float
    reason: str
    is_conservative: bool  # True if adjustment makes entry harder


class DelayAdapter:
    """
    Delay Compensation Adapter.
    
    Handles 15-minute delayed quotes by:
    - Estimating price movement using ATR
    - Applying conservative buffers to entry prices
    - Blocking entries during high-risk periods
    - Adjusting Greeks for stale data
    
    Usage:
        adapter = DelayAdapter(delay_minutes=15)
        adjusted = adapter.adjust_entry_price(
            quoted_price=4.50,
            is_credit=True,
            atr=0.25,
            underlying_price=450
        )
    """
    
    def __init__(
        self,
        delay_minutes: int = PRICE_DELAY_MINUTES,
        credit_buffer_sigma: float = CREDIT_ENTRY_BUFFER_SIGMA,
        debit_buffer_sigma: float = DEBIT_ENTRY_BUFFER_SIGMA
    ):
        """
        Initialize delay adapter.
        
        Args:
            delay_minutes: Quote delay in minutes
            credit_buffer_sigma: Buffer for credit entries (std devs)
            debit_buffer_sigma: Buffer for debit entries (std devs)
        """
        self.delay_minutes = delay_minutes
        self.credit_buffer_sigma = credit_buffer_sigma
        self.debit_buffer_sigma = debit_buffer_sigma
        
        # Price movement cache (symbol -> ATR)
        self.atr_cache: Dict[str, float] = {}
        
        logger.info(
            f"Delay Adapter initialized: {delay_minutes}min delay, "
            f"buffers: {credit_buffer_sigma:.1f}σ credit, {debit_buffer_sigma:.1f}σ debit"
        )
    
    def estimate_price_movement(
        self,
        quoted_price: float,
        atr: float,
        underlying_price: float,
        delay_minutes: Optional[int] = None
    ) -> DelayedPrice:
        """
        Estimate current price given delayed quote.
        
        Uses ATR to estimate typical price movement over delay period.
        
        Args:
            quoted_price: Price from delayed quote
            atr: Average True Range of underlying
            underlying_price: Current underlying price
            delay_minutes: Quote delay (defaults to self.delay_minutes)
            
        Returns:
            DelayedPrice with estimate
        """
        delay = delay_minutes or self.delay_minutes
        
        # Estimate intraday movement as fraction of daily ATR
        # Assume ATR is daily, scale to minutes: sqrt(delay_min / 390_min)
        intraday_factor = np.sqrt(delay / 390)
        expected_move = atr * intraday_factor
        
        # Convert to option price movement (rough approximation)
        # Options move ~10-50% of underlying move depending on delta
        # Use conservative 30% for estimation
        option_move_estimate = expected_move * 0.30 * (quoted_price / underlying_price)
        
        # Confidence decreases with delay and volatility
        confidence = max(0.3, 1.0 - (delay / 60) - (atr / underlying_price))
        
        return DelayedPrice(
            quoted_price=quoted_price,
            estimated_current=quoted_price,  # Central estimate = quote
            buffer_amount=option_move_estimate,
            confidence=confidence,
            delay_minutes=delay,
            timestamp=datetime.now()
        )
    
    def adjust_entry_price(
        self,
        quoted_price: float,
        is_credit: bool,
        atr: float,
        underlying_price: float,
        symbol: Optional[str] = None
    ) -> EntryAdjustment:
        """
        Calculate conservative entry price accounting for delay.
        
        For credit spreads (selling): Require LOWER price to enter
        For debit spreads (buying): Accept HIGHER price to enter
        
        Args:
            quoted_price: Option price from delayed quote
            is_credit: True if selling (credit), False if buying (debit)
            atr: Average True Range of underlying
            underlying_price: Current underlying price
            symbol: Symbol (for caching ATR)
            
        Returns:
            EntryAdjustment with recommended price
        """
        # Cache ATR if symbol provided
        if symbol:
            self.atr_cache[symbol] = atr
        
        # Estimate price movement
        delayed = self.estimate_price_movement(quoted_price, atr, underlying_price)
        
        # Apply buffer based on strategy type
        if is_credit:
            # Selling premium: require price to be LOWER (worse fill for us)
            # This protects against price rising during delay
            buffer_sigma = self.credit_buffer_sigma
            adjusted_price = quoted_price - (delayed.buffer_amount * buffer_sigma)
            reason = f"Credit: subtract {buffer_sigma:.1f}σ buffer (price may have risen)"
            is_conservative = True
        else:
            # Buying premium: accept price to be HIGHER (worse fill for us)
            # This protects against price falling during delay
            buffer_sigma = self.debit_buffer_sigma
            adjusted_price = quoted_price + (delayed.buffer_amount * buffer_sigma)
            reason = f"Debit: add {buffer_sigma:.1f}σ buffer (price may have fallen)"
            is_conservative = True
        
        # Ensure price is positive
        adjusted_price = max(0.01, adjusted_price)
        
        adjustment_pct = ((adjusted_price - quoted_price) / quoted_price) * 100
        
        logger.debug(
            f"Entry adjustment: ${quoted_price:.2f} → ${adjusted_price:.2f} "
            f"({adjustment_pct:+.1f}%) - {reason}"
        )
        
        return EntryAdjustment(
            original_price=quoted_price,
            adjusted_price=adjusted_price,
            adjustment_pct=adjustment_pct,
            reason=reason,
            is_conservative=is_conservative
        )
    
    def get_market_period(self, current_time: Optional[datetime] = None) -> MarketPeriod:
        """
        Determine current market period.
        
        Args:
            current_time: Time to check (defaults to now)
            
        Returns:
            MarketPeriod
        """
        now = current_time or datetime.now()
        current = now.time()
        
        # Market hours: 9:30 AM - 4:00 PM ET
        market_open = time(9, 30)
        market_close = time(16, 0)
        
        # Define periods
        open_volatility_end = time(10, 0)  # First 30 minutes
        close_volatility_start = time(15, 30)  # Last 30 minutes
        
        # Check if market day (M-F, simplified - doesn't check holidays)
        if now.weekday() >= 5:  # Saturday=5, Sunday=6
            return MarketPeriod.CLOSED
        
        # Check time periods
        if current < market_open:
            return MarketPeriod.PRE_MARKET
        elif current < open_volatility_end:
            return MarketPeriod.OPEN_VOLATILITY
        elif current < close_volatility_start:
            return MarketPeriod.SAFE_TRADING
        elif current < market_close:
            return MarketPeriod.CLOSE_VOLATILITY
        else:
            return MarketPeriod.AFTER_HOURS
    
    def is_safe_to_trade(
        self,
        current_time: Optional[datetime] = None,
        vix_level: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Check if it's safe to enter new positions with delayed data.
        
        Args:
            current_time: Time to check (defaults to now)
            vix_level: Current VIX level (optional)
            
        Returns:
            (is_safe: bool, reason: str)
        """
        period = self.get_market_period(current_time)
        
        # Block during high-volatility periods
        if period in [MarketPeriod.OPEN_VOLATILITY, MarketPeriod.CLOSE_VOLATILITY]:
            return False, f"Avoid {period.value} (high volatility, delayed data risky)"
        
        # Block when market closed
        if period in [MarketPeriod.CLOSED, MarketPeriod.PRE_MARKET, MarketPeriod.AFTER_HOURS]:
            return False, f"Market {period.value}"
        
        # Check VIX if provided
        if vix_level is not None and vix_level > MAX_VIX_FOR_ENTRY:
            return False, f"VIX too high ({vix_level:.1f} > {MAX_VIX_FOR_ENTRY})"
        
        # Safe to trade
        return True, f"Safe trading window ({period.value})"
    
    def should_reduce_position_size(
        self,
        standard_size: int,
        vix_level: float,
        reduction_pct: float = 0.20
    ) -> Tuple[int, str]:
        """
        Recommend position size reduction for delayed data.
        
        Args:
            standard_size: Normal position size (contracts)
            vix_level: Current VIX level
            reduction_pct: % reduction to apply (default 20%)
            
        Returns:
            (adjusted_size: int, reason: str)
        """
        # Reduce size by 20% to account for delayed data uncertainty
        reduced_size = int(standard_size * (1 - reduction_pct))
        reduced_size = max(1, reduced_size)  # At least 1 contract
        
        reason = f"Reduced {reduction_pct:.0%} for 15-min delay"
        
        # Further reduction in high VIX environment
        if vix_level > 25:
            additional_reduction = 0.10
            reduced_size = int(reduced_size * (1 - additional_reduction))
            reduced_size = max(1, reduced_size)
            reason += f", additional {additional_reduction:.0%} for VIX {vix_level:.1f}"
        
        return reduced_size, reason
    
    def adjust_greeks_for_delay(
        self,
        symbol: str,
        quoted_greeks: Dict[str, float],
        dte: int,
        iv: float
    ) -> Dict[str, float]:
        """
        Adjust Greeks to account for data staleness.
        
        Greeks change over time (especially theta), so delayed data may
        underestimate current risk.
        
        Args:
            symbol: Underlying symbol
            quoted_greeks: Greeks from delayed quote
            dte: Days to expiration
            iv: Implied volatility
            
        Returns:
            Adjusted Greeks
        """
        adjusted = quoted_greeks.copy()
        
        # Theta accelerates as expiration approaches
        # For options near expiration, theta may have increased significantly
        if dte <= 7:
            # Near expiration: theta accelerates rapidly
            theta_adjustment = 1.15  # Assume 15% higher theta
        elif dte <= 21:
            # Acceleration zone: moderate adjustment
            theta_adjustment = 1.08
        else:
            # Far from expiration: minimal adjustment
            theta_adjustment = 1.02
        
        if 'theta' in adjusted:
            adjusted['theta'] *= theta_adjustment
        
        # Gamma also increases near expiration (more convexity risk)
        if dte <= 7 and 'gamma' in adjusted:
            adjusted['gamma'] *= 1.10
        
        # Vega decreases near expiration
        if dte <= 7 and 'vega' in adjusted:
            adjusted['vega'] *= 0.90
        
        logger.debug(
            f"{symbol} Greeks adjusted for delay (DTE={dte}): "
            f"theta×{theta_adjustment:.2f}"
        )
        
        return adjusted
    
    def check_dte_sufficient(self, dte: int) -> Tuple[bool, str]:
        """
        Check if DTE is sufficient for delayed data trading.
        
        With delayed data, avoid very short-dated options where
        gamma/theta risk changes rapidly.
        
        Args:
            dte: Days to expiration
            
        Returns:
            (is_sufficient: bool, reason: str)
        """
        min_dte = MIN_DTE_FOR_DELAYED_DATA
        
        if dte < min_dte:
            return False, (
                f"DTE too low for delayed data ({dte} < {min_dte} days). "
                "Gamma/theta risk changes too rapidly."
            )
        
        return True, f"DTE {dte} sufficient"
    
    def get_safe_exit_price(
        self,
        current_bid: float,
        current_ask: float,
        position_is_long: bool,
        use_conservative: bool = True
    ) -> float:
        """
        Get conservative exit price for delayed quotes.
        
        Args:
            current_bid: Bid price from delayed quote
            current_ask: Ask price from delayed quote
            position_is_long: True if closing long position
            use_conservative: Use conservative assumption (default True)
            
        Returns:
            Recommended exit price
        """
        if position_is_long:
            # Closing long: we're selling, use bid
            # Conservative: assume bid has moved down
            if use_conservative:
                return current_bid * 0.95  # 5% haircut
            return current_bid
        else:
            # Closing short: we're buying, use ask
            # Conservative: assume ask has moved up
            if use_conservative:
                return current_ask * 1.05  # 5% markup
            return current_ask
    
    def get_delay_report(self) -> Dict:
        """Get summary of delay adapter configuration and cached data."""
        return {
            'delay_minutes': self.delay_minutes,
            'credit_buffer_sigma': self.credit_buffer_sigma,
            'debit_buffer_sigma': self.debit_buffer_sigma,
            'cached_atrs': len(self.atr_cache),
            'safe_windows': SAFE_TRADING_WINDOWS,
            'max_vix': MAX_VIX_FOR_ENTRY,
            'min_dte': MIN_DTE_FOR_DELAYED_DATA,
        }
