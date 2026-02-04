"""
Position Manager
================

Manage option positions with Kelly-based sizing and risk controls.

Handles:
- Position sizing with Kelly criterion
- Capital availability checks
- Margin requirement calculations
- Position tracking and P&L
- Portfolio construction limits
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum

from .utils.risk_metrics import calculate_kelly_fraction
from .utils.black_scholes import OptionType, Greeks
from .utils.constants import (
    MAX_POSITION_PCT,
    MAX_POSITIONS,
    KELLY_FRACTION_BASE,
    MAX_KELLY_FRACTION,
    RESERVE_CAPITAL_PCT,
    MARGIN_REQUIREMENT_PCT,
)

logger = logging.getLogger(__name__)


class PositionStatus(Enum):
    """Status of an option position."""
    PENDING = "pending"
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"
    EXPIRED = "expired"
    ASSIGNED = "assigned"


@dataclass
class Position:
    """An open option position."""
    symbol: str
    strike: float
    expiration: datetime
    option_type: OptionType
    quantity: int  # Number of contracts
    entry_price: float  # Price paid/received per contract
    entry_time: datetime
    
    current_price: float = 0.0
    status: PositionStatus = PositionStatus.OPEN
    
    greeks: Optional[Greeks] = None
    underlying_price: float = 0.0
    iv: float = 0.0
    
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    realized_pnl: Optional[float] = None
    
    notes: str = ""
    
    def unrealized_pnl(self) -> float:
        """Calculate current unrealized P&L."""
        if self.status == PositionStatus.CLOSED:
            return self.realized_pnl or 0.0
        
        # For short positions (negative quantity), profit when price decreases
        return (self.current_price - self.entry_price) * self.quantity * 100
    
    def pnl_percent(self) -> float:
        """Calculate P&L as percentage of risk."""
        risk = abs(self.entry_price * self.quantity * 100)
        if risk == 0:
            return 0.0
        return (self.unrealized_pnl() / risk) * 100
    
    def close(self, exit_price: float, exit_time: Optional[datetime] = None) -> float:
        """
        Close the position and calculate realized P&L.
        
        Returns:
            Realized P&L
        """
        self.exit_price = exit_price
        self.exit_time = exit_time or datetime.now()
        self.realized_pnl = (exit_price - self.entry_price) * self.quantity * 100
        self.status = PositionStatus.CLOSED
        
        logger.info(
            f"Position closed: {self.symbol} {self.strike} {self.option_type.value} "
            f"P&L: ${self.realized_pnl:+,.2f} ({self.pnl_percent():+.1f}%)"
        )
        
        return self.realized_pnl
    
    def __str__(self) -> str:
        pnl = self.realized_pnl if self.status == PositionStatus.CLOSED else self.unrealized_pnl()
        return (
            f"{self.symbol} {self.strike} {self.option_type.value} x{self.quantity:+d} "
            f"@ ${self.entry_price:.2f} → ${self.current_price:.2f} "
            f"P&L: ${pnl:+,.2f} ({self.pnl_percent():+.1f}%)"
        )


@dataclass
class PositionSizing:
    """Position size recommendation."""
    num_contracts: int
    capital_required: float
    risk_per_contract: float
    max_loss: float
    kelly_fraction: float
    reasoning: str


class PositionManager:
    """
    Position Manager.
    
    Manages option positions with Kelly-based sizing and risk controls.
    
    Usage:
        manager = PositionManager(account_value=100000, buying_power=50000)
        sizing = manager.calculate_position_size(win_rate=0.65, avg_win=150, avg_loss=100, ...)
        can_open, reason = manager.can_open_position(symbol='SPY', capital_required=5000)
        manager.open_position(...)
    """
    
    def __init__(
        self,
        account_value: float,
        buying_power: float,
        max_positions: int = MAX_POSITIONS,
        max_position_pct: float = MAX_POSITION_PCT,
        reserve_pct: float = RESERVE_CAPITAL_PCT
    ):
        """
        Initialize position manager.
        
        Args:
            account_value: Total account equity
            buying_power: Available buying power
            max_positions: Maximum number of concurrent positions
            max_position_pct: Max % of account per position
            reserve_pct: % of capital to reserve (not allocate)
        """
        self.account_value = account_value
        self.buying_power = buying_power
        self.max_positions = max_positions
        self.max_position_pct = max_position_pct
        self.reserve_pct = reserve_pct
        
        # Active positions
        self.positions: List[Position] = []
        self.closed_positions: List[Position] = []
        
        # Performance tracking
        self.total_realized_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        logger.info(
            f"Position Manager initialized: Account=${account_value:,.0f}, "
            f"BP=${buying_power:,.0f}, Max positions={max_positions}"
        )
    
    def calculate_position_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        option_price: float,
        max_contracts: Optional[int] = None,
        kelly_multiplier: float = KELLY_FRACTION_BASE
    ) -> PositionSizing:
        """
        Calculate optimal position size using Kelly criterion.
        
        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade ($)
            avg_loss: Average losing trade ($, positive number)
            option_price: Price per option contract
            max_contracts: Maximum contracts allowed (optional)
            kelly_multiplier: Fraction of Kelly to use (0-1)
            
        Returns:
            PositionSizing recommendation
        """
        # Calculate Kelly fraction
        kelly_frac = calculate_kelly_fraction(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_fraction=MAX_KELLY_FRACTION
        )
        
        # Apply multiplier for safety
        adjusted_kelly = kelly_frac * kelly_multiplier
        
        # Calculate available capital (excluding reserve)
        available_capital = self.buying_power * (1 - self.reserve_pct)
        
        # Max capital per position
        max_position_capital = self.account_value * self.max_position_pct
        
        # Kelly-based capital allocation
        kelly_capital = min(
            available_capital * adjusted_kelly,
            max_position_capital
        )
        
        # Convert to number of contracts
        risk_per_contract = option_price * 100  # Options are 100 shares
        num_contracts = int(kelly_capital / risk_per_contract)
        
        # Apply max contracts limit if provided
        if max_contracts is not None:
            num_contracts = min(num_contracts, max_contracts)
        
        # Ensure at least 1 contract if capital allows
        if num_contracts == 0 and available_capital >= risk_per_contract:
            num_contracts = 1
            reasoning = "Minimum 1 contract (Kelly suggests less)"
        else:
            reasoning = f"Kelly {adjusted_kelly:.1%} of ${available_capital:,.0f}"
        
        actual_capital = num_contracts * risk_per_contract
        max_loss = actual_capital  # For buying options
        
        return PositionSizing(
            num_contracts=num_contracts,
            capital_required=actual_capital,
            risk_per_contract=risk_per_contract,
            max_loss=max_loss,
            kelly_fraction=adjusted_kelly,
            reasoning=reasoning
        )
    
    def calculate_margin_requirement(
        self,
        symbol: str,
        strike: float,
        option_type: OptionType,
        quantity: int,
        underlying_price: float,
        is_cash_secured: bool = True
    ) -> float:
        """
        Calculate margin requirement for selling options.
        
        Args:
            symbol: Underlying symbol
            strike: Option strike
            option_type: CALL or PUT
            quantity: Number of contracts (negative for short)
            underlying_price: Current underlying price
            is_cash_secured: True for CSP/covered call
            
        Returns:
            Margin requirement ($)
        """
        if quantity >= 0:
            # Long options: no margin, just premium
            return 0.0
        
        num_contracts = abs(quantity)
        
        if is_cash_secured:
            if option_type == OptionType.PUT:
                # Cash-secured put: need cash to buy stock
                return strike * 100 * num_contracts
            else:
                # Covered call: need to own stock (already accounted for)
                return 0.0
        else:
            # Naked option: use standard margin requirement
            # Simplified: 20% of underlying value
            return underlying_price * 100 * num_contracts * MARGIN_REQUIREMENT_PCT
    
    def can_open_position(
        self,
        symbol: str,
        capital_required: float,
        margin_required: float = 0.0
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a new position can be opened.
        
        Args:
            symbol: Underlying symbol
            capital_required: Capital needed for position
            margin_required: Margin requirement (for short positions)
            
        Returns:
            (can_open: bool, reason: Optional[str])
        """
        # Check position limit
        if len(self.positions) >= self.max_positions:
            return False, f"Max positions reached ({self.max_positions})"
        
        # Check buying power
        total_required = capital_required + margin_required
        if total_required > self.buying_power:
            return False, (
                f"Insufficient buying power: need ${total_required:,.0f}, "
                f"have ${self.buying_power:,.0f}"
            )
        
        # Check position size limit
        max_position_capital = self.account_value * self.max_position_pct
        if capital_required > max_position_capital:
            return False, (
                f"Position too large: ${capital_required:,.0f} > "
                f"${max_position_capital:,.0f} ({self.max_position_pct:.0%} limit)"
            )
        
        # Check concentration (don't put too much in one symbol)
        existing_exposure = sum(
            abs(pos.entry_price * pos.quantity * 100)
            for pos in self.positions
            if pos.symbol == symbol and pos.status == PositionStatus.OPEN
        )
        total_exposure = existing_exposure + capital_required
        max_symbol_exposure = self.account_value * 0.25  # Max 25% in one symbol
        
        if total_exposure > max_symbol_exposure:
            return False, (
                f"Too much exposure to {symbol}: ${total_exposure:,.0f} > "
                f"${max_symbol_exposure:,.0f} (25% limit)"
            )
        
        return True, None
    
    def open_position(
        self,
        symbol: str,
        strike: float,
        expiration: datetime,
        option_type: OptionType,
        quantity: int,
        entry_price: float,
        greeks: Optional[Greeks] = None,
        underlying_price: float = 0.0,
        iv: float = 0.0,
        notes: str = ""
    ) -> Optional[Position]:
        """
        Open a new position.
        
        Args:
            symbol: Underlying symbol
            strike: Option strike price
            expiration: Expiration date
            option_type: CALL or PUT
            quantity: Number of contracts (+ for long, - for short)
            entry_price: Entry price per contract
            greeks: Option Greeks (optional)
            underlying_price: Current underlying price
            iv: Implied volatility
            notes: Additional notes
            
        Returns:
            Position object if opened, None if rejected
        """
        capital_required = abs(entry_price * quantity * 100)
        margin_required = self.calculate_margin_requirement(
            symbol, strike, option_type, quantity, underlying_price
        )
        
        can_open, reason = self.can_open_position(symbol, capital_required, margin_required)
        if not can_open:
            logger.warning(f"Cannot open position: {reason}")
            return None
        
        position = Position(
            symbol=symbol,
            strike=strike,
            expiration=expiration,
            option_type=option_type,
            quantity=quantity,
            entry_price=entry_price,
            entry_time=datetime.now(),
            current_price=entry_price,
            greeks=greeks,
            underlying_price=underlying_price,
            iv=iv,
            notes=notes
        )
        
        self.positions.append(position)
        self.buying_power -= (capital_required + margin_required)
        
        logger.info(
            f"Position opened: {position.symbol} {position.strike} "
            f"{position.option_type.value} x{quantity:+d} @ ${entry_price:.2f}"
        )
        
        return position
    
    def close_position(
        self,
        position: Position,
        exit_price: float,
        exit_time: Optional[datetime] = None
    ) -> float:
        """
        Close an existing position.
        
        Args:
            position: Position to close
            exit_price: Exit price per contract
            exit_time: Exit timestamp
            
        Returns:
            Realized P&L
        """
        # Calculate P&L
        realized_pnl = position.close(exit_price, exit_time)
        
        # Update statistics
        self.total_realized_pnl += realized_pnl
        self.total_trades += 1
        if realized_pnl > 0:
            self.winning_trades += 1
        elif realized_pnl < 0:
            self.losing_trades += 1
        
        # Return capital
        capital_returned = abs(exit_price * position.quantity * 100)
        margin_returned = self.calculate_margin_requirement(
            position.symbol,
            position.strike,
            position.option_type,
            position.quantity,
            position.underlying_price
        )
        self.buying_power += (capital_returned + margin_returned)
        
        # Move to closed positions
        if position in self.positions:
            self.positions.remove(position)
        self.closed_positions.append(position)
        
        return realized_pnl
    
    def update_position_prices(
        self,
        symbol: str,
        strike: float,
        expiration: datetime,
        option_type: OptionType,
        current_price: float,
        greeks: Optional[Greeks] = None,
        underlying_price: Optional[float] = None
    ) -> None:
        """Update market data for a position."""
        for pos in self.positions:
            if (pos.symbol == symbol and pos.strike == strike and
                pos.expiration == expiration and pos.option_type == option_type):
                pos.current_price = current_price
                if greeks:
                    pos.greeks = greeks
                if underlying_price:
                    pos.underlying_price = underlying_price
                return
        
        logger.warning(f"Position not found for update: {symbol} {strike} {option_type.value}")
    
    def get_total_unrealized_pnl(self) -> float:
        """Get total unrealized P&L across all open positions."""
        return sum(pos.unrealized_pnl() for pos in self.positions)
    
    def get_total_pnl(self) -> float:
        """Get total P&L (realized + unrealized)."""
        return self.total_realized_pnl + self.get_total_unrealized_pnl()
    
    def get_win_rate(self) -> float:
        """Calculate win rate from closed positions."""
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades
    
    def get_performance_summary(self) -> Dict:
        """Get detailed performance metrics."""
        return {
            'account_value': self.account_value,
            'buying_power': self.buying_power,
            'open_positions': len(self.positions),
            'total_trades': self.total_trades,
            'win_rate': self.get_win_rate(),
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'total_realized_pnl': self.total_realized_pnl,
            'total_unrealized_pnl': self.get_total_unrealized_pnl(),
            'total_pnl': self.get_total_pnl(),
            'pnl_percent': (self.get_total_pnl() / self.account_value * 100) if self.account_value > 0 else 0
        }
    
    def get_positions_summary(self) -> str:
        """Get human-readable summary of all positions."""
        if not self.positions:
            return "No open positions"
        
        lines = [f"Open Positions ({len(self.positions)}):"]
        total_pnl = 0.0
        
        for pos in self.positions:
            lines.append(f"  {pos}")
            total_pnl += pos.unrealized_pnl()
        
        lines.append(f"\nTotal Unrealized P&L: ${total_pnl:+,.2f}")
        lines.append(f"Total Realized P&L: ${self.total_realized_pnl:+,.2f}")
        lines.append(f"Win Rate: {self.get_win_rate():.1%} ({self.winning_trades}W/{self.losing_trades}L)")
        
        return "\n".join(lines)
    
    def update_account_value(self, new_value: float) -> None:
        """Update account value (after deposits/withdrawals or market changes)."""
        old_value = self.account_value
        self.account_value = new_value
        
        logger.info(f"Account value updated: ${old_value:,.0f} → ${new_value:,.0f}")
    
    def update_buying_power(self, new_bp: float) -> None:
        """Update available buying power."""
        old_bp = self.buying_power
        self.buying_power = new_bp
        
        logger.debug(f"Buying power updated: ${old_bp:,.0f} → ${new_bp:,.0f}")
