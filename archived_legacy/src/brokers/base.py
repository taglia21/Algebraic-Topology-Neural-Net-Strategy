"""
Base Broker Client
==================

Abstract base class and shared dataclasses for broker integrations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class AccountInfo:
    """Broker account summary."""
    cash: float
    buying_power: float
    portfolio_value: float
    currency: str = "USD"


@dataclass
class Position:
    """Open position."""
    symbol: str
    qty: float
    avg_cost: float
    market_value: float
    unrealized_pnl: float
    side: str = "long"  # 'long' or 'short'


@dataclass
class Order:
    """Order receipt."""
    order_id: str
    symbol: str
    qty: float
    side: str
    order_type: str
    status: str
    filled_qty: float = 0.0
    filled_avg_price: float = 0.0
    created_at: Optional[datetime] = None


@dataclass
class Bar:
    """OHLCV bar."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class OptionContract:
    """Option contract with live Greeks."""
    symbol: str
    expiry: str
    strike: float
    right: str  # 'C' or 'P'
    bid: float = 0.0
    ask: float = 0.0
    last: float = 0.0
    volume: int = 0
    open_interest: int = 0
    implied_volatility: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0


class BaseBrokerClient(ABC):
    """Abstract interface every broker adapter must implement."""

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def get_account(self) -> AccountInfo: ...

    @abstractmethod
    def get_positions(self) -> List[Position]: ...

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        limit_price: Optional[float] = None,
    ) -> Order: ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool: ...

    @abstractmethod
    def close_position(self, symbol: str) -> Optional[Order]: ...

    @abstractmethod
    def get_bars(self, symbol: str, timeframe: str, limit: int) -> List[Bar]: ...

    @abstractmethod
    def get_option_chain(
        self, symbol: str, expiry: str
    ) -> List[OptionContract]: ...

    @abstractmethod
    def get_vix(self) -> float: ...
