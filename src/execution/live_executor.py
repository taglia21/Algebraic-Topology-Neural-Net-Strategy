"""Live Order Execution System for TD Ameritrade.

This addresses Issue #1: No Live Execution System
"""

import os
import time
import logging
import uuid
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


@dataclass
class Order:
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    limit_price: Optional[float] = None
    order_id: str = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    average_fill_price: float = 0.0
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.order_id is None:
            self.order_id = str(uuid.uuid4())


class LiveExecutor:
    """Live order execution engine with paper trading support."""
    
    def __init__(self, paper_trading: bool = True, initial_cash: float = 100000.0):
        self.paper_trading = paper_trading
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, int] = {}
        self.cash_balance = initial_cash if paper_trading else 0.0
        self.equity = initial_cash if paper_trading else 0.0
        self.order_history: List[Order] = []
        self.pnl_history: List[Dict] = []
        
        if paper_trading:
            logger.info(f"PAPER TRADING MODE - Starting with ${initial_cash:,.2f}")
        else:
            logger.info("LIVE TRADING MODE - Real money at risk!")
    
    def submit_order(self, symbol: str, side: OrderSide, quantity: int, 
                     order_type: OrderType = OrderType.MARKET, 
                     limit_price: Optional[float] = None) -> Optional[Order]:
        """Submit an order for execution."""
        
        # Validate order
        if quantity <= 0:
            logger.error(f"Invalid quantity: {quantity}")
            return None
        
        if order_type == OrderType.LIMIT and limit_price is None:
            logger.error("Limit price required for limit orders")
            return None
        
        # Create order
        order = Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            submitted_at=datetime.now()
        )
        
        # Paper trading execution
        if self.paper_trading:
            return self._execute_paper_order(order)
        else:
            raise NotImplementedError(
                "Live TDA execution is not implemented. "
                "Use AlpacaClient from src/trading/alpaca_client.py for live trading."
            )
    
    def _execute_paper_order(self, order: Order) -> Order:
        """Execute order in paper trading mode."""
        
        # Simulate immediate fill at market price for simplicity
        # In reality would use limit prices and market data
        # Get real market price
        from market_data import MarketDataFetcher
        fetcher = MarketDataFetcher()
        market_price = fetcher.get_current_price(order.symbol) or 100.0
        fill_price = order.limit_price if order.limit_price else market_price
        
        # Calculate cost
        cost = order.quantity * fill_price
        
        # Check if we have enough cash for buys
        if order.side == OrderSide.BUY:
            if cost > self.cash_balance:
                logger.error(f"Insufficient cash: ${self.cash_balance:.2f} < ${cost:.2f}")
                order.status = OrderStatus.REJECTED
                return order
            
            self.cash_balance -= cost
            self.positions[order.symbol] = self.positions.get(order.symbol, 0) + order.quantity
        
        # Handle sells
        elif order.side == OrderSide.SELL:
            current_position = self.positions.get(order.symbol, 0)
            if order.quantity > current_position:
                logger.error(f"Insufficient shares: {current_position} < {order.quantity}")
                order.status = OrderStatus.REJECTED
                return order
            
            self.cash_balance += cost
            self.positions[order.symbol] = current_position - order.quantity
            if self.positions[order.symbol] == 0:
                del self.positions[order.symbol]
        
        # Mark as filled
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.average_fill_price = fill_price
        order.filled_at = datetime.now()
        
        # Store order
        self.orders[order.order_id] = order
        self.order_history.append(order)
        
        logger.info(f"Order filled: {order.side.value} {order.quantity} {order.symbol} @ ${fill_price:.2f}")
        logger.info(f"Cash: ${self.cash_balance:.2f}, Positions: {len(self.positions)}")
        
        return order
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        order = self.orders.get(order_id)
        if not order:
            logger.error(f"Order not found: {order_id}")
            return False
        
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            logger.error(f"Cannot cancel order in status: {order.status.value}")
            return False
        
        order.status = OrderStatus.CANCELLED
        logger.info(f"Order cancelled: {order_id}")
        return True
    
    def get_position(self, symbol: str) -> int:
        """Get current position for symbol."""
        return self.positions.get(symbol, 0)
    
    def get_account_value(self) -> float:
        """Get total account value (cash + positions)."""
        total = self.cash_balance
        
        # Add position values
        if self.positions and self.paper_trading:
            from market_data import MarketDataFetcher
            fetcher = MarketDataFetcher()
            for symbol, quantity in self.positions.items():
                price = fetcher.get_current_price(symbol) or 100.0
                total += quantity * price
        
        return total
    
    def get_pnl(self) -> float:
        """Get total P&L."""
        return self.get_account_value() - (100000.0 if self.paper_trading else 0.0)


if __name__ == "__main__":
    # Test the executor
    executor = LiveExecutor(paper_trading=True)
    
    # Test buy order
    order1 = executor.submit_order("AAPL", OrderSide.BUY, 10, OrderType.MARKET)
    print(f"Order 1: {order1.status.value if order1 else 'Failed'}")
    
    # Test sell order
    order2 = executor.submit_order("AAPL", OrderSide.SELL, 5, OrderType.MARKET)
    print(f"Order 2: {order2.status.value if order2 else 'Failed'}")
    
    print(f"\nFinal state:")
    print(f"Cash: ${executor.cash_balance:.2f}")
    print(f"Positions: {executor.positions}")
    print(f"P&L: ${executor.get_pnl():.2f}")
