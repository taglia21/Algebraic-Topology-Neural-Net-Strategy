#!/usr/bin/env python3
"""
V23 Execution Engine
=====================
Production-grade order management for the V21 mean reversion strategy.

Features:
- Order management with pending/filled tracking
- Entry optimization (market open, limit, TWAP)
- Slippage tracking and analysis
- Alpaca API integration (paper + live)

Configuration via environment variables:
- ALPACA_API_KEY
- ALPACA_API_SECRET
- ALPACA_BASE_URL (paper: https://paper-api.alpaca.markets)
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('V23_Execution')


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    MARKET_WITH_LIMIT = "market_with_limit"  # Market with tolerance
    TWAP = "twap"  # Time-weighted average price


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Order:
    """Represents a trading order."""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    limit_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    filled_qty: float = 0.0
    avg_fill_price: Optional[float] = None
    expected_price: Optional[float] = None
    slippage_bps: Optional[float] = None
    broker_order_id: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'order_type': self.order_type.value,
            'limit_price': self.limit_price,
            'status': self.status.value,
            'submitted_at': self.submitted_at.isoformat() if self.submitted_at else None,
            'filled_at': self.filled_at.isoformat() if self.filled_at else None,
            'filled_qty': self.filled_qty,
            'avg_fill_price': self.avg_fill_price,
            'expected_price': self.expected_price,
            'slippage_bps': self.slippage_bps,
            'broker_order_id': self.broker_order_id,
            'error_message': self.error_message
        }


@dataclass
class Quote:
    """Market quote data."""
    symbol: str
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    last_price: float
    timestamp: datetime
    
    @property
    def mid_price(self) -> float:
        return (self.bid + self.ask) / 2
    
    @property
    def spread_bps(self) -> float:
        if self.mid_price == 0:
            return 0
        return (self.ask - self.bid) / self.mid_price * 10000


# =============================================================================
# BROKER API INTERFACE
# =============================================================================

class BrokerAPI(ABC):
    """Abstract broker API interface."""
    
    @abstractmethod
    def get_quote(self, symbol: str) -> Quote:
        """Get current quote for symbol."""
        pass
    
    @abstractmethod
    def submit_market_order(self, symbol: str, side: str, qty: float) -> str:
        """Submit market order, returns broker order ID."""
        pass
    
    @abstractmethod
    def submit_limit_order(self, symbol: str, side: str, qty: float, 
                          limit_price: float) -> str:
        """Submit limit order, returns broker order ID."""
        pass
    
    @abstractmethod
    def get_order_status(self, broker_order_id: str) -> Tuple[str, float, float]:
        """Get order status, filled qty, avg price."""
        pass
    
    @abstractmethod
    def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel order, returns success."""
        pass
    
    @abstractmethod
    def get_positions(self) -> Dict[str, float]:
        """Get current positions {symbol: quantity}."""
        pass
    
    @abstractmethod
    def get_account_value(self) -> float:
        """Get total account value."""
        pass
    
    @abstractmethod
    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        pass


class AlpacaAPI(BrokerAPI):
    """Alpaca Markets API implementation."""
    
    def __init__(self, paper_mode: bool = True):
        self.paper_mode = paper_mode
        self.api_key = os.environ.get('ALPACA_API_KEY', '')
        self.api_secret = os.environ.get('ALPACA_API_SECRET', '')
        
        if paper_mode:
            self.base_url = os.environ.get(
                'ALPACA_BASE_URL', 
                'https://paper-api.alpaca.markets'
            )
        else:
            self.base_url = 'https://api.alpaca.markets'
        
        self._validate_credentials()
        self.api = self._init_api()
        
        logger.info(f"AlpacaAPI initialized (paper_mode={paper_mode})")
    
    def _validate_credentials(self):
        """Validate API credentials are set."""
        if not self.api_key or not self.api_secret:
            logger.warning("Alpaca credentials not set. Using simulation mode.")
            self.simulation_mode = True
        else:
            self.simulation_mode = False
    
    def _init_api(self):
        """Initialize Alpaca API client."""
        if self.simulation_mode:
            return None
        
        try:
            import alpaca_trade_api as tradeapi
            return tradeapi.REST(
                self.api_key,
                self.api_secret,
                self.base_url,
                api_version='v2'
            )
        except ImportError:
            logger.warning("alpaca-trade-api not installed. Using simulation mode.")
            self.simulation_mode = True
            return None
    
    def get_quote(self, symbol: str) -> Quote:
        """Get current quote."""
        if self.simulation_mode:
            # Simulate quote
            base_price = 100.0 + hash(symbol) % 900
            spread = base_price * 0.001  # 10bps spread
            return Quote(
                symbol=symbol,
                bid=base_price - spread/2,
                ask=base_price + spread/2,
                bid_size=1000,
                ask_size=1000,
                last_price=base_price,
                timestamp=datetime.now()
            )
        
        try:
            quote = self.api.get_latest_quote(symbol)
            return Quote(
                symbol=symbol,
                bid=float(quote.bp),
                ask=float(quote.ap),
                bid_size=int(quote.bs),
                ask_size=int(quote.as_),
                last_price=float((quote.bp + quote.ap) / 2),
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {e}")
            raise
    
    def submit_market_order(self, symbol: str, side: str, qty: float) -> str:
        """Submit market order."""
        if self.simulation_mode:
            order_id = f"SIM-{symbol}-{int(time.time()*1000)}"
            logger.info(f"[SIMULATION] Market order: {side} {qty} {symbol}")
            return order_id
        
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='market',
                time_in_force='day'
            )
            return order.id
        except Exception as e:
            logger.error(f"Error submitting market order: {e}")
            raise
    
    def submit_limit_order(self, symbol: str, side: str, qty: float,
                          limit_price: float) -> str:
        """Submit limit order."""
        if self.simulation_mode:
            order_id = f"SIM-{symbol}-{int(time.time()*1000)}"
            logger.info(f"[SIMULATION] Limit order: {side} {qty} {symbol} @ {limit_price:.2f}")
            return order_id
        
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='limit',
                limit_price=round(limit_price, 2),
                time_in_force='day'
            )
            return order.id
        except Exception as e:
            logger.error(f"Error submitting limit order: {e}")
            raise
    
    def get_order_status(self, broker_order_id: str) -> Tuple[str, float, float]:
        """Get order status."""
        if self.simulation_mode:
            # Simulate filled
            return ('filled', 100.0, 150.00)
        
        try:
            order = self.api.get_order(broker_order_id)
            return (
                order.status,
                float(order.filled_qty or 0),
                float(order.filled_avg_price or 0)
            )
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            raise
    
    def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel order."""
        if self.simulation_mode:
            logger.info(f"[SIMULATION] Cancelled order {broker_order_id}")
            return True
        
        try:
            self.api.cancel_order(broker_order_id)
            return True
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False
    
    def get_positions(self) -> Dict[str, float]:
        """Get current positions."""
        if self.simulation_mode:
            return {}
        
        try:
            positions = self.api.list_positions()
            return {p.symbol: float(p.qty) for p in positions}
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return {}
    
    def get_account_value(self) -> float:
        """Get account value."""
        if self.simulation_mode:
            return 100000.0
        
        try:
            account = self.api.get_account()
            return float(account.portfolio_value)
        except Exception as e:
            logger.error(f"Error getting account value: {e}")
            return 0.0
    
    def is_market_open(self) -> bool:
        """Check if market is open."""
        if self.simulation_mode:
            now = datetime.now()
            # Assume market open 9:30-16:00 ET on weekdays
            if now.weekday() >= 5:
                return False
            hour = now.hour
            return 9 <= hour < 16
        
        try:
            clock = self.api.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Error checking market status: {e}")
            return False


# =============================================================================
# EXECUTION ENGINE
# =============================================================================

class ExecutionEngine:
    """
    Production-grade order execution engine.
    
    Features:
    - Multiple order types (market, limit, TWAP)
    - Entry optimization based on spread
    - Slippage tracking
    - Order lifecycle management
    """
    
    # Configuration
    SPREAD_THRESHOLD_BPS = 30  # Use limit order if spread > 30bps
    LARGE_ORDER_THRESHOLD = 10000  # Dollar value for TWAP
    TWAP_TRANCHES = 3
    TWAP_INTERVAL_SECONDS = 300  # 5 minutes between tranches
    MARKET_WITH_LIMIT_TOLERANCE = 0.005  # 0.5% tolerance
    
    def __init__(self, broker: BrokerAPI, paper_mode: bool = True):
        self.broker = broker
        self.paper_mode = paper_mode
        
        self.pending_orders: Dict[str, Order] = {}
        self.filled_orders: Dict[str, Order] = {}
        self.cancelled_orders: Dict[str, Order] = {}
        
        self.slippage_log: List[Dict] = []
        self.order_counter = 0
        
        # State persistence
        self.state_dir = Path('state/execution')
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ExecutionEngine initialized (paper_mode={paper_mode})")
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        self.order_counter += 1
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        return f"V23-{timestamp}-{self.order_counter:06d}"
    
    def submit_order(self, 
                    symbol: str, 
                    side: OrderSide,
                    quantity: float,
                    order_type: Optional[OrderType] = None,
                    dollar_value: Optional[float] = None) -> Order:
        """
        Submit order with intelligent routing.
        
        Args:
            symbol: Stock symbol
            side: Buy or sell
            quantity: Number of shares
            order_type: Force specific order type (optional)
            dollar_value: Dollar value for TWAP decision (optional)
        
        Returns:
            Order object with tracking info
        """
        # Get current quote
        try:
            quote = self.broker.get_quote(symbol)
        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {e}")
            order = Order(
                order_id=self._generate_order_id(),
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type or OrderType.MARKET,
                status=OrderStatus.REJECTED,
                error_message=str(e)
            )
            return order
        
        # Determine order type if not specified
        if order_type is None:
            order_type = self._select_order_type(quote, dollar_value)
        
        # Create order object
        order = Order(
            order_id=self._generate_order_id(),
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            expected_price=quote.mid_price,
            submitted_at=datetime.now()
        )
        
        # Execute based on order type
        try:
            if order_type == OrderType.MARKET:
                self._execute_market_order(order, quote)
            elif order_type == OrderType.LIMIT:
                self._execute_limit_order(order, quote)
            elif order_type == OrderType.MARKET_WITH_LIMIT:
                self._execute_market_with_limit(order, quote)
            elif order_type == OrderType.TWAP:
                self._execute_twap_order(order, quote)
            
            self.pending_orders[order.order_id] = order
            logger.info(f"Order submitted: {order.order_id} {side.value} {quantity} {symbol} "
                       f"type={order_type.value}")
            
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.error_message = str(e)
            logger.error(f"Order submission failed: {e}")
        
        return order
    
    def _select_order_type(self, quote: Quote, dollar_value: Optional[float]) -> OrderType:
        """Select optimal order type based on market conditions."""
        # Large orders use TWAP
        if dollar_value and dollar_value > self.LARGE_ORDER_THRESHOLD:
            logger.info(f"Large order (${dollar_value:,.0f}) - using TWAP")
            return OrderType.TWAP
        
        # Wide spread uses limit
        if quote.spread_bps > self.SPREAD_THRESHOLD_BPS:
            logger.info(f"Wide spread ({quote.spread_bps:.1f}bps) - using limit")
            return OrderType.LIMIT
        
        # Default to market
        return OrderType.MARKET
    
    def _execute_market_order(self, order: Order, quote: Quote):
        """Execute market order."""
        broker_id = self.broker.submit_market_order(
            order.symbol,
            order.side.value,
            order.quantity
        )
        order.broker_order_id = broker_id
        order.status = OrderStatus.SUBMITTED
    
    def _execute_limit_order(self, order: Order, quote: Quote):
        """Execute limit order at midpoint."""
        limit_price = quote.mid_price
        
        # Adjust for side (slightly improve price for fills)
        if order.side == OrderSide.BUY:
            limit_price = quote.mid_price + (quote.ask - quote.mid_price) * 0.3
        else:
            limit_price = quote.mid_price - (quote.mid_price - quote.bid) * 0.3
        
        order.limit_price = limit_price
        
        broker_id = self.broker.submit_limit_order(
            order.symbol,
            order.side.value,
            order.quantity,
            limit_price
        )
        order.broker_order_id = broker_id
        order.status = OrderStatus.SUBMITTED
    
    def _execute_market_with_limit(self, order: Order, quote: Quote):
        """Execute market order with price tolerance."""
        # Set limit at 0.5% from current price
        if order.side == OrderSide.BUY:
            limit_price = quote.ask * (1 + self.MARKET_WITH_LIMIT_TOLERANCE)
        else:
            limit_price = quote.bid * (1 - self.MARKET_WITH_LIMIT_TOLERANCE)
        
        order.limit_price = limit_price
        
        broker_id = self.broker.submit_limit_order(
            order.symbol,
            order.side.value,
            order.quantity,
            limit_price
        )
        order.broker_order_id = broker_id
        order.status = OrderStatus.SUBMITTED
    
    def _execute_twap_order(self, order: Order, quote: Quote):
        """Execute TWAP order (split into tranches)."""
        # For TWAP, we submit the first tranche and schedule the rest
        tranche_qty = order.quantity / self.TWAP_TRANCHES
        
        # Store TWAP metadata
        order.metadata['twap'] = {
            'total_tranches': self.TWAP_TRANCHES,
            'completed_tranches': 0,
            'tranche_qty': tranche_qty,
            'tranche_orders': []
        }
        
        # Submit first tranche as market
        broker_id = self.broker.submit_market_order(
            order.symbol,
            order.side.value,
            tranche_qty
        )
        
        order.metadata['twap']['tranche_orders'].append({
            'broker_id': broker_id,
            'qty': tranche_qty,
            'submitted_at': datetime.now().isoformat()
        })
        
        order.broker_order_id = broker_id
        order.status = OrderStatus.PARTIAL
        
        logger.info(f"TWAP tranche 1/{self.TWAP_TRANCHES} submitted: {tranche_qty} shares")
    
    def process_twap_tranches(self, order: Order):
        """Process remaining TWAP tranches (call periodically)."""
        if order.order_type != OrderType.TWAP:
            return
        
        twap = order.metadata.get('twap', {})
        completed = twap.get('completed_tranches', 0)
        total = twap.get('total_tranches', self.TWAP_TRANCHES)
        
        if completed >= total - 1:  # First tranche already submitted
            return
        
        # Submit next tranche
        quote = self.broker.get_quote(order.symbol)
        tranche_qty = twap['tranche_qty']
        
        broker_id = self.broker.submit_market_order(
            order.symbol,
            order.side.value,
            tranche_qty
        )
        
        twap['tranche_orders'].append({
            'broker_id': broker_id,
            'qty': tranche_qty,
            'submitted_at': datetime.now().isoformat()
        })
        twap['completed_tranches'] = completed + 1
        
        logger.info(f"TWAP tranche {completed+2}/{total} submitted: {tranche_qty} shares")
    
    def update_order_status(self, order: Order) -> Order:
        """Update order status from broker."""
        if not order.broker_order_id:
            return order
        
        try:
            status, filled_qty, avg_price = self.broker.get_order_status(
                order.broker_order_id
            )
            
            # Map broker status to our status
            status_map = {
                'new': OrderStatus.SUBMITTED,
                'accepted': OrderStatus.SUBMITTED,
                'pending_new': OrderStatus.SUBMITTED,
                'partially_filled': OrderStatus.PARTIAL,
                'filled': OrderStatus.FILLED,
                'canceled': OrderStatus.CANCELLED,
                'cancelled': OrderStatus.CANCELLED,
                'rejected': OrderStatus.REJECTED,
                'expired': OrderStatus.EXPIRED
            }
            
            order.status = status_map.get(status.lower(), OrderStatus.PENDING)
            order.filled_qty = filled_qty
            order.avg_fill_price = avg_price if avg_price > 0 else None
            
            # Calculate slippage if filled
            if order.status == OrderStatus.FILLED and order.avg_fill_price:
                order.filled_at = datetime.now()
                self._calculate_slippage(order)
                
                # Move to filled orders
                if order.order_id in self.pending_orders:
                    del self.pending_orders[order.order_id]
                self.filled_orders[order.order_id] = order
                
        except Exception as e:
            logger.error(f"Error updating order status: {e}")
        
        return order
    
    def _calculate_slippage(self, order: Order):
        """Calculate and log slippage."""
        if not order.expected_price or not order.avg_fill_price:
            return
        
        if order.side == OrderSide.BUY:
            # For buys, positive slippage means we paid more
            slippage = (order.avg_fill_price - order.expected_price) / order.expected_price
        else:
            # For sells, positive slippage means we received less
            slippage = (order.expected_price - order.avg_fill_price) / order.expected_price
        
        order.slippage_bps = slippage * 10000
        
        # Log for analysis
        self.slippage_log.append({
            'order_id': order.order_id,
            'symbol': order.symbol,
            'side': order.side.value,
            'quantity': order.quantity,
            'expected_price': order.expected_price,
            'fill_price': order.avg_fill_price,
            'slippage_bps': order.slippage_bps,
            'order_type': order.order_type.value,
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"Order {order.order_id} filled: slippage={order.slippage_bps:.1f}bps")
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order."""
        order = self.pending_orders.get(order_id)
        if not order:
            logger.warning(f"Order {order_id} not found in pending orders")
            return False
        
        if order.broker_order_id:
            success = self.broker.cancel_order(order.broker_order_id)
            if success:
                order.status = OrderStatus.CANCELLED
                del self.pending_orders[order_id]
                self.cancelled_orders[order_id] = order
                logger.info(f"Order {order_id} cancelled")
                return True
        
        return False
    
    def cancel_all_pending(self) -> int:
        """Cancel all pending orders. Returns count cancelled."""
        cancelled = 0
        order_ids = list(self.pending_orders.keys())
        
        for order_id in order_ids:
            if self.cancel_order(order_id):
                cancelled += 1
        
        logger.info(f"Cancelled {cancelled} pending orders")
        return cancelled
    
    def get_slippage_stats(self) -> Dict:
        """Get slippage statistics."""
        if not self.slippage_log:
            return {'count': 0}
        
        slippages = [log['slippage_bps'] for log in self.slippage_log]
        
        return {
            'count': len(slippages),
            'mean_bps': np.mean(slippages),
            'median_bps': np.median(slippages),
            'std_bps': np.std(slippages),
            'max_bps': max(slippages),
            'min_bps': min(slippages),
            'total_cost_bps': sum(slippages)
        }
    
    def save_state(self):
        """Save engine state to disk."""
        state = {
            'pending_orders': {k: v.to_dict() for k, v in self.pending_orders.items()},
            'filled_orders': {k: v.to_dict() for k, v in self.filled_orders.items()},
            'slippage_log': self.slippage_log,
            'order_counter': self.order_counter,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.state_dir / 'execution_state.json', 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info("Execution state saved")
    
    def load_state(self):
        """Load engine state from disk."""
        state_file = self.state_dir / 'execution_state.json'
        if not state_file.exists():
            return
        
        with open(state_file) as f:
            state = json.load(f)
        
        self.slippage_log = state.get('slippage_log', [])
        self.order_counter = state.get('order_counter', 0)
        
        logger.info("Execution state loaded")
    
    def get_status_summary(self) -> Dict:
        """Get execution engine status summary."""
        return {
            'pending_orders': len(self.pending_orders),
            'filled_orders': len(self.filled_orders),
            'cancelled_orders': len(self.cancelled_orders),
            'slippage_stats': self.get_slippage_stats(),
            'paper_mode': self.paper_mode,
            'market_open': self.broker.is_market_open()
        }


# =============================================================================
# EXECUTION MANAGER (High-Level Interface)
# =============================================================================

class ExecutionManager:
    """
    High-level execution manager for V21 strategy.
    Coordinates signal generation, order execution, and position tracking.
    """
    
    def __init__(self, paper_mode: bool = True):
        self.paper_mode = paper_mode
        self.broker = AlpacaAPI(paper_mode=paper_mode)
        self.engine = ExecutionEngine(self.broker, paper_mode=paper_mode)
        
        # Trade log
        self.trade_log: List[Dict] = []
        
        logger.info(f"ExecutionManager initialized (paper_mode={paper_mode})")
    
    def execute_rebalance(self, 
                         target_positions: Dict[str, float],
                         account_value: Optional[float] = None) -> List[Order]:
        """
        Execute rebalance to target positions.
        
        Args:
            target_positions: {symbol: target_weight} where weight is 0-1
            account_value: Total account value (fetched if not provided)
        
        Returns:
            List of executed orders
        """
        if account_value is None:
            account_value = self.broker.get_account_value()
        
        current_positions = self.broker.get_positions()
        orders = []
        
        # Calculate target shares for each position
        for symbol, target_weight in target_positions.items():
            try:
                quote = self.broker.get_quote(symbol)
                target_value = account_value * target_weight
                target_shares = target_value / quote.mid_price
                
                current_shares = current_positions.get(symbol, 0)
                delta_shares = target_shares - current_shares
                
                if abs(delta_shares) < 1:
                    continue  # Skip tiny trades
                
                side = OrderSide.BUY if delta_shares > 0 else OrderSide.SELL
                qty = abs(delta_shares)
                dollar_value = qty * quote.mid_price
                
                order = self.engine.submit_order(
                    symbol=symbol,
                    side=side,
                    quantity=qty,
                    dollar_value=dollar_value
                )
                orders.append(order)
                
                self.trade_log.append({
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'side': side.value,
                    'quantity': qty,
                    'target_weight': target_weight,
                    'order_id': order.order_id
                })
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
        
        return orders
    
    def close_all_positions(self) -> List[Order]:
        """Emergency close all positions."""
        current_positions = self.broker.get_positions()
        orders = []
        
        for symbol, qty in current_positions.items():
            if qty != 0:
                order = self.engine.submit_order(
                    symbol=symbol,
                    side=OrderSide.SELL if qty > 0 else OrderSide.BUY,
                    quantity=abs(qty),
                    order_type=OrderType.MARKET
                )
                orders.append(order)
        
        logger.warning(f"EMERGENCY: Closed all positions ({len(orders)} orders)")
        return orders
    
    def get_execution_report(self) -> Dict:
        """Generate execution report."""
        return {
            'account_value': self.broker.get_account_value(),
            'positions': self.broker.get_positions(),
            'engine_status': self.engine.get_status_summary(),
            'recent_trades': self.trade_log[-20:],
            'timestamp': datetime.now().isoformat()
        }


# =============================================================================
# MAIN / TESTING
# =============================================================================

def main():
    """Test execution engine."""
    logger.info("=" * 70)
    logger.info("🚀 V23 EXECUTION ENGINE TEST")
    logger.info("=" * 70)
    
    # Initialize in paper mode
    manager = ExecutionManager(paper_mode=True)
    
    # Test order submission
    logger.info("\n📋 Testing order types...")
    
    # Market order (small)
    order1 = manager.engine.submit_order(
        symbol='AAPL',
        side=OrderSide.BUY,
        quantity=10
    )
    logger.info(f"   Market order: {order1.order_id} - {order1.status.value}")
    
    # Limit order (simulate wide spread)
    order2 = manager.engine.submit_order(
        symbol='MSFT',
        side=OrderSide.BUY,
        quantity=20,
        order_type=OrderType.LIMIT
    )
    logger.info(f"   Limit order: {order2.order_id} - {order2.status.value}")
    
    # TWAP order (large)
    order3 = manager.engine.submit_order(
        symbol='GOOGL',
        side=OrderSide.BUY,
        quantity=100,
        dollar_value=15000  # Triggers TWAP
    )
    logger.info(f"   TWAP order: {order3.order_id} - {order3.status.value}")
    
    # Test rebalance
    logger.info("\n📊 Testing rebalance...")
    target_positions = {
        'AAPL': 0.10,
        'MSFT': 0.10,
        'GOOGL': 0.10
    }
    orders = manager.execute_rebalance(target_positions)
    logger.info(f"   Rebalance orders: {len(orders)}")
    
    # Get status
    logger.info("\n📈 Execution Status:")
    status = manager.engine.get_status_summary()
    for key, value in status.items():
        if isinstance(value, dict):
            logger.info(f"   {key}:")
            for k, v in value.items():
                logger.info(f"      {k}: {v}")
        else:
            logger.info(f"   {key}: {value}")
    
    # Save state
    manager.engine.save_state()
    
    logger.info("\n✅ Execution engine test complete")
    
    return manager


if __name__ == "__main__":
    main()
