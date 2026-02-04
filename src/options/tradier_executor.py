"""
Tradier Executor
================

Execute option orders via Tradier API with retry logic and error handling.

Features:
- Single-leg orders (buy/sell calls/puts)
- Multi-leg spreads
- Order status monitoring
- Retry with exponential backoff
- Position closure
"""

import logging
import time
import requests
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum

from .utils.black_scholes import OptionType
from .utils.constants import (
    ORDER_TIMEOUT_SECONDS,
    MAX_RETRY_ATTEMPTS,
    RETRY_BACKOFF_SECONDS,
)

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    """Order side."""
    BUY_TO_OPEN = "buy_to_open"
    BUY_TO_CLOSE = "buy_to_close"
    SELL_TO_OPEN = "sell_to_open"
    SELL_TO_CLOSE = "sell_to_close"


class OrderType(Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class OrderResult:
    """Result of order execution."""
    order_id: Optional[str]
    status: OrderStatus
    filled_quantity: int
    average_fill_price: Optional[float]
    commission: float
    error_message: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    @property
    def is_success(self) -> bool:
        """Check if order succeeded."""
        return self.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]
    
    @property
    def is_complete(self) -> bool:
        """Check if order is in terminal state."""
        return self.status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED
        ]


@dataclass
class OptionLeg:
    """Single leg of an option order."""
    symbol: str  # Underlying symbol
    strike: float
    expiration: str  # Format: YYYY-MM-DD
    option_type: OptionType
    side: OrderSide
    quantity: int
    
    def to_tradier_symbol(self) -> str:
        """
        Convert to Tradier option symbol format.
        
        Format: SYMBOL YYMMDD C/P STRIKE
        Example: SPY 240315C00450000 (SPY March 15, 2024, Call, $450)
        """
        # Parse expiration
        exp_parts = self.expiration.split('-')
        year = exp_parts[0][2:]  # Last 2 digits
        month = exp_parts[1]
        day = exp_parts[2]
        
        # Option type: C or P
        opt_type = 'C' if self.option_type == OptionType.CALL else 'P'
        
        # Strike: 8 digits (dollars * 1000)
        strike_formatted = f"{int(self.strike * 1000):08d}"
        
        return f"{self.symbol}{year}{month}{day}{opt_type}{strike_formatted}"


class TradierExecutor:
    """
    Tradier API Order Executor.
    
    Handles order placement, monitoring, and management via Tradier API.
    
    Usage:
        executor = TradierExecutor(api_key='...', account_id='...')
        result = executor.place_option_order(
            symbol='SPY',
            strike=450,
            expiration='2024-03-15',
            option_type=OptionType.PUT,
            side=OrderSide.SELL_TO_OPEN,
            quantity=1,
            limit_price=4.50
        )
    """
    
    def __init__(
        self,
        api_key: str,
        account_id: str,
        sandbox: bool = True,
        timeout: int = ORDER_TIMEOUT_SECONDS
    ):
        """
        Initialize Tradier executor.
        
        Args:
            api_key: Tradier API key
            account_id: Tradier account ID
            sandbox: Use sandbox environment (default True)
            timeout: Order timeout in seconds
        """
        self.api_key = api_key
        self.account_id = account_id
        self.sandbox = sandbox
        self.timeout = timeout
        
        # API endpoints
        if sandbox:
            self.base_url = "https://sandbox.tradier.com/v1"
        else:
            self.base_url = "https://api.tradier.com/v1"
        
        # Request headers
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Accept': 'application/json'
        }
        
        logger.info(
            f"Tradier Executor initialized: account={account_id}, "
            f"sandbox={sandbox}"
        )
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict:
        """
        Make API request with error handling.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (e.g., '/accounts/{account_id}/orders')
            data: Request body data
            params: Query parameters
            
        Returns:
            Response JSON
            
        Raises:
            Exception: If request fails
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=self.headers,
                data=data,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
    
    def place_option_order(
        self,
        symbol: str,
        strike: float,
        expiration: str,  # YYYY-MM-DD
        option_type: OptionType,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.LIMIT,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        duration: str = "day"
    ) -> OrderResult:
        """
        Place single-leg option order.
        
        Args:
            symbol: Underlying symbol
            strike: Option strike price
            expiration: Expiration date (YYYY-MM-DD)
            option_type: CALL or PUT
            side: Order side (buy_to_open, sell_to_open, etc.)
            quantity: Number of contracts
            order_type: Market or Limit
            limit_price: Limit price (required for limit orders)
            stop_price: Stop price (for stop orders)
            duration: Order duration ("day", "gtc", etc.)
            
        Returns:
            OrderResult
        """
        # Create option leg
        leg = OptionLeg(
            symbol=symbol,
            strike=strike,
            expiration=expiration,
            option_type=option_type,
            side=side,
            quantity=quantity
        )
        
        # Build order data
        order_data = {
            'class': 'option',
            'symbol': symbol,
            'option_symbol': leg.to_tradier_symbol(),
            'side': side.value,
            'quantity': quantity,
            'type': order_type.value,
            'duration': duration
        }
        
        # Add price for limit orders
        if order_type == OrderType.LIMIT:
            if limit_price is None:
                logger.error("Limit price required for limit order")
                return OrderResult(
                    order_id=None,
                    status=OrderStatus.REJECTED,
                    filled_quantity=0,
                    average_fill_price=None,
                    commission=0.0,
                    error_message="Limit price required"
                )
            order_data['price'] = f"{limit_price:.2f}"
        
        # Add stop price for stop orders
        if order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            if stop_price is None:
                return OrderResult(
                    order_id=None,
                    status=OrderStatus.REJECTED,
                    filled_quantity=0,
                    average_fill_price=None,
                    commission=0.0,
                    error_message="Stop price required"
                )
            order_data['stop'] = f"{stop_price:.2f}"
        
        # Place order with retry logic
        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                logger.info(
                    f"Placing order (attempt {attempt + 1}/{MAX_RETRY_ATTEMPTS}): "
                    f"{symbol} {strike} {option_type.value} {side.value} x{quantity}"
                )
                
                response = self._make_request(
                    method='POST',
                    endpoint=f'/accounts/{self.account_id}/orders',
                    data=order_data
                )
                
                # Parse response
                order_info = response.get('order', {})
                order_id = order_info.get('id')
                status_str = order_info.get('status', 'pending')
                
                # Map status
                status = self._parse_order_status(status_str)
                
                logger.info(f"Order placed: ID={order_id}, status={status.value}")
                
                return OrderResult(
                    order_id=str(order_id) if order_id else None,
                    status=status,
                    filled_quantity=0,  # Will be updated when monitored
                    average_fill_price=None,
                    commission=0.0
                )
            
            except Exception as e:
                logger.error(f"Order placement failed (attempt {attempt + 1}): {e}")
                
                if attempt < MAX_RETRY_ATTEMPTS - 1:
                    # Exponential backoff
                    wait_time = RETRY_BACKOFF_SECONDS * (2 ** attempt)
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    # Final attempt failed
                    return OrderResult(
                        order_id=None,
                        status=OrderStatus.REJECTED,
                        filled_quantity=0,
                        average_fill_price=None,
                        commission=0.0,
                        error_message=str(e)
                    )
        
        # Should not reach here
        return OrderResult(
            order_id=None,
            status=OrderStatus.REJECTED,
            filled_quantity=0,
            average_fill_price=None,
            commission=0.0,
            error_message="Max retries exceeded"
        )
    
    def place_spread_order(
        self,
        legs: List[OptionLeg],
        order_type: OrderType = OrderType.LIMIT,
        limit_price: Optional[float] = None,
        duration: str = "day"
    ) -> OrderResult:
        """
        Place multi-leg spread order.
        
        Args:
            legs: List of option legs
            order_type: Order type
            limit_price: Net debit/credit limit
            duration: Order duration
            
        Returns:
            OrderResult
        """
        if len(legs) < 2:
            return OrderResult(
                order_id=None,
                status=OrderStatus.REJECTED,
                filled_quantity=0,
                average_fill_price=None,
                commission=0.0,
                error_message="Spread requires at least 2 legs"
            )
        
        # Build multi-leg order data
        order_data = {
            'class': 'multileg',
            'symbol': legs[0].symbol,
            'type': order_type.value,
            'duration': duration
        }
        
        # Add legs
        for i, leg in enumerate(legs):
            order_data[f'option_symbol[{i}]'] = leg.to_tradier_symbol()
            order_data[f'side[{i}]'] = leg.side.value
            order_data[f'quantity[{i}]'] = leg.quantity
        
        # Add price for limit orders
        if order_type == OrderType.LIMIT and limit_price is not None:
            order_data['price'] = f"{limit_price:.2f}"
        
        # Place order (similar retry logic as single-leg)
        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                response = self._make_request(
                    method='POST',
                    endpoint=f'/accounts/{self.account_id}/orders',
                    data=order_data
                )
                
                order_info = response.get('order', {})
                order_id = order_info.get('id')
                status = self._parse_order_status(order_info.get('status', 'pending'))
                
                logger.info(f"Spread order placed: ID={order_id}")
                
                return OrderResult(
                    order_id=str(order_id) if order_id else None,
                    status=status,
                    filled_quantity=0,
                    average_fill_price=None,
                    commission=0.0
                )
            
            except Exception as e:
                if attempt < MAX_RETRY_ATTEMPTS - 1:
                    time.sleep(RETRY_BACKOFF_SECONDS * (2 ** attempt))
                else:
                    return OrderResult(
                        order_id=None,
                        status=OrderStatus.REJECTED,
                        filled_quantity=0,
                        average_fill_price=None,
                        commission=0.0,
                        error_message=str(e)
                    )
    
    def get_order_status(self, order_id: str) -> OrderResult:
        """
        Get current status of an order.
        
        Args:
            order_id: Order ID
            
        Returns:
            OrderResult with current status
        """
        try:
            response = self._make_request(
                method='GET',
                endpoint=f'/accounts/{self.account_id}/orders/{order_id}'
            )
            
            order_info = response.get('order', {})
            status = self._parse_order_status(order_info.get('status', 'pending'))
            
            # Extract fill information
            filled_qty = int(order_info.get('exec_quantity', 0))
            avg_fill_price = float(order_info.get('avg_fill_price', 0.0)) if filled_qty > 0 else None
            commission = float(order_info.get('commission', 0.0))
            
            return OrderResult(
                order_id=order_id,
                status=status,
                filled_quantity=filled_qty,
                average_fill_price=avg_fill_price,
                commission=commission
            )
        
        except Exception as e:
            logger.error(f"Failed to get order status for {order_id}: {e}")
            return OrderResult(
                order_id=order_id,
                status=OrderStatus.PENDING,
                filled_quantity=0,
                average_fill_price=None,
                commission=0.0,
                error_message=str(e)
            )
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.
        
        Args:
            order_id: Order ID
            
        Returns:
            True if canceled successfully
        """
        try:
            response = self._make_request(
                method='DELETE',
                endpoint=f'/accounts/{self.account_id}/orders/{order_id}'
            )
            
            logger.info(f"Order {order_id} canceled")
            return True
        
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    def _parse_order_status(self, status_str: str) -> OrderStatus:
        """Map Tradier status string to OrderStatus enum."""
        status_map = {
            'pending': OrderStatus.PENDING,
            'open': OrderStatus.OPEN,
            'partially_filled': OrderStatus.PARTIALLY_FILLED,
            'filled': OrderStatus.FILLED,
            'canceled': OrderStatus.CANCELED,
            'rejected': OrderStatus.REJECTED,
            'expired': OrderStatus.EXPIRED,
        }
        return status_map.get(status_str.lower(), OrderStatus.PENDING)
    
    def wait_for_fill(
        self,
        order_id: str,
        max_wait_seconds: Optional[int] = None,
        poll_interval: int = 2
    ) -> OrderResult:
        """
        Wait for order to fill or reach terminal state.
        
        Args:
            order_id: Order ID
            max_wait_seconds: Maximum time to wait (defaults to self.timeout)
            poll_interval: Seconds between status checks
            
        Returns:
            Final OrderResult
        """
        max_wait = max_wait_seconds or self.timeout
        start_time = time.time()
        
        while (time.time() - start_time) < max_wait:
            result = self.get_order_status(order_id)
            
            if result.is_complete:
                return result
            
            time.sleep(poll_interval)
        
        # Timeout reached
        logger.warning(f"Order {order_id} wait timeout after {max_wait}s")
        return self.get_order_status(order_id)
    
    def close_position(
        self,
        symbol: str,
        strike: float,
        expiration: str,
        option_type: OptionType,
        quantity: int,
        is_long: bool,
        limit_price: Optional[float] = None
    ) -> OrderResult:
        """
        Close an existing position.
        
        Args:
            symbol: Underlying symbol
            strike: Option strike
            expiration: Expiration date
            option_type: CALL or PUT
            quantity: Number of contracts to close
            is_long: True if closing long position
            limit_price: Limit price (use market order if None)
            
        Returns:
            OrderResult
        """
        # Determine side
        if is_long:
            side = OrderSide.SELL_TO_CLOSE
        else:
            side = OrderSide.BUY_TO_CLOSE
        
        # Determine order type
        order_type = OrderType.LIMIT if limit_price else OrderType.MARKET
        
        logger.info(
            f"Closing position: {symbol} {strike} {option_type.value} x{quantity} "
            f"({side.value})"
        )
        
        return self.place_option_order(
            symbol=symbol,
            strike=strike,
            expiration=expiration,
            option_type=option_type,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price
        )
