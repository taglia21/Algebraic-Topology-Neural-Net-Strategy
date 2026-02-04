"""
Alpaca Options Trade Executor
==============================

Executes options trades via Alpaca API with:
- Single-leg orders (calls, puts)
- Multi-leg spreads (credit spreads, debit spreads)
- Complex strategies (iron condors, straddles)
- Bracket orders with stop-loss and take-profit
- Retry logic and error handling
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple
import os

from .config import get_config


# ============================================================================
# DATA MODELS
# ============================================================================

class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type."""
    LIMIT = "limit"
    MARKET = "market"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


@dataclass
class OrderLeg:
    """Single leg of an order."""
    symbol: str  # Option symbol
    side: OrderSide
    quantity: int
    limit_price: Optional[float] = None


@dataclass
class ExecutionResult:
    """Result of order execution."""
    success: bool
    order_id: Optional[str]
    status: OrderStatus
    filled_quantity: int
    average_fill_price: float
    timestamp: datetime
    error_message: str = ""
    legs: List[OrderLeg] = None


# ============================================================================
# ALPACA OPTIONS EXECUTOR
# ============================================================================

class AlpacaOptionsExecutor:
    """
    Execute options trades via Alpaca API.
    
    Features:
    - Submit single-leg and multi-leg orders
    - Bracket orders with stops/targets
    - Limit orders at mid-price
    - Retry logic for transient failures
    - Comprehensive logging
    """
    
    def __init__(self, api_key: str = None, api_secret: str = None, paper: bool = True):
        """
        Initialize executor.
        
        Args:
            api_key: Alpaca API key (or from env)
            api_secret: Alpaca API secret (or from env)
            paper: Use paper trading endpoint
        """
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # Get credentials
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.api_secret = api_secret or os.getenv("ALPACA_API_SECRET")
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Alpaca API credentials not provided")
        
        # Set endpoint
        if paper:
            self.base_url = "https://paper-api.alpaca.markets"
        else:
            self.base_url = "https://api.alpaca.markets"
        
        self.logger.info(f"Initialized executor (paper={paper})")
    
    async def submit_single_leg_order(
        self,
        option_symbol: str,
        side: OrderSide,
        quantity: int,
        limit_price: Optional[float] = None,
        with_bracket: bool = True,
    ) -> ExecutionResult:
        """
        Submit single-leg option order.
        
        Args:
            option_symbol: Option symbol (OCC format)
            side: BUY or SELL
            quantity: Number of contracts
            limit_price: Limit price (None for market)
            with_bracket: Add stop-loss and take-profit
            
        Returns:
            ExecutionResult
        """
        self.logger.info(
            f"Submitting {side.value} {quantity} {option_symbol} @ {limit_price or 'market'}"
        )
        
        leg = OrderLeg(
            symbol=option_symbol,
            side=side,
            quantity=quantity,
            limit_price=limit_price,
        )
        
        # Build order payload
        order_data = {
            "symbol": option_symbol,
            "qty": quantity,
            "side": side.value,
            "type": "limit" if limit_price else "market",
            "time_in_force": "day",
        }
        
        if limit_price:
            order_data["limit_price"] = limit_price
        
        # Add bracket orders if requested
        if with_bracket and limit_price:
            # Calculate stop-loss and take-profit prices
            if side == OrderSide.BUY:
                # Buying: Stop if price drops, profit if price rises
                stop_loss = limit_price * (1 - self.config["stop_loss_pct"])
                take_profit = limit_price * (1 + self.config["target_profit_pct"])
            else:
                # Selling: Stop if price rises, profit if price drops
                stop_loss = limit_price * (1 + self.config["stop_loss_pct"])
                take_profit = limit_price * (1 - self.config["target_profit_pct"])
            
            order_data["order_class"] = "bracket"
            order_data["take_profit"] = {"limit_price": take_profit}
            order_data["stop_loss"] = {"stop_price": stop_loss}
        
        # Execute with retry
        result = await self._execute_with_retry(order_data, [leg])
        
        return result
    
    async def submit_spread_order(
        self,
        long_symbol: str,
        short_symbol: str,
        quantity: int,
        net_credit: float = None,
        net_debit: float = None,
    ) -> ExecutionResult:
        """
        Submit 2-leg spread order (credit or debit spread).
        
        Args:
            long_symbol: Option to buy
            short_symbol: Option to sell
            quantity: Number of spreads
            net_credit: Net credit received (for credit spreads)
            net_debit: Net debit paid (for debit spreads)
            
        Returns:
            ExecutionResult
        """
        self.logger.info(f"Submitting spread: LONG {long_symbol}, SHORT {short_symbol}")
        
        legs = [
            OrderLeg(symbol=long_symbol, side=OrderSide.BUY, quantity=quantity),
            OrderLeg(symbol=short_symbol, side=OrderSide.SELL, quantity=quantity),
        ]
        
        # Multi-leg order
        order_data = {
            "symbol": long_symbol.split()[0],  # Underlying symbol
            "qty": quantity,
            "side": "buy",  # Net position side
            "type": "limit",
            "time_in_force": "day",
            "order_class": "oto",  # One-triggers-other
            "legs": [
                {
                    "symbol": long_symbol,
                    "side": "buy",
                    "qty": quantity,
                },
                {
                    "symbol": short_symbol,
                    "side": "sell",
                    "qty": quantity,
                },
            ],
        }
        
        # Set limit price based on credit/debit
        if net_credit:
            order_data["limit_price"] = net_credit
        elif net_debit:
            order_data["limit_price"] = net_debit
        
        result = await self._execute_with_retry(order_data, legs)
        
        return result
    
    async def submit_iron_condor(
        self,
        underlying: str,
        put_buy_strike: float,
        put_sell_strike: float,
        call_sell_strike: float,
        call_buy_strike: float,
        quantity: int,
        net_credit: float,
    ) -> ExecutionResult:
        """
        Submit 4-leg iron condor order.
        
        Args:
            underlying: Underlying symbol (e.g., "SPY")
            put_buy_strike: Long put strike
            put_sell_strike: Short put strike
            call_sell_strike: Short call strike
            call_buy_strike: Long call strike
            quantity: Number of iron condors
            net_credit: Net credit received
            
        Returns:
            ExecutionResult
        """
        self.logger.info(
            f"Submitting iron condor on {underlying}: "
            f"Puts {put_buy_strike}/{put_sell_strike}, "
            f"Calls {call_sell_strike}/{call_buy_strike}"
        )
        
        # NOTE: This is simplified - actual OCC symbols would need to be constructed
        # based on expiration, strike, and option type
        
        legs = [
            OrderLeg(symbol=f"{underlying}_PUT_{put_buy_strike}", side=OrderSide.BUY, quantity=quantity),
            OrderLeg(symbol=f"{underlying}_PUT_{put_sell_strike}", side=OrderSide.SELL, quantity=quantity),
            OrderLeg(symbol=f"{underlying}_CALL_{call_sell_strike}", side=OrderSide.SELL, quantity=quantity),
            OrderLeg(symbol=f"{underlying}_CALL_{call_buy_strike}", side=OrderSide.BUY, quantity=quantity),
        ]
        
        order_data = {
            "symbol": underlying,
            "qty": quantity,
            "side": "buy",
            "type": "limit",
            "limit_price": net_credit,
            "time_in_force": "day",
            "order_class": "oto",
            "legs": [
                {"symbol": leg.symbol, "side": leg.side.value, "qty": leg.quantity}
                for leg in legs
            ],
        }
        
        result = await self._execute_with_retry(order_data, legs)
        
        return result
    
    async def submit_straddle(
        self,
        underlying: str,
        strike: float,
        quantity: int,
        net_debit: float,
        buy: bool = True,
    ) -> ExecutionResult:
        """
        Submit straddle order (buy or sell call + put at same strike).
        
        Args:
            underlying: Underlying symbol
            strike: Strike price
            quantity: Number of straddles
            net_debit: Net debit paid (for long) or credit (for short)
            buy: True for long straddle, False for short
            
        Returns:
            ExecutionResult
        """
        side = OrderSide.BUY if buy else OrderSide.SELL
        
        self.logger.info(
            f"Submitting {'long' if buy else 'short'} straddle on {underlying} @ {strike}"
        )
        
        legs = [
            OrderLeg(symbol=f"{underlying}_CALL_{strike}", side=side, quantity=quantity),
            OrderLeg(symbol=f"{underlying}_PUT_{strike}", side=side, quantity=quantity),
        ]
        
        order_data = {
            "symbol": underlying,
            "qty": quantity,
            "side": side.value,
            "type": "limit",
            "limit_price": net_debit,
            "time_in_force": "day",
            "order_class": "oto",
            "legs": [
                {"symbol": leg.symbol, "side": leg.side.value, "qty": leg.quantity}
                for leg in legs
            ],
        }
        
        result = await self._execute_with_retry(order_data, legs)
        
        return result
    
    async def _execute_with_retry(
        self,
        order_data: Dict,
        legs: List[OrderLeg],
    ) -> ExecutionResult:
        """
        Execute order with retry logic.
        
        Args:
            order_data: Order payload for Alpaca API
            legs: Order legs for logging
            
        Returns:
            ExecutionResult
        """
        max_retries = self.config["order_retry_attempts"]
        
        for attempt in range(max_retries):
            try:
                # Simulate API call (replace with actual Alpaca API call)
                result = await self._submit_order_to_alpaca(order_data)
                
                if result.success:
                    self.logger.info(
                        f"Order filled: {result.order_id} @ {result.average_fill_price}"
                    )
                    return result
                else:
                    self.logger.warning(f"Order attempt {attempt + 1} failed: {result.error_message}")
                    
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
            
            except Exception as e:
                self.logger.error(f"Order execution error (attempt {attempt + 1}): {e}")
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return ExecutionResult(
                        success=False,
                        order_id=None,
                        status=OrderStatus.REJECTED,
                        filled_quantity=0,
                        average_fill_price=0.0,
                        timestamp=datetime.now(),
                        error_message=str(e),
                        legs=legs,
                    )
        
        # Max retries exceeded
        return ExecutionResult(
            success=False,
            order_id=None,
            status=OrderStatus.REJECTED,
            filled_quantity=0,
            average_fill_price=0.0,
            timestamp=datetime.now(),
            error_message="Max retries exceeded",
            legs=legs,
        )
    
    async def _submit_order_to_alpaca(self, order_data: Dict) -> ExecutionResult:
        """
        Submit order to Alpaca API.
        
        This is a mock implementation. In production, this would:
        1. Make HTTP POST to /v2/orders
        2. Poll order status until filled
        3. Return actual execution details
        
        Args:
            order_data: Order payload
            
        Returns:
            ExecutionResult
        """
        # TODO: Replace with actual Alpaca API call
        # For now, simulate successful execution
        
        import random
        
        await asyncio.sleep(0.5)  # Simulate network delay
        
        # Simulate 95% success rate
        if random.random() < 0.95:
            return ExecutionResult(
                success=True,
                order_id=f"ORDER_{datetime.now().timestamp()}",
                status=OrderStatus.FILLED,
                filled_quantity=order_data.get("qty", 1),
                average_fill_price=order_data.get("limit_price", 1.0),
                timestamp=datetime.now(),
                error_message="",
            )
        else:
            return ExecutionResult(
                success=False,
                order_id=None,
                status=OrderStatus.REJECTED,
                filled_quantity=0,
                average_fill_price=0.0,
                timestamp=datetime.now(),
                error_message="Simulated rejection",
            )
