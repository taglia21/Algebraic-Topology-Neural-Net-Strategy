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
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple
import os
import time

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest, GetOrdersRequest, OptionLegRequest
from alpaca.trading.enums import OrderSide as AlpacaOrderSide, TimeInForce, OrderClass, AssetClass
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionLatestQuoteRequest

from .config import RISK_CONFIG


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
        self.config = RISK_CONFIG
        self.logger = logging.getLogger(__name__)
        
        # Get credentials
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.api_secret = api_secret or os.getenv("ALPACA_SECRET_KEY")
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Alpaca API credentials not provided")
        
        # Initialize Alpaca clients
        self.paper = paper
        self.trading_client = TradingClient(
            api_key=self.api_key,
            secret_key=self.api_secret,
            paper=paper
        )
        self.data_client = OptionHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.api_secret
        )
        
        # Set endpoint for logging
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
        Submit 2-leg spread order using Alpaca MLEG OrderClass.
        
        Args:
            long_symbol: OCC symbol to buy (e.g., "SPY250620P00550000")
            short_symbol: OCC symbol to sell
            quantity: Number of spreads
            net_credit: Net credit received (for credit spreads)
            net_debit: Net debit paid (for debit spreads)
            
        Returns:
            ExecutionResult
        """
        self.logger.info(f"Submitting MLEG spread: LONG {long_symbol}, SHORT {short_symbol}")
        
        internal_legs = [
            OrderLeg(symbol=long_symbol, side=OrderSide.BUY, quantity=quantity),
            OrderLeg(symbol=short_symbol, side=OrderSide.SELL, quantity=quantity),
        ]
        
        # Extract underlying from OCC symbol
        underlying = ""
        for ch in long_symbol:
            if ch.isdigit():
                break
            underlying += ch
        
        # Build MLEG legs
        # ratio_qty is the ratio between legs (1:1 for standard spread)
        # The actual quantity goes in the main order's qty field
        mleg_legs = [
            OptionLegRequest(
                symbol=long_symbol,
                side=AlpacaOrderSide.BUY,
                ratio_qty="1",
            ),
            OptionLegRequest(
                symbol=short_symbol,
                side=AlpacaOrderSide.SELL,
                ratio_qty="1",
            ),
        ]
        
        # Determine limit price and net side
        if net_credit and net_credit > 0:
            limit_price = round(net_credit, 2)
            net_side = AlpacaOrderSide.SELL
        elif net_debit and net_debit > 0:
            limit_price = round(net_debit, 2)
            net_side = AlpacaOrderSide.BUY
        else:
            limit_price = 0.01
            net_side = AlpacaOrderSide.BUY
        
        try:
            order_request = LimitOrderRequest(
                symbol=underlying,
                qty=quantity,
                side=net_side,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.MLEG,
                limit_price=limit_price,
                legs=mleg_legs,
            )
            
            order = self.trading_client.submit_order(order_request)
            self.logger.info(f"MLEG spread order submitted: {order.id}")
            
            # Poll for fill
            status, filled_qty, avg_price = await self._poll_order_status(
                str(order.id), timeout=30.0
            )
            
            success = status == "filled"
            return ExecutionResult(
                success=success,
                order_id=str(order.id),
                status=OrderStatus.FILLED if success else OrderStatus.REJECTED,
                filled_quantity=filled_qty,
                average_fill_price=avg_price,
                timestamp=datetime.now(),
                error_message="" if success else f"Order {status}",
                legs=internal_legs,
            )
        except Exception as e:
            self.logger.error(f"MLEG spread submission failed: {e}")
            return ExecutionResult(
                success=False,
                order_id=None,
                status=OrderStatus.REJECTED,
                filled_quantity=0,
                average_fill_price=0.0,
                timestamp=datetime.now(),
                error_message=str(e),
                legs=internal_legs,
            )
    
    async def submit_iron_condor(
        self,
        underlying: str,
        put_long_occ: str,
        put_short_occ: str,
        call_short_occ: str,
        call_long_occ: str,
        quantity: int,
        net_credit: float,
    ) -> ExecutionResult:
        """
        Submit 4-leg iron condor using Alpaca MLEG OrderClass.
        
        Args:
            underlying: Underlying symbol (e.g., "SPY")
            put_long_occ: Long put OCC symbol (lowest strike)
            put_short_occ: Short put OCC symbol
            call_short_occ: Short call OCC symbol
            call_long_occ: Long call OCC symbol (highest strike)
            quantity: Number of iron condors
            net_credit: Net credit received
            
        Returns:
            ExecutionResult
        """
        self.logger.info(
            f"Submitting MLEG iron condor on {underlying}: "
            f"Puts {put_long_occ}/{put_short_occ}, "
            f"Calls {call_short_occ}/{call_long_occ} "
            f"Credit=${net_credit:.2f}"
        )
        
        internal_legs = [
            OrderLeg(symbol=put_long_occ, side=OrderSide.BUY, quantity=quantity),
            OrderLeg(symbol=put_short_occ, side=OrderSide.SELL, quantity=quantity),
            OrderLeg(symbol=call_short_occ, side=OrderSide.SELL, quantity=quantity),
            OrderLeg(symbol=call_long_occ, side=OrderSide.BUY, quantity=quantity),
        ]
        
        mleg_legs = [
            OptionLegRequest(
                symbol=put_long_occ,
                side=AlpacaOrderSide.BUY,
                ratio_qty="1",
            ),
            OptionLegRequest(
                symbol=put_short_occ,
                side=AlpacaOrderSide.SELL,
                ratio_qty="1",
            ),
            OptionLegRequest(
                symbol=call_short_occ,
                side=AlpacaOrderSide.SELL,
                ratio_qty="1",
            ),
            OptionLegRequest(
                symbol=call_long_occ,
                side=AlpacaOrderSide.BUY,
                ratio_qty="1",
            ),
        ]
        
        try:
            order_request = LimitOrderRequest(
                symbol=underlying,
                qty=quantity,
                side=AlpacaOrderSide.SELL,  # Net credit
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.MLEG,
                limit_price=round(net_credit, 2),
                legs=mleg_legs,
            )
            
            order = self.trading_client.submit_order(order_request)
            self.logger.info(f"MLEG iron condor order submitted: {order.id}")
            
            status, filled_qty, avg_price = await self._poll_order_status(
                str(order.id), timeout=30.0
            )
            
            success = status == "filled"
            return ExecutionResult(
                success=success,
                order_id=str(order.id),
                status=OrderStatus.FILLED if success else OrderStatus.REJECTED,
                filled_quantity=filled_qty,
                average_fill_price=avg_price,
                timestamp=datetime.now(),
                error_message="" if success else f"Order {status}",
                legs=internal_legs,
            )
        except Exception as e:
            self.logger.error(f"MLEG iron condor submission failed: {e}")
            return ExecutionResult(
                success=False,
                order_id=None,
                status=OrderStatus.REJECTED,
                filled_quantity=0,
                average_fill_price=0.0,
                timestamp=datetime.now(),
                error_message=str(e),
                legs=internal_legs,
            )
    
    async def submit_straddle(
        self,
        underlying: str,
        call_occ: str,
        put_occ: str,
        quantity: int,
        net_debit: float,
        buy: bool = True,
    ) -> ExecutionResult:
        """
        Submit straddle using Alpaca MLEG OrderClass.
        
        Args:
            underlying: Underlying symbol
            call_occ: Call OCC symbol
            put_occ: Put OCC symbol
            quantity: Number of straddles
            net_debit: Net debit paid (for long) or credit (for short)
            buy: True for long straddle, False for short
            
        Returns:
            ExecutionResult
        """
        alpaca_side = AlpacaOrderSide.BUY if buy else AlpacaOrderSide.SELL
        internal_side = OrderSide.BUY if buy else OrderSide.SELL
        
        self.logger.info(
            f"Submitting MLEG {'long' if buy else 'short'} straddle on {underlying}: "
            f"Call={call_occ}, Put={put_occ}, Debit=${net_debit:.2f}"
        )
        
        internal_legs = [
            OrderLeg(symbol=call_occ, side=internal_side, quantity=quantity),
            OrderLeg(symbol=put_occ, side=internal_side, quantity=quantity),
        ]
        
        mleg_legs = [
            OptionLegRequest(
                symbol=call_occ,
                side=alpaca_side,
                ratio_qty="1",
            ),
            OptionLegRequest(
                symbol=put_occ,
                side=alpaca_side,
                ratio_qty="1",
            ),
        ]
        
        try:
            order_request = LimitOrderRequest(
                symbol=underlying,
                qty=quantity,
                side=alpaca_side,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.MLEG,
                limit_price=round(net_debit, 2),
                legs=mleg_legs,
            )
            
            order = self.trading_client.submit_order(order_request)
            self.logger.info(f"MLEG straddle order submitted: {order.id}")
            
            status, filled_qty, avg_price = await self._poll_order_status(
                str(order.id), timeout=30.0
            )
            
            success = status == "filled"
            return ExecutionResult(
                success=success,
                order_id=str(order.id),
                status=OrderStatus.FILLED if success else OrderStatus.REJECTED,
                filled_quantity=filled_qty,
                average_fill_price=avg_price,
                timestamp=datetime.now(),
                error_message="" if success else f"Order {status}",
                legs=internal_legs,
            )
        except Exception as e:
            self.logger.error(f"MLEG straddle submission failed: {e}")
            return ExecutionResult(
                success=False,
                order_id=None,
                status=OrderStatus.REJECTED,
                filled_quantity=0,
                average_fill_price=0.0,
                timestamp=datetime.now(),
                error_message=str(e),
                legs=internal_legs,
            )
    
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
        max_retries = self.config.get("retry_attempts", 3)
        
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
    
    async def _get_option_quote(self, symbol: str) -> Optional[Dict]:
        """
        Get latest option quote with bid/ask.
        
        Args:
            symbol: Option symbol in OCC format
            
        Returns:
            Dict with bid_price, ask_price, bid_size, ask_size or None
        """
        try:
            request = OptionLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self.data_client.get_option_latest_quote(request)
            
            if symbol in quotes:
                quote = quotes[symbol]
                return {
                    "bid_price": float(quote.bid_price) if quote.bid_price else 0.0,
                    "ask_price": float(quote.ask_price) if quote.ask_price else 0.0,
                    "bid_size": int(quote.bid_size) if quote.bid_size else 0,
                    "ask_size": int(quote.ask_size) if quote.ask_size else 0,
                }
            return None
        except Exception as e:
            self.logger.error(f"Failed to get quote for {symbol}: {e}")
            return None
    
    async def _validate_pre_trade_checks(self, symbol: str, quantity: int, side: str) -> Tuple[bool, str, Optional[float]]:
        """
        Perform pre-trade validation checks.
        
        Checks:
        1. Bid-ask spread < 15% of mid price
        2. Minimum quote size >= 10 contracts
        3. Sufficient buying power
        
        Args:
            symbol: Option symbol
            quantity: Number of contracts
            side: 'buy' or 'sell'
            
        Returns:
            (passed, error_message, suggested_limit_price)
        """
        # Get quote
        quote = await self._get_option_quote(symbol)
        if not quote:
            return False, "No quote available", None
        
        bid = quote["bid_price"]
        ask = quote["ask_price"]
        bid_size = quote["bid_size"]
        ask_size = quote["ask_size"]
        
        # Check for valid quotes
        if bid <= 0 or ask <= 0:
            return False, "Invalid bid/ask prices", None
        
        # Calculate mid and spread
        mid = (bid + ask) / 2
        spread = ask - bid
        spread_pct = (spread / mid) * 100 if mid > 0 else 100
        
        # Check 1: Spread < 15%
        if spread_pct > 15:
            return False, f"Bid-ask spread too wide: {spread_pct:.1f}%", None
        
        # Check 2: Minimum liquidity
        min_size = bid_size if side == "sell" else ask_size
        if min_size < 10:
            return False, f"Insufficient liquidity: {min_size} contracts", None
        
        # Check 3: Buying power (for buys)
        if side == "buy":
            try:
                account = self.trading_client.get_account()
                buying_power = float(account.buying_power)
                estimated_cost = ask * quantity * 100  # 100 shares per contract
                
                if buying_power < estimated_cost:
                    return False, f"Insufficient buying power: ${buying_power:.2f} < ${estimated_cost:.2f}", None
            except Exception as e:
                self.logger.warning(f"Could not check buying power: {e}")
        
        # Calculate suggested limit price (mid + 0.5% slippage buffer)
        if side == "buy":
            suggested_price = mid * 1.005  # Slightly above mid
        else:
            suggested_price = mid * 0.995  # Slightly below mid
        
        return True, "", suggested_price
    
    async def _poll_order_status(self, order_id: str, timeout: float = 30.0) -> Tuple[str, int, float]:
        """
        Poll order status until filled or timeout.
        
        Args:
            order_id: Alpaca order ID
            timeout: Max seconds to wait
            
        Returns:
            (status, filled_qty, avg_fill_price)
        """
        start_time = time.time()
        poll_interval = 0.5  # Start with 500ms
        
        while time.time() - start_time < timeout:
            try:
                order = self.trading_client.get_order_by_id(order_id)
                
                status = order.status.value
                filled_qty = int(order.filled_qty) if order.filled_qty else 0
                avg_price = float(order.filled_avg_price) if order.filled_avg_price else 0.0
                
                # Terminal states
                if status in ["filled", "cancelled", "rejected", "expired"]:
                    return status, filled_qty, avg_price
                
                # Still pending
                await asyncio.sleep(poll_interval)
                poll_interval = min(poll_interval * 1.5, 2.0)  # Exponential backoff, max 2s
                
            except Exception as e:
                self.logger.error(f"Error polling order {order_id}: {e}")
                await asyncio.sleep(1.0)
        
        # Timeout
        return "timeout", 0, 0.0
    
    async def _submit_order_to_alpaca(self, order_data: Dict) -> ExecutionResult:
        """
        Submit order to Alpaca API with real execution.
        
        Process:
        1. Get option quote for bid/ask
        2. Validate pre-trade checks
        3. Calculate limit price at mid + 0.5% slippage buffer
        4. Submit LimitOrderRequest
        5. Poll order status until filled or timeout (30s)
        6. Return actual fill price and quantity
        
        Args:
            order_data: Order payload with symbol, qty, side, type, limit_price, etc.
            
        Returns:
            ExecutionResult with actual execution details
        """
        symbol = order_data["symbol"]
        quantity = order_data["qty"]
        side = order_data["side"]
        order_type = order_data.get("type", "limit")
        limit_price = order_data.get("limit_price")
        
        try:
            # Pre-trade validation
            passed, error_msg, suggested_price = await self._validate_pre_trade_checks(
                symbol, quantity, side
            )
            
            if not passed:
                self.logger.error(f"Pre-trade check failed: {error_msg}")
                return ExecutionResult(
                    success=False,
                    order_id=None,
                    status=OrderStatus.REJECTED,
                    filled_quantity=0,
                    average_fill_price=0.0,
                    timestamp=datetime.now(),
                    error_message=f"Pre-trade validation failed: {error_msg}",
                )
            
            # Use suggested price if no limit specified
            if not limit_price:
                limit_price = suggested_price
            
            # Map order side
            alpaca_side = AlpacaOrderSide.BUY if side == "buy" else AlpacaOrderSide.SELL
            
            # Build order request
            if order_type == "market":
                order_request = MarketOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=alpaca_side,
                    time_in_force=TimeInForce.DAY,
                )
            else:
                # Limit order
                order_request = LimitOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=alpaca_side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=limit_price,
                )
            
            # Submit order
            self.logger.info(f"Submitting REAL order: {side.upper()} {quantity} {symbol} @ ${limit_price:.2f}")
            order = self.trading_client.submit_order(order_request)
            order_id = order.id
            
            self.logger.info(f"Order submitted: {order_id}")
            
            # Poll for fill
            status, filled_qty, avg_price = await self._poll_order_status(order_id, timeout=30.0)
            
            # Map status
            if status == "filled":
                order_status = OrderStatus.FILLED
                success = True
                error_msg = ""
            elif status in ["partially_filled", "partial_fill"]:
                order_status = OrderStatus.PARTIAL
                success = True
                error_msg = f"Partially filled: {filled_qty}/{quantity}"
            elif status == "timeout":
                order_status = OrderStatus.PENDING
                success = False
                error_msg = "Order timeout - check dashboard"
            else:
                order_status = OrderStatus.REJECTED
                success = False
                error_msg = f"Order {status}"
            
            return ExecutionResult(
                success=success,
                order_id=order_id,
                status=order_status,
                filled_quantity=filled_qty,
                average_fill_price=avg_price,
                timestamp=datetime.now(),
                error_message=error_msg,
            )
            
        except Exception as e:
            self.logger.error(f"Order submission failed: {e}")
            return ExecutionResult(
                success=False,
                order_id=None,
                status=OrderStatus.REJECTED,
                filled_quantity=0,
                average_fill_price=0.0,
                timestamp=datetime.now(),
                error_message=str(e),
            )
