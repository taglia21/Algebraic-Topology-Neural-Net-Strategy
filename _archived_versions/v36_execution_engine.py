#!/usr/bin/env python3
"""
V36 Execution Engine
====================
Smart order routing with TWAP, VWAP, and limit order strategies.

Features:
- TWAP: Time-weighted slicing for even distribution
- VWAP: Volume-weighted slicing based on intraday profile
- Smart limit orders with price improvement and market fallback
- Execution quality tracking (arrival price vs fill price)
- Alpaca trading API integration

Usage:
    engine = ExecutionEngine()
    orders = await engine.split_order_twap('AAPL', 1000, duration_minutes=30, num_slices=6)
    await engine.execute_orders(orders)
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any
from zoneinfo import ZoneInfo

import aiohttp

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('V36_Execution')

EST = ZoneInfo("America/New_York")


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIAL = "partial_fill"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class OrderSlice:
    """A single order slice for TWAP/VWAP execution."""
    symbol: str
    shares: int
    side: OrderSide
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    scheduled_time: Optional[datetime] = None
    weight: float = 1.0  # For VWAP weighting
    
    # Execution tracking
    status: OrderStatus = OrderStatus.PENDING
    order_id: Optional[str] = None
    fill_price: Optional[float] = None
    filled_shares: int = 0
    filled_at: Optional[datetime] = None


@dataclass
class ExecutionQuality:
    """Execution quality metrics."""
    symbol: str
    total_shares: int
    arrival_price: float
    avg_fill_price: float
    slippage_bps: float
    execution_time_seconds: float
    fill_rate: float  # Percentage filled


@dataclass
class ExecutionConfig:
    """Configuration for execution engine."""
    alpaca_api_key: str = field(default_factory=lambda: os.getenv('ALPACA_API_KEY', ''))
    alpaca_secret_key: str = field(default_factory=lambda: os.getenv('ALPACA_SECRET_KEY', ''))
    alpaca_base_url: str = field(
        default_factory=lambda: os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    )
    limit_timeout_seconds: int = 30  # Fallback to market after this
    default_improvement_bps: int = 2  # Default price improvement for limits


# Default intraday volume profile (hourly buckets, 9:30-16:00)
DEFAULT_VOLUME_PROFILE = [
    0.12,  # 9:30-10:30 - High open volume
    0.08,  # 10:30-11:30
    0.07,  # 11:30-12:30 - Lunch lull
    0.07,  # 12:30-13:30
    0.08,  # 13:30-14:30
    0.10,  # 14:30-15:30
    0.15,  # 15:30-16:00 - High close volume (30 min bucket)
]


class AlpacaTradingClient:
    """Async Alpaca trading API client."""

    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure HTTP session exists."""
        if self.session is None or self.session.closed:
            headers = {
                'APCA-API-KEY-ID': self.config.alpaca_api_key,
                'APCA-API-SECRET-KEY': self.config.alpaca_secret_key,
            }
            self.session = aiohttp.ClientSession(headers=headers)
        return self.session

    async def close(self) -> None:
        """Close the HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()

    async def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest quote for a symbol."""
        session = await self._ensure_session()
        url = f"https://data.alpaca.markets/v2/stocks/{symbol}/quotes/latest"
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get('quote', {})
        except aiohttp.ClientError as e:
            logger.error(f"Quote fetch error for {symbol}: {e}")
        return None

    async def submit_order(
        self, symbol: str, shares: int, side: OrderSide,
        order_type: OrderType, limit_price: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Submit an order to Alpaca."""
        session = await self._ensure_session()
        url = f"{self.config.alpaca_base_url}/v2/orders"
        
        payload = {
            'symbol': symbol,
            'qty': str(abs(shares)),
            'side': side.value,
            'type': order_type.value,
            'time_in_force': 'day'
        }
        if order_type == OrderType.LIMIT and limit_price:
            payload['limit_price'] = str(round(limit_price, 2))
        
        try:
            async with session.post(url, json=payload) as resp:
                if resp.status in (200, 201):
                    return await resp.json()
                else:
                    error = await resp.text()
                    logger.error(f"Order submit failed: {resp.status} - {error}")
        except aiohttp.ClientError as e:
            logger.error(f"Order submit error: {e}")
        return None

    async def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get order status."""
        session = await self._ensure_session()
        url = f"{self.config.alpaca_base_url}/v2/orders/{order_id}"
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    return await resp.json()
        except aiohttp.ClientError as e:
            logger.error(f"Order fetch error: {e}")
        return None

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        session = await self._ensure_session()
        url = f"{self.config.alpaca_base_url}/v2/orders/{order_id}"
        try:
            async with session.delete(url) as resp:
                return resp.status in (200, 204)
        except aiohttp.ClientError:
            return False


class ExecutionEngine:
    """
    Smart order routing engine with TWAP, VWAP, and limit strategies.
    
    Args:
        config: Execution configuration
    
    Example:
        engine = ExecutionEngine()
        orders = await engine.split_order_twap('AAPL', 1000, 30, 6)
        quality = await engine.execute_orders(orders)
    """

    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()
        self.client = AlpacaTradingClient(self.config)
        self._execution_log: List[ExecutionQuality] = []

    async def close(self) -> None:
        """Close client connections."""
        await self.client.close()

    async def _get_arrival_price(self, symbol: str, side: OrderSide) -> float:
        """Get arrival price (mid-quote at decision time)."""
        quote = await self.client.get_quote(symbol)
        if quote:
            bid = quote.get('bp', 0)
            ask = quote.get('ap', 0)
            if bid and ask:
                return (bid + ask) / 2
            return ask if side == OrderSide.BUY else bid
        return 0.0

    def split_order_twap(
        self, symbol: str, shares: int, duration_minutes: int, num_slices: int
    ) -> List[OrderSlice]:
        """
        Split order using TWAP (Time-Weighted Average Price) strategy.
        
        Args:
            symbol: Stock symbol
            shares: Total shares (positive=buy, negative=sell)
            duration_minutes: Total execution window
            num_slices: Number of order slices
        
        Returns:
            List of OrderSlice objects with scheduled times
        """
        side = OrderSide.BUY if shares > 0 else OrderSide.SELL
        abs_shares = abs(shares)
        base_size = abs_shares // num_slices
        remainder = abs_shares % num_slices
        
        interval = timedelta(minutes=duration_minutes / num_slices)
        now = datetime.now(EST)
        
        slices = []
        for i in range(num_slices):
            slice_shares = base_size + (1 if i < remainder else 0)
            if slice_shares > 0:
                slices.append(OrderSlice(
                    symbol=symbol,
                    shares=slice_shares,
                    side=side,
                    scheduled_time=now + (interval * i),
                    weight=1.0 / num_slices
                ))
        
        logger.info(f"TWAP: {symbol} {shares} shares -> {len(slices)} slices over {duration_minutes}min")
        return slices

    def split_order_vwap(
        self, symbol: str, shares: int, volume_profile: Optional[List[float]] = None
    ) -> List[OrderSlice]:
        """
        Split order using VWAP (Volume-Weighted Average Price) strategy.
        
        Args:
            symbol: Stock symbol
            shares: Total shares
            volume_profile: Hourly volume weights (defaults to typical profile)
        
        Returns:
            List of OrderSlice objects weighted by volume
        """
        profile = volume_profile or DEFAULT_VOLUME_PROFILE
        profile_sum = sum(profile)
        normalized = [p / profile_sum for p in profile]
        
        side = OrderSide.BUY if shares > 0 else OrderSide.SELL
        abs_shares = abs(shares)
        
        slices = []
        allocated = 0
        for i, weight in enumerate(normalized):
            slice_shares = int(abs_shares * weight)
            if i == len(normalized) - 1:
                slice_shares = abs_shares - allocated  # Ensure exact total
            
            if slice_shares > 0:
                slices.append(OrderSlice(
                    symbol=symbol,
                    shares=slice_shares,
                    side=side,
                    weight=weight
                ))
                allocated += slice_shares
        
        logger.info(f"VWAP: {symbol} {shares} shares -> {len(slices)} weighted slices")
        return slices

    async def place_limit_with_improvement(
        self, symbol: str, shares: int, side: OrderSide, improvement_bps: int = 2
    ) -> OrderSlice:
        """
        Place limit order with price improvement, fallback to market on timeout.
        
        Args:
            symbol: Stock symbol
            shares: Number of shares
            side: Buy or sell
            improvement_bps: Basis points improvement from NBBO
        
        Returns:
            OrderSlice with execution result
        """
        quote = await self.client.get_quote(symbol)
        if not quote:
            logger.warning(f"No quote for {symbol}, using market order")
            return OrderSlice(symbol=symbol, shares=shares, side=side, order_type=OrderType.MARKET)
        
        bid, ask = quote.get('bp', 0), quote.get('ap', 0)
        improvement = improvement_bps / 10000
        
        if side == OrderSide.BUY:
            limit_price = ask * (1 - improvement)
        else:
            limit_price = bid * (1 + improvement)
        
        order_slice = OrderSlice(
            symbol=symbol,
            shares=shares,
            side=side,
            order_type=OrderType.LIMIT,
            limit_price=round(limit_price, 2)
        )
        
        logger.info(f"Limit order: {side.value} {shares} {symbol} @ ${limit_price:.2f}")
        return order_slice

    async def execute_orders(self, orders: List[OrderSlice]) -> ExecutionQuality:
        """
        Execute a list of order slices and track quality.
        
        Args:
            orders: List of OrderSlice to execute
        
        Returns:
            ExecutionQuality metrics
        """
        if not orders:
            raise ValueError("No orders to execute")
        
        symbol = orders[0].symbol
        side = orders[0].side
        total_shares = sum(o.shares for o in orders)
        arrival_price = await self._get_arrival_price(symbol, side)
        start_time = datetime.now(EST)
        
        filled_shares = 0
        total_value = 0.0
        
        for order in orders:
            if order.scheduled_time:
                wait_seconds = (order.scheduled_time - datetime.now(EST)).total_seconds()
                if wait_seconds > 0:
                    await asyncio.sleep(wait_seconds)
            
            result = await self.client.submit_order(
                order.symbol, order.shares, order.side,
                order.order_type, order.limit_price
            )
            
            if result:
                order.order_id = result.get('id')
                order.status = OrderStatus.SUBMITTED
                
                # Wait for fill or timeout
                filled = await self._wait_for_fill(order)
                if filled:
                    filled_shares += order.filled_shares
                    if order.fill_price:
                        total_value += order.filled_shares * order.fill_price
        
        avg_fill = total_value / filled_shares if filled_shares > 0 else 0
        slippage = ((avg_fill - arrival_price) / arrival_price * 10000) if arrival_price else 0
        if side == OrderSide.SELL:
            slippage = -slippage
        
        quality = ExecutionQuality(
            symbol=symbol,
            total_shares=total_shares,
            arrival_price=arrival_price,
            avg_fill_price=avg_fill,
            slippage_bps=slippage,
            execution_time_seconds=(datetime.now(EST) - start_time).total_seconds(),
            fill_rate=filled_shares / total_shares if total_shares > 0 else 0
        )
        
        self._execution_log.append(quality)
        logger.info(f"Execution complete: {symbol} filled {filled_shares}/{total_shares}, slippage={slippage:.1f}bps")
        return quality

    async def _wait_for_fill(self, order: OrderSlice, max_wait: int = 30) -> bool:
        """Wait for order fill with market fallback."""
        if not order.order_id:
            return False
        
        for _ in range(max_wait):
            status = await self.client.get_order(order.order_id)
            if status:
                if status.get('status') == 'filled':
                    order.status = OrderStatus.FILLED
                    order.fill_price = float(status.get('filled_avg_price', 0))
                    order.filled_shares = int(float(status.get('filled_qty', 0)))
                    return True
            await asyncio.sleep(1)
        
        # Timeout - cancel and submit market order
        if order.order_type == OrderType.LIMIT:
            await self.client.cancel_order(order.order_id)
            logger.warning(f"Limit timeout, converting to market: {order.symbol}")
            result = await self.client.submit_order(
                order.symbol, order.shares, order.side, OrderType.MARKET
            )
            if result:
                order.order_id = result.get('id')
                return await self._wait_for_fill(order, max_wait=10)
        
        return False

    def get_execution_log(self) -> List[ExecutionQuality]:
        """Get execution quality log."""
        return self._execution_log.copy()


async def main() -> None:
    """Example usage of ExecutionEngine."""
    engine = ExecutionEngine()
    
    # TWAP example
    twap_orders = engine.split_order_twap('AAPL', 1000, duration_minutes=30, num_slices=6)
    print(f"TWAP slices: {len(twap_orders)}")
    for o in twap_orders:
        print(f"  {o.shares} shares @ {o.scheduled_time}")
    
    # VWAP example
    vwap_orders = engine.split_order_vwap('MSFT', 500)
    print(f"\nVWAP slices: {len(vwap_orders)}")
    for o in vwap_orders:
        print(f"  {o.shares} shares (weight={o.weight:.2%})")
    
    await engine.close()


if __name__ == "__main__":
    asyncio.run(main())
