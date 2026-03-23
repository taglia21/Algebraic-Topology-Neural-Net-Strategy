"""
Equity order execution on IBKR.

DORMANT by default — all orders are logged but not submitted until
the equities engine is explicitly authorized. This is controlled by
the `enabled` flag.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from ib_async import (
        IB, Contract, Stock, Order as IBOrder, MarketOrder,
        LimitOrder, StopOrder, BracketOrder, Trade,
    )
except ImportError:
    IB = Contract = Stock = IBOrder = MarketOrder = None
    LimitOrder = StopOrder = BracketOrder = Trade = None


@dataclass
class OrderResult:
    """Result of an order placement attempt."""
    order_id: int
    symbol: str
    action: str            # BUY or SELL
    quantity: float
    order_type: str        # MKT, LMT, STP
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: str = "SUBMITTED"   # SUBMITTED, SIMULATED, REJECTED, FILLED
    fill_price: Optional[float] = None
    timestamp: str = ""
    simulated: bool = False


class EquityTrader:
    """
    Stock order execution engine.

    SAFETY: When enabled=False (default), all order methods log the
    intended order but do NOT submit to IBKR. They return an OrderResult
    with status='SIMULATED'. This is the DORMANT mode.
    """

    def __init__(self, client, enabled: bool = False) -> None:
        """
        Args:
            client: IBKRClient instance
            enabled: If False, orders are simulated only (DORMANT mode)
        """
        self._client = client
        self._enabled = enabled
        self._order_counter = 0

        mode = "LIVE" if enabled else "DORMANT (simulated)"
        logger.info("EquityTrader initialized in %s mode", mode)

    @property
    def ib(self) -> IB:
        return self._client.ib

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        """Activate live order execution. Requires explicit authorization."""
        self._enabled = True
        logger.warning("EquityTrader ENABLED — live orders will be submitted to IBKR")

    def disable(self) -> None:
        """Deactivate live order execution. Orders will be simulated."""
        self._enabled = False
        logger.info("EquityTrader DISABLED — orders will be simulated")

    # --- Order Placement ---

    async def place_market_order(
        self, symbol: str, quantity: float, action: str
    ) -> OrderResult:
        """Place a market order. BUY or SELL."""
        return await self._place_order(symbol, quantity, action, "MKT")

    async def place_limit_order(
        self, symbol: str, quantity: float, action: str, limit_price: float
    ) -> OrderResult:
        """Place a limit order."""
        return await self._place_order(
            symbol, quantity, action, "LMT", limit_price=limit_price
        )

    async def place_stop_order(
        self, symbol: str, quantity: float, action: str, stop_price: float
    ) -> OrderResult:
        """Place a stop order."""
        return await self._place_order(
            symbol, quantity, action, "STP", stop_price=stop_price
        )

    async def place_bracket_order(
        self,
        symbol: str,
        quantity: float,
        action: str,
        limit_price: float,
        take_profit: float,
        stop_loss: float,
    ) -> list[OrderResult]:
        """
        Place a bracket order (entry + take profit + stop loss).

        Returns list of 3 OrderResults: [parent, take_profit, stop_loss].
        """
        contract = Stock(symbol, "SMART", "USD") if Stock else None

        if not self._enabled:
            logger.info(
                "[SIMULATED] Bracket %s %g %s @ LMT %.2f | TP=%.2f SL=%.2f",
                action, quantity, symbol, limit_price, take_profit, stop_loss,
            )
            ts = datetime.now().isoformat()
            return [
                OrderResult(self._next_id(), symbol, action, quantity, "LMT",
                            limit_price=limit_price, status="SIMULATED", timestamp=ts, simulated=True),
                OrderResult(self._next_id(), symbol, "SELL" if action == "BUY" else "BUY",
                            quantity, "LMT", limit_price=take_profit, status="SIMULATED", timestamp=ts, simulated=True),
                OrderResult(self._next_id(), symbol, "SELL" if action == "BUY" else "BUY",
                            quantity, "STP", stop_price=stop_loss, status="SIMULATED", timestamp=ts, simulated=True),
            ]

        bracket = BracketOrder(action, quantity, limit_price, take_profit, stop_loss)
        results = []
        submitted_orders = []
        try:
            for order in bracket:
                trade = self.ib.placeOrder(contract, order)
                submitted_orders.append(trade)
                results.append(OrderResult(
                    order_id=trade.order.orderId,
                    symbol=symbol,
                    action=order.action,
                    quantity=quantity,
                    order_type=order.orderType,
                    limit_price=getattr(order, "lmtPrice", None),
                    stop_price=getattr(order, "auxPrice", None),
                    status="SUBMITTED",
                    timestamp=datetime.now().isoformat(),
                ))
        except Exception as partial_err:
            # C-09 fix: If any leg fails, cancel all previously submitted legs
            logger.error(
                "Bracket order partially failed for %s: %s. Cancelling %d submitted legs.",
                symbol, partial_err, len(submitted_orders),
            )
            for prev_trade in submitted_orders:
                try:
                    self.ib.cancelOrder(prev_trade.order)
                except Exception:
                    pass
            raise  # Re-raise so caller knows the bracket failed
        return results

    # --- Order Management ---

    async def cancel_order(self, order_id: int) -> None:
        """Cancel a pending order by ID."""
        if not self._enabled:
            logger.info("[SIMULATED] Cancel order %d", order_id)
            return
        for trade in self.ib.openTrades():
            if trade.order.orderId == order_id:
                self.ib.cancelOrder(trade.order)
                logger.info("Cancelled order %d", order_id)
                return
        logger.warning("Order %d not found in open trades", order_id)

    async def cancel_all_orders(self) -> None:
        """Cancel all pending equity orders."""
        if not self._enabled:
            logger.info("[SIMULATED] Cancel all equity orders")
            return
        self.ib.reqGlobalCancel()
        logger.info("Requested global cancel for all orders")

    async def get_open_orders(self) -> list:
        """Get all open orders."""
        return self.ib.openOrders()

    async def flatten_all(self) -> list[OrderResult]:
        """Close all equity positions with market orders.

        C-07 fix: Temporarily force-enables trading to ensure flatten
        actually executes even if the trader was disabled.
        """
        was_enabled = self._enabled
        self._enabled = True  # Force-enable for emergency flatten
        try:
            positions = await self._client.get_positions()
            results = []
            for pos in positions:
                if pos.contract.secType == "STK" and pos.position != 0:
                    action = "SELL" if pos.position > 0 else "BUY"
                    qty = abs(pos.position)
                    result = await self.place_market_order(
                        pos.contract.symbol, qty, action
                    )
                    results.append(result)
            return results
        finally:
            self._enabled = was_enabled  # Restore original state

    # --- Internal ---

    async def _place_order(
        self,
        symbol: str,
        quantity: float,
        action: str,
        order_type: str,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> OrderResult:
        """Internal order placement with dormant/live branching."""
        ts = datetime.now().isoformat()

        if not self._enabled:
            logger.info(
                "[SIMULATED] %s %s %g %s%s%s",
                order_type,
                action,
                quantity,
                symbol,
                f" @ {limit_price}" if limit_price else "",
                f" stop {stop_price}" if stop_price else "",
            )
            return OrderResult(
                order_id=self._next_id(),
                symbol=symbol,
                action=action,
                quantity=quantity,
                order_type=order_type,
                limit_price=limit_price,
                stop_price=stop_price,
                status="SIMULATED",
                timestamp=ts,
                simulated=True,
            )

        contract = Stock(symbol, "SMART", "USD")
        if order_type == "MKT":
            order = MarketOrder(action, quantity)
        elif order_type == "LMT":
            order = LimitOrder(action, quantity, limit_price)
        elif order_type == "STP":
            order = StopOrder(action, quantity, stop_price)
        else:
            raise ValueError(f"Unknown order type: {order_type}")

        trade = self.ib.placeOrder(contract, order)
        logger.info(
            "LIVE ORDER: %s %s %g %s (id=%d)",
            order_type, action, quantity, symbol, trade.order.orderId,
        )

        return OrderResult(
            order_id=trade.order.orderId,
            symbol=symbol,
            action=action,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            status="SUBMITTED",
            timestamp=ts,
        )

    def _next_id(self) -> int:
        """Generate a simulated order ID."""
        self._order_counter += 1
        return 900000 + self._order_counter
