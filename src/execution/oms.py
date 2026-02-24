"""
Phase Q — Order Management System.

Item 13: OrderManagementSystem — full OMS, order states, event bus, persist to JSON.
Item 14: PreTradeRiskChecker — restricted list, max size, daily limit, duplicate/fat finger.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Order State Machine
# ---------------------------------------------------------------------------

class OrderState(str, Enum):
    """Order lifecycle states."""
    PENDING = "PENDING"
    VALIDATED = "VALIDATED"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


@dataclass
class Order:
    """Represents a trading order."""
    order_id: str = ""
    symbol: str = ""
    side: str = "BUY"
    order_type: str = "MARKET"
    quantity: int = 0
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    state: str = "PENDING"
    filled_quantity: int = 0
    filled_avg_price: float = 0.0
    created_at: str = ""
    updated_at: str = ""
    strategy: str = ""
    reason: str = ""


@dataclass
class OrderEvent:
    """Event emitted on order state changes."""
    event_type: str  # "created", "submitted", "filled", "cancelled", "rejected"
    order: Order
    timestamp: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Item 14 — PreTradeRiskChecker
# ---------------------------------------------------------------------------

@dataclass
class PreTradeCheckResult:
    """Result of pre-trade risk check."""
    approved: bool = True
    violations: List[str] = field(default_factory=list)
    checks_passed: int = 0
    checks_total: int = 0


class PreTradeRiskChecker:
    """Pre-trade risk checks before order submission.

    Checks:
      1. Restricted list (banned symbols).
      2. Maximum order size (shares and notional).
      3. Daily order count limit.
      4. Fat finger check (price deviation > 5%).
      5. Duplicate order detection (same symbol/side within cooldown).
    """

    def __init__(
        self,
        restricted_symbols: Optional[Set[str]] = None,
        max_order_shares: int = 10000,
        max_order_notional: float = 500000.0,
        max_daily_orders: int = 200,
        fat_finger_pct: float = 0.05,
        duplicate_cooldown_sec: float = 5.0,
    ):
        self.restricted_symbols = restricted_symbols or set()
        self.max_order_shares = max_order_shares
        self.max_order_notional = max_order_notional
        self.max_daily_orders = max_daily_orders
        self.fat_finger_pct = fat_finger_pct
        self.duplicate_cooldown_sec = duplicate_cooldown_sec

        self._daily_count: int = 0
        self._recent_orders: List[Dict[str, Any]] = []

    def check(
        self,
        order: Order,
        current_price: Optional[float] = None,
    ) -> PreTradeCheckResult:
        """Run all pre-trade checks.

        Args:
            order: Order to validate.
            current_price: Current market price for fat-finger check.

        Returns:
            PreTradeCheckResult with approval and violations.
        """
        violations = []
        checks = 0
        total = 5

        # 1. Restricted list
        checks += 1
        if order.symbol.upper() in self.restricted_symbols:
            violations.append(f"Symbol {order.symbol} is restricted")

        # 2. Max shares
        checks += 1
        if order.quantity > self.max_order_shares:
            violations.append(
                f"Order size {order.quantity} exceeds max {self.max_order_shares}"
            )

        # 3. Max notional
        checks += 1
        if current_price and order.quantity * current_price > self.max_order_notional:
            notional = order.quantity * current_price
            violations.append(
                f"Notional ${notional:,.0f} exceeds max ${self.max_order_notional:,.0f}"
            )

        # 4. Daily limit
        checks += 1
        if self._daily_count >= self.max_daily_orders:
            violations.append(
                f"Daily order count {self._daily_count} at limit {self.max_daily_orders}"
            )

        # 5. Fat finger + duplicate
        checks += 1
        if current_price and order.limit_price:
            deviation = abs(order.limit_price - current_price) / max(current_price, 1e-6)
            if deviation > self.fat_finger_pct:
                violations.append(
                    f"Fat finger: limit ${order.limit_price:.2f} deviates "
                    f"{deviation:.1%} from market ${current_price:.2f}"
                )

        # Duplicate check
        now = time.time()
        for recent in self._recent_orders:
            if (
                recent["symbol"] == order.symbol
                and recent["side"] == order.side
                and now - recent["time"] < self.duplicate_cooldown_sec
            ):
                violations.append(
                    f"Duplicate order: {order.side} {order.symbol} within "
                    f"{self.duplicate_cooldown_sec}s cooldown"
                )
                break

        passed = checks - len(violations)

        if not violations:
            self._daily_count += 1
            self._recent_orders.append({
                "symbol": order.symbol,
                "side": order.side,
                "time": now,
            })
            # Prune old entries
            self._recent_orders = [
                r for r in self._recent_orders
                if now - r["time"] < self.duplicate_cooldown_sec * 10
            ]

        result = PreTradeCheckResult(
            approved=len(violations) == 0,
            violations=violations,
            checks_passed=passed,
            checks_total=total,
        )

        if violations:
            logger.warning("Pre-trade REJECTED: %s", "; ".join(violations))
        else:
            logger.debug("Pre-trade APPROVED for %s %s %d", order.side, order.symbol, order.quantity)

        return result

    def reset_daily(self) -> None:
        """Reset daily counters (call at market open)."""
        self._daily_count = 0
        self._recent_orders.clear()


# ---------------------------------------------------------------------------
# Item 13 — OrderManagementSystem
# ---------------------------------------------------------------------------

class OrderManagementSystem:
    """Full Order Management System with state machine, event bus, and persistence.

    Order lifecycle:
      PENDING → VALIDATED → SUBMITTED → PARTIALLY_FILLED → FILLED
                                      → CANCELLED
      PENDING → REJECTED (if pre-trade check fails)

    Event bus: subscribe to order events with callbacks.
    Persistence: save/load order book to JSON.
    """

    def __init__(
        self,
        pre_trade_checker: Optional[PreTradeRiskChecker] = None,
        persist_path: str = "state/oms_orders.json",
    ):
        self.pre_trade_checker = pre_trade_checker or PreTradeRiskChecker()
        self.persist_path = persist_path
        self._orders: Dict[str, Order] = {}
        self._event_listeners: Dict[str, List[Callable]] = {}
        self._fill_history: List[Dict[str, Any]] = []

    # --- Order Creation ---

    def create_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str = "MARKET",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        strategy: str = "",
        current_price: Optional[float] = None,
    ) -> Order:
        """Create a new order and run pre-trade checks.

        Returns:
            Order in VALIDATED or REJECTED state.
        """
        order = Order(
            order_id=str(uuid.uuid4())[:12],
            symbol=symbol.upper(),
            side=side.upper(),
            order_type=order_type.upper(),
            quantity=quantity,
            limit_price=limit_price,
            stop_price=stop_price,
            state=OrderState.PENDING.value,
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
            strategy=strategy,
        )

        # Pre-trade risk checks
        check = self.pre_trade_checker.check(order, current_price)
        if check.approved:
            order.state = OrderState.VALIDATED.value
            self._orders[order.order_id] = order
            self._emit_event("created", order)
            logger.info("Order created: %s %s %d %s", side, symbol, quantity, order.order_id)
        else:
            order.state = OrderState.REJECTED.value
            order.reason = "; ".join(check.violations)
            self._orders[order.order_id] = order
            self._emit_event("rejected", order, {"violations": check.violations})
            logger.warning("Order rejected: %s — %s", order.order_id, order.reason)

        return order

    # --- State Transitions ---

    def submit_order(self, order_id: str) -> Optional[Order]:
        """Move order from VALIDATED to SUBMITTED."""
        order = self._orders.get(order_id)
        if not order or order.state != OrderState.VALIDATED.value:
            return None
        order.state = OrderState.SUBMITTED.value
        order.updated_at = datetime.now(timezone.utc).isoformat()
        self._emit_event("submitted", order)
        return order

    def record_fill(
        self,
        order_id: str,
        quantity: int,
        price: float,
    ) -> Optional[Order]:
        """Record a fill (partial or complete).

        Args:
            order_id: Order ID to fill.
            quantity: Shares filled in this execution.
            price: Fill price.

        Returns:
            Updated Order or None if not found.
        """
        order = self._orders.get(order_id)
        if not order:
            return None
        if order.state not in (
            OrderState.SUBMITTED.value,
            OrderState.PARTIALLY_FILLED.value,
        ):
            return None

        # Update fill info
        prev_qty = order.filled_quantity
        prev_cost = order.filled_avg_price * prev_qty
        new_qty = prev_qty + quantity
        new_avg = (prev_cost + price * quantity) / max(new_qty, 1)

        order.filled_quantity = new_qty
        order.filled_avg_price = round(new_avg, 4)
        order.updated_at = datetime.now(timezone.utc).isoformat()

        if new_qty >= order.quantity:
            order.state = OrderState.FILLED.value
            self._emit_event("filled", order, {"fill_price": price, "fill_qty": quantity})
        else:
            order.state = OrderState.PARTIALLY_FILLED.value
            self._emit_event("partial_fill", order, {"fill_price": price, "fill_qty": quantity})

        self._fill_history.append({
            "order_id": order_id,
            "quantity": quantity,
            "price": price,
            "timestamp": order.updated_at,
        })

        return order

    def cancel_order(self, order_id: str, reason: str = "") -> Optional[Order]:
        """Cancel an order."""
        order = self._orders.get(order_id)
        if not order:
            return None
        if order.state in (OrderState.FILLED.value, OrderState.CANCELLED.value):
            return None
        order.state = OrderState.CANCELLED.value
        order.reason = reason
        order.updated_at = datetime.now(timezone.utc).isoformat()
        self._emit_event("cancelled", order, {"reason": reason})
        return order

    # --- Event Bus ---

    def subscribe(self, event_type: str, callback: Callable) -> None:
        """Subscribe to order events.

        Args:
            event_type: One of "created", "submitted", "filled", "partial_fill",
                       "cancelled", "rejected".
            callback: Function(OrderEvent) to call.
        """
        if event_type not in self._event_listeners:
            self._event_listeners[event_type] = []
        self._event_listeners[event_type].append(callback)

    def _emit_event(
        self,
        event_type: str,
        order: Order,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit an event to all subscribers."""
        event = OrderEvent(
            event_type=event_type,
            order=order,
            timestamp=datetime.now(timezone.utc).isoformat(),
            details=details or {},
        )
        for listener in self._event_listeners.get(event_type, []):
            try:
                listener(event)
            except Exception as e:
                logger.error("Event listener error: %s", e)

    # --- Query ---

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self._orders.get(order_id)

    def get_open_orders(self) -> List[Order]:
        """Get all non-terminal orders."""
        open_states = {
            OrderState.PENDING.value,
            OrderState.VALIDATED.value,
            OrderState.SUBMITTED.value,
            OrderState.PARTIALLY_FILLED.value,
        }
        return [o for o in self._orders.values() if o.state in open_states]

    def get_all_orders(self) -> List[Order]:
        """Get all orders."""
        return list(self._orders.values())

    @property
    def fill_history(self) -> List[Dict[str, Any]]:
        return self._fill_history

    # --- Persistence ---

    def save(self) -> None:
        """Persist order book to JSON."""
        path = Path(self.persist_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "orders": {oid: asdict(o) for oid, o in self._orders.items()},
            "fill_history": self._fill_history,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("OMS state saved to %s (%d orders)", self.persist_path, len(self._orders))

    def load(self) -> int:
        """Load order book from JSON. Returns number of orders loaded."""
        path = Path(self.persist_path)
        if not path.exists():
            return 0
        with open(path) as f:
            data = json.load(f)
        for oid, odata in data.get("orders", {}).items():
            self._orders[oid] = Order(**{
                k: v for k, v in odata.items() if k in Order.__dataclass_fields__
            })
        self._fill_history = data.get("fill_history", [])
        logger.info("OMS loaded %d orders from %s", len(self._orders), self.persist_path)
        return len(self._orders)
