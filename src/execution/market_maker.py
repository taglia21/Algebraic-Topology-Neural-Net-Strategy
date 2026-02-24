"""
Phase N — Market Making & Adverse Selection.

Item 4: AvellanedaStoikovMarketMaker — A-S model, reservation price, optimal spread.
Item 6: InventoryManager — mean-reversion pressure, max_inventory=500, 2x hard stop.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Item 6 — InventoryManager
# ---------------------------------------------------------------------------

@dataclass
class InventoryState:
    """Current inventory state."""
    position: int = 0
    max_inventory: int = 500
    utilization: float = 0.0  # |position| / max_inventory
    skew: float = 0.0  # mean-reversion pressure
    is_hard_stop: bool = False  # |position| >= 2 * max_inventory


class InventoryManager:
    """Manage market-maker inventory with mean-reversion pressure.

    - max_inventory: soft limit, skew quotes beyond this.
    - 2x max_inventory: HARD STOP, flatten immediately.
    - Skew = gamma * position (pushes quotes to reduce inventory).
    """

    def __init__(
        self,
        max_inventory: int = 500,
        gamma: float = 0.01,
    ):
        self.max_inventory = max_inventory
        self.gamma = gamma
        self._position: int = 0

    def update(self, fill_qty: int) -> InventoryState:
        """Update inventory after a fill.

        Args:
            fill_qty: Positive for buy, negative for sell.

        Returns:
            Updated InventoryState.
        """
        self._position += fill_qty
        return self.state()

    def state(self) -> InventoryState:
        """Get current inventory state."""
        abs_pos = abs(self._position)
        utilization = abs_pos / max(self.max_inventory, 1)
        skew = self.gamma * self._position
        is_hard_stop = abs_pos >= 2 * self.max_inventory

        if is_hard_stop:
            logger.critical(
                "INVENTORY HARD STOP: position=%d (limit=%d)",
                self._position, 2 * self.max_inventory,
            )

        return InventoryState(
            position=self._position,
            max_inventory=self.max_inventory,
            utilization=utilization,
            skew=skew,
            is_hard_stop=is_hard_stop,
        )

    def get_skew(self) -> float:
        """Return quote skew based on inventory.

        Positive position → negative skew (lower ask to sell).
        Negative position → positive skew (raise bid to buy).
        """
        return -self.gamma * self._position

    def should_flatten(self) -> bool:
        """Return True if at hard stop (2x max_inventory)."""
        return abs(self._position) >= 2 * self.max_inventory

    def reset(self) -> None:
        """Reset inventory to zero."""
        self._position = 0

    @property
    def position(self) -> int:
        return self._position


# ---------------------------------------------------------------------------
# Item 4 — AvellanedaStoikovMarketMaker
# ---------------------------------------------------------------------------

@dataclass
class ASQuote:
    """Avellaneda-Stoikov market maker quote."""
    bid: float = 0.0
    ask: float = 0.0
    spread: float = 0.0
    reservation_price: float = 0.0
    mid: float = 0.0
    inventory_skew: float = 0.0


class AvellanedaStoikovMarketMaker:
    """Avellaneda-Stoikov optimal market making model.

    Reference: Avellaneda & Stoikov (2008) "High-frequency trading in a
    limit order book."

    Reservation price: r(s, q, t) = s - q * gamma * sigma^2 * (T - t)
    Optimal spread: delta = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/k)

    Where:
        s = mid price
        q = inventory
        gamma = risk aversion
        sigma = volatility
        T - t = time remaining
        k = order arrival intensity
    """

    def __init__(
        self,
        gamma: float = 0.1,
        k: float = 1.5,
        sigma: float = 0.02,
        T: float = 1.0,
        inventory_manager: Optional[InventoryManager] = None,
    ):
        """
        Args:
            gamma: Risk aversion parameter.
            k: Order arrival intensity (higher = more aggressive).
            sigma: Volatility (e.g., daily).
            T: Trading session length (1.0 = full day).
            inventory_manager: Optional InventoryManager for position tracking.
        """
        self.gamma = gamma
        self.k = k
        self.sigma = sigma
        self.T = T
        self.inventory = inventory_manager or InventoryManager()

    def reservation_price(self, mid: float, time_remaining: float = 1.0) -> float:
        """Calculate reservation price given current inventory.

        r = s - q * gamma * sigma^2 * (T - t)
        """
        q = self.inventory.position
        return mid - q * self.gamma * (self.sigma ** 2) * time_remaining

    def optimal_spread(self, time_remaining: float = 1.0) -> float:
        """Calculate optimal spread.

        delta = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/k)
        """
        variance_term = self.gamma * (self.sigma ** 2) * time_remaining
        intensity_term = (2.0 / self.gamma) * math.log(1.0 + self.gamma / self.k)
        return variance_term + intensity_term

    def compute_quotes(
        self,
        mid: float,
        time_remaining: float = 1.0,
    ) -> ASQuote:
        """Compute optimal bid/ask quotes.

        Args:
            mid: Current mid price.
            time_remaining: Fraction of trading session remaining [0, 1].

        Returns:
            ASQuote with bid, ask, spread, and reservation price.
        """
        r = self.reservation_price(mid, time_remaining)
        spread = self.optimal_spread(time_remaining)
        half_spread = spread / 2.0

        # Apply inventory skew from InventoryManager
        inv_skew = self.inventory.get_skew()

        bid = r - half_spread + inv_skew
        ask = r + half_spread + inv_skew

        # Safety: bid must be < ask
        if bid >= ask:
            center = (bid + ask) / 2.0
            bid = center - 0.005
            ask = center + 0.005

        quote = ASQuote(
            bid=round(bid, 4),
            ask=round(ask, 4),
            spread=round(ask - bid, 4),
            reservation_price=round(r, 4),
            mid=mid,
            inventory_skew=round(inv_skew, 6),
        )

        logger.debug(
            "AS Quote: bid=%.4f ask=%.4f spread=%.4f r=%.4f q=%d",
            quote.bid, quote.ask, quote.spread, quote.reservation_price,
            self.inventory.position,
        )
        return quote

    def on_fill(self, qty: int) -> InventoryState:
        """Process a fill and update inventory.

        Args:
            qty: Positive for buy fill, negative for sell fill.

        Returns:
            Updated InventoryState.
        """
        state = self.inventory.update(qty)
        if state.is_hard_stop:
            logger.critical("HARD STOP triggered — flatten inventory immediately!")
        return state
