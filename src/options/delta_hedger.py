"""
Delta Hedger — Real-Time Greek Hedging (Phase J, Item 13)
==========================================================

Every 5 minutes during market hours, compute net portfolio delta from
all open option positions; if |net_delta| > 0.10, submit SPY hedge
order to neutralize.  Log all hedges.
"""

import asyncio
import logging
import time as _time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["DeltaHedger", "HedgeRecord"]


@dataclass
class HedgeRecord:
    """Record of a delta hedge execution."""
    timestamp: float
    net_delta_before: float
    hedge_qty: int
    hedge_symbol: str
    hedge_side: str
    net_delta_after: float


class DeltaHedger:
    """Automatic delta-neutral hedging via SPY shares.

    Parameters
    ----------
    delta_threshold : float
        Hedge when |net_delta| exceeds this (default 0.10).
    hedge_symbol : str
        Instrument used for hedging (default "SPY").
    interval_seconds : int
        Check interval (default 300 = 5 min).
    broker_client : object or None
        Broker client for order submission.
    """

    def __init__(
        self,
        delta_threshold: float = 0.10,
        hedge_symbol: str = "SPY",
        interval_seconds: int = 300,
        broker_client=None,
    ):
        self.delta_threshold = delta_threshold
        self.hedge_symbol = hedge_symbol
        self.interval_seconds = interval_seconds
        self.broker = broker_client
        self._hedges: List[HedgeRecord] = []
        self._running = False
        self._cumulative_hedge_qty: int = 0

    def compute_net_delta(self, positions: List[Dict]) -> float:
        """Compute net portfolio delta from option positions.

        Parameters
        ----------
        positions : list of dict
            Each dict should have ``delta`` and ``quantity`` keys.

        Returns
        -------
        float
            Net portfolio delta.
        """
        net = 0.0
        for pos in positions:
            delta = float(pos.get("delta", 0.0))
            qty = int(pos.get("quantity", pos.get("qty", 0)))
            net += delta * qty
        return net

    def should_hedge(self, net_delta: float) -> bool:
        """Check if hedging is needed."""
        return abs(net_delta) > self.delta_threshold

    def compute_hedge_order(self, net_delta: float) -> Dict:
        """Compute the hedge order to neutralize delta.

        A positive net_delta means we need to sell shares (short delta);
        negative means buy shares.

        Returns
        -------
        dict with ``symbol``, ``side``, ``quantity`` keys.
        """
        # Round to integer shares; 1 delta ≈ 100 shares
        shares = int(round(-net_delta * 100))
        side = "buy" if shares > 0 else "sell"
        return {
            "symbol": self.hedge_symbol,
            "side": side,
            "quantity": abs(shares),
        }

    async def check_and_hedge(self, positions: List[Dict]) -> Optional[HedgeRecord]:
        """Check delta and hedge if needed.

        Parameters
        ----------
        positions : list of dict

        Returns
        -------
        HedgeRecord or None
        """
        net_delta = self.compute_net_delta(positions)

        if not self.should_hedge(net_delta):
            logger.debug("Delta OK: %.4f (threshold %.2f)", net_delta, self.delta_threshold)
            return None

        order = self.compute_hedge_order(net_delta)
        logger.info(
            "DELTA HEDGE: net_delta=%.4f → %s %d %s",
            net_delta, order["side"], order["quantity"], order["symbol"],
        )

        # Execute hedge
        if self.broker and hasattr(self.broker, "submit_order"):
            try:
                await self.broker.submit_order(**order)
            except Exception as exc:
                logger.error("Hedge order failed: %s", exc)

        new_delta = net_delta + (order["quantity"] / 100.0 * (1 if order["side"] == "buy" else -1))
        self._cumulative_hedge_qty += order["quantity"]

        record = HedgeRecord(
            timestamp=_time.time(),
            net_delta_before=net_delta,
            hedge_qty=order["quantity"],
            hedge_symbol=order["symbol"],
            hedge_side=order["side"],
            net_delta_after=new_delta,
        )
        self._hedges.append(record)
        return record

    @property
    def hedge_history(self) -> List[HedgeRecord]:
        return list(self._hedges)

    @property
    def total_hedges(self) -> int:
        return len(self._hedges)

    async def run_loop(self, get_positions_fn) -> None:
        """Run continuous hedging loop (call in background task).

        Parameters
        ----------
        get_positions_fn : callable
            Returns list of position dicts with delta/quantity.
        """
        self._running = True
        logger.info("Delta hedger started (interval=%ds, threshold=%.2f)",
                     self.interval_seconds, self.delta_threshold)
        while self._running:
            try:
                positions = get_positions_fn()
                await self.check_and_hedge(positions)
            except Exception as exc:
                logger.error("Delta hedger error: %s", exc)
            await asyncio.sleep(self.interval_seconds)

    def stop(self):
        """Stop the hedging loop."""
        self._running = False
