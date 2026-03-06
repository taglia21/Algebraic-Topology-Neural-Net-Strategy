"""
Transaction Cost Model — Effective Spread (Phase F, Item 2)
============================================================

Model realized half-spread using Lee-Ready tick rule on trade prints.
Subtract from expected edge before entry to gate trades with negative net alpha.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["TransactionCostModel", "SpreadCostResult"]


@dataclass
class SpreadCostResult:
    """Result of effective spread cost estimation."""
    symbol: str
    effective_half_spread_bps: float
    trade_direction: str  # "buy" or "sell" inferred via Lee-Ready
    mid_price: float
    trade_price: float
    net_edge_bps: float  # expected_edge - cost
    should_trade: bool   # True if net_edge > 0


class TransactionCostModel:
    """Lee-Ready tick rule effective spread estimator.

    The Lee-Ready algorithm classifies each trade as buyer- or
    seller-initiated by comparing the trade price to the prevailing
    mid-quote.  The effective half-spread is:

        effective_hs = |trade_price - mid_price| / mid_price * 10_000  (bps)

    A trade is only allowed if ``expected_edge_bps - effective_hs > 0``.

    Parameters
    ----------
    min_net_edge_bps : float
        Minimum net edge (expected - cost) to allow trade (default 0).
    lookback : int
        Rolling window of recent trades for averaging (default 20).
    """

    def __init__(self, min_net_edge_bps: float = 0.0, lookback: int = 20):
        self.min_net_edge_bps = min_net_edge_bps
        self.lookback = lookback
        self._recent_spreads: Dict[str, List[float]] = {}

    def effective_spread_cost(
        self,
        trade_price: float,
        bid: float,
        ask: float,
        expected_edge_bps: float = 5.0,
        symbol: str = "UNK",
    ) -> SpreadCostResult:
        """Compute realized half-spread using Lee-Ready tick rule.

        Parameters
        ----------
        trade_price : float
            Last trade price.
        bid : float
            Current best bid.
        ask : float
            Current best ask.
        expected_edge_bps : float
            Strategy's expected edge in basis points.
        symbol : str
            Symbol for logging/tracking.

        Returns
        -------
        SpreadCostResult
        """
        if bid <= 0 or ask <= 0 or ask < bid:
            return SpreadCostResult(
                symbol=symbol, effective_half_spread_bps=0.0,
                trade_direction="unknown", mid_price=0.0,
                trade_price=trade_price, net_edge_bps=expected_edge_bps,
                should_trade=True,
            )

        mid = (bid + ask) / 2.0
        if mid <= 0:
            mid = trade_price

        # Lee-Ready classification
        if trade_price > mid:
            direction = "buy"
        elif trade_price < mid:
            direction = "sell"
        else:
            direction = "buy"  # tie → default buy

        # Effective half-spread in bps
        eff_hs_bps = abs(trade_price - mid) / mid * 10_000

        # Track rolling average
        if symbol not in self._recent_spreads:
            self._recent_spreads[symbol] = []
        self._recent_spreads[symbol].append(eff_hs_bps)
        if len(self._recent_spreads[symbol]) > self.lookback:
            self._recent_spreads[symbol] = self._recent_spreads[symbol][-self.lookback:]

        net_edge = expected_edge_bps - eff_hs_bps
        should_trade = net_edge > self.min_net_edge_bps

        if not should_trade:
            logger.warning(
                "NEGATIVE ALPHA GATE: %s eff_spread=%.2f bps > edge=%.2f bps → REJECT",
                symbol, eff_hs_bps, expected_edge_bps,
            )

        return SpreadCostResult(
            symbol=symbol,
            effective_half_spread_bps=eff_hs_bps,
            trade_direction=direction,
            mid_price=mid,
            trade_price=trade_price,
            net_edge_bps=net_edge,
            should_trade=should_trade,
        )

    def get_avg_spread_bps(self, symbol: str) -> float:
        """Return rolling average half-spread for a symbol."""
        history = self._recent_spreads.get(symbol, [])
        return float(np.mean(history)) if history else 0.0
