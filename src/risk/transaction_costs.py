"""
Transaction Cost Model
=======================
Estimates the full round-trip cost of a trade BEFORE execution,
enabling the engine to skip trades whose expected alpha < expected cost.

Components
----------
1. Spread cost      — half bid-ask spread
2. Market impact    — Almgren-Chriss square-root temporary impact
3. Commission/fees  — flat per-share + % of notional
4. Slippage buffer  — configurable safety margin

Usage::

    from src.risk.transaction_costs import TransactionCostModel

    tcm = TransactionCostModel()
    cost = tcm.estimate_cost(
        symbol="AAPL",
        qty=100,
        price=185.0,
        side="buy",
        spread_bps=2.0,
        adv=50_000_000,
        volatility=0.018,
    )
    if cost.total_bps > expected_alpha_bps:
        skip trade ...
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CostEstimate:
    """Breakdown of estimated transaction costs."""
    symbol: str
    side: str
    qty: int
    notional: float
    spread_cost_bps: float       # half-spread
    impact_cost_bps: float       # temporary market impact
    commission_bps: float        # broker fees
    slippage_buffer_bps: float   # safety margin
    total_bps: float             # sum
    total_dollars: float         # total_bps * notional / 10_000

    @property
    def round_trip_bps(self) -> float:
        """Estimated round-trip cost (entry + exit)."""
        return self.total_bps * 2


@dataclass
class TCMConfig:
    """Tunables for TransactionCostModel."""
    # Spread
    default_spread_bps: float = 3.0          # fallback if spread unknown
    # Market impact — Almgren-Chriss params
    impact_eta: float = 0.1                  # temporary impact coefficient
    impact_exponent: float = 0.5             # square-root model
    # Commission
    commission_per_share: float = 0.0        # most brokers zero now
    commission_pct: float = 0.0              # % of notional
    min_commission: float = 0.0
    # Slippage buffer
    slippage_buffer_bps: float = 1.0         # extra safety margin
    # Gate
    max_cost_bps: float = 25.0               # refuse trades costing > this


class TransactionCostModel:
    """
    Pre-trade cost estimator.

    Combines spread, market impact, commission, and slippage buffer
    into a single basis-point cost number.

    The market impact model follows Almgren-Chriss (2001):
        impact = η · σ · (Q / ADV) ^ 0.5

    where η is a calibrated coefficient, σ is daily volatility,
    Q is order quantity in shares, ADV is average daily volume.
    """

    def __init__(self, config: Optional[TCMConfig] = None):
        self.cfg = config or TCMConfig()

    def estimate_cost(
        self,
        symbol: str,
        qty: int,
        price: float,
        side: str = "buy",
        spread_bps: Optional[float] = None,
        adv: Optional[float] = None,
        volatility: Optional[float] = None,
    ) -> CostEstimate:
        """
        Estimate total one-way transaction cost.

        Parameters
        ----------
        symbol : str
        qty : int          Order size in shares
        price : float      Reference price
        side : str         "buy" or "sell"
        spread_bps : float Bid-ask spread in basis points (half-spread used)
        adv : float        Average daily volume (shares)
        volatility : float Daily return volatility (e.g. 0.015 = 1.5%)

        Returns
        -------
        CostEstimate
        """
        notional = qty * price
        if notional <= 0:
            return CostEstimate(
                symbol=symbol, side=side, qty=qty, notional=0,
                spread_cost_bps=0, impact_cost_bps=0, commission_bps=0,
                slippage_buffer_bps=0, total_bps=0, total_dollars=0,
            )

        # 1. Spread cost — half the bid-ask spread
        half_spread = (spread_bps if spread_bps is not None
                       else self.cfg.default_spread_bps) / 2.0

        # 2. Market impact — Almgren-Chriss square-root model
        impact_bps = 0.0
        if adv and adv > 0 and volatility and volatility > 0:
            participation_rate = qty / adv
            impact_bps = (
                self.cfg.impact_eta
                * volatility
                * (participation_rate ** self.cfg.impact_exponent)
                * 10_000  # convert to bps
            )

        # 3. Commission
        comm_dollars = max(
            self.cfg.min_commission,
            qty * self.cfg.commission_per_share + notional * self.cfg.commission_pct,
        )
        comm_bps = comm_dollars / notional * 10_000 if notional > 0 else 0

        # 4. Slippage buffer
        slippage = self.cfg.slippage_buffer_bps

        total = half_spread + impact_bps + comm_bps + slippage
        total_dollars = notional * total / 10_000

        estimate = CostEstimate(
            symbol=symbol,
            side=side,
            qty=qty,
            notional=notional,
            spread_cost_bps=round(half_spread, 2),
            impact_cost_bps=round(impact_bps, 2),
            commission_bps=round(comm_bps, 2),
            slippage_buffer_bps=round(slippage, 2),
            total_bps=round(total, 2),
            total_dollars=round(total_dollars, 2),
        )

        logger.debug(
            f"TCA {symbol} {side} {qty}sh: spread={half_spread:.1f} "
            f"impact={impact_bps:.1f} comm={comm_bps:.1f} "
            f"total={total:.1f} bps (${total_dollars:.2f})"
        )
        return estimate

    def should_trade(
        self,
        cost: CostEstimate,
        expected_alpha_bps: float = 0.0,
    ) -> tuple[bool, str]:
        """
        Gate: reject trades where cost exceeds expected alpha or hard cap.

        Returns (allowed, reason).
        """
        if cost.total_bps > self.cfg.max_cost_bps:
            return False, f"cost {cost.total_bps:.1f} bps > cap {self.cfg.max_cost_bps:.1f}"
        if expected_alpha_bps > 0 and cost.total_bps > expected_alpha_bps:
            return False, (
                f"cost {cost.total_bps:.1f} bps > alpha {expected_alpha_bps:.1f} bps"
            )
        return True, "ok"
