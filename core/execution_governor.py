"""
core/execution_governor.py
==========================
ORIA-inspired Execution Governor.

From Joshua Aalampour's ORIA Part 3:
    Target Weights → Staged orders under liquidity, speed, Participation limits

    Forward position: ξ_{t,t+1}^for = (V_t · w_{t+1}^arg) / P_{t,t+1}^ref
    Execution delta:  Δξ_{t,t+1}^ex = ξ_{t,t+1}^for − ξ_t

The Execution Governor controls:
1. Order staging — split large orders into smaller child orders
2. Participation rate limits — don't exceed X% of volume
3. Urgency-based execution speed
4. Transaction cost estimation
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ExecutionPlan:
    """Execution plan for a single signal."""
    ticker: str
    direction: str  # "BUY" or "SELL"
    target_qty: int
    staged_orders: List[Dict]
    estimated_cost: float
    urgency: str  # "LOW", "MEDIUM", "HIGH"
    participation_pct: float
    notes: str


@dataclass
class GovernorConfig:
    """Execution Governor configuration."""
    max_participation_pct: float = 0.05    # Max 5% of avg daily volume per order
    max_single_order_shares: int = 100     # Max shares per child order
    min_order_value: float = 50.0          # Minimum order value ($)
    urgency_threshold_high: float = 0.8    # Signal strength > 0.8 = HIGH urgency
    urgency_threshold_med: float = 0.4     # Signal strength > 0.4 = MEDIUM
    estimated_slippage_bps: float = 5.0    # 5 bps estimated slippage
    estimated_commission: float = 1.0      # $1 per trade (IBKR tiered)


class ExecutionGovernor:
    """ORIA-inspired Execution Governor.

    Converts approved risk-box signals into staged execution plans
    that minimize market impact and transaction costs.

    For our current account size (~$6K), most orders are 1-5 shares,
    so staging is minimal. But the framework is built to scale.

    Parameters
    ----------
    config : GovernorConfig
        Execution configuration.
    """

    def __init__(self, config: Optional[GovernorConfig] = None):
        self.config = config or GovernorConfig()
        self._avg_daily_volume: Dict[str, float] = {}

    def update_volume_data(self, volume_data: Dict[str, float]):
        """Update average daily volume estimates.

        Parameters
        ----------
        volume_data : dict
            {ticker: avg_daily_volume_shares}
        """
        self._avg_daily_volume.update(volume_data)

    def _estimate_adv(self, ticker: str) -> float:
        """Get average daily volume for a ticker."""
        return self._avg_daily_volume.get(ticker, 1_000_000)  # default 1M

    def _compute_urgency(self, signal_strength: float) -> str:
        """Map signal strength to execution urgency."""
        cfg = self.config
        if signal_strength >= cfg.urgency_threshold_high:
            return "HIGH"
        elif signal_strength >= cfg.urgency_threshold_med:
            return "MEDIUM"
        return "LOW"

    def _compute_participation_limit(self, ticker: str, qty: int) -> float:
        """Compute what % of ADV this order represents."""
        adv = self._estimate_adv(ticker)
        if adv <= 0:
            return 1.0
        return qty / adv

    def _estimate_transaction_cost(
        self,
        qty: int,
        price: float,
        n_child_orders: int = 1,
    ) -> float:
        """Estimate total transaction cost (commission + slippage).

        Parameters
        ----------
        qty : int
            Total shares.
        price : float
            Current price per share.
        n_child_orders : int
            Number of child orders.

        Returns
        -------
        float
            Estimated total cost in dollars.
        """
        cfg = self.config

        # Commission: per order
        commission = cfg.estimated_commission * n_child_orders

        # Slippage: proportional to order value
        order_value = qty * price
        slippage = order_value * (cfg.estimated_slippage_bps / 10_000)

        return round(commission + slippage, 2)

    def _stage_order(
        self,
        ticker: str,
        direction: str,
        total_qty: int,
        price: float,
        signal_strength: float,
    ) -> ExecutionPlan:
        """Create a staged execution plan for one signal.

        For small accounts, most orders are 1-5 shares → single order.
        For larger sizes, splits into child orders respecting participation limits.

        Parameters
        ----------
        ticker : str
            Symbol.
        direction : str
            "BUY" or "SELL".
        total_qty : int
            Total shares to trade.
        price : float
            Current price.
        signal_strength : float
            Original signal strength (for urgency).

        Returns
        -------
        ExecutionPlan
        """
        cfg = self.config
        urgency = self._compute_urgency(signal_strength)
        participation = self._compute_participation_limit(ticker, total_qty)

        # Check participation rate
        if participation > cfg.max_participation_pct:
            # Scale down to participation limit
            adv = self._estimate_adv(ticker)
            max_shares = max(1, int(adv * cfg.max_participation_pct))
            logger.warning(
                "Governor: %s %d shares = %.2f%% ADV (limit %.1f%%). "
                "Capping to %d shares.",
                ticker, total_qty, participation * 100,
                cfg.max_participation_pct * 100, max_shares,
            )
            total_qty = min(total_qty, max_shares)
            participation = self._compute_participation_limit(ticker, total_qty)

        # Stage into child orders
        if total_qty <= cfg.max_single_order_shares:
            # Single order (typical for our account size)
            staged = [{
                "ticker": ticker,
                "direction": direction,
                "qty": total_qty,
                "order_type": "BRACKET" if urgency != "LOW" else "LIMIT",
                "stage": 1,
                "of_total": 1,
            }]
        else:
            # Split into child orders
            n_children = math.ceil(total_qty / cfg.max_single_order_shares)
            staged = []
            remaining = total_qty
            for i in range(n_children):
                child_qty = min(remaining, cfg.max_single_order_shares)
                staged.append({
                    "ticker": ticker,
                    "direction": direction,
                    "qty": child_qty,
                    "order_type": "LIMIT",
                    "stage": i + 1,
                    "of_total": n_children,
                })
                remaining -= child_qty

        cost = self._estimate_transaction_cost(total_qty, price, len(staged))

        # Cost check: skip if estimated cost > 5% of order value
        order_value = total_qty * price
        cost_pct = cost / order_value if order_value > 0 else 1.0
        notes = f"cost={cost_pct:.1%} of order"

        if order_value < cfg.min_order_value:
            notes += f" | WARNING: order value ${order_value:.0f} below minimum ${cfg.min_order_value:.0f}"

        return ExecutionPlan(
            ticker=ticker,
            direction=direction,
            target_qty=total_qty,
            staged_orders=staged,
            estimated_cost=cost,
            urgency=urgency,
            participation_pct=round(participation * 100, 4),
            notes=notes,
        )

    def plan_execution(
        self,
        approved_signals: List[Dict],
        prices: Dict[str, float],
    ) -> List[ExecutionPlan]:
        """Create execution plans for all approved signals.

        Parameters
        ----------
        approved_signals : list of dict
            From RiskBox output. Each has: ticker, direction, position_value, etc.
        prices : dict
            {ticker: current_price}.

        Returns
        -------
        list of ExecutionPlan
            Ordered by urgency (HIGH first).
        """
        plans = []

        for sig in approved_signals:
            ticker = sig.get("ticker", "???")
            direction = sig.get("direction", "LONG")
            pos_value = sig.get("position_value", 0)
            signal_strength = sig.get("signal_strength", 0.5)
            price = prices.get(ticker, 0)

            if price <= 0:
                logger.warning("Governor: no price for %s, skipping", ticker)
                continue

            # Compute target shares
            target_qty = max(1, int(pos_value / price))

            # Map direction
            action = "BUY" if direction == "LONG" else "SELL"

            plan = self._stage_order(
                ticker=ticker,
                direction=action,
                total_qty=target_qty,
                price=price,
                signal_strength=signal_strength,
            )
            plans.append(plan)

            logger.info(
                "Governor: %s %s %d @ $%.2f (urgency=%s, cost=$%.2f, part=%.2f%%)",
                action, ticker, target_qty, price,
                plan.urgency, plan.estimated_cost, plan.participation_pct,
            )

        # Sort by urgency: HIGH first
        urgency_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        plans.sort(key=lambda p: urgency_order.get(p.urgency, 2))

        return plans
