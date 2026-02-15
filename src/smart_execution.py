"""
Smart Execution Engine
=======================
TWAP / VWAP order splitting with slippage tracking.

Splits a parent order into child slices over 5–15 minutes,
submitting each slice as a limit order with a small buffer.

Usage:
    from src.smart_execution import SmartExecutor, ExecutionPlan

    executor = SmartExecutor(submit_fn=submit_limit_order)
    plan = executor.plan_execution("AAPL", qty=200, side="buy",
                                   ref_price=185.0, strategy="twap")
    # Execute slices (call repeatedly on a timer, or all-at-once in backtest)
    for sl in plan.slices:
        executor.execute_slice(plan, sl)
    report = executor.get_report(plan)
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
from datetime import datetime, timedelta

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class OrderSlice:
    """One child slice of a parent order."""
    slice_id: int
    qty: int
    target_time_offset_sec: float  # seconds from plan start
    limit_price: float = 0.0
    status: str = "pending"        # pending / submitted / filled / failed
    fill_price: float = 0.0
    fill_time: Optional[datetime] = None
    slippage_bps: float = 0.0


@dataclass
class ExecutionPlan:
    """A full parent order split into slices."""
    plan_id: str
    symbol: str
    side: str                      # buy / sell
    total_qty: int
    ref_price: float               # NBBO at plan creation
    strategy: str                  # twap / vwap
    slices: List[OrderSlice] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    duration_sec: float = 300.0    # 5 min default
    buffer_pct: float = 0.001     # 10 bps limit buffer
    filled_qty: int = 0
    avg_fill_price: float = 0.0
    total_slippage_bps: float = 0.0
    status: str = "planned"        # planned / active / complete / aborted


@dataclass
class ExecutionReport:
    """Post-execution analytics."""
    plan_id: str
    symbol: str
    side: str
    total_qty: int
    filled_qty: int
    ref_price: float
    avg_fill_price: float
    slippage_bps: float
    num_slices: int
    slices_filled: int
    duration_sec: float
    strategy: str


@dataclass
class SmartExecConfig:
    """Tunables for the execution engine."""
    default_strategy: str = "twap"
    default_duration_sec: float = 300.0    # 5 min
    max_duration_sec: float = 900.0        # 15 min
    min_slices: int = 3
    max_slices: int = 10
    slice_size_variation: float = 0.20     # ±20% randomisation
    limit_buffer_buy_pct: float = 0.001    # +10 bps for buys
    limit_buffer_sell_pct: float = 0.001   # -10 bps for sells
    min_qty_per_slice: int = 1


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def _generate_twap_schedule(
    total_qty: int, n_slices: int, variation: float = 0.20,
) -> List[int]:
    """
    Split ``total_qty`` into ``n_slices`` roughly-equal child quantities
    with ±``variation`` randomisation.
    """
    base = total_qty / n_slices
    raw = [max(1, int(base * (1 + np.random.uniform(-variation, variation))))
           for _ in range(n_slices)]
    # Adjust last slice to hit exact total
    raw[-1] = total_qty - sum(raw[:-1])
    if raw[-1] <= 0:
        raw[-1] = 1
        excess = sum(raw) - total_qty
        # Trim excess from earlier slices
        for i in range(len(raw) - 2, -1, -1):
            trim = min(excess, raw[i] - 1)
            raw[i] -= trim
            excess -= trim
            if excess <= 0:
                break
    return raw


def _generate_vwap_schedule(
    total_qty: int, n_slices: int,
) -> List[int]:
    """
    VWAP-style schedule: front-load slightly (U-shape), reflecting
    typical intraday volume patterns.
    """
    # U-shape weight: more at start and end, less in the middle
    x = np.linspace(0, 1, n_slices)
    weights = 1.0 + 0.5 * (4.0 * (x - 0.5) ** 2)
    weights /= weights.sum()
    raw = [max(1, int(total_qty * w)) for w in weights]
    raw[-1] = total_qty - sum(raw[:-1])
    if raw[-1] <= 0:
        raw[-1] = 1
    return raw


# ─────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────

class SmartExecutor:
    """
    TWAP / VWAP smart execution engine.

    Parameters
    ----------
    submit_fn : callable
        ``submit_fn(symbol, qty, side, limit_price) -> dict | None``
        The function used to actually submit orders (e.g. Alpaca REST).
    config : SmartExecConfig
        Tunable parameters.
    """

    def __init__(
        self,
        submit_fn: Optional[Callable] = None,
        config: SmartExecConfig = None,
    ):
        self.submit_fn = submit_fn
        self.cfg = config or SmartExecConfig()
        self._plans: Dict[str, ExecutionPlan] = {}

    # ── Plan creation ────────────────────────────────────────────

    def plan_execution(
        self,
        symbol: str,
        qty: int,
        side: str,
        ref_price: float,
        strategy: str = None,
        duration_sec: float = None,
    ) -> ExecutionPlan:
        """
        Create an execution plan (without submitting anything yet).

        Parameters
        ----------
        symbol : str
        qty : int   Total shares to trade.
        side : str  "buy" or "sell".
        ref_price : float   Current mid/last price.
        strategy : str   "twap" or "vwap".
        duration_sec : float   Execution window in seconds.

        Returns
        -------
        ExecutionPlan
        """
        strategy = strategy or self.cfg.default_strategy
        duration_sec = duration_sec or self.cfg.default_duration_sec
        duration_sec = min(duration_sec, self.cfg.max_duration_sec)

        n_slices = max(
            self.cfg.min_slices,
            min(self.cfg.max_slices, qty // 10),
        )
        # Ensure we don't end up with slices < min_qty
        while qty // n_slices < self.cfg.min_qty_per_slice and n_slices > 1:
            n_slices -= 1

        # Generate child quantities
        if strategy == "vwap":
            qtys = _generate_vwap_schedule(qty, n_slices)
        else:
            qtys = _generate_twap_schedule(qty, n_slices, self.cfg.slice_size_variation)

        # Time offsets
        interval = duration_sec / n_slices
        slices: List[OrderSlice] = []
        for i, q in enumerate(qtys):
            buf = self.cfg.limit_buffer_buy_pct if side == "buy" else self.cfg.limit_buffer_sell_pct
            if side == "buy":
                limit = round(ref_price * (1 + buf), 2)
            else:
                limit = round(ref_price * (1 - buf), 2)

            slices.append(OrderSlice(
                slice_id=i,
                qty=q,
                target_time_offset_sec=interval * i,
                limit_price=limit,
            ))

        plan_id = f"exec_{symbol}_{uuid.uuid4().hex[:8]}"
        plan = ExecutionPlan(
            plan_id=plan_id,
            symbol=symbol,
            side=side,
            total_qty=qty,
            ref_price=ref_price,
            strategy=strategy,
            slices=slices,
            duration_sec=duration_sec,
            buffer_pct=buf,
            status="planned",
        )
        self._plans[plan_id] = plan
        logger.info(
            f"Execution plan {plan_id}: {side.upper()} {qty} {symbol} "
            f"via {strategy.upper()} in {n_slices} slices over "
            f"{duration_sec:.0f}s"
        )
        return plan

    # ── Slice execution ──────────────────────────────────────────

    def execute_slice(
        self,
        plan: ExecutionPlan,
        sl: OrderSlice,
        current_price: Optional[float] = None,
    ) -> bool:
        """
        Submit one slice.  Returns True if submission succeeded.

        If ``current_price`` is given, the limit price is refreshed
        relative to the live price rather than the stale ``ref_price``.
        """
        if sl.status != "pending":
            return False

        plan.status = "active"

        # Refresh limit if we have a live price
        if current_price is not None:
            buf = self.cfg.limit_buffer_buy_pct if plan.side == "buy" else self.cfg.limit_buffer_sell_pct
            if plan.side == "buy":
                sl.limit_price = round(current_price * (1 + buf), 2)
            else:
                sl.limit_price = round(current_price * (1 - buf), 2)

        if self.submit_fn is None:
            # Dry-run / backtest mode — simulate instant fill
            sl.status = "filled"
            sl.fill_price = sl.limit_price
            sl.fill_time = datetime.now()
            sl.slippage_bps = 0.0
        else:
            result = self.submit_fn(plan.symbol, sl.qty, plan.side, sl.limit_price)
            if result is None:
                sl.status = "failed"
                logger.warning(f"Slice {sl.slice_id} failed for {plan.symbol}")
                return False
            sl.status = "submitted"
            # Assume fill at limit for now (real tracking needs order-status poll)
            sl.fill_price = sl.limit_price
            sl.fill_time = datetime.now()

        # Compute per-slice slippage
        if plan.ref_price > 0:
            if plan.side == "buy":
                sl.slippage_bps = (sl.fill_price - plan.ref_price) / plan.ref_price * 10_000
            else:
                sl.slippage_bps = (plan.ref_price - sl.fill_price) / plan.ref_price * 10_000

        sl.status = "filled"
        plan.filled_qty += sl.qty

        # Update plan-level average fill
        fills = [s for s in plan.slices if s.status == "filled"]
        if fills:
            total_cost = sum(s.fill_price * s.qty for s in fills)
            total_q = sum(s.qty for s in fills)
            plan.avg_fill_price = total_cost / total_q if total_q > 0 else plan.ref_price

        # Check if all slices are done
        if all(s.status in ("filled", "failed") for s in plan.slices):
            plan.status = "complete"
            self._compute_plan_slippage(plan)
            logger.info(
                f"Plan {plan.plan_id} COMPLETE: "
                f"{plan.filled_qty}/{plan.total_qty} filled, "
                f"avg=${plan.avg_fill_price:.2f}, "
                f"slippage={plan.total_slippage_bps:+.1f} bps"
            )

        return True

    def execute_all_slices(
        self,
        plan: ExecutionPlan,
        current_price: Optional[float] = None,
        inter_slice_delay: float = 0.0,
    ) -> ExecutionReport:
        """
        Execute every slice sequentially (convenience for backtest / fast
        execution).  For live trading, call ``execute_slice`` on a timer.
        """
        for sl in plan.slices:
            self.execute_slice(plan, sl, current_price=current_price)
            if inter_slice_delay > 0:
                time.sleep(inter_slice_delay)
        return self.get_report(plan)

    # ── Reporting ────────────────────────────────────────────────

    def get_report(self, plan: ExecutionPlan) -> ExecutionReport:
        """Build a post-execution report."""
        self._compute_plan_slippage(plan)
        elapsed = (datetime.now() - plan.created_at).total_seconds()
        filled_count = sum(1 for s in plan.slices if s.status == "filled")
        return ExecutionReport(
            plan_id=plan.plan_id,
            symbol=plan.symbol,
            side=plan.side,
            total_qty=plan.total_qty,
            filled_qty=plan.filled_qty,
            ref_price=plan.ref_price,
            avg_fill_price=plan.avg_fill_price,
            slippage_bps=plan.total_slippage_bps,
            num_slices=len(plan.slices),
            slices_filled=filled_count,
            duration_sec=elapsed,
            strategy=plan.strategy,
        )

    # ── Internals ────────────────────────────────────────────────

    def _compute_plan_slippage(self, plan: ExecutionPlan):
        """Aggregate slippage across all filled slices."""
        if plan.ref_price <= 0:
            plan.total_slippage_bps = 0.0
            return
        if plan.side == "buy":
            plan.total_slippage_bps = (
                (plan.avg_fill_price - plan.ref_price) / plan.ref_price * 10_000
            )
        else:
            plan.total_slippage_bps = (
                (plan.ref_price - plan.avg_fill_price) / plan.ref_price * 10_000
            )
