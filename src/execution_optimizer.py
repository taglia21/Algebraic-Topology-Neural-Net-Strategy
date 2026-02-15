"""
Execution Optimiser — Smart Order Routing & Slippage Minimisation (TIER 4)
===========================================================================

Provides intelligent order execution with multiple algorithms:
1. **TWAP** — Time-Weighted Average Price
2. **VWAP** — Volume-Weighted Average Price (with intraday profile)
3. **Iceberg** — Hidden-size orders to reduce market impact
4. **Adaptive** — Dynamically picks strategy based on urgency / size

Slippage prediction model estimates expected cost *before* execution.

Usage:
    from src.execution_optimizer import ExecutionOptimizer, ExecConfig

    opt = ExecutionOptimizer()
    result = opt.execute_optimal("AAPL", qty=500, side="buy",
                                  ref_price=185.0, urgency=0.5)
"""

import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS & CONFIG
# =============================================================================

class ExecStrategy(Enum):
    TWAP = "twap"
    VWAP = "vwap"
    ICEBERG = "iceberg"
    ADAPTIVE = "adaptive"


class SliceStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class ExecConfig:
    """Configuration for execution optimiser."""
    # Timing
    default_duration_sec: float = 300.0     # 5 minutes
    min_slice_interval_sec: float = 10.0
    max_slices: int = 20

    # Sizing
    iceberg_show_pct: float = 0.15          # show 15% of remaining
    iceberg_random_noise: float = 0.05      # ±5% random on show size

    # VWAP profile (normalised intraday bins — 13 x 30-min from 9:30–16:00 ET)
    vwap_profile: List[float] = field(default_factory=lambda: [
        0.11, 0.08, 0.07, 0.06, 0.06, 0.06, 0.06,
        0.06, 0.07, 0.07, 0.07, 0.09, 0.14,
    ])

    # Limit buffer
    limit_buffer_bps: float = 10.0          # 10 bps

    # Slippage model coefficients (simple linear model)
    # slippage_bps ≈ α + β₁ * participation_rate + β₂ * volatility + β₃ * urgency
    slip_alpha: float = 0.5
    slip_beta_participation: float = 30.0
    slip_beta_volatility: float = 200.0
    slip_beta_urgency: float = 5.0

    # Cost threshold — warn if estimated cost exceeds
    cost_warn_bps: float = 20.0

    # Adaptive strategy thresholds
    adaptive_large_pct: float = 0.01        # > 1% ADV → iceberg
    adaptive_urgent_threshold: float = 0.8  # urgency > 0.8 → TWAP fast


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class OrderSlice:
    """Single child order in an execution plan."""
    slice_id: int = 0
    qty: int = 0
    target_time_offset_sec: float = 0.0
    limit_price: float = 0.0
    status: SliceStatus = SliceStatus.PENDING
    fill_price: float = 0.0
    fill_qty: int = 0
    submitted_at: Optional[str] = None
    filled_at: Optional[str] = None
    slippage_bps: float = 0.0


@dataclass
class ExecutionPlan:
    """Full execution plan for a parent order."""
    plan_id: str = ""
    symbol: str = ""
    side: str = "buy"
    total_qty: int = 0
    ref_price: float = 0.0
    strategy: ExecStrategy = ExecStrategy.TWAP
    slices: List[OrderSlice] = field(default_factory=list)
    duration_sec: float = 300.0
    created_at: str = ""
    estimated_slippage_bps: float = 0.0
    estimated_cost_bps: float = 0.0
    status: str = "planned"


@dataclass
class ExecutionResult:
    """Result after (simulated) execution."""
    plan_id: str = ""
    symbol: str = ""
    side: str = "buy"
    total_qty: int = 0
    filled_qty: int = 0
    ref_price: float = 0.0
    avg_fill_price: float = 0.0
    realised_slippage_bps: float = 0.0
    num_slices: int = 0
    strategy: str = "twap"
    duration_sec: float = 0.0
    timestamp: str = ""


# =============================================================================
# SLIPPAGE PREDICTOR
# =============================================================================

class SlippagePredictor:
    """Linear slippage model: predicts cost (bps) before execution."""

    def __init__(self, config: ExecConfig):
        self.cfg = config

    def predict(
        self, qty: int, avg_daily_volume: float,
        volatility: float, urgency: float,
    ) -> float:
        """
        Estimate execution cost in bps.

        Parameters
        ----------
        qty : order quantity
        avg_daily_volume : ADV shares
        volatility : daily return volatility (e.g. 0.02)
        urgency : 0 (patient) .. 1 (immediate)
        """
        adv = max(avg_daily_volume, 1)
        participation = qty / adv
        slip = (
            self.cfg.slip_alpha
            + self.cfg.slip_beta_participation * participation
            + self.cfg.slip_beta_volatility * volatility
            + self.cfg.slip_beta_urgency * urgency
        )
        return max(slip, 0.0)


# =============================================================================
# EXECUTION OPTIMIZER
# =============================================================================

class ExecutionOptimizer:
    """
    Intelligent order execution engine.

    Usage:
        opt = ExecutionOptimizer()
        result = opt.execute_optimal("AAPL", 500, "buy", 185.0, urgency=0.5)
    """

    def __init__(self, config: Optional[ExecConfig] = None,
                 submit_fn: Optional[Callable] = None):
        """
        Parameters
        ----------
        config : ExecConfig
        submit_fn : optional callback(symbol, qty, side, limit_price) → fill_price.
                    If None, uses internal simulation.
        """
        self.config = config or ExecConfig()
        self._submit_fn = submit_fn
        self._predictor = SlippagePredictor(self.config)
        self._plans: Dict[str, ExecutionPlan] = {}
        self._results: Dict[str, ExecutionResult] = {}
        logger.info("ExecutionOptimizer initialised (strategies: %s)",
                     [s.value for s in ExecStrategy])

    # ── Public API ───────────────────────────────────────────────────────

    def execute_optimal(
        self,
        symbol: str,
        qty: int,
        side: str = "buy",
        ref_price: float = 0.0,
        urgency: float = 0.5,
        avg_daily_volume: float = 1_000_000,
        volatility: float = 0.02,
        strategy: Optional[ExecStrategy] = None,
        duration_sec: Optional[float] = None,
    ) -> ExecutionResult:
        """
        Plan and execute (or simulate) an order optimally.

        Parameters
        ----------
        symbol : ticker
        qty : shares to trade
        side : "buy" or "sell"
        ref_price : reference / arrival price
        urgency : 0 (patient) .. 1 (immediate)
        avg_daily_volume : ADV for participation calc
        volatility : daily return vol
        strategy : force a strategy; None = adaptive
        duration_sec : override duration
        """
        if strategy is None:
            strategy = self._pick_strategy(qty, avg_daily_volume, urgency)

        dur = duration_sec or self._pick_duration(urgency)
        plan = self._build_plan(symbol, qty, side, ref_price, strategy, dur,
                                avg_daily_volume, volatility, urgency)
        self._plans[plan.plan_id] = plan

        result = self._execute_plan(plan)
        self._results[plan.plan_id] = result

        logger.info("Exec %s %s %d %s @ %.2f → avg %.2f (slip %.1f bps, %d slices)",
                     plan.plan_id[:8], symbol, qty, side,
                     ref_price, result.avg_fill_price,
                     result.realised_slippage_bps, result.num_slices)
        return result

    def estimate_cost(
        self, qty: int, avg_daily_volume: float = 1_000_000,
        volatility: float = 0.02, urgency: float = 0.5,
    ) -> float:
        """Estimate execution cost in bps without executing."""
        return self._predictor.predict(qty, avg_daily_volume, volatility, urgency)

    def get_plan(self, plan_id: str) -> Optional[ExecutionPlan]:
        return self._plans.get(plan_id)

    def get_result(self, plan_id: str) -> Optional[ExecutionResult]:
        return self._results.get(plan_id)

    def get_all_results(self) -> List[ExecutionResult]:
        return list(self._results.values())

    # ── Strategy selection ───────────────────────────────────────────────

    def _pick_strategy(
        self, qty: int, adv: float, urgency: float,
    ) -> ExecStrategy:
        """Adaptive strategy picker."""
        participation = qty / max(adv, 1)

        if participation > self.config.adaptive_large_pct:
            return ExecStrategy.ICEBERG
        if urgency > self.config.adaptive_urgent_threshold:
            return ExecStrategy.TWAP
        if participation > 0.001:
            return ExecStrategy.VWAP
        return ExecStrategy.TWAP

    def _pick_duration(self, urgency: float) -> float:
        """Map urgency to duration."""
        # urgency 0 → 10 min, urgency 1 → 30 sec
        max_dur = self.config.default_duration_sec * 2  # 10 min
        min_dur = 30.0
        dur = max_dur - (max_dur - min_dur) * urgency
        return max(dur, min_dur)

    # ── Plan building ────────────────────────────────────────────────────

    def _build_plan(
        self, symbol: str, qty: int, side: str, ref_price: float,
        strategy: ExecStrategy, duration_sec: float,
        adv: float, vol: float, urgency: float,
    ) -> ExecutionPlan:
        plan = ExecutionPlan(
            plan_id=uuid.uuid4().hex[:12],
            symbol=symbol,
            side=side,
            total_qty=qty,
            ref_price=ref_price,
            strategy=strategy,
            duration_sec=duration_sec,
            created_at=datetime.now(timezone.utc).isoformat(),
            estimated_slippage_bps=self._predictor.predict(qty, adv, vol, urgency),
        )

        if strategy == ExecStrategy.TWAP:
            plan.slices = self._build_twap_slices(qty, duration_sec, ref_price, side)
        elif strategy == ExecStrategy.VWAP:
            plan.slices = self._build_vwap_slices(qty, duration_sec, ref_price, side)
        elif strategy == ExecStrategy.ICEBERG:
            plan.slices = self._build_iceberg_slices(qty, duration_sec, ref_price, side)
        else:  # ADAPTIVE fallback
            plan.slices = self._build_twap_slices(qty, duration_sec, ref_price, side)

        plan.estimated_cost_bps = plan.estimated_slippage_bps
        if plan.estimated_cost_bps > self.config.cost_warn_bps:
            logger.warning("High estimated cost: %.1f bps for %s %d %s",
                           plan.estimated_cost_bps, symbol, qty, side)

        return plan

    def _build_twap_slices(
        self, qty: int, dur: float, ref: float, side: str,
    ) -> List[OrderSlice]:
        """Equal-sized slices spread over duration."""
        n = min(self.config.max_slices,
                max(3, int(dur / self.config.min_slice_interval_sec)))
        interval = dur / n
        slice_qty_base = qty // n
        remainder = qty - slice_qty_base * n

        slices = []
        for i in range(n):
            q = slice_qty_base + (1 if i < remainder else 0)
            if q <= 0:
                continue
            buf = self.config.limit_buffer_bps / 10_000.0
            lim = ref * (1 + buf) if side == "buy" else ref * (1 - buf)
            slices.append(OrderSlice(
                slice_id=i,
                qty=q,
                target_time_offset_sec=i * interval,
                limit_price=round(lim, 4),
            ))
        return slices

    def _build_vwap_slices(
        self, qty: int, dur: float, ref: float, side: str,
    ) -> List[OrderSlice]:
        """Volume-profile-weighted slices."""
        profile = np.array(self.config.vwap_profile, dtype=float)
        n = min(len(profile), self.config.max_slices,
                max(3, int(dur / self.config.min_slice_interval_sec)))
        profile = profile[:n]
        profile /= profile.sum()
        interval = dur / n

        slices = []
        remaining = qty
        for i in range(n):
            q = int(round(qty * profile[i]))
            q = min(q, remaining)
            if q <= 0 and i < n - 1:
                continue
            if i == n - 1:
                q = remaining  # sweep remainder
            remaining -= q
            buf = self.config.limit_buffer_bps / 10_000.0
            lim = ref * (1 + buf) if side == "buy" else ref * (1 - buf)
            slices.append(OrderSlice(
                slice_id=i,
                qty=q,
                target_time_offset_sec=i * interval,
                limit_price=round(lim, 4),
            ))
        return slices

    def _build_iceberg_slices(
        self, qty: int, dur: float, ref: float, side: str,
    ) -> List[OrderSlice]:
        """Iceberg: small visible clips, random noise on size."""
        remaining = qty
        slices = []
        t = 0.0
        interval = max(self.config.min_slice_interval_sec, dur / self.config.max_slices)
        idx = 0

        while remaining > 0 and idx < self.config.max_slices:
            show = max(1, int(remaining * self.config.iceberg_show_pct))
            noise = np.random.uniform(
                1 - self.config.iceberg_random_noise,
                1 + self.config.iceberg_random_noise,
            )
            show = max(1, int(show * noise))
            show = min(show, remaining)

            buf = self.config.limit_buffer_bps / 10_000.0
            lim = ref * (1 + buf) if side == "buy" else ref * (1 - buf)

            slices.append(OrderSlice(
                slice_id=idx,
                qty=show,
                target_time_offset_sec=t,
                limit_price=round(lim, 4),
            ))
            remaining -= show
            t += interval
            idx += 1

        return slices

    # ── Execution (simulation or real) ───────────────────────────────────

    def _execute_plan(self, plan: ExecutionPlan) -> ExecutionResult:
        """Execute all slices and produce a result."""
        plan.status = "active"
        total_cost = 0.0
        total_filled = 0

        for sl in plan.slices:
            fill_price = self._fill_slice(plan.symbol, sl, plan.side, plan.ref_price)
            sl.fill_price = fill_price
            sl.fill_qty = sl.qty
            sl.status = SliceStatus.FILLED
            sl.filled_at = datetime.now(timezone.utc).isoformat()

            sl_slip = self._calc_slippage(plan.ref_price, fill_price, plan.side)
            sl.slippage_bps = sl_slip

            total_cost += fill_price * sl.qty
            total_filled += sl.qty

        plan.status = "complete"

        avg_fill = total_cost / total_filled if total_filled > 0 else plan.ref_price
        real_slip = self._calc_slippage(plan.ref_price, avg_fill, plan.side)

        return ExecutionResult(
            plan_id=plan.plan_id,
            symbol=plan.symbol,
            side=plan.side,
            total_qty=plan.total_qty,
            filled_qty=total_filled,
            ref_price=plan.ref_price,
            avg_fill_price=round(avg_fill, 4),
            realised_slippage_bps=round(real_slip, 2),
            num_slices=len(plan.slices),
            strategy=plan.strategy.value,
            duration_sec=plan.duration_sec,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _fill_slice(
        self, symbol: str, sl: OrderSlice, side: str, ref_price: float,
    ) -> float:
        """Fill a single slice — real or simulated."""
        if self._submit_fn:
            try:
                return float(self._submit_fn(symbol, sl.qty, side, sl.limit_price))
            except Exception as e:
                logger.error("Submit failed for slice %d: %s", sl.slice_id, e)
                return ref_price

        # ── Simulation ──
        # Random walk around limit price with small market-impact noise
        noise = np.random.normal(0, ref_price * 0.0002)
        impact = ref_price * 0.0001 * (1 if side == "buy" else -1)
        fill = sl.limit_price + noise + impact
        return round(fill, 4)

    @staticmethod
    def _calc_slippage(ref: float, fill: float, side: str) -> float:
        """Slippage in bps (positive = unfavourable)."""
        if ref <= 0:
            return 0.0
        if side == "buy":
            return (fill - ref) / ref * 10_000
        return (ref - fill) / ref * 10_000


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    opt = ExecutionOptimizer()

    # TWAP
    r1 = opt.execute_optimal("AAPL", 500, "buy", 185.0, urgency=0.3)
    print(f"\nTWAP: filled {r1.filled_qty}/{r1.total_qty}, "
          f"avg {r1.avg_fill_price:.2f}, slip {r1.realised_slippage_bps:.1f} bps, "
          f"{r1.num_slices} slices ({r1.strategy})")

    # VWAP
    r2 = opt.execute_optimal("MSFT", 1000, "sell", 410.0, urgency=0.5,
                              strategy=ExecStrategy.VWAP)
    print(f"VWAP: filled {r2.filled_qty}/{r2.total_qty}, "
          f"avg {r2.avg_fill_price:.2f}, slip {r2.realised_slippage_bps:.1f} bps, "
          f"{r2.num_slices} slices ({r2.strategy})")

    # Iceberg (large order)
    r3 = opt.execute_optimal("NVDA", 5000, "buy", 700.0, urgency=0.2,
                              avg_daily_volume=50_000_000,
                              strategy=ExecStrategy.ICEBERG)
    print(f"ICE : filled {r3.filled_qty}/{r3.total_qty}, "
          f"avg {r3.avg_fill_price:.2f}, slip {r3.realised_slippage_bps:.1f} bps, "
          f"{r3.num_slices} slices ({r3.strategy})")

    # Cost estimate
    est = opt.estimate_cost(1000, avg_daily_volume=2_000_000, volatility=0.025, urgency=0.6)
    print(f"\nEstimated cost for 1000 shares: {est:.1f} bps")
