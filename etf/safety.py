"""
etf/safety.py
=============
ETF-native live-trading safety layer (Phase 5). Two concerns, both implemented
as **pure, dependency-free functions** so they are fully unit-testable without a
live IBKR connection:

1. **Pre-trade kill-switch** (:func:`pretrade_safety_check`) — a hard gate run
   before any rebalance. It HALTS (kill-switch) on catastrophic conditions
   (book drawdown beyond ``hard_halt_drawdown``, single-day loss beyond
   ``max_daily_loss``) and BLOCKS (skip this cycle, retry next) on recoverable
   problems (stale data, an unresolved reconciliation mismatch, gross exposure
   over the leverage cap). This is distinct from the smooth ``dd_derisk``
   overlay: the overlay gently scales exposure down; the kill-switch slams the
   brakes and requires a human reset.

2. **Slippage telemetry** (:func:`compute_slippage`) — after a fill, compare the
   realised price to the price the plan assumed and record adverse slippage in
   bps and dollars. The Paper→Live gate requires realised slippage to stay
   within 20% of the modeled assumption; this is how we measure it.

Design note: the ETF engine is intentionally self-contained (no dependency on
the equities/Alpaca ``core.kill_switch``), so this module stands alone.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from etf.config import ETFConfig

logger = logging.getLogger("etf.safety")


# ---------------------------------------------------------------------------
# Pre-trade kill-switch
# ---------------------------------------------------------------------------
@dataclass
class SafetyDecision:
    """Outcome of the pre-trade safety gate.

    Attributes
    ----------
    allowed :
        True only when it is safe to submit orders this cycle.
    halt :
        True when a CATASTROPHIC condition tripped the kill-switch — trading
        must stay stopped until a human resets (do not auto-resume next cycle).
    reasons :
        Human-readable explanations for every block/halt trigger (for the
        runbook and audit log). Empty when ``allowed`` is True.
    """

    allowed: bool
    halt: bool
    reasons: List[str] = field(default_factory=list)


def pretrade_safety_check(
    cfg: ETFConfig,
    *,
    current_drawdown: float,
    daily_pnl_pct: float,
    gross_exposure: float,
    reconciliation_ok: bool = True,
    data_is_fresh: bool = True,
) -> SafetyDecision:
    """Run the pre-trade kill-switch.

    Parameters
    ----------
    current_drawdown :
        Current book drawdown from peak, as a POSITIVE fraction (0.10 = −10%).
    daily_pnl_pct :
        Today's P&L as a SIGNED fraction of start-of-day equity
        (−0.05 = down 5% on the day).
    gross_exposure :
        Planned gross exposure (sum of |weights|) for the target book.
    reconciliation_ok :
        Result of the prior cycle's reconciliation. False means the live book
        drifted from intent and the drift is unresolved — do not trade on top of
        an inconsistent book.
    data_is_fresh :
        False if price/account data is stale or missing (fail-safe).

    Returns
    -------
    SafetyDecision
        ``allowed`` is True only if no trigger fired. ``halt`` is True if a
        catastrophic (kill-switch) trigger fired.
    """
    reasons: List[str] = []
    halt = False

    r = cfg.risk
    # --- Catastrophic halts (kill-switch — require human reset) ----------
    if current_drawdown >= r.hard_halt_drawdown:
        halt = True
        reasons.append(
            f"KILL-SWITCH: drawdown {current_drawdown:.1%} >= hard halt "
            f"{r.hard_halt_drawdown:.1%}"
        )
    if daily_pnl_pct <= -r.max_daily_loss:
        halt = True
        reasons.append(
            f"KILL-SWITCH: daily loss {daily_pnl_pct:.1%} <= -{r.max_daily_loss:.1%}"
        )

    # --- Recoverable blocks (skip this cycle, retry next) ----------------
    if not data_is_fresh:
        reasons.append("BLOCK: market/account data is stale or missing (fail-safe)")
    if not reconciliation_ok:
        reasons.append("BLOCK: unresolved reconciliation mismatch from prior cycle")
    # Allow a tiny float tolerance so a book exactly at the cap is not rejected.
    cap = r.max_gross_leverage * 1.001
    if gross_exposure > cap:
        reasons.append(
            f"BLOCK: gross exposure {gross_exposure:.2f} exceeds leverage cap "
            f"{r.max_gross_leverage:.2f}"
        )

    allowed = len(reasons) == 0
    return SafetyDecision(allowed=allowed, halt=halt, reasons=reasons)


# ---------------------------------------------------------------------------
# Slippage telemetry
# ---------------------------------------------------------------------------
@dataclass
class SlippageRecord:
    symbol: str
    action: str            # "BUY" / "SELL"
    expected_price: float  # price the plan assumed
    fill_price: float      # realised fill
    quantity: int
    slippage_bps: float    # SIGNED: positive = adverse (cost), negative = price improvement
    cost_usd: float        # signed dollar slippage vs expected


@dataclass
class SlippageReport:
    records: List[SlippageRecord]
    total_notional: float
    total_cost_usd: float          # signed; positive = net adverse slippage
    avg_slippage_bps: float        # notional-weighted, signed
    worst_slippage_bps: float      # most adverse single fill (max signed)
    within_tolerance: bool         # avg adverse slippage within modeled budget


def _adverse_slippage_bps(action: str, expected: float, fill: float) -> float:
    """Signed slippage in bps where POSITIVE means we paid worse than expected.

    BUY  adverse when fill > expected -> (fill - expected) / expected
    SELL adverse when fill < expected -> (expected - fill) / expected
    """
    if expected <= 0:
        raise ValueError("expected price must be positive")
    if action.upper() == "BUY":
        rel = (fill - expected) / expected
    else:
        rel = (expected - fill) / expected
    return rel * 1e4


def compute_slippage(
    orders: List["object"],
    fills: Dict[str, float],
    cfg: ETFConfig,
) -> SlippageReport:
    """Compare realised fills to the plan's expected prices.

    Parameters
    ----------
    orders :
        Planned orders (objects exposing ``symbol``, ``action``, ``quantity``
        and ``est_price`` — i.e. ``etf.ibkr_broker.PlannedOrder``). Kept duck-typed
        to avoid a circular import.
    fills :
        Realised fill price per symbol. Symbols missing from ``fills`` are
        skipped (treated as un-filled / cancelled — they did not cost slippage).
    cfg :
        Provides ``execution.slippage_bps`` as the modeled budget. Realised
        average adverse slippage within ``1.20 ×`` that budget passes the gate.

    Returns
    -------
    SlippageReport
    """
    records: List[SlippageRecord] = []
    total_notional = 0.0
    total_cost = 0.0
    weighted_bps = 0.0
    worst = float("-inf")

    for o in orders:
        fill = fills.get(o.symbol)
        if fill is None or fill <= 0:
            continue
        expected = float(o.est_price)
        qty = int(o.quantity)
        bps = _adverse_slippage_bps(o.action, expected, fill)
        notional = expected * qty
        cost = (bps / 1e4) * notional
        records.append(SlippageRecord(
            symbol=o.symbol,
            action=o.action,
            expected_price=expected,
            fill_price=float(fill),
            quantity=qty,
            slippage_bps=float(bps),
            cost_usd=float(cost),
        ))
        total_notional += notional
        total_cost += cost
        weighted_bps += bps * notional
        worst = max(worst, bps)

    avg_bps = (weighted_bps / total_notional) if total_notional > 0 else 0.0
    worst_bps = worst if records else 0.0
    # Gate: realised adverse slippage within 20% of the modeled assumption.
    budget = cfg.execution.slippage_bps * 1.20
    within = avg_bps <= budget

    return SlippageReport(
        records=records,
        total_notional=float(total_notional),
        total_cost_usd=float(total_cost),
        avg_slippage_bps=float(avg_bps),
        worst_slippage_bps=float(worst_bps),
        within_tolerance=bool(within),
    )


def log_slippage(report: SlippageReport, path: str | Path, *, as_of: str = "") -> None:
    """Append a slippage report to a JSONL telemetry file (one line per cycle).

    Best-effort: any I/O error is logged, never raised, so telemetry can never
    break the trading loop.
    """
    try:
        ts = as_of or datetime.now(timezone.utc).isoformat()
        row = {
            "as_of": ts,
            "n_fills": len(report.records),
            "total_notional": report.total_notional,
            "total_cost_usd": report.total_cost_usd,
            "avg_slippage_bps": report.avg_slippage_bps,
            "worst_slippage_bps": report.worst_slippage_bps,
            "within_tolerance": report.within_tolerance,
            "fills": [
                {
                    "symbol": r.symbol,
                    "action": r.action,
                    "expected": r.expected_price,
                    "fill": r.fill_price,
                    "qty": r.quantity,
                    "slippage_bps": r.slippage_bps,
                }
                for r in report.records
            ],
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
    except Exception as exc:  # pragma: no cover - telemetry must never crash trading
        logger.error("Failed to write slippage telemetry: %s", exc)
