"""
etf/state.py
============
Persistent equity-state tracking for the live ETF engine (Phase 5).

The kill-switch in :mod:`etf.safety` needs two live inputs it cannot derive from
a single snapshot:

- **current_drawdown** — how far below the running peak equity the book is.
- **daily_pnl_pct**    — today's P&L vs the start-of-day equity.

Both require *memory across cycles*, which must survive process restarts (a live
trader is long-running and will be bounced for deploys). This module persists the
running peak and the start-of-day baseline to a small JSON file, written
atomically so a crash mid-write can never corrupt it.

The core :func:`update_state` is a **pure function** (state in -> state + derived
metrics out) so the drawdown / daily-P&L logic is fully unit-testable without any
broker or filesystem.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger("etf.state")


@dataclass
class EquityState:
    """Persisted equity memory.

    Attributes
    ----------
    peak_equity :
        Highest equity ever observed (high-water mark) — denominator of drawdown.
    sod_equity :
        Start-of-day equity baseline — denominator of daily P&L.
    sod_date :
        ISO date (YYYY-MM-DD) the ``sod_equity`` baseline belongs to.
    last_equity :
        Most recently observed equity (becomes tomorrow's start-of-day baseline).
    updated_at :
        ISO-8601 UTC timestamp of the last update (audit trail).
    """

    peak_equity: float
    sod_equity: float
    sod_date: str
    last_equity: float
    updated_at: str


def update_state(
    prev: Optional[EquityState],
    equity: float,
    *,
    now: Optional[datetime] = None,
) -> Tuple[EquityState, float, float]:
    """Advance the equity state with a fresh equity reading.

    Parameters
    ----------
    prev :
        Previous persisted state, or None on first ever observation.
    equity :
        Current net-liquidation equity from the broker snapshot.
    now :
        Override for the current time (testing). Defaults to UTC now.

    Returns
    -------
    (state, current_drawdown, daily_pnl_pct)
        ``current_drawdown`` is a POSITIVE fraction (0.10 == −10% from peak).
        ``daily_pnl_pct`` is SIGNED (−0.05 == down 5% on the day). Both feed the
        kill-switch directly.
    """
    if equity <= 0:
        raise ValueError("equity must be positive")
    now = now or datetime.now(timezone.utc)
    today = now.date().isoformat()

    if prev is None:
        # First observation: no history -> no drawdown, no daily move yet.
        state = EquityState(
            peak_equity=equity,
            sod_equity=equity,
            sod_date=today,
            last_equity=equity,
            updated_at=now.isoformat(),
        )
        return state, 0.0, 0.0

    peak = max(prev.peak_equity, equity)
    # Roll the start-of-day baseline when the calendar day changes: the baseline
    # becomes the LAST equity observed before today (prior close).
    if today != prev.sod_date:
        sod_equity = prev.last_equity
        sod_date = today
    else:
        sod_equity = prev.sod_equity
        sod_date = prev.sod_date

    drawdown = 1.0 - equity / peak if peak > 0 else 0.0
    daily_pnl_pct = (equity / sod_equity - 1.0) if sod_equity > 0 else 0.0

    state = EquityState(
        peak_equity=float(peak),
        sod_equity=float(sod_equity),
        sod_date=sod_date,
        last_equity=float(equity),
        updated_at=now.isoformat(),
    )
    return state, float(max(0.0, drawdown)), float(daily_pnl_pct)


def load_state(path: str | Path) -> Optional[EquityState]:
    """Load persisted state. Returns None if absent or unreadable (fail-safe)."""
    try:
        p = Path(path)
        if not p.exists():
            return None
        data = json.loads(p.read_text())
        return EquityState(
            peak_equity=float(data["peak_equity"]),
            sod_equity=float(data["sod_equity"]),
            sod_date=str(data["sod_date"]),
            last_equity=float(data["last_equity"]),
            updated_at=str(data.get("updated_at", "")),
        )
    except Exception as exc:
        logger.error("Failed to load equity state from %s: %s", path, exc)
        return None


def save_state(state: EquityState, path: str | Path) -> None:
    """Persist state atomically (temp file + os.replace) so a crash mid-write
    cannot corrupt the file. Best-effort: errors are logged, never raised, so
    state I/O can never break the trading loop."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(asdict(state), fh, indent=2)
            os.replace(tmp, p)  # atomic on POSIX
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    except Exception as exc:  # pragma: no cover - state I/O must never crash trading
        logger.error("Failed to save equity state to %s: %s", path, exc)


# ===========================================================================
# Reconciliation state (Phase 5 — cross-cycle safety memory)
# ===========================================================================
@dataclass
class ReconciliationState:
    """Persisted outcome of the most recent post-trade reconciliation.

    The pre-trade kill-switch BLOCKS a new cycle when the *prior* cycle left an
    unresolved reconciliation mismatch (the live book drifted from intent), so
    we never trade on top of an inconsistent book. That decision needs memory
    across process restarts, which this small file provides.

    Attributes
    ----------
    ok :
        True if the last reconciliation matched target within tolerance.
    as_of :
        ISO-8601 timestamp of the reconciliation that produced this state.
    mismatches :
        symbol -> |realised_w - target_w| for any drift beyond tolerance
        (empty when ``ok``). Carried for the runbook / audit trail.
    updated_at :
        ISO-8601 UTC timestamp of the last write.
    """

    ok: bool
    as_of: str
    mismatches: dict
    updated_at: str


def load_reconciliation_state(path: str | Path) -> Optional[ReconciliationState]:
    """Load the persisted reconciliation state.

    Returns None if the file is absent or unreadable. A missing file means "no
    prior cycle" — the caller should treat that as reconciled (nothing to block
    on yet), NOT as a mismatch, so a fresh deployment can trade.
    """
    try:
        p = Path(path)
        if not p.exists():
            return None
        data = json.loads(p.read_text())
        return ReconciliationState(
            ok=bool(data["ok"]),
            as_of=str(data.get("as_of", "")),
            mismatches=dict(data.get("mismatches", {})),
            updated_at=str(data.get("updated_at", "")),
        )
    except Exception as exc:
        logger.error("Failed to load reconciliation state from %s: %s", path, exc)
        return None


def save_reconciliation_state(
    ok: bool,
    mismatches: Optional[dict],
    path: str | Path,
    *,
    as_of: str = "",
    now: Optional[datetime] = None,
) -> None:
    """Persist the reconciliation outcome atomically.

    Best-effort: errors are logged, never raised, so reconciliation bookkeeping
    can never break the trading loop.
    """
    try:
        now = now or datetime.now(timezone.utc)
        state = ReconciliationState(
            ok=bool(ok),
            as_of=as_of or now.isoformat(),
            mismatches={str(k): float(v) for k, v in (mismatches or {}).items()},
            updated_at=now.isoformat(),
        )
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(asdict(state), fh, indent=2)
            os.replace(tmp, p)  # atomic on POSIX
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    except Exception as exc:  # pragma: no cover - state I/O must never crash trading
        logger.error("Failed to save reconciliation state to %s: %s", path, exc)

