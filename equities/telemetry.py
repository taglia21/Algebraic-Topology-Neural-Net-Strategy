"""
equities/telemetry.py
======================
Telemetry collection for promotion gate evidence.

This module provides a central telemetry collector for metrics required by
promotion gates: paper->live and live scale-up.

Metrics collected:
- Trading days (unique dates)
- Slippage (realized vs. modeled)
- Order rejections
- Reconciliation mismatches
- Circuit breaker halts (by reason)
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TelemetrySnapshot:
    """A single point-in-time telemetry snapshot."""

    timestamp: str
    trading_days_count: int
    mean_realized_slippage_bps: float
    total_orders_submitted: int
    orders_rejected: int
    rejection_rate: float
    unresolved_reconciliation_mismatches: int
    software_defect_halts: int
    total_halts: int

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return asdict(self)


class TelemetryCollector:
    """Central collector for trading telemetry metrics."""

    def __init__(self) -> None:
        # Trading days: set of unique date strings (YYYY-MM-DD)
        self.trading_days: set[str] = set()

        # Slippage: list of realized slippages in bps
        self.slippage_samples: List[float] = []

        # Orders
        self.orders_submitted: int = 0
        self.orders_rejected: int = 0
        self.orders_rejected_list: List[dict] = []  # [{"symbol": "...", "qty": ..., "reason": "..."}]

        # Reconciliation
        self.reconciliation_mismatches_by_date: Dict[str, int] = defaultdict(int)

        # Circuit breaker halts
        self.halts: List[dict] = []  # [{"timestamp": "...", "reason": "...", "is_defect": bool}]

        # Snapshots (for archival)
        self.snapshots: List[TelemetrySnapshot] = []

    # ── Trading days ──

    def record_trading_day(self, date_str: str) -> None:
        """Record a trading day (YYYY-MM-DD format)."""
        self.trading_days.add(date_str)

    def get_trading_days_count(self) -> int:
        return len(self.trading_days)

    # ── Slippage ──

    def record_slippage(self, slippage_bps: float) -> None:
        """Record a single fill's slippage in basis points."""
        self.slippage_samples.append(float(slippage_bps))

    def get_mean_slippage_bps(self) -> float:
        """Return mean realized slippage across all fills."""
        if not self.slippage_samples:
            return 0.0
        return sum(self.slippage_samples) / len(self.slippage_samples)

    # ── Order rejections ──

    def record_order_submitted(self) -> None:
        """Increment order submission counter."""
        self.orders_submitted += 1

    def record_order_rejected(self, symbol: str, qty: int, reason: str = "") -> None:
        """Record a rejected order."""
        self.orders_rejected += 1
        self.orders_rejected_list.append(
            {"symbol": symbol, "qty": qty, "reason": reason, "timestamp": datetime.now(timezone.utc).isoformat()}
        )

    def get_rejection_rate(self) -> float:
        """Return order rejection rate (0.0 to 1.0)."""
        if self.orders_submitted == 0:
            return 0.0
        return self.orders_rejected / self.orders_submitted

    # ── Reconciliation ──

    def record_reconciliation_mismatch(self, count: int, date_str: Optional[str] = None) -> None:
        """Record count of unresolved reconciliation mismatches for a date."""
        if date_str is None:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.reconciliation_mismatches_by_date[date_str] = count

    def get_total_unresolved_mismatches(self) -> int:
        """Return total unresolved mismatches across all dates (max on any given day)."""
        if not self.reconciliation_mismatches_by_date:
            return 0
        return max(self.reconciliation_mismatches_by_date.values())

    # ── Circuit breaker halts ──

    def record_halt(self, reason: str, is_software_defect: bool = False) -> None:
        """Record a circuit breaker halt.

        Args:
            reason: Textual reason for halt (e.g., "drawdown exceeded", "error: API timeout")
            is_software_defect: True if halt was caused by a code bug (vs. legitimate risk condition)
        """
        self.halts.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reason": reason,
                "is_software_defect": is_software_defect,
            }
        )

    def get_software_defect_halts_count(self) -> int:
        """Return count of halts caused by software defects."""
        return sum(1 for h in self.halts if h.get("is_software_defect", False))

    def get_total_halts_count(self) -> int:
        """Return total halt count."""
        return len(self.halts)

    # ── Snapshots ──

    def take_snapshot(self) -> TelemetrySnapshot:
        """Create a point-in-time snapshot of current metrics."""
        snap = TelemetrySnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            trading_days_count=self.get_trading_days_count(),
            mean_realized_slippage_bps=self.get_mean_slippage_bps(),
            total_orders_submitted=self.orders_submitted,
            orders_rejected=self.orders_rejected,
            rejection_rate=self.get_rejection_rate(),
            unresolved_reconciliation_mismatches=self.get_total_unresolved_mismatches(),
            software_defect_halts=self.get_software_defect_halts_count(),
            total_halts=self.get_total_halts_count(),
        )
        self.snapshots.append(snap)
        return snap

    # ── Export ──

    def to_dict(self) -> dict:
        """Export current state as dict for JSON serialization."""
        return {
            "trading_days_count": self.get_trading_days_count(),
            "trading_days_list": sorted(list(self.trading_days)),
            "mean_realized_slippage_bps": self.get_mean_slippage_bps(),
            "slippage_samples": self.slippage_samples,
            "orders_submitted": self.orders_submitted,
            "orders_rejected": self.orders_rejected,
            "rejection_rate": self.get_rejection_rate(),
            "orders_rejected_list": self.orders_rejected_list,
            "unresolved_reconciliation_mismatches": self.get_total_unresolved_mismatches(),
            "reconciliation_mismatches_by_date": dict(self.reconciliation_mismatches_by_date),
            "software_defect_halts": self.get_software_defect_halts_count(),
            "total_halts": self.get_total_halts_count(),
            "halts": self.halts,
            "snapshots": [snap.to_dict() for snap in self.snapshots],
        }

    def export_json(self, path: str | Path) -> None:
        """Export telemetry to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        logger.info(f"Telemetry exported to {path}")

    def export_promotion_gate_evidence(
        self,
        path: str | Path,
        modeled_slippage_bps: float = 7.0,
        runbook_documented: bool = False,
    ) -> dict:
        """Export evidence for paper->live promotion gate.

        Args:
            path: Output JSON file path
            modeled_slippage_bps: Modeled slippage from config (for comparison)
            runbook_documented: Manual flag: True if runbook/rollback is documented

        Returns:
            Dict matching promotion_gate_evidence.paper_to_live schema
        """
        evidence = {
            "paper_trading_days": self.get_trading_days_count(),
            "modeled_slippage_bps": modeled_slippage_bps,
            "realized_slippage_bps": self.get_mean_slippage_bps(),
            "order_rejection_rate": self.get_rejection_rate(),
            "unresolved_reconciliation_mismatches": self.get_total_unresolved_mismatches(),
            "kill_switch_halts_due_to_software_defect": self.get_software_defect_halts_count(),
            "runbook_and_rollback_documented": runbook_documented,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(evidence, indent=2))
        logger.info(f"Promotion gate evidence exported to {path}")

        return evidence


# Global singleton instance
_global_telemetry: Optional[TelemetryCollector] = None


def get_telemetry() -> TelemetryCollector:
    """Get or create the global telemetry collector."""
    global _global_telemetry
    if _global_telemetry is None:
        _global_telemetry = TelemetryCollector()
    return _global_telemetry


def reset_telemetry() -> None:
    """Reset the global telemetry collector (for testing)."""
    global _global_telemetry
    _global_telemetry = None
