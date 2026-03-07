"""
core/reconciliation.py
======================
Position reconciliation between the internal execution engine and the
live broker (Alpaca).

Run after every trading cycle (or on a schedule) to detect and resolve
discrepancies between what the system *thinks* it holds and what the
broker *actually* reports.

Reconciliation Modes
--------------------
- **Soft**: Log discrepancies only — no automatic corrections.
- **Hard**: Automatically adjust internal state to match broker truth,
  and optionally submit corrective orders.

Usage
-----
    from core.reconciliation import Reconciler

    reconciler = Reconciler(broker=alpaca_broker, mode="soft")
    report = reconciler.reconcile(internal_positions)
    if report.has_discrepancies:
        logger.warning(f"Discrepancies found: {report.summary()}")
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List

from equities.models import Position

logger = logging.getLogger(__name__)


@dataclass
class Discrepancy:
    """A single position discrepancy between internal and broker state.

    Attributes
    ----------
    symbol :
        Ticker symbol.
    internal_qty :
        Quantity the system thinks it holds.
    broker_qty :
        Quantity the broker reports.
    internal_avg_entry :
        Internal average entry price.
    broker_avg_entry :
        Broker-reported average entry price.
    discrepancy_type :
        ``"qty_mismatch"``, ``"missing_internal"``, ``"missing_broker"``,
        or ``"entry_price_mismatch"``.
    """

    symbol: str
    internal_qty: int
    broker_qty: int
    internal_avg_entry: float
    broker_avg_entry: float
    discrepancy_type: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def qty_delta(self) -> int:
        """broker_qty − internal_qty; positive means broker has more."""
        return self.broker_qty - self.internal_qty


@dataclass
class ReconciliationReport:
    """Result of a reconciliation run.

    Attributes
    ----------
    timestamp :
        When reconciliation was performed.
    discrepancies :
        List of individual discrepancies found.
    symbols_checked :
        Total symbols compared.
    mode :
        ``"soft"`` or ``"hard"``.
    corrections_applied :
        Number of automatic corrections (hard mode only).
    """

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    discrepancies: List[Discrepancy] = field(default_factory=list)
    symbols_checked: int = 0
    mode: str = "soft"
    corrections_applied: int = 0

    @property
    def has_discrepancies(self) -> bool:
        return len(self.discrepancies) > 0

    @property
    def is_clean(self) -> bool:
        return not self.has_discrepancies

    def summary(self) -> str:
        """Human-readable summary of the reconciliation."""
        if self.is_clean:
            return (
                f"Reconciliation CLEAN: {self.symbols_checked} positions "
                f"match broker state."
            )

        lines = [
            f"Reconciliation MISMATCH: {len(self.discrepancies)} "
            f"discrepancies in {self.symbols_checked} symbols."
        ]
        for d in self.discrepancies:
            lines.append(
                f"  {d.symbol}: internal={d.internal_qty} vs broker={d.broker_qty} "
                f"({d.discrepancy_type}, delta={d.qty_delta:+d})"
            )
        if self.corrections_applied > 0:
            lines.append(f"  Corrections applied: {self.corrections_applied}")
        return "\n".join(lines)


class Reconciler:
    """Position reconciliation engine.

    Parameters
    ----------
    broker :
        A :class:`~equities.execution.Broker` implementation that provides
        ``get_positions()`` for the broker's ground truth.
    mode :
        ``"soft"`` (log only) or ``"hard"`` (auto-correct internal state).
    qty_tolerance :
        Allowable share quantity difference before flagging a discrepancy.
        Default 0 (exact match required).
    price_tolerance_pct :
        Allowable average entry price difference (as fraction) before
        flagging. Default 0.01 (1%).
    """

    def __init__(
        self,
        broker,
        mode: str = "soft",
        qty_tolerance: float = 0,
        price_tolerance_pct: float = 0.01,
    ) -> None:
        self._broker = broker
        self._mode = mode
        self._qty_tolerance = qty_tolerance
        self._price_tolerance_pct = price_tolerance_pct
        self._lock = threading.Lock()

        logger.info(f"Reconciler initialised: mode={mode}")

    def reconcile(
        self,
        internal_positions: Dict[str, Position],
    ) -> ReconciliationReport:
        """Compare internal positions against broker truth.

        Parameters
        ----------
        internal_positions :
            The system's internal position map (symbol → Position).

        Returns
        -------
        ReconciliationReport
        """
        report = ReconciliationReport(mode=self._mode)

        # Fetch broker positions
        try:
            broker_positions = self._broker.get_positions()
        except Exception as exc:
            logger.error(f"Reconciler: failed to fetch broker positions: {exc}")
            report.discrepancies.append(
                Discrepancy(
                    symbol="SYSTEM",
                    internal_qty=0,
                    broker_qty=0,
                    internal_avg_entry=0.0,
                    broker_avg_entry=0.0,
                    discrepancy_type="broker_fetch_failed",
                )
            )
            return report

        # All symbols in either set
        all_symbols = set(internal_positions.keys()) | set(broker_positions.keys())
        report.symbols_checked = len(all_symbols)

        for symbol in sorted(all_symbols):
            internal = internal_positions.get(symbol)
            broker = broker_positions.get(symbol)

            int_qty = internal.qty if internal else 0
            brk_qty = broker.qty if broker else 0
            int_entry = internal.avg_entry if internal else 0.0
            brk_entry = broker.avg_entry if broker else 0.0

            # Check quantity
            if abs(int_qty - brk_qty) > self._qty_tolerance:
                if internal is None:
                    disc_type = "missing_internal"
                elif broker is None:
                    disc_type = "missing_broker"
                else:
                    disc_type = "qty_mismatch"

                disc = Discrepancy(
                    symbol=symbol,
                    internal_qty=int_qty,
                    broker_qty=brk_qty,
                    internal_avg_entry=int_entry,
                    broker_avg_entry=brk_entry,
                    discrepancy_type=disc_type,
                )
                report.discrepancies.append(disc)

                logger.warning(
                    f"Reconciler: {disc_type} on {symbol} — "
                    f"internal={int_qty}, broker={brk_qty}"
                )
                continue

            # Check entry price (only if both exist and qty matches)
            if internal and broker and int_qty != 0:
                if int_entry > 0 and brk_entry > 0:
                    price_diff = abs(int_entry - brk_entry) / brk_entry
                    if price_diff > self._price_tolerance_pct:
                        report.discrepancies.append(
                            Discrepancy(
                                symbol=symbol,
                                internal_qty=int_qty,
                                broker_qty=brk_qty,
                                internal_avg_entry=int_entry,
                                broker_avg_entry=brk_entry,
                                discrepancy_type="entry_price_mismatch",
                            )
                        )

        # Hard mode: apply corrections (under lock to prevent concurrent mutation)
        if self._mode == "hard" and report.has_discrepancies:
            with self._lock:
                report.corrections_applied = self._apply_corrections(
                    internal_positions, broker_positions, report.discrepancies
                )

        # Log summary
        log_fn = logger.info if report.is_clean else logger.warning
        log_fn(report.summary())

        return report

    def _apply_corrections(
        self,
        internal_positions: Dict[str, Position],
        broker_positions: Dict[str, Position],
        discrepancies: List[Discrepancy],
    ) -> int:
        """Apply corrections to internal state to match broker truth.

        In hard mode, the broker is the source of truth.  We update the
        internal position map to match.

        Parameters
        ----------
        internal_positions :
            Mutable internal position dict.
        broker_positions :
            Broker ground truth.
        discrepancies :
            Detected discrepancies.

        Returns
        -------
        int
            Number of corrections applied.
        """
        corrections = 0
        for disc in discrepancies:
            symbol = disc.symbol
            broker_pos = broker_positions.get(symbol)

            if disc.discrepancy_type == "missing_internal" and broker_pos:
                # Broker has position we don't track — adopt it
                internal_positions[symbol] = Position(
                    symbol=symbol,
                    qty=broker_pos.qty,
                    avg_entry=broker_pos.avg_entry,
                    current_price=broker_pos.current_price,
                    unrealized_pnl=broker_pos.unrealized_pnl,
                    strategy="reconciled",
                )
                corrections += 1
                logger.info(f"Reconciler: adopted broker position for {symbol}")

            elif disc.discrepancy_type == "missing_broker":
                # We think we have a position the broker doesn't — remove it
                if symbol in internal_positions:
                    del internal_positions[symbol]
                    corrections += 1
                    logger.info(f"Reconciler: removed phantom position for {symbol}")

            elif disc.discrepancy_type == "qty_mismatch" and broker_pos:
                # Qty differs — trust the broker
                if symbol in internal_positions:
                    internal_positions[symbol].qty = broker_pos.qty
                    internal_positions[symbol].avg_entry = broker_pos.avg_entry
                    corrections += 1
                    logger.info(
                        f"Reconciler: corrected {symbol} qty "
                        f"{disc.internal_qty} → {disc.broker_qty}"
                    )

        return corrections
