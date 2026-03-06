"""
Phase Q — Post-Trade Reconciliation.

Item 15: PostTradeReconciler — OMS vs IBKR fills, discrepancy > $1 flag, > $100 Discord alert.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class FillRecord:
    """A single fill record."""
    order_id: str = ""
    symbol: str = ""
    side: str = ""
    quantity: int = 0
    price: float = 0.0
    timestamp: str = ""
    source: str = ""  # "oms" or "broker"


@dataclass
class Discrepancy:
    """A reconciliation discrepancy."""
    order_id: str = ""
    symbol: str = ""
    field: str = ""  # "price", "quantity", "missing"
    oms_value: float = 0.0
    broker_value: float = 0.0
    difference: float = 0.0
    severity: str = "info"  # "info", "warning", "critical"
    requires_alert: bool = False


@dataclass
class ReconciliationReport:
    """Reconciliation report."""
    matched: int = 0
    discrepancies: List[Discrepancy] = field(default_factory=list)
    missing_in_broker: List[str] = field(default_factory=list)  # order_ids
    missing_in_oms: List[str] = field(default_factory=list)
    total_oms_fills: int = 0
    total_broker_fills: int = 0
    max_price_diff: float = 0.0
    alerts_triggered: int = 0
    timestamp: str = ""


class PostTradeReconciler:
    """Reconcile OMS fills with broker (IBKR) fills.

    - Discrepancy > $1: FLAG for review.
    - Discrepancy > $100: ALERT via Discord webhook.
    """

    def __init__(
        self,
        flag_threshold: float = 1.0,
        alert_threshold: float = 100.0,
        report_path: str = "logs/reconciliation.json",
        discord_webhook_url: Optional[str] = None,
    ):
        """
        Args:
            flag_threshold: Dollar threshold to flag discrepancy.
            alert_threshold: Dollar threshold to trigger Discord alert.
            report_path: Path to save reconciliation reports.
            discord_webhook_url: Discord webhook for critical alerts.
        """
        self.flag_threshold = flag_threshold
        self.alert_threshold = alert_threshold
        self.report_path = report_path
        self.discord_webhook_url = discord_webhook_url
        self._reports: List[ReconciliationReport] = []

    def reconcile(
        self,
        oms_fills: List[FillRecord],
        broker_fills: List[FillRecord],
    ) -> ReconciliationReport:
        """Reconcile OMS fills against broker fills.

        Matches by order_id. Checks price and quantity differences.

        Args:
            oms_fills: Fills from internal OMS.
            broker_fills: Fills from broker (IBKR).

        Returns:
            ReconciliationReport with all discrepancies.
        """
        # Index by order_id
        oms_by_id = {f.order_id: f for f in oms_fills}
        broker_by_id = {f.order_id: f for f in broker_fills}

        report = ReconciliationReport(
            total_oms_fills=len(oms_fills),
            total_broker_fills=len(broker_fills),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        all_ids = set(oms_by_id.keys()) | set(broker_by_id.keys())

        for oid in all_ids:
            oms_fill = oms_by_id.get(oid)
            broker_fill = broker_by_id.get(oid)

            if oms_fill and not broker_fill:
                report.missing_in_broker.append(oid)
                report.discrepancies.append(Discrepancy(
                    order_id=oid,
                    symbol=oms_fill.symbol,
                    field="missing",
                    oms_value=oms_fill.price,
                    broker_value=0.0,
                    difference=oms_fill.price * oms_fill.quantity,
                    severity="critical",
                    requires_alert=True,
                ))
                continue

            if broker_fill and not oms_fill:
                report.missing_in_oms.append(oid)
                report.discrepancies.append(Discrepancy(
                    order_id=oid,
                    symbol=broker_fill.symbol,
                    field="missing",
                    oms_value=0.0,
                    broker_value=broker_fill.price,
                    difference=broker_fill.price * broker_fill.quantity,
                    severity="critical",
                    requires_alert=True,
                ))
                continue

            # Both exist — compare
            assert oms_fill is not None and broker_fill is not None

            # Price discrepancy
            price_diff = abs(oms_fill.price * oms_fill.quantity - broker_fill.price * broker_fill.quantity)

            if price_diff > self.flag_threshold:
                severity = "warning"
                requires_alert = False

                if price_diff > self.alert_threshold:
                    severity = "critical"
                    requires_alert = True
                    report.alerts_triggered += 1

                report.discrepancies.append(Discrepancy(
                    order_id=oid,
                    symbol=oms_fill.symbol,
                    field="notional",
                    oms_value=oms_fill.price * oms_fill.quantity,
                    broker_value=broker_fill.price * broker_fill.quantity,
                    difference=price_diff,
                    severity=severity,
                    requires_alert=requires_alert,
                ))
                report.max_price_diff = max(report.max_price_diff, price_diff)
            else:
                report.matched += 1

            # Quantity discrepancy
            qty_diff = abs(oms_fill.quantity - broker_fill.quantity)
            if qty_diff > 0:
                report.discrepancies.append(Discrepancy(
                    order_id=oid,
                    symbol=oms_fill.symbol,
                    field="quantity",
                    oms_value=float(oms_fill.quantity),
                    broker_value=float(broker_fill.quantity),
                    difference=float(qty_diff),
                    severity="warning",
                ))

        # Send alerts for critical discrepancies
        critical = [d for d in report.discrepancies if d.requires_alert]
        if critical and self.discord_webhook_url:
            self._send_discord_alert(critical)

        self._reports.append(report)

        logger.info(
            "Reconciliation: %d matched, %d discrepancies, %d alerts",
            report.matched, len(report.discrepancies), report.alerts_triggered,
        )
        return report

    def _send_discord_alert(self, discrepancies: List[Discrepancy]) -> None:
        """Send Discord alert for critical discrepancies."""
        try:
            import urllib.request

            message = "🚨 **RECONCILIATION ALERT** 🚨\n"
            for d in discrepancies[:5]:
                message += (
                    f"• {d.symbol} ({d.field}): OMS=${d.oms_value:,.2f} vs "
                    f"Broker=${d.broker_value:,.2f} (diff=${d.difference:,.2f})\n"
                )

            payload = json.dumps({"content": message}).encode("utf-8")
            req = urllib.request.Request(
                self.discord_webhook_url,  # type: ignore
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5)
            logger.info("Discord reconciliation alert sent")
        except Exception as e:
            logger.error("Discord alert failed: %s", e)

    def save_report(self, report: ReconciliationReport) -> None:
        """Save reconciliation report to JSON."""
        path = Path(self.report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "matched": report.matched,
            "discrepancies": [
                {
                    "order_id": d.order_id,
                    "symbol": d.symbol,
                    "field": d.field,
                    "oms_value": d.oms_value,
                    "broker_value": d.broker_value,
                    "difference": d.difference,
                    "severity": d.severity,
                }
                for d in report.discrepancies
            ],
            "missing_in_broker": report.missing_in_broker,
            "missing_in_oms": report.missing_in_oms,
            "alerts_triggered": report.alerts_triggered,
            "timestamp": report.timestamp,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @property
    def reports(self) -> List[ReconciliationReport]:
        return self._reports
