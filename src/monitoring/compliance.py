"""
Phase T — Compliance Engine & Regulatory Reporter.

Item 21: ComplianceEngine — position/sector concentration, gross/net exposure, options notional.
Item 22: RegulatoryReporter — Form PF-style, large trader, TCA summary.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Item 21 — ComplianceEngine
# ---------------------------------------------------------------------------

@dataclass
class ComplianceViolation:
    """A compliance rule violation."""
    rule: str
    description: str
    severity: str  # "warning", "breach", "critical"
    current_value: float
    limit: float
    symbol: Optional[str] = None
    timestamp: str = ""


@dataclass
class ComplianceReport:
    """Compliance check result."""
    compliant: bool = True
    violations: List[ComplianceViolation] = field(default_factory=list)
    checks_run: int = 0
    position_concentration_max: float = 0.0
    sector_concentration_max: float = 0.0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    options_notional: float = 0.0
    timestamp: str = ""


class ComplianceEngine:
    """Portfolio compliance engine.

    Rules:
      1. Single position concentration: max 15% of NAV.
      2. Sector concentration: max 30% of NAV.
      3. Gross exposure: max 200% of NAV.
      4. Net exposure: between -30% and +130% of NAV.
      5. Options notional exposure: max 50% of NAV.
    """

    def __init__(
        self,
        max_position_pct: float = 0.15,
        max_sector_pct: float = 0.30,
        max_gross_exposure: float = 2.00,
        net_exposure_range: tuple = (-0.30, 1.30),
        max_options_notional_pct: float = 0.50,
    ):
        self.max_position_pct = max_position_pct
        self.max_sector_pct = max_sector_pct
        self.max_gross_exposure = max_gross_exposure
        self.net_exposure_range = net_exposure_range
        self.max_options_notional_pct = max_options_notional_pct
        self._violations_log: List[ComplianceViolation] = []

    def check(
        self,
        positions: Dict[str, float],
        nav: float,
        sector_map: Optional[Dict[str, str]] = None,
        options_notional: float = 0.0,
    ) -> ComplianceReport:
        """Run all compliance checks.

        Args:
            positions: Dict of symbol → position value (positive=long, negative=short).
            nav: Net Asset Value.
            sector_map: Dict of symbol → sector name.
            options_notional: Total options notional exposure.

        Returns:
            ComplianceReport with any violations.
        """
        violations = []
        checks = 0
        now = datetime.now(timezone.utc).isoformat()

        if nav <= 0:
            return ComplianceReport(timestamp=now)

        # 1. Position concentration
        checks += 1
        max_pos_conc = 0.0
        for sym, val in positions.items():
            conc = abs(val) / nav
            max_pos_conc = max(max_pos_conc, conc)
            if conc > self.max_position_pct:
                violations.append(ComplianceViolation(
                    rule="position_concentration",
                    description=f"{sym} concentration {conc:.1%} exceeds {self.max_position_pct:.0%}",
                    severity="breach",
                    current_value=conc,
                    limit=self.max_position_pct,
                    symbol=sym,
                    timestamp=now,
                ))

        # 2. Sector concentration
        checks += 1
        max_sec_conc = 0.0
        if sector_map:
            sector_exposures: Dict[str, float] = {}
            for sym, val in positions.items():
                sector = sector_map.get(sym, "unknown")
                sector_exposures[sector] = sector_exposures.get(sector, 0) + abs(val)

            for sector, exp in sector_exposures.items():
                conc = exp / nav
                max_sec_conc = max(max_sec_conc, conc)
                if conc > self.max_sector_pct:
                    violations.append(ComplianceViolation(
                        rule="sector_concentration",
                        description=f"Sector '{sector}' at {conc:.1%} exceeds {self.max_sector_pct:.0%}",
                        severity="breach",
                        current_value=conc,
                        limit=self.max_sector_pct,
                        timestamp=now,
                    ))

        # 3. Gross exposure
        checks += 1
        gross = sum(abs(v) for v in positions.values()) / nav
        if gross > self.max_gross_exposure:
            violations.append(ComplianceViolation(
                rule="gross_exposure",
                description=f"Gross exposure {gross:.1%} exceeds {self.max_gross_exposure:.0%}",
                severity="critical",
                current_value=gross,
                limit=self.max_gross_exposure,
                timestamp=now,
            ))

        # 4. Net exposure
        checks += 1
        net = sum(positions.values()) / nav
        lo, hi = self.net_exposure_range
        if net < lo or net > hi:
            violations.append(ComplianceViolation(
                rule="net_exposure",
                description=f"Net exposure {net:.1%} outside [{lo:.0%}, {hi:.0%}]",
                severity="breach",
                current_value=net,
                limit=hi if net > hi else lo,
                timestamp=now,
            ))

        # 5. Options notional
        checks += 1
        opt_pct = options_notional / nav if nav > 0 else 0
        if opt_pct > self.max_options_notional_pct:
            violations.append(ComplianceViolation(
                rule="options_notional",
                description=f"Options notional {opt_pct:.1%} exceeds {self.max_options_notional_pct:.0%}",
                severity="breach",
                current_value=opt_pct,
                limit=self.max_options_notional_pct,
                timestamp=now,
            ))

        self._violations_log.extend(violations)

        report = ComplianceReport(
            compliant=len(violations) == 0,
            violations=violations,
            checks_run=checks,
            position_concentration_max=max_pos_conc,
            sector_concentration_max=max_sec_conc,
            gross_exposure=gross,
            net_exposure=net,
            options_notional=opt_pct,
            timestamp=now,
        )

        if not report.compliant:
            logger.warning(
                "COMPLIANCE BREACH: %d violations (%s)",
                len(violations), ", ".join(v.rule for v in violations),
            )
        return report

    @property
    def violations_log(self) -> List[ComplianceViolation]:
        return self._violations_log


# ---------------------------------------------------------------------------
# Item 22 — RegulatoryReporter
# ---------------------------------------------------------------------------

@dataclass
class RegulatoryReport:
    """Regulatory report data."""
    report_type: str = ""
    timestamp: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


class RegulatoryReporter:
    """Generate regulatory reports.

    Reports:
      - Form PF-style: AUM, leverage, sector exposure, liquidity profile.
      - Large trader report: daily gross notional.
      - TCA summary: execution quality, slippage analysis.
    """

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        self._reports: List[RegulatoryReport] = []

    def form_pf_report(
        self,
        nav: float,
        positions: Dict[str, float],
        leverage: float,
        sector_exposures: Optional[Dict[str, float]] = None,
    ) -> RegulatoryReport:
        """Generate Form PF-style report.

        Args:
            nav: Net asset value.
            positions: Symbol → value.
            leverage: Gross leverage ratio.
            sector_exposures: Sector → exposure.

        Returns:
            RegulatoryReport with Form PF data.
        """
        long_val = sum(v for v in positions.values() if v > 0)
        short_val = sum(abs(v) for v in positions.values() if v < 0)

        data = {
            "reporting_period": datetime.now(timezone.utc).strftime("%Y-%m"),
            "nav": nav,
            "gross_asset_value": long_val + short_val,
            "long_exposure": long_val,
            "short_exposure": short_val,
            "leverage": leverage,
            "n_positions": len(positions),
            "sector_exposures": sector_exposures or {},
            "top_5_positions": dict(
                sorted(positions.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            ),
        }

        report = RegulatoryReport(
            report_type="form_pf",
            timestamp=datetime.now(timezone.utc).isoformat(),
            data=data,
        )
        self._reports.append(report)
        logger.info("Form PF report generated: NAV=$%,.0f, leverage=%.2fx", nav, leverage)
        return report

    def large_trader_report(
        self,
        daily_trades: List[Dict[str, Any]],
        threshold: float = 20_000_000.0,
    ) -> RegulatoryReport:
        """Generate large trader report (SEC Rule 13h-1 style).

        Args:
            daily_trades: List of trade dicts with 'symbol', 'quantity', 'price'.
            threshold: Daily notional threshold for large trader status.

        Returns:
            RegulatoryReport with large trader data.
        """
        total_notional = sum(
            abs(t.get("quantity", 0) * t.get("price", 0))
            for t in daily_trades
        )

        data = {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "total_notional": total_notional,
            "n_trades": len(daily_trades),
            "exceeds_threshold": total_notional > threshold,
            "threshold": threshold,
            "top_symbols": {},
        }

        # Aggregate by symbol
        by_sym: Dict[str, float] = {}
        for t in daily_trades:
            sym = t.get("symbol", "")
            by_sym[sym] = by_sym.get(sym, 0) + abs(t.get("quantity", 0) * t.get("price", 0))
        data["top_symbols"] = dict(sorted(by_sym.items(), key=lambda x: x[1], reverse=True)[:10])

        report = RegulatoryReport(
            report_type="large_trader",
            timestamp=datetime.now(timezone.utc).isoformat(),
            data=data,
        )
        self._reports.append(report)
        return report

    def tca_summary(
        self,
        fills: List[Dict[str, Any]],
    ) -> RegulatoryReport:
        """Generate Transaction Cost Analysis summary.

        Args:
            fills: List of fill dicts with 'price', 'benchmark_price', 'quantity', 'commission'.

        Returns:
            RegulatoryReport with TCA data.
        """
        if not fills:
            return RegulatoryReport(report_type="tca", timestamp=datetime.now(timezone.utc).isoformat())

        slippages = []
        total_commission = 0.0
        total_notional = 0.0

        for fill in fills:
            price = fill.get("price", 0)
            benchmark = fill.get("benchmark_price", price)
            qty = abs(fill.get("quantity", 0))
            commission = fill.get("commission", 0)

            if benchmark > 0:
                slippage_bps = (price - benchmark) / benchmark * 10000
                slippages.append(slippage_bps)

            total_commission += commission
            total_notional += price * qty

        data = {
            "period": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "n_fills": len(fills),
            "total_notional": total_notional,
            "total_commission": total_commission,
            "avg_slippage_bps": float(sum(slippages) / len(slippages)) if slippages else 0,
            "median_slippage_bps": float(sorted(slippages)[len(slippages) // 2]) if slippages else 0,
            "p95_slippage_bps": float(sorted(slippages)[int(len(slippages) * 0.95)]) if len(slippages) > 1 else (slippages[0] if slippages else 0),
            "cost_ratio_bps": total_commission / max(total_notional, 1e-6) * 10000,
        }

        report = RegulatoryReport(
            report_type="tca",
            timestamp=datetime.now(timezone.utc).isoformat(),
            data=data,
        )
        self._reports.append(report)
        logger.info(
            "TCA summary: %d fills, avg slippage=%.1f bps, commission=$%,.2f",
            len(fills), data["avg_slippage_bps"], total_commission,
        )
        return report

    def save_report(self, report: RegulatoryReport) -> str:
        """Save report to JSON file."""
        path = Path(self.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        filename = f"{report.report_type}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        filepath = path / filename
        with open(filepath, "w") as f:
            json.dump({
                "report_type": report.report_type,
                "timestamp": report.timestamp,
                "data": report.data,
            }, f, indent=2, default=str)
        return str(filepath)

    @property
    def reports(self) -> List[RegulatoryReport]:
        return self._reports
