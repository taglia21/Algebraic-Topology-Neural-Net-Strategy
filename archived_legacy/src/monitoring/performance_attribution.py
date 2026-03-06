"""
Performance Attribution (Phase L, Item 19)
============================================

Break down daily P&L into:
  - Strategy contribution (gamma scalp, vol arb, skew, etc.)
  - Greeks P&L (delta, theta, vega)
  - Transaction costs
  - Slippage

Writes daily JSON report to ``logs/attribution/YYYY-MM-DD.json``.
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["PerformanceAttributor", "DailyAttribution"]


@dataclass
class StrategyPnL:
    """P&L for a single strategy."""
    strategy_name: str
    gross_pnl: float = 0.0
    n_trades: int = 0
    win_rate: float = 0.0


@dataclass
class GreeksPnL:
    """P&L decomposition by Greeks."""
    delta_pnl: float = 0.0
    theta_pnl: float = 0.0
    vega_pnl: float = 0.0
    gamma_pnl: float = 0.0


@dataclass
class DailyAttribution:
    """Full daily P&L attribution report."""
    date: str
    total_pnl: float = 0.0
    strategy_pnl: List[StrategyPnL] = field(default_factory=list)
    greeks_pnl: GreeksPnL = field(default_factory=GreeksPnL)
    transaction_costs: float = 0.0
    slippage: float = 0.0
    net_pnl: float = 0.0


class PerformanceAttributor:
    """Daily P&L attribution engine.

    Parameters
    ----------
    output_dir : str
        Directory for daily JSON reports (default ``logs/attribution``).
    """

    def __init__(self, output_dir: str = "logs/attribution"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._current_date: Optional[str] = None
        self._trades: List[Dict] = []
        self._daily_reports: List[DailyAttribution] = []

    def record_trade(
        self,
        strategy: str,
        pnl: float,
        delta_pnl: float = 0.0,
        theta_pnl: float = 0.0,
        vega_pnl: float = 0.0,
        gamma_pnl: float = 0.0,
        costs: float = 0.0,
        slippage: float = 0.0,
        won: bool = True,
    ) -> None:
        """Record a single trade's P&L components."""
        today = date.today().isoformat()
        if self._current_date != today:
            if self._current_date is not None:
                self._flush_day()
            self._current_date = today
            self._trades = []

        self._trades.append({
            "strategy": strategy,
            "pnl": pnl,
            "delta_pnl": delta_pnl,
            "theta_pnl": theta_pnl,
            "vega_pnl": vega_pnl,
            "gamma_pnl": gamma_pnl,
            "costs": costs,
            "slippage": slippage,
            "won": won,
        })

    def generate_report(self, report_date: Optional[str] = None) -> DailyAttribution:
        """Generate attribution report for the current day.

        Parameters
        ----------
        report_date : str or None
            Override date string (default: today).

        Returns
        -------
        DailyAttribution
        """
        dt = report_date or date.today().isoformat()

        # Aggregate by strategy
        strat_map: Dict[str, List[Dict]] = {}
        for t in self._trades:
            strat_map.setdefault(t["strategy"], []).append(t)

        strategies = []
        for name, trades in strat_map.items():
            gross = sum(t["pnl"] for t in trades)
            wins = sum(1 for t in trades if t["won"])
            strategies.append(StrategyPnL(
                strategy_name=name,
                gross_pnl=gross,
                n_trades=len(trades),
                win_rate=wins / len(trades) if trades else 0.0,
            ))

        # Greeks aggregate
        greeks = GreeksPnL(
            delta_pnl=sum(t["delta_pnl"] for t in self._trades),
            theta_pnl=sum(t["theta_pnl"] for t in self._trades),
            vega_pnl=sum(t["vega_pnl"] for t in self._trades),
            gamma_pnl=sum(t["gamma_pnl"] for t in self._trades),
        )

        total_costs = sum(t["costs"] for t in self._trades)
        total_slippage = sum(t["slippage"] for t in self._trades)
        total_pnl = sum(t["pnl"] for t in self._trades)
        net_pnl = total_pnl - total_costs - total_slippage

        report = DailyAttribution(
            date=dt,
            total_pnl=total_pnl,
            strategy_pnl=strategies,
            greeks_pnl=greeks,
            transaction_costs=total_costs,
            slippage=total_slippage,
            net_pnl=net_pnl,
        )

        self._daily_reports.append(report)
        return report

    def save_report(self, report: Optional[DailyAttribution] = None) -> str:
        """Save daily attribution report to JSON.

        Returns path to saved file.
        """
        if report is None:
            report = self.generate_report()

        path = self.output_dir / f"{report.date}.json"
        data = asdict(report)

        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info("Attribution report saved: %s", path)
        return str(path)

    def _flush_day(self):
        """Flush and save the current day's report."""
        try:
            report = self.generate_report(self._current_date)
            self.save_report(report)
        except Exception as exc:
            logger.error("Failed to flush daily report: %s", exc)

    @property
    def daily_reports(self) -> List[DailyAttribution]:
        return list(self._daily_reports)
