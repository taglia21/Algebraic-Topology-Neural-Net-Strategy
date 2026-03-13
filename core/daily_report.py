"""
core/daily_report.py
====================
Generate daily performance report as formatted text.
Can be logged, emailed, or displayed in a dashboard.
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class DailyReporter:
    """Generate daily trading performance reports."""

    def generate_report(
        self,
        nav: float,
        daily_pnl: float,
        positions: List[Dict],
        trades_today: List[Dict],
        kelly_params: Dict[str, float],
        regime: str = "NORMAL",
        signals_generated: int = 0,
        signals_actionable: int = 0,
    ) -> str:
        """Generate a formatted daily report string."""
        pnl_pct = (daily_pnl / nav * 100) if nav > 0 else 0.0
        pnl_emoji = "+" if daily_pnl >= 0 else ""

        lines = [
            f"{'=' * 50}",
            f"  ATNN v2 -- DAILY REPORT",
            f"  {datetime.now().strftime('%A, %B %d, %Y')}",
            f"{'=' * 50}",
            f"",
            f"  NAV:          ${nav:,.2f}",
            f"  Daily P&L:    {pnl_emoji}${daily_pnl:,.2f} ({pnl_emoji}{pnl_pct:.2f}%)",
            f"  Regime:       {regime}",
            f"",
            f"  SIGNALS",
            f"  Generated:    {signals_generated}",
            f"  Actionable:   {signals_actionable}",
            f"  Trades Today: {len(trades_today)}",
            f"",
        ]

        if positions:
            lines.append(f"  OPEN POSITIONS ({len(positions)})")
            lines.append(f"  {'Ticker':<8} {'Qty':>6} {'Entry':>10} {'Strategy':<10}")
            lines.append(f"  {'-' * 40}")
            for pos in positions:
                lines.append(
                    f"  {pos.get('ticker', '???'):<8} "
                    f"{pos.get('quantity', 0):>6.0f} "
                    f"${pos.get('entry_price', 0):>9.2f} "
                    f"{pos.get('strategy_source', 'TDA'):<10}"
                )
        else:
            lines.append("  OPEN POSITIONS: None")

        lines.extend([
            f"",
            f"  KELLY PARAMETERS (rolling)",
            f"  Win Rate:     {kelly_params.get('win_rate', 0.55):.1%}",
            f"  Avg Win:      {kelly_params.get('avg_win', 0.02):.2%}",
            f"  Avg Loss:     {kelly_params.get('avg_loss', 0.015):.2%}",
            f"",
            f"{'=' * 50}",
        ])

        return "\n".join(lines)

    def log_report(self, report: str) -> None:
        """Log the report to the trade logger."""
        for line in report.split("\n"):
            logger.info(line)
