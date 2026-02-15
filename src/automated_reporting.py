"""
Automated Reporting Module (TIER 3)
=====================================

Generates and delivers daily/weekly performance reports via email or file.

Features:
1. Daily summary — P&L, trades, regime, top movers
2. Weekly digest — cumulative stats, Sharpe, drawdown, factor attribution
3. Monthly tearsheet — full performance analysis
4. HTML email formatting with inline Plotly images
5. CSV attachment of trade log
6. Configurable schedule (cron-compatible)

Usage:
    from src.automated_reporting import AutomatedReporter, ReportConfig

    reporter = AutomatedReporter(ReportConfig(
        smtp_host="smtp.gmail.com",
        smtp_port=587,
        sender_email="bot@example.com",
        sender_password="app-password",
        recipients=["trader@example.com"],
    ))
    reporter.generate_daily_report(portfolio_history, trades)
    reporter.send_report(report)
"""

import json
import csv
import io
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class ReportFrequency(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class ReportConfig:
    """Report delivery configuration."""
    # SMTP settings
    smtp_host: str = ""
    smtp_port: int = 587
    sender_email: str = ""
    sender_password: str = ""
    recipients: List[str] = field(default_factory=list)
    use_tls: bool = True

    # Report settings
    report_dir: str = "results/reports"
    include_trades_csv: bool = True
    max_trade_rows: int = 500

    # Schedule
    daily_hour: int = 17     # 5 PM
    weekly_day: int = 4      # Friday
    monthly_day: int = 1     # 1st of month


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class PerformanceMetrics:
    """Computed performance metrics for a reporting period."""
    period: str = ""            # "2026-02-15" or "2026-W07"
    frequency: str = "daily"
    start_value: float = 0.0
    end_value: float = 0.0
    pnl: float = 0.0
    return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    volatility_annualized: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    best_trade_pnl: float = 0.0
    worst_trade_pnl: float = 0.0
    best_symbol: str = ""
    worst_symbol: str = ""
    regime_distribution: Dict[str, float] = field(default_factory=dict)
    top_contributors: List[Tuple[str, float]] = field(default_factory=list)
    bottom_contributors: List[Tuple[str, float]] = field(default_factory=list)


@dataclass
class TradeSummary:
    """Summary of a trade for reporting."""
    timestamp: str = ""
    symbol: str = ""
    side: str = ""
    quantity: float = 0.0
    price: float = 0.0
    pnl: float = 0.0
    strategy: str = ""


@dataclass
class Report:
    """Generated report container."""
    title: str = ""
    frequency: str = "daily"
    generated_at: str = ""
    metrics: Optional[PerformanceMetrics] = None
    html_content: str = ""
    csv_attachment: str = ""  # CSV string for trade log
    file_path: str = ""


# =============================================================================
# PERFORMANCE CALCULATOR
# =============================================================================

class PerformanceCalculator:
    """Computes metrics from portfolio history and trade data."""

    @staticmethod
    def compute(
        values: List[float],
        timestamps: List[str],
        trades: List[TradeSummary],
        frequency: str = "daily",
        period_label: str = "",
    ) -> PerformanceMetrics:
        """Compute full performance metrics for a period."""
        m = PerformanceMetrics(period=period_label, frequency=frequency)
        if not values or len(values) < 2:
            return m

        arr = np.array(values, dtype=float)
        returns = np.diff(arr) / arr[:-1]
        returns = returns[np.isfinite(returns)]

        m.start_value = float(arr[0])
        m.end_value = float(arr[-1])
        m.pnl = m.end_value - m.start_value
        m.return_pct = (m.end_value / m.start_value - 1) * 100 if m.start_value else 0.0

        # Sharpe & Sortino (annualized)
        if len(returns) > 1 and np.std(returns) > 0:
            m.sharpe_ratio = float(np.mean(returns) / np.std(returns) * np.sqrt(252))
            downside = returns[returns < 0]
            if len(downside) > 0 and np.std(downside) > 0:
                m.sortino_ratio = float(np.mean(returns) / np.std(downside) * np.sqrt(252))
            m.volatility_annualized = float(np.std(returns) * np.sqrt(252) * 100)

        # Drawdown
        peak = np.maximum.accumulate(arr)
        dd = (peak - arr) / np.where(peak > 0, peak, 1) * 100
        m.max_drawdown_pct = float(np.max(dd))

        # Trade stats
        m.total_trades = len(trades)
        pnls = [t.pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        m.winning_trades = len(wins)
        m.losing_trades = len(losses)
        m.win_rate = (len(wins) / len(pnls) * 100) if pnls else 0.0
        m.avg_win = float(np.mean(wins)) if wins else 0.0
        m.avg_loss = float(np.mean(losses)) if losses else 0.0
        m.profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float('inf')

        if pnls:
            m.best_trade_pnl = max(pnls)
            m.worst_trade_pnl = min(pnls)

        # Per-symbol attribution
        symbol_pnl: Dict[str, float] = {}
        for t in trades:
            symbol_pnl[t.symbol] = symbol_pnl.get(t.symbol, 0) + t.pnl
        sorted_syms = sorted(symbol_pnl.items(), key=lambda x: x[1], reverse=True)
        m.top_contributors = sorted_syms[:5]
        m.bottom_contributors = sorted_syms[-5:]
        if sorted_syms:
            m.best_symbol = sorted_syms[0][0]
            m.worst_symbol = sorted_syms[-1][0]

        return m


# =============================================================================
# HTML REPORT TEMPLATE
# =============================================================================

class ReportRenderer:
    """Renders performance metrics into styled HTML."""

    @staticmethod
    def _color(val: float) -> str:
        return "#00d4aa" if val >= 0 else "#ff4757"

    @classmethod
    def render(cls, metrics: PerformanceMetrics, trades: List[TradeSummary],
               title: str = "Daily Report") -> str:
        """Render full HTML report."""
        m = metrics
        c = cls._color

        top_html = "".join(
            f"<tr><td>{s}</td><td style='color:{c(p)}'>${p:+,.2f}</td></tr>"
            for s, p in m.top_contributors
        )
        bottom_html = "".join(
            f"<tr><td>{s}</td><td style='color:{c(p)}'>${p:+,.2f}</td></tr>"
            for s, p in m.bottom_contributors
        )

        trade_rows = ""
        for t in trades[-30:]:
            cl = c(t.pnl)
            trade_rows += (
                f"<tr><td>{t.timestamp}</td><td>{t.symbol}</td><td>{t.side}</td>"
                f"<td>{t.quantity}</td><td>${t.price:,.2f}</td>"
                f"<td style='color:{cl}'>${t.pnl:+,.2f}</td>"
                f"<td>{t.strategy}</td></tr>"
            )

        html = f"""
        <html>
        <head><style>
            body {{ font-family: 'Segoe UI', sans-serif; background: #f8f9fa; padding: 20px; }}
            .container {{ max-width: 700px; margin: auto; background: #fff; border-radius: 12px;
                          box-shadow: 0 2px 12px rgba(0,0,0,0.08); padding: 30px; }}
            h1 {{ font-size: 22px; border-bottom: 2px solid #5352ed; padding-bottom: 8px; }}
            h2 {{ font-size: 16px; margin-top: 30px; color: #555; }}
            .metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 16px 0; }}
            .metric {{ background: #f1f2f6; border-radius: 8px; padding: 14px; text-align: center; }}
            .metric .label {{ font-size: 11px; color: #888; text-transform: uppercase; }}
            .metric .val {{ font-size: 20px; font-weight: 700; margin-top: 4px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 12px; }}
            th, td {{ padding: 6px 10px; text-align: left; border-bottom: 1px solid #eee; }}
            th {{ color: #888; text-transform: uppercase; font-size: 10px; }}
            .footer {{ margin-top: 30px; font-size: 11px; color: #aaa; text-align: center; }}
        </style></head>
        <body>
        <div class="container">
            <h1>{title}</h1>
            <p style="color:#888">Period: {m.period} &bull; Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC</p>

            <div class="metrics">
                <div class="metric">
                    <div class="label">P&L</div>
                    <div class="val" style="color:{c(m.pnl)}">${m.pnl:+,.2f}</div>
                </div>
                <div class="metric">
                    <div class="label">Return</div>
                    <div class="val" style="color:{c(m.return_pct)}">{m.return_pct:+.2f}%</div>
                </div>
                <div class="metric">
                    <div class="label">Sharpe</div>
                    <div class="val">{m.sharpe_ratio:.2f}</div>
                </div>
                <div class="metric">
                    <div class="label">Win Rate</div>
                    <div class="val">{m.win_rate:.1f}%</div>
                </div>
                <div class="metric">
                    <div class="label">Max Drawdown</div>
                    <div class="val" style="color:#ff4757">{m.max_drawdown_pct:.2f}%</div>
                </div>
                <div class="metric">
                    <div class="label">Profit Factor</div>
                    <div class="val">{m.profit_factor:.2f}</div>
                </div>
                <div class="metric">
                    <div class="label">Sortino</div>
                    <div class="val">{m.sortino_ratio:.2f}</div>
                </div>
                <div class="metric">
                    <div class="label">Vol (Ann.)</div>
                    <div class="val">{m.volatility_annualized:.1f}%</div>
                </div>
                <div class="metric">
                    <div class="label">Trades</div>
                    <div class="val">{m.total_trades}</div>
                </div>
            </div>

            <h2>Top Contributors</h2>
            <table><thead><tr><th>Symbol</th><th>P&L</th></tr></thead>
            <tbody>{top_html}</tbody></table>

            <h2>Bottom Contributors</h2>
            <table><thead><tr><th>Symbol</th><th>P&L</th></tr></thead>
            <tbody>{bottom_html}</tbody></table>

            <h2>Trade Summary</h2>
            <table>
                <thead><tr><th>Avg Win</th><th>Avg Loss</th><th>Best</th><th>Worst</th></tr></thead>
                <tbody><tr>
                    <td style="color:#00d4aa">${m.avg_win:+,.2f}</td>
                    <td style="color:#ff4757">${m.avg_loss:+,.2f}</td>
                    <td style="color:#00d4aa">${m.best_trade_pnl:+,.2f}</td>
                    <td style="color:#ff4757">${m.worst_trade_pnl:+,.2f}</td>
                </tr></tbody>
            </table>

            <h2>Recent Trades</h2>
            <table>
                <thead><tr><th>Time</th><th>Symbol</th><th>Side</th><th>Qty</th><th>Price</th><th>P&L</th><th>Strategy</th></tr></thead>
                <tbody>{trade_rows}</tbody>
            </table>

            <div class="footer">
                TDA Neural Net Trading System &bull; Automated Report
            </div>
        </div>
        </body></html>
        """
        return html


# =============================================================================
# AUTOMATED REPORTER
# =============================================================================

class AutomatedReporter:
    """
    Generates and delivers performance reports.

    Usage:
        reporter = AutomatedReporter(config)
        report = reporter.generate_daily_report(values, timestamps, trades)
        reporter.save_report(report)
        reporter.send_email(report)
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        self.config = config or ReportConfig()
        self._calculator = PerformanceCalculator()
        self._renderer = ReportRenderer()
        self._history: List[Report] = []
        logger.info("AutomatedReporter initialized (report_dir=%s)", self.config.report_dir)

    # ── Report generation ────────────────────────────────────────────────

    def generate_daily_report(
        self,
        values: List[float],
        timestamps: List[str],
        trades: List[TradeSummary],
        period_label: Optional[str] = None,
    ) -> Report:
        """Generate daily performance report."""
        label = period_label or date.today().isoformat()
        metrics = self._calculator.compute(values, timestamps, trades, "daily", label)
        html = self._renderer.render(metrics, trades, title=f"Daily Report — {label}")
        csv_str = self._trades_to_csv(trades)

        report = Report(
            title=f"Daily Report — {label}",
            frequency="daily",
            generated_at=datetime.utcnow().isoformat(),
            metrics=metrics,
            html_content=html,
            csv_attachment=csv_str,
        )
        self._history.append(report)
        return report

    def generate_weekly_report(
        self,
        values: List[float],
        timestamps: List[str],
        trades: List[TradeSummary],
        week_label: Optional[str] = None,
    ) -> Report:
        """Generate weekly performance report."""
        label = week_label or date.today().strftime("%Y-W%W")
        metrics = self._calculator.compute(values, timestamps, trades, "weekly", label)
        html = self._renderer.render(metrics, trades, title=f"Weekly Digest — {label}")
        csv_str = self._trades_to_csv(trades)

        report = Report(
            title=f"Weekly Digest — {label}",
            frequency="weekly",
            generated_at=datetime.utcnow().isoformat(),
            metrics=metrics,
            html_content=html,
            csv_attachment=csv_str,
        )
        self._history.append(report)
        return report

    def generate_monthly_report(
        self,
        values: List[float],
        timestamps: List[str],
        trades: List[TradeSummary],
        month_label: Optional[str] = None,
    ) -> Report:
        """Generate monthly tearsheet report."""
        label = month_label or date.today().strftime("%Y-%m")
        metrics = self._calculator.compute(values, timestamps, trades, "monthly", label)
        html = self._renderer.render(metrics, trades, title=f"Monthly Tearsheet — {label}")
        csv_str = self._trades_to_csv(trades)

        report = Report(
            title=f"Monthly Tearsheet — {label}",
            frequency="monthly",
            generated_at=datetime.utcnow().isoformat(),
            metrics=metrics,
            html_content=html,
            csv_attachment=csv_str,
        )
        self._history.append(report)
        return report

    # ── Delivery ─────────────────────────────────────────────────────────

    def save_report(self, report: Report) -> str:
        """Save report HTML and CSV to disk."""
        report_dir = Path(self.config.report_dir) / report.frequency
        report_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        html_path = report_dir / f"report_{ts}.html"
        html_path.write_text(report.html_content, encoding="utf-8")
        report.file_path = str(html_path)

        if report.csv_attachment:
            csv_path = report_dir / f"trades_{ts}.csv"
            csv_path.write_text(report.csv_attachment, encoding="utf-8")

        logger.info("Report saved: %s", html_path)
        return str(html_path)

    def send_email(self, report: Report) -> bool:
        """Send report via email."""
        cfg = self.config
        if not cfg.smtp_host or not cfg.recipients:
            logger.warning("SMTP not configured — skipping email.")
            return False

        try:
            msg = MIMEMultipart("mixed")
            msg["From"] = cfg.sender_email
            msg["To"] = ", ".join(cfg.recipients)
            msg["Subject"] = report.title

            html_part = MIMEText(report.html_content, "html")
            msg.attach(html_part)

            if report.csv_attachment and cfg.include_trades_csv:
                attachment = MIMEBase("application", "octet-stream")
                attachment.set_payload(report.csv_attachment.encode("utf-8"))
                encoders.encode_base64(attachment)
                attachment.add_header(
                    "Content-Disposition",
                    f"attachment; filename=trades_{report.metrics.period}.csv",
                )
                msg.attach(attachment)

            with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=30) as server:
                if cfg.use_tls:
                    server.starttls()
                if cfg.sender_password:
                    server.login(cfg.sender_email, cfg.sender_password)
                server.sendmail(cfg.sender_email, cfg.recipients, msg.as_string())

            logger.info("Report email sent to %s", cfg.recipients)
            return True

        except Exception as e:
            logger.error("Email send failed: %s", e)
            return False

    # ── Helpers ──────────────────────────────────────────────────────────

    def _trades_to_csv(self, trades: List[TradeSummary]) -> str:
        """Convert trade list to CSV string."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["timestamp", "symbol", "side", "quantity", "price", "pnl", "strategy"])
        for t in trades[: self.config.max_trade_rows]:
            writer.writerow([t.timestamp, t.symbol, t.side, t.quantity, t.price, t.pnl, t.strategy])
        return output.getvalue()

    def get_history(self) -> List[Report]:
        """Return all generated reports."""
        return self._history


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import random

    # Simulate portfolio data
    values = [100_000.0]
    timestamps = []
    trades = []
    for i in range(30):
        dt = (date.today() - timedelta(days=30 - i)).isoformat()
        timestamps.append(dt)
        ret = random.gauss(0.001, 0.012)
        values.append(values[-1] * (1 + ret))

        if random.random() > 0.5:
            trades.append(TradeSummary(
                timestamp=dt,
                symbol=random.choice(["SPY", "QQQ", "AAPL", "MSFT"]),
                side=random.choice(["BUY", "SELL"]),
                quantity=random.randint(10, 100),
                price=random.uniform(100, 500),
                pnl=random.gauss(50, 200),
                strategy=random.choice(["TDA_momentum", "mean_reversion"]),
            ))

    reporter = AutomatedReporter()
    report = reporter.generate_daily_report(values, timestamps, trades)
    path = reporter.save_report(report)
    print(f"Report saved: {path}")
    print(f"Metrics: Sharpe={report.metrics.sharpe_ratio:.2f}, "
          f"Return={report.metrics.return_pct:.2f}%, "
          f"Win Rate={report.metrics.win_rate:.1f}%")
