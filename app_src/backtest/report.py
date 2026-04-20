"""
backtest/report.py
==================
HTML report generation for backtest results.

Generates standalone HTML files with embedded matplotlib charts
(base64 PNGs) and formatted tables. Dark-themed quant fund aesthetic.
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# CSS Theme
# ---------------------------------------------------------------------------

_CSS = """
:root {
    --bg-primary: #0d1117;
    --bg-secondary: #161b22;
    --bg-tertiary: #21262d;
    --text-primary: #c9d1d9;
    --text-secondary: #8b949e;
    --accent-green: #3fb950;
    --accent-red: #f85149;
    --accent-blue: #58a6ff;
    --accent-purple: #bc8cff;
    --accent-orange: #d29922;
    --border-color: #30363d;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    background: var(--bg-primary);
    color: var(--text-primary);
    line-height: 1.6;
    padding: 2rem;
}

.container { max-width: 1200px; margin: 0 auto; }

h1 {
    font-size: 1.8rem;
    color: var(--accent-blue);
    border-bottom: 2px solid var(--accent-blue);
    padding-bottom: 0.5rem;
    margin-bottom: 1.5rem;
}

h2 {
    font-size: 1.2rem;
    color: var(--accent-purple);
    margin: 1.5rem 0 0.75rem 0;
}

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
}

.metric-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 1rem;
}

.metric-label {
    font-size: 0.75rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.metric-value {
    font-size: 1.4rem;
    font-weight: 700;
    margin-top: 0.25rem;
}

.positive { color: var(--accent-green); }
.negative { color: var(--accent-red); }
.neutral  { color: var(--text-primary); }

.chart-container {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1.5rem;
    text-align: center;
}

.chart-container img {
    max-width: 100%;
    height: auto;
    border-radius: 4px;
}

table {
    width: 100%;
    border-collapse: collapse;
    background: var(--bg-secondary);
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 1.5rem;
}

th {
    background: var(--bg-tertiary);
    color: var(--accent-blue);
    padding: 0.75rem;
    text-align: left;
    font-size: 0.8rem;
    text-transform: uppercase;
}

td {
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid var(--border-color);
    font-size: 0.85rem;
}

tr:hover { background: var(--bg-tertiary); }

.footer {
    text-align: center;
    color: var(--text-secondary);
    font-size: 0.75rem;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border-color);
}
"""


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _fig_to_base64(fig) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="#0d1117", edgecolor="none")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


def _setup_axes(ax) -> None:
    """Apply dark theme to axes."""
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#8b949e", labelsize=8)
    ax.spines["bottom"].set_color("#30363d")
    ax.spines["left"].set_color("#30363d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.label.set_color("#8b949e")
    ax.yaxis.label.set_color("#8b949e")
    ax.title.set_color("#c9d1d9")


def _chart_equity_curve(
    equity: pd.Series,
    benchmark: Optional[pd.Series] = None,
) -> str:
    """Equity curve with optional benchmark overlay."""
    fig, ax = plt.subplots(figsize=(10, 4))
    _setup_axes(ax)
    ax.plot(equity.index, equity.values, color="#58a6ff", linewidth=1.5, label="Portfolio")
    if benchmark is not None and len(benchmark) > 0:
        # Normalize benchmark to same starting value
        scale = float(equity.iloc[0]) / float(benchmark.iloc[0]) if benchmark.iloc[0] != 0 else 1
        ax.plot(benchmark.index, benchmark.values * scale, color="#8b949e",
                linewidth=1, alpha=0.7, label="SPY (scaled)")
        ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
    ax.set_title("Equity Curve")
    ax.set_ylabel("NAV ($)")
    ax.grid(True, alpha=0.15, color="#30363d")
    return _fig_to_base64(fig)


def _chart_drawdown(equity: pd.Series) -> str:
    """Underwater/drawdown chart."""
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max.replace(0, np.nan)

    fig, ax = plt.subplots(figsize=(10, 3))
    _setup_axes(ax)
    ax.fill_between(dd.index, dd.values, 0, color="#f85149", alpha=0.4)
    ax.plot(dd.index, dd.values, color="#f85149", linewidth=0.8)
    ax.set_title("Drawdown (Underwater Curve)")
    ax.set_ylabel("Drawdown %")
    ax.grid(True, alpha=0.15, color="#30363d")
    return _fig_to_base64(fig)


def _chart_monthly_heatmap(equity: pd.Series) -> str:
    """Monthly returns heatmap."""
    monthly = equity.resample("ME").last()
    monthly_ret = monthly.pct_change().dropna()

    if len(monthly_ret) < 2:
        return ""

    years = sorted(monthly_ret.index.year.unique())
    data = np.full((len(years), 12), np.nan)
    for idx, val in monthly_ret.items():
        y_idx = years.index(idx.year)
        m_idx = idx.month - 1
        data[y_idx, m_idx] = val

    fig, ax = plt.subplots(figsize=(10, max(2, len(years) * 0.5 + 1)))
    _setup_axes(ax)

    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-0.15, vmax=0.15)
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], fontsize=7)
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels([str(y) for y in years], fontsize=8)

    # Annotate cells
    for i in range(len(years)):
        for j in range(12):
            v = data[i, j]
            if not np.isnan(v):
                color = "#0d1117" if abs(v) > 0.05 else "#c9d1d9"
                ax.text(j, i, f"{v:.1%}", ha="center", va="center",
                        fontsize=7, color=color)

    ax.set_title("Monthly Returns Heatmap")
    fig.colorbar(im, ax=ax, shrink=0.8)
    return _fig_to_base64(fig)


def _chart_rolling_sharpe(equity: pd.Series, window: int = 252) -> str:
    """Rolling Sharpe ratio (252-day)."""
    returns = equity.pct_change().dropna()
    if len(returns) < window:
        return ""

    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    sharpe = (rolling_mean / rolling_std.replace(0, np.nan)) * np.sqrt(252)
    sharpe = sharpe.dropna()

    if len(sharpe) == 0:
        return ""

    fig, ax = plt.subplots(figsize=(10, 3))
    _setup_axes(ax)
    ax.plot(sharpe.index, sharpe.values, color="#bc8cff", linewidth=1)
    ax.axhline(y=0, color="#30363d", linewidth=0.8, linestyle="--")
    ax.axhline(y=1, color="#3fb950", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axhline(y=-1, color="#f85149", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_title(f"Rolling Sharpe Ratio ({window}d)")
    ax.grid(True, alpha=0.15, color="#30363d")
    return _fig_to_base64(fig)


def _chart_trade_distribution(trades: pd.DataFrame) -> str:
    """Win/loss P&L histogram."""
    if trades is None or len(trades) == 0 or "pnl" not in trades.columns:
        return ""

    pnls = trades["pnl"].dropna().values

    fig, ax = plt.subplots(figsize=(10, 3))
    _setup_axes(ax)

    colors = ["#3fb950" if p > 0 else "#f85149" for p in pnls]
    ax.bar(range(len(pnls)), sorted(pnls, reverse=True), color=colors, width=0.8)
    ax.axhline(y=0, color="#30363d", linewidth=0.8)
    ax.set_title("Trade P&L Distribution")
    ax.set_ylabel("P&L ($)")
    ax.set_xlabel("Trades (sorted)")
    ax.grid(True, alpha=0.15, color="#30363d", axis="y")
    return _fig_to_base64(fig)


def _chart_strategy_pnl(trades: pd.DataFrame) -> str:
    """P&L by strategy."""
    if trades is None or len(trades) == 0:
        return ""

    strat_col = "strategy" if "strategy" in trades.columns else "strategy_type"
    if strat_col not in trades.columns or "pnl" not in trades.columns:
        return ""

    by_strat = trades.groupby(strat_col)["pnl"].sum()
    if len(by_strat) == 0:
        return ""

    fig, ax = plt.subplots(figsize=(10, 3))
    _setup_axes(ax)

    colors = ["#3fb950" if v > 0 else "#f85149" for v in by_strat.values]
    ax.barh(by_strat.index, by_strat.values, color=colors, height=0.5)
    ax.axvline(x=0, color="#30363d", linewidth=0.8)
    ax.set_title("P&L by Strategy")
    ax.set_xlabel("Total P&L ($)")
    ax.grid(True, alpha=0.15, color="#30363d", axis="x")
    return _fig_to_base64(fig)


# ---------------------------------------------------------------------------
# BacktestReport
# ---------------------------------------------------------------------------

class BacktestReport:
    """Generate a standalone HTML backtest report.

    Parameters
    ----------
    title : str
        Report title.
    """

    def __init__(self, title: str = "ATNN Backtest Report") -> None:
        self.title = title

    def generate(
        self,
        result: Any,
        output_path: str,
        benchmark: Optional[pd.Series] = None,
    ) -> str:
        """Generate HTML report and write to file.

        Parameters
        ----------
        result : BacktestResult or OptionsBacktestResult
            The backtest result object.
        output_path : str
            Path to write the HTML file.
        benchmark : pd.Series, optional
            Benchmark equity curve for overlay.

        Returns
        -------
        str
            Path to the generated HTML file.
        """
        equity = result.equity_curve
        trades = result.trades if isinstance(result.trades, pd.DataFrame) else pd.DataFrame(result.trades)
        metrics = result.metrics

        html_parts = [
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            f"<title>{self.title}</title>",
            "<meta charset='utf-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1'>",
            f"<style>{_CSS}</style>",
            "</head>",
            "<body>",
            "<div class='container'>",
            f"<h1>{self.title}</h1>",
        ]

        # Summary metrics cards
        html_parts.append(self._metrics_section(metrics))

        # Charts
        if HAS_MPL and equity is not None and len(equity) > 1:
            html_parts.append("<h2>Equity Curve</h2>")
            html_parts.append(self._chart_html(_chart_equity_curve(equity, benchmark)))

            html_parts.append("<h2>Drawdown</h2>")
            html_parts.append(self._chart_html(_chart_drawdown(equity)))

            heatmap = _chart_monthly_heatmap(equity)
            if heatmap:
                html_parts.append("<h2>Monthly Returns</h2>")
                html_parts.append(self._chart_html(heatmap))

            rolling = _chart_rolling_sharpe(equity)
            if rolling:
                html_parts.append("<h2>Rolling Sharpe (252d)</h2>")
                html_parts.append(self._chart_html(rolling))

            trade_dist = _chart_trade_distribution(trades)
            if trade_dist:
                html_parts.append("<h2>Trade Distribution</h2>")
                html_parts.append(self._chart_html(trade_dist))

            strat_chart = _chart_strategy_pnl(trades)
            if strat_chart:
                html_parts.append("<h2>P&L by Strategy</h2>")
                html_parts.append(self._chart_html(strat_chart))

        # Tables
        html_parts.append(self._summary_table(metrics))

        if trades is not None and len(trades) > 0 and "pnl" in trades.columns:
            html_parts.append(self._top_trades_table(trades))

        # Monthly returns table
        if equity is not None and len(equity) > 30:
            html_parts.append(self._monthly_table(equity))

        # Footer
        html_parts.append(
            "<div class='footer'>Generated by ATNN Quant Powerhouse v2</div>"
        )
        html_parts.append("</div></body></html>")

        html = "\n".join(html_parts)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(html)

        logger.info("Report written to %s", output_path)
        return output_path

    @staticmethod
    def _chart_html(b64: str) -> str:
        if not b64:
            return ""
        return f"<div class='chart-container'><img src='data:image/png;base64,{b64}'/></div>"

    @staticmethod
    def _metrics_section(metrics: dict) -> str:
        """Build the metrics cards grid."""

        def _card(label: str, value: Any, is_pct: bool = False) -> str:
            if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
                display = "N/A"
                css_class = "neutral"
            elif is_pct:
                display = f"{value:+.2%}"
                css_class = "positive" if value > 0 else "negative" if value < 0 else "neutral"
            elif isinstance(value, float):
                display = f"{value:.2f}"
                css_class = "positive" if value > 0 else "negative" if value < 0 else "neutral"
            elif isinstance(value, int):
                display = str(value)
                css_class = "neutral"
            else:
                display = str(value)
                css_class = "neutral"

            return (
                f"<div class='metric-card'>"
                f"<div class='metric-label'>{label}</div>"
                f"<div class='metric-value {css_class}'>{display}</div>"
                f"</div>"
            )

        cards = [
            _card("Total Return", metrics.get("total_return"), True),
            _card("CAGR", metrics.get("cagr"), True),
            _card("Sharpe Ratio", metrics.get("sharpe_ratio")),
            _card("Sortino Ratio", metrics.get("sortino_ratio")),
            _card("Max Drawdown", metrics.get("max_drawdown"), True),
            _card("Calmar Ratio", metrics.get("calmar_ratio")),
            _card("Win Rate", metrics.get("win_rate"), True),
            _card("Profit Factor", metrics.get("profit_factor")),
            _card("Total Trades", metrics.get("total_trades", 0)),
            _card("Volatility", metrics.get("volatility"), True),
        ]

        return "<div class='metrics-grid'>" + "\n".join(cards) + "</div>"

    @staticmethod
    def _summary_table(metrics: dict) -> str:
        """Build the summary metrics table."""
        rows = []
        display_keys = [
            ("total_return", "Total Return", True),
            ("annual_return", "Annual Return", True),
            ("cagr", "CAGR", True),
            ("sharpe_ratio", "Sharpe Ratio", False),
            ("sortino_ratio", "Sortino Ratio", False),
            ("calmar_ratio", "Calmar Ratio", False),
            ("max_drawdown", "Max Drawdown", True),
            ("max_drawdown_duration", "DD Duration (days)", False),
            ("volatility", "Volatility", True),
            ("win_rate", "Win Rate", True),
            ("profit_factor", "Profit Factor", False),
            ("avg_win", "Avg Win ($)", False),
            ("avg_loss", "Avg Loss ($)", False),
            ("total_trades", "Total Trades", False),
            ("avg_holding_period", "Avg Holding (days)", False),
            ("var_95", "VaR 95%", True),
            ("cvar_95", "CVaR 95%", True),
        ]

        for key, label, is_pct in display_keys:
            val = metrics.get(key)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                formatted = "N/A"
            elif is_pct:
                formatted = f"{val:+.2%}"
            elif isinstance(val, float):
                formatted = f"{val:.4f}"
            else:
                formatted = str(val)
            rows.append(f"<tr><td>{label}</td><td>{formatted}</td></tr>")

        return (
            "<h2>Summary Metrics</h2>"
            "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead>"
            "<tbody>" + "\n".join(rows) + "</tbody></table>"
        )

    @staticmethod
    def _top_trades_table(trades: pd.DataFrame) -> str:
        """Best and worst trades tables."""
        if "pnl" not in trades.columns:
            return ""

        sorted_trades = trades.sort_values("pnl", ascending=False)
        best = sorted_trades.head(10)
        worst = sorted_trades.tail(10).iloc[::-1]

        def _trade_rows(df: pd.DataFrame) -> str:
            rows = []
            for _, t in df.iterrows():
                sym = t.get("symbol", "")
                pnl = t.get("pnl", 0)
                entry = str(t.get("entry_date", ""))[:10]
                exit_ = str(t.get("exit_date", ""))[:10]
                css = "positive" if pnl > 0 else "negative"
                rows.append(
                    f"<tr><td>{sym}</td><td>{entry}</td><td>{exit_}</td>"
                    f"<td class='{css}'>${pnl:+.2f}</td></tr>"
                )
            return "\n".join(rows)

        html = "<h2>Top 10 Best Trades</h2>"
        html += ("<table><thead><tr><th>Symbol</th><th>Entry</th>"
                 "<th>Exit</th><th>P&amp;L</th></tr></thead><tbody>")
        html += _trade_rows(best) + "</tbody></table>"

        html += "<h2>Top 10 Worst Trades</h2>"
        html += ("<table><thead><tr><th>Symbol</th><th>Entry</th>"
                 "<th>Exit</th><th>P&amp;L</th></tr></thead><tbody>")
        html += _trade_rows(worst) + "</tbody></table>"

        return html

    @staticmethod
    def _monthly_table(equity: pd.Series) -> str:
        """Monthly returns table."""
        monthly = equity.resample("ME").last()
        monthly_ret = monthly.pct_change().dropna()

        if len(monthly_ret) < 1:
            return ""

        years = sorted(monthly_ret.index.year.unique())
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        header = "<th>Year</th>" + "".join(f"<th>{m}</th>" for m in months) + "<th>Annual</th>"

        rows = []
        for year in years:
            cells = f"<td><strong>{year}</strong></td>"
            year_total = 0.0
            has_data = False
            for month in range(1, 13):
                mask = (monthly_ret.index.year == year) & (monthly_ret.index.month == month)
                vals = monthly_ret.loc[mask]
                if len(vals) > 0:
                    v = float(vals.iloc[0])
                    year_total += v
                    has_data = True
                    css = "positive" if v > 0 else "negative"
                    cells += f"<td class='{css}'>{v:+.1%}</td>"
                else:
                    cells += "<td>-</td>"
            if has_data:
                css = "positive" if year_total > 0 else "negative"
                cells += f"<td class='{css}'><strong>{year_total:+.1%}</strong></td>"
            else:
                cells += "<td>-</td>"
            rows.append(f"<tr>{cells}</tr>")

        return (
            "<h2>Monthly Returns</h2>"
            f"<table><thead><tr>{header}</tr></thead>"
            "<tbody>" + "\n".join(rows) + "</tbody></table>"
        )
