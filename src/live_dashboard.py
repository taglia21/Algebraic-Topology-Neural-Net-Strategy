"""
Live Dashboard Module (TIER 3)
===============================

Real-time P&L charts and metrics dashboard using Flask + Plotly.

Features:
1. Real-time P&L curve with drawdown overlay
2. Position heatmap across portfolio
3. Regime indicator panel
4. Risk metrics gauges (Sharpe, VaR, max DD)
5. Trade log table with filtering
6. WebSocket push for live updates

Tech Stack:
- Flask for HTTP server
- Flask-SocketIO for real-time push
- Plotly for interactive charts
- Jinja2 templates (inline)

Usage:
    from src.live_dashboard import LiveDashboard, DashboardConfig

    dashboard = LiveDashboard(DashboardConfig(port=5050))
    dashboard.update_portfolio(portfolio_state)
    dashboard.start()  # non-blocking
"""

import json
import time
import logging
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)

try:
    from flask import Flask, render_template_string, jsonify, request
    from flask_socketio import SocketIO
    from flask_cors import CORS
    import plotly
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    logger.warning("Flask/SocketIO/Plotly not installed. Dashboard disabled.")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    host: str = "0.0.0.0"
    port: int = 5050
    debug: bool = False
    max_history_points: int = 5000
    update_interval_ms: int = 1000
    title: str = "TDA Neural Net Trading Dashboard"
    theme: str = "dark"  # dark or light


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class PortfolioSnapshot:
    """Point-in-time portfolio state."""
    timestamp: str = ""
    portfolio_value: float = 0.0
    cash: float = 0.0
    daily_pnl: float = 0.0
    daily_return_pct: float = 0.0
    total_return_pct: float = 0.0
    positions: Dict[str, Dict[str, float]] = field(default_factory=dict)
    regime: str = "unknown"
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    var_95: float = 0.0
    win_rate: float = 0.0
    num_trades_today: int = 0
    open_positions: int = 0


@dataclass
class TradeRecord:
    """Single trade record."""
    timestamp: str = ""
    symbol: str = ""
    side: str = ""  # BUY / SELL
    quantity: float = 0.0
    price: float = 0.0
    pnl: float = 0.0
    strategy: str = ""
    regime: str = ""


# =============================================================================
# PLOTLY CHART BUILDERS
# =============================================================================

class ChartBuilder:
    """Builds Plotly charts from portfolio data."""

    DARK_THEME = dict(
        paper_bgcolor="#1e1e2f",
        plot_bgcolor="#1e1e2f",
        font_color="#e0e0e0",
        gridcolor="#2d2d44",
    )

    LIGHT_THEME = dict(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8f9fa",
        font_color="#333333",
        gridcolor="#dee2e6",
    )

    @classmethod
    def _theme(cls, theme: str) -> dict:
        return cls.DARK_THEME if theme == "dark" else cls.LIGHT_THEME

    @classmethod
    def equity_curve(
        cls,
        timestamps: List[str],
        values: List[float],
        drawdowns: List[float],
        theme: str = "dark",
    ) -> str:
        """Build equity curve with drawdown overlay."""
        t = cls._theme(theme)
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.05,
            subplot_titles=("Portfolio Value", "Drawdown %"),
        )
        fig.add_trace(
            go.Scatter(
                x=timestamps, y=values,
                mode="lines", name="Portfolio",
                line=dict(color="#00d4aa", width=2),
                fill="tozeroy", fillcolor="rgba(0,212,170,0.1)",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=timestamps, y=drawdowns,
                mode="lines", name="Drawdown",
                line=dict(color="#ff4757", width=1.5),
                fill="tozeroy", fillcolor="rgba(255,71,87,0.15)",
            ),
            row=2, col=1,
        )
        fig.update_layout(
            height=480, margin=dict(l=40, r=20, t=40, b=30),
            paper_bgcolor=t["paper_bgcolor"],
            plot_bgcolor=t["plot_bgcolor"],
            font=dict(color=t["font_color"], size=11),
            showlegend=False,
            xaxis2=dict(gridcolor=t["gridcolor"]),
            yaxis=dict(gridcolor=t["gridcolor"]),
            yaxis2=dict(gridcolor=t["gridcolor"], ticksuffix="%"),
        )
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    @classmethod
    def position_heatmap(
        cls,
        positions: Dict[str, Dict[str, float]],
        theme: str = "dark",
    ) -> str:
        """Build position heatmap (symbol vs weight/pnl)."""
        t = cls._theme(theme)
        symbols = list(positions.keys()) or ["(none)"]
        weights = [positions.get(s, {}).get("weight_pct", 0) for s in symbols]
        pnls = [positions.get(s, {}).get("unrealized_pnl", 0) for s in symbols]
        colors = ["#00d4aa" if p >= 0 else "#ff4757" for p in pnls]

        fig = go.Figure(go.Bar(
            x=symbols, y=weights,
            marker_color=colors,
            text=[f"${p:+,.0f}" for p in pnls],
            textposition="outside",
        ))
        fig.update_layout(
            height=300, margin=dict(l=40, r=20, t=30, b=30),
            paper_bgcolor=t["paper_bgcolor"],
            plot_bgcolor=t["plot_bgcolor"],
            font=dict(color=t["font_color"], size=11),
            yaxis=dict(title="Weight %", gridcolor=t["gridcolor"]),
            title="Position Weights",
        )
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    @classmethod
    def daily_pnl_bar(
        cls,
        dates: List[str],
        pnls: List[float],
        theme: str = "dark",
    ) -> str:
        """Daily P&L bar chart."""
        t = cls._theme(theme)
        colors = ["#00d4aa" if p >= 0 else "#ff4757" for p in pnls]
        fig = go.Figure(go.Bar(x=dates, y=pnls, marker_color=colors))
        fig.update_layout(
            height=280, margin=dict(l=40, r=20, t=30, b=30),
            paper_bgcolor=t["paper_bgcolor"],
            plot_bgcolor=t["plot_bgcolor"],
            font=dict(color=t["font_color"], size=11),
            yaxis=dict(title="Daily P&L ($)", gridcolor=t["gridcolor"]),
            title="Daily P&L",
        )
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    @classmethod
    def risk_gauges(
        cls,
        sharpe: float,
        max_dd: float,
        var95: float,
        win_rate: float,
        theme: str = "dark",
    ) -> str:
        """Risk metric gauges."""
        t = cls._theme(theme)
        fig = make_subplots(
            rows=1, cols=4,
            specs=[[{"type": "indicator"}] * 4],
        )
        gauges = [
            ("Sharpe", sharpe, [-1, 4], [[0, 1, "#ff4757"], [1, 2, "#ffa502"], [2, 4, "#00d4aa"]]),
            ("Max DD %", max_dd, [0, 20], [[0, 5, "#00d4aa"], [5, 10, "#ffa502"], [10, 20, "#ff4757"]]),
            ("VaR 95%", var95, [0, 10], [[0, 3, "#00d4aa"], [3, 6, "#ffa502"], [6, 10, "#ff4757"]]),
            ("Win Rate %", win_rate, [30, 80], [[30, 50, "#ff4757"], [50, 60, "#ffa502"], [60, 80, "#00d4aa"]]),
        ]
        for i, (title, value, rng, steps) in enumerate(gauges, 1):
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=value,
                    title=dict(text=title),
                    gauge=dict(
                        axis=dict(range=rng),
                        bar=dict(color="#5352ed"),
                        steps=[dict(range=s[:2], color=s[2]) for s in steps],
                    ),
                ),
                row=1, col=i,
            )
        fig.update_layout(
            height=220, margin=dict(l=20, r=20, t=40, b=10),
            paper_bgcolor=t["paper_bgcolor"],
            font=dict(color=t["font_color"], size=10),
        )
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


# =============================================================================
# HTML TEMPLATE
# =============================================================================

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.min.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Segoe UI', system-ui, sans-serif;
               background: {{ bg_color }}; color: {{ text_color }}; }
        .header { padding: 12px 24px; background: {{ header_bg }};
                  display: flex; justify-content: space-between; align-items: center;
                  border-bottom: 1px solid {{ border_color }}; }
        .header h1 { font-size: 18px; }
        .status { font-size: 13px; display: flex; gap: 16px; }
        .status .dot { width: 8px; height: 8px; border-radius: 50%;
                       display: inline-block; margin-right: 4px; }
        .dot.green { background: #00d4aa; }
        .dot.red { background: #ff4757; }
        .metrics-bar { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
                       gap: 12px; padding: 16px 24px; }
        .metric-card { background: {{ card_bg }}; border-radius: 8px; padding: 14px;
                       border: 1px solid {{ border_color }}; }
        .metric-card .label { font-size: 11px; opacity: 0.7; text-transform: uppercase; }
        .metric-card .value { font-size: 22px; font-weight: 700; margin-top: 4px; }
        .metric-card .value.positive { color: #00d4aa; }
        .metric-card .value.negative { color: #ff4757; }
        .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
                  padding: 0 24px 16px; }
        .chart-box { background: {{ card_bg }}; border-radius: 8px;
                     border: 1px solid {{ border_color }}; padding: 8px; }
        .chart-box.full { grid-column: 1 / -1; }
        .trades-section { padding: 0 24px 24px; }
        .trades-section h3 { margin-bottom: 8px; }
        table { width: 100%; border-collapse: collapse; font-size: 12px; }
        th, td { padding: 6px 10px; text-align: left;
                 border-bottom: 1px solid {{ border_color }}; }
        th { opacity: 0.7; text-transform: uppercase; font-size: 10px; }
        .pnl-pos { color: #00d4aa; } .pnl-neg { color: #ff4757; }
        @media (max-width: 800px) { .charts { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ title }}</h1>
        <div class="status">
            <span><span class="dot green" id="ws-dot"></span> <span id="ws-status">Connecting...</span></span>
            <span id="last-update">--</span>
        </div>
    </div>

    <div class="metrics-bar" id="metrics-bar"></div>

    <div class="charts">
        <div class="chart-box full" id="equity-chart"></div>
        <div class="chart-box" id="position-chart"></div>
        <div class="chart-box" id="dailypnl-chart"></div>
        <div class="chart-box full" id="risk-chart"></div>
    </div>

    <div class="trades-section">
        <h3>Recent Trades</h3>
        <table>
            <thead><tr>
                <th>Time</th><th>Symbol</th><th>Side</th><th>Qty</th>
                <th>Price</th><th>P&L</th><th>Strategy</th><th>Regime</th>
            </tr></thead>
            <tbody id="trades-body"></tbody>
        </table>
    </div>

    <script>
    const socket = io();
    socket.on('connect', () => {
        document.getElementById('ws-dot').className = 'dot green';
        document.getElementById('ws-status').textContent = 'Live';
    });
    socket.on('disconnect', () => {
        document.getElementById('ws-dot').className = 'dot red';
        document.getElementById('ws-status').textContent = 'Disconnected';
    });

    function renderMetrics(snap) {
        const bar = document.getElementById('metrics-bar');
        const pairs = [
            ['Portfolio', '$' + snap.portfolio_value.toLocaleString(undefined, {minimumFractionDigits:0}), snap.daily_pnl >= 0],
            ['Daily P&L', (snap.daily_pnl >= 0 ? '+$' : '-$') + Math.abs(snap.daily_pnl).toLocaleString(undefined, {minimumFractionDigits:0}), snap.daily_pnl >= 0],
            ['Return', snap.total_return_pct.toFixed(2) + '%', snap.total_return_pct >= 0],
            ['Sharpe', snap.sharpe_ratio.toFixed(2), snap.sharpe_ratio >= 1],
            ['Max DD', snap.max_drawdown_pct.toFixed(2) + '%', snap.max_drawdown_pct < 5],
            ['Win Rate', snap.win_rate.toFixed(1) + '%', snap.win_rate >= 50],
            ['Positions', snap.open_positions, true],
            ['Regime', snap.regime, snap.regime !== 'crisis'],
        ];
        bar.innerHTML = pairs.map(([l,v,ok]) =>
            `<div class="metric-card"><div class="label">${l}</div>`+
            `<div class="value ${ok?'positive':'negative'}">${v}</div></div>`
        ).join('');
    }

    socket.on('snapshot', (data) => {
        document.getElementById('last-update').textContent = data.snapshot.timestamp;
        renderMetrics(data.snapshot);
        if (data.equity_chart) Plotly.react('equity-chart', JSON.parse(data.equity_chart).data, JSON.parse(data.equity_chart).layout);
        if (data.position_chart) Plotly.react('position-chart', JSON.parse(data.position_chart).data, JSON.parse(data.position_chart).layout);
        if (data.dailypnl_chart) Plotly.react('dailypnl-chart', JSON.parse(data.dailypnl_chart).data, JSON.parse(data.dailypnl_chart).layout);
        if (data.risk_chart) Plotly.react('risk-chart', JSON.parse(data.risk_chart).data, JSON.parse(data.risk_chart).layout);
    });

    socket.on('trade', (tr) => {
        const tb = document.getElementById('trades-body');
        const cls = tr.pnl >= 0 ? 'pnl-pos' : 'pnl-neg';
        tb.insertAdjacentHTML('afterbegin',
            `<tr><td>${tr.timestamp}</td><td>${tr.symbol}</td><td>${tr.side}</td>`+
            `<td>${tr.quantity}</td><td>$${tr.price.toFixed(2)}</td>`+
            `<td class="${cls}">$${tr.pnl.toFixed(2)}</td>`+
            `<td>${tr.strategy}</td><td>${tr.regime}</td></tr>`);
        if (tb.children.length > 50) tb.lastChild.remove();
    });

    // initial load
    fetch('/api/state').then(r => r.json()).then(data => {
        if (data.snapshot) renderMetrics(data.snapshot);
        if (data.equity_chart) Plotly.newPlot('equity-chart', JSON.parse(data.equity_chart).data, JSON.parse(data.equity_chart).layout);
        if (data.position_chart) Plotly.newPlot('position-chart', JSON.parse(data.position_chart).data, JSON.parse(data.position_chart).layout);
        if (data.dailypnl_chart) Plotly.newPlot('dailypnl-chart', JSON.parse(data.dailypnl_chart).data, JSON.parse(data.dailypnl_chart).layout);
        if (data.risk_chart) Plotly.newPlot('risk-chart', JSON.parse(data.risk_chart).data, JSON.parse(data.risk_chart).layout);
        if (data.trades) {
            const tb = document.getElementById('trades-body');
            data.trades.forEach(tr => {
                const cls = tr.pnl >= 0 ? 'pnl-pos' : 'pnl-neg';
                tb.insertAdjacentHTML('beforeend',
                    `<tr><td>${tr.timestamp}</td><td>${tr.symbol}</td><td>${tr.side}</td>`+
                    `<td>${tr.quantity}</td><td>$${(tr.price||0).toFixed(2)}</td>`+
                    `<td class="${cls}">$${(tr.pnl||0).toFixed(2)}</td>`+
                    `<td>${tr.strategy||''}</td><td>${tr.regime||''}</td></tr>`);
            });
        }
    });
    </script>
</body>
</html>
"""


# =============================================================================
# LIVE DASHBOARD
# =============================================================================

class LiveDashboard:
    """
    Real-time trading dashboard with Flask + SocketIO + Plotly.

    Usage:
        dashboard = LiveDashboard()
        dashboard.start()                    # starts background Flask server
        dashboard.update_portfolio(snapshot)  # push new data
        dashboard.record_trade(trade)         # push trade
        dashboard.stop()
    """

    def __init__(self, config: Optional[DashboardConfig] = None):
        self.config = config or DashboardConfig()
        self._history_timestamps: deque = deque(maxlen=self.config.max_history_points)
        self._history_values: deque = deque(maxlen=self.config.max_history_points)
        self._history_drawdowns: deque = deque(maxlen=self.config.max_history_points)
        self._daily_pnl_dates: deque = deque(maxlen=252)
        self._daily_pnl_values: deque = deque(maxlen=252)
        self._trades: deque = deque(maxlen=200)
        self._latest_snapshot: Optional[PortfolioSnapshot] = None
        self._peak_value: float = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Flask app
        self.app: Optional[Any] = None
        self.socketio: Optional[Any] = None

        if FLASK_AVAILABLE:
            self._build_app()

        logger.info("LiveDashboard initialized (port=%d)", self.config.port)

    # ── Flask app construction ───────────────────────────────────────────

    def _build_app(self):
        self.app = Flask(__name__)
        CORS(self.app)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode="threading")

        theme = self.config.theme
        theme_vars = {
            "dark": dict(bg_color="#121220", text_color="#e0e0e0", header_bg="#1e1e2f",
                         card_bg="#1e1e2f", border_color="#2d2d44"),
            "light": dict(bg_color="#f8f9fa", text_color="#333", header_bg="#fff",
                          card_bg="#fff", border_color="#dee2e6"),
        }
        tv = theme_vars.get(theme, theme_vars["dark"])

        dashboard_ref = self  # closure reference

        @self.app.route("/")
        def index():
            return render_template_string(DASHBOARD_HTML, title=dashboard_ref.config.title, **tv)

        @self.app.route("/api/state")
        def api_state():
            return jsonify(dashboard_ref._build_state_payload())

        @self.app.route("/api/health")
        def api_health():
            return jsonify({"status": "healthy", "uptime": time.time()})

    # ── State payload ────────────────────────────────────────────────────

    def _build_state_payload(self) -> dict:
        """Build full state for initial load or polling."""
        snap = self._latest_snapshot or PortfolioSnapshot()
        ts = list(self._history_timestamps)
        vals = list(self._history_values)
        dds = list(self._history_drawdowns)
        theme = self.config.theme

        payload: Dict[str, Any] = {"snapshot": asdict(snap)}

        if FLASK_AVAILABLE and ts:
            payload["equity_chart"] = ChartBuilder.equity_curve(ts, vals, dds, theme)
            payload["position_chart"] = ChartBuilder.position_heatmap(snap.positions, theme)
            payload["dailypnl_chart"] = ChartBuilder.daily_pnl_bar(
                list(self._daily_pnl_dates), list(self._daily_pnl_values), theme,
            )
            payload["risk_chart"] = ChartBuilder.risk_gauges(
                snap.sharpe_ratio, snap.max_drawdown_pct,
                snap.var_95, snap.win_rate, theme,
            )

        payload["trades"] = [asdict(t) for t in list(self._trades)[-50:]]
        return payload

    # ── Public API ───────────────────────────────────────────────────────

    def update_portfolio(self, snapshot: PortfolioSnapshot) -> None:
        """Push a new portfolio snapshot; emits via WebSocket."""
        if not snapshot.timestamp:
            snapshot.timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        self._latest_snapshot = snapshot
        self._history_timestamps.append(snapshot.timestamp)
        self._history_values.append(snapshot.portfolio_value)

        # Compute drawdown
        if snapshot.portfolio_value > self._peak_value:
            self._peak_value = snapshot.portfolio_value
        dd = 0.0 if self._peak_value == 0 else (
            (self._peak_value - snapshot.portfolio_value) / self._peak_value * 100
        )
        self._history_drawdowns.append(dd)
        snapshot.max_drawdown_pct = max(snapshot.max_drawdown_pct, dd)

        # Daily P&L tracking
        today = snapshot.timestamp[:10]
        if not self._daily_pnl_dates or self._daily_pnl_dates[-1] != today:
            self._daily_pnl_dates.append(today)
            self._daily_pnl_values.append(snapshot.daily_pnl)
        else:
            self._daily_pnl_values[-1] = snapshot.daily_pnl

        # Push via WebSocket
        if self.socketio and self._running:
            try:
                self.socketio.emit("snapshot", self._build_state_payload())
            except Exception as e:
                logger.warning("WebSocket emit failed: %s", e)

    def record_trade(self, trade: TradeRecord) -> None:
        """Record a trade and push it to connected clients."""
        if not trade.timestamp:
            trade.timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        self._trades.append(trade)

        if self.socketio and self._running:
            try:
                self.socketio.emit("trade", asdict(trade))
            except Exception as e:
                logger.warning("Trade emit failed: %s", e)

    def get_snapshot(self) -> Optional[PortfolioSnapshot]:
        """Return current snapshot."""
        return self._latest_snapshot

    def get_trades(self, n: int = 50) -> List[TradeRecord]:
        """Return recent trades."""
        return list(self._trades)[-n:]

    def get_equity_history(self) -> Tuple[List[str], List[float], List[float]]:
        """Return (timestamps, values, drawdowns) history."""
        return (
            list(self._history_timestamps),
            list(self._history_values),
            list(self._history_drawdowns),
        )

    # ── Server lifecycle ─────────────────────────────────────────────────

    def start(self, blocking: bool = False) -> None:
        """Start dashboard server."""
        if not FLASK_AVAILABLE:
            logger.error("Cannot start dashboard — Flask not installed.")
            return

        self._running = True
        logger.info("Starting dashboard on http://%s:%d", self.config.host, self.config.port)

        if blocking:
            self.socketio.run(
                self.app, host=self.config.host, port=self.config.port,
                debug=self.config.debug, use_reloader=False,
            )
        else:
            self._thread = threading.Thread(
                target=self.socketio.run,
                kwargs=dict(
                    app=self.app, host=self.config.host, port=self.config.port,
                    debug=False, use_reloader=False, log_output=False,
                ),
                daemon=True,
            )
            self._thread.start()
            logger.info("Dashboard running in background thread.")

    def stop(self) -> None:
        """Stop dashboard server."""
        self._running = False
        logger.info("Dashboard stopped.")

    @property
    def is_running(self) -> bool:
        return self._running


# =============================================================================
# STANDALONE ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    dash = LiveDashboard(DashboardConfig(port=5050, theme="dark"))

    # Simulate data
    import random
    value = 100_000.0
    for i in range(60):
        ret = random.gauss(0.001, 0.015)
        value *= (1 + ret)
        snap = PortfolioSnapshot(
            portfolio_value=value,
            cash=value * 0.2,
            daily_pnl=value * ret,
            daily_return_pct=ret * 100,
            total_return_pct=(value / 100_000 - 1) * 100,
            positions={
                "SPY": {"weight_pct": 30, "unrealized_pnl": random.uniform(-500, 800)},
                "QQQ": {"weight_pct": 25, "unrealized_pnl": random.uniform(-400, 600)},
                "IWM": {"weight_pct": 15, "unrealized_pnl": random.uniform(-300, 400)},
            },
            regime="bull" if ret > 0 else "bear",
            sharpe_ratio=1.8 + random.gauss(0, 0.3),
            max_drawdown_pct=random.uniform(1, 5),
            var_95=random.uniform(1, 4),
            win_rate=random.uniform(50, 65),
            num_trades_today=random.randint(0, 5),
            open_positions=3,
        )
        dash.update_portfolio(snap)

    dash.start(blocking=True)
