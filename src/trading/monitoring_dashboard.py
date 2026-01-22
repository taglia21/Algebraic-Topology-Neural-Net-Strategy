"""
V2.1 Production Monitoring Dashboard
=====================================

Real-time monitoring dashboard with:
- Sharpe ratio tracking
- Drawdown alerts
- Execution quality metrics
- Component health status
- Trade history visualization

Usage:
    dashboard = MonitoringDashboard(port=8080)
    dashboard.start_server()
    dashboard.update_metrics({"equity": 100000, "sharpe": 1.5})
"""

import json
import logging
import threading
import http.server
import socketserver
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
import os

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Real-time performance metrics."""
    timestamp: str = ""
    equity: float = 0.0
    cash: float = 0.0
    buying_power: float = 0.0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    positions_count: int = 0
    
    # Risk metrics
    drawdown: float = 0.0
    max_drawdown: float = 0.0
    sharpe_30d: float = 0.0
    sharpe_60d: float = 0.0
    volatility_20d: float = 0.0
    
    # Execution metrics
    trades_today: int = 0
    win_rate_30d: float = 0.0
    avg_slippage_bp: float = 0.0
    fill_rate: float = 0.0
    
    # Component status
    ensemble_regime: bool = False
    transformer: bool = False
    tda_generator: bool = False
    is_halted: bool = False
    halt_reason: str = ""
    
    # Market regime
    current_regime: str = "neutral"
    regime_confidence: float = 0.5


@dataclass
class AlertLevel:
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """System alert."""
    timestamp: str
    level: str
    title: str
    message: str
    acknowledged: bool = False


class MonitoringDashboard:
    """
    Real-time monitoring dashboard server.
    
    Provides:
    - HTTP server on configurable port
    - Real-time metrics updates
    - Alert management
    - Trade history
    - Component health
    """
    
    def __init__(self, port: int = 8080):
        self.port = port
        self.metrics = PerformanceMetrics()
        self.alerts: List[Alert] = []
        self.trade_history: List[Dict] = []
        self.daily_returns: List[float] = []
        
        self._server: Optional[socketserver.TCPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._is_running = False
        
        # Thresholds for alerts
        self.alert_thresholds = {
            "drawdown_warning": 0.03,
            "drawdown_critical": 0.05,
            "sharpe_warning": 0.8,
            "sharpe_critical": 0.5,
            "slippage_warning": 10,  # basis points
            "slippage_critical": 20,
        }
        
        logger.info(f"MonitoringDashboard initialized on port {port}")
        
    def update_metric(self, key: str, value: Any):
        """Update a single metric."""
        if hasattr(self.metrics, key):
            setattr(self.metrics, key, value)
            self._check_alerts(key, value)
        self.metrics.timestamp = datetime.now().isoformat()
        
    def update_metrics(self, metrics_dict: Dict[str, Any]):
        """Update multiple metrics at once."""
        for key, value in metrics_dict.items():
            if hasattr(self.metrics, key):
                setattr(self.metrics, key, value)
                self._check_alerts(key, value)
        self.metrics.timestamp = datetime.now().isoformat()
        
    def add_trade(self, trade: Dict):
        """Add trade to history."""
        self.trade_history.append({
            **trade,
            "dashboard_timestamp": datetime.now().isoformat()
        })
        
        # Keep last 1000 trades
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-1000:]
            
    def add_daily_return(self, ret: float):
        """Add daily return for Sharpe calculation."""
        self.daily_returns.append(ret)
        
        # Keep 90 days
        if len(self.daily_returns) > 90:
            self.daily_returns = self.daily_returns[-90:]
            
        # Update rolling Sharpe
        self._update_rolling_sharpe()
        
    def _update_rolling_sharpe(self):
        """Calculate rolling Sharpe ratios."""
        import numpy as np
        
        if len(self.daily_returns) >= 30:
            returns_30d = self.daily_returns[-30:]
            mean_30 = np.mean(returns_30d)
            std_30 = np.std(returns_30d) + 1e-8
            self.metrics.sharpe_30d = (mean_30 / std_30) * np.sqrt(252)
            
        if len(self.daily_returns) >= 60:
            returns_60d = self.daily_returns[-60:]
            mean_60 = np.mean(returns_60d)
            std_60 = np.std(returns_60d) + 1e-8
            self.metrics.sharpe_60d = (mean_60 / std_60) * np.sqrt(252)
            
    def _check_alerts(self, key: str, value: Any):
        """Check if value triggers an alert."""
        now = datetime.now().isoformat()
        
        if key == "drawdown":
            if value >= self.alert_thresholds["drawdown_critical"]:
                self._add_alert(AlertLevel.CRITICAL, "Critical Drawdown",
                              f"Drawdown at {value:.1%} - Circuit breaker threshold reached")
            elif value >= self.alert_thresholds["drawdown_warning"]:
                self._add_alert(AlertLevel.WARNING, "Drawdown Warning",
                              f"Drawdown at {value:.1%} - Approaching threshold")
                              
        elif key == "sharpe_30d":
            if value < self.alert_thresholds["sharpe_critical"]:
                self._add_alert(AlertLevel.CRITICAL, "Low Sharpe Ratio",
                              f"30-day Sharpe at {value:.2f} - Below critical threshold")
            elif value < self.alert_thresholds["sharpe_warning"]:
                self._add_alert(AlertLevel.WARNING, "Sharpe Warning",
                              f"30-day Sharpe at {value:.2f} - Below target")
                              
        elif key == "avg_slippage_bp":
            if value > self.alert_thresholds["slippage_critical"]:
                self._add_alert(AlertLevel.CRITICAL, "High Slippage",
                              f"Average slippage at {value:.1f}bp - Review execution")
            elif value > self.alert_thresholds["slippage_warning"]:
                self._add_alert(AlertLevel.WARNING, "Slippage Warning",
                              f"Average slippage at {value:.1f}bp")
                              
        elif key == "is_halted" and value:
            self._add_alert(AlertLevel.CRITICAL, "Trading Halted",
                          f"Trading has been halted: {self.metrics.halt_reason}")
                          
    def _add_alert(self, level: str, title: str, message: str):
        """Add new alert."""
        alert = Alert(
            timestamp=datetime.now().isoformat(),
            level=level,
            title=title,
            message=message,
        )
        self.alerts.append(alert)
        
        # Keep last 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
            
        logger.warning(f"Alert: [{level}] {title} - {message}")
        
    def acknowledge_alert(self, index: int):
        """Acknowledge an alert."""
        if 0 <= index < len(self.alerts):
            self.alerts[index].acknowledged = True
            
    def get_dashboard_html(self) -> str:
        """Generate dashboard HTML."""
        metrics = asdict(self.metrics)
        
        # Determine status colors
        dd_color = "#ff4444" if self.metrics.drawdown > 0.03 else "#00ff88" if self.metrics.drawdown < 0.01 else "#ffaa00"
        sharpe_color = "#00ff88" if self.metrics.sharpe_30d > 1.0 else "#ffaa00" if self.metrics.sharpe_30d > 0.5 else "#ff4444"
        pnl_color = "#00ff88" if self.metrics.daily_pnl >= 0 else "#ff4444"
        
        # Component status badges
        ensemble_badge = "✅" if self.metrics.ensemble_regime else "❌"
        transformer_badge = "✅" if self.metrics.transformer else "❌"
        tda_badge = "✅" if self.metrics.tda_generator else "❌"
        halt_badge = "🔴 HALTED" if self.metrics.is_halted else "🟢 ACTIVE"
        
        # Recent alerts
        recent_alerts = self.alerts[-5:][::-1]  # Last 5, newest first
        alerts_html = ""
        for alert in recent_alerts:
            alert_color = "#ff4444" if alert.level == "critical" else "#ffaa00" if alert.level == "warning" else "#00d4ff"
            alerts_html += f'''
            <div class="alert" style="border-left: 3px solid {alert_color};">
                <strong>{alert.title}</strong>
                <p>{alert.message}</p>
                <small>{alert.timestamp}</small>
            </div>
            '''
            
        # Recent trades
        recent_trades = self.trade_history[-10:][::-1]
        trades_html = ""
        for trade in recent_trades:
            side_color = "#00ff88" if trade.get("side") == "buy" else "#ff4444"
            trades_html += f'''
            <tr>
                <td>{trade.get("timestamp", "")[:19]}</td>
                <td>{trade.get("symbol", "")}</td>
                <td style="color: {side_color};">{trade.get("side", "").upper()}</td>
                <td>{trade.get("qty", 0)}</td>
                <td>${trade.get("price", 0):,.2f}</td>
            </tr>
            '''
        
        return f'''
<!DOCTYPE html>
<html>
<head>
    <title>V2.1 Production Dashboard</title>
    <meta http-equiv="refresh" content="30">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', Arial, sans-serif; 
            background: #0d1117; 
            color: #c9d1d9; 
            padding: 20px;
            min-height: 100vh;
        }}
        .header {{ 
            text-align: center; 
            margin-bottom: 30px; 
            border-bottom: 1px solid #30363d;
            padding-bottom: 20px;
        }}
        .header h1 {{ color: #58a6ff; font-size: 2em; margin-bottom: 10px; }}
        .header .status {{ font-size: 1.2em; }}
        .grid {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
            gap: 15px; 
            margin-bottom: 30px; 
        }}
        .card {{ 
            background: #161b22; 
            border: 1px solid #30363d;
            border-radius: 8px; 
            padding: 20px; 
            text-align: center; 
        }}
        .card h3 {{ color: #8b949e; font-size: 0.8em; text-transform: uppercase; margin-bottom: 10px; }}
        .card .value {{ font-size: 1.8em; font-weight: bold; }}
        .section {{ 
            background: #161b22; 
            border: 1px solid #30363d;
            border-radius: 8px; 
            padding: 20px; 
            margin-bottom: 20px;
        }}
        .section h2 {{ color: #58a6ff; margin-bottom: 15px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #30363d; }}
        th {{ color: #8b949e; font-size: 0.85em; }}
        .alert {{ 
            background: #21262d; 
            padding: 15px; 
            margin-bottom: 10px; 
            border-radius: 6px; 
        }}
        .alert p {{ margin: 5px 0; color: #8b949e; }}
        .alert small {{ color: #6e7681; }}
        .components {{ display: flex; gap: 20px; flex-wrap: wrap; }}
        .component {{ 
            background: #21262d; 
            padding: 10px 20px; 
            border-radius: 6px; 
        }}
        .footer {{ 
            text-align: center; 
            color: #6e7681; 
            margin-top: 30px; 
            font-size: 0.9em; 
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 V2.1 Production Dashboard</h1>
        <div class="status">{halt_badge} | Regime: {self.metrics.current_regime.upper()} ({self.metrics.regime_confidence:.0%})</div>
        <small>Last updated: {self.metrics.timestamp}</small>
    </div>
    
    <div class="grid">
        <div class="card">
            <h3>Equity</h3>
            <div class="value" style="color: #58a6ff;">${self.metrics.equity:,.0f}</div>
        </div>
        <div class="card">
            <h3>Daily P&L</h3>
            <div class="value" style="color: {pnl_color};">${self.metrics.daily_pnl:+,.0f}</div>
        </div>
        <div class="card">
            <h3>Drawdown</h3>
            <div class="value" style="color: {dd_color};">{self.metrics.drawdown:.2%}</div>
        </div>
        <div class="card">
            <h3>30-Day Sharpe</h3>
            <div class="value" style="color: {sharpe_color};">{self.metrics.sharpe_30d:.2f}</div>
        </div>
        <div class="card">
            <h3>Positions</h3>
            <div class="value">{self.metrics.positions_count}</div>
        </div>
        <div class="card">
            <h3>Trades Today</h3>
            <div class="value">{self.metrics.trades_today}</div>
        </div>
        <div class="card">
            <h3>Win Rate (30d)</h3>
            <div class="value">{self.metrics.win_rate_30d:.0%}</div>
        </div>
        <div class="card">
            <h3>Avg Slippage</h3>
            <div class="value">{self.metrics.avg_slippage_bp:.1f}bp</div>
        </div>
    </div>
    
    <div class="section">
        <h2>🔧 Component Status</h2>
        <div class="components">
            <div class="component">{ensemble_badge} Ensemble Regime</div>
            <div class="component">{transformer_badge} Transformer</div>
            <div class="component">{tda_badge} TDA Generator</div>
        </div>
    </div>
    
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
        <div class="section">
            <h2>⚠️ Recent Alerts</h2>
            {alerts_html if alerts_html else '<p style="color: #6e7681;">No recent alerts</p>'}
        </div>
        
        <div class="section">
            <h2>📈 Recent Trades</h2>
            <table>
                <thead>
                    <tr><th>Time</th><th>Symbol</th><th>Side</th><th>Qty</th><th>Price</th></tr>
                </thead>
                <tbody>
                    {trades_html if trades_html else '<tr><td colspan="5" style="color: #6e7681;">No trades today</td></tr>'}
                </tbody>
            </table>
        </div>
    </div>
    
    <div class="footer">
        V2.1 Production System | Auto-refresh every 30s | 
        <a href="/api/metrics" style="color: #58a6ff;">API Endpoint</a>
    </div>
</body>
</html>
        '''
        
    def get_metrics_json(self) -> str:
        """Get metrics as JSON."""
        return json.dumps({
            "metrics": asdict(self.metrics),
            "alerts": [asdict(a) for a in self.alerts[-10:]],
            "recent_trades": self.trade_history[-20:],
        }, indent=2)
        
    def start_server(self):
        """Start the HTTP server in a background thread."""
        if self._is_running:
            return
            
        dashboard = self
        
        class DashboardHandler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/" or self.path == "/dashboard":
                    self.send_response(200)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()
                    self.wfile.write(dashboard.get_dashboard_html().encode())
                elif self.path == "/api/metrics":
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(dashboard.get_metrics_json().encode())
                elif self.path == "/health":
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "ok"}).encode())
                else:
                    self.send_response(404)
                    self.end_headers()
                    
            def log_message(self, format, *args):
                pass  # Suppress HTTP logs
                
        def run_server():
            try:
                with socketserver.TCPServer(("", self.port), DashboardHandler) as httpd:
                    self._server = httpd
                    self._is_running = True
                    logger.info(f"Dashboard server started on port {self.port}")
                    httpd.serve_forever()
            except Exception as e:
                logger.error(f"Dashboard server error: {e}")
                self._is_running = False
                
        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()
        
    def stop_server(self):
        """Stop the HTTP server."""
        if self._server:
            self._server.shutdown()
            self._is_running = False
            logger.info("Dashboard server stopped")


# =============================================================================
# EXECUTION QUALITY TRACKER
# =============================================================================

class ExecutionQualityTracker:
    """
    Track execution quality metrics:
    - Slippage analysis
    - Fill rates
    - Timing impact
    """
    
    def __init__(self):
        self.executions: List[Dict] = []
        
    def record_execution(self, order: Dict, fill: Dict):
        """Record an order execution for analysis."""
        expected_price = order.get("expected_price", 0)
        fill_price = fill.get("price", 0)
        
        if expected_price > 0 and fill_price > 0:
            slippage_bp = ((fill_price - expected_price) / expected_price) * 10000
            if order.get("side") == "sell":
                slippage_bp = -slippage_bp
                
            self.executions.append({
                "timestamp": datetime.now().isoformat(),
                "symbol": order.get("symbol"),
                "side": order.get("side"),
                "expected_price": expected_price,
                "fill_price": fill_price,
                "slippage_bp": slippage_bp,
                "qty": fill.get("qty", 0),
                "fill_rate": fill.get("filled_qty", 0) / order.get("qty", 1),
            })
            
        # Keep last 1000
        if len(self.executions) > 1000:
            self.executions = self.executions[-1000:]
            
    def get_stats(self, days: int = 30) -> Dict[str, float]:
        """Get execution quality stats for period."""
        import numpy as np
        
        cutoff = datetime.now() - timedelta(days=days)
        recent = [e for e in self.executions 
                 if datetime.fromisoformat(e["timestamp"]) > cutoff]
        
        if not recent:
            return {
                "avg_slippage_bp": 0.0,
                "slippage_std_bp": 0.0,
                "fill_rate": 1.0,
                "execution_count": 0,
            }
            
        slippages = [e["slippage_bp"] for e in recent]
        fill_rates = [e["fill_rate"] for e in recent]
        
        return {
            "avg_slippage_bp": float(np.mean(slippages)),
            "slippage_std_bp": float(np.std(slippages)),
            "fill_rate": float(np.mean(fill_rates)),
            "execution_count": len(recent),
        }
