#!/usr/bin/env python3
"""
V36 Monitoring Dashboard
========================
Flask-based real-time trading dashboard with Alpaca integration.

Features:
- Real-time P&L display with 30-second auto-refresh
- Current positions table with unrealized P&L
- HMM regime indicator (Bull/Bear/Sideways/Crisis)
- Circuit breaker status panel (green/yellow/red)
- Recent trades log (last 20 trades)
- REST API endpoint: GET /api/status

Usage:
    python v36_monitoring_dashboard.py
    # Open http://localhost:8080 in browser
"""

import logging
import os
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

import requests
from flask import Flask, render_template_string, jsonify, request

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('V36_Dashboard')


class MarketRegime(Enum):
    """Market regime from HMM."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    CRISIS = "crisis"


class CircuitBreakerStatus(Enum):
    """Circuit breaker status levels."""
    GREEN = "green"    # Normal trading
    YELLOW = "yellow"  # Reduced position sizing
    RED = "red"        # Trading halted


@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    port: int = 8080
    refresh_seconds: int = 30
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    
    def __post_init__(self):
        self.alpaca_api_key = os.getenv('ALPACA_API_KEY', self.alpaca_api_key)
        self.alpaca_secret_key = os.getenv('ALPACA_SECRET_KEY', self.alpaca_secret_key)
        self.alpaca_base_url = os.getenv('ALPACA_BASE_URL', self.alpaca_base_url)


@dataclass
class Position:
    """Trading position."""
    symbol: str
    qty: int
    side: str
    entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float


@dataclass
class Trade:
    """Trade record."""
    symbol: str
    side: str
    qty: int
    price: float
    timestamp: str
    status: str


class AlpacaClient:
    """Alpaca API client for dashboard data."""

    def __init__(self, config: DashboardConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'APCA-API-KEY-ID': config.alpaca_api_key,
            'APCA-API-SECRET-KEY': config.alpaca_secret_key
        })

    def _request(self, method: str, endpoint: str, **kwargs) -> Optional[Any]:
        """Make API request."""
        try:
            url = f"{self.config.alpaca_base_url}{endpoint}"
            resp = self.session.request(method, url, timeout=10, **kwargs)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.error(f"Alpaca API error: {e}")
            return None

    def get_account(self) -> Dict[str, Any]:
        """Get account information."""
        return self._request('GET', '/v2/account') or {}

    def get_positions(self) -> List[Position]:
        """Get all positions."""
        data = self._request('GET', '/v2/positions') or []
        positions = []
        for p in data:
            positions.append(Position(
                symbol=p.get('symbol', ''),
                qty=int(float(p.get('qty', 0))),
                side=p.get('side', 'long'),
                entry_price=float(p.get('avg_entry_price', 0)),
                current_price=float(p.get('current_price', 0)),
                market_value=float(p.get('market_value', 0)),
                unrealized_pnl=float(p.get('unrealized_pl', 0)),
                unrealized_pnl_pct=float(p.get('unrealized_plpc', 0)) * 100
            ))
        return positions

    def get_orders(self, limit: int = 20) -> List[Trade]:
        """Get recent orders."""
        data = self._request('GET', '/v2/orders', params={'status': 'all', 'limit': limit}) or []
        trades = []
        for o in data:
            trades.append(Trade(
                symbol=o.get('symbol', ''),
                side=o.get('side', ''),
                qty=int(float(o.get('filled_qty', 0) or o.get('qty', 0))),
                price=float(o.get('filled_avg_price', 0) or 0),
                timestamp=o.get('filled_at', o.get('created_at', '')),
                status=o.get('status', '')
            ))
        return trades


class DashboardState:
    """Shared dashboard state."""

    def __init__(self):
        self.equity: float = 0.0
        self.cash: float = 0.0
        self.buying_power: float = 0.0
        self.day_pnl: float = 0.0
        self.day_pnl_pct: float = 0.0
        self.total_pnl: float = 0.0
        self.positions: List[Position] = []
        self.trades: List[Trade] = []
        self.regime: MarketRegime = MarketRegime.SIDEWAYS
        self.circuit_breaker: CircuitBreakerStatus = CircuitBreakerStatus.GREEN
        self.last_update: str = ""
        self.drawdown_pct: float = 0.0
        self._lock = threading.Lock()

    def update(self, client: AlpacaClient) -> None:
        """Update state from Alpaca."""
        with self._lock:
            account = client.get_account()
            if account:
                self.equity = float(account.get('equity', 0))
                self.cash = float(account.get('cash', 0))
                self.buying_power = float(account.get('buying_power', 0))
                last_equity = float(account.get('last_equity', self.equity))
                self.day_pnl = self.equity - last_equity
                self.day_pnl_pct = (self.day_pnl / last_equity * 100) if last_equity else 0
            
            self.positions = client.get_positions()
            self.trades = client.get_orders(limit=20)
            self._update_circuit_breaker()
            self.last_update = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def _update_circuit_breaker(self) -> None:
        """Update circuit breaker based on drawdown."""
        if self.day_pnl_pct <= -5.0:
            self.circuit_breaker = CircuitBreakerStatus.RED
        elif self.day_pnl_pct <= -2.0:
            self.circuit_breaker = CircuitBreakerStatus.YELLOW
        else:
            self.circuit_breaker = CircuitBreakerStatus.GREEN

    def set_regime(self, regime: MarketRegime) -> None:
        """Set current market regime."""
        with self._lock:
            self.regime = regime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API."""
        with self._lock:
            return {
                'equity': self.equity,
                'cash': self.cash,
                'buying_power': self.buying_power,
                'day_pnl': self.day_pnl,
                'day_pnl_pct': self.day_pnl_pct,
                'positions': [asdict(p) for p in self.positions],
                'trades': [asdict(t) for t in self.trades[:20]],
                'regime': self.regime.value,
                'circuit_breaker': self.circuit_breaker.value,
                'last_update': self.last_update
            }


# HTML template with Bootstrap
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="refresh" content="{{ refresh_seconds }}">
    <title>V36 Trading Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #1a1a2e; color: #eee; }
        .card { background-color: #16213e; border: 1px solid #0f3460; }
        .card-header { background-color: #0f3460; border-bottom: 1px solid #0f3460; }
        .table { color: #eee; }
        .positive { color: #00ff88; }
        .negative { color: #ff4757; }
        .regime-bull { background-color: #00ff88; color: #000; }
        .regime-bear { background-color: #ff4757; color: #fff; }
        .regime-sideways { background-color: #ffa502; color: #000; }
        .regime-crisis { background-color: #8b0000; color: #fff; }
        .cb-green { background-color: #00ff88; }
        .cb-yellow { background-color: #ffa502; }
        .cb-red { background-color: #ff4757; }
        .status-badge { padding: 8px 16px; border-radius: 4px; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container-fluid py-4">
        <div class="row mb-4">
            <div class="col-12">
                <h2>V36 Trading Dashboard <small class="text-muted">Last update: {{ state.last_update }}</small></h2>
            </div>
        </div>
        
        <!-- P&L and Status Cards -->
        <div class="row mb-4">
            <div class="col-md-3">
                <div class="card">
                    <div class="card-header">Portfolio Value</div>
                    <div class="card-body">
                        <h3>${{ "%.2f"|format(state.equity) }}</h3>
                        <p class="mb-0">Cash: ${{ "%.2f"|format(state.cash) }}</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card">
                    <div class="card-header">Day P&L</div>
                    <div class="card-body">
                        <h3 class="{{ 'positive' if state.day_pnl >= 0 else 'negative' }}">
                            ${{ "%.2f"|format(state.day_pnl) }} ({{ "%.2f"|format(state.day_pnl_pct) }}%)
                        </h3>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card">
                    <div class="card-header">Market Regime</div>
                    <div class="card-body">
                        <span class="status-badge regime-{{ state.regime }}">{{ state.regime|upper }}</span>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card">
                    <div class="card-header">Circuit Breaker</div>
                    <div class="card-body">
                        <span class="status-badge cb-{{ state.circuit_breaker }}">{{ state.circuit_breaker|upper }}</span>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Positions Table -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">Current Positions ({{ state.positions|length }})</div>
                    <div class="card-body">
                        <table class="table table-sm table-hover">
                            <thead>
                                <tr>
                                    <th>Symbol</th><th>Side</th><th>Qty</th>
                                    <th>Entry</th><th>Current</th><th>Value</th><th>P&L</th><th>P&L %</th>
                                </tr>
                            </thead>
                            <tbody>
                            {% for p in state.positions %}
                                <tr>
                                    <td><strong>{{ p.symbol }}</strong></td>
                                    <td>{{ p.side }}</td>
                                    <td>{{ p.qty }}</td>
                                    <td>${{ "%.2f"|format(p.entry_price) }}</td>
                                    <td>${{ "%.2f"|format(p.current_price) }}</td>
                                    <td>${{ "%.2f"|format(p.market_value) }}</td>
                                    <td class="{{ 'positive' if p.unrealized_pnl >= 0 else 'negative' }}">
                                        ${{ "%.2f"|format(p.unrealized_pnl) }}
                                    </td>
                                    <td class="{{ 'positive' if p.unrealized_pnl_pct >= 0 else 'negative' }}">
                                        {{ "%.2f"|format(p.unrealized_pnl_pct) }}%
                                    </td>
                                </tr>
                            {% else %}
                                <tr><td colspan="8" class="text-center">No open positions</td></tr>
                            {% endfor %}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Recent Trades -->
        <div class="row">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">Recent Trades (Last 20)</div>
                    <div class="card-body">
                        <table class="table table-sm table-hover">
                            <thead>
                                <tr><th>Time</th><th>Symbol</th><th>Side</th><th>Qty</th><th>Price</th><th>Status</th></tr>
                            </thead>
                            <tbody>
                            {% for t in state.trades[:20] %}
                                <tr>
                                    <td>{{ t.timestamp[:19] if t.timestamp else 'N/A' }}</td>
                                    <td><strong>{{ t.symbol }}</strong></td>
                                    <td class="{{ 'positive' if t.side == 'buy' else 'negative' }}">{{ t.side|upper }}</td>
                                    <td>{{ t.qty }}</td>
                                    <td>${{ "%.2f"|format(t.price) if t.price else 'N/A' }}</td>
                                    <td>{{ t.status }}</td>
                                </tr>
                            {% else %}
                                <tr><td colspan="6" class="text-center">No recent trades</td></tr>
                            {% endfor %}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""


def create_app(config: Optional[DashboardConfig] = None) -> Flask:
    """Create Flask application."""
    config = config or DashboardConfig()
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.urandom(24)
    
    client = AlpacaClient(config)
    state = DashboardState()
    
    def background_update():
        """Background thread for updating state."""
        while True:
            try:
                state.update(client)
                logger.debug("Dashboard state updated")
            except Exception as e:
                logger.error(f"Update error: {e}")
            time.sleep(config.refresh_seconds)
    
    update_thread = threading.Thread(target=background_update, daemon=True)
    update_thread.start()
    
    @app.route('/')
    def dashboard():
        """Main dashboard page."""
        return render_template_string(
            DASHBOARD_HTML, 
            state=state.to_dict(), 
            refresh_seconds=config.refresh_seconds
        )
    
    @app.route('/api/status')
    def api_status():
        """API endpoint returning JSON status."""
        return jsonify(state.to_dict())
    
    @app.route('/api/regime', methods=['POST'])
    def set_regime():
        """Set market regime (for external integration)."""
        data = request.get_json() if hasattr(request, 'get_json') else {}
        regime_str = data.get('regime', 'sideways')
        try:
            regime = MarketRegime(regime_str.lower())
            state.set_regime(regime)
            return jsonify({'status': 'ok', 'regime': regime.value})
        except ValueError:
            return jsonify({'status': 'error', 'message': 'Invalid regime'}), 400
    
    return app


def main() -> None:
    """Run the dashboard."""
    config = DashboardConfig()
    app = create_app(config)
    
    logger.info(f"Starting V36 Dashboard on http://localhost:{config.port}")
    app.run(host='0.0.0.0', port=config.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
