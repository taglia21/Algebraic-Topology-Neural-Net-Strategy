#!/usr/bin/env python3
"""
V28 Real-Time Dashboard API
============================
WebSocket and REST API endpoints for live trading dashboard integration.

Features:
- WebSocket endpoints for live P&L streaming
- REST API endpoints for metrics, positions, trades
- Health check and monitoring endpoints
- Redis-backed caching for performance

Endpoints:
- GET  /api/metrics       - Sharpe, CAGR, MaxDD, WinRate
- GET  /api/positions     - Current holdings
- GET  /api/trades        - Recent executions
- GET  /api/regime        - Current market regime
- GET  /api/health        - System health status
- WS   /ws/pnl            - Live P&L streaming
- WS   /ws/trades         - Live trade notifications
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from pathlib import Path
import threading

import numpy as np
import pandas as pd

# Optional imports with fallbacks
try:
    from aiohttp import web
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('V28_Dashboard')


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class PerformanceMetrics:
    """Trading performance metrics."""
    sharpe_ratio: float = 0.0
    cagr: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    total_trades: int = 0
    total_pnl: float = 0.0
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    monthly_pnl: float = 0.0
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Position:
    """Current position information."""
    symbol: str
    side: str  # 'long' or 'short'
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    entry_time: str
    holding_period_hours: float
    regime_at_entry: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Trade:
    """Trade execution record."""
    trade_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    timestamp: str
    pnl: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    signal_source: str = ""
    regime: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RegimeInfo:
    """Current market regime information."""
    regime: str  # 'bull', 'bear', 'sideways'
    volatility: str  # 'high', 'normal', 'low'
    hmm_state: int = 0
    hmm_state_name: str = ""
    garch_forecast: float = 0.0
    vix_level: float = 0.0
    confidence: float = 0.0
    strategy_allocation: Dict[str, float] = field(default_factory=dict)
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HealthStatus:
    """System health status."""
    status: str  # 'healthy', 'degraded', 'unhealthy'
    uptime_seconds: float = 0.0
    last_trade_time: str = ""
    last_data_update: str = ""
    api_latency_ms: float = 0.0
    websocket_connections: int = 0
    memory_usage_mb: float = 0.0
    cpu_usage_pct: float = 0.0
    errors_last_hour: int = 0
    warnings_last_hour: int = 0
    redis_connected: bool = False
    broker_connected: bool = False
    data_feed_connected: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# METRICS CALCULATOR
# =============================================================================

class MetricsCalculator:
    """Calculate trading performance metrics from trade history."""
    
    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate
        self.daily_returns: List[float] = []
        self.equity_curve: List[float] = []
        self.trades: List[Trade] = []
        self.start_equity: float = 100000.0
        self.current_equity: float = 100000.0
        
    def update_equity(self, equity: float):
        """Update current equity and track return."""
        if self.equity_curve:
            daily_ret = (equity - self.equity_curve[-1]) / self.equity_curve[-1]
            self.daily_returns.append(daily_ret)
        self.equity_curve.append(equity)
        self.current_equity = equity
        
    def add_trade(self, trade: Trade):
        """Add completed trade to history."""
        self.trades.append(trade)
        
    def calculate_sharpe(self, annualized: bool = True) -> float:
        """Calculate Sharpe ratio."""
        if len(self.daily_returns) < 2:
            return 0.0
        
        returns = np.array(self.daily_returns)
        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)
        
        if std_ret == 0:
            return 0.0
        
        sharpe = (mean_ret - self.risk_free_rate / 252) / std_ret
        
        if annualized:
            sharpe *= np.sqrt(252)
        
        return float(sharpe)
    
    def calculate_sortino(self, annualized: bool = True) -> float:
        """Calculate Sortino ratio (downside deviation only)."""
        if len(self.daily_returns) < 2:
            return 0.0
        
        returns = np.array(self.daily_returns)
        mean_ret = np.mean(returns)
        
        # Only consider negative returns for downside deviation
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return float('inf') if mean_ret > 0 else 0.0
        
        downside_std = np.std(downside_returns, ddof=1)
        
        if downside_std == 0:
            return 0.0
        
        sortino = (mean_ret - self.risk_free_rate / 252) / downside_std
        
        if annualized:
            sortino *= np.sqrt(252)
        
        return float(sortino)
    
    def calculate_cagr(self) -> float:
        """Calculate Compound Annual Growth Rate."""
        if len(self.equity_curve) < 2:
            return 0.0
        
        start_val = self.start_equity
        end_val = self.current_equity
        n_days = len(self.equity_curve)
        years = n_days / 252
        
        if years == 0 or start_val == 0:
            return 0.0
        
        cagr = (end_val / start_val) ** (1 / years) - 1
        return float(cagr)
    
    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if len(self.equity_curve) < 2:
            return 0.0
        
        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        
        return float(np.min(drawdown))
    
    def calculate_calmar(self) -> float:
        """Calculate Calmar ratio (CAGR / MaxDD)."""
        cagr = self.calculate_cagr()
        max_dd = abs(self.calculate_max_drawdown())
        
        if max_dd == 0:
            return 0.0
        
        return float(cagr / max_dd)
    
    def calculate_win_rate(self) -> float:
        """Calculate win rate from trades."""
        if not self.trades:
            return 0.0
        
        wins = sum(1 for t in self.trades if t.pnl > 0)
        return float(wins / len(self.trades))
    
    def calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        if not self.trades:
            return 0.0
        
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return float(gross_profit / gross_loss)
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get all performance metrics."""
        wins = [t.pnl for t in self.trades if t.pnl > 0]
        losses = [t.pnl for t in self.trades if t.pnl < 0]
        
        # Calculate period P&Ls
        daily_pnl = self.daily_returns[-1] * self.current_equity if self.daily_returns else 0.0
        weekly_pnl = sum(self.daily_returns[-5:]) * self.current_equity if len(self.daily_returns) >= 5 else 0.0
        monthly_pnl = sum(self.daily_returns[-21:]) * self.current_equity if len(self.daily_returns) >= 21 else 0.0
        
        return PerformanceMetrics(
            sharpe_ratio=round(self.calculate_sharpe(), 3),
            cagr=round(self.calculate_cagr(), 4),
            max_drawdown=round(self.calculate_max_drawdown(), 4),
            win_rate=round(self.calculate_win_rate(), 4),
            profit_factor=round(self.calculate_profit_factor(), 3),
            sortino_ratio=round(self.calculate_sortino(), 3),
            calmar_ratio=round(self.calculate_calmar(), 3),
            avg_win=round(np.mean(wins), 2) if wins else 0.0,
            avg_loss=round(np.mean(losses), 2) if losses else 0.0,
            total_trades=len(self.trades),
            total_pnl=round(self.current_equity - self.start_equity, 2),
            daily_pnl=round(daily_pnl, 2),
            weekly_pnl=round(weekly_pnl, 2),
            monthly_pnl=round(monthly_pnl, 2),
            timestamp=datetime.now().isoformat()
        )


# =============================================================================
# REDIS CACHE MANAGER
# =============================================================================

class CacheManager:
    """Redis-backed cache for metrics and data."""
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
        self.redis_client: Optional[Any] = None
        self.local_cache: Dict[str, Any] = {}
        self.cache_ttl: int = 60  # Default TTL in seconds
        
        self._connect()
        
    def _connect(self):
        """Connect to Redis."""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, using local cache")
            return
        
        try:
            self.redis_client = redis.from_url(self.redis_url)
            self.redis_client.ping()
            logger.info("Connected to Redis")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}, using local cache")
            self.redis_client = None
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if self.redis_client:
            try:
                value = self.redis_client.get(key)
                if value:
                    return json.loads(value)
            except Exception:
                pass
        return self.local_cache.get(key)
    
    def set(self, key: str, value: Any, ttl: int = None):
        """Set value in cache."""
        ttl = ttl or self.cache_ttl
        
        if self.redis_client:
            try:
                self.redis_client.setex(key, ttl, json.dumps(value))
            except Exception:
                pass
        
        self.local_cache[key] = value
    
    def delete(self, key: str):
        """Delete key from cache."""
        if self.redis_client:
            try:
                self.redis_client.delete(key)
            except Exception:
                pass
        self.local_cache.pop(key, None)
    
    def is_connected(self) -> bool:
        """Check if Redis is connected."""
        if not self.redis_client:
            return False
        try:
            self.redis_client.ping()
            return True
        except Exception:
            return False


# =============================================================================
# WEBSOCKET MANAGER
# =============================================================================

class WebSocketManager:
    """Manage WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.pnl_subscribers: Set[Any] = set()
        self.trade_subscribers: Set[Any] = set()
        self.position_subscribers: Set[Any] = set()
        self.regime_subscribers: Set[Any] = set()
        self._lock = threading.Lock()
        
    def add_subscriber(self, ws: Any, channel: str):
        """Add WebSocket subscriber to channel."""
        with self._lock:
            if channel == 'pnl':
                self.pnl_subscribers.add(ws)
            elif channel == 'trades':
                self.trade_subscribers.add(ws)
            elif channel == 'positions':
                self.position_subscribers.add(ws)
            elif channel == 'regime':
                self.regime_subscribers.add(ws)
    
    def remove_subscriber(self, ws: Any, channel: str = None):
        """Remove WebSocket subscriber."""
        with self._lock:
            if channel is None:
                # Remove from all channels
                self.pnl_subscribers.discard(ws)
                self.trade_subscribers.discard(ws)
                self.position_subscribers.discard(ws)
                self.regime_subscribers.discard(ws)
            elif channel == 'pnl':
                self.pnl_subscribers.discard(ws)
            elif channel == 'trades':
                self.trade_subscribers.discard(ws)
            elif channel == 'positions':
                self.position_subscribers.discard(ws)
            elif channel == 'regime':
                self.regime_subscribers.discard(ws)
    
    async def broadcast_pnl(self, pnl_data: Dict[str, Any]):
        """Broadcast P&L update to all subscribers."""
        await self._broadcast(self.pnl_subscribers, 'pnl', pnl_data)
    
    async def broadcast_trade(self, trade: Trade):
        """Broadcast new trade to all subscribers."""
        await self._broadcast(self.trade_subscribers, 'trade', trade.to_dict())
    
    async def broadcast_positions(self, positions: List[Position]):
        """Broadcast position updates."""
        await self._broadcast(self.position_subscribers, 'positions', 
                              [p.to_dict() for p in positions])
    
    async def broadcast_regime(self, regime: RegimeInfo):
        """Broadcast regime change."""
        await self._broadcast(self.regime_subscribers, 'regime', regime.to_dict())
    
    async def _broadcast(self, subscribers: Set[Any], event_type: str, data: Any):
        """Broadcast message to subscribers."""
        if not AIOHTTP_AVAILABLE:
            return
        
        message = json.dumps({
            'type': event_type,
            'data': data,
            'timestamp': datetime.now().isoformat()
        })
        
        dead_connections = set()
        
        for ws in subscribers.copy():
            try:
                await ws.send_str(message)
            except Exception:
                dead_connections.add(ws)
        
        # Clean up dead connections
        with self._lock:
            for ws in dead_connections:
                subscribers.discard(ws)
    
    def get_connection_count(self) -> int:
        """Get total number of WebSocket connections."""
        return (len(self.pnl_subscribers) + len(self.trade_subscribers) + 
                len(self.position_subscribers) + len(self.regime_subscribers))


# =============================================================================
# DASHBOARD API SERVER
# =============================================================================

class DashboardAPIServer:
    """
    REST and WebSocket API server for dashboard integration.
    """
    
    def __init__(
        self,
        host: str = '0.0.0.0',
        port: int = 8080,
        redis_url: str = None
    ):
        self.host = host
        self.port = port
        
        # Components
        self.cache = CacheManager(redis_url)
        self.ws_manager = WebSocketManager()
        self.metrics_calc = MetricsCalculator()
        
        # State
        self.positions: Dict[str, Position] = {}
        self.recent_trades: List[Trade] = []
        self.current_regime: Optional[RegimeInfo] = None
        self.start_time = datetime.now()
        self.errors_count = 0
        self.warnings_count = 0
        
        # Web application
        self.app: Optional[Any] = None
        self._running = False
        
    def setup_routes(self):
        """Set up API routes."""
        if not AIOHTTP_AVAILABLE:
            logger.error("aiohttp not available, cannot set up routes")
            return
        
        self.app = web.Application()
        
        # REST endpoints
        self.app.router.add_get('/api/metrics', self.handle_metrics)
        self.app.router.add_get('/api/positions', self.handle_positions)
        self.app.router.add_get('/api/trades', self.handle_trades)
        self.app.router.add_get('/api/regime', self.handle_regime)
        self.app.router.add_get('/api/health', self.handle_health)
        self.app.router.add_get('/api/equity', self.handle_equity)
        
        # WebSocket endpoints
        self.app.router.add_get('/ws/pnl', self.handle_ws_pnl)
        self.app.router.add_get('/ws/trades', self.handle_ws_trades)
        self.app.router.add_get('/ws/positions', self.handle_ws_positions)
        self.app.router.add_get('/ws/regime', self.handle_ws_regime)
        
        logger.info("API routes configured")
    
    # -------------------------------------------------------------------------
    # REST Handlers
    # -------------------------------------------------------------------------
    
    async def handle_metrics(self, request) -> web.Response:
        """GET /api/metrics - Return performance metrics."""
        try:
            # Check cache first
            cached = self.cache.get('metrics')
            if cached:
                return web.json_response(cached)
            
            # Calculate fresh metrics
            metrics = self.metrics_calc.get_metrics()
            metrics_dict = metrics.to_dict()
            
            # Cache for 5 seconds
            self.cache.set('metrics', metrics_dict, ttl=5)
            
            return web.json_response(metrics_dict)
        except Exception as e:
            self.errors_count += 1
            logger.error(f"Error in /api/metrics: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def handle_positions(self, request) -> web.Response:
        """GET /api/positions - Return current positions."""
        try:
            positions_list = [p.to_dict() for p in self.positions.values()]
            return web.json_response({
                'positions': positions_list,
                'count': len(positions_list),
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            self.errors_count += 1
            logger.error(f"Error in /api/positions: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def handle_trades(self, request) -> web.Response:
        """GET /api/trades - Return recent trades."""
        try:
            # Get query params
            limit = int(request.query.get('limit', 100))
            
            trades_list = [t.to_dict() for t in self.recent_trades[-limit:]]
            return web.json_response({
                'trades': trades_list,
                'count': len(trades_list),
                'total_trades': len(self.recent_trades),
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            self.errors_count += 1
            logger.error(f"Error in /api/trades: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def handle_regime(self, request) -> web.Response:
        """GET /api/regime - Return current market regime."""
        try:
            if self.current_regime:
                return web.json_response(self.current_regime.to_dict())
            else:
                return web.json_response({
                    'regime': 'unknown',
                    'volatility': 'unknown',
                    'timestamp': datetime.now().isoformat()
                })
        except Exception as e:
            self.errors_count += 1
            logger.error(f"Error in /api/regime: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def handle_health(self, request) -> web.Response:
        """GET /api/health - Return system health status."""
        try:
            import psutil
            
            # Calculate uptime
            uptime = (datetime.now() - self.start_time).total_seconds()
            
            # Get system stats
            memory = psutil.Process().memory_info()
            cpu_pct = psutil.cpu_percent(interval=0.1)
            
            # Determine overall status
            if self.errors_count > 10:
                status = 'unhealthy'
            elif self.errors_count > 0 or self.warnings_count > 5:
                status = 'degraded'
            else:
                status = 'healthy'
            
            health = HealthStatus(
                status=status,
                uptime_seconds=uptime,
                last_trade_time=self.recent_trades[-1].timestamp if self.recent_trades else '',
                last_data_update=datetime.now().isoformat(),
                api_latency_ms=0.0,  # Would measure actual latency in production
                websocket_connections=self.ws_manager.get_connection_count(),
                memory_usage_mb=memory.rss / 1024 / 1024,
                cpu_usage_pct=cpu_pct,
                errors_last_hour=self.errors_count,
                warnings_last_hour=self.warnings_count,
                redis_connected=self.cache.is_connected(),
                broker_connected=True,  # Would check actual broker connection
                data_feed_connected=True  # Would check actual data feed
            )
            
            return web.json_response(health.to_dict())
        except Exception as e:
            self.errors_count += 1
            logger.error(f"Error in /api/health: {e}")
            return web.json_response({
                'status': 'unhealthy',
                'error': str(e)
            }, status=500)
    
    async def handle_equity(self, request) -> web.Response:
        """GET /api/equity - Return equity curve data."""
        try:
            return web.json_response({
                'equity_curve': self.metrics_calc.equity_curve[-252:],  # Last year
                'current_equity': self.metrics_calc.current_equity,
                'start_equity': self.metrics_calc.start_equity,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            self.errors_count += 1
            return web.json_response({'error': str(e)}, status=500)
    
    # -------------------------------------------------------------------------
    # WebSocket Handlers
    # -------------------------------------------------------------------------
    
    async def handle_ws_pnl(self, request) -> web.WebSocketResponse:
        """WebSocket /ws/pnl - Live P&L streaming."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        self.ws_manager.add_subscriber(ws, 'pnl')
        logger.info("New P&L WebSocket connection")
        
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    if msg.data == 'close':
                        break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    break
        finally:
            self.ws_manager.remove_subscriber(ws, 'pnl')
        
        return ws
    
    async def handle_ws_trades(self, request) -> web.WebSocketResponse:
        """WebSocket /ws/trades - Live trade notifications."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        self.ws_manager.add_subscriber(ws, 'trades')
        logger.info("New trades WebSocket connection")
        
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    if msg.data == 'close':
                        break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    break
        finally:
            self.ws_manager.remove_subscriber(ws, 'trades')
        
        return ws
    
    async def handle_ws_positions(self, request) -> web.WebSocketResponse:
        """WebSocket /ws/positions - Live position updates."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        self.ws_manager.add_subscriber(ws, 'positions')
        logger.info("New positions WebSocket connection")
        
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    if msg.data == 'close':
                        break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    break
        finally:
            self.ws_manager.remove_subscriber(ws, 'positions')
        
        return ws
    
    async def handle_ws_regime(self, request) -> web.WebSocketResponse:
        """WebSocket /ws/regime - Live regime updates."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        self.ws_manager.add_subscriber(ws, 'regime')
        logger.info("New regime WebSocket connection")
        
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    if msg.data == 'close':
                        break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    break
        finally:
            self.ws_manager.remove_subscriber(ws, 'regime')
        
        return ws
    
    # -------------------------------------------------------------------------
    # Data Update Methods (called by trading engine)
    # -------------------------------------------------------------------------
    
    def update_position(self, position: Position):
        """Update a position."""
        self.positions[position.symbol] = position
        self.cache.set(f'position:{position.symbol}', position.to_dict())
    
    def remove_position(self, symbol: str):
        """Remove a closed position."""
        self.positions.pop(symbol, None)
        self.cache.delete(f'position:{symbol}')
    
    def add_trade(self, trade: Trade):
        """Add new trade and broadcast."""
        self.recent_trades.append(trade)
        self.metrics_calc.add_trade(trade)
        self.cache.set(f'trade:{trade.trade_id}', trade.to_dict())
        
        # Keep only last 1000 trades in memory
        if len(self.recent_trades) > 1000:
            self.recent_trades = self.recent_trades[-1000:]
    
    def update_equity(self, equity: float):
        """Update equity for P&L calculation."""
        self.metrics_calc.update_equity(equity)
    
    def update_regime(self, regime: RegimeInfo):
        """Update current market regime."""
        self.current_regime = regime
        self.cache.set('regime', regime.to_dict())
    
    # -------------------------------------------------------------------------
    # Server Control
    # -------------------------------------------------------------------------
    
    async def start(self):
        """Start the API server."""
        if not AIOHTTP_AVAILABLE:
            logger.error("aiohttp not installed, cannot start server")
            return
        
        self.setup_routes()
        
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        
        self._running = True
        logger.info(f"🚀 V28 Dashboard API running on http://{self.host}:{self.port}")
    
    def stop(self):
        """Stop the API server."""
        self._running = False


# =============================================================================
# MAIN / DEMO
# =============================================================================

async def demo():
    """Demo the dashboard API."""
    server = DashboardAPIServer(port=8080)
    
    # Add some sample data
    server.metrics_calc.start_equity = 100000
    
    # Simulate equity curve
    equity = 100000
    for i in range(100):
        daily_return = np.random.randn() * 0.02 + 0.001  # Slight positive drift
        equity *= (1 + daily_return)
        server.metrics_calc.update_equity(equity)
    
    # Simulate trades
    for i in range(20):
        pnl = np.random.randn() * 500 + 100
        trade = Trade(
            trade_id=f"T{i:05d}",
            symbol=['SPY', 'QQQ', 'AAPL', 'NVDA'][i % 4],
            side='buy' if i % 2 == 0 else 'sell',
            quantity=100,
            price=450.0 + np.random.randn() * 10,
            timestamp=datetime.now().isoformat(),
            pnl=pnl,
            pnl_pct=pnl / 50000,
            regime='bull'
        )
        server.add_trade(trade)
    
    # Simulate positions
    server.update_position(Position(
        symbol='SPY',
        side='long',
        quantity=100,
        entry_price=445.50,
        current_price=448.75,
        unrealized_pnl=325.0,
        unrealized_pnl_pct=0.73,
        entry_time=datetime.now().isoformat(),
        holding_period_hours=4.5,
        regime_at_entry='bull'
    ))
    
    # Set regime
    server.update_regime(RegimeInfo(
        regime='bull',
        volatility='low',
        hmm_state=0,
        hmm_state_name='LowVolTrend',
        garch_forecast=0.15,
        vix_level=14.5,
        confidence=0.85,
        strategy_allocation={'momentum': 0.6, 'mean_reversion': 0.4},
        timestamp=datetime.now().isoformat()
    ))
    
    await server.start()
    
    print("\n📊 V28 Dashboard API Demo")
    print("=" * 50)
    print(f"REST Endpoints:")
    print(f"  GET http://localhost:8080/api/metrics")
    print(f"  GET http://localhost:8080/api/positions")
    print(f"  GET http://localhost:8080/api/trades")
    print(f"  GET http://localhost:8080/api/regime")
    print(f"  GET http://localhost:8080/api/health")
    print(f"\nWebSocket Endpoints:")
    print(f"  WS ws://localhost:8080/ws/pnl")
    print(f"  WS ws://localhost:8080/ws/trades")
    print("=" * 50)
    
    # Keep running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        server.stop()


if __name__ == '__main__':
    asyncio.run(demo())
