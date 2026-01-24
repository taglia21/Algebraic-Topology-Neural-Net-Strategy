#!/usr/bin/env python3
"""
V29 Unified Performance Dashboard
================================
Monitors equity trading (Alpaca) and prediction markets (Polymarket/Kalshi)
with real-time updates, interactive charts, and combined metrics.

Author: Trading Bot System
Version: 29.0
"""

import os
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import hmac
import requests
from functools import wraps

# Flask and extensions
from flask import Flask, render_template_string, jsonify, request, Response
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# Data processing
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/v29_dashboard.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('V29Dashboard')

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Dashboard configuration from environment variables."""
    
    # Alpaca API
    ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', '')
    ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', '')
    ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    
    # Polymarket API
    POLYMARKET_API_KEY = os.getenv('POLYMARKET_API_KEY', '')
    POLYMARKET_SECRET = os.getenv('POLYMARKET_SECRET', '')
    POLYMARKET_BASE_URL = os.getenv('POLYMARKET_BASE_URL', 'https://clob.polymarket.com')
    
    # Kalshi API
    KALSHI_API_KEY = os.getenv('KALSHI_API_KEY', '')
    KALSHI_SECRET = os.getenv('KALSHI_SECRET', '')
    KALSHI_BASE_URL = os.getenv('KALSHI_BASE_URL', 'https://trading-api.kalshi.com/trade-api/v2')
    
    # Dashboard settings
    FLASK_SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'v29-dashboard-secret-key')
    DASHBOARD_PORT = int(os.getenv('DASHBOARD_PORT', '5029'))
    UPDATE_INTERVAL = int(os.getenv('UPDATE_INTERVAL', '5'))  # seconds
    
    # Alert thresholds
    DRAWDOWN_ALERT_PCT = float(os.getenv('DRAWDOWN_ALERT_PCT', '5.0'))
    POSITION_SIZE_ALERT_PCT = float(os.getenv('POSITION_SIZE_ALERT_PCT', '20.0'))
    PNL_ALERT_THRESHOLD = float(os.getenv('PNL_ALERT_THRESHOLD', '1000.0'))


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    qty: float
    side: str
    entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    sector: str = "Unknown"
    platform: str = "alpaca"


@dataclass
class Trade:
    """Represents a completed trade."""
    id: str
    symbol: str
    side: str
    qty: float
    price: float
    timestamp: datetime
    pnl: float = 0.0
    platform: str = "alpaca"


@dataclass
class AccountMetrics:
    """Account performance metrics."""
    equity: float = 0.0
    cash: float = 0.0
    buying_power: float = 0.0
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    monthly_pnl: float = 0.0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    positions_count: int = 0
    trades_today: int = 0


@dataclass
class PredictionMarketPosition:
    """Prediction market position."""
    market_id: str
    title: str
    outcome: str
    shares: float
    avg_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    platform: str


@dataclass
class ArbitrageOpportunity:
    """Cross-platform arbitrage opportunity."""
    market_title: str
    polymarket_price: Optional[float]
    kalshi_price: Optional[float]
    spread: float
    expected_profit: float
    timestamp: datetime
    executed: bool = False


@dataclass
class Alert:
    """System alert."""
    id: str
    level: str  # 'info', 'warning', 'critical'
    message: str
    timestamp: datetime
    acknowledged: bool = False


# =============================================================================
# API CLIENTS
# =============================================================================

class AlpacaClient:
    """Alpaca trading API client."""
    
    def __init__(self):
        self.api_key = Config.ALPACA_API_KEY
        self.secret_key = Config.ALPACA_SECRET_KEY
        self.base_url = Config.ALPACA_BASE_URL
        self.session = requests.Session()
        self.session.headers.update({
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.secret_key
        })
        
    def _request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Make API request with error handling."""
        try:
            url = f"{self.base_url}{endpoint}"
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Alpaca API error: {e}")
            return {}
    
    def get_account(self) -> Dict:
        """Get account information."""
        return self._request('GET', '/v2/account')
    
    def get_positions(self) -> List[Dict]:
        """Get all positions."""
        result = self._request('GET', '/v2/positions')
        return result if isinstance(result, list) else []
    
    def get_orders(self, status: str = 'all', limit: int = 100) -> List[Dict]:
        """Get orders."""
        result = self._request('GET', '/v2/orders', params={
            'status': status,
            'limit': limit
        })
        return result if isinstance(result, list) else []
    
    def get_portfolio_history(self, period: str = '1M') -> Dict:
        """Get portfolio history."""
        return self._request('GET', '/v2/account/portfolio/history', params={
            'period': period,
            'timeframe': '1D'
        })
    
    def get_asset(self, symbol: str) -> Dict:
        """Get asset information."""
        return self._request('GET', f'/v2/assets/{symbol}')


class PolymarketClient:
    """Polymarket CLOB API client."""
    
    def __init__(self):
        self.api_key = Config.POLYMARKET_API_KEY
        self.secret = Config.POLYMARKET_SECRET
        self.base_url = Config.POLYMARKET_BASE_URL
        self.session = requests.Session()
        
    def _sign_request(self, timestamp: str, method: str, path: str, body: str = '') -> str:
        """Sign API request."""
        message = f"{timestamp}{method}{path}{body}"
        signature = hmac.new(
            self.secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Make API request."""
        try:
            timestamp = str(int(time.time() * 1000))
            body = json.dumps(kwargs.get('json', {})) if kwargs.get('json') else ''
            signature = self._sign_request(timestamp, method, endpoint, body)
            
            headers = {
                'POLY_API_KEY': self.api_key,
                'POLY_TIMESTAMP': timestamp,
                'POLY_SIGNATURE': signature,
                'Content-Type': 'application/json'
            }
            
            url = f"{self.base_url}{endpoint}"
            response = self.session.request(method, url, headers=headers, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Polymarket API error: {e}")
            return {}
    
    def get_balance(self) -> Dict:
        """Get account balance."""
        return self._request('GET', '/balance')
    
    def get_positions(self) -> List[Dict]:
        """Get open positions."""
        result = self._request('GET', '/positions')
        return result.get('positions', []) if isinstance(result, dict) else []
    
    def get_trades(self, limit: int = 100) -> List[Dict]:
        """Get trade history."""
        result = self._request('GET', '/trades', params={'limit': limit})
        return result.get('trades', []) if isinstance(result, dict) else []
    
    def get_markets(self) -> List[Dict]:
        """Get available markets."""
        try:
            response = self.session.get(f"{self.base_url}/markets")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching markets: {e}")
            return []


class KalshiClient:
    """Kalshi trading API client."""
    
    def __init__(self):
        self.api_key = Config.KALSHI_API_KEY
        self.secret = Config.KALSHI_SECRET
        self.base_url = Config.KALSHI_BASE_URL
        self.session = requests.Session()
        self.token = None
        self.token_expiry = None
        
    def _authenticate(self) -> bool:
        """Authenticate with Kalshi API."""
        try:
            response = self.session.post(
                f"{self.base_url}/login",
                json={
                    'email': self.api_key,
                    'password': self.secret
                }
            )
            response.raise_for_status()
            data = response.json()
            self.token = data.get('token')
            self.session.headers.update({'Authorization': f'Bearer {self.token}'})
            return True
        except Exception as e:
            logger.error(f"Kalshi authentication error: {e}")
            return False
    
    def _request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Make authenticated API request."""
        try:
            if not self.token:
                self._authenticate()
            
            url = f"{self.base_url}{endpoint}"
            response = self.session.request(method, url, **kwargs)
            
            if response.status_code == 401:
                self._authenticate()
                response = self.session.request(method, url, **kwargs)
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Kalshi API error: {e}")
            return {}
    
    def get_balance(self) -> Dict:
        """Get account balance."""
        return self._request('GET', '/portfolio/balance')
    
    def get_positions(self) -> List[Dict]:
        """Get open positions."""
        result = self._request('GET', '/portfolio/positions')
        return result.get('market_positions', []) if isinstance(result, dict) else []
    
    def get_fills(self, limit: int = 100) -> List[Dict]:
        """Get trade fills."""
        result = self._request('GET', '/portfolio/fills', params={'limit': limit})
        return result.get('fills', []) if isinstance(result, dict) else []
    
    def get_markets(self) -> List[Dict]:
        """Get available markets."""
        result = self._request('GET', '/markets', params={'status': 'open'})
        return result.get('markets', []) if isinstance(result, dict) else []


# =============================================================================
# METRICS ENGINE
# =============================================================================

class MetricsEngine:
    """Calculate and track performance metrics."""
    
    def __init__(self):
        self.equity_history: List[Tuple[datetime, float]] = []
        self.trade_history: List[Trade] = []
        self.pnl_history: List[Tuple[datetime, float]] = []
        
        # Sector mapping for equities
        self.sector_map = {
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
            'AMZN': 'Consumer', 'TSLA': 'Automotive', 'META': 'Technology',
            'NVDA': 'Technology', 'JPM': 'Financial', 'BAC': 'Financial',
            'XOM': 'Energy', 'CVX': 'Energy', 'JNJ': 'Healthcare',
            'PFE': 'Healthcare', 'UNH': 'Healthcare', 'WMT': 'Consumer',
            'KO': 'Consumer', 'PEP': 'Consumer', 'DIS': 'Media',
            'NFLX': 'Media', 'VZ': 'Telecom', 'T': 'Telecom'
        }
    
    def calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.05) -> float:
        """Calculate Sharpe ratio from returns."""
        if not returns or len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        excess_returns = returns_array - (risk_free_rate / 252)  # Daily risk-free rate
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        return float(sharpe)
    
    def calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown percentage."""
        if not equity_curve or len(equity_curve) < 2:
            return 0.0
        
        peak = equity_curve[0]
        max_dd = 0.0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def calculate_win_rate(self, trades: List[Trade]) -> float:
        """Calculate win rate from trades."""
        if not trades:
            return 0.0
        
        winning_trades = sum(1 for t in trades if t.pnl > 0)
        return winning_trades / len(trades) * 100
    
    def calculate_profit_factor(self, trades: List[Trade]) -> float:
        """Calculate profit factor from trades."""
        if not trades:
            return 0.0
        
        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def get_sector(self, symbol: str) -> str:
        """Get sector for a symbol."""
        return self.sector_map.get(symbol.upper(), 'Other')
    
    def calculate_sector_breakdown(self, positions: List[Position]) -> Dict[str, float]:
        """Calculate portfolio breakdown by sector."""
        sector_values: Dict[str, float] = {}
        
        for pos in positions:
            sector = pos.sector or self.get_sector(pos.symbol)
            sector_values[sector] = sector_values.get(sector, 0) + pos.market_value
        
        total = sum(sector_values.values())
        if total > 0:
            return {k: v / total * 100 for k, v in sector_values.items()}
        return sector_values


# =============================================================================
# ARBITRAGE DETECTOR
# =============================================================================

class ArbitrageDetector:
    """Detect arbitrage opportunities across prediction markets."""
    
    def __init__(self, polymarket: PolymarketClient, kalshi: KalshiClient):
        self.polymarket = polymarket
        self.kalshi = kalshi
        self.opportunities: List[ArbitrageOpportunity] = []
        self.min_spread = 0.02  # Minimum 2% spread for opportunity
        
    def find_opportunities(self) -> List[ArbitrageOpportunity]:
        """Find arbitrage opportunities between platforms."""
        opportunities = []
        
        try:
            poly_markets = self.polymarket.get_markets()
            kalshi_markets = self.kalshi.get_markets()
            
            # Match markets by title/description (simplified matching)
            poly_map = {m.get('question', '').lower(): m for m in poly_markets}
            
            for km in kalshi_markets:
                kalshi_title = km.get('title', '').lower()
                
                # Try to find matching Polymarket market
                for poly_title, pm in poly_map.items():
                    if self._titles_match(kalshi_title, poly_title):
                        spread = self._calculate_spread(pm, km)
                        if spread and spread.spread >= self.min_spread:
                            opportunities.append(spread)
                            break
            
            self.opportunities = opportunities
            
        except Exception as e:
            logger.error(f"Error detecting arbitrage: {e}")
        
        return opportunities
    
    def _titles_match(self, title1: str, title2: str) -> bool:
        """Check if two market titles likely refer to same event."""
        # Simplified matching - check for significant word overlap
        words1 = set(title1.split())
        words2 = set(title2.split())
        overlap = len(words1 & words2)
        return overlap >= 3
    
    def _calculate_spread(self, poly_market: Dict, kalshi_market: Dict) -> Optional[ArbitrageOpportunity]:
        """Calculate spread between two matching markets."""
        try:
            poly_yes = float(poly_market.get('outcomePrices', [0.5])[0])
            kalshi_yes = float(kalshi_market.get('yes_price', 50)) / 100
            
            spread = abs(poly_yes - kalshi_yes)
            
            if spread >= self.min_spread:
                return ArbitrageOpportunity(
                    market_title=kalshi_market.get('title', 'Unknown'),
                    polymarket_price=poly_yes,
                    kalshi_price=kalshi_yes,
                    spread=spread,
                    expected_profit=spread * 100,  # Per $100 position
                    timestamp=datetime.now()
                )
        except Exception as e:
            logger.error(f"Error calculating spread: {e}")
        
        return None


# =============================================================================
# ALERT MANAGER
# =============================================================================

class AlertManager:
    """Manage system alerts."""
    
    def __init__(self):
        self.alerts: List[Alert] = []
        self.alert_callbacks: List[callable] = []
        
    def add_alert(self, level: str, message: str) -> Alert:
        """Add new alert."""
        alert = Alert(
            id=hashlib.md5(f"{datetime.now()}{message}".encode()).hexdigest()[:8],
            level=level,
            message=message,
            timestamp=datetime.now()
        )
        self.alerts.append(alert)
        
        # Keep only last 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
        
        # Trigger callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
        
        logger.log(
            logging.CRITICAL if level == 'critical' else 
            logging.WARNING if level == 'warning' else logging.INFO,
            f"Alert [{level}]: {message}"
        )
        
        return alert
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                return True
        return False
    
    def get_unacknowledged(self) -> List[Alert]:
        """Get unacknowledged alerts."""
        return [a for a in self.alerts if not a.acknowledged]
    
    def check_thresholds(self, metrics: AccountMetrics, 
                         equity_positions: List[Position],
                         pred_positions: List[PredictionMarketPosition]):
        """Check metrics against thresholds and generate alerts."""
        
        # Drawdown alert
        if metrics.max_drawdown >= Config.DRAWDOWN_ALERT_PCT:
            self.add_alert('warning', 
                f"Max drawdown {metrics.max_drawdown:.1f}% exceeds threshold")
        
        # Large daily P&L alert
        if abs(metrics.daily_pnl) >= Config.PNL_ALERT_THRESHOLD:
            level = 'info' if metrics.daily_pnl > 0 else 'warning'
            self.add_alert(level, 
                f"Large daily P&L: ${metrics.daily_pnl:,.2f}")
        
        # Position concentration alert
        total_value = sum(p.market_value for p in equity_positions)
        if total_value > 0:
            for pos in equity_positions:
                pct = pos.market_value / total_value * 100
                if pct >= Config.POSITION_SIZE_ALERT_PCT:
                    self.add_alert('warning',
                        f"Large position: {pos.symbol} is {pct:.1f}% of portfolio")


# =============================================================================
# DASHBOARD DATA AGGREGATOR
# =============================================================================

class DashboardDataAggregator:
    """Aggregate data from all sources for dashboard."""
    
    def __init__(self):
        self.alpaca = AlpacaClient()
        self.polymarket = PolymarketClient()
        self.kalshi = KalshiClient()
        self.metrics_engine = MetricsEngine()
        self.arbitrage_detector = ArbitrageDetector(self.polymarket, self.kalshi)
        self.alert_manager = AlertManager()
        
        # Cached data
        self._equity_metrics: Optional[AccountMetrics] = None
        self._equity_positions: List[Position] = []
        self._pred_positions: List[PredictionMarketPosition] = []
        self._polymarket_balance: float = 0.0
        self._kalshi_balance: float = 0.0
        self._last_update: Optional[datetime] = None
        
    def refresh_all(self) -> Dict[str, Any]:
        """Refresh all data from APIs."""
        try:
            # Equity data
            equity_data = self._fetch_equity_data()
            
            # Prediction market data
            pred_data = self._fetch_prediction_market_data()
            
            # Combined metrics
            combined = self._calculate_combined_metrics(equity_data, pred_data)
            
            # Check for alerts
            if self._equity_metrics:
                self.alert_manager.check_thresholds(
                    self._equity_metrics,
                    self._equity_positions,
                    self._pred_positions
                )
            
            # Arbitrage opportunities
            arb_opportunities = self.arbitrage_detector.find_opportunities()
            
            self._last_update = datetime.now()
            
            return {
                'equity': equity_data,
                'prediction_markets': pred_data,
                'combined': combined,
                'arbitrage': [asdict(a) for a in arb_opportunities],
                'alerts': [asdict(a) for a in self.alert_manager.get_unacknowledged()],
                'last_update': self._last_update.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error refreshing data: {e}")
            return {'error': str(e)}
    
    def _fetch_equity_data(self) -> Dict[str, Any]:
        """Fetch equity trading data from Alpaca."""
        account = self.alpaca.get_account()
        positions = self.alpaca.get_positions()
        orders = self.alpaca.get_orders(status='closed', limit=100)
        history = self.alpaca.get_portfolio_history(period='1M')
        
        # Parse positions
        self._equity_positions = []
        for p in positions:
            pos = Position(
                symbol=p.get('symbol', ''),
                qty=float(p.get('qty', 0)),
                side='long' if float(p.get('qty', 0)) > 0 else 'short',
                entry_price=float(p.get('avg_entry_price', 0)),
                current_price=float(p.get('current_price', 0)),
                market_value=float(p.get('market_value', 0)),
                unrealized_pnl=float(p.get('unrealized_pl', 0)),
                unrealized_pnl_pct=float(p.get('unrealized_plpc', 0)) * 100,
                sector=self.metrics_engine.get_sector(p.get('symbol', '')),
                platform='alpaca'
            )
            self._equity_positions.append(pos)
        
        # Calculate metrics
        equity = float(account.get('equity', 0))
        last_equity = float(account.get('last_equity', equity))
        
        # Calculate P&L periods
        equity_values = history.get('equity', [])
        timestamps = history.get('timestamp', [])
        
        daily_pnl = equity - last_equity
        weekly_pnl = self._calculate_period_pnl(equity_values, timestamps, 7)
        monthly_pnl = self._calculate_period_pnl(equity_values, timestamps, 30)
        
        # Calculate trade metrics
        trades = []
        for o in orders:
            if o.get('filled_at'):
                trade = Trade(
                    id=o.get('id', ''),
                    symbol=o.get('symbol', ''),
                    side=o.get('side', ''),
                    qty=float(o.get('filled_qty', 0)),
                    price=float(o.get('filled_avg_price', 0)),
                    timestamp=datetime.fromisoformat(o['filled_at'].replace('Z', '+00:00')),
                    platform='alpaca'
                )
                trades.append(trade)
        
        # Calculate returns for Sharpe
        returns = []
        if len(equity_values) > 1:
            for i in range(1, len(equity_values)):
                if equity_values[i-1] > 0:
                    ret = (equity_values[i] - equity_values[i-1]) / equity_values[i-1]
                    returns.append(ret)
        
        self._equity_metrics = AccountMetrics(
            equity=equity,
            cash=float(account.get('cash', 0)),
            buying_power=float(account.get('buying_power', 0)),
            daily_pnl=daily_pnl,
            weekly_pnl=weekly_pnl,
            monthly_pnl=monthly_pnl,
            total_pnl=equity - float(account.get('initial_margin', equity)),
            win_rate=self.metrics_engine.calculate_win_rate(trades),
            profit_factor=self.metrics_engine.calculate_profit_factor(trades),
            sharpe_ratio=self.metrics_engine.calculate_sharpe_ratio(returns),
            max_drawdown=self.metrics_engine.calculate_max_drawdown(equity_values),
            positions_count=len(self._equity_positions),
            trades_today=sum(1 for t in trades if t.timestamp.date() == datetime.now().date())
        )
        
        # Sector breakdown
        sector_breakdown = self.metrics_engine.calculate_sector_breakdown(self._equity_positions)
        
        return {
            'metrics': asdict(self._equity_metrics),
            'positions': [asdict(p) for p in self._equity_positions],
            'sector_breakdown': sector_breakdown,
            'equity_history': {
                'timestamps': timestamps,
                'values': equity_values
            },
            'recent_trades': [asdict(t) for t in trades[:20]]
        }
    
    def _fetch_prediction_market_data(self) -> Dict[str, Any]:
        """Fetch prediction market data from Polymarket and Kalshi."""
        
        # Polymarket data
        poly_balance = self.polymarket.get_balance()
        poly_positions = self.polymarket.get_positions()
        poly_trades = self.polymarket.get_trades()
        
        self._polymarket_balance = float(poly_balance.get('balance', 0))
        
        poly_pos_list = []
        for p in poly_positions:
            pos = PredictionMarketPosition(
                market_id=p.get('market_id', ''),
                title=p.get('market_title', 'Unknown'),
                outcome=p.get('outcome', ''),
                shares=float(p.get('size', 0)),
                avg_price=float(p.get('avg_price', 0)),
                current_price=float(p.get('current_price', 0)),
                market_value=float(p.get('size', 0)) * float(p.get('current_price', 0)),
                unrealized_pnl=(float(p.get('current_price', 0)) - float(p.get('avg_price', 0))) * float(p.get('size', 0)),
                platform='polymarket'
            )
            poly_pos_list.append(pos)
        
        # Kalshi data
        kalshi_balance = self.kalshi.get_balance()
        kalshi_positions = self.kalshi.get_positions()
        kalshi_fills = self.kalshi.get_fills()
        
        self._kalshi_balance = float(kalshi_balance.get('balance', 0)) / 100  # Cents to dollars
        
        kalshi_pos_list = []
        for p in kalshi_positions:
            pos = PredictionMarketPosition(
                market_id=p.get('ticker', ''),
                title=p.get('market_title', 'Unknown'),
                outcome='Yes' if p.get('position', 0) > 0 else 'No',
                shares=abs(float(p.get('position', 0))),
                avg_price=float(p.get('average_price', 0)) / 100,
                current_price=float(p.get('market_price', 0)) / 100,
                market_value=abs(float(p.get('position', 0))) * float(p.get('market_price', 0)) / 100,
                unrealized_pnl=float(p.get('realized_pnl', 0)) / 100,
                platform='kalshi'
            )
            kalshi_pos_list.append(pos)
        
        self._pred_positions = poly_pos_list + kalshi_pos_list
        
        # Calculate win rates
        poly_wins = sum(1 for t in poly_trades if float(t.get('pnl', 0)) > 0)
        poly_win_rate = poly_wins / len(poly_trades) * 100 if poly_trades else 0
        
        kalshi_wins = sum(1 for f in kalshi_fills if float(f.get('realized_pnl', 0)) > 0)
        kalshi_win_rate = kalshi_wins / len(kalshi_fills) * 100 if kalshi_fills else 0
        
        return {
            'polymarket': {
                'balance': self._polymarket_balance,
                'positions': [asdict(p) for p in poly_pos_list],
                'positions_count': len(poly_pos_list),
                'total_pnl': sum(p.unrealized_pnl for p in poly_pos_list),
                'win_rate': poly_win_rate,
                'recent_trades': poly_trades[:20]
            },
            'kalshi': {
                'balance': self._kalshi_balance,
                'positions': [asdict(p) for p in kalshi_pos_list],
                'positions_count': len(kalshi_pos_list),
                'total_pnl': sum(p.unrealized_pnl for p in kalshi_pos_list),
                'win_rate': kalshi_win_rate,
                'recent_trades': kalshi_fills[:20]
            },
            'combined_balance': self._polymarket_balance + self._kalshi_balance,
            'combined_positions': len(self._pred_positions)
        }
    
    def _calculate_combined_metrics(self, equity_data: Dict, pred_data: Dict) -> Dict[str, Any]:
        """Calculate combined metrics across all systems."""
        equity_value = equity_data.get('metrics', {}).get('equity', 0)
        pred_value = pred_data.get('combined_balance', 0)
        
        total_portfolio = equity_value + pred_value
        
        equity_pnl = equity_data.get('metrics', {}).get('daily_pnl', 0)
        pred_pnl = (pred_data.get('polymarket', {}).get('total_pnl', 0) + 
                   pred_data.get('kalshi', {}).get('total_pnl', 0))
        
        combined_pnl = equity_pnl + pred_pnl
        
        # Risk exposure
        equity_exposure = sum(abs(p.market_value) for p in self._equity_positions)
        pred_exposure = sum(abs(p.market_value) for p in self._pred_positions)
        
        return {
            'total_portfolio_value': total_portfolio,
            'equity_allocation': equity_value / total_portfolio * 100 if total_portfolio > 0 else 0,
            'prediction_allocation': pred_value / total_portfolio * 100 if total_portfolio > 0 else 0,
            'combined_daily_pnl': combined_pnl,
            'combined_pnl_pct': combined_pnl / total_portfolio * 100 if total_portfolio > 0 else 0,
            'total_exposure': equity_exposure + pred_exposure,
            'exposure_ratio': (equity_exposure + pred_exposure) / total_portfolio if total_portfolio > 0 else 0,
            'total_positions': len(self._equity_positions) + len(self._pred_positions),
            'systems_status': {
                'equity': 'active' if equity_value > 0 else 'inactive',
                'polymarket': 'active' if self._polymarket_balance > 0 else 'inactive',
                'kalshi': 'active' if self._kalshi_balance > 0 else 'inactive'
            }
        }
    
    def _calculate_period_pnl(self, values: List[float], timestamps: List[int], 
                              days: int) -> float:
        """Calculate P&L for a specific period."""
        if not values or not timestamps:
            return 0.0
        
        cutoff = datetime.now() - timedelta(days=days)
        cutoff_ts = int(cutoff.timestamp())
        
        start_value = values[0]
        for i, ts in enumerate(timestamps):
            if ts >= cutoff_ts:
                start_value = values[i]
                break
        
        return values[-1] - start_value if values else 0.0


# =============================================================================
# FLASK APPLICATION
# =============================================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = Config.FLASK_SECRET_KEY
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialize aggregator
data_aggregator = DashboardDataAggregator()

# Background update thread
update_thread = None
update_running = False


def background_updater():
    """Background thread for real-time updates."""
    global update_running
    while update_running:
        try:
            data = data_aggregator.refresh_all()
            socketio.emit('data_update', data, namespace='/dashboard')
        except Exception as e:
            logger.error(f"Background update error: {e}")
        time.sleep(Config.UPDATE_INTERVAL)


# =============================================================================
# HTML TEMPLATE
# =============================================================================

DASHBOARD_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>V29 Unified Trading Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.6.0/socket.io.min.js"></script>
    <style>
        :root {
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --text-primary: #f0f6fc;
            --text-secondary: #8b949e;
            --accent-green: #3fb950;
            --accent-red: #f85149;
            --accent-blue: #58a6ff;
            --accent-yellow: #d29922;
            --accent-purple: #a371f7;
            --border-color: #30363d;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
        }
        
        .header {
            background: var(--bg-secondary);
            padding: 1rem 2rem;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1rem;
        }
        
        .header h1 {
            font-size: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .status-indicator {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: var(--accent-green);
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .header-stats {
            display: flex;
            gap: 2rem;
            flex-wrap: wrap;
        }
        
        .header-stat {
            text-align: right;
        }
        
        .header-stat-label {
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-transform: uppercase;
        }
        
        .header-stat-value {
            font-size: 1.25rem;
            font-weight: 600;
        }
        
        .header-stat-value.positive { color: var(--accent-green); }
        .header-stat-value.negative { color: var(--accent-red); }
        
        .main-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 1rem;
            padding: 1rem;
        }
        
        @media (max-width: 768px) {
            .main-container {
                grid-template-columns: 1fr;
            }
        }
        
        .card {
            background: var(--bg-secondary);
            border-radius: 8px;
            border: 1px solid var(--border-color);
            overflow: hidden;
        }
        
        .card-header {
            padding: 1rem;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .card-header h2 {
            font-size: 1rem;
            font-weight: 600;
        }
        
        .card-badge {
            font-size: 0.75rem;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            background: var(--bg-tertiary);
        }
        
        .card-body {
            padding: 1rem;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 1rem;
        }
        
        .metric-box {
            background: var(--bg-tertiary);
            padding: 1rem;
            border-radius: 6px;
            text-align: center;
        }
        
        .metric-value {
            font-size: 1.25rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        
        .metric-label {
            font-size: 0.75rem;
            color: var(--text-secondary);
        }
        
        .chart-container {
            height: 300px;
            margin-top: 1rem;
        }
        
        .positions-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.875rem;
        }
        
        .positions-table th,
        .positions-table td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }
        
        .positions-table th {
            color: var(--text-secondary);
            font-weight: 500;
            text-transform: uppercase;
            font-size: 0.7rem;
        }
        
        .positions-table tr:hover {
            background: var(--bg-tertiary);
        }
        
        .pnl-positive { color: var(--accent-green); }
        .pnl-negative { color: var(--accent-red); }
        
        .alert-container {
            position: fixed;
            top: 1rem;
            right: 1rem;
            z-index: 1000;
            max-width: 400px;
        }
        
        .alert {
            padding: 1rem;
            margin-bottom: 0.5rem;
            border-radius: 6px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            animation: slideIn 0.3s ease;
        }
        
        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        
        .alert-info { background: var(--accent-blue); }
        .alert-warning { background: var(--accent-yellow); color: #000; }
        .alert-critical { background: var(--accent-red); }
        
        .alert-dismiss {
            background: none;
            border: none;
            color: inherit;
            cursor: pointer;
            font-size: 1.25rem;
            padding: 0 0.5rem;
        }
        
        .arbitrage-card {
            background: var(--bg-tertiary);
            padding: 1rem;
            border-radius: 6px;
            margin-bottom: 0.5rem;
            border-left: 3px solid var(--accent-purple);
        }
        
        .arbitrage-title {
            font-weight: 500;
            margin-bottom: 0.5rem;
        }
        
        .arbitrage-details {
            display: flex;
            justify-content: space-between;
            font-size: 0.875rem;
            color: var(--text-secondary);
        }
        
        .system-status {
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
        }
        
        .system-badge {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            background: var(--bg-tertiary);
            border-radius: 4px;
            font-size: 0.875rem;
        }
        
        .system-badge.active::before {
            content: '';
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--accent-green);
        }
        
        .system-badge.inactive::before {
            content: '';
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--text-secondary);
        }
        
        .tab-container {
            display: flex;
            border-bottom: 1px solid var(--border-color);
        }
        
        .tab {
            padding: 0.75rem 1.5rem;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            transition: all 0.2s;
        }
        
        .tab:hover {
            background: var(--bg-tertiary);
        }
        
        .tab.active {
            border-bottom-color: var(--accent-blue);
            color: var(--accent-blue);
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .loading {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 200px;
            color: var(--text-secondary);
        }
        
        .spinner {
            width: 40px;
            height: 40px;
            border: 3px solid var(--border-color);
            border-top-color: var(--accent-blue);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 1rem;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>
            <span class="status-indicator" id="connectionStatus"></span>
            V29 Unified Dashboard
        </h1>
        <div class="header-stats">
            <div class="header-stat">
                <div class="header-stat-label">Total Portfolio</div>
                <div class="header-stat-value" id="totalPortfolio">$0.00</div>
            </div>
            <div class="header-stat">
                <div class="header-stat-label">Daily P&L</div>
                <div class="header-stat-value" id="dailyPnl">$0.00</div>
            </div>
            <div class="header-stat">
                <div class="header-stat-label">Last Update</div>
                <div class="header-stat-value" id="lastUpdate" style="font-size: 0.875rem;">--:--:--</div>
            </div>
        </div>
    </div>
    
    <div class="alert-container" id="alertContainer"></div>
    
    <div class="main-container">
        <!-- Combined Overview -->
        <div class="card" style="grid-column: span 2;">
            <div class="card-header">
                <h2>📊 Combined Overview</h2>
                <div class="system-status" id="systemStatus">
                    <span class="system-badge active">Equity</span>
                    <span class="system-badge inactive">Polymarket</span>
                    <span class="system-badge inactive">Kalshi</span>
                </div>
            </div>
            <div class="card-body">
                <div class="metrics-grid" id="combinedMetrics">
                    <div class="metric-box">
                        <div class="metric-value" id="equityAllocation">0%</div>
                        <div class="metric-label">Equity Allocation</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value" id="predAllocation">0%</div>
                        <div class="metric-label">Prediction Markets</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value" id="totalExposure">$0</div>
                        <div class="metric-label">Total Exposure</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value" id="exposureRatio">0x</div>
                        <div class="metric-label">Exposure Ratio</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value" id="totalPositions">0</div>
                        <div class="metric-label">Total Positions</div>
                    </div>
                </div>
                <div class="chart-container" id="portfolioChart"></div>
            </div>
        </div>
        
        <!-- Equity Engine -->
        <div class="card">
            <div class="card-header">
                <h2>📈 Equity Engine (Alpaca)</h2>
                <span class="card-badge" id="equityBadge">Paper Trading</span>
            </div>
            <div class="card-body">
                <div class="metrics-grid">
                    <div class="metric-box">
                        <div class="metric-value" id="equityValue">$0</div>
                        <div class="metric-label">Equity</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value" id="equityCash">$0</div>
                        <div class="metric-label">Cash</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value" id="equityWinRate">0%</div>
                        <div class="metric-label">Win Rate</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value" id="equitySharpe">0.00</div>
                        <div class="metric-label">Sharpe Ratio</div>
                    </div>
                </div>
                <div class="tab-container">
                    <div class="tab active" onclick="switchTab('equity', 'pnl')">P&L</div>
                    <div class="tab" onclick="switchTab('equity', 'positions')">Positions</div>
                    <div class="tab" onclick="switchTab('equity', 'sectors')">Sectors</div>
                </div>
                <div id="equity-pnl" class="tab-content active">
                    <div class="metrics-grid" style="margin-top: 1rem;">
                        <div class="metric-box">
                            <div class="metric-value pnl-positive" id="equityDailyPnl">$0</div>
                            <div class="metric-label">Daily</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value" id="equityWeeklyPnl">$0</div>
                            <div class="metric-label">Weekly</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value" id="equityMonthlyPnl">$0</div>
                            <div class="metric-label">Monthly</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value" id="equityMaxDD">0%</div>
                            <div class="metric-label">Max DD</div>
                        </div>
                    </div>
                    <div class="chart-container" id="equityChart"></div>
                </div>
                <div id="equity-positions" class="tab-content">
                    <table class="positions-table">
                        <thead>
                            <tr>
                                <th>Symbol</th>
                                <th>Qty</th>
                                <th>Value</th>
                                <th>P&L</th>
                            </tr>
                        </thead>
                        <tbody id="equityPositionsTable"></tbody>
                    </table>
                </div>
                <div id="equity-sectors" class="tab-content">
                    <div class="chart-container" id="sectorChart"></div>
                </div>
            </div>
        </div>
        
        <!-- Prediction Markets -->
        <div class="card">
            <div class="card-header">
                <h2>🎲 Prediction Markets</h2>
                <span class="card-badge">Live</span>
            </div>
            <div class="card-body">
                <div class="metrics-grid">
                    <div class="metric-box">
                        <div class="metric-value" id="predBalance">$0</div>
                        <div class="metric-label">Total Balance</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value" id="predPositions">0</div>
                        <div class="metric-label">Positions</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value" id="predWinRate">0%</div>
                        <div class="metric-label">Win Rate</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value" id="predPnl">$0</div>
                        <div class="metric-label">Total P&L</div>
                    </div>
                </div>
                <div class="tab-container">
                    <div class="tab active" onclick="switchTab('pred', 'polymarket')">Polymarket</div>
                    <div class="tab" onclick="switchTab('pred', 'kalshi')">Kalshi</div>
                </div>
                <div id="pred-polymarket" class="tab-content active">
                    <div class="metrics-grid" style="margin-top: 1rem;">
                        <div class="metric-box">
                            <div class="metric-value" id="polyBalance">$0</div>
                            <div class="metric-label">Balance</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value" id="polyPositions">0</div>
                            <div class="metric-label">Positions</div>
                        </div>
                    </div>
                    <table class="positions-table">
                        <thead>
                            <tr>
                                <th>Market</th>
                                <th>Side</th>
                                <th>Shares</th>
                                <th>P&L</th>
                            </tr>
                        </thead>
                        <tbody id="polyPositionsTable"></tbody>
                    </table>
                </div>
                <div id="pred-kalshi" class="tab-content">
                    <div class="metrics-grid" style="margin-top: 1rem;">
                        <div class="metric-box">
                            <div class="metric-value" id="kalshiBalance">$0</div>
                            <div class="metric-label">Balance</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value" id="kalshiPositions">0</div>
                            <div class="metric-label">Positions</div>
                        </div>
                    </div>
                    <table class="positions-table">
                        <thead>
                            <tr>
                                <th>Market</th>
                                <th>Side</th>
                                <th>Contracts</th>
                                <th>P&L</th>
                            </tr>
                        </thead>
                        <tbody id="kalshiPositionsTable"></tbody>
                    </table>
                </div>
            </div>
        </div>
        
        <!-- Arbitrage Opportunities -->
        <div class="card">
            <div class="card-header">
                <h2>⚡ Arbitrage Opportunities</h2>
                <span class="card-badge" id="arbCount">0 found</span>
            </div>
            <div class="card-body">
                <div id="arbitrageList">
                    <div class="loading">
                        <div class="spinner"></div>
                        Scanning markets...
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Recent Alerts -->
        <div class="card">
            <div class="card-header">
                <h2>🔔 Recent Alerts</h2>
                <span class="card-badge" id="alertCount">0 unread</span>
            </div>
            <div class="card-body">
                <div id="alertList">
                    <p style="color: var(--text-secondary); text-align: center; padding: 2rem;">
                        No alerts
                    </p>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Socket.IO connection
        const socket = io('/dashboard');
        let isConnected = false;
        
        socket.on('connect', () => {
            isConnected = true;
            document.getElementById('connectionStatus').style.background = '#3fb950';
            console.log('Connected to dashboard');
        });
        
        socket.on('disconnect', () => {
            isConnected = false;
            document.getElementById('connectionStatus').style.background = '#f85149';
        });
        
        socket.on('data_update', (data) => {
            updateDashboard(data);
        });
        
        // Initial data fetch
        fetch('/api/data')
            .then(res => res.json())
            .then(data => updateDashboard(data))
            .catch(err => console.error('Error fetching initial data:', err));
        
        function updateDashboard(data) {
            if (data.error) {
                console.error('Data error:', data.error);
                return;
            }
            
            // Update timestamp
            if (data.last_update) {
                const time = new Date(data.last_update).toLocaleTimeString();
                document.getElementById('lastUpdate').textContent = time;
            }
            
            // Update combined metrics
            if (data.combined) {
                const c = data.combined;
                document.getElementById('totalPortfolio').textContent = formatCurrency(c.total_portfolio_value);
                document.getElementById('dailyPnl').textContent = formatCurrency(c.combined_daily_pnl);
                document.getElementById('dailyPnl').className = 
                    'header-stat-value ' + (c.combined_daily_pnl >= 0 ? 'positive' : 'negative');
                
                document.getElementById('equityAllocation').textContent = c.equity_allocation.toFixed(1) + '%';
                document.getElementById('predAllocation').textContent = c.prediction_allocation.toFixed(1) + '%';
                document.getElementById('totalExposure').textContent = formatCurrency(c.total_exposure);
                document.getElementById('exposureRatio').textContent = c.exposure_ratio.toFixed(2) + 'x';
                document.getElementById('totalPositions').textContent = c.total_positions;
                
                // Update system status
                updateSystemStatus(c.systems_status);
                
                // Update portfolio chart
                updatePortfolioChart(data);
            }
            
            // Update equity metrics
            if (data.equity && data.equity.metrics) {
                const e = data.equity.metrics;
                document.getElementById('equityValue').textContent = formatCurrency(e.equity);
                document.getElementById('equityCash').textContent = formatCurrency(e.cash);
                document.getElementById('equityWinRate').textContent = e.win_rate.toFixed(1) + '%';
                document.getElementById('equitySharpe').textContent = e.sharpe_ratio.toFixed(2);
                
                updatePnlValue('equityDailyPnl', e.daily_pnl);
                updatePnlValue('equityWeeklyPnl', e.weekly_pnl);
                updatePnlValue('equityMonthlyPnl', e.monthly_pnl);
                document.getElementById('equityMaxDD').textContent = e.max_drawdown.toFixed(1) + '%';
                
                // Update equity chart
                if (data.equity.equity_history) {
                    updateEquityChart(data.equity.equity_history);
                }
                
                // Update positions table
                if (data.equity.positions) {
                    updatePositionsTable('equityPositionsTable', data.equity.positions);
                }
                
                // Update sector chart
                if (data.equity.sector_breakdown) {
                    updateSectorChart(data.equity.sector_breakdown);
                }
            }
            
            // Update prediction markets
            if (data.prediction_markets) {
                const pm = data.prediction_markets;
                document.getElementById('predBalance').textContent = formatCurrency(pm.combined_balance);
                document.getElementById('predPositions').textContent = pm.combined_positions;
                
                const avgWinRate = ((pm.polymarket?.win_rate || 0) + (pm.kalshi?.win_rate || 0)) / 2;
                document.getElementById('predWinRate').textContent = avgWinRate.toFixed(1) + '%';
                
                const totalPnl = (pm.polymarket?.total_pnl || 0) + (pm.kalshi?.total_pnl || 0);
                updatePnlValue('predPnl', totalPnl);
                
                // Polymarket
                if (pm.polymarket) {
                    document.getElementById('polyBalance').textContent = formatCurrency(pm.polymarket.balance);
                    document.getElementById('polyPositions').textContent = pm.polymarket.positions_count;
                    updatePredPositionsTable('polyPositionsTable', pm.polymarket.positions);
                }
                
                // Kalshi
                if (pm.kalshi) {
                    document.getElementById('kalshiBalance').textContent = formatCurrency(pm.kalshi.balance);
                    document.getElementById('kalshiPositions').textContent = pm.kalshi.positions_count;
                    updatePredPositionsTable('kalshiPositionsTable', pm.kalshi.positions);
                }
            }
            
            // Update arbitrage
            if (data.arbitrage) {
                updateArbitrageList(data.arbitrage);
            }
            
            // Update alerts
            if (data.alerts) {
                updateAlerts(data.alerts);
            }
        }
        
        function formatCurrency(value) {
            return new Intl.NumberFormat('en-US', {
                style: 'currency',
                currency: 'USD',
                minimumFractionDigits: 2
            }).format(value || 0);
        }
        
        function updatePnlValue(elementId, value) {
            const el = document.getElementById(elementId);
            el.textContent = formatCurrency(value);
            el.className = 'metric-value ' + (value >= 0 ? 'pnl-positive' : 'pnl-negative');
        }
        
        function updateSystemStatus(status) {
            const container = document.getElementById('systemStatus');
            container.innerHTML = Object.entries(status).map(([name, state]) => 
                `<span class="system-badge ${state}">${name.charAt(0).toUpperCase() + name.slice(1)}</span>`
            ).join('');
        }
        
        function updatePositionsTable(tableId, positions) {
            const tbody = document.getElementById(tableId);
            if (!positions || positions.length === 0) {
                tbody.innerHTML = '<tr><td colspan="4" style="text-align: center; color: var(--text-secondary);">No positions</td></tr>';
                return;
            }
            
            tbody.innerHTML = positions.map(p => `
                <tr>
                    <td><strong>${p.symbol}</strong><br><small style="color: var(--text-secondary)">${p.sector}</small></td>
                    <td>${p.qty}</td>
                    <td>${formatCurrency(p.market_value)}</td>
                    <td class="${p.unrealized_pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}">
                        ${formatCurrency(p.unrealized_pnl)}<br>
                        <small>${p.unrealized_pnl_pct.toFixed(2)}%</small>
                    </td>
                </tr>
            `).join('');
        }
        
        function updatePredPositionsTable(tableId, positions) {
            const tbody = document.getElementById(tableId);
            if (!positions || positions.length === 0) {
                tbody.innerHTML = '<tr><td colspan="4" style="text-align: center; color: var(--text-secondary);">No positions</td></tr>';
                return;
            }
            
            tbody.innerHTML = positions.map(p => `
                <tr>
                    <td>${p.title.substring(0, 30)}...</td>
                    <td>${p.outcome}</td>
                    <td>${p.shares.toFixed(2)}</td>
                    <td class="${p.unrealized_pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}">
                        ${formatCurrency(p.unrealized_pnl)}
                    </td>
                </tr>
            `).join('');
        }
        
        function updateArbitrageList(opportunities) {
            const container = document.getElementById('arbitrageList');
            document.getElementById('arbCount').textContent = opportunities.length + ' found';
            
            if (!opportunities || opportunities.length === 0) {
                container.innerHTML = '<p style="color: var(--text-secondary); text-align: center; padding: 2rem;">No arbitrage opportunities detected</p>';
                return;
            }
            
            container.innerHTML = opportunities.map(opp => `
                <div class="arbitrage-card">
                    <div class="arbitrage-title">${opp.market_title}</div>
                    <div class="arbitrage-details">
                        <span>Poly: ${(opp.polymarket_price * 100).toFixed(1)}¢</span>
                        <span>Kalshi: ${(opp.kalshi_price * 100).toFixed(1)}¢</span>
                        <span style="color: var(--accent-green)">Spread: ${(opp.spread * 100).toFixed(1)}%</span>
                        <span>Est. Profit: $${opp.expected_profit.toFixed(2)}</span>
                    </div>
                </div>
            `).join('');
        }
        
        function updateAlerts(alerts) {
            const container = document.getElementById('alertList');
            document.getElementById('alertCount').textContent = alerts.length + ' unread';
            
            if (!alerts || alerts.length === 0) {
                container.innerHTML = '<p style="color: var(--text-secondary); text-align: center; padding: 2rem;">No alerts</p>';
                return;
            }
            
            container.innerHTML = alerts.slice(0, 10).map(alert => `
                <div class="arbitrage-card" style="border-left-color: ${
                    alert.level === 'critical' ? 'var(--accent-red)' : 
                    alert.level === 'warning' ? 'var(--accent-yellow)' : 'var(--accent-blue)'
                }">
                    <div style="display: flex; justify-content: space-between;">
                        <span>${alert.message}</span>
                        <button class="alert-dismiss" onclick="acknowledgeAlert('${alert.id}')">&times;</button>
                    </div>
                    <small style="color: var(--text-secondary)">${new Date(alert.timestamp).toLocaleString()}</small>
                </div>
            `).join('');
        }
        
        function acknowledgeAlert(alertId) {
            fetch('/api/alerts/' + alertId + '/acknowledge', { method: 'POST' })
                .then(() => {
                    // Alert will be removed on next update
                })
                .catch(err => console.error('Error acknowledging alert:', err));
        }
        
        function updatePortfolioChart(data) {
            if (!data.equity?.equity_history?.timestamps) return;
            
            const timestamps = data.equity.equity_history.timestamps.map(ts => new Date(ts * 1000));
            const values = data.equity.equity_history.values;
            
            const trace = {
                x: timestamps,
                y: values,
                type: 'scatter',
                mode: 'lines',
                fill: 'tozeroy',
                line: { color: '#58a6ff', width: 2 },
                fillcolor: 'rgba(88, 166, 255, 0.1)'
            };
            
            const layout = {
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                margin: { t: 20, r: 20, b: 40, l: 60 },
                xaxis: { 
                    gridcolor: '#30363d',
                    tickfont: { color: '#8b949e' }
                },
                yaxis: { 
                    gridcolor: '#30363d',
                    tickfont: { color: '#8b949e' },
                    tickformat: '$,.0f'
                },
                showlegend: false
            };
            
            Plotly.react('portfolioChart', [trace], layout, { responsive: true, displayModeBar: false });
        }
        
        function updateEquityChart(history) {
            if (!history.timestamps) return;
            
            const timestamps = history.timestamps.map(ts => new Date(ts * 1000));
            const values = history.values;
            
            const trace = {
                x: timestamps,
                y: values,
                type: 'scatter',
                mode: 'lines',
                line: { color: '#3fb950', width: 2 }
            };
            
            const layout = {
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                margin: { t: 10, r: 10, b: 30, l: 50 },
                xaxis: { 
                    gridcolor: '#30363d',
                    tickfont: { color: '#8b949e', size: 10 }
                },
                yaxis: { 
                    gridcolor: '#30363d',
                    tickfont: { color: '#8b949e', size: 10 },
                    tickformat: '$,.0f'
                },
                showlegend: false
            };
            
            Plotly.react('equityChart', [trace], layout, { responsive: true, displayModeBar: false });
        }
        
        function updateSectorChart(breakdown) {
            const labels = Object.keys(breakdown);
            const values = Object.values(breakdown);
            
            const trace = {
                labels: labels,
                values: values,
                type: 'pie',
                hole: 0.4,
                textinfo: 'label+percent',
                textfont: { color: '#f0f6fc', size: 11 },
                marker: {
                    colors: ['#58a6ff', '#3fb950', '#d29922', '#f85149', '#a371f7', '#8b949e']
                }
            };
            
            const layout = {
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                margin: { t: 20, r: 20, b: 20, l: 20 },
                showlegend: false
            };
            
            Plotly.react('sectorChart', [trace], layout, { responsive: true, displayModeBar: false });
        }
        
        function switchTab(section, tab) {
            // Update tab buttons
            document.querySelectorAll(`#${section === 'equity' ? 'equity-pnl' : 'pred-polymarket'}`).forEach(el => {
                const container = el.closest('.card-body');
                container.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                container.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            });
            
            event.target.classList.add('active');
            document.getElementById(`${section}-${tab}`).classList.add('active');
        }
    </script>
</body>
</html>
'''


# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def index():
    """Serve dashboard page."""
    return render_template_string(DASHBOARD_TEMPLATE)


@app.route('/api/data')
def get_data():
    """Get all dashboard data."""
    try:
        data = data_aggregator.refresh_all()
        return jsonify(data)
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/equity')
def get_equity():
    """Get equity data only."""
    try:
        data = data_aggregator._fetch_equity_data()
        return jsonify(data)
    except Exception as e:
        logger.error(f"Equity API error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/prediction-markets')
def get_prediction_markets():
    """Get prediction market data only."""
    try:
        data = data_aggregator._fetch_prediction_market_data()
        return jsonify(data)
    except Exception as e:
        logger.error(f"Prediction markets API error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/arbitrage')
def get_arbitrage():
    """Get arbitrage opportunities."""
    try:
        opportunities = data_aggregator.arbitrage_detector.find_opportunities()
        return jsonify([asdict(o) for o in opportunities])
    except Exception as e:
        logger.error(f"Arbitrage API error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/alerts')
def get_alerts():
    """Get all alerts."""
    try:
        alerts = data_aggregator.alert_manager.alerts
        return jsonify([asdict(a) for a in alerts])
    except Exception as e:
        logger.error(f"Alerts API error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/alerts/<alert_id>/acknowledge', methods=['POST'])
def acknowledge_alert(alert_id):
    """Acknowledge an alert."""
    try:
        success = data_aggregator.alert_manager.acknowledge_alert(alert_id)
        return jsonify({'success': success})
    except Exception as e:
        logger.error(f"Alert acknowledge error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'update_interval': Config.UPDATE_INTERVAL
    })


# =============================================================================
# WEBSOCKET EVENTS
# =============================================================================

@socketio.on('connect', namespace='/dashboard')
def handle_connect():
    """Handle WebSocket connection."""
    logger.info("Client connected to dashboard")
    emit('connected', {'status': 'ok'})


@socketio.on('disconnect', namespace='/dashboard')
def handle_disconnect():
    """Handle WebSocket disconnection."""
    logger.info("Client disconnected from dashboard")


@socketio.on('request_update', namespace='/dashboard')
def handle_update_request():
    """Handle manual update request."""
    try:
        data = data_aggregator.refresh_all()
        emit('data_update', data)
    except Exception as e:
        logger.error(f"Update request error: {e}")
        emit('error', {'message': str(e)})


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def start_dashboard(host: str = '0.0.0.0', port: int = None, debug: bool = False):
    """Start the dashboard server."""
    global update_thread, update_running
    
    port = port or Config.DASHBOARD_PORT
    
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    logger.info(f"Starting V29 Unified Dashboard on http://{host}:{port}")
    logger.info(f"Update interval: {Config.UPDATE_INTERVAL} seconds")
    
    # Start background updater
    update_running = True
    update_thread = threading.Thread(target=background_updater, daemon=True)
    update_thread.start()
    
    try:
        socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)
    finally:
        update_running = False
        if update_thread:
            update_thread.join(timeout=5)


def stop_dashboard():
    """Stop the dashboard server."""
    global update_running
    update_running = False
    logger.info("Dashboard stopped")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='V29 Unified Trading Dashboard')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=Config.DASHBOARD_PORT, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║           V29 Unified Performance Dashboard                  ║
    ║                                                              ║
    ║   Monitoring: Equity (Alpaca) + Prediction Markets           ║
    ║              (Polymarket, Kalshi)                            ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    print(f"\n📊 Dashboard URL: http://localhost:{args.port}")
    print(f"📡 API Base URL: http://localhost:{args.port}/api")
    print("\n🔑 Required Environment Variables:")
    print("   - ALPACA_API_KEY, ALPACA_SECRET_KEY")
    print("   - POLYMARKET_API_KEY, POLYMARKET_SECRET")
    print("   - KALSHI_API_KEY, KALSHI_SECRET")
    print("\n⚙️  Optional Settings:")
    print(f"   - UPDATE_INTERVAL: {Config.UPDATE_INTERVAL}s")
    print(f"   - DRAWDOWN_ALERT_PCT: {Config.DRAWDOWN_ALERT_PCT}%")
    print(f"   - POSITION_SIZE_ALERT_PCT: {Config.POSITION_SIZE_ALERT_PCT}%")
    print("\nPress Ctrl+C to stop the server.\n")
    
    start_dashboard(host=args.host, port=args.port, debug=args.debug)
