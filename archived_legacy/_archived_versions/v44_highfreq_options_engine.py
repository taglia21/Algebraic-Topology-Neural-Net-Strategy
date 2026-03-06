#!/usr/bin/env python3
"""
v44_highfreq_options_engine.py - High-Frequency Aggressive Options Trading Engine

Features:
- 60-second continuous scanning loop
- Websocket streaming for real-time prices
- Async execution for speed
- Aggressive options scalping (calls/puts)
- Premium selling strategies
- Strict risk controls (Kelly/4, max 2% per trade)

Author: Trading System v44
Date: 2026-01-26
"""

import os
import sys
import json
import logging
import argparse
import asyncio
import signal
from datetime import datetime, timedelta, time as dt_time
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple, Any, Callable
from pathlib import Path
from collections import deque
from enum import Enum
import threading
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

class OptionType(Enum):
    CALL = 'call'
    PUT = 'put'


class TradeDirection(Enum):
    BUY = 'buy'
    SELL = 'sell'


@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_loss_per_trade_pct: float = 0.02      # 2% max per trade
    max_daily_loss_pct: float = 0.05          # 5% max daily loss
    max_positions: int = 3                     # Max concurrent options
    kelly_fraction: float = 0.25               # Quarter-Kelly
    stop_loss_pct: float = 0.50               # 50% of premium
    take_profit_pct: float = 0.30             # 30% profit target
    min_dte_exit: int = 2                      # Exit if < 2 DTE and losing
    no_trade_before_close_min: int = 15       # No trading last 15 min


@dataclass
class ScalpConfig:
    """Scalping options configuration."""
    # Entry thresholds
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    volume_spike_mult: float = 2.0
    signal_threshold: float = 0.6
    
    # Option parameters - Scalp
    scalp_delta_min: float = 0.40
    scalp_delta_max: float = 0.50
    scalp_dte_min: int = 0
    scalp_dte_max: int = 7
    
    # Option parameters - Premium Selling
    sell_delta_min: float = 0.20
    sell_delta_max: float = 0.30
    sell_dte_min: int = 14
    sell_dte_max: int = 30


@dataclass
class EngineConfig:
    """Main engine configuration."""
    # Trading settings
    paper_trading: bool = True
    feed: str = 'iex'
    scan_interval_sec: int = 60
    
    # Universe
    symbols: List[str] = field(default_factory=lambda: [
        'SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA', 'NVDA', 
        'AMD', 'META', 'GOOGL', 'AMZN', 'IWM', 'DIA'
    ])
    
    # Rate limiting
    max_requests_per_min: int = 150
    request_buffer: int = 20  # Keep 20 requests buffer
    
    # Risk config
    risk: RiskConfig = field(default_factory=RiskConfig)
    scalp: ScalpConfig = field(default_factory=ScalpConfig)
    
    # State file
    state_file: str = 'v44_state.json'
    
    def effective_rate_limit(self) -> int:
        return self.max_requests_per_min - self.request_buffer


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure logging."""
    logger = logging.getLogger('v44_hf_options')
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

logger = setup_logging()


# =============================================================================
# RATE LIMITER
# =============================================================================

class RateLimiter:
    """Token bucket rate limiter for API requests."""
    
    def __init__(self, max_requests: int = 130, window_sec: int = 60):
        self.max_requests = max_requests
        self.window_sec = window_sec
        self.requests: deque = deque()
        self._lock = threading.Lock()
    
    def acquire(self) -> bool:
        """Acquire a request token. Returns True if allowed."""
        with self._lock:
            now = datetime.now()
            cutoff = now - timedelta(seconds=self.window_sec)
            
            # Remove old requests
            while self.requests and self.requests[0] < cutoff:
                self.requests.popleft()
            
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            return False
    
    async def wait_for_token(self) -> None:
        """Wait until a request token is available."""
        while not self.acquire():
            await asyncio.sleep(0.1)
    
    def remaining(self) -> int:
        """Get remaining requests in current window."""
        with self._lock:
            now = datetime.now()
            cutoff = now - timedelta(seconds=self.window_sec)
            while self.requests and self.requests[0] < cutoff:
                self.requests.popleft()
            return self.max_requests - len(self.requests)


# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

class TechnicalAnalyzer:
    """Calculate technical indicators for trading signals."""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_macd(
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD, signal line, and histogram."""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    @staticmethod
    def calculate_bollinger(
        prices: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + std_dev * std
        lower = sma - std_dev * std
        return upper, sma, lower
    
    @staticmethod
    def calculate_volume_spike(volumes: pd.Series, period: int = 20) -> pd.Series:
        """Calculate volume relative to moving average."""
        vol_sma = volumes.rolling(window=period).mean()
        return volumes / vol_sma
    
    @staticmethod
    def detect_support_resistance(
        prices: pd.DataFrame,
        window: int = 20
    ) -> Tuple[float, float]:
        """Detect support and resistance levels."""
        high = prices['high'].rolling(window).max().iloc[-1]
        low = prices['low'].rolling(window).min().iloc[-1]
        return float(low), float(high)  # support, resistance
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run full technical analysis."""
        if len(data) < 30:
            return {}
        
        close = data['close']
        volume = data['volume']
        
        rsi = self.calculate_rsi(close)
        macd, macd_signal, macd_hist = self.calculate_macd(close)
        bb_upper, bb_mid, bb_lower = self.calculate_bollinger(close)
        vol_spike = self.calculate_volume_spike(volume)
        support, resistance = self.detect_support_resistance(data)
        
        current_price = close.iloc[-1]
        
        # MACD crossover detection
        macd_bullish_cross = (macd.iloc[-2] < macd_signal.iloc[-2] and 
                              macd.iloc[-1] > macd_signal.iloc[-1])
        macd_bearish_cross = (macd.iloc[-2] > macd_signal.iloc[-2] and 
                              macd.iloc[-1] < macd_signal.iloc[-1])
        
        # Support/resistance proximity
        near_support = current_price < support * 1.02
        near_resistance = current_price > resistance * 0.98
        
        return {
            'rsi': float(rsi.iloc[-1]),
            'macd': float(macd.iloc[-1]),
            'macd_signal': float(macd_signal.iloc[-1]),
            'macd_hist': float(macd_hist.iloc[-1]),
            'macd_bullish_cross': macd_bullish_cross,
            'macd_bearish_cross': macd_bearish_cross,
            'bb_upper': float(bb_upper.iloc[-1]),
            'bb_mid': float(bb_mid.iloc[-1]),
            'bb_lower': float(bb_lower.iloc[-1]),
            'volume_spike': float(vol_spike.iloc[-1]),
            'support': support,
            'resistance': resistance,
            'near_support': near_support,
            'near_resistance': near_resistance,
            'price': float(current_price)
        }


# =============================================================================
# OPTIONS SIGNAL GENERATOR
# =============================================================================

class OptionsSignalGenerator:
    """Generate options trading signals based on technical analysis."""
    
    def __init__(self, config: EngineConfig):
        self.config = config
        self.scalp = config.scalp
        
    def generate_call_signal(self, analysis: Dict) -> Tuple[float, str]:
        """
        Generate bullish call signal.
        Returns: (signal_strength 0-1, rationale)
        """
        if not analysis:
            return 0.0, "No data"
        
        signals = []
        rationale_parts = []
        
        # RSI oversold
        rsi = analysis.get('rsi', 50)
        if rsi < self.scalp.rsi_oversold:
            signals.append(0.35)
            rationale_parts.append(f"RSI oversold ({rsi:.1f})")
        
        # Near support
        if analysis.get('near_support', False):
            signals.append(0.25)
            rationale_parts.append("Near support")
        
        # MACD bullish crossover
        if analysis.get('macd_bullish_cross', False):
            signals.append(0.25)
            rationale_parts.append("MACD bullish cross")
        
        # Volume spike
        vol_spike = analysis.get('volume_spike', 1.0)
        if vol_spike > self.scalp.volume_spike_mult:
            signals.append(0.15)
            rationale_parts.append(f"Volume spike ({vol_spike:.1f}x)")
        
        strength = min(sum(signals), 1.0)
        rationale = " + ".join(rationale_parts) if rationale_parts else "No signals"
        
        return strength, rationale
    
    def generate_put_signal(self, analysis: Dict) -> Tuple[float, str]:
        """
        Generate bearish put signal.
        Returns: (signal_strength 0-1, rationale)
        """
        if not analysis:
            return 0.0, "No data"
        
        signals = []
        rationale_parts = []
        
        # RSI overbought
        rsi = analysis.get('rsi', 50)
        if rsi > self.scalp.rsi_overbought:
            signals.append(0.35)
            rationale_parts.append(f"RSI overbought ({rsi:.1f})")
        
        # Near resistance
        if analysis.get('near_resistance', False):
            signals.append(0.25)
            rationale_parts.append("Near resistance")
        
        # MACD bearish crossover
        if analysis.get('macd_bearish_cross', False):
            signals.append(0.25)
            rationale_parts.append("MACD bearish cross")
        
        # Volume spike on down move
        vol_spike = analysis.get('volume_spike', 1.0)
        macd_hist = analysis.get('macd_hist', 0)
        if vol_spike > self.scalp.volume_spike_mult and macd_hist < 0:
            signals.append(0.15)
            rationale_parts.append(f"Volume spike on down ({vol_spike:.1f}x)")
        
        strength = min(sum(signals), 1.0)
        rationale = " + ".join(rationale_parts) if rationale_parts else "No signals"
        
        return strength, rationale
    
    def generate_premium_sell_signal(
        self, 
        analysis: Dict,
        has_shares: bool = False
    ) -> Tuple[OptionType, float, str]:
        """
        Generate premium selling signal (covered call or cash-secured put).
        Returns: (option_type, signal_strength, rationale)
        """
        if not analysis:
            return OptionType.CALL, 0.0, "No data"
        
        rsi = analysis.get('rsi', 50)
        vol_spike = analysis.get('volume_spike', 1.0)
        
        # Prefer covered calls if we have shares and RSI > 60
        if has_shares and rsi > 60:
            strength = min(0.4 + (rsi - 60) / 100, 0.8)
            return OptionType.CALL, strength, f"Covered call: RSI={rsi:.1f}"
        
        # Cash-secured puts if RSI < 40 (want to buy lower)
        if rsi < 40:
            strength = min(0.4 + (40 - rsi) / 100, 0.8)
            return OptionType.PUT, strength, f"Cash-secured put: RSI={rsi:.1f}"
        
        return OptionType.CALL, 0.0, "No premium sell signal"


# =============================================================================
# RISK MANAGER
# =============================================================================

class RiskManager:
    """Comprehensive risk management for options trading."""
    
    def __init__(self, config: EngineConfig):
        self.config = config
        self.risk = config.risk
        self.daily_pnl: float = 0.0
        self.trade_count: int = 0
        self.open_positions: List[Dict] = []
        self.daily_reset_date: datetime = datetime.now().date()
    
    def reset_daily_if_needed(self) -> None:
        """Reset daily counters if new day."""
        today = datetime.now().date()
        if today > self.daily_reset_date:
            logger.info("📅 New trading day - resetting daily limits")
            self.daily_pnl = 0.0
            self.trade_count = 0
            self.daily_reset_date = today
    
    def can_trade(self, portfolio_value: float) -> Tuple[bool, str]:
        """Check if trading is allowed."""
        self.reset_daily_if_needed()
        
        # Check market hours
        now = datetime.now()
        market_close = now.replace(hour=16, minute=0, second=0)
        no_trade_cutoff = market_close - timedelta(minutes=self.risk.no_trade_before_close_min)
        
        if now >= no_trade_cutoff:
            return False, f"No trading last {self.risk.no_trade_before_close_min} min before close"
        
        # Check daily loss limit
        max_daily_loss = portfolio_value * self.risk.max_daily_loss_pct
        if self.daily_pnl <= -max_daily_loss:
            return False, f"Daily loss limit reached (${-self.daily_pnl:.2f})"
        
        # Check max positions
        if len(self.open_positions) >= self.risk.max_positions:
            return False, f"Max positions ({self.risk.max_positions}) reached"
        
        return True, "OK"
    
    def calculate_position_size(
        self,
        portfolio_value: float,
        confidence: float,
        option_price: float
    ) -> Tuple[int, float]:
        """
        Calculate position size using quarter-Kelly.
        Returns: (num_contracts, notional_value)
        """
        # Max loss per trade
        max_loss = portfolio_value * self.risk.max_loss_per_trade_pct
        
        # Quarter-Kelly adjustment
        adjusted_confidence = confidence * self.risk.kelly_fraction
        
        # Position size in dollars
        position_value = max_loss / self.risk.stop_loss_pct  # Size based on stop
        position_value *= adjusted_confidence
        
        # Calculate contracts (options are 100 shares)
        contract_cost = option_price * 100
        num_contracts = max(1, int(position_value / contract_cost))
        
        # Cap to not exceed max loss
        while num_contracts * contract_cost * self.risk.stop_loss_pct > max_loss:
            num_contracts -= 1
            if num_contracts <= 0:
                return 0, 0.0
        
        return num_contracts, num_contracts * contract_cost
    
    def check_exit_conditions(
        self,
        position: Dict,
        current_price: float
    ) -> Tuple[bool, str]:
        """
        Check if position should be exited.
        Returns: (should_exit, reason)
        """
        entry_price = position.get('entry_price', 0)
        if entry_price <= 0:
            return False, ""
        
        pnl_pct = (current_price - entry_price) / entry_price
        
        # Take profit
        if pnl_pct >= self.risk.take_profit_pct:
            return True, f"Take profit ({pnl_pct:.1%})"
        
        # Stop loss
        if pnl_pct <= -self.risk.stop_loss_pct:
            return True, f"Stop loss ({pnl_pct:.1%})"
        
        # Time decay exit
        dte = position.get('dte', 99)
        if dte < self.risk.min_dte_exit and pnl_pct < 0:
            return True, f"Time decay exit (DTE={dte}, P&L={pnl_pct:.1%})"
        
        return False, ""
    
    def record_trade(self, pnl: float) -> None:
        """Record trade P&L."""
        self.daily_pnl += pnl
        self.trade_count += 1
    
    def add_position(self, position: Dict) -> None:
        """Add open position."""
        self.open_positions.append(position)
    
    def remove_position(self, symbol: str) -> None:
        """Remove closed position."""
        self.open_positions = [p for p in self.open_positions if p.get('symbol') != symbol]
    
    def get_status(self) -> Dict:
        """Get risk manager status."""
        return {
            'daily_pnl': self.daily_pnl,
            'trade_count': self.trade_count,
            'open_positions': len(self.open_positions),
            'daily_reset_date': str(self.daily_reset_date)
        }


# =============================================================================
# DATA MANAGER (IEX + WEBSOCKET)
# =============================================================================

class DataManager:
    """
    Manages data feeds from Alpaca with IEX and websocket streaming.
    """
    
    def __init__(self, config: EngineConfig):
        self.config = config
        self.rate_limiter = RateLimiter(config.effective_rate_limit())
        self.trading_client = None
        self.stock_client = None
        self.options_client = None
        self.ws_running = False
        self.latest_quotes: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        
        self._init_clients()
    
    def _init_clients(self) -> None:
        """Initialize Alpaca clients."""
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data import StockHistoricalDataClient
            
            api_key = os.environ.get('APCA_API_KEY_ID')
            api_secret = os.environ.get('APCA_API_SECRET_KEY')
            
            if not api_key or not api_secret:
                logger.error("✗ Missing Alpaca credentials")
                return
            
            self.trading_client = TradingClient(
                api_key, api_secret,
                paper=self.config.paper_trading
            )
            
            self.stock_client = StockHistoricalDataClient(api_key, api_secret)
            
            # Options client (if available)
            try:
                from alpaca.data import OptionHistoricalDataClient
                self.options_client = OptionHistoricalDataClient(api_key, api_secret)
                logger.info("✓ Options data client initialized")
            except ImportError:
                logger.warning("⚠ Options data client not available")
            
            logger.info("✓ Alpaca clients initialized (IEX feed)")
            
        except ImportError as e:
            logger.error(f"✗ alpaca-py not installed: {e}")
        except Exception as e:
            logger.error(f"✗ Client init error: {e}")
    
    async def fetch_bars(
        self,
        symbols: List[str],
        days: int = 30
    ) -> Dict[str, pd.DataFrame]:
        """Fetch historical bars using IEX feed."""
        if not self.stock_client:
            return await self._fetch_yfinance(symbols, days)
        
        await self.rate_limiter.wait_for_token()
        
        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            
            end = datetime.now()
            start = end - timedelta(days=days + 5)
            
            request = StockBarsRequest(
                symbol_or_symbols=symbols,
                start=start,
                end=end,
                timeframe=TimeFrame.Day,
                feed='iex'  # Always use IEX
            )
            
            bars = self.stock_client.get_stock_bars(request)
            
            data = {}
            for symbol in symbols:
                if symbol in bars.data:
                    df = pd.DataFrame([{
                        'open': b.open,
                        'high': b.high,
                        'low': b.low,
                        'close': b.close,
                        'volume': b.volume,
                        'timestamp': b.timestamp
                    } for b in bars.data[symbol]])
                    
                    if not df.empty:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df.set_index('timestamp', inplace=True)
                        data[symbol] = df.tail(days)
            
            logger.debug(f"Fetched bars for {len(data)} symbols from IEX")
            return data
            
        except Exception as e:
            logger.warning(f"IEX bars error: {e}, falling back to yfinance")
            return await self._fetch_yfinance(symbols, days)
    
    async def _fetch_yfinance(
        self,
        symbols: List[str],
        days: int
    ) -> Dict[str, pd.DataFrame]:
        """Fallback to yfinance for historical data."""
        data = {}
        try:
            import yfinance as yf
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(period=f'{days}d')
                    if not df.empty:
                        df.columns = df.columns.str.lower()
                        df = df[['open', 'high', 'low', 'close', 'volume']]
                        data[symbol] = df
                except Exception:
                    pass
                    
        except ImportError:
            logger.error("yfinance not installed")
        
        return data
    
    async def fetch_intraday_bars(
        self,
        symbols: List[str],
        timeframe: str = '5Min'
    ) -> Dict[str, pd.DataFrame]:
        """Fetch intraday bars for signals."""
        if not self.stock_client:
            return {}
        
        await self.rate_limiter.wait_for_token()
        
        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            
            tf_map = {
                '1Min': TimeFrame.Minute,
                '5Min': TimeFrame(5, 'Min'),
                '15Min': TimeFrame(15, 'Min'),
                '1Hour': TimeFrame.Hour
            }
            tf = tf_map.get(timeframe, TimeFrame(5, 'Min'))
            
            end = datetime.now()
            start = end - timedelta(hours=8)  # Today's data
            
            request = StockBarsRequest(
                symbol_or_symbols=symbols,
                start=start,
                end=end,
                timeframe=tf,
                feed='iex'
            )
            
            bars = self.stock_client.get_stock_bars(request)
            
            data = {}
            for symbol in symbols:
                if symbol in bars.data:
                    df = pd.DataFrame([{
                        'open': b.open,
                        'high': b.high,
                        'low': b.low,
                        'close': b.close,
                        'volume': b.volume,
                        'timestamp': b.timestamp
                    } for b in bars.data[symbol]])
                    
                    if not df.empty:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df.set_index('timestamp', inplace=True)
                        data[symbol] = df
            
            return data
            
        except Exception as e:
            logger.warning(f"Intraday bars error: {e}")
            return {}
    
    async def get_options_chain(
        self,
        symbol: str,
        dte_min: int = 0,
        dte_max: int = 30
    ) -> List[Dict]:
        """Fetch options chain for a symbol."""
        await self.rate_limiter.wait_for_token()
        
        try:
            from alpaca.trading.requests import GetOptionContractsRequest
            from alpaca.trading.enums import AssetStatus
            
            if not self.trading_client:
                return []
            
            request = GetOptionContractsRequest(
                underlying_symbols=[symbol],
                status=AssetStatus.ACTIVE,
                expiration_date_gte=(datetime.now() + timedelta(days=dte_min)).strftime('%Y-%m-%d'),
                expiration_date_lte=(datetime.now() + timedelta(days=dte_max)).strftime('%Y-%m-%d')
            )
            
            contracts = self.trading_client.get_option_contracts(request)
            
            return [{
                'symbol': c.symbol,
                'underlying': c.underlying_symbol,
                'type': c.type.value,
                'strike': float(c.strike_price),
                'expiration': str(c.expiration_date),
                'dte': (c.expiration_date - datetime.now().date()).days
            } for c in contracts.option_contracts]
            
        except Exception as e:
            logger.debug(f"Options chain error for {symbol}: {e}")
            return []
    
    async def get_option_quote(self, option_symbol: str) -> Optional[Dict]:
        """Get quote for an option contract."""
        await self.rate_limiter.wait_for_token()
        
        try:
            if not self.options_client:
                return None
            
            from alpaca.data.requests import OptionLatestQuoteRequest
            
            request = OptionLatestQuoteRequest(symbol_or_symbols=[option_symbol])
            quotes = self.options_client.get_option_latest_quote(request)
            
            if option_symbol in quotes:
                q = quotes[option_symbol]
                return {
                    'bid': float(q.bid_price),
                    'ask': float(q.ask_price),
                    'mid': (float(q.bid_price) + float(q.ask_price)) / 2
                }
                
        except Exception as e:
            logger.debug(f"Option quote error: {e}")
        
        return None
    
    def get_account(self) -> Optional[Dict]:
        """Get account information."""
        if not self.trading_client:
            return None
        
        try:
            account = self.trading_client.get_account()
            return {
                'equity': float(account.equity),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value)
            }
        except Exception as e:
            logger.error(f"Account fetch error: {e}")
            return None
    
    def get_positions(self) -> List[Dict]:
        """Get current positions."""
        if not self.trading_client:
            return []
        
        try:
            positions = self.trading_client.get_all_positions()
            return [{
                'symbol': p.symbol,
                'qty': float(p.qty),
                'market_value': float(p.market_value),
                'unrealized_pl': float(p.unrealized_pl),
                'asset_class': p.asset_class.value
            } for p in positions]
        except Exception as e:
            logger.error(f"Positions fetch error: {e}")
            return []
    
    async def start_websocket(self, symbols: List[str]) -> None:
        """Start websocket streaming for real-time quotes."""
        try:
            from alpaca.data.live import StockDataStream
            
            api_key = os.environ.get('APCA_API_KEY_ID')
            api_secret = os.environ.get('APCA_API_SECRET_KEY')
            
            if not api_key or not api_secret:
                return
            
            stream = StockDataStream(api_key, api_secret, feed='iex')
            
            async def quote_handler(quote):
                with self._lock:
                    self.latest_quotes[quote.symbol] = {
                        'bid': float(quote.bid_price),
                        'ask': float(quote.ask_price),
                        'timestamp': quote.timestamp
                    }
            
            for symbol in symbols:
                stream.subscribe_quotes(quote_handler, symbol)
            
            self.ws_running = True
            logger.info(f"✓ Websocket streaming started for {len(symbols)} symbols")
            
            await stream.run()
            
        except Exception as e:
            logger.warning(f"Websocket error: {e}")
            self.ws_running = False
    
    def get_latest_quote(self, symbol: str) -> Optional[Dict]:
        """Get latest quote from websocket cache."""
        with self._lock:
            return self.latest_quotes.get(symbol)


# =============================================================================
# ORDER EXECUTION ENGINE
# =============================================================================

class ExecutionEngine:
    """Execute options orders via Alpaca."""
    
    def __init__(self, data_manager: DataManager, risk_manager: RiskManager):
        self.data = data_manager
        self.risk = risk_manager
    
    async def submit_option_order(
        self,
        option_symbol: str,
        side: TradeDirection,
        qty: int,
        order_type: str = 'market'
    ) -> Optional[Dict]:
        """Submit an options order."""
        if not self.data.trading_client:
            logger.warning("Trading client not available")
            return None
        
        try:
            from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce
            
            alpaca_side = OrderSide.BUY if side == TradeDirection.BUY else OrderSide.SELL
            
            if order_type == 'market':
                request = MarketOrderRequest(
                    symbol=option_symbol,
                    qty=qty,
                    side=alpaca_side,
                    time_in_force=TimeInForce.DAY
                )
            else:
                # Get current quote for limit order
                quote = await self.data.get_option_quote(option_symbol)
                if not quote:
                    return None
                
                limit_price = quote['ask'] if side == TradeDirection.BUY else quote['bid']
                
                request = LimitOrderRequest(
                    symbol=option_symbol,
                    qty=qty,
                    side=alpaca_side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=limit_price
                )
            
            order = self.data.trading_client.submit_order(request)
            
            logger.info(f"✓ Order submitted: {side.value} {qty}x {option_symbol}")
            
            return {
                'order_id': str(order.id),
                'symbol': option_symbol,
                'side': side.value,
                'qty': qty,
                'status': order.status.value
            }
            
        except Exception as e:
            logger.error(f"Order submission error: {e}")
            return None
    
    async def close_position(self, symbol: str) -> Optional[Dict]:
        """Close an options position."""
        try:
            if not self.data.trading_client:
                return None
            
            result = self.data.trading_client.close_position(symbol)
            logger.info(f"✓ Position closed: {symbol}")
            
            return {'symbol': symbol, 'status': 'closed'}
            
        except Exception as e:
            logger.error(f"Close position error: {e}")
            return None


# =============================================================================
# STATE PERSISTENCE
# =============================================================================

class StateManager:
    """JSON state persistence."""
    
    def __init__(self, filepath: str = 'v44_state.json'):
        self.filepath = Path(filepath)
    
    def save(self, state: Dict) -> bool:
        try:
            state['timestamp'] = datetime.now().isoformat()
            with open(self.filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            return True
        except Exception as e:
            logger.error(f"State save error: {e}")
            return False
    
    def load(self) -> Dict:
        try:
            if self.filepath.exists():
                with open(self.filepath, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"State load error: {e}")
        return {}


# =============================================================================
# MAIN TRADING ENGINE
# =============================================================================

class HighFreqOptionsEngine:
    """
    Main high-frequency options trading engine.
    Runs continuous scanning loop with 60-second intervals.
    """
    
    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        
        logger.info("=" * 60)
        logger.info("V44 HIGH-FREQUENCY OPTIONS ENGINE")
        logger.info("=" * 60)
        
        self.data = DataManager(self.config)
        self.risk = RiskManager(self.config)
        self.tech = TechnicalAnalyzer()
        self.signals = OptionsSignalGenerator(self.config)
        self.execution = ExecutionEngine(self.data, self.risk)
        self.state = StateManager(self.config.state_file)
        
        self._running = False
        self._stop_event = asyncio.Event()
        
        logger.info("=" * 60)
    
    async def scan_for_signals(self) -> List[Dict]:
        """Scan all symbols for trading signals."""
        logger.info("🔍 Scanning for signals...")
        
        # Fetch historical data for technical analysis
        bars = await self.data.fetch_bars(self.config.symbols, days=30)
        
        if not bars:
            logger.warning("No data available")
            return []
        
        opportunities = []
        
        for symbol, df in bars.items():
            analysis = self.tech.analyze(df)
            if not analysis:
                continue
            
            # Check for call signal
            call_strength, call_rationale = self.signals.generate_call_signal(analysis)
            if call_strength >= self.config.scalp.signal_threshold:
                opportunities.append({
                    'symbol': symbol,
                    'type': OptionType.CALL,
                    'direction': TradeDirection.BUY,
                    'strength': call_strength,
                    'rationale': call_rationale,
                    'analysis': analysis
                })
            
            # Check for put signal
            put_strength, put_rationale = self.signals.generate_put_signal(analysis)
            if put_strength >= self.config.scalp.signal_threshold:
                opportunities.append({
                    'symbol': symbol,
                    'type': OptionType.PUT,
                    'direction': TradeDirection.BUY,
                    'strength': put_strength,
                    'rationale': put_rationale,
                    'analysis': analysis
                })
            
            # Check for premium selling
            positions = self.data.get_positions()
            has_shares = any(p['symbol'] == symbol and p['asset_class'] == 'us_equity' 
                           for p in positions)
            
            sell_type, sell_strength, sell_rationale = self.signals.generate_premium_sell_signal(
                analysis, has_shares
            )
            if sell_strength >= 0.5:  # Lower threshold for selling premium
                opportunities.append({
                    'symbol': symbol,
                    'type': sell_type,
                    'direction': TradeDirection.SELL,
                    'strength': sell_strength,
                    'rationale': sell_rationale,
                    'analysis': analysis
                })
        
        # Sort by signal strength
        opportunities.sort(key=lambda x: x['strength'], reverse=True)
        
        logger.info(f"Found {len(opportunities)} potential opportunities")
        return opportunities
    
    async def find_option_contract(
        self,
        symbol: str,
        option_type: OptionType,
        direction: TradeDirection
    ) -> Optional[Dict]:
        """Find suitable option contract based on strategy."""
        config = self.config.scalp
        
        # Determine DTE range based on direction
        if direction == TradeDirection.BUY:
            dte_min, dte_max = config.scalp_dte_min, config.scalp_dte_max
            delta_min, delta_max = config.scalp_delta_min, config.scalp_delta_max
        else:
            dte_min, dte_max = config.sell_dte_min, config.sell_dte_max
            delta_min, delta_max = config.sell_delta_min, config.sell_delta_max
        
        # Get options chain
        chain = await self.data.get_options_chain(symbol, dte_min, dte_max)
        
        if not chain:
            return None
        
        # Filter by type
        type_str = option_type.value
        filtered = [c for c in chain if c['type'] == type_str]
        
        if not filtered:
            return None
        
        # Get current price for ATM selection
        quote = self.data.get_latest_quote(symbol)
        if quote:
            current_price = (quote['bid'] + quote['ask']) / 2
        else:
            # Use last close from bars
            bars = await self.data.fetch_bars([symbol], days=1)
            if symbol in bars and not bars[symbol].empty:
                current_price = bars[symbol]['close'].iloc[-1]
            else:
                return None
        
        # Find ATM option (closest to current price)
        best_contract = min(
            filtered,
            key=lambda c: abs(c['strike'] - current_price)
        )
        
        # Get option quote
        option_quote = await self.data.get_option_quote(best_contract['symbol'])
        if option_quote:
            best_contract['bid'] = option_quote['bid']
            best_contract['ask'] = option_quote['ask']
            best_contract['mid'] = option_quote['mid']
        
        return best_contract
    
    async def execute_opportunity(self, opportunity: Dict) -> Optional[Dict]:
        """Execute a trading opportunity."""
        symbol = opportunity['symbol']
        option_type = opportunity['type']
        direction = opportunity['direction']
        
        # Find suitable contract
        contract = await self.find_option_contract(symbol, option_type, direction)
        
        if not contract:
            logger.debug(f"No suitable contract found for {symbol}")
            return None
        
        # Get account and check risk
        account = self.data.get_account()
        if not account:
            return None
        
        can_trade, reason = self.risk.can_trade(account['portfolio_value'])
        if not can_trade:
            logger.warning(f"Cannot trade: {reason}")
            return None
        
        # Calculate position size
        option_price = contract.get('mid', contract.get('ask', 1.0))
        num_contracts, notional = self.risk.calculate_position_size(
            account['portfolio_value'],
            opportunity['strength'],
            option_price
        )
        
        if num_contracts <= 0:
            logger.debug("Position size too small")
            return None
        
        # Execute order
        result = await self.execution.submit_option_order(
            contract['symbol'],
            direction,
            num_contracts
        )
        
        if result:
            # Track position
            position = {
                'symbol': contract['symbol'],
                'underlying': symbol,
                'type': option_type.value,
                'direction': direction.value,
                'qty': num_contracts,
                'entry_price': option_price,
                'dte': contract.get('dte', 7),
                'rationale': opportunity['rationale'],
                'timestamp': datetime.now().isoformat()
            }
            self.risk.add_position(position)
            
            logger.info(
                f"✅ TRADE: {direction.value} {num_contracts}x {contract['symbol']} "
                f"@ ${option_price:.2f} | {opportunity['rationale']}"
            )
        
        return result
    
    async def monitor_positions(self) -> List[Dict]:
        """Monitor open positions for exit signals."""
        exits = []
        
        for position in self.risk.open_positions.copy():
            symbol = position['symbol']
            
            # Get current quote
            quote = await self.data.get_option_quote(symbol)
            if not quote:
                continue
            
            current_price = quote['mid']
            
            # Check exit conditions
            should_exit, reason = self.risk.check_exit_conditions(position, current_price)
            
            if should_exit:
                result = await self.execution.close_position(symbol)
                if result:
                    pnl = (current_price - position['entry_price']) * position['qty'] * 100
                    self.risk.record_trade(pnl)
                    self.risk.remove_position(symbol)
                    
                    logger.info(f"🔔 EXIT: {symbol} | {reason} | P&L: ${pnl:+.2f}")
                    exits.append({**result, 'reason': reason, 'pnl': pnl})
        
        return exits
    
    async def run_scan_cycle(self) -> Dict:
        """Run a single scan cycle."""
        cycle_start = datetime.now()
        results = {
            'timestamp': cycle_start.isoformat(),
            'opportunities': 0,
            'trades': 0,
            'exits': 0
        }
        
        try:
            # Monitor existing positions
            exits = await self.monitor_positions()
            results['exits'] = len(exits)
            
            # Scan for new signals
            opportunities = await self.scan_for_signals()
            results['opportunities'] = len(opportunities)
            
            # Execute top opportunities
            for opp in opportunities[:3]:  # Max 3 per cycle
                result = await self.execute_opportunity(opp)
                if result:
                    results['trades'] += 1
            
            # Save state
            state = {
                'last_scan': results,
                'risk': self.risk.get_status(),
                'rate_limit_remaining': self.data.rate_limiter.remaining()
            }
            self.state.save(state)
            
        except Exception as e:
            logger.error(f"Scan cycle error: {e}")
            results['error'] = str(e)
        
        elapsed = (datetime.now() - cycle_start).total_seconds()
        logger.info(
            f"📊 Cycle complete: {results['opportunities']} opps, "
            f"{results['trades']} trades, {results['exits']} exits "
            f"({elapsed:.1f}s)"
        )
        
        return results
    
    async def run_continuous(self) -> None:
        """Run continuous scanning loop."""
        logger.info(f"🚀 Starting continuous scan (interval: {self.config.scan_interval_sec}s)")
        self._running = True
        
        # Start websocket in background
        ws_task = asyncio.create_task(
            self.data.start_websocket(self.config.symbols)
        )
        
        cycle_count = 0
        
        try:
            while self._running and not self._stop_event.is_set():
                cycle_count += 1
                logger.info(f"\n{'='*40}")
                logger.info(f"SCAN CYCLE #{cycle_count}")
                logger.info(f"{'='*40}")
                
                await self.run_scan_cycle()
                
                # Wait for next interval
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self.config.scan_interval_sec
                    )
                except asyncio.TimeoutError:
                    pass  # Normal - continue to next cycle
                    
        except asyncio.CancelledError:
            logger.info("Scan loop cancelled")
        finally:
            self._running = False
            ws_task.cancel()
    
    def stop(self) -> None:
        """Stop the engine."""
        logger.info("⏹️ Stopping engine...")
        self._running = False
        self._stop_event.set()
    
    async def run_test(self) -> Dict:
        """Run a single test scan without trading."""
        logger.info("🧪 Running test scan (no trades)...")
        
        opportunities = await self.scan_for_signals()
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'mode': 'test',
            'symbols_scanned': len(self.config.symbols),
            'opportunities': []
        }
        
        for opp in opportunities[:10]:
            results['opportunities'].append({
                'symbol': opp['symbol'],
                'type': opp['type'].value,
                'direction': opp['direction'].value,
                'strength': round(opp['strength'], 3),
                'rationale': opp['rationale']
            })
        
        return results
    
    def get_status(self) -> Dict:
        """Get current engine status."""
        status = {
            'timestamp': datetime.now().isoformat(),
            'engine': 'v44_highfreq_options',
            'running': self._running
        }
        
        # Account info
        account = self.data.get_account()
        if account:
            status['account'] = account
        
        # Positions
        positions = self.data.get_positions()
        status['positions'] = positions
        status['options_positions'] = [p for p in positions if p.get('asset_class') == 'us_option']
        
        # Risk status
        status['risk'] = self.risk.get_status()
        
        # Rate limiting
        status['rate_limit_remaining'] = self.data.rate_limiter.remaining()
        
        # Websocket status
        status['websocket_running'] = self.data.ws_running
        status['latest_quotes'] = len(self.data.latest_quotes)
        
        return status
    
    def print_status(self) -> None:
        """Print formatted status."""
        status = self.get_status()
        
        print("\n" + "=" * 60)
        print("V44 HIGH-FREQUENCY OPTIONS ENGINE STATUS")
        print("=" * 60)
        print(f"Timestamp: {status['timestamp']}")
        print(f"Engine Running: {'✅ YES' if status['running'] else '❌ NO'}")
        print()
        
        if 'account' in status:
            acc = status['account']
            print("💰 ACCOUNT:")
            print(f"  Equity:       ${acc['equity']:>12,.2f}")
            print(f"  Cash:         ${acc['cash']:>12,.2f}")
            print(f"  Buying Power: ${acc['buying_power']:>12,.2f}")
        
        print()
        print("📊 RISK STATUS:")
        risk = status.get('risk', {})
        print(f"  Daily P&L:     ${risk.get('daily_pnl', 0):>+10,.2f}")
        print(f"  Trade Count:   {risk.get('trade_count', 0)}")
        print(f"  Open Positions:{risk.get('open_positions', 0)}")
        
        print()
        print("⚡ RATE LIMITING:")
        print(f"  Remaining:     {status.get('rate_limit_remaining', 0)}/min")
        
        print()
        print(f"📡 WEBSOCKET: {'✅ Connected' if status.get('websocket_running') else '❌ Disconnected'}")
        print(f"   Live Quotes:  {status.get('latest_quotes', 0)}")
        
        print()
        print("📈 OPTIONS POSITIONS:")
        opts = status.get('options_positions', [])
        if opts:
            for p in opts:
                print(f"  {p['symbol']:20} | {p['qty']:>5} | ${p.get('market_value', 0):>10,.2f}")
        else:
            print("  (none)")
        
        print("=" * 60)


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='V44 High-Frequency Options Trading Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python v44_highfreq_options_engine.py --status
  python v44_highfreq_options_engine.py --test
  python v44_highfreq_options_engine.py --trade
  python v44_highfreq_options_engine.py --trade --interval 30
        """
    )
    
    parser.add_argument(
        '--trade',
        action='store_true',
        help='Start continuous trading daemon'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run single test scan without trading'
    )
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show current status'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=60,
        help='Scan interval in seconds (default: 60)'
    )
    parser.add_argument(
        '--live',
        action='store_true',
        help='Enable live trading (not paper)'
    )
    parser.add_argument(
        '--symbols',
        type=str,
        help='Comma-separated symbols'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        setup_logging(logging.DEBUG)
    
    # Build config
    config = EngineConfig()
    config.scan_interval_sec = args.interval
    
    if args.live:
        config.paper_trading = False
        logger.warning("⚠️  LIVE TRADING ENABLED")
    
    if args.symbols:
        config.symbols = [s.strip().upper() for s in args.symbols.split(',')]
    
    # Initialize engine
    engine = HighFreqOptionsEngine(config)
    
    # Handle shutdown signals
    def signal_handler(sig, frame):
        logger.info("\n🛑 Shutdown signal received")
        engine.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Execute command
    if args.status:
        engine.print_status()
        
    elif args.test:
        results = asyncio.run(engine.run_test())
        
        print("\n" + "=" * 60)
        print("TEST SCAN RESULTS")
        print("=" * 60)
        print(f"Symbols Scanned: {results['symbols_scanned']}")
        print(f"Opportunities: {len(results['opportunities'])}")
        print()
        
        for opp in results['opportunities']:
            emoji = "📈" if opp['direction'] == 'buy' else "📉"
            print(f"{emoji} {opp['symbol']:6} {opp['type']:4} {opp['direction']:4} "
                  f"| Strength: {opp['strength']:.2f}")
            print(f"   → {opp['rationale']}")
        
        print("=" * 60)
        
    elif args.trade:
        logger.info("🚀 Starting trading daemon...")
        logger.info(f"📊 Scan interval: {config.scan_interval_sec}s")
        logger.info(f"📈 Symbols: {', '.join(config.symbols)}")
        logger.info(f"⚠️  Max positions: {config.risk.max_positions}")
        logger.info(f"⚠️  Max loss/trade: {config.risk.max_loss_per_trade_pct:.1%}")
        logger.info(f"⚠️  Max daily loss: {config.risk.max_daily_loss_pct:.1%}")
        
        if not config.paper_trading:
            logger.warning("=" * 60)
            logger.warning("🔴 LIVE TRADING MODE - REAL MONEY AT RISK 🔴")
            logger.warning("=" * 60)
        
        asyncio.run(engine.run_continuous())
        
    else:
        engine.print_status()
        print("\nUse --help for available commands")


if __name__ == '__main__':
    main()
