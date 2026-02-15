#!/usr/bin/env python3
"""
PROFIT TRADER - Aggressive Momentum + Mean Reversion Stock Trader
=================================================================
Designed to actively trade and generate profit on Alpaca Paper account.

Strategies:
1. MOMENTUM BREAKOUT: Buy stocks breaking out on high relative volume
2. MEAN REVERSION: Buy oversold bounces on quality names
3. VWAP RECLAIM: Buy when price reclaims VWAP from below
4. TREND FOLLOWING: Ride strong intraday trends with trailing stops

Risk Management:
- Max 10 simultaneous positions
- 8-12% of portfolio per position
- 1.5% hard stop loss per trade
- 3% trailing stop on winners
- Take profit at 2-4% gain
- Daily loss limit: 3% of portfolio
"""

import os
import sys
import json
import time
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# ── Setup ──────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('profit_trader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ProfitTrader')

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False
    logger.warning("yfinance not available - using Alpaca data only")

# OVERHAUL FIX #6: HMM Regime Filter
try:
    from src.trading.regime_ensemble import EnsembleRegimeDetector, RegimeType, RegimeState
    HAS_REGIME = True
except ImportError:
    HAS_REGIME = False
    logger.warning("Regime ensemble not available - trading without regime filter")

# OVERHAUL FIX #9: Options Wheel Strategy
try:
    from src.alpaca_options_engine import AlpacaOptionsEngine, OptionContract
    HAS_OPTIONS = True
except ImportError:
    HAS_OPTIONS = False
    logger.warning("Options engine not available - wheel strategy disabled")

# OVERHAUL FIX #11: TDA Signal Confirmation
try:
    from src.tda_features import TDAFeatureGenerator
    HAS_TDA = True
except ImportError:
    HAS_TDA = False
    logger.warning("TDA features not available - trading without TDA confirmation")

# OVERHAUL FIX #14: Health Monitoring + Discord Alerts
try:
    from src.health_monitor import HealthMonitor, HealthCheckResult, HealthStatus, ComponentType
    HAS_HEALTH = True
except ImportError:
    HAS_HEALTH = False
    logger.warning("Health monitor not available - running without health checks")

# ── Configuration ──────────────────────────────────────────────────────────────

@dataclass
class TraderConfig:
    # Universe - high-beta liquid stocks for maximum profit opportunity
    universe: List[str] = field(default_factory=lambda: [
        # Mega-cap tech (liquid, moves with conviction)
        'NVDA', 'TSLA', 'META', 'AAPL', 'AMZN', 'MSFT', 'GOOG', 'AMD', 'NFLX', 'AVGO',
        # High-beta growth (bigger swings = more opportunity)
        'PLTR', 'COIN', 'MSTR', 'ARM', 'SMCI', 'HOOD', 'SOFI',
        'SNOW', 'CRM', 'PANW', 'SHOP', 'ROKU', 'UBER',
        # Crypto-adjacent (volatile)
        'MARA', 'RIOT',
        # ETFs for trend trades
        'SPY', 'QQQ', 'TQQQ', 'SOXL',
    ])

    # Position sizing
    max_positions: int = 10
    position_pct: float = 0.10         # 10% of portfolio per trade
    max_single_position: float = 0.15  # Never more than 15% in one name

    # Entry thresholds
    momentum_threshold: float = 0.015   # 1.5% move to trigger momentum entry
    mean_rev_threshold: float = -0.025  # -2.5% drop for mean reversion entry
    vwap_reclaim_pct: float = 0.003     # 0.3% above VWAP to confirm reclaim
    min_volume_ratio: float = 1.3       # Volume must be 1.3x average

    # Exit thresholds
    stop_loss_pct: float = 0.015       # 1.5% hard stop
    trailing_stop_pct: float = 0.025    # 2.5% trailing stop
    take_profit_pct: float = 0.03      # 3% take profit
    quick_scalp_pct: float = 0.015     # 1.5% quick scalp target
    time_stop_minutes: int = 120       # Exit after 2 hours if flat
    swing_mode: bool = True            # OVERHAUL FIX #7: Hold positions overnight (no EOD close)

    # Risk limits
    daily_loss_limit_pct: float = 0.03  # 3% daily loss limit
    max_correlation_positions: int = 3   # Max 3 in same "group"
    min_cash_pct: float = 0.15          # OVERHAUL FIX #5: Keep 15% cash buffer
    kelly_fraction: float = 0.5         # OVERHAUL FIX #8: Half-Kelly (conservative)
    min_position_pct: float = 0.03      # Minimum position size (3% of equity)
    max_position_pct: float = 0.12      # Maximum even if Kelly says more

    # Scan interval
    scan_interval_seconds: int = 120    # Scan every 2 minutes
    
    # Market hours (ET)
    market_open_hour: int = 9
    market_open_minute: int = 30
    market_close_hour: int = 16
    market_close_minute: int = 0
    
    # Strategy weights (probability of using each)
    momentum_weight: float = 0.40
    mean_rev_weight: float = 0.25
    vwap_weight: float = 0.20
    trend_weight: float = 0.15

    # OVERHAUL FIX #10: Team of Rivals veto
    team_veto_enabled: bool = True
    team_min_approvals: int = 4          # Need 4/6 checks to pass

    # OVERHAUL FIX #11: TDA confirmation
    tda_enabled: bool = True
    tda_window: int = 30                 # Rolling window for TDA features
    tda_turbulence_veto: float = 0.80    # Veto momentum/trend if turbulence > 80th pctl
    tda_betti1_min: float = 2.0          # Min betti_1 for mean-reversion (needs structure)

    # OVERHAUL FIX #12: ML Ensemble confirmation
    ml_ensemble_enabled: bool = True
    ml_min_samples: int = 30             # Min completed trades before ML starts voting
    ml_retrain_every: int = 10           # Retrain after every N new outcomes
    ml_veto_threshold: float = 0.45      # Veto if ML probability < 45%

    # OVERHAUL FIX #13: State persistence (crash recovery)
    state_file: str = 'state/trader_state.json'
    state_save_interval: int = 5         # Save state every N scan cycles

    # OVERHAUL FIX #14: Health monitoring + Discord alerts
    health_enabled: bool = True
    health_check_interval: int = 60      # Seconds between background health checks
    health_heartbeat_interval: int = 120 # Seconds between heartbeats
    drawdown_alert_pct: float = 0.03     # Alert on 3% drawdown
    drawdown_halt_pct: float = 0.05      # Halt on 5% drawdown
    consecutive_api_fail_limit: int = 5  # Alert after 5 consecutive API failures

    # OVERHAUL FIX #15: Trade journal (CSV audit trail)
    journal_enabled: bool = True
    journal_file: str = 'state/trade_journal.csv'

    # OVERHAUL FIX #16: EOD performance report + Discord summary
    eod_report_enabled: bool = True
    eod_report_file: str = 'state/daily_reports.jsonl'

    # OVERHAUL FIX #9: Wheel Strategy (cash-secured puts → covered calls)
    wheel_enabled: bool = True
    wheel_tickers: List[str] = field(default_factory=lambda: ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA'])
    wheel_dte_min: int = 25            # Min days to expiration
    wheel_dte_max: int = 45            # Max days to expiration
    wheel_otm_pct: float = 0.05        # Target ~5% OTM (≈0.30 delta)
    wheel_min_premium: float = 0.50    # Minimum $0.50 premium per contract
    wheel_max_contracts: int = 2        # Max contracts per underlying
    wheel_capital_pct: float = 0.30    # Max 30% of equity for wheel
    wheel_run_hour: int = 10           # Run wheel scan once/day at 10:xx ET


# OVERHAUL FIX #4: Sector groupings for correlation-based position caps
SECTOR_MAP = {
    'crypto_adjacent':  ['MARA', 'RIOT', 'COIN', 'MSTR', 'HOOD'],
    'mega_tech':        ['NVDA', 'AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'NFLX'],
    'high_beta_growth': ['PLTR', 'TSLA', 'ARM', 'SMCI', 'SOFI', 'SNOW', 'CRM', 'PANW', 'SHOP', 'ROKU', 'UBER'],
    'semis':            ['AMD', 'AVGO', 'SOXL'],
    'etfs':             ['SPY', 'QQQ', 'TQQQ'],
}


# ── Alpaca Client ──────────────────────────────────────────────────────────────

class AlpacaClient:
    """Direct REST client for Alpaca - no SDK dependency issues."""

    def __init__(self):
        self.key = os.getenv('APCA_API_KEY_ID') or os.getenv('ALPACA_API_KEY')
        self.secret = os.getenv('APCA_API_SECRET_KEY') or os.getenv('ALPACA_SECRET_KEY')
        self.base = 'https://paper-api.alpaca.markets/v2'
        self.data_base = 'https://data.alpaca.markets/v2'

        if not self.key or not self.secret:
            raise ValueError("Missing Alpaca API keys! Set APCA_API_KEY_ID and APCA_API_SECRET_KEY")

        self.headers = {
            'APCA-API-KEY-ID': self.key,
            'APCA-API-SECRET-KEY': self.secret,
            'Content-Type': 'application/json'
        }
        # Verify connection
        acct = self.get_account()
        logger.info(f"Connected to Alpaca | Portfolio: ${float(acct['equity']):,.2f} | Cash: ${float(acct['cash']):,.2f}")

    def get_account(self) -> dict:
        r = requests.get(f'{self.base}/account', headers=self.headers, timeout=10)
        r.raise_for_status()
        return r.json()

    def get_positions(self) -> List[dict]:
        r = requests.get(f'{self.base}/positions', headers=self.headers, timeout=10)
        r.raise_for_status()
        return r.json()

    def get_position(self, symbol: str) -> Optional[dict]:
        r = requests.get(f'{self.base}/positions/{symbol}', headers=self.headers, timeout=10)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()

    # OVERHAUL FIX #2: Default to limit orders to reduce slippage
    def place_order(self, symbol: str, qty: int, side: str, order_type: str = 'limit',
                    time_in_force: str = 'day', limit_price: float = None,
                    stop_price: float = None, trail_percent: float = None) -> Optional[dict]:
        """Place an order. Limit orders auto-compute price from current quote."""

        # Auto-compute limit price from quote if not provided
        if order_type == 'limit' and limit_price is None:
            snapshot = self.get_snapshot(symbol)
            if snapshot and 'latestQuote' in snapshot:
                quote = snapshot['latestQuote']
                if side == 'buy':
                    ask = float(quote.get('ap', 0))
                    if ask > 0:
                        # Pad: +$0.02 for <$100, +0.05% for >=$100
                        limit_price = round(ask + (0.02 if ask < 100 else ask * 0.0005), 2)
                        logger.debug(f"Limit price for {symbol}: ask=${ask:.2f} -> limit=${limit_price:.2f}")
                    else:
                        order_type = 'market'  # no ask -> fall back
                elif side == 'sell':
                    bid = float(quote.get('bp', 0))
                    if bid > 0:
                        limit_price = round(bid - (0.02 if bid < 100 else bid * 0.0005), 2)
                        logger.debug(f"Limit price for {symbol}: bid=${bid:.2f} -> limit=${limit_price:.2f}")
                    else:
                        order_type = 'market'
            else:
                logger.warning(f"No quote for {symbol}, falling back to market order")
                order_type = 'market'

        data = {
            'symbol': symbol,
            'qty': str(qty),
            'side': side,
            'type': order_type,
            'time_in_force': time_in_force,
        }
        if limit_price:
            data['limit_price'] = str(round(limit_price, 2))
        if stop_price:
            data['stop_price'] = str(round(stop_price, 2))
        if trail_percent:
            data['trail_percent'] = str(round(trail_percent, 2))

        try:
            r = requests.post(f'{self.base}/orders', headers=self.headers, json=data, timeout=10)
            if r.status_code in [200, 201]:
                order = r.json()
                logger.info(f"ORDER PLACED: {side.upper()} {qty} {symbol} @ {order_type} | ID: {order['id'][:8]}")
                return order
            else:
                logger.error(f"Order failed for {symbol}: {r.status_code} - {r.text[:200]}")
                return None
        except Exception as e:
            logger.error(f"Order exception for {symbol}: {e}")
            return None

    def close_position(self, symbol: str) -> Optional[dict]:
        """Close entire position in a symbol."""
        r = requests.delete(f'{self.base}/positions/{symbol}', headers=self.headers, timeout=10)
        if r.status_code == 200:
            logger.info(f"CLOSED position: {symbol}")
            return r.json()
        else:
            logger.warning(f"Failed to close {symbol}: {r.status_code}")
            return None

    def get_orders(self, status='open') -> List[dict]:
        r = requests.get(f'{self.base}/orders?status={status}&limit=50', headers=self.headers, timeout=10)
        r.raise_for_status()
        return r.json()

    def cancel_all_orders(self):
        r = requests.delete(f'{self.base}/orders', headers=self.headers, timeout=10)
        logger.info(f"Cancelled all open orders: {r.status_code}")

    # OVERHAUL FIX #2: Helpers for limit order fill monitoring
    def get_order(self, order_id: str) -> Optional[dict]:
        """Get order status by ID."""
        try:
            r = requests.get(f'{self.base}/orders/{order_id}', headers=self.headers, timeout=10)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a single order by ID."""
        try:
            r = requests.delete(f'{self.base}/orders/{order_id}', headers=self.headers, timeout=10)
            return r.status_code in [200, 204]
        except Exception:
            return False

    def get_bars(self, symbol: str, timeframe: str = '5Min', limit: int = 100) -> List[dict]:
        """Get bars from Alpaca data API."""
        params = {'timeframe': timeframe, 'limit': limit, 'feed': 'iex'}
        r = requests.get(f'{self.data_base}/stocks/{symbol}/bars',
                        headers=self.headers, params=params, timeout=10)
        if r.status_code == 200:
            return r.json().get('bars', [])
        return []

    def get_snapshot(self, symbol: str) -> Optional[dict]:
        """Get latest snapshot (quote + trade + bar)."""
        r = requests.get(f'{self.data_base}/stocks/{symbol}/snapshot',
                        headers=self.headers, params={'feed': 'iex'}, timeout=10)
        if r.status_code == 200:
            return r.json()
        return None


# ── Market Data ────────────────────────────────────────────────────────────────

class MarketData:
    """Fetch and analyze market data for trading signals."""

    # OVERHAUL FIX #1: Primary data from Alpaca API, yfinance fallback only
    def __init__(self, client: 'AlpacaClient' = None):
        self.client = client

    def get_stock_data(self, ticker: str, period: str = '5d', interval: str = '5m') -> Optional[pd.DataFrame]:
        """Get OHLCV data — Alpaca primary, yfinance fallback."""

        # ── Alpaca primary path (no rate limits, official API) ──
        if self.client is not None:
            try:
                # Map period/interval to Alpaca params
                tf_map = {'1m': '1Min', '5m': '5Min', '15m': '15Min', '1h': '1Hour', '1d': '1Day'}
                timeframe = tf_map.get(interval, '5Min')
                limit_map = {'1d': 78, '2d': 156, '5d': 390, '10d': 780}
                limit = limit_map.get(period, 390)

                bars = self.client.get_bars(ticker, timeframe=timeframe, limit=limit)
                if bars and len(bars) > 0:
                    df = pd.DataFrame(bars)
                    # Alpaca bar keys: t, o, h, l, c, v, n, vw
                    rename = {'t': 'timestamp', 'o': 'Open', 'h': 'High',
                              'l': 'Low', 'c': 'Close', 'v': 'Volume'}
                    df = df.rename(columns=rename)
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df = df.set_index('timestamp')
                    # Keep only OHLCV columns
                    ohlcv_cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
                    df = df[ohlcv_cols]
                    if len(df) >= 10:
                        logger.debug(f"Alpaca data OK for {ticker}: {len(df)} bars")
                        return df
            except Exception as e:
                logger.debug(f"Alpaca data failed for {ticker}: {e}")

        # ── yfinance fallback ──
        if not HAS_YF:
            return None
        try:
            data = yf.download(ticker, period=period, interval=interval, progress=False, timeout=10)
            if data.empty:
                return None
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            logger.debug(f"yfinance fallback used for {ticker}")
            return data
        except Exception as e:
            logger.debug(f"yfinance fallback failed for {ticker}: {e}")
            return None

    def analyze(self, ticker: str) -> Optional[Dict]:
        """Full technical analysis for a ticker. Returns signals dict."""
        df = self.get_stock_data(ticker, period='5d', interval='5m')
        if df is None or len(df) < 30:
            return None

        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        volume = df['Volume'].values

        current = float(close[-1])
        prev_close = float(close[-2]) if len(close) > 1 else current

        # ── Momentum metrics ──
        ret_5m = (close[-1] / close[-2] - 1) if len(close) > 1 else 0
        ret_15m = (close[-1] / close[-4] - 1) if len(close) > 3 else 0
        ret_30m = (close[-1] / close[-7] - 1) if len(close) > 6 else 0
        ret_1h = (close[-1] / close[-13] - 1) if len(close) > 12 else 0
        ret_2h = (close[-1] / close[-25] - 1) if len(close) > 24 else 0

        # Today's OHLC (last ~78 bars for 5-min intervals in a trading day)
        today_bars = min(len(close), 78)
        today_close = close[-today_bars:]
        today_high = high[-today_bars:]
        today_low = low[-today_bars:]
        today_volume = volume[-today_bars:]

        today_open = float(today_close[0])
        today_return = (current / today_open - 1) if today_open > 0 else 0

        # ── VWAP ──
        typical_price = (today_high + today_low + today_close) / 3
        cum_vol = np.cumsum(today_volume)
        cum_tp_vol = np.cumsum(typical_price * today_volume)
        vwap = cum_tp_vol[-1] / cum_vol[-1] if cum_vol[-1] > 0 else current
        vwap_distance = (current - vwap) / vwap

        # ── Moving averages ──
        sma_20 = np.mean(close[-20:]) if len(close) >= 20 else current
        sma_50 = np.mean(close[-50:]) if len(close) >= 50 else current
        ema_9 = self._ema(close, 9)
        ema_21 = self._ema(close, 21)

        # ── RSI (14-period) ──
        rsi = self._rsi(close, 14)

        # ── MACD ──
        ema_12 = self._ema(close, 12)
        ema_26 = self._ema(close, 26)
        macd = ema_12 - ema_26
        signal_line = self._ema(np.array([macd]), 9) if isinstance(macd, (int, float)) else macd
        macd_histogram = macd - signal_line

        # ── Bollinger Bands ──
        bb_sma = np.mean(close[-20:]) if len(close) >= 20 else current
        bb_std = np.std(close[-20:]) if len(close) >= 20 else 0
        bb_upper = bb_sma + 2 * bb_std
        bb_lower = bb_sma - 2 * bb_std
        bb_position = (current - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5

        # ── Volume analysis ──
        vol_avg = np.mean(volume[-50:]) if len(volume) >= 50 else np.mean(volume)
        vol_current = float(volume[-1])
        vol_ratio = vol_current / vol_avg if vol_avg > 0 else 1.0
        vol_today_avg = np.mean(today_volume) if len(today_volume) > 0 else 0
        vol_surge = vol_current / vol_today_avg if vol_today_avg > 0 else 1.0

        # ── ATR (for stop sizing) ──
        atr = self._atr(high, low, close, 14)

        # ── Price action ──
        # Higher highs / higher lows check (last 5 bars)
        recent_highs = high[-5:]
        recent_lows = low[-5:]
        higher_highs = all(recent_highs[i] >= recent_highs[i-1] for i in range(1, len(recent_highs)))
        higher_lows = all(recent_lows[i] >= recent_lows[i-1] for i in range(1, len(recent_lows)))
        uptrend = higher_highs and higher_lows

        # ── Day's range position ──
        day_high = float(np.max(today_high))
        day_low = float(np.min(today_low))
        day_range = day_high - day_low
        range_position = (current - day_low) / day_range if day_range > 0 else 0.5

        return {
            'ticker': ticker,
            'price': current,
            'today_return': today_return,
            'ret_5m': ret_5m,
            'ret_15m': ret_15m,
            'ret_30m': ret_30m,
            'ret_1h': ret_1h,
            'ret_2h': ret_2h,
            'vwap': vwap,
            'vwap_distance': vwap_distance,
            'sma_20': sma_20,
            'sma_50': sma_50,
            'ema_9': ema_9,
            'ema_21': ema_21,
            'rsi': rsi,
            'macd': macd,
            'macd_histogram': macd_histogram,
            'bb_position': bb_position,
            'bb_upper': bb_upper,
            'bb_lower': bb_lower,
            'vol_ratio': vol_ratio,
            'vol_surge': vol_surge,
            'atr': atr,
            'uptrend': uptrend,
            'range_position': range_position,
            'day_high': day_high,
            'day_low': day_low,
        }

    def _ema(self, data: np.ndarray, period: int) -> float:
        if len(data) < period:
            return float(data[-1])
        multiplier = 2 / (period + 1)
        ema = float(data[-period])
        for i in range(-period + 1, 0):
            ema = (data[i] - ema) * multiplier + ema
        return ema

    def _rsi(self, data: np.ndarray, period: int = 14) -> float:
        if len(data) < period + 1:
            return 50.0
        deltas = np.diff(data[-(period+1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        if len(high) < period + 1:
            return float(high[-1] - low[-1])
        trs = []
        for i in range(-period, 0):
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            trs.append(tr)
        return float(np.mean(trs))


# ── Signal Generator ───────────────────────────────────────────────────────────

@dataclass
class Signal:
    ticker: str
    strategy: str       # 'momentum', 'mean_rev', 'vwap', 'trend'
    side: str           # 'buy' or 'sell'
    confidence: float   # 0-1
    price: float
    stop_loss: float
    take_profit: float
    reason: str
    atr: float = 0.0    # OVERHAUL FIX #3: ATR for dynamic stop sizing

class SignalGenerator:
    """Generate trading signals from market data analysis."""

    def __init__(self, config: TraderConfig):
        self.config = config

    def generate(self, analysis: Dict) -> Optional[Signal]:
        """Generate a trading signal from analysis data."""
        signals = []

        # Try each strategy
        s = self._check_momentum(analysis)
        if s:
            signals.append(s)
        s = self._check_mean_reversion(analysis)
        if s:
            signals.append(s)
        s = self._check_vwap_reclaim(analysis)
        if s:
            signals.append(s)
        s = self._check_trend(analysis)
        if s:
            signals.append(s)

        if not signals:
            return None

        # Return highest confidence signal
        return max(signals, key=lambda x: x.confidence)

    def _check_momentum(self, a: Dict) -> Optional[Signal]:
        """Momentum breakout: strong move + volume confirmation."""
        price = a['price']
        
        # Need strong recent move
        if a['ret_30m'] < self.config.momentum_threshold:
            return None
        
        # Volume must confirm
        if a['vol_ratio'] < self.config.min_volume_ratio:
            return None
        
        # RSI not overbought yet (room to run)
        if a['rsi'] > 78:
            return None
        
        # Price above VWAP (buying into strength)
        if a['vwap_distance'] < 0:
            return None

        # Confidence based on multiple factors
        conf = 0.5
        if a['ret_1h'] > 0.02:  # Strong 1h momentum
            conf += 0.15
        if a['vol_surge'] > 2.0:  # Volume spike
            conf += 0.1
        if a['uptrend']:  # Price action confirms
            conf += 0.1
        if a['ema_9'] > a['ema_21']:  # EMA alignment
            conf += 0.1
        if a['range_position'] > 0.7:  # Near day high (breakout)
            conf += 0.05

        # OVERHAUL FIX #3: ATR-based stop (2.5x ATR for momentum)
        atr = a['atr']
        stop = price - (2.5 * atr) if atr > 0 else price * (1 - self.config.stop_loss_pct)
        target = price + (3.0 * atr) if atr > 0 else price * (1 + self.config.take_profit_pct)

        return Signal(
            ticker=a['ticker'], strategy='momentum', side='buy',
            confidence=min(conf, 0.95), price=price,
            stop_loss=stop, take_profit=target,
            reason=f"Momentum breakout: 30m={a['ret_30m']*100:+.1f}%, vol={a['vol_ratio']:.1f}x, RSI={a['rsi']:.0f}, ATR=${atr:.2f}",
            atr=atr
        )

    def _check_mean_reversion(self, a: Dict) -> Optional[Signal]:
        """Mean reversion: oversold bounce on quality name."""
        price = a['price']
        
        # Need significant drop
        if a['ret_2h'] > self.config.mean_rev_threshold:
            return None
        
        # But showing signs of recovery (5m positive)
        if a['ret_5m'] < 0:
            return None
        
        # RSI oversold
        if a['rsi'] > 40:
            return None
        
        # Price near Bollinger lower band
        if a['bb_position'] > 0.3:
            return None

        conf = 0.5
        if a['rsi'] < 30:
            conf += 0.15
        if a['bb_position'] < 0.1:
            conf += 0.1
        if a['ret_5m'] > 0.003:  # Strong bounce
            conf += 0.1
        if a['vwap_distance'] < -0.01:  # Well below VWAP (snap-back likely)
            conf += 0.1

        # OVERHAUL FIX #3: ATR-based stop (1.5x ATR for mean reversion — tighter)
        atr = a['atr']
        stop = price - (1.5 * atr) if atr > 0 else price * (1 - self.config.stop_loss_pct)
        target = price + (2.0 * atr) if atr > 0 else price * (1 + self.config.quick_scalp_pct)

        return Signal(
            ticker=a['ticker'], strategy='mean_rev', side='buy',
            confidence=min(conf, 0.95), price=price,
            stop_loss=stop, take_profit=target,
            reason=f"Mean reversion: 2h={a['ret_2h']*100:+.1f}%, RSI={a['rsi']:.0f}, BB={a['bb_position']:.2f}, ATR=${atr:.2f}",
            atr=atr
        )

    def _check_vwap_reclaim(self, a: Dict) -> Optional[Signal]:
        """VWAP reclaim: price crosses above VWAP with volume."""
        price = a['price']
        
        # Price just crossed above VWAP
        if a['vwap_distance'] < 0 or a['vwap_distance'] > 0.01:
            return None  # Too far below or already extended above
        
        # Positive recent momentum
        if a['ret_15m'] < 0:
            return None
        
        # Volume confirmation
        if a['vol_ratio'] < 1.1:
            return None

        conf = 0.5
        if a['ret_5m'] > 0.002:
            conf += 0.1
        if a['vol_surge'] > 1.5:
            conf += 0.1
        if a['rsi'] > 45 and a['rsi'] < 65:  # Neutral RSI = room to run
            conf += 0.1
        if a['ema_9'] > a['ema_21']:
            conf += 0.1

        # OVERHAUL FIX #3: ATR-based stop (2.0x ATR for VWAP, floored at VWAP -0.2%)
        atr = a['atr']
        atr_stop = price - (2.0 * atr) if atr > 0 else price * (1 - self.config.stop_loss_pct)
        stop = min(atr_stop, a['vwap'] * 0.998)
        target = price + (2.5 * atr) if atr > 0 else price * (1 + self.config.take_profit_pct)

        return Signal(
            ticker=a['ticker'], strategy='vwap', side='buy',
            confidence=min(conf, 0.95), price=price,
            stop_loss=stop, take_profit=target,
            reason=f"VWAP reclaim: dist={a['vwap_distance']*100:+.2f}%, 15m={a['ret_15m']*100:+.1f}%, vol={a['vol_ratio']:.1f}x, ATR=${atr:.2f}",
            atr=atr
        )

    def _check_trend(self, a: Dict) -> Optional[Signal]:
        """Trend following: strong intraday trend continuation."""
        price = a['price']
        
        # Need established trend
        if a['today_return'] < 0.01:  # At least +1% on the day
            return None
        
        # EMA alignment (9 > 21 > 50 for strong trend)
        if not (a['ema_9'] > a['ema_21']):
            return None
        
        # Not too extended (still has room)
        if a['rsi'] > 75:
            return None
        
        # Above VWAP
        if a['vwap_distance'] < 0:
            return None

        conf = 0.45
        if a['uptrend']:
            conf += 0.15
        if a['today_return'] > 0.025:
            conf += 0.1
        if a['vol_ratio'] > 1.2:
            conf += 0.1
        if a['range_position'] > 0.6:
            conf += 0.1
        if a['macd_histogram'] > 0:
            conf += 0.05

        # OVERHAUL FIX #3: ATR-based stop (3.0x ATR for trends — widest, let winners run)
        atr = a['atr']
        stop = price - (3.0 * atr) if atr > 0 else price * (1 - self.config.stop_loss_pct * 1.5)
        target = price + (4.0 * atr) if atr > 0 else price * (1 + self.config.take_profit_pct * 1.5)

        return Signal(
            ticker=a['ticker'], strategy='trend', side='buy',
            confidence=min(conf, 0.95), price=price,
            stop_loss=stop, take_profit=target,
            reason=f"Trend follow: day={a['today_return']*100:+.1f}%, range_pos={a['range_position']:.2f}, RSI={a['rsi']:.0f}, ATR=${atr:.2f}",
            atr=atr
        )


# ── Position Manager ───────────────────────────────────────────────────────────

@dataclass
class TrackedPosition:
    symbol: str
    entry_price: float
    qty: int
    side: str
    strategy: str
    entry_time: datetime
    stop_loss: float
    take_profit: float
    trailing_stop: float
    high_water: float      # Highest price since entry (for trailing stop)

class PositionManager:
    """Manage open positions with active stop/target monitoring."""

    def __init__(self, config: TraderConfig, client: AlpacaClient):
        self.config = config
        self.client = client
        self.tracked: Dict[str, TrackedPosition] = {}
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.wins_today = 0
        self.losses_today = 0
        # Auto-sync on startup
        self._initial_sync()

    def _initial_sync(self):
        """Load existing Alpaca positions into tracking on startup."""
        try:
            alpaca_positions = self.client.get_positions()
            for p in alpaca_positions:
                sym = p['symbol']
                # Skip options (contain numbers in symbol beyond normal tickers)
                if len(sym) > 10:
                    continue
                entry = float(p['avg_entry_price'])
                qty = int(float(p['qty']))
                self.tracked[sym] = TrackedPosition(
                    symbol=sym,
                    entry_price=entry,
                    qty=abs(qty),
                    side='buy' if qty > 0 else 'sell',
                    strategy='synced',
                    entry_time=datetime.now(timezone.utc) - timedelta(hours=1),  # Assume recent
                    stop_loss=entry * (1 - self.config.stop_loss_pct),
                    take_profit=entry * (1 + self.config.take_profit_pct),
                    trailing_stop=entry * (1 - self.config.trailing_stop_pct),
                    high_water=float(p.get('current_price', entry)),
                )
            if self.tracked:
                logger.info(f"Synced {len(self.tracked)} existing positions from Alpaca: {list(self.tracked.keys())}")
        except Exception as e:
            logger.warning(f"Initial position sync failed: {e}")

    def sync_positions(self):
        """Sync tracked positions with what Alpaca shows."""
        alpaca_positions = {p['symbol']: p for p in self.client.get_positions()
                          if len(p['symbol']) <= 10}  # Skip options
        
        # Remove tracked positions that are no longer in Alpaca
        for sym in list(self.tracked.keys()):
            if sym not in alpaca_positions:
                logger.info(f"Position {sym} no longer in Alpaca - removing from tracking")
                del self.tracked[sym]
        
        # Add any Alpaca positions we're not tracking yet
        for sym, p in alpaca_positions.items():
            if sym not in self.tracked:
                entry = float(p['avg_entry_price'])
                qty = int(float(p['qty']))
                self.tracked[sym] = TrackedPosition(
                    symbol=sym,
                    entry_price=entry,
                    qty=abs(qty),
                    side='buy' if qty > 0 else 'sell',
                    strategy='synced',
                    entry_time=datetime.now(timezone.utc),
                    stop_loss=entry * (1 - self.config.stop_loss_pct),
                    take_profit=entry * (1 + self.config.take_profit_pct),
                    trailing_stop=entry * (1 - self.config.trailing_stop_pct),
                    high_water=float(p.get('current_price', entry)),
                )
                logger.info(f"Added untracked position {sym} to monitoring")

    def add_position(self, signal: Signal, qty: int):
        """Track a new position."""
        self.tracked[signal.ticker] = TrackedPosition(
            symbol=signal.ticker,
            entry_price=signal.price,
            qty=qty,
            side=signal.side,
            strategy=signal.strategy,
            entry_time=datetime.now(timezone.utc),
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            trailing_stop=signal.price * (1 - self.config.trailing_stop_pct),
            high_water=signal.price,
        )
        self.trades_today += 1

    def check_exits(self, market_data: MarketData) -> List[Tuple[str, str]]:
        """Check all positions for exit conditions. Returns list of (symbol, reason)."""
        exits = []
        now = datetime.now(timezone.utc)

        for sym, pos in list(self.tracked.items()):
            df = market_data.get_stock_data(sym, period='1d', interval='1m')
            if df is None or len(df) == 0:
                continue

            current_price = float(df['Close'].iloc[-1])

            # Update high water mark
            if current_price > pos.high_water:
                pos.high_water = current_price
                # Ratchet trailing stop up
                new_trail = current_price * (1 - self.config.trailing_stop_pct)
                if new_trail > pos.trailing_stop:
                    pos.trailing_stop = new_trail

            pnl_pct = (current_price / pos.entry_price - 1)

            # 1. Hard stop loss
            if current_price <= pos.stop_loss:
                exits.append((sym, f"STOP LOSS hit @ ${current_price:.2f} (loss: {pnl_pct*100:+.1f}%)"))
                self.daily_pnl += pnl_pct * pos.entry_price * pos.qty
                self.losses_today += 1
                continue

            # 2. Take profit
            if current_price >= pos.take_profit:
                exits.append((sym, f"TAKE PROFIT hit @ ${current_price:.2f} (gain: {pnl_pct*100:+.1f}%)"))
                self.daily_pnl += pnl_pct * pos.entry_price * pos.qty
                self.wins_today += 1
                continue

            # 3. Trailing stop (only after some profit)
            if pnl_pct > 0.005 and current_price <= pos.trailing_stop:
                exits.append((sym, f"TRAILING STOP hit @ ${current_price:.2f} (gain: {pnl_pct*100:+.1f}%, peak: ${pos.high_water:.2f})"))
                self.daily_pnl += pnl_pct * pos.entry_price * pos.qty
                self.wins_today += 1
                continue

            # 4. Time stop (exit if flat after too long)
            age_minutes = (now - pos.entry_time).total_seconds() / 60
            if age_minutes > self.config.time_stop_minutes and abs(pnl_pct) < 0.005:
                exits.append((sym, f"TIME STOP after {age_minutes:.0f}min (flat: {pnl_pct*100:+.1f}%)"))
                continue

        return exits

    @property
    def position_count(self) -> int:
        return len(self.tracked)

    @property
    def can_open_new(self) -> bool:
        return self.position_count < self.config.max_positions


# ── TDA Signal Confirmation ────────────────────────────────────────────────────

class TDAConfirm:
    """OVERHAUL FIX #11: Use persistent homology to confirm/veto trade signals.

    - High turbulence (chaotic topology) → veto momentum/trend entries
    - Low Betti-1 (no loops/cycles) → veto mean-reversion (no structure to revert to)
    - High entropy → confidence penalty (uncertain regime)
    """

    def __init__(self, config: TraderConfig):
        self.config = config
        self.generator = TDAFeatureGenerator(window=config.tda_window, feature_mode='v1.2')
        # Cache: ticker -> (timestamp, features_dict)
        self._cache: Dict[str, Tuple[float, dict]] = {}
        self._cache_ttl = 300  # 5 min cache

    def compute_features(self, ticker: str, close_prices: np.ndarray) -> Optional[dict]:
        """Compute TDA features from close prices with caching."""
        now = time.time()
        if ticker in self._cache:
            ts, feats = self._cache[ticker]
            if now - ts < self._cache_ttl:
                return feats

        if len(close_prices) < self.config.tda_window + 5:
            return None
        try:
            log_prices = np.log(close_prices + 1e-10)
            returns = np.diff(log_prices)
            feats = self.generator.compute_persistence_features(returns[-self.config.tda_window:])
            self._cache[ticker] = (now, feats)
            return feats
        except Exception as e:
            logger.debug(f"TDA compute failed for {ticker}: {e}")
            return None

    def confirm(self, signal: 'Signal', close_prices: np.ndarray) -> Tuple[bool, str]:
        """Check TDA topology against the signal strategy.

        Returns (approved, reason).
        """
        feats = self.compute_features(signal.ticker, close_prices)
        if feats is None:
            return True, "TDA:no-data"  # approve if can't compute

        # ── Turbulence proxy: sqrt(persistence_l0² + persistence_l1²) ──
        p0 = feats.get('persistence_l0', 0)
        p1 = feats.get('persistence_l1', 0)
        turbulence = np.sqrt(p0**2 + p1**2)

        # ── Betti-1 (loop/cycle count) ──
        betti1 = feats.get('betti_1', 0)

        # ── Entropy (topological uncertainty) ──
        e0 = feats.get('entropy_l0', 0)
        e1 = feats.get('entropy_l1', 0)
        entropy = (e0 + e1) / 2.0

        # Rule 1: High turbulence → veto momentum/trend (market too chaotic)
        if signal.strategy in ('momentum', 'trend') and turbulence > self.config.tda_turbulence_veto:
            return False, f"TDA:turbulence={turbulence:.2f}>{self.config.tda_turbulence_veto}"

        # Rule 2: Low Betti-1 → veto mean-reversion (no cyclical structure to revert to)
        if signal.strategy == 'mean_rev' and betti1 < self.config.tda_betti1_min:
            return False, f"TDA:betti1={betti1}<{self.config.tda_betti1_min}"

        # Rule 3: Very high entropy → reduce confidence note (informational)
        if entropy > 2.5:
            return True, f"TDA:OK(entropy={entropy:.1f},caution)"

        return True, f"TDA:OK(turb={turbulence:.2f},b1={betti1},ent={entropy:.1f})"


# ── ML Ensemble Confirmation ──────────────────────────────────────────────────

class MLConfirm:
    """OVERHAUL FIX #12: Lightweight RF + GBM ensemble trained on trade outcomes.

    Cold start: auto-approves until `ml_min_samples` outcomes recorded.
    Retrains every `ml_retrain_every` new outcomes.
    """

    _FEATURE_KEYS = [
        'rsi', 'vwap_distance', 'bb_position', 'vol_ratio', 'vol_surge',
        'today_return', 'range_position',
    ]

    def __init__(self, config: TraderConfig):
        self.config = config
        self.X: List[List[float]] = []   # feature rows
        self.y: List[int] = []           # 1=win, 0=loss
        self._new_since_train: int = 0
        self._rf = None   # RandomForestClassifier
        self._gbm = None  # GradientBoostingClassifier
        self._trained = False

    # ── helpers ──

    def _extract(self, analysis: Dict) -> Optional[List[float]]:
        """Pull numeric features from an analysis dict."""
        try:
            row = []
            for k in self._FEATURE_KEYS:
                v = analysis.get(k, 0.0)
                row.append(float(v) if v is not None else 0.0)
            # Normalise ATR by price so it's scale-free
            atr = float(analysis.get('atr', 0) or 0)
            price = float(analysis.get('price', 1) or 1)
            row.append(atr / price if price else 0.0)
            return row
        except Exception:
            return None

    # ── public API ──

    def record_outcome(self, analysis: Dict, won: bool) -> None:
        """Record a completed trade for future training."""
        feats = self._extract(analysis)
        if feats is None:
            return
        self.X.append(feats)
        self.y.append(int(won))
        self._new_since_train += 1
        if (self._trained and self._new_since_train >= self.config.ml_retrain_every) or \
           (not self._trained and len(self.y) >= self.config.ml_min_samples):
            self._retrain()

    def confirm(self, signal: 'Signal', analysis: Dict) -> Tuple[bool, str]:
        """Approve/veto a signal. Returns (approved, reason)."""
        if not self._trained:
            return True, f"ML:cold_start({len(self.y)}/{self.config.ml_min_samples})"
        feats = self._extract(analysis)
        if feats is None:
            return True, "ML:no_features"
        try:
            import numpy as _np
            X = _np.array([feats])
            p_rf = self._rf.predict_proba(X)[0][1]
            p_gbm = self._gbm.predict_proba(X)[0][1]
            prob = (p_rf + p_gbm) / 2.0
            if prob < self.config.ml_veto_threshold:
                return False, f"ML:veto(prob={prob:.2f}<{self.config.ml_veto_threshold})"
            return True, f"ML:OK(prob={prob:.2f})"
        except Exception as e:
            return True, f"ML:error({e})"

    # ── internal ──

    def _retrain(self) -> None:
        """Fit RF + GBM on accumulated outcomes."""
        try:
            import numpy as _np
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

            X = _np.array(self.X)
            y = _np.array(self.y)
            if len(set(y)) < 2:
                return  # need both classes

            self._rf = RandomForestClassifier(
                n_estimators=80, max_depth=4, min_samples_leaf=3, random_state=42
            )
            self._gbm = GradientBoostingClassifier(
                n_estimators=80, max_depth=3, learning_rate=0.05, random_state=42
            )
            self._rf.fit(X, y)
            self._gbm.fit(X, y)
            self._trained = True
            self._new_since_train = 0
            logger.info(f"ML ensemble retrained on {len(y)} samples "
                       f"(wins={int(y.sum())}, losses={len(y)-int(y.sum())})")
        except Exception as e:
            logger.warning(f"ML retrain failed: {e}")


# ── State Persistence (crash recovery) ─────────────────────────────────────────

class StatePersistence:
    """OVERHAUL FIX #13: Save/restore trader state to survive restarts.

    Persists tracked positions, ML training data, and counters to JSON.
    On startup, restores everything so the bot resumes seamlessly.
    """

    def __init__(self, config: TraderConfig):
        self.path = config.state_file
        os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)

    def save(self, trader: 'ProfitTrader') -> None:
        """Serialize trader state to JSON."""
        try:
            positions = {}
            for sym, pos in trader.positions.tracked.items():
                positions[sym] = {
                    'symbol': pos.symbol,
                    'entry_price': pos.entry_price,
                    'qty': pos.qty,
                    'side': pos.side,
                    'strategy': pos.strategy,
                    'entry_time': pos.entry_time.isoformat(),
                    'stop_loss': pos.stop_loss,
                    'take_profit': pos.take_profit,
                    'trailing_stop': pos.trailing_stop,
                    'high_water': pos.high_water,
                }

            ml_state = {}
            if trader.ml_confirm:
                ml_state = {
                    'X': trader.ml_confirm.X,
                    'y': trader.ml_confirm.y,
                    'trained': trader.ml_confirm._trained,
                }

            state = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'cycle': trader.cycle,
                'tracked_positions': positions,
                'ml_state': ml_state,
                'wins_today': trader.positions.wins_today,
                'losses_today': trader.positions.losses_today,
                'trades_today': trader.positions.trades_today,
                'daily_pnl': trader.positions.daily_pnl,
            }

            tmp = self.path + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(state, f, indent=2)
            os.replace(tmp, self.path)  # atomic on POSIX
            logger.info(f"State saved: {len(positions)} positions, cycle={trader.cycle}")
        except Exception as e:
            logger.warning(f"State save failed: {e}")

    def load(self, trader: 'ProfitTrader') -> None:
        """Restore trader state from JSON."""
        if not os.path.exists(self.path):
            logger.info("No saved state found — starting fresh")
            return
        try:
            with open(self.path, 'r') as f:
                state = json.load(f)

            # Restore cycle count
            trader.cycle = state.get('cycle', 0)

            # Restore tracked positions
            positions = state.get('tracked_positions', {})
            for sym, pdata in positions.items():
                trader.positions.tracked[sym] = TrackedPosition(
                    symbol=pdata['symbol'],
                    entry_price=pdata['entry_price'],
                    qty=pdata['qty'],
                    side=pdata['side'],
                    strategy=pdata['strategy'],
                    entry_time=datetime.fromisoformat(pdata['entry_time']),
                    stop_loss=pdata['stop_loss'],
                    take_profit=pdata['take_profit'],
                    trailing_stop=pdata['trailing_stop'],
                    high_water=pdata['high_water'],
                )

            # Restore ML training data
            ml_state = state.get('ml_state', {})
            if trader.ml_confirm and ml_state.get('X'):
                trader.ml_confirm.X = ml_state['X']
                trader.ml_confirm.y = ml_state['y']
                if len(trader.ml_confirm.y) >= trader.config.ml_min_samples:
                    trader.ml_confirm._retrain()
                    logger.info(f"ML state restored and retrained ({len(trader.ml_confirm.y)} samples)")

            # Restore counters
            trader.positions.wins_today = state.get('wins_today', 0)
            trader.positions.losses_today = state.get('losses_today', 0)
            trader.positions.trades_today = state.get('trades_today', 0)
            trader.positions.daily_pnl = state.get('daily_pnl', 0.0)

            ts = state.get('timestamp', 'unknown')
            logger.info(f"State restored from {ts}: {len(positions)} positions, cycle={trader.cycle}")
        except Exception as e:
            logger.warning(f"State load failed (starting fresh): {e}")


# ── Trade Journal (CSV audit trail) ───────────────────────────────────────────

class TradeJournal:
    """OVERHAUL FIX #15: Append-only CSV trade journal for post-hoc analysis.

    Records every entry and exit with full context: strategy, regime, P&L, etc.
    """

    COLUMNS = [
        'timestamp', 'action', 'ticker', 'strategy', 'qty', 'price',
        'stop_loss', 'take_profit', 'confidence', 'pnl_pct',
        'exit_reason', 'regime', 'cycle',
    ]

    def __init__(self, config: TraderConfig):
        self.path = config.journal_file
        os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)
        # Write header if file doesn't exist yet
        if not os.path.exists(self.path):
            with open(self.path, 'w') as f:
                f.write(','.join(self.COLUMNS) + '\n')
            logger.info(f"Trade journal created: {self.path}")

    def _append(self, row: Dict) -> None:
        """Append a single row to the CSV."""
        try:
            vals = [str(row.get(c, '')) for c in self.COLUMNS]
            with open(self.path, 'a') as f:
                f.write(','.join(vals) + '\n')
        except Exception as e:
            logger.debug(f"Journal write failed: {e}")

    def log_entry(self, signal: 'Signal', qty: int, fill_price: float,
                  regime: str = '', cycle: int = 0) -> None:
        """Record a trade entry."""
        self._append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'action': 'ENTRY',
            'ticker': signal.ticker,
            'strategy': signal.strategy,
            'qty': qty,
            'price': f"{fill_price:.2f}",
            'stop_loss': f"{signal.stop_loss:.2f}",
            'take_profit': f"{signal.take_profit:.2f}",
            'confidence': f"{signal.confidence:.2f}",
            'pnl_pct': '',
            'exit_reason': '',
            'regime': regime,
            'cycle': cycle,
        })

    def log_exit(self, sym: str, reason: str, entry_price: float,
                 exit_price: float, qty: int, strategy: str,
                 regime: str = '', cycle: int = 0) -> None:
        """Record a trade exit with P&L."""
        pnl_pct = (exit_price / entry_price - 1) if entry_price > 0 else 0
        self._append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'action': 'EXIT',
            'ticker': sym,
            'strategy': strategy,
            'qty': qty,
            'price': f"{exit_price:.2f}",
            'stop_loss': '',
            'take_profit': '',
            'confidence': '',
            'pnl_pct': f"{pnl_pct:.4f}",
            'exit_reason': reason.replace(',', ';'),  # Escape commas for CSV
            'regime': regime,
            'cycle': cycle,
        })


# ── EOD Performance Report ───────────────────────────────────────────────

class DailyReporter:
    """OVERHAUL FIX #16: End-of-day performance summary.

    Compiles trade journal into a daily report, saves to JSONL,
    and optionally sends a Discord message.
    """

    def __init__(self, config: TraderConfig):
        self.config = config
        self.report_path = config.eod_report_file
        self.journal_path = config.journal_file
        self._reported_today = False
        os.makedirs(os.path.dirname(self.report_path) or '.', exist_ok=True)

    def generate_report(self, equity: float, daily_return: float,
                        wins: int, losses: int, positions_held: int,
                        regime: str = '') -> Dict:
        """Build EOD report from today's journal entries and live stats."""
        today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')

        # Parse today's journal entries
        entries, exits = [], []
        try:
            if os.path.exists(self.journal_path):
                df = pd.read_csv(self.journal_path)
                df_today = df[df['timestamp'].str.startswith(today_str)]
                entries = df_today[df_today['action'] == 'ENTRY'].to_dict('records')
                exits = df_today[df_today['action'] == 'EXIT'].to_dict('records')
        except Exception:
            pass

        # Compute strategy breakdown
        strategy_stats: Dict[str, Dict] = {}
        for ex in exits:
            strat = str(ex.get('strategy', 'unknown'))
            pnl = float(ex.get('pnl_pct', 0) or 0)
            if strat not in strategy_stats:
                strategy_stats[strat] = {'wins': 0, 'losses': 0, 'total_pnl': 0.0}
            if pnl > 0:
                strategy_stats[strat]['wins'] += 1
            else:
                strategy_stats[strat]['losses'] += 1
            strategy_stats[strat]['total_pnl'] += pnl

        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0

        report = {
            'date': today_str,
            'equity': round(equity, 2),
            'daily_return_pct': round(daily_return * 100, 2),
            'trades_entered': len(entries),
            'trades_exited': len(exits),
            'wins': wins,
            'losses': losses,
            'win_rate': round(win_rate, 3),
            'positions_held_eod': positions_held,
            'regime': regime,
            'strategy_breakdown': strategy_stats,
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }

        # Append to JSONL file
        try:
            with open(self.report_path, 'a') as f:
                f.write(json.dumps(report) + '\n')
        except Exception as e:
            logger.debug(f"Report save failed: {e}")

        return report

    def send_discord(self, report: Dict) -> None:
        """Send report summary to Discord webhook (if configured)."""
        webhook = os.getenv('DISCORD_WEBHOOK_URL')
        if not webhook:
            return
        try:
            emoji = '🟢' if report['daily_return_pct'] >= 0 else '🔴'
            lines = [
                f"{emoji} **Daily Report — {report['date']}**",
                f"Equity: **${report['equity']:,.2f}** ({report['daily_return_pct']:+.2f}%)",
                f"Trades: {report['trades_entered']} entered, {report['trades_exited']} exited",
                f"W/L: {report['wins']}/{report['losses']} ({report['win_rate']:.0%} win rate)",
                f"Holding {report['positions_held_eod']} positions overnight",
                f"Regime: {report['regime'] or 'N/A'}",
            ]
            if report['strategy_breakdown']:
                lines.append('**Strategy P&L:**')
                for strat, s in report['strategy_breakdown'].items():
                    lines.append(f"  {strat}: {s['wins']}W/{s['losses']}L ({s['total_pnl']*100:+.2f}%)")

            requests.post(webhook, json={'content': '\n'.join(lines)}, timeout=10)
            logger.info("EOD report sent to Discord")
        except Exception as e:
            logger.debug(f"Discord send failed: {e}")

    def maybe_report(self, equity: float, daily_return: float,
                     wins: int, losses: int, positions_held: int,
                     regime: str = '') -> None:
        """Generate and send report once per day (idempotent guard)."""
        if self._reported_today:
            return
        report = self.generate_report(equity, daily_return, wins, losses,
                                       positions_held, regime)
        logger.info(f"EOD REPORT: equity=${report['equity']:,.2f} | "
                   f"return={report['daily_return_pct']:+.2f}% | "
                   f"W/L={report['wins']}/{report['losses']} | "
                   f"win_rate={report['win_rate']:.0%}")
        self.send_discord(report)
        self._reported_today = True

    def reset_daily(self):
        """Reset the daily guard (call at market open)."""
        self._reported_today = False


# ── Team of Rivals Veto ────────────────────────────────────────────────────────

class TeamVeto:
    """OVERHAUL FIX #10: 6-agent confirmation layer — need 4/6 to approve a trade."""

    AGENTS = ['marcus', 'victoria', 'james', 'elena', 'derek', 'sophia']

    def __init__(self, config: TraderConfig):
        self.config = config

    def evaluate(self, signal: 'Signal', analysis: Dict,
                 regime_state=None, sector_count: int = 0,
                 spread_pct: float = 0.0) -> Tuple[bool, str]:
        """Run 6 independent checks. Returns (approved, summary)."""
        votes: Dict[str, Tuple[bool, str]] = {}
        votes['marcus']   = self._marcus(signal)
        votes['victoria'] = self._victoria(signal)
        votes['james']    = self._james(signal, analysis)
        votes['elena']    = self._elena(signal, regime_state)
        votes['derek']    = self._derek(signal, spread_pct)
        votes['sophia']   = self._sophia(signal, sector_count)

        approvals = sum(1 for ok, _ in votes.values() if ok)
        passed = approvals >= self.config.team_min_approvals
        vetoes = [f"{a}({r})" for a, (ok, r) in votes.items() if not ok]
        summary = f"{approvals}/6 approve" + (f" | vetoed by: {', '.join(vetoes)}" if vetoes else "")
        return passed, summary

    # ── Agent checks ───────────────────────────────────────────────────────

    def _marcus(self, signal: 'Signal') -> Tuple[bool, str]:
        """Strategy: confidence must be meaningfully high."""
        if signal.confidence >= 0.58:
            return True, "conf OK"
        return False, f"conf {signal.confidence:.0%}<58%"

    def _victoria(self, signal: 'Signal') -> Tuple[bool, str]:
        """Risk: reward-to-risk ratio >= 1.5."""
        risk = abs(signal.price - signal.stop_loss)
        reward = abs(signal.take_profit - signal.price)
        if risk <= 0:
            return True, "no-risk"  # auto-approve if no stop set
        rr = reward / risk
        if rr >= 1.5:
            return True, f"R:R {rr:.1f}"
        return False, f"R:R {rr:.1f}<1.5"

    def _james(self, signal: 'Signal', analysis: Dict) -> Tuple[bool, str]:
        """Quant: price above 20-bar SMA (don't buy into downtrend)."""
        sma20 = analysis.get('sma_20', 0)
        price = signal.price
        if sma20 <= 0:
            return True, "no-SMA"  # data unavailable → approve
        if price >= sma20:
            return True, f"price≥SMA20"
        return False, f"price<SMA20"

    def _elena(self, signal: 'Signal', regime_state) -> Tuple[bool, str]:
        """Market: regime must not be hostile to strategy."""
        if not regime_state or not HAS_REGIME:
            return True, "no-regime"
        regime = regime_state.regime
        # Bear/Crisis hostile to all longs; High-vol hostile to momentum
        if regime in (RegimeType.BEAR, RegimeType.CRISIS):
            return False, f"regime={regime.value}"
        if regime == RegimeType.HIGH_VOL and signal.strategy == 'momentum':
            return False, f"high-vol+momentum"
        return True, f"regime={regime.value}"

    def _derek(self, signal: 'Signal', spread_pct: float) -> Tuple[bool, str]:
        """Execution: bid-ask spread must be < 0.5% for clean fills."""
        if spread_pct <= 0:
            return True, "no-spread"  # data unavailable → approve
        if spread_pct < 0.005:
            return True, f"spread={spread_pct:.2%}"
        return False, f"spread={spread_pct:.2%}≥0.5%"

    def _sophia(self, signal: 'Signal', sector_count: int) -> Tuple[bool, str]:
        """Compliance: sector concentration below limit."""
        limit = self.config.max_correlation_positions
        if sector_count < limit:
            return True, f"{sector_count}/{limit}"
        return False, f"{sector_count}/{limit} full"


# ── Wheel Strategy Manager ──────────────────────────────────────────────────────

class WheelManager:
    """OVERHAUL FIX #9: Sell cash-secured puts; if assigned, sell covered calls."""

    def __init__(self, config: TraderConfig, equity_client: 'AlpacaClient'):
        self.config = config
        self.equity_client = equity_client
        self.engine: Optional['AlpacaOptionsEngine'] = None
        self._last_run_date: Optional[str] = None  # YYYY-MM-DD of last wheel cycle
        self._init_engine()

    def _init_engine(self):
        """Create AlpacaOptionsEngine (uses same .env creds)."""
        try:
            self.engine = AlpacaOptionsEngine(paper=True)
            logger.info("Wheel: AlpacaOptionsEngine initialised")
        except Exception as e:
            logger.warning(f"Wheel: engine init failed — {e}")
            self.engine = None

    # ── helpers ─────────────────────────────────────────────────────────────

    def _target_expiration(self) -> str:
        """Pick a Friday between dte_min and dte_max days out."""
        today = datetime.now(timezone.utc).date()
        target = today + timedelta(days=(self.config.wheel_dte_min + self.config.wheel_dte_max) // 2)
        # Roll to next Friday (weekday 4)
        days_ahead = (4 - target.weekday()) % 7
        target = target + timedelta(days=days_ahead)
        return target.strftime('%Y-%m-%d')

    def _shares_held(self, ticker: str) -> int:
        """How many shares of `ticker` do we hold (from equity positions)?"""
        pos = self.equity_client.get_position(ticker)
        if pos and int(float(pos.get('qty', 0))) > 0:
            return int(float(pos['qty']))
        return 0

    def _has_open_option(self, ticker: str) -> bool:
        """Check if we already have an open option position on this underlying."""
        try:
            positions = self.engine.get_positions()
            for p in positions:
                if p.underlying == ticker:
                    return True
        except Exception:
            pass
        return False

    # ── core logic ──────────────────────────────────────────────────────────

    def find_csp_candidate(self, ticker: str, price: float) -> Optional[dict]:
        """Find a cash-secured put to sell: ~5% OTM, DTE window, min premium."""
        exp = self._target_expiration()
        target_strike = round(price * (1 - self.config.wheel_otm_pct), 2)
        try:
            strike_lo = target_strike * 0.97
            strike_hi = target_strike * 1.03
            contracts = self.engine.get_options_chain(
                underlying=ticker,
                expiration_date=exp,
                strike_range=(strike_lo, strike_hi),
            )
            puts = [c for c in contracts if c.option_type == 'put' and c.bid >= self.config.wheel_min_premium]
            if not puts:
                return None
            # Pick the put closest to target strike
            best = min(puts, key=lambda c: abs(c.strike - target_strike))
            return {'contract': best, 'type': 'csp'}
        except Exception as e:
            logger.debug(f"Wheel CSP scan {ticker}: {e}")
            return None

    def find_cc_candidate(self, ticker: str, price: float, avg_cost: float) -> Optional[dict]:
        """Find a covered call to sell: strike above avg cost, DTE window."""
        exp = self._target_expiration()
        target_strike = round(price * (1 + self.config.wheel_otm_pct), 2)
        # Ensure strike is above avg_cost so assignment is profitable
        target_strike = max(target_strike, round(avg_cost * 1.01, 2))
        try:
            strike_lo = target_strike * 0.97
            strike_hi = target_strike * 1.05
            contracts = self.engine.get_options_chain(
                underlying=ticker,
                expiration_date=exp,
                strike_range=(strike_lo, strike_hi),
            )
            calls = [c for c in contracts if c.option_type == 'call' and c.bid >= self.config.wheel_min_premium]
            if not calls:
                return None
            best = min(calls, key=lambda c: abs(c.strike - target_strike))
            return {'contract': best, 'type': 'cc'}
        except Exception as e:
            logger.debug(f"Wheel CC scan {ticker}: {e}")
            return None

    def run_wheel_cycle(self, equity: float, cash: float, regime_state=None):
        """Main wheel cycle: sell CSPs or CCs for each wheel ticker."""
        if not self.engine:
            logger.debug("Wheel: engine not available, skipping")
            return

        today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        if self._last_run_date == today_str:
            return  # Already ran today

        # Regime guard: don't sell puts in BEAR/CRISIS
        regime_block_puts = False
        if regime_state and HAS_REGIME:
            if regime_state.regime in (RegimeType.BEAR, RegimeType.CRISIS):
                regime_block_puts = True
                logger.info(f"Wheel: regime {regime_state.regime.value} — skipping new CSPs")

        max_wheel_capital = equity * self.config.wheel_capital_pct
        capital_used = 0.0

        logger.info(f"{'─'*40}")
        logger.info(f"WHEEL CYCLE | Tickers: {self.config.wheel_tickers} | Budget: ${max_wheel_capital:,.0f}")

        for ticker in self.config.wheel_tickers:
            if self._has_open_option(ticker):
                logger.info(f"  {ticker}: already has open option — skip")
                continue

            shares = self._shares_held(ticker)
            # Get current price
            snap = self.equity_client.get_snapshot(ticker)
            if not snap:
                continue
            price = float(snap.get('latestTrade', {}).get('p', 0))
            if price <= 0:
                continue

            if shares >= 100:
                # ── Covered call path ──
                pos = self.equity_client.get_position(ticker)
                avg_cost = float(pos.get('avg_entry_price', price)) if pos else price
                candidate = self.find_cc_candidate(ticker, price, avg_cost)
                if candidate:
                    c = candidate['contract']
                    n_contracts = min(shares // 100, self.config.wheel_max_contracts)
                    result = self.engine.place_option_order(
                        symbol=c.symbol, quantity=n_contracts,
                        side='sell', order_type='limit', limit_price=round(c.bid, 2)
                    )
                    if result:
                        logger.info(f"  WHEEL CC: SELL {n_contracts} {c.symbol} @ ${c.bid:.2f} "
                                   f"(strike ${c.strike}, exp {c.expiration})")
                    else:
                        logger.warning(f"  WHEEL CC: order failed for {ticker}")
                else:
                    logger.info(f"  {ticker}: no suitable covered call found")
            else:
                # ── Cash-secured put path ──
                if regime_block_puts:
                    continue
                collateral = price * (1 - self.config.wheel_otm_pct) * 100  # per contract
                if capital_used + collateral > max_wheel_capital:
                    logger.info(f"  {ticker}: wheel capital limit reached")
                    continue
                # Ensure enough cash for assignment
                if cash - collateral < equity * self.config.min_cash_pct:
                    logger.info(f"  {ticker}: insufficient cash for CSP collateral")
                    continue

                candidate = self.find_csp_candidate(ticker, price)
                if candidate:
                    c = candidate['contract']
                    n_contracts = min(
                        self.config.wheel_max_contracts,
                        int((max_wheel_capital - capital_used) / collateral)
                    )
                    n_contracts = max(1, n_contracts)
                    result = self.engine.place_option_order(
                        symbol=c.symbol, quantity=n_contracts,
                        side='sell', order_type='limit', limit_price=round(c.bid, 2)
                    )
                    if result:
                        capital_used += collateral * n_contracts
                        logger.info(f"  WHEEL CSP: SELL {n_contracts} {c.symbol} @ ${c.bid:.2f} "
                                   f"(strike ${c.strike}, exp {c.expiration})")
                    else:
                        logger.warning(f"  WHEEL CSP: order failed for {ticker}")
                else:
                    logger.info(f"  {ticker}: no suitable CSP found")

        self._last_run_date = today_str
        logger.info(f"WHEEL CYCLE COMPLETE | Capital deployed: ${capital_used:,.0f}")


# ── Main Trader ────────────────────────────────────────────────────────────────

class ProfitTrader:
    """Main trading engine - scans, signals, executes, manages."""

    def __init__(self, config: TraderConfig = None):
        self.config = config or TraderConfig()
        self.client = AlpacaClient()
        self.data = MarketData(client=self.client)
        self.signals = SignalGenerator(self.config)
        self.positions = PositionManager(self.config, self.client)
        self.running = False
        self.cycle = 0
        # OVERHAUL FIX #6: HMM Regime Filter
        self.regime_state = None
        self._init_regime_detector()
        # OVERHAUL FIX #10: Team of Rivals veto
        self.veto: Optional[TeamVeto] = None
        if self.config.team_veto_enabled:
            self.veto = TeamVeto(self.config)
            logger.info("Team of Rivals veto: enabled (need %d/6 approvals)", self.config.team_min_approvals)
        # OVERHAUL FIX #11: TDA confirmation
        self.tda_confirm: Optional['TDAConfirm'] = None
        if HAS_TDA and self.config.tda_enabled:
            try:
                self.tda_confirm = TDAConfirm(self.config)
                logger.info("TDA confirmation: enabled (window=%d)", self.config.tda_window)
            except Exception as e:
                logger.warning(f"TDA confirmation init failed: {e}")
        # OVERHAUL FIX #12: ML Ensemble confirmation
        self.ml_confirm: Optional['MLConfirm'] = None
        if self.config.ml_ensemble_enabled:
            try:
                self.ml_confirm = MLConfirm(self.config)
                logger.info("ML ensemble: enabled (min_samples=%d, retrain_every=%d)",
                           self.config.ml_min_samples, self.config.ml_retrain_every)
            except Exception as e:
                logger.warning(f"ML ensemble init failed: {e}")
        # OVERHAUL FIX #13: State persistence (crash recovery)
        self.state_persistence = StatePersistence(self.config)
        self.state_persistence.load(self)  # Restore state from previous run
        # OVERHAUL FIX #14: Health monitoring
        self.health: Optional['HealthMonitor'] = None
        self._api_fail_streak = 0
        self._peak_equity = 0.0
        self._health_halted = False
        if HAS_HEALTH and self.config.health_enabled:
            try:
                discord_url = os.getenv('DISCORD_WEBHOOK_URL')
                self.health = HealthMonitor(
                    heartbeat_interval=self.config.health_heartbeat_interval,
                    check_interval=self.config.health_check_interval,
                    discord_webhook=discord_url,
                )
                self._register_health_checks()
                logger.info("Health monitor: enabled (check=%ds, heartbeat=%ds, discord=%s)",
                           self.config.health_check_interval,
                           self.config.health_heartbeat_interval,
                           'YES' if discord_url else 'no')
            except Exception as e:
                logger.warning(f"Health monitor init failed: {e}")
        # OVERHAUL FIX #15: Trade journal
        self.journal: Optional['TradeJournal'] = None
        if self.config.journal_enabled:
            try:
                self.journal = TradeJournal(self.config)
                logger.info("Trade journal: enabled (%s)", self.config.journal_file)
            except Exception as e:
                logger.warning(f"Trade journal init failed: {e}")
        # OVERHAUL FIX #16: EOD performance reporter
        self.reporter: Optional['DailyReporter'] = None
        if self.config.eod_report_enabled:
            try:
                self.reporter = DailyReporter(self.config)
                logger.info("EOD reporter: enabled (%s)", self.config.eod_report_file)
            except Exception as e:
                logger.warning(f"EOD reporter init failed: {e}")
        # OVERHAUL FIX #9: Wheel strategy
        self.wheel: Optional[WheelManager] = None
        if HAS_OPTIONS and self.config.wheel_enabled:
            try:
                self.wheel = WheelManager(self.config, self.client)
            except Exception as e:
                logger.warning(f"Wheel manager init failed: {e}")

    # OVERHAUL FIX #14: Health check registrations
    def _register_health_checks(self):
        """Register components with the health monitor."""
        if not self.health:
            return

        def check_alpaca_api():
            """Check Alpaca API connectivity."""
            t0 = time.time()
            try:
                acct = self.client.get_account()
                latency = (time.time() - t0) * 1000
                if acct and 'equity' in acct:
                    self._api_fail_streak = 0
                    return HealthCheckResult(
                        component='alpaca_api', component_type=ComponentType.API_CONNECTION,
                        status=HealthStatus.HEALTHY, timestamp=datetime.now(timezone.utc),
                        message=f"OK (equity=${float(acct['equity']):,.0f})",
                        latency_ms=latency,
                    )
                self._api_fail_streak += 1
                return HealthCheckResult(
                    component='alpaca_api', component_type=ComponentType.API_CONNECTION,
                    status=HealthStatus.DEGRADED, timestamp=datetime.now(timezone.utc),
                    message='No equity in response', latency_ms=latency,
                )
            except Exception as e:
                self._api_fail_streak += 1
                status = HealthStatus.CRITICAL if self._api_fail_streak >= self.config.consecutive_api_fail_limit else HealthStatus.UNHEALTHY
                return HealthCheckResult(
                    component='alpaca_api', component_type=ComponentType.API_CONNECTION,
                    status=status, timestamp=datetime.now(timezone.utc),
                    message=f"API fail #{self._api_fail_streak}: {e}",
                    latency_ms=(time.time() - t0) * 1000,
                )

        def check_drawdown():
            """Monitor portfolio drawdown."""
            try:
                acct = self.client.get_account()
                equity = float(acct['equity'])
                last_eq = float(acct.get('last_equity', equity))
                self._peak_equity = max(self._peak_equity, equity, last_eq)
                dd = (self._peak_equity - equity) / self._peak_equity if self._peak_equity > 0 else 0

                if dd >= self.config.drawdown_halt_pct:
                    self._health_halted = True
                    return HealthCheckResult(
                        component='drawdown', component_type=ComponentType.RISK_MANAGER,
                        status=HealthStatus.CRITICAL, timestamp=datetime.now(timezone.utc),
                        message=f"HALT: drawdown {dd:.1%} >= {self.config.drawdown_halt_pct:.0%}",
                        metadata={'drawdown': dd, 'equity': equity, 'peak': self._peak_equity},
                    )
                elif dd >= self.config.drawdown_alert_pct:
                    return HealthCheckResult(
                        component='drawdown', component_type=ComponentType.RISK_MANAGER,
                        status=HealthStatus.UNHEALTHY, timestamp=datetime.now(timezone.utc),
                        message=f"WARNING: drawdown {dd:.1%}",
                        metadata={'drawdown': dd, 'equity': equity, 'peak': self._peak_equity},
                    )
                else:
                    self._health_halted = False
                    return HealthCheckResult(
                        component='drawdown', component_type=ComponentType.RISK_MANAGER,
                        status=HealthStatus.HEALTHY, timestamp=datetime.now(timezone.utc),
                        message=f"DD {dd:.1%} (peak=${self._peak_equity:,.0f})",
                    )
            except Exception as e:
                return HealthCheckResult(
                    component='drawdown', component_type=ComponentType.RISK_MANAGER,
                    status=HealthStatus.UNKNOWN, timestamp=datetime.now(timezone.utc),
                    message=str(e),
                )

        def check_positions():
            """Monitor position health."""
            n = self.positions.position_count
            status = HealthStatus.HEALTHY if n <= self.config.max_positions else HealthStatus.DEGRADED
            return HealthCheckResult(
                component='positions', component_type=ComponentType.POSITION_TRACKER,
                status=status, timestamp=datetime.now(timezone.utc),
                message=f"{n}/{self.config.max_positions} positions",
                metadata={'count': n, 'tracked': list(self.positions.tracked.keys())},
            )

        self.health.register_component('alpaca_api', check_alpaca_api, ComponentType.API_CONNECTION)
        self.health.register_component('drawdown', check_drawdown, ComponentType.RISK_MANAGER)
        self.health.register_component('positions', check_positions, ComponentType.POSITION_TRACKER)

    def _init_regime_detector(self):
        """Initialize regime detector with SPY daily returns."""
        if not HAS_REGIME:
            logger.info("Regime detector: disabled (import unavailable)")
            return
        try:
            self.regime_detector = EnsembleRegimeDetector(n_regimes=3, n_features=5)
            # Fetch SPY daily bars for regime fitting
            bars = self.client.get_bars('SPY', timeframe='1Day', limit=500)
            if bars and len(bars) >= 60:
                closes = np.array([float(b.get('c', b.get('Close', 0))) for b in bars])
                returns = np.diff(closes) / closes[:-1]
                self.regime_detector.fit(returns)
                # Get current regime
                features = self.regime_detector.compute_features(returns)
                if len(features) > 0:
                    self.regime_state = self.regime_detector.predict(features[-1:])
                    logger.info(f"Regime detector: {self.regime_state.regime.value} "
                               f"(confidence={self.regime_state.confidence:.0%})")
            else:
                logger.warning("Regime detector: insufficient SPY data for fitting")
        except Exception as e:
            logger.warning(f"Regime detector init failed: {e}")
            self.regime_state = None

    def _update_regime(self):
        """Update regime state from latest SPY data."""
        if not HAS_REGIME or not hasattr(self, 'regime_detector') or not self.regime_detector.is_fitted:
            return
        try:
            bars = self.client.get_bars('SPY', timeframe='1Day', limit=60)
            if bars and len(bars) >= 40:
                closes = np.array([float(b.get('c', b.get('Close', 0))) for b in bars])
                returns = np.diff(closes) / closes[:-1]
                features = self.regime_detector.compute_features(returns)
                if len(features) > 0:
                    self.regime_state = self.regime_detector.predict(features[-1:])
        except Exception as e:
            logger.debug(f"Regime update failed: {e}")
        """Check if market is currently open."""
        try:
            acct = self.client.get_account()
            # Also check via clock
            r = requests.get('https://paper-api.alpaca.markets/v2/clock',
                           headers=self.client.headers, timeout=5)
            if r.status_code == 200:
                clock = r.json()
                return clock.get('is_open', False)
        except:
            pass
        # Fallback: check time (ET = UTC-5)
        now = datetime.now(timezone.utc) - timedelta(hours=5)
        if now.weekday() >= 5:
            return False
        market_open = now.replace(hour=9, minute=30, second=0)
        market_close = now.replace(hour=16, minute=0, second=0)
        return market_open <= now <= market_close

    # OVERHAUL FIX #8: Kelly-based position sizing with cash buffer
    def calculate_position_size(self, price: float, signal: 'Signal' = None) -> int:
        """Calculate number of shares using Kelly criterion, preserving cash buffer."""
        acct = self.client.get_account()
        equity = float(acct['equity'])
        cash = float(acct['cash'])
        buying_power = float(acct['buying_power'])

        # Default: use config percentage of equity
        target_value = equity * self.config.position_pct

        # Kelly sizing: f* = (p*b - q) / b  where p=win_prob, b=win/loss, q=1-p
        if signal and signal.atr > 0:
            win_prob = signal.confidence
            risk = abs(price - signal.stop_loss)
            reward = abs(signal.take_profit - price)
            if risk > 0:
                b = reward / risk  # payoff ratio
                q = 1 - win_prob
                kelly = (win_prob * b - q) / b
                kelly = max(0, kelly) * self.config.kelly_fraction  # Half-Kelly
                kelly_pct = np.clip(kelly, self.config.min_position_pct, self.config.max_position_pct)
                target_value = equity * kelly_pct
                logger.debug(f"Kelly sizing {signal.ticker}: f*={kelly:.3f}, "
                            f"pct={kelly_pct:.1%}, val=${target_value:,.0f}")

        # Don't exceed buying power / remaining positions
        available_per = buying_power / max(1, self.config.max_positions - self.positions.position_count)
        target_value = min(target_value, available_per * 0.9)

        # Enforce cash buffer: never let cash drop below min_cash_pct of equity
        min_cash = equity * self.config.min_cash_pct
        if cash - target_value < min_cash:
            target_value = max(0, cash - min_cash)
            if target_value < equity * 0.03:  # <3% position not worth it
                logger.warning(f"Cash buffer: need ${min_cash:,.0f} reserved, have ${cash:,.0f} — skipping trade")
                return 0

        shares = int(target_value / price)
        return max(1, shares) if shares > 0 else 0

    # OVERHAUL FIX #4: Sector-aware position limits
    def get_sector(self, symbol: str) -> str:
        """Return sector group for a symbol."""
        for sector, tickers in SECTOR_MAP.items():
            if symbol in tickers:
                return sector
        return 'other'

    def count_positions_in_sector(self, sector: str) -> int:
        """Count how many tracked positions are in a given sector."""
        return sum(1 for sym in self.positions.tracked if self.get_sector(sym) == sector)

    # OVERHAUL FIX #9: Wheel strategy integration
    def _run_wheel_cycle(self, equity: float, cash: float):
        """Run wheel strategy once per day at the configured hour."""
        if not self.wheel:
            return
        try:
            now_utc = datetime.now(timezone.utc)
            et_hour = (now_utc - timedelta(hours=5)).hour  # rough ET
            if et_hour != self.config.wheel_run_hour:
                return  # Not the right hour
            self.wheel.run_wheel_cycle(equity, cash, regime_state=self.regime_state)
        except Exception as e:
            logger.warning(f"Wheel cycle error: {e}")

    def scan_and_trade(self):
        """Main scan cycle: analyze universe, generate signals, execute."""
        self.cycle += 1
        logger.info(f"{'='*60}")
        logger.info(f"SCAN CYCLE #{self.cycle} | Positions: {self.positions.position_count}/{self.config.max_positions}")
        logger.info(f"{'='*60}")

        # ── Step 1: Check exits on existing positions ──
        exits = self.positions.check_exits(self.data)
        for sym, reason in exits:
            logger.info(f"EXIT {sym}: {reason}")
            # OVERHAUL FIX #15: Record exit in trade journal
            pos = self.positions.tracked.get(sym)
            if self.journal and pos:
                try:
                    snap = self.client.get_snapshot(sym)
                    exit_price = float(snap['latestTrade']['p']) if snap and 'latestTrade' in snap else pos.entry_price
                    regime_str = self.regime_state.regime.value if self.regime_state and HAS_REGIME else ''
                    self.journal.log_exit(sym, reason, pos.entry_price, exit_price,
                                          pos.qty, pos.strategy, regime_str, self.cycle)
                except Exception:
                    pass
            # OVERHAUL FIX #12: record outcome for ML ensemble
            won = 'TAKE PROFIT' in reason or ('TRAILING STOP' in reason and 'gain' in reason)
            if self.ml_confirm:
                try:
                    analysis = self.data.analyze(sym)
                    if analysis:
                        self.ml_confirm.record_outcome(analysis, won)
                except Exception:
                    pass
            self.client.close_position(sym)
            if sym in self.positions.tracked:
                del self.positions.tracked[sym]

        # ── Step 2: Sync positions ──
        self.positions.sync_positions()

        # ── Step 3: Check daily loss limit ──
        acct = self.client.get_account()
        equity = float(acct['equity'])
        last_equity = float(acct.get('last_equity', equity))
        daily_return = (equity / last_equity - 1) if last_equity > 0 else 0

        if daily_return < -self.config.daily_loss_limit_pct:
            logger.warning(f"DAILY LOSS LIMIT HIT: {daily_return*100:.1f}% | Stopping new entries")
            return

        # ── Step 3b: Update regime state ──
        self._update_regime()
        regime_block_strategies = set()
        if self.regime_state and HAS_REGIME:
            regime = self.regime_state.regime
            if regime in (RegimeType.BEAR, RegimeType.CRISIS):
                regime_block_strategies = {'momentum', 'trend'}
                logger.warning(f"REGIME FILTER: {regime.value} (conf={self.regime_state.confidence:.0%}) "
                              f"— blocking momentum/trend entries")
            elif regime == RegimeType.HIGH_VOL:
                regime_block_strategies = {'momentum'}
                logger.info(f"REGIME FILTER: {regime.value} — blocking momentum entries")

        # ── Step 4: Scan for new opportunities ──
        if not self.positions.can_open_new:
            logger.info(f"Max positions reached ({self.positions.position_count}). Waiting for exits.")
            return

        candidates = []
        held_symbols = set(self.positions.tracked.keys())

        for ticker in self.config.universe:
            if ticker in held_symbols:
                continue
            try:
                analysis = self.data.analyze(ticker)
                if analysis is None:
                    continue
                signal = self.signals.generate(analysis)
                if signal and signal.confidence >= 0.55:
                    candidates.append(signal)
            except Exception as e:
                logger.debug(f"Error analyzing {ticker}: {e}")

        # Sort by confidence
        candidates.sort(key=lambda s: s.confidence, reverse=True)

        # ── Step 5: Execute top signals ──
        slots_available = self.config.max_positions - self.positions.position_count
        to_execute = candidates[:slots_available]

        if not to_execute:
            logger.info("No qualifying signals this cycle.")
        
        for signal in to_execute:
            # OVERHAUL FIX #6: Regime filter — skip blocked strategies
            if signal.strategy in regime_block_strategies:
                logger.info(f"SKIPPED {signal.ticker}: {signal.strategy} blocked by regime filter ({self.regime_state.regime.value})")
                continue

            # OVERHAUL FIX #4: Enforce sector cap before entry
            signal_sector = self.get_sector(signal.ticker)
            sector_count = self.count_positions_in_sector(signal_sector)
            if sector_count >= self.config.max_correlation_positions:
                logger.info(f"SKIPPED {signal.ticker}: sector '{signal_sector}' full ({sector_count}/{self.config.max_correlation_positions})")
                continue

            # OVERHAUL FIX #10: Team of Rivals veto
            if self.veto:
                # Compute bid-ask spread %
                spread_pct = 0.0
                snap = self.client.get_snapshot(signal.ticker)
                if snap and 'latestQuote' in snap:
                    q = snap['latestQuote']
                    bid, ask = float(q.get('bp', 0)), float(q.get('ap', 0))
                    mid = (bid + ask) / 2
                    if mid > 0:
                        spread_pct = (ask - bid) / mid
                # Retrieve analysis for SMA check (use cached if possible)
                veto_analysis = self.data.analyze(signal.ticker) or {}
                approved, veto_summary = self.veto.evaluate(
                    signal, veto_analysis,
                    regime_state=self.regime_state,
                    sector_count=sector_count,
                    spread_pct=spread_pct,
                )
                if not approved:
                    logger.info(f"VETOED {signal.ticker}: {veto_summary}")
                    continue
                else:
                    logger.debug(f"APPROVED {signal.ticker}: {veto_summary}")

            # OVERHAUL FIX #11: TDA topological confirmation
            if self.tda_confirm:
                try:
                    df = self.data.get_stock_data(signal.ticker, period='10d', interval='5m')
                    if df is not None and len(df) >= self.config.tda_window + 5:
                        close_arr = df['Close'].values.astype(float)
                        tda_ok, tda_reason = self.tda_confirm.confirm(signal, close_arr)
                        if not tda_ok:
                            logger.info(f"TDA VETO {signal.ticker}: {tda_reason}")
                            continue
                        else:
                            logger.debug(f"TDA PASS {signal.ticker}: {tda_reason}")
                except Exception as e:
                    logger.debug(f"TDA check failed for {signal.ticker}: {e}")

            # OVERHAUL FIX #12: ML ensemble confirmation
            if self.ml_confirm:
                try:
                    ml_ok, ml_reason = self.ml_confirm.confirm(signal, analysis)
                    if not ml_ok:
                        logger.info(f"ML VETO {signal.ticker}: {ml_reason}")
                        continue
                    else:
                        logger.debug(f"ML PASS {signal.ticker}: {ml_reason}")
                except Exception as e:
                    logger.debug(f"ML check failed for {signal.ticker}: {e}")

            qty = self.calculate_position_size(signal.price, signal=signal)
            if qty == 0:
                logger.info(f"SKIPPED {signal.ticker}: insufficient capital (cash buffer)")
                continue
            logger.info(f"SIGNAL: {signal.strategy.upper()} {signal.ticker} | "
                       f"Conf={signal.confidence:.0%} | Qty={qty} | "
                       f"Stop=${signal.stop_loss:.2f} | Target=${signal.take_profit:.2f}")
            logger.info(f"  Reason: {signal.reason}")

            order = self.client.place_order(
                symbol=signal.ticker,
                qty=qty,
                side='buy',
                order_type='limit',
                time_in_force='day'
            )

            if order:
                # OVERHAUL FIX #2: Monitor limit order fill (30s timeout)
                order_id = order['id']
                filled = False
                for _wait in range(6):  # 6 x 5s = 30s
                    time.sleep(5)
                    status = self.client.get_order(order_id)
                    if status and status.get('status') == 'filled':
                        fill_price = float(status.get('filled_avg_price', signal.price))
                        logger.info(f"  FILLED: {signal.ticker} x{qty} @ ${fill_price:.2f} (limit)")
                        self.positions.add_position(signal, qty)
                        # OVERHAUL FIX #15: Record entry in trade journal
                        if self.journal:
                            regime_str = self.regime_state.regime.value if self.regime_state and HAS_REGIME else ''
                            self.journal.log_entry(signal, qty, fill_price, regime_str, self.cycle)
                        filled = True
                        break
                    elif status and status.get('status') in ('canceled', 'expired', 'rejected'):
                        logger.warning(f"  Order {signal.ticker} {status.get('status')}")
                        break
                if not filled:
                    # Cancel stale limit, replace with market
                    self.client.cancel_order(order_id)
                    logger.info(f"  Limit not filled in 30s, replacing with market order")
                    mkt = self.client.place_order(
                        symbol=signal.ticker, qty=qty, side='buy',
                        order_type='market', time_in_force='day'
                    )
                    if mkt:
                        self.positions.add_position(signal, qty)
                        # OVERHAUL FIX #15: Record market-order entry in trade journal
                        if self.journal:
                            regime_str = self.regime_state.regime.value if self.regime_state and HAS_REGIME else ''
                            self.journal.log_entry(signal, qty, signal.price, regime_str, self.cycle)
                        logger.info(f"  FILLED: {signal.ticker} x{qty} @ market (fallback)")

        # ── Step 6: Status summary ──
        logger.info(f"\nPORTFOLIO STATUS:")
        logger.info(f"  Equity: ${equity:,.2f} | Daily: {daily_return*100:+.2f}%")
        logger.info(f"  Positions: {self.positions.position_count}/{self.config.max_positions}")
        logger.info(f"  Trades today: {self.positions.trades_today} | W:{self.positions.wins_today} L:{self.positions.losses_today}")
        for sym, pos in self.positions.tracked.items():
            logger.info(f"    {sym}: {pos.qty} @ ${pos.entry_price:.2f} ({pos.strategy}) | Stop=${pos.stop_loss:.2f} | TP=${pos.take_profit:.2f}")

        # ── Step 6b: OVERHAUL FIX #13 — Periodic state save ──
        if self.cycle % self.config.state_save_interval == 0:
            self.state_persistence.save(self)

        # ── Step 7: OVERHAUL FIX #9 — Wheel strategy (once per day) ──
        self._run_wheel_cycle(equity, float(acct['cash']))

    def run_once(self):
        """Run a single scan cycle (for testing or cron)."""
        if not self.is_market_open():
            logger.warning("Market is closed. Use run_once_force() to bypass.")
            return False
        self.scan_and_trade()
        return True

    def run_once_force(self):
        """Run a single scan cycle regardless of market hours."""
        self.scan_and_trade()
        return True

    def run_continuous(self):
        """Run continuously during market hours."""
        self.running = True
        # OVERHAUL FIX #14: Start health monitoring background thread
        if self.health:
            self.health.start()
            logger.info("Health monitor: background thread started")
        logger.info("="*70)
        logger.info("PROFIT TRADER STARTING - CONTINUOUS MODE")
        logger.info(f"Universe: {len(self.config.universe)} stocks")
        logger.info(f"Scan interval: {self.config.scan_interval_seconds}s")
        logger.info(f"Position size: {self.config.position_pct:.0%} | Max positions: {self.config.max_positions}")
        logger.info("="*70)

        while self.running:
            try:
                # OVERHAUL FIX #14: Skip trading if health monitor halted us
                if self._health_halted:
                    logger.warning("HEALTH HALT: drawdown limit reached — skipping trades")
                    time.sleep(self.config.scan_interval_seconds)
                    continue
                if self.is_market_open():
                    # OVERHAUL FIX #16: Reset daily report guard at market open
                    if self.reporter:
                        self.reporter.reset_daily()
                    self.scan_and_trade()
                else:
                    logger.info("Market closed. Waiting...")
                    # OVERHAUL FIX #16: Generate EOD report once when market closes
                    if self.reporter:
                        try:
                            acct = self.client.get_account()
                            eq = float(acct['equity'])
                            last_eq = float(acct.get('last_equity', eq))
                            dr = (eq / last_eq - 1) if last_eq > 0 else 0
                            regime_str = self.regime_state.regime.value if self.regime_state and HAS_REGIME else ''
                            self.reporter.maybe_report(
                                eq, dr,
                                self.positions.wins_today, self.positions.losses_today,
                                self.positions.position_count, regime_str,
                            )
                        except Exception as e:
                            logger.debug(f"EOD report failed: {e}")
                    # OVERHAUL FIX #7: Only close EOD if NOT in swing mode
                    if not self.config.swing_mode and self.positions.position_count > 0:
                        logger.info("EOD: Closing all positions (day-trade mode)")
                        for sym in list(self.positions.tracked.keys()):
                            self.client.close_position(sym)
                        self.positions.tracked.clear()
                    elif self.positions.position_count > 0:
                        logger.info(f"SWING MODE: Holding {self.positions.position_count} positions overnight")

                time.sleep(self.config.scan_interval_seconds)

            except KeyboardInterrupt:
                logger.info("Shutting down — saving state...")
                self.state_persistence.save(self)
                if self.health:
                    self.health.stop()
                self.running = False
                break
            except Exception as e:
                logger.error(f"Cycle error: {e}")
                time.sleep(30)

    def emergency_close_all(self):
        """Close all positions immediately."""
        logger.warning("EMERGENCY CLOSE ALL")
        self.client.cancel_all_orders()
        positions = self.client.get_positions()
        for p in positions:
            self.client.close_position(p['symbol'])
        self.positions.tracked.clear()


# ── Entry Point ────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Profit Trader - Aggressive Momentum Trading')
    parser.add_argument('--mode', choices=['scan', 'continuous', 'status', 'close-all'],
                       default='scan', help='Operating mode')
    parser.add_argument('--interval', type=int, default=120, help='Scan interval in seconds')
    parser.add_argument('--max-positions', type=int, default=10, help='Max simultaneous positions')
    parser.add_argument('--position-pct', type=float, default=0.10, help='Position size as pct of equity')
    args = parser.parse_args()

    config = TraderConfig(
        scan_interval_seconds=args.interval,
        max_positions=args.max_positions,
        position_pct=args.position_pct,
    )

    trader = ProfitTrader(config)

    if args.mode == 'status':
        acct = trader.client.get_account()
        print(f"\nPortfolio: ${float(acct['equity']):,.2f}")
        print(f"Cash: ${float(acct['cash']):,.2f}")
        print(f"Buying Power: ${float(acct['buying_power']):,.2f}")
        positions = trader.client.get_positions()
        print(f"\nPositions ({len(positions)}):")
        for p in positions:
            print(f"  {p['symbol']}: {p['qty']} @ ${float(p['avg_entry_price']):,.2f} | P&L: ${float(p['unrealized_pl']):+,.2f}")
        return

    if args.mode == 'close-all':
        trader.emergency_close_all()
        print("All positions closed.")
        return

    if args.mode == 'scan':
        # Single scan - run immediately regardless of market hours for testing
        trader.run_once_force()
        return

    if args.mode == 'continuous':
        trader.run_continuous()


if __name__ == '__main__':
    main()
