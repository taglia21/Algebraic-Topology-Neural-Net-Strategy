#!/usr/bin/env python3
"""Continuous Trading Bot - Signal-based trading during market hours"""
import os
import requests
from dotenv import load_dotenv
import logging
import time
from datetime import datetime
import pytz
import numpy as np

# Import trading gate for circuit breaker protection
try:
    from src.risk.trading_gate import check_trading_allowed, update_breaker_state
    HAS_TRADING_GATE = True
except ImportError:
    HAS_TRADING_GATE = False

# Import process lock to prevent multiple bots
try:
    from src.risk.process_lock import acquire_trading_lock, release_trading_lock
    HAS_PROCESS_LOCK = True
except ImportError:
    HAS_PROCESS_LOCK = False

# Import regime filter to avoid buying in downtrends
try:
    from src.risk.regime_filter import is_bullish_regime, get_position_scale
    HAS_REGIME_FILTER = True
except ImportError:
    HAS_REGIME_FILTER = False

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
ALPACA_KEY = os.getenv('APCA_API_KEY_ID')
ALPACA_SECRET = os.getenv('APCA_API_SECRET_KEY')
ALPACA_BASE = 'https://paper-api.alpaca.markets/v2'
ALPACA_DATA_BASE = 'https://data.alpaca.markets/v2'

alpaca_headers = {
    'APCA-API-KEY-ID': ALPACA_KEY,
    'APCA-API-SECRET-KEY': ALPACA_SECRET
}

# Trading symbols
SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'SPY', 'QQQ', 'IWM']
TRADE_INTERVAL = 120  # Trade every 2 minutes (reduced from 60s to avoid overtrading)
MAX_POSITIONS = 6
POSITION_SIZE_PCT = 0.05  # 5% of equity per position

# Risk parameters - FIXED: take profit > stop loss for positive expectancy
TAKE_PROFIT_PCT = 0.03   # 3% take profit
STOP_LOSS_PCT = 0.015     # 1.5% stop loss (2:1 reward:risk ratio)

def get_account():
    """Get account information"""
    r = requests.get(f'{ALPACA_BASE}/account', headers=alpaca_headers)
    return r.json() if r.status_code == 200 else None

def get_positions():
    """Get current positions"""
    r = requests.get(f'{ALPACA_BASE}/positions', headers=alpaca_headers)
    return r.json() if r.status_code == 200 else []

def get_bars(symbol, timeframe='1Day', limit=30):
    """Get historical bars for analysis"""
    r = requests.get(
        f'{ALPACA_DATA_BASE}/stocks/{symbol}/bars',
        headers=alpaca_headers,
        params={'timeframe': timeframe, 'limit': limit}
    )
    if r.status_code == 200:
        return r.json().get('bars', [])
    return []

def calculate_rsi(prices, period=14):
    """Calculate RSI from price array"""
    if len(prices) < period + 1:
        return 50.0  # Neutral
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss < 1e-10:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def analyze_symbol(symbol):
    """Analyze symbol using RSI + momentum for buy/sell signals"""
    bars = get_bars(symbol, '1Day', 30)
    if not bars or len(bars) < 20:
        return None, 0.0
    
    closes = [float(b['c']) for b in bars]
    current_price = closes[-1]
    
    # RSI
    rsi = calculate_rsi(np.array(closes))
    
    # Momentum: 5-day and 10-day
    mom_5 = (closes[-1] / closes[-5] - 1) if len(closes) >= 5 else 0
    mom_10 = (closes[-1] / closes[-10] - 1) if len(closes) >= 10 else 0
    
    # SMA crossover: 5-day vs 15-day
    sma5 = np.mean(closes[-5:])
    sma15 = np.mean(closes[-15:]) if len(closes) >= 15 else sma5
    
    # Score the opportunity
    score = 0
    if rsi < 35:     # Oversold
        score += 2
    elif rsi < 45:
        score += 1
    
    if mom_5 > 0.005:  # Positive 5-day momentum > 0.5%
        score += 1
    if mom_10 > 0.01:  # Positive 10-day momentum > 1%
        score += 1
    
    if sma5 > sma15:   # Short-term trend up
        score += 1
    
    # Require score >= 3 for a buy
    if score >= 3:
        confidence = min(score / 5.0, 1.0)
        return 'BUY', confidence
    
    return 'HOLD', 0.0

def get_last_price(symbol):
    """Get the latest trade price for limit order pricing."""
    try:
        r = requests.get(
            f'{ALPACA_DATA_BASE}/stocks/{symbol}/trades/latest',
            headers=alpaca_headers
        )
        if r.status_code == 200:
            return float(r.json().get('trade', {}).get('p', 0))
    except Exception:
        pass
    return None

def place_order(symbol, qty, side):
    """Place limit order at latest price (avoid market order slippage)."""
    price = get_last_price(symbol)
    if price and price > 0:
        # Limit order: buy slightly above / sell slightly below last trade
        if side == 'buy':
            limit_price = round(price * 1.001, 2)  # 0.1% above
        else:
            limit_price = round(price * 0.999, 2)  # 0.1% below
        order_data = {
            'symbol': symbol,
            'qty': qty,
            'side': side,
            'type': 'limit',
            'limit_price': str(limit_price),
            'time_in_force': 'day'
        }
    else:
        # Fallback to market only if we can't get a price
        logger.warning(f"Could not get price for {symbol}, using market order")
        order_data = {
            'symbol': symbol,
            'qty': qty,
            'side': side,
            'type': 'market',
            'time_in_force': 'day'
        }
    r = requests.post(
        f'{ALPACA_BASE}/orders',
        headers={**alpaca_headers, 'Content-Type': 'application/json'},
        json=order_data
    )
    return r.json() if r.status_code in [200, 201] else None

def should_sell(position):
    """Check stop-loss / take-profit with POSITIVE risk-reward ratio"""
    unrealized_plpc = float(position.get('unrealized_plpc', 0))
    # FIXED: Take profit (3%) > Stop loss (1.5%) = 2:1 reward:risk ratio
    return unrealized_plpc >= TAKE_PROFIT_PCT or unrealized_plpc <= -STOP_LOSS_PCT

def is_market_open_est():
    """Check if market is open using proper EST timezone"""
    est = pytz.timezone('US/Eastern')
    now = datetime.now(est)
    if now.weekday() >= 5:  # Weekend
        return False
    hour, minute = now.hour, now.minute
    # Market hours: 9:30 AM - 4:00 PM EST
    if hour < 9 or (hour == 9 and minute < 30) or hour >= 16:
        return False
    return True

logger.info('='*70)
logger.info('CONTINUOUS TRADING BOT - SIGNAL-BASED (FIXED)')
logger.info(f'Risk params: TP={TAKE_PROFIT_PCT*100}% / SL={STOP_LOSS_PCT*100}% (2:1 R:R)')
logger.info(f'Position size: {POSITION_SIZE_PCT*100}% of equity')
logger.info('='*70)

# Acquire exclusive trading lock
_trading_lock = None
if HAS_PROCESS_LOCK:
    _trading_lock = acquire_trading_lock('continuous_trader')
    if _trading_lock is None:
        logger.error('Another trading bot is already running! Exiting.')
        exit(1)

account = get_account()
if account:
    logger.info(f"Account: ${float(account['equity']):,.2f} equity")
else:
    logger.error('Failed to get account info')
    exit(1)

trade_count = 0

try:
    while True:
        # FIXED: Use EST timezone for market hours check
        if not is_market_open_est():
            logger.info('Market closed - waiting...')
            time.sleep(300)
            continue
        
        # Circuit breaker check
        if HAS_TRADING_GATE:
            allowed, reason = check_trading_allowed()
            if not allowed:
                logger.warning(f'⚠️ CIRCUIT BREAKER: {reason} - skipping cycle')
                time.sleep(60)
                continue
        
        # Get current positions
        positions = get_positions()
        position_symbols = {p['symbol']: p for p in positions}
        
        logger.info(f'\nPositions: {len(positions)}')
        
        # Check for sells first (stop-loss / take-profit)
        for symbol, position in position_symbols.items():
            if should_sell(position):
                qty = int(float(position['qty']))
                plpc = float(position['unrealized_plpc'])
                action = 'PROFIT' if plpc > 0 else 'STOP'
                logger.info(f'{action}: {symbol} ({plpc*100:.2f}%)')
                result = place_order(symbol, qty, 'sell')
                if result:
                    trade_count += 1
                time.sleep(2)
        
        # Look for buy opportunities using REAL signals (not random)
        if len(positions) < MAX_POSITIONS:
            # Regime filter: don't buy in bear markets
            if HAS_REGIME_FILTER and not is_bullish_regime():
                logger.info('📉 Bear regime detected — skipping new long entries')
            else:
                regime_scale = get_position_scale() if HAS_REGIME_FILTER else 1.0
                best_signal = None
                best_conf = 0.0
                best_symbol = None
                
                for symbol in SYMBOLS:
                    if symbol in position_symbols:
                        continue
                    signal, confidence = analyze_symbol(symbol)
                    if signal == 'BUY' and confidence > best_conf:
                        best_signal = signal
                        best_conf = confidence
                        best_symbol = symbol
                
                if best_symbol and best_conf >= 0.5:
                    # Position size based on equity percentage, scaled by regime
                    account = get_account()
                    if account:
                        equity = float(account['equity'])
                        bars = get_bars(best_symbol, '1Day', 1)
                        if bars:
                            price = float(bars[-1]['c'])
                            dollar_amount = equity * POSITION_SIZE_PCT * regime_scale
                            qty = max(1, int(dollar_amount / price))
                            logger.info(f'BUY: {best_symbol} ({qty} shares, conf={best_conf:.2f})')
                            result = place_order(best_symbol, qty, 'buy')
                            if result:
                                trade_count += 1
                time.sleep(2)
        
        # Refresh account
        account = get_account()
        if account:
            equity = float(account['equity'])
            cash = float(account['cash'])
            logger.info(f'Portfolio: ${equity:,.2f} | Cash: ${cash:,.2f} | Trades: {trade_count}')
        
        logger.info(f'Next scan in {TRADE_INTERVAL}s...')
        time.sleep(TRADE_INTERVAL)
        
except KeyboardInterrupt:
    logger.info('\nBot stopped by user')
    logger.info(f'Total trades executed: {trade_count}')
except Exception as e:
    logger.error(f'Error: {e}')
    import traceback
    logger.error(traceback.format_exc())
