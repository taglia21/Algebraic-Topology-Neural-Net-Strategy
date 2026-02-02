#!/usr/bin/env python3
"""Smart Strategy-Based Trading Bot - Uses actual market signals"""
import os
import requests
from dotenv import load_dotenv
import logging
import time
from datetime import datetime, timedelta
import json

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API Configuration
ALPACA_KEY = os.getenv('APCA_API_KEY_ID')
ALPACA_SECRET = os.getenv('APCA_API_SECRET_KEY')
ALPACA_BASE = 'https://paper-api.alpaca.markets/v2'

alpaca_headers = {
    'APCA-API-KEY-ID': ALPACA_KEY,
    'APCA-API-SECRET-KEY': ALPACA_SECRET
}

# Trading universe - major stocks with good liquidity
UNIVERSE = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'SPY', 'QQQ']
SCAN_INTERVAL = 120  # Scan every 2 minutes
MAX_POSITIONS = 5
POSITION_SIZE = 2  # shares per position
PROFIT_TARGET = 0.015  # 1.5% profit target
STOP_LOSS = 0.02  # 2% stop loss

def get_account():
    r = requests.get(f'{ALPACA_BASE}/account', headers=alpaca_headers)
    return r.json() if r.status_code == 200 else None

def get_positions():
    r = requests.get(f'{ALPACA_BASE}/positions', headers=alpaca_headers)
    return r.json() if r.status_code == 200 else []

def get_bars(symbol, timeframe='5Min', limit=20):
    """Get recent price bars for technical analysis"""
    r = requests.get(
        f'{ALPACA_BASE}/stocks/{symbol}/bars',
        headers=alpaca_headers,
        params={'timeframe': timeframe, 'limit': limit}
    )
    if r.status_code == 200:
        data = r.json()
        return data.get('bars', [])
    return []

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    if len(prices) < period + 1:
        return 50  # neutral
    
    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]
    
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_momentum(prices):
    """Calculate price momentum"""
    if len(prices) < 2:
        return 0
    return (prices[-1] - prices[0]) / prices[0]

def analyze_symbol(symbol):
    """Analyze symbol for trading signals"""
    bars = get_bars(symbol)
    if not bars or len(bars) < 15:
        return None
    
    closes = [float(bar['c']) for bar in bars]
    current_price = closes[-1]
    
    # Calculate indicators
    rsi = calculate_rsi(closes)
    momentum = calculate_momentum(closes)
    
    # Simple moving average
    sma_short = sum(closes[-5:]) / 5
    sma_long = sum(closes[-15:]) / 15
    
    # Trading signals
    signal = {
        'symbol': symbol,
        'price': current_price,
        'rsi': rsi,
        'momentum': momentum,
        'sma_cross': sma_short > sma_long,
        'score': 0
    }
    
    # Score the opportunity
    if rsi < 30:  # Oversold
        signal['score'] += 2
    elif rsi < 40:
        signal['score'] += 1
    
    if momentum > 0.01:  # Positive momentum
        signal['score'] += 1
    
    if signal['sma_cross']:  # Golden cross
        signal['score'] += 2
    
    # Strong buy signal requires score >= 3
    signal['action'] = 'BUY' if signal['score'] >= 3 else 'HOLD'
    
    return signal

def place_order(symbol, qty, side):
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

logger.info('='*80)
logger.info('🧠 SMART STRATEGY-BASED TRADING BOT')
logger.info('='*80)
logger.info('Strategy: RSI + Momentum + SMA Crossover')
logger.info(f'Universe: {len(UNIVERSE)} stocks')
logger.info(f'Max Positions: {MAX_POSITIONS}')
logger.info(f'Profit Target: {PROFIT_TARGET*100:.1f}% | Stop Loss: {STOP_LOSS*100:.1f}%')
logger.info('='*80)

account = get_account()
if account:
    logger.info(f"💰 Starting Balance: ${float(account['cash']):,.2f}")

trade_count = 0

try:
    while True:
        now = datetime.now()
        
        # Market hours check
        if now.hour < 9 or (now.hour == 9 and now.minute < 30) or now.hour >= 16:
            logger.info('⏳ Market closed')
            time.sleep(300)
            continue
        
        logger.info(f'\n🔍 Scanning {len(UNIVERSE)} stocks for opportunities...')
        
        # Get current positions
        positions = get_positions()
        position_dict = {p['symbol']: p for p in positions}
        
        # Check exit signals for existing positions
        for symbol, pos in position_dict.items():
            unrealized_plpc = float(pos.get('unrealized_plpc', 0))
            qty = int(float(pos['qty']))
            
            if unrealized_plpc >= PROFIT_TARGET:
                logger.info(f'🎯 PROFIT TARGET HIT: {symbol} (+{unrealized_plpc*100:.2f}%)')
                result = place_order(symbol, qty, 'sell')
                if result:
                    trade_count += 1
                    logger.info(f'✅ Order {trade_count}: SOLD {qty} {symbol} at profit')
            elif unrealized_plpc <= -STOP_LOSS:
                logger.info(f'🛑 STOP LOSS HIT: {symbol} ({unrealized_plpc*100:.2f}%)')
                result = place_order(symbol, qty, 'sell')
                if result:
                    trade_count += 1
                    logger.info(f'✅ Order {trade_count}: SOLD {qty} {symbol} at loss')
        
        # Look for new opportunities if we have room
        if len(positions) < MAX_POSITIONS:
            signals = []
            for symbol in UNIVERSE:
                if symbol not in position_dict:
                    signal = analyze_symbol(symbol)
                    if signal and signal['action'] == 'BUY':
                        signals.append(signal)
            
            # Sort by score and take best opportunity
            signals.sort(key=lambda x: x['score'], reverse=True)
            
            if signals:
                best = signals[0]
                logger.info(f"\n🟢 BUY SIGNAL: {best['symbol']}")
                logger.info(f"   Price: ${best['price']:.2f}")
                logger.info(f"   RSI: {best['rsi']:.1f}")
                logger.info(f"   Momentum: {best['momentum']*100:.2f}%")
                logger.info(f"   Score: {best['score']}/5")
                
                result = place_order(best['symbol'], POSITION_SIZE, 'buy')
                if result:
                    trade_count += 1
                    logger.info(f"✅ Order {trade_count}: BOUGHT {POSITION_SIZE} {best['symbol']}")
        
        account = get_account()
        if account:
            equity = float(account['equity'])
            cash = float(account['cash'])
            logger.info(f'\n📊 Portfolio: ${equity:,.2f} | Cash: ${cash:,.2f} | Positions: {len(positions)} | Trades: {trade_count}')
        
        logger.info(f'⏱️  Next scan in {SCAN_INTERVAL}s...')
        time.sleep(SCAN_INTERVAL)
        
except KeyboardInterrupt:
    logger.info('\n\n🛑 Bot stopped')
    logger.info(f'Total trades: {trade_count}')
