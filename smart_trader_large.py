#!/usr/bin/env python3
"""Smart Strategy-Based Trading Bot - LARGE UNIVERSE"""
import os
import requests
from dotenv import load_dotenv
import logging
import time
from datetime import datetime

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# API Configuration
ALPACA_KEY = os.getenv('APCA_API_KEY_ID')
ALPACA_SECRET = os.getenv('APCA_API_SECRET_KEY')
ALPACA_BASE = 'https://paper-api.alpaca.markets/v2'

alpaca_headers = {
    'APCA-API-KEY-ID': ALPACA_KEY,
    'APCA-API-SECRET-KEY': ALPACA_SECRET
}

# EXPANDED UNIVERSE - 100+ liquid stocks across sectors
UNIVERSE = [
    # Mega Cap Tech
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'NFLX', 'AMD', 'INTC',
    'ORCL', 'CSCO', 'ADBE', 'CRM', 'AVGO', 'TXN', 'QCOM', 'AMAT', 'MU', 'NOW',
    # Finance
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'USB',
    # Healthcare
    'JNJ', 'UNH', 'PFE', 'ABBV', 'LLY', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY',
    # Consumer
    'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'COST', 'DIS', 'CMCSA',
    # Energy
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PSX', 'VLO', 'MPC', 'OXY', 'HAL',
    # Industrial
    'BA', 'HON', 'UPS', 'CAT', 'GE', 'MMM', 'LMT', 'RTX', 'DE', 'UNP',
    # ETFs for diversification
    'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'EEM', 'XLF', 'XLE', 'XLK', 'XLV',
    # Growth Stocks
    'SHOP', 'SQ', 'COIN', 'PLTR', 'SNAP', 'UBER', 'LYFT', 'ABNB', 'RBLX', 'ZM',
    # More opportunities
    'F', 'GM', 'T', 'VZ', 'PG', 'KO', 'PEP', 'WBA', 'CVS', 'MA', 'V', 'PYPL'
]

SCAN_INTERVAL = 90  # Scan every 90 seconds
MAX_POSITIONS = 10  # Increased from 5
POSITION_SIZE = 2
PROFIT_TARGET = 0.015
STOP_LOSS = 0.02

def get_account():
    r = requests.get(f'{ALPACA_BASE}/account', headers=alpaca_headers)
    return r.json() if r.status_code == 200 else None

def get_positions():
    r = requests.get(f'{ALPACA_BASE}/positions', headers=alpaca_headers)
    return r.json() if r.status_code == 200 else []

def get_bars(symbol, timeframe='5Min', limit=20):
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
    if len(prices) < period + 1:
        return 50
    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_momentum(prices):
    if len(prices) < 2:
        return 0
    return (prices[-1] - prices[0]) / prices[0]

def analyze_symbol(symbol):
    try:
        bars = get_bars(symbol)
        if not bars or len(bars) < 15:
            return None
        closes = [float(bar['c']) for bar in bars]
        current_price = closes[-1]
        rsi = calculate_rsi(closes)
        momentum = calculate_momentum(closes)
        sma_short = sum(closes[-5:]) / 5
        sma_long = sum(closes[-15:]) / 15
        
        signal = {
            'symbol': symbol,
            'price': current_price,
            'rsi': rsi,
            'momentum': momentum,
            'sma_cross': sma_short > sma_long,
            'score': 0
        }
        
        if rsi < 30:
            signal['score'] += 2
        elif rsi < 40:
            signal['score'] += 1
        if momentum > 0.01:
            signal['score'] += 1
        if signal['sma_cross']:
            signal['score'] += 2
        
        signal['action'] = 'BUY' if signal['score'] >= 3 else 'HOLD'
        return signal
    except:
        return None

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
logger.info('🌍 SMART TRADER - LARGE UNIVERSE (100+ STOCKS)')
logger.info('='*80)
logger.info(f'Universe: {len(UNIVERSE)} stocks')
logger.info(f'Max Positions: {MAX_POSITIONS}')
logger.info(f'Strategy: RSI + Momentum + SMA')
logger.info('='*80)

account = get_account()
if account:
    logger.info(f"💰 Balance: ${float(account['cash']):,.2f}")

trade_count = 0

try:
    while True:
        now = datetime.now()
        if now.hour < 9 or (now.hour == 9 and now.minute < 30) or now.hour >= 16:
            logger.info('⏳ Market closed')
            time.sleep(300)
            continue
        
        logger.info(f'\n🔍 Scanning {len(UNIVERSE)} stocks...')
        
        positions = get_positions()
        position_dict = {p['symbol']: p for p in positions}
        
        # Check exits
        for symbol, pos in position_dict.items():
            unrealized_plpc = float(pos.get('unrealized_plpc', 0))
            qty = int(float(pos['qty']))
            if unrealized_plpc >= PROFIT_TARGET:
                logger.info(f'🎯 {symbol} +{unrealized_plpc*100:.2f}% - TAKING PROFIT')
                result = place_order(symbol, qty, 'sell')
                if result:
                    trade_count += 1
                    logger.info(f'✅ SOLD {qty} {symbol}')
            elif unrealized_plpc <= -STOP_LOSS:
                logger.info(f'🛑 {symbol} {unrealized_plpc*100:.2f}% - STOP LOSS')
                result = place_order(symbol, qty, 'sell')
                if result:
                    trade_count += 1
                    logger.info(f'✅ SOLD {qty} {symbol}')
        
        # Look for opportunities
        if len(positions) < MAX_POSITIONS:
            signals = []
            for symbol in UNIVERSE:
                if symbol not in position_dict:
                    signal = analyze_symbol(symbol)
                    if signal and signal['action'] == 'BUY':
                        signals.append(signal)
            
            signals.sort(key=lambda x: x['score'], reverse=True)
            
            if signals:
                best = signals[0]
                logger.info(f"\n🟢 BUY: {best['symbol']} ${best['price']:.2f}")
                logger.info(f"   RSI:{best['rsi']:.0f} Mom:{best['momentum']*100:.1f}% Score:{best['score']}/5")
                result = place_order(best['symbol'], POSITION_SIZE, 'buy')
                if result:
                    trade_count += 1
                    logger.info(f'✅ BOUGHT {POSITION_SIZE} {best['symbol']}')
        
        account = get_account()
        if account:
            equity = float(account['equity'])
            logger.info(f'\n📊 ${equity:,.2f} | {len(positions)} pos | {trade_count} trades')
        
        logger.info(f'⏱️  Next in {SCAN_INTERVAL}s')
        time.sleep(SCAN_INTERVAL)
        
except KeyboardInterrupt:
    logger.info('\n🛑 Stopped')
    logger.info(f'Total trades: {trade_count}')
