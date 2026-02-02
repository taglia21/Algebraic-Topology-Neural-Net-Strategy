#!/usr/bin/env python3
"""Continuous Trading Bot - Actively trades throughout market hours"""
import os
import requests
from dotenv import load_dotenv
import logging
import time
from datetime import datetime
import random

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
ALPACA_KEY = os.getenv('APCA_API_KEY_ID')
ALPACA_SECRET = os.getenv('APCA_API_SECRET_KEY')
ALPACA_BASE = 'https://paper-api.alpaca.markets/v2'

alpaca_headers = {
    'APCA-API-KEY-ID': ALPACA_KEY,
    'APCA-API-SECRET-KEY': ALPACA_SECRET
}

# Trading symbols
SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'SPY', 'QQQ', 'IWM']
TRADE_INTERVAL = 60  # Trade every 60 seconds
MIN_QTY = 1
MAX_QTY = 5

def get_account():
    """Get account information"""
    r = requests.get(f'{ALPACA_BASE}/account', headers=alpaca_headers)
    return r.json() if r.status_code == 200 else None

def get_positions():
    """Get current positions"""
    r = requests.get(f'{ALPACA_BASE}/positions', headers=alpaca_headers)
    return r.json() if r.status_code == 200 else []

def get_quote(symbol):
    """Get latest quote for symbol"""
    r = requests.get(f'{ALPACA_BASE}/stocks/{symbol}/quotes/latest', headers=alpaca_headers)
    return r.json() if r.status_code == 200 else None

def place_order(symbol, qty, side):
    """Place market order"""
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

def should_buy(symbol, positions):
    """Simple strategy: buy if we don't have position"""
    return symbol not in [p['symbol'] for p in positions]

def should_sell(symbol, position):
    """Simple strategy: sell if profit > 1% or loss > 2%"""
    unrealized_plpc = float(position.get('unrealized_plpc', 0))
    return unrealized_plpc > 0.01 or unrealized_plpc < -0.02

logger.info('='*70)
logger.info('🚀 CONTINUOUS TRADING BOT - STARTING')
logger.info('='*70)

# Get initial account status
account = get_account()
if account:
    logger.info(f"💰 Account: ${float(account['cash']):,.2f} cash")
    logger.info(f"💰 Buying Power: ${float(account['buying_power']):,.2f}")
else:
    logger.error('❌ Failed to get account info')
    exit(1)

logger.info(f"🔄 Trading {len(SYMBOLS)} symbols every {TRADE_INTERVAL}s")
logger.info(f"📊 Symbols: {', '.join(SYMBOLS[:5])}...")
logger.info('='*70)

trade_count = 0

try:
    while True:
        now = datetime.now()
        hour = now.hour
        
        # Trading hours: 9:30 AM - 4:00 PM EST
        if hour < 9 or (hour == 9 and now.minute < 30) or hour >= 16:
            logger.info('⏳ Market closed - waiting...')
            time.sleep(300)  # Check every 5 minutes
            continue
        
        # Get current positions
        positions = get_positions()
        position_symbols = {p['symbol']: p for p in positions}
        
        logger.info(f'\n📊 Current Positions: {len(positions)}')
        
        # Check for sells first
        for symbol, position in position_symbols.items():
            if should_sell(symbol, position):
                qty = int(float(position['qty']))
                logger.info(f'🟥 SELL Signal: {symbol} ({qty} shares) - P/L: {float(position["unrealized_plpc"])*100:.2f}%')
                result = place_order(symbol, qty, 'sell')
                if result:
                    trade_count += 1
                    logger.info(f'✅ Order {trade_count}: SOLD {qty} {symbol}')
                time.sleep(2)
        
        # Look for buy opportunities
        for symbol in random.sample(SYMBOLS, min(3, len(SYMBOLS))):
            if should_buy(symbol, positions) and len(positions) < 8:
                qty = random.randint(MIN_QTY, MAX_QTY)
                logger.info(f'🟩 BUY Signal: {symbol} ({qty} shares)')
                result = place_order(symbol, qty, 'buy')
                if result:
                    trade_count += 1
                    logger.info(f'✅ Order {trade_count}: BOUGHT {qty} {symbol}')
                time.sleep(2)
                break  # Only one buy per cycle
        
        # Refresh account
        account = get_account()
        if account:
            cash = float(account['cash'])
            equity = float(account['equity'])
            logger.info(f'\n💵 Portfolio: ${equity:,.2f} | Cash: ${cash:,.2f} | Trades: {trade_count}')
        
        logger.info(f'⏱️  Next scan in {TRADE_INTERVAL}s...')
        time.sleep(TRADE_INTERVAL)
        
except KeyboardInterrupt:
    logger.info('\n\n🛑 Bot stopped by user')
    logger.info(f'Total trades executed: {trade_count}')
except Exception as e:
    logger.error(f'\n❌ Error: {e}')
    import traceback
    logger.error(traceback.format_exc())
