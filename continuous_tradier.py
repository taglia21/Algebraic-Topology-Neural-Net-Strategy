#!/usr/bin/env python3
"""Continuous Tradier Trading Bot - Actively trades on Tradier Sandbox"""
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

# Tradier Configuration
TRADIER_TOKEN = os.getenv('TRADIER_API_TOKEN')
TRADIER_ACCOUNT = os.getenv('TRADIER_ACCOUNT_ID')
TRADIER_BASE = 'https://sandbox.tradier.com/v1'

tradier_headers = {
    'Authorization': f'Bearer {TRADIER_TOKEN}',
    'Accept': 'application/json'
}

# Trading symbols - different from Alpaca to diversify
SYMBOLS = ['MSFT', 'AMZN', 'TSLA', 'NFLX', 'DIS', 'BABA', 'AMD', 'INTC', 'BA', 'V']
TRADE_INTERVAL = 45  # Trade every 45 seconds
MIN_QTY = 1
MAX_QTY = 3

def get_balances():
    """Get account balances"""
    r = requests.get(
        f'{TRADIER_BASE}/accounts/{TRADIER_ACCOUNT}/balances',
        headers=tradier_headers
    )
    return r.json() if r.status_code == 200 else None

def get_positions():
    """Get current positions"""
    r = requests.get(
        f'{TRADIER_BASE}/accounts/{TRADIER_ACCOUNT}/positions',
        headers=tradier_headers
    )
    data = r.json() if r.status_code == 200 else {}
    positions = data.get('positions', {})
    if positions == 'null' or not positions:
        return []
    position_list = positions.get('position', [])
    return position_list if isinstance(position_list, list) else [position_list]

def place_order(symbol, qty, side):
    """Place market order on Tradier"""
    order_data = {
        'class': 'equity',
        'symbol': symbol,
        'side': side.lower(),
        'quantity': str(qty),
        'type': 'market',
        'duration': 'day'
    }
    r = requests.post(
        f'{TRADIER_BASE}/accounts/{TRADIER_ACCOUNT}/orders',
        headers=tradier_headers,
        data=order_data
    )
    return r.json() if r.status_code == 200 else None

logger.info('='*70)
logger.info('🚀 TRADIER CONTINUOUS TRADING BOT - STARTING')
logger.info('='*70)

# Get initial balances
balances = get_balances()
if balances and 'balances' in balances:
    bal = balances['balances']
    logger.info(f"💰 Total Value: ${float(bal.get('total_equity', 0)):,.2f}")
    logger.info(f"💰 Buying Power: ${float(bal.get('option_buying_power', 0)):,.2f}")
else:
    logger.warning('⚠️ Could not get balances (sandbox limitation)')

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
            time.sleep(300)
            continue
        
        # Get current positions
        positions = get_positions()
        position_symbols = {p['symbol']: p for p in positions}
        
        logger.info(f'\n📊 TRADIER Positions: {len(positions)}')
        
        # Simple strategy: buy if we don't have it, sell randomly for profit taking
        if random.random() < 0.3 and positions:  # 30% chance to sell a position
            position = random.choice(positions)
            symbol = position['symbol']
            qty = int(float(position['quantity']))
            logger.info(f'🟥 SELL Signal: {symbol} ({qty} shares)')
            result = place_order(symbol, qty, 'sell')
            if result:
                trade_count += 1
                logger.info(f'✅ TRADIER Order {trade_count}: SOLD {qty} {symbol}')
            time.sleep(2)
        
        # Look for buy opportunities
        if len(positions) < 7:  # Keep max 7 positions
            symbol = random.choice(SYMBOLS)
            if symbol not in position_symbols:
                qty = random.randint(MIN_QTY, MAX_QTY)
                logger.info(f'🟩 BUY Signal: {symbol} ({qty} shares)')
                result = place_order(symbol, qty, 'buy')
                if result:
                    trade_count += 1
                    logger.info(f'✅ TRADIER Order {trade_count}: BOUGHT {qty} {symbol}')
                time.sleep(2)
        
        logger.info(f'📊 Total TRADIER trades: {trade_count}')
        logger.info(f'⏱️  Next scan in {TRADE_INTERVAL}s...')
        time.sleep(TRADE_INTERVAL)
        
except KeyboardInterrupt:
    logger.info('\n\n🛑 Tradier bot stopped')
    logger.info(f'Total trades: {trade_count}')
except Exception as e:
    logger.error(f'❌ Error: {e}')
    import traceback
    logger.error(traceback.format_exc())
