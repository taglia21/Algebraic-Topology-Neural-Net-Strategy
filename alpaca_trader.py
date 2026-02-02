#!/usr/bin/env python3
import os
import requests
from dotenv import load_dotenv
import logging
import json

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALPACA_KEY = os.getenv('APCA_API_KEY_ID')
ALPACA_SECRET = os.getenv('APCA_API_SECRET_KEY')
BASE_URL = 'https://paper-api.alpaca.markets/v2'

headers = {
    'APCA-API-KEY-ID': ALPACA_KEY,
    'APCA-API-SECRET-KEY': ALPACA_SECRET
}

logger.info('='*70)
logger.info('ALPACA PAPER TRADING - LIVE TEST')
logger.info('='*70)
logger.info(f'Account: ATNN Paper Trading')
logger.info(f'Base URL: {BASE_URL}')
logger.info('='*70)

# Test 1: Get account
response = requests.get(f'{BASE_URL}/account', headers=headers)
if response.status_code == 200:
    account = response.json()
    logger.info('\n✓ Account Retrieved Successfully!')
    logger.info(f'Account Number: {account["account_number"]}')
    logger.info(f'Cash: ${account["cash"]}')
    logger.info(f'Buying Power: ${account["buying_power"]}')
    logger.info(f'Status: {account["status"]}')
else:
    logger.error(f'Account fetch failed: {response.status_code} - {response.text}')

# Test 2: Place MARKET ORDER for 1 share of AAPL
logger.info('\n' + '='*70)
logger.info('PLACING LIVE ORDER ON ALPACA: BUY 1 AAPL')
logger.info('='*70)

order_data = {
    'symbol': 'AAPL',
    'qty': 1,
    'side': 'buy',
    'type': 'market',
    'time_in_force': 'day'
}

response = requests.post(
    f'{BASE_URL}/orders',
    headers={**headers, 'Content-Type': 'application/json'},
    json=order_data
)

if response.status_code in [200, 201]:
    order = response.json()
    logger.info('\n✓✓✓ ORDER SUCCESSFULLY PLACED ON ALPACA! ✓✓✓')
    logger.info(f'Order ID: {order["id"]}')
    logger.info(f'Symbol: {order["symbol"]}')
    logger.info(f'Qty: {order["qty"]}')
    logger.info(f'Side: {order["side"]}')
    logger.info(f'Type: {order["type"]}')
    logger.info(f'Status: {order["status"]}')
    logger.info('='*70)
    logger.info('✅ ALPACA PAPER TRADING IS OPERATIONAL!')
    logger.info('✅ YOU WILL SEE THIS ORDER IN YOUR ALPACA DASHBOARD!')
    logger.info('='*70)
else:
    logger.error(f'Order failed: {response.status_code} - {response.text}')

logger.info('\nTest complete. Check your Alpaca dashboard for the order!')
