#!/usr/bin/env python3
import os
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get credentials
api_key = os.getenv('APCA_API_KEY_ID')
api_secret = os.getenv('APCA_API_SECRET_KEY')
base_url = os.getenv('APCA_API_BASE_URL')

logger.info('='*60)
logger.info('ALPACA PAPER TRADING TEST')
logger.info('='*60)
logger.info(f'API Key: {api_key}')
logger.info(f'Secret (first 10): {api_secret[:10]}...')
logger.info(f'Base URL: {base_url}')
logger.info('='*60)

# Initialize API with explicit paper=True parameter
try:
    api = tradeapi.REST(
        key_id=api_key,
        secret_key=api_secret,
        base_url=base_url
    )
    
    logger.info('\n✓ API object created')
    
    # Test 1: Get account
    account = api.get_account()
    logger.info(f'\n✓ Account Retrieved!')
    logger.info(f'Account Status: {account.status}')
    logger.info(f'Account Number: {account.account_number}')
    logger.info(f'Buying Power: ${account.buying_power}')
    logger.info(f'Cash: ${account.cash}')
    
    # Test 2: Get current positions
    positions = api.list_positions()
    logger.info(f'\nCurrent Positions: {len(positions)}')
    
    # Test 3: Place a market order for 1 share of AAPL
    logger.info('\n' + '='*60)
    logger.info('PLACING TEST ORDER: BUY 1 AAPL (Market Order)')
    logger.info('='*60)
    
    order = api.submit_order(
        symbol='AAPL',
        qty=1,
        side='buy',
        type='market',
        time_in_force='day'
    )
    
    logger.info(f'\n✓✓✓ ORDER PLACED ON ALPACA! ✓✓✓')
    logger.info(f'Order ID: {order.id}')
    logger.info(f'Symbol: {order.symbol}')
    logger.info(f'Qty: {order.qty}')
    logger.info(f'Side: {order.side}')
    logger.info(f'Status: {order.status}')
    logger.info('='*60)
    logger.info('SUCCESS! Alpaca paper trading is OPERATIONAL!')
    logger.info('='*60)
    
except Exception as e:
    logger.error(f'\n❌ ERROR: {type(e).__name__}')
    logger.error(f'Message: {str(e)}')
    import traceback
    logger.error(f'\nFull traceback:\n{traceback.format_exc()}')
