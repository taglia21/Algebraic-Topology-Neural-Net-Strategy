#!/usr/bin/env python3
import os
import requests
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tradier API Configuration
TRADIER_TOKEN = os.getenv('TRADIER_API_TOKEN')
TRADIER_ACCOUNT = os.getenv('TRADIER_ACCOUNT_ID')
BASE_URL = 'https://sandbox.tradier.com/v1'

headers = {
    'Authorization': f'Bearer {TRADIER_TOKEN}',
    'Accept': 'application/json'
}

logger.info('='*60)
logger.info('LIVE TRADING BOT - TRADIER INTEGRATION')
logger.info('='*60)
logger.info(f'Account ID: {TRADIER_ACCOUNT}')
logger.info(f'Base URL: {BASE_URL}')
logger.info('='*60)

# Test 1: Get account profile to verify authentication
try:
    response = requests.get(
        f'{BASE_URL}/user/profile',
        headers=headers
    )
    if response.status_code == 200:
        profile = response.json()
        logger.info('✓ API Authentication Successful!')
        logger.info(f'Profile: {profile}')
    else:
        logger.error(f'Authentication failed: {response.status_code} - {response.text}')
except Exception as e:
    logger.error(f'Error getting profile: {e}')

# Test 2: Get account balances
try:
    response = requests.get(
        f'{BASE_URL}/accounts/{TRADIER_ACCOUNT}/balances',
        headers=headers
    )
    if response.status_code == 200:
        balances = response.json()
        logger.info('✓ Account Balances Retrieved!')
        logger.info(f'Balances: {balances}')
    else:
        logger.error(f'Failed to get balances: {response.status_code} - {response.text}')
except Exception as e:
    logger.error(f'Error getting balances: {e}')

# Test 3: Place a simple BUY order for 1 share of AAPL
logger.info('\nAttempting to place TEST ORDER: BUY 1 share of AAPL...')

try:
    order_data = {
        'class': 'equity',
        'symbol': 'AAPL',
        'side': 'buy',
        'quantity': '1',
        'type': 'market',
        'duration': 'day'
    }
    
    response = requests.post(
        f'{BASE_URL}/accounts/{TRADIER_ACCOUNT}/orders',
        headers=headers,
        data=order_data
    )
    
    if response.status_code == 200:
        order = response.json()
        logger.info('✓✓✓ ORDER PLACED SUCCESSFULLY! ✓✓✓')
        logger.info(f'Order Response: {order}')
        logger.info('='*60)
        logger.info('THE BOT CAN TRADE! System is operational!')
        logger.info('='*60)
    else:
        logger.error(f'Order failed: {response.status_code} - {response.text}')
except Exception as e:
    logger.error(f'Error placing order: {e}')

logger.info('\nBot test complete.')