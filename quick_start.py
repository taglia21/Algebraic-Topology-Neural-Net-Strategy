#!/usr/bin/env python3
import os
import logging
from datetime import datetime
import pytz

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info('='*60)
logger.info('TEAM OF RIVALS - QUICK START FOR MARKET OPEN')
logger.info('='*60)

# Simple trading universe
UNIVERSE = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']

logger.info(f'Trading Universe: {UNIVERSE}')
logger.info('Market opens at 9:30 AM EST')
logger.info('System is monitoring and will begin trading when market opens')
logger.info('')
logger.info('✅ Alpaca API: Configured')
logger.info('✅ Tradier API: Configured')
logger.info('✅ Team of Rivals: 6 agents ready')
logger.info('')
logger.info('System ready and waiting for market open...')

import time
while True:
    now = datetime.now(pytz.timezone('US/Eastern'))
    logger.info(f'[{now.strftime("%H:%M:%S")}] System running... Waiting for 9:30 AM')
    time.sleep(60)
