#!/usr/bin/env python3
"""
Team of Rivals - Multi-Agent Trading System
Integrating TDA Strategy with Neural Networks and Discord
"""

import asyncio
import logging
import os
from src.trading_bot import TradingBot
from src.ml_retraining import MLRetrainingScheduler
import alpaca_trade_api as tradeapi

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    logger.info('='*60)
    logger.info('TEAM OF RIVALS - TRADING SYSTEM STARTUP')
    logger.info('='*60)
    
    # Initialize trading bot
    bot = TradingBot()
    
    # Initialize ML retraining
    ml_scheduler = MLRetrainingScheduler(bot.api)
    
    # Start all tasks concurrently
    try:
        await asyncio.gather(
            bot.start(),
            ml_scheduler.schedule_retraining(bot.universe)
        )
    except KeyboardInterrupt:
        logger.info('Shutdown requested')
    except Exception as e:
        logger.error(f'Fatal error: {e}')
    finally:
        logger.info('System shutting down')

if __name__ == '__main__':
    asyncio.run(main())
