import os
import asyncio
import logging
from datetime import datetime, time
import pytz
import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from config.universe import UNIVERSES
from config.tda_strategy import demo_tda_strategy
from src.team_of_rivals import TeamOfRivals
from src.discord_bot import DiscordBot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self):
        self.api = tradeapi.REST(
            os.getenv('APCA_API_KEY_ID'),
            os.getenv('APCA_API_SECRET_KEY'),
            os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')
        )
        self.universe = UNIVERSES['UNIVERSEEOF']
        self.team = TeamOfRivals()
        self.discord = None
        self.est = pytz.timezone('US/Eastern')
        
    async def initialize(self):
        """Initialize Discord bot and team"""
        self.discord = DiscordBot(self.team)
        await self.discord.start(os.getenv('DISCORD_BOT_TOKEN'))
        
    def get_market_data(self, symbols, timeframe='1Min', limit=100):
        """Get real-time market data from Alpaca"""
        try:
            barsets = self.api.get_bars(
                symbols,
                timeframe,
                limit=limit
            ).df
            return barsets
        except Exception as e:
            logger.error(f'Error fetching market data: {e}')
            return None
            
    def analyze_with_tda(self, market_data):
        """Run TDA analysis on market data"""
        try:
            signals = demo_tda_strategy(market_data)
            return signals
        except Exception as e:
            logger.error(f'TDA analysis error: {e}')
            return None
            
    async def execute_trade(self, symbol, side, qty, signal_strength):
        """Execute trade with team veto mechanism"""
        # Get team consensus with veto power
        trade_proposal = {
            'symbol': symbol,
            'side': side,
            'qty': qty,
            'signal_strength': signal_strength,
            'timestamp': datetime.now(self.est)
        }
        
        # Team deliberation
        approved = await self.team.deliberate_trade(trade_proposal)
        
        if not approved:
            logger.info(f'Trade vetoed by team: {symbol} {side} {qty}')
            if self.discord:
                await self.discord.log_veto(trade_proposal)
            return None
            
        # Execute approved trade
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='market',
                time_in_force='day'
            )
            logger.info(f'Trade executed: {order}')
            if self.discord:
                await self.discord.log_trade(order)
            return order
        except Exception as e:
            logger.error(f'Trade execution error: {e}')
            return None
            
    def calculate_position_size(self, symbol, signal_strength, account_value):
        """Calculate position size with risk management"""
        # Risk 1-2% of account per trade based on signal strength
        risk_pct = 0.01 + (0.01 * signal_strength)  # 1-2%
        position_value = account_value * risk_pct
        
        # Get current price
        try:
            quote = self.api.get_latest_trade(symbol)
            price = quote.price
            qty = int(position_value / price)
            return max(1, qty)  # At least 1 share
        except Exception as e:
            logger.error(f'Position sizing error: {e}')
            return 1
            
    async def run_trading_cycle(self):
        """Main trading cycle"""
        logger.info('Starting trading cycle')
        
        # Get account info
        account = self.api.get_account()
        account_value = float(account.equity)
        
        # Get market data for universe
        market_data = self.get_market_data(self.universe)
        if market_data is None:
            logger.error('Failed to get market data')
            return
            
        # Analyze with TDA
        signals = self.analyze_with_tda(market_data)
        if signals is None:
            return
            
        # Execute trades on signals
        for symbol, signal in signals.items():
            if abs(signal['strength']) > 0.3:  # Threshold
                side = 'buy' if signal['strength'] > 0 else 'sell'
                qty = self.calculate_position_size(
                    symbol, 
                    abs(signal['strength']),
                    account_value
                )
                await self.execute_trade(symbol, side, qty, signal['strength'])
                
    async def morning_standup(self):
        """Run daily 9am standup meeting"""
        logger.info('Starting morning standup')
        if self.discord:
            await self.discord.run_standup()
            
    async def schedule_tasks(self):
        """Schedule recurring tasks"""
        while True:
            now = datetime.now(self.est)
            
            # Morning standup at 9am EST
            if now.hour == 9 and now.minute == 0:
                await self.morning_standup()
                await asyncio.sleep(60)  # Wait a minute
                
            # Trading cycles during market hours (9:30am - 4pm)
            if now.hour >= 9 and now.hour < 16:
                if now.hour == 9 and now.minute >= 30:
                    await self.run_trading_cycle()
                elif now.hour > 9:
                    await self.run_trading_cycle()
                await asyncio.sleep(300)  # Run every 5 minutes
            else:
                await asyncio.sleep(60)  # Check every minute outside market hours
                
    async def start(self):
        """Start the trading bot"""
        logger.info('Initializing Trading Bot')
        await self.initialize()
        logger.info('Starting scheduled tasks')
        await self.schedule_tasks()

if __name__ == '__main__':
    bot = TradingBot()
    asyncio.run(bot.start())
