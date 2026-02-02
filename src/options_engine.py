import os
import asyncio
import logging
import requests
from datetime import datetime, timedelta
import pytz
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class TradierOptionsEngine:
    """Options trading engine for Tradier paper trading"""
    def __init__(self):
        self.api_token = os.getenv('TRADIER_API_TOKEN')
        self.account_id = os.getenv('TRADIER_ACCOUNT_ID')
        self.base_url = 'https://sandbox.tradier.com/v1'
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_token}',
            'Accept': 'application/json'
        })
        self.est = pytz.timezone('US/Eastern')
        
    def get_options_chain(self, symbol, expiration=None):
        """Get options chain for symbol"""
        if not expiration:
            # Get next Friday expiration
            today = datetime.now(self.est)
            days_ahead = 4 - today.weekday()  # Friday is 4
            if days_ahead <= 0:
                days_ahead += 7
            expiration = (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
            
        try:
            response = self.session.get(
                f'{self.base_url}/markets/options/chains',
                params={'symbol': symbol, 'expiration': expiration}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f'Error fetching options chain: {e}')
            return None
            
    def get_quote(self, symbols):
        """Get quote for options symbols"""
        try:
            response = self.session.get(
                f'{self.base_url}/markets/quotes',
                params={'symbols': ','.join(symbols) if isinstance(symbols, list) else symbols}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f'Error fetching quote: {e}')
            return None
            
    def find_optimal_options(self, symbol, strategy='wheel'):
        """Find optimal options for strategy"""
        chain = self.get_options_chain(symbol)
        if not chain:
            return None
            
        # Get underlying price
        quote = self.get_quote(symbol)
        if not quote:
            return None
            
        underlying_price = quote['quotes']['quote']['last']
        
        # For wheel strategy, sell cash-secured puts
        if strategy == 'wheel':
            # Find put ~0.3 delta (30% OTM)
            target_strike = underlying_price * 0.97
            
            puts = [opt for opt in chain.get('options', {}).get('option', []) 
                   if opt.get('option_type') == 'put']
            
            # Find closest strike to target
            optimal = min(puts, 
                         key=lambda x: abs(x.get('strike', 0) - target_strike))
            
            return optimal
            
        return None
        
    def execute_option_trade(self, symbol, side, quantity, option_symbol=None):
        """Execute options trade on Tradier"""
        try:
            data = {
                'class': 'option',
                'symbol': symbol,
                'option_symbol': option_symbol,
                'side': side,  # 'buy_to_open', 'sell_to_open', etc.
                'quantity': quantity,
                'type': 'market',
                'duration': 'day'
            }
            
            response = self.session.post(
                f'{self.base_url}/accounts/{self.account_id}/orders',
                data=data
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f'Options trade executed: {result}')
            return result
        except Exception as e:
            logger.error(f'Options trade execution error: {e}')
            return None
            
    def get_positions(self):
        """Get current options positions"""
        try:
            response = self.session.get(
                f'{self.base_url}/accounts/{self.account_id}/positions'
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f'Error fetching positions: {e}')
            return None
            
    async def run_wheel_strategy(self, symbols, capital_per_trade=1000):
        """Run the Wheel strategy"""
        logger.info(f'Running Wheel strategy on {len(symbols)} symbols')
        
        for symbol in symbols:
            # Get current positions
            positions = self.get_positions()
            
            # Check if we have shares (from assignment)
            has_shares = False
            if positions and 'positions' in positions:
                for pos in positions.get('positions', {}).get('position', []):
                    if pos.get('symbol') == symbol:
                        has_shares = True
                        # Sell covered calls
                        logger.info(f'Have shares of {symbol}, selling covered call')
                        # Implementation would go here
                        break
            
            if not has_shares:
                # Sell cash-secured puts
                optimal_put = self.find_optimal_options(symbol, 'wheel')
                if optimal_put:
                    contracts = capital_per_trade // (optimal_put.get('strike', 1) * 100)
                    if contracts > 0:
                        logger.info(f'Selling {contracts} puts on {symbol}')
                        # Execute the trade
                        # self.execute_option_trade(
                        #     symbol, 'sell_to_open', contracts, 
                        #     optimal_put.get('symbol')
                        # )
                        
            await asyncio.sleep(1)  # Rate limiting
