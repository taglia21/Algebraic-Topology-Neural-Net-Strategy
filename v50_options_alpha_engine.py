#!/usr/bin/env python3
"""
V50 Options Alpha Engine - Full Options Execution
Builds on V49 with actual Alpaca Options API integration
"""

import os
import sys
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

# Alpaca imports
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, 
    LimitOrderRequest,
    GetOptionContractsRequest
)
from alpaca.trading.enums import (
    OrderSide, 
    OrderType, 
    TimeInForce,
    AssetClass,
    ContractType,
    ExerciseStyle
)
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

@dataclass
class OptionsConfig:
    """Options trading configuration"""
    # Wheel strategy params
    min_iv_rank: float = 30.0
    max_iv_rank: float = 80.0
    min_premium_pct: float = 0.5  # 0.5% minimum premium
    target_delta: float = 0.30   # ~30 delta puts
    min_dte: int = 14
    max_dte: int = 45
    max_contracts: int = 5
    max_options_positions: int = 10
    
    # Risk management
    max_portfolio_risk: float = 0.20  # 20% max in options
    position_size_pct: float = 0.03   # 3% per position
    
    # Wheel-eligible stocks (high liquidity, optionable)
    wheel_stocks: List[str] = field(default_factory=lambda: [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'AMD', 'INTC',
        'SPY', 'QQQ', 'IWM', 'DIA', 'XLF', 'XLE', 'XLK',
        'BAC', 'JPM', 'WFC', 'C', 'GS',
        'F', 'GM', 'T', 'VZ', 'PFE'
    ])

class OptionsAlphaEngine:
    """V50 Options Alpha Engine with full Alpaca Options API"""
    
    def __init__(self, paper: bool = True):
        self.api_key = os.getenv('ALPACA_API_KEY', os.getenv('APCA_API_KEY_ID'))
        self.api_secret = os.getenv('ALPACA_SECRET_KEY', os.getenv('APCA_API_SECRET_KEY'))
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Alpaca API credentials not found")
        
        self.trading_client = TradingClient(
            self.api_key, 
            self.api_secret, 
            paper=paper
        )
        self.data_client = StockHistoricalDataClient(
            self.api_key,
            self.api_secret
        )
        
        self.config = OptionsConfig()
        self.logger = self._setup_logger()
        self.options_positions = []
        self.trade_history = []
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('V50_Options')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        if not logger.handlers:
            logger.addHandler(handler)
        return logger

    def get_account_info(self) -> dict:
        """Get account information"""
        account = self.trading_client.get_account()
        return {
            'equity': float(account.equity),
            'buying_power': float(account.buying_power),
            'cash': float(account.cash),
            'options_buying_power': float(getattr(account, 'options_buying_power', account.buying_power))
        }
    
    def get_option_contracts(self, symbol: str, contract_type: str = 'put', 
                             min_dte: int = 14, max_dte: int = 45) -> List[dict]:
        """
        Fetch available option contracts from Alpaca
        
        Args:
            symbol: Underlying stock symbol
            contract_type: 'put' or 'call'
            min_dte: Minimum days to expiration
            max_dte: Maximum days to expiration
        """
        try:
            # Calculate expiration date range
            min_expiry = (datetime.now() + timedelta(days=min_dte)).strftime('%Y-%m-%d')
            max_expiry = (datetime.now() + timedelta(days=max_dte)).strftime('%Y-%m-%d')
            
            # Build request
            req = GetOptionContractsRequest(
                underlying_symbols=[symbol],
                status='active',
                type=ContractType.PUT if contract_type == 'put' else ContractType.CALL,
                style=ExerciseStyle.AMERICAN,
                expiration_date_gte=min_expiry,
                expiration_date_lte=max_expiry,
                limit=100
            )
            
            # Fetch contracts
            contracts = self.trading_client.get_option_contracts(req)
            
            result = []
            for contract in contracts.option_contracts if hasattr(contracts, 'option_contracts') else contracts:
                result.append({
                    'symbol': contract.symbol,
                    'underlying': symbol,
                    'strike': float(contract.strike_price),
                    'expiration': str(contract.expiration_date),
                    'type': contract_type,
                    'open_interest': getattr(contract, 'open_interest', 0),
                    'close_price': float(getattr(contract, 'close_price', 0) or 0)
                })
            
            self.logger.info(f"Found {len(result)} {contract_type} contracts for {symbol}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error fetching option contracts for {symbol}: {e}")
            return []
    
    def get_current_price(self, symbol: str) -> float:
        """Get current stock price using latest trade"""
        try:
            from alpaca.data.requests import StockLatestBarRequest
            req = StockLatestBarRequest(symbol_or_symbols=[symbol])
            bars = self.data_client.get_stock_latest_bar(req)
            if symbol in bars:
                return float(bars[symbol].close)
            return 0.0
        except Exception as e:
            self.logger.error(f"Error getting price for {symbol}: {e}")
            return 0.0
    
    def find_best_put_to_sell(self, symbol: str) -> Optional[dict]:
        """
        Find the best put option to sell for wheel strategy
        Target: ~30 delta, good premium, adequate liquidity
        """
        try:
            # Get current stock price
            stock_price = self.get_current_price(symbol)
            if stock_price <= 0:
                return None
            
            # Target strike ~5-10% OTM for ~30 delta
            target_strike = stock_price * 0.92  # 8% OTM
            
            # Get put contracts
            contracts = self.get_option_contracts(
                symbol, 'put', 
                self.config.min_dte, 
                self.config.max_dte
            )
            
            if not contracts:
                return None
            
            # Score and rank contracts
            best_contract = None
            best_score = -float('inf')
            
            for contract in contracts:
                strike = contract['strike']
                premium = contract['close_price']
                
                # Skip if no premium data
                if premium <= 0:
                    continue
                
                # Calculate metrics
                otm_pct = (stock_price - strike) / stock_price
                premium_pct = (premium / strike) * 100
                
                # Skip if too ITM or too far OTM
                if otm_pct < 0 or otm_pct > 0.15:
                    continue
                
                # Skip if premium too low
                if premium_pct < self.config.min_premium_pct:
                    continue
                
                # Score: balance premium vs risk
                score = premium_pct * 10 - abs(otm_pct - 0.08) * 50
                
                if score > best_score:
                    best_score = score
                    best_contract = {
                        **contract,
                        'stock_price': stock_price,
                        'otm_pct': otm_pct,
                        'premium_pct': premium_pct,
                        'score': score
                    }
            
            if best_contract:
                self.logger.info(
                    f"Best put for {symbol}: Strike ${best_contract['strike']:.2f} "
                    f"({best_contract['otm_pct']*100:.1f}% OTM), "
                    f"Premium ${best_contract['close_price']:.2f} "
                    f"({best_contract['premium_pct']:.2f}%)"
                )
            
            return best_contract
            
        except Exception as e:
            self.logger.error(f"Error finding put for {symbol}: {e}")
            return None

    def sell_put(self, contract: dict, num_contracts: int = 1) -> Optional[str]:
        """
        EXECUTE: Sell a cash-secured put option
        
        This is the actual order submission to Alpaca!
        
        Args:
            contract: Contract info from find_best_put_to_sell()
            num_contracts: Number of contracts to sell
        
        Returns:
            Order ID if successful, None otherwise
        """
        try:
            option_symbol = contract['symbol']
            strike = contract['strike']
            premium = contract.get('close_price', 0)
            
            # Calculate cash needed for cash-secured put
            cash_needed = strike * 100 * num_contracts
            
            # Verify we have enough buying power
            account = self.get_account_info()
            if account['buying_power'] < cash_needed:
                self.logger.warning(
                    f"Insufficient buying power for {option_symbol}. "
                    f"Need ${cash_needed:.2f}, have ${account['buying_power']:.2f}"
                )
                return None
            
            self.logger.info(
                f"EXECUTING SELL PUT: {num_contracts}x {option_symbol} "
                f"Strike ${strike:.2f} @ ~${premium:.2f} premium"
            )
            
            # Create market order to SELL the put (open short position)
            order_request = MarketOrderRequest(
                symbol=option_symbol,
                qty=num_contracts,
                side=OrderSide.SELL,  # Sell to open
                type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY
            )
            
            # SUBMIT THE ORDER!
            order = self.trading_client.submit_order(order_request)
            
            self.logger.info(
                f"ORDER SUBMITTED! ID: {order.id} | "
                f"Status: {order.status} | "
                f"Symbol: {option_symbol}"
            )
            
            # Record trade
            self.trade_history.append({
                'timestamp': datetime.now().isoformat(),
                'action': 'sell_put',
                'symbol': option_symbol,
                'underlying': contract.get('underlying'),
                'strike': strike,
                'expiration': contract.get('expiration'),
                'contracts': num_contracts,
                'premium': premium,
                'order_id': str(order.id),
                'status': str(order.status)
            })
            
            return str(order.id)
            
        except Exception as e:
            self.logger.error(f"SELL PUT FAILED: {e}")
            return None
    
    def sell_covered_call(self, symbol: str, num_contracts: int = 1) -> Optional[str]:
        """
        EXECUTE: Sell a covered call against existing stock position
        
        Args:
            symbol: Underlying stock symbol (must own 100+ shares)
            num_contracts: Number of contracts to sell
        
        Returns:
            Order ID if successful, None otherwise
        """
        try:
            # Verify we have shares
            positions = self.trading_client.get_all_positions()
            stock_position = None
            for pos in positions:
                if pos.symbol == symbol:
                    stock_position = pos
                    break
            
            if not stock_position:
                self.logger.warning(f"No position in {symbol} for covered call")
                return None
            
            shares_owned = int(float(stock_position.qty))
            if shares_owned < num_contracts * 100:
                self.logger.warning(
                    f"Not enough shares for covered call. "
                    f"Have {shares_owned}, need {num_contracts * 100}"
                )
                return None
            
            # Find best call to sell (slightly OTM)
            stock_price = self.get_current_price(symbol)
            target_strike = stock_price * 1.05  # 5% OTM
            
            contracts = self.get_option_contracts(symbol, 'call', 14, 45)
            
            # Find closest strike above current price
            best_call = None
            for contract in contracts:
                if contract['strike'] >= target_strike:
                    if best_call is None or contract['strike'] < best_call['strike']:
                        best_call = contract
            
            if not best_call:
                self.logger.warning(f"No suitable call contracts found for {symbol}")
                return None
            
            self.logger.info(
                f"EXECUTING COVERED CALL: {num_contracts}x {best_call['symbol']} "
                f"Strike ${best_call['strike']:.2f}"
            )
            
            # Sell the call
            order_request = MarketOrderRequest(
                symbol=best_call['symbol'],
                qty=num_contracts,
                side=OrderSide.SELL,
                type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY
            )
            
            order = self.trading_client.submit_order(order_request)
            
            self.logger.info(f"COVERED CALL ORDER SUBMITTED! ID: {order.id}")
            
            self.trade_history.append({
                'timestamp': datetime.now().isoformat(),
                'action': 'sell_call',
                'symbol': best_call['symbol'],
                'underlying': symbol,
                'strike': best_call['strike'],
                'expiration': best_call.get('expiration'),
                'contracts': num_contracts,
                'order_id': str(order.id)
            })
            
            return str(order.id)
            
        except Exception as e:
            self.logger.error(f"COVERED CALL FAILED: {e}")
            return None

    def get_options_positions(self) -> List[dict]:
        """Get current options positions"""
        try:
            positions = self.trading_client.get_all_positions()
            options_pos = []
            for pos in positions:
                # Options symbols are longer and contain dates
                if len(pos.symbol) > 10:  # Likely an option
                    options_pos.append({
                        'symbol': pos.symbol,
                        'qty': int(float(pos.qty)),
                        'side': 'short' if float(pos.qty) < 0 else 'long',
                        'market_value': float(pos.market_value),
                        'cost_basis': float(pos.cost_basis),
                        'unrealized_pl': float(pos.unrealized_pl)
                    })
            return options_pos
        except Exception as e:
            self.logger.error(f"Error getting options positions: {e}")
            return []
    
    def run_wheel_strategy(self) -> dict:
        """
        Execute the wheel strategy:
        1. Scan wheel-eligible stocks
        2. Find best puts to sell
        3. Execute sell orders
        4. Track positions
        """
        results = {
            'scanned': 0,
            'signals': 0,
            'orders_placed': 0,
            'orders': []
        }
        
        try:
            account = self.get_account_info()
            self.logger.info(f"Account Equity: ${account['equity']:,.2f}")
            self.logger.info(f"Buying Power: ${account['buying_power']:,.2f}")
            
            # Check existing options positions
            current_options = self.get_options_positions()
            self.logger.info(f"Current options positions: {len(current_options)}")
            
            if len(current_options) >= self.config.max_options_positions:
                self.logger.info("Max options positions reached")
                return results
            
            # Calculate available capital for new positions
            max_options_capital = account['equity'] * self.config.max_portfolio_risk
            current_options_value = sum(abs(p['market_value']) for p in current_options)
            available_capital = max_options_capital - current_options_value
            
            self.logger.info(f"Available capital for options: ${available_capital:,.2f}")
            
            # Scan wheel stocks
            for symbol in self.config.wheel_stocks:
                results['scanned'] += 1
                
                # Find best put to sell
                best_put = self.find_best_put_to_sell(symbol)
                
                if best_put:
                    results['signals'] += 1
                    
                    # Calculate position size
                    strike = best_put['strike']
                    cash_per_contract = strike * 100
                    position_capital = account['equity'] * self.config.position_size_pct
                    num_contracts = min(
                        self.config.max_contracts,
                        int(position_capital / cash_per_contract),
                        int(available_capital / cash_per_contract)
                    )
                    
                    if num_contracts >= 1:
                        # EXECUTE THE TRADE!
                        order_id = self.sell_put(best_put, num_contracts)
                        
                        if order_id:
                            results['orders_placed'] += 1
                            results['orders'].append({
                                'symbol': best_put['symbol'],
                                'underlying': symbol,
                                'strike': strike,
                                'contracts': num_contracts,
                                'order_id': order_id
                            })
                            
                            # Update available capital
                            available_capital -= cash_per_contract * num_contracts
                            
                            # Stop if we've placed enough orders
                            if results['orders_placed'] >= 3:  # Max 3 new positions per scan
                                break
            
            self.logger.info(
                f"Wheel scan complete: {results['scanned']} scanned, "
                f"{results['signals']} signals, {results['orders_placed']} orders placed"
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Wheel strategy error: {e}")
            return results

    async def run_continuous(self, interval_seconds: int = 60):
        """Run continuous options scanning and trading"""
        self.logger.info("="*60)
        self.logger.info("V50 OPTIONS ALPHA ENGINE")
        self.logger.info("Full Alpaca Options API Integration")
        self.logger.info("="*60)
        
        scan_count = 0
        
        while True:
            try:
                scan_count += 1
                self.logger.info(f"\n--- OPTIONS SCAN #{scan_count} ---")
                
                # Run wheel strategy
                results = self.run_wheel_strategy()
                
                # Report
                account = self.get_account_info()
                self.logger.info(
                    f"Equity: ${account['equity']:,.2f} | "
                    f"Orders: {results['orders_placed']} | "
                    f"Next scan in {interval_seconds}s"
                )
                
                await asyncio.sleep(interval_seconds)
                
            except KeyboardInterrupt:
                self.logger.info("Shutting down...")
                break
            except Exception as e:
                self.logger.error(f"Scan error: {e}")
                await asyncio.sleep(interval_seconds)


async def main():
    import argparse
    parser = argparse.ArgumentParser(description='V50 Options Alpha Engine')
    parser.add_argument('--test', action='store_true', help='Run single test scan')
    parser.add_argument('--trade', action='store_true', help='Run continuous trading')
    parser.add_argument('--interval', type=int, default=60, help='Scan interval in seconds')
    args = parser.parse_args()
    
    engine = OptionsAlphaEngine(paper=True)
    
    if args.test:
        print("\n=== V50 OPTIONS ENGINE TEST ===")
        account = engine.get_account_info()
        print(f"Account Equity: ${account['equity']:,.2f}")
        print(f"Buying Power: ${account['buying_power']:,.2f}")
        
        # Test getting contracts
        print("\nTesting option contract fetch...")
        contracts = engine.get_option_contracts('AAPL', 'put', 14, 45)
        print(f"Found {len(contracts)} AAPL put contracts")
        
        if contracts:
            print(f"Sample contract: {contracts[0]}")
        
        # Test finding best put
        print("\nFinding best put to sell...")
        best_put = engine.find_best_put_to_sell('AAPL')
        if best_put:
            print(f"Best put: {best_put['symbol']}")
            print(f"Strike: ${best_put['strike']:.2f}")
            print(f"Premium: ${best_put.get('close_price', 0):.2f}")
        
        print("\n[TEST COMPLETE] - Use --trade to enable live trading")
        
    elif args.trade:
        await engine.run_continuous(args.interval)
    else:
        print("Use --test or --trade")


if __name__ == '__main__':
    asyncio.run(main())
