#!/usr/bin/env python3
"""AGGRESSIVE AUTO TRADER - Runs every hour during market hours
Executes high-conviction trades based on momentum and mean reversion signals
"""

import os
import logging
import argparse
from datetime import datetime, timedelta
import pytz
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

import alpaca_trade_api as tradeapi

class AggressiveAutoTrader:
    """Aggressive automated trading system"""
    
    # High-momentum stocks to trade
    MOMENTUM_UNIVERSE = [
        'NVDA', 'TSLA', 'AMD', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
        'NFLX', 'CRM', 'SHOP', 'SQ', 'PYPL', 'COIN', 'MSTR', 'PLTR'
    ]
    
    # Leveraged ETFs for max gains
    LEVERAGED_ETFS = [
        'TQQQ', 'SOXL', 'FNGU', 'TECL', 'SPXL', 'UPRO', 'UDOW', 'TNA'
    ]
    
    # Pairs for stat arb
    PAIRS = [
        ('XOM', 'CVX'), ('JPM', 'BAC'), ('KO', 'PEP'),
        ('V', 'MA'), ('HD', 'LOW'), ('MSFT', 'GOOGL')
    ]
    
    def __init__(self, paper=True):
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.api_secret = os.getenv('ALPACA_SECRET_KEY')
        base_url = 'https://paper-api.alpaca.markets' if paper else 'https://api.alpaca.markets'
        self.api = tradeapi.REST(self.api_key, self.api_secret, base_url, api_version='v2')
        self.paper = paper
        self.est = pytz.timezone('US/Eastern')
        logger.info(f"AutoTrader initialized ({'PAPER' if paper else 'LIVE'})")
        
    def is_market_open(self):
        """Check if market is open"""
        try:
            clock = self.api.get_clock()
            return clock.is_open
        except:
            return False
            
    def get_account(self):
        """Get account info"""
        account = self.api.get_account()
        return {
            'equity': float(account.equity),
            'cash': float(account.cash),
            'buying_power': float(account.buying_power)
        }
        
    def get_positions(self):
        """Get current positions"""
        positions = self.api.list_positions()
        return {p.symbol: {'qty': float(p.qty), 'market_value': float(p.market_value),
                          'unrealized_pl': float(p.unrealized_pl), 'pct': float(p.unrealized_plpc)}
                for p in positions}


    def get_bars(self, symbol, days=20):
        """Get historical bars"""
        try:
            end = datetime.now()
            start = end - timedelta(days=days+5)
            bars = self.api.get_bars(symbol, '1Day', start=start.strftime('%Y-%m-%d'),
                                     end=end.strftime('%Y-%m-%d')).df
            return bars
        except Exception as e:
            logger.warning(f"Failed to get bars for {symbol}: {e}")
            return None
            
    def calculate_momentum_signal(self, symbol):
        """Calculate momentum signal (-1 to 1)"""
        bars = self.get_bars(symbol, 20)
        if bars is None or len(bars) < 10:
            return 0, 0
            
        closes = bars['close'].values
        
        # Multi-timeframe momentum
        mom_5 = (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 5 else 0
        mom_10 = (closes[-1] - closes[-10]) / closes[-10] if len(closes) >= 10 else 0
        mom_20 = (closes[-1] - closes[-20]) / closes[-20] if len(closes) >= 20 else 0
        
        # RSI
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
        avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0.0001
        rs = avg_gain / avg_loss if avg_loss > 0 else 1
        rsi = 100 - (100 / (1 + rs))
        
        # Combined signal
        signal = (mom_5 * 0.4 + mom_10 * 0.35 + mom_20 * 0.25) * 10
        signal = max(-1, min(1, signal))
        
        # Confidence based on RSI extremes
        if rsi > 70:
            confidence = 0.8  # Overbought but trending
        elif rsi < 30:
            confidence = 0.9  # Oversold - high conviction buy
        else:
            confidence = 0.6
            
        return signal, confidence
        
    def calculate_pair_signal(self, sym1, sym2):
        """Calculate pairs trading signal"""
        bars1 = self.get_bars(sym1, 60)
        bars2 = self.get_bars(sym2, 60)
        
        if bars1 is None or bars2 is None or len(bars1) < 30 or len(bars2) < 30:
            return 0, 0
            
        # Calculate spread z-score
        spread = np.log(bars1['close'].values) - np.log(bars2['close'].values)
        mean = np.mean(spread[-30:])
        std = np.std(spread[-30:])
        
        if std == 0:
            return 0, 0
            
        zscore = (spread[-1] - mean) / std
        
        # Signal: positive = long sym1/short sym2, negative = short sym1/long sym2
        if zscore > 2:
            return -1, 0.8  # Short spread
        elif zscore < -2:
            return 1, 0.8   # Long spread
        elif abs(zscore) < 0.5:
            return 0, 0.9   # Close positions
        return 0, 0


    def execute_trade(self, symbol, side, notional):
        """Execute a trade"""
        try:
            order = self.api.submit_order(
                symbol=symbol,
                notional=notional,
                side=side,
                type='market',
                time_in_force='day'
            )
            logger.info(f"ORDER EXECUTED: {side.upper()} ${notional:.2f} of {symbol}")
            return order
        except Exception as e:
            logger.error(f"Order failed for {symbol}: {e}")
            return None
            
    def close_position(self, symbol):
        """Close a position"""
        try:
            self.api.close_position(symbol)
            logger.info(f"CLOSED position in {symbol}")
            return True
        except Exception as e:
            logger.error(f"Failed to close {symbol}: {e}")
            return False
            
    def check_stop_loss(self, positions):
        """Check and execute stop losses"""
        STOP_LOSS_PCT = -0.07  # 7% stop loss
        TAKE_PROFIT_PCT = 0.15  # 15% take profit
        
        for symbol, pos in positions.items():
            pct = pos['pct']
            if pct <= STOP_LOSS_PCT:
                logger.warning(f"STOP LOSS triggered for {symbol} at {pct:.2%}")
                self.close_position(symbol)
            elif pct >= TAKE_PROFIT_PCT:
                logger.info(f"TAKE PROFIT triggered for {symbol} at {pct:.2%}")
                self.close_position(symbol)
                
    def run_momentum_strategy(self):
        """Run momentum strategy on high-beta stocks"""
        account = self.get_account()
        positions = self.get_positions()
        
        # Allocate 60% of buying power to momentum
        momentum_budget = account['buying_power'] * 0.6
        per_stock = momentum_budget / len(self.MOMENTUM_UNIVERSE)
        per_stock = min(per_stock, 5000)  # Max $5k per position
        
        logger.info(f"Running momentum strategy - Budget: ${momentum_budget:.2f}")
        
        signals = []
        for symbol in self.MOMENTUM_UNIVERSE:
            signal, confidence = self.calculate_momentum_signal(symbol)
            if abs(signal) > 0.3 and confidence > 0.5:
                signals.append((symbol, signal, confidence))
                
        # Sort by signal strength
        signals.sort(key=lambda x: abs(x[1]) * x[2], reverse=True)
        
        # Execute top 5 signals
        for symbol, signal, confidence in signals[:5]:
            if symbol in positions:
                continue
            if signal > 0.3:
                self.execute_trade(symbol, 'buy', per_stock * confidence)
            elif signal < -0.3 and symbol in positions:
                self.close_position(symbol)
                
    def run_leveraged_etf_strategy(self):
        """Aggressive leveraged ETF strategy"""
        account = self.get_account()
        positions = self.get_positions()
        
        # Allocate 30% to leveraged ETFs
        etf_budget = account['buying_power'] * 0.3
        per_etf = etf_budget / 4  # Top 4 ETFs
        per_etf = min(per_etf, 10000)  # Max $10k per ETF
        
        logger.info(f"Running leveraged ETF strategy - Budget: ${etf_budget:.2f}")
        
        signals = []
        for symbol in self.LEVERAGED_ETFS:
            signal, confidence = self.calculate_momentum_signal(symbol)
            signals.append((symbol, signal, confidence))
            
        # Buy top 4 with positive momentum
        signals.sort(key=lambda x: x[1] * x[2], reverse=True)
        
        for symbol, signal, confidence in signals[:4]:
            if symbol not in positions and signal > 0:
                self.execute_trade(symbol, 'buy', per_etf)


    def run(self):
        """Main trading loop"""
        logger.info("="*60)
        logger.info("AGGRESSIVE AUTO TRADER - STARTING RUN")
        logger.info("="*60)
        
        # Check market status
        if not self.is_market_open():
            logger.info("Market is closed - skipping run")
            return
            
        # Get current state
        account = self.get_account()
        positions = self.get_positions()
        
        logger.info(f"Account Equity: ${account['equity']:,.2f}")
        logger.info(f"Buying Power: ${account['buying_power']:,.2f}")
        logger.info(f"Current Positions: {len(positions)}")
        
        # 1. Check stop losses first
        logger.info("\n--- Checking Stop Losses ---")
        self.check_stop_loss(positions)
        
        # 2. Run momentum strategy
        logger.info("\n--- Running Momentum Strategy ---")
        self.run_momentum_strategy()
        
        # 3. Run leveraged ETF strategy
        logger.info("\n--- Running Leveraged ETF Strategy ---")
        self.run_leveraged_etf_strategy()
        
        # Report final state
        final_positions = self.get_positions()
        logger.info(f"\nFinal Position Count: {len(final_positions)}")
        logger.info("="*60)
        logger.info("RUN COMPLETE")
        logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(description='Aggressive Auto Trader')
    parser.add_argument('--run', action='store_true', help='Execute trading cycle')
    parser.add_argument('--status', action='store_true', help='Show status only')
    parser.add_argument('--live', action='store_true', help='Use live trading (default: paper)')
    args = parser.parse_args()
    
    trader = AggressiveAutoTrader(paper=not args.live)
    
    if args.status:
        account = trader.get_account()
        positions = trader.get_positions()
        print(f"\nAccount Equity: ${account['equity']:,.2f}")
        print(f"Buying Power: ${account['buying_power']:,.2f}")
        print(f"Positions: {len(positions)}")
        for sym, pos in positions.items():
            print(f"  {sym}: ${pos['market_value']:,.2f} ({pos['pct']:.2%})")
    elif args.run:
        trader.run()
    else:
        trader.run()

if __name__ == '__main__':
    main()
