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

# Import trading gate for circuit breaker protection
try:
    from src.risk.trading_gate import check_trading_allowed, update_breaker_state
    HAS_TRADING_GATE = True
except ImportError:
    HAS_TRADING_GATE = False

# Import process lock to prevent multiple bots
try:
    from src.risk.process_lock import acquire_trading_lock, release_trading_lock
    HAS_PROCESS_LOCK = True
except ImportError:
    HAS_PROCESS_LOCK = False

class AggressiveAutoTrader:
    """Aggressive automated trading system"""
    
    # High-momentum stocks to trade
    MOMENTUM_UNIVERSE = [
        'NVDA', 'TSLA', 'AMD', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
        'NFLX', 'CRM', 'SHOP', 'SQ', 'PYPL', 'COIN', 'MSTR', 'PLTR'
    ]
    
    # Leveraged ETFs DISABLED - daily decay makes them unsuitable for swing trading
    # Holding 3x ETFs overnight bleeds capital via volatility drag
    LEVERAGED_ETFS = []
    
    # Pairs for stat arb
    PAIRS = [
        ('XOM', 'CVX'), ('JPM', 'BAC'), ('KO', 'PEP'),
        ('V', 'MA'), ('HD', 'LOW'), ('MSFT', 'GOOGL')
    ]
    
    def __init__(self, paper=True):
        # Support both env var naming conventions
        self.api_key = os.getenv('ALPACA_API_KEY') or os.getenv('APCA_API_KEY_ID')
        self.api_secret = os.getenv('ALPACA_SECRET_KEY') or os.getenv('APCA_API_SECRET_KEY')
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
        
        # Confidence based on RSI — FIXED: align RSI interpretation with signal direction
        # For a momentum BUY signal:
        #   RSI 30-50 = oversold recovering, high confidence
        #   RSI 50-70 = normal, moderate confidence
        #   RSI > 70 = overbought, LOW confidence (likely to reverse)
        # For a momentum SELL signal:
        #   RSI > 70 = overbought, high confidence for sell
        #   RSI < 30 = oversold, LOW confidence for sell (likely to bounce)
        if signal > 0:  # Buy signal
            if rsi < 40:
                confidence = 0.85  # Oversold + positive momentum = strong buy
            elif rsi < 60:
                confidence = 0.65  # Normal range
            elif rsi > 70:
                confidence = 0.30  # Overbought - low confidence for buying
            else:
                confidence = 0.50
        else:  # Sell signal
            if rsi > 70:
                confidence = 0.85  # Overbought + negative momentum = strong sell
            elif rsi > 50:
                confidence = 0.65
            elif rsi < 30:
                confidence = 0.30  # Oversold - low confidence for selling
            else:
                confidence = 0.50
            
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
        """Execute a trade using limit order at latest price."""
        try:
            # Get latest quote for limit pricing
            quote = self.api.get_latest_trade(symbol)
            last_price = float(quote.price) if quote else None
            
            if last_price and last_price > 0:
                # Calculate qty from notional and use limit order
                qty = max(1, int(notional / last_price))
                if side == 'buy':
                    limit_price = round(last_price * 1.001, 2)  # 0.1% above
                else:
                    limit_price = round(last_price * 0.999, 2)  # 0.1% below
                
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    type='limit',
                    limit_price=limit_price,
                    time_in_force='day'
                )
            else:
                # Fallback: use notional market order only if no price available
                logger.warning(f"No price for {symbol}, using market order")
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
        
        # FIXED: Use equity (not buying_power which includes margin)
        momentum_budget = account['equity'] * 0.40  # 40% of equity (was 60% of buying_power)
        per_stock = momentum_budget / len(self.MOMENTUM_UNIVERSE)
        per_stock = min(per_stock, 3000)  # Max $3k per position (was $5k)
        
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
        """Leveraged ETF strategy - FIXED: require strong signals and use equity-based budget"""
        account = self.get_account()
        positions = self.get_positions()
        
        # FIXED: Use equity (not buying_power), reduced allocation since these are 3x leveraged
        etf_budget = account['equity'] * 0.10  # Only 10% of equity to leveraged ETFs (was 30% of buying_power)
        per_etf = etf_budget / 4  # Top 4 ETFs
        per_etf = min(per_etf, 3000)  # Max $3k per ETF (was $10k)
        
        logger.info(f"Running leveraged ETF strategy - Budget: ${etf_budget:.2f}")
        
        signals = []
        for symbol in self.LEVERAGED_ETFS:
            signal, confidence = self.calculate_momentum_signal(symbol)
            signals.append((symbol, signal, confidence))
            
        # FIXED: Require strong positive signal AND minimum confidence
        signals.sort(key=lambda x: x[1] * x[2], reverse=True)
        
        for symbol, signal, confidence in signals[:4]:
            if symbol not in positions and signal > 0.4 and confidence > 0.5:
                self.execute_trade(symbol, 'buy', per_etf * confidence)


    def run(self):
        """Main trading loop"""
        logger.info("="*60)
        logger.info("AGGRESSIVE AUTO TRADER - STARTING RUN")
        logger.info("="*60)
        
        # Check market status
        if not self.is_market_open():
            logger.info("Market is closed - skipping run")
            return
        
        # Circuit breaker check
        if HAS_TRADING_GATE:
            allowed, reason = check_trading_allowed()
            if not allowed:
                logger.warning(f'⚠️ CIRCUIT BREAKER: {reason} - skipping run')
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
        
        # 3. Leveraged ETF strategy DISABLED (daily decay makes them unsuitable)
        # logger.info("\n--- Running Leveraged ETF Strategy ---")
        # self.run_leveraged_etf_strategy()
        
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
    
    # Acquire exclusive trading lock (skip for status-only)
    _trading_lock = None
    if not args.status and HAS_PROCESS_LOCK:
        _trading_lock = acquire_trading_lock('auto_trader')
        if _trading_lock is None:
            logger.error('Another trading bot is already running! Exiting.')
            return
    
    try:
        trader = AggressiveAutoTrader(paper=not args.live)
        
        if args.status:
            account = trader.get_account()
            positions = trader.get_positions()
            print(f"\nAccount Equity: ${account['equity']:,.2f}")
            print(f"Buying Power: ${account['buying_power']:,.2f}")
            print(f"Positions: {len(positions)}")
            for sym, pos in positions.items():
                print(f"  {sym}: ${pos['market_value']:,.2f} ({pos['pct']:.2%})")
        else:
            trader.run()
    finally:
        if _trading_lock:
            release_trading_lock(_trading_lock)

if __name__ == '__main__':
    main()
