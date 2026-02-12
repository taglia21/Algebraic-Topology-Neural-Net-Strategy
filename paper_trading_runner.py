#!/usr/bin/env python3
"""
⚠️  DEPRECATED — use run_v28_production.py --mode=paper instead.

Paper Trading Runner with Enhanced ML  (legacy — not used by V28)
=================================================================
Runs paper trading with the enhanced ML system.
"""

import os
import sys
import json
import time
import signal
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import threading

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('paper_trading_results.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from src.ml_integration import MLIntegration, get_ml_signal, record_trade, get_ml_stats
from config.strategy_overrides import get_overrides

# Import trading gate for circuit breaker protection
try:
    from src.risk.trading_gate import check_trading_allowed, update_breaker_state
    HAS_TRADING_GATE = True
except ImportError:
    HAS_TRADING_GATE = False

# Try to import Alpaca
try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logger.warning("Alpaca not available - using mock trading")

# Try to import data fetching
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not available")

import numpy as np
import pandas as pd


def get_universe_from_overrides():
    """Get trading universe from strategy overrides."""
    overrides = get_overrides()
    # Default universe excluding QQQ
    base_universe = ['SPY', 'IWM', 'XLK', 'XLF']
    excluded = overrides.excluded_tickers if hasattr(overrides, 'excluded_tickers') else {'QQQ'}
    return [t for t in base_universe if t not in excluded]


def get_thresholds_from_overrides():
    """Get signal thresholds from overrides."""
    overrides = get_overrides()
    return {
        'buy': getattr(overrides, 'nn_buy_threshold', 0.55),
        'sell': getattr(overrides, 'nn_sell_threshold', 0.45)
    }


class PaperTradingMetrics:
    """Track paper trading metrics."""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.trades: List[Dict] = []
        self.daily_pnl: Dict[str, float] = {}
        self.positions: Dict[str, Dict] = {}
        self.cash = 100000.0  # Starting capital
        self.initial_capital = 100000.0
        
    def record_trade(self, ticker: str, signal: str, price: float, shares: int, pnl: float = 0):
        """Record a trade."""
        trade = {
            'timestamp': datetime.now().isoformat(),
            'ticker': ticker,
            'signal': signal,
            'price': price,
            'shares': shares,
            'pnl': pnl
        }
        self.trades.append(trade)
        
        # Update daily P&L
        today = datetime.now().strftime('%Y-%m-%d')
        self.daily_pnl[today] = self.daily_pnl.get(today, 0) + pnl
        
        logger.info(f"TRADE: {signal.upper()} {shares} {ticker} @ ${price:.2f} | P&L: ${pnl:.2f}")
    
    def get_win_rate(self) -> float:
        """Calculate win rate."""
        if not self.trades:
            return 0.0
        winning = sum(1 for t in self.trades if t['pnl'] > 0)
        return winning / len(self.trades)
    
    def get_total_pnl(self) -> float:
        """Get total P&L."""
        return sum(t['pnl'] for t in self.trades)
    
    def get_sharpe(self) -> float:
        """Calculate rolling Sharpe ratio."""
        if len(self.daily_pnl) < 2:
            return 0.0
        
        returns = list(self.daily_pnl.values())
        if np.std(returns) == 0:
            return 0.0
        return np.mean(returns) / np.std(returns) * np.sqrt(252)
    
    def get_summary(self) -> Dict:
        """Get metrics summary."""
        return {
            'runtime_minutes': (datetime.now() - self.start_time).total_seconds() / 60,
            'total_trades': len(self.trades),
            'win_rate': self.get_win_rate(),
            'total_pnl': self.get_total_pnl(),
            'sharpe_ratio': self.get_sharpe(),
            'cash': self.cash,
            'positions': len(self.positions),
            'daily_pnl': self.daily_pnl
        }
    
    def save(self, filepath: str = 'paper_trading_metrics.json'):
        """Save metrics to file."""
        with open(filepath, 'w') as f:
            json.dump(self.get_summary(), f, indent=2, default=str)


class PaperTradingBot:
    """Paper trading bot with enhanced ML."""
    
    def __init__(self):
        self.running = False
        self.metrics = PaperTradingMetrics()
        self.ml = MLIntegration.get_instance()
        
        # Get universe from overrides
        self.universe = get_universe_from_overrides()
        self.thresholds = get_thresholds_from_overrides()
        
        # Trading parameters
        self.position_size = 10000  # $10k per position
        self.max_positions = 5
        
        # Initialize Alpaca if available — support both env var naming conventions
        self.alpaca = None
        api_key = os.environ.get('ALPACA_API_KEY') or os.environ.get('APCA_API_KEY_ID')
        api_secret = os.environ.get('ALPACA_SECRET_KEY') or os.environ.get('APCA_API_SECRET_KEY')
        if ALPACA_AVAILABLE and api_key:
            try:
                self.alpaca = tradeapi.REST(
                    api_key,
                    api_secret,
                    'https://paper-api.alpaca.markets',  # Paper trading endpoint
                    api_version='v2'
                )
                logger.info("✅ Connected to Alpaca Paper Trading")
            except Exception as e:
                logger.warning(f"Alpaca connection failed: {e}")
                self.alpaca = None
        
        logger.info(f"Paper Trading Bot initialized")
        logger.info(f"  Universe: {self.universe}")
        logger.info(f"  Position size: ${self.position_size}")
        logger.info(f"  Enhanced ML: {self.ml.enhanced_ml is not None}")
    
    def fetch_prices(self, ticker: str) -> Optional[Dict]:
        """Fetch recent price data for a ticker."""
        try:
            if YFINANCE_AVAILABLE:
                data = yf.download(ticker, period='3mo', progress=False)
                if len(data) > 0:
                    # Handle both single and multi-index columns
                    if isinstance(data.columns, pd.MultiIndex):
                        close_col = data['Close'][ticker] if ticker in data['Close'].columns else data['Close'].iloc[:, 0]
                        high_col = data['High'][ticker] if ticker in data['High'].columns else data['High'].iloc[:, 0]
                        low_col = data['Low'][ticker] if ticker in data['Low'].columns else data['Low'].iloc[:, 0]
                        vol_col = data['Volume'][ticker] if ticker in data['Volume'].columns else data['Volume'].iloc[:, 0]
                    else:
                        close_col = data['Close']
                        high_col = data['High']
                        low_col = data['Low']
                        vol_col = data['Volume']
                    
                    return {
                        'close': close_col.values.tolist(),
                        'high': high_col.values.tolist(),
                        'low': low_col.values.tolist(),
                        'volume': vol_col.values.tolist(),
                        'current_price': float(close_col.iloc[-1])
                    }
            
            # FIXED: Do NOT fall back to mock data — trading on fake prices loses real money
            logger.error(f"Cannot fetch real price data for {ticker} — refusing to generate mock data")
            return None
        except Exception as e:
            logger.error(f"Error fetching prices for {ticker}: {e}")
            return None
    
    def execute_trade(self, ticker: str, signal: str, price: float) -> bool:
        """Execute a paper trade."""
        try:
            shares = int(self.position_size / price)
            
            if self.alpaca:
                # Real Alpaca paper trade
                side = 'buy' if signal == 'long' else 'sell'
                order = self.alpaca.submit_order(
                    symbol=ticker,
                    qty=shares,
                    side=side,
                    type='market',
                    time_in_force='day'
                )
                logger.info(f"Alpaca order submitted: {order.id}")
            
            # Record in our metrics
            self.metrics.record_trade(ticker, signal, price, shares)
            
            # Track position
            self.metrics.positions[ticker] = {
                'signal': signal,
                'entry_price': price,
                'shares': shares,
                'entry_time': datetime.now().isoformat()
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return False
    
    def close_position(self, ticker: str) -> float:
        """Close an existing position."""
        if ticker not in self.metrics.positions:
            return 0.0
        
        position = self.metrics.positions[ticker]
        price_data = self.fetch_prices(ticker)
        
        if not price_data:
            return 0.0
        
        current_price = price_data['current_price']
        entry_price = position['entry_price']
        shares = position['shares']
        signal = position['signal']
        
        # Calculate P&L
        if signal == 'long':
            pnl = (current_price - entry_price) * shares
        else:
            pnl = (entry_price - current_price) * shares
        
        # Record the closing trade
        self.metrics.record_trade(
            ticker, 
            'close_' + signal, 
            current_price, 
            shares, 
            pnl
        )
        
        # Feed back to ML
        record_trade(ticker, signal, entry_price, current_price, self.position_size)
        
        # Remove position
        del self.metrics.positions[ticker]
        self.metrics.cash += self.position_size + pnl
        
        return pnl
    
    def trading_cycle(self):
        """Run one trading cycle."""
        logger.info("-" * 60)
        logger.info("TRADING CYCLE START")
        
        # Circuit breaker check
        if HAS_TRADING_GATE:
            allowed, reason = check_trading_allowed()
            if not allowed:
                logger.warning(f'⚠️ CIRCUIT BREAKER: {reason} - skipping cycle')
                return
        
        signals_generated = 0
        trades_executed = 0
        
        for ticker in self.universe:
            # Fetch prices
            price_data = self.fetch_prices(ticker)
            if not price_data:
                continue
            
            # Get ML signal
            signal, prob, conf = get_ml_signal(ticker, price_data)
            signals_generated += 1
            
            logger.info(f"{ticker}: signal={signal}, prob={prob:.3f}, conf={conf:.3f}")
            
            # Check if we already have a position
            if ticker in self.metrics.positions:
                position = self.metrics.positions[ticker]
                # Close if signal reversed
                if (position['signal'] == 'long' and signal == 'short') or \
                   (position['signal'] == 'short' and signal == 'long'):
                    pnl = self.close_position(ticker)
                    trades_executed += 1
                    logger.info(f"  Closed position: P&L=${pnl:.2f}")
                continue
            
            # Only trade signals with some confidence (lowered for demo)
            if conf < 0.1:
                continue
            
            # Check position limits
            if len(self.metrics.positions) >= self.max_positions:
                continue
            
            # Check cash
            if self.metrics.cash < self.position_size:
                continue
            
            # Execute trade
            if signal in ['long', 'short']:
                current_price = price_data['current_price']
                if self.execute_trade(ticker, signal, current_price):
                    trades_executed += 1
                    self.metrics.cash -= self.position_size
        
        # Print summary
        summary = self.metrics.get_summary()
        logger.info("-" * 60)
        logger.info(f"CYCLE COMPLETE: {signals_generated} signals, {trades_executed} trades")
        logger.info(f"  Total P&L: ${summary['total_pnl']:.2f}")
        logger.info(f"  Win Rate: {summary['win_rate']:.1%}")
        logger.info(f"  Sharpe: {summary['sharpe_ratio']:.3f}")
        logger.info(f"  Positions: {summary['positions']}")
        logger.info("-" * 60)
        
        # Save metrics
        self.metrics.save()
        
        return trades_executed
    
    def run(self, interval_minutes: int = 5):
        """Run the paper trading bot continuously."""
        self.running = True
        
        logger.info("=" * 60)
        logger.info("PAPER TRADING BOT STARTED")
        logger.info(f"  Interval: {interval_minutes} minutes")
        logger.info(f"  Universe: {', '.join(self.universe)}")
        logger.info("=" * 60)
        
        # Handle graceful shutdown
        def signal_handler(sig, frame):
            logger.info("\nShutting down gracefully...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        cycle_count = 0
        while self.running:
            try:
                cycle_count += 1
                logger.info(f"\n{'='*60}")
                logger.info(f"CYCLE {cycle_count} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"{'='*60}")
                
                # Run trading cycle
                trades = self.trading_cycle()
                
                # Print ML stats
                ml_stats = get_ml_stats()
                logger.info(f"ML Stats: {json.dumps(ml_stats, indent=2, default=str)}")
                
                # Wait for next cycle
                if self.running:
                    logger.info(f"Sleeping {interval_minutes} minutes until next cycle...")
                    for _ in range(interval_minutes * 60):
                        if not self.running:
                            break
                        time.sleep(1)
                        
            except Exception as e:
                logger.error(f"Error in trading cycle: {e}")
                time.sleep(60)  # Wait a minute before retrying
        
        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("PAPER TRADING SESSION ENDED")
        logger.info("=" * 60)
        summary = self.metrics.get_summary()
        logger.info(f"Runtime: {summary['runtime_minutes']:.1f} minutes")
        logger.info(f"Total Trades: {summary['total_trades']}")
        logger.info(f"Win Rate: {summary['win_rate']:.1%}")
        logger.info(f"Total P&L: ${summary['total_pnl']:.2f}")
        logger.info(f"Sharpe Ratio: {summary['sharpe_ratio']:.3f}")
        logger.info("=" * 60)
        
        self.metrics.save()


def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("ENHANCED ML PAPER TRADING SYSTEM")
    print("=" * 60 + "\n")
    
    # Show configuration
    overrides = get_overrides()
    thresholds = get_thresholds_from_overrides()
    universe = get_universe_from_overrides()
    
    print("Configuration:")
    print(f"  TDA Enabled: {getattr(overrides, 'enable_tda', False)}")
    print(f"  Risk Parity Enabled: {getattr(overrides, 'enable_risk_parity', False)}")
    print(f"  Buy Threshold: {thresholds['buy']}")
    print(f"  Sell Threshold: {thresholds['sell']}")
    print(f"  Universe: {universe}")
    print()
    
    # Create and run bot
    bot = PaperTradingBot()
    
    # Use 1 minute intervals for demo, 5 minutes for production
    interval = int(os.environ.get('TRADING_INTERVAL', 1))
    
    bot.run(interval_minutes=interval)


if __name__ == "__main__":
    main()
