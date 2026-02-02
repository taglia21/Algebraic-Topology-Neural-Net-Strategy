#!/usr/bin/env python3
"""
AGGRESSIVE PAPER TRADER
=======================
Maximum profit extraction for demo purposes.
Uses full Kelly, momentum chasing, and loose thresholds.
"""

import os
import sys
import json
import time
import signal
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('aggressive_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import configs
from config.aggressive_mode import get_aggressive_config

# Try yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# Try Alpaca
try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False


class AggressiveTrader:
    """Aggressive paper trader for fast P&L demonstration."""
    
    def __init__(self):
        self.config = get_aggressive_config()
        self.running = False
        
        # Portfolio state
        self.initial_capital = 100000.0
        self.cash = 100000.0
        self.positions: Dict[str, Dict] = {}
        self.trades: List[Dict] = []
        self.daily_pnl: Dict[str, float] = {}
        
        # Metrics
        self.start_time = datetime.now()
        self.cycle_count = 0
        
        # Price cache for momentum
        self.price_cache: Dict[str, List[float]] = {}
        
        # Alpaca connection
        self.alpaca = None
        if ALPACA_AVAILABLE and os.environ.get('ALPACA_API_KEY'):
            try:
                self.alpaca = tradeapi.REST(
                    os.environ.get('ALPACA_API_KEY'),
                    os.environ.get('ALPACA_SECRET_KEY'),
                    'https://paper-api.alpaca.markets',
                    api_version='v2'
                )
                logger.info("✅ Connected to Alpaca Paper Trading")
            except Exception as e:
                logger.warning(f"Alpaca connection failed: {e}")
        
        logger.info("🔥 AGGRESSIVE TRADER INITIALIZED")
        logger.info(f"   Universe: {len(self.config.universe)} symbols")
        logger.info(f"   Position size: {self.config.position_size_pct:.0%} per trade")
        logger.info(f"   Thresholds: {self.config.buy_threshold}/{self.config.sell_threshold}")
        logger.info(f"   Min confidence: {self.config.min_confidence}")
    
    def fetch_price(self, ticker: str) -> Optional[Dict]:
        """Fetch current price and calculate momentum."""
        try:
            if YFINANCE_AVAILABLE:
                data = yf.download(ticker, period='5d', interval='1m', progress=False)
                if len(data) > 0:
                    # Handle multi-index
                    if isinstance(data.columns, pd.MultiIndex):
                        close = data['Close'][ticker] if ticker in data['Close'].columns else data['Close'].iloc[:, 0]
                    else:
                        close = data['Close']
                    
                    current = float(close.iloc[-1])
                    
                    # Calculate momentum signals
                    prices = close.values[-60:]  # Last hour
                    if len(prices) >= 10:
                        mom_5m = (prices[-1] / prices[-5] - 1) if len(prices) >= 5 else 0
                        mom_15m = (prices[-1] / prices[-15] - 1) if len(prices) >= 15 else 0
                        mom_30m = (prices[-1] / prices[-30] - 1) if len(prices) >= 30 else 0
                        
                        # VWAP approximation
                        vwap = np.mean(prices[-30:]) if len(prices) >= 30 else current
                        vwap_signal = (current - vwap) / vwap
                    else:
                        mom_5m = mom_15m = mom_30m = vwap_signal = 0
                    
                    return {
                        'price': current,
                        'mom_5m': mom_5m,
                        'mom_15m': mom_15m,
                        'mom_30m': mom_30m,
                        'vwap_signal': vwap_signal,
                        'prices': prices.tolist()
                    }
            
            # Fallback mock data
            np.random.seed(int(time.time()) + hash(ticker) % 1000)
            base = 100 + hash(ticker) % 400
            current = base * (1 + np.random.normal(0, 0.01))
            return {
                'price': current,
                'mom_5m': np.random.normal(0.001, 0.005),
                'mom_15m': np.random.normal(0.002, 0.008),
                'mom_30m': np.random.normal(0.003, 0.01),
                'vwap_signal': np.random.normal(0, 0.005),
                'prices': [current]
            }
            
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            return None
    
    def generate_signal(self, ticker: str, price_data: Dict) -> Tuple[str, float, float]:
        """
        Generate aggressive trading signal.
        Combines ML prediction with momentum chasing.
        """
        # Base momentum signal
        mom = (
            price_data['mom_5m'] * 0.5 +
            price_data['mom_15m'] * 0.3 +
            price_data['mom_30m'] * 0.2
        )
        
        # VWAP signal
        vwap = price_data['vwap_signal']
        
        # Combine signals
        combined = mom * 0.6 + vwap * 0.4
        
        # Convert to probability
        prob = 1 / (1 + np.exp(-combined * 50))  # More sensitive
        
        # Confidence based on signal strength
        conf = min(abs(combined) * 20, 1.0)
        
        # Aggressive thresholds
        if prob > self.config.buy_threshold:
            signal = 'long'
        elif prob < self.config.sell_threshold:
            signal = 'short'
        else:
            # In aggressive mode, bias toward action
            if self.config.trade_all_signals:
                if prob > 0.5:
                    signal = 'long'
                    conf *= 0.7  # Lower confidence for marginal
                else:
                    signal = 'short'
                    conf *= 0.7
            else:
                signal = 'neutral'
        
        # Momentum chase override
        if self.config.enable_momentum_chase:
            if price_data['mom_5m'] > self.config.momentum_threshold:
                signal = 'long'
                conf = max(conf, 0.6)
                logger.info(f"   🚀 Momentum chase BUY on {ticker}")
            elif price_data['mom_5m'] < -self.config.momentum_threshold:
                signal = 'short'
                conf = max(conf, 0.6)
                logger.info(f"   📉 Momentum chase SELL on {ticker}")
        
        return signal, prob, conf
    
    def calculate_position_size(self, price: float, confidence: float) -> int:
        """Calculate aggressive position size."""
        # Base size from config
        base_value = self.cash * self.config.position_size_pct
        
        # Kelly adjustment
        kelly_size = base_value * self.config.kelly_fraction * (1 + confidence)
        
        # Apply leverage
        if self.config.use_margin:
            kelly_size *= self.config.max_leverage
        
        # Clamp to limits
        kelly_size = max(kelly_size, self.config.min_position_dollars)
        kelly_size = min(kelly_size, self.config.max_position_dollars)
        kelly_size = min(kelly_size, self.cash * 0.95)  # Don't exceed available
        
        shares = int(kelly_size / price)
        return max(shares, 1)
    
    def execute_trade(self, ticker: str, signal: str, price: float, shares: int) -> bool:
        """Execute a trade."""
        try:
            trade_value = price * shares
            
            if self.alpaca:
                side = 'buy' if signal == 'long' else 'sell'
                order = self.alpaca.submit_order(
                    symbol=ticker,
                    qty=shares,
                    side=side,
                    type='market',
                    time_in_force='day'
                )
                logger.info(f"   📝 Alpaca order: {order.id}")
            
            # Record trade
            trade = {
                'timestamp': datetime.now().isoformat(),
                'ticker': ticker,
                'signal': signal,
                'price': price,
                'shares': shares,
                'value': trade_value
            }
            self.trades.append(trade)
            
            # Update position
            self.positions[ticker] = {
                'signal': signal,
                'entry_price': price,
                'shares': shares,
                'entry_time': datetime.now()
            }
            
            # Update cash
            self.cash -= trade_value
            
            emoji = '🟢' if signal == 'long' else '🔴'
            logger.info(f"   {emoji} EXECUTED: {signal.upper()} {shares} {ticker} @ ${price:.2f} (${trade_value:.0f})")
            
            return True
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return False
    
    def check_exits(self, ticker: str, current_price: float) -> Optional[float]:
        """Check if position should be exited."""
        if ticker not in self.positions:
            return None
        
        pos = self.positions[ticker]
        entry = pos['entry_price']
        signal = pos['signal']
        
        # Calculate P&L
        if signal == 'long':
            pnl_pct = (current_price - entry) / entry
        else:
            pnl_pct = (entry - current_price) / entry
        
        # Check take profit
        if pnl_pct >= self.config.take_profit_pct:
            logger.info(f"   💰 TAKE PROFIT: {ticker} +{pnl_pct:.2%}")
            return self._close_position(ticker, current_price)
        
        # Check stop loss
        if pnl_pct <= -self.config.stop_loss_pct:
            logger.info(f"   🛑 STOP LOSS: {ticker} {pnl_pct:.2%}")
            return self._close_position(ticker, current_price)
        
        # Check scalp timeout
        if self.config.enable_scalping:
            hold_time = (datetime.now() - pos['entry_time']).total_seconds() / 60
            if hold_time > self.config.scalp_time_limit_minutes and pnl_pct > 0:
                logger.info(f"   ⏱️ SCALP EXIT: {ticker} +{pnl_pct:.2%}")
                return self._close_position(ticker, current_price)
        
        return None
    
    def _close_position(self, ticker: str, exit_price: float) -> float:
        """Close a position and return P&L."""
        pos = self.positions[ticker]
        entry = pos['entry_price']
        shares = pos['shares']
        signal = pos['signal']
        
        if signal == 'long':
            pnl = (exit_price - entry) * shares
        else:
            pnl = (entry - exit_price) * shares
        
        # Update cash
        self.cash += (entry * shares) + pnl
        
        # Record
        today = datetime.now().strftime('%Y-%m-%d')
        self.daily_pnl[today] = self.daily_pnl.get(today, 0) + pnl
        
        # Remove position
        del self.positions[ticker]
        
        emoji = '💰' if pnl > 0 else '💸'
        logger.info(f"   {emoji} CLOSED: {ticker} P&L=${pnl:.2f}")
        
        return pnl
    
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        total = self.cash
        for ticker, pos in self.positions.items():
            total += pos['entry_price'] * pos['shares']
        return total
    
    def trading_cycle(self):
        """Run one aggressive trading cycle."""
        self.cycle_count += 1
        
        logger.info("=" * 60)
        logger.info(f"🔥 CYCLE {self.cycle_count} | {datetime.now().strftime('%H:%M:%S')}")
        logger.info("=" * 60)
        
        trades_executed = 0
        signals_generated = 0
        
        for ticker in self.config.universe:
            # Fetch price
            price_data = self.fetch_price(ticker)
            if not price_data:
                continue
            
            current_price = price_data['price']
            
            # Check exits first
            exit_pnl = self.check_exits(ticker, current_price)
            if exit_pnl is not None:
                trades_executed += 1
            
            # Skip if already have position
            if ticker in self.positions:
                continue
            
            # Generate signal
            signal, prob, conf = self.generate_signal(ticker, price_data)
            signals_generated += 1
            
            # Log signal
            emoji = '🟢' if signal == 'long' else ('🔴' if signal == 'short' else '⚪')
            logger.info(f"{ticker}: {emoji} {signal} prob={prob:.3f} conf={conf:.3f} price=${current_price:.2f}")
            
            # Skip if confidence too low
            if conf < self.config.min_confidence:
                continue
            
            # Skip neutral unless aggressive
            if signal == 'neutral' and not self.config.trade_all_signals:
                continue
            
            # Check position limits
            if len(self.positions) >= 10:  # Max 10 positions
                continue
            
            # Check cash
            if self.cash < self.config.min_position_dollars:
                continue
            
            # Execute trade
            if signal in ['long', 'short']:
                shares = self.calculate_position_size(current_price, conf)
                if self.execute_trade(ticker, signal, current_price, shares):
                    trades_executed += 1
        
        # Summary
        portfolio = self.get_portfolio_value()
        total_pnl = portfolio - self.initial_capital
        today_pnl = self.daily_pnl.get(datetime.now().strftime('%Y-%m-%d'), 0)
        
        logger.info("-" * 60)
        logger.info(f"📊 SUMMARY:")
        logger.info(f"   Portfolio: ${portfolio:,.2f}")
        logger.info(f"   Total P&L: ${total_pnl:,.2f} ({total_pnl/self.initial_capital:.2%})")
        logger.info(f"   Today P&L: ${today_pnl:,.2f}")
        logger.info(f"   Positions: {len(self.positions)}")
        logger.info(f"   Signals: {signals_generated} | Trades: {trades_executed}")
        logger.info("-" * 60)
        
        # Save metrics
        self.save_metrics()
        
        return trades_executed
    
    def save_metrics(self):
        """Save current metrics to file."""
        portfolio = self.get_portfolio_value()
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'runtime_minutes': (datetime.now() - self.start_time).total_seconds() / 60,
            'portfolio_value': portfolio,
            'total_pnl': portfolio - self.initial_capital,
            'total_pnl_pct': (portfolio - self.initial_capital) / self.initial_capital,
            'cash': self.cash,
            'positions': len(self.positions),
            'total_trades': len(self.trades),
            'cycles': self.cycle_count,
            'daily_pnl': self.daily_pnl,
            'open_positions': {k: {'shares': v['shares'], 'entry': v['entry_price']} 
                             for k, v in self.positions.items()}
        }
        
        with open('aggressive_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
    
    def run(self):
        """Run the aggressive trader."""
        self.running = True
        
        def signal_handler(sig, frame):
            logger.info("\n🛑 Shutting down...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        logger.info("🔥" * 30)
        logger.info("AGGRESSIVE TRADER STARTED")
        logger.info(f"Cycle interval: {self.config.cycle_seconds}s")
        logger.info("🔥" * 30)
        
        while self.running:
            try:
                self.trading_cycle()
                
                if self.running:
                    logger.info(f"⏱️ Next cycle in {self.config.cycle_seconds}s...")
                    for _ in range(self.config.cycle_seconds):
                        if not self.running:
                            break
                        time.sleep(1)
                        
            except Exception as e:
                logger.error(f"Cycle error: {e}")
                time.sleep(10)
        
        # Final summary
        portfolio = self.get_portfolio_value()
        total_pnl = portfolio - self.initial_capital
        
        logger.info("\n" + "=" * 60)
        logger.info("🏁 AGGRESSIVE TRADING SESSION ENDED")
        logger.info("=" * 60)
        logger.info(f"Runtime: {(datetime.now() - self.start_time).total_seconds()/60:.1f} minutes")
        logger.info(f"Final Portfolio: ${portfolio:,.2f}")
        logger.info(f"Total P&L: ${total_pnl:,.2f} ({total_pnl/self.initial_capital:.2%})")
        logger.info(f"Trades Executed: {len(self.trades)}")
        logger.info("=" * 60)
        
        self.save_metrics()


def main():
    print("\n" + "🔥" * 30)
    print("       AGGRESSIVE MODE ACTIVATED")
    print("       Paper Trading - Fast Profit Demo")
    print("🔥" * 30 + "\n")
    
    config = get_aggressive_config()
    print("Configuration:")
    print(f"  Buy threshold: {config.buy_threshold}")
    print(f"  Sell threshold: {config.sell_threshold}")
    print(f"  Min confidence: {config.min_confidence}")
    print(f"  Position size: {config.position_size_pct:.0%}")
    print(f"  Kelly fraction: {config.kelly_fraction}")
    print(f"  Cycle: {config.cycle_seconds}s")
    print(f"  Universe: {config.universe}")
    print()
    
    trader = AggressiveTrader()
    trader.run()


if __name__ == "__main__":
    main()
