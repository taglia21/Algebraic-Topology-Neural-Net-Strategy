#!/usr/bin/env python3
"""
V39 Crypto Trader - 24/7 BTC/ETH Trading Module

Compact cryptocurrency trading system using Alpaca Crypto API.
Features momentum-based signals with 5-minute trading loop.

Author: V39 Trading System
Date: 2026-01-26
"""

from dotenv import load_dotenv
load_dotenv()

import os
import sys
import time
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json

import numpy as np
import pandas as pd

# Alpaca imports
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
    from alpaca.data.historical.crypto import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("Warning: Alpaca SDK not installed. Run: pip install alpaca-py")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('v39_crypto.log')
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CryptoConfig:
    """Configuration for crypto trading."""
    # Trading universe
    symbols: List[str] = field(default_factory=lambda: ["BTC/USD", "ETH/USD"])
    
    # Position limits
    max_position_pct: float = 0.45  # Max 45% per crypto
    total_deployment: float = 0.90  # 90% total deployment
    min_trade_usd: float = 100.0    # Minimum trade size
    
    # Signal parameters
    fast_period: int = 12           # Fast EMA period
    slow_period: int = 26           # Slow EMA period
    signal_period: int = 9          # Signal line period
    rsi_period: int = 14            # RSI period
    rsi_oversold: float = 30.0      # RSI oversold threshold
    rsi_overbought: float = 70.0    # RSI overbought threshold
    
    # Trading parameters
    loop_interval: int = 300        # 5 minutes in seconds
    lookback_days: int = 7          # Days of historical data
    momentum_threshold: float = 0.02  # 2% momentum threshold
    
    # Risk management
    stop_loss_pct: float = 0.05     # 5% stop loss
    take_profit_pct: float = 0.10   # 10% take profit


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CryptoSignal:
    """Trading signal for a cryptocurrency."""
    symbol: str
    timestamp: datetime
    signal: str  # 'BUY', 'SELL', 'HOLD'
    strength: float  # -1.0 to 1.0
    price: float
    rsi: float
    macd: float
    macd_signal: float
    momentum: float
    reason: str


@dataclass
class CryptoPosition:
    """Current position in a cryptocurrency."""
    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float


# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

class TechnicalAnalyzer:
    """Calculate technical indicators for crypto."""
    
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.inf)
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, 
             signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD, Signal, and Histogram."""
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def momentum(data: pd.Series, period: int = 10) -> pd.Series:
        """Calculate price momentum."""
        return data.pct_change(periods=period)
    
    @staticmethod
    def volatility(data: pd.Series, period: int = 20) -> pd.Series:
        """Calculate rolling volatility."""
        returns = data.pct_change()
        return returns.rolling(window=period).std() * np.sqrt(252 * 24)  # Annualized


# =============================================================================
# CRYPTO TRADER
# =============================================================================

class CryptoTrader:
    """24/7 Cryptocurrency trading system."""
    
    def __init__(self, config: Optional[CryptoConfig] = None, paper: bool = True):
        """Initialize the crypto trader."""
        self.config = config or CryptoConfig()
        self.paper = paper
        self.analyzer = TechnicalAnalyzer()
        
        # Initialize Alpaca clients
        api_key = os.getenv('ALPACA_API_KEY') or os.getenv('APCA_API_KEY_ID')
        secret_key = os.getenv('ALPACA_SECRET_KEY') or os.getenv('APCA_API_SECRET_KEY')
        
        if not api_key or not secret_key:
            logger.warning("Alpaca API keys not found in environment")
            self.trading_client = None
            self.data_client = None
        elif ALPACA_AVAILABLE:
            self.trading_client = TradingClient(api_key, secret_key, paper=paper)
            self.data_client = CryptoHistoricalDataClient(api_key, secret_key)
            logger.info(f"Initialized Alpaca clients (paper={paper})")
        else:
            self.trading_client = None
            self.data_client = None
        
        # State tracking
        self.signals: Dict[str, CryptoSignal] = {}
        self.positions: Dict[str, CryptoPosition] = {}
        self.last_prices: Dict[str, float] = {}
        self.running = False
    
    def get_account_info(self) -> Dict:
        """Get account information."""
        if not self.trading_client:
            return {"cash": 100000, "equity": 100000, "buying_power": 100000}
        
        try:
            account = self.trading_client.get_account()
            return {
                "cash": float(account.cash),
                "equity": float(account.equity),
                "buying_power": float(account.buying_power),
                "crypto_status": getattr(account, 'crypto_status', 'ACTIVE')
            }
        except Exception as e:
            logger.error(f"Error getting account: {e}")
            return {}
    
    def get_crypto_bars(self, symbol: str, days: int = 7) -> Optional[pd.DataFrame]:
        """Fetch historical crypto bars."""
        if not self.data_client:
            return self._generate_mock_data(symbol, days)
        
        try:
            # Convert symbol format for API
            api_symbol = symbol.replace("/", "")  # BTC/USD -> BTCUSD
            
            request = CryptoBarsRequest(
                symbol_or_symbols=api_symbol,
                timeframe=TimeFrame.Hour,
                start=datetime.now() - timedelta(days=days)
            )
            bars = self.data_client.get_crypto_bars(request)
            
            if api_symbol in bars:
                df = bars[api_symbol].df
                df = df.reset_index()
                return df
            return None
        except Exception as e:
            logger.error(f"Error fetching bars for {symbol}: {e}")
            return self._generate_mock_data(symbol, days)
    
    def _generate_mock_data(self, symbol: str, days: int) -> pd.DataFrame:
        """Generate mock data for testing."""
        base_price = 95000 if "BTC" in symbol else 3200
        periods = days * 24  # Hourly data
        
        timestamps = pd.date_range(end=datetime.now(), periods=periods, freq='H')
        
        # Random walk with drift
        returns = np.random.normal(0.0001, 0.02, periods)
        prices = base_price * np.cumprod(1 + returns)
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices * (1 + np.random.uniform(-0.005, 0.005, periods)),
            'high': prices * (1 + np.random.uniform(0, 0.02, periods)),
            'low': prices * (1 - np.random.uniform(0, 0.02, periods)),
            'close': prices,
            'volume': np.random.uniform(100, 1000, periods)
        })
        return df
    
    def calculate_signals(self, symbol: str) -> Optional[CryptoSignal]:
        """Calculate trading signals for a symbol."""
        df = self.get_crypto_bars(symbol, self.config.lookback_days)
        if df is None or len(df) < 50:
            return None
        
        close = df['close']
        current_price = float(close.iloc[-1])
        self.last_prices[symbol] = current_price
        
        # Calculate indicators
        rsi = self.analyzer.rsi(close, self.config.rsi_period)
        macd_line, signal_line, histogram = self.analyzer.macd(
            close, self.config.fast_period, 
            self.config.slow_period, self.config.signal_period
        )
        momentum = self.analyzer.momentum(close, 10)
        
        current_rsi = float(rsi.iloc[-1])
        current_macd = float(macd_line.iloc[-1])
        current_signal = float(signal_line.iloc[-1])
        current_momentum = float(momentum.iloc[-1])
        
        # Generate signal
        signal = "HOLD"
        strength = 0.0
        reasons = []
        
        # RSI signals
        if current_rsi < self.config.rsi_oversold:
            strength += 0.3
            reasons.append(f"RSI oversold ({current_rsi:.1f})")
        elif current_rsi > self.config.rsi_overbought:
            strength -= 0.3
            reasons.append(f"RSI overbought ({current_rsi:.1f})")
        
        # MACD signals
        if current_macd > current_signal and histogram.iloc[-1] > histogram.iloc[-2]:
            strength += 0.4
            reasons.append("MACD bullish crossover")
        elif current_macd < current_signal and histogram.iloc[-1] < histogram.iloc[-2]:
            strength -= 0.4
            reasons.append("MACD bearish crossover")
        
        # Momentum signals
        if current_momentum > self.config.momentum_threshold:
            strength += 0.3
            reasons.append(f"Strong momentum ({current_momentum:.2%})")
        elif current_momentum < -self.config.momentum_threshold:
            strength -= 0.3
            reasons.append(f"Weak momentum ({current_momentum:.2%})")
        
        # Determine final signal
        if strength >= 0.5:
            signal = "BUY"
        elif strength <= -0.5:
            signal = "SELL"
        
        crypto_signal = CryptoSignal(
            symbol=symbol,
            timestamp=datetime.now(),
            signal=signal,
            strength=np.clip(strength, -1.0, 1.0),
            price=current_price,
            rsi=current_rsi,
            macd=current_macd,
            macd_signal=current_signal,
            momentum=current_momentum,
            reason="; ".join(reasons) if reasons else "No clear signal"
        )
        
        self.signals[symbol] = crypto_signal
        return crypto_signal
    
    def get_positions(self) -> Dict[str, CryptoPosition]:
        """Get current crypto positions."""
        if not self.trading_client:
            return {}
        
        try:
            positions = self.trading_client.get_all_positions()
            crypto_positions = {}
            
            for pos in positions:
                symbol = pos.symbol
                if any(crypto in symbol for crypto in ["BTC", "ETH"]):
                    # Normalize symbol format
                    if "/" not in symbol:
                        symbol = f"{symbol[:3]}/{symbol[3:]}" if len(symbol) == 6 else symbol
                    
                    crypto_positions[symbol] = CryptoPosition(
                        symbol=symbol,
                        qty=float(pos.qty),
                        avg_entry_price=float(pos.avg_entry_price),
                        current_price=float(pos.current_price),
                        market_value=float(pos.market_value),
                        unrealized_pnl=float(pos.unrealized_pl),
                        unrealized_pnl_pct=float(pos.unrealized_plpc) * 100
                    )
            
            self.positions = crypto_positions
            return crypto_positions
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return {}
    
    def calculate_position_size(self, symbol: str, signal: CryptoSignal) -> float:
        """Calculate position size based on signal strength."""
        account = self.get_account_info()
        equity = account.get("equity", 0)
        
        if equity <= 0:
            return 0
        
        # Base allocation
        max_allocation = equity * self.config.max_position_pct
        
        # Adjust by signal strength
        allocation = max_allocation * abs(signal.strength)
        
        # Check existing position
        current_pos = self.positions.get(symbol)
        if current_pos:
            allocation -= current_pos.market_value
        
        # Ensure minimum trade size
        if allocation < self.config.min_trade_usd:
            return 0
        
        return allocation
    
    def execute_trade(self, symbol: str, signal: CryptoSignal) -> bool:
        """Execute a trade based on signal."""
        if not self.trading_client:
            logger.info(f"[SIMULATED] {signal.signal} {symbol} @ ${signal.price:,.2f}")
            return True
        
        try:
            # Calculate position size
            if signal.signal == "BUY":
                usd_amount = self.calculate_position_size(symbol, signal)
                if usd_amount <= 0:
                    return False
                
                qty = usd_amount / signal.price
                order = MarketOrderRequest(
                    symbol=symbol.replace("/", ""),
                    qty=round(qty, 8),
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.GTC
                )
                
            elif signal.signal == "SELL":
                pos = self.positions.get(symbol)
                if not pos or pos.qty <= 0:
                    return False
                
                order = MarketOrderRequest(
                    symbol=symbol.replace("/", ""),
                    qty=round(pos.qty, 8),
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC
                )
            else:
                return False
            
            result = self.trading_client.submit_order(order)
            logger.info(f"Order submitted: {signal.signal} {symbol} - ID: {result.id}")
            return True
            
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return False
    
    def check_stop_loss_take_profit(self) -> List[str]:
        """Check positions for stop loss or take profit triggers."""
        triggered = []
        
        for symbol, pos in self.positions.items():
            pnl_pct = pos.unrealized_pnl_pct / 100
            
            if pnl_pct <= -self.config.stop_loss_pct:
                logger.warning(f"STOP LOSS triggered for {symbol}: {pnl_pct:.2%}")
                triggered.append(f"STOP_LOSS:{symbol}")
                
            elif pnl_pct >= self.config.take_profit_pct:
                logger.info(f"TAKE PROFIT triggered for {symbol}: {pnl_pct:.2%}")
                triggered.append(f"TAKE_PROFIT:{symbol}")
        
        return triggered
    
    def run_cycle(self) -> Dict:
        """Run one trading cycle."""
        logger.info("=" * 50)
        logger.info(f"Trading cycle: {datetime.now().isoformat()}")
        
        # Get current positions
        self.get_positions()
        
        # Check stop loss / take profit
        triggers = self.check_stop_loss_take_profit()
        for trigger in triggers:
            action, symbol = trigger.split(":")
            # Create sell signal for triggered positions
            if symbol in self.positions:
                sell_signal = CryptoSignal(
                    symbol=symbol, timestamp=datetime.now(),
                    signal="SELL", strength=-1.0,
                    price=self.positions[symbol].current_price,
                    rsi=50, macd=0, macd_signal=0, momentum=0,
                    reason=action
                )
                self.execute_trade(symbol, sell_signal)
        
        results = {"signals": [], "trades": []}
        
        # Calculate signals and execute
        for symbol in self.config.symbols:
            signal = self.calculate_signals(symbol)
            if signal:
                results["signals"].append({
                    "symbol": symbol,
                    "signal": signal.signal,
                    "strength": signal.strength,
                    "price": signal.price,
                    "reason": signal.reason
                })
                
                logger.info(f"{symbol}: {signal.signal} (strength: {signal.strength:.2f})")
                logger.info(f"  RSI: {signal.rsi:.1f}, MACD: {signal.macd:.4f}")
                logger.info(f"  Reason: {signal.reason}")
                
                if signal.signal != "HOLD":
                    success = self.execute_trade(symbol, signal)
                    results["trades"].append({
                        "symbol": symbol,
                        "action": signal.signal,
                        "success": success
                    })
        
        return results
    
    def run_loop(self):
        """Run continuous trading loop."""
        logger.info("Starting 24/7 crypto trading loop...")
        logger.info(f"Symbols: {self.config.symbols}")
        logger.info(f"Interval: {self.config.loop_interval} seconds")
        
        self.running = True
        
        while self.running:
            try:
                self.run_cycle()
                
                # Wait for next cycle
                logger.info(f"Sleeping {self.config.loop_interval}s until next cycle...")
                time.sleep(self.config.loop_interval)
                
            except KeyboardInterrupt:
                logger.info("Received shutdown signal")
                self.running = False
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(60)  # Wait 1 minute on error
        
        logger.info("Trading loop stopped")
    
    def get_status(self) -> Dict:
        """Get current trading status."""
        account = self.get_account_info()
        positions = self.get_positions()
        
        status = {
            "timestamp": datetime.now().isoformat(),
            "account": account,
            "positions": [],
            "signals": [],
            "config": {
                "symbols": self.config.symbols,
                "max_position_pct": self.config.max_position_pct,
                "paper_trading": self.paper
            }
        }
        
        for symbol, pos in positions.items():
            status["positions"].append({
                "symbol": symbol,
                "qty": pos.qty,
                "avg_entry": pos.avg_entry_price,
                "current_price": pos.current_price,
                "market_value": pos.market_value,
                "pnl": pos.unrealized_pnl,
                "pnl_pct": pos.unrealized_pnl_pct
            })
        
        for symbol, signal in self.signals.items():
            status["signals"].append({
                "symbol": symbol,
                "signal": signal.signal,
                "strength": signal.strength,
                "price": signal.price,
                "rsi": signal.rsi,
                "reason": signal.reason
            })
        
        return status
    
    def print_status(self):
        """Print formatted status to console."""
        status = self.get_status()
        
        print("\n" + "=" * 60)
        print("V39 CRYPTO TRADER STATUS")
        print("=" * 60)
        print(f"Timestamp: {status['timestamp']}")
        print(f"Paper Trading: {status['config']['paper_trading']}")
        
        print("\n--- ACCOUNT ---")
        acc = status['account']
        print(f"Cash: ${acc.get('cash', 0):,.2f}")
        print(f"Equity: ${acc.get('equity', 0):,.2f}")
        print(f"Buying Power: ${acc.get('buying_power', 0):,.2f}")
        
        print("\n--- POSITIONS ---")
        if status['positions']:
            for pos in status['positions']:
                print(f"{pos['symbol']}: {pos['qty']:.6f} @ ${pos['avg_entry']:,.2f}")
                print(f"  Current: ${pos['current_price']:,.2f} | "
                      f"Value: ${pos['market_value']:,.2f} | "
                      f"P&L: ${pos['pnl']:,.2f} ({pos['pnl_pct']:.2f}%)")
        else:
            print("No open positions")
        
        print("\n--- LATEST SIGNALS ---")
        for symbol in self.config.symbols:
            signal = self.calculate_signals(symbol)
            if signal:
                print(f"{symbol}: {signal.signal} (strength: {signal.strength:.2f})")
                print(f"  Price: ${signal.price:,.2f} | RSI: {signal.rsi:.1f}")
                print(f"  {signal.reason}")
        
        print("=" * 60 + "\n")


# =============================================================================
# CLI
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="V39 Crypto Trader - BTC/ETH Trading")
    parser.add_argument("--trade", action="store_true", help="Start trading loop")
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--cycle", action="store_true", help="Run single trading cycle")
    parser.add_argument("--paper", action="store_true", default=True, help="Paper trading mode")
    parser.add_argument("--live", action="store_true", help="Live trading mode")
    
    args = parser.parse_args()
    
    paper = not args.live
    trader = CryptoTrader(paper=paper)
    
    if args.status:
        trader.print_status()
    elif args.cycle:
        result = trader.run_cycle()
        print(json.dumps(result, indent=2, default=str))
    elif args.trade:
        trader.run_loop()
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python v39_crypto_trader.py --status")
        print("  python v39_crypto_trader.py --cycle")
        print("  python v39_crypto_trader.py --trade")
        print("  python v39_crypto_trader.py --trade --live")


if __name__ == "__main__":
    main()
