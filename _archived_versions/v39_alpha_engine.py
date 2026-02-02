#!/usr/bin/env python3
"""
V39 Alpha Engine - Main Trading Orchestrator

Combines all V39 strategies: Crypto, Deep ML, and V38 Alpha Core.
Unified interface for multi-asset automated trading.

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
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

# Alpaca imports
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

# Import V39 modules
try:
    from v39_crypto_trader import CryptoTrader, CryptoConfig
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

try:
    from v39_deep_ml_engine import DeepMLEngine, MLConfig
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

try:
    from v38_alpha_core import ExpandedUniverse, RegimeDetector, MLEnsemble
    V38_AVAILABLE = True
except ImportError:
    V38_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('v39_alpha.log')
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AlphaConfig:
    """Main configuration for alpha engine."""
    # Equity universe
    equity_symbols: List[str] = field(default_factory=lambda: [
        'SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META',
        'TSLA', 'AMD', 'COIN', 'MSTR', 'TQQQ', 'SOXL'
    ])
    
    # Crypto symbols
    crypto_symbols: List[str] = field(default_factory=lambda: ['BTC/USD', 'ETH/USD'])
    
    # Allocation percentages
    equity_allocation: float = 0.60   # 60% to equities
    crypto_allocation: float = 0.30   # 30% to crypto
    cash_reserve: float = 0.10        # 10% cash reserve
    
    # Trading parameters
    equity_loop_interval: int = 300   # 5 minutes
    max_position_pct: float = 0.10    # Max 10% per position
    min_trade_usd: float = 100.0      # Minimum trade size
    
    # Risk management
    portfolio_stop_loss: float = 0.15  # 15% portfolio stop
    position_stop_loss: float = 0.05   # 5% position stop
    
    # ML settings
    use_ml_signals: bool = True
    ml_confidence_threshold: float = 0.6


# =============================================================================
# ALPHA ENGINE
# =============================================================================

class AlphaEngine:
    """
    Main orchestrator combining crypto, ML, and equity strategies.
    """
    
    def __init__(self, config: Optional[AlphaConfig] = None, paper: bool = True):
        """Initialize the alpha engine."""
        self.config = config or AlphaConfig()
        self.paper = paper
        self.running = False
        
        # Initialize Alpaca clients
        api_key = os.getenv('ALPACA_API_KEY') or os.getenv('APCA_API_KEY_ID')
        secret_key = os.getenv('ALPACA_SECRET_KEY') or os.getenv('APCA_API_SECRET_KEY')
        
        if api_key and secret_key and ALPACA_AVAILABLE:
            self.trading_client = TradingClient(api_key, secret_key, paper=paper)
            self.data_client = StockHistoricalDataClient(api_key, secret_key)
            logger.info(f"Alpaca clients initialized (paper={paper})")
        else:
            self.trading_client = None
            self.data_client = None
            logger.warning("Alpaca API not available")
        
        # Initialize sub-modules
        self.crypto_trader = None
        self.ml_engine = None
        self.v38_universe = None
        
        self._init_modules()
        
        # State tracking
        self.equity_signals: Dict[str, Dict] = {}
        self.portfolio_value = 0.0
        self.daily_pnl = 0.0
    
    def _init_modules(self):
        """Initialize trading sub-modules."""
        # Crypto trader
        if CRYPTO_AVAILABLE:
            crypto_config = CryptoConfig(symbols=self.config.crypto_symbols)
            self.crypto_trader = CryptoTrader(config=crypto_config, paper=self.paper)
            logger.info("Crypto trader initialized")
        
        # ML engine
        if ML_AVAILABLE:
            self.ml_engine = DeepMLEngine()
            logger.info("ML engine initialized")
        
        # V38 Universe
        if V38_AVAILABLE:
            self.v38_universe = ExpandedUniverse()
            logger.info("V38 universe initialized")
    
    def get_account(self) -> Dict:
        """Get account information."""
        if not self.trading_client:
            return {"equity": 100000, "cash": 100000, "buying_power": 100000}
        
        try:
            account = self.trading_client.get_account()
            return {
                "equity": float(account.equity),
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "portfolio_value": float(account.portfolio_value),
                "day_pnl": float(getattr(account, 'daily_pnl', 0) or 0)
            }
        except Exception as e:
            logger.error(f"Error getting account: {e}")
            return {}
    
    def get_positions(self) -> Dict[str, Dict]:
        """Get all current positions."""
        if not self.trading_client:
            return {}
        
        try:
            positions = self.trading_client.get_all_positions()
            return {
                pos.symbol: {
                    "qty": float(pos.qty),
                    "avg_entry": float(pos.avg_entry_price),
                    "current_price": float(pos.current_price),
                    "market_value": float(pos.market_value),
                    "unrealized_pnl": float(pos.unrealized_pl),
                    "unrealized_pnl_pct": float(pos.unrealized_plpc) * 100
                }
                for pos in positions
            }
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return {}
    
    def get_stock_bars(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """Fetch historical stock bars."""
        if not self.data_client:
            return self._generate_mock_bars(symbol, days)
        
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Hour,
                start=datetime.now() - timedelta(days=days)
            )
            bars = self.data_client.get_stock_bars(request)
            
            if symbol in bars:
                df = bars[symbol].df.reset_index()
                return df
            return None
        except Exception as e:
            logger.error(f"Error fetching bars for {symbol}: {e}")
            return self._generate_mock_bars(symbol, days)
    
    def _generate_mock_bars(self, symbol: str, days: int) -> pd.DataFrame:
        """Generate mock data for testing."""
        base_prices = {'SPY': 580, 'QQQ': 520, 'AAPL': 240, 'NVDA': 140}
        base = base_prices.get(symbol, 100)
        periods = days * 8
        
        returns = np.random.normal(0.0002, 0.015, periods)
        close = base * np.cumprod(1 + returns)
        
        return pd.DataFrame({
            'timestamp': pd.date_range(end=datetime.now(), periods=periods, freq='H'),
            'open': close * (1 + np.random.uniform(-0.005, 0.005, periods)),
            'high': close * (1 + np.random.uniform(0, 0.01, periods)),
            'low': close * (1 - np.random.uniform(0, 0.01, periods)),
            'close': close,
            'volume': np.random.uniform(1e6, 5e6, periods)
        })
    
    def generate_equity_signals(self) -> Dict[str, Dict]:
        """Generate trading signals for equities."""
        signals = {}
        
        for symbol in self.config.equity_symbols:
            df = self.get_stock_bars(symbol, days=30)
            if df is None or len(df) < 50:
                continue
            
            close = df['close']
            
            # Technical indicators
            rsi = self._rsi(close, 14).iloc[-1]
            sma_20 = close.rolling(20).mean().iloc[-1]
            sma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else sma_20
            current_price = close.iloc[-1]
            momentum = (current_price / close.iloc[-20] - 1) * 100 if len(close) >= 20 else 0
            
            # Generate signal
            signal = "HOLD"
            strength = 0.0
            
            # RSI signals
            if rsi < 30:
                strength += 0.4
            elif rsi > 70:
                strength -= 0.4
            
            # Trend signals
            if current_price > sma_20 > sma_50:
                strength += 0.3
            elif current_price < sma_20 < sma_50:
                strength -= 0.3
            
            # Momentum
            if momentum > 5:
                strength += 0.3
            elif momentum < -5:
                strength -= 0.3
            
            if strength >= 0.5:
                signal = "BUY"
            elif strength <= -0.5:
                signal = "SELL"
            
            signals[symbol] = {
                "signal": signal,
                "strength": np.clip(strength, -1, 1),
                "price": current_price,
                "rsi": rsi,
                "momentum": momentum,
                "above_sma20": current_price > sma_20
            }
        
        # Add ML signals if available
        if self.ml_engine and self.config.use_ml_signals:
            signals = self._enhance_with_ml(signals)
        
        self.equity_signals = signals
        return signals
    
    def _enhance_with_ml(self, signals: Dict) -> Dict:
        """Enhance signals with ML predictions."""
        try:
            for symbol in signals:
                df = self.get_stock_bars(symbol, days=60)
                if df is not None and len(df) >= 100:
                    if not self.ml_engine.is_trained:
                        self.ml_engine.train(df, verbose=False)
                    
                    preds = self.ml_engine.predict(df.tail(10))
                    if len(preds) > 0:
                        ml_signal = preds['signal'].iloc[-1]
                        ml_conf = preds.get('ensemble_conf', preds.get('neural_net_conf'))
                        if ml_conf is not None:
                            ml_conf = ml_conf.iloc[-1]
                            if ml_conf >= self.config.ml_confidence_threshold:
                                signals[symbol]['ml_signal'] = ml_signal
                                signals[symbol]['ml_confidence'] = ml_conf
        except Exception as e:
            logger.warning(f"ML enhancement failed: {e}")
        
        return signals
    
    @staticmethod
    def _rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.inf)
        return 100 - (100 / (1 + rs))
    
    def execute_equity_trade(self, symbol: str, side: str, qty: float) -> bool:
        """Execute an equity trade."""
        if not self.trading_client:
            logger.info(f"[SIMULATED] {side} {qty:.2f} {symbol}")
            return True
        
        try:
            order = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side == "BUY" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            result = self.trading_client.submit_order(order)
            logger.info(f"Order submitted: {side} {qty} {symbol} - ID: {result.id}")
            return True
        except Exception as e:
            logger.error(f"Trade error: {e}")
            return False
    
    def run_equity_cycle(self) -> Dict:
        """Run one equity trading cycle."""
        logger.info("=" * 50)
        logger.info(f"Equity cycle: {datetime.now().isoformat()}")
        
        signals = self.generate_equity_signals()
        positions = self.get_positions()
        account = self.get_account()
        
        equity = account.get("equity", 0)
        max_per_position = equity * self.config.max_position_pct
        
        results = {"signals": [], "trades": []}
        
        for symbol, sig in signals.items():
            results["signals"].append({
                "symbol": symbol, "signal": sig["signal"],
                "strength": sig["strength"], "price": sig["price"]
            })
            
            logger.info(f"{symbol}: {sig['signal']} (str={sig['strength']:.2f}, "
                       f"RSI={sig['rsi']:.1f}, mom={sig['momentum']:.1f}%)")
            
            # Execute trades
            current_pos = positions.get(symbol, {})
            current_value = current_pos.get("market_value", 0)
            
            if sig["signal"] == "BUY" and current_value < max_per_position:
                allocation = min(max_per_position - current_value, 
                               self.config.min_trade_usd * 5)
                if allocation >= self.config.min_trade_usd:
                    qty = allocation / sig["price"]
                    success = self.execute_equity_trade(symbol, "BUY", qty)
                    results["trades"].append({
                        "symbol": symbol, "side": "BUY",
                        "qty": qty, "success": success
                    })
            
            elif sig["signal"] == "SELL" and current_value > 0:
                qty = current_pos.get("qty", 0)
                if qty > 0:
                    success = self.execute_equity_trade(symbol, "SELL", qty)
                    results["trades"].append({
                        "symbol": symbol, "side": "SELL",
                        "qty": qty, "success": success
                    })
        
        return results
    
    def run_crypto_cycle(self) -> Dict:
        """Run crypto trading cycle."""
        if not self.crypto_trader:
            return {"error": "Crypto trader not available"}
        
        return self.crypto_trader.run_cycle()
    
    def run_all_cycles(self) -> Dict:
        """Run both equity and crypto cycles."""
        results = {
            "timestamp": datetime.now().isoformat(),
            "equity": {},
            "crypto": {}
        }
        
        # Run cycles in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            equity_future = executor.submit(self.run_equity_cycle)
            crypto_future = executor.submit(self.run_crypto_cycle)
            
            results["equity"] = equity_future.result()
            results["crypto"] = crypto_future.result()
        
        return results
    
    def run_trading_loop(self):
        """Run continuous trading loop."""
        logger.info("=" * 60)
        logger.info("V39 ALPHA ENGINE STARTING")
        logger.info("=" * 60)
        logger.info(f"Equities: {self.config.equity_symbols}")
        logger.info(f"Crypto: {self.config.crypto_symbols}")
        logger.info(f"Paper mode: {self.paper}")
        
        self.running = True
        
        while self.running:
            try:
                results = self.run_all_cycles()
                
                # Log summary
                eq_trades = len(results.get("equity", {}).get("trades", []))
                cr_trades = len(results.get("crypto", {}).get("trades", []))
                logger.info(f"Cycle complete: {eq_trades} equity trades, {cr_trades} crypto trades")
                
                # Sleep until next cycle
                logger.info(f"Sleeping {self.config.equity_loop_interval}s...")
                time.sleep(self.config.equity_loop_interval)
                
            except KeyboardInterrupt:
                logger.info("Shutdown signal received")
                self.running = False
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(60)
        
        logger.info("Trading loop stopped")
    
    def get_status(self) -> Dict:
        """Get complete system status."""
        account = self.get_account()
        positions = self.get_positions()
        
        # Categorize positions
        equity_positions = {k: v for k, v in positions.items() 
                          if not any(c in k for c in ['BTC', 'ETH'])}
        crypto_positions = {k: v for k, v in positions.items() 
                          if any(c in k for c in ['BTC', 'ETH'])}
        
        total_equity_value = sum(p["market_value"] for p in equity_positions.values())
        total_crypto_value = sum(p["market_value"] for p in crypto_positions.values())
        
        return {
            "timestamp": datetime.now().isoformat(),
            "paper_mode": self.paper,
            "account": account,
            "positions": {
                "equity": equity_positions,
                "crypto": crypto_positions
            },
            "totals": {
                "equity_value": total_equity_value,
                "crypto_value": total_crypto_value,
                "total_invested": total_equity_value + total_crypto_value
            },
            "modules": {
                "crypto_available": CRYPTO_AVAILABLE,
                "ml_available": ML_AVAILABLE,
                "v38_available": V38_AVAILABLE
            }
        }
    
    def print_status(self):
        """Print formatted status."""
        status = self.get_status()
        
        print("\n" + "=" * 60)
        print("V39 ALPHA ENGINE STATUS")
        print("=" * 60)
        print(f"Timestamp: {status['timestamp']}")
        print(f"Mode: {'PAPER' if status['paper_mode'] else 'LIVE'}")
        
        print("\n--- ACCOUNT ---")
        acc = status['account']
        print(f"Equity: ${acc.get('equity', 0):,.2f}")
        print(f"Cash: ${acc.get('cash', 0):,.2f}")
        print(f"Buying Power: ${acc.get('buying_power', 0):,.2f}")
        
        print("\n--- EQUITY POSITIONS ---")
        for sym, pos in status['positions']['equity'].items():
            print(f"  {sym}: {pos['qty']:.2f} @ ${pos['avg_entry']:.2f} | "
                  f"P&L: ${pos['unrealized_pnl']:.2f} ({pos['unrealized_pnl_pct']:.1f}%)")
        
        print("\n--- CRYPTO POSITIONS ---")
        for sym, pos in status['positions']['crypto'].items():
            print(f"  {sym}: {pos['qty']:.6f} @ ${pos['avg_entry']:,.2f} | "
                  f"P&L: ${pos['unrealized_pnl']:.2f} ({pos['unrealized_pnl_pct']:.1f}%)")
        
        print("\n--- TOTALS ---")
        totals = status['totals']
        print(f"Equity Value: ${totals['equity_value']:,.2f}")
        print(f"Crypto Value: ${totals['crypto_value']:,.2f}")
        print(f"Total Invested: ${totals['total_invested']:,.2f}")
        
        print("\n--- MODULES ---")
        mods = status['modules']
        print(f"Crypto: {'✓' if mods['crypto_available'] else '✗'}")
        print(f"ML Engine: {'✓' if mods['ml_available'] else '✗'}")
        print(f"V38 Core: {'✓' if mods['v38_available'] else '✗'}")
        
        print("=" * 60 + "\n")


# =============================================================================
# CLI
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="V39 Alpha Engine - Trading Orchestrator")
    parser.add_argument("--trade-all", action="store_true", help="Start trading all strategies")
    parser.add_argument("--trade-equity", action="store_true", help="Trade equities only")
    parser.add_argument("--trade-crypto", action="store_true", help="Trade crypto only")
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--signals", action="store_true", help="Generate and show signals")
    parser.add_argument("--cycle", action="store_true", help="Run single trading cycle")
    parser.add_argument("--paper", action="store_true", default=True, help="Paper trading")
    parser.add_argument("--live", action="store_true", help="Live trading mode")
    
    args = parser.parse_args()
    
    paper = not args.live
    engine = AlphaEngine(paper=paper)
    
    if args.status:
        engine.print_status()
    
    elif args.signals:
        signals = engine.generate_equity_signals()
        print("\n=== EQUITY SIGNALS ===")
        for sym, sig in signals.items():
            print(f"{sym}: {sig['signal']} (strength={sig['strength']:.2f})")
        
        if engine.crypto_trader:
            print("\n=== CRYPTO SIGNALS ===")
            for sym in engine.config.crypto_symbols:
                signal = engine.crypto_trader.calculate_signals(sym)
                if signal:
                    print(f"{sym}: {signal.signal} (strength={signal.strength:.2f})")
    
    elif args.cycle:
        results = engine.run_all_cycles()
        print(json.dumps(results, indent=2, default=str))
    
    elif args.trade_all:
        engine.run_trading_loop()
    
    elif args.trade_equity:
        engine.running = True
        while engine.running:
            try:
                engine.run_equity_cycle()
                time.sleep(engine.config.equity_loop_interval)
            except KeyboardInterrupt:
                engine.running = False
    
    elif args.trade_crypto:
        if engine.crypto_trader:
            engine.crypto_trader.run_loop()
        else:
            print("Crypto trader not available")
    
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python v39_alpha_engine.py --status")
        print("  python v39_alpha_engine.py --signals")
        print("  python v39_alpha_engine.py --trade-all")
        print("  python v39_alpha_engine.py --trade-all --live")


if __name__ == "__main__":
    main()
