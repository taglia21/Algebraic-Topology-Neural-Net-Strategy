#!/usr/bin/env python3
"""
Paper Trading Runner
====================
Main entry point for running the Team of Rivals trading system in paper mode.

Usage:
    python run_paper_trading.py
"""

import asyncio
import sys
import os
from datetime import datetime
import logging

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.tda_strategy import TDAStrategy
from src.v50_options_alpha_engine import V50OptionsAlphaEngine
from src.risk.risk_manager import RiskManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_full_system_test():
    """Run a complete test of all system components."""
    import yfinance as yf
    
    print("=" * 70)
    print("TEAM OF RIVALS - FULL SYSTEM TEST")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)
    
    # Initialize components
    print("\n[1] Initializing Components...")
    tda = TDAStrategy(lookback_window=60)
    options = V50OptionsAlphaEngine(paper_trading=True)
    risk = RiskManager()
    print("   ✓ TDA Strategy")
    print("   ✓ Options Alpha Engine")
    print("   ✓ Risk Manager")
    
    # Test universe
    universe = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA']
    print(f"\n[2] Analyzing Universe: {universe}")
    
    for symbol in universe:
        print(f"\n--- {symbol} ---")
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period='3mo', interval='1d')
            
            if df.empty:
                print(f"   No data for {symbol}")
                continue
            
            current_price = df['Close'].iloc[-1]
            print(f"   Price: ${current_price:.2f}")
            
            # TDA Analysis
            analysis = tda.analyze(df)
            print(f"   Regime: {analysis['regime']}")
            print(f"   Persistence: {analysis['persistence_score']:.2f}")
            print(f"   Volatility: {analysis['volatility']:.2%}")
            
            # NN Prediction
            prediction = tda.predict(df)
            print(f"   NN Signal: {prediction:.4f}")
            
            # Generate options signal
            tda_signal = {
                'persistence_score': analysis['persistence_score'],
                'trend_strength': analysis['trend_strength'],
                'regime': analysis['regime']
            }
            
            signal = options.generate_signal(
                symbol=symbol,
                underlying_price=current_price,
                tda_signal=tda_signal,
                nn_prediction=prediction
            )
            
            if signal:
                print(f"   SIGNAL: {signal.strategy.value}")
                print(f"   Direction: {signal.direction}")
                print(f"   Confidence: {signal.confidence:.2%}")
            else:
                print("   SIGNAL: None (below threshold)")
            
        except Exception as e:
            print(f"   Error: {e}")
    
    # System status
    print("\n[3] System Status")
    status = options.get_status()
    print(f"   Portfolio Value: ${status['portfolio_value']:,.2f}")
    print(f"   Paper Trading: {status['paper_trading']}")
    print(f"   Signals Generated: {status['signals_generated']}")
    
    print("\n" + "=" * 70)
    print("SYSTEM TEST COMPLETE - READY FOR PAPER TRADING")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    run_full_system_test()

