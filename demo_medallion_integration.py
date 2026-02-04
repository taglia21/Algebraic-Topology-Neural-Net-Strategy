#!/usr/bin/env python3
"""
Quick Demo: Medallion-Enhanced Trading Engine
==============================================

Shows a live trading decision with Medallion mathematical analysis.
"""

import logging
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.enhanced_trading_engine import EnhancedTradingEngine
from src.position_sizer import PerformanceMetrics

# Configure logging for demo
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)

def demo():
    print("\n" + "="*70)
    print("MEDALLION-ENHANCED TRADING ENGINE DEMO")
    print("="*70)
    
    # Initialize engine
    print("\n🚀 Initializing Enhanced Trading Engine with Medallion Math...")
    engine = EnhancedTradingEngine()
    
    # Sample performance metrics
    metrics = PerformanceMetrics(
        total_trades=100,
        winning_trades=58,
        losing_trades=42,
        total_profit=14500,
        total_loss=-9200
    )
    
    # Analyze a symbol
    symbol = 'AAPL'
    portfolio_value = 100000
    
    print(f"\n📊 Analyzing {symbol} with $100,000 portfolio...")
    print("-"*70)
    
    decision = engine.analyze_opportunity(symbol, portfolio_value, metrics)
    
    # Display results
    print("\n" + "="*70)
    print(f"TRADING DECISION: {symbol}")
    print("="*70)
    
    print(f"\n📈 Signal: {decision.signal.value.upper()}")
    print(f"{'✅ TRADEABLE' if decision.is_tradeable else '❌ NOT TRADEABLE'}")
    
    print(f"\n📊 Analysis Scores:")
    print(f"  Multi-Timeframe:  {decision.mtf_score:.1f}/100")
    print(f"  Sentiment:        {decision.sentiment_score:+.2f}")
    print(f"  Combined:         {decision.combined_score:.2f}")
    print(f"  Confidence:       {decision.confidence:.1%}")
    
    # Show Medallion analysis
    if decision.metadata.get('medallion_analysis'):
        medallion = decision.metadata['medallion_analysis']
        print(f"\n🎯 Medallion Mathematical Analysis:")
        print(f"  Hurst Exponent:   {medallion['hurst_exponent']:.3f}")
        
        h = medallion['hurst_exponent']
        if h > 0.55:
            print(f"    → Trending market (use momentum)")
        elif h < 0.45:
            print(f"    → Mean-reverting (use stat arb)")
        else:
            print(f"    → Random walk (reduce exposure)")
        
        print(f"\n  Market Regime:    {medallion['regime']}")
        regime = medallion['regime']
        if regime == 'Bull':
            print(f"    → Bullish conditions")
        elif regime == 'Bear':
            print(f"    → Bearish conditions (position reduced 30%)")
        elif regime == 'HighVol':
            print(f"    → High volatility (position reduced 50%)")
        else:
            print(f"    → Sideways market")
        
        print(f"\n  Strategy:         {medallion['recommended_strategy']}")
        print(f"  Confidence:       {medallion['strategy_confidence']:.1%}")
        print(f"  O-U Z-Score:      {medallion['ou_signal']['z_score']:.2f}")
        print(f"  O-U Action:       {medallion['ou_signal']['action']}")
        print(f"  Half-Life:        {medallion.get('half_life_days', 0):.1f} days")
        
        # Regime probabilities
        print(f"\n  Regime Probabilities:")
        probs = medallion['regime_probabilities']
        regime_names = ['Bull', 'Bear', 'HighVol', 'Sideways']
        for name, prob in zip(regime_names, probs):
            bar = '█' * int(prob * 40)
            print(f"    {name:12s}: {prob:.1%} {bar}")
    else:
        print(f"\n⚠️  Medallion analysis unavailable (insufficient data)")
    
    print(f"\n💰 Position Sizing:")
    print(f"  Position Value:   ${decision.recommended_position_value:,.2f}")
    print(f"  Quantity:         {decision.recommended_quantity} shares")
    print(f"  Entry Price:      ${decision.entry_price:.2f}")
    
    print(f"\n🛡️  Risk Management:")
    print(f"  Stop Loss:        ${decision.stop_loss:.2f} ({((decision.stop_loss/decision.entry_price-1)*100):+.1f}%)")
    print(f"  Take Profits:")
    for i, tp in enumerate(decision.take_profits, 1):
        pct = (tp/decision.entry_price - 1) * 100
        print(f"    TP{i}: ${tp:.2f} (+{pct:.1f}%)")
    
    if decision.rejection_reasons:
        print(f"\n❌ Rejection Reasons:")
        for reason in decision.rejection_reasons:
            print(f"  • {reason}")
    
    print("\n" + "="*70)
    
    # Summary
    print("\n📝 Summary:")
    if decision.is_tradeable:
        print(f"  ✅ EXECUTE: Buy {decision.recommended_quantity} shares of {symbol} @ ${decision.entry_price:.2f}")
        print(f"  🎯 Target: ${decision.take_profits[0]:.2f} (+{((decision.take_profits[0]/decision.entry_price-1)*100):.1f}%)")
        print(f"  🛡️  Stop: ${decision.stop_loss:.2f} ({((decision.stop_loss/decision.entry_price-1)*100):+.1f}%)")
    else:
        print(f"  ❌ SKIP: Do not trade {symbol} at this time")
        print(f"  Reasons: {len(decision.rejection_reasons)} issues identified")
    
    print("\n" + "="*70)
    print("Demo complete! Medallion mathematical foundations are active.")
    print("="*70 + "\n")

if __name__ == "__main__":
    demo()
