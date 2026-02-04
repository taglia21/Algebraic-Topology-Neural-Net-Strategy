"""
Demo: Autonomous Trading System
================================

Simple demonstration of the autonomous options trading engine.

This script shows how to:
1. Initialize the autonomous engine
2. Run a few trading cycles
3. Display statistics

Usage:
    python demo_autonomous_system.py
"""

import asyncio
import logging
from datetime import datetime

from src.options.autonomous_engine import AutonomousTradingEngine


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


async def demo_autonomous_trading():
    """
    Demonstrate autonomous trading for a few cycles.
    """
    print("="*70)
    print("AUTONOMOUS TRADING SYSTEM DEMO")
    print("="*70)
    print()
    
    # Initialize engine with $10,000 portfolio
    engine = AutonomousTradingEngine(
        portfolio_value=10000.0,
        paper=True,
        state_file="demo_trading_state.json",
    )
    
    print("Engine initialized!")
    print(f"Portfolio: ${engine.portfolio_value:,.0f}")
    print(f"Mode: {'PAPER' if engine.paper else 'LIVE'}")
    print()
    
    # Run 3 trading cycles
    print("Running 3 trading cycles...")
    print("(In production, this runs continuously during market hours)")
    print()
    
    try:
        for i in range(3):
            print(f"\n{'='*70}")
            print(f"DEMO CYCLE #{i+1}")
            print(f"{'='*70}\n")
            
            # Run one trading cycle
            await engine._trading_cycle()
            
            # Display stats
            print("\n📊 Current Statistics:")
            print(f"  Cycles Run: {engine.stats['cycles_run']}")
            print(f"  Signals Generated: {engine.stats['signals_generated']}")
            print(f"  Trades Executed: {engine.stats['trades_executed']}")
            print(f"  Trades Failed: {engine.stats['trades_failed']}")
            print(f"  Positions Closed: {engine.stats['positions_closed']}")
            print(f"  Open Positions: {len(engine.current_positions)}")
            
            if i < 2:
                print("\nWaiting 5 seconds before next cycle...")
                await asyncio.sleep(5)
    
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    
    # Final summary
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("\n📈 Final Statistics:")
    for key, value in engine.stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ State saved to: demo_trading_state.json")
    print("\nTo run the full autonomous system:")
    print("  python alpaca_options_monitor.py --mode autonomous --portfolio 10000")
    print()


async def demo_signal_generation():
    """
    Demonstrate just the signal generation component.
    """
    from src.options.signal_generator import SignalGenerator
    from src.options.universe import get_universe
    
    print("\n" + "="*70)
    print("SIGNAL GENERATION DEMO")
    print("="*70)
    
    generator = SignalGenerator()
    
    # Get full universe
    symbols = get_universe()
    print(f"\nScanning {len(symbols)} symbols from universe...")
    print(f"Symbols: {', '.join(symbols)}")
    
    # Generate signals
    signals = await generator.generate_all_signals(
        symbols=symbols[:3],  # Just first 3 for demo
        portfolio_delta=0.0,
    )
    
    print(f"\n✅ Generated {len(signals)} signals\n")
    
    # Display top 5 signals
    for i, signal in enumerate(signals[:5], 1):
        print(f"{i}. {signal.symbol} - {signal.strategy}")
        print(f"   Type: {signal.signal_type.value}")
        print(f"   Source: {signal.signal_source.value}")
        print(f"   Confidence: {signal.confidence:.1%}")
        print(f"   Reason: {signal.reason}")
        print()


async def demo_position_sizing():
    """
    Demonstrate position sizing with Kelly Criterion.
    """
    from src.options.position_sizer import MedallionPositionSizer
    
    print("\n" + "="*70)
    print("POSITION SIZING DEMO")
    print("="*70)
    
    sizer = MedallionPositionSizer()
    
    # Example: Size a position
    portfolio_value = 10000.0
    max_loss_per_contract = 200.0  # $200 max loss per spread
    signal_confidence = 0.75  # 75% confidence
    probability_of_profit = 0.65  # 65% PoP
    iv_rank = 60.0  # High IV
    
    print(f"\nPortfolio Value: ${portfolio_value:,.0f}")
    print(f"Max Loss per Contract: ${max_loss_per_contract:,.0f}")
    print(f"Signal Confidence: {signal_confidence:.1%}")
    print(f"Probability of Profit: {probability_of_profit:.1%}")
    print(f"IV Rank: {iv_rank:.0f}")
    
    position_size = sizer.calculate_position_size(
        portfolio_value=portfolio_value,
        max_loss_per_contract=max_loss_per_contract,
        signal_confidence=signal_confidence,
        probability_of_profit=probability_of_profit,
        iv_rank=iv_rank,
        current_portfolio_delta=0.0,
        position_delta_per_contract=0.0,
    )
    
    print(f"\n✅ Position Size Calculated:")
    print(f"   Contracts: {position_size.contracts}")
    print(f"   Dollar Amount: ${position_size.dollar_amount:,.0f}")
    print(f"   Risk Amount: ${position_size.risk_dollar_amount:,.0f}")
    print(f"   Risk Percent: {position_size.risk_percent:.2%}")
    print(f"   Kelly Fraction: {position_size.kelly_fraction:.2%}")
    print(f"   Confidence Mult: {position_size.confidence_multiplier:.2%}")
    print(f"   Volatility Mult: {position_size.volatility_multiplier:.2f}x")
    print(f"   Reason: {position_size.reason}")
    print()


async def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("🤖 AUTONOMOUS OPTIONS TRADING SYSTEM")
    print("Full Demo")
    print("="*70)
    print()
    
    # Demo 1: Signal Generation
    await demo_signal_generation()
    
    # Demo 2: Position Sizing
    await demo_position_sizing()
    
    # Demo 3: Full Autonomous Trading
    print("\n" + "="*70)
    print("Would you like to run the full autonomous trading demo?")
    print("This will run 3 trading cycles with signal generation,")
    print("position sizing, and simulated order execution.")
    print("="*70)
    
    response = input("\nRun full demo? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        await demo_autonomous_trading()
    else:
        print("\nFull demo skipped.")
    
    print("\n" + "="*70)
    print("✅ DEMO COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Review the code in src/options/")
    print("2. Run tests: pytest tests/test_signal_generator.py -v")
    print("3. Start autonomous trading:")
    print("   python alpaca_options_monitor.py --mode autonomous --portfolio 10000")
    print()


if __name__ == "__main__":
    asyncio.run(main())
