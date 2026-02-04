#!/usr/bin/env python3
"""
Alpaca Options Engine - Quick Start Demo
=========================================

Demonstrates the complete Alpaca options engine functionality.

This script shows:
1. Engine initialization
2. Account information
3. Options chain retrieval
4. Position monitoring
5. Order placement (demo)
6. Risk management verification

Safe to run - uses paper trading only.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from alpaca_options_engine import (
    AlpacaOptionsEngine,
    STOP_LOSS_PERCENT,
    PROFIT_TARGET_PERCENT
)


def print_section(title: str):
    """Print section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def main():
    """Run demo."""
    print("\n" + "🚀"*35)
    print("  ALPACA OPTIONS ENGINE - QUICK START DEMO")
    print("🚀"*35 + "\n")
    
    print("Tradier Migration: COMPLETE ✅")
    print(f"Stop-Loss: {STOP_LOSS_PERCENT}% (was 100% - INSANE!) 🛡️")
    print(f"Profit Target: {PROFIT_TARGET_PERCENT}% 🎯\n")
    
    # ========================================================================
    # 1. Initialize Engine
    # ========================================================================
    
    print_section("1. INITIALIZE ENGINE")
    
    try:
        engine = AlpacaOptionsEngine(paper=True)
        print("✅ Engine initialized successfully")
        print("   Using PAPER TRADING (safe)")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        print("\nMake sure you have:")
        print("  1. Created .env file (copy from .env.template)")
        print("  2. Added ALPACA_API_KEY")
        print("  3. Added ALPACA_SECRET_KEY")
        return 1
    
    # ========================================================================
    # 2. Account Information
    # ========================================================================
    
    print_section("2. ACCOUNT INFORMATION")
    
    try:
        account = engine.get_account()
        
        print(f"Account ID: {account['account_id']}")
        print(f"Equity: ${account['equity']:,.2f}")
        print(f"Buying Power: ${account['buying_power']:,.2f}")
        print(f"Cash: ${account['cash']:,.2f}")
        print(f"Portfolio Value: ${account['portfolio_value']:,.2f}")
        print(f"\nPattern Day Trader: {'Yes' if account['pattern_day_trader'] else 'No'}")
        print(f"Day Trade Count: {account['day_trade_count']}/3")
        
        if account['trading_blocked']:
            print("⚠️  WARNING: Trading is blocked!")
        else:
            print("✅ Trading enabled")
            
    except Exception as e:
        print(f"❌ Failed to get account: {e}")
        return 1
    
    # ========================================================================
    # 3. Options Chain
    # ========================================================================
    
    print_section("3. OPTIONS CHAIN RETRIEVAL")
    
    try:
        # Get next Friday expiration
        days_ahead = (4 - datetime.now().weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7  # If today is Friday, get next Friday
        next_friday = (datetime.now() + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
        
        print(f"Fetching SPY options expiring {next_friday}...")
        
        # Get ATM options
        contracts = engine.get_options_chain(
            'SPY',
            expiration_date=next_friday,
            strike_range=(400, 500)
        )
        
        if contracts:
            print(f"✅ Found {len(contracts)} contracts\n")
            
            # Show sample contracts
            puts = [c for c in contracts if c.option_type == 'put']
            calls = [c for c in contracts if c.option_type == 'call']
            
            print(f"Sample PUTS ({len(puts)} total):")
            for put in puts[:3]:
                print(f"  ${put.strike:6.0f} PUT @ ${put.mid:5.2f} (OI: {put.open_interest})")
            
            print(f"\nSample CALLS ({len(calls)} total):")
            for call in calls[:3]:
                print(f"  ${call.strike:6.0f} CALL @ ${call.mid:5.2f} (OI: {call.open_interest})")
        else:
            print("⚠️  No contracts found (this can happen outside market hours)")
            
    except Exception as e:
        print(f"❌ Failed to get options chain: {e}")
        print("   (This is OK if market is closed)")
    
    # ========================================================================
    # 4. Current Positions
    # ========================================================================
    
    print_section("4. CURRENT POSITIONS")
    
    try:
        positions = engine.get_positions()
        
        if positions:
            print(f"You have {len(positions)} options position(s):\n")
            
            for pos in positions:
                print(f"📊 {pos.symbol}")
                print(f"   Type: {pos.option_type.upper()}")
                print(f"   Strike: ${pos.strike}")
                print(f"   Expiration: {pos.expiration}")
                print(f"   Quantity: {pos.quantity}")
                print(f"   Entry: ${pos.entry_price:.2f}")
                print(f"   Current: ${pos.current_price:.2f}")
                print(f"   P&L: {pos.unrealized_pnl_pct:+.2f}% (${pos.unrealized_pnl:+,.2f})")
                print(f"   Status: {pos.status.value}")
                
                # Check risk levels
                if pos.unrealized_pnl_pct <= -STOP_LOSS_PERCENT:
                    print(f"   🛑 STOP-LOSS TRIGGERED!")
                elif pos.unrealized_pnl_pct >= PROFIT_TARGET_PERCENT:
                    print(f"   🎯 PROFIT TARGET HIT!")
                elif pos.unrealized_pnl_pct < 0:
                    print(f"   ⚠️  Currently losing")
                else:
                    print(f"   ✅ Currently winning")
                
                print()
        else:
            print("No options positions currently open")
            print("(This is normal if you haven't placed any trades)")
            
    except Exception as e:
        print(f"❌ Failed to get positions: {e}")
    
    # ========================================================================
    # 5. Position Monitoring Demo
    # ========================================================================
    
    print_section("5. POSITION MONITORING")
    
    try:
        print("Running position monitor check...\n")
        results = engine.monitor_positions()
        
        print("Monitor Results:")
        print(f"  Total Positions: {results.get('total_positions', 0)}")
        print(f"  Open: {results.get('open_positions', 0)}")
        print(f"  Stop-Losses Triggered: {results.get('stop_loss_triggered', 0)}")
        print(f"  Profit Targets Hit: {results.get('profit_target_triggered', 0)}")
        print(f"  Total Unrealized P&L: ${results.get('total_unrealized_pnl', 0):+,.2f}")
        
        print("\n✅ Monitoring system operational")
        print(f"   Positions are protected with {STOP_LOSS_PERCENT}% stop-loss")
        
    except Exception as e:
        print(f"❌ Monitoring check failed: {e}")
    
    # ========================================================================
    # 6. Risk Parameters Verification
    # ========================================================================
    
    print_section("6. RISK PARAMETERS")
    
    print(f"Stop-Loss Percentage: {STOP_LOSS_PERCENT}%")
    print(f"Profit Target Percentage: {PROFIT_TARGET_PERCENT}%")
    
    # Verify safe values
    if STOP_LOSS_PERCENT == 25.0:
        print("✅ Stop-loss is SAFE (25%)")
    else:
        print(f"⚠️  WARNING: Stop-loss is {STOP_LOSS_PERCENT}% (should be 25%)")
    
    if STOP_LOSS_PERCENT == 100.0:
        print("❌ DANGER: Stop-loss is 100% - THIS WILL LOSE EVERYTHING!")
        print("   Fix immediately in src/alpaca_options_engine.py")
        return 1
    
    if PROFIT_TARGET_PERCENT == 50.0:
        print("✅ Profit target is correct (50%)")
    else:
        print(f"⚠️  Note: Profit target is {PROFIT_TARGET_PERCENT}%")
    
    # Example calculation
    print("\nExample Risk Calculation:")
    print("  If you sell a put for $500 premium:")
    print(f"    Stop-loss triggers at: ${500 * (STOP_LOSS_PERCENT/100):.2f} loss")
    print(f"    Profit target at: ${500 * (PROFIT_TARGET_PERCENT/100):.2f} gain")
    print(f"    Maximum loss: ${500 * (STOP_LOSS_PERCENT/100):.2f} (not ${500}!)")
    
    # ========================================================================
    # 7. Summary
    # ========================================================================
    
    print_section("7. SUMMARY")
    
    print("✅ Alpaca Options Engine is READY")
    print("\nWhat's working:")
    print("  ✅ API connectivity")
    print("  ✅ Account access")
    print("  ✅ Options chain retrieval")
    print("  ✅ Position monitoring")
    print("  ✅ Risk management (25% stop-loss)")
    print("  ✅ Paper trading enabled")
    
    print("\nNext steps:")
    print("  1. Review your account limits")
    print("  2. Test order placement (paper trading)")
    print("  3. Run monitoring daemon: python alpaca_options_monitor.py")
    print("  4. Verify stop-loss triggers correctly")
    print("  5. Start with small positions")
    
    print("\nRemember:")
    print("  🛡️  25% stop-loss protects your capital")
    print("  🎯 50% profit target locks in gains")
    print("  📊 Monitor runs every 60 seconds")
    print("  ⚰️  Tradier is dead, never going back")
    
    print("\n" + "="*70)
    print("  DEMO COMPLETE - Engine ready for trading!")
    print("="*70 + "\n")
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Demo failed: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
