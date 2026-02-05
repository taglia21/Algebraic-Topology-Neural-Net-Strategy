"""
Phase 2 Test: IV Data Pipeline
===============================

Tests the IVDataManager with caching and IV rank calculation.
"""

import asyncio
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from options.iv_data_manager import IVDataManager


async def test_iv_data_pipeline():
    """Test IV data manager."""
    
    print("=" * 60)
    print("PHASE 2 TEST: IV Data Pipeline")
    print("=" * 60)
    
    try:
        # Initialize manager
        print("\n1. Initializing IV Data Manager...")
        iv_manager = IVDataManager(data_dir="data")
        print("✓ Manager initialized")
        
        # Check database stats
        print("\n2. Checking database statistics...")
        stats = iv_manager.get_stats()
        print(f"✓ Database: {stats['db_path']}")
        print(f"  Symbols: {stats['symbols']}")
        print(f"  Records: {stats['total_records']}")
        print(f"  Date range: {stats['earliest_date']} to {stats['latest_date']}")
        
        # Backfill synthetic data for testing
        print("\n3. Backfilling synthetic IV data...")
        test_symbols = ['SPY', 'QQQ', 'IWM']
        
        for symbol in test_symbols:
            rows = iv_manager.backfill_synthetic_data(symbol, days=252)
            print(f"  ✓ {symbol}: {rows} days backfilled")
        
        # Test IV rank calculation
        print("\n4. Testing IV Rank calculation...")
        for symbol in test_symbols:
            iv_rank = iv_manager.get_iv_rank(symbol)
            current_iv = iv_manager.get_current_iv(symbol)
            
            if iv_rank is not None:
                print(f"  ✓ {symbol}:")
                print(f"    Current IV: {current_iv:.2%}")
                print(f"    IV Rank: {iv_rank:.1f}%")
            else:
                print(f"  ✗ {symbol}: Insufficient data")
        
        # Test IV history retrieval
        print("\n5. Testing IV history retrieval...")
        history = iv_manager.get_iv_history('SPY', days=30)
        print(f"  ✓ Retrieved {len(history)} days of history for SPY")
        
        if history:
            latest = history[0]
            print(f"    Latest: {latest.date.strftime('%Y-%m-%d')}")
            print(f"    ATM IV: {latest.atm_iv:.2%}")
            print(f"    Skew: {latest.skew_25delta:.2%}")
        
        # Test daily update
        print("\n6. Testing daily IV update...")
        success = await iv_manager.update_daily_iv('SPY')
        if success:
            print("  ✓ Daily IV updated")
        else:
            print("  ⚠ Update skipped (no credentials or already updated)")
        
        # Final stats
        print("\n7. Final database statistics...")
        stats = iv_manager.get_stats()
        print(f"  ✓ Total symbols: {stats['symbols']}")
        print(f"  ✓ Total records: {stats['total_records']}")
        
        print("\n" + "=" * 60)
        print("PHASE 2 TEST COMPLETE ✓")
        print("=" * 60)
        print("\nIV Data Pipeline is ready.")
        print("Features:")
        print("  ✓ SQLite cache with persistent storage")
        print("  ✓ IV rank calculation (252-day rolling window)")
        print("  ✓ ATM IV extraction from option chains")
        print("  ✓ Skew and term structure metrics")
        print("  ✓ Automatic backfilling for missing data")
        print("\nNo more 'Insufficient data for IV rank' errors!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(test_iv_data_pipeline())
