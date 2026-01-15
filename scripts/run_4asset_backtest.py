#!/usr/bin/env python3
"""Run full 3-year backtest with 4-asset portfolio (no QQQ).

Tests the optimized portfolio composition from Iteration 3 analysis.
"""

import os
import sys
import subprocess
from datetime import datetime

# Temporarily modify TICKERS in main_multiasset.py, run, then restore
MAIN_FILE = '/workspaces/Algebraic-Topology-Neural-Net-Strategy/main_multiasset.py'
RESULTS_DIR = '/workspaces/Algebraic-Topology-Neural-Net-Strategy/results'


def run_4asset_backtest():
    """Run backtest with 4-asset portfolio."""
    
    print("=" * 60)
    print("ITERATION 3: 4-ASSET PORTFOLIO VALIDATION")
    print("=" * 60)
    print("\nConfiguration:")
    print("  Assets: SPY, IWM, XLF, XLK (no QQQ)")
    print("  Period: 2022-01-01 to 2025-12-31 (full 3 years)")
    print("  Features: V1.3 TDA + Regime Detection + Signal Filters")
    
    # Read current file
    with open(MAIN_FILE, 'r') as f:
        original_content = f.read()
    
    # Modify TICKERS to remove QQQ
    modified_content = original_content.replace(
        'TICKERS = ["SPY", "QQQ", "IWM", "XLF", "XLK"]',
        'TICKERS = ["SPY", "IWM", "XLF", "XLK"]  # V1.4: 4-asset portfolio (no QQQ)'
    )
    
    try:
        # Write modified file
        with open(MAIN_FILE, 'w') as f:
            f.write(modified_content)
        
        print("\n  Modified TICKERS to 4-asset portfolio")
        print("  Running backtest...")
        
        # Run backtest
        result = subprocess.run(
            ['python', 'main_multiasset.py', '--start=2022-01-01', '--end=2025-12-31'],
            cwd='/workspaces/Algebraic-Topology-Neural-Net-Strategy',
            capture_output=True,
            text=True,
            timeout=600
        )
        
        # Save output
        output_path = os.path.join(RESULTS_DIR, '4asset_backtest_output.txt')
        with open(output_path, 'w') as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n\nSTDERR:\n" + result.stderr)
        
        print(f"\n  Output saved to: {output_path}")
        
        # Print key results from output
        output_lines = result.stdout.split('\n')
        in_summary = False
        for line in output_lines:
            if 'PORTF_WGT' in line or 'PORTF_EQ' in line:
                print(f"  {line.strip()}")
            if 'WGT Sharpe_net' in line or 'WGT Return_net' in line:
                print(f"  {line.strip()}")
            if 'Total Trades' in line:
                print(f"  {line.strip()}")
        
        return result.returncode == 0
        
    finally:
        # Restore original file
        with open(MAIN_FILE, 'w') as f:
            f.write(original_content)
        print("\n  Restored original TICKERS configuration")


def run_5asset_qqq_filtered_backtest():
    """Run backtest with QQQ filtered to FAVORABLE-only conditions."""
    
    print("\n" + "=" * 60)
    print("ALTERNATIVE: 5-ASSET WITH QQQ FAVORABLE-ONLY FILTER")
    print("=" * 60)
    print("\nConfiguration:")
    print("  Assets: SPY, QQQ, IWM, XLF, XLK")
    print("  QQQ Filter: Trade only in FAVORABLE conditions")
    print("  Period: 2022-01-01 to 2025-12-31")
    
    # This would require more complex code changes to ensemble_strategy.py
    # For now, skip this and use the 4-asset results
    print("\n  (Skipping - requires ensemble_strategy modification)")
    print("  Recommendation: Use 4-asset portfolio based on analysis")
    

if __name__ == '__main__':
    success = run_4asset_backtest()
    
    if success:
        print("\n" + "=" * 60)
        print("4-ASSET BACKTEST COMPLETE")
        print("=" * 60)
    else:
        print("\n  ERROR: Backtest failed")
