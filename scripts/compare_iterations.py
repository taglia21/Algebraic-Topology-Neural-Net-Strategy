#!/usr/bin/env python3
"""Compare results across optimization iterations.

Compares baseline, Iteration 1, and Iteration 2 results.
"""

import json
import os
from datetime import datetime

RESULTS_DIR = '/workspaces/Algebraic-Topology-Neural-Net-Strategy/results'

def load_json_safe(path: str) -> dict:
    """Load JSON file safely."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def format_pct(val: float) -> str:
    """Format as percentage."""
    return f"{val:.2f}%" if val is not None else "N/A"

def format_float(val: float, decimals: int = 2) -> str:
    """Format float with specified decimals."""
    return f"{val:.{decimals}f}" if val is not None else "N/A"

def main():
    """Generate comparison report."""
    print("=" * 80)
    print("OPTIMIZATION ITERATION COMPARISON REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Load results
    baseline_path = os.path.join(RESULTS_DIR, 'multiasset_backtest_baseline.json')
    iteration1_path = os.path.join(RESULTS_DIR, 'multiasset_backtest.json')  # Current results
    
    baseline = load_json_safe(baseline_path)
    current = load_json_safe(iteration1_path)
    
    if current is None:
        print("ERROR: No current results found!")
        return
    
    # Get metadata and results
    metadata = current.get('metadata', {})
    per_asset = current.get('per_asset', {})
    portfolio_eq = current.get('portfolio_equal_weight', {})
    portfolio_wgt = current.get('portfolio_performance_weighted', {})
    
    print(f"\nScenario: {metadata.get('train_period', 'N/A')} → {metadata.get('test_period', 'N/A')}")
    print("-" * 80)
    
    print("\n" + "=" * 80)
    print("PORTFOLIO COMPARISON (Equal Weight)")
    print("=" * 80)
    print(f"{'Metric':<30} {'Current':>15}")
    print("-" * 50)
    print(f"{'Sharpe Ratio (gross):':<30} {format_float(portfolio_eq.get('sharpe_ratio')):>15}")
    print(f"{'Sharpe Ratio (net):':<30} {format_float(portfolio_eq.get('sharpe_ratio_net')):>15}")
    print(f"{'Return (gross):':<30} {format_pct(portfolio_eq.get('total_return', 0) * 100):>15}")
    print(f"{'Return (net):':<30} {format_pct(portfolio_eq.get('total_return_net', 0) * 100):>15}")
    print(f"{'Max Drawdown:':<30} {format_pct(portfolio_eq.get('max_drawdown', 0) * 100):>15}")
    print(f"{'Total Trades:':<30} {portfolio_eq.get('total_trades', 0):>15}")
    print(f"{'Turnover:':<30} {format_float(portfolio_eq.get('turnover', 0)):>15}x")
    
    print("\n" + "=" * 80)
    print("PORTFOLIO COMPARISON (Risk-Weighted)")
    print("=" * 80)
    print(f"{'Metric':<30} {'Current':>15}")
    print("-" * 50)
    print(f"{'Sharpe Ratio (gross):':<30} {format_float(portfolio_wgt.get('sharpe_ratio')):>15}")
    print(f"{'Sharpe Ratio (net):':<30} {format_float(portfolio_wgt.get('sharpe_ratio_net')):>15}")
    print(f"{'Return (gross):':<30} {format_pct(portfolio_wgt.get('total_return', 0) * 100):>15}")
    print(f"{'Return (net):':<30} {format_pct(portfolio_wgt.get('total_return_net', 0) * 100):>15}")
    print(f"{'Max Drawdown:':<30} {format_pct(portfolio_wgt.get('max_drawdown', 0) * 100):>15}")
    print(f"{'Weights:':<30} {portfolio_wgt.get('weights', {})}'")
    
    # Per-asset breakdown
    assets = per_asset
    print("\n" + "=" * 80)
    print("PER-ASSET PERFORMANCE")
    print("=" * 80)
    print(f"{'Asset':<8} {'Sharpe':>10} {'Sh_net':>10} {'Return':>10} {'Ret_net':>10} {'Trades':>8}")
    print("-" * 60)
    
    positive_sharpe_count = 0
    for ticker, metrics in assets.items():
        sharpe = metrics.get('sharpe_ratio', 0)
        sharpe_net = metrics.get('sharpe_ratio_net', 0)
        ret = metrics.get('total_return', 0) * 100
        ret_net = metrics.get('total_return_net', 0) * 100
        trades = metrics.get('num_trades', 0)
        
        if sharpe_net > 0:
            positive_sharpe_count += 1
        
        print(f"{ticker:<8} {format_float(sharpe):>10} {format_float(sharpe_net):>10} "
              f"{format_pct(ret):>10} {format_pct(ret_net):>10} {trades:>8}")
    
    print("-" * 60)
    print(f"Assets with positive Sharpe (net): {positive_sharpe_count}/{len(assets)}")
    
    # Optimization settings
    print("\n" + "=" * 80)
    print("ITERATION 2 OPTIMIZATION SETTINGS")
    print("=" * 80)
    print("""
    Regime Detection: ENABLED
    - MarketRegimeDetector with TradingCondition (FAVORABLE/NEUTRAL/UNFAVORABLE)
    - Skip trades in UNFAVORABLE conditions (high vol + bear market)
    - Adjust thresholds based on regime (+0.02 for UNFAVORABLE, +0.01 for NEUTRAL)
    - Position size multiplier (0.0-1.25) based on regime
    
    Volatility-Adaptive Sizing: ENABLED
    - Target volatility = historical median
    - Position scaling = target_vol / current_vol (capped 0.5x - 1.5x)
    - Integrated with regime multiplier
    
    Signal Filters (from Iteration 1): ENABLED
    - RSI filter: period=14, oversold=45, overbought=55
    - Volatility filter: threshold=35%
    
    Position Sizing:
    - Half-Kelly: kelly_fraction=0.50
    - Max position: 15% of account
    """)
    
    # Summary assessment
    print("\n" + "=" * 80)
    print("ASSESSMENT")
    print("=" * 80)
    
    wgt_sharpe_net = portfolio_wgt.get('sharpe_ratio_net', 0)
    total_trades = portfolio_eq.get('total_trades', 0)
    max_dd = portfolio_eq.get('max_drawdown', 0) * 100
    
    print(f"\nTarget: Portfolio Sharpe > 1.5, 4/5 assets Sharpe > 0.2, Max DD < 5%")
    print(f"\nResults:")
    
    # Portfolio Sharpe check
    if wgt_sharpe_net >= 1.5:
        print(f"  ✓ Portfolio Sharpe (weighted, net): {format_float(wgt_sharpe_net)} >= 1.5")
    else:
        print(f"  ✗ Portfolio Sharpe (weighted, net): {format_float(wgt_sharpe_net)} < 1.5")
    
    # Asset Sharpe check
    assets_above_threshold = sum(1 for m in assets.values() if m.get('sharpe_ratio_net', 0) > 0.2)
    if assets_above_threshold >= 4:
        print(f"  ✓ Assets with Sharpe > 0.2: {assets_above_threshold}/5 >= 4")
    else:
        print(f"  ✗ Assets with Sharpe > 0.2: {assets_above_threshold}/5 < 4")
    
    # Max DD check
    if max_dd <= 5:
        print(f"  ✓ Max Drawdown: {format_pct(max_dd)} <= 5%")
    else:
        print(f"  ✗ Max Drawdown: {format_pct(max_dd)} > 5%")
    
    # Trade reduction
    print(f"\n  Trade count: {total_trades} (target: quality over quantity)")
    
    # Save report
    report_path = os.path.join(RESULTS_DIR, 'iteration2_comparison.txt')
    
    # Re-run with output to file
    import sys
    from io import StringIO
    
    # Capture all output
    old_stdout = sys.stdout
    sys.stdout = buffer = StringIO()
    
    # Re-print everything
    print("=" * 80)
    print("OPTIMIZATION ITERATION 2 COMPARISON REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    train_period = metadata.get('train_period', 'N/A')
    test_period = metadata.get('test_period', 'N/A')
    print(f"\nScenario: {train_period} → {test_period}")
    print("-" * 80)
    print(f"\nPortfolio (Equal Weight):")
    print(f"  Sharpe (net): {format_float(portfolio_eq.get('sharpe_ratio_net'))}")
    print(f"  Return (net): {format_pct(portfolio_eq.get('total_return_net', 0) * 100)}")
    print(f"  Max DD: {format_pct(portfolio_eq.get('max_drawdown', 0) * 100)}")
    print(f"  Trades: {portfolio_eq.get('total_trades', 0)}")
    print(f"\nPortfolio (Risk-Weighted):")
    print(f"  Sharpe (net): {format_float(portfolio_wgt.get('sharpe_ratio_net'))}")
    print(f"  Return (net): {format_pct(portfolio_wgt.get('total_return_net', 0) * 100)}")
    print(f"  Weights: {portfolio_wgt.get('weights', {})}")
    print(f"\nPer-Asset:")
    for ticker, metrics in assets.items():
        print(f"  {ticker}: Sharpe={format_float(metrics.get('sharpe_ratio_net'))}, "
              f"Return={format_pct(metrics.get('total_return_net', 0) * 100)}, "
              f"Trades={metrics.get('num_trades', 0)}")
    print(f"\nTarget Achievement:")
    print(f"  Portfolio Sharpe > 1.5: {'✓' if wgt_sharpe_net >= 1.5 else '✗'} ({format_float(wgt_sharpe_net)})")
    print(f"  4/5 assets Sharpe > 0.2: {'✓' if assets_above_threshold >= 4 else '✗'} ({assets_above_threshold}/5)")
    print(f"  Max DD < 5%: {'✓' if max_dd <= 5 else '✗'} ({format_pct(max_dd)})")
    
    report_content = buffer.getvalue()
    sys.stdout = old_stdout
    
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    print(f"\nReport saved to: {report_path}")
    

if __name__ == '__main__':
    main()
