#!/usr/bin/env python3
"""
Compare baseline vs optimized backtest results.

This script generates a detailed comparison report showing the impact
of optimization changes (Half-Kelly, signal filters) on trading performance.

Usage:
    python scripts/compare_optimization.py [--baseline FILE] [--optimized FILE]
"""

import json
import sys
import os
from datetime import datetime


def load_results(filepath):
    """Load backtest results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def format_pct(value, mult=100):
    """Format value as percentage."""
    return f"{value * mult:.2f}%"


def format_change(before, after, as_pct=False):
    """Format change with +/- prefix."""
    change = after - before
    pct_change = (change / before * 100) if before != 0 else 0
    
    if as_pct:
        return f"{change:+.2f}%", f"{pct_change:+.1f}%"
    return f"{change:+.4f}", f"{pct_change:+.1f}%"


def compare_backtests(baseline_file, optimized_file, output_file=None):
    """Generate detailed comparison report."""
    
    baseline = load_results(baseline_file)
    optimized = load_results(optimized_file)
    
    # Build report
    lines = []
    
    lines.append("=" * 80)
    lines.append(" OPTIMIZATION IMPACT REPORT".center(80))
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Baseline: {baseline_file}")
    lines.append(f"Optimized: {optimized_file}")
    lines.append("")
    
    # Configuration comparison
    lines.append("=" * 80)
    lines.append(" CONFIGURATION CHANGES")
    lines.append("=" * 80)
    lines.append("")
    lines.append("BEFORE (Baseline - Quarter-Kelly):")
    lines.append("  - Kelly Fraction: 0.25")
    lines.append("  - Max Position Size: 10%")
    lines.append("  - Max Portfolio Heat: 20%")
    lines.append("  - Risk Per Trade: 1%")
    lines.append("  - Signal Filter: Disabled")
    lines.append("")
    lines.append("AFTER (Optimized - Half-Kelly + Filters):")
    lines.append("  - Kelly Fraction: 0.50")
    lines.append("  - Max Position Size: 15%")
    lines.append("  - Max Portfolio Heat: 35%")
    lines.append("  - Risk Per Trade: 2%")
    lines.append("  - Signal Filter: RSI(35/65) + Vol(30%)")
    lines.append("")
    
    # Portfolio-level comparison
    lines.append("=" * 80)
    lines.append(" PORTFOLIO PERFORMANCE")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"{'Metric':<28} {'Baseline':>12} {'Optimized':>12} {'Change':>10} {'% Chg':>8}")
    lines.append("-" * 80)
    
    # Extract portfolio metrics
    b_port = baseline.get('portfolio', baseline.get('summary', {}))
    o_port = optimized.get('portfolio', optimized.get('summary', {}))
    
    metrics = [
        ('Total Return', 'total_return', 100, '%'),
        ('Sharpe Ratio', 'sharpe_ratio', 1, ''),
        ('Sharpe Ratio (Net)', 'sharpe_ratio_net', 1, ''),
        ('Max Drawdown', 'max_drawdown', 100, '%'),
        ('Win Rate', 'win_rate', 100, '%'),
        ('Total Trades', 'num_trades', 1, ''),
        ('Avg Trade Return', 'avg_trade_return', 100, '%'),
    ]
    
    for name, key, mult, unit in metrics:
        b_val = b_port.get(key, 0) * mult
        o_val = o_port.get(key, 0) * mult
        change = o_val - b_val
        pct_chg = (change / b_val * 100) if b_val != 0 else 0
        
        b_str = f"{b_val:.2f}{unit}" if unit else f"{b_val:.2f}"
        o_str = f"{o_val:.2f}{unit}" if unit else f"{o_val:.2f}"
        chg_str = f"{change:+.2f}"
        pct_str = f"{pct_chg:+.1f}%"
        
        lines.append(f"{name:<28} {b_str:>12} {o_str:>12} {chg_str:>10} {pct_str:>8}")
    
    # Risk Management metrics
    lines.append("")
    lines.append("=" * 80)
    lines.append(" RISK MANAGEMENT METRICS")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"{'Metric':<35} {'Baseline':>12} {'Optimized':>12}")
    lines.append("-" * 65)
    
    risk_metrics = [
        ('Max Portfolio Heat Reached', 'max_portfolio_heat_reached', 100, '%'),
        ('Stop-Loss Exits', 'num_stopped_out', 1, ''),
        ('Take-Profit Exits', 'num_take_profit_hits', 1, ''),
    ]
    
    for name, key, mult, unit in risk_metrics:
        b_val = b_port.get(key, 0) * mult
        o_val = o_port.get(key, 0) * mult
        
        b_str = f"{b_val:.2f}{unit}" if unit else f"{b_val:.2f}"
        o_str = f"{o_val:.2f}{unit}" if unit else f"{o_val:.2f}"
        
        lines.append(f"{name:<35} {b_str:>12} {o_str:>12}")
    
    # Per-asset breakdown
    lines.append("")
    lines.append("=" * 80)
    lines.append(" PER-ASSET SHARPE RATIO COMPARISON")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"{'Asset':<8} {'Baseline':>12} {'Optimized':>12} {'Improvement':>12} {'Win Rate Δ':>12}")
    lines.append("-" * 60)
    
    assets = ['SPY', 'QQQ', 'IWM', 'XLF', 'XLK']
    
    for asset in assets:
        b_asset = baseline.get('per_asset', {}).get(asset, {})
        o_asset = optimized.get('per_asset', {}).get(asset, {})
        
        b_sharpe = b_asset.get('sharpe_ratio_net', b_asset.get('sharpe_ratio', 0))
        o_sharpe = o_asset.get('sharpe_ratio_net', o_asset.get('sharpe_ratio', 0))
        sharpe_improvement = o_sharpe - b_sharpe
        
        b_wr = b_asset.get('win_rate', 0) * 100
        o_wr = o_asset.get('win_rate', 0) * 100
        wr_change = o_wr - b_wr
        
        lines.append(f"{asset:<8} {b_sharpe:>12.3f} {o_sharpe:>12.3f} {sharpe_improvement:>+12.3f} {wr_change:>+11.1f}%")
    
    # Calculate aggregates
    b_sharpes = [baseline.get('per_asset', {}).get(a, {}).get('sharpe_ratio_net', 0) for a in assets]
    o_sharpes = [optimized.get('per_asset', {}).get(a, {}).get('sharpe_ratio_net', 0) for a in assets]
    
    lines.append("-" * 60)
    lines.append(f"{'AVERAGE':<8} {sum(b_sharpes)/len(b_sharpes):>12.3f} {sum(o_sharpes)/len(o_sharpes):>12.3f} {(sum(o_sharpes)-sum(b_sharpes))/len(o_sharpes):>+12.3f}")
    
    # Trade count comparison
    lines.append("")
    lines.append("=" * 80)
    lines.append(" TRADE COUNT COMPARISON (Filter Impact)")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"{'Asset':<8} {'Baseline':>12} {'Optimized':>12} {'Reduction':>12} {'Reduction %':>12}")
    lines.append("-" * 60)
    
    total_b_trades = 0
    total_o_trades = 0
    
    for asset in assets:
        b_trades = baseline.get('per_asset', {}).get(asset, {}).get('num_trades', 0)
        o_trades = optimized.get('per_asset', {}).get(asset, {}).get('num_trades', 0)
        
        total_b_trades += b_trades
        total_o_trades += o_trades
        
        reduction = b_trades - o_trades
        reduction_pct = (reduction / b_trades * 100) if b_trades > 0 else 0
        
        lines.append(f"{asset:<8} {b_trades:>12} {o_trades:>12} {reduction:>12} {reduction_pct:>+11.1f}%")
    
    lines.append("-" * 60)
    total_reduction = total_b_trades - total_o_trades
    total_reduction_pct = (total_reduction / total_b_trades * 100) if total_b_trades > 0 else 0
    lines.append(f"{'TOTAL':<8} {total_b_trades:>12} {total_o_trades:>12} {total_reduction:>12} {total_reduction_pct:>+11.1f}%")
    
    # Summary
    lines.append("")
    lines.append("=" * 80)
    lines.append(" SUMMARY")
    lines.append("=" * 80)
    lines.append("")
    
    # Calculate key improvements
    avg_b_sharpe = sum(b_sharpes) / len(b_sharpes)
    avg_o_sharpe = sum(o_sharpes) / len(o_sharpes)
    sharpe_improvement = avg_o_sharpe - avg_b_sharpe
    
    b_total_return = b_port.get('total_return', 0)
    o_total_return = o_port.get('total_return', 0)
    return_improvement = (o_total_return - b_total_return) * 100
    
    lines.append(f"✓ Average Sharpe Improvement: {sharpe_improvement:+.3f}")
    lines.append(f"✓ Total Return Improvement: {return_improvement:+.2f}%")
    lines.append(f"✓ Trade Reduction (Filter Impact): {total_reduction_pct:.1f}%")
    lines.append("")
    
    # Recommendations
    if avg_o_sharpe >= 1.0:
        lines.append("🎯 TARGET ACHIEVED: Average Sharpe >= 1.0 (Production Ready)")
    elif avg_o_sharpe >= 0.7:
        lines.append("⚠️  PARTIAL SUCCESS: Average Sharpe 0.7-1.0 (Needs Fine-tuning)")
    else:
        lines.append("❌ BELOW TARGET: Average Sharpe < 0.7 (Further Optimization Required)")
    
    lines.append("")
    lines.append("=" * 80)
    
    # Print report
    report = "\n".join(lines)
    print(report)
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {output_file}")
    
    return {
        'avg_sharpe_before': avg_b_sharpe,
        'avg_sharpe_after': avg_o_sharpe,
        'sharpe_improvement': sharpe_improvement,
        'trade_reduction_pct': total_reduction_pct
    }


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare baseline vs optimized backtest results')
    parser.add_argument('--baseline', default='results/multiasset_backtest_baseline.json',
                       help='Baseline results JSON file')
    parser.add_argument('--optimized', default='results/multiasset_backtest.json',
                       help='Optimized results JSON file')
    parser.add_argument('--output', default='results/optimization_comparison.txt',
                       help='Output file for comparison report')
    
    args = parser.parse_args()
    
    # Check files exist
    if not os.path.exists(args.baseline):
        print(f"Error: Baseline file not found: {args.baseline}")
        sys.exit(1)
    
    if not os.path.exists(args.optimized):
        print(f"Error: Optimized file not found: {args.optimized}")
        sys.exit(1)
    
    compare_backtests(args.baseline, args.optimized, args.output)


if __name__ == "__main__":
    main()
