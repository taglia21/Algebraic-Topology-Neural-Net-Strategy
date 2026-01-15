#!/usr/bin/env python3
"""Full 3-Year Analysis Report Generator.

Generates comprehensive year-by-year analysis of portfolio performance
across 2022-2024 (full 3 years of data available at the time).
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, '/workspaces/Algebraic-Topology-Neural-Net-Strategy')

RESULTS_DIR = '/workspaces/Algebraic-Topology-Neural-Net-Strategy/results'


def load_backtest_results() -> Dict:
    """Load the most recent backtest results."""
    path = os.path.join(RESULTS_DIR, 'multiasset_backtest.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}


def download_benchmark_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download benchmark data for comparison."""
    import yfinance as yf
    t = yf.Ticker(ticker)
    df = t.history(start=start, end=end, auto_adjust=True)
    if not df.empty:
        df.columns = [c.lower() for c in df.columns]
    return df


def calculate_yearly_metrics(df: pd.DataFrame) -> Dict[str, Dict]:
    """Calculate metrics for each year in the dataframe."""
    close = df['close'] if 'close' in df.columns else df['Close']
    df = df.copy()
    df['year'] = pd.to_datetime(df.index).year
    df['return'] = close.pct_change()
    
    yearly_metrics = {}
    
    for year in df['year'].unique():
        year_data = df[df['year'] == year]
        returns = year_data['return'].dropna()
        
        if len(returns) < 10:
            continue
        
        # Total return
        total_return = (1 + returns).prod() - 1
        
        # Annualized metrics
        ann_return = total_return  # Already yearly
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        
        # Max drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        max_dd = abs(drawdown.min())
        
        yearly_metrics[year] = {
            'total_return_pct': round(total_return * 100, 2),
            'annualized_vol_pct': round(ann_vol * 100, 2),
            'sharpe_ratio': round(sharpe, 2),
            'max_drawdown_pct': round(max_dd * 100, 2),
            'trading_days': len(returns)
        }
    
    return yearly_metrics


def generate_full_analysis_report() -> str:
    """Generate comprehensive 3-year analysis report."""
    
    print("=" * 60)
    print("FULL 3-YEAR ANALYSIS REPORT GENERATOR")
    print("=" * 60)
    
    report = []
    report.append("=" * 80)
    report.append("FULL 3-YEAR BACKTEST ANALYSIS REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)
    
    # Load current backtest results
    results = load_backtest_results()
    
    # Executive Summary
    report.append("\n" + "=" * 80)
    report.append("EXECUTIVE SUMMARY")
    report.append("=" * 80)
    
    metadata = results.get('metadata', {})
    portfolio_eq = results.get('portfolio_equal_weight', {})
    portfolio_wgt = results.get('portfolio_performance_weighted', {})
    per_asset = results.get('per_asset', {})
    
    report.append(f"\nTest Period: {metadata.get('test_period', 'N/A')}")
    report.append(f"Train Period: {metadata.get('train_period', 'N/A')}")
    report.append(f"Data Provider: {metadata.get('data_provider', 'N/A')}")
    report.append(f"TDA Feature Mode: {metadata.get('tda_feature_mode', 'N/A')}")
    
    report.append("\nPortfolio Performance (Risk-Weighted):")
    report.append(f"  Sharpe Ratio (net): {portfolio_wgt.get('sharpe_ratio_net', 'N/A')}")
    report.append(f"  Total Return (net): {portfolio_wgt.get('total_return_net', 0)*100:.2f}%")
    report.append(f"  Max Drawdown: {portfolio_wgt.get('max_drawdown', 0)*100:.2f}%")
    report.append(f"  Weights: {results.get('weights', {})}")
    
    # Download benchmark data for comparison
    print("\n[1/4] Downloading benchmark data...")
    spy_data = download_benchmark_data('SPY', '2022-01-01', '2025-12-31')
    
    # Year-by-Year Benchmark Analysis
    report.append("\n" + "=" * 80)
    report.append("YEAR-BY-YEAR BENCHMARK ANALYSIS (SPY Buy-and-Hold)")
    report.append("=" * 80)
    
    if not spy_data.empty:
        yearly_benchmark = calculate_yearly_metrics(spy_data)
        
        report.append(f"\n{'Year':<8} {'Return%':>10} {'Volatility%':>12} {'Sharpe':>10} {'MaxDD%':>10}")
        report.append("-" * 55)
        
        for year, metrics in sorted(yearly_benchmark.items()):
            report.append(
                f"{year:<8} {metrics['total_return_pct']:>10.2f} "
                f"{metrics['annualized_vol_pct']:>12.2f} "
                f"{metrics['sharpe_ratio']:>10.2f} "
                f"{metrics['max_drawdown_pct']:>10.2f}"
            )
        
        # Summary
        all_returns = [m['total_return_pct'] for m in yearly_benchmark.values()]
        all_sharpes = [m['sharpe_ratio'] for m in yearly_benchmark.values()]
        avg_return = np.mean(all_returns) if all_returns else 0
        avg_sharpe = np.mean(all_sharpes) if all_sharpes else 0
        
        report.append("-" * 55)
        report.append(f"{'AVERAGE':<8} {avg_return:>10.2f} {'':>12} {avg_sharpe:>10.2f}")
    
    # Per-Asset Analysis
    report.append("\n" + "=" * 80)
    report.append("PER-ASSET STRATEGY PERFORMANCE")
    report.append("=" * 80)
    
    report.append(f"\n{'Asset':<8} {'Sharpe':>10} {'Sh_net':>10} {'Return%':>10} {'Trades':>8} {'WinRate':>10}")
    report.append("-" * 60)
    
    for ticker, data in per_asset.items():
        sharpe = data.get('sharpe_ratio', 0)
        sharpe_net = data.get('sharpe_ratio_net', 0)
        ret = data.get('total_return', 0) * 100
        trades = data.get('num_trades', 0)
        win_rate = data.get('win_rate', 0) * 100
        
        report.append(
            f"{ticker:<8} {sharpe:>10.2f} {sharpe_net:>10.2f} "
            f"{ret:>10.2f} {trades:>8} {win_rate:>9.1f}%"
        )
    
    # QQQ Analysis Summary
    report.append("\n" + "=" * 80)
    report.append("QQQ ANALYSIS SUMMARY")
    report.append("=" * 80)
    
    qqq_data = per_asset.get('QQQ', {})
    report.append(f"\nQQQ Strategy Performance:")
    report.append(f"  Sharpe (net): {qqq_data.get('sharpe_ratio_net', 'N/A')}")
    report.append(f"  Return (net): {qqq_data.get('total_return_net', 0)*100:.2f}%")
    report.append(f"  Trades: {qqq_data.get('num_trades', 0)}")
    report.append(f"  Win Rate: {qqq_data.get('win_rate', 0)*100:.1f}%")
    
    report.append("\nKey Findings from QQQ Diagnostic Report:")
    report.append("  • QQQ performs well only in FAVORABLE conditions (Sharpe 4.0)")
    report.append("  • QQQ fails in NEUTRAL conditions (Sharpe -1.05)")
    report.append("  • QQQ has 0.96 correlation with SPY (high overlap)")
    report.append("  • QQQ has 0.97 correlation with XLK (tech sector overlap)")
    report.append("  • QQQ volatility 32% higher than SPY")
    
    report.append("\nMitigation Applied:")
    report.append("  • Risk-weighted allocation reduces QQQ exposure automatically")
    report.append("  • Current weights exclude underperformers (QQQ gets 0% when negative)")
    report.append("  • Regime detection filters reduce trades in unfavorable conditions")
    
    # Portfolio Composition Analysis
    report.append("\n" + "=" * 80)
    report.append("OPTIMAL PORTFOLIO COMPOSITION")
    report.append("=" * 80)
    
    weights = results.get('weights', {})
    report.append(f"\nRisk-Weighted Allocation: {weights}")
    report.append(f"Risk Scale: {results.get('risk_scale', 'N/A')}")
    report.append(f"Cash Weight: {results.get('cash_weight', 0)*100:.0f}%")
    
    report.append("\nPortfolio Strategy:")
    report.append("  1. Keep 5-asset universe for neural network training diversity")
    report.append("  2. Let risk-weighting naturally reduce allocation to underperformers")
    report.append("  3. Regime detection filters reduce trades in unfavorable conditions")
    report.append("  4. Signal quality filters (RSI, volatility) reduce low-quality trades")
    
    # Success Criteria Check
    report.append("\n" + "=" * 80)
    report.append("SUCCESS CRITERIA ASSESSMENT")
    report.append("=" * 80)
    
    wgt_sharpe = portfolio_wgt.get('sharpe_ratio_net', 0)
    max_dd = portfolio_wgt.get('max_drawdown', 0) * 100
    total_trades = portfolio_eq.get('total_trades', 0)
    
    report.append("\nMinimum Requirements:")
    report.append(f"  [{'✓' if wgt_sharpe > 1.3 else '✗'}] Portfolio Sharpe > 1.3: {wgt_sharpe:.2f}")
    report.append(f"  [{'✓' if max_dd < 8 else '✗'}] Max Drawdown < 8%: {max_dd:.2f}%")
    
    positive_assets = sum(1 for d in per_asset.values() if d.get('sharpe_ratio_net', 0) > 0)
    report.append(f"  [{'✓' if positive_assets >= 2 else '✗'}] Assets with Sharpe > 0: {positive_assets}/5")
    
    report.append("\nStretch Goals:")
    report.append(f"  [{'✓' if wgt_sharpe > 1.5 else '✗'}] Portfolio Sharpe > 1.5: {wgt_sharpe:.2f}")
    report.append(f"  [{'✓' if max_dd < 5 else '✗'}] Max Drawdown < 5%: {max_dd:.2f}%")
    
    # Risk Management Metrics
    report.append("\n" + "=" * 80)
    report.append("RISK MANAGEMENT METRICS")
    report.append("=" * 80)
    
    report.append("\nPosition Sizing:")
    report.append("  • Half-Kelly: kelly_fraction=0.50")
    report.append("  • Max position: 15% of account per asset")
    report.append("  • Volatility-adaptive scaling: 0.5x to 1.5x")
    
    report.append("\nSignal Filters:")
    report.append("  • RSI filter: period=14, oversold=45, overbought=55")
    report.append("  • Volatility filter: threshold=35%")
    report.append("  • Regime detection: skip UNFAVORABLE conditions")
    
    report.append("\nTransaction Costs:")
    report.append("  • Cost model: 5 bp/side (0.001 per trade)")
    report.append(f"  • Total trades: {total_trades}")
    report.append(f"  • Turnover: {portfolio_eq.get('turnover', 0):.2f}x")
    
    # Recommendations
    report.append("\n" + "=" * 80)
    report.append("RECOMMENDATIONS & NEXT STEPS")
    report.append("=" * 80)
    
    report.append("\n1. Keep Current Configuration:")
    report.append("   • 5-asset portfolio with risk-weighting provides optimal results")
    report.append("   • Regime detection + signal filters reduce low-quality trades")
    report.append("   • Risk-weighting automatically handles underperforming assets")
    
    report.append("\n2. QQQ Strategy:")
    report.append("   • DECISION: Keep QQQ in training data but let risk-weighting reduce allocation")
    report.append("   • The neural network benefits from diverse training data")
    report.append("   • When QQQ underperforms, it gets 0% allocation automatically")
    
    report.append("\n3. Production Deployment Considerations:")
    report.append("   • Add real-time data feed integration")
    report.append("   • Implement paper trading validation before live deployment")
    report.append("   • Set up monitoring dashboards for performance tracking")
    report.append("   • Define drawdown circuit breakers (e.g., pause at -5%)")
    
    report.append("\n4. Future Enhancements:")
    report.append("   • Consider adding sector rotation signals")
    report.append("   • Test with intraday data (hourly timeframe)")
    report.append("   • Explore options overlay for additional income")
    
    report.append("\n" + "=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)
    
    return "\n".join(report)


def main():
    """Generate and save the analysis report."""
    
    report = generate_full_analysis_report()
    
    # Save report
    report_path = os.path.join(RESULTS_DIR, 'full_3year_analysis.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\n[4/4] Report saved to: {report_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("REPORT GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nFull report available at: {report_path}")
    
    return report


if __name__ == '__main__':
    main()
