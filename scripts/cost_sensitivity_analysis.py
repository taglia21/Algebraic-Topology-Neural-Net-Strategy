"""Cost Sensitivity Analysis Script.

Analyzes how different transaction cost scenarios affect backtest performance.

Scenarios tested:
1. Low cost: $0.005/share, 3 bps spread, 2 bps slippage
2. Baseline: $1/trade, 5 bps spread, 3 bps slippage
3. High cost: $5/trade, 10 bps spread, 5 bps slippage
4. Extreme: $10/trade, 20 bps spread, 10 bps slippage
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transaction_costs import CostModel, CostScenario, COST_SCENARIOS


def run_cost_sensitivity_analysis(
    base_results_path: str = 'results/multiasset_backtest.json',
    output_path: str = 'results/cost_sensitivity_report.json'
) -> Dict[str, Any]:
    """
    Run cost sensitivity analysis on existing backtest results.
    
    This script takes the gross performance metrics and applies different
    cost scenarios to estimate how costs affect net returns.
    
    Args:
        base_results_path: Path to baseline backtest results
        output_path: Path to save sensitivity analysis results
        
    Returns:
        Dict with cost sensitivity analysis results
    """
    print("\n" + "=" * 70)
    print("TRANSACTION COST SENSITIVITY ANALYSIS")
    print("=" * 70)
    
    # Load base results
    results_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.dirname(results_dir)  # Go up one level
    
    full_base_path = os.path.join(results_dir, base_results_path)
    
    if not os.path.exists(full_base_path):
        print(f"  ⚠️  Base results not found at {full_base_path}")
        print("  Please run main_multiasset.py first to generate baseline results.")
        return {}
    
    with open(full_base_path, 'r') as f:
        base_results = json.load(f)
    
    per_asset = base_results.get('per_asset', {})
    portfolio_ew = base_results.get('portfolio_equal_weight', {})
    
    print(f"\n  Loaded baseline results from: {full_base_path}")
    print(f"  Assets: {list(per_asset.keys())}")
    print(f"  Portfolio gross return: {portfolio_ew.get('total_return', 0)*100:.2f}%")
    
    # Collect trade data for cost calculations
    trade_data = []
    
    for ticker, metrics in per_asset.items():
        num_trades = metrics.get('num_trades', 0)
        notional_traded = metrics.get('total_notional_traded', 0)
        
        if num_trades > 0:
            avg_trade_size = notional_traded / (num_trades * 2)  # Divide by 2 for entry/exit
            trade_data.append({
                'ticker': ticker,
                'num_trades': num_trades,
                'total_notional': notional_traded,
                'avg_trade_value': avg_trade_size,
                'avg_shares': avg_trade_size / 100  # Rough estimate
            })
    
    if not trade_data:
        print("  ⚠️  No trades found in baseline results.")
        return {}
    
    # Run analysis for each cost scenario
    scenarios = ['low_cost', 'baseline', 'high_cost', 'extreme']
    scenario_results = {}
    
    print("\n  Running cost scenarios...")
    print("-" * 70)
    print(f"  {'Scenario':<12} {'Total Costs':>12} {'Cost/Trade':>12} {'Net Return':>12} {'Degradation':>12}")
    print("-" * 70)
    
    gross_return = portfolio_ew.get('total_return', 0)
    initial_capital = portfolio_ew.get('initial_cash', 100000)
    
    for scenario_name in scenarios:
        scenario = COST_SCENARIOS[scenario_name]
        cost_model = CostModel(
            commission_per_trade=scenario.commission_per_trade,
            commission_per_share=scenario.commission_per_share,
            min_commission=scenario.min_commission,
            bid_ask_spread_bps=scenario.bid_ask_spread_bps,
            slippage_bps=scenario.slippage_bps
        )
        
        # Calculate total costs across all trades
        total_costs = 0
        total_trades = 0
        
        for trade in trade_data:
            # Estimate entry and exit costs
            shares = max(1, int(trade['avg_shares']))
            price = trade['avg_trade_value'] / shares if shares > 0 else 100
            atr = price * 0.02  # Rough 2% ATR estimate
            
            for _ in range(trade['num_trades']):
                # Entry cost
                entry_cost, _ = cost_model.calculate_total_cost(shares, price, atr, 'entry')
                # Exit cost
                exit_cost, _ = cost_model.calculate_total_cost(shares, price, atr, 'exit')
                
                total_costs += entry_cost + exit_cost
                total_trades += 1
        
        # Calculate net returns
        gross_pnl = initial_capital * gross_return
        net_pnl = gross_pnl - total_costs
        net_return = net_pnl / initial_capital
        
        # Calculate degradation
        if gross_return != 0:
            degradation_pct = (1 - net_return / gross_return) * 100 if gross_return > 0 else 0
        else:
            degradation_pct = 0
        
        cost_per_trade = total_costs / total_trades if total_trades > 0 else 0
        
        scenario_results[scenario_name] = {
            'scenario_params': {
                'commission_per_trade': scenario.commission_per_trade,
                'commission_per_share': scenario.commission_per_share,
                'bid_ask_spread_bps': scenario.bid_ask_spread_bps,
                'slippage_bps': scenario.slippage_bps,
            },
            'total_costs': round(total_costs, 2),
            'cost_per_trade': round(cost_per_trade, 2),
            'total_trades': total_trades,
            'gross_return': round(gross_return, 4),
            'net_return': round(net_return, 4),
            'degradation_pct': round(degradation_pct, 2),
            'costs_as_pct_of_gross_pnl': round(total_costs / abs(gross_pnl) * 100, 2) if gross_pnl != 0 else 0
        }
        
        print(f"  {scenario_name:<12} ${total_costs:>10.2f} ${cost_per_trade:>10.2f} "
              f"{net_return*100:>11.2f}% {degradation_pct:>11.1f}%")
    
    print("-" * 70)
    
    # Summary analysis
    print("\n  Analysis Summary:")
    print("  " + "=" * 50)
    
    baseline_net = scenario_results['baseline']['net_return']
    extreme_net = scenario_results['extreme']['net_return']
    
    cost_impact = baseline_net - extreme_net
    print(f"  Baseline to Extreme cost impact: {cost_impact*100:.2f}%")
    
    # Breakeven analysis
    print("\n  Cost Breakeven Analysis:")
    for scenario_name in scenarios:
        sr = scenario_results[scenario_name]
        if sr['net_return'] <= 0 and gross_return > 0:
            print(f"    {scenario_name}: Strategy becomes UNPROFITABLE (net return = {sr['net_return']*100:.2f}%)")
        else:
            remaining_pct = (sr['net_return'] / gross_return * 100) if gross_return > 0 else 0
            print(f"    {scenario_name}: Retains {remaining_pct:.1f}% of gross return")
    
    # Build full report
    report = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'base_results_path': base_results_path,
            'initial_capital': initial_capital,
            'gross_return': gross_return,
            'total_trades_analyzed': sum(t['num_trades'] for t in trade_data),
        },
        'scenarios': scenario_results,
        'summary': {
            'baseline_net_return': scenario_results['baseline']['net_return'],
            'baseline_total_costs': scenario_results['baseline']['total_costs'],
            'extreme_net_return': scenario_results['extreme']['net_return'],
            'extreme_total_costs': scenario_results['extreme']['total_costs'],
            'cost_sensitivity': cost_impact,  # How much return lost from baseline to extreme
        },
        'recommendations': []
    }
    
    # Generate recommendations
    baseline_degrade = scenario_results['baseline']['degradation_pct']
    if baseline_degrade > 50:
        report['recommendations'].append(
            "WARNING: Transaction costs consume >50% of gross returns. "
            "Consider reducing trade frequency or using limit orders."
        )
    elif baseline_degrade > 25:
        report['recommendations'].append(
            "CAUTION: Transaction costs are significant (>25% of gross returns). "
            "Monitor execution quality and consider cost reduction strategies."
        )
    else:
        report['recommendations'].append(
            "Transaction costs are within acceptable range (<25% of gross returns)."
        )
    
    # Save report
    full_output_path = os.path.join(results_dir, output_path)
    os.makedirs(os.path.dirname(full_output_path), exist_ok=True)
    
    with open(full_output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n  Report saved to: {full_output_path}")
    print("=" * 70)
    
    return report


def create_cost_comparison_table(report: Dict[str, Any]) -> pd.DataFrame:
    """Create a comparison table from the cost sensitivity report."""
    scenarios = report.get('scenarios', {})
    
    rows = []
    for name, data in scenarios.items():
        rows.append({
            'Scenario': name,
            'Commission': f"${data['scenario_params']['commission_per_trade']:.2f}/trade + ${data['scenario_params']['commission_per_share']:.3f}/sh",
            'Spread (bps)': data['scenario_params']['bid_ask_spread_bps'],
            'Slippage (bps)': data['scenario_params']['slippage_bps'],
            'Total Costs': f"${data['total_costs']:.2f}",
            'Cost/Trade': f"${data['cost_per_trade']:.2f}",
            'Net Return': f"{data['net_return']*100:.2f}%",
            'Degradation': f"{data['degradation_pct']:.1f}%"
        })
    
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run cost sensitivity analysis')
    parser.add_argument('--base-results', type=str, default='results/multiasset_backtest.json',
                       help='Path to baseline backtest results')
    parser.add_argument('--output', type=str, default='results/cost_sensitivity_report.json',
                       help='Path to save analysis report')
    
    args = parser.parse_args()
    
    report = run_cost_sensitivity_analysis(
        base_results_path=args.base_results,
        output_path=args.output
    )
    
    if report:
        # Print comparison table
        print("\n  Cost Comparison Table:")
        df = create_cost_comparison_table(report)
        print(df.to_string(index=False))
