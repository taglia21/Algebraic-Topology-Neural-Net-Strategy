#!/usr/bin/env python3
"""
V2 Enhancement Report Generator

Generates comprehensive comparison report from backtest results:
- Executive summary table (V1.3 vs V2.0)
- Ablation analysis with enhancement contributions
- Equity curves visualization
- Drawdown analysis
- Trade analysis
- Recommendations

Usage:
    python scripts/generate_v2_report.py

Output:
    results/V2_ENHANCEMENT_REPORT.md
    results/equity_curves.png
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available - equity curves will not be generated")


def load_results(results_path: str = 'results/backtest_results_latest.json') -> Dict[str, Any]:
    """
    Load backtest results from JSON file.
    
    Args:
        results_path: Path to JSON results file
    
    Returns:
        Dictionary with backtest results
    """
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results not found: {results_path}. Run run_v2_backtest_ablation.py first.")
    
    with open(results_path, 'r') as f:
        return json.load(f)


def generate_summary_table(results: Dict[str, Any]) -> str:
    """
    Generate markdown summary table comparing V1.3 and V2.0.
    
    Returns:
        Markdown formatted table string
    """
    strategies = results.get('results', {})
    
    if 'V1.3_baseline' not in strategies or 'V2.0_full' not in strategies:
        return "⚠️ Incomplete results - missing baseline or V2.0 data\n"
    
    v13 = strategies['V1.3_baseline']
    v2 = strategies['V2.0_full']
    
    # Calculate improvements
    sharpe_delta = v2['sharpe_ratio'] - v13['sharpe_ratio']
    sharpe_pct = (sharpe_delta / abs(v13['sharpe_ratio']) * 100) if v13['sharpe_ratio'] != 0 else 0
    
    cagr_delta = v2['cagr'] - v13['cagr']
    
    # Max DD improvement (less negative is better)
    dd_delta = v2['max_drawdown'] - v13['max_drawdown']
    dd_pct = (dd_delta / abs(v13['max_drawdown']) * 100) if v13['max_drawdown'] != 0 else 0
    
    table = """
| Metric | V1.3 Baseline | V2.0 Full | Change | Δ% |
|--------|---------------|-----------|--------|-----|
| **Sharpe Ratio** | {v13_sharpe:.3f} | {v2_sharpe:.3f} | {sharpe_delta:+.3f} | {sharpe_pct:+.1f}% |
| **CAGR** | {v13_cagr:.2%} | {v2_cagr:.2%} | {cagr_delta:+.2%} | - |
| **Max Drawdown** | {v13_dd:.2%} | {v2_dd:.2%} | {dd_delta:+.2%} | {dd_pct:+.1f}% |
| **Calmar Ratio** | {v13_calmar:.2f} | {v2_calmar:.2f} | {calmar_delta:+.2f} | - |
| **Win Rate** | {v13_win:.1%} | {v2_win:.1%} | {win_delta:+.1%} | - |
| **Volatility** | {v13_vol:.1%} | {v2_vol:.1%} | {vol_delta:+.1%} | - |
| **Total Return** | {v13_ret:.1%} | {v2_ret:.1%} | {ret_delta:+.1%} | - |
| **Trades** | {v13_trades} | {v2_trades} | {trades_delta:+d} | - |
""".format(
        v13_sharpe=v13['sharpe_ratio'],
        v2_sharpe=v2['sharpe_ratio'],
        sharpe_delta=sharpe_delta,
        sharpe_pct=sharpe_pct,
        v13_cagr=v13['cagr'],
        v2_cagr=v2['cagr'],
        cagr_delta=cagr_delta,
        v13_dd=v13['max_drawdown'],
        v2_dd=v2['max_drawdown'],
        dd_delta=dd_delta,
        dd_pct=dd_pct,
        v13_calmar=v13['calmar_ratio'],
        v2_calmar=v2['calmar_ratio'],
        calmar_delta=v2['calmar_ratio'] - v13['calmar_ratio'],
        v13_win=v13['win_rate'],
        v2_win=v2['win_rate'],
        win_delta=v2['win_rate'] - v13['win_rate'],
        v13_vol=v13['volatility'],
        v2_vol=v2['volatility'],
        vol_delta=v2['volatility'] - v13['volatility'],
        v13_ret=v13['total_return'],
        v2_ret=v2['total_return'],
        ret_delta=v2['total_return'] - v13['total_return'],
        v13_trades=v13['n_trades'],
        v2_trades=v2['n_trades'],
        trades_delta=v2['n_trades'] - v13['n_trades'],
    )
    
    return table


def generate_ablation_table(results: Dict[str, Any]) -> str:
    """
    Generate markdown table showing ablation study results.
    
    Returns:
        Markdown formatted table string
    """
    contributions = results.get('ablation_contributions', {})
    strategies = results.get('results', {})
    
    if not contributions:
        return "⚠️ No ablation data available\n"
    
    v2_sharpe = strategies.get('V2.0_full', {}).get('sharpe_ratio', 0)
    
    # Sort by contribution (descending)
    sorted_contribs = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
    
    table = """
| Enhancement | Sharpe Contribution | % of V2 Sharpe | Impact |
|-------------|---------------------|----------------|--------|
"""
    
    for component, contrib in sorted_contribs:
        pct = (contrib / v2_sharpe * 100) if v2_sharpe != 0 else 0
        
        if contrib > 0.05:
            impact = "🟢 Strong positive"
        elif contrib > 0.01:
            impact = "🟡 Moderate positive"
        elif contrib > -0.01:
            impact = "⚪ Neutral"
        elif contrib > -0.05:
            impact = "🟠 Moderate negative"
        else:
            impact = "🔴 Strong negative"
        
        table += f"| {component.replace('_', ' ').title()} | {contrib:+.4f} | {pct:+.1f}% | {impact} |\n"
    
    return table


def generate_ablation_details(results: Dict[str, Any]) -> str:
    """
    Generate detailed ablation analysis for each component.
    
    Returns:
        Markdown formatted analysis string
    """
    strategies = results.get('results', {})
    contributions = results.get('ablation_contributions', {})
    
    if 'V2.0_full' not in strategies:
        return ""
    
    v2 = strategies['V2.0_full']
    
    analysis = ""
    
    component_info = {
        'transformer': {
            'name': 'Transformer Predictor',
            'ablation_key': 'V2_no_transformer',
            'description': 'Multi-head self-attention for capturing long-range dependencies in price patterns.',
        },
        'sac': {
            'name': 'SAC Position Sizing',
            'ablation_key': 'V2_no_sac',
            'description': 'Soft Actor-Critic for continuous, entropy-regularized position sizing.',
        },
        'tda': {
            'name': 'Persistent Laplacian TDA',
            'ablation_key': 'V2_no_tda',
            'description': 'Topological data analysis for detecting volatility regimes and market structure.',
        },
        'ensemble_regime': {
            'name': 'Ensemble Regime Detection',
            'ablation_key': 'V2_no_ensemble',
            'description': 'Multi-model consensus (HMM + GMM + Clustering) for robust regime classification.',
        },
        'order_flow': {
            'name': 'Order Flow Analysis',
            'ablation_key': 'V2_no_orderflow',
            'description': 'Microstructure signals from volume, bid-ask spread, and trade classification.',
        },
        'enhanced_momentum': {
            'name': 'Enhanced Momentum',
            'ablation_key': 'V2_no_enhanced_mom',
            'description': 'Multi-scale momentum with mean-reversion overlay for overbought/oversold detection.',
        },
        'risk_parity': {
            'name': 'Risk Parity Weighting',
            'ablation_key': 'V2_no_risk_parity',
            'description': 'Inverse volatility weighting for balanced risk contribution across assets.',
        },
    }
    
    for comp_key, info in component_info.items():
        ablation_key = info['ablation_key']
        contrib = contributions.get(comp_key, 0)
        
        if ablation_key in strategies:
            abl = strategies[ablation_key]
            
            analysis += f"""
### {info['name']}

**Description:** {info['description']}

**Contribution:** {'+' if contrib >= 0 else ''}{contrib:.4f} Sharpe points

| Metric | V2 Full | Without {info['name']} | Impact |
|--------|---------|------------------------|--------|
| Sharpe | {v2['sharpe_ratio']:.3f} | {abl['sharpe_ratio']:.3f} | {v2['sharpe_ratio'] - abl['sharpe_ratio']:+.3f} |
| CAGR | {v2['cagr']:.2%} | {abl['cagr']:.2%} | {v2['cagr'] - abl['cagr']:+.2%} |
| Max DD | {v2['max_drawdown']:.2%} | {abl['max_drawdown']:.2%} | {v2['max_drawdown'] - abl['max_drawdown']:+.2%} |

"""
    
    return analysis


def plot_equity_curves(results: Dict[str, Any], output_path: str = 'results/equity_curves.png') -> bool:
    """
    Generate equity curve comparison chart.
    
    Args:
        results: Backtest results dictionary
        output_path: Path to save PNG file
    
    Returns:
        True if chart was generated, False otherwise
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available - skipping equity curve generation")
        return False
    
    strategies = results.get('results', {})
    
    # Since we don't have actual equity curves in the results,
    # we'll simulate them based on the metrics
    
    np.random.seed(42)
    n_days = 252  # ~1 year of trading days
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Equity curves
    ax1 = axes[0]
    
    strategies_to_plot = ['V1.3_baseline', 'V2.0_full']
    colors = {'V1.3_baseline': '#1f77b4', 'V2.0_full': '#ff7f0e'}
    
    for name in strategies_to_plot:
        if name not in strategies:
            continue
        
        s = strategies[name]
        
        # Simulate daily returns based on annual metrics
        daily_return = s['cagr'] / 252
        daily_vol = s['volatility'] / np.sqrt(252)
        
        returns = np.random.normal(daily_return, daily_vol, n_days)
        equity = 100000 * np.cumprod(1 + returns)
        
        label = f"{name} (Sharpe: {s['sharpe_ratio']:.2f})"
        ax1.plot(equity, label=label, color=colors.get(name, None), linewidth=2)
    
    ax1.set_title('Simulated Equity Curves (V1.3 vs V2.0)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Trading Days')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=100000, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    
    # Ablation comparison (bar chart)
    ax2 = axes[1]
    
    contributions = results.get('ablation_contributions', {})
    if contributions:
        components = list(contributions.keys())
        values = list(contributions.values())
        
        colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in values]
        
        bars = ax2.barh(components, values, color=colors)
        ax2.set_xlabel('Sharpe Contribution')
        ax2.set_title('Enhancement Contributions (Ablation Study)', fontsize=14, fontweight='bold')
        ax2.axvline(x=0, color='black', linewidth=0.5)
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax2.text(val + 0.002 if val >= 0 else val - 0.002,
                     bar.get_y() + bar.get_height()/2,
                     f'{val:+.3f}',
                     ha='left' if val >= 0 else 'right',
                     va='center',
                     fontsize=10)
    
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Equity curves saved to {output_path}")
    return True


def generate_recommendations(results: Dict[str, Any]) -> str:
    """
    Generate actionable recommendations based on results.
    
    Returns:
        Markdown formatted recommendations string
    """
    strategies = results.get('results', {})
    contributions = results.get('ablation_contributions', {})
    
    if 'V1.3_baseline' not in strategies or 'V2.0_full' not in strategies:
        return "⚠️ Insufficient data for recommendations\n"
    
    v13 = strategies['V1.3_baseline']
    v2 = strategies['V2.0_full']
    
    recommendations = []
    
    # Overall performance
    sharpe_delta = v2['sharpe_ratio'] - v13['sharpe_ratio']
    
    if sharpe_delta > 0.1:
        recommendations.append("✅ **Deploy V2.0**: Significant improvement over baseline. Proceed with production deployment.")
    elif sharpe_delta > 0:
        recommendations.append("✅ **Deploy V2.0 with monitoring**: Marginal improvement. Deploy with enhanced monitoring.")
    else:
        recommendations.append("⚠️ **Review V2.0**: No improvement over baseline. Consider selective enhancements.")
    
    # Component-specific recommendations
    positive_components = [c for c, v in contributions.items() if v > 0.02]
    negative_components = [c for c, v in contributions.items() if v < -0.02]
    
    if positive_components:
        recommendations.append(f"🟢 **Keep enabled**: {', '.join(positive_components)} - these contribute positively to Sharpe.")
    
    if negative_components:
        recommendations.append(f"🔴 **Consider disabling**: {', '.join(negative_components)} - these reduce Sharpe ratio.")
    
    # Risk metrics
    if v2['max_drawdown'] < v13['max_drawdown']:  # Less negative = better
        recommendations.append(f"📉 **Drawdown improved**: V2.0 reduces max drawdown from {v13['max_drawdown']:.1%} to {v2['max_drawdown']:.1%}")
    else:
        recommendations.append(f"⚠️ **Drawdown increased**: V2.0 increases max drawdown to {v2['max_drawdown']:.1%}. Review risk controls.")
    
    # Volatility
    vol_delta = v2['volatility'] - v13['volatility']
    if vol_delta < -0.02:
        recommendations.append("📊 **Lower volatility**: V2.0 achieves lower portfolio volatility - good for risk-adjusted returns.")
    elif vol_delta > 0.02:
        recommendations.append("⚠️ **Higher volatility**: V2.0 has higher volatility. Consider reducing position sizes.")
    
    # Target check
    target_sharpe = 1.50
    target_dd = -0.015
    target_cagr = 0.18
    
    targets_met = []
    targets_missed = []
    
    if v2['sharpe_ratio'] >= target_sharpe:
        targets_met.append(f"Sharpe ≥ {target_sharpe} ✓")
    else:
        targets_missed.append(f"Sharpe: {v2['sharpe_ratio']:.2f} < {target_sharpe}")
    
    if v2['max_drawdown'] >= target_dd:  # Less negative = better
        targets_met.append(f"Max DD ≤ {target_dd:.1%} ✓")
    else:
        targets_missed.append(f"Max DD: {v2['max_drawdown']:.1%} < {target_dd:.1%}")
    
    if v2['cagr'] >= target_cagr:
        targets_met.append(f"CAGR ≥ {target_cagr:.0%} ✓")
    else:
        targets_missed.append(f"CAGR: {v2['cagr']:.1%} < {target_cagr:.0%}")
    
    if targets_met:
        recommendations.append(f"🎯 **Targets met**: {', '.join(targets_met)}")
    if targets_missed:
        recommendations.append(f"❌ **Targets missed**: {', '.join(targets_missed)}")
    
    return "\n".join([f"- {r}" for r in recommendations])


def create_markdown_report(results: Dict[str, Any], output_path: str = 'results/V2_ENHANCEMENT_REPORT.md') -> str:
    """
    Create comprehensive markdown report.
    
    Args:
        results: Backtest results dictionary
        output_path: Path to save markdown file
    
    Returns:
        Path to saved report
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    report = f"""# V2.0 Enhancement Report

**Generated:** {timestamp}

---

## Executive Summary

This report compares the V2.0 trading system (with 7 enhancements) against the V1.3 baseline,
and provides ablation analysis to identify the contribution of each enhancement.

{generate_summary_table(results)}

---

## Ablation Study

The following table shows the Sharpe ratio contribution of each enhancement.
A positive value means removing the enhancement reduces Sharpe (the enhancement helps).
A negative value means removing the enhancement improves Sharpe (the enhancement hurts).

{generate_ablation_table(results)}

---

## Detailed Enhancement Analysis

{generate_ablation_details(results)}

---

## Equity Curves

![Equity Curves](equity_curves.png)

*Note: Equity curves are simulated based on reported metrics for visualization purposes.*

---

## Recommendations

{generate_recommendations(results)}

---

## Configuration

```json
{json.dumps(results.get('config', {}), indent=2, default=str)}
```

---

## Raw Results

<details>
<summary>Click to expand full results JSON</summary>

```json
{json.dumps(results.get('results', {}), indent=2, default=str)}
```

</details>

---

*Report generated by V2 Enhancement Analysis Pipeline*
"""
    
    # Save report
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Report saved to {output_path}")
    return output_path


def main():
    """Main entry point."""
    print("=" * 60)
    print("V2 Enhancement Report Generator")
    print("=" * 60)
    
    # Load results
    logger.info("Loading backtest results...")
    try:
        results = load_results()
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    
    logger.info(f"Loaded {len(results.get('results', {}))} strategy results")
    
    # Generate equity curves
    logger.info("Generating equity curves...")
    chart_generated = plot_equity_curves(results)
    
    # Generate report
    logger.info("Generating markdown report...")
    report_path = create_markdown_report(results)
    
    print("\n" + "=" * 60)
    print("REPORT GENERATED")
    print("=" * 60)
    print(f"\nReport: {report_path}")
    if chart_generated:
        print(f"Chart: results/equity_curves.png")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
