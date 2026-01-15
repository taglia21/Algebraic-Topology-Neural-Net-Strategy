"""Comprehensive 5-Year Performance Analysis & SPY Benchmark Comparison.

Generates deployment readiness report for TDA+NN Trading Strategy.
Answers critical questions:
1. Does strategy beat SPY on absolute returns?
2. Does strategy beat SPY on risk-adjusted basis?
3. Can it work with $2K capital?
4. What are realistic expectations?
5. Should user proceed with deployment?
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_provider import get_ohlcv_hybrid

RESULTS_DIR = '/workspaces/Algebraic-Topology-Neural-Net-Strategy/results'

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_json(filepath: str) -> Dict:
    """Load JSON file safely."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        return {'error': str(e)}


def load_walk_forward_results() -> Dict:
    """Load and parse walk-forward validation results."""
    wf = load_json(f'{RESULTS_DIR}/multiasset_walkforward_report.json')
    if 'error' in wf:
        return wf
    return wf


def load_baseline_results() -> Dict:
    """Load baseline backtest results."""
    return load_json(f'{RESULTS_DIR}/multiasset_backtest.json')


# ============================================================================
# SPY BENCHMARK CALCULATION
# ============================================================================

def calculate_spy_benchmark(start_date: str = '2020-01-01', 
                            end_date: str = '2025-01-15') -> Dict:
    """
    Calculate SPY buy-and-hold benchmark performance.
    
    Returns comprehensive metrics for comparison.
    """
    print("Fetching SPY data for benchmark calculation...")
    
    df = get_ohlcv_hybrid('SPY', start_date, end_date)
    
    if df.empty:
        return {'error': 'Failed to fetch SPY data'}
    
    # Calculate daily returns
    df['returns'] = df['close'].pct_change()
    df = df.dropna()
    
    # Basic metrics
    total_return = (df['close'].iloc[-1] / df['close'].iloc[0]) - 1
    
    # Annualized return (CAGR)
    years = (df.index[-1] - df.index[0]).days / 365.25
    cagr = (1 + total_return) ** (1 / years) - 1
    
    # Volatility
    daily_vol = df['returns'].std()
    annual_vol = daily_vol * np.sqrt(252)
    
    # Sharpe Ratio (assuming 4% risk-free rate)
    rf_rate = 0.04
    excess_return = cagr - rf_rate
    sharpe = excess_return / annual_vol if annual_vol > 0 else 0
    
    # Sortino Ratio (downside deviation)
    downside_returns = df['returns'][df['returns'] < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino = excess_return / downside_std if downside_std > 0 else 0
    
    # Maximum Drawdown
    cumulative = (1 + df['returns']).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdowns = cumulative / rolling_max - 1
    max_drawdown = drawdowns.min()
    
    # Find drawdown duration
    dd_end = drawdowns.idxmin()
    dd_start = cumulative[:dd_end].idxmax()
    dd_duration_days = (dd_end - dd_start).days
    
    # Recovery time (if recovered)
    post_dd = cumulative[dd_end:]
    recovery_idx = post_dd[post_dd >= cumulative[dd_start]].first_valid_index()
    if recovery_idx:
        recovery_days = (recovery_idx - dd_end).days
    else:
        recovery_days = None  # Still in drawdown or not recovered
    
    # Calmar Ratio
    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Year-by-year breakdown
    yearly_returns = {}
    for year in range(2020, 2026):
        year_data = df[df.index.year == year]
        if len(year_data) > 0:
            yr_return = (year_data['close'].iloc[-1] / year_data['close'].iloc[0]) - 1
            yr_vol = year_data['returns'].std() * np.sqrt(252)
            yr_sharpe = (yr_return - rf_rate/12*len(year_data)/21) / yr_vol if yr_vol > 0 else 0
            
            # Year max drawdown
            yr_cum = (1 + year_data['returns']).cumprod()
            yr_peak = yr_cum.expanding().max()
            yr_dd = (yr_cum / yr_peak - 1).min()
            
            yearly_returns[year] = {
                'return': float(yr_return),
                'volatility': float(yr_vol),
                'sharpe': float(yr_sharpe),
                'max_drawdown': float(yr_dd),
                'trading_days': len(year_data)
            }
    
    # Monthly stats
    monthly_returns = df['returns'].resample('ME').apply(lambda x: (1+x).prod()-1)
    best_month = monthly_returns.max()
    worst_month = monthly_returns.min()
    winning_months = (monthly_returns > 0).sum()
    losing_months = (monthly_returns < 0).sum()
    
    return {
        'ticker': 'SPY',
        'period': f'{start_date} to {end_date}',
        'years': round(years, 2),
        'total_bars': len(df),
        
        # Return metrics
        'total_return': float(total_return),
        'cagr': float(cagr),
        
        # Risk metrics
        'annual_volatility': float(annual_vol),
        'max_drawdown': float(max_drawdown),
        'max_drawdown_start': str(dd_start.date()) if dd_start else None,
        'max_drawdown_end': str(dd_end.date()) if dd_end else None,
        'drawdown_duration_days': dd_duration_days,
        'recovery_days': recovery_days,
        
        # Risk-adjusted metrics
        'sharpe_ratio': float(sharpe),
        'sortino_ratio': float(sortino),
        'calmar_ratio': float(calmar),
        
        # Yearly breakdown
        'yearly_returns': yearly_returns,
        
        # Monthly stats
        'best_month': float(best_month),
        'worst_month': float(worst_month),
        'winning_months': int(winning_months),
        'losing_months': int(losing_months),
        'monthly_win_rate': float(winning_months / (winning_months + losing_months))
    }


# ============================================================================
# STRATEGY PERFORMANCE EXTRACTION
# ============================================================================

def extract_strategy_performance() -> Dict:
    """
    Extract and compile strategy performance from all available data.
    """
    baseline = load_baseline_results()
    wf = load_walk_forward_results()
    
    # Use walk-forward as primary (covers full 2020-2025 period)
    if 'error' not in wf:
        agg = wf.get('aggregate', {})
        folds = wf.get('folds', [])
        
        # Calculate cumulative return across all folds
        cumulative_eq = 1.0
        cumulative_wgt = 1.0
        all_trades = 0
        all_wins = 0
        yearly_data = {}
        
        for fold in folds:
            eq = fold.get('portfolio_equal_weight', {})
            wgt = fold.get('portfolio_performance_weighted', {})
            
            eq_ret = eq.get('total_return_net', 0)
            wgt_ret = wgt.get('total_return_net', 0)
            
            cumulative_eq *= (1 + eq_ret)
            cumulative_wgt *= (1 + wgt_ret)
            
            trades = eq.get('num_trades', 0)
            win_rate = eq.get('win_rate', 0)
            all_trades += trades
            all_wins += int(trades * win_rate)
            
            # Extract year from fold name
            test_start = fold.get('test_start', '')
            if test_start:
                year = int(test_start[:4])
                half = 'H1' if 'H1' in fold.get('name', '') or test_start[5:7] in ['01','02','03','04','05','06'] else 'H2'
                key = f"{year}_{half}"
                yearly_data[key] = {
                    'period': f"{fold.get('test_start')} to {fold.get('test_end')}",
                    'eq_return': eq_ret,
                    'wgt_return': wgt_ret,
                    'eq_sharpe': eq.get('sharpe_ratio_net', 0),
                    'wgt_sharpe': wgt.get('sharpe_ratio_net', 0),
                    'max_drawdown': eq.get('max_drawdown', 0),
                    'trades': trades,
                    'win_rate': win_rate
                }
        
        total_eq_return = cumulative_eq - 1
        total_wgt_return = cumulative_wgt - 1
        
        # Annualized returns (8 folds = ~4 years)
        years_covered = len(folds) * 0.5  # Each fold is 6 months
        cagr_eq = (cumulative_eq ** (1/years_covered)) - 1 if years_covered > 0 else 0
        cagr_wgt = (cumulative_wgt ** (1/years_covered)) - 1 if years_covered > 0 else 0
        
        # Max drawdown across all folds
        max_dd = max([f.get('portfolio_equal_weight', {}).get('max_drawdown', 0) for f in folds])
        
        return {
            'source': 'walk_forward',
            'period': f"{wf.get('metadata', {}).get('config', {}).get('start')} to {wf.get('metadata', {}).get('config', {}).get('end')}",
            'n_folds': len(folds),
            'years_covered': years_covered,
            
            # Equal-weight portfolio
            'eq_total_return': total_eq_return,
            'eq_cagr': cagr_eq,
            'eq_sharpe_mean': agg.get('eq_sharpe_net_mean', 0),
            'eq_sharpe_std': agg.get('eq_sharpe_net_std', 0),
            
            # Performance-weighted portfolio
            'wgt_total_return': total_wgt_return,
            'wgt_cagr': cagr_wgt,
            'wgt_sharpe_mean': agg.get('wgt_sharpe_net_mean', 0),
            'wgt_sharpe_std': agg.get('wgt_sharpe_net_std', 0),
            
            # Risk metrics
            'max_drawdown': max_dd,
            
            # Trade stats
            'total_trades': all_trades,
            'overall_win_rate': all_wins / all_trades if all_trades > 0 else 0,
            
            # Fold consistency
            'positive_eq_folds': agg.get('n_positive_eq_sharpe_net', 0),
            'positive_wgt_folds': agg.get('n_positive_wgt_sharpe_net', 0),
            
            # Yearly breakdown
            'yearly_data': yearly_data
        }
    
    return {'error': 'No walk-forward data available'}


# ============================================================================
# MICRO CAPITAL ANALYSIS
# ============================================================================

def analyze_micro_capital(capital: float = 2000.0) -> Dict:
    """
    Analyze feasibility of trading with micro capital.
    """
    # Current approximate prices (would fetch live in production)
    prices = {
        'IWM': 225.0,  # Russell 2000 ETF
        'XLF': 48.0,   # Financial Select Sector
        'SPY': 590.0,  # S&P 500 ETF
        'QQQ': 520.0,  # Nasdaq 100 ETF
        'XLK': 230.0,  # Technology Select Sector
    }
    
    # Strategy allocation (from backtest: 100% IWM or 60/40 IWM/XLF)
    allocations = {
        'conservative': {'IWM': 0.6, 'XLF': 0.4},
        'concentrated': {'IWM': 1.0}
    }
    
    results = {}
    
    for alloc_name, weights in allocations.items():
        positions = {}
        total_invested = 0
        
        for ticker, weight in weights.items():
            allocation = capital * weight
            shares = int(allocation / prices[ticker])
            actual_value = shares * prices[ticker]
            positions[ticker] = {
                'target_allocation': allocation,
                'shares': shares,
                'actual_value': actual_value,
                'price_per_share': prices[ticker]
            }
            total_invested += actual_value
        
        cash_remainder = capital - total_invested
        
        # Transaction cost analysis at micro scale
        # IB commission: $1 minimum or $0.005/share
        ib_commission_per_trade = {}
        for ticker, pos in positions.items():
            shares = pos['shares']
            # IB tiered: $0.0035/share (min $0.35, max 1% of trade value)
            per_share_cost = max(0.35, shares * 0.0035)
            pct_of_value = per_share_cost / pos['actual_value'] * 100 if pos['actual_value'] > 0 else 0
            ib_commission_per_trade[ticker] = {
                'shares': shares,
                'commission': per_share_cost,
                'pct_of_trade': pct_of_value,
                'roundtrip_cost_pct': pct_of_value * 2
            }
        
        # Compare to backtest assumption
        backtest_cost_bps = 10  # 0.1% per trade assumed
        actual_avg_cost_pct = np.mean([v['roundtrip_cost_pct'] for v in ib_commission_per_trade.values()])
        cost_multiplier = actual_avg_cost_pct / (backtest_cost_bps / 100) if backtest_cost_bps > 0 else 1
        
        results[alloc_name] = {
            'capital': capital,
            'positions': positions,
            'total_invested': total_invested,
            'cash_remainder': cash_remainder,
            'utilization': total_invested / capital,
            'commission_analysis': ib_commission_per_trade,
            'avg_roundtrip_cost_pct': actual_avg_cost_pct,
            'backtest_assumed_cost_pct': backtest_cost_bps / 100,
            'cost_multiplier_vs_backtest': cost_multiplier
        }
    
    # Minimum viable capital calculation
    min_shares_per_position = 10  # Reasonable minimum for trading
    min_capital_needed = {}
    for ticker, price in prices.items():
        min_capital_needed[ticker] = min_shares_per_position * price
    
    # For the strategy (IWM-focused)
    min_viable = min_shares_per_position * prices['IWM']
    recommended_capital = min_viable * 2  # 2x for diversification and cash buffer
    
    return {
        'capital_tested': capital,
        'allocations': results,
        'minimum_viable_capital': min_viable,
        'recommended_capital': recommended_capital,
        'prices_used': prices,
        'feasibility_verdict': capital >= min_viable
    }


# ============================================================================
# COMPARISON & VERDICT
# ============================================================================

def generate_comparison(strategy: Dict, benchmark: Dict) -> Dict:
    """
    Generate head-to-head comparison between strategy and SPY benchmark.
    """
    comparison = {
        'metric_comparison': {},
        'absolute_verdict': None,
        'risk_adjusted_verdict': None,
        'overall_recommendation': None
    }
    
    # Extract key metrics
    strat_return = strategy.get('wgt_total_return', 0)
    strat_cagr = strategy.get('wgt_cagr', 0)
    strat_sharpe = strategy.get('wgt_sharpe_mean', 0)
    strat_dd = strategy.get('max_drawdown', 0)
    
    spy_return = benchmark.get('total_return', 0)
    spy_cagr = benchmark.get('cagr', 0)
    spy_sharpe = benchmark.get('sharpe_ratio', 0)
    spy_dd = abs(benchmark.get('max_drawdown', 0))
    spy_calmar = benchmark.get('calmar_ratio', 0)
    
    # Calculate Calmar for strategy
    strat_calmar = strat_cagr / strat_dd if strat_dd > 0 else 0
    
    comparison['metric_comparison'] = {
        'total_return_5yr': {
            'strategy': f"{strat_return*100:.2f}%",
            'spy': f"{spy_return*100:.2f}%",
            'winner': 'Strategy' if strat_return > spy_return else 'SPY',
            'difference': f"{(strat_return - spy_return)*100:.2f}%"
        },
        'annualized_return': {
            'strategy': f"{strat_cagr*100:.2f}%",
            'spy': f"{spy_cagr*100:.2f}%",
            'winner': 'Strategy' if strat_cagr > spy_cagr else 'SPY',
            'difference': f"{(strat_cagr - spy_cagr)*100:.2f}%"
        },
        'sharpe_ratio': {
            'strategy': f"{strat_sharpe:.3f}",
            'spy': f"{spy_sharpe:.3f}",
            'winner': 'Strategy' if strat_sharpe > spy_sharpe else 'SPY',
            'difference': f"{strat_sharpe - spy_sharpe:.3f}"
        },
        'max_drawdown': {
            'strategy': f"{strat_dd*100:.2f}%",
            'spy': f"{spy_dd*100:.2f}%",
            'winner': 'Strategy' if strat_dd < spy_dd else 'SPY',
            'note': 'Lower is better'
        },
        'calmar_ratio': {
            'strategy': f"{strat_calmar:.3f}",
            'spy': f"{spy_calmar:.3f}",
            'winner': 'Strategy' if strat_calmar > spy_calmar else 'SPY'
        }
    }
    
    # Absolute returns verdict
    if strat_return > spy_return:
        comparison['absolute_verdict'] = {
            'result': 'STRATEGY BEATS SPY',
            'margin': f"{(strat_return - spy_return)*100:.2f}%",
            'explanation': f"Strategy returned {strat_return*100:.2f}% vs SPY's {spy_return*100:.2f}%"
        }
    else:
        comparison['absolute_verdict'] = {
            'result': 'SPY BEATS STRATEGY',
            'margin': f"{(spy_return - strat_return)*100:.2f}%",
            'explanation': f"SPY returned {spy_return*100:.2f}% vs Strategy's {strat_return*100:.2f}%"
        }
    
    # Risk-adjusted verdict
    if strat_sharpe > spy_sharpe:
        comparison['risk_adjusted_verdict'] = {
            'result': 'STRATEGY BEATS SPY (Risk-Adjusted)',
            'sharpe_advantage': f"{strat_sharpe - spy_sharpe:.3f}",
            'explanation': f"Strategy Sharpe {strat_sharpe:.3f} vs SPY Sharpe {spy_sharpe:.3f}"
        }
    else:
        comparison['risk_adjusted_verdict'] = {
            'result': 'SPY BEATS STRATEGY (Risk-Adjusted)',
            'sharpe_advantage': f"{spy_sharpe - strat_sharpe:.3f}",
            'explanation': f"SPY Sharpe {spy_sharpe:.3f} vs Strategy Sharpe {strat_sharpe:.3f}"
        }
    
    # Drawdown comparison
    comparison['drawdown_comparison'] = {
        'strategy_max_dd': f"{strat_dd*100:.2f}%",
        'spy_max_dd': f"{spy_dd*100:.2f}%",
        'user_tolerance': "5-8%",
        'strategy_within_tolerance': strat_dd <= 0.08,
        'spy_within_tolerance': spy_dd <= 0.08
    }
    
    return comparison


# ============================================================================
# DEPLOYMENT RECOMMENDATION
# ============================================================================

def generate_deployment_recommendation(strategy: Dict, benchmark: Dict, 
                                        comparison: Dict, micro_capital: Dict) -> Dict:
    """
    Generate final deployment recommendation.
    """
    rec = {
        'verdict': None,
        'confidence': None,
        'minimum_capital': None,
        'expected_performance': {},
        'risks': [],
        'mitigations': [],
        'action_items': []
    }
    
    # Decision criteria
    beats_spy_absolute = 'STRATEGY BEATS' in comparison.get('absolute_verdict', {}).get('result', '')
    beats_spy_risk_adj = 'STRATEGY BEATS' in comparison.get('risk_adjusted_verdict', {}).get('result', '')
    within_dd_tolerance = comparison.get('drawdown_comparison', {}).get('strategy_within_tolerance', False)
    capital_feasible = micro_capital.get('feasibility_verdict', False)
    
    strat_sharpe = strategy.get('wgt_sharpe_mean', 0)
    strat_return = strategy.get('wgt_total_return', 0)
    spy_return = benchmark.get('total_return', 0)
    
    # Scoring
    score = 0
    max_score = 5
    
    if beats_spy_absolute:
        score += 1
    if beats_spy_risk_adj:
        score += 1
    if within_dd_tolerance:
        score += 1
    if strat_sharpe > 0.5:
        score += 1
    if strategy.get('positive_wgt_folds', 0) >= 5:  # Majority of folds positive
        score += 1
    
    # Determine verdict
    if score >= 4 and beats_spy_absolute:
        rec['verdict'] = 'PROCEED WITH CAUTION'
        rec['confidence'] = 'Medium'
    elif score >= 3 and strat_sharpe > 0.5:
        rec['verdict'] = 'PAPER TRADE FIRST'
        rec['confidence'] = 'Low-Medium'
    elif strat_return < 0 or strat_sharpe < 0:
        rec['verdict'] = 'DO NOT DEPLOY - OPTIMIZE FURTHER'
        rec['confidence'] = 'High (against deployment)'
    else:
        rec['verdict'] = 'OPTIMIZE FURTHER BEFORE DEPLOYMENT'
        rec['confidence'] = 'Low'
    
    # Capital recommendation
    if capital_feasible:
        rec['minimum_capital'] = micro_capital.get('minimum_viable_capital', 2500)
        rec['recommended_capital'] = micro_capital.get('recommended_capital', 5000)
    else:
        rec['minimum_capital'] = micro_capital.get('recommended_capital', 5000)
        rec['recommended_capital'] = micro_capital.get('recommended_capital', 5000) * 2
    
    # Expected performance degradation
    backtest_return = strat_return
    
    # Realistic expectations with live trading degradation
    degradation_factors = {
        'optimistic': 0.20,  # 20% degradation
        'realistic': 0.40,   # 40% degradation
        'conservative': 0.60 # 60% degradation
    }
    
    rec['expected_performance'] = {
        'backtest_annual_return': f"{strategy.get('wgt_cagr', 0)*100:.2f}%",
        'optimistic_live': f"{strategy.get('wgt_cagr', 0)*(1-degradation_factors['optimistic'])*100:.2f}%",
        'realistic_live': f"{strategy.get('wgt_cagr', 0)*(1-degradation_factors['realistic'])*100:.2f}%",
        'conservative_live': f"{strategy.get('wgt_cagr', 0)*(1-degradation_factors['conservative'])*100:.2f}%",
        'note': 'Live trading typically underperforms backtest by 20-60%'
    }
    
    # Dollar expectations at $2K
    capital = 2000
    annual_return_realistic = strategy.get('wgt_cagr', 0) * (1 - degradation_factors['realistic'])
    expected_annual_pnl = capital * annual_return_realistic
    
    rec['dollar_expectations'] = {
        'starting_capital': f"${capital:,}",
        'expected_annual_pnl_optimistic': f"${capital * strategy.get('wgt_cagr', 0) * (1-0.2):.0f}",
        'expected_annual_pnl_realistic': f"${expected_annual_pnl:.0f}",
        'expected_annual_pnl_conservative': f"${capital * strategy.get('wgt_cagr', 0) * (1-0.6):.0f}",
        'is_worth_effort': expected_annual_pnl > 50  # More than $50/year
    }
    
    # Risks
    rec['risks'] = [
        'Strategy shows high variance across time periods (Sharpe std > 1.3)',
        'Only 5-6 out of 8 walk-forward folds were profitable',
        'Current backtest period may not represent future conditions',
        'Small capital amplifies transaction cost impact',
        'Limited diversification with single-asset (IWM) concentration'
    ]
    
    # Mitigations
    rec['mitigations'] = [
        'Start with paper trading for 3-6 months to verify signals',
        'Use larger capital ($5K+) to reduce transaction cost drag',
        'Implement strict position sizing (max 2% risk per trade)',
        'Monitor for regime changes that invalidate model',
        'Set monthly review checkpoints to evaluate performance'
    ]
    
    # Action items
    if 'PROCEED' in rec['verdict']:
        rec['action_items'] = [
            '1. Paper trade for minimum 3 months',
            '2. Track actual vs predicted signals',
            '3. Verify transaction costs match assumptions',
            '4. Start with $2K only if paper trading is successful',
            '5. Scale to full position sizing after 6 months profitable'
        ]
    else:
        rec['action_items'] = [
            '1. Investigate why strategy underperforms SPY',
            '2. Consider alternative feature engineering',
            '3. Test different asset allocation strategies',
            '4. Evaluate longer training periods',
            '5. Compare against other benchmarks (QQQ, IWM itself)'
        ]
    
    return rec


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(strategy: Dict, benchmark: Dict, comparison: Dict, 
                    micro_capital: Dict, recommendation: Dict) -> str:
    """
    Generate comprehensive text report.
    """
    lines = []
    
    # Header
    lines.append("=" * 80)
    lines.append("TDA+NN TRADING STRATEGY - DEPLOYMENT ANALYSIS REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    lines.append("")
    
    # Executive Summary
    lines.append("EXECUTIVE SUMMARY")
    lines.append("=" * 40)
    abs_verdict = comparison.get('absolute_verdict', {}).get('result', 'UNKNOWN')
    risk_verdict = comparison.get('risk_adjusted_verdict', {}).get('result', 'UNKNOWN')
    
    lines.append(f"Absolute Returns: {abs_verdict}")
    lines.append(f"Risk-Adjusted: {risk_verdict}")
    lines.append(f"Deployment Recommendation: {recommendation.get('verdict', 'UNKNOWN')}")
    lines.append(f"Confidence: {recommendation.get('confidence', 'UNKNOWN')}")
    lines.append(f"Minimum Capital: ${recommendation.get('minimum_capital', 0):,.0f}")
    lines.append(f"Recommended Capital: ${recommendation.get('recommended_capital', 0):,.0f}")
    lines.append("")
    
    # Key Metrics Comparison
    lines.append("KEY METRICS COMPARISON")
    lines.append("=" * 40)
    lines.append(f"{'Metric':<25} {'Strategy':>12} {'SPY B&H':>12} {'Winner':>10}")
    lines.append("-" * 60)
    
    for metric, data in comparison.get('metric_comparison', {}).items():
        name = metric.replace('_', ' ').title()
        strat_val = data.get('strategy', 'N/A')
        spy_val = data.get('spy', 'N/A')
        winner = data.get('winner', 'N/A')
        lines.append(f"{name:<25} {strat_val:>12} {spy_val:>12} {winner:>10}")
    lines.append("")
    
    # Strategy Performance Detail
    lines.append("5-YEAR STRATEGY PERFORMANCE (Walk-Forward Validated)")
    lines.append("=" * 40)
    lines.append(f"Period: {strategy.get('period', 'N/A')}")
    lines.append(f"Walk-Forward Folds: {strategy.get('n_folds', 0)}")
    lines.append(f"Years Covered: {strategy.get('years_covered', 0):.1f}")
    lines.append("")
    lines.append("Performance-Weighted Portfolio:")
    lines.append(f"  Total Return: {strategy.get('wgt_total_return', 0)*100:.2f}%")
    lines.append(f"  CAGR: {strategy.get('wgt_cagr', 0)*100:.2f}%")
    lines.append(f"  Sharpe Ratio (mean): {strategy.get('wgt_sharpe_mean', 0):.3f}")
    lines.append(f"  Sharpe Ratio (std): {strategy.get('wgt_sharpe_std', 0):.3f}")
    lines.append(f"  Max Drawdown: {strategy.get('max_drawdown', 0)*100:.2f}%")
    lines.append(f"  Total Trades: {strategy.get('total_trades', 0)}")
    lines.append(f"  Win Rate: {strategy.get('overall_win_rate', 0)*100:.1f}%")
    lines.append(f"  Profitable Folds: {strategy.get('positive_wgt_folds', 0)}/{strategy.get('n_folds', 0)}")
    lines.append("")
    
    # SPY Benchmark Detail
    lines.append("SPY BUY-AND-HOLD BENCHMARK")
    lines.append("=" * 40)
    lines.append(f"Period: {benchmark.get('period', 'N/A')}")
    lines.append(f"Total Return: {benchmark.get('total_return', 0)*100:.2f}%")
    lines.append(f"CAGR: {benchmark.get('cagr', 0)*100:.2f}%")
    lines.append(f"Sharpe Ratio: {benchmark.get('sharpe_ratio', 0):.3f}")
    lines.append(f"Sortino Ratio: {benchmark.get('sortino_ratio', 0):.3f}")
    lines.append(f"Max Drawdown: {benchmark.get('max_drawdown', 0)*100:.2f}%")
    lines.append(f"Calmar Ratio: {benchmark.get('calmar_ratio', 0):.3f}")
    lines.append("")
    
    # Year-by-year SPY
    lines.append("SPY Year-by-Year Returns:")
    yearly = benchmark.get('yearly_returns', {})
    for year in sorted(yearly.keys()):
        yr_data = yearly[year]
        lines.append(f"  {year}: {yr_data.get('return', 0)*100:>6.1f}% (Sharpe: {yr_data.get('sharpe', 0):.2f}, MaxDD: {yr_data.get('max_drawdown', 0)*100:.1f}%)")
    lines.append("")
    
    # Micro Capital Analysis
    lines.append("MICRO CAPITAL ANALYSIS ($2,000)")
    lines.append("=" * 40)
    alloc = micro_capital.get('allocations', {}).get('conservative', {})
    lines.append(f"Capital: ${micro_capital.get('capital_tested', 0):,.0f}")
    lines.append(f"Feasibility: {'YES' if micro_capital.get('feasibility_verdict') else 'NO'}")
    lines.append(f"Minimum Viable Capital: ${micro_capital.get('minimum_viable_capital', 0):,.0f}")
    lines.append(f"Recommended Capital: ${micro_capital.get('recommended_capital', 0):,.0f}")
    lines.append("")
    lines.append("Position Sizes at $2K (60/40 IWM/XLF):")
    for ticker, pos in alloc.get('positions', {}).items():
        lines.append(f"  {ticker}: {pos.get('shares', 0)} shares @ ${pos.get('price_per_share', 0):.0f} = ${pos.get('actual_value', 0):.0f}")
    lines.append(f"  Cash Remainder: ${alloc.get('cash_remainder', 0):.0f}")
    lines.append("")
    lines.append("Transaction Cost Impact:")
    lines.append(f"  Backtest Assumed: {alloc.get('backtest_assumed_cost_pct', 0)*100:.2f}% per trade")
    lines.append(f"  Actual at $2K: {alloc.get('avg_roundtrip_cost_pct', 0):.2f}% per roundtrip")
    lines.append(f"  Cost Multiplier: {alloc.get('cost_multiplier_vs_backtest', 0):.1f}x backtest assumption")
    lines.append("")
    
    # Expected Performance
    lines.append("EXPECTED LIVE PERFORMANCE")
    lines.append("=" * 40)
    exp = recommendation.get('expected_performance', {})
    lines.append(f"Backtest Annual Return: {exp.get('backtest_annual_return', 'N/A')}")
    lines.append(f"Optimistic (20% degradation): {exp.get('optimistic_live', 'N/A')}")
    lines.append(f"Realistic (40% degradation): {exp.get('realistic_live', 'N/A')}")
    lines.append(f"Conservative (60% degradation): {exp.get('conservative_live', 'N/A')}")
    lines.append("")
    dollar_exp = recommendation.get('dollar_expectations', {})
    lines.append(f"At $2,000 Starting Capital:")
    lines.append(f"  Optimistic Annual P&L: {dollar_exp.get('expected_annual_pnl_optimistic', 'N/A')}")
    lines.append(f"  Realistic Annual P&L: {dollar_exp.get('expected_annual_pnl_realistic', 'N/A')}")
    lines.append(f"  Conservative Annual P&L: {dollar_exp.get('expected_annual_pnl_conservative', 'N/A')}")
    lines.append(f"  Worth the Effort: {'YES' if dollar_exp.get('is_worth_effort') else 'NO'}")
    lines.append("")
    
    # Drawdown Analysis
    lines.append("DRAWDOWN TOLERANCE CHECK")
    lines.append("=" * 40)
    dd = comparison.get('drawdown_comparison', {})
    lines.append(f"User Tolerance: {dd.get('user_tolerance', '5-8%')}")
    lines.append(f"Strategy Max DD: {dd.get('strategy_max_dd', 'N/A')}")
    lines.append(f"SPY Max DD: {dd.get('spy_max_dd', 'N/A')}")
    lines.append(f"Strategy Within Tolerance: {'YES ✓' if dd.get('strategy_within_tolerance') else 'NO ✗'}")
    lines.append(f"SPY Within Tolerance: {'YES ✓' if dd.get('spy_within_tolerance') else 'NO ✗'}")
    lines.append("")
    
    # Risks
    lines.append("IDENTIFIED RISKS")
    lines.append("=" * 40)
    for i, risk in enumerate(recommendation.get('risks', []), 1):
        lines.append(f"  {i}. {risk}")
    lines.append("")
    
    # Mitigations
    lines.append("RECOMMENDED MITIGATIONS")
    lines.append("=" * 40)
    for i, mitigation in enumerate(recommendation.get('mitigations', []), 1):
        lines.append(f"  {i}. {mitigation}")
    lines.append("")
    
    # Final Recommendation
    lines.append("=" * 80)
    lines.append("FINAL RECOMMENDATION")
    lines.append("=" * 80)
    lines.append(f"VERDICT: {recommendation.get('verdict', 'UNKNOWN')}")
    lines.append(f"CONFIDENCE: {recommendation.get('confidence', 'UNKNOWN')}")
    lines.append("")
    lines.append("Action Items:")
    for item in recommendation.get('action_items', []):
        lines.append(f"  {item}")
    lines.append("")
    
    # Critical Bottom Line
    lines.append("=" * 80)
    lines.append("CRITICAL BOTTOM LINE FOR USER")
    lines.append("=" * 80)
    
    spy_return = benchmark.get('total_return', 0)
    strat_return = strategy.get('wgt_total_return', 0)
    
    if strat_return > spy_return:
        lines.append(f"✓ Strategy DOES beat SPY on absolute returns (+{(strat_return-spy_return)*100:.2f}%)")
    else:
        lines.append(f"✗ Strategy DOES NOT beat SPY on absolute returns ({(strat_return-spy_return)*100:.2f}%)")
    
    strat_sharpe = strategy.get('wgt_sharpe_mean', 0)
    spy_sharpe = benchmark.get('sharpe_ratio', 0)
    
    if strat_sharpe > spy_sharpe:
        lines.append(f"✓ Strategy DOES beat SPY on risk-adjusted basis (Sharpe +{strat_sharpe-spy_sharpe:.3f})")
    else:
        lines.append(f"✗ Strategy DOES NOT beat SPY on risk-adjusted basis (Sharpe {strat_sharpe-spy_sharpe:.3f})")
    
    if micro_capital.get('feasibility_verdict'):
        lines.append(f"✓ $2K capital IS feasible (minimum ${micro_capital.get('minimum_viable_capital', 0):,.0f})")
    else:
        lines.append(f"✗ $2K capital NOT recommended (need ${micro_capital.get('recommended_capital', 0):,.0f})")
    
    strat_dd = strategy.get('max_drawdown', 0)
    if strat_dd <= 0.08:
        lines.append(f"✓ Max drawdown {strat_dd*100:.1f}% is within 5-8% tolerance")
    else:
        lines.append(f"✗ Max drawdown {strat_dd*100:.1f}% exceeds 5-8% tolerance")
    
    lines.append("")
    lines.append("=" * 80)
    
    return '\n'.join(lines)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run complete deployment analysis."""
    
    print("=" * 60)
    print("DEPLOYMENT ANALYSIS - 5-Year Performance & SPY Comparison")
    print("=" * 60)
    
    # 1. Calculate SPY benchmark
    print("\n[1/5] Calculating SPY benchmark...")
    benchmark = calculate_spy_benchmark('2020-01-01', '2025-01-15')
    if 'error' in benchmark:
        print(f"ERROR: {benchmark['error']}")
        return
    print(f"  SPY Total Return: {benchmark['total_return']*100:.2f}%")
    print(f"  SPY Sharpe: {benchmark['sharpe_ratio']:.3f}")
    
    # 2. Extract strategy performance
    print("\n[2/5] Extracting strategy performance...")
    strategy = extract_strategy_performance()
    if 'error' in strategy:
        print(f"ERROR: {strategy['error']}")
        return
    print(f"  Strategy Total Return: {strategy['wgt_total_return']*100:.2f}%")
    print(f"  Strategy Sharpe: {strategy['wgt_sharpe_mean']:.3f}")
    
    # 3. Generate comparison
    print("\n[3/5] Generating comparison...")
    comparison = generate_comparison(strategy, benchmark)
    print(f"  Absolute: {comparison['absolute_verdict']['result']}")
    print(f"  Risk-Adjusted: {comparison['risk_adjusted_verdict']['result']}")
    
    # 4. Micro capital analysis
    print("\n[4/5] Analyzing $2K capital feasibility...")
    micro_capital = analyze_micro_capital(2000)
    print(f"  Feasible: {micro_capital['feasibility_verdict']}")
    print(f"  Recommended: ${micro_capital['recommended_capital']:,.0f}")
    
    # 5. Generate recommendation
    print("\n[5/5] Generating deployment recommendation...")
    recommendation = generate_deployment_recommendation(
        strategy, benchmark, comparison, micro_capital
    )
    print(f"  Verdict: {recommendation['verdict']}")
    print(f"  Confidence: {recommendation['confidence']}")
    
    # Generate full report
    report = generate_report(strategy, benchmark, comparison, micro_capital, recommendation)
    
    # Save report
    report_path = f'{RESULTS_DIR}/deployment_analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\n✓ Report saved to: {report_path}")
    
    # Save JSON data
    json_data = {
        'generated_at': datetime.now().isoformat(),
        'benchmark': benchmark,
        'strategy': strategy,
        'comparison': comparison,
        'micro_capital': micro_capital,
        'recommendation': recommendation
    }
    json_path = f'{RESULTS_DIR}/deployment_analysis_data.json'
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"✓ Data saved to: {json_path}")
    
    # Print report summary
    print("\n" + "=" * 60)
    print("REPORT SUMMARY")
    print("=" * 60)
    print(report)
    
    return json_data


if __name__ == "__main__":
    main()
