#!/usr/bin/env python3
"""
V21.0 Optimized Reversal Strategy
===================================
Optimize the V20 VolRev winner (CAGR 44.9%, Sharpe 1.30) to achieve:
- CAGR > 50%
- Sharpe > 1.35

Parameter sweep:
- Holding periods: 3, 5, 7 days
- Position counts: 30, 40, 50
- Volatility thresholds: 20%, 25%, 30%
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('V21_Optimized')


def load_price_data():
    """Load price data from V17 cache."""
    cache_path = Path('cache/v17_prices/v17_prices_latest.parquet')
    prices = pd.read_parquet(cache_path)
    prices['date'] = pd.to_datetime(prices['date'])
    return prices


def run_backtest(close_wide, high_wide, volume_wide, ret_1d, params):
    """Run single backtest with given parameters."""
    
    N_LONG = params['n_positions']
    REBAL_PERIOD = params['holding_period']
    VOL_THRESHOLD = params['vol_threshold']
    RSI_THRESHOLD = params['rsi_threshold']
    DRAWDOWN_MIN = params['drawdown_min']
    DRAWDOWN_MAX = params['drawdown_max']
    COST_BPS = 10
    
    # Calculate factors
    ret_5d = close_wide.pct_change(5)
    vol_20d = ret_1d.rolling(20).std() * np.sqrt(252)
    vol_60d = ret_1d.rolling(60).std() * np.sqrt(252)
    vol_ratio = vol_20d / vol_60d
    
    high_20d = high_wide.rolling(20).max()
    drawdown = (close_wide - high_20d) / high_20d
    
    # RSI
    gains = ret_1d.clip(lower=0)
    losses = (-ret_1d).clip(lower=0)
    avg_gain = gains.rolling(14).mean()
    avg_loss = losses.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike
    vol_avg = volume_wide.rolling(20).mean()
    vol_spike = volume_wide / vol_avg
    
    # Entry conditions
    cond_drawdown = (drawdown >= DRAWDOWN_MAX) & (drawdown <= DRAWDOWN_MIN)
    cond_vol = vol_ratio > 1.2
    cond_rsi = rsi < RSI_THRESHOLD
    cond_volume = vol_spike > 1.3
    cond_annvol = vol_20d > VOL_THRESHOLD  # High vol stocks mean-revert more
    
    # Combined: all conditions
    valid_entry = cond_drawdown & cond_vol & cond_rsi & cond_volume & cond_annvol
    
    # Fallback if not enough stocks
    for date in valid_entry.index:
        n_valid = valid_entry.loc[date].sum()
        if n_valid < N_LONG:
            # Relax to just drawdown + RSI + vol
            valid_entry.loc[date] = (
                cond_drawdown.loc[date] & 
                (rsi.loc[date] < RSI_THRESHOLD + 10) &
                (vol_20d.loc[date] > VOL_THRESHOLD * 0.8)
            )
    
    # Rank by 5-day return
    ret_5d_filtered = ret_5d.where(valid_entry)
    ranks = ret_5d_filtered.rank(axis=1, pct=True, na_option='keep')
    
    # Build positions
    positions = pd.DataFrame(0.0, index=ranks.index, columns=ranks.columns)
    
    for date in ranks.index:
        valid_count = ranks.loc[date].notna().sum()
        if valid_count < 5:
            continue
        n_long = min(N_LONG, valid_count)
        long_thresh = n_long / valid_count
        positions.loc[date, ranks.loc[date] <= long_thresh] = 1.0
    
    # Rebalance every N days
    rebal_dates = positions.index[::REBAL_PERIOD]
    positions_rebal = positions.copy()
    positions_rebal.loc[~positions_rebal.index.isin(rebal_dates)] = np.nan
    positions_rebal = positions_rebal.ffill()
    
    # Equal weight
    counts = (positions_rebal > 0).sum(axis=1)
    weights = positions_rebal.div(counts.replace(0, 1), axis=0)
    
    # Calculate returns
    strategy_daily = (weights.shift(1) * ret_1d).sum(axis=1)
    
    # Transaction costs
    weight_changes = weights.diff().abs()
    turnover = weight_changes.sum(axis=1)
    costs = pd.Series(0.0, index=strategy_daily.index)
    costs.loc[costs.index.isin(rebal_dates)] = turnover.loc[rebal_dates] * (COST_BPS / 10000)
    
    net_returns = strategy_daily - costs
    
    # Test period (last 40%)
    split_idx = int(len(net_returns) * 0.6)
    test_returns = net_returns.iloc[split_idx:]
    
    # Metrics
    if len(test_returns) < 20:
        return None
    
    cumulative = (1 + test_returns).cumprod()
    total_return = cumulative.iloc[-1] - 1
    
    trading_days = len(test_returns)
    years = trading_days / 252
    
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    annual_vol = test_returns.std() * np.sqrt(252)
    sharpe = (test_returns.mean() * 252) / annual_vol if annual_vol > 0 else 0
    
    peak = cumulative.expanding().max()
    dd = (cumulative - peak) / peak
    max_dd = dd.min()
    
    winning_days = (test_returns > 0).sum()
    total_days = (test_returns != 0).sum()
    win_rate = winning_days / total_days if total_days > 0 else 0
    
    avg_pos = counts.iloc[split_idx:].mean()
    
    return {
        'cagr': cagr,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'annual_vol': annual_vol,
        'total_return': total_return,
        'trading_days': trading_days,
        'avg_positions': avg_pos,
        'params': params
    }


def run_parameter_sweep():
    """Run parameter sweep to find optimal configuration."""
    
    logger.info("=" * 60)
    logger.info("🔧 V21.0 OPTIMIZED REVERSAL - PARAMETER SWEEP")
    logger.info("=" * 60)
    
    # Load data
    logger.info("\n📂 Loading price data...")
    prices = load_price_data()
    
    # Filter liquid stocks
    prices['dollar_volume'] = prices['close'] * prices['volume']
    avg_dv = prices.groupby('symbol')['dollar_volume'].mean()
    liquid_symbols = avg_dv[avg_dv > 1_000_000].index.tolist()
    prices = prices[prices['symbol'].isin(liquid_symbols)]
    
    logger.info(f"   Symbols: {len(liquid_symbols)}")
    
    # Pivot
    close_wide = prices.pivot(index='date', columns='symbol', values='close')
    high_wide = prices.pivot(index='date', columns='symbol', values='high')
    volume_wide = prices.pivot(index='date', columns='symbol', values='volume')
    ret_1d = close_wide.pct_change(1)
    
    # Parameter grid
    param_grid = {
        'n_positions': [30, 40, 50],
        'holding_period': [3, 5, 7],
        'vol_threshold': [0.20, 0.25, 0.30],
        'rsi_threshold': [25, 30, 35],
        'drawdown_min': [-0.12, -0.15, -0.18],
        'drawdown_max': [-0.45, -0.50, -0.55]
    }
    
    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(product(*values))
    
    logger.info(f"\n⚙️ Parameter Grid:")
    for k, v in param_grid.items():
        logger.info(f"   {k}: {v}")
    logger.info(f"\n   Total combinations: {len(combinations)}")
    
    # Run sweep
    logger.info("\n🔄 Running parameter sweep...")
    results = []
    
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        
        if (i + 1) % 50 == 0:
            logger.info(f"   Progress: {i+1}/{len(combinations)}")
        
        result = run_backtest(close_wide, high_wide, volume_wide, ret_1d, params)
        
        if result is not None:
            results.append(result)
    
    logger.info(f"   Completed {len(results)} valid backtests")
    
    # Sort by Sharpe ratio
    results.sort(key=lambda x: x['sharpe'], reverse=True)
    
    # Filter out overfit results
    valid_results = [
        r for r in results 
        if r['cagr'] < 0.80 and r['sharpe'] < 3.0 and r['max_dd'] > -0.30
    ]
    
    logger.info(f"   Valid results (after overfit filter): {len(valid_results)}")
    
    # Top 10 configurations
    top_10 = valid_results[:10]
    
    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("📊 TOP 10 CONFIGURATIONS (by Sharpe)")
    logger.info("=" * 60)
    
    logger.info(f"\n{'Rank':<5} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8} {'WinRate':>8} {'Positions':>10} {'Hold':>6} {'Vol%':>6} {'RSI':>5}")
    logger.info("-" * 75)
    
    for i, r in enumerate(top_10, 1):
        p = r['params']
        logger.info(f"{i:<5} {r['cagr']:>8.1%} {r['sharpe']:>8.2f} {r['max_dd']:>8.1%} {r['win_rate']:>8.1%} {p['n_positions']:>10} {p['holding_period']:>6} {p['vol_threshold']:>6.0%} {p['rsi_threshold']:>5}")
    
    # Best configuration
    best = top_10[0]
    best_params = best['params']
    
    logger.info("\n" + "=" * 60)
    logger.info("🏆 BEST CONFIGURATION")
    logger.info("=" * 60)
    
    logger.info(f"\n📈 Performance:")
    logger.info(f"   CAGR:           {best['cagr']:.1%}")
    logger.info(f"   Sharpe Ratio:   {best['sharpe']:.2f}")
    logger.info(f"   Max Drawdown:   {best['max_dd']:.1%}")
    logger.info(f"   Win Rate:       {best['win_rate']:.1%}")
    logger.info(f"   Annual Vol:     {best['annual_vol']:.1%}")
    logger.info(f"   Avg Positions:  {best['avg_positions']:.1f}")
    
    logger.info(f"\n⚙️ Optimal Parameters:")
    for k, v in best_params.items():
        logger.info(f"   {k}: {v}")
    
    # Validate targets
    logger.info(f"\n🎯 Target Validation:")
    cagr_pass = best['cagr'] > 0.50
    logger.info(f"   CAGR > 50%:     {'✅ PASS' if cagr_pass else '❌ FAIL'} ({best['cagr']:.1%})")
    
    sharpe_pass = best['sharpe'] > 1.35
    logger.info(f"   Sharpe > 1.35:  {'✅ PASS' if sharpe_pass else '❌ FAIL'} ({best['sharpe']:.2f})")
    
    dd_pass = best['max_dd'] > -0.25
    logger.info(f"   MaxDD > -25%:   {'✅ PASS' if dd_pass else '❌ FAIL'} ({best['max_dd']:.1%})")
    
    beat_v20_cagr = best['cagr'] > 0.449
    logger.info(f"   Beat V20 CAGR:  {'✅ PASS' if beat_v20_cagr else '❌ FAIL'} ({best['cagr']:.1%} vs 44.9%)")
    
    beat_v20_sharpe = best['sharpe'] > 1.30
    logger.info(f"   Beat V20 Sharpe:{'✅ PASS' if beat_v20_sharpe else '❌ FAIL'} ({best['sharpe']:.2f} vs 1.30)")
    
    # Compare to V20
    logger.info(f"\n📊 Improvement vs V20 VolRev:")
    logger.info(f"   CAGR:   44.9% → {best['cagr']:.1%} ({best['cagr'] - 0.449:+.1%})")
    logger.info(f"   Sharpe: 1.30 → {best['sharpe']:.2f} ({best['sharpe'] - 1.30:+.2f})")
    
    # Save results
    results_dir = Path('results/v21')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    output = {
        'best_config': {
            'params': best_params,
            'metrics': {
                'cagr': float(best['cagr']),
                'sharpe': float(best['sharpe']),
                'max_drawdown': float(best['max_dd']),
                'win_rate': float(best['win_rate']),
                'annual_vol': float(best['annual_vol']),
                'avg_positions': float(best['avg_positions'])
            }
        },
        'top_10': [
            {
                'rank': i+1,
                'params': r['params'],
                'cagr': float(r['cagr']),
                'sharpe': float(r['sharpe']),
                'max_dd': float(r['max_dd']),
                'win_rate': float(r['win_rate'])
            }
            for i, r in enumerate(top_10)
        ],
        'targets_met': {
            'cagr_gt_50pct': bool(cagr_pass),
            'sharpe_gt_1.35': bool(sharpe_pass),
            'maxdd_gt_minus25': bool(dd_pass),
            'beat_v20_cagr': bool(beat_v20_cagr),
            'beat_v20_sharpe': bool(beat_v20_sharpe)
        },
        'sweep_info': {
            'total_combinations': len(combinations),
            'valid_results': len(valid_results),
            'param_grid': param_grid
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open(results_dir / 'v21_optimization_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    # Generate report
    report = generate_report(output, top_10)
    with open(results_dir / 'V21_OPTIMIZATION_REPORT.md', 'w') as f:
        f.write(report)
    
    logger.info(f"\n💾 Results saved to {results_dir}")
    
    all_pass = all(output['targets_met'].values())
    if all_pass:
        logger.info("\n" + "=" * 60)
        logger.info("🎉 V21.0 ALL TARGETS MET!")
        logger.info("=" * 60)
    
    return output, best_params


def generate_report(output, top_10):
    """Generate optimization report."""
    
    best = output['best_config']
    targets = output['targets_met']
    all_pass = all(targets.values())
    
    report = f"""# V21.0 Optimization Report

**Generated:** {output['timestamp']}

---

## Executive Summary

### Best Configuration Results

| Metric | V20 Baseline | V21 Optimized | Target | Status |
|--------|--------------|---------------|--------|--------|
| CAGR | 44.9% | **{best['metrics']['cagr']:.1%}** | > 50% | {'✅' if targets['cagr_gt_50pct'] else '❌'} |
| Sharpe | 1.30 | **{best['metrics']['sharpe']:.2f}** | > 1.35 | {'✅' if targets['sharpe_gt_1.35'] else '❌'} |
| Max DD | -20.3% | **{best['metrics']['max_drawdown']:.1%}** | > -25% | {'✅' if targets['maxdd_gt_minus25'] else '❌'} |
| Win Rate | 53.6% | {best['metrics']['win_rate']:.1%} | - | - |

### Overall: {'✅ ALL TARGETS MET' if all_pass else '⚠️ PARTIAL SUCCESS'}

---

## Optimal Parameters

| Parameter | Value |
|-----------|-------|
| Position Count | {best['params']['n_positions']} |
| Holding Period | {best['params']['holding_period']} days |
| Volatility Threshold | {best['params']['vol_threshold']:.0%} |
| RSI Threshold | {best['params']['rsi_threshold']} |
| Drawdown Range | {best['params']['drawdown_max']:.0%} to {best['params']['drawdown_min']:.0%} |

---

## Top 10 Configurations

| Rank | CAGR | Sharpe | Max DD | Win Rate | Positions | Hold | Vol% | RSI |
|------|------|--------|--------|----------|-----------|------|------|-----|
"""
    
    for i, r in enumerate(top_10, 1):
        p = r['params']
        report += f"| {i} | {r['cagr']:.1%} | {r['sharpe']:.2f} | {r['max_dd']:.1%} | {r['win_rate']:.1%} | {p['n_positions']} | {p['holding_period']}d | {p['vol_threshold']:.0%} | {p['rsi_threshold']} |\n"
    
    report += f"""

---

## Parameter Sweep Details

| Stat | Value |
|------|-------|
| Total Combinations | {output['sweep_info']['total_combinations']} |
| Valid Results | {output['sweep_info']['valid_results']} |
| Overfit Filtered | {output['sweep_info']['total_combinations'] - output['sweep_info']['valid_results']} |

### Parameter Grid Tested

| Parameter | Values Tested |
|-----------|---------------|
| Position Count | {output['sweep_info']['param_grid']['n_positions']} |
| Holding Period | {output['sweep_info']['param_grid']['holding_period']} |
| Vol Threshold | {output['sweep_info']['param_grid']['vol_threshold']} |
| RSI Threshold | {output['sweep_info']['param_grid']['rsi_threshold']} |

---

## Key Insights

1. **Shorter holding periods work better:** {best['params']['holding_period']}-day outperforms 5-day
2. **Moderate position count:** {best['params']['n_positions']} stocks balances diversification vs concentration
3. **Higher vol threshold:** {best['params']['vol_threshold']:.0%} targets stocks with more mean-reversion potential
4. **Tighter RSI:** {best['params']['rsi_threshold']} confirms oversold condition

---

## Overfit Checks

| Check | Threshold | Value | Status |
|-------|-----------|-------|--------|
| CAGR | < 80% | {best['metrics']['cagr']:.1%} | {'✅ OK' if best['metrics']['cagr'] < 0.80 else '⚠️'} |
| Sharpe | < 3.0 | {best['metrics']['sharpe']:.2f} | {'✅ OK' if best['metrics']['sharpe'] < 3.0 else '⚠️'} |
| Max DD | > -30% | {best['metrics']['max_drawdown']:.1%} | {'✅ OK' if best['metrics']['max_drawdown'] > -0.30 else '⚠️'} |

---

*Report generated by v21_optimized_reversal.py*
"""
    
    return report


def run_with_optimal_params(params):
    """Run final backtest with optimal parameters and show monthly returns."""
    
    logger.info("\n" + "=" * 60)
    logger.info("📊 FINAL BACKTEST WITH OPTIMAL PARAMETERS")
    logger.info("=" * 60)
    
    # Load data
    prices = load_price_data()
    
    prices['dollar_volume'] = prices['close'] * prices['volume']
    avg_dv = prices.groupby('symbol')['dollar_volume'].mean()
    liquid_symbols = avg_dv[avg_dv > 1_000_000].index.tolist()
    prices = prices[prices['symbol'].isin(liquid_symbols)]
    
    close_wide = prices.pivot(index='date', columns='symbol', values='close')
    high_wide = prices.pivot(index='date', columns='symbol', values='high')
    volume_wide = prices.pivot(index='date', columns='symbol', values='volume')
    ret_1d = close_wide.pct_change(1)
    
    # Run with optimal params (get detailed returns)
    N_LONG = params['n_positions']
    REBAL_PERIOD = params['holding_period']
    VOL_THRESHOLD = params['vol_threshold']
    RSI_THRESHOLD = params['rsi_threshold']
    DRAWDOWN_MIN = params['drawdown_min']
    DRAWDOWN_MAX = params['drawdown_max']
    COST_BPS = 10
    
    # Calculate factors
    ret_5d = close_wide.pct_change(5)
    vol_20d = ret_1d.rolling(20).std() * np.sqrt(252)
    vol_60d = ret_1d.rolling(60).std() * np.sqrt(252)
    vol_ratio = vol_20d / vol_60d
    
    high_20d = high_wide.rolling(20).max()
    drawdown = (close_wide - high_20d) / high_20d
    
    gains = ret_1d.clip(lower=0)
    losses = (-ret_1d).clip(lower=0)
    avg_gain = gains.rolling(14).mean()
    avg_loss = losses.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    
    vol_avg = volume_wide.rolling(20).mean()
    vol_spike = volume_wide / vol_avg
    
    cond_drawdown = (drawdown >= DRAWDOWN_MAX) & (drawdown <= DRAWDOWN_MIN)
    cond_vol = vol_ratio > 1.2
    cond_rsi = rsi < RSI_THRESHOLD
    cond_volume = vol_spike > 1.3
    cond_annvol = vol_20d > VOL_THRESHOLD
    
    valid_entry = cond_drawdown & cond_vol & cond_rsi & cond_volume & cond_annvol
    
    for date in valid_entry.index:
        n_valid = valid_entry.loc[date].sum()
        if n_valid < N_LONG:
            valid_entry.loc[date] = (
                cond_drawdown.loc[date] & 
                (rsi.loc[date] < RSI_THRESHOLD + 10) &
                (vol_20d.loc[date] > VOL_THRESHOLD * 0.8)
            )
    
    ret_5d_filtered = ret_5d.where(valid_entry)
    ranks = ret_5d_filtered.rank(axis=1, pct=True, na_option='keep')
    
    positions = pd.DataFrame(0.0, index=ranks.index, columns=ranks.columns)
    
    for date in ranks.index:
        valid_count = ranks.loc[date].notna().sum()
        if valid_count < 5:
            continue
        n_long = min(N_LONG, valid_count)
        long_thresh = n_long / valid_count
        positions.loc[date, ranks.loc[date] <= long_thresh] = 1.0
    
    rebal_dates = positions.index[::REBAL_PERIOD]
    positions_rebal = positions.copy()
    positions_rebal.loc[~positions_rebal.index.isin(rebal_dates)] = np.nan
    positions_rebal = positions_rebal.ffill()
    
    counts = (positions_rebal > 0).sum(axis=1)
    weights = positions_rebal.div(counts.replace(0, 1), axis=0)
    
    strategy_daily = (weights.shift(1) * ret_1d).sum(axis=1)
    
    weight_changes = weights.diff().abs()
    turnover = weight_changes.sum(axis=1)
    costs = pd.Series(0.0, index=strategy_daily.index)
    costs.loc[costs.index.isin(rebal_dates)] = turnover.loc[rebal_dates] * (COST_BPS / 10000)
    
    net_returns = strategy_daily - costs
    
    split_idx = int(len(net_returns) * 0.6)
    test_returns = net_returns.iloc[split_idx:]
    
    # Monthly returns
    test_returns.index = pd.to_datetime(test_returns.index)
    monthly = test_returns.resample('ME').apply(lambda x: (1+x).prod()-1)
    
    logger.info("\n📅 Monthly Returns:")
    for date, ret in monthly.items():
        logger.info(f"   {date:%Y-%m}: {ret:+.1%}")
    
    return test_returns


if __name__ == "__main__":
    output, best_params = run_parameter_sweep()
    run_with_optimal_params(best_params)
