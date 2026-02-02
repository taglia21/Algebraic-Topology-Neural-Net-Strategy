#!/usr/bin/env python3
"""
V19.0 Phase 1: Pure Reversal Strategy
======================================
Exploit the negative IC of momentum factors by going LONG losers, SHORT winners.

Logic:
- Rank all stocks by 5-day return (worst to best)
- LONG: Bottom 30 performers (biggest losers → expect reversion UP)
- SHORT: Top 30 performers (biggest winners → expect reversion DOWN)
- Holding period: 5 days, then rebalance
- Equal weight positions

Based on diagnostic finding: reversal_5d has IC = +0.061
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('V19_Reversal')


def load_price_data():
    """Load price data from V17 cache."""
    cache_path = Path('cache/v17_prices/v17_prices_latest.parquet')
    
    if not cache_path.exists():
        raise FileNotFoundError(f"Price data not found at {cache_path}")
    
    prices = pd.read_parquet(cache_path)
    prices['date'] = pd.to_datetime(prices['date'])
    
    return prices


def run_reversal_backtest():
    """Run pure reversal strategy backtest."""
    
    logger.info("=" * 60)
    logger.info("🔄 V19.0 PHASE 1: PURE REVERSAL STRATEGY")
    logger.info("=" * 60)
    
    # Load data
    logger.info("\n📂 Loading price data...")
    prices = load_price_data()
    
    # Filter to liquid stocks (average daily volume > $1M)
    prices['dollar_volume'] = prices['close'] * prices['volume']
    avg_dv = prices.groupby('symbol')['dollar_volume'].mean()
    liquid_symbols = avg_dv[avg_dv > 1_000_000].index.tolist()
    prices = prices[prices['symbol'].isin(liquid_symbols)]
    
    logger.info(f"   Total symbols: {len(liquid_symbols)}")
    logger.info(f"   Date range: {prices['date'].min():%Y-%m-%d} to {prices['date'].max():%Y-%m-%d}")
    
    # Pivot to wide format for vectorized operations
    logger.info("\n📊 Preparing data matrices...")
    close_wide = prices.pivot(index='date', columns='symbol', values='close')
    
    # Calculate 5-day returns (lookback for ranking)
    ret_5d = close_wide.pct_change(5)
    
    # Calculate forward 5-day returns (for P&L)
    fwd_ret_5d = close_wide.pct_change(5).shift(-5)
    
    # Calculate daily returns for Sharpe calculation
    daily_ret = close_wide.pct_change(1)
    
    logger.info(f"   Data shape: {close_wide.shape}")
    
    # Strategy parameters
    N_LONG = 50   # Number of long positions
    N_SHORT = 0   # LONG-ONLY (shorts hurt in trending regime)
    REBAL_PERIOD = 5  # Rebalance every 5 days
    COST_BPS = 10  # Transaction cost in basis points
    
    # Additional filters for quality reversals
    MIN_DRAWDOWN = -0.10  # Stock must be down at least 10% from 20d high
    MAX_DRAWDOWN = -0.40  # But not more than 40% (distressed)
    
    logger.info(f"\n⚙️ Strategy Parameters:")
    logger.info(f"   Long positions: {N_LONG} (biggest 5d losers)")
    logger.info(f"   Mode: LONG-ONLY (shorts hurt in trending regimes)")
    logger.info(f"   Rebalance period: {REBAL_PERIOD} days")
    logger.info(f"   Transaction cost: {COST_BPS} bps round-trip")
    logger.info(f"   Drawdown filter: {MIN_DRAWDOWN:.0%} to {MAX_DRAWDOWN:.0%}")
    
    # Generate signals using vectorized ranking
    logger.info("\n🎯 Generating reversal signals...")
    
    # Calculate 20-day high for drawdown filter
    high_20d = close_wide.rolling(20).max()
    drawdown_from_high = (close_wide - high_20d) / high_20d
    
    # Filter: Only consider stocks with meaningful drawdown (not distressed)
    valid_for_reversal = (drawdown_from_high >= MAX_DRAWDOWN) & (drawdown_from_high <= MIN_DRAWDOWN)
    
    # Mask 5-day returns for invalid stocks
    ret_5d_filtered = ret_5d.where(valid_for_reversal)
    
    # Rank stocks cross-sectionally by 5-day return (ascending: lowest first)
    # Lower rank = bigger loser = LONG candidate
    ranks = ret_5d_filtered.rank(axis=1, pct=True, na_option='keep')
    
    # Create position matrix - LONG ONLY
    positions = pd.DataFrame(0.0, index=ranks.index, columns=ranks.columns)
    
    # Vectorized position assignment - Long bottom N_LONG only
    for date in ranks.index:
        valid_count = ranks.loc[date].notna().sum()
        if valid_count < N_LONG:
            continue
        
        long_thresh = N_LONG / valid_count
        positions.loc[date, ranks.loc[date] <= long_thresh] = 1.0
    
    # Only rebalance every REBAL_PERIOD days
    rebal_dates = positions.index[::REBAL_PERIOD]
    
    # Forward-fill positions between rebalance dates
    positions_rebal = positions.copy()
    positions_rebal.loc[~positions_rebal.index.isin(rebal_dates)] = np.nan
    positions_rebal = positions_rebal.ffill()
    
    # Equal weight within long leg (100% long-only)
    long_counts = (positions_rebal == 1).sum(axis=1)
    
    # Normalize to equal weight: 100% long
    weights = positions_rebal.copy()
    for date in weights.index:
        if long_counts[date] > 0:
            weights.loc[date, weights.loc[date] == 1] = 1.0 / long_counts[date]
    
    # Calculate strategy returns
    logger.info("\n📈 Running backtest...")
    
    # Daily strategy returns: sum of (weight * next-day return)
    strategy_daily = (weights.shift(1) * daily_ret).sum(axis=1)
    
    # Transaction costs: only on rebalance dates
    weight_changes = weights.diff().abs()
    turnover = weight_changes.sum(axis=1)
    
    # Apply cost only on rebalance days
    costs = pd.Series(0.0, index=strategy_daily.index)
    costs.loc[costs.index.isin(rebal_dates)] = turnover.loc[rebal_dates] * (COST_BPS / 10000)
    
    net_returns = strategy_daily - costs
    
    # Filter to test period (walk-forward: train on first 60%, test on last 40%)
    split_idx = int(len(net_returns) * 0.6)
    test_returns = net_returns.iloc[split_idx:]
    
    # Calculate metrics
    cumulative = (1 + test_returns).cumprod()
    total_return = cumulative.iloc[-1] - 1
    
    trading_days = len(test_returns)
    years = trading_days / 252
    
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    annual_vol = test_returns.std() * np.sqrt(252)
    sharpe = (test_returns.mean() * 252) / annual_vol if annual_vol > 0 else 0
    
    # Drawdown
    peak = cumulative.expanding().max()
    drawdown = (cumulative - peak) / peak
    max_dd = drawdown.min()
    
    # Win rate
    winning_days = (test_returns > 0).sum()
    total_days = (test_returns != 0).sum()
    win_rate = winning_days / total_days if total_days > 0 else 0
    
    # Average turnover
    avg_turnover = turnover.mean()
    
    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("📊 V19.0 PHASE 1: REVERSAL STRATEGY RESULTS")
    logger.info("=" * 60)
    
    logger.info(f"\n📈 Performance (Test Period - {trading_days} days):")
    logger.info(f"   Total Return:   {total_return:.1%}")
    logger.info(f"   CAGR:           {cagr:.1%}")
    logger.info(f"   Sharpe Ratio:   {sharpe:.2f}")
    logger.info(f"   Max Drawdown:   {max_dd:.1%}")
    logger.info(f"   Annual Vol:     {annual_vol:.1%}")
    logger.info(f"   Win Rate:       {win_rate:.1%}")
    logger.info(f"   Avg Turnover:   {avg_turnover:.1%} daily")
    
    # Validate against targets
    logger.info(f"\n🎯 Phase 1 Targets:")
    cagr_pass = cagr > 0.05
    logger.info(f"   CAGR > 5%:      {'✅ PASS' if cagr_pass else '❌ FAIL'} ({cagr:.1%})")
    
    sharpe_pass = sharpe > 0.5
    logger.info(f"   Sharpe > 0.5:   {'✅ PASS' if sharpe_pass else '❌ FAIL'} ({sharpe:.2f})")
    
    dd_pass = max_dd > -0.30
    logger.info(f"   MaxDD > -30%:   {'✅ PASS' if dd_pass else '❌ FAIL'} ({max_dd:.1%})")
    
    # Monthly returns breakdown
    logger.info("\n📅 Monthly Returns:")
    test_returns.index = pd.to_datetime(test_returns.index)
    monthly = test_returns.resample('M').apply(lambda x: (1+x).prod()-1)
    for date, ret in monthly.tail(6).items():
        logger.info(f"   {date:%Y-%m}: {ret:+.1%}")
    
    # Save results
    results_dir = Path('results/v19')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'phase': 'Phase 1: Pure Reversal',
        'total_return': float(total_return),
        'cagr': float(cagr),
        'sharpe': float(sharpe),
        'max_drawdown': float(max_dd),
        'annual_vol': float(annual_vol),
        'win_rate': float(win_rate),
        'trading_days': int(trading_days),
        'avg_turnover': float(avg_turnover),
        'n_long': N_LONG,
        'n_short': N_SHORT,
        'rebal_period': REBAL_PERIOD,
        'cost_bps': COST_BPS,
        'n_symbols': len(liquid_symbols),
        'targets_met': {
            'cagr_gt_5pct': bool(cagr_pass),
            'sharpe_gt_0.5': bool(sharpe_pass),
            'maxdd_gt_minus30': bool(dd_pass)
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open(results_dir / 'v19_reversal_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate report
    report = generate_report(results, monthly, cumulative)
    with open(results_dir / 'V19_REVERSAL_REPORT.md', 'w') as f:
        f.write(report)
    
    logger.info(f"\n💾 Results saved to {results_dir}")
    
    # Return for ensemble use
    return {
        'results': results,
        'daily_returns': test_returns,
        'cumulative': cumulative,
        'positions': weights
    }


def generate_report(results, monthly, cumulative):
    """Generate markdown report for Phase 1."""
    
    all_pass = all(results['targets_met'].values())
    
    report = f"""# V19.0 Phase 1: Pure Reversal Strategy Report

**Generated:** {results['timestamp']}

---

## Executive Summary

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| CAGR | {results['cagr']:.1%} | > 5% | {'✅' if results['targets_met']['cagr_gt_5pct'] else '❌'} |
| Sharpe | {results['sharpe']:.2f} | > 0.5 | {'✅' if results['targets_met']['sharpe_gt_0.5'] else '❌'} |
| Max Drawdown | {results['max_drawdown']:.1%} | > -30% | {'✅' if results['targets_met']['maxdd_gt_minus30'] else '❌'} |
| Win Rate | {results['win_rate']:.1%} | - | - |
| Annual Volatility | {results['annual_vol']:.1%} | - | - |

**Overall:** {'✅ PROCEED TO PHASE 2' if all_pass else '⚠️ REVIEW BEFORE PROCEEDING'}

---

## Strategy Logic

```
1. Calculate 5-day returns for all stocks
2. Filter: Only stocks down 10-40% from 20-day high (quality dip)
3. Rank filtered stocks cross-sectionally (worst to best)
4. LONG: Bottom {results['n_long']} performers (biggest losers)
5. Mode: LONG-ONLY (shorts removed - hurt in trending regimes)
6. Equal weight positions (100% invested)
7. Rebalance every {results['rebal_period']} days
8. Transaction cost: {results['cost_bps']} bps round-trip
```

---

## Backtest Details

| Parameter | Value |
|-----------|-------|
| Universe | {results['n_symbols']} liquid stocks |
| Test Period | {results['trading_days']} days (~{results['trading_days']/252:.1f} years) |
| Average Turnover | {results['avg_turnover']:.1%} daily |

---

## Monthly Returns (Last 6 Months)

| Month | Return |
|-------|--------|
"""
    
    for date, ret in monthly.tail(6).items():
        report += f"| {date:%Y-%m} | {ret:+.1%} |\n"
    
    report += f"""

---

## Equity Curve

```
Starting Value: $1.00
Ending Value:   ${cumulative.iloc[-1]:.2f}
Total Return:   {results['total_return']:.1%}
```

---

## Why Reversal Works

Based on V18 diagnostic:
- **reversal_5d** factor has IC = +0.061 (positive predictive power)
- All **momentum** factors have NEGATIVE IC (momentum fails in current regime)
- Current HMM regime: **LowVolMeanRevert** (42% of time)

The market is rewarding **mean reversion**, not **trend following**.

---

## Next Steps

{'**Phase 1 targets met!** Proceed to Phase 2: Mean Reversion Overlay.' if all_pass else '**Targets not fully met.** Consider adjusting parameters before Phase 2.'}

---

*Report generated by v19_reversal_strategy.py*
"""
    
    return report


if __name__ == "__main__":
    run_reversal_backtest()
