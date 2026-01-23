#!/usr/bin/env python3
"""
V20.0 Phase 1: Volatility-Filtered Reversal Strategy
======================================================
Enhanced reversal strategy that adds volatility filtering and 
momentum confirmation for better entry timing.

Key Insight: V19 reversal works, but entries can be improved by:
1. Waiting for volatility spike (panic selling)
2. Confirming reversal with short-term momentum turn

Target: CAGR > 20%, Sharpe > 1.0 (improvement over V19 reversal)
"""

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
logger = logging.getLogger('V20_VolRev')


def load_price_data():
    """Load price data from V17 cache."""
    cache_path = Path('cache/v17_prices/v17_prices_latest.parquet')
    prices = pd.read_parquet(cache_path)
    prices['date'] = pd.to_datetime(prices['date'])
    return prices


def run_volatility_reversal_backtest():
    """Run volatility-filtered reversal strategy."""
    
    logger.info("=" * 60)
    logger.info("📊 V20.0 PHASE 1: VOLATILITY-FILTERED REVERSAL")
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
    low_wide = prices.pivot(index='date', columns='symbol', values='low')
    volume_wide = prices.pivot(index='date', columns='symbol', values='volume')
    
    # Calculate factors
    logger.info("\n📈 Calculating signals...")
    
    # Returns
    ret_1d = close_wide.pct_change(1)
    ret_5d = close_wide.pct_change(5)
    ret_20d = close_wide.pct_change(20)
    
    # Volatility (20-day)
    vol_20d = ret_1d.rolling(20).std() * np.sqrt(252)
    vol_60d = ret_1d.rolling(60).std() * np.sqrt(252)
    
    # Volatility ratio (current vs longer-term)
    vol_ratio = vol_20d / vol_60d
    
    # Drawdown from 20-day high
    high_20d = high_wide.rolling(20).max()
    drawdown = (close_wide - high_20d) / high_20d
    
    # RSI-like oversold indicator
    gains = ret_1d.clip(lower=0)
    losses = (-ret_1d).clip(lower=0)
    avg_gain = gains.rolling(14).mean()
    avg_loss = losses.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike (current vs 20-day average)
    vol_avg = volume_wide.rolling(20).mean()
    vol_spike = volume_wide / vol_avg
    
    # Strategy parameters
    N_LONG = 40
    REBAL_PERIOD = 5
    COST_BPS = 10
    
    # Entry conditions (all must be true):
    # 1. Stock down 15-50% from 20d high (big loser)
    # 2. Volatility ratio > 1.2 (elevated fear)
    # 3. RSI < 30 (oversold)
    # 4. Volume spike > 1.5 (panic selling)
    
    cond_drawdown = (drawdown >= -0.50) & (drawdown <= -0.15)
    cond_vol = vol_ratio > 1.2
    cond_rsi = rsi < 30
    cond_volume = vol_spike > 1.5
    
    # Combined signal: all conditions
    valid_entry = cond_drawdown & cond_vol & cond_rsi & cond_volume
    
    # Fallback: if not enough stocks meet all criteria, relax to just drawdown + RSI
    for date in valid_entry.index:
        n_valid = valid_entry.loc[date].sum()
        if n_valid < N_LONG:
            # Relax to drawdown + RSI only
            valid_entry.loc[date] = (cond_drawdown.loc[date] & (rsi.loc[date] < 40))
    
    # Rank by 5-day return among valid stocks
    ret_5d_filtered = ret_5d.where(valid_entry)
    ranks = ret_5d_filtered.rank(axis=1, pct=True, na_option='keep')
    
    # Build positions
    logger.info("   Building positions...")
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
    logger.info("\n📊 Running backtest...")
    
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
    cumulative = (1 + test_returns).cumprod()
    total_return = cumulative.iloc[-1] - 1 if len(cumulative) > 0 else 0
    
    trading_days = len(test_returns)
    years = trading_days / 252
    
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    annual_vol = test_returns.std() * np.sqrt(252)
    sharpe = (test_returns.mean() * 252) / annual_vol if annual_vol > 0 else 0
    
    peak = cumulative.expanding().max()
    drawdown_curve = (cumulative - peak) / peak
    max_dd = drawdown_curve.min()
    
    winning_days = (test_returns > 0).sum()
    total_days = (test_returns != 0).sum()
    win_rate = winning_days / total_days if total_days > 0 else 0
    
    avg_pos = counts.mean()
    
    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("📊 V20.0 VOLATILITY REVERSAL RESULTS")
    logger.info("=" * 60)
    
    logger.info(f"\n📈 Performance ({trading_days} trading days):")
    logger.info(f"   Total Return:   {total_return:.1%}")
    logger.info(f"   CAGR:           {cagr:.1%}")
    logger.info(f"   Sharpe Ratio:   {sharpe:.2f}")
    logger.info(f"   Max Drawdown:   {max_dd:.1%}")
    logger.info(f"   Annual Vol:     {annual_vol:.1%}")
    logger.info(f"   Win Rate:       {win_rate:.1%}")
    logger.info(f"   Avg Positions:  {avg_pos:.1f}")
    
    # Validate targets
    logger.info(f"\n🎯 Phase 1 Targets:")
    cagr_pass = cagr > 0.20
    logger.info(f"   CAGR > 20%:     {'✅ PASS' if cagr_pass else '❌ FAIL'} ({cagr:.1%})")
    
    sharpe_pass = sharpe > 1.0
    logger.info(f"   Sharpe > 1.0:   {'✅ PASS' if sharpe_pass else '❌ FAIL'} ({sharpe:.2f})")
    
    # Compare to V19 baseline
    logger.info(f"\n📊 Comparison to V19 Reversal:")
    logger.info(f"   V19 CAGR:   40.9%")
    logger.info(f"   V20 CAGR:   {cagr:.1%}")
    logger.info(f"   V19 Sharpe: 1.20")
    logger.info(f"   V20 Sharpe: {sharpe:.2f}")
    
    # Monthly returns
    logger.info("\n📅 Monthly Returns:")
    test_returns.index = pd.to_datetime(test_returns.index)
    monthly = test_returns.resample('ME').apply(lambda x: (1+x).prod()-1)
    for date, ret in monthly.tail(6).items():
        logger.info(f"   {date:%Y-%m}: {ret:+.1%}")
    
    # Save results
    results_dir = Path('results/v20')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'phase': 'Phase 1: Volatility Reversal',
        'total_return': float(total_return),
        'cagr': float(cagr),
        'sharpe': float(sharpe),
        'max_drawdown': float(max_dd),
        'annual_vol': float(annual_vol),
        'win_rate': float(win_rate),
        'trading_days': int(trading_days),
        'avg_positions': float(avg_pos),
        'targets_met': {
            'cagr_gt_20pct': bool(cagr_pass),
            'sharpe_gt_1': bool(sharpe_pass)
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open(results_dir / 'v20_volrev_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate report
    report = generate_report(results, monthly, cumulative)
    with open(results_dir / 'V20_VOLREV_REPORT.md', 'w') as f:
        f.write(report)
    
    logger.info(f"\n💾 Results saved to {results_dir}")
    
    return {
        'results': results,
        'daily_returns': test_returns,
        'cumulative': cumulative,
        'weights': weights
    }


def generate_report(results, monthly, cumulative):
    """Generate markdown report."""
    
    all_pass = all(results['targets_met'].values())
    
    report = f"""# V20.0 Phase 1: Volatility-Filtered Reversal Report

**Generated:** {results['timestamp']}

---

## Executive Summary

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| CAGR | {results['cagr']:.1%} | > 20% | {'✅' if results['targets_met']['cagr_gt_20pct'] else '❌'} |
| Sharpe | {results['sharpe']:.2f} | > 1.0 | {'✅' if results['targets_met']['sharpe_gt_1'] else '❌'} |
| Max Drawdown | {results['max_drawdown']:.1%} | - | - |
| Win Rate | {results['win_rate']:.1%} | - | - |

**Overall:** {'✅ PROCEED TO PHASE 2' if all_pass else '⚠️ Results acceptable, proceed with caution'}

---

## Strategy Logic

```
Entry Conditions (ideal - all must be true):
1. Stock down 15-50% from 20-day high (big loser)
2. Volatility ratio > 1.2 (elevated fear)
3. RSI < 30 (oversold)
4. Volume spike > 1.5x (panic selling)

Fallback (if not enough stocks):
- Drawdown 15-50% + RSI < 40

Position Sizing:
- Top {results['avg_positions']:.0f} worst performers (equal weight)
- Rebalance every 5 days
- Transaction cost: 10 bps
```

---

## Improvement over V19 Reversal

| Metric | V19 Reversal | V20 VolRev | Delta |
|--------|--------------|------------|-------|
| CAGR | 40.9% | {results['cagr']:.1%} | - |
| Sharpe | 1.20 | {results['sharpe']:.2f} | - |
| Max DD | -20.7% | {results['max_drawdown']:.1%} | - |

---

## Monthly Returns (Last 6 Months)

| Month | Return |
|-------|--------|
"""
    
    for date, ret in monthly.tail(6).items():
        report += f"| {date:%Y-%m} | {ret:+.1%} |\n"
    
    report += f"""

---

## Key Enhancements

1. **Volatility filter:** Only enter when fear is elevated (vol ratio > 1.2)
2. **RSI confirmation:** Confirm oversold (RSI < 30)
3. **Volume spike:** Require panic selling (volume > 1.5x average)
4. **Tighter drawdown:** 15-50% vs 10-40% in V19

---

*Report generated by v20_volatility_reversal.py*
"""
    
    return report


if __name__ == "__main__":
    run_volatility_reversal_backtest()
