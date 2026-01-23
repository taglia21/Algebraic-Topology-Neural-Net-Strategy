#!/usr/bin/env python3
"""
V19.0 Phase 2: Mean Reversion Strategy
=======================================
Use z-score to identify oversold/overbought conditions.

Logic:
- Calculate z-score: (price - 20d_mean) / 20d_std for each stock
- LONG: Stocks with z-score < -2.0 (oversold)
- SHORT: Stocks with z-score > +2.0 (overbought) [OPTIONAL]
- Exit when z-score crosses 0

Based on diagnostic finding: overbought_oversold has IC = +0.071
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
logger = logging.getLogger('V19_MeanRev')


def load_price_data():
    """Load price data from V17 cache."""
    cache_path = Path('cache/v17_prices/v17_prices_latest.parquet')
    
    if not cache_path.exists():
        raise FileNotFoundError(f"Price data not found at {cache_path}")
    
    prices = pd.read_parquet(cache_path)
    prices['date'] = pd.to_datetime(prices['date'])
    
    return prices


def run_meanrev_backtest():
    """Run mean reversion strategy backtest."""
    
    logger.info("=" * 60)
    logger.info("📊 V19.0 PHASE 2: MEAN REVERSION STRATEGY")
    logger.info("=" * 60)
    
    # Load data
    logger.info("\n📂 Loading price data...")
    prices = load_price_data()
    
    # Filter to liquid stocks
    prices['dollar_volume'] = prices['close'] * prices['volume']
    avg_dv = prices.groupby('symbol')['dollar_volume'].mean()
    liquid_symbols = avg_dv[avg_dv > 1_000_000].index.tolist()
    prices = prices[prices['symbol'].isin(liquid_symbols)]
    
    logger.info(f"   Total symbols: {len(liquid_symbols)}")
    logger.info(f"   Date range: {prices['date'].min():%Y-%m-%d} to {prices['date'].max():%Y-%m-%d}")
    
    # Pivot to wide format
    logger.info("\n📊 Preparing data matrices...")
    close_wide = prices.pivot(index='date', columns='symbol', values='close')
    
    # Calculate z-score
    ma_20 = close_wide.rolling(20).mean()
    std_20 = close_wide.rolling(20).std()
    zscore = (close_wide - ma_20) / std_20
    
    # Daily returns for P&L
    daily_ret = close_wide.pct_change(1)
    
    logger.info(f"   Data shape: {close_wide.shape}")
    
    # Strategy parameters
    ZSCORE_ENTRY_LONG = -2.0   # Enter long when z < -2
    ZSCORE_EXIT = 0.0          # Exit when z crosses 0
    MAX_POSITIONS = 50         # Max concurrent positions
    COST_BPS = 10              # Transaction cost
    LONG_ONLY = True           # Skip shorts for now
    
    logger.info(f"\n⚙️ Strategy Parameters:")
    logger.info(f"   Entry threshold: z-score < {ZSCORE_ENTRY_LONG}")
    logger.info(f"   Exit threshold: z-score > {ZSCORE_EXIT}")
    logger.info(f"   Max positions: {MAX_POSITIONS}")
    logger.info(f"   Mode: {'LONG-ONLY' if LONG_ONLY else 'Long/Short'}")
    logger.info(f"   Transaction cost: {COST_BPS} bps round-trip")
    
    # Generate signals
    logger.info("\n🎯 Generating mean reversion signals...")
    
    # Entry: z-score < -2 and was above yesterday (fresh cross down)
    entry_signal = (zscore < ZSCORE_ENTRY_LONG) & (zscore.shift(1) >= ZSCORE_ENTRY_LONG)
    
    # Exit: z-score crosses above 0
    exit_signal = (zscore >= ZSCORE_EXIT) & (zscore.shift(1) < ZSCORE_EXIT)
    
    # Build position matrix with holding logic
    positions = pd.DataFrame(0.0, index=zscore.index, columns=zscore.columns)
    
    # Track positions day by day (need loop for holding logic)
    current_positions = {}
    
    for i, date in enumerate(zscore.index):
        if i == 0:
            continue
            
        # Check exits first
        exits_today = []
        for symbol in current_positions:
            if exit_signal.loc[date, symbol] if symbol in exit_signal.columns else False:
                exits_today.append(symbol)
        
        for symbol in exits_today:
            del current_positions[symbol]
        
        # Check entries (limit to MAX_POSITIONS)
        if len(current_positions) < MAX_POSITIONS:
            # Get all entry signals for today, sorted by z-score (most oversold first)
            entries = entry_signal.loc[date]
            entries = entries[entries].index.tolist()
            
            # Filter out already held
            entries = [s for s in entries if s not in current_positions]
            
            # Sort by z-score (most negative first)
            entries_with_z = [(s, zscore.loc[date, s]) for s in entries if pd.notna(zscore.loc[date, s])]
            entries_with_z.sort(key=lambda x: x[1])
            
            # Take up to MAX_POSITIONS - current
            slots = MAX_POSITIONS - len(current_positions)
            for symbol, z in entries_with_z[:slots]:
                current_positions[symbol] = 1.0
        
        # Set positions for today
        for symbol, weight in current_positions.items():
            positions.loc[date, symbol] = weight
    
    # Equal weight normalization
    position_counts = (positions != 0).sum(axis=1)
    weights = positions.div(position_counts.replace(0, 1), axis=0)
    
    # Calculate strategy returns
    logger.info("\n📈 Running backtest...")
    
    strategy_daily = (weights.shift(1) * daily_ret).sum(axis=1)
    
    # Transaction costs
    weight_changes = weights.diff().abs()
    turnover = weight_changes.sum(axis=1)
    costs = turnover * (COST_BPS / 10000)
    
    net_returns = strategy_daily - costs
    
    # Filter to test period
    split_idx = int(len(net_returns) * 0.6)
    test_returns = net_returns.iloc[split_idx:]
    
    # Calculate metrics
    cumulative = (1 + test_returns).cumprod()
    total_return = cumulative.iloc[-1] - 1 if len(cumulative) > 0 else 0
    
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
    
    # Average positions
    avg_positions = position_counts.mean()
    
    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("📊 V19.0 PHASE 2: MEAN REVERSION RESULTS")
    logger.info("=" * 60)
    
    logger.info(f"\n📈 Performance (Test Period - {trading_days} days):")
    logger.info(f"   Total Return:   {total_return:.1%}")
    logger.info(f"   CAGR:           {cagr:.1%}")
    logger.info(f"   Sharpe Ratio:   {sharpe:.2f}")
    logger.info(f"   Max Drawdown:   {max_dd:.1%}")
    logger.info(f"   Annual Vol:     {annual_vol:.1%}")
    logger.info(f"   Win Rate:       {win_rate:.1%}")
    logger.info(f"   Avg Positions:  {avg_positions:.1f}")
    
    # Validate targets
    logger.info(f"\n🎯 Phase 2 Targets:")
    cagr_pass = cagr > 0
    logger.info(f"   CAGR > 0%:      {'✅ PASS' if cagr_pass else '❌ FAIL'} ({cagr:.1%})")
    
    sharpe_pass = sharpe > 0
    logger.info(f"   Sharpe > 0:     {'✅ PASS' if sharpe_pass else '❌ FAIL'} ({sharpe:.2f})")
    
    # Monthly returns
    logger.info("\n📅 Monthly Returns:")
    test_returns.index = pd.to_datetime(test_returns.index)
    monthly = test_returns.resample('ME').apply(lambda x: (1+x).prod()-1)
    for date, ret in monthly.tail(6).items():
        logger.info(f"   {date:%Y-%m}: {ret:+.1%}")
    
    # Save results
    results_dir = Path('results/v19')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'phase': 'Phase 2: Mean Reversion',
        'total_return': float(total_return),
        'cagr': float(cagr),
        'sharpe': float(sharpe),
        'max_drawdown': float(max_dd),
        'annual_vol': float(annual_vol),
        'win_rate': float(win_rate),
        'trading_days': int(trading_days),
        'avg_positions': float(avg_positions),
        'zscore_entry': ZSCORE_ENTRY_LONG,
        'zscore_exit': ZSCORE_EXIT,
        'max_positions': MAX_POSITIONS,
        'cost_bps': COST_BPS,
        'n_symbols': len(liquid_symbols),
        'targets_met': {
            'cagr_positive': bool(cagr_pass),
            'sharpe_positive': bool(sharpe_pass)
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open(results_dir / 'v19_meanrev_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate report
    report = generate_report(results, monthly, cumulative)
    with open(results_dir / 'V19_MEANREV_REPORT.md', 'w') as f:
        f.write(report)
    
    logger.info(f"\n💾 Results saved to {results_dir}")
    
    return {
        'results': results,
        'daily_returns': test_returns,
        'cumulative': cumulative,
        'positions': weights
    }


def generate_report(results, monthly, cumulative):
    """Generate markdown report for Phase 2."""
    
    all_pass = all(results['targets_met'].values())
    
    report = f"""# V19.0 Phase 2: Mean Reversion Strategy Report

**Generated:** {results['timestamp']}

---

## Executive Summary

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| CAGR | {results['cagr']:.1%} | > 0% | {'✅' if results['targets_met']['cagr_positive'] else '❌'} |
| Sharpe | {results['sharpe']:.2f} | > 0 | {'✅' if results['targets_met']['sharpe_positive'] else '❌'} |
| Max Drawdown | {results['max_drawdown']:.1%} | - | - |
| Win Rate | {results['win_rate']:.1%} | - | - |
| Annual Volatility | {results['annual_vol']:.1%} | - | - |

**Overall:** {'✅ PROCEED TO PHASE 3' if all_pass else '⚠️ REVIEW BEFORE PROCEEDING'}

---

## Strategy Logic

```
1. Calculate z-score = (price - 20d_mean) / 20d_std
2. ENTRY: z-score crosses below {results['zscore_entry']}
3. EXIT: z-score crosses above {results['zscore_exit']}
4. Max {results['max_positions']} concurrent positions
5. Equal weight across all positions
6. Transaction cost: {results['cost_bps']} bps round-trip
```

---

## Backtest Details

| Parameter | Value |
|-----------|-------|
| Universe | {results['n_symbols']} liquid stocks |
| Test Period | {results['trading_days']} days (~{results['trading_days']/252:.1f} years) |
| Average Positions | {results['avg_positions']:.1f} |

---

## Monthly Returns (Last 6 Months)

| Month | Return |
|-------|--------|
"""
    
    for date, ret in monthly.tail(6).items():
        report += f"| {date:%Y-%m} | {ret:+.1%} |\n"
    
    report += f"""

---

## Why Mean Reversion Works

Based on V18 diagnostic:
- **overbought_oversold** factor has IC = +0.071 (positive predictive power)
- Current HMM regime: **LowVolMeanRevert** (42% of time)
- Z-score extremes tend to revert to mean

---

## Comparison to Phase 1 (Reversal)

| Metric | Reversal | Mean Reversion |
|--------|----------|----------------|
| CAGR | See Phase 1 | {results['cagr']:.1%} |
| Sharpe | See Phase 1 | {results['sharpe']:.2f} |
| Max DD | See Phase 1 | {results['max_drawdown']:.1%} |

---

## Next Steps

{'**Phase 2 targets met!** Proceed to Phase 3: Ensemble Combination.' if all_pass else '**Targets not fully met.** Consider adjusting z-score thresholds.'}

---

*Report generated by v19_mean_reversion.py*
"""
    
    return report


if __name__ == "__main__":
    run_meanrev_backtest()
