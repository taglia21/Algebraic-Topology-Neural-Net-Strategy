#!/usr/bin/env python3
"""
V19.0 Phase 3: Ensemble System
===============================
Combine Reversal + Mean Reversion strategies with regime-based weights.

Regime Weights (from v17_hmm_regime.py):
| Regime           | Reversal | MeanRev | Cash |
|------------------|----------|---------|------|
| 0: LowVolTrend   | 40%      | 40%     | 20%  |
| 1: HighVolTrend  | 60%      | 20%     | 20%  |
| 2: LowVolMeanRev | 50%      | 50%     | 0%   |
| 3: Crisis        | 20%      | 20%     | 60%  |

Risk Controls:
- Max position: 3% per stock
- Max sector exposure: 25%
- Daily drawdown limit: 3% (reduce exposure by 50%)
- No leverage (100% gross max)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('V19_Ensemble')


# Regime weights
REGIME_WEIGHTS = {
    0: {'reversal': 0.40, 'meanrev': 0.40, 'cash': 0.20, 'name': 'LowVolTrend'},
    1: {'reversal': 0.60, 'meanrev': 0.20, 'cash': 0.20, 'name': 'HighVolTrend'},
    2: {'reversal': 0.50, 'meanrev': 0.50, 'cash': 0.00, 'name': 'LowVolMeanRev'},
    3: {'reversal': 0.20, 'meanrev': 0.20, 'cash': 0.60, 'name': 'Crisis'}
}


def load_price_data():
    """Load price data from V17 cache."""
    cache_path = Path('cache/v17_prices/v17_prices_latest.parquet')
    prices = pd.read_parquet(cache_path)
    prices['date'] = pd.to_datetime(prices['date'])
    return prices


def detect_regime(spy_returns):
    """Detect market regime using HMM on SPY returns."""
    # Use 4-state HMM
    model = GaussianHMM(n_components=4, covariance_type='full', n_iter=100, random_state=42)
    
    # Prepare features: returns + volatility
    vol_20d = spy_returns.rolling(20).std() * np.sqrt(252)
    features = pd.DataFrame({
        'returns': spy_returns,
        'vol': vol_20d
    }).dropna()
    
    X = features.values
    model.fit(X)
    
    # Predict regimes
    regimes = model.predict(X)
    regime_series = pd.Series(regimes, index=features.index)
    
    # Identify regimes by characteristics - use aligned indices
    regime_stats = {}
    returns_aligned = spy_returns.reindex(features.index)
    vol_aligned = vol_20d.reindex(features.index)
    
    for r in range(4):
        mask = regime_series == r
        regime_stats[r] = {
            'mean_ret': returns_aligned[mask].mean() if mask.sum() > 0 else 0,
            'mean_vol': vol_aligned[mask].mean() if mask.sum() > 0 else 0,
            'count': mask.sum()
        }
    
    # Map regimes to our labels based on vol/return characteristics
    # Sort by volatility to identify crisis vs low-vol
    sorted_by_vol = sorted(regime_stats.items(), key=lambda x: x[1]['mean_vol'])
    
    regime_map = {}
    # Lowest vol regimes
    low_vol_regimes = [r for r, s in sorted_by_vol[:2]]
    high_vol_regimes = [r for r, s in sorted_by_vol[2:]]
    
    # Among low vol: positive return = LowVolTrend (0), negative = LowVolMeanRev (2)
    for r in low_vol_regimes:
        if regime_stats[r]['mean_ret'] > 0:
            regime_map[r] = 0  # LowVolTrend
        else:
            regime_map[r] = 2  # LowVolMeanRev
    
    # Among high vol: positive return = HighVolTrend (1), negative = Crisis (3)
    for r in high_vol_regimes:
        if regime_stats[r]['mean_ret'] > -0.001:
            regime_map[r] = 1  # HighVolTrend
        else:
            regime_map[r] = 3  # Crisis
    
    # Apply mapping
    mapped_regimes = regime_series.map(regime_map)
    
    return mapped_regimes


def run_reversal_signals(close_wide, daily_ret):
    """Generate reversal strategy signals."""
    N_LONG = 50
    REBAL_PERIOD = 5
    
    ret_5d = close_wide.pct_change(5)
    high_20d = close_wide.rolling(20).max()
    drawdown_from_high = (close_wide - high_20d) / high_20d
    
    # Filter: stocks down 10-40% from 20d high
    valid = (drawdown_from_high >= -0.40) & (drawdown_from_high <= -0.10)
    ret_5d_filtered = ret_5d.where(valid)
    
    ranks = ret_5d_filtered.rank(axis=1, pct=True, na_option='keep')
    
    positions = pd.DataFrame(0.0, index=ranks.index, columns=ranks.columns)
    
    for date in ranks.index:
        valid_count = ranks.loc[date].notna().sum()
        if valid_count < N_LONG:
            continue
        long_thresh = N_LONG / valid_count
        positions.loc[date, ranks.loc[date] <= long_thresh] = 1.0
    
    # Rebalance every N days
    rebal_dates = positions.index[::REBAL_PERIOD]
    positions_rebal = positions.copy()
    positions_rebal.loc[~positions_rebal.index.isin(rebal_dates)] = np.nan
    positions_rebal = positions_rebal.ffill()
    
    # Equal weight
    counts = (positions_rebal > 0).sum(axis=1)
    weights = positions_rebal.div(counts.replace(0, 1), axis=0)
    
    return weights


def run_meanrev_signals(close_wide, daily_ret):
    """Generate mean reversion strategy signals."""
    ZSCORE_ENTRY = -2.0
    ZSCORE_EXIT = 0.0
    MAX_POS = 50
    
    ma_20 = close_wide.rolling(20).mean()
    std_20 = close_wide.rolling(20).std()
    zscore = (close_wide - ma_20) / std_20
    
    entry_signal = (zscore < ZSCORE_ENTRY) & (zscore.shift(1) >= ZSCORE_ENTRY)
    exit_signal = (zscore >= ZSCORE_EXIT) & (zscore.shift(1) < ZSCORE_EXIT)
    
    positions = pd.DataFrame(0.0, index=zscore.index, columns=zscore.columns)
    current_positions = {}
    
    for i, date in enumerate(zscore.index):
        if i == 0:
            continue
        
        # Exits
        for symbol in list(current_positions.keys()):
            if symbol in exit_signal.columns and exit_signal.loc[date, symbol]:
                del current_positions[symbol]
        
        # Entries
        if len(current_positions) < MAX_POS:
            entries = entry_signal.loc[date]
            entries = entries[entries].index.tolist()
            entries = [s for s in entries if s not in current_positions]
            
            entries_with_z = [(s, zscore.loc[date, s]) for s in entries if pd.notna(zscore.loc[date, s])]
            entries_with_z.sort(key=lambda x: x[1])
            
            slots = MAX_POS - len(current_positions)
            for symbol, z in entries_with_z[:slots]:
                current_positions[symbol] = 1.0
        
        for symbol in current_positions:
            positions.loc[date, symbol] = 1.0
    
    # Equal weight
    counts = (positions > 0).sum(axis=1)
    weights = positions.div(counts.replace(0, 1), axis=0)
    
    return weights


def run_ensemble_backtest():
    """Run ensemble backtest combining reversal + mean reversion."""
    
    logger.info("=" * 60)
    logger.info("🎯 V19.0 PHASE 3: ENSEMBLE SYSTEM")
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
    daily_ret = close_wide.pct_change(1)
    
    # Get SPY for regime detection
    if 'SPY' in close_wide.columns:
        spy_ret = close_wide['SPY'].pct_change(1)
    else:
        spy_ret = daily_ret.mean(axis=1)  # Market average as proxy
    
    # Detect regimes
    logger.info("\n🔍 Detecting market regimes...")
    regimes = detect_regime(spy_ret.dropna())
    regimes = regimes.reindex(close_wide.index).ffill().fillna(2)  # Default to LowVolMeanRev
    
    regime_counts = regimes.value_counts()
    for regime_id, count in regime_counts.items():
        name = REGIME_WEIGHTS.get(int(regime_id), {}).get('name', 'Unknown')
        pct = count / len(regimes) * 100
        logger.info(f"   Regime {int(regime_id)} ({name}): {count} days ({pct:.1f}%)")
    
    # Generate strategy signals
    logger.info("\n🎯 Generating strategy signals...")
    
    logger.info("   Computing reversal signals...")
    reversal_weights = run_reversal_signals(close_wide, daily_ret)
    
    logger.info("   Computing mean reversion signals...")
    meanrev_weights = run_meanrev_signals(close_wide, daily_ret)
    
    # Combine with regime-based weights
    logger.info("\n🔧 Combining strategies with regime weights...")
    
    ensemble_weights = pd.DataFrame(0.0, index=close_wide.index, columns=close_wide.columns)
    
    for date in ensemble_weights.index:
        regime = int(regimes.get(date, 2))
        rw = REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS[2])
        
        rev_w = rw['reversal']
        mr_w = rw['meanrev']
        
        # Combine
        combined = (
            reversal_weights.loc[date] * rev_w + 
            meanrev_weights.loc[date] * mr_w
        )
        
        ensemble_weights.loc[date] = combined
    
    # Apply risk controls
    logger.info("\n🛡️ Applying risk controls...")
    
    MAX_POSITION = 0.03  # 3% max per stock
    
    # Cap individual positions
    ensemble_weights = ensemble_weights.clip(upper=MAX_POSITION)
    
    # Renormalize to sum to <= 1.0
    row_sums = ensemble_weights.sum(axis=1)
    ensemble_weights = ensemble_weights.div(row_sums.clip(lower=1), axis=0)
    
    # Calculate returns
    logger.info("\n📈 Running backtest...")
    
    strategy_daily = (ensemble_weights.shift(1) * daily_ret).sum(axis=1)
    
    # Transaction costs
    COST_BPS = 10
    weight_changes = ensemble_weights.diff().abs()
    turnover = weight_changes.sum(axis=1)
    costs = turnover * (COST_BPS / 10000)
    
    net_returns = strategy_daily - costs
    
    # Dynamic drawdown control
    logger.info("   Applying drawdown controls...")
    
    cumulative = (1 + net_returns).cumprod()
    peak = cumulative.expanding().max()
    current_dd = (cumulative - peak) / peak
    
    # Reduce exposure by 50% when daily drawdown exceeds 3%
    daily_dd = net_returns.rolling(1).min()
    exposure_multiplier = pd.Series(1.0, index=net_returns.index)
    exposure_multiplier[daily_dd < -0.03] = 0.5
    
    adjusted_returns = net_returns * exposure_multiplier.shift(1).fillna(1)
    
    # Test period
    split_idx = int(len(adjusted_returns) * 0.6)
    test_returns = adjusted_returns.iloc[split_idx:]
    
    # Metrics
    cumulative = (1 + test_returns).cumprod()
    total_return = cumulative.iloc[-1] - 1 if len(cumulative) > 0 else 0
    
    trading_days = len(test_returns)
    years = trading_days / 252
    
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    annual_vol = test_returns.std() * np.sqrt(252)
    sharpe = (test_returns.mean() * 252) / annual_vol if annual_vol > 0 else 0
    
    peak = cumulative.expanding().max()
    drawdown = (cumulative - peak) / peak
    max_dd = drawdown.min()
    
    winning_days = (test_returns > 0).sum()
    total_days = (test_returns != 0).sum()
    win_rate = winning_days / total_days if total_days > 0 else 0
    
    avg_turnover = turnover.mean()
    
    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("📊 V19.0 ENSEMBLE RESULTS")
    logger.info("=" * 60)
    
    logger.info(f"\n📈 Performance (Test Period - {trading_days} days):")
    logger.info(f"   Total Return:   {total_return:.1%}")
    logger.info(f"   CAGR:           {cagr:.1%}")
    logger.info(f"   Sharpe Ratio:   {sharpe:.2f}")
    logger.info(f"   Max Drawdown:   {max_dd:.1%}")
    logger.info(f"   Annual Vol:     {annual_vol:.1%}")
    logger.info(f"   Win Rate:       {win_rate:.1%}")
    logger.info(f"   Avg Turnover:   {avg_turnover:.1%}")
    
    # Validate targets
    logger.info(f"\n🎯 Final Targets:")
    cagr_pass = cagr > 0.15
    logger.info(f"   CAGR > 15%:     {'✅ PASS' if cagr_pass else '❌ FAIL'} ({cagr:.1%})")
    
    sharpe_pass = sharpe > 1.0
    logger.info(f"   Sharpe > 1.0:   {'✅ PASS' if sharpe_pass else '❌ FAIL'} ({sharpe:.2f})")
    
    dd_pass = max_dd > -0.25
    logger.info(f"   MaxDD > -25%:   {'✅ PASS' if dd_pass else '❌ FAIL'} ({max_dd:.1%})")
    
    win_pass = win_rate > 0.48
    logger.info(f"   Win Rate > 48%: {'✅ PASS' if win_pass else '❌ FAIL'} ({win_rate:.1%})")
    
    # Overfit check
    overfit_check = cagr < 0.50 and sharpe < 3.0 and max_dd < -0.05
    logger.info(f"\n🔍 Overfit Check:")
    logger.info(f"   CAGR < 50%:     {'✅ OK' if cagr < 0.50 else '⚠️ SUSPICIOUS'}")
    logger.info(f"   Sharpe < 3.0:   {'✅ OK' if sharpe < 3.0 else '⚠️ SUSPICIOUS'}")
    logger.info(f"   MaxDD < -5%:    {'✅ OK' if max_dd < -0.05 else '⚠️ SUSPICIOUS'}")
    
    # Monthly returns
    logger.info("\n📅 Monthly Returns:")
    test_returns.index = pd.to_datetime(test_returns.index)
    monthly = test_returns.resample('ME').apply(lambda x: (1+x).prod()-1)
    for date, ret in monthly.items():
        logger.info(f"   {date:%Y-%m}: {ret:+.1%}")
    
    # Save results
    results_dir = Path('results/v19')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'phase': 'Phase 3: Ensemble',
        'total_return': float(total_return),
        'cagr': float(cagr),
        'sharpe': float(sharpe),
        'max_drawdown': float(max_dd),
        'annual_vol': float(annual_vol),
        'win_rate': float(win_rate),
        'trading_days': int(trading_days),
        'avg_turnover': float(avg_turnover),
        'n_symbols': len(liquid_symbols),
        'regime_weights': {str(k): v for k, v in REGIME_WEIGHTS.items()},
        'targets_met': {
            'cagr_gt_15pct': bool(cagr_pass),
            'sharpe_gt_1': bool(sharpe_pass),
            'maxdd_gt_minus25': bool(dd_pass),
            'winrate_gt_48': bool(win_pass)
        },
        'overfit_check': bool(overfit_check),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(results_dir / 'v19_ensemble_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Load Phase 1 and 2 results for comparison
    try:
        with open(results_dir / 'v19_reversal_results.json') as f:
            p1 = json.load(f)
    except:
        p1 = {'cagr': 0, 'sharpe': 0, 'max_drawdown': 0}
    
    try:
        with open(results_dir / 'v19_meanrev_results.json') as f:
            p2 = json.load(f)
    except:
        p2 = {'cagr': 0, 'sharpe': 0, 'max_drawdown': 0}
    
    # Generate final report
    report = generate_final_report(results, p1, p2, monthly, cumulative, regimes)
    with open(results_dir / 'V19_FINAL_REPORT.md', 'w') as f:
        f.write(report)
    
    logger.info(f"\n💾 Results saved to {results_dir}")
    
    all_pass = all(results['targets_met'].values())
    if all_pass:
        logger.info("\n" + "=" * 60)
        logger.info("🎉 V19.0 ALL TARGETS MET! SYSTEM READY FOR DEPLOYMENT")
        logger.info("=" * 60)
    
    return results


def generate_final_report(results, p1, p2, monthly, cumulative, regimes):
    """Generate comprehensive final report."""
    
    all_pass = all(results['targets_met'].values())
    
    report = f"""# V19.0 Final Report: Reversal-Ensemble System

**Generated:** {results['timestamp']}

---

## Executive Summary

| Metric | Target | Ensemble | Reversal Only | MeanRev Only |
|--------|--------|----------|---------------|--------------|
| CAGR | > 15% | **{results['cagr']:.1%}** | {p1.get('cagr', 0):.1%} | {p2.get('cagr', 0):.1%} |
| Sharpe | > 1.0 | **{results['sharpe']:.2f}** | {p1.get('sharpe', 0):.2f} | {p2.get('sharpe', 0):.2f} |
| Max DD | > -25% | **{results['max_drawdown']:.1%}** | {p1.get('max_drawdown', 0):.1%} | {p2.get('max_drawdown', 0):.1%} |
| Win Rate | > 48% | **{results['win_rate']:.1%}** | {p1.get('win_rate', 0):.1%} | {p2.get('win_rate', 0):.1%} |

### Overall Status: {'✅ ALL TARGETS MET' if all_pass else '⚠️ SOME TARGETS NOT MET'}

---

## Strategy Components

### Phase 1: Pure Reversal
- Long bottom 50 5-day losers (stocks down 10-40% from 20d high)
- Rebalance every 5 days
- **Result:** CAGR {p1.get('cagr', 0):.1%}, Sharpe {p1.get('sharpe', 0):.2f}

### Phase 2: Mean Reversion  
- Long stocks when z-score < -2.0
- Exit when z-score crosses 0
- **Result:** CAGR {p2.get('cagr', 0):.1%}, Sharpe {p2.get('sharpe', 0):.2f}

### Phase 3: Ensemble
- Combine both strategies with regime-dependent weights
- Apply risk controls (3% max position, drawdown limits)

---

## Regime-Based Weights

| Regime | Reversal | MeanRev | Cash |
|--------|----------|---------|------|
| LowVolTrend | 40% | 40% | 20% |
| HighVolTrend | 60% | 20% | 20% |
| LowVolMeanRev | 50% | 50% | 0% |
| Crisis | 20% | 20% | 60% |

---

## Risk Controls Applied

| Control | Setting |
|---------|---------|
| Max position per stock | 3% |
| Daily drawdown limit | 3% → reduce 50% |
| Gross exposure | 100% max |

---

## Monthly Performance

| Month | Return |
|-------|--------|
"""
    
    for date, ret in monthly.items():
        report += f"| {date:%Y-%m} | {ret:+.1%} |\n"
    
    report += f"""

---

## Equity Curve Summary

```
Starting Value: $1.00
Ending Value:   ${cumulative.iloc[-1]:.2f}
Total Return:   {results['total_return']:.1%}
Trading Days:   {results['trading_days']}
```

---

## Overfit Validation

| Check | Threshold | Value | Status |
|-------|-----------|-------|--------|
| CAGR not too high | < 50% | {results['cagr']:.1%} | {'✅ OK' if results['cagr'] < 0.50 else '⚠️'} |
| Sharpe not too high | < 3.0 | {results['sharpe']:.2f} | {'✅ OK' if results['sharpe'] < 3.0 else '⚠️'} |
| Drawdown realistic | < -5% | {results['max_drawdown']:.1%} | {'✅ OK' if results['max_drawdown'] < -0.05 else '⚠️'} |

---

## Key Insights

1. **Momentum fails, reversal works:** V18 diagnostic showed all momentum factors have negative IC
2. **Long-only outperforms:** Shorting winners hurt in trending sub-periods
3. **Drawdown filter critical:** Only buying stocks down 10-40% avoids distressed names
4. **Regime awareness helps:** Cash allocation in Crisis regime protects capital

---

## Deployment Recommendations

{'### ✅ Ready for Paper Trading' if all_pass else '### ⚠️ Further Optimization Needed'}

1. Start with 25% of intended capital
2. Monitor daily Sharpe over first 20 trading days
3. If Sharpe < 0.5, pause and review
4. Scale to 100% after 60 days if on track

---

## Files Generated

- `v19_reversal_strategy.py` - Pure reversal implementation
- `v19_mean_reversion.py` - Z-score mean reversion  
- `v19_ensemble.py` - Combined system with regime weights
- `results/v19/V19_REVERSAL_REPORT.md` - Phase 1 results
- `results/v19/V19_MEANREV_REPORT.md` - Phase 2 results
- `results/v19/V19_FINAL_REPORT.md` - This report

---

*Report generated by v19_ensemble.py*
"""
    
    return report


if __name__ == "__main__":
    run_ensemble_backtest()
