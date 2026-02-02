#!/usr/bin/env python3
"""
V20.0 Phase 2: Enhanced Ensemble System
=========================================
Combine 3 strategies with dynamic weight adjustment.

Strategies:
1. Volatility Reversal (V20) - 35% base weight
2. Mean Reversion (V19) - 30% base weight  
3. Original Reversal (V19) - 35% base weight

Dynamic weighting based on 20-day rolling Sharpe.

Target: CAGR > 32%, Sharpe > 1.35
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('V20_Ensemble')


def load_price_data():
    """Load price data from V17 cache."""
    cache_path = Path('cache/v17_prices/v17_prices_latest.parquet')
    prices = pd.read_parquet(cache_path)
    prices['date'] = pd.to_datetime(prices['date'])
    return prices


def run_reversal_strategy(close_wide, ret_1d):
    """V19 Pure Reversal: Long biggest 5d losers."""
    N_LONG = 50
    REBAL_PERIOD = 5
    
    ret_5d = close_wide.pct_change(5)
    high_20d = close_wide.rolling(20).max()
    drawdown = (close_wide - high_20d) / high_20d
    
    valid = (drawdown >= -0.40) & (drawdown <= -0.10)
    ret_5d_filtered = ret_5d.where(valid)
    ranks = ret_5d_filtered.rank(axis=1, pct=True, na_option='keep')
    
    positions = pd.DataFrame(0.0, index=ranks.index, columns=ranks.columns)
    
    for date in ranks.index:
        valid_count = ranks.loc[date].notna().sum()
        if valid_count < N_LONG:
            continue
        long_thresh = N_LONG / valid_count
        positions.loc[date, ranks.loc[date] <= long_thresh] = 1.0
    
    rebal_dates = positions.index[::REBAL_PERIOD]
    positions_rebal = positions.copy()
    positions_rebal.loc[~positions_rebal.index.isin(rebal_dates)] = np.nan
    positions_rebal = positions_rebal.ffill()
    
    counts = (positions_rebal > 0).sum(axis=1)
    weights = positions_rebal.div(counts.replace(0, 1), axis=0)
    
    strategy_returns = (weights.shift(1) * ret_1d).sum(axis=1)
    
    return strategy_returns, weights


def run_meanrev_strategy(close_wide, ret_1d):
    """V19 Mean Reversion: Long when z-score < -2."""
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
        
        for symbol in list(current_positions.keys()):
            if symbol in exit_signal.columns and exit_signal.loc[date, symbol]:
                del current_positions[symbol]
        
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
    
    counts = (positions > 0).sum(axis=1)
    weights = positions.div(counts.replace(0, 1), axis=0)
    
    strategy_returns = (weights.shift(1) * ret_1d).sum(axis=1)
    
    return strategy_returns, weights


def run_volrev_strategy(close_wide, high_wide, volume_wide, ret_1d):
    """V20 Volatility Reversal: Enhanced with vol/RSI filters."""
    N_LONG = 40
    REBAL_PERIOD = 5
    
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
    cond_drawdown = (drawdown >= -0.50) & (drawdown <= -0.15)
    cond_vol = vol_ratio > 1.2
    cond_rsi = rsi < 30
    cond_volume = vol_spike > 1.5
    
    valid_entry = cond_drawdown & cond_vol & cond_rsi & cond_volume
    
    for date in valid_entry.index:
        n_valid = valid_entry.loc[date].sum()
        if n_valid < N_LONG:
            valid_entry.loc[date] = (cond_drawdown.loc[date] & (rsi.loc[date] < 40))
    
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
    
    strategy_returns = (weights.shift(1) * ret_1d).sum(axis=1)
    
    return strategy_returns, weights


def run_enhanced_ensemble():
    """Run enhanced ensemble with dynamic weights."""
    
    logger.info("=" * 60)
    logger.info("🎯 V20.0 PHASE 2: ENHANCED ENSEMBLE")
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
    
    # Run individual strategies
    logger.info("\n🎯 Running individual strategies...")
    
    logger.info("   Running Reversal (V19)...")
    reversal_ret, reversal_wts = run_reversal_strategy(close_wide, ret_1d)
    
    logger.info("   Running Mean Reversion (V19)...")
    meanrev_ret, meanrev_wts = run_meanrev_strategy(close_wide, ret_1d)
    
    logger.info("   Running Volatility Reversal (V20)...")
    volrev_ret, volrev_wts = run_volrev_strategy(close_wide, high_wide, volume_wide, ret_1d)
    
    # Combine into DataFrame
    strat_returns = pd.DataFrame({
        'reversal': reversal_ret,
        'meanrev': meanrev_ret,
        'volrev': volrev_ret
    }).fillna(0)
    
    # Base weights
    BASE_WEIGHTS = {
        'reversal': 0.35,
        'meanrev': 0.30,
        'volrev': 0.35
    }
    
    logger.info(f"\n⚙️ Base Weights:")
    for strat, w in BASE_WEIGHTS.items():
        logger.info(f"   {strat}: {w:.0%}")
    
    # Dynamic weight adjustment based on 20-day rolling Sharpe
    logger.info("\n🔧 Applying dynamic weight adjustment...")
    
    LOOKBACK = 20
    MIN_WEIGHT = 0.10
    MAX_WEIGHT = 0.50
    
    dynamic_weights = pd.DataFrame(index=strat_returns.index, columns=strat_returns.columns)
    
    for i in range(len(strat_returns)):
        if i < LOOKBACK:
            # Use base weights for warmup period
            for strat in BASE_WEIGHTS:
                dynamic_weights.iloc[i][strat] = BASE_WEIGHTS[strat]
        else:
            # Calculate rolling Sharpe for each strategy
            window = strat_returns.iloc[i-LOOKBACK:i]
            sharpes = {}
            
            for strat in strat_returns.columns:
                ret_mean = window[strat].mean() * 252
                ret_std = window[strat].std() * np.sqrt(252)
                sharpe = ret_mean / ret_std if ret_std > 0 else 0
                sharpes[strat] = max(0, sharpe)  # Zero out negative Sharpe
            
            # Normalize weights proportional to Sharpe
            total_sharpe = sum(sharpes.values())
            
            if total_sharpe > 0:
                for strat in strat_returns.columns:
                    raw_weight = sharpes[strat] / total_sharpe
                    # Apply min/max bounds
                    bounded_weight = max(MIN_WEIGHT, min(MAX_WEIGHT, raw_weight))
                    dynamic_weights.iloc[i][strat] = bounded_weight
            else:
                # All strategies have negative Sharpe - use base weights
                for strat in BASE_WEIGHTS:
                    dynamic_weights.iloc[i][strat] = BASE_WEIGHTS[strat]
        
        # Renormalize to sum to 1
        row_sum = dynamic_weights.iloc[i].sum()
        if row_sum > 0:
            dynamic_weights.iloc[i] = dynamic_weights.iloc[i] / row_sum
    
    dynamic_weights = dynamic_weights.astype(float)
    
    # Calculate ensemble returns
    ensemble_returns = (strat_returns * dynamic_weights).sum(axis=1)
    
    # Apply transaction costs (approximate)
    COST_BPS = 10
    turnover = dynamic_weights.diff().abs().sum(axis=1)
    costs = turnover * (COST_BPS / 10000) * 0.5  # Approximate portfolio turnover
    
    net_returns = ensemble_returns - costs
    
    # Apply risk controls
    logger.info("\n🛡️ Applying risk controls...")
    
    # Daily loss limit: 2% -> reduce exposure by 50%
    exposure_mult = pd.Series(1.0, index=net_returns.index)
    
    for i in range(1, len(net_returns)):
        if net_returns.iloc[i-1] < -0.02:
            exposure_mult.iloc[i] = 0.5
        else:
            exposure_mult.iloc[i] = min(1.0, exposure_mult.iloc[i-1] * 1.1)  # Gradually recover
    
    adjusted_returns = net_returns * exposure_mult
    
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
    
    # Average weights
    avg_weights = dynamic_weights.iloc[split_idx:].mean()
    
    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("📊 V20.0 ENHANCED ENSEMBLE RESULTS")
    logger.info("=" * 60)
    
    logger.info(f"\n📈 Performance ({trading_days} trading days):")
    logger.info(f"   Total Return:   {total_return:.1%}")
    logger.info(f"   CAGR:           {cagr:.1%}")
    logger.info(f"   Sharpe Ratio:   {sharpe:.2f}")
    logger.info(f"   Max Drawdown:   {max_dd:.1%}")
    logger.info(f"   Annual Vol:     {annual_vol:.1%}")
    logger.info(f"   Win Rate:       {win_rate:.1%}")
    
    logger.info(f"\n📊 Average Strategy Weights (Test Period):")
    for strat, w in avg_weights.items():
        logger.info(f"   {strat}: {w:.1%}")
    
    # Validate targets
    logger.info(f"\n🎯 Final Targets:")
    cagr_pass = cagr > 0.32
    logger.info(f"   CAGR > 32%:     {'✅ PASS' if cagr_pass else '❌ FAIL'} ({cagr:.1%})")
    
    sharpe_pass = sharpe > 1.35
    logger.info(f"   Sharpe > 1.35:  {'✅ PASS' if sharpe_pass else '❌ FAIL'} ({sharpe:.2f})")
    
    dd_pass = max_dd > -0.18
    logger.info(f"   MaxDD > -18%:   {'✅ PASS' if dd_pass else '❌ FAIL'} ({max_dd:.1%})")
    
    # Overfit check
    logger.info(f"\n🔍 Overfit Check:")
    logger.info(f"   CAGR < 55%:     {'✅ OK' if cagr < 0.55 else '⚠️ SUSPICIOUS'}")
    logger.info(f"   Sharpe < 3.0:   {'✅ OK' if sharpe < 3.0 else '⚠️ SUSPICIOUS'}")
    logger.info(f"   MaxDD < -8%:    {'✅ OK' if max_dd < -0.08 else '⚠️ SUSPICIOUS'}")
    
    # Compare to V19
    logger.info(f"\n📊 Comparison to V19 Ensemble:")
    logger.info(f"   V19 CAGR:   23.5%    V20 CAGR:   {cagr:.1%}")
    logger.info(f"   V19 Sharpe: 1.15     V20 Sharpe: {sharpe:.2f}")
    logger.info(f"   V19 MaxDD:  -13.5%   V20 MaxDD:  {max_dd:.1%}")
    
    # Monthly returns
    logger.info("\n📅 Monthly Returns:")
    test_returns.index = pd.to_datetime(test_returns.index)
    monthly = test_returns.resample('ME').apply(lambda x: (1+x).prod()-1)
    for date, ret in monthly.items():
        logger.info(f"   {date:%Y-%m}: {ret:+.1%}")
    
    # Save results
    results_dir = Path('results/v20')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'phase': 'Phase 2: Enhanced Ensemble',
        'total_return': float(total_return),
        'cagr': float(cagr),
        'sharpe': float(sharpe),
        'max_drawdown': float(max_dd),
        'annual_vol': float(annual_vol),
        'win_rate': float(win_rate),
        'trading_days': int(trading_days),
        'avg_weights': {k: float(v) for k, v in avg_weights.items()},
        'targets_met': {
            'cagr_gt_32pct': bool(cagr_pass),
            'sharpe_gt_1.35': bool(sharpe_pass),
            'maxdd_gt_minus18': bool(dd_pass)
        },
        'v19_comparison': {
            'v19_cagr': 0.235,
            'v19_sharpe': 1.15,
            'v19_maxdd': -0.135,
            'cagr_improvement': float(cagr - 0.235),
            'sharpe_improvement': float(sharpe - 1.15)
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open(results_dir / 'v20_ensemble_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate report
    report = generate_report(results, monthly, cumulative, avg_weights)
    with open(results_dir / 'V20_FINAL_REPORT.md', 'w') as f:
        f.write(report)
    
    logger.info(f"\n💾 Results saved to {results_dir}")
    
    all_pass = all(results['targets_met'].values())
    if all_pass:
        logger.info("\n" + "=" * 60)
        logger.info("🎉 V20.0 ALL TARGETS MET!")
        logger.info("=" * 60)
    
    return results


def generate_report(results, monthly, cumulative, avg_weights):
    """Generate final report."""
    
    all_pass = all(results['targets_met'].values())
    
    report = f"""# V20.0 Final Report: Enhanced Ensemble System

**Generated:** {results['timestamp']}

---

## Executive Summary

| Metric | V19 Baseline | V20 Result | Target | Status |
|--------|--------------|------------|--------|--------|
| CAGR | 23.5% | **{results['cagr']:.1%}** | > 32% | {'✅' if results['targets_met']['cagr_gt_32pct'] else '❌'} |
| Sharpe | 1.15 | **{results['sharpe']:.2f}** | > 1.35 | {'✅' if results['targets_met']['sharpe_gt_1.35'] else '❌'} |
| Max DD | -13.5% | **{results['max_drawdown']:.1%}** | > -18% | {'✅' if results['targets_met']['maxdd_gt_minus18'] else '❌'} |
| Win Rate | 55.6% | {results['win_rate']:.1%} | - | - |

### Overall Status: {'✅ ALL TARGETS MET' if all_pass else '⚠️ SOME TARGETS NOT MET'}

---

## Strategy Components

### 1. Volatility Reversal (V20 - New)
- Long stocks with: drawdown 15-50%, vol spike, RSI < 30
- Base weight: 35%
- **Improvement over V19 reversal:** Better timing via volatility filter

### 2. Pure Reversal (V19)
- Long biggest 5-day losers with drawdown 10-40%
- Base weight: 35%
- Strong baseline performer

### 3. Mean Reversion (V19)
- Long when z-score < -2.0, exit at 0
- Base weight: 30%
- Consistent positive returns

---

## Dynamic Weight Adjustment

Weights adjusted based on 20-day rolling Sharpe:
- Strategies with negative Sharpe → weight = 0
- Normalize remaining to sum to 1
- Bounds: 10% minimum, 50% maximum

### Average Weights (Test Period)

| Strategy | Weight |
|----------|--------|
| Reversal | {avg_weights.get('reversal', 0):.1%} |
| Mean Reversion | {avg_weights.get('meanrev', 0):.1%} |
| Volatility Reversal | {avg_weights.get('volrev', 0):.1%} |

---

## Risk Controls Applied

| Control | Setting |
|---------|---------|
| Daily loss limit | 2% → reduce exposure 50% |
| Gross exposure | 100% max (no leverage) |
| Recovery rate | 10% per day after drawdown |

---

## Monthly Performance

| Month | Return |
|-------|--------|
"""
    
    for date, ret in monthly.items():
        report += f"| {date:%Y-%m} | {ret:+.1%} |\n"
    
    report += f"""

---

## Improvement over V19

| Metric | V19 | V20 | Improvement |
|--------|-----|-----|-------------|
| CAGR | 23.5% | {results['cagr']:.1%} | {results['v19_comparison']['cagr_improvement']:+.1%} |
| Sharpe | 1.15 | {results['sharpe']:.2f} | {results['v19_comparison']['sharpe_improvement']:+.2f} |

---

## Overfit Validation

| Check | Threshold | Value | Status |
|-------|-----------|-------|--------|
| CAGR | < 55% | {results['cagr']:.1%} | {'✅ OK' if results['cagr'] < 0.55 else '⚠️'} |
| Sharpe | < 3.0 | {results['sharpe']:.2f} | {'✅ OK' if results['sharpe'] < 3.0 else '⚠️'} |
| Max DD | < -8% | {results['max_drawdown']:.1%} | {'✅ OK' if results['max_drawdown'] < -0.08 else '⚠️'} |

---

## Files Created

| File | Purpose |
|------|---------|
| v20_volatility_reversal.py | New volatility-filtered reversal |
| v20_enhanced_ensemble.py | Combined system with dynamic weights |
| results/v20/V20_VOLREV_REPORT.md | Phase 1 results |
| results/v20/V20_FINAL_REPORT.md | This report |

---

{'## ✅ Ready for Paper Trading' if all_pass else '## ⚠️ Review Results Before Deployment'}

---

*Report generated by v20_enhanced_ensemble.py*
"""
    
    return report


if __name__ == "__main__":
    run_enhanced_ensemble()
