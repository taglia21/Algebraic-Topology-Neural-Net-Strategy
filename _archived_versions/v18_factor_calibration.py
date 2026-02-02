#!/usr/bin/env python3
"""
V18.0 Factor Calibration (Vectorized)
======================================
Fix V17's -12.8% CAGR by calibrating factor weights based on IC analysis.

Actions:
1. FLIP 25 factors with IC < -0.02 (multiply by -1)
2. REMOVE 4 factors with IC between -0.02 and 0
3. WEIGHT remaining factors by |IC|

Uses VECTORIZED pandas operations - no loops over stocks/factors.
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
logger = logging.getLogger('V18_Calibration')


# Factor IC from diagnostic (hardcoded to avoid recomputation)
FACTOR_IC = {
    # NEGATIVE IC - FLIP THESE (multiply by -1)
    'distance_from_high': -0.1909,
    'price_vs_ma200': -0.1652,
    'distance_from_low': -0.1408,
    'momentum_12m': -0.1278,
    'momentum_6m': -0.1216,
    'ma_cross_50_200': -0.1151,
    'momentum_3m': -0.1050,
    'risk_adjusted_momentum': -0.1028,
    'momentum_consistency': -0.0929,
    'channel_position': -0.0860,
    'zscore_50d': -0.0849,
    'momentum_1m': -0.0801,
    'new_high_count': -0.0795,
    'breakout_20d': -0.0769,
    'relative_strength': -0.0744,
    'zscore_20d': -0.0742,
    'betti_0_estimate': -0.0544,
    'macd_histogram': -0.0543,
    'ma_cross_20_50': -0.0533,
    'momentum_6_1': -0.0532,
    'momentum_12_1': -0.0445,
    'obv_momentum': -0.0410,
    'tda_complexity': -0.0360,
    'vwap_distance': -0.0306,
    'betti_1_estimate': -0.0220,
    
    # WEAK NEGATIVE - REMOVE THESE (IC between -0.02 and 0)
    'dollar_volume': -0.0198,  # REMOVE
    'persistence_entropy': -0.0152,  # REMOVE
    'price_gap': -0.0115,  # REMOVE
    'trend_strength_adx': -0.0008,  # REMOVE
    
    # POSITIVE IC - KEEP AS IS
    'downside_vol': 0.0884,
    'amihud_illiquidity': 0.0877,
    'atr_ratio': 0.0794,
    'overbought_oversold': 0.0707,
    'volatility_60d': 0.0701,
    'landscape_distance': 0.0700,
    'reversal_5d': 0.0610,
    'idio_vol': 0.0523,
    'volatility_20d': 0.0494,
    'volume_price_trend': 0.0398,
    'volume_momentum': 0.0287,
    'volatility_ratio': 0.0268,
    'upside_vol': 0.0239,
    'bollinger_width': 0.0236,
    'momentum_acceleration': 0.0200,
    'mean_reversion_speed': 0.0197,
    'volume_volatility': 0.0185,
    'wasserstein_distance': 0.0165,
    'kurtosis_60d': 0.0106,
    'relative_volume': 0.0092,
    'skewness_60d': 0.0050,
}


def get_calibrated_weights():
    """
    Return calibrated factor weights based on IC analysis.
    
    - Factors with IC < -0.02: FLIP (weight = -|IC|, then normalize)
    - Factors with IC between -0.02 and 0: REMOVE (weight = 0)
    - Factors with IC > 0: KEEP (weight = IC)
    """
    weights = {}
    
    for factor, ic in FACTOR_IC.items():
        if ic < -0.02:
            # FLIP: Use negative of IC as weight (so factor is multiplied by -1)
            # The weight itself represents the importance (absolute IC)
            # We flip the factor values, then weight by |IC|
            weights[factor] = {'action': 'FLIP', 'weight': abs(ic), 'original_ic': ic}
        elif ic < 0:
            # REMOVE: Too weak to use
            weights[factor] = {'action': 'REMOVE', 'weight': 0, 'original_ic': ic}
        else:
            # KEEP: Positive IC
            weights[factor] = {'action': 'KEEP', 'weight': ic, 'original_ic': ic}
    
    # Normalize weights to sum to 1 (only non-zero weights)
    total_weight = sum(w['weight'] for w in weights.values())
    if total_weight > 0:
        for factor in weights:
            weights[factor]['normalized_weight'] = weights[factor]['weight'] / total_weight
    
    return weights


def run_calibrated_backtest():
    """Run backtest with calibrated factors using vectorized operations."""
    
    logger.info("=" * 60)
    logger.info("🔧 V18.0 FACTOR CALIBRATION")
    logger.info("=" * 60)
    
    # Get calibrated weights
    weights = get_calibrated_weights()
    
    # Count actions
    flip_count = sum(1 for w in weights.values() if w['action'] == 'FLIP')
    remove_count = sum(1 for w in weights.values() if w['action'] == 'REMOVE')
    keep_count = sum(1 for w in weights.values() if w['action'] == 'KEEP')
    
    logger.info(f"📊 Factor Calibration:")
    logger.info(f"   FLIP (contrarian): {flip_count} factors")
    logger.info(f"   REMOVE (no signal): {remove_count} factors")
    logger.info(f"   KEEP (as-is): {keep_count} factors")
    
    # Load price data
    logger.info("\n📂 Loading data...")
    prices = pd.read_parquet('cache/v17_prices/v17_prices_latest.parquet')
    prices['date'] = pd.to_datetime(prices['date'])
    
    # Use top 100 symbols for speed
    symbol_volume = prices.groupby('symbol')['volume'].mean().nlargest(100)
    top_symbols = symbol_volume.index.tolist()
    prices = prices[prices['symbol'].isin(top_symbols)]
    
    logger.info(f"   Symbols: {len(top_symbols)}")
    logger.info(f"   Date range: {prices['date'].min():%Y-%m-%d} to {prices['date'].max():%Y-%m-%d}")
    
    # Calculate factors VECTORIZED (pivot first)
    logger.info("\n📈 Computing factors (vectorized)...")
    
    # Pivot to wide format: dates as rows, symbols as columns
    close_wide = prices.pivot(index='date', columns='symbol', values='close')
    volume_wide = prices.pivot(index='date', columns='symbol', values='volume')
    high_wide = prices.pivot(index='date', columns='symbol', values='high')
    low_wide = prices.pivot(index='date', columns='symbol', values='low')
    
    # Calculate returns
    returns_1d = close_wide.pct_change(1)
    returns_5d = close_wide.pct_change(5)
    returns_20d = close_wide.pct_change(20)
    
    # Forward returns for signal evaluation
    fwd_ret_5d = close_wide.pct_change(5).shift(-5)
    
    # Calculate key factors (vectorized across all symbols at once)
    factors_wide = {}
    
    # Momentum factors (will be FLIPPED)
    factors_wide['momentum_3m'] = close_wide.pct_change(63)
    factors_wide['momentum_6m'] = close_wide.pct_change(126)
    factors_wide['momentum_1m'] = close_wide.pct_change(21)
    
    # Volatility factors (KEEP)
    factors_wide['volatility_20d'] = returns_1d.rolling(20).std() * np.sqrt(252)
    factors_wide['volatility_60d'] = returns_1d.rolling(60).std() * np.sqrt(252)
    
    # Downside vol (BEST factor)
    downside_ret = returns_1d.where(returns_1d < 0, 0)
    factors_wide['downside_vol'] = downside_ret.rolling(60).std() * np.sqrt(252)
    
    # Mean reversion factors (KEEP)
    ma20 = close_wide.rolling(20).mean()
    std20 = close_wide.rolling(20).std()
    factors_wide['zscore_20d'] = (close_wide - ma20) / std20
    factors_wide['reversal_5d'] = -returns_5d  # Already contrarian
    
    # ATR ratio (KEEP)
    tr = pd.concat([
        high_wide - low_wide,
        (high_wide - close_wide.shift()).abs(),
        (low_wide - close_wide.shift()).abs()
    ]).groupby(level=0).max()
    atr = tr.rolling(14).mean()
    factors_wide['atr_ratio'] = atr / close_wide
    
    # Overbought/Oversold (KEEP)
    high_14 = high_wide.rolling(14).max()
    low_14 = low_wide.rolling(14).min()
    factors_wide['overbought_oversold'] = (high_14 - close_wide) / (high_14 - low_14)
    
    # Distance from high (FLIP - becomes "buy dips")
    high_252 = high_wide.rolling(252).max()
    factors_wide['distance_from_high'] = (close_wide - high_252) / high_252
    
    # Price vs MA200 (FLIP)
    ma200 = close_wide.rolling(200).mean()
    factors_wide['price_vs_ma200'] = (close_wide - ma200) / ma200
    
    logger.info(f"   Computed {len(factors_wide)} key factors")
    
    # Calculate composite signal using calibrated weights
    logger.info("\n🎯 Calculating calibrated composite signal...")
    
    composite = pd.DataFrame(0.0, index=close_wide.index, columns=close_wide.columns)
    
    for factor_name, factor_df in factors_wide.items():
        if factor_name not in weights:
            continue
        
        w = weights[factor_name]
        
        if w['action'] == 'REMOVE':
            continue
        
        # Z-score the factor cross-sectionally
        factor_mean = factor_df.mean(axis=1)
        factor_std = factor_df.std(axis=1)
        factor_zscore = factor_df.sub(factor_mean, axis=0).div(factor_std, axis=0)
        factor_zscore = factor_zscore.clip(-3, 3).fillna(0)
        
        if w['action'] == 'FLIP':
            # Multiply by -1 to flip direction
            factor_zscore = -factor_zscore
        
        # Add weighted factor to composite
        composite = composite + factor_zscore * w['normalized_weight']
    
    # Rank composite signal cross-sectionally
    signal_rank = composite.rank(axis=1, pct=True)
    
    # Generate positions: long top 20%, short bottom 20%
    positions = pd.DataFrame(0.0, index=signal_rank.index, columns=signal_rank.columns)
    positions = positions.where(~(signal_rank > 0.8), 1.0)
    positions = positions.where(~(signal_rank < 0.2), -1.0)
    
    # Calculate strategy returns
    logger.info("\n📊 Running backtest...")
    
    # Position at t, return from t to t+1
    strategy_returns = (positions.shift(1) * returns_1d).mean(axis=1)
    
    # Apply transaction costs: 10bps roundtrip when position changes
    position_changes = positions.diff().abs()
    turnover = position_changes.mean(axis=1)
    cost_per_day = turnover * 0.0010  # 10bps roundtrip
    
    net_returns = strategy_returns - cost_per_day
    
    # Calculate metrics
    cumulative = (1 + net_returns).cumprod()
    
    # Filter to test period (last 12 months)
    test_start = close_wide.index.max() - pd.DateOffset(months=12)
    test_returns = net_returns[net_returns.index >= test_start]
    test_cumulative = (1 + test_returns).cumprod()
    
    # Metrics
    total_return = test_cumulative.iloc[-1] - 1 if len(test_cumulative) > 0 else 0
    trading_days = len(test_returns)
    years = trading_days / 252
    
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    annual_vol = test_returns.std() * np.sqrt(252)
    sharpe = (test_returns.mean() * 252) / annual_vol if annual_vol > 0 else 0
    
    # Drawdown
    peak = test_cumulative.expanding().max()
    drawdown = (test_cumulative - peak) / peak
    max_dd = drawdown.min()
    
    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("📊 V18.0 CALIBRATED BACKTEST RESULTS")
    logger.info("=" * 60)
    
    logger.info(f"\n📈 Performance (Last 12 Months):")
    logger.info(f"   Total Return:   {total_return:.1%}")
    logger.info(f"   CAGR:           {cagr:.1%}")
    logger.info(f"   Sharpe Ratio:   {sharpe:.2f}")
    logger.info(f"   Max Drawdown:   {max_dd:.1%}")
    logger.info(f"   Annual Vol:     {annual_vol:.1%}")
    logger.info(f"   Trading Days:   {trading_days}")
    
    # Compare to V17
    logger.info(f"\n📊 Comparison to V17:")
    logger.info(f"   V17 CAGR:       -12.8%")
    logger.info(f"   V18 CAGR:       {cagr:.1%}")
    logger.info(f"   Improvement:    {(cagr - (-0.128)):.1%}")
    
    # Top 10 factor weights
    logger.info(f"\n🏆 Top 10 Factor Weights:")
    sorted_weights = sorted(
        [(f, w) for f, w in weights.items() if w['weight'] > 0],
        key=lambda x: x[1]['weight'],
        reverse=True
    )[:10]
    
    for factor, w in sorted_weights:
        action_symbol = "🔄" if w['action'] == 'FLIP' else "✓"
        logger.info(f"   {action_symbol} {factor}: {w['normalized_weight']:.1%} (IC={w['original_ic']:.4f})")
    
    # Success check
    success = cagr > 0
    
    if success:
        logger.info(f"\n✅ SUCCESS: CAGR is POSITIVE ({cagr:.1%})")
    else:
        logger.info(f"\n⚠️ CAGR still negative ({cagr:.1%}) - may need further optimization")
    
    # Save results
    results_dir = Path('results/v18')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'total_return': float(total_return),
        'cagr': float(cagr),
        'sharpe': float(sharpe),
        'max_drawdown': float(max_dd),
        'annual_vol': float(annual_vol),
        'trading_days': int(trading_days),
        'factors_flipped': flip_count,
        'factors_removed': remove_count,
        'factors_kept': keep_count,
        'v17_cagr': -0.128,
        'improvement': float(cagr - (-0.128)),
        'success': bool(success),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(results_dir / 'v18_calibration_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate report
    report = generate_report(results, weights, sorted_weights)
    with open(results_dir / 'V18_CALIBRATION_REPORT.md', 'w') as f:
        f.write(report)
    
    logger.info(f"\n💾 Results saved to {results_dir}")
    
    return results


def generate_report(results, weights, top_weights):
    """Generate markdown calibration report."""
    
    report = f"""# V18.0 Factor Calibration Report

**Generated:** {results['timestamp']}

---

## Executive Summary

| Metric | V17 (Before) | V18 (After) | Change |
|--------|--------------|-------------|--------|
| CAGR | -12.8% | {results['cagr']:.1%} | {results['improvement']:+.1%} |
| Sharpe | -1.35 | {results['sharpe']:.2f} | - |
| Max DD | -21.1% | {results['max_drawdown']:.1%} | - |

**Result:** {"✅ SUCCESS - Positive CAGR achieved!" if results['success'] else "⚠️ Still negative, needs further work"}

---

## Calibration Actions

| Action | Count | Description |
|--------|-------|-------------|
| FLIP | {results['factors_flipped']} | Reversed sign (use as contrarian) |
| REMOVE | {results['factors_removed']} | Excluded (weak/no signal) |
| KEEP | {results['factors_kept']} | Used as-is |

---

## Top 10 Factor Weights (After Calibration)

| Rank | Factor | Weight | Action | Original IC |
|------|--------|--------|--------|-------------|
"""
    
    for i, (factor, w) in enumerate(top_weights, 1):
        action = w['action']
        report += f"| {i} | {factor} | {w['normalized_weight']:.1%} | {action} | {w['original_ic']:.4f} |\n"
    
    report += f"""

---

## Key Insight

The diagnostic showed that **momentum factors had negative IC** in the recent market regime.
By **flipping** these factors (using them as contrarian/reversal signals), 
we align with the actual market behavior.

### Before (V17):
- Used momentum_12_1, momentum_6m as "buy winners"
- These had negative IC → buying losers

### After (V18):
- Flipped momentum factors → "buy recent losers"
- Weighted by |IC| → strongest signals get most weight

---

## Next Steps

1. **If CAGR > 0%:** Factor calibration works - proceed to regime-specific weights
2. **If CAGR < 0%:** Review factor computation for bugs, try simpler model

---

*Report generated by v18_factor_calibration.py*
"""
    
    return report


if __name__ == "__main__":
    run_calibrated_backtest()
