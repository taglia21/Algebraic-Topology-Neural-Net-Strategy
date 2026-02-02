#!/usr/bin/env python3
"""
V24 Low-Beta Momentum V5 - OPTIMAL BALANCE
============================================
70% long / 30% short to achieve:
1. LOW correlation with V21 (< 0.3)
2. POSITIVE returns (unlike full market-neutral)

Key insight from V4:
- Full market-neutral (50/50): correlation -0.17 ✅ but CAGR -1.2% ❌
- Need to tilt slightly long to capture market returns while staying decorrelated

Strategy:
- 70% gross long (top momentum quintile)
- 30% gross short (bottom momentum quintile)
- Net exposure: ~40% (vs V21's ~100%)
- This should give correlation ~0.2 while having positive returns
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('V24_LowBeta')


class V24Config:
    """Low-beta momentum configuration."""
    # Position sizing - 70/30 long/short
    LONG_WEIGHT = 0.70   # 70% gross long
    SHORT_WEIGHT = 0.30  # 30% gross short
    # Net exposure = 40%
    
    LONG_PCT = 0.20   # Long top 20%
    SHORT_PCT = 0.20  # Short bottom 20%
    MAX_POSITIONS_PER_LEG = 25
    
    REBAL_PERIOD = 15  # Every 15 trading days
    MOM_LOOKBACK = 60  # 60-day momentum (3 months)
    
    MIN_PRICE = 10.0
    MIN_DOLLAR_VOLUME = 5_000_000
    
    COST_BPS_LONG = 10
    COST_BPS_SHORT = 25


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and prepare price data."""
    cache_path = Path('cache/v17_prices/v17_prices_latest.parquet')
    prices = pd.read_parquet(cache_path)
    prices['date'] = pd.to_datetime(prices['date'])
    
    symbol_col = 'symbol' if 'symbol' in prices.columns else 'ticker'
    
    close_wide = prices.pivot(index='date', columns=symbol_col, values='close')
    volume_wide = prices.pivot(index='date', columns=symbol_col, values='volume')
    
    close_wide = close_wide.ffill(limit=5)
    volume_wide = volume_wide.ffill(limit=5)
    
    logger.info(f"Data: {close_wide.shape[0]} days x {close_wide.shape[1]} stocks")
    
    return close_wide, volume_wide


def run_v24_low_beta(close_wide: pd.DataFrame,
                     volume_wide: pd.DataFrame,
                     config: V24Config) -> pd.Series:
    """
    Run 70/30 long/short momentum strategy.
    """
    ret_1d = close_wide.pct_change(fill_method=None)
    
    # Filters
    price_ok = close_wide > config.MIN_PRICE
    dollar_vol = close_wide * volume_wide
    avg_dollar_vol = dollar_vol.rolling(20).mean()
    liquid = avg_dollar_vol > config.MIN_DOLLAR_VOLUME
    tradeable = price_ok & liquid
    
    # Momentum (60-day)
    momentum = close_wide.pct_change(config.MOM_LOOKBACK, fill_method=None)
    momentum_filtered = momentum.where(tradeable)
    ranks = momentum_filtered.rank(axis=1, pct=True, ascending=True, na_option='keep')
    
    # Positions
    long_weights = pd.DataFrame(0.0, index=ranks.index, columns=ranks.columns)
    short_weights = pd.DataFrame(0.0, index=ranks.index, columns=ranks.columns)
    
    valid_dates = ranks.index[config.MOM_LOOKBACK:]
    rebal_dates = valid_dates[::config.REBAL_PERIOD]
    
    current_long = pd.Series(0.0, index=ranks.columns)
    current_short = pd.Series(0.0, index=ranks.columns)
    
    for date in valid_dates:
        if date in rebal_dates:
            day_ranks = ranks.loc[date].dropna()
            
            if len(day_ranks) < 20:
                current_long = pd.Series(0.0, index=ranks.columns)
                current_short = pd.Series(0.0, index=ranks.columns)
            else:
                # Long: top 20%
                long_stocks = day_ranks.nlargest(min(len(day_ranks) // 5, config.MAX_POSITIONS_PER_LEG)).index
                
                # Short: bottom 20%
                short_stocks = day_ranks.nsmallest(min(len(day_ranks) // 5, config.MAX_POSITIONS_PER_LEG)).index
                
                current_long = pd.Series(0.0, index=ranks.columns)
                current_short = pd.Series(0.0, index=ranks.columns)
                
                if len(long_stocks) > 0:
                    current_long[long_stocks] = config.LONG_WEIGHT / len(long_stocks)
                if len(short_stocks) > 0:
                    current_short[short_stocks] = config.SHORT_WEIGHT / len(short_stocks)
        
        long_weights.loc[date] = current_long
        short_weights.loc[date] = current_short
    
    # Returns
    long_returns = (long_weights.shift(1) * ret_1d).sum(axis=1)
    short_returns = -(short_weights.shift(1) * ret_1d).sum(axis=1)
    strategy_returns = long_returns + short_returns
    
    # Costs
    long_turnover = long_weights.diff().abs().sum(axis=1)
    short_turnover = short_weights.diff().abs().sum(axis=1)
    costs = pd.Series(0.0, index=strategy_returns.index)
    costs.loc[rebal_dates] = (
        long_turnover.loc[rebal_dates] * (config.COST_BPS_LONG / 10000) +
        short_turnover.loc[rebal_dates] * (config.COST_BPS_SHORT / 10000)
    )
    
    net_returns = strategy_returns - costs
    net_returns = net_returns.loc[valid_dates].dropna()
    
    # Log exposure
    net_exposure = (long_weights.sum(axis=1) - short_weights.sum(axis=1))
    logger.info(f"Net exposure: {net_exposure.loc[valid_dates].mean():.1%}")
    
    return net_returns


def simulate_v21(close_wide: pd.DataFrame) -> pd.Series:
    """Simulate V21."""
    ret_1d = close_wide.pct_change(fill_method=None)
    
    delta = close_wide.diff()
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)
    avg_gain = gains.rolling(14).mean()
    avg_loss = losses.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    
    vol_20d = ret_1d.rolling(20).std() * np.sqrt(252)
    
    v21_entry = (rsi < 35) & (vol_20d > 0.30)
    rsi_filtered = rsi.where(v21_entry)
    ranks = rsi_filtered.rank(axis=1, pct=True, ascending=True, na_option='keep')
    
    positions = pd.DataFrame(0.0, index=ranks.index, columns=ranks.columns)
    
    for date in ranks.index:
        valid_count = ranks.loc[date].notna().sum()
        if valid_count < 5:
            continue
        n_long = min(30, valid_count)
        threshold = n_long / valid_count
        positions.loc[date, ranks.loc[date] <= threshold] = 1.0
    
    rebal_dates = positions.index[::5]
    positions_held = positions.copy()
    positions_held.loc[~positions_held.index.isin(rebal_dates)] = np.nan
    positions_held = positions_held.ffill()
    
    counts = (positions_held > 0).sum(axis=1).replace(0, 1)
    weights = positions_held.div(counts, axis=0)
    
    v21_returns = (weights.shift(1) * ret_1d).sum(axis=1)
    
    weight_changes = weights.diff().abs().sum(axis=1)
    costs = pd.Series(0.0, index=v21_returns.index)
    costs.loc[rebal_dates] = weight_changes.loc[rebal_dates] * 0.001
    
    return (v21_returns - costs).dropna()


def calculate_metrics(ret: pd.Series, name: str) -> Dict:
    """Calculate metrics."""
    if len(ret) < 30 or ret.std() == 0:
        return {'name': name, 'cagr': 0, 'vol': 0, 'sharpe': 0, 'max_dd': 0}
    
    cum = (1 + ret).cumprod()
    years = len(ret) / 252
    cagr = cum.iloc[-1] ** (1/years) - 1 if years > 0 else 0
    vol = ret.std() * np.sqrt(252)
    sharpe = ret.mean() / ret.std() * np.sqrt(252)
    max_dd = (cum / cum.expanding().max() - 1).min()
    
    return {'name': name, 'cagr': float(cagr), 'vol': float(vol), 
            'sharpe': float(sharpe), 'max_dd': float(max_dd)}


def main():
    """Run V24 Low-Beta Momentum."""
    logger.info("=" * 70)
    logger.info("V24 LOW-BETA MOMENTUM V5 (70% Long / 30% Short)")
    logger.info("=" * 70)
    
    config = V24Config()
    close_wide, volume_wide = load_data()
    
    # V24 backtest
    v24_returns = run_v24_low_beta(close_wide, volume_wide, config)
    v24_metrics = calculate_metrics(v24_returns, 'V24')
    
    logger.info(f"\nV24 LOW-BETA METRICS:")
    logger.info(f"  CAGR: {v24_metrics['cagr']:.1%}")
    logger.info(f"  Vol: {v24_metrics['vol']:.1%}")
    logger.info(f"  Sharpe: {v24_metrics['sharpe']:.2f}")
    logger.info(f"  MaxDD: {v24_metrics['max_dd']:.1%}")
    
    # V21 simulation
    v21_returns = simulate_v21(close_wide)
    v21_metrics = calculate_metrics(v21_returns, 'V21')
    
    logger.info(f"\nV21 METRICS:")
    logger.info(f"  CAGR: {v21_metrics['cagr']:.1%}")
    logger.info(f"  Sharpe: {v21_metrics['sharpe']:.2f}")
    
    # Correlation
    common = v24_returns.index.intersection(v21_returns.index)
    v24 = v24_returns.loc[common]
    v21 = v21_returns.loc[common]
    
    correlation = v24.corr(v21)
    
    logger.info(f"\n★ CORRELATION: {correlation:.3f}")
    logger.info(f"  Target: < 0.3")
    logger.info(f"  Status: {'✅ PASS' if abs(correlation) < 0.3 else '❌ FAIL'}")
    
    # Combined portfolio
    combined = 0.5 * v24 + 0.5 * v21
    combined_metrics = calculate_metrics(combined, 'Combined')
    
    logger.info(f"\nCOMBINED PORTFOLIO (50/50):")
    logger.info(f"  CAGR: {combined_metrics['cagr']:.1%}")
    logger.info(f"  Sharpe: {combined_metrics['sharpe']:.2f}")
    logger.info(f"  MaxDD: {combined_metrics['max_dd']:.1%}")
    logger.info(f"  Sharpe improvement: {combined_metrics['sharpe'] - v21_metrics['sharpe']:+.2f}")
    
    # Save
    results = {
        'timestamp': datetime.now().isoformat(),
        'strategy': 'V24_LowBeta_V5',
        'v24': v24_metrics,
        'v21': v21_metrics,
        'correlation': float(correlation),
        'combined': combined_metrics,
        'passed_correlation': bool(abs(correlation) < 0.3),
        'passed_returns': bool(v24_metrics['cagr'] > 0)
    }
    
    results_dir = Path('results/v24')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / 'v24_v5_low_beta_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    v24_returns.to_frame('returns').to_parquet(results_dir / 'v24_v5_daily_returns.parquet')
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    
    if abs(correlation) < 0.3 and v24_metrics['cagr'] > 0:
        logger.info("🎉 SUCCESS! V24 achieves low correlation AND positive returns!")
    elif abs(correlation) < 0.3:
        logger.info("✅ Correlation target met, but returns negative")
    else:
        logger.info("❌ Correlation still too high")
    
    return results


if __name__ == "__main__":
    main()
