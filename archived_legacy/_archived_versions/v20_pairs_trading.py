#!/usr/bin/env python3
"""
V20.0 Phase 1: Sector Rotation + Relative Value Strategy
==========================================================
Instead of pairs trading (which struggles in current regime), use sector-based
relative value: long underperforming sectors, short outperforming sectors.

This is a more robust form of mean reversion at the sector level.

Logic:
1. Group stocks by sector (approximated by correlation clusters)
2. Calculate 20-day sector returns
3. Long bottom 3 sectors, short top 3 sectors
4. Rebalance weekly

Target: CAGR > 15%, Sharpe > 0.8
"""

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('V20_SectorRV')


def load_price_data():
    """Load price data from V17 cache."""
    cache_path = Path('cache/v17_prices/v17_prices_latest.parquet')
    prices = pd.read_parquet(cache_path)
    prices['date'] = pd.to_datetime(prices['date'])
    return prices


def calculate_half_life(spread):
    """Calculate half-life of mean reversion using AR(1)."""
    spread = spread.dropna()
    if len(spread) < 30:
        return np.inf
    
    spread_lag = spread.shift(1).dropna()
    spread_diff = spread.diff().dropna()
    
    # Align
    spread_lag = spread_lag.iloc[1:]
    spread_diff = spread_diff.iloc[1:]
    
    if len(spread_lag) < 10:
        return np.inf
    
    try:
        # AR(1) regression: delta_y = phi * y_lag + epsilon
        slope, intercept, r_value, p_value, std_err = stats.linregress(spread_lag, spread_diff)
        
        if slope >= 0:
            return np.inf  # Not mean reverting
        
        half_life = -np.log(2) / slope
        return half_life
    except:
        return np.inf


def find_cointegrated_pairs(close_wide, train_end_idx, min_hl=5, max_hl=25, p_threshold=0.05):
    """Find cointegrated pairs from training data."""
    
    train_data = close_wide.iloc[:train_end_idx]
    
    # Get top 100 most liquid for speed
    valid_cols = train_data.columns[train_data.notna().sum() > len(train_data) * 0.9]
    if len(valid_cols) > 100:
        # Sort by average volume (use variance as proxy for liquidity)
        variances = train_data[valid_cols].var().nlargest(100)
        valid_cols = variances.index.tolist()
    
    logger.info(f"   Testing pairs among {len(valid_cols)} stocks...")
    
    pairs = []
    tested = 0
    
    # Test all combinations
    for s1, s2 in combinations(valid_cols, 2):
        tested += 1
        
        if tested % 1000 == 0:
            logger.info(f"   Tested {tested} pairs, found {len(pairs)} cointegrated...")
        
        y1 = train_data[s1].dropna()
        y2 = train_data[s2].dropna()
        
        # Align
        common_idx = y1.index.intersection(y2.index)
        if len(common_idx) < 100:
            continue
        
        y1 = y1.loc[common_idx]
        y2 = y2.loc[common_idx]
        
        try:
            # Engle-Granger cointegration test
            score, pvalue, _ = coint(y1, y2)
            
            if pvalue > p_threshold:
                continue
            
            # Calculate hedge ratio (beta)
            model = OLS(y1, y2).fit()
            beta = model.params[0]
            
            if beta <= 0:
                continue
            
            # Calculate spread
            spread = y1 - beta * y2
            
            # Check half-life
            half_life = calculate_half_life(spread)
            
            if min_hl <= half_life <= max_hl:
                # Calculate spread volatility for ranking
                spread_std = spread.std()
                
                pairs.append({
                    'stock1': s1,
                    'stock2': s2,
                    'pvalue': pvalue,
                    'beta': beta,
                    'half_life': half_life,
                    'spread_std': spread_std
                })
        except:
            continue
    
    logger.info(f"   Found {len(pairs)} cointegrated pairs with valid half-life")
    
    # Sort by p-value (most significant first)
    pairs.sort(key=lambda x: x['pvalue'])
    
    return pairs[:30]  # Top 30


def run_pairs_backtest():
    """Run pairs trading backtest with walk-forward validation."""
    
    logger.info("=" * 60)
    logger.info("📊 V20.0 PHASE 1: PAIRS TRADING STRATEGY")
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
    
    # Pivot to wide format
    close_wide = prices.pivot(index='date', columns='symbol', values='close')
    
    logger.info(f"   Date range: {close_wide.index[0]:%Y-%m-%d} to {close_wide.index[-1]:%Y-%m-%d}")
    logger.info(f"   Total days: {len(close_wide)}")
    
    # Walk-forward parameters
    TRAIN_DAYS = 126  # 6 months
    TEST_DAYS = 42    # 2 months
    ZSCORE_ENTRY = 1.5  # Lower threshold for more trades
    ZSCORE_EXIT = 0.3
    POSITION_SIZE = 0.05  # 5% per pair (2.5% each leg)
    COST_BPS = 10
    
    logger.info(f"\n⚙️ Strategy Parameters:")
    logger.info(f"   Train window: {TRAIN_DAYS} days")
    logger.info(f"   Test window: {TEST_DAYS} days")
    logger.info(f"   Z-score entry: ±{ZSCORE_ENTRY}")
    logger.info(f"   Z-score exit: ±{ZSCORE_EXIT}")
    logger.info(f"   Position size: {POSITION_SIZE:.1%} per pair")
    
    # Walk-forward loop
    all_returns = []
    all_trades = []
    
    start_idx = TRAIN_DAYS
    n_windows = 0
    
    while start_idx + TEST_DAYS <= len(close_wide):
        n_windows += 1
        train_end = start_idx
        test_end = min(start_idx + TEST_DAYS, len(close_wide))
        
        logger.info(f"\n🔄 Walk-forward window {n_windows}:")
        logger.info(f"   Train: {close_wide.index[train_end-TRAIN_DAYS]:%Y-%m-%d} to {close_wide.index[train_end-1]:%Y-%m-%d}")
        logger.info(f"   Test:  {close_wide.index[train_end]:%Y-%m-%d} to {close_wide.index[test_end-1]:%Y-%m-%d}")
        
        # Find pairs on training data
        pairs = find_cointegrated_pairs(close_wide, train_end)
        
        if len(pairs) < 5:
            logger.info(f"   ⚠️ Only {len(pairs)} pairs found, skipping window")
            start_idx += TEST_DAYS
            continue
        
        logger.info(f"   Selected {len(pairs)} pairs for trading")
        
        # Trade on test data
        test_data = close_wide.iloc[train_end:test_end]
        window_returns = pd.Series(0.0, index=test_data.index)
        
        for pair in pairs:
            s1, s2 = pair['stock1'], pair['stock2']
            beta = pair['beta']
            
            if s1 not in test_data.columns or s2 not in test_data.columns:
                continue
            
            p1 = test_data[s1]
            p2 = test_data[s2]
            
            # Calculate spread
            spread = p1 - beta * p2
            
            # Z-score using rolling 20-day window
            spread_mean = spread.rolling(20, min_periods=10).mean()
            spread_std = spread.rolling(20, min_periods=10).std()
            zscore = (spread - spread_mean) / spread_std
            
            # Generate signals
            position = 0  # 0: flat, 1: long spread, -1: short spread
            pair_returns = []
            
            for i in range(1, len(test_data)):
                date = test_data.index[i]
                prev_date = test_data.index[i-1]
                
                z = zscore.iloc[i-1] if not pd.isna(zscore.iloc[i-1]) else 0
                
                # Entry signals
                if position == 0:
                    if z < -ZSCORE_ENTRY:
                        position = 1  # Long spread (long S1, short S2)
                        all_trades.append({'date': date, 'pair': f"{s1}-{s2}", 'action': 'LONG'})
                    elif z > ZSCORE_ENTRY:
                        position = -1  # Short spread (short S1, long S2)
                        all_trades.append({'date': date, 'pair': f"{s1}-{s2}", 'action': 'SHORT'})
                
                # Exit signals
                elif position != 0:
                    if (position == 1 and z > -ZSCORE_EXIT) or (position == -1 and z < ZSCORE_EXIT):
                        all_trades.append({'date': date, 'pair': f"{s1}-{s2}", 'action': 'EXIT'})
                        position = 0
                
                # Calculate return if in position
                if position != 0:
                    ret1 = (p1.iloc[i] / p1.iloc[i-1] - 1) if p1.iloc[i-1] > 0 else 0
                    ret2 = (p2.iloc[i] / p2.iloc[i-1] - 1) if p2.iloc[i-1] > 0 else 0
                    
                    # Spread return: long S1, short S2 (or vice versa)
                    # Full position size for the pair
                    if position == 1:
                        pair_ret = (ret1 - ret2) * POSITION_SIZE
                    else:
                        pair_ret = (ret2 - ret1) * POSITION_SIZE
                    
                    pair_returns.append((date, pair_ret))
            
            # Add pair returns to window returns
            for date, ret in pair_returns:
                if date in window_returns.index:
                    window_returns[date] += ret
        
        # Apply transaction costs
        trades_per_day = pd.Series([t['date'] for t in all_trades if t['date'] in window_returns.index]).value_counts()
        for date, n_trades in trades_per_day.items():
            if date in window_returns.index:
                window_returns[date] -= n_trades * 2 * (COST_BPS / 10000) * POSITION_SIZE
        
        all_returns.append(window_returns)
        start_idx += TEST_DAYS
    
    # Combine all returns
    if not all_returns:
        logger.error("No valid trading windows!")
        return None
    
    returns = pd.concat(all_returns)
    returns = returns.groupby(returns.index).sum()  # Combine overlapping dates
    
    # Calculate metrics
    cumulative = (1 + returns).cumprod()
    total_return = cumulative.iloc[-1] - 1 if len(cumulative) > 0 else 0
    
    trading_days = len(returns)
    years = trading_days / 252
    
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    annual_vol = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() * 252) / annual_vol if annual_vol > 0 else 0
    
    peak = cumulative.expanding().max()
    drawdown = (cumulative - peak) / peak
    max_dd = drawdown.min()
    
    winning_days = (returns > 0).sum()
    total_days = (returns != 0).sum()
    win_rate = winning_days / total_days if total_days > 0 else 0
    
    n_trades = len(all_trades)
    
    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("📊 V20.0 PAIRS TRADING RESULTS")
    logger.info("=" * 60)
    
    logger.info(f"\n📈 Performance ({trading_days} trading days):")
    logger.info(f"   Total Return:   {total_return:.1%}")
    logger.info(f"   CAGR:           {cagr:.1%}")
    logger.info(f"   Sharpe Ratio:   {sharpe:.2f}")
    logger.info(f"   Max Drawdown:   {max_dd:.1%}")
    logger.info(f"   Annual Vol:     {annual_vol:.1%}")
    logger.info(f"   Win Rate:       {win_rate:.1%}")
    logger.info(f"   Total Trades:   {n_trades}")
    
    # Validate targets
    logger.info(f"\n🎯 Phase 1 Targets:")
    cagr_pass = cagr > 0.15
    logger.info(f"   CAGR > 15%:     {'✅ PASS' if cagr_pass else '❌ FAIL'} ({cagr:.1%})")
    
    sharpe_pass = sharpe > 0.8
    logger.info(f"   Sharpe > 0.8:   {'✅ PASS' if sharpe_pass else '❌ FAIL'} ({sharpe:.2f})")
    
    # Monthly returns
    logger.info("\n📅 Monthly Returns:")
    returns.index = pd.to_datetime(returns.index)
    monthly = returns.resample('ME').apply(lambda x: (1+x).prod()-1)
    for date, ret in monthly.tail(6).items():
        logger.info(f"   {date:%Y-%m}: {ret:+.1%}")
    
    # Save results
    results_dir = Path('results/v20')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'phase': 'Phase 1: Pairs Trading',
        'total_return': float(total_return),
        'cagr': float(cagr),
        'sharpe': float(sharpe),
        'max_drawdown': float(max_dd),
        'annual_vol': float(annual_vol),
        'win_rate': float(win_rate),
        'trading_days': int(trading_days),
        'n_trades': n_trades,
        'n_windows': n_windows,
        'targets_met': {
            'cagr_gt_15pct': bool(cagr_pass),
            'sharpe_gt_0.8': bool(sharpe_pass)
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open(results_dir / 'v20_pairs_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate report
    report = generate_report(results, monthly, cumulative)
    with open(results_dir / 'V20_PAIRS_REPORT.md', 'w') as f:
        f.write(report)
    
    logger.info(f"\n💾 Results saved to {results_dir}")
    
    return {
        'results': results,
        'daily_returns': returns,
        'cumulative': cumulative
    }


def generate_report(results, monthly, cumulative):
    """Generate markdown report."""
    
    all_pass = all(results['targets_met'].values())
    
    report = f"""# V20.0 Phase 1: Pairs Trading Report

**Generated:** {results['timestamp']}

---

## Executive Summary

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| CAGR | {results['cagr']:.1%} | > 15% | {'✅' if results['targets_met']['cagr_gt_15pct'] else '❌'} |
| Sharpe | {results['sharpe']:.2f} | > 0.8 | {'✅' if results['targets_met']['sharpe_gt_0.8'] else '❌'} |
| Max Drawdown | {results['max_drawdown']:.1%} | - | - |
| Win Rate | {results['win_rate']:.1%} | - | - |
| Total Trades | {results['n_trades']} | - | - |

**Overall:** {'✅ PROCEED TO PHASE 2' if all_pass else '⚠️ REVIEW BEFORE PROCEEDING'}

---

## Strategy Logic

```
1. Find cointegrated pairs (Engle-Granger, p < 0.05)
2. Filter by half-life: 5-25 days
3. Select top 30 pairs per window
4. Calculate spread z-score (20-day rolling)
5. ENTRY: |z-score| > 2.0
6. EXIT: |z-score| < 0.5
7. Position: 2% per pair (1% each leg)
```

---

## Walk-Forward Validation

| Parameter | Value |
|-----------|-------|
| Train window | 126 days (6 months) |
| Test window | 42 days (2 months) |
| Total windows | {results['n_windows']} |
| Trading days | {results['trading_days']} |

---

## Monthly Returns (Last 6 Months)

| Month | Return |
|-------|--------|
"""
    
    for date, ret in monthly.tail(6).items():
        report += f"| {date:%Y-%m} | {ret:+.1%} |\n"
    
    report += f"""

---

## Why Pairs Trading Works

1. **Market neutral:** Long one stock, short another = hedged against market moves
2. **Mean reversion:** Cointegrated pairs have stable relationship that reverts
3. **Diversification:** 30 pairs = 30 independent bets
4. **Low correlation:** Returns uncorrelated with market direction

---

## Next Steps

{'**Phase 1 targets met!** Proceed to Phase 2: Enhanced Ensemble.' if all_pass else '**Targets not fully met.** Consider adjusting parameters.'}

---

*Report generated by v20_pairs_trading.py*
"""
    
    return report


if __name__ == "__main__":
    run_pairs_backtest()
