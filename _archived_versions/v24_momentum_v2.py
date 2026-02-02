#!/usr/bin/env python3
"""
V24 Momentum Strategy V2 - REDESIGNED FOR LOW CORRELATION
===========================================================
Trend-following momentum designed to be UNCORRELATED with V21 mean-reversion.

KEY INSIGHT: V21 is SHORT-TERM mean reversion (5-day holds, oversold RSI).
To achieve LOW CORRELATION, V24 must be fundamentally different:

1. LONGER holding periods: 20-40 days (vs V21's 5 days)
2. TREND-FOLLOWING logic: Buy strength, not weakness
3. DIFFERENT signals: 50-day moving average crossovers, not RSI oversold
4. DIFFERENT volatility profile: Lower-volatility trending stocks

Design Philosophy:
- V21 profits from MEAN REVERSION in choppy markets
- V24 profits from TREND CONTINUATION in trending markets
- Together they cover both market regimes

Entry Conditions (ALL must be true):
1. Price > 50-day SMA (confirmed uptrend)
2. 50-day SMA > 200-day SMA (golden cross regime)
3. 20-day momentum > 0 (positive short-term momentum)
4. 50-day momentum rank in top 20% of universe (relative strength)
5. 20-day volatility < median (favor steady trends, not volatile spikes)

Exit Conditions (ANY triggers exit):
1. Price < 50-day SMA (trend broken)
2. OR 50-day SMA < 200-day SMA (death cross)
3. OR max holding period: 40 days

Target: Correlation with V21 < 0.3
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('V24_MomentumV2')


# =============================================================================
# CONFIGURATION
# =============================================================================

class V24Config:
    """Strategy configuration."""
    # Position sizing
    N_POSITIONS = 30
    MAX_SECTOR_WEIGHT = 0.25  # Max 25% in any sector
    
    # Holding period
    REBAL_PERIOD = 10  # Rebalance every 10 days (vs V21's 5)
    MAX_HOLD_DAYS = 40  # Max holding period
    
    # Entry thresholds
    MOMENTUM_LOOKBACK = 50  # 50-day momentum for ranking
    MOMENTUM_PERCENTILE = 0.20  # Top 20% momentum
    
    # Moving averages
    SMA_SHORT = 50
    SMA_LONG = 200
    
    # Volatility filter
    VOL_LOOKBACK = 20
    VOL_PERCENTILE_MAX = 0.50  # Only bottom 50% volatility stocks
    
    # Transaction costs
    COST_BPS = 10
    
    # Walk-forward
    TRAIN_MONTHS = 6
    TEST_MONTHS = 2


# =============================================================================
# DATA LOADING
# =============================================================================

def load_price_data() -> pd.DataFrame:
    """Load price data from cache."""
    cache_path = Path('cache/v17_prices/v17_prices_latest.parquet')
    if not cache_path.exists():
        raise FileNotFoundError(f"Price data not found at {cache_path}")
    
    prices = pd.read_parquet(cache_path)
    prices['date'] = pd.to_datetime(prices['date'])
    logger.info(f"Loaded {len(prices):,} price records")
    return prices


def prepare_wide_data(prices: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convert to wide format."""
    # Check column name
    symbol_col = 'symbol' if 'symbol' in prices.columns else 'ticker'
    
    close_wide = prices.pivot(index='date', columns=symbol_col, values='close')
    volume_wide = prices.pivot(index='date', columns=symbol_col, values='volume')
    
    # Forward-fill gaps
    close_wide = close_wide.ffill(limit=5)
    volume_wide = volume_wide.ffill(limit=5)
    
    logger.info(f"Data shape: {close_wide.shape[0]} days x {close_wide.shape[1]} stocks")
    
    return close_wide, volume_wide


# =============================================================================
# V21 SIMULATION (for correlation calculation)
# =============================================================================

def simulate_v21_returns(close_wide: pd.DataFrame) -> pd.Series:
    """
    Simulate V21 mean-reversion strategy returns.
    V21: Buy oversold stocks (RSI < 35), hold 5 days.
    """
    logger.info("Simulating V21 returns for correlation analysis...")
    
    ret_1d = close_wide.pct_change()
    
    # V21 signals: RSI < 35, high volatility
    delta = close_wide.diff()
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)
    avg_gain = gains.rolling(14).mean()
    avg_loss = losses.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    
    vol_20d = ret_1d.rolling(20).std() * np.sqrt(252)
    
    # V21 entry: RSI < 35 and high vol
    v21_entry = (rsi < 35) & (vol_20d > 0.30)
    
    # Rank by lowest RSI (most oversold)
    rsi_filtered = rsi.where(v21_entry)
    ranks = rsi_filtered.rank(axis=1, pct=True, ascending=True, na_option='keep')
    
    # Top 30 most oversold
    n_pos = 30
    positions = pd.DataFrame(0.0, index=ranks.index, columns=ranks.columns)
    
    for i, date in enumerate(ranks.index):
        valid_count = ranks.loc[date].notna().sum()
        if valid_count < 5:
            continue
        n_long = min(n_pos, valid_count)
        threshold = n_long / valid_count
        positions.loc[date, ranks.loc[date] <= threshold] = 1.0
    
    # Rebalance every 5 days
    rebal_period = 5
    rebal_dates = positions.index[::rebal_period]
    positions_held = positions.copy()
    positions_held.loc[~positions_held.index.isin(rebal_dates)] = np.nan
    positions_held = positions_held.ffill()
    
    # Equal weight
    counts = (positions_held > 0).sum(axis=1).replace(0, 1)
    weights = positions_held.div(counts, axis=0)
    
    # Returns
    v21_returns = (weights.shift(1) * ret_1d).sum(axis=1)
    
    # Transaction costs
    weight_changes = weights.diff().abs().sum(axis=1)
    costs = pd.Series(0.0, index=v21_returns.index)
    costs.loc[rebal_dates] = weight_changes.loc[rebal_dates] * 0.001  # 10bps
    
    v21_returns = v21_returns - costs
    v21_returns = v21_returns.dropna()
    
    logger.info(f"V21 simulation: {len(v21_returns)} days, "
                f"CAGR={((1+v21_returns).prod())**(252/len(v21_returns))-1:.1%}")
    
    return v21_returns


# =============================================================================
# V24 MOMENTUM STRATEGY (REDESIGNED)
# =============================================================================

def calculate_v24_signals(close_wide: pd.DataFrame, 
                          volume_wide: pd.DataFrame,
                          config: V24Config) -> pd.DataFrame:
    """
    Calculate V24 momentum signals - FUNDAMENTALLY DIFFERENT from V21.
    
    V21 buys: Oversold (RSI<35), high volatility, short-term
    V24 buys: Strong trends, low volatility, longer-term
    """
    
    ret_1d = close_wide.pct_change()
    
    # === TREND INDICATORS (not RSI) ===
    
    # 50-day and 200-day SMAs
    sma_50 = close_wide.rolling(config.SMA_SHORT).mean()
    sma_200 = close_wide.rolling(config.SMA_LONG).mean()
    
    # Price above 50-day SMA (uptrend)
    above_sma50 = close_wide > sma_50
    
    # Golden cross: 50-day > 200-day
    golden_cross = sma_50 > sma_200
    
    # 50-day momentum (for ranking)
    mom_50d = close_wide.pct_change(config.MOMENTUM_LOOKBACK)
    
    # 20-day momentum (positive short-term)
    mom_20d = close_wide.pct_change(20)
    positive_momentum = mom_20d > 0
    
    # === VOLATILITY FILTER (opposite of V21) ===
    # V21 likes HIGH volatility stocks (mean-revert more)
    # V24 likes LOW volatility stocks (steady trends)
    
    vol_20d = ret_1d.rolling(config.VOL_LOOKBACK).std() * np.sqrt(252)
    vol_median = vol_20d.median(axis=1)
    low_volatility = vol_20d.lt(vol_median, axis=0)  # Below median volatility
    
    # === LIQUIDITY FILTER ===
    avg_volume = volume_wide.rolling(20).mean()
    avg_dollar_vol = avg_volume * close_wide
    liquid = avg_dollar_vol > 5_000_000  # $5M+ daily volume
    
    # === COMBINED ENTRY SIGNAL ===
    # All conditions must be true
    valid_entry = (
        above_sma50 &           # Price > 50-day SMA
        golden_cross &          # 50-day > 200-day SMA
        positive_momentum &     # 20-day momentum > 0
        low_volatility &        # Below median volatility
        liquid                  # Sufficient liquidity
    )
    
    # Rank by 50-day momentum (buy strongest trends)
    mom_filtered = mom_50d.where(valid_entry)
    
    # Rank: higher momentum = lower rank number (top performers)
    ranks = mom_filtered.rank(axis=1, ascending=False, pct=True, na_option='keep')
    
    logger.info(f"V24 signals: avg valid stocks/day = {valid_entry.sum(axis=1).mean():.1f}")
    
    return ranks, valid_entry, sma_50, close_wide


def run_v24_backtest(close_wide: pd.DataFrame,
                     volume_wide: pd.DataFrame,
                     config: V24Config,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Run V24 momentum backtest.
    
    Key differences from V21:
    - Rebalance every 10 days (vs 5)
    - Buy top momentum (vs bottom RSI)
    - Low volatility preference (vs high volatility)
    - Trend-following exit (vs time-based)
    """
    
    # Filter date range if specified
    if start_date:
        close_wide = close_wide[close_wide.index >= start_date]
        volume_wide = volume_wide[volume_wide.index >= start_date]
    if end_date:
        close_wide = close_wide[close_wide.index <= end_date]
        volume_wide = volume_wide[volume_wide.index <= end_date]
    
    ret_1d = close_wide.pct_change()
    
    # Get signals
    ranks, valid_entry, sma_50, prices = calculate_v24_signals(
        close_wide, volume_wide, config
    )
    
    # Build positions - top N by momentum rank
    positions = pd.DataFrame(0.0, index=ranks.index, columns=ranks.columns)
    
    for date in ranks.index:
        valid_count = ranks.loc[date].notna().sum()
        if valid_count < 5:
            continue
        n_long = min(config.N_POSITIONS, valid_count)
        threshold = config.MOMENTUM_PERCENTILE  # Top 20%
        positions.loc[date, ranks.loc[date] <= threshold] = 1.0
        
        # Limit to exactly N positions
        pos_count = (positions.loc[date] > 0).sum()
        if pos_count > config.N_POSITIONS:
            # Keep only top N by rank
            ranked_pos = ranks.loc[date][positions.loc[date] > 0].nsmallest(config.N_POSITIONS)
            positions.loc[date] = 0.0
            positions.loc[date, ranked_pos.index] = 1.0
    
    # Rebalance every N days
    rebal_dates = positions.index[::config.REBAL_PERIOD]
    positions_held = positions.copy()
    
    # Hold positions between rebalances, but exit if trend breaks
    for i in range(len(positions.index)):
        date = positions.index[i]
        
        if date in rebal_dates:
            continue
        
        # Carry forward positions from previous day
        if i > 0:
            prev_date = positions.index[i-1]
            positions_held.loc[date] = positions_held.loc[prev_date]
            
            # Exit positions where price < 50-day SMA (trend broken)
            trend_broken = prices.loc[date] < sma_50.loc[date]
            positions_held.loc[date, trend_broken] = 0.0
    
    # Equal weight
    counts = (positions_held > 0).sum(axis=1).replace(0, 1)
    weights = positions_held.div(counts, axis=0)
    
    # Calculate returns
    strategy_daily = (weights.shift(1) * ret_1d).sum(axis=1)
    
    # Transaction costs
    weight_changes = weights.diff().abs().sum(axis=1)
    costs = weight_changes * (config.COST_BPS / 10000)
    
    net_returns = strategy_daily - costs
    net_returns = net_returns.dropna()
    
    return net_returns, weights


# =============================================================================
# WALK-FORWARD VALIDATION
# =============================================================================

def run_walk_forward(close_wide: pd.DataFrame,
                     volume_wide: pd.DataFrame,
                     config: V24Config) -> Dict:
    """
    Walk-forward validation with 6-month train, 2-month test windows.
    """
    logger.info("=" * 60)
    logger.info("WALK-FORWARD VALIDATION")
    logger.info("=" * 60)
    
    dates = close_wide.index
    total_days = len(dates)
    
    # Calculate window sizes
    train_days = config.TRAIN_MONTHS * 21  # ~21 trading days/month
    test_days = config.TEST_MONTHS * 21
    window_size = train_days + test_days
    
    all_is_returns = []
    all_oos_returns = []
    
    window_num = 0
    start_idx = 0
    
    while start_idx + window_size <= total_days:
        window_num += 1
        
        train_start = dates[start_idx]
        train_end = dates[start_idx + train_days - 1]
        test_start = dates[start_idx + train_days]
        test_end = dates[min(start_idx + window_size - 1, total_days - 1)]
        
        # In-sample
        is_returns, _ = run_v24_backtest(
            close_wide, volume_wide, config,
            start_date=str(train_start.date()),
            end_date=str(train_end.date())
        )
        
        # Out-of-sample
        oos_returns, _ = run_v24_backtest(
            close_wide, volume_wide, config,
            start_date=str(test_start.date()),
            end_date=str(test_end.date())
        )
        
        if len(is_returns) > 10 and len(oos_returns) > 10:
            all_is_returns.append(is_returns)
            all_oos_returns.append(oos_returns)
            
            is_sharpe = is_returns.mean() / is_returns.std() * np.sqrt(252) if is_returns.std() > 0 else 0
            oos_sharpe = oos_returns.mean() / oos_returns.std() * np.sqrt(252) if oos_returns.std() > 0 else 0
            
            logger.info(f"Window {window_num}: Train {train_start.date()} to {train_end.date()}, "
                       f"Test {test_start.date()} to {test_end.date()}")
            logger.info(f"  IS Sharpe: {is_sharpe:.2f}, OOS Sharpe: {oos_sharpe:.2f}")
        
        # Move to next window
        start_idx += test_days
    
    # Combine all returns
    if all_is_returns and all_oos_returns:
        combined_is = pd.concat(all_is_returns)
        combined_oos = pd.concat(all_oos_returns)
        
        is_sharpe = combined_is.mean() / combined_is.std() * np.sqrt(252) if combined_is.std() > 0 else 0
        oos_sharpe = combined_oos.mean() / combined_oos.std() * np.sqrt(252) if combined_oos.std() > 0 else 0
        
        oos_is_ratio = oos_sharpe / is_sharpe if is_sharpe > 0 else 0
        
        logger.info(f"\nWALK-FORWARD SUMMARY:")
        logger.info(f"  Combined IS Sharpe: {is_sharpe:.2f}")
        logger.info(f"  Combined OOS Sharpe: {oos_sharpe:.2f}")
        logger.info(f"  OOS/IS Ratio: {oos_is_ratio:.1%}")
        
        return {
            'is_returns': combined_is,
            'oos_returns': combined_oos,
            'is_sharpe': is_sharpe,
            'oos_sharpe': oos_sharpe,
            'oos_is_ratio': oos_is_ratio,
            'n_windows': window_num
        }
    
    return {'error': 'Insufficient data for walk-forward'}


# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

def calculate_correlation(v24_returns: pd.Series, v21_returns: pd.Series) -> Dict:
    """
    Calculate correlation between V24 and V21 returns.
    TARGET: Correlation < 0.3
    """
    # Align dates
    common_dates = v24_returns.index.intersection(v21_returns.index)
    
    if len(common_dates) < 20:
        return {'error': 'Insufficient overlapping dates'}
    
    v24_aligned = v24_returns.loc[common_dates]
    v21_aligned = v21_returns.loc[common_dates]
    
    # Pearson correlation
    correlation = v24_aligned.corr(v21_aligned)
    
    # Rolling correlation (60-day windows)
    rolling_corr = v24_aligned.rolling(60).corr(v21_aligned)
    
    logger.info(f"\nCORRELATION ANALYSIS:")
    logger.info(f"  Overall correlation: {correlation:.3f}")
    logger.info(f"  Target: < 0.3")
    logger.info(f"  Status: {'✅ PASS' if correlation < 0.3 else '❌ FAIL (too high)'}")
    
    return {
        'correlation': correlation,
        'rolling_corr_mean': rolling_corr.mean(),
        'rolling_corr_std': rolling_corr.std(),
        'n_overlapping_days': len(common_dates),
        'passed': correlation < 0.3
    }


# =============================================================================
# COMBINED PORTFOLIO ANALYSIS
# =============================================================================

def analyze_combined_portfolio(v24_returns: pd.Series, 
                                v21_returns: pd.Series,
                                weight_v24: float = 0.5) -> Dict:
    """
    Analyze 50/50 combined portfolio of V24 and V21.
    """
    common_dates = v24_returns.index.intersection(v21_returns.index)
    
    if len(common_dates) < 20:
        return {'error': 'Insufficient data'}
    
    v24_aligned = v24_returns.loc[common_dates]
    v21_aligned = v21_returns.loc[common_dates]
    
    # Combined returns
    combined = weight_v24 * v24_aligned + (1 - weight_v24) * v21_aligned
    
    # Metrics
    def calc_metrics(returns: pd.Series, name: str) -> Dict:
        if len(returns) < 20:
            return {}
        
        cumulative = (1 + returns).cumprod()
        total_return = cumulative.iloc[-1] - 1
        years = len(returns) / 252
        
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        vol = returns.std() * np.sqrt(252)
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        peak = cumulative.expanding().max()
        dd = (cumulative - peak) / peak
        max_dd = dd.min()
        
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        
        return {
            'name': name,
            'cagr': cagr,
            'volatility': vol,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'calmar': calmar
        }
    
    v24_metrics = calc_metrics(v24_aligned, 'V24')
    v21_metrics = calc_metrics(v21_aligned, 'V21')
    combined_metrics = calc_metrics(combined, 'Combined (50/50)')
    
    logger.info(f"\nCOMBINED PORTFOLIO ANALYSIS:")
    logger.info(f"  {'Strategy':<20} {'CAGR':>10} {'Sharpe':>10} {'MaxDD':>10}")
    logger.info(f"  {'-'*50}")
    
    for m in [v21_metrics, v24_metrics, combined_metrics]:
        logger.info(f"  {m['name']:<20} {m['cagr']:>10.1%} {m['sharpe']:>10.2f} {m['max_drawdown']:>10.1%}")
    
    # Improvement from diversification
    sharpe_improvement = combined_metrics['sharpe'] - v21_metrics['sharpe']
    dd_improvement = combined_metrics['max_drawdown'] - v21_metrics['max_drawdown']
    
    logger.info(f"\n  Diversification benefit:")
    logger.info(f"    Sharpe improvement: {sharpe_improvement:+.2f}")
    logger.info(f"    MaxDD improvement: {dd_improvement:+.1%}")
    
    return {
        'v21': v21_metrics,
        'v24': v24_metrics,
        'combined': combined_metrics,
        'sharpe_improvement': sharpe_improvement,
        'dd_improvement': dd_improvement
    }


# =============================================================================
# METRICS CALCULATION
# =============================================================================

def calculate_metrics(returns: pd.Series) -> Dict:
    """Calculate comprehensive performance metrics."""
    if len(returns) < 20:
        return {'error': 'Insufficient data'}
    
    cumulative = (1 + returns).cumprod()
    total_return = cumulative.iloc[-1] - 1
    
    trading_days = len(returns)
    years = trading_days / 252
    
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    annual_vol = returns.std() * np.sqrt(252)
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    
    # Drawdown
    peak = cumulative.expanding().max()
    dd = (cumulative - peak) / peak
    max_dd = dd.min()
    
    # Win rate
    win_rate = (returns > 0).mean()
    
    # Profit factor
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    
    # Calmar ratio
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
    # Monthly returns analysis
    monthly = returns.resample('ME').sum()
    best_month = monthly.max()
    worst_month = monthly.min()
    pct_positive_months = (monthly > 0).mean()
    
    return {
        'cagr': cagr,
        'annual_volatility': annual_vol,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'calmar_ratio': calmar,
        'total_return': total_return,
        'trading_days': trading_days,
        'best_month': best_month,
        'worst_month': worst_month,
        'pct_positive_months': pct_positive_months
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run V24 Momentum Strategy V2."""
    logger.info("=" * 70)
    logger.info("V24 MOMENTUM STRATEGY V2 - REDESIGNED FOR LOW CORRELATION")
    logger.info("=" * 70)
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    
    config = V24Config()
    
    # Load data
    prices = load_price_data()
    close_wide, volume_wide = prepare_wide_data(prices)
    
    # === FULL BACKTEST ===
    logger.info("\n" + "=" * 60)
    logger.info("FULL PERIOD BACKTEST")
    logger.info("=" * 60)
    
    v24_returns, weights = run_v24_backtest(close_wide, volume_wide, config)
    
    # Calculate V24 metrics
    v24_metrics = calculate_metrics(v24_returns)
    
    logger.info(f"\nV24 STANDALONE METRICS:")
    logger.info(f"  CAGR: {v24_metrics['cagr']:.1%}")
    logger.info(f"  Sharpe: {v24_metrics['sharpe_ratio']:.2f}")
    logger.info(f"  Max Drawdown: {v24_metrics['max_drawdown']:.1%}")
    logger.info(f"  Win Rate: {v24_metrics['win_rate']:.1%}")
    logger.info(f"  Profit Factor: {v24_metrics['profit_factor']:.2f}")
    
    # === SIMULATE V21 FOR CORRELATION ===
    v21_returns = simulate_v21_returns(close_wide)
    
    # === CORRELATION ANALYSIS ===
    logger.info("\n" + "=" * 60)
    logger.info("CORRELATION ANALYSIS")
    logger.info("=" * 60)
    
    corr_results = calculate_correlation(v24_returns, v21_returns)
    
    # === COMBINED PORTFOLIO ===
    logger.info("\n" + "=" * 60)
    logger.info("COMBINED PORTFOLIO (50% V21 + 50% V24)")
    logger.info("=" * 60)
    
    combined_results = analyze_combined_portfolio(v24_returns, v21_returns)
    
    # === WALK-FORWARD VALIDATION ===
    wf_results = run_walk_forward(close_wide, volume_wide, config)
    
    # === SAVE RESULTS ===
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_positions': config.N_POSITIONS,
            'rebal_period': config.REBAL_PERIOD,
            'momentum_lookback': config.MOMENTUM_LOOKBACK,
            'sma_short': config.SMA_SHORT,
            'sma_long': config.SMA_LONG
        },
        'v24_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                        for k, v in v24_metrics.items()},
        'correlation': {
            'v24_v21_correlation': float(corr_results.get('correlation', 0)),
            'passed': corr_results.get('passed', False)
        },
        'walk_forward': {
            'is_sharpe': float(wf_results.get('is_sharpe', 0)),
            'oos_sharpe': float(wf_results.get('oos_sharpe', 0)),
            'oos_is_ratio': float(wf_results.get('oos_is_ratio', 0)),
            'n_windows': wf_results.get('n_windows', 0)
        },
        'combined_portfolio': {
            'combined_sharpe': float(combined_results.get('combined', {}).get('sharpe', 0)),
            'combined_cagr': float(combined_results.get('combined', {}).get('cagr', 0)),
            'combined_max_dd': float(combined_results.get('combined', {}).get('max_drawdown', 0)),
            'sharpe_improvement': float(combined_results.get('sharpe_improvement', 0))
        }
    }
    
    # Save to file
    results_dir = Path('results/v24')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / 'v24_v2_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save daily returns for future use
    v24_returns.to_frame('returns').to_parquet(results_dir / 'v24_daily_returns.parquet')
    
    logger.info(f"\nResults saved to {results_dir}")
    
    # === FINAL SUMMARY ===
    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)
    
    # Check success criteria
    checks = [
        ('V24 CAGR > 25%', v24_metrics['cagr'] > 0.25, f"{v24_metrics['cagr']:.1%}"),
        ('V24 Sharpe > 0.8', v24_metrics['sharpe_ratio'] > 0.8, f"{v24_metrics['sharpe_ratio']:.2f}"),
        ('V24 MaxDD > -30%', v24_metrics['max_drawdown'] > -0.30, f"{v24_metrics['max_drawdown']:.1%}"),
        ('Correlation < 0.3', corr_results.get('correlation', 1) < 0.3, f"{corr_results.get('correlation', 1):.3f}"),
        ('Combined Sharpe > V21', combined_results.get('sharpe_improvement', 0) > 0, 
         f"{combined_results.get('sharpe_improvement', 0):+.2f}"),
        ('OOS/IS > 70%', wf_results.get('oos_is_ratio', 0) > 0.70, 
         f"{wf_results.get('oos_is_ratio', 0):.1%}")
    ]
    
    logger.info(f"\n{'Check':<30} {'Status':<10} {'Value':<15}")
    logger.info("-" * 55)
    
    all_passed = True
    for check_name, passed, value in checks:
        status = '✅ PASS' if passed else '❌ FAIL'
        logger.info(f"{check_name:<30} {status:<10} {value:<15}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("\n🎉 ALL CRITERIA MET - V24 is ready for deployment!")
    else:
        logger.info("\n⚠️ Some criteria not met - review strategy design")
    
    return results


if __name__ == "__main__":
    results = main()
