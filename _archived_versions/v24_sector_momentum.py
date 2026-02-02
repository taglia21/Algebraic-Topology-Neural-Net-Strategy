#!/usr/bin/env python3
"""
V24 Momentum Strategy V3 - RADICALLY DIFFERENT DESIGN
======================================================
Cross-sectional sector rotation momentum for MAXIMUM DECORRELATION from V21.

PROBLEM: V24 v1 (0.621) and v2 (0.407) both too correlated with V21.

ROOT CAUSE ANALYSIS:
- V21 and V24 both trade the SAME universe (individual stocks)
- Both rebalance frequently (5-10 days)
- Market beta dominates both strategies

RADICAL SOLUTION: Trade DIFFERENT ASSETS ENTIRELY
- V24 trades SECTOR ETF PROXIES (simulated from stock sectors)
- V24 holds for 20+ days (vs V21's 5 days)
- V24 is LONG-ONLY RELATIVE STRENGTH (vs V21's individual stock selection)

EXPECTED DECORRELATION SOURCES:
1. Different holding period (20 days vs 5 days)
2. Different signal type (relative sector strength vs oversold RSI)
3. Aggregated sector-level decisions vs individual stock-level
4. NO volatility filter (V21 loves high-vol, V24 neutral)

Design:
- Group stocks by sector
- Calculate sector-level momentum (50-day returns)
- Long top 3 sectors (equal weight)
- Monthly rebalance (every 20 trading days)
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
logger = logging.getLogger('V24_SectorMom')


# =============================================================================
# SECTOR PROXY MAPPING (simulated sectors from stocks)
# =============================================================================

SECTOR_STOCKS = {
    'Technology': ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'AVGO', 'AMD', 'INTC', 'CRM', 'ORCL', 
                   'ADBE', 'CSCO', 'NOW', 'IBM', 'QCOM', 'AMAT', 'MU', 'INTU', 'SNPS', 'CDNS'],
    'Healthcare': ['UNH', 'JNJ', 'LLY', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY',
                   'CVS', 'AMGN', 'GILD', 'CI', 'ISRG', 'REGN', 'HUM', 'VRTX', 'MDT', 'ZTS'],
    'Financials': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'USB',
                   'PNC', 'TFC', 'COF', 'CME', 'ICE', 'MCO', 'MMC', 'AON', 'MET', 'PRU'],
    'Consumer': ['AMZN', 'TSLA', 'HD', 'WMT', 'MCD', 'NKE', 'COST', 'LOW', 'SBUX', 'TGT',
                 'DIS', 'BKNG', 'CMCSA', 'NFLX', 'TJX', 'ORLY', 'AZO', 'ROST', 'YUM', 'DPZ'],
    'Industrials': ['GE', 'CAT', 'RTX', 'HON', 'UPS', 'BA', 'DE', 'LMT', 'UNP', 'MMM',
                    'ETN', 'ITW', 'EMR', 'GD', 'NOC', 'WM', 'CSX', 'NSC', 'FDX', 'PH'],
    'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HAL',
               'DVN', 'KMI', 'WMB', 'OKE', 'FANG', 'HES', 'PXD', 'TRGP', 'BKR', 'LNG'],
    'Materials': ['LIN', 'SHW', 'APD', 'ECL', 'FCX', 'NEM', 'NUE', 'VMC', 'MLM', 'DD',
                  'DOW', 'CTVA', 'PPG', 'ALB', 'CF', 'MOS', 'STLD', 'X', 'AA', 'CLF'],
    'Utilities': ['NEE', 'SO', 'DUK', 'D', 'SRE', 'AEP', 'EXC', 'XEL', 'PCG', 'ED',
                  'WEC', 'PEG', 'AWK', 'AES', 'ES', 'CEG', 'DTE', 'FE', 'PPL', 'ETR'],
}


# =============================================================================
# CONFIGURATION
# =============================================================================

class V24Config:
    """Sector momentum configuration."""
    # Positions
    N_TOP_SECTORS = 3  # Long top 3 sectors
    STOCKS_PER_SECTOR = 10  # Top 10 most liquid from each sector
    
    # Rebalance
    REBAL_PERIOD = 20  # Monthly (vs V21's 5 days)
    
    # Momentum
    MOM_LOOKBACK = 50  # 50-day momentum for sector ranking
    
    # Entry thresholds
    MIN_MOMENTUM = 0.0  # Must have positive momentum
    
    # Costs
    COST_BPS = 10


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
    symbol_col = 'symbol' if 'symbol' in prices.columns else 'ticker'
    
    close_wide = prices.pivot(index='date', columns=symbol_col, values='close')
    volume_wide = prices.pivot(index='date', columns=symbol_col, values='volume')
    
    close_wide = close_wide.ffill(limit=5)
    volume_wide = volume_wide.ffill(limit=5)
    
    logger.info(f"Data shape: {close_wide.shape[0]} days x {close_wide.shape[1]} stocks")
    
    return close_wide, volume_wide


# =============================================================================
# SECTOR CALCULATIONS
# =============================================================================

def calculate_sector_returns(close_wide: pd.DataFrame, 
                            volume_wide: pd.DataFrame,
                            config: V24Config) -> pd.DataFrame:
    """
    Calculate equal-weight sector returns from constituent stocks.
    """
    sector_returns = {}
    
    for sector_name, stocks in SECTOR_STOCKS.items():
        # Find which stocks are in our data
        available = [s for s in stocks if s in close_wide.columns]
        
        if len(available) < 3:
            logger.warning(f"{sector_name}: Only {len(available)} stocks available, skipping")
            continue
        
        # Take top N by average dollar volume
        avg_vol = (volume_wide[available] * close_wide[available]).rolling(20).mean()
        top_stocks = avg_vol.iloc[-1].nlargest(min(config.STOCKS_PER_SECTOR, len(available))).index.tolist()
        
        # Equal-weight daily returns
        stock_returns = close_wide[top_stocks].pct_change(fill_method=None)
        sector_ret = stock_returns.mean(axis=1)
        sector_returns[sector_name] = sector_ret
    
    sector_df = pd.DataFrame(sector_returns)
    logger.info(f"Sector returns: {list(sector_df.columns)}")
    
    return sector_df


def calculate_sector_momentum(sector_returns: pd.DataFrame, 
                              config: V24Config) -> pd.DataFrame:
    """
    Calculate cumulative sector momentum over lookback period.
    """
    # Cumulative return over lookback
    sector_mom = sector_returns.rolling(config.MOM_LOOKBACK).sum()
    
    return sector_mom


# =============================================================================
# V24 SECTOR MOMENTUM STRATEGY
# =============================================================================

def run_v24_sector_backtest(close_wide: pd.DataFrame,
                            volume_wide: pd.DataFrame,
                            config: V24Config,
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Run sector rotation momentum backtest.
    
    Strategy:
    - Calculate sector momentum (50-day cumulative return)
    - Go long top 3 sectors (equal weight)
    - Rebalance monthly (every 20 trading days)
    """
    
    # Filter date range
    if start_date:
        close_wide = close_wide[close_wide.index >= start_date]
        volume_wide = volume_wide[volume_wide.index >= start_date]
    if end_date:
        close_wide = close_wide[close_wide.index <= end_date]
        volume_wide = volume_wide[volume_wide.index <= end_date]
    
    if len(close_wide) < 60:
        return pd.Series(dtype=float), pd.DataFrame()
    
    # Calculate sector returns
    sector_returns = calculate_sector_returns(close_wide, volume_wide, config)
    
    if len(sector_returns.columns) < 4:
        logger.warning("Insufficient sectors available")
        return pd.Series(dtype=float), pd.DataFrame()
    
    # Calculate sector momentum
    sector_mom = calculate_sector_momentum(sector_returns, config)
    
    # Build positions
    sector_positions = pd.DataFrame(0.0, index=sector_mom.index, columns=sector_mom.columns)
    
    # Rebalance dates
    rebal_dates = sector_mom.index[config.MOM_LOOKBACK::config.REBAL_PERIOD]
    
    current_positions = pd.Series(0.0, index=sector_mom.columns)
    
    for date in sector_mom.index[config.MOM_LOOKBACK:]:
        if date in rebal_dates:
            # Rank sectors by momentum
            mom_today = sector_mom.loc[date].dropna()
            
            # Filter: only positive momentum
            positive_mom = mom_today[mom_today > config.MIN_MOMENTUM]
            
            if len(positive_mom) >= config.N_TOP_SECTORS:
                # Top N sectors
                top_sectors = positive_mom.nlargest(config.N_TOP_SECTORS).index
                current_positions = pd.Series(0.0, index=sector_mom.columns)
                for s in top_sectors:
                    current_positions[s] = 1.0 / config.N_TOP_SECTORS
            else:
                # Not enough positive momentum sectors - stay flat
                current_positions = pd.Series(0.0, index=sector_mom.columns)
        
        sector_positions.loc[date] = current_positions
    
    # Calculate strategy returns
    strategy_returns = (sector_positions.shift(1) * sector_returns).sum(axis=1)
    
    # Transaction costs
    weight_changes = sector_positions.diff().abs().sum(axis=1)
    costs = pd.Series(0.0, index=strategy_returns.index)
    costs.loc[rebal_dates] = weight_changes.loc[rebal_dates] * (config.COST_BPS / 10000)
    
    net_returns = strategy_returns - costs
    net_returns = net_returns.dropna()
    net_returns = net_returns[net_returns.index >= sector_mom.index[config.MOM_LOOKBACK]]
    
    return net_returns, sector_positions


# =============================================================================
# V21 SIMULATION (for correlation calculation)
# =============================================================================

def simulate_v21_returns(close_wide: pd.DataFrame) -> pd.Series:
    """
    Simulate V21 mean-reversion strategy returns.
    V21: Buy oversold stocks (RSI < 35), hold 5 days.
    """
    logger.info("Simulating V21 returns...")
    
    ret_1d = close_wide.pct_change(fill_method=None)
    
    # RSI
    delta = close_wide.diff()
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)
    avg_gain = gains.rolling(14).mean()
    avg_loss = losses.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    
    # Volatility
    vol_20d = ret_1d.rolling(20).std() * np.sqrt(252)
    
    # V21 entry: RSI < 35 and high vol
    v21_entry = (rsi < 35) & (vol_20d > 0.30)
    
    # Rank by lowest RSI
    rsi_filtered = rsi.where(v21_entry)
    ranks = rsi_filtered.rank(axis=1, pct=True, ascending=True, na_option='keep')
    
    # Positions
    n_pos = 30
    positions = pd.DataFrame(0.0, index=ranks.index, columns=ranks.columns)
    
    for date in ranks.index:
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
    
    # Costs
    weight_changes = weights.diff().abs().sum(axis=1)
    costs = pd.Series(0.0, index=v21_returns.index)
    costs.loc[rebal_dates] = weight_changes.loc[rebal_dates] * 0.001
    
    v21_returns = v21_returns - costs
    v21_returns = v21_returns.dropna()
    
    if len(v21_returns) > 0:
        logger.info(f"V21: {len(v21_returns)} days, CAGR={(1+v21_returns).prod()**(252/len(v21_returns))-1:.1%}")
    
    return v21_returns


# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

def calculate_correlation(v24_returns: pd.Series, v21_returns: pd.Series) -> Dict:
    """Calculate correlation between V24 and V21."""
    common_dates = v24_returns.index.intersection(v21_returns.index)
    
    if len(common_dates) < 20:
        return {'correlation': 1.0, 'passed': False, 'error': 'Insufficient overlap'}
    
    v24_aligned = v24_returns.loc[common_dates]
    v21_aligned = v21_returns.loc[common_dates]
    
    correlation = v24_aligned.corr(v21_aligned)
    
    # Rolling correlation
    rolling_corr = v24_aligned.rolling(60).corr(v21_aligned)
    
    logger.info(f"\nCORRELATION ANALYSIS:")
    logger.info(f"  Overall correlation: {correlation:.3f}")
    logger.info(f"  Target: < 0.3")
    logger.info(f"  Status: {'✅ PASS' if abs(correlation) < 0.3 else '❌ FAIL'}")
    
    return {
        'correlation': float(correlation),
        'rolling_mean': float(rolling_corr.mean()) if not rolling_corr.isna().all() else 0.0,
        'n_days': len(common_dates),
        'passed': abs(correlation) < 0.3
    }


# =============================================================================
# COMBINED PORTFOLIO
# =============================================================================

def analyze_combined_portfolio(v24_returns: pd.Series, 
                               v21_returns: pd.Series,
                               weight_v24: float = 0.5) -> Dict:
    """Analyze 50/50 combined portfolio."""
    common_dates = v24_returns.index.intersection(v21_returns.index)
    
    if len(common_dates) < 20:
        return {'error': 'Insufficient data'}
    
    v24_aligned = v24_returns.loc[common_dates]
    v21_aligned = v21_returns.loc[common_dates]
    combined = weight_v24 * v24_aligned + (1 - weight_v24) * v21_aligned
    
    def metrics(ret, name):
        if len(ret) < 20 or ret.std() == 0:
            return {'name': name, 'cagr': 0, 'sharpe': 0, 'max_dd': 0}
        cum = (1 + ret).cumprod()
        years = len(ret) / 252
        cagr = (cum.iloc[-1]) ** (1/years) - 1 if years > 0 else 0
        sharpe = ret.mean() / ret.std() * np.sqrt(252)
        max_dd = (cum / cum.expanding().max() - 1).min()
        return {'name': name, 'cagr': cagr, 'sharpe': sharpe, 'max_dd': max_dd}
    
    m21 = metrics(v21_aligned, 'V21')
    m24 = metrics(v24_aligned, 'V24')
    mc = metrics(combined, 'Combined')
    
    logger.info(f"\nCOMBINED PORTFOLIO:")
    logger.info(f"  {'Strategy':<15} {'CAGR':>10} {'Sharpe':>10} {'MaxDD':>10}")
    logger.info(f"  {'-'*45}")
    for m in [m21, m24, mc]:
        logger.info(f"  {m['name']:<15} {m['cagr']:>10.1%} {m['sharpe']:>10.2f} {m['max_dd']:>10.1%}")
    
    return {
        'v21': m21, 
        'v24': m24, 
        'combined': mc,
        'sharpe_improvement': mc['sharpe'] - m21['sharpe'],
        'dd_improvement': mc['max_dd'] - m21['max_dd']
    }


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(returns: pd.Series) -> Dict:
    """Calculate performance metrics."""
    if len(returns) < 20 or returns.std() == 0:
        return {'error': 'Insufficient data'}
    
    cum = (1 + returns).cumprod()
    total_ret = cum.iloc[-1] - 1
    years = len(returns) / 252
    
    cagr = (1 + total_ret) ** (1/years) - 1 if years > 0 else 0
    vol = returns.std() * np.sqrt(252)
    sharpe = returns.mean() / returns.std() * np.sqrt(252)
    
    peak = cum.expanding().max()
    max_dd = (cum / peak - 1).min()
    
    win_rate = (returns > 0).mean()
    
    return {
        'cagr': float(cagr),
        'volatility': float(vol),
        'sharpe': float(sharpe),
        'max_dd': float(max_dd),
        'win_rate': float(win_rate),
        'n_days': len(returns)
    }


# =============================================================================
# WALK-FORWARD VALIDATION  
# =============================================================================

def run_walk_forward(close_wide: pd.DataFrame,
                     volume_wide: pd.DataFrame,
                     config: V24Config) -> Dict:
    """Walk-forward with 3-month train, 1-month test."""
    logger.info("\n" + "=" * 60)
    logger.info("WALK-FORWARD VALIDATION")
    logger.info("=" * 60)
    
    # Use shorter windows for limited data
    train_days = 3 * 21  # 3 months
    test_days = 1 * 21   # 1 month
    window_size = train_days + test_days
    
    dates = close_wide.index
    all_is, all_oos = [], []
    
    window = 0
    start_idx = 0
    
    while start_idx + window_size <= len(dates):
        window += 1
        
        train_end = dates[start_idx + train_days - 1]
        test_start = dates[start_idx + train_days]
        test_end = dates[min(start_idx + window_size - 1, len(dates) - 1)]
        
        is_ret, _ = run_v24_sector_backtest(
            close_wide, volume_wide, config,
            end_date=str(train_end.date())
        )
        
        oos_ret, _ = run_v24_sector_backtest(
            close_wide, volume_wide, config,
            start_date=str(test_start.date()),
            end_date=str(test_end.date())
        )
        
        if len(is_ret) > 10 and len(oos_ret) > 5:
            all_is.append(is_ret)
            all_oos.append(oos_ret)
            
            is_sr = is_ret.mean() / is_ret.std() * np.sqrt(252) if is_ret.std() > 0 else 0
            oos_sr = oos_ret.mean() / oos_ret.std() * np.sqrt(252) if oos_ret.std() > 0 else 0
            
            logger.info(f"  Window {window}: IS={is_sr:.2f}, OOS={oos_sr:.2f}")
        
        start_idx += test_days
    
    if all_is and all_oos:
        combined_is = pd.concat(all_is)
        combined_oos = pd.concat(all_oos)
        
        is_sharpe = combined_is.mean() / combined_is.std() * np.sqrt(252) if combined_is.std() > 0 else 0
        oos_sharpe = combined_oos.mean() / combined_oos.std() * np.sqrt(252) if combined_oos.std() > 0 else 0
        
        ratio = oos_sharpe / is_sharpe if is_sharpe > 0 else 0
        
        logger.info(f"\n  Combined IS Sharpe: {is_sharpe:.2f}")
        logger.info(f"  Combined OOS Sharpe: {oos_sharpe:.2f}")
        logger.info(f"  OOS/IS Ratio: {ratio:.1%}")
        
        return {
            'is_sharpe': float(is_sharpe),
            'oos_sharpe': float(oos_sharpe),
            'oos_is_ratio': float(ratio),
            'n_windows': window
        }
    
    return {'is_sharpe': 0, 'oos_sharpe': 0, 'oos_is_ratio': 0, 'n_windows': 0}


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run V24 Sector Momentum Strategy."""
    logger.info("=" * 70)
    logger.info("V24 SECTOR MOMENTUM STRATEGY V3")
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
    
    v24_returns, sector_positions = run_v24_sector_backtest(close_wide, volume_wide, config)
    
    if len(v24_returns) < 20:
        logger.error("V24 backtest returned insufficient data")
        return None
    
    v24_metrics = calculate_metrics(v24_returns)
    
    logger.info(f"\nV24 SECTOR MOMENTUM METRICS:")
    logger.info(f"  CAGR: {v24_metrics.get('cagr', 0):.1%}")
    logger.info(f"  Sharpe: {v24_metrics.get('sharpe', 0):.2f}")
    logger.info(f"  Max Drawdown: {v24_metrics.get('max_dd', 0):.1%}")
    logger.info(f"  Win Rate: {v24_metrics.get('win_rate', 0):.1%}")
    
    # === V21 SIMULATION ===
    v21_returns = simulate_v21_returns(close_wide)
    
    # === CORRELATION ===
    corr_results = calculate_correlation(v24_returns, v21_returns)
    
    # === COMBINED PORTFOLIO ===
    combined_results = analyze_combined_portfolio(v24_returns, v21_returns)
    
    # === WALK-FORWARD ===
    wf_results = run_walk_forward(close_wide, volume_wide, config)
    
    # === SAVE RESULTS ===
    results = {
        'timestamp': datetime.now().isoformat(),
        'strategy': 'V24_SectorMomentum_V3',
        'config': {
            'n_top_sectors': config.N_TOP_SECTORS,
            'stocks_per_sector': config.STOCKS_PER_SECTOR,
            'rebal_period': config.REBAL_PERIOD,
            'momentum_lookback': config.MOM_LOOKBACK
        },
        'v24_metrics': v24_metrics,
        'correlation': corr_results,
        'combined': {
            'v21': combined_results.get('v21', {}),
            'v24': combined_results.get('v24', {}),
            'combined': combined_results.get('combined', {}),
            'sharpe_improvement': float(combined_results.get('sharpe_improvement', 0)),
            'dd_improvement': float(combined_results.get('dd_improvement', 0))
        },
        'walk_forward': wf_results
    }
    
    results_dir = Path('results/v24')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / 'v24_v3_sector_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    v24_returns.to_frame('returns').to_parquet(results_dir / 'v24_v3_daily_returns.parquet')
    
    logger.info(f"\nResults saved to {results_dir}")
    
    # === FINAL SUMMARY ===
    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)
    
    checks = [
        ('V24 CAGR > 15%', v24_metrics.get('cagr', 0) > 0.15, f"{v24_metrics.get('cagr', 0):.1%}"),
        ('V24 Sharpe > 0.5', v24_metrics.get('sharpe', 0) > 0.5, f"{v24_metrics.get('sharpe', 0):.2f}"),
        ('V24 MaxDD > -25%', v24_metrics.get('max_dd', -1) > -0.25, f"{v24_metrics.get('max_dd', 0):.1%}"),
        ('★ Correlation < 0.3', abs(corr_results.get('correlation', 1)) < 0.3, f"{corr_results.get('correlation', 1):.3f}"),
        ('Combined Sharpe > V21', combined_results.get('sharpe_improvement', 0) > 0, 
         f"{combined_results.get('sharpe_improvement', 0):+.2f}"),
    ]
    
    logger.info(f"\n{'Check':<30} {'Status':<10} {'Value':<15}")
    logger.info("-" * 55)
    
    all_passed = True
    correlation_passed = False
    
    for check_name, passed, value in checks:
        status = '✅ PASS' if passed else '❌ FAIL'
        logger.info(f"{check_name:<30} {status:<10} {value:<15}")
        if '★' in check_name:
            correlation_passed = passed
        if not passed:
            all_passed = False
    
    if correlation_passed:
        logger.info("\n🎉 CORRELATION TARGET MET! V24 can diversify V21!")
    else:
        logger.info("\n⚠️ Correlation still too high - need different approach")
    
    return results


if __name__ == "__main__":
    results = main()
