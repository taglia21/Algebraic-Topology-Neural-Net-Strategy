#!/usr/bin/env python3
"""
V24 Time-Series Momentum V4 - MARKET-NEUTRAL LONG/SHORT
=========================================================
TRULY DIFFERENT from V21 to achieve correlation < 0.3

ANALYSIS: Why previous V24 versions failed:
- V1 (0.621): Same stocks, same timeframe as V21
- V2 (0.407): Same stocks, slightly different signals
- V3 (0.563): Sector rotation, but still long-only → high beta

ROOT CAUSE: MARKET BETA
- V21 is net-long → profits when market rises
- All V24 versions were net-long → also profit when market rises
- Correlation driven by shared market exposure

SOLUTION: MARKET-NEUTRAL LONG/SHORT
- V24 is DOLLAR-NEUTRAL: 50% long, 50% short
- V24 profits from RELATIVE momentum spread
- V24 has ~0 market beta → decorrelated from V21

Strategy Design:
1. Rank all stocks by 90-day momentum
2. LONG top 20% (winners)
3. SHORT bottom 20% (losers)
4. Equal-weight within long/short legs
5. Rebalance monthly (20 trading days)

Expected decorrelation:
- V21: Net long, profits from oversold bounces (market up)
- V24: Market neutral, profits from momentum spread (any market)
- Low/zero market beta should give correlation < 0.3
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
logger = logging.getLogger('V24_MarketNeutral')


# =============================================================================
# CONFIGURATION
# =============================================================================

class V24Config:
    """Market-neutral momentum configuration."""
    # Position sizing
    LONG_PCT = 0.20   # Long top 20%
    SHORT_PCT = 0.20  # Short bottom 20%
    MAX_POSITIONS_PER_LEG = 30  # Cap at 30 long, 30 short
    
    # Holding period
    REBAL_PERIOD = 20  # Monthly rebalance
    
    # Momentum
    MOM_LOOKBACK = 90  # 90-day momentum (longer term)
    
    # Filters
    MIN_PRICE = 10.0
    MIN_DOLLAR_VOLUME = 5_000_000
    
    # Costs (higher for shorts)
    COST_BPS_LONG = 10
    COST_BPS_SHORT = 25  # Borrow cost + execution


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
# V24 MARKET-NEUTRAL MOMENTUM
# =============================================================================

def run_v24_market_neutral(close_wide: pd.DataFrame,
                           volume_wide: pd.DataFrame,
                           config: V24Config,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Run market-neutral long/short momentum strategy.
    
    Key design:
    - 50% long (top momentum) + 50% short (bottom momentum)
    - Net market exposure ~0 (dollar neutral)
    - Profit from momentum spread, not market direction
    """
    
    # Filter dates
    if start_date:
        close_wide = close_wide[close_wide.index >= start_date]
        volume_wide = volume_wide[volume_wide.index >= start_date]
    if end_date:
        close_wide = close_wide[close_wide.index <= end_date]
        volume_wide = volume_wide[volume_wide.index <= end_date]
    
    if len(close_wide) < config.MOM_LOOKBACK + 20:
        return pd.Series(dtype=float), pd.DataFrame()
    
    ret_1d = close_wide.pct_change(fill_method=None)
    
    # === FILTERS ===
    # Price filter
    price_ok = close_wide > config.MIN_PRICE
    
    # Liquidity filter
    dollar_vol = close_wide * volume_wide
    avg_dollar_vol = dollar_vol.rolling(20).mean()
    liquid = avg_dollar_vol > config.MIN_DOLLAR_VOLUME
    
    # Combined filter
    tradeable = price_ok & liquid
    
    # === MOMENTUM ===
    # 90-day total return
    momentum = close_wide.pct_change(config.MOM_LOOKBACK, fill_method=None)
    
    # Apply tradeable filter
    momentum_filtered = momentum.where(tradeable)
    
    # Rank by momentum (higher = better)
    # pct=True gives percentile rank 0-1
    ranks = momentum_filtered.rank(axis=1, pct=True, ascending=True, na_option='keep')
    
    # === BUILD POSITIONS ===
    long_weights = pd.DataFrame(0.0, index=ranks.index, columns=ranks.columns)
    short_weights = pd.DataFrame(0.0, index=ranks.index, columns=ranks.columns)
    
    # Rebalance dates (after momentum lookback window)
    valid_dates = ranks.index[config.MOM_LOOKBACK:]
    rebal_dates = valid_dates[::config.REBAL_PERIOD]
    
    current_long = pd.Series(0.0, index=ranks.columns)
    current_short = pd.Series(0.0, index=ranks.columns)
    
    for date in valid_dates:
        if date in rebal_dates:
            day_ranks = ranks.loc[date].dropna()
            n_stocks = len(day_ranks)
            
            if n_stocks < 20:
                # Not enough stocks - stay flat
                current_long = pd.Series(0.0, index=ranks.columns)
                current_short = pd.Series(0.0, index=ranks.columns)
            else:
                # Long: top 20% by momentum
                long_threshold = 1 - config.LONG_PCT  # e.g., 0.80
                long_stocks = day_ranks[day_ranks >= long_threshold].index
                n_long = min(len(long_stocks), config.MAX_POSITIONS_PER_LEG)
                long_stocks = day_ranks.nlargest(n_long).index
                
                # Short: bottom 20% by momentum
                short_threshold = config.SHORT_PCT  # e.g., 0.20
                short_stocks = day_ranks[day_ranks <= short_threshold].index
                n_short = min(len(short_stocks), config.MAX_POSITIONS_PER_LEG)
                short_stocks = day_ranks.nsmallest(n_short).index
                
                # Equal weight within each leg
                current_long = pd.Series(0.0, index=ranks.columns)
                current_short = pd.Series(0.0, index=ranks.columns)
                
                if len(long_stocks) > 0:
                    current_long[long_stocks] = 0.5 / len(long_stocks)  # 50% gross long
                if len(short_stocks) > 0:
                    current_short[short_stocks] = 0.5 / len(short_stocks)  # 50% gross short
        
        long_weights.loc[date] = current_long
        short_weights.loc[date] = current_short
    
    # === CALCULATE RETURNS ===
    # Long leg returns
    long_returns = (long_weights.shift(1) * ret_1d).sum(axis=1)
    
    # Short leg returns (inverted - profit when stocks go down)
    short_returns = -(short_weights.shift(1) * ret_1d).sum(axis=1)
    
    # Combined returns
    strategy_returns = long_returns + short_returns
    
    # === TRANSACTION COSTS ===
    long_turnover = long_weights.diff().abs().sum(axis=1)
    short_turnover = short_weights.diff().abs().sum(axis=1)
    
    costs = pd.Series(0.0, index=strategy_returns.index)
    costs.loc[rebal_dates] = (
        long_turnover.loc[rebal_dates] * (config.COST_BPS_LONG / 10000) +
        short_turnover.loc[rebal_dates] * (config.COST_BPS_SHORT / 10000)
    )
    
    net_returns = strategy_returns - costs
    net_returns = net_returns.loc[valid_dates]
    net_returns = net_returns.dropna()
    
    # Calculate net exposure for logging
    net_exposure = (long_weights.sum(axis=1) - short_weights.sum(axis=1))
    avg_net_exposure = net_exposure.loc[valid_dates].mean()
    logger.info(f"Average net exposure: {avg_net_exposure:.1%} (target: 0%)")
    
    # Combined weights for reporting
    combined_weights = long_weights - short_weights
    
    return net_returns, combined_weights


# =============================================================================
# V21 SIMULATION
# =============================================================================

def simulate_v21_returns(close_wide: pd.DataFrame) -> pd.Series:
    """Simulate V21 mean-reversion returns."""
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
    
    # V21 entry
    v21_entry = (rsi < 35) & (vol_20d > 0.30)
    
    rsi_filtered = rsi.where(v21_entry)
    ranks = rsi_filtered.rank(axis=1, pct=True, ascending=True, na_option='keep')
    
    n_pos = 30
    positions = pd.DataFrame(0.0, index=ranks.index, columns=ranks.columns)
    
    for date in ranks.index:
        valid_count = ranks.loc[date].notna().sum()
        if valid_count < 5:
            continue
        n_long = min(n_pos, valid_count)
        threshold = n_long / valid_count
        positions.loc[date, ranks.loc[date] <= threshold] = 1.0
    
    rebal_period = 5
    rebal_dates = positions.index[::rebal_period]
    positions_held = positions.copy()
    positions_held.loc[~positions_held.index.isin(rebal_dates)] = np.nan
    positions_held = positions_held.ffill()
    
    counts = (positions_held > 0).sum(axis=1).replace(0, 1)
    weights = positions_held.div(counts, axis=0)
    
    v21_returns = (weights.shift(1) * ret_1d).sum(axis=1)
    
    weight_changes = weights.diff().abs().sum(axis=1)
    costs = pd.Series(0.0, index=v21_returns.index)
    costs.loc[rebal_dates] = weight_changes.loc[rebal_dates] * 0.001
    
    v21_returns = v21_returns - costs
    v21_returns = v21_returns.dropna()
    
    if len(v21_returns) > 0:
        logger.info(f"V21: CAGR={(1+v21_returns).prod()**(252/len(v21_returns))-1:.1%}")
    
    return v21_returns


# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

def calculate_correlation(v24_returns: pd.Series, v21_returns: pd.Series) -> Dict:
    """Calculate correlation."""
    common = v24_returns.index.intersection(v21_returns.index)
    
    if len(common) < 30:
        return {'correlation': 0, 'passed': False, 'error': 'Insufficient overlap'}
    
    v24 = v24_returns.loc[common]
    v21 = v21_returns.loc[common]
    
    corr = v24.corr(v21)
    
    # Also compute beta to market (using V21 as proxy)
    cov = v24.cov(v21)
    var_v21 = v21.var()
    beta = cov / var_v21 if var_v21 > 0 else 0
    
    logger.info(f"\n★ CORRELATION ANALYSIS:")
    logger.info(f"  Correlation: {corr:.3f}")
    logger.info(f"  Beta to V21: {beta:.3f}")
    logger.info(f"  Target: correlation < 0.3")
    logger.info(f"  Status: {'✅ PASS!' if abs(corr) < 0.3 else '❌ FAIL'}")
    
    return {
        'correlation': float(corr),
        'beta': float(beta),
        'n_days': len(common),
        'passed': abs(corr) < 0.3
    }


# =============================================================================
# COMBINED PORTFOLIO
# =============================================================================

def analyze_combined(v24_returns: pd.Series, v21_returns: pd.Series) -> Dict:
    """Analyze combined portfolio."""
    common = v24_returns.index.intersection(v21_returns.index)
    
    if len(common) < 30:
        return {}
    
    v24 = v24_returns.loc[common]
    v21 = v21_returns.loc[common]
    combined = 0.5 * v24 + 0.5 * v21
    
    def metrics(ret, name):
        if len(ret) < 20 or ret.std() == 0:
            return {'name': name, 'cagr': 0, 'sharpe': 0, 'max_dd': 0, 'vol': 0}
        cum = (1 + ret).cumprod()
        years = len(ret) / 252
        cagr = cum.iloc[-1] ** (1/years) - 1 if years > 0 else 0
        vol = ret.std() * np.sqrt(252)
        sharpe = ret.mean() / ret.std() * np.sqrt(252)
        max_dd = (cum / cum.expanding().max() - 1).min()
        return {'name': name, 'cagr': cagr, 'sharpe': sharpe, 'max_dd': max_dd, 'vol': vol}
    
    m21 = metrics(v21, 'V21 (long-only)')
    m24 = metrics(v24, 'V24 (L/S neutral)')
    mc = metrics(combined, 'Combined 50/50')
    
    logger.info(f"\nCOMBINED PORTFOLIO:")
    logger.info(f"  {'Strategy':<20} {'CAGR':>8} {'Vol':>8} {'Sharpe':>8} {'MaxDD':>8}")
    logger.info(f"  {'-'*55}")
    for m in [m21, m24, mc]:
        logger.info(f"  {m['name']:<20} {m['cagr']:>8.1%} {m['vol']:>8.1%} {m['sharpe']:>8.2f} {m['max_dd']:>8.1%}")
    
    sharpe_imp = mc['sharpe'] - m21['sharpe']
    dd_imp = mc['max_dd'] - m21['max_dd']
    
    logger.info(f"\n  Diversification benefit:")
    logger.info(f"    Sharpe: {sharpe_imp:+.2f}")
    logger.info(f"    MaxDD: {dd_imp:+.1%}")
    
    return {
        'v21': m21, 
        'v24': m24, 
        'combined': mc,
        'sharpe_improvement': float(sharpe_imp),
        'dd_improvement': float(dd_imp)
    }


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(returns: pd.Series) -> Dict:
    """Calculate strategy metrics."""
    if len(returns) < 30 or returns.std() == 0:
        return {'error': 'Insufficient data'}
    
    cum = (1 + returns).cumprod()
    years = len(returns) / 252
    
    cagr = cum.iloc[-1] ** (1/years) - 1 if years > 0 else 0
    vol = returns.std() * np.sqrt(252)
    sharpe = returns.mean() / returns.std() * np.sqrt(252)
    max_dd = (cum / cum.expanding().max() - 1).min()
    
    win_rate = (returns > 0).mean()
    
    # Sortino ratio
    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = returns.mean() * 252 / downside if downside > 0 else 0
    
    return {
        'cagr': float(cagr),
        'volatility': float(vol),
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'max_dd': float(max_dd),
        'win_rate': float(win_rate),
        'n_days': len(returns)
    }


# =============================================================================
# WALK-FORWARD
# =============================================================================

def run_walk_forward(close_wide: pd.DataFrame,
                     volume_wide: pd.DataFrame,
                     config: V24Config) -> Dict:
    """Walk-forward validation."""
    logger.info("\n" + "=" * 60)
    logger.info("WALK-FORWARD VALIDATION")
    logger.info("=" * 60)
    
    train_days = 4 * 21  # 4 months
    test_days = 1 * 21   # 1 month
    
    dates = close_wide.index
    all_oos = []
    
    window = 0
    start_idx = 0
    
    while start_idx + train_days + test_days <= len(dates):
        window += 1
        
        test_start = dates[start_idx + train_days]
        test_end = dates[min(start_idx + train_days + test_days - 1, len(dates) - 1)]
        
        # OOS only (strategy has no parameters to fit)
        oos_ret, _ = run_v24_market_neutral(
            close_wide, volume_wide, config,
            start_date=str(test_start.date()),
            end_date=str(test_end.date())
        )
        
        if len(oos_ret) > 5:
            all_oos.append(oos_ret)
        
        start_idx += test_days
    
    if all_oos:
        combined_oos = pd.concat(all_oos)
        oos_sharpe = combined_oos.mean() / combined_oos.std() * np.sqrt(252) if combined_oos.std() > 0 else 0
        
        logger.info(f"  Windows: {window}")
        logger.info(f"  OOS Sharpe: {oos_sharpe:.2f}")
        
        return {'oos_sharpe': float(oos_sharpe), 'n_windows': window}
    
    return {'oos_sharpe': 0, 'n_windows': 0}


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run V24 Market-Neutral Momentum Strategy."""
    logger.info("=" * 70)
    logger.info("V24 MARKET-NEUTRAL LONG/SHORT MOMENTUM V4")
    logger.info("=" * 70)
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("\nDesign rationale:")
    logger.info("  - V21 is NET-LONG → correlated with market")
    logger.info("  - V24 is MARKET-NEUTRAL (50% long, 50% short)")
    logger.info("  - V24 profits from momentum SPREAD, not market direction")
    logger.info("  - Expected correlation < 0.3 due to ~0 market beta")
    
    config = V24Config()
    
    # Load data
    prices = load_price_data()
    close_wide, volume_wide = prepare_wide_data(prices)
    
    # === FULL BACKTEST ===
    logger.info("\n" + "=" * 60)
    logger.info("FULL PERIOD BACKTEST")
    logger.info("=" * 60)
    
    v24_returns, weights = run_v24_market_neutral(close_wide, volume_wide, config)
    
    if len(v24_returns) < 30:
        logger.error("Insufficient data for V24 backtest")
        return None
    
    v24_metrics = calculate_metrics(v24_returns)
    
    logger.info(f"\nV24 MARKET-NEUTRAL METRICS:")
    logger.info(f"  CAGR: {v24_metrics.get('cagr', 0):.1%}")
    logger.info(f"  Volatility: {v24_metrics.get('volatility', 0):.1%}")
    logger.info(f"  Sharpe: {v24_metrics.get('sharpe', 0):.2f}")
    logger.info(f"  Sortino: {v24_metrics.get('sortino', 0):.2f}")
    logger.info(f"  Max Drawdown: {v24_metrics.get('max_dd', 0):.1%}")
    logger.info(f"  Win Rate: {v24_metrics.get('win_rate', 0):.1%}")
    
    # === V21 SIMULATION ===
    v21_returns = simulate_v21_returns(close_wide)
    
    # === CORRELATION ===
    corr_results = calculate_correlation(v24_returns, v21_returns)
    
    # === COMBINED ===
    combined_results = analyze_combined(v24_returns, v21_returns)
    
    # === WALK-FORWARD ===
    wf_results = run_walk_forward(close_wide, volume_wide, config)
    
    # === SAVE RESULTS ===
    results = {
        'timestamp': datetime.now().isoformat(),
        'strategy': 'V24_MarketNeutral_V4',
        'design': 'Long/Short momentum, 50% each leg, monthly rebalance',
        'config': {
            'long_pct': config.LONG_PCT,
            'short_pct': config.SHORT_PCT,
            'max_positions': config.MAX_POSITIONS_PER_LEG,
            'rebal_period': config.REBAL_PERIOD,
            'momentum_lookback': config.MOM_LOOKBACK
        },
        'v24_metrics': v24_metrics,
        'correlation': corr_results,
        'combined': {
            'v21': combined_results.get('v21', {}),
            'v24': combined_results.get('v24', {}),
            'combined': combined_results.get('combined', {}),
            'sharpe_improvement': combined_results.get('sharpe_improvement', 0),
            'dd_improvement': combined_results.get('dd_improvement', 0)
        },
        'walk_forward': wf_results
    }
    
    results_dir = Path('results/v24')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / 'v24_v4_market_neutral_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    v24_returns.to_frame('returns').to_parquet(results_dir / 'v24_v4_daily_returns.parquet')
    
    # === FINAL SUMMARY ===
    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)
    
    corr_val = corr_results.get('correlation', 1)
    
    checks = [
        ('V24 CAGR > 5%', v24_metrics.get('cagr', 0) > 0.05, f"{v24_metrics.get('cagr', 0):.1%}"),
        ('V24 Sharpe > 0.3', v24_metrics.get('sharpe', 0) > 0.3, f"{v24_metrics.get('sharpe', 0):.2f}"),
        ('V24 MaxDD > -20%', v24_metrics.get('max_dd', -1) > -0.20, f"{v24_metrics.get('max_dd', 0):.1%}"),
        ('★★★ CORRELATION < 0.3', abs(corr_val) < 0.3, f"{corr_val:.3f}"),
        ('Combined Sharpe improved', combined_results.get('sharpe_improvement', 0) > 0, 
         f"{combined_results.get('sharpe_improvement', 0):+.2f}"),
    ]
    
    logger.info(f"\n{'Check':<30} {'Status':<10} {'Value':<15}")
    logger.info("-" * 55)
    
    correlation_passed = False
    
    for check_name, passed, value in checks:
        status = '✅ PASS' if passed else '❌ FAIL'
        logger.info(f"{check_name:<30} {status:<10} {value:<15}")
        if '★★★' in check_name:
            correlation_passed = passed
    
    if correlation_passed:
        logger.info("\n🎉🎉🎉 CORRELATION TARGET MET! V24 is truly decorrelated from V21!")
        logger.info("    V24 Market-Neutral can effectively diversify V21 Mean-Reversion!")
    else:
        logger.info("\n⚠️ Need to try more radical approach (cross-asset, different timing, etc.)")
    
    return results


if __name__ == "__main__":
    results = main()
