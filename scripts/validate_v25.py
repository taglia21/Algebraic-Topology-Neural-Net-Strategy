#!/usr/bin/env python3
"""
V25 Phase Validation Script
============================

Validates each phase of V25 Adaptive Profit Maximization System.

Usage:
    python scripts/validate_v25.py --phase 1
    python scripts/validate_v25.py --phase 2
    python scripts/validate_v25.py --phase all
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('V25_Validation')


# =============================================================================
# DATA LOADING
# =============================================================================

def load_strategy_returns():
    """Load V21 and V24 daily returns from existing backtests."""
    
    # Try to load actual V24 returns
    v24_path = Path('results/v24/v24_v5_daily_returns.parquet')
    
    if v24_path.exists():
        v24_df = pd.read_parquet(v24_path)
        v24_returns = v24_df['returns'].values
        logger.info(f"Loaded V24 returns: {len(v24_returns)} days")
    else:
        v24_returns = None
        logger.warning("V24 returns not found, will simulate")
        
    # Load price data for V21 simulation and regime detection
    prices_path = Path('cache/v17_prices/v17_prices_latest.parquet')
    
    if prices_path.exists():
        prices_df = pd.read_parquet(prices_path)
        prices_df['date'] = pd.to_datetime(prices_df['date'])
        logger.info(f"Loaded prices: {len(prices_df)} records")
    else:
        raise FileNotFoundError(f"Price data not found at {prices_path}")
        
    return v24_returns, prices_df


def simulate_v21_returns(prices_df: pd.DataFrame) -> np.ndarray:
    """
    Simulate V21 mean-reversion returns.
    
    V21 characteristics:
    - RSI < 35 entry (oversold)
    - 5-day holding period
    - High volatility preference
    - Sharpe ~0.70, CAGR ~15%
    """
    symbol_col = 'symbol' if 'symbol' in prices_df.columns else 'ticker'
    
    # Get wide format
    close_wide = prices_df.pivot(index='date', columns=symbol_col, values='close')
    close_wide = close_wide.ffill(limit=5)
    
    ret_1d = close_wide.pct_change()
    
    # RSI calculation
    delta = close_wide.diff()
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)
    avg_gain = gains.rolling(14).mean()
    avg_loss = losses.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    
    # Volatility
    vol_20d = ret_1d.rolling(20).std() * np.sqrt(252)
    
    # V21 entry signal
    v21_entry = (rsi < 35) & (vol_20d > 0.30)
    
    # Rank by lowest RSI
    rsi_filtered = rsi.where(v21_entry)
    ranks = rsi_filtered.rank(axis=1, pct=True, ascending=True, na_option='keep')
    
    # Build positions
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
    rebal_dates = positions.index[::5]
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
    
    return v21_returns.values


def simulate_v24_returns(prices_df: pd.DataFrame) -> np.ndarray:
    """
    Simulate V24 low-beta momentum returns.
    
    V24 characteristics:
    - 70% long / 30% short
    - 60-day momentum
    - 15-day rebalance
    - Sharpe ~0.55, CAGR ~8%
    """
    symbol_col = 'symbol' if 'symbol' in prices_df.columns else 'ticker'
    
    close_wide = prices_df.pivot(index='date', columns=symbol_col, values='close')
    volume_wide = prices_df.pivot(index='date', columns=symbol_col, values='volume')
    
    close_wide = close_wide.ffill(limit=5)
    volume_wide = volume_wide.ffill(limit=5)
    
    ret_1d = close_wide.pct_change()
    
    # Filters
    price_ok = close_wide > 10
    dollar_vol = close_wide * volume_wide
    avg_dollar_vol = dollar_vol.rolling(20).mean()
    liquid = avg_dollar_vol > 5_000_000
    tradeable = price_ok & liquid
    
    # 60-day momentum
    momentum = close_wide.pct_change(60)
    momentum_filtered = momentum.where(tradeable)
    ranks = momentum_filtered.rank(axis=1, pct=True, ascending=True, na_option='keep')
    
    # Positions
    long_weight = 0.70
    short_weight = 0.30
    max_pos = 25
    rebal_period = 15
    
    long_weights = pd.DataFrame(0.0, index=ranks.index, columns=ranks.columns)
    short_weights = pd.DataFrame(0.0, index=ranks.index, columns=ranks.columns)
    
    valid_dates = ranks.index[60:]
    rebal_dates = valid_dates[::rebal_period]
    
    current_long = pd.Series(0.0, index=ranks.columns)
    current_short = pd.Series(0.0, index=ranks.columns)
    
    for date in valid_dates:
        if date in rebal_dates:
            day_ranks = ranks.loc[date].dropna()
            
            if len(day_ranks) >= 20:
                long_stocks = day_ranks.nlargest(min(len(day_ranks) // 5, max_pos)).index
                short_stocks = day_ranks.nsmallest(min(len(day_ranks) // 5, max_pos)).index
                
                current_long = pd.Series(0.0, index=ranks.columns)
                current_short = pd.Series(0.0, index=ranks.columns)
                
                if len(long_stocks) > 0:
                    current_long[long_stocks] = long_weight / len(long_stocks)
                if len(short_stocks) > 0:
                    current_short[short_stocks] = short_weight / len(short_stocks)
        
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
    costs.loc[rebal_dates] = long_turnover.loc[rebal_dates] * 0.001 + short_turnover.loc[rebal_dates] * 0.0025
    
    net_returns = strategy_returns - costs
    net_returns = net_returns.loc[valid_dates].dropna()
    
    return net_returns.values


# =============================================================================
# PHASE VALIDATIONS
# =============================================================================

def validate_phase1() -> bool:
    """
    Phase 1: Enhanced Regime Detection
    
    Success: Combined Sharpe >= 0.80
    """
    logger.info("=" * 70)
    logger.info("PHASE 1: Enhanced Regime Detection")
    logger.info("=" * 70)
    
    from src.regime.v25_adaptive_allocator import V25AdaptiveAllocator, run_v25_backtest
    
    # Load data
    _, prices_df = load_strategy_returns()
    
    # Simulate both strategies
    logger.info("Simulating V21 returns...")
    v21_returns = simulate_v21_returns(prices_df)
    logger.info(f"V21: {len(v21_returns)} days")
    
    logger.info("Simulating V24 returns...")
    v24_returns = simulate_v24_returns(prices_df)
    logger.info(f"V24: {len(v24_returns)} days")
    
    # Align returns
    min_len = min(len(v21_returns), len(v24_returns))
    v21_returns = v21_returns[-min_len:]
    v24_returns = v24_returns[-min_len:]
    
    # Get market prices for regime detection
    symbol_col = 'symbol' if 'symbol' in prices_df.columns else 'ticker'
    close_wide = prices_df.pivot(index='date', columns=symbol_col, values='close')
    market_prices = close_wide.mean(axis=1).values[-min_len-100:]  # Extra for warmup
    
    logger.info(f"Aligned data: {min_len} days")
    
    # Initialize allocator
    allocator = V25AdaptiveAllocator(
        log_dir="logs/v25",
        window_size=60,
        learning_rate=0.1
    )
    
    # Run backtest
    results = run_v25_backtest(v21_returns, v24_returns, market_prices, allocator)
    
    # Display results
    static = results['static']
    adaptive = results['adaptive']
    
    logger.info(f"\n{'Strategy':<20} {'CAGR':>10} {'Sharpe':>10} {'MaxDD':>10}")
    logger.info("-" * 50)
    logger.info(f"{'Static 50/50':<20} {static['cagr']:>10.1%} {static['sharpe']:>10.2f} {static['max_dd']:>10.1%}")
    logger.info(f"{'V25 Adaptive':<20} {adaptive['cagr']:>10.1%} {adaptive['sharpe']:>10.2f} {adaptive['max_dd']:>10.1%}")
    
    logger.info(f"\nSharpe Improvement: {results['sharpe_improvement']:+.3f}")
    
    # Accuracy
    acc = results['accuracy']
    logger.info(f"Regime Prediction Accuracy: {acc['accuracy']:.1%}")
    
    # Save allocator state
    allocator.save_state()
    
    # Validation
    target = 0.80
    passed = adaptive['sharpe'] >= target
    
    logger.info("\n" + "-" * 50)
    logger.info(f"Target Sharpe: >= {target}")
    logger.info(f"Achieved: {adaptive['sharpe']:.2f}")
    logger.info(f"Status: {'✅ PASS' if passed else '❌ FAIL'}")
    
    # Save results
    results_path = Path('results/v25')
    results_path.mkdir(parents=True, exist_ok=True)
    
    with open(results_path / 'phase1_results.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'phase': 1,
            'static_sharpe': float(static['sharpe']),
            'adaptive_sharpe': float(adaptive['sharpe']),
            'improvement': float(results['sharpe_improvement']),
            'target': target,
            'passed': bool(passed)
        }, f, indent=2)
    
    return passed


def validate_phase2() -> bool:
    """
    Phase 2: Pattern Memory System
    
    Success: Win rate on pattern trades >= 60%
    """
    logger.info("=" * 70)
    logger.info("PHASE 2: Pattern Memory System")
    logger.info("=" * 70)
    logger.info("⏳ Phase 2 not yet implemented")
    return False


def validate_phase3() -> bool:
    """
    Phase 3: Adaptive Position Sizing
    
    Success: Drawdown reduction >= 2pp
    """
    logger.info("=" * 70)
    logger.info("PHASE 3: Adaptive Position Sizing")
    logger.info("=" * 70)
    logger.info("⏳ Phase 3 not yet implemented")
    return False


def validate_phase4() -> bool:
    """
    Phase 4: Continuous Learning Loop
    
    Success: Positive accuracy trend over 30 days
    """
    logger.info("=" * 70)
    logger.info("PHASE 4: Continuous Learning Loop")
    logger.info("=" * 70)
    logger.info("⏳ Phase 4 not yet implemented")
    return False


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='V25 Phase Validation')
    parser.add_argument('--phase', type=str, default='1',
                       help='Phase to validate: 1, 2, 3, 4, or all')
    args = parser.parse_args()
    
    phase = args.phase.lower()
    
    if phase == '1':
        success = validate_phase1()
    elif phase == '2':
        success = validate_phase2()
    elif phase == '3':
        success = validate_phase3()
    elif phase == '4':
        success = validate_phase4()
    elif phase == 'all':
        results = []
        for p in [1, 2, 3, 4]:
            logger.info(f"\n{'='*70}")
            logger.info(f"RUNNING PHASE {p}")
            logger.info(f"{'='*70}\n")
            
            if p == 1:
                passed = validate_phase1()
            elif p == 2:
                passed = validate_phase2()
            elif p == 3:
                passed = validate_phase3()
            else:
                passed = validate_phase4()
                
            results.append((p, passed))
            
            if not passed:
                logger.warning(f"Phase {p} failed. Stopping.")
                break
                
        logger.info("\n" + "=" * 70)
        logger.info("SUMMARY")
        logger.info("=" * 70)
        for p, passed in results:
            status = '✅ PASS' if passed else '❌ FAIL'
            logger.info(f"Phase {p}: {status}")
            
        success = all(passed for _, passed in results)
    else:
        logger.error(f"Unknown phase: {phase}")
        success = False
        
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
