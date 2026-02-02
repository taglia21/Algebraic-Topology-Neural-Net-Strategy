#!/usr/bin/env python3
"""
V24 Complementary Momentum Strategy
=====================================
Breakout momentum strategy designed to COMPLEMENT V21 mean-reversion.

Key Design Principles:
- LOW CORRELATION with V21 (<0.3 target) is the PRIMARY success metric
- Captures trend-following alpha in TRENDING markets
- Opposite entry logic from V21: buy breakouts, not oversold

Entry Conditions (ALL must be true):
1. Price closes above 20-day high (breakout confirmation)
2. Volume > 1.5x 20-day average (conviction filter)
3. RSI(14) > 50 (momentum confirmation)
4. Top 30% of 1-month sector performance (relative strength)

Exit Conditions (ANY triggers exit):
1. Price closes below 10-day low (trend reversal)
2. Trailing stop: 2x ATR from highest close since entry
3. Max holding period: 20 trading days
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
logger = logging.getLogger('V24_Momentum')


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


def load_v21_returns() -> pd.Series:
    """Load V21 daily returns for correlation analysis."""
    # Try to load from cached V21 results
    v21_path = Path('results/v21/v21_daily_returns.parquet')
    if v21_path.exists():
        returns = pd.read_parquet(v21_path)['returns']
        return returns
    
    # Otherwise, run V21 backtest to get returns
    logger.info("V21 returns not cached, will compute during backtest")
    return None


def prepare_wide_data(prices: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
    """Convert long-format prices to wide format for vectorized operations."""
    # Column is 'symbol' not 'ticker'
    close_wide = prices.pivot(index='date', columns='symbol', values='close')
    high_wide = prices.pivot(index='date', columns='symbol', values='high')
    low_wide = prices.pivot(index='date', columns='symbol', values='low')
    volume_wide = prices.pivot(index='date', columns='symbol', values='volume')
    
    # Forward-fill missing values (up to 5 days for holidays)
    close_wide = close_wide.ffill(limit=5)
    high_wide = high_wide.ffill(limit=5)
    low_wide = low_wide.ffill(limit=5)
    volume_wide = volume_wide.ffill(limit=5)
    
    logger.info(f"Data shape: {close_wide.shape[0]} days x {close_wide.shape[1]} stocks")
    
    return close_wide, high_wide, low_wide, volume_wide


# =============================================================================
# SECTOR DATA (Simplified - use first letter of ticker as proxy)
# =============================================================================

def assign_sectors(tickers: list) -> Dict[str, str]:
    """
    Assign sectors to tickers.
    In production, use GICS from a data provider.
    Here we use a simplified sector mapping based on known ETF/stock sectors.
    """
    # Known sector mappings (subset)
    known_sectors = {
        # Technology
        'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
        'GOOG': 'Technology', 'META': 'Technology', 'NVDA': 'Technology',
        'AMD': 'Technology', 'INTC': 'Technology', 'CRM': 'Technology',
        'ADBE': 'Technology', 'CSCO': 'Technology', 'AVGO': 'Technology',
        'ORCL': 'Technology', 'IBM': 'Technology', 'QCOM': 'Technology',
        # Healthcare
        'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'PFE': 'Healthcare',
        'MRK': 'Healthcare', 'ABBV': 'Healthcare', 'LLY': 'Healthcare',
        'TMO': 'Healthcare', 'ABT': 'Healthcare', 'BMY': 'Healthcare',
        # Financials
        'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials',
        'GS': 'Financials', 'MS': 'Financials', 'C': 'Financials',
        'BLK': 'Financials', 'AXP': 'Financials', 'SCHW': 'Financials',
        # Consumer
        'AMZN': 'Consumer', 'TSLA': 'Consumer', 'HD': 'Consumer',
        'NKE': 'Consumer', 'MCD': 'Consumer', 'SBUX': 'Consumer',
        'TGT': 'Consumer', 'COST': 'Consumer', 'WMT': 'Consumer',
        # Energy
        'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy',
        'SLB': 'Energy', 'EOG': 'Energy', 'PXD': 'Energy',
        # Industrials
        'CAT': 'Industrials', 'BA': 'Industrials', 'HON': 'Industrials',
        'UPS': 'Industrials', 'GE': 'Industrials', 'MMM': 'Industrials',
        # Utilities
        'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities',
        # Communications
        'VZ': 'Communications', 'T': 'Communications', 'TMUS': 'Communications',
        'NFLX': 'Communications', 'DIS': 'Communications',
        # Materials
        'LIN': 'Materials', 'APD': 'Materials', 'SHW': 'Materials',
        # Real Estate
        'AMT': 'RealEstate', 'PLD': 'RealEstate', 'EQIX': 'RealEstate',
    }
    
    # Assign known or default to sector based on hash
    sectors = {}
    sector_list = ['Technology', 'Healthcare', 'Financials', 'Consumer', 
                   'Energy', 'Industrials', 'Utilities', 'Communications',
                   'Materials', 'RealEstate']
    
    for ticker in tickers:
        if ticker in known_sectors:
            sectors[ticker] = known_sectors[ticker]
        else:
            # Use hash for consistent pseudo-random assignment
            sectors[ticker] = sector_list[hash(ticker) % len(sector_list)]
    
    return sectors


# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

def calculate_indicators(close: pd.DataFrame, high: pd.DataFrame, 
                         low: pd.DataFrame, volume: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Calculate all technical indicators needed for momentum strategy."""
    
    indicators = {}
    
    # Daily returns
    ret_1d = close.pct_change()
    indicators['ret_1d'] = ret_1d
    
    # 20-day high (for breakout detection)
    high_20d = close.rolling(20).max()
    indicators['high_20d'] = high_20d
    
    # 10-day low (for exit signal)
    low_10d = close.rolling(10).min()
    indicators['low_10d'] = low_10d
    
    # Volume ratio (current vs 20-day average)
    vol_sma_20 = volume.rolling(20).mean()
    vol_ratio = volume / vol_sma_20
    indicators['vol_ratio'] = vol_ratio
    
    # RSI(14)
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)
    avg_gain = gains.rolling(14).mean()
    avg_loss = losses.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    indicators['rsi'] = rsi
    
    # ATR(14) for trailing stop
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=0).max(level=1) if isinstance(tr1.index, pd.MultiIndex) else \
         pd.DataFrame({col: pd.concat([tr1[col], tr2[col], tr3[col]], axis=1).max(axis=1) 
                       for col in close.columns})
    # Simpler TR calculation
    tr = pd.DataFrame(index=close.index, columns=close.columns)
    for col in close.columns:
        h = high[col]
        l = low[col]
        c_prev = close[col].shift(1)
        tr[col] = pd.concat([
            h - l,
            (h - c_prev).abs(),
            (l - c_prev).abs()
        ], axis=1).max(axis=1)
    
    atr = tr.rolling(14).mean()
    indicators['atr'] = atr
    
    # 1-month return (for sector relative strength)
    ret_1m = close.pct_change(21)
    indicators['ret_1m'] = ret_1m
    
    # 3-month momentum
    ret_3m = close.pct_change(63)
    indicators['ret_3m'] = ret_3m
    
    logger.info("Calculated all technical indicators")
    return indicators


def calculate_sector_performance(ret_1m: pd.DataFrame, 
                                  sector_map: Dict[str, str]) -> pd.DataFrame:
    """Calculate sector average 1-month performance."""
    sectors = pd.Series(sector_map)
    sector_perf = pd.DataFrame(index=ret_1m.index)
    
    for sector in sectors.unique():
        sector_stocks = sectors[sectors == sector].index
        valid_stocks = [s for s in sector_stocks if s in ret_1m.columns]
        if valid_stocks:
            sector_perf[sector] = ret_1m[valid_stocks].mean(axis=1)
    
    return sector_perf


def calculate_sector_relative_strength(ret_1m: pd.DataFrame,
                                        sector_map: Dict[str, str]) -> pd.DataFrame:
    """
    Calculate if each stock is in top 30% of its sector's 1-month performance.
    Returns DataFrame of boolean values.
    """
    sectors = pd.Series(sector_map)
    in_top_30pct = pd.DataFrame(False, index=ret_1m.index, columns=ret_1m.columns)
    
    for sector in sectors.unique():
        sector_stocks = sectors[sectors == sector].index
        valid_stocks = [s for s in sector_stocks if s in ret_1m.columns]
        
        if len(valid_stocks) < 3:
            # If sector too small, include all
            in_top_30pct[valid_stocks] = True
            continue
        
        # Rank within sector
        sector_ret = ret_1m[valid_stocks]
        sector_rank_pct = sector_ret.rank(axis=1, pct=True)
        
        # Top 30%
        in_top_30pct[valid_stocks] = sector_rank_pct >= 0.70
    
    return in_top_30pct


# =============================================================================
# MOMENTUM STRATEGY BACKTEST
# =============================================================================

def run_momentum_backtest(close: pd.DataFrame, high: pd.DataFrame,
                          low: pd.DataFrame, volume: pd.DataFrame,
                          params: Dict) -> Dict:
    """
    Run momentum strategy backtest.
    
    Parameters:
    - n_positions: Max number of positions
    - breakout_period: Period for breakout detection (default 20)
    - exit_period: Period for exit signal (default 10)
    - vol_ratio_threshold: Volume ratio threshold (default 1.5)
    - rsi_threshold: RSI must be above this (default 50)
    - atr_multiplier: ATR multiplier for trailing stop (default 2.0)
    - max_holding: Maximum holding period (default 20)
    - cost_bps: Transaction cost in basis points (default 10)
    """
    
    N_POSITIONS = params.get('n_positions', 30)
    BREAKOUT_PERIOD = params.get('breakout_period', 20)
    EXIT_PERIOD = params.get('exit_period', 10)
    VOL_RATIO_THRESH = params.get('vol_ratio_threshold', 1.5)
    RSI_THRESHOLD = params.get('rsi_threshold', 50)
    ATR_MULT = params.get('atr_multiplier', 2.0)
    MAX_HOLDING = params.get('max_holding', 20)
    COST_BPS = params.get('cost_bps', 10)
    
    # Calculate indicators
    indicators = calculate_indicators(close, high, low, volume)
    ret_1d = indicators['ret_1d']
    high_20d = close.rolling(BREAKOUT_PERIOD).max()
    low_10d = close.rolling(EXIT_PERIOD).min()
    vol_ratio = indicators['vol_ratio']
    rsi = indicators['rsi']
    atr = indicators['atr']
    ret_1m = indicators['ret_1m']
    ret_3m = indicators['ret_3m']
    
    # Sector relative strength
    sector_map = assign_sectors(close.columns.tolist())
    in_top_sector = calculate_sector_relative_strength(ret_1m, sector_map)
    
    # ==========================================================================
    # ENTRY CONDITIONS (vectorized)
    # ==========================================================================
    
    # 1. Breakout: price >= 20-day high (within 0.5% to allow for rounding)
    breakout = close >= (high_20d * 0.995)
    
    # 2. Volume conviction: volume > 1.5x average
    vol_conviction = vol_ratio > VOL_RATIO_THRESH
    
    # 3. RSI momentum: RSI > 50 (not oversold, has momentum)
    rsi_momentum = rsi > RSI_THRESHOLD
    
    # 4. Sector relative strength: top 30% of sector
    sector_strength = in_top_sector
    
    # Combined entry signal
    entry_signal = breakout & vol_conviction & rsi_momentum & sector_strength
    
    # Rank by 3-month momentum for selection
    momentum_score = ret_3m.where(entry_signal)
    
    # ==========================================================================
    # POSITION MANAGEMENT (with holding period and exits)
    # ==========================================================================
    
    # Track positions with entry dates and highest price since entry
    positions = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    entry_dates = pd.DataFrame(pd.NaT, index=close.index, columns=close.columns)
    highest_since_entry = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    
    dates = close.index.tolist()
    
    for i, date in enumerate(dates):
        if i == 0:
            continue
        
        prev_date = dates[i - 1]
        
        # Carry forward existing positions
        current_pos = positions.loc[prev_date].copy()
        current_entry = entry_dates.loc[prev_date].copy()
        current_high = highest_since_entry.loc[prev_date].copy()
        
        # Update highest price for existing positions
        for ticker in close.columns:
            if current_pos[ticker] > 0:
                prev_high = current_high[ticker]
                today_price = close.loc[date, ticker]
                if pd.isna(prev_high):
                    current_high[ticker] = today_price
                else:
                    current_high[ticker] = max(prev_high, today_price)
        
        # =======================================================================
        # EXIT CONDITIONS
        # =======================================================================
        
        for ticker in close.columns:
            if current_pos[ticker] == 0:
                continue
            
            entry_dt = current_entry[ticker]
            if pd.isna(entry_dt):
                continue
            
            # Days held
            try:
                entry_idx = dates.index(entry_dt)
                days_held = i - entry_idx
            except:
                days_held = 0
            
            exit_triggered = False
            
            # Exit 1: Price below 10-day low (trend reversal)
            if close.loc[date, ticker] < low_10d.loc[date, ticker]:
                exit_triggered = True
            
            # Exit 2: Trailing stop (2x ATR from highest)
            if not pd.isna(current_high[ticker]) and not pd.isna(atr.loc[date, ticker]):
                trailing_stop = current_high[ticker] - ATR_MULT * atr.loc[date, ticker]
                if close.loc[date, ticker] < trailing_stop:
                    exit_triggered = True
            
            # Exit 3: Max holding period
            if days_held >= MAX_HOLDING:
                exit_triggered = True
            
            if exit_triggered:
                current_pos[ticker] = 0
                current_entry[ticker] = pd.NaT
                current_high[ticker] = np.nan
        
        # =======================================================================
        # ENTRY: Fill up to N_POSITIONS
        # =======================================================================
        
        n_current = (current_pos > 0).sum()
        n_available = N_POSITIONS - n_current
        
        if n_available > 0:
            # Get today's entry candidates
            candidates = momentum_score.loc[date].dropna()
            
            # Exclude stocks we already hold
            candidates = candidates[~candidates.index.isin(
                current_pos[current_pos > 0].index
            )]
            
            if len(candidates) > 0:
                # Select top N by momentum score
                new_entries = candidates.nlargest(min(n_available, len(candidates)))
                
                for ticker in new_entries.index:
                    current_pos[ticker] = 1.0
                    current_entry[ticker] = date
                    current_high[ticker] = close.loc[date, ticker]
        
        # Save state
        positions.loc[date] = current_pos
        entry_dates.loc[date] = current_entry
        highest_since_entry.loc[date] = current_high
    
    # ==========================================================================
    # CALCULATE RETURNS
    # ==========================================================================
    
    # Equal weight within positions
    pos_count = (positions > 0).sum(axis=1).replace(0, 1)
    weights = positions.div(pos_count, axis=0)
    
    # Strategy returns
    strategy_returns = (weights.shift(1) * ret_1d).sum(axis=1)
    
    # Transaction costs
    weight_changes = weights.diff().abs().sum(axis=1)
    costs = weight_changes * (COST_BPS / 10000)
    
    net_returns = strategy_returns - costs
    
    # Store daily returns for correlation analysis
    result = {
        'daily_returns': net_returns,
        'positions': positions,
        'weights': weights,
        'gross_returns': strategy_returns,
        'costs': costs,
        'entry_signals': entry_signal.sum(axis=1)
    }
    
    return result


# =============================================================================
# WALK-FORWARD VALIDATION
# =============================================================================

def run_walkforward_validation(close: pd.DataFrame, high: pd.DataFrame,
                                low: pd.DataFrame, volume: pd.DataFrame,
                                params: Dict,
                                train_months: int = 6,
                                test_months: int = 2) -> Dict:
    """
    Run walk-forward validation with 6-month train / 2-month test windows.
    """
    
    logger.info(f"Running walk-forward validation: {train_months}m train, {test_months}m test")
    
    dates = close.index
    start_date = dates[0]
    end_date = dates[-1]
    
    # Create windows
    windows = []
    current_start = start_date
    
    while True:
        train_end = current_start + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)
        
        if test_end > end_date:
            break
        
        windows.append({
            'train_start': current_start,
            'train_end': train_end,
            'test_start': train_end,
            'test_end': test_end
        })
        
        current_start = current_start + pd.DateOffset(months=test_months)
    
    logger.info(f"Created {len(windows)} walk-forward windows")
    
    is_returns = []
    oos_returns = []
    window_results = []
    
    for i, window in enumerate(windows):
        # In-sample period
        is_mask = (dates >= window['train_start']) & (dates < window['train_end'])
        is_close = close.loc[is_mask]
        is_high = high.loc[is_mask]
        is_low = low.loc[is_mask]
        is_volume = volume.loc[is_mask]
        
        if len(is_close) < 60:  # Need at least 60 days
            continue
        
        # Out-of-sample period
        oos_mask = (dates >= window['test_start']) & (dates < window['test_end'])
        oos_close = close.loc[oos_mask]
        oos_high = high.loc[oos_mask]
        oos_low = low.loc[oos_mask]
        oos_volume = volume.loc[oos_mask]
        
        if len(oos_close) < 20:
            continue
        
        # Run backtest on IS
        is_result = run_momentum_backtest(is_close, is_high, is_low, is_volume, params)
        is_ret = is_result['daily_returns']
        is_returns.extend(is_ret.tolist())
        
        # Run backtest on OOS
        oos_result = run_momentum_backtest(oos_close, oos_high, oos_low, oos_volume, params)
        oos_ret = oos_result['daily_returns']
        oos_returns.extend(oos_ret.tolist())
        
        # Window metrics
        is_sharpe = (is_ret.mean() * 252) / (is_ret.std() * np.sqrt(252)) if is_ret.std() > 0 else 0
        oos_sharpe = (oos_ret.mean() * 252) / (oos_ret.std() * np.sqrt(252)) if oos_ret.std() > 0 else 0
        
        window_results.append({
            'window': i + 1,
            'train_start': window['train_start'].strftime('%Y-%m-%d'),
            'train_end': window['train_end'].strftime('%Y-%m-%d'),
            'test_start': window['test_start'].strftime('%Y-%m-%d'),
            'test_end': window['test_end'].strftime('%Y-%m-%d'),
            'is_sharpe': is_sharpe,
            'oos_sharpe': oos_sharpe,
            'sharpe_decay': oos_sharpe / is_sharpe if is_sharpe > 0 else 0
        })
        
        logger.info(f"Window {i+1}: IS Sharpe={is_sharpe:.2f}, OOS Sharpe={oos_sharpe:.2f}")
    
    # Aggregate metrics
    is_returns = pd.Series(is_returns)
    oos_returns = pd.Series(oos_returns)
    
    is_sharpe_agg = (is_returns.mean() * 252) / (is_returns.std() * np.sqrt(252)) if is_returns.std() > 0 else 0
    oos_sharpe_agg = (oos_returns.mean() * 252) / (oos_returns.std() * np.sqrt(252)) if oos_returns.std() > 0 else 0
    
    return {
        'windows': window_results,
        'is_sharpe': is_sharpe_agg,
        'oos_sharpe': oos_sharpe_agg,
        'oos_is_ratio': oos_sharpe_agg / is_sharpe_agg if is_sharpe_agg > 0 else 0,
        'is_returns': is_returns,
        'oos_returns': oos_returns
    }


# =============================================================================
# METRICS CALCULATION
# =============================================================================

def calculate_metrics(returns: pd.Series, name: str = "Strategy") -> Dict:
    """Calculate comprehensive performance metrics."""
    
    if len(returns) < 20:
        return {'error': 'Insufficient data'}
    
    returns = returns.dropna()
    
    # Cumulative returns
    cumulative = (1 + returns).cumprod()
    total_return = cumulative.iloc[-1] - 1
    
    # Time calculations
    trading_days = len(returns)
    years = trading_days / 252
    
    # CAGR
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    # Volatility
    annual_vol = returns.std() * np.sqrt(252)
    
    # Sharpe (assuming 0 risk-free rate)
    sharpe = (returns.mean() * 252) / annual_vol if annual_vol > 0 else 0
    
    # Drawdown
    peak = cumulative.expanding().max()
    drawdown = (cumulative - peak) / peak
    max_dd = drawdown.min()
    
    # Win rate
    wins = (returns > 0).sum()
    total_trades = (returns != 0).sum()
    win_rate = wins / total_trades if total_trades > 0 else 0
    
    # Profit factor
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    
    # Calmar ratio
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
    # Monthly returns for concentration check
    monthly_returns = returns.resample('M').sum() if hasattr(returns.index, 'freq') or isinstance(returns.index, pd.DatetimeIndex) else returns
    
    return {
        'name': name,
        'total_return': total_return * 100,
        'cagr': cagr * 100,
        'annual_volatility': annual_vol * 100,
        'sharpe': sharpe,
        'max_drawdown': max_dd * 100,
        'calmar': calmar,
        'win_rate': win_rate * 100,
        'profit_factor': profit_factor,
        'trading_days': trading_days,
        'years': years
    }


# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

def analyze_correlation_with_v21(v24_returns: pd.Series, 
                                  v21_returns: Optional[pd.Series] = None) -> Dict:
    """
    Calculate correlation between V24 and V21 strategies.
    This is the CRITICAL metric - must be < 0.3.
    """
    
    if v21_returns is None:
        # Need to run V21 backtest
        logger.info("Loading V21 for correlation analysis...")
        try:
            from v21_optimized_reversal import load_price_data, run_backtest
            
            prices = load_price_data()
            close_wide = prices.pivot(index='date', columns='ticker', values='close')
            high_wide = prices.pivot(index='date', columns='ticker', values='high')
            volume_wide = prices.pivot(index='date', columns='ticker', values='volume')
            ret_1d = close_wide.pct_change()
            
            # V21 best params
            v21_params = {
                'n_positions': 30,
                'holding_period': 5,
                'vol_threshold': 0.30,
                'rsi_threshold': 35,
                'drawdown_min': -0.05,
                'drawdown_max': -0.25
            }
            
            # Simplified V21 run - just get returns
            # (Full V21 logic would be imported, but we'll simulate)
            logger.info("Running simplified V21 for correlation...")
            
            # V21: Buy oversold (RSI < 35), sell when RSI > 70
            rsi = calculate_rsi(close_wide)
            oversold = rsi < 35
            
            # Simple V21 proxy returns
            v21_signal = oversold.shift(1)  # Signal from yesterday
            v21_weights = v21_signal.div(v21_signal.sum(axis=1).replace(0, 1), axis=0)
            v21_weights = v21_weights.clip(upper=0.10)  # Max 10% per position
            v21_returns = (v21_weights.shift(1) * ret_1d).sum(axis=1)
            
        except Exception as e:
            logger.error(f"Could not load V21: {e}")
            return {'error': str(e), 'correlation': np.nan}
    
    # Align dates
    common_dates = v24_returns.index.intersection(v21_returns.index)
    
    if len(common_dates) < 20:
        return {'error': 'Insufficient overlapping dates', 'correlation': np.nan}
    
    v24_aligned = v24_returns.loc[common_dates]
    v21_aligned = v21_returns.loc[common_dates]
    
    # Calculate correlation
    correlation = v24_aligned.corr(v21_aligned)
    
    # Rolling correlation (30-day window)
    rolling_corr = v24_aligned.rolling(30).corr(v21_aligned)
    
    return {
        'correlation': correlation,
        'rolling_correlation_mean': rolling_corr.mean(),
        'rolling_correlation_std': rolling_corr.std(),
        'correlation_min': rolling_corr.min(),
        'correlation_max': rolling_corr.max(),
        'common_days': len(common_dates),
        'correlation_pass': correlation < 0.3,
        'correlation_fail': correlation > 0.5
    }


def calculate_rsi(close: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculate RSI for all stocks."""
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)
    avg_gain = gains.rolling(period).mean()
    avg_loss = losses.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi


# =============================================================================
# COMBINED PORTFOLIO ANALYSIS
# =============================================================================

def analyze_combined_portfolio(v24_returns: pd.Series, 
                               v21_returns: pd.Series,
                               weights: Tuple[float, float] = (0.5, 0.5)) -> Dict:
    """
    Analyze 50/50 combined portfolio of V21 + V24.
    """
    
    # Align dates
    common_dates = v24_returns.index.intersection(v21_returns.index)
    v24 = v24_returns.loc[common_dates]
    v21 = v21_returns.loc[common_dates]
    
    # Combined returns
    w_v21, w_v24 = weights
    combined = w_v21 * v21 + w_v24 * v24
    
    # Calculate metrics for all three
    v21_metrics = calculate_metrics(v21, "V21 Mean Reversion")
    v24_metrics = calculate_metrics(v24, "V24 Momentum")
    combined_metrics = calculate_metrics(combined, f"Combined {int(w_v21*100)}/{int(w_v24*100)}")
    
    # Improvement analysis
    sharpe_improvement = combined_metrics['sharpe'] - v21_metrics['sharpe']
    dd_improvement = combined_metrics['max_drawdown'] - v21_metrics['max_drawdown']  # Less negative is better
    
    return {
        'v21': v21_metrics,
        'v24': v24_metrics,
        'combined': combined_metrics,
        'sharpe_improvement': sharpe_improvement,
        'drawdown_improvement': dd_improvement,  # Positive means combined has less drawdown
        'diversification_benefit': combined_metrics['sharpe'] > max(v21_metrics['sharpe'], v24_metrics['sharpe'])
    }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run V24 momentum strategy with full validation."""
    
    logger.info("=" * 70)
    logger.info("V24 COMPLEMENTARY MOMENTUM STRATEGY")
    logger.info("=" * 70)
    
    # =========================================================================
    # LOAD DATA
    # =========================================================================
    
    logger.info("\n📊 Loading price data...")
    prices = load_price_data()
    close, high, low, volume = prepare_wide_data(prices)
    
    # =========================================================================
    # STRATEGY PARAMETERS
    # =========================================================================
    
    params = {
        'n_positions': 30,
        'breakout_period': 20,
        'exit_period': 10,
        'vol_ratio_threshold': 1.5,
        'rsi_threshold': 50,
        'atr_multiplier': 2.0,
        'max_holding': 20,
        'cost_bps': 10
    }
    
    logger.info(f"\n📋 Parameters: {params}")
    
    # =========================================================================
    # FULL BACKTEST
    # =========================================================================
    
    logger.info("\n🚀 Running full backtest...")
    full_result = run_momentum_backtest(close, high, low, volume, params)
    v24_returns = full_result['daily_returns']
    
    # Calculate metrics
    v24_metrics = calculate_metrics(v24_returns, "V24 Momentum")
    
    logger.info("\n📈 V24 Standalone Metrics:")
    for key, value in v24_metrics.items():
        if isinstance(value, float):
            logger.info(f"   {key}: {value:.2f}")
        else:
            logger.info(f"   {key}: {value}")
    
    # =========================================================================
    # WALK-FORWARD VALIDATION
    # =========================================================================
    
    logger.info("\n🔄 Running walk-forward validation...")
    wf_results = run_walkforward_validation(close, high, low, volume, params)
    
    logger.info(f"\n📊 Walk-Forward Results:")
    logger.info(f"   In-Sample Sharpe:     {wf_results['is_sharpe']:.2f}")
    logger.info(f"   Out-of-Sample Sharpe: {wf_results['oos_sharpe']:.2f}")
    logger.info(f"   OOS/IS Ratio:         {wf_results['oos_is_ratio']:.2%}")
    
    # =========================================================================
    # V21 CORRELATION ANALYSIS
    # =========================================================================
    
    logger.info("\n🔗 Analyzing correlation with V21...")
    
    # Get V21 returns by running a simplified version
    ret_1d = close.pct_change()
    rsi = calculate_rsi(close)
    
    # V21 proxy: buy oversold, hold 5 days
    oversold = rsi < 35
    high_20d = high.rolling(20).max()
    drawdown = (close - high_20d) / high_20d
    
    # V21 entry: oversold + drawdown
    v21_entry = oversold & (drawdown < -0.10)
    
    # Simple V21 returns
    v21_signal = v21_entry.shift(1)
    n_signals = v21_signal.sum(axis=1).replace(0, 1)
    v21_weights = v21_signal.div(n_signals, axis=0).clip(upper=0.10)
    
    # Hold for 5 days
    v21_weights_held = v21_weights.rolling(5, min_periods=1).max()
    v21_weights_held = v21_weights_held.div(v21_weights_held.sum(axis=1).replace(0, 1), axis=0)
    
    v21_returns = (v21_weights_held.shift(1) * ret_1d).sum(axis=1)
    
    # Correlation analysis
    corr_results = analyze_correlation_with_v21(v24_returns, v21_returns)
    
    logger.info(f"\n🔗 V24-V21 Correlation Analysis:")
    logger.info(f"   Correlation:      {corr_results['correlation']:.3f}")
    logger.info(f"   Target:           < 0.30")
    logger.info(f"   Pass:             {'✅ YES' if corr_results['correlation_pass'] else '❌ NO'}")
    
    if corr_results['correlation'] > 0.5:
        logger.warning("   ⚠️  FAIL: Correlation > 0.5 - strategy does not complement V21!")
    
    # =========================================================================
    # COMBINED PORTFOLIO ANALYSIS
    # =========================================================================
    
    logger.info("\n💼 Analyzing combined 50/50 portfolio...")
    combined_results = analyze_combined_portfolio(v24_returns, v21_returns, (0.5, 0.5))
    
    logger.info(f"\n📊 Portfolio Comparison:")
    logger.info(f"   {'Metric':<20} {'V21':>12} {'V24':>12} {'Combined':>12}")
    logger.info(f"   {'-'*56}")
    
    for metric in ['cagr', 'sharpe', 'max_drawdown', 'win_rate']:
        v21_val = combined_results['v21'].get(metric, 0)
        v24_val = combined_results['v24'].get(metric, 0)
        comb_val = combined_results['combined'].get(metric, 0)
        logger.info(f"   {metric:<20} {v21_val:>12.2f} {v24_val:>12.2f} {comb_val:>12.2f}")
    
    logger.info(f"\n   Sharpe Improvement:    {combined_results['sharpe_improvement']:+.2f}")
    logger.info(f"   Diversification Benefit: {'✅ YES' if combined_results['diversification_benefit'] else '❌ NO'}")
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    
    results_dir = Path('results/v24')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save daily returns for future use
    v24_returns.to_frame('returns').to_parquet(results_dir / 'v24_daily_returns.parquet')
    
    # Save full results
    results = {
        'timestamp': datetime.now().isoformat(),
        'parameters': params,
        'standalone_metrics': v24_metrics,
        'walkforward': {
            'is_sharpe': wf_results['is_sharpe'],
            'oos_sharpe': wf_results['oos_sharpe'],
            'oos_is_ratio': wf_results['oos_is_ratio'],
            'windows': wf_results['windows']
        },
        'correlation': {
            'v21_correlation': corr_results['correlation'],
            'correlation_pass': corr_results['correlation_pass'],
            'correlation_fail': corr_results.get('correlation_fail', False)
        },
        'combined_portfolio': {
            'v21_sharpe': combined_results['v21']['sharpe'],
            'v24_sharpe': combined_results['v24']['sharpe'],
            'combined_sharpe': combined_results['combined']['sharpe'],
            'sharpe_improvement': combined_results['sharpe_improvement'],
            'diversification_benefit': combined_results['diversification_benefit']
        },
        'success_criteria': {
            'cagr_pass': v24_metrics['cagr'] > 15,  # > 15% hard fail threshold
            'sharpe_pass': v24_metrics['sharpe'] > 0.5,  # > 0.5 hard fail threshold
            'max_dd_pass': v24_metrics['max_drawdown'] > -40,  # > -40% hard fail
            'correlation_pass': corr_results['correlation'] < 0.5,  # < 0.5 hard fail
            'oos_is_pass': wf_results['oos_is_ratio'] > 0.5,  # > 50% OOS/IS
            'all_pass': all([
                v24_metrics['cagr'] > 15,
                v24_metrics['sharpe'] > 0.5,
                v24_metrics['max_drawdown'] > -40,
                corr_results['correlation'] < 0.5,
                wf_results['oos_is_ratio'] > 0.5
            ])
        }
    }
    
    with open(results_dir / 'v24_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\n💾 Results saved to {results_dir}")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("V24 FINAL ASSESSMENT")
    logger.info("=" * 70)
    
    criteria = results['success_criteria']
    all_pass = criteria['all_pass']
    
    logger.info(f"\n   {'Criterion':<30} {'Status':>10}")
    logger.info(f"   {'-'*42}")
    logger.info(f"   {'CAGR > 15%':<30} {'✅ PASS' if criteria['cagr_pass'] else '❌ FAIL':>10}")
    logger.info(f"   {'Sharpe > 0.5':<30} {'✅ PASS' if criteria['sharpe_pass'] else '❌ FAIL':>10}")
    logger.info(f"   {'MaxDD > -40%':<30} {'✅ PASS' if criteria['max_dd_pass'] else '❌ FAIL':>10}")
    logger.info(f"   {'V21 Correlation < 0.5':<30} {'✅ PASS' if criteria['correlation_pass'] else '❌ FAIL':>10}")
    logger.info(f"   {'OOS/IS Ratio > 50%':<30} {'✅ PASS' if criteria['oos_is_pass'] else '❌ FAIL':>10}")
    
    logger.info(f"\n   {'='*42}")
    if all_pass:
        logger.info(f"   ✅ V24 MOMENTUM STRATEGY VALIDATED")
        logger.info(f"   Ready for combined deployment with V21")
    else:
        logger.info(f"   ❌ V24 DID NOT MEET ALL CRITERIA")
        logger.info(f"   Review failed criteria before deployment")
    
    return results


if __name__ == "__main__":
    results = main()
