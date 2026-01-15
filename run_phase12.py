"""
Phase 12: All-Weather Regime-Switching Strategy
================================================

BIDIRECTIONAL ALPHA: Profit in BOTH bull and bear markets

This script tests the regime-switching strategy across the FULL 2022-2025 period,
including the 2022 bear market that devastated long-only leveraged strategies.

Key Innovation:
- Bull regime: Long 3x ETFs (TQQQ, SPXL, SOXL)
- Bear regime: Inverse 3x ETFs (SQQQ, SPXU, SOXS)

Targets:
- CAGR: ≥28%
- Max DD: ≤22%
- Sharpe: ≥1.5
- Alpha vs SPY: ≥5%
"""

import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# =============================================================================
# DATA LOADING
# =============================================================================

def download_data(tickers: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    """Download OHLCV data for tickers."""
    data = {}
    
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if len(df) > 0:
                # Normalize column names
                df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
                data[ticker] = df
                logger.info(f"  {ticker}: {len(df)} days")
        except Exception as e:
            logger.warning(f"  {ticker}: Failed - {e}")
    
    return data


def compute_sma(prices: pd.Series, window: int) -> pd.Series:
    """Compute Simple Moving Average."""
    return prices.rolling(window=window, min_periods=window).mean()


def compute_vix_estimate(spy_data: pd.DataFrame) -> pd.Series:
    """Estimate VIX-like volatility from SPY returns."""
    returns = spy_data['close'].pct_change()
    # 20-day rolling volatility annualized
    vol = returns.rolling(20).std() * np.sqrt(252) * 100
    # Scale to approximate VIX levels
    return vol.fillna(18)


# =============================================================================
# REGIME DETECTION (Simplified for backtest)
# =============================================================================

def classify_regime(spy_data: pd.DataFrame, date_idx: int) -> Tuple[str, float]:
    """
    Classify market regime based on multiple signals.
    
    Returns:
        regime: 'strong_bull', 'mild_bull', 'neutral', 'mild_bear', 'strong_bear'
        confidence: 0.0 to 1.0
    """
    if date_idx < 200:
        return 'neutral', 0.5
    
    close = spy_data['close'].iloc[:date_idx+1]
    
    # Compute SMAs
    sma_20 = compute_sma(close, 20)
    sma_50 = compute_sma(close, 50)
    sma_200 = compute_sma(close, 200)
    
    current_price = close.iloc[-1]
    
    # Signal 1: Price vs SMAs
    above_20 = current_price > sma_20.iloc[-1] if not np.isnan(sma_20.iloc[-1]) else False
    above_50 = current_price > sma_50.iloc[-1] if not np.isnan(sma_50.iloc[-1]) else False
    above_200 = current_price > sma_200.iloc[-1] if not np.isnan(sma_200.iloc[-1]) else False
    
    # Signal 2: SMA alignment
    sma_20_above_50 = sma_20.iloc[-1] > sma_50.iloc[-1] if not np.isnan(sma_20.iloc[-1]) and not np.isnan(sma_50.iloc[-1]) else False
    sma_50_above_200 = sma_50.iloc[-1] > sma_200.iloc[-1] if not np.isnan(sma_50.iloc[-1]) and not np.isnan(sma_200.iloc[-1]) else False
    
    # Signal 3: Momentum (20-day return)
    ret_20 = (current_price / close.iloc[-20] - 1) if len(close) >= 20 else 0
    
    # Signal 4: Trend strength (price distance from 200 SMA)
    distance_200 = (current_price - sma_200.iloc[-1]) / sma_200.iloc[-1] if not np.isnan(sma_200.iloc[-1]) and sma_200.iloc[-1] > 0 else 0
    
    # Count bullish/bearish signals
    bullish_count = sum([above_20, above_50, above_200, sma_20_above_50, sma_50_above_200, ret_20 > 0.02])
    bearish_count = sum([not above_20, not above_50, not above_200, not sma_20_above_50, not sma_50_above_200, ret_20 < -0.02])
    
    # Strong trend confirmation
    strong_uptrend = distance_200 > 0.05 and ret_20 > 0.03
    strong_downtrend = distance_200 < -0.05 and ret_20 < -0.03
    
    # Classify regime
    if bullish_count >= 5 or strong_uptrend:
        regime = 'strong_bull'
        confidence = min(1.0, 0.6 + bullish_count * 0.08)
    elif bullish_count >= 4:
        regime = 'mild_bull'
        confidence = 0.5 + bullish_count * 0.06
    elif bearish_count >= 5 or strong_downtrend:
        regime = 'strong_bear'
        confidence = min(1.0, 0.6 + bearish_count * 0.08)
    elif bearish_count >= 4:
        regime = 'mild_bear'
        confidence = 0.5 + bearish_count * 0.06
    else:
        regime = 'neutral'
        confidence = 0.3
    
    return regime, confidence


# =============================================================================
# ALLOCATION LOGIC
# =============================================================================

# ETF mappings
LONG_ETFS = {'TQQQ': 0.50, 'SPXL': 0.30, 'SOXL': 0.20}
INVERSE_ETFS = {'SQQQ': 0.50, 'SPXU': 0.30, 'SOXS': 0.20}


def get_allocation(
    regime: str,
    confidence: float,
    vix_level: float,
    current_drawdown: float,
    consecutive_losses: int,
) -> Tuple[Dict[str, float], str]:
    """
    Get ETF allocation based on regime.
    
    Returns:
        weights: Dict of ticker -> weight
        direction: 'long', 'inverse', or 'neutral'
    """
    # Base allocation by regime
    if regime in ['strong_bull', 'mild_bull']:
        direction = 'long'
        base_etfs = LONG_ETFS
    elif regime in ['strong_bear', 'mild_bear']:
        direction = 'inverse'
        base_etfs = INVERSE_ETFS
    else:
        return {}, 'neutral'
    
    # Regime strength scaling
    if 'strong' in regime:
        regime_scale = 0.65
    else:  # mild
        regime_scale = 0.45
    
    # Confidence scaling
    conf_scale = 0.6 + 0.4 * confidence
    
    # VIX scaling (reduce in high vol)
    if vix_level > 40:
        vix_scale = 0.30
    elif vix_level > 30:
        vix_scale = 0.50
    elif vix_level > 25:
        vix_scale = 0.70
    else:
        vix_scale = 1.0
    
    # Drawdown protection
    if current_drawdown > 0.15:
        dd_scale = 0.30
    elif current_drawdown > 0.10:
        dd_scale = 0.50
    elif current_drawdown > 0.05:
        dd_scale = 0.75
    else:
        dd_scale = 1.0
    
    # Consecutive loss penalty
    if consecutive_losses >= 4:
        loss_scale = 0.30
    elif consecutive_losses >= 3:
        loss_scale = 0.50
    elif consecutive_losses >= 2:
        loss_scale = 0.75
    else:
        loss_scale = 1.0
    
    # Combined scale
    total_scale = regime_scale * conf_scale * vix_scale * dd_scale * loss_scale
    total_scale = max(0.10, min(0.65, total_scale))
    
    # Apply to ETFs
    weights = {ticker: weight * total_scale for ticker, weight in base_etfs.items()}
    
    return weights, direction


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def run_backtest(
    data: Dict[str, pd.DataFrame],
    spy_data: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
    rebalance_freq: int = 5,  # Weekly
    initial_capital: float = 100000,
) -> Dict:
    """
    Run regime-switching backtest.
    """
    logger.info(f"\nRunning backtest from {start_date.date()} to {end_date.date()}")
    
    # Align dates
    trading_dates = spy_data.loc[start_date:end_date].index
    if len(trading_dates) == 0:
        logger.error("No trading dates found!")
        return {}
    
    logger.info(f"Trading days: {len(trading_dates)}")
    
    # VIX estimate
    vix_series = compute_vix_estimate(spy_data)
    
    # State tracking
    equity = initial_capital
    peak_equity = equity
    equity_curve = [equity]
    dates = []
    
    current_weights = {}
    last_rebalance_idx = 0
    consecutive_losses = 0
    
    # Regime tracking
    regime_history = []
    regime_confirmation = {}  # Regime -> days confirmed
    confirmed_regime = 'neutral'
    CONFIRMATION_DAYS = 3
    
    # Trade log
    trades = []
    
    # Daily loop
    for i, date in enumerate(trading_dates):
        # Get SPY index for full history access
        spy_idx = spy_data.index.get_loc(date)
        
        # Classify regime
        raw_regime, confidence = classify_regime(spy_data, spy_idx)
        
        # Regime confirmation logic (prevent whipsaws)
        if raw_regime != confirmed_regime:
            if raw_regime not in regime_confirmation:
                regime_confirmation = {raw_regime: 1}
            else:
                regime_confirmation[raw_regime] += 1
            
            if regime_confirmation.get(raw_regime, 0) >= CONFIRMATION_DAYS:
                confirmed_regime = raw_regime
                regime_confirmation = {}
                logger.info(f"{date.date()}: Regime CONFIRMED -> {confirmed_regime}")
        else:
            regime_confirmation = {}
        
        # Get VIX
        vix_level = vix_series.loc[date] if date in vix_series.index else 18
        
        # Current drawdown
        current_drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
        
        # Rebalance check
        should_rebalance = (i - last_rebalance_idx >= rebalance_freq) or (i == 0)
        
        if should_rebalance:
            new_weights, direction = get_allocation(
                regime=confirmed_regime,
                confidence=confidence,
                vix_level=vix_level,
                current_drawdown=current_drawdown,
                consecutive_losses=consecutive_losses,
            )
            
            # Only change if different
            if new_weights != current_weights:
                trades.append({
                    'date': date,
                    'regime': confirmed_regime,
                    'direction': direction,
                    'weights': new_weights.copy(),
                    'vix': vix_level,
                })
                current_weights = new_weights
                last_rebalance_idx = i
        
        # Calculate daily return
        daily_return = 0.0
        for ticker, weight in current_weights.items():
            if ticker in data and date in data[ticker].index:
                # Get ticker return
                ticker_idx = data[ticker].index.get_loc(date)
                if ticker_idx > 0:
                    prev_close = data[ticker]['close'].iloc[ticker_idx - 1]
                    curr_close = data[ticker]['close'].iloc[ticker_idx]
                    ticker_return = (curr_close - prev_close) / prev_close
                    daily_return += weight * ticker_return
        
        # Cash portion (1 - sum of weights)
        total_weight = sum(current_weights.values())
        # Cash earns ~0% for simplicity
        
        # Update equity
        equity *= (1 + daily_return)
        equity_curve.append(equity)
        dates.append(date)
        
        # Update peak
        if equity > peak_equity:
            peak_equity = equity
        
        # Track consecutive losses
        if daily_return < -0.001:
            consecutive_losses += 1
        else:
            consecutive_losses = 0
        
        # Regime history
        regime_history.append({'date': date, 'regime': confirmed_regime, 'raw': raw_regime})
    
    # Calculate metrics
    returns = pd.Series(equity_curve[1:], index=dates).pct_change().dropna()
    
    total_return = (equity - initial_capital) / initial_capital
    years = len(trading_dates) / 252
    cagr = (equity / initial_capital) ** (1/years) - 1 if years > 0 else 0
    
    # Max drawdown
    equity_series = pd.Series(equity_curve[1:], index=dates)
    rolling_max = equity_series.expanding().max()
    drawdowns = (equity_series - rolling_max) / rolling_max
    max_drawdown = abs(drawdowns.min())
    
    # Sharpe ratio
    if returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
    else:
        sharpe = 0
    
    # SPY benchmark
    spy_returns = spy_data['close'].pct_change().loc[start_date:end_date]
    spy_total = (spy_data['close'].loc[end_date] / spy_data['close'].loc[start_date] - 1) if start_date in spy_data['close'].index else 0
    spy_cagr = (1 + spy_total) ** (1/years) - 1 if years > 0 else 0
    alpha = cagr - spy_cagr
    
    # Regime analysis
    regime_df = pd.DataFrame(regime_history)
    regime_counts = regime_df['regime'].value_counts()
    
    results = {
        'total_return': total_return,
        'cagr': cagr,
        'max_drawdown': max_drawdown,
        'sharpe': sharpe,
        'spy_cagr': spy_cagr,
        'alpha': alpha,
        'years': years,
        'trading_days': len(trading_dates),
        'trades': len(trades),
        'regime_counts': regime_counts.to_dict(),
        'equity_curve': equity_series,
        'final_equity': equity,
    }
    
    return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 80)
    print("PHASE 12: ALL-WEATHER REGIME-SWITCHING STRATEGY")
    print("=" * 80)
    print("\nObjective: Profit in BOTH bull and bear markets")
    print("Key: Use inverse ETFs (SQQQ, SPXU, SOXS) in bear regimes")
    print("\nTargets:")
    print("  - CAGR: ≥28%")
    print("  - Max DD: ≤22%")
    print("  - Sharpe: ≥1.5")
    print("  - Alpha vs SPY: ≥5%")
    print("-" * 80)
    
    # Tickers
    all_tickers = ['SPY', 'QQQ'] + list(LONG_ETFS.keys()) + list(INVERSE_ETFS.keys())
    print(f"\nTickers: {all_tickers}")
    
    # Download data
    print("\n1. Downloading data...")
    start = "2021-06-01"  # Extra history for indicators
    end = "2025-06-01"
    
    data = download_data(all_tickers, start, end)
    
    if 'SPY' not in data:
        print("ERROR: Could not download SPY data!")
        return
    
    spy_data = data['SPY']
    
    # Test periods
    test_periods = [
        ("FULL 2022-2025 (WITH BEAR)", "2022-01-03", "2025-05-30"),
        ("2023-2025 (Bull comparison)", "2023-01-03", "2025-05-30"),
        ("2022 Bear Market Only", "2022-01-03", "2022-12-30"),
    ]
    
    results_summary = []
    
    for period_name, start_str, end_str in test_periods:
        print(f"\n{'='*80}")
        print(f"Testing: {period_name}")
        print(f"{'='*80}")
        
        start_date = pd.Timestamp(start_str)
        end_date = pd.Timestamp(end_str)
        
        results = run_backtest(
            data=data,
            spy_data=spy_data,
            start_date=start_date,
            end_date=end_date,
            rebalance_freq=5,  # Weekly
        )
        
        if results:
            print(f"\n--- RESULTS: {period_name} ---")
            print(f"  Total Return:  {results['total_return']:>8.1%}")
            print(f"  CAGR:          {results['cagr']:>8.1%}  {'✓' if results['cagr'] >= 0.28 else '✗'} (target: ≥28%)")
            print(f"  Max Drawdown:  {results['max_drawdown']:>8.1%}  {'✓' if results['max_drawdown'] <= 0.22 else '✗'} (target: ≤22%)")
            print(f"  Sharpe:        {results['sharpe']:>8.2f}  {'✓' if results['sharpe'] >= 1.5 else '✗'} (target: ≥1.5)")
            print(f"  SPY CAGR:      {results['spy_cagr']:>8.1%}")
            print(f"  Alpha:         {results['alpha']:>+8.1%}  {'✓' if results['alpha'] >= 0.05 else '✗'} (target: ≥+5%)")
            print(f"\n  Regime Distribution:")
            for regime, count in results['regime_counts'].items():
                pct = count / results['trading_days'] * 100
                print(f"    {regime:15s}: {count:4d} days ({pct:5.1f}%)")
            
            results['period'] = period_name
            results_summary.append(results)
    
    # Summary
    print("\n" + "=" * 80)
    print("PHASE 12 SUMMARY")
    print("=" * 80)
    
    print("\n                      | CAGR    | Max DD  | Sharpe  | Alpha   | Score")
    print("-" * 80)
    
    for r in results_summary:
        score = sum([
            r['cagr'] >= 0.28,
            r['max_drawdown'] <= 0.22,
            r['sharpe'] >= 1.5,
            r['alpha'] >= 0.05,
        ])
        print(f"{r['period']:22s} | {r['cagr']:6.1%} | {r['max_drawdown']:6.1%} | {r['sharpe']:6.2f} | {r['alpha']:+6.1%} | {score}/4")
    
    # Check for 2022 bear market specific
    if len(results_summary) >= 3:
        bear_2022 = results_summary[2]
        print(f"\n2022 Bear Market Analysis:")
        print(f"  Strategy return: {bear_2022['total_return']:+.1%}")
        print(f"  SPY return:      {bear_2022['spy_cagr']:.1%} (CAGR-adjusted)")
        
        # TQQQ comparison
        if 'TQQQ' in data:
            tqqq = data['TQQQ']
            tqqq_2022_start = tqqq.loc['2022-01-03':'2022-01-07']['close'].iloc[0] if '2022-01-03' <= tqqq.index[-1] else None
            tqqq_2022_end = tqqq.loc['2022-12-29':'2022-12-31']['close'].iloc[-1] if len(tqqq.loc['2022-12-29':'2022-12-31']) > 0 else None
            if tqqq_2022_start and tqqq_2022_end:
                tqqq_ret = tqqq_2022_end / tqqq_2022_start - 1
                print(f"  TQQQ return:     {tqqq_ret:+.1%} (long-only 3x)")
                print(f"  Outperformance:  {bear_2022['total_return'] - tqqq_ret:+.1%} vs TQQQ")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
