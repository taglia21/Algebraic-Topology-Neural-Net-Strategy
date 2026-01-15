"""
Phase 12 v2: All-Weather Regime-Switching Strategy (IMPROVED)
==============================================================

Key improvements over v1:
1. Faster regime detection (2-day confirmation vs 3)
2. More aggressive allocation in strong regimes
3. Better momentum confirmation before switching
4. Hold positions longer in clear trends
5. Reduced whipsaw with trend strength filter

BIDIRECTIONAL ALPHA: Profit in BOTH bull and bear markets
"""

import logging
import warnings
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# =============================================================================
# ETF MAPPINGS
# =============================================================================

LONG_ETFS = {'TQQQ': 0.50, 'SPXL': 0.30, 'SOXL': 0.20}
INVERSE_ETFS = {'SQQQ': 0.50, 'SPXU': 0.30, 'SOXS': 0.20}

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
                df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
                data[ticker] = df
                logger.info(f"  {ticker}: {len(df)} days")
        except Exception as e:
            logger.warning(f"  {ticker}: Failed - {e}")
    return data


# =============================================================================
# IMPROVED REGIME DETECTION v2
# =============================================================================

class RegimeDetectorV2:
    """
    Improved regime detection with:
    - Multiple timeframe confirmation
    - Trend strength weighting
    - Reduced whipsaw via momentum filters
    """
    
    def __init__(self):
        self.current_regime = 'neutral'
        self.regime_days = 0
        self.last_switch_date = None
        self.min_hold_days = 5  # Minimum days before switching
        
    def classify(self, prices: pd.Series, vix_estimate: float = 18) -> Tuple[str, float, Dict]:
        """
        Classify market regime with multiple signals.
        
        Returns:
            regime: str
            confidence: float (0-1)
            signals: dict of individual signals
        """
        if len(prices) < 200:
            return 'neutral', 0.3, {}
        
        current_price = prices.iloc[-1]
        
        # Compute moving averages
        sma_10 = prices.rolling(10).mean().iloc[-1]
        sma_20 = prices.rolling(20).mean().iloc[-1]
        sma_50 = prices.rolling(50).mean().iloc[-1]
        sma_200 = prices.rolling(200).mean().iloc[-1]
        
        # Compute momentum
        mom_5 = (current_price / prices.iloc[-5] - 1) if len(prices) >= 5 else 0
        mom_20 = (current_price / prices.iloc[-20] - 1) if len(prices) >= 20 else 0
        mom_60 = (current_price / prices.iloc[-60] - 1) if len(prices) >= 60 else 0
        
        # Compute distance from 200 SMA (trend strength)
        trend_distance = (current_price - sma_200) / sma_200 if sma_200 > 0 else 0
        
        # Compute 20-day volatility
        returns = prices.pct_change()
        vol_20 = returns.rolling(20).std().iloc[-1] * np.sqrt(252) if len(returns) >= 20 else 0.15
        
        # Higher high / Lower low (trend continuation)
        high_10 = prices.iloc[-10:].max() if len(prices) >= 10 else current_price
        low_10 = prices.iloc[-10:].min() if len(prices) >= 10 else current_price
        making_highs = current_price >= high_10 * 0.99
        making_lows = current_price <= low_10 * 1.01
        
        signals = {
            'price_above_10sma': current_price > sma_10,
            'price_above_20sma': current_price > sma_20,
            'price_above_50sma': current_price > sma_50,
            'price_above_200sma': current_price > sma_200,
            'sma_10_above_20': sma_10 > sma_20,
            'sma_20_above_50': sma_20 > sma_50,
            'sma_50_above_200': sma_50 > sma_200,
            'mom_5_positive': mom_5 > 0.01,
            'mom_20_positive': mom_20 > 0.02,
            'mom_60_positive': mom_60 > 0.05,
            'trend_strong_up': trend_distance > 0.05,
            'trend_strong_down': trend_distance < -0.05,
            'making_highs': making_highs,
            'making_lows': making_lows,
            'low_vol': vol_20 < 0.20,
            'high_vol': vol_20 > 0.30,
            'vix_low': vix_estimate < 20,
            'vix_high': vix_estimate > 30,
        }
        
        # Count bullish/bearish signals
        bullish = sum([
            signals['price_above_200sma'],
            signals['price_above_50sma'],
            signals['sma_50_above_200'],
            signals['sma_20_above_50'],
            signals['mom_20_positive'],
            signals['mom_60_positive'],
            signals['trend_strong_up'],
            signals['making_highs'],
            signals['vix_low'] and not signals['high_vol'],
        ])
        
        bearish = sum([
            not signals['price_above_200sma'],
            not signals['price_above_50sma'],
            not signals['sma_50_above_200'],
            not signals['sma_20_above_50'],
            not signals['mom_20_positive'] and mom_20 < -0.02,
            not signals['mom_60_positive'] and mom_60 < -0.05,
            signals['trend_strong_down'],
            signals['making_lows'],
            signals['vix_high'],
        ])
        
        # Classify regime
        if bullish >= 7:
            raw_regime = 'strong_bull'
            confidence = min(1.0, 0.7 + bullish * 0.03)
        elif bullish >= 5:
            raw_regime = 'mild_bull'
            confidence = 0.5 + bullish * 0.05
        elif bearish >= 7:
            raw_regime = 'strong_bear'
            confidence = min(1.0, 0.7 + bearish * 0.03)
        elif bearish >= 5:
            raw_regime = 'mild_bear'
            confidence = 0.5 + bearish * 0.05
        else:
            raw_regime = 'neutral'
            confidence = 0.4
        
        # Apply regime stickiness (reduce whipsaw)
        final_regime = self._apply_stickiness(raw_regime, confidence)
        
        return final_regime, confidence, signals
    
    def _apply_stickiness(self, raw_regime: str, confidence: float) -> str:
        """Apply regime stickiness to reduce whipsaws."""
        self.regime_days += 1
        
        # If in a strong trend, require more conviction to switch
        if 'strong' in self.current_regime:
            if raw_regime != self.current_regime:
                # Need to see opposite strong signal OR high confidence switch
                if 'strong' in raw_regime or confidence > 0.75:
                    if self.regime_days >= self.min_hold_days:
                        self.current_regime = raw_regime
                        self.regime_days = 0
        else:
            # More easily switch from neutral/mild
            if raw_regime != self.current_regime:
                if self.regime_days >= 2:  # 2-day confirmation
                    self.current_regime = raw_regime
                    self.regime_days = 0
        
        return self.current_regime
    
    def reset(self):
        self.current_regime = 'neutral'
        self.regime_days = 0


# =============================================================================
# ALLOCATION v2
# =============================================================================

def get_allocation_v2(
    regime: str,
    confidence: float,
    vix_level: float,
    current_drawdown: float,
    trend_strength: float,
    consecutive_days_in_regime: int,
) -> Tuple[Dict[str, float], str]:
    """
    Get ETF allocation with improved position sizing.
    """
    # Direction
    if regime in ['strong_bull', 'mild_bull']:
        direction = 'long'
        base_etfs = LONG_ETFS
    elif regime in ['strong_bear', 'mild_bear']:
        direction = 'inverse'
        base_etfs = INVERSE_ETFS
    else:
        return {}, 'neutral'
    
    # Base allocation - more aggressive in strong regimes
    if regime == 'strong_bull':
        base_alloc = 0.75  # Increased from 0.65
    elif regime == 'mild_bull':
        base_alloc = 0.50
    elif regime == 'strong_bear':
        base_alloc = 0.70  # Strong inverse position
    elif regime == 'mild_bear':
        base_alloc = 0.45
    else:
        base_alloc = 0.0
    
    # Confidence bonus
    if confidence > 0.8:
        base_alloc *= 1.10
    elif confidence > 0.6:
        base_alloc *= 1.0
    else:
        base_alloc *= 0.85
    
    # Regime duration bonus (reward staying in trend)
    if consecutive_days_in_regime >= 20:
        base_alloc *= 1.10  # 10% bonus for established trend
    elif consecutive_days_in_regime >= 10:
        base_alloc *= 1.05
    
    # VIX adjustment
    if vix_level > 45:
        vix_mult = 0.25  # Crisis - very low exposure
    elif vix_level > 35:
        vix_mult = 0.45
    elif vix_level > 28:
        vix_mult = 0.65
    elif vix_level > 22:
        vix_mult = 0.85
    else:
        vix_mult = 1.0
    
    # In bear regime, high VIX is actually good for inverse (volatility helps)
    if direction == 'inverse' and vix_level > 25:
        vix_mult = min(1.0, vix_mult * 1.3)  # Boost inverse exposure
    
    base_alloc *= vix_mult
    
    # Drawdown protection (tighter for long, looser for inverse)
    if direction == 'long':
        if current_drawdown > 0.12:
            base_alloc *= 0.30
        elif current_drawdown > 0.08:
            base_alloc *= 0.50
        elif current_drawdown > 0.05:
            base_alloc *= 0.75
    else:  # inverse - looser DD limits
        if current_drawdown > 0.18:
            base_alloc *= 0.40
        elif current_drawdown > 0.12:
            base_alloc *= 0.60
    
    # Cap final allocation
    base_alloc = max(0.10, min(0.80, base_alloc))
    
    # Apply to ETFs
    weights = {ticker: weight * base_alloc for ticker, weight in base_etfs.items()}
    
    return weights, direction


# =============================================================================
# BACKTEST ENGINE v2
# =============================================================================

def run_backtest_v2(
    data: Dict[str, pd.DataFrame],
    spy_data: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
    rebalance_freq: int = 3,  # Faster rebalancing
    initial_capital: float = 100000,
) -> Dict:
    """Run improved regime-switching backtest."""
    
    logger.info(f"\nRunning V2 backtest from {start_date.date()} to {end_date.date()}")
    
    # Align dates
    trading_dates = spy_data.loc[start_date:end_date].index
    if len(trading_dates) == 0:
        return {}
    
    logger.info(f"Trading days: {len(trading_dates)}")
    
    # Initialize detector
    detector = RegimeDetectorV2()
    
    # State tracking
    equity = initial_capital
    peak_equity = equity
    equity_curve = [equity]
    dates = []
    
    current_weights = {}
    last_rebalance_idx = 0
    consecutive_regime_days = 0
    last_regime = 'neutral'
    
    # Daily loop
    for i, date in enumerate(trading_dates):
        # Get historical prices up to this date
        prices = spy_data['close'].loc[:date]
        
        # Estimate VIX from volatility
        returns = prices.pct_change()
        vol_20 = returns.iloc[-20:].std() * np.sqrt(252) * 100 if len(returns) >= 20 else 18
        vix_estimate = vol_20
        
        # Classify regime
        regime, confidence, signals = detector.classify(prices, vix_estimate)
        
        # Track consecutive days in regime
        if regime == last_regime:
            consecutive_regime_days += 1
        else:
            consecutive_regime_days = 1
            last_regime = regime
        
        # Current drawdown
        current_drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
        
        # Trend strength
        trend_strength = signals.get('trend_strong_up', 0) * 0.5 + signals.get('trend_strong_down', 0) * (-0.5)
        
        # Rebalance check
        should_rebalance = (i - last_rebalance_idx >= rebalance_freq) or (i == 0)
        
        # Force rebalance on regime change
        if regime != last_regime and 'strong' in regime:
            should_rebalance = True
        
        if should_rebalance:
            new_weights, direction = get_allocation_v2(
                regime=regime,
                confidence=confidence,
                vix_level=vix_estimate,
                current_drawdown=current_drawdown,
                trend_strength=trend_strength,
                consecutive_days_in_regime=consecutive_regime_days,
            )
            
            if new_weights != current_weights:
                current_weights = new_weights
                last_rebalance_idx = i
        
        # Calculate daily return
        daily_return = 0.0
        for ticker, weight in current_weights.items():
            if ticker in data and date in data[ticker].index:
                ticker_idx = data[ticker].index.get_loc(date)
                if ticker_idx > 0:
                    prev_close = data[ticker]['close'].iloc[ticker_idx - 1]
                    curr_close = data[ticker]['close'].iloc[ticker_idx]
                    ticker_return = (curr_close - prev_close) / prev_close
                    daily_return += weight * ticker_return
        
        # Update equity
        equity *= (1 + daily_return)
        equity_curve.append(equity)
        dates.append(date)
        
        # Update peak
        if equity > peak_equity:
            peak_equity = equity
    
    # Calculate metrics
    equity_series = pd.Series(equity_curve[1:], index=dates)
    returns = equity_series.pct_change().dropna()
    
    total_return = (equity - initial_capital) / initial_capital
    years = len(trading_dates) / 252
    cagr = (equity / initial_capital) ** (1/years) - 1 if years > 0 else 0
    
    # Max drawdown
    rolling_max = equity_series.expanding().max()
    drawdowns = (equity_series - rolling_max) / rolling_max
    max_drawdown = abs(drawdowns.min())
    
    # Sharpe
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    
    # SPY benchmark
    spy_start = spy_data['close'].loc[start_date:start_date].iloc[0] if len(spy_data['close'].loc[start_date:start_date]) > 0 else spy_data['close'].iloc[0]
    spy_end = spy_data['close'].loc[:end_date].iloc[-1]
    spy_total = spy_end / spy_start - 1
    spy_cagr = (1 + spy_total) ** (1/years) - 1 if years > 0 else 0
    alpha = cagr - spy_cagr
    
    return {
        'total_return': total_return,
        'cagr': cagr,
        'max_drawdown': max_drawdown,
        'sharpe': sharpe,
        'spy_cagr': spy_cagr,
        'alpha': alpha,
        'years': years,
        'trading_days': len(trading_dates),
        'equity_curve': equity_series,
        'final_equity': equity,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("PHASE 12 v2: ALL-WEATHER REGIME-SWITCHING (IMPROVED)")
    print("=" * 80)
    print("\nImprovements:")
    print("  - Faster 2-day regime confirmation")
    print("  - More aggressive allocation in strong trends")
    print("  - Regime duration bonus (reward established trends)")
    print("  - VIX boost for inverse positions in bear markets")
    print("-" * 80)
    
    # Tickers
    all_tickers = ['SPY', 'QQQ'] + list(LONG_ETFS.keys()) + list(INVERSE_ETFS.keys())
    print(f"\nTickers: {all_tickers}")
    
    # Download data
    print("\n1. Downloading data...")
    start = "2021-06-01"
    end = "2025-06-01"
    data = download_data(all_tickers, start, end)
    
    if 'SPY' not in data:
        print("ERROR: Could not download SPY data!")
        return
    
    spy_data = data['SPY']
    
    # Test periods
    test_periods = [
        ("FULL 2022-2025 (WITH BEAR)", "2022-01-03", "2025-05-30"),
        ("2023-2025 (Bull)", "2023-01-03", "2025-05-30"),
        ("2022 Bear Market", "2022-01-03", "2022-12-30"),
    ]
    
    results_summary = []
    
    for period_name, start_str, end_str in test_periods:
        print(f"\n{'='*80}")
        print(f"Testing: {period_name}")
        print(f"{'='*80}")
        
        start_date = pd.Timestamp(start_str)
        end_date = pd.Timestamp(end_str)
        
        results = run_backtest_v2(
            data=data,
            spy_data=spy_data,
            start_date=start_date,
            end_date=end_date,
            rebalance_freq=3,
        )
        
        if results:
            print(f"\n--- RESULTS: {period_name} ---")
            print(f"  Total Return:  {results['total_return']:>8.1%}")
            print(f"  CAGR:          {results['cagr']:>8.1%}  {'✓' if results['cagr'] >= 0.28 else '✗'}")
            print(f"  Max Drawdown:  {results['max_drawdown']:>8.1%}  {'✓' if results['max_drawdown'] <= 0.22 else '✗'}")
            print(f"  Sharpe:        {results['sharpe']:>8.2f}  {'✓' if results['sharpe'] >= 1.5 else '✗'}")
            print(f"  SPY CAGR:      {results['spy_cagr']:>8.1%}")
            print(f"  Alpha:         {results['alpha']:>+8.1%}  {'✓' if results['alpha'] >= 0.05 else '✗'}")
            
            results['period'] = period_name
            results_summary.append(results)
    
    # Summary
    print("\n" + "=" * 80)
    print("PHASE 12 v2 SUMMARY")
    print("=" * 80)
    
    print("\n                          | CAGR    | Max DD  | Sharpe  | Alpha   | Score")
    print("-" * 80)
    
    for r in results_summary:
        score = sum([
            r['cagr'] >= 0.28,
            r['max_drawdown'] <= 0.22,
            r['sharpe'] >= 1.5,
            r['alpha'] >= 0.05,
        ])
        print(f"{r['period']:26s} | {r['cagr']:6.1%} | {r['max_drawdown']:6.1%} | {r['sharpe']:6.2f} | {r['alpha']:+6.1%} | {score}/4")
    
    # TQQQ comparison for 2022
    if 'TQQQ' in data and len(results_summary) >= 3:
        tqqq = data['TQQQ']
        try:
            tqqq_start = tqqq.loc['2022-01-03':'2022-01-10']['close'].iloc[0]
            tqqq_end = tqqq.loc['2022-12-28':'2022-12-31']['close'].iloc[-1]
            tqqq_ret = tqqq_end / tqqq_start - 1
            
            bear_2022 = results_summary[2]
            print(f"\n2022 Bear Market Comparison:")
            print(f"  Strategy:  {bear_2022['total_return']:+.1%}")
            print(f"  TQQQ:      {tqqq_ret:+.1%} (long-only 3x)")
            print(f"  Advantage: {bear_2022['total_return'] - tqqq_ret:+.1%}")
        except Exception as e:
            print(f"\nCould not compute TQQQ comparison: {e}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
