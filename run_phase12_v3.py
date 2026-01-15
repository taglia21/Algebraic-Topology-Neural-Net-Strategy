"""
Phase 12 v3: All-Weather Regime-Switching Strategy (TREND-FOLLOWING)
=====================================================================

Key insight from Phase 11 v3: Trend-following with fast exits works.

This version combines:
1. Clear trend signals from SMA alignment
2. Cash as default (no position in unclear markets)
3. Fast exit when trend breaks (protective stops)
4. Inverse positions ONLY in confirmed downtrends
5. Strict drawdown limits per position

The strategy is more conservative but aims for consistent alpha.
"""

import logging
import warnings
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# =============================================================================
# CONSTANTS
# =============================================================================

# Long 3x ETFs for uptrends
LONG_ETFS = {'TQQQ': 0.50, 'SPXL': 0.30, 'SOXL': 0.20}

# Inverse 3x ETFs for downtrends
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
                print(f"  {ticker}: {len(df)} days")
        except Exception as e:
            print(f"  {ticker}: Failed - {e}")
    return data


# =============================================================================
# TREND-FOLLOWING REGIME DETECTOR v3
# =============================================================================

class TrendFollowingDetector:
    """
    Simple but effective trend detection:
    
    UPTREND: Price > 20 SMA > 50 SMA > 200 SMA + positive momentum
    DOWNTREND: Price < 20 SMA < 50 SMA < 200 SMA + negative momentum
    NEUTRAL: Everything else (go to cash)
    """
    
    def __init__(self):
        self.position = 'cash'  # 'long', 'inverse', 'cash'
        self.entry_price = 0
        self.days_in_position = 0
        
    def get_signal(self, prices: pd.Series, high: pd.Series, low: pd.Series) -> Tuple[str, float, Dict]:
        """
        Get trading signal based on trend-following rules.
        
        Returns:
            position: 'long', 'inverse', 'cash'
            allocation: 0.0 to 1.0
            signals: dict of indicators
        """
        if len(prices) < 200:
            return 'cash', 0.0, {}
        
        current = prices.iloc[-1]
        
        # SMAs
        sma_20 = prices.rolling(20).mean().iloc[-1]
        sma_50 = prices.rolling(50).mean().iloc[-1]
        sma_200 = prices.rolling(200).mean().iloc[-1]
        
        # Momentum
        mom_10 = (current / prices.iloc[-10] - 1) if len(prices) >= 10 else 0
        mom_20 = (current / prices.iloc[-20] - 1) if len(prices) >= 20 else 0
        
        # Volatility (for position sizing)
        returns = prices.pct_change()
        vol_20 = returns.iloc[-20:].std() * np.sqrt(252) if len(returns) >= 20 else 0.2
        
        # ATR for stops
        atr_20 = (high - low).rolling(20).mean().iloc[-1] if len(high) >= 20 else current * 0.02
        atr_pct = atr_20 / current
        
        # Trend distance (% from 200 SMA)
        trend_distance = (current - sma_200) / sma_200
        
        signals = {
            'price': current,
            'sma_20': sma_20,
            'sma_50': sma_50,
            'sma_200': sma_200,
            'mom_10': mom_10,
            'mom_20': mom_20,
            'vol_20': vol_20,
            'atr_pct': atr_pct,
            'trend_distance': trend_distance,
        }
        
        # UPTREND CONDITIONS
        uptrend_sma = current > sma_20 > sma_50 > sma_200
        uptrend_momentum = mom_20 > 0.01  # Some positive momentum
        strong_uptrend = trend_distance > 0.05 and mom_10 > 0.02
        
        # DOWNTREND CONDITIONS  
        downtrend_sma = current < sma_20 < sma_50 < sma_200
        downtrend_momentum = mom_20 < -0.01  # Some negative momentum
        strong_downtrend = trend_distance < -0.05 and mom_10 < -0.02
        
        # Mixed/weak signals
        above_200 = current > sma_200
        below_200 = current < sma_200
        
        # Determine position
        if uptrend_sma and uptrend_momentum:
            new_position = 'long'
            # Allocation based on trend strength
            if strong_uptrend:
                allocation = 0.70
            else:
                allocation = 0.50
        elif downtrend_sma and downtrend_momentum:
            new_position = 'inverse'
            # Allocation based on trend strength
            if strong_downtrend:
                allocation = 0.65
            else:
                allocation = 0.45
        else:
            # Partial allocation for weak signals
            if above_200 and mom_20 > 0.02:
                new_position = 'long'
                allocation = 0.30  # Small long
            elif below_200 and mom_20 < -0.02:
                new_position = 'inverse'
                allocation = 0.30  # Small inverse
            else:
                new_position = 'cash'
                allocation = 0.0
        
        # Volatility adjustment
        if vol_20 > 0.35:  # Very high vol
            allocation *= 0.50
        elif vol_20 > 0.25:  # High vol
            allocation *= 0.70
        
        # Position tracking
        if new_position == self.position:
            self.days_in_position += 1
        else:
            self.position = new_position
            self.days_in_position = 1
            self.entry_price = current
        
        signals['position'] = new_position
        signals['allocation'] = allocation
        
        return new_position, allocation, signals
    
    def check_stop(self, current_price: float, allocation: float) -> Tuple[bool, float]:
        """
        Check if stop-loss is triggered.
        
        Returns:
            stopped: bool
            new_allocation: float
        """
        if self.position == 'cash' or self.entry_price == 0:
            return False, allocation
        
        pnl = (current_price - self.entry_price) / self.entry_price
        
        if self.position == 'long':
            # Stop if down 5% from entry
            if pnl < -0.05:
                self.position = 'cash'
                return True, 0.0
            # Reduce if down 3%
            elif pnl < -0.03:
                return False, allocation * 0.50
        
        elif self.position == 'inverse':
            # Inverse profits when price goes down
            # Stop if price goes UP 5% (our inverse loses)
            if pnl > 0.05:
                self.position = 'cash'
                return True, 0.0
            elif pnl > 0.03:
                return False, allocation * 0.50
        
        return False, allocation
    
    def reset(self):
        self.position = 'cash'
        self.entry_price = 0
        self.days_in_position = 0


# =============================================================================
# BACKTEST ENGINE v3
# =============================================================================

def run_backtest_v3(
    data: Dict[str, pd.DataFrame],
    spy_data: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
    initial_capital: float = 100000,
) -> Dict:
    """Run trend-following regime-switching backtest."""
    
    print(f"\nRunning V3 backtest from {start_date.date()} to {end_date.date()}")
    
    # Align dates
    trading_dates = spy_data.loc[start_date:end_date].index
    if len(trading_dates) == 0:
        return {}
    
    print(f"Trading days: {len(trading_dates)}")
    
    # Initialize detector
    detector = TrendFollowingDetector()
    
    # State tracking
    equity = initial_capital
    peak_equity = equity
    equity_curve = [equity]
    dates = []
    
    current_weights = {}
    position_log = []
    
    # Drawdown tracking per regime
    regime_peak = {'long': equity, 'inverse': equity}
    
    # Daily loop
    for i, date in enumerate(trading_dates):
        # Get historical data up to this date
        prices = spy_data['close'].loc[:date]
        highs = spy_data['high'].loc[:date] if 'high' in spy_data.columns else prices
        lows = spy_data['low'].loc[:date] if 'low' in spy_data.columns else prices
        
        # Get signal
        position, allocation, signals = detector.get_signal(prices, highs, lows)
        
        # Check stop-loss
        stopped, allocation = detector.check_stop(prices.iloc[-1], allocation)
        if stopped:
            position = 'cash'
        
        # Portfolio drawdown protection
        current_drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
        
        # Reduce allocation during drawdown
        if current_drawdown > 0.15:
            allocation *= 0.30  # Aggressive reduction
        elif current_drawdown > 0.10:
            allocation *= 0.50
        elif current_drawdown > 0.05:
            allocation *= 0.75
        
        # Build weights
        if position == 'long':
            current_weights = {t: w * allocation for t, w in LONG_ETFS.items()}
        elif position == 'inverse':
            current_weights = {t: w * allocation for t, w in INVERSE_ETFS.items()}
        else:
            current_weights = {}
        
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
        
        # Log position changes
        if i == 0 or (position_log and position_log[-1]['position'] != position):
            position_log.append({
                'date': date,
                'position': position,
                'allocation': allocation,
                'spy_price': prices.iloc[-1],
            })
    
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
    spy_start = spy_data['close'].loc[start_date:].iloc[0]
    spy_end = spy_data['close'].loc[:end_date].iloc[-1]
    spy_total = spy_end / spy_start - 1
    spy_cagr = (1 + spy_total) ** (1/years) - 1 if years > 0 else 0
    alpha = cagr - spy_cagr
    
    # Position analysis
    position_df = pd.DataFrame(position_log)
    position_counts = position_df['position'].value_counts() if len(position_df) > 0 else {}
    
    return {
        'total_return': total_return,
        'cagr': cagr,
        'max_drawdown': max_drawdown,
        'sharpe': sharpe,
        'spy_cagr': spy_cagr,
        'alpha': alpha,
        'years': years,
        'trading_days': len(trading_dates),
        'position_changes': len(position_log),
        'position_counts': position_counts,
        'equity_curve': equity_series,
        'final_equity': equity,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("PHASE 12 v3: TREND-FOLLOWING REGIME-SWITCHING")
    print("=" * 80)
    print("\nStrategy:")
    print("  - UPTREND (SMA aligned up): Long 3x ETFs (TQQQ, SPXL, SOXL)")
    print("  - DOWNTREND (SMA aligned down): Inverse 3x ETFs (SQQQ, SPXU, SOXS)")
    print("  - NEUTRAL: Cash (no position)")
    print("  - 5% stop-loss on all positions")
    print("  - Aggressive DD protection")
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
        
        results = run_backtest_v3(
            data=data,
            spy_data=spy_data,
            start_date=start_date,
            end_date=end_date,
        )
        
        if results:
            print(f"\n--- RESULTS: {period_name} ---")
            print(f"  Total Return:  {results['total_return']:>8.1%}")
            print(f"  CAGR:          {results['cagr']:>8.1%}  {'✓' if results['cagr'] >= 0.28 else '✗'}")
            print(f"  Max Drawdown:  {results['max_drawdown']:>8.1%}  {'✓' if results['max_drawdown'] <= 0.22 else '✗'}")
            print(f"  Sharpe:        {results['sharpe']:>8.2f}  {'✓' if results['sharpe'] >= 1.5 else '✗'}")
            print(f"  SPY CAGR:      {results['spy_cagr']:>8.1%}")
            print(f"  Alpha:         {results['alpha']:>+8.1%}  {'✓' if results['alpha'] >= 0.05 else '✗'}")
            print(f"  Position changes: {results['position_changes']}")
            
            results['period'] = period_name
            results_summary.append(results)
    
    # Summary
    print("\n" + "=" * 80)
    print("PHASE 12 v3 SUMMARY")
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
            print(f"  SPY:       {bear_2022['spy_cagr']:+.1%}")
            print(f"  TQQQ:      {tqqq_ret:+.1%} (long-only 3x)")
            print(f"  Advantage vs TQQQ: {bear_2022['total_return'] - tqqq_ret:+.1%}")
        except Exception as e:
            print(f"\nCould not compute TQQQ comparison: {e}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
