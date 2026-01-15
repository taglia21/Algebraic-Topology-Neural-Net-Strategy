#!/usr/bin/env python3
"""
Phase 6 Momentum + TDA Strategy

Uses momentum as the primary factor (70%) with TDA for structure validation (30%).
Removes sector constraints to maximize performance.

Key learnings from tuning:
- Pure momentum outperforms multi-factor approach in this period
- Sector constraints hurt returns significantly
- TDA adds marginal value for stock selection (better for timing)
- Quality/Value factors dilute momentum signal
"""

import sys
import os
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_provider import get_ohlcv_data
from src.tda_features import TDAFeatureGenerator


@dataclass
class StockScore:
    """Stock with scoring components."""
    ticker: str
    momentum_score: float
    tda_score: float
    combined_score: float
    
    # Metadata
    price: float = 0.0
    return_1y: float = 0.0


class MomentumTDAStrategy:
    """
    Momentum-first strategy with TDA validation.
    
    Factor weights:
    - Momentum (12-1 month): 70%
    - TDA structure quality: 30%
    """
    
    def __init__(
        self,
        n_stocks: int = 20,
        momentum_weight: float = 0.70,
        tda_weight: float = 0.30,
        lookback_12m: int = 252,
        lookback_skip: int = 21,  # Skip recent month
    ):
        self.n_stocks = n_stocks
        self.momentum_weight = momentum_weight
        self.tda_weight = tda_weight
        self.lookback_12m = lookback_12m
        self.lookback_skip = lookback_skip
        
        self.tda_gen = TDAFeatureGenerator(window=30, feature_mode='v1.3')
    
    def compute_momentum(self, prices: np.ndarray) -> float:
        """Compute 12-1 month momentum."""
        if len(prices) < self.lookback_12m:
            return 0.0
        
        price_12m = prices[-self.lookback_12m]
        price_1m = prices[-self.lookback_skip] if len(prices) > self.lookback_skip else prices[-1]
        
        if price_12m <= 0:
            return 0.0
        
        return (price_1m / price_12m) - 1.0
    
    def compute_tda_score(self, features: Dict[str, float]) -> float:
        """
        Convert TDA features to a bullish score (0-1).
        
        Key TDA indicators:
        - h0_persistence_ratio: Higher = stronger trend structure
        - structure_entropy: Lower = cleaner pattern
        - mean_lifetime: Moderate = healthy dynamics
        """
        if not features:
            return 0.5
        
        score = 0.5  # Neutral base
        
        # H0 persistence ratio - indicates trend strength
        h0_ratio = features.get('h0_persistence_ratio', 0.5)
        score += 0.25 * (h0_ratio - 0.5)
        
        # Structure entropy - lower is better (cleaner structure)
        entropy = features.get('structure_entropy', 1.0)
        if entropy > 0:
            # Normalize: entropy ~0.5-2.0 typical
            entropy_score = max(0, 1.0 - (entropy - 0.5) / 1.5)
            score += 0.15 * (entropy_score - 0.5)
        
        # Betti-1 count - some cycles indicate healthy volatility
        # Too many = chaos, too few = stagnation
        betti_1 = features.get('betti_1', 0)
        if 1 <= betti_1 <= 5:
            score += 0.1
        elif betti_1 > 10:
            score -= 0.05
        
        return max(0.0, min(1.0, score))
    
    def select_stocks(
        self,
        ohlcv_dict: Dict[str, pd.DataFrame],
        tda_features: Dict[str, Dict[str, float]],
        as_of_date: Optional[pd.Timestamp] = None,
    ) -> List[StockScore]:
        """
        Select top N stocks by combined momentum + TDA score.
        
        Args:
            ohlcv_dict: Dict mapping ticker to OHLCV DataFrame
            tda_features: Dict mapping ticker to TDA features
            as_of_date: Date for point-in-time selection
            
        Returns:
            List of top N StockScore objects
        """
        scores = []
        
        for ticker, df in ohlcv_dict.items():
            try:
                # Get data up to as_of_date
                if as_of_date is not None:
                    mask = df.index <= as_of_date
                    if not mask.any():
                        continue
                    df_subset = df[mask]
                else:
                    df_subset = df
                
                if len(df_subset) < self.lookback_12m:
                    continue
                
                close = df_subset['close'].values if 'close' in df_subset.columns else df_subset['Close'].values
                
                # Compute momentum
                momentum = self.compute_momentum(close)
                
                # Normalize momentum to 0-1 scale
                # Typical range: -50% to +100%
                mom_normalized = (momentum + 0.5) / 1.5
                mom_normalized = max(0.0, min(1.0, mom_normalized))
                
                # Get TDA score
                tda_score = 0.5
                if ticker in tda_features:
                    tda_score = self.compute_tda_score(tda_features[ticker])
                
                # Combined score
                combined = (self.momentum_weight * mom_normalized + 
                           self.tda_weight * tda_score)
                
                scores.append(StockScore(
                    ticker=ticker,
                    momentum_score=momentum,
                    tda_score=tda_score,
                    combined_score=combined,
                    price=close[-1],
                    return_1y=momentum,
                ))
                
            except Exception as e:
                continue
        
        # Sort by combined score and return top N
        scores.sort(key=lambda x: -x.combined_score)
        return scores[:self.n_stocks]


def run_backtest(
    universe: List[str],
    start_date: str = '2019-01-01',
    end_date: str = '2025-01-01',
    backtest_start: str = '2021-01-01',
    n_stocks: int = 20,
    rebalance_freq: str = 'monthly',
    initial_capital: float = 100000,
) -> Dict[str, Any]:
    """
    Run momentum + TDA backtest.
    
    Args:
        universe: List of tickers
        start_date: Data start date (for warmup)
        end_date: Data end date
        backtest_start: Backtest start date (after warmup)
        n_stocks: Number of stocks to hold
        rebalance_freq: Rebalancing frequency
        initial_capital: Starting capital
        
    Returns:
        Dict with backtest results
    """
    print("="*60)
    print("PHASE 6: MOMENTUM + TDA STRATEGY")
    print("="*60)
    
    # Fetch data
    print(f"\nFetching data for {len(universe)} stocks...")
    start_time = time.time()
    
    ohlcv_data = {}
    failed = []
    for i, ticker in enumerate(universe):
        try:
            df = get_ohlcv_data(ticker, start_date, end_date, provider='yfinance')
            if len(df) > 504:  # At least 2 years
                ohlcv_data[ticker] = df
        except Exception as e:
            failed.append(ticker)
        
        if (i + 1) % 25 == 0:
            print(f"  Progress: {i+1}/{len(universe)}")
    
    elapsed = time.time() - start_time
    print(f"Fetched {len(ohlcv_data)} stocks in {elapsed:.1f}s ({len(failed)} failed)")
    
    if len(ohlcv_data) < n_stocks:
        raise ValueError(f"Not enough stocks: {len(ohlcv_data)} < {n_stocks}")
    
    # Compute TDA features for all stocks
    print("\nComputing TDA features...")
    tda_gen = TDAFeatureGenerator(window=30, feature_mode='v1.3')
    tda_rolling = {}
    
    for ticker, df in ohlcv_data.items():
        try:
            features_df = tda_gen.generate_features(df)
            if len(features_df) > 0:
                tda_rolling[ticker] = features_df
        except Exception as e:
            pass
    
    print(f"TDA features computed for {len(tda_rolling)} stocks")
    
    # Initialize strategy
    strategy = MomentumTDAStrategy(n_stocks=n_stocks)
    
    # Initialize portfolio
    cash = initial_capital
    holdings = {}  # ticker -> shares
    
    # Get rebalance dates
    sample_df = list(ohlcv_data.values())[0]
    all_dates = sample_df.index.to_series().groupby(
        [sample_df.index.year, sample_df.index.month]
    ).first()
    monthly_dates = all_dates[all_dates >= backtest_start]
    
    equity_curve = []
    trade_log = []
    total_trades = 0
    
    print(f"\n{'='*60}")
    print(f"BACKTEST: {backtest_start} to {end_date}")
    print(f"{'='*60}")
    
    for date in monthly_dates:
        # Calculate current portfolio value
        portfolio_value = cash
        for ticker, shares in holdings.items():
            if ticker in ohlcv_data and date in ohlcv_data[ticker].index:
                portfolio_value += shares * ohlcv_data[ticker].loc[date]['close']
        
        equity_curve.append({
            'date': date,
            'value': portfolio_value,
            'cash': cash,
            'n_positions': len(holdings),
        })
        
        # Get TDA features as of this date
        tda_as_of_date = {}
        for ticker, df in tda_rolling.items():
            if isinstance(df.index, pd.DatetimeIndex):
                mask = df.index <= date
                if mask.any():
                    tda_as_of_date[ticker] = df[mask].iloc[-1].to_dict()
        
        # Get data subset up to this date
        ohlcv_subset = {}
        for ticker, df in ohlcv_data.items():
            mask = df.index <= date
            if mask.any():
                ohlcv_subset[ticker] = df[mask]
        
        # Select stocks
        selected = strategy.select_stocks(
            ohlcv_dict=ohlcv_subset,
            tda_features=tda_as_of_date,
            as_of_date=date,
        )
        
        target_tickers = set(s.ticker for s in selected)
        
        # Close positions not in target
        for ticker in list(holdings.keys()):
            if ticker not in target_tickers:
                if ticker in ohlcv_data and date in ohlcv_data[ticker].index:
                    sell_price = ohlcv_data[ticker].loc[date]['close']
                    sell_value = holdings[ticker] * sell_price
                    cash += sell_value
                    trade_log.append({
                        'date': date,
                        'ticker': ticker,
                        'action': 'SELL',
                        'shares': holdings[ticker],
                        'price': sell_price,
                    })
                    total_trades += 1
                del holdings[ticker]
        
        # Open/adjust positions
        if selected:
            current_value = cash + sum(
                holdings.get(t, 0) * ohlcv_data[t].loc[date]['close']
                for t in holdings if t in ohlcv_data and date in ohlcv_data[t].index
            )
            weight = 0.98 / len(selected)  # 2% cash buffer
            
            for stock_score in selected:
                ticker = stock_score.ticker
                if ticker not in ohlcv_data or date not in ohlcv_data[ticker].index:
                    continue
                
                price = ohlcv_data[ticker].loc[date]['close']
                target_shares = int((current_value * weight) / price)
                current_shares = holdings.get(ticker, 0)
                diff = target_shares - current_shares
                
                if diff > 0 and cash >= diff * price:
                    # Buy
                    cash -= diff * price
                    holdings[ticker] = current_shares + diff
                    if current_shares == 0:
                        trade_log.append({
                            'date': date,
                            'ticker': ticker,
                            'action': 'BUY',
                            'shares': diff,
                            'price': price,
                        })
                        total_trades += 1
                elif diff < 0:
                    # Reduce position
                    cash += abs(diff) * price
                    holdings[ticker] = current_shares + diff
                    if holdings[ticker] <= 0:
                        del holdings[ticker]
    
    # Final portfolio value
    final_date = monthly_dates.iloc[-1]
    final_value = cash
    for ticker, shares in holdings.items():
        if ticker in ohlcv_data and final_date in ohlcv_data[ticker].index:
            final_value += shares * ohlcv_data[ticker].loc[final_date]['close']
    
    # Calculate metrics
    total_return = (final_value / initial_capital) - 1
    years = (monthly_dates.iloc[-1] - monthly_dates.iloc[0]).days / 365.25
    cagr = (final_value / initial_capital) ** (1/years) - 1 if years > 0 else 0
    
    eq_df = pd.DataFrame(equity_curve)
    eq_df['return'] = eq_df['value'].pct_change()
    monthly_vol = eq_df['return'].std()
    sharpe = (eq_df['return'].mean() / monthly_vol) * np.sqrt(12) if monthly_vol > 0 else 0
    
    peak = eq_df['value'].expanding().max()
    drawdown = (eq_df['value'] - peak) / peak
    max_dd = drawdown.min()
    
    # SPY benchmark
    spy = get_ohlcv_data('SPY', start_date, end_date, provider='yfinance')
    spy_start = spy.loc[monthly_dates.iloc[0]]['close'] if monthly_dates.iloc[0] in spy.index else spy.iloc[0]['close']
    spy_end = spy.loc[monthly_dates.iloc[-1]]['close'] if monthly_dates.iloc[-1] in spy.index else spy.iloc[-1]['close']
    spy_return = (spy_end / spy_start) - 1
    
    # Print results
    print(f"\n{'='*60}")
    print("PHASE 6 RESULTS")
    print(f"{'='*60}")
    print(f"  Period: {monthly_dates.iloc[0].date()} to {monthly_dates.iloc[-1].date()}")
    print(f"  Universe: {len(ohlcv_data)} stocks")
    print(f"  Positions: {n_stocks}")
    print(f"\n  PERFORMANCE:")
    print(f"    Total Return: {total_return*100:.1f}%")
    print(f"    SPY Return:   {spy_return*100:.1f}%")
    print(f"    Alpha:        {(total_return - spy_return)*100:.1f}%")
    print(f"    CAGR:         {cagr*100:.1f}%")
    print(f"    Sharpe Ratio: {sharpe:.2f}")
    print(f"    Max Drawdown: {max_dd*100:.1f}%")
    print(f"    Total Trades: {total_trades}")
    
    print(f"\n  CURRENT HOLDINGS:")
    for ticker, shares in sorted(holdings.items()):
        if ticker in ohlcv_data and final_date in ohlcv_data[ticker].index:
            price = ohlcv_data[ticker].loc[final_date]['close']
            value = shares * price
            print(f"    {ticker}: {shares} shares @ ${price:.2f} = ${value:,.0f}")
    
    # Check targets
    print(f"\n{'='*60}")
    print("PHASE 6 TARGETS:")
    print(f"{'='*60}")
    cagr_pct = cagr * 100
    dd_pct = abs(max_dd * 100)
    print(f"  CAGR > 17%:     {cagr_pct:.1f}% {'✓' if cagr_pct > 17 else '✗'}")
    print(f"  Sharpe > 1.0:   {sharpe:.2f} {'✓' if sharpe > 1.0 else '✗'}")
    print(f"  Max DD < 17%:   {dd_pct:.1f}% {'✓' if dd_pct < 17 else '✗'}")
    print(f"  Beat SPY:       {(total_return - spy_return)*100:.1f}% {'✓' if total_return > spy_return else '✗'}")
    
    results = {
        'phase': 6,
        'strategy': 'momentum_tda',
        'universe_size': len(ohlcv_data),
        'n_positions': n_stocks,
        'period': {
            'start': str(monthly_dates.iloc[0].date()),
            'end': str(monthly_dates.iloc[-1].date()),
        },
        'performance': {
            'total_return_pct': round(total_return * 100, 1),
            'spy_return_pct': round(spy_return * 100, 1),
            'alpha_pct': round((total_return - spy_return) * 100, 1),
            'cagr_pct': round(cagr * 100, 1),
            'sharpe_ratio': round(sharpe, 2),
            'max_drawdown_pct': round(max_dd * 100, 1),
        },
        'trading': {
            'total_trades': total_trades,
            'avg_trades_per_month': round(total_trades / len(monthly_dates), 1),
        },
        'holdings': {ticker: shares for ticker, shares in holdings.items()},
        'targets_met': {
            'cagr_gt_17': cagr_pct > 17,
            'sharpe_gt_1': sharpe > 1.0,
            'max_dd_lt_17': dd_pct < 17,
            'beat_spy': total_return > spy_return,
        },
    }
    
    return results


if __name__ == '__main__':
    # 100 stock universe
    UNIVERSE = [
        # Technology (25)
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC', 'CRM',
        'ORCL', 'ADBE', 'CSCO', 'QCOM', 'AVGO', 'TXN', 'IBM', 'AMAT', 'MU', 'NOW',
        'INTU', 'LRCX', 'ADI', 'MCHP', 'SNPS',
        # Healthcare (20)
        'UNH', 'JNJ', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT', 'DHR', 'BMY',
        'AMGN', 'GILD', 'CVS', 'CI', 'ISRG', 'VRTX', 'REGN', 'MRNA', 'BIIB', 'ZTS',
        # Financial (15)
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'SCHW', 'BLK', 'AXP', 'C', 'USB',
        'PNC', 'BK', 'COF', 'MMC', 'AIG',
        # Consumer Cyclical (15)
        'HD', 'MCD', 'NKE', 'SBUX', 'TJX', 'COST', 'WMT', 'TGT', 'DG', 'ORLY',
        'LOW', 'ROST', 'BBY', 'DLTR', 'AZO',
        # Industrial (10)
        'CAT', 'HON', 'UPS', 'RTX', 'BA', 'LMT', 'GE', 'MMM', 'DE', 'EMR',
        # Energy (5)
        'XOM', 'CVX', 'COP', 'SLB', 'EOG',
        # Consumer Defensive (10)
        'PG', 'KO', 'PEP', 'PM', 'MO', 'MDLZ', 'CL', 'KMB', 'EL', 'STZ',
    ]
    
    results = run_backtest(
        universe=UNIVERSE,
        start_date='2019-01-01',
        end_date='2025-01-01',
        backtest_start='2021-01-01',
        n_stocks=20,
        initial_capital=100000,
    )
    
    # Save results
    output_path = 'results/phase6_momentum_tda.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")
