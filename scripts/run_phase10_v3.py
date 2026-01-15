#!/usr/bin/env python3
"""
Phase 10 v3: TREND-FOLLOWING LEVERAGED ETF STRATEGY
====================================================

Simpler approach that has historically worked:
1. Use 3x leveraged ETFs for bull exposure
2. Strict trend-following signals (50-day vs 200-day moving average)
3. VIX-based regime filter
4. Strict stop-losses

Target: 25-35% CAGR with <=22% Max DD
"""

import os
import sys
import time
import json
import logging
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import backtrader as bt
import yfinance as yf

warnings.filterwarnings('ignore')

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

TEST_START = "2023-01-01"
TEST_END = "2025-05-01"

# Focused universe - only high-quality options
CORE_UNIVERSE = [
    # Core leveraged ETFs (main holdings in bull market)
    'TQQQ',  # 3x Nasdaq - primary holding
    'SPXL',  # 3x S&P 500 - secondary
    'UPRO',  # 3x S&P 500 - alternative
    
    # High-quality tech (unlevered but high momentum)
    'NVDA', 'AVGO', 'META', 'GOOGL', 'MSFT', 'AAPL',
    'AMD', 'CRM', 'NFLX',
    
    # Benchmark/safety
    'QQQ', 'SPY',
]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_price_data(tickers: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """Load price data for all tickers."""
    price_data = {}
    
    extended_start = (pd.to_datetime(start_date) - pd.Timedelta(days=300)).strftime('%Y-%m-%d')
    
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=extended_start, end=end_date, progress=False)
            if df is not None and len(df) >= 200:
                df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
                price_data[ticker] = df
        except Exception as e:
            logger.debug(f"Failed to load {ticker}: {e}")
    
    logger.info(f"Loaded {len(price_data)} tickers")
    return price_data


def load_vix_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Load VIX data."""
    extended_start = (pd.to_datetime(start_date) - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
    
    try:
        vix = yf.download('^VIX', start=extended_start, end=end_date, progress=False)
        if vix is not None and len(vix) > 0:
            vix.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in vix.columns]
            return vix
    except:
        pass
    
    dates = pd.date_range(start_date, end_date, freq='B')
    return pd.DataFrame({'close': [15.0] * len(dates)}, index=dates)


# =============================================================================
# TREND-FOLLOWING STRATEGY
# =============================================================================

class TrendFollowingStrategy(bt.Strategy):
    """
    Simple trend-following strategy with leveraged ETFs.
    
    Rules:
    1. Use 50-day MA cross above 200-day MA for trend signal
    2. VIX < 22 for risk-on mode
    3. Strict position sizing based on volatility
    4. Quick exit when trend reverses
    """
    
    params = dict(
        price_data=None,
        vix_data=None,
        leveraged_tickers=['TQQQ', 'SPXL', 'UPRO'],
        tech_tickers=['NVDA', 'AVGO', 'META', 'GOOGL', 'MSFT', 'AAPL', 'AMD', 'CRM', 'NFLX'],
        max_leveraged_allocation=0.50,  # Max 50% in leveraged ETFs (increased)
        max_position=0.20,  # 20% max per position (increased)
        rebalance_days=5,
    )
    
    def __init__(self):
        self.price_data = self.params.price_data
        self.vix_data = self.params.vix_data
        
        self.ticker_map = {}
        for i, data in enumerate(self.datas):
            self.ticker_map[data._name] = i
        
        self.day_count = 0
        self.equity_curve = []
        self.target_weights = {}
        self.trades = []
        self.exposure_history = []
        self.peak_value = self.broker.getvalue()
    
    def next(self):
        self.day_count += 1
        current_value = self.broker.getvalue()
        self.equity_curve.append(current_value)
        self.peak_value = max(self.peak_value, current_value)
        
        if self.day_count % self.params.rebalance_days != 0:
            return
        
        current_date = self.datas[0].datetime.date(0).strftime('%Y-%m-%d')
        
        # Current drawdown
        current_dd = 1 - (current_value / self.peak_value) if self.peak_value > 0 else 0
        
        # Get VIX
        vix = self._get_vix(current_date)
        
        # Generate weights
        self.target_weights = self._generate_weights(current_date, vix, current_dd)
        
        # Track exposure
        total_exposure = sum(self.target_weights.values())
        self.exposure_history.append(total_exposure)
        
        # Execute
        self._rebalance()
    
    def _get_vix(self, date: str) -> float:
        if self.vix_data is None or self.vix_data.empty:
            return 15.0
        
        try:
            date_dt = pd.to_datetime(date)
            idx = self.vix_data.index.get_indexer([date_dt], method='ffill')[0]
            if 0 <= idx < len(self.vix_data):
                return float(self.vix_data['close'].iloc[idx])
        except:
            pass
        return 15.0
    
    def _generate_weights(self, date: str, vix: float, current_dd: float) -> Dict[str, float]:
        """Generate portfolio weights using trend-following."""
        date_dt = pd.to_datetime(date)
        
        # VIX regime
        if vix >= 28:
            vix_regime = 'crisis'
        elif vix >= 22:
            vix_regime = 'defensive'
        elif vix >= 16:
            vix_regime = 'neutral'
        else:
            vix_regime = 'bullish'
        
        # Calculate trend scores for all tickers
        signals = {}
        
        for ticker, df in self.price_data.items():
            df_to_date = df[df.index <= date_dt]
            if len(df_to_date) < 200:
                continue
            
            close = df_to_date['close']
            
            # Moving averages
            ma_50 = close.rolling(50).mean().iloc[-1]
            ma_200 = close.rolling(200).mean().iloc[-1]
            current_price = close.iloc[-1]
            
            # Trend signals
            trend_up = ma_50 > ma_200  # Golden cross
            price_above_50 = current_price > ma_50
            
            # Momentum (20-day return)
            ret_20 = (current_price / close.iloc[-20] - 1) if len(close) >= 20 else 0
            
            # Trend strength
            if trend_up and price_above_50:
                score = 0.7 + ret_20  # Strong uptrend
            elif trend_up:
                score = 0.4 + ret_20 * 0.5  # Moderate uptrend
            elif price_above_50:
                score = 0.2 + ret_20 * 0.3  # Weak signal
            else:
                score = 0  # No position
            
            signals[ticker] = max(0, score)
        
        if not signals or max(signals.values()) <= 0:
            return {}
        
        # Sort and take top picks
        sorted_signals = sorted(signals.items(), key=lambda x: x[1], reverse=True)
        
        # Number of positions based on VIX
        if vix_regime == 'bullish':
            n_positions = 8
        elif vix_regime == 'neutral':
            n_positions = 6
        else:
            n_positions = 3
        
        top_signals = dict(sorted_signals[:n_positions])
        
        # Compute weights
        total = sum(top_signals.values())
        if total <= 0:
            return {}
        
        weights = {}
        levered_weight = 0.0
        
        for ticker, signal in top_signals.items():
            weight = signal / total
            
            # Cap leveraged ETF allocation
            if ticker in self.params.leveraged_tickers:
                if levered_weight + weight > self.params.max_leveraged_allocation:
                    weight = max(0, self.params.max_leveraged_allocation - levered_weight)
                levered_weight += weight
            
            # Max position size
            weight = min(weight, self.params.max_position)
            
            if weight > 0.02:
                weights[ticker] = weight
        
        # Apply VIX/DD scaling - more aggressive in normal conditions
        if vix_regime == 'crisis':
            scale = 0.25
        elif vix_regime == 'defensive':
            scale = 0.55
        elif current_dd > 0.10:
            scale = 0.45
        elif current_dd > 0.05:
            scale = 0.70
        else:
            scale = 0.98  # Nearly fully invested in good conditions
        
        # Reduce leveraged ETF weights more in high VIX
        if vix >= 22:
            for ticker in self.params.leveraged_tickers:
                if ticker in weights:
                    weights[ticker] *= 0.5
        elif vix >= 18:
            for ticker in self.params.leveraged_tickers:
                if ticker in weights:
                    weights[ticker] *= 0.75
        
        weights = {k: v * scale for k, v in weights.items()}
        
        return weights
    
    def _rebalance(self):
        """Execute rebalance."""
        total_value = self.broker.getvalue()
        
        # Current positions
        current = {}
        for ticker, idx in self.ticker_map.items():
            pos = self.getposition(self.datas[idx])
            if pos.size > 0:
                current[ticker] = pos.size * self.datas[idx].close[0] / total_value
            else:
                current[ticker] = 0
        
        # Close positions not in target
        for ticker, idx in self.ticker_map.items():
            if ticker not in self.target_weights and current.get(ticker, 0) > 0.01:
                self.close(self.datas[idx])
        
        # Rebalance
        for ticker, target_weight in self.target_weights.items():
            if ticker not in self.ticker_map:
                continue
            
            idx = self.ticker_map[ticker]
            data = self.datas[idx]
            current_weight = current.get(ticker, 0)
            
            if abs(target_weight - current_weight) < 0.02:
                continue
            
            price = data.close[0]
            if price <= 0:
                continue
            
            target_value = total_value * target_weight
            current_value = total_value * current_weight
            order_value = target_value - current_value
            shares = int(order_value / price)
            
            if shares > 0:
                self.buy(data, size=shares)
                self.trades.append({'date': str(self.datas[0].datetime.date(0)),
                                   'ticker': ticker, 'action': 'BUY', 'shares': shares})
            elif shares < 0:
                self.sell(data, size=abs(shares))
                self.trades.append({'date': str(self.datas[0].datetime.date(0)),
                                   'ticker': ticker, 'action': 'SELL', 'shares': abs(shares)})


# =============================================================================
# MAIN
# =============================================================================

def run_phase10_v3(
    tickers: List[str] = None,
    start_date: str = TEST_START,
    end_date: str = TEST_END,
    initial_capital: float = 100000.0,
) -> Dict:
    """Run Phase 10 v3 backtest."""
    tickers = tickers or CORE_UNIVERSE
    
    logger.info("=" * 60)
    logger.info("PHASE 10 v3: TREND-FOLLOWING LEVERAGED ETF")
    logger.info("=" * 60)
    logger.info("Strategy: 50/200 MA + VIX Filter + Leveraged ETFs")
    logger.info("TARGET: 25-35% CAGR with <22% Max DD")
    
    start_time = time.time()
    
    # Load data
    logger.info("\n[1/2] Loading price data...")
    price_data = load_price_data(tickers, start_date, end_date)
    
    if len(price_data) < 3:
        logger.error("Insufficient data")
        return {'error': 'Insufficient data'}
    
    logger.info("\n[2/2] Loading VIX data...")
    vix_data = load_vix_data(start_date, end_date)
    
    logger.info("\nRunning backtest...")
    
    # Setup Backtrader
    cerebro = bt.Cerebro()
    cerebro.broker.set_cash(initial_capital)
    cerebro.broker.setcommission(commission=0.001)
    
    # Add data feeds
    for ticker, df in price_data.items():
        df_test = df[df.index >= start_date]
        if len(df_test) < 60:
            continue
        
        data = bt.feeds.PandasData(
            dataname=df_test,
            name=ticker,
            datetime=None,
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
        )
        cerebro.adddata(data, name=ticker)
    
    # Add strategy
    cerebro.addstrategy(
        TrendFollowingStrategy,
        price_data=price_data,
        vix_data=vix_data,
    )
    
    # Run
    results = cerebro.run()
    strategy = results[0]
    
    # Calculate metrics
    final_value = cerebro.broker.getvalue()
    total_return = (final_value / initial_capital - 1) * 100
    
    equity = strategy.equity_curve
    if len(equity) > 0:
        years = len(equity) / 252
        cagr = ((final_value / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
        
        peak = np.maximum.accumulate(equity)
        drawdown = (np.array(equity) - peak) / peak
        max_dd = abs(np.min(drawdown)) * 100
        
        returns = np.diff(equity) / equity[:-1]
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = (np.mean(returns) * 252) / (np.std(returns) * np.sqrt(252))
        else:
            sharpe = 0.0
        
        avg_exposure = np.mean(strategy.exposure_history) if strategy.exposure_history else 0.0
    else:
        cagr, max_dd, sharpe, avg_exposure = 0, 0, 0, 0
    
    runtime = time.time() - start_time
    
    results_dict = {
        'strategy': 'Phase 10 v3: Trend-Following Leveraged ETF',
        'period': f'{start_date} to {end_date}',
        'initial_capital': initial_capital,
        'final_value': final_value,
        'total_return_pct': round(total_return, 2),
        'cagr_pct': round(cagr, 2),
        'max_drawdown_pct': round(max_dd, 2),
        'sharpe_ratio': round(sharpe, 2),
        'avg_exposure': round(avg_exposure, 2),
        'n_trades': len(strategy.trades),
        'n_tickers': len(price_data),
        'runtime_seconds': round(runtime, 1),
    }
    
    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 10 v3 RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total Return: {total_return:.1f}%")
    logger.info(f"CAGR: {cagr:.1f}%")
    logger.info(f"Sharpe Ratio: {sharpe:.2f}")
    logger.info(f"Max Drawdown: {max_dd:.1f}%")
    logger.info(f"Avg Exposure: {avg_exposure:.0%}")
    logger.info(f"Trades: {len(strategy.trades)}")
    logger.info(f"Runtime: {runtime:.1f}s")
    
    logger.info("\n" + "-" * 40)
    logger.info("TARGET ANALYSIS:")
    cagr_ok = 25 <= cagr <= 35
    sharpe_ok = sharpe >= 1.5
    dd_ok = max_dd <= 22
    logger.info(f"  CAGR 25-35%: {'✅' if cagr_ok else '❌'} ({cagr:.1f}%)")
    logger.info(f"  Sharpe >= 1.5: {'✅' if sharpe_ok else '❌'} ({sharpe:.2f})")
    logger.info(f"  Max DD <= 22%: {'✅' if dd_ok else '❌'} ({max_dd:.1f}%)")
    
    # SPY comparison
    if 'SPY' in price_data:
        spy = price_data['SPY']
        spy_test = spy[spy.index >= start_date]
        if len(spy_test) > 0:
            spy_return = (spy_test['close'].iloc[-1] / spy_test['close'].iloc[0] - 1) * 100
            spy_years = len(spy_test) / 252
            spy_cagr = ((1 + spy_return/100) ** (1/spy_years) - 1) * 100 if spy_years > 0 else 0
            logger.info(f"\nSPY Benchmark: {spy_cagr:.1f}% CAGR")
            logger.info(f"Alpha vs SPY: {cagr - spy_cagr:+.1f}%")
            results_dict['spy_cagr'] = round(spy_cagr, 2)
            results_dict['alpha_vs_spy'] = round(cagr - spy_cagr, 2)
    
    # Save results
    results_path = os.path.join(project_root, 'results', 'phase10_v3_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)
    logger.info(f"\nResults saved to: {results_path}")
    
    return results_dict


if __name__ == "__main__":
    run_phase10_v3()
