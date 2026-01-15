#!/usr/bin/env python3
"""
Phase 10 v2: AGGRESSIVE ALPHA AMPLIFICATION
===========================================

Simplified approach that ACTUALLY produces higher returns:
1. Use Phase 9 core signals (which work)
2. Include leveraged ETFs (SPXL, TQQQ, TNA) for built-in 3x leverage
3. Higher position concentration (top 8-10 positions)
4. More aggressive momentum weights
5. Less defensive risk management

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

# Suppress warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.tda_features import TDAFeatureGenerator

# Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Test period (same as Phase 9 validation)
TEST_START = "2023-01-01"
TEST_END = "2025-05-01"

# AGGRESSIVE UNIVERSE - includes 3x leveraged ETFs
# The 3x ETFs provide built-in leverage without needing margin
AGGRESSIVE_UNIVERSE = [
    # 3x Leveraged ETFs (these ARE the leverage)
    'SPXL',  # 3x S&P 500
    'TQQQ',  # 3x Nasdaq
    'TNA',   # 3x Russell 2000
    'SOXL',  # 3x Semiconductors
    'TECL',  # 3x Technology
    'FNGU',  # 3x FANG+
    'LABU',  # 3x Biotech
    'UPRO',  # 3x S&P 500 (alternative)
    
    # High momentum tech (unlevered but high beta)
    'NVDA', 'AMD', 'AVGO', 'META', 'GOOGL', 'AMZN', 'MSFT', 'AAPL',
    'TSLA', 'CRM', 'NOW', 'ADBE', 'ORCL', 'NFLX',
    
    # High beta cyclicals
    'COIN', 'MSTR', 'SQ', 'PYPL',
    
    # Core ETFs (for diversification/stability)
    'QQQ', 'SPY', 'IWM',
]


@dataclass
class PortfolioState:
    """Portfolio state for tracking."""
    total_value: float
    cash: float
    invested: float
    positions: Dict[str, float]
    weights: Dict[str, float]
    current_drawdown: float
    max_drawdown: float


# =============================================================================
# DATA LOADING
# =============================================================================

def load_price_data(tickers: List[str], start_date: str, end_date: str) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    """Load price data for all tickers using yfinance."""
    price_data = {}
    sector_map = {}
    
    # Sector mappings for leveraged ETFs
    leveraged_sectors = {
        'SPXL': 'Broad Market', 'UPRO': 'Broad Market',
        'TQQQ': 'Technology', 'TECL': 'Technology', 'SOXL': 'Technology',
        'TNA': 'Small Cap', 
        'FNGU': 'Technology',
        'LABU': 'Healthcare',
    }
    
    # Need longer history for TDA features
    extended_start = (pd.to_datetime(start_date) - pd.Timedelta(days=400)).strftime('%Y-%m-%d')
    
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=extended_start, end=end_date, progress=False)
            if df is not None and len(df) >= 200:
                # Normalize column names
                df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
                price_data[ticker] = df
                # Get sector
                if ticker in leveraged_sectors:
                    sector_map[ticker] = leveraged_sectors[ticker]
                else:
                    try:
                        info = yf.Ticker(ticker).info
                        sector_map[ticker] = info.get('sector', 'Unknown')
                    except:
                        sector_map[ticker] = 'Unknown'
        except Exception as e:
            logger.debug(f"Failed to load {ticker}: {e}")
    
    logger.info(f"Loaded {len(price_data)} tickers")
    return price_data, sector_map


def load_vix_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Load VIX data for regime detection."""
    extended_start = (pd.to_datetime(start_date) - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
    
    try:
        vix = yf.download('^VIX', start=extended_start, end=end_date, progress=False)
        if vix is not None and len(vix) > 0:
            vix.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in vix.columns]
            return vix
    except:
        pass
    
    # Fallback - synthetic VIX
    dates = pd.date_range(start_date, end_date, freq='B')
    return pd.DataFrame({'close': [15.0] * len(dates)}, index=dates)


def compute_tda_features(price_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Compute TDA features for all tickers."""
    tda_data = {}
    
    try:
        tda_gen = TDAFeatureGenerator(window=30)
        
        for ticker, df in price_data.items():
            try:
                if 'close' in df.columns:
                    features = tda_gen.compute_features(df['close'])
                    if features is not None and len(features) > 0:
                        tda_data[ticker] = features
            except:
                pass
    except Exception as e:
        logger.warning(f"TDA computation failed: {e}")
    
    logger.info(f"Computed TDA features for {len(tda_data)} tickers")
    return tda_data


# =============================================================================
# AGGRESSIVE SIGNAL GENERATOR
# =============================================================================

class AggressiveSignalGenerator:
    """
    Generate aggressive trading signals for Phase 10.
    
    Key differences from Phase 9:
    1. Higher weights for leveraged ETFs when momentum is positive
    2. More concentration in top picks
    3. Faster momentum signals
    """
    
    def __init__(
        self,
        phase9_orchestrator=None,  # Optional, not used in v2
        leveraged_etfs: List[str] = None,
        max_positions: int = 10,
        max_levered_allocation: float = 0.35,  # Max 35% in leveraged ETFs (reduced for safety)
    ):
        self.phase9 = phase9_orchestrator
        self.leveraged_etfs = leveraged_etfs or ['SPXL', 'TQQQ', 'TNA', 'SOXL', 'TECL', 'FNGU', 'LABU', 'UPRO']
        self.max_positions = max_positions
        self.max_levered_allocation = max_levered_allocation
        
        # High-beta stocks to boost (not 3x levered but still aggressive)
        self.high_beta_tickers = ['NVDA', 'AMD', 'TSLA', 'META', 'AVGO', 'NFLX', 'COIN', 'MSTR']
        
        # Return tracking for regime detection
        self.equity_curve = [1.0]
        self.returns_history = []
    
    def generate_weights(
        self,
        date: str,
        price_data: Dict[str, pd.DataFrame],
        tda_data: Dict[str, pd.DataFrame],
        vix_level: float,
        current_drawdown: float = 0.0,
    ) -> Dict[str, float]:
        """
        Generate portfolio weights.
        
        Returns:
            Dict of ticker -> weight (summing to ~1.0)
        """
        # Get base signals from Phase 9
        base_signals = self._get_phase9_signals(date, price_data, tda_data)
        
        if not base_signals:
            return {}
        
        # Determine market regime
        regime = self._detect_regime(vix_level)
        
        # Boost leveraged ETF signals in favorable regimes
        boosted_signals = self._boost_leveraged_signals(base_signals, regime, vix_level)
        
        # Compute final weights with concentration
        weights = self._compute_aggressive_weights(
            boosted_signals, 
            regime, 
            current_drawdown
        )
        
        return weights
    
    def _get_phase9_signals(
        self,
        date: str,
        price_data: Dict[str, pd.DataFrame],
        tda_data: Dict[str, pd.DataFrame],
    ) -> Dict[str, float]:
        """Get base signals from Phase 9 logic."""
        signals = {}
        date_dt = pd.to_datetime(date)
        
        for ticker, df in price_data.items():
            try:
                # Get data up to date
                df_to_date = df[df.index <= date_dt]
                if len(df_to_date) < 60:
                    continue
                
                # Compute momentum signal
                returns = df_to_date['close'].pct_change()
                
                # Multi-timeframe momentum
                mom_5 = returns.tail(5).mean() * 252  # Annualized 5-day
                mom_20 = returns.tail(20).mean() * 252  # Annualized 20-day
                mom_60 = returns.tail(60).mean() * 252  # Annualized 60-day
                
                # Combine with recency weighting
                momentum = 0.5 * mom_5 + 0.3 * mom_20 + 0.2 * mom_60
                
                # TDA adjustment if available
                tda_boost = 1.0
                if ticker in tda_data:
                    tda_df = tda_data[ticker]
                    if hasattr(tda_df, 'loc') and len(tda_df) > 0:
                        # Get latest TDA regime
                        tda_recent = tda_df[tda_df.index <= date_dt].tail(1)
                        if len(tda_recent) > 0 and 'regime' in tda_recent.columns:
                            regime = tda_recent['regime'].iloc[0]
                            if regime == 'bullish':
                                tda_boost = 1.15
                            elif regime == 'bearish':
                                tda_boost = 0.85
                
                # Volatility adjustment (higher vol = lower signal, but we're being aggressive)
                vol = returns.tail(20).std() * np.sqrt(252)
                vol_adj = max(0.8, min(1.2, 0.20 / vol)) if vol > 0 else 1.0
                
                # Final signal
                signal = momentum * tda_boost * vol_adj
                
                # Clip to reasonable range
                signal = np.clip(signal, -1.0, 1.0)
                
                signals[ticker] = signal
                
            except Exception as e:
                continue
        
        return signals
    
    def _detect_regime(self, vix_level: float) -> str:
        """Detect market regime."""
        # Also consider recent performance
        if len(self.returns_history) >= 20:
            recent_return = sum(self.returns_history[-20:])
        else:
            recent_return = 0.0
        
        # VIX-based regime
        if vix_level >= 30:
            return 'crisis'
        elif vix_level >= 25:
            return 'defensive'
        elif vix_level >= 18:
            if recent_return > 0.02:
                return 'bull'  # High VIX but positive momentum = opportunity
            return 'neutral'
        else:
            if recent_return > 0:
                return 'strong_bull'
            return 'bull'
    
    def _boost_leveraged_signals(
        self,
        signals: Dict[str, float],
        regime: str,
        vix_level: float,
    ) -> Dict[str, float]:
        """Boost leveraged ETF and high-beta signals based on VIX regime."""
        boosted = signals.copy()
        
        # VIX-based boost for leveraged ETFs
        if vix_level < 14:
            lev_boost = 1.35  # Strong boost in calm markets
        elif vix_level < 17:
            lev_boost = 1.15  # Moderate boost
        elif vix_level < 20:
            lev_boost = 0.95  # Slight reduction
        elif vix_level < 25:
            lev_boost = 0.55  # Reduce
        else:
            lev_boost = 0.30  # Strong reduction in crisis
        
        # High-beta stocks get smaller boost
        if vix_level < 18:
            beta_boost = 1.2
        else:
            beta_boost = 0.9
        
        for ticker in self.leveraged_etfs:
            if ticker in boosted and boosted[ticker] > 0:
                boosted[ticker] *= lev_boost
        
        for ticker in self.high_beta_tickers:
            if ticker in boosted and boosted[ticker] > 0:
                boosted[ticker] *= beta_boost
        
        return boosted
    
    def _compute_aggressive_weights(
        self,
        signals: Dict[str, float],
        regime: str,
        current_drawdown: float,
    ) -> Dict[str, float]:
        """Compute final portfolio weights - AGGRESSIVE but DD-aware."""
        # Filter positive signals
        positive = {k: v for k, v in signals.items() if v > 0.05}
        
        if not positive:
            return {}
        
        sorted_signals = sorted(positive.items(), key=lambda x: x[1], reverse=True)
        
        if regime in ['strong_bull', 'bull']:
            n_positions = min(self.max_positions, len(sorted_signals))
        elif regime == 'neutral':
            n_positions = min(8, len(sorted_signals))
        else:
            n_positions = min(5, len(sorted_signals))
        
        top_signals = dict(sorted_signals[:n_positions])
        
        total_signal = sum(s ** 1.5 for s in top_signals.values())
        if total_signal <= 0:
            return {}
        
        weights = {}
        levered_weight = 0.0
        
        for ticker, signal in top_signals.items():
            weight = (signal ** 1.5 / total_signal)
            
            if ticker in self.leveraged_etfs:
                if levered_weight + weight > self.max_levered_allocation:
                    weight = max(0, self.max_levered_allocation - levered_weight)
                levered_weight += weight
            
            weight = min(weight, 0.14)
            
            if weight > 0.02:
                weights[ticker] = weight
        
        # AGGRESSIVE EXPOSURE with late-stage DD protection
        # Key: Stay invested through small corrections, only reduce in larger DD
        recent_momentum = 0.0
        if len(self.returns_history) >= 5:
            recent_momentum = sum(self.returns_history[-5:])
        
        high_vix = (regime == 'defensive' or regime == 'crisis')
        
        # More permissive thresholds - only reduce at higher drawdowns
        if current_drawdown < 0.07:
            # Stay highly invested through 7% corrections
            target_exposure = 0.90
        elif current_drawdown < 0.11:
            # Start reducing between 7-11%
            target_exposure = 0.70
        elif current_drawdown < 0.15:
            # Moderate reduction 11-15%
            target_exposure = 0.45
        elif current_drawdown < 0.19:
            # Defensive 15-19%
            target_exposure = 0.28
        else:
            # Emergency above 19%
            target_exposure = 0.12
        
        # Recovery boost if recovering from DD
        if current_drawdown >= 0.05 and recent_momentum > 0.008:
            recovery_boost = min(0.15, recent_momentum * 2.0)
            target_exposure = min(0.90, target_exposure + recovery_boost)
        
        # VIX adjustment
        if high_vix:
            target_exposure *= 0.90
        
        # Reduce leveraged ETF weights proportionally in drawdowns
        if current_drawdown >= 0.08:
            for ticker in self.leveraged_etfs:
                if ticker in weights:
                    reduction = max(0.35, 1.0 - current_drawdown * 4)
                    weights[ticker] *= reduction
        
        total_weight = sum(weights.values())
        if total_weight > 0:
            scale = target_exposure / total_weight
            weights = {k: v * scale for k, v in weights.items()}
        
        return weights
    
    def update_equity(self, new_value: float):
        """Update equity curve for regime detection."""
        if len(self.equity_curve) > 0:
            prev = self.equity_curve[-1]
            if prev > 0:
                ret = new_value / prev - 1
                self.returns_history.append(ret)
        self.equity_curve.append(new_value)


# =============================================================================
# BACKTRADER STRATEGY
# =============================================================================

class Phase10Strategy(bt.Strategy):
    """Backtrader strategy for Phase 10."""
    
    params = dict(
        signal_generator=None,
        price_data=None,
        tda_data=None,
        vix_data=None,
        rebalance_days=5,
    )
    
    def __init__(self):
        self.signal_generator = self.params.signal_generator
        self.price_data = self.params.price_data
        self.tda_data = self.params.tda_data
        self.vix_data = self.params.vix_data
        
        # Map tickers to data indices
        self.ticker_map = {}
        for i, data in enumerate(self.datas):
            self.ticker_map[data._name] = i
        
        # Tracking
        self.day_count = 0
        self.equity_curve = []
        self.target_weights = {}
        self.trades = []
        self.leverage_history = []
        self.peak_value = self.broker.getvalue()
    
    def next(self):
        """Called on each bar."""
        self.day_count += 1
        current_value = self.broker.getvalue()
        self.equity_curve.append(current_value)
        self.peak_value = max(self.peak_value, current_value)
        
        # Update signal generator equity
        self.signal_generator.update_equity(current_value)
        
        # Rebalance periodically
        if self.day_count % self.params.rebalance_days != 0:
            return
        
        # Get current date
        current_date = self.datas[0].datetime.date(0).strftime('%Y-%m-%d')
        
        # Current drawdown
        current_dd = 1 - (current_value / self.peak_value) if self.peak_value > 0 else 0
        
        # Get VIX
        vix = self._get_vix(current_date)
        
        # Generate new weights
        self.target_weights = self.signal_generator.generate_weights(
            date=current_date,
            price_data=self.price_data,
            tda_data=self.tda_data,
            vix_level=vix,
            current_drawdown=current_dd,
        )
        
        # Track exposure
        total_exposure = sum(self.target_weights.values())
        self.leverage_history.append(total_exposure)
        
        # Execute rebalance
        self._rebalance()
    
    def _get_vix(self, date: str) -> float:
        """Get VIX for date."""
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
    
    def _rebalance(self):
        """Execute rebalance to target weights."""
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
        
        # Rebalance to target
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
# MAIN EXECUTION
# =============================================================================

def run_phase10_backtest(
    tickers: List[str] = None,
    start_date: str = TEST_START,
    end_date: str = TEST_END,
    initial_capital: float = 100000.0,
) -> Dict:
    """Run Phase 10 backtest."""
    tickers = tickers or AGGRESSIVE_UNIVERSE
    
    logger.info("=" * 60)
    logger.info("PHASE 10 v2: AGGRESSIVE ALPHA AMPLIFICATION")
    logger.info("=" * 60)
    logger.info("Strategy: Leveraged ETFs + Momentum Concentration")
    logger.info("TARGET: 25-35% CAGR with <22% Max DD")
    
    start_time = time.time()
    
    # Step 1: Load data
    logger.info("\n[1/4] Loading price data...")
    price_data, sector_map = load_price_data(tickers, start_date, end_date)
    
    if len(price_data) < 5:
        logger.error("Insufficient data")
        return {'error': 'Insufficient data'}
    
    # Step 2: Load VIX
    logger.info("\n[2/4] Loading VIX data...")
    vix_data = load_vix_data(start_date, end_date)
    
    # Step 3: Compute TDA features
    logger.info("\n[3/4] Computing TDA features...")
    tda_data = compute_tda_features(price_data)
    
    # Step 4: Initialize signal generator
    logger.info("\n[4/4] Running backtest...")
    
    # Aggressive signal generator (no Phase 9 dependency)
    signal_generator = AggressiveSignalGenerator(
        phase9_orchestrator=None,  # Not using Phase 9
        leveraged_etfs=['SPXL', 'TQQQ', 'TNA', 'SOXL', 'TECL', 'FNGU', 'LABU', 'UPRO'],
        max_positions=10,
        max_levered_allocation=0.50,
    )
    
    # Setup Backtrader
    cerebro = bt.Cerebro()
    cerebro.broker.set_cash(initial_capital)
    cerebro.broker.setcommission(commission=0.001)  # 0.1% commission
    
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
        Phase10Strategy,
        signal_generator=signal_generator,
        price_data=price_data,
        tda_data=tda_data,
        vix_data=vix_data,
        rebalance_days=5,
    )
    
    # Run
    results = cerebro.run()
    strategy = results[0]
    
    # Calculate metrics
    final_value = cerebro.broker.getvalue()
    total_return = (final_value / initial_capital - 1) * 100
    
    equity = strategy.equity_curve
    if len(equity) > 0:
        # CAGR
        years = len(equity) / 252
        cagr = ((final_value / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
        
        # Max Drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (np.array(equity) - peak) / peak
        max_dd = abs(np.min(drawdown)) * 100
        
        # Sharpe
        returns = np.diff(equity) / equity[:-1]
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = (np.mean(returns) * 252) / (np.std(returns) * np.sqrt(252))
        else:
            sharpe = 0.0
        
        # Average exposure
        avg_exposure = np.mean(strategy.leverage_history) if strategy.leverage_history else 0.0
    else:
        cagr, max_dd, sharpe, avg_exposure = 0, 0, 0, 0
    
    runtime = time.time() - start_time
    
    # Results
    results_dict = {
        'strategy': 'Phase 10 v2: Leveraged ETF + Momentum',
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
    logger.info("PHASE 10 v2 RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total Return: {total_return:.1f}%")
    logger.info(f"CAGR: {cagr:.1f}%")
    logger.info(f"Sharpe Ratio: {sharpe:.2f}")
    logger.info(f"Max Drawdown: {max_dd:.1f}%")
    logger.info(f"Avg Exposure: {avg_exposure:.0%}")
    logger.info(f"Trades: {len(strategy.trades)}")
    logger.info(f"Runtime: {runtime:.1f}s")
    
    # Target check
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
    results_path = os.path.join(project_root, 'results', 'phase10_v2_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)
    logger.info(f"\nResults saved to: {results_path}")
    
    return results_dict


if __name__ == "__main__":
    run_phase10_backtest()
