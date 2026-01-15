#!/usr/bin/env python3
"""Phase 10: Aggressive Alpha Amplification.

Main execution script for Phase 10 trading system targeting 25-35% CAGR.

KEY CHANGES FROM PHASE 9:
- Dynamic leverage (0.5x - 1.5x) based on regime
- Kelly-optimal position sizing (45% fractional Kelly)
- Concentrated portfolio (15 positions max)
- Higher volatility target (22% vs 15%)
- Aggressive drawdown controls (22% max DD acceptable)

TARGETS:
- CAGR: 25-35% (vs Phase 9's 12.2%)
- Sharpe Ratio: > 1.5 (accepting lower Sharpe for higher returns)
- Max Drawdown: < 22% (vs Phase 9's 14.7%)
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import backtrader as bt

# Setup path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Phase 10 imports
from src.phase10 import (
    Phase10Orchestrator,
    Phase10Config,
    LeverageState,
)
from src.phase10.dynamic_leverage import LeverageRegime

# Phase 9 imports for components
from src.phase9.dynamic_optimizer import PortfolioState
from src.tda_features import TDAFeatureGenerator

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('Phase10')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Expanded universe with high-beta stocks for aggressive returns
AGGRESSIVE_UNIVERSE = [
    # Core ETFs
    'SPY', 'QQQ', 'IWM', 'DIA',
    # Sector ETFs
    'XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLY', 'XLP', 'XLU',
    # High-beta tech leaders
    'NVDA', 'AMD', 'TSLA', 'META', 'GOOGL', 'AMZN', 'MSFT', 'AAPL',
    # High-momentum growth
    'AVGO', 'CRM', 'NFLX', 'ADBE', 'NOW', 'PANW', 'CRWD',
    # Large-cap momentum
    'LLY', 'UNH', 'V', 'MA', 'COST', 'HD',
    # Leveraged ETFs (for aggressive phases)
    'SPXL', 'TQQQ',
]

# Date configuration
TEST_START = "2023-01-01"
TEST_END = "2025-12-31"

# Output paths
RESULTS_DIR = '/workspaces/Algebraic-Topology-Neural-Net-Strategy/results'
RESULTS_JSON = f'{RESULTS_DIR}/phase10_results.json'
REPORT_PATH = f'{RESULTS_DIR}/PHASE10_REPORT.md'


# =============================================================================
# DATA LOADING
# =============================================================================

def load_price_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    """Load price data for all tickers."""
    import yfinance as yf
    
    logger.info(f"Loading data for {len(tickers)} tickers: {start_date} to {end_date}")
    
    # Add buffer for lookback
    buffer_start = pd.to_datetime(start_date) - pd.Timedelta(days=300)
    
    price_data = {}
    sector_map = {}
    
    # Sector mapping for ETFs
    etf_sectors = {
        'SPY': 'Market', 'QQQ': 'Technology', 'IWM': 'SmallCap', 'DIA': 'LargeCap',
        'XLK': 'Technology', 'XLF': 'Financial', 'XLV': 'Healthcare',
        'XLE': 'Energy', 'XLI': 'Industrial', 'XLY': 'ConsumerDiscretionary',
        'XLP': 'ConsumerStaples', 'XLU': 'Utilities',
        'SPXL': 'Leveraged', 'TQQQ': 'Leveraged',
    }
    
    failed = []
    
    for ticker in tickers:
        try:
            df = yf.download(
                ticker,
                start=buffer_start.strftime('%Y-%m-%d'),
                end=end_date,
                progress=False,
            )
            
            if df.empty or len(df) < 100:
                logger.warning(f"Insufficient data for {ticker}")
                failed.append(ticker)
                continue
            
            # Normalize column names
            df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
            
            price_data[ticker] = df
            
            # Get sector
            if ticker in etf_sectors:
                sector_map[ticker] = etf_sectors[ticker]
            else:
                try:
                    info = yf.Ticker(ticker).info
                    sector_map[ticker] = info.get('sector', 'Technology')
                except:
                    sector_map[ticker] = 'Technology'
                    
        except Exception as e:
            logger.warning(f"Failed to load {ticker}: {e}")
            failed.append(ticker)
    
    logger.info(f"Loaded {len(price_data)}/{len(tickers)} tickers. Failed: {failed}")
    
    return price_data, sector_map


def load_vix_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Load VIX data for volatility regime detection."""
    import yfinance as yf
    
    buffer_start = pd.to_datetime(start_date) - pd.Timedelta(days=60)
    
    try:
        vix = yf.download('^VIX', start=buffer_start, end=end_date, progress=False)
        if not vix.empty:
            vix.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in vix.columns]
            return vix
    except Exception as e:
        logger.warning(f"Failed to load VIX: {e}")
    
    return pd.DataFrame()


def compute_tda_features(
    price_data: Dict[str, pd.DataFrame],
    window: int = 30,
) -> Dict[str, pd.DataFrame]:
    """Compute TDA features for all tickers."""
    logger.info(f"Computing TDA features for {len(price_data)} tickers...")
    
    tda_gen = TDAFeatureGenerator(
        window=window,
        feature_mode='v1.3',
    )
    
    tda_data = {}
    
    for ticker, df in price_data.items():
        try:
            features = tda_gen.generate_features(df)
            if features is not None and len(features) > 0:
                tda_data[ticker] = features
        except Exception as e:
            logger.debug(f"TDA failed for {ticker}: {e}")
    
    logger.info(f"Computed TDA for {len(tda_data)}/{len(price_data)} tickers")
    return tda_data


# =============================================================================
# PHASE 10 BACKTEST STRATEGY
# =============================================================================

class Phase10Strategy(bt.Strategy):
    """Backtrader strategy with dynamic leverage."""
    
    params = (
        ('phase10_system', None),
        ('vix_data', None),
        ('tda_data', None),
        ('sector_map', None),
        ('initial_capital', 100000.0),
        ('rebalance_frequency', 2),  # Every 2 days
        ('max_position', 0.15),
        ('verbose', False),
    )
    
    def __init__(self):
        self.order = None
        self.bar_count = 0
        self.trades = []
        self.equity_curve = []
        self.leverage_history = []
        
        # Phase 10 system
        self.p10 = self.params.phase10_system
        
        # Track data feeds by ticker
        self.ticker_map = {}
        for i, data in enumerate(self.datas):
            ticker = data._name
            self.ticker_map[ticker] = i
        
        # State
        self.target_weights = {}
        self.current_leverage = 1.0
        
        logger.info(f"Phase10Strategy initialized with {len(self.datas)} data feeds")
    
    def next(self):
        self.bar_count += 1
        
        # Get current date
        current_date = self.datas[0].datetime.date(0).isoformat()
        
        # Track equity
        portfolio_value = self.broker.getvalue()
        self.equity_curve.append(portfolio_value)
        
        # Update Phase 10 with current portfolio value
        self.p10.update_portfolio_value(portfolio_value)
        
        # Rebalance check
        if self.bar_count % self.params.rebalance_frequency != 0:
            return
        
        # Build current price data
        prices = {}
        volumes = {}
        for ticker, idx in self.ticker_map.items():
            data = self.datas[idx]
            
            # Get historical prices
            close_prices = []
            for i in range(-min(252, len(data)), 1):
                try:
                    close_prices.append(data.close[i])
                except:
                    pass
            
            if len(close_prices) >= 60:
                prices[ticker] = np.array(close_prices)
                volumes[ticker] = np.ones_like(close_prices) * 1e6
        
        if len(prices) < 10:
            return
        
        # Get SPY data for regime
        spy_prices = pd.DataFrame({'close': prices.get('SPY', np.array([100.0]))})
        
        # Get VIX level
        vix_level = self._get_current_vix(current_date)
        
        # Build portfolio state
        portfolio_state = self._get_portfolio_state()
        
        # Run Phase 10 analysis
        state = self.p10.process_day(
            date=current_date,
            spy_prices=spy_prices,
            universe_prices=prices,
            universe_volumes=volumes,
            sector_map=self.params.sector_map or {},
            vix_level=vix_level,
            tda_data=self.params.tda_data,
            current_portfolio=portfolio_state,
        )
        
        # Get leverage and weights
        self.current_leverage = state.actual_leverage
        self.leverage_history.append(self.current_leverage)
        self.target_weights = state.adjusted_weights
        
        # Execute rebalance
        self._rebalance(current_date)
    
    def _get_current_vix(self, date: str) -> float:
        """Get VIX level for current date."""
        if self.params.vix_data is None or self.params.vix_data.empty:
            return 15.0  # Default
        
        try:
            vix_df = self.params.vix_data
            date_dt = pd.to_datetime(date)
            
            # Find closest date
            idx = vix_df.index.get_indexer([date_dt], method='ffill')[0]
            if idx >= 0 and idx < len(vix_df):
                return float(vix_df['close'].iloc[idx])
        except:
            pass
        
        return 15.0
    
    def _get_portfolio_state(self) -> PortfolioState:
        """Get current portfolio state."""
        total_value = self.broker.getvalue()
        cash = self.broker.getcash()
        
        positions = {}
        weights = {}
        
        for ticker, idx in self.ticker_map.items():
            pos = self.getposition(self.datas[idx])
            if pos.size > 0:
                value = pos.size * self.datas[idx].close[0]
                positions[ticker] = value
                weights[ticker] = value / total_value if total_value > 0 else 0
        
        # Drawdown
        if self.equity_curve:
            peak = max(self.equity_curve)
            drawdown = (total_value / peak - 1) if peak > 0 else 0
            max_dd = min(0, min([(v / max(self.equity_curve[:i+1]) - 1) 
                                  for i, v in enumerate(self.equity_curve)]))
        else:
            drawdown = 0
            max_dd = 0
        
        return PortfolioState(
            total_value=total_value,
            cash=cash,
            invested=total_value - cash,
            positions=positions,
            weights=weights,
            current_drawdown=drawdown,
            max_drawdown=max_dd,
            portfolio_beta=1.0,
            portfolio_volatility=0.20,
        )
    
    def _rebalance(self, date: str):
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
        
        # Calculate trades
        for ticker, target_weight in self.target_weights.items():
            if ticker not in self.ticker_map:
                continue
            
            idx = self.ticker_map[ticker]
            data = self.datas[idx]
            current_weight = current.get(ticker, 0)
            
            weight_diff = target_weight - current_weight
            
            if abs(weight_diff) < 0.02:  # 2% threshold
                continue
            
            # Calculate order
            price = data.close[0]
            if price <= 0:
                continue
            
            target_value = total_value * target_weight
            current_value = total_value * current_weight
            order_value = target_value - current_value
            shares = int(order_value / price)
            
            if shares != 0:
                self.order = self.order_target_percent(data, target_weight)
                
                self.trades.append({
                    'date': date,
                    'ticker': ticker,
                    'action': 'buy' if shares > 0 else 'sell',
                    'shares': abs(shares),
                    'price': price,
                    'value': abs(order_value),
                    'leverage': self.current_leverage,
                })
                
                if self.params.verbose:
                    logger.info(f"{date}: {ticker} -> {target_weight:.1%} (lev: {self.current_leverage:.2f}x)")
        
        # Close positions not in target
        for ticker in current:
            if ticker not in self.target_weights and current[ticker] > 0.01:
                idx = self.ticker_map[ticker]
                self.order = self.order_target_percent(self.datas[idx], 0)
                
                if self.params.verbose:
                    logger.info(f"{date}: Closing {ticker}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_phase10_backtest(
    tickers: List[str] = None,
    start_date: str = TEST_START,
    end_date: str = TEST_END,
    initial_capital: float = 100000.0,
    verbose: bool = False,
) -> Dict:
    """
    Run Phase 10 backtest.
    
    Returns:
        Results dictionary with performance metrics
    """
    tickers = tickers or AGGRESSIVE_UNIVERSE
    
    logger.info("=" * 60)
    logger.info("PHASE 10: AGGRESSIVE ALPHA AMPLIFICATION")
    logger.info("=" * 60)
    logger.info("TARGET: 25-35% CAGR with <22% Max DD")
    
    start_time = time.time()
    
    # Step 1: Load data
    logger.info("\n[1/5] Loading price data...")
    price_data, sector_map = load_price_data(tickers, start_date, end_date)
    
    if len(price_data) < 10:
        logger.error("Insufficient data loaded")
        return {'error': 'Insufficient data'}
    
    # Step 2: Load VIX
    logger.info("\n[2/5] Loading VIX data...")
    vix_data = load_vix_data(start_date, end_date)
    
    # Step 3: Compute TDA features
    logger.info("\n[3/5] Computing TDA features...")
    tda_data = compute_tda_features(price_data)
    
    # Step 4: Initialize Phase 10 system
    logger.info("\n[4/5] Initializing Phase 10 system...")
    config = Phase10Config(
        max_leverage=1.5,
        min_leverage=0.5,
        kelly_fraction=0.45,
        bull_leverage=1.45,
        moderate_leverage=1.25,
        defensive_leverage=0.70,
        max_positions=15,
        max_position_weight=0.15,
        target_volatility=0.22,
        dd_start_reduction=0.06,
        dd_aggressive_reduction=0.12,
        max_acceptable_dd=0.22,
        initial_capital=initial_capital,
    )
    phase10_system = Phase10Orchestrator(config)
    
    # Step 5: Run backtest
    logger.info("\n[5/5] Running backtest with dynamic leverage...")
    
    # Setup Backtrader
    cerebro = bt.Cerebro()
    cerebro.broker.set_cash(initial_capital)
    cerebro.broker.setcommission(commission=0.001)  # 0.1% commission
    
    # Add data feeds
    for ticker, df in price_data.items():
        # Filter to test period
        df_test = df[df.index >= start_date]
        if len(df_test) < 60:
            continue
        
        data = bt.feeds.PandasData(
            dataname=df_test,
            name=ticker,
        )
        cerebro.adddata(data)
    
    # Add strategy
    cerebro.addstrategy(
        Phase10Strategy,
        phase10_system=phase10_system,
        vix_data=vix_data,
        tda_data=tda_data,
        sector_map=sector_map,
        initial_capital=initial_capital,
        verbose=verbose,
    )
    
    # Run
    results = cerebro.run()
    strategy = results[0]
    
    # Calculate metrics
    final_value = cerebro.broker.getvalue()
    total_return = (final_value / initial_capital - 1) * 100
    
    # Calculate CAGR
    equity_curve = np.array(strategy.equity_curve)
    trading_days = len(equity_curve)
    years = trading_days / 252
    cagr = ((final_value / initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0
    
    # Calculate Sharpe
    if len(equity_curve) > 1:
        returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe = (np.mean(returns) * 252) / (np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
    else:
        sharpe = 0
    
    # Calculate Max Drawdown
    rolling_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - rolling_max) / rolling_max
    max_dd = np.min(drawdowns) * 100
    
    # Average leverage
    avg_leverage = np.mean(strategy.leverage_history) if strategy.leverage_history else 1.0
    
    runtime = time.time() - start_time
    
    # Results
    results_dict = {
        'phase': 'Phase 10',
        'description': 'Aggressive Alpha Amplification with Dynamic Leverage',
        'period': f'{start_date} to {end_date}',
        'initial_capital': initial_capital,
        'final_value': final_value,
        'performance': {
            'total_return': f'{total_return:.1f}%',
            'cagr': f'{cagr:.1f}%',
            'sharpe_ratio': f'{sharpe:.2f}',
            'max_drawdown': f'{max_dd:.1f}%',
        },
        'leverage': {
            'average_leverage': f'{avg_leverage:.2f}x',
            'max_leverage': f'{max(strategy.leverage_history) if strategy.leverage_history else 1.0:.2f}x',
            'min_leverage': f'{min(strategy.leverage_history) if strategy.leverage_history else 1.0:.2f}x',
        },
        'targets': {
            'cagr_target': '25-35%',
            'sharpe_target': '>1.5',
            'max_dd_target': '<22%',
        },
        'target_met': {
            'cagr': bool(25 <= cagr <= 35),
            'sharpe': bool(sharpe > 1.5),
            'max_dd': bool(abs(max_dd) < 22),
        },
        'trades': len(strategy.trades),
        'runtime_seconds': runtime,
        'timestamp': datetime.now().isoformat(),
    }
    
    # Log results
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 10 RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total Return: {total_return:.1f}%")
    logger.info(f"CAGR: {cagr:.1f}%")
    logger.info(f"Sharpe Ratio: {sharpe:.2f}")
    logger.info(f"Max Drawdown: {max_dd:.1f}%")
    logger.info(f"Avg Leverage: {avg_leverage:.2f}x")
    logger.info(f"Trades: {len(strategy.trades)}")
    logger.info(f"Runtime: {runtime:.1f}s")
    logger.info("=" * 60)
    
    logger.info("\nTARGET ANALYSIS:")
    logger.info(f"  CAGR 25-35%: {'✅ PASS' if 25 <= cagr <= 35 else ('⚠️ HIGH' if cagr > 35 else '❌ MISS')} ({cagr:.1f}%)")
    logger.info(f"  Sharpe >= 1.5: {'✅ PASS' if sharpe >= 1.5 else '❌ MISS'} ({sharpe:.2f})")
    logger.info(f"  Max DD <= 22%: {'✅ PASS' if abs(max_dd) <= 22 else '❌ MISS'} ({abs(max_dd):.1f}%)")
    
    # Save results
    with open(RESULTS_JSON, 'w') as f:
        json.dump(results_dict, f, indent=2)
    logger.info(f"\nResults saved to {RESULTS_JSON}")
    
    # Generate report
    generate_report(results_dict, strategy)
    logger.info(f"Report saved to {REPORT_PATH}")
    
    return results_dict


def generate_report(results: Dict, strategy) -> None:
    """Generate Phase 10 markdown report."""
    
    report = f"""# Phase 10: Aggressive Alpha Amplification Report

## Executive Summary

**Phase 10** transforms Phase 9's excellent risk-adjusted returns into aggressive absolute returns
using dynamic leverage and concentrated positions.

### Performance Results

| Metric | Actual | Target | Status |
|--------|--------|--------|--------|
| **Total Return** | {results['performance']['total_return']} | - | - |
| **CAGR** | {results['performance']['cagr']} | 25-35% | {'✅' if results['target_met']['cagr'] else '❌'} |
| **Sharpe Ratio** | {results['performance']['sharpe_ratio']} | > 1.5 | {'✅' if results['target_met']['sharpe'] else '❌'} |
| **Max Drawdown** | {results['performance']['max_drawdown']} | < 22% | {'✅' if results['target_met']['max_dd'] else '❌'} |

### Leverage Statistics

| Metric | Value |
|--------|-------|
| **Average Leverage** | {results['leverage']['average_leverage']} |
| **Max Leverage** | {results['leverage']['max_leverage']} |
| **Min Leverage** | {results['leverage']['min_leverage']} |

## System Architecture

### Key Enhancements from Phase 9

1. **Dynamic Leverage Engine**
   - Kelly-optimal leverage calculation (45% fractional Kelly)
   - Regime-conditional scaling (Bull: 1.45x, Neutral: 1.0x, Bear: 0.7x)
   - Drawdown-triggered de-leveraging
   - Volatility-based adjustments

2. **Concentrated Portfolio**
   - Maximum 15 positions (vs 20+ in Phase 9)
   - Up to 15% per position (vs 12%)
   - Focus on highest-conviction signals

3. **Aggressive Parameters**
   - Target volatility: 22% (vs 15%)
   - Max acceptable drawdown: 22% (vs 15%)
   - Bull regime leverage: 1.45x

4. **VIX Integration**
   - Real-time volatility regime detection
   - Leverage reduction at VIX > 25
   - Crisis mode at VIX > 30

## Technical Configuration

- **Period**: {results['period']}
- **Initial Capital**: ${results['initial_capital']:,.0f}
- **Final Value**: ${results['final_value']:,.2f}
- **Total Trades**: {results['trades']}
- **Runtime**: {results['runtime_seconds']:.1f} seconds

## Files Created

```
src/phase10/
├── __init__.py                 # Package exports
├── dynamic_leverage.py         # Kelly + regime leverage engine
└── phase10_orchestrator.py     # Main orchestrator

scripts/
└── run_phase10.py              # Execution script
```

---
*Generated: {results['timestamp']}*
"""
    
    with open(REPORT_PATH, 'w') as f:
        f.write(report)


if __name__ == '__main__':
    run_phase10_backtest(verbose=False)
