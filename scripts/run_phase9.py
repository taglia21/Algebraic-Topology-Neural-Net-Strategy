#!/usr/bin/env python3
"""Phase 9: Advanced Alpha Generation & Regime-Adaptive Optimization.

Main execution script for the Phase 9 trading system.

MISSION CRITICAL OBJECTIVE:
Transform from GOOD (6-20% CAGR) to EXCEPTIONAL (30-50%+ CAGR)
- Sharpe Ratio target: > 2.0
- Max Drawdown target: < 15%

This script:
1. Loads price data for multi-asset universe
2. Computes TDA features
3. Runs hierarchical regime detection
4. Generates advanced alpha signals
5. Executes regime-adaptive backtest
6. Reports comprehensive performance metrics
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

# Phase 9 imports
from src.phase9 import (
    Phase9Orchestrator, HierarchicalRegimeStrategy, AdvancedAlphaEngine,
    AdaptiveUniverseScreener, DynamicPositionOptimizer
)
from src.phase9.phase9_orchestrator import Phase9Config
from src.phase9.regime_meta_strategy import MacroRegime, RegimeMeta
from src.phase9.dynamic_optimizer import PortfolioState

# Existing imports
from src.tda_features import TDAFeatureGenerator
from src.ensemble_strategy import EnsembleStrategy

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('Phase9')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Universe configuration
CORE_TICKERS = ['SPY', 'QQQ', 'IWM', 'DIA', 'XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLY']

EXPANDED_UNIVERSE = [
    # Core ETFs
    'SPY', 'QQQ', 'IWM', 'DIA',
    # Sector ETFs
    'XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLC',
    # Large-cap growth
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
    # Large-cap value
    'BRK-B', 'JPM', 'JNJ', 'UNH', 'V', 'PG', 'HD', 'MA',
    # Momentum plays
    'AMD', 'AVGO', 'CRM', 'ADBE', 'NFLX', 'COST', 'LLY', 'MRK',
    # Diversifiers
    'GLD', 'TLT', 'VNQ',
]

# Date configuration
TRAIN_START = "2020-01-01"
TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END = "2025-12-31"

# Output paths
RESULTS_DIR = '/workspaces/Algebraic-Topology-Neural-Net-Strategy/results'
RESULTS_JSON = f'{RESULTS_DIR}/phase9_results.json'
REPORT_PATH = f'{RESULTS_DIR}/PHASE9_REPORT.md'

# =============================================================================
# DATA LOADING
# =============================================================================

def load_price_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    """
    Load price data for all tickers.
    
    Returns:
        (price_data, sector_map)
    """
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
        'XLP': 'ConsumerStaples', 'XLU': 'Utilities', 'XLB': 'Materials',
        'XLRE': 'RealEstate', 'XLC': 'Communication',
        'GLD': 'Commodity', 'TLT': 'Bond', 'VNQ': 'RealEstate',
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
                    sector_map[ticker] = info.get('sector', 'Other')
                except:
                    sector_map[ticker] = 'Other'
                    
        except Exception as e:
            logger.warning(f"Failed to load {ticker}: {e}")
            failed.append(ticker)
    
    logger.info(f"Loaded {len(price_data)}/{len(tickers)} tickers. Failed: {failed}")
    
    return price_data, sector_map


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
# PHASE 9 BACKTEST STRATEGY
# =============================================================================

class Phase9Strategy(bt.Strategy):
    """Backtrader strategy integrating Phase 9 components."""
    
    params = (
        ('phase9_system', None),
        ('spy_data', None),
        ('vix_data', None),
        ('tda_data', None),
        ('sector_map', None),
        ('initial_capital', 100000.0),
        ('rebalance_frequency', 3),  # Rebalance every 3 days
        ('max_position', 0.12),  # Allow 12% max per position
        ('verbose', False),
    )
    
    def __init__(self):
        self.order = None
        self.bar_count = 0
        self.trades = []
        self.equity_curve = []
        
        # Phase 9 system
        self.p9 = self.params.phase9_system
        
        # Track data feeds by ticker
        self.ticker_map = {}
        for i, data in enumerate(self.datas):
            ticker = data._name
            self.ticker_map[ticker] = i
        
        # State
        self.current_regime = None
        self.target_weights = {}
        
        logger.info(f"Phase9Strategy initialized with {len(self.datas)} data feeds")
    
    def next(self):
        self.bar_count += 1
        
        # Get current date
        current_date = self.datas[0].datetime.date(0).isoformat()
        
        # Track equity
        portfolio_value = self.broker.getvalue()
        self.equity_curve.append(portfolio_value)
        
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
                volumes[ticker] = np.ones_like(close_prices) * 1e6  # Placeholder
        
        if len(prices) < 10:
            return
        
        # Get SPY data for regime
        spy_prices = pd.DataFrame({'close': prices.get('SPY', np.array([100.0]))})
        
        # Build portfolio state
        portfolio_state = self._get_portfolio_state()
        
        # Run Phase 9 analysis
        state = self.p9.process_day(
            date=current_date,
            spy_prices=spy_prices,
            universe_prices=prices,
            universe_volumes=volumes,
            sector_map=self.params.sector_map or {},
            vix_data=self.params.vix_data,
            tda_data=self.params.tda_data,
            current_portfolio=portfolio_state,
        )
        
        self.current_regime = state.regime_meta
        
        # Check if trading allowed
        if state.regime_meta and not state.regime_meta.trade_allowed:
            if self.params.verbose:
                logger.info(f"{current_date}: Trading blocked by regime")
            return
        
        # Get target weights from signals
        self.target_weights = self._compute_target_weights(state.signals)
        
        # Execute rebalance
        self._rebalance(current_date)
    
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
            portfolio_volatility=0.15,
        )
    
    def _compute_target_weights(self, signals: Dict[str, float]) -> Dict[str, float]:
        """Compute target weights from signals."""
        if not signals:
            return {}
        
        # Filter positive signals (balanced threshold)
        positive = {k: v for k, v in signals.items() if v > 0.02}
        
        if not positive:
            return {}
        
        # Sort by signal strength and take top 20 positions for concentration
        sorted_signals = sorted(positive.items(), key=lambda x: x[1], reverse=True)[:20]
        positive = dict(sorted_signals)
        
        # Normalize weights  
        total = sum(positive.values())
        weights = {k: (v / total) * 0.98 for k, v in positive.items()}  # 98% invested max
        
        # Apply drawdown scaling - balanced for upside capture while controlling DD
        if self.equity_curve:
            peak = max(self.equity_curve)
            current = self.equity_curve[-1]
            drawdown = (peak - current) / peak if peak > 0 else 0
            
            # Scale down positions when in drawdown
            if drawdown > 0.045:  # Start scaling at 4.5% DD
                # At 4.5% DD: 0.94x, at 9% DD: 0.5x, at 11% DD: 0.3x
                dd_scale = max(0.18, 1.0 - (drawdown - 0.045) * 8.5)
                weights = {k: v * dd_scale for k, v in weights.items()}
        
        # Apply position limits
        return {k: min(v, self.params.max_position) for k, v in weights.items()}
    
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
                })
                
                if self.params.verbose:
                    logger.info(f"{date}: {ticker} -> {target_weight:.1%} (diff: {weight_diff:+.1%})")
        
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

def run_phase9_backtest(
    tickers: List[str] = None,
    start_date: str = TEST_START,
    end_date: str = TEST_END,
    initial_capital: float = 100000.0,
    verbose: bool = False,
) -> Dict:
    """
    Run Phase 9 backtest.
    
    Returns:
        Results dictionary with performance metrics
    """
    tickers = tickers or EXPANDED_UNIVERSE
    
    logger.info("=" * 60)
    logger.info("PHASE 9: ADVANCED ALPHA GENERATION")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    # Step 1: Load data
    logger.info("\n[1/4] Loading price data...")
    price_data, sector_map = load_price_data(tickers, start_date, end_date)
    
    if len(price_data) < 10:
        logger.error("Insufficient data loaded")
        return {'error': 'Insufficient data'}
    
    # Step 2: Compute TDA features
    logger.info("\n[2/4] Computing TDA features...")
    tda_data = compute_tda_features(price_data)
    
    # Step 3: Initialize Phase 9 system
    logger.info("\n[3/4] Initializing Phase 9 system...")
    config = Phase9Config(
        target_universe_size=30,
        target_volatility=0.15,
        max_position=0.10,
        initial_capital=initial_capital,
    )
    phase9_system = Phase9Orchestrator(config)
    
    # Step 4: Run backtest
    logger.info("\n[4/4] Running backtest...")
    
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
            datetime=None,
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
            openinterest=-1,
        )
        cerebro.adddata(data)
    
    # Add strategy
    cerebro.addstrategy(
        Phase9Strategy,
        phase9_system=phase9_system,
        tda_data=tda_data,
        sector_map=sector_map,
        initial_capital=initial_capital,
        verbose=verbose,
    )
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    # Run
    logger.info(f"Starting backtest: {start_date} to {end_date}")
    results = cerebro.run()
    strategy = results[0]
    
    elapsed = time.time() - start_time
    
    # Extract results
    final_value = cerebro.broker.getvalue()
    total_return = (final_value / initial_capital - 1)
    
    # Analyzers
    sharpe_analysis = strategy.analyzers.sharpe.get_analysis()
    dd_analysis = strategy.analyzers.drawdown.get_analysis()
    returns_analysis = strategy.analyzers.returns.get_analysis()
    trade_analysis = strategy.analyzers.trades.get_analysis()
    
    # Calculate CAGR
    equity_curve = strategy.equity_curve
    n_days = len(equity_curve)
    n_years = n_days / 252 if n_days > 0 else 1
    cagr = (final_value / initial_capital) ** (1 / n_years) - 1 if n_years > 0 else 0
    
    # Sharpe
    sharpe = sharpe_analysis.get('sharperatio', 0) or 0
    
    # Drawdown
    max_dd = dd_analysis.get('max', {}).get('drawdown', 0) / 100
    
    # Compile results
    results_dict = {
        'phase': 'Phase 9',
        'description': 'Advanced Alpha Generation & Regime-Adaptive Optimization',
        'period': f'{start_date} to {end_date}',
        'initial_capital': initial_capital,
        'final_value': final_value,
        'performance': {
            'total_return': f'{total_return:.1%}',
            'cagr': f'{cagr:.1%}',
            'sharpe_ratio': f'{sharpe:.2f}',
            'max_drawdown': f'{-max_dd:.1%}',
        },
        'targets': {
            'cagr_target': '30-50%',
            'sharpe_target': '>2.0',
            'max_dd_target': '<15%',
        },
        'target_met': {
            'cagr': cagr >= 0.30,
            'sharpe': sharpe >= 2.0,
            'max_dd': max_dd <= 0.15,
        },
        'trades': len(strategy.trades),
        'runtime_seconds': elapsed,
        'timestamp': datetime.now().isoformat(),
    }
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 9 RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total Return: {total_return:.1%}")
    logger.info(f"CAGR: {cagr:.1%}")
    logger.info(f"Sharpe Ratio: {sharpe:.2f}")
    logger.info(f"Max Drawdown: {-max_dd:.1%}")
    logger.info(f"Trades: {len(strategy.trades)}")
    logger.info(f"Runtime: {elapsed:.1f}s")
    logger.info("=" * 60)
    
    # Target check
    logger.info("\nTARGET ANALYSIS:")
    logger.info(f"  CAGR >= 30%: {'✅ PASS' if cagr >= 0.30 else '❌ MISS'} ({cagr:.1%})")
    logger.info(f"  Sharpe >= 2.0: {'✅ PASS' if sharpe >= 2.0 else '❌ MISS'} ({sharpe:.2f})")
    logger.info(f"  Max DD <= 15%: {'✅ PASS' if max_dd <= 0.15 else '❌ MISS'} ({max_dd:.1%})")
    
    return results_dict


def generate_report(results: Dict) -> str:
    """Generate Markdown report."""
    report = f"""# Phase 9: Advanced Alpha Generation Report

## Executive Summary

**Phase 9** implements a comprehensive 5-pillar transformation targeting institutional-grade alpha:
- **CAGR Target**: 30-50%
- **Sharpe Target**: > 2.0
- **Max Drawdown Target**: < 15%

## Performance Results

| Metric | Actual | Target | Status |
|--------|--------|--------|--------|
| **Total Return** | {results['performance']['total_return']} | - | - |
| **CAGR** | {results['performance']['cagr']} | 30-50% | {'✅' if results['target_met']['cagr'] else '❌'} |
| **Sharpe Ratio** | {results['performance']['sharpe_ratio']} | > 2.0 | {'✅' if results['target_met']['sharpe'] else '❌'} |
| **Max Drawdown** | {results['performance']['max_drawdown']} | < 15% | {'✅' if results['target_met']['max_dd'] else '❌'} |

## System Architecture

### 5-Pillar Transformation

1. **Hierarchical Regime Meta-Strategy**
   - Layer 1: HMM-based macro regime detection (Bull/Bear/HighVol/LowVol)
   - Layer 2: TDA regime correlation (topology analysis)
   - Layer 3: Dynamic factor allocation

2. **Advanced Alpha Engine**
   - Multi-horizon momentum (1w/1m/3m/6m/12m)
   - Mean reversion capture (RSI/Bollinger)
   - TDA-enhanced signals
   - Cross-sectional momentum

3. **Adaptive Universe Screener**
   - Quality-based filtering
   - Momentum ranking
   - Sector diversification

4. **Dynamic Position Optimizer**
   - Regime-adaptive Kelly criterion
   - Risk-parity allocation
   - Drawdown scaling
   - Correlation adjustment

5. **Integrated Risk Management**
   - ATR-based stops
   - Portfolio heat limits
   - Regime-based leverage

## Technical Details

- **Period**: {results['period']}
- **Initial Capital**: ${results['initial_capital']:,.0f}
- **Final Value**: ${results['final_value']:,.0f}
- **Total Trades**: {results['trades']}
- **Runtime**: {results['runtime_seconds']:.1f} seconds

## Files Created

```
src/phase9/
├── __init__.py                 # Package exports
├── regime_meta_strategy.py     # Hierarchical regime detection
├── alpha_engine.py             # Advanced alpha generation
├── adaptive_screener.py        # Universe screening
├── dynamic_optimizer.py        # Position optimization
└── phase9_orchestrator.py      # Main orchestrator

scripts/
└── run_phase9.py               # Execution script
```

---
*Generated: {results['timestamp']}*
"""
    return report


if __name__ == '__main__':
    # Run backtest
    results = run_phase9_backtest(
        tickers=EXPANDED_UNIVERSE,
        start_date=TEST_START,
        end_date=TEST_END,
        initial_capital=100000.0,
        verbose=False,
    )
    
    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    with open(RESULTS_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {RESULTS_JSON}")
    
    # Generate report
    report = generate_report(results)
    with open(REPORT_PATH, 'w') as f:
        f.write(report)
    logger.info(f"Report saved to {REPORT_PATH}")
