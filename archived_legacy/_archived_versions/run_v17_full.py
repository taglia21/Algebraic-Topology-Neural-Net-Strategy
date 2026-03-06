#!/usr/bin/env python3
"""
V17.0 Full System Runner
=========================
Integrates all V17.0 components:
1. Universe Builder (1,000+ symbols)
2. Data Pipeline (vectorized fetch)
3. HMM Regime Detection (4 states)
4. Strategy Router (regime-specific strategies)
5. Factor Zoo (50+ factors)
6. Walk-Forward Backtest (realistic costs)

Realistic Targets:
- Sharpe: 1.5-3.0
- CAGR: 25-50%
- MaxDD: -15% to -25%

Red Flags:
- Sharpe >5.0 = overfit
- CAGR >100% = overfit
"""

import os
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

# V17 Components
from v17_universe_builder import UniverseBuilder
from v17_data_pipeline import DataPipeline
from v17_hmm_regime import HMMRegimeDetector, REGIME_NAMES, REGIME_STRATEGIES
from v17_strategy_router import StrategyRouter, STRATEGY_CONFIGS
from v17_factor_zoo import FactorZoo
from v17_walkforward import WalkForwardEngine, BacktestConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('V17_Full')


class V17System:
    """
    V17.0 Integrated Trading System
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        
        # Components
        self.universe_builder = UniverseBuilder()
        self.data_pipeline = DataPipeline()
        self.hmm_detector = HMMRegimeDetector()
        self.strategy_router = StrategyRouter()
        self.factor_zoo = FactorZoo()
        self.backtest_engine = None
        
        # State
        self.universe: list = []
        self.prices: pd.DataFrame = None
        self.factors: pd.DataFrame = None
        self.regime: int = 2  # Default to MeanRevert
        self.regime_history: pd.DataFrame = None
        
        # Results
        self.backtest_results: Dict = {}
        
    def _default_config(self) -> Dict:
        return {
            'initial_capital': 1_000_000,
            'train_months': 12,
            'test_months': 3,
            'roll_months': 1,
            'max_positions': 50,
            'max_position_size': 0.04,
            'vol_target': 0.15,
            'use_cache': True,
            'top_n_symbols': 200  # For faster testing
        }
    
    def run(self, refresh_data: bool = False) -> Dict[str, Any]:
        """
        Run the full V17 system.
        """
        print("\n" + "=" * 70)
        print("🚀 V17.0 INTEGRATED TRADING SYSTEM")
        print("=" * 70)
        print(f"   Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"   Initial Capital: ${self.config['initial_capital']:,.0f}")
        print(f"   Max Positions: {self.config['max_positions']}")
        print("=" * 70)
        
        # Step 1: Universe
        logger.info("\n📊 Step 1: Building Universe...")
        self._build_universe(refresh=refresh_data)
        
        # Step 2: Data
        logger.info("\n📥 Step 2: Fetching Price Data...")
        self._fetch_data(refresh=refresh_data)
        
        # Step 3: Regime Detection
        logger.info("\n🧠 Step 3: Detecting Market Regime...")
        self._detect_regime()
        
        # Step 4: Calculate Factors
        logger.info("\n🦁 Step 4: Computing Factors...")
        self._compute_factors()
        
        # Step 5: Generate Signals
        logger.info("\n📈 Step 5: Generating Trading Signals...")
        signals = self._generate_signals()
        
        # Step 6: Walk-Forward Backtest
        logger.info("\n🔄 Step 6: Running Walk-Forward Backtest...")
        self._run_backtest(signals)
        
        # Step 7: Report
        self._generate_report()
        
        return self.backtest_results
    
    def _build_universe(self, refresh: bool = False):
        """Build tradeable universe"""
        cache_file = 'cache/universe/universe_latest.json'
        
        if not refresh and os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                data = json.load(f)
                self.universe = data.get('symbols', [])
            logger.info(f"   📂 Loaded cached universe: {len(self.universe)} symbols")
        else:
            result = self.universe_builder.build_universe()
            self.universe = result['symbols']
            logger.info(f"   ✅ Built universe: {len(self.universe)} symbols")
        
        # Limit for faster testing
        if self.config.get('top_n_symbols'):
            self.universe = self.universe[:self.config['top_n_symbols']]
            logger.info(f"   📊 Using top {len(self.universe)} symbols")
    
    def _fetch_data(self, refresh: bool = False):
        """Fetch price data"""
        cache_file = 'cache/v17_prices/v17_prices_latest.parquet'
        
        if not refresh and os.path.exists(cache_file):
            self.prices = pd.read_parquet(cache_file)
            logger.info(f"   📂 Loaded cached prices: {len(self.prices)} rows")
        else:
            self.prices = self.data_pipeline.fetch_all(self.universe)
            logger.info(f"   ✅ Fetched prices: {len(self.prices)} rows")
        
        # Filter to our universe
        self.prices = self.prices[self.prices['symbol'].isin(self.universe)]
        logger.info(f"   📊 Filtered to {self.prices['symbol'].nunique()} symbols")
    
    def _detect_regime(self):
        """Detect current market regime using HMM"""
        cache_file = 'cache/v17_hmm_regime.pkl'
        
        # Load or train HMM
        if os.path.exists(cache_file):
            self.hmm_detector.load(cache_file)
            logger.info("   📂 Loaded cached HMM model")
        else:
            # Get SPY data for training
            import yfinance as yf
            spy = yf.download('SPY', period='5y', progress=False)
            if isinstance(spy.columns, pd.MultiIndex):
                spy.columns = spy.columns.get_level_values(0)
            spy.columns = [c.lower() for c in spy.columns]
            spy = spy.reset_index()
            spy.columns = [c.lower() if c != 'Date' else 'date' for c in spy.columns]
            
            self.hmm_detector.fit(spy)
            self.hmm_detector.save(cache_file)
            logger.info("   ✅ Trained and saved HMM model")
        
        # Get current regime
        # Use SPY from our price data
        spy_data = self.prices[self.prices['symbol'] == 'SPY']
        if spy_data.empty:
            # Fetch SPY separately
            import yfinance as yf
            spy_data = yf.download('SPY', period='1y', progress=False)
            if isinstance(spy_data.columns, pd.MultiIndex):
                spy_data.columns = spy_data.columns.get_level_values(0)
            spy_data.columns = [c.lower() for c in spy_data.columns]
            spy_data = spy_data.reset_index()
            spy_data.columns = [c.lower() if c != 'Date' else 'date' for c in spy_data.columns]
        
        self.regime, regime_name = self.hmm_detector.get_current_regime(spy_data)
        self.regime_history = self.hmm_detector.predict(spy_data)
        
        logger.info(f"   🎯 Current Regime: {regime_name} (state {self.regime})")
        logger.info(f"   📋 Strategy: {REGIME_STRATEGIES[self.regime]}")
        
        # Update router
        self.strategy_router.set_regime(self.regime)
    
    def _compute_factors(self):
        """Compute factors for all symbols"""
        logger.info(f"   Computing {len(self.factor_zoo.get_all_factors())} factors...")
        
        # Group by symbol and compute factors
        factor_dfs = []
        
        for symbol in self.prices['symbol'].unique():
            sym_data = self.prices[self.prices['symbol'] == symbol].copy()
            
            if len(sym_data) < 60:
                continue
            
            sym_data = sym_data.sort_values('date')
            
            try:
                sym_factors = self.factor_zoo.compute_all_factors(sym_data)
                sym_factors['symbol'] = symbol
                factor_dfs.append(sym_factors)
            except Exception as e:
                logger.warning(f"   ⚠️ Failed to compute factors for {symbol}: {e}")
        
        if factor_dfs:
            self.factors = pd.concat(factor_dfs, ignore_index=True)
            logger.info(f"   ✅ Computed factors for {len(factor_dfs)} symbols")
        else:
            self.factors = pd.DataFrame()
            logger.warning("   ⚠️ No factors computed")
    
    def _generate_signals(self) -> pd.DataFrame:
        """Generate trading signals based on regime and factors"""
        logger.info(f"   Using {REGIME_NAMES[self.regime]} strategy...")
        
        signals_list = []
        
        # Get strategy-specific factor weights
        strategy_name = REGIME_STRATEGIES[self.regime]
        
        # Define factor weights by strategy
        factor_weights = {
            'v17_momentum_xsection': {
                'momentum_12_1': 0.3,
                'momentum_6_1': 0.2,
                'risk_adjusted_momentum': 0.2,
                'relative_strength': 0.15,
                'momentum_consistency': 0.15
            },
            'v17_trend_follow': {
                'ma_cross_50_200': 0.25,
                'breakout_20d': 0.25,
                'trend_strength_adx': 0.2,
                'price_vs_ma200': 0.15,
                'channel_position': 0.15
            },
            'v17_stat_arb': {
                'zscore_20d': -0.3,  # Negative for mean reversion
                'reversal_5d': 0.25,
                'mean_reversion_speed': 0.2,
                'overbought_oversold': -0.15,
                'betti_1_estimate': 0.1  # TDA cycle detection
            },
            'v17_defensive': {
                'volatility_20d': -0.4,  # Low vol preferred
                'downside_vol': -0.3,
                'distance_from_high': 0.2,
                'amihud_illiquidity': -0.1
            }
        }
        
        weights = factor_weights.get(strategy_name, {})
        
        if not weights or self.factors.empty:
            # Fallback to simple momentum
            logger.info("   Using fallback momentum signals")
            weights = {'momentum_3m': 1.0}
        
        # Get latest factors for each symbol
        latest_factors = self.factors.groupby('symbol').last().reset_index()
        
        # Calculate composite score
        latest_factors['alpha_score'] = 0
        for factor, weight in weights.items():
            if factor in latest_factors.columns:
                # Winsorize and z-score
                values = latest_factors[factor].fillna(0)
                values = values.clip(-3, 3)
                if values.std() > 0:
                    values = (values - values.mean()) / values.std()
                latest_factors['alpha_score'] += weight * values
        
        # Rank and generate signals
        latest_factors['rank'] = latest_factors['alpha_score'].rank(pct=True)
        
        # Long top quintile, short bottom quintile
        latest_factors['signal'] = 0.0
        latest_factors.loc[latest_factors['rank'] > 0.8, 'signal'] = 1.0
        latest_factors.loc[latest_factors['rank'] < 0.2, 'signal'] = -1.0
        
        # Expand to all dates
        for _, row in latest_factors.iterrows():
            symbol = row['symbol']
            signal = row['signal']
            alpha = row['alpha_score']
            
            # Get all dates for this symbol
            sym_dates = self.prices[self.prices['symbol'] == symbol]['date'].unique()
            
            for date in sym_dates[-252:]:  # Last year of dates
                signals_list.append({
                    'date': date,
                    'symbol': symbol,
                    'signal': signal,
                    'alpha_score': alpha
                })
        
        signals = pd.DataFrame(signals_list)
        signals['date'] = pd.to_datetime(signals['date'])
        
        logger.info(f"   ✅ Generated {len(signals)} signal observations")
        logger.info(f"   📊 Long: {(signals['signal'] > 0).sum()}, Short: {(signals['signal'] < 0).sum()}")
        
        return signals
    
    def _run_backtest(self, signals: pd.DataFrame):
        """Run walk-forward backtest"""
        config = BacktestConfig(
            train_months=self.config['train_months'],
            test_months=self.config['test_months'],
            roll_months=self.config['roll_months'],
            initial_capital=self.config['initial_capital'],
            max_position_size=self.config['max_position_size'],
            vol_target=self.config['vol_target']
        )
        
        self.backtest_engine = WalkForwardEngine(config)
        self.backtest_results = self.backtest_engine.run_backtest(self.prices, signals)
    
    def _generate_report(self):
        """Generate and save comprehensive report"""
        if not self.backtest_engine:
            return
        
        self.backtest_engine.print_report()
        
        # Save results
        results_dir = Path('results/v17')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics = self.backtest_engine.final_metrics
        metrics['regime'] = int(self.regime)
        metrics['regime_name'] = REGIME_NAMES[self.regime]
        metrics['strategy'] = REGIME_STRATEGIES[self.regime]
        metrics['n_symbols'] = len(self.universe)
        metrics['n_factors'] = len(self.factor_zoo.get_all_factors())
        metrics['timestamp'] = datetime.now().isoformat()
        
        with open(results_dir / 'v17_full_results.json', 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        # Save equity curve
        if self.backtest_engine.daily_equity is not None:
            self.backtest_engine.daily_equity.to_parquet(
                results_dir / 'v17_equity_curve.parquet', 
                index=False
            )
        
        # Generate markdown report
        report = self._generate_markdown_report(metrics)
        with open(results_dir / 'V17_REPORT.md', 'w') as f:
            f.write(report)
        
        logger.info(f"\n💾 Results saved to {results_dir}")
        
        # Quality assessment
        self._assess_quality(metrics)
    
    def _generate_markdown_report(self, metrics: Dict) -> str:
        """Generate markdown report"""
        report = f"""# V17.0 Trading System Report

**Generated:** {metrics.get('timestamp', 'N/A')}

## System Configuration
- Initial Capital: ${self.config['initial_capital']:,.0f}
- Universe Size: {metrics.get('n_symbols', 0)} symbols
- Factors: {metrics.get('n_factors', 0)} factors
- Current Regime: {metrics.get('regime_name', 'Unknown')}
- Active Strategy: {metrics.get('strategy', 'Unknown')}

## Performance Summary

| Metric | Value | Target Range |
|--------|-------|--------------|
| Sharpe Ratio | {metrics.get('sharpe', 0):.2f} | 1.5 - 3.0 |
| CAGR | {metrics.get('cagr', 0):.1%} | 25% - 50% |
| Max Drawdown | {metrics.get('max_drawdown', 0):.1%} | -15% to -25% |
| Annual Vol | {metrics.get('annual_volatility', 0):.1%} | 10% - 20% |
| Sortino Ratio | {metrics.get('sortino', 0):.2f} | >2.0 |
| Calmar Ratio | {metrics.get('calmar', 0):.2f} | >1.0 |

## Trading Statistics
- Total Return: {metrics.get('total_return', 0):.1%}
- Trading Days: {metrics.get('trading_days', 0)}
- Total Trades: {metrics.get('total_trades', 0)}
- Win Rate: {metrics.get('win_rate', 0):.1%}
- Annual Turnover: {metrics.get('annualized_turnover', 0):.0%}

## Transaction Costs
- Total Commission: ${metrics.get('total_commission', 0):,.0f}
- Total Slippage: ${metrics.get('total_slippage', 0):,.0f}
- Cost Drag: {metrics.get('cost_drag', 0):.2%}

## Quality Assessment

{"✅ **REALISTIC**: Metrics within expected bounds" if metrics.get('is_realistic', False) else "⚠️ **WARNING**: Metrics may indicate overfitting"}

### Red Flags Checked:
- Sharpe > 5.0: {"❌ FLAGGED" if metrics.get('sharpe', 0) >= 5.0 else "✅ OK"}
- CAGR > 100%: {"❌ FLAGGED" if metrics.get('cagr', 0) >= 1.0 else "✅ OK"}
- Max DD < -50%: {"❌ FLAGGED" if metrics.get('max_drawdown', 0) < -0.50 else "✅ OK"}

## Notes
- Walk-forward validation with {self.config['train_months']}-month train, {self.config['test_months']}-month test
- Transaction costs: 5bps commission + 5-20bps slippage
- Position limits: {self.config['max_position_size']:.0%} max per position
- Vol target: {self.config['vol_target']:.0%} annual
"""
        return report
    
    def _assess_quality(self, metrics: Dict):
        """Assess result quality"""
        print("\n" + "=" * 50)
        print("🔍 QUALITY ASSESSMENT")
        print("=" * 50)
        
        sharpe = metrics.get('sharpe', 0)
        cagr = metrics.get('cagr', 0)
        max_dd = metrics.get('max_drawdown', 0)
        
        issues = []
        
        # Check for overfitting
        if sharpe >= 5.0:
            issues.append(f"❌ Sharpe {sharpe:.2f} too high - likely overfit")
        elif sharpe >= 3.0:
            print(f"⚠️ Sharpe {sharpe:.2f} - verify with out-of-sample")
        elif sharpe >= 1.5:
            print(f"✅ Sharpe {sharpe:.2f} - realistic")
        else:
            issues.append(f"❌ Sharpe {sharpe:.2f} too low")
        
        if cagr >= 1.0:
            issues.append(f"❌ CAGR {cagr:.1%} too high - likely overfit")
        elif cagr >= 0.50:
            print(f"⚠️ CAGR {cagr:.1%} - verify with out-of-sample")
        elif cagr >= 0.25:
            print(f"✅ CAGR {cagr:.1%} - realistic")
        else:
            print(f"⚠️ CAGR {cagr:.1%} - below target")
        
        if max_dd < -0.50:
            issues.append(f"❌ MaxDD {max_dd:.1%} too severe")
        elif max_dd < -0.25:
            print(f"⚠️ MaxDD {max_dd:.1%} - within tolerance")
        else:
            print(f"✅ MaxDD {max_dd:.1%} - acceptable")
        
        if issues:
            print("\n⚠️ ISSUES DETECTED:")
            for issue in issues:
                print(f"   {issue}")
        else:
            print("\n✅ All quality checks passed")


def main():
    """Run V17 full system"""
    print("\n" + "=" * 70)
    print("🎯 V17.0 FULL SYSTEM EXECUTION")
    print("=" * 70)
    
    # Configuration
    config = {
        'initial_capital': 1_000_000,
        'train_months': 9,    # 9-month training
        'test_months': 3,     # 3-month testing
        'roll_months': 1,
        'max_positions': 40,
        'max_position_size': 0.04,
        'vol_target': 0.15,
        'use_cache': True,
        'top_n_symbols': 100  # Use 100 symbols for faster execution
    }
    
    # Initialize and run
    system = V17System(config)
    results = system.run(refresh_data=False)
    
    print("\n" + "=" * 70)
    print("✅ V17.0 EXECUTION COMPLETE")
    print("=" * 70)
    
    return system


if __name__ == "__main__":
    main()
