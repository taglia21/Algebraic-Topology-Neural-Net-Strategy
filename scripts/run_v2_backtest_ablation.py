#!/usr/bin/env python3
"""
V2 Backtest Runner with Ablation Study

Runs comprehensive backtests comparing:
- V1.3 baseline (simple momentum + regime detection)
- V2.0 full (all 7 enhancements enabled)
- 7 ablation variants (V2.0 with each enhancement disabled)

Usage:
    python scripts/run_v2_backtest_ablation.py

Output:
    results/backtest_results_{timestamp}.json - All backtest metrics
    Console progress with tqdm
"""

import os
import sys
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try tqdm, fallback to simple progress
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None, total=None):
        if desc:
            print(f"  {desc}...")
        for item in iterable:
            yield item


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 100_000.0
    transaction_cost_bps: float = 5.0  # 5 basis points
    rebalance_frequency: str = 'monthly'  # 'daily', 'weekly', 'monthly'
    max_position_pct: float = 0.15  # 15% max per asset
    max_cash_pct: float = 0.50  # 50% max cash in risk_off
    risk_free_rate: float = 0.04  # 4% annual
    
    train_start: str = '2022-01-01'
    train_end: str = '2023-12-31'
    test_start: str = '2024-01-01'
    test_end: str = '2025-01-20'
    
    tickers: List[str] = field(default_factory=lambda: ['SPY', 'QQQ', 'IWM', 'XLK', 'XLF'])


@dataclass  
class BacktestMetrics:
    """Metrics from a single backtest run."""
    name: str
    sharpe_ratio: float
    cagr: float
    max_drawdown: float
    total_return: float
    win_rate: float
    calmar_ratio: float
    n_trades: int
    avg_holding_days: float
    volatility: float
    
    # Regime-specific metrics
    bull_return: float = 0.0
    bear_return: float = 0.0
    neutral_return: float = 0.0
    
    # Timing metrics
    processing_time_s: float = 0.0


class VectorizedBacktester:
    """
    Fast vectorized backtester for portfolio strategies.
    
    Uses numpy operations instead of event-driven loops for speed.
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.transaction_cost = config.transaction_cost_bps / 10000
        
    def load_price_data(self, data_path: str = 'data/mock_backtest_prices.parquet') -> Dict[str, pd.DataFrame]:
        """Load price data from parquet file."""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Price data not found: {data_path}. Run generate_mock_backtest_data.py first.")
        
        combined = pd.read_parquet(data_path)
        
        # Convert MultiIndex to dict of DataFrames
        price_data = {}
        for ticker in combined.index.get_level_values('Ticker').unique():
            df = combined.loc[ticker].copy()
            price_data[ticker] = df
        
        return price_data
    
    def get_rebalance_dates(self, dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Get rebalance dates based on frequency."""
        if self.config.rebalance_frequency == 'daily':
            return dates
        elif self.config.rebalance_frequency == 'weekly':
            # First trading day of each week
            return dates[dates.dayofweek == dates.dayofweek.min()]
        elif self.config.rebalance_frequency == 'monthly':
            # First trading day of each month
            monthly = dates.to_period('M').unique()
            rebalance = []
            for period in monthly:
                month_dates = dates[(dates.year == period.year) & (dates.month == period.month)]
                if len(month_dates) > 0:
                    rebalance.append(month_dates[0])
            return pd.DatetimeIndex(rebalance)
        else:
            raise ValueError(f"Unknown frequency: {self.config.rebalance_frequency}")
    
    def compute_signals_v13(self, price_data: Dict[str, pd.DataFrame], date: pd.Timestamp) -> Dict[str, float]:
        """
        V1.3 baseline signal generation.
        
        Simple momentum + volatility regime detection.
        """
        signals = {}
        
        # Get SPY for regime detection
        spy_df = price_data.get('SPY')
        if spy_df is None or date not in spy_df.index:
            return {ticker: 0.0 for ticker in self.config.tickers}
        
        # Regime: based on SPY 50-day momentum and volatility
        spy_hist = spy_df.loc[:date]
        if len(spy_hist) < 50:
            regime = 'neutral'
        else:
            spy_close = spy_hist['Close'].values
            momentum_50 = spy_close[-1] / spy_close[-50] - 1
            vol_20 = np.std(np.diff(np.log(spy_close[-20:]))) * np.sqrt(252)
            
            if momentum_50 > 0.05 and vol_20 < 0.25:
                regime = 'bull'
            elif momentum_50 < -0.05 or vol_20 > 0.35:
                regime = 'bear'
            else:
                regime = 'neutral'
        
        # Position sizing based on regime
        if regime == 'bear':
            base_allocation = 0.10  # 10% per asset in bear
        elif regime == 'bull':
            base_allocation = 0.18  # 18% per asset in bull
        else:
            base_allocation = 0.15  # 15% per asset in neutral
        
        # Momentum-based allocation per asset
        for ticker in self.config.tickers:
            df = price_data.get(ticker)
            if df is None or date not in df.index:
                signals[ticker] = 0.0
                continue
            
            hist = df.loc[:date]
            if len(hist) < 20:
                signals[ticker] = 0.0
                continue
            
            close = hist['Close'].values
            
            # 20-day momentum
            mom_20 = close[-1] / close[-20] - 1
            
            # Adjust allocation by momentum
            if mom_20 > 0.02:
                signals[ticker] = min(base_allocation * 1.2, self.config.max_position_pct)
            elif mom_20 < -0.02:
                signals[ticker] = max(base_allocation * 0.5, 0.02)
            else:
                signals[ticker] = base_allocation
        
        return signals
    
    def compute_signals_v2(self, 
                           price_data: Dict[str, pd.DataFrame], 
                           date: pd.Timestamp,
                           config_overrides: Dict[str, bool] = None) -> Dict[str, float]:
        """
        V2.0 signal generation with all enhancements.
        
        Args:
            price_data: OHLCV data
            date: Current date
            config_overrides: Override component flags for ablation study
        """
        # Default: all components enabled
        use_transformer = True
        use_sac = True
        use_persistent_laplacian = True
        use_ensemble_regime = True
        use_order_flow = True
        use_enhanced_momentum = True
        use_risk_parity = True
        
        if config_overrides:
            use_transformer = config_overrides.get('use_transformer', True)
            use_sac = config_overrides.get('use_sac', True)
            use_persistent_laplacian = config_overrides.get('use_persistent_laplacian', True)
            use_ensemble_regime = config_overrides.get('use_ensemble_regime', True)
            use_order_flow = config_overrides.get('use_order_flow', True)
            use_enhanced_momentum = config_overrides.get('use_enhanced_momentum', True)
            use_risk_parity = config_overrides.get('use_risk_parity', True)
        
        signals = {}
        
        # === REGIME DETECTION ===
        spy_df = price_data.get('SPY')
        if spy_df is None or date not in spy_df.index:
            return {ticker: 0.0 for ticker in self.config.tickers}
        
        spy_hist = spy_df.loc[:date]
        if len(spy_hist) < 50:
            regime = 'neutral'
            regime_confidence = 0.5
        else:
            spy_close = spy_hist['Close'].values
            
            if use_ensemble_regime:
                # Ensemble regime detection (simplified without hmmlearn)
                # Use multiple momentum + volatility signals
                mom_20 = spy_close[-1] / spy_close[-20] - 1
                mom_50 = spy_close[-1] / spy_close[-50] - 1
                vol_20 = np.std(np.diff(np.log(spy_close[-20:]))) * np.sqrt(252)
                
                # "GMM-like" clustering based on momentum
                bull_score = (mom_20 > 0.02) + (mom_50 > 0.05) + (vol_20 < 0.20)
                bear_score = (mom_20 < -0.02) + (mom_50 < -0.05) + (vol_20 > 0.30)
                
                if bull_score >= 2:
                    regime = 'bull'
                    regime_confidence = 0.7 + 0.1 * bull_score
                elif bear_score >= 2:
                    regime = 'bear'
                    regime_confidence = 0.7 + 0.1 * bear_score
                else:
                    regime = 'neutral'
                    regime_confidence = 0.5
            else:
                # Simple regime (V1.3 style)
                mom_50 = spy_close[-1] / spy_close[-50] - 1
                vol_20 = np.std(np.diff(np.log(spy_close[-20:]))) * np.sqrt(252)
                
                if mom_50 > 0.05 and vol_20 < 0.25:
                    regime = 'bull'
                elif mom_50 < -0.05 or vol_20 > 0.35:
                    regime = 'bear'
                else:
                    regime = 'neutral'
                regime_confidence = 0.5
        
        # Base allocation by regime
        if regime == 'bear':
            base_allocation = 0.08
        elif regime == 'bull':
            base_allocation = 0.20
        else:
            base_allocation = 0.15
        
        # === PER-ASSET SIGNALS ===
        asset_scores = {}
        asset_volatilities = {}
        
        for ticker in self.config.tickers:
            df = price_data.get(ticker)
            if df is None or date not in df.index:
                asset_scores[ticker] = 0.0
                asset_volatilities[ticker] = 0.20
                continue
            
            hist = df.loc[:date]
            if len(hist) < 50:
                asset_scores[ticker] = 0.0
                asset_volatilities[ticker] = 0.20
                continue
            
            close = hist['Close'].values
            high = hist['High'].values
            low = hist['Low'].values
            volume = hist['Volume'].values
            
            # === TRANSFORMER PREDICTION (simplified) ===
            if use_transformer:
                # Multi-scale momentum (transformer-like feature extraction)
                mom_5 = close[-1] / close[-5] - 1
                mom_10 = close[-1] / close[-10] - 1
                mom_20 = close[-1] / close[-20] - 1
                mom_50 = close[-1] / close[-50] - 1
                
                # RSI
                returns = np.diff(close[-15:])
                gains = np.sum(np.maximum(returns, 0))
                losses = np.sum(np.maximum(-returns, 0))
                rsi = 100 * gains / (gains + losses + 1e-10)
                
                # Weighted prediction
                transformer_score = 0.3 * np.tanh(mom_5 * 20) + \
                                    0.3 * np.tanh(mom_20 * 10) + \
                                    0.2 * np.tanh(mom_50 * 5) + \
                                    0.2 * (rsi - 50) / 50
            else:
                transformer_score = 0.0
            
            # === PERSISTENT LAPLACIAN TDA ===
            if use_persistent_laplacian:
                # Simplified TDA: volatility structure
                vol_short = np.std(np.diff(np.log(close[-10:]))) * np.sqrt(252)
                vol_long = np.std(np.diff(np.log(close[-50:]))) * np.sqrt(252)
                
                # "Topological" score: vol compression is bullish
                if vol_short < vol_long * 0.8:
                    tda_score = 0.5  # Compression = breakout potential
                elif vol_short > vol_long * 1.2:
                    tda_score = -0.3  # Expansion = risk
                else:
                    tda_score = 0.0
            else:
                tda_score = 0.0
            
            # === ORDER FLOW ===
            if use_order_flow:
                # Simplified order flow: volume trend
                vol_ma_5 = np.mean(volume[-5:])
                vol_ma_20 = np.mean(volume[-20:])
                
                if vol_ma_5 > vol_ma_20 * 1.5 and close[-1] > close[-5]:
                    orderflow_score = 0.3  # High volume + up = buying pressure
                elif vol_ma_5 > vol_ma_20 * 1.5 and close[-1] < close[-5]:
                    orderflow_score = -0.3  # High volume + down = selling pressure
                else:
                    orderflow_score = 0.0
            else:
                orderflow_score = 0.0
            
            # === ENHANCED MOMENTUM ===
            if use_enhanced_momentum:
                # Mean reversion component for overbought/oversold
                returns_20 = np.log(close[-1] / close[-20])
                z_score = returns_20 / (np.std(np.diff(np.log(close[-60:]))) * np.sqrt(20) + 1e-10)
                
                if z_score > 2.0:
                    enhanced_mom_score = -0.2  # Overbought
                elif z_score < -2.0:
                    enhanced_mom_score = 0.2  # Oversold (contrarian)
                else:
                    enhanced_mom_score = 0.0
            else:
                enhanced_mom_score = 0.0
            
            # Combine scores
            total_score = transformer_score + tda_score + orderflow_score + enhanced_mom_score
            asset_scores[ticker] = total_score
            
            # Volatility for risk parity
            asset_volatilities[ticker] = np.std(np.diff(np.log(close[-20:]))) * np.sqrt(252)
        
        # === SAC POSITION SIZING ===
        for ticker in self.config.tickers:
            score = asset_scores[ticker]
            vol = asset_volatilities[ticker]
            
            # Base allocation
            alloc = base_allocation
            
            # SAC-style: adjust by score
            if use_sac:
                # Continuous action: multiply by score-derived factor
                sac_multiplier = 1.0 + 0.5 * np.tanh(score)  # Range: [0.5, 1.5]
                alloc *= sac_multiplier
            
            # Risk parity: inverse volatility weighting
            if use_risk_parity:
                target_vol = 0.15
                vol_scalar = target_vol / (vol + 0.01)
                alloc *= min(vol_scalar, 2.0)
            
            # Clip to limits
            signals[ticker] = np.clip(alloc, 0.0, self.config.max_position_pct)
        
        # Normalize if total > 1
        total_alloc = sum(signals.values())
        if total_alloc > 1.0:
            for ticker in signals:
                signals[ticker] /= total_alloc
        
        return signals
    
    def run_backtest(self, 
                     price_data: Dict[str, pd.DataFrame],
                     signal_fn,
                     name: str,
                     test_start: str = None,
                     test_end: str = None) -> BacktestMetrics:
        """
        Run vectorized backtest.
        
        Args:
            price_data: Dict of ticker -> DataFrame
            signal_fn: Function(price_data, date) -> Dict[ticker, allocation]
            name: Name for this backtest run
            test_start: Start date for test period
            test_end: End date for test period
        
        Returns:
            BacktestMetrics with all calculated metrics
        """
        start_time = time.time()
        
        test_start = pd.Timestamp(test_start or self.config.test_start)
        test_end = pd.Timestamp(test_end or self.config.test_end)
        
        # Get common date index
        ref_ticker = self.config.tickers[0]
        dates = price_data[ref_ticker].loc[test_start:test_end].index
        rebalance_dates = self.get_rebalance_dates(dates)
        
        logger.debug(f"Running {name}: {len(dates)} days, {len(rebalance_dates)} rebalances")
        
        # Initialize
        cash = self.config.initial_capital
        positions = {ticker: 0.0 for ticker in self.config.tickers}  # Number of shares
        portfolio_values = []
        trade_log = []
        
        # Get price matrix
        price_matrix = np.zeros((len(dates), len(self.config.tickers)))
        for i, ticker in enumerate(self.config.tickers):
            price_matrix[:, i] = price_data[ticker].loc[dates, 'Close'].values
        
        for t, date in enumerate(dates):
            current_prices = price_matrix[t]
            
            # Calculate portfolio value = cash + sum(shares * price)
            position_value = sum(positions[ticker] * current_prices[i] 
                                 for i, ticker in enumerate(self.config.tickers))
            portfolio_value = cash + position_value
            portfolio_values.append(portfolio_value)
            
            # Rebalance?
            if date in rebalance_dates:
                # Get target allocations (as fraction of portfolio)
                target_allocs = signal_fn(price_data, date)
                
                # Calculate and execute trades
                for i, ticker in enumerate(self.config.tickers):
                    price = current_prices[i]
                    if price <= 0:
                        continue
                    
                    target_value = portfolio_value * target_allocs.get(ticker, 0.0)
                    current_value = positions[ticker] * price
                    trade_value = target_value - current_value
                    
                    if abs(trade_value) > 100:  # Min trade size
                        # Calculate shares to trade
                        shares_to_trade = trade_value / price
                        
                        # Apply transaction costs
                        cost = abs(trade_value) * self.transaction_cost
                        cash -= cost
                        
                        # Update positions
                        positions[ticker] += shares_to_trade
                        cash -= trade_value  # Buy = decrease cash, Sell = increase cash
                        
                        trade_log.append({
                            'date': date,
                            'ticker': ticker,
                            'trade_value': trade_value,
                            'shares': shares_to_trade,
                            'cost': cost,
                        })
        
        # Calculate metrics
        portfolio_values = np.array(portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        returns = np.nan_to_num(returns, 0)
        
        # Sharpe
        excess_returns = returns - self.config.risk_free_rate / 252
        sharpe = np.mean(excess_returns) / (np.std(excess_returns) + 1e-10) * np.sqrt(252)
        
        # CAGR
        n_years = len(dates) / 252
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        # Max Drawdown
        cummax = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - cummax) / cummax
        max_dd = np.min(drawdowns)
        
        # Win rate
        positive_days = np.sum(returns > 0)
        win_rate = positive_days / len(returns) if len(returns) > 0 else 0
        
        # Calmar ratio
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        
        # Volatility
        volatility = np.std(returns) * np.sqrt(252)
        
        # Processing time
        processing_time = time.time() - start_time
        
        return BacktestMetrics(
            name=name,
            sharpe_ratio=float(sharpe),
            cagr=float(cagr),
            max_drawdown=float(max_dd),
            total_return=float(total_return),
            win_rate=float(win_rate),
            calmar_ratio=float(calmar),
            n_trades=len(trade_log),
            avg_holding_days=252 / max(len(rebalance_dates), 1),
            volatility=float(volatility),
            processing_time_s=processing_time,
        )


class V2BacktestRunner:
    """
    Runs V1.3 baseline, V2.0 full, and ablation study backtests.
    """
    
    ABLATION_CONFIGS = {
        'V2_no_transformer': {'use_transformer': False},
        'V2_no_sac': {'use_sac': False},
        'V2_no_tda': {'use_persistent_laplacian': False},
        'V2_no_ensemble': {'use_ensemble_regime': False},
        'V2_no_orderflow': {'use_order_flow': False},
        'V2_no_enhanced_mom': {'use_enhanced_momentum': False},
        'V2_no_risk_parity': {'use_risk_parity': False},
    }
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.backtester = VectorizedBacktester(self.config)
        self.results = {}
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load price data."""
        return self.backtester.load_price_data()
    
    def run_baseline(self, price_data: Dict[str, pd.DataFrame]) -> BacktestMetrics:
        """Run V1.3 baseline backtest."""
        logger.info("Running V1.3 baseline...")
        
        metrics = self.backtester.run_backtest(
            price_data,
            lambda pd, d: self.backtester.compute_signals_v13(pd, d),
            name='V1.3_baseline'
        )
        
        self.results['V1.3_baseline'] = metrics
        logger.info(f"  V1.3: Sharpe={metrics.sharpe_ratio:.3f}, CAGR={metrics.cagr:.2%}, MaxDD={metrics.max_drawdown:.2%}")
        
        return metrics
    
    def run_v2_full(self, price_data: Dict[str, pd.DataFrame]) -> BacktestMetrics:
        """Run V2.0 with all enhancements."""
        logger.info("Running V2.0 full...")
        
        metrics = self.backtester.run_backtest(
            price_data,
            lambda pd, d: self.backtester.compute_signals_v2(pd, d, config_overrides=None),
            name='V2.0_full'
        )
        
        self.results['V2.0_full'] = metrics
        logger.info(f"  V2.0: Sharpe={metrics.sharpe_ratio:.3f}, CAGR={metrics.cagr:.2%}, MaxDD={metrics.max_drawdown:.2%}")
        
        return metrics
    
    def run_ablation_study(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, BacktestMetrics]:
        """Run all ablation variants."""
        logger.info("Running ablation study (7 variants)...")
        
        ablation_results = {}
        
        for name, config in tqdm(self.ABLATION_CONFIGS.items(), desc="Ablation variants"):
            metrics = self.backtester.run_backtest(
                price_data,
                lambda pd, d, cfg=config: self.backtester.compute_signals_v2(pd, d, config_overrides=cfg),
                name=name
            )
            
            ablation_results[name] = metrics
            self.results[name] = metrics
            
            logger.debug(f"  {name}: Sharpe={metrics.sharpe_ratio:.3f}")
        
        return ablation_results
    
    def compute_ablation_contributions(self) -> Dict[str, float]:
        """Compute the contribution of each enhancement to Sharpe."""
        if 'V2.0_full' not in self.results:
            return {}
        
        v2_sharpe = self.results['V2.0_full'].sharpe_ratio
        contributions = {}
        
        component_map = {
            'transformer': 'V2_no_transformer',
            'sac': 'V2_no_sac',
            'tda': 'V2_no_tda',
            'ensemble_regime': 'V2_no_ensemble',
            'order_flow': 'V2_no_orderflow',
            'enhanced_momentum': 'V2_no_enhanced_mom',
            'risk_parity': 'V2_no_risk_parity',
        }
        
        for component, ablation_name in component_map.items():
            if ablation_name in self.results:
                ablation_sharpe = self.results[ablation_name].sharpe_ratio
                # Contribution = how much Sharpe drops when this component is removed
                contributions[component] = v2_sharpe - ablation_sharpe
        
        return contributions
    
    def save_results(self, output_dir: str = 'results') -> str:
        """Save results to JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(output_dir, f'backtest_results_{timestamp}.json')
        
        # Convert to serializable format
        results_dict = {
            'timestamp': timestamp,
            'config': asdict(self.config),
            'results': {name: asdict(metrics) for name, metrics in self.results.items()},
            'ablation_contributions': self.compute_ablation_contributions(),
        }
        
        # Also save a "latest" copy
        latest_path = os.path.join(output_dir, 'backtest_results_latest.json')
        
        for path in [output_path, latest_path]:
            with open(path, 'w') as f:
                json.dump(results_dict, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_path}")
        return output_path
    
    def run_all(self) -> Dict[str, BacktestMetrics]:
        """Run all backtests: baseline + V2 + ablation."""
        print("=" * 60)
        print("V2 Backtest Runner with Ablation Study")
        print("=" * 60)
        
        total_start = time.time()
        
        # Load data
        logger.info("Loading price data...")
        try:
            price_data = self.load_data()
        except FileNotFoundError:
            logger.warning("Mock data not found. Generating...")
            from scripts.generate_mock_backtest_data import MockDataGenerator
            gen = MockDataGenerator(seed=42)
            price_data = gen.generate_prices()
            order_flow = gen.generate_order_flow(price_data)
            gen.save_data(price_data, order_flow)
        
        logger.info(f"Loaded {len(price_data)} assets, {len(next(iter(price_data.values())))} days each")
        
        # Run backtests
        self.run_baseline(price_data)
        self.run_v2_full(price_data)
        self.run_ablation_study(price_data)
        
        # Save results
        self.save_results()
        
        # Summary
        total_time = time.time() - total_start
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        
        print(f"\n{'Strategy':<25} {'Sharpe':>8} {'CAGR':>10} {'MaxDD':>10} {'Trades':>8}")
        print("-" * 65)
        
        for name in ['V1.3_baseline', 'V2.0_full'] + list(self.ABLATION_CONFIGS.keys()):
            if name in self.results:
                m = self.results[name]
                print(f"{name:<25} {m.sharpe_ratio:>8.3f} {m.cagr:>9.2%} {m.max_drawdown:>9.2%} {m.n_trades:>8}")
        
        # Ablation contributions
        contributions = self.compute_ablation_contributions()
        if contributions:
            print("\n" + "-" * 65)
            print("ABLATION CONTRIBUTIONS (Sharpe impact when removed)")
            print("-" * 65)
            for comp, contrib in sorted(contributions.items(), key=lambda x: x[1], reverse=True):
                sign = "+" if contrib > 0 else ""
                print(f"  {comp:<25} {sign}{contrib:>.4f}")
        
        print(f"\nTotal time: {total_time:.1f}s")
        
        return self.results


def main():
    """Main entry point."""
    runner = V2BacktestRunner()
    runner.run_all()
    return 0


if __name__ == '__main__':
    sys.exit(main())
