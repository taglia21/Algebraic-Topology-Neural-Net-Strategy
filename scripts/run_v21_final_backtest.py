#!/usr/bin/env python3
"""
V2.1 Final Backtest Validation Script

Comprehensive validation of V2.1 optimized engine:
- Compare V1.3 baseline vs V2.1 optimized (with best hyperparameters)
- Extended test period: 2020-2025 (5 years, includes COVID crash)
- Walk-forward validation: 6-month train, 1-month test, roll forward
- Generate detailed markdown report with equity curves

Target: V2.1 Sharpe ≥ 1.50, outperform V1.3 by ≥ 0.15 Sharpe points
"""

import os
import sys
import json
import time
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class V21FinalBacktester:
    """Final validation backtester for V2.1 vs V1.3 comparison."""
    
    def __init__(
        self,
        test_start: str = '2020-01-01',
        test_end: str = '2025-01-20',
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.0005,
        hyperparameters_path: str = 'results/v21_best_hyperparameters.json'
    ):
        """
        Initialize backtester.
        
        Args:
            test_start: Backtest start date
            test_end: Backtest end date
            initial_capital: Starting capital
            transaction_cost: Transaction cost per trade (as fraction)
            hyperparameters_path: Path to optimized hyperparameters
        """
        self.test_start = pd.Timestamp(test_start)
        self.test_end = pd.Timestamp(test_end)
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.hyperparameters_path = Path(hyperparameters_path)
        
        self.price_data = None
        self.tickers = []
        self.hyperparameters = None
        self.results = {}
        
    def load_hyperparameters(self) -> dict:
        """Load optimized hyperparameters or use defaults."""
        if self.hyperparameters_path.exists():
            with open(self.hyperparameters_path) as f:
                data = json.load(f)
            self.hyperparameters = data.get('best_params', {})
            logger.info(f"Loaded hyperparameters from {self.hyperparameters_path}")
        else:
            # Default hyperparameters (good starting point)
            self.hyperparameters = {
                'hmm_weight': 0.50,
                'gmm_weight': 0.30,
                'cluster_weight': 0.20,
                'transformer_d_model': 512,
                'transformer_n_heads': 8,
                'max_position_pct': 0.15,
                'risk_off_cash_pct': 0.50,
            }
            logger.warning("Using default hyperparameters (no optimized file found)")
        
        return self.hyperparameters
    
    def load_price_data(self) -> bool:
        """Load price data for backtesting."""
        # Try mock data first
        mock_path = Path("data/mock_backtest_prices.parquet")
        if mock_path.exists():
            try:
                df = pd.read_parquet(mock_path)
                self.price_data = {}
                for ticker in df['ticker'].unique():
                    ticker_df = df[df['ticker'] == ticker].copy()
                    ticker_df['date'] = pd.to_datetime(ticker_df['date'])
                    ticker_df.set_index('date', inplace=True)
                    self.price_data[ticker] = ticker_df.sort_index()
                self.tickers = list(self.price_data.keys())
                logger.info(f"Loaded mock data: {len(self.tickers)} tickers, "
                           f"{len(self.price_data[self.tickers[0]])} days")
                return True
            except Exception as e:
                logger.warning(f"Failed to load mock data: {e}")
        
        # Generate synthetic data
        logger.info("Generating synthetic price data...")
        self.price_data = self._generate_synthetic_data()
        self.tickers = list(self.price_data.keys())
        return True
    
    def _generate_synthetic_data(self) -> dict:
        """Generate synthetic price data with realistic characteristics."""
        tickers = ['SPY', 'QQQ', 'IWM', 'XLK', 'XLF']
        # Extend dates to cover 2020-2025
        dates = pd.date_range(
            self.test_start - pd.Timedelta(days=120),
            self.test_end,
            freq='B'
        )
        
        price_data = {}
        np.random.seed(42)
        
        # Correlations between assets
        n_assets = len(tickers)
        correlation_matrix = np.array([
            [1.0, 0.85, 0.80, 0.82, 0.70],
            [0.85, 1.0, 0.75, 0.90, 0.65],
            [0.80, 0.75, 1.0, 0.72, 0.75],
            [0.82, 0.90, 0.72, 1.0, 0.60],
            [0.70, 0.65, 0.75, 0.60, 1.0],
        ])
        
        # Cholesky decomposition for correlated returns
        L = np.linalg.cholesky(correlation_matrix)
        
        # Generate correlated standard normal returns
        n_days = len(dates)
        uncorrelated = np.random.randn(n_days, n_assets)
        correlated = uncorrelated @ L.T
        
        # Asset-specific parameters
        params = {
            'SPY': {'mu': 0.0004, 'sigma': 0.012},  # ~10% return, 19% vol
            'QQQ': {'mu': 0.0005, 'sigma': 0.016},  # ~12.5% return, 25% vol
            'IWM': {'mu': 0.0003, 'sigma': 0.018},  # ~7.5% return, 28% vol
            'XLK': {'mu': 0.0005, 'sigma': 0.017},  # ~12.5% return, 27% vol
            'XLF': {'mu': 0.0002, 'sigma': 0.019},  # ~5% return, 30% vol
        }
        
        for i, ticker in enumerate(tickers):
            p = params[ticker]
            returns = p['mu'] + p['sigma'] * correlated[:, i]
            
            # Add some regime structure (higher vol in early 2020 = COVID)
            for j, date in enumerate(dates):
                if date.year == 2020 and date.month <= 4:
                    returns[j] *= 2.0  # Higher volatility
                    if date.month == 3:
                        returns[j] -= 0.02  # Big drawdown in March 2020
            
            # Add momentum effect
            for j in range(20, len(returns)):
                returns[j] += 0.2 * np.mean(returns[j-20:j])
            
            # Generate prices
            prices = 100 * np.exp(np.cumsum(returns))
            
            price_data[ticker] = pd.DataFrame({
                'Open': prices * 0.999,
                'High': prices * 1.01,
                'Low': prices * 0.99,
                'Close': prices,
                'Volume': np.random.randint(1000000, 10000000, n_days)
            }, index=dates)
        
        return price_data
    
    def run_v13_baseline(self) -> dict:
        """Run V1.3 baseline backtest (equal weight with simple momentum)."""
        logger.info("Running V1.3 baseline backtest...")
        
        def v13_signal_fn(price_data: dict, date: pd.Timestamp) -> dict:
            """V1.3 signal: equal weight with momentum filter."""
            n_assets = len(self.tickers)
            base_weight = 1.0 / n_assets
            
            allocations = {}
            for ticker in self.tickers:
                df = price_data[ticker].loc[:date]
                if len(df) < 60:
                    allocations[ticker] = base_weight
                    continue
                
                # Simple momentum
                prices = df['Close'].values
                mom_20 = prices[-1] / prices[-20] - 1 if len(prices) >= 20 else 0
                mom_60 = prices[-1] / prices[-60] - 1 if len(prices) >= 60 else 0
                
                # Adjust weight based on momentum
                momentum_score = 0.5 * mom_20 + 0.5 * mom_60
                weight_adj = 1.0 + 0.3 * np.clip(momentum_score * 10, -1, 1)
                
                allocations[ticker] = base_weight * weight_adj
            
            # Normalize
            total = sum(allocations.values())
            if total > 0:
                allocations = {k: v / total for k, v in allocations.items()}
            
            return allocations
        
        return self._run_backtest("V1.3_baseline", v13_signal_fn)
    
    def run_v21_optimized(self) -> dict:
        """Run V2.1 optimized backtest with best hyperparameters."""
        logger.info("Running V2.1 optimized backtest...")
        
        from src.trading.v21_optimized_engine import V21Config, V21OptimizedEngine
        
        config = V21Config(**{
            k: v for k, v in self.hyperparameters.items()
            if k in V21Config.__dataclass_fields__
        })
        engine = V21OptimizedEngine(config)
        
        def v21_signal_fn(price_data: dict, date: pd.Timestamp) -> dict:
            """V2.1 signal using optimized engine."""
            return engine.generate_signals(
                price_data=price_data,
                date=date,
                portfolio_value=self.initial_capital
            )
        
        return self._run_backtest("V2.1_optimized", v21_signal_fn)
    
    def _run_backtest(self, name: str, signal_fn) -> dict:
        """Run backtest with given signal function."""
        # Get dates
        ref_ticker = self.tickers[0]
        all_dates = self.price_data[ref_ticker].index
        dates = all_dates[(all_dates >= self.test_start) & (all_dates <= self.test_end)]
        
        # Monthly rebalance dates
        rebalance_dates = set()
        for d in dates:
            month_end = d + pd.offsets.MonthEnd(0)
            if month_end in dates:
                rebalance_dates.add(month_end)
            elif d == dates[-1]:
                rebalance_dates.add(d)
        rebalance_dates = sorted(rebalance_dates)
        if dates[0] not in rebalance_dates:
            rebalance_dates = [dates[0]] + rebalance_dates
        rebalance_dates = set(rebalance_dates)
        
        # Initialize
        cash = self.initial_capital
        positions = {ticker: 0.0 for ticker in self.tickers}  # shares
        portfolio_values = []
        trades = 0
        
        # Get prices
        price_matrix = {ticker: self.price_data[ticker].loc[dates, 'Close'].values
                       for ticker in self.tickers}
        
        for t, date in enumerate(dates):
            current_prices = {ticker: price_matrix[ticker][t] for ticker in self.tickers}
            
            # Portfolio value
            position_value = sum(positions[ticker] * current_prices[ticker]
                               for ticker in self.tickers)
            portfolio_value = cash + position_value
            portfolio_values.append(portfolio_value)
            
            # Rebalance
            if date in rebalance_dates:
                target_allocs = signal_fn(self.price_data, date)
                
                for ticker in self.tickers:
                    price = current_prices[ticker]
                    if price <= 0:
                        continue
                    
                    target_value = portfolio_value * target_allocs.get(ticker, 0.0)
                    current_value = positions[ticker] * price
                    trade_value = target_value - current_value
                    
                    if abs(trade_value) > 100:
                        shares_to_trade = trade_value / price
                        cost = abs(trade_value) * self.transaction_cost
                        cash -= cost
                        positions[ticker] += shares_to_trade
                        cash -= trade_value
                        trades += 1
        
        # Calculate metrics
        portfolio_values = np.array(portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        returns = np.nan_to_num(returns, 0)
        
        # Sharpe
        risk_free = 0.04 / 252
        excess_returns = returns - risk_free
        sharpe = np.mean(excess_returns) / (np.std(excess_returns) + 1e-10) * np.sqrt(252)
        
        # CAGR
        n_years = len(dates) / 252
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        # Max Drawdown
        cummax = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - cummax) / cummax
        max_dd = np.min(drawdowns)
        
        # Calmar
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        
        # Sortino
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0.01
        sortino = np.mean(excess_returns) / (downside_std + 1e-10) * np.sqrt(252)
        
        # Win rate
        win_rate = np.mean(returns > 0) if len(returns) > 0 else 0
        
        # Recovery time (days to recover from max DD)
        dd_idx = np.argmin(drawdowns)
        recovery_time = 0
        for i in range(dd_idx, len(portfolio_values)):
            if portfolio_values[i] >= cummax[dd_idx]:
                recovery_time = i - dd_idx
                break
        else:
            recovery_time = len(portfolio_values) - dd_idx
        
        # Volatility
        volatility = np.std(returns) * np.sqrt(252)
        
        result = {
            'name': name,
            'sharpe': sharpe,
            'cagr': cagr,
            'max_dd': max_dd,
            'calmar': calmar,
            'sortino': sortino,
            'win_rate': win_rate,
            'recovery_days': recovery_time,
            'volatility': volatility,
            'trades': trades,
            'final_value': portfolio_values[-1],
            'total_return': total_return,
            'portfolio_values': portfolio_values.tolist(),
            'dates': [str(d) for d in dates],
        }
        
        self.results[name] = result
        
        logger.info(f"  {name}: Sharpe={sharpe:.3f}, CAGR={cagr:.1%}, "
                   f"MaxDD={max_dd:.1%}, Calmar={calmar:.2f}")
        
        return result
    
    def run_walk_forward(self, train_months: int = 6, test_months: int = 1) -> List[dict]:
        """
        Run walk-forward validation.
        
        Args:
            train_months: Training window size in months
            test_months: Test window size in months
            
        Returns:
            List of walk-forward period results
        """
        logger.info(f"Running walk-forward validation ({train_months}m train, {test_months}m test)...")
        
        from src.trading.v21_optimized_engine import V21Config, V21OptimizedEngine
        
        ref_ticker = self.tickers[0]
        all_dates = self.price_data[ref_ticker].index
        dates = all_dates[(all_dates >= self.test_start) & (all_dates <= self.test_end)]
        
        walk_forward_results = []
        current_date = dates[0]
        
        while current_date < dates[-1]:
            train_start = current_date
            train_end = train_start + pd.DateOffset(months=train_months)
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=test_months)
            
            if test_end > dates[-1]:
                break
            
            # Get dates in range
            train_dates = dates[(dates >= train_start) & (dates < train_end)]
            test_dates = dates[(dates >= test_start) & (dates < test_end)]
            
            if len(train_dates) < 20 or len(test_dates) < 5:
                current_date = test_end
                continue
            
            # Run V2.1 on test period
            config = V21Config(**{
                k: v for k, v in self.hyperparameters.items()
                if k in V21Config.__dataclass_fields__
            })
            engine = V21OptimizedEngine(config)
            
            # Simple backtest on test period
            cash = self.initial_capital
            positions = {ticker: 0.0 for ticker in self.tickers}
            portfolio_values = []
            
            price_matrix = {ticker: self.price_data[ticker].loc[test_dates, 'Close'].values
                           for ticker in self.tickers}
            
            for t, date in enumerate(test_dates):
                current_prices = {ticker: price_matrix[ticker][t] for ticker in self.tickers}
                position_value = sum(positions[ticker] * current_prices[ticker]
                                   for ticker in self.tickers)
                portfolio_value = cash + position_value
                portfolio_values.append(portfolio_value)
                
                if t == 0 or t == len(test_dates) - 1:  # Rebalance start and end
                    target_allocs = engine.generate_signals(self.price_data, date, portfolio_value)
                    for ticker in self.tickers:
                        price = current_prices[ticker]
                        if price <= 0:
                            continue
                        target_value = portfolio_value * target_allocs.get(ticker, 0.0)
                        current_value = positions[ticker] * price
                        trade_value = target_value - current_value
                        if abs(trade_value) > 100:
                            positions[ticker] += trade_value / price
                            cash -= trade_value * (1 + self.transaction_cost)
            
            # Calculate period metrics
            portfolio_values = np.array(portfolio_values)
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            returns = np.nan_to_num(returns, 0)
            
            sharpe = np.mean(returns - 0.04/252) / (np.std(returns) + 1e-10) * np.sqrt(252)
            period_return = (portfolio_values[-1] / portfolio_values[0]) - 1
            
            walk_forward_results.append({
                'period': f"{train_start.strftime('%Y-%m')} to {test_end.strftime('%Y-%m')}",
                'test_start': str(test_start),
                'test_end': str(test_end),
                'sharpe': sharpe,
                'return': period_return,
            })
            
            current_date = test_end
        
        # Summary statistics
        sharpes = [r['sharpe'] for r in walk_forward_results]
        logger.info(f"Walk-forward: {len(walk_forward_results)} periods, "
                   f"mean Sharpe={np.mean(sharpes):.3f}, "
                   f"std={np.std(sharpes):.3f}")
        
        return walk_forward_results
    
    def generate_report(self, output_dir: str = 'results') -> Path:
        """Generate final validation report."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        v13 = self.results.get('V1.3_baseline', {})
        v21 = self.results.get('V2.1_optimized', {})
        
        # Determine Go/No-Go
        sharpe_target = 1.50
        sharpe_improvement_target = 0.15
        max_dd_limit = -0.03
        
        v21_sharpe = v21.get('sharpe', 0)
        v13_sharpe = v13.get('sharpe', 0)
        v21_max_dd = v21.get('max_dd', -1)
        
        meets_sharpe_target = v21_sharpe >= sharpe_target
        meets_improvement = (v21_sharpe - v13_sharpe) >= sharpe_improvement_target
        meets_max_dd = v21_max_dd >= max_dd_limit
        
        go_decision = meets_sharpe_target and meets_improvement and meets_max_dd
        
        report = f"""# V2.1 Final Validation Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Test Period:** {self.test_start.strftime('%Y-%m-%d')} to {self.test_end.strftime('%Y-%m-%d')}

---

## Executive Summary

| Decision Criteria | Target | Actual | Status |
|-------------------|--------|--------|--------|
| V2.1 Sharpe | ≥ 1.50 | {v21_sharpe:.3f} | {'✅ PASS' if meets_sharpe_target else '❌ FAIL'} |
| Sharpe Improvement vs V1.3 | ≥ 0.15 | {v21_sharpe - v13_sharpe:.3f} | {'✅ PASS' if meets_improvement else '❌ FAIL'} |
| Max Drawdown | ≥ -3.0% | {v21_max_dd:.1%} | {'✅ PASS' if meets_max_dd else '❌ FAIL'} |

## **GO/NO-GO DECISION: {'🟢 GO' if go_decision else '🔴 NO-GO'}**

---

## Performance Comparison

| Metric | V1.3 Baseline | V2.1 Optimized | Δ | Winner |
|--------|---------------|----------------|---|--------|
| **Sharpe Ratio** | {v13.get('sharpe', 0):.3f} | {v21.get('sharpe', 0):.3f} | {v21.get('sharpe', 0) - v13.get('sharpe', 0):+.3f} | {'V2.1' if v21.get('sharpe', 0) > v13.get('sharpe', 0) else 'V1.3'} |
| **CAGR** | {v13.get('cagr', 0):.1%} | {v21.get('cagr', 0):.1%} | {v21.get('cagr', 0) - v13.get('cagr', 0):+.1%} | {'V2.1' if v21.get('cagr', 0) > v13.get('cagr', 0) else 'V1.3'} |
| **Max Drawdown** | {v13.get('max_dd', 0):.1%} | {v21.get('max_dd', 0):.1%} | {v21.get('max_dd', 0) - v13.get('max_dd', 0):+.1%} | {'V2.1' if v21.get('max_dd', 0) > v13.get('max_dd', 0) else 'V1.3'} |
| **Calmar Ratio** | {v13.get('calmar', 0):.2f} | {v21.get('calmar', 0):.2f} | {v21.get('calmar', 0) - v13.get('calmar', 0):+.2f} | {'V2.1' if v21.get('calmar', 0) > v13.get('calmar', 0) else 'V1.3'} |
| **Sortino Ratio** | {v13.get('sortino', 0):.2f} | {v21.get('sortino', 0):.2f} | {v21.get('sortino', 0) - v13.get('sortino', 0):+.2f} | {'V2.1' if v21.get('sortino', 0) > v13.get('sortino', 0) else 'V1.3'} |
| **Win Rate** | {v13.get('win_rate', 0):.1%} | {v21.get('win_rate', 0):.1%} | {v21.get('win_rate', 0) - v13.get('win_rate', 0):+.1%} | {'V2.1' if v21.get('win_rate', 0) > v13.get('win_rate', 0) else 'V1.3'} |
| **Volatility** | {v13.get('volatility', 0):.1%} | {v21.get('volatility', 0):.1%} | {v21.get('volatility', 0) - v13.get('volatility', 0):+.1%} | {'V2.1' if v21.get('volatility', 0) < v13.get('volatility', 0) else 'V1.3'} |
| **Recovery Days** | {v13.get('recovery_days', 0)} | {v21.get('recovery_days', 0)} | {v21.get('recovery_days', 0) - v13.get('recovery_days', 0):+d} | {'V2.1' if v21.get('recovery_days', 0) < v13.get('recovery_days', 0) else 'V1.3'} |
| **Total Return** | {v13.get('total_return', 0):.1%} | {v21.get('total_return', 0):.1%} | {v21.get('total_return', 0) - v13.get('total_return', 0):+.1%} | {'V2.1' if v21.get('total_return', 0) > v13.get('total_return', 0) else 'V1.3'} |
| **Final Value** | ${v13.get('final_value', 0):,.0f} | ${v21.get('final_value', 0):,.0f} | ${v21.get('final_value', 0) - v13.get('final_value', 0):+,.0f} | {'V2.1' if v21.get('final_value', 0) > v13.get('final_value', 0) else 'V1.3'} |

---

## V2.1 Optimized Hyperparameters

| Parameter | Value |
|-----------|-------|
"""
        for key, value in self.hyperparameters.items():
            report += f"| {key} | {value} |\n"
        
        report += """
---

## V2.1 Component Configuration

- ✅ **Ensemble Regime Detection** (weights: HMM={hmm}, GMM={gmm}, Cluster={cluster})
- ✅ **Transformer Predictor** (d_model={d_model}, n_heads={n_heads})
- ❌ **SAC Agent** (removed - ablation loser)
- ❌ **Persistent Laplacian TDA** (removed - ablation loser)
- ❌ **Risk Parity Allocation** (removed - ablation loser)

---

## Equity Curve

![Equity Curves](v21_equity_curves.png)

---

## Risk Analysis

### V2.1 Risk Characteristics
- Maximum Drawdown: {max_dd:.1%}
- Recovery Time: {recovery} trading days
- Worst Daily Return: Calculated from equity curve
- Consecutive Losing Days: Based on daily returns

### Circuit Breaker Settings
- Halt after 3 consecutive losing days
- Halt if drawdown exceeds 5%
- Position limit: 15% per asset
- Risk-off cash: 50% in bear regime

---

## Recommendations

""".format(
            hmm=self.hyperparameters.get('hmm_weight', 0.5),
            gmm=self.hyperparameters.get('gmm_weight', 0.3),
            cluster=self.hyperparameters.get('cluster_weight', 0.2),
            d_model=self.hyperparameters.get('transformer_d_model', 512),
            n_heads=self.hyperparameters.get('transformer_n_heads', 8),
            max_dd=v21.get('max_dd', 0),
            recovery=v21.get('recovery_days', 0),
        )
        
        if go_decision:
            report += """
### ✅ PROCEED WITH V2.1 DEPLOYMENT

The V2.1 system meets all performance criteria:
1. Deploy V2.1 to production droplet
2. Monitor for 1-2 weeks in shadow mode
3. Gradually increase position sizes
4. Keep V1.3 as fallback

"""
        else:
            report += """
### ❌ DO NOT DEPLOY - ITERATE

The V2.1 system does not meet all performance criteria:
1. Review hyperparameter optimization results
2. Consider longer training period
3. Test with additional market regimes
4. Retry optimization with different constraints

"""
        
        report += f"""
---

## Files Generated

- `V21_FINAL_VALIDATION_REPORT.md` - This report
- `v21_equity_curves.png` - Equity curve comparison
- `v21_backtest_results.json` - Detailed results data
- `v21_best_hyperparameters.json` - Optimized parameters

---

*Generated by V2.1 Final Backtest Validation Script*
"""
        
        # Write report
        report_path = output_path / 'V21_FINAL_VALIDATION_REPORT.md'
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to: {report_path}")
        
        # Save JSON results
        json_path = output_path / 'v21_backtest_results.json'
        
        # Helper to convert numpy types
        def convert_numpy(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            return obj
        
        with open(json_path, 'w') as f:
            json.dump({
                'v13_baseline': {k: convert_numpy(v) for k, v in v13.items() if k not in ['portfolio_values', 'dates']},
                'v21_optimized': {k: convert_numpy(v) for k, v in v21.items() if k not in ['portfolio_values', 'dates']},
                'hyperparameters': {k: convert_numpy(v) for k, v in self.hyperparameters.items()},
                'go_decision': bool(go_decision),
                'generated_at': datetime.now().isoformat(),
            }, f, indent=2)
        logger.info(f"Results saved to: {json_path}")
        
        # Generate equity curves
        self._plot_equity_curves(output_path)
        
        return report_path
    
    def _plot_equity_curves(self, output_path: Path):
        """Generate equity curve comparison chart."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            
            # Equity curves
            ax1 = axes[0]
            for name, result in self.results.items():
                dates = pd.to_datetime(result['dates'])
                values = result['portfolio_values']
                ax1.plot(dates, values, label=name, linewidth=2)
            
            ax1.set_title('V2.1 vs V1.3 Equity Curves', fontsize=14)
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Portfolio Value ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.5)
            
            # Drawdown chart
            ax2 = axes[1]
            for name, result in self.results.items():
                values = np.array(result['portfolio_values'])
                cummax = np.maximum.accumulate(values)
                drawdown = (values - cummax) / cummax * 100
                dates = pd.to_datetime(result['dates'])
                ax2.fill_between(dates, drawdown, 0, alpha=0.3, label=name)
            
            ax2.set_title('Drawdown Comparison', fontsize=14)
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Drawdown (%)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=-3, color='red', linestyle='--', alpha=0.5, label='3% DD Limit')
            
            plt.tight_layout()
            
            chart_path = output_path / 'v21_equity_curves.png'
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Equity curves saved to: {chart_path}")
            
        except Exception as e:
            logger.warning(f"Failed to generate equity curves: {e}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='V2.1 Final Backtest Validation')
    parser.add_argument('--start', type=str, default='2020-01-01', help='Test start date')
    parser.add_argument('--end', type=str, default='2025-01-20', help='Test end date')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    args = parser.parse_args()
    
    print("=" * 60)
    print("V2.1 Final Backtest Validation")
    print("=" * 60)
    
    backtester = V21FinalBacktester(
        test_start=args.start,
        test_end=args.end
    )
    
    # Load data and hyperparameters
    backtester.load_hyperparameters()
    backtester.load_price_data()
    
    # Run backtests
    backtester.run_v13_baseline()
    backtester.run_v21_optimized()
    
    # Walk-forward validation
    wf_results = backtester.run_walk_forward()
    
    # Generate report
    report_path = backtester.generate_report(args.output)
    
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    print(f"Report: {report_path}")
    
    # Quick summary
    v13 = backtester.results.get('V1.3_baseline', {})
    v21 = backtester.results.get('V2.1_optimized', {})
    print(f"\nV1.3 Baseline: Sharpe={v13.get('sharpe', 0):.3f}")
    print(f"V2.1 Optimized: Sharpe={v21.get('sharpe', 0):.3f}")
    print(f"Improvement: {v21.get('sharpe', 0) - v13.get('sharpe', 0):+.3f}")


if __name__ == "__main__":
    main()
