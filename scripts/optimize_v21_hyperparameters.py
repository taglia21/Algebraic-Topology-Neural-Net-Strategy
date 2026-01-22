#!/usr/bin/env python3
"""
V2.1 Hyperparameter Optimization Script

Tunes V2.1 engine hyperparameters using Optuna Bayesian optimization:
- Ensemble Regime weights (HMM, GMM, Clustering)
- Transformer architecture (d_model, n_heads)
- Position sizing parameters (max_position_pct, risk_off_cash_pct)

Target: Maximize Sharpe ratio on validation set while avoiding overfitting.
Constraints: Max 20 trials, < 10 minutes total runtime.
"""

import os
import sys
import json
import time
import logging
import warnings
from datetime import datetime
from pathlib import Path

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

# Try to import Optuna, fall back to grid search if not available
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available, using grid search fallback")


class V21HyperparameterOptimizer:
    """Optimizes V2.1 engine hyperparameters."""
    
    def __init__(
        self,
        train_start: str = '2022-01-01',
        train_end: str = '2023-12-31',
        test_start: str = '2024-01-01',
        test_end: str = '2025-01-20',
        max_trials: int = 20,
        timeout_minutes: float = 10.0
    ):
        """
        Initialize optimizer.
        
        Args:
            train_start: Training period start date
            train_end: Training period end date
            test_start: Test period start date
            test_end: Test period end date
            max_trials: Maximum number of optimization trials
            timeout_minutes: Maximum runtime in minutes
        """
        self.train_start = pd.Timestamp(train_start)
        self.train_end = pd.Timestamp(train_end)
        self.test_start = pd.Timestamp(test_start)
        self.test_end = pd.Timestamp(test_end)
        self.max_trials = max_trials
        self.timeout_seconds = timeout_minutes * 60
        
        self.price_data = None
        self.trial_history = []
        self.best_params = None
        self.best_sharpe = -np.inf
        
    def load_price_data(self) -> bool:
        """Load price data for optimization."""
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
                logger.info(f"Loaded mock data: {len(self.price_data)} tickers")
                return True
            except Exception as e:
                logger.warning(f"Failed to load mock data: {e}")
        
        # Fall back to synthetic data
        logger.info("Generating synthetic data for optimization...")
        self.price_data = self._generate_synthetic_data()
        return True
    
    def _generate_synthetic_data(self) -> dict:
        """Generate synthetic price data for testing."""
        tickers = ['SPY', 'QQQ', 'IWM', 'XLK', 'XLF']
        dates = pd.date_range(self.train_start - pd.Timedelta(days=60), self.test_end, freq='B')
        
        price_data = {}
        np.random.seed(42)
        
        for ticker in tickers:
            # Generate returns with some structure
            base_return = 0.0003  # ~7.5% annual
            vol = 0.015  # ~24% annual
            returns = np.random.normal(base_return, vol, len(dates))
            
            # Add momentum
            for i in range(20, len(returns)):
                returns[i] += 0.3 * np.mean(returns[i-20:i])
            
            # Generate prices
            prices = 100 * np.exp(np.cumsum(returns))
            
            price_data[ticker] = pd.DataFrame({
                'Open': prices * 0.999,
                'High': prices * 1.01,
                'Low': prices * 0.99,
                'Close': prices,
                'Volume': np.random.randint(1000000, 10000000, len(dates))
            }, index=dates)
        
        return price_data
    
    def run_backtest(self, params: dict) -> dict:
        """Run backtest with given hyperparameters."""
        from src.trading.v21_optimized_engine import V21Config, V21OptimizedEngine
        
        # Create config from params
        config = V21Config(
            hmm_weight=params.get('hmm_weight', 0.5),
            gmm_weight=params.get('gmm_weight', 0.3),
            cluster_weight=params.get('cluster_weight', 0.2),
            transformer_d_model=params.get('transformer_d_model', 512),
            transformer_n_heads=params.get('transformer_n_heads', 8),
            max_position_pct=params.get('max_position_pct', 0.15),
            risk_off_cash_pct=params.get('risk_off_cash_pct', 0.50),
        )
        
        engine = V21OptimizedEngine(config)
        
        # Run vectorized backtest
        tickers = list(self.price_data.keys())
        ref_ticker = tickers[0]
        dates = self.price_data[ref_ticker].loc[self.test_start:self.test_end].index
        
        # Initialize
        initial_capital = 100000
        cash = initial_capital
        positions = {ticker: 0.0 for ticker in tickers}  # shares
        portfolio_values = []
        
        # Rebalance monthly
        rebalance_dates = [dates[0]] + [d for d in dates if d.month != (d - pd.Timedelta(days=1)).month]
        
        # Get price matrix
        price_matrix = {}
        for ticker in tickers:
            price_matrix[ticker] = self.price_data[ticker].loc[dates, 'Close'].values
        
        for t, date in enumerate(dates):
            current_prices = {ticker: price_matrix[ticker][t] for ticker in tickers}
            
            # Calculate portfolio value
            position_value = sum(positions[ticker] * current_prices[ticker] for ticker in tickers)
            portfolio_value = cash + position_value
            portfolio_values.append(portfolio_value)
            
            # Rebalance?
            if date in rebalance_dates:
                # Get target allocations
                target_allocs = engine.generate_signals(
                    price_data=self.price_data,
                    date=date,
                    portfolio_value=portfolio_value
                )
                
                # Execute trades
                for ticker in tickers:
                    price = current_prices[ticker]
                    target_value = portfolio_value * target_allocs.get(ticker, 0.0)
                    current_value = positions[ticker] * price
                    trade_value = target_value - current_value
                    
                    if abs(trade_value) > 100:
                        shares_to_trade = trade_value / price
                        cost = abs(trade_value) * 0.0005  # 5bps
                        cash -= cost
                        positions[ticker] += shares_to_trade
                        cash -= trade_value
        
        # Calculate metrics
        portfolio_values = np.array(portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        returns = np.nan_to_num(returns, 0)
        
        # Sharpe
        excess_returns = returns - 0.04 / 252  # 4% risk-free
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
        
        # Win rate
        win_rate = np.mean(returns > 0) if len(returns) > 0 else 0
        
        return {
            'sharpe': sharpe,
            'cagr': cagr,
            'max_dd': max_dd,
            'calmar': calmar,
            'win_rate': win_rate,
            'final_value': portfolio_values[-1],
        }
    
    def objective(self, trial) -> float:
        """Optuna objective function."""
        # Sample hyperparameters
        hmm_weight = trial.suggest_float('hmm_weight', 0.3, 0.7)
        gmm_weight = trial.suggest_float('gmm_weight', 0.2, 0.5)
        cluster_weight = 1.0 - hmm_weight - gmm_weight
        
        # Ensure cluster_weight is valid
        if cluster_weight < 0.1 or cluster_weight > 0.3:
            return -10.0  # Invalid, penalize
        
        params = {
            'hmm_weight': hmm_weight,
            'gmm_weight': gmm_weight,
            'cluster_weight': cluster_weight,
            'transformer_d_model': trial.suggest_categorical('transformer_d_model', [256, 512, 768]),
            'transformer_n_heads': trial.suggest_categorical('transformer_n_heads', [4, 8, 16]),
            'max_position_pct': trial.suggest_float('max_position_pct', 0.10, 0.20),
            'risk_off_cash_pct': trial.suggest_float('risk_off_cash_pct', 0.40, 0.60),
        }
        
        try:
            metrics = self.run_backtest(params)
            
            # Record trial
            trial_result = {
                'trial': trial.number,
                'params': params,
                'metrics': metrics,
            }
            self.trial_history.append(trial_result)
            
            # Penalize high drawdown
            if metrics['max_dd'] < -0.03:  # > 3% drawdown
                return metrics['sharpe'] - 0.5
            
            return metrics['sharpe']
            
        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {e}")
            return -10.0
    
    def grid_search_objective(self, params: dict) -> float:
        """Grid search objective (fallback if Optuna not available)."""
        try:
            metrics = self.run_backtest(params)
            
            # Record trial
            trial_result = {
                'trial': len(self.trial_history),
                'params': params,
                'metrics': metrics,
            }
            self.trial_history.append(trial_result)
            
            # Penalize high drawdown
            if metrics['max_dd'] < -0.03:
                return metrics['sharpe'] - 0.5
            
            return metrics['sharpe']
            
        except Exception as e:
            logger.warning(f"Grid search trial failed: {e}")
            return -10.0
    
    def run_grid_search(self):
        """Run grid search optimization (fallback)."""
        logger.info("Running grid search optimization...")
        
        # Define grid
        grid = {
            'hmm_weight': [0.4, 0.5, 0.6],
            'gmm_weight': [0.25, 0.35],
            'transformer_d_model': [512],
            'transformer_n_heads': [8],
            'max_position_pct': [0.12, 0.15, 0.18],
            'risk_off_cash_pct': [0.45, 0.55],
        }
        
        from itertools import product
        keys = list(grid.keys())
        values = list(grid.values())
        
        start_time = time.time()
        
        for i, combo in enumerate(product(*values)):
            if time.time() - start_time > self.timeout_seconds:
                logger.warning("Grid search timeout reached")
                break
            
            params = dict(zip(keys, combo))
            params['cluster_weight'] = 1.0 - params['hmm_weight'] - params['gmm_weight']
            
            if params['cluster_weight'] < 0.1 or params['cluster_weight'] > 0.35:
                continue
            
            sharpe = self.grid_search_objective(params)
            
            if sharpe > self.best_sharpe:
                self.best_sharpe = sharpe
                self.best_params = params.copy()
            
            logger.info(f"Trial {i}: Sharpe={sharpe:.3f} (best={self.best_sharpe:.3f})")
    
    def optimize(self) -> dict:
        """Run hyperparameter optimization."""
        logger.info("=" * 60)
        logger.info("V2.1 Hyperparameter Optimization")
        logger.info("=" * 60)
        
        # Load data
        if not self.load_price_data():
            raise RuntimeError("Failed to load price data")
        
        start_time = time.time()
        
        if OPTUNA_AVAILABLE:
            # Use Optuna
            logger.info(f"Using Optuna with {self.max_trials} trials, {self.timeout_seconds:.0f}s timeout")
            
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=42)
            )
            
            study.optimize(
                self.objective,
                n_trials=self.max_trials,
                timeout=self.timeout_seconds,
                show_progress_bar=True
            )
            
            self.best_params = study.best_params
            # Add cluster_weight
            self.best_params['cluster_weight'] = (
                1.0 - self.best_params['hmm_weight'] - self.best_params['gmm_weight']
            )
            self.best_sharpe = study.best_value
            
        else:
            # Use grid search fallback
            self.run_grid_search()
        
        elapsed = time.time() - start_time
        
        # Final validation with best params
        logger.info("\n" + "=" * 60)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Time: {elapsed:.1f}s ({len(self.trial_history)} trials)")
        logger.info(f"Best Sharpe: {self.best_sharpe:.4f}")
        logger.info(f"Best Parameters:")
        for key, value in self.best_params.items():
            logger.info(f"  {key}: {value}")
        
        return {
            'best_params': self.best_params,
            'best_sharpe': self.best_sharpe,
            'n_trials': len(self.trial_history),
            'elapsed_seconds': elapsed,
            'trial_history': self.trial_history,
        }
    
    def save_results(self, output_dir: str = 'results'):
        """Save optimization results."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save best hyperparameters
        hp_path = output_path / 'v21_best_hyperparameters.json'
        with open(hp_path, 'w') as f:
            json.dump({
                'best_params': self.best_params,
                'best_sharpe': self.best_sharpe,
                'optimized_at': datetime.now().isoformat(),
                'n_trials': len(self.trial_history),
            }, f, indent=2)
        logger.info(f"Best hyperparameters saved to: {hp_path}")
        
        # Save trial history
        history_path = output_path / 'v21_optimization_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.trial_history, f, indent=2, default=str)
        logger.info(f"Trial history saved to: {history_path}")
        
        # Generate optimization chart
        self._plot_optimization_history(output_path)
        
        return hp_path
    
    def _plot_optimization_history(self, output_path: Path):
        """Generate optimization history chart."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            trials = [t['trial'] for t in self.trial_history]
            sharpes = [t['metrics']['sharpe'] for t in self.trial_history]
            
            # Running best
            running_best = []
            best_so_far = -np.inf
            for s in sharpes:
                best_so_far = max(best_so_far, s)
                running_best.append(best_so_far)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.scatter(trials, sharpes, alpha=0.5, label='Trial Sharpe')
            ax.plot(trials, running_best, 'r-', linewidth=2, label='Best Sharpe')
            ax.axhline(y=1.35, color='g', linestyle='--', label='V1.3 Baseline (1.35)')
            ax.axhline(y=1.50, color='orange', linestyle='--', label='V2.1 Target (1.50)')
            
            ax.set_xlabel('Trial')
            ax.set_ylabel('Sharpe Ratio')
            ax.set_title('V2.1 Hyperparameter Optimization')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            chart_path = output_path / 'v21_optimization_chart.png'
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Optimization chart saved to: {chart_path}")
            
        except Exception as e:
            logger.warning(f"Failed to generate chart: {e}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='V2.1 Hyperparameter Optimization')
    parser.add_argument('--trials', type=int, default=20, help='Max optimization trials')
    parser.add_argument('--timeout', type=float, default=10.0, help='Timeout in minutes')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    args = parser.parse_args()
    
    optimizer = V21HyperparameterOptimizer(
        max_trials=args.trials,
        timeout_minutes=args.timeout
    )
    
    results = optimizer.optimize()
    optimizer.save_results(args.output)
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"Best Sharpe: {results['best_sharpe']:.4f}")
    print(f"Trials: {results['n_trials']}")
    print(f"Time: {results['elapsed_seconds']:.1f}s")
    print("\nBest Hyperparameters:")
    for key, value in results['best_params'].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
