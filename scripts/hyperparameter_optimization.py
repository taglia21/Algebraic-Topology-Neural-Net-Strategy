"""Hyperparameter Optimization Script for TDA+NN Trading System.

Uses RandomizedSearchCV-style approach to find optimal parameters for:
- LSTM architecture (sequence_length, units, learning_rate, etc.)
- TDA features (persistence_threshold, max_dimension, n_bins)
- Strategy parameters (buy/sell thresholds, min_confidence)

Objective: Maximize weighted score combining Sharpe, return, and drawdown.
"""

import os
import sys
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import logging
import csv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Results directory
RESULTS_DIR = '/workspaces/Algebraic-Topology-Neural-Net-Strategy/results'


@dataclass
class ParameterGrid:
    """Define parameter search space."""
    
    # LSTM parameters
    lstm_params = {
        'sequence_length': [15, 30, 60, 90],
        'lstm_units': [32, 64, 128],
        'learning_rate': [0.0005, 0.001, 0.005],
        'batch_size': [16, 32, 64],
        'epochs': [50, 100, 150]
    }
    
    # TDA parameters
    tda_params = {
        'persistence_threshold': [0.01, 0.05, 0.1],
        'max_dimension': [1, 2],
        'n_bins': [5, 10, 15],
        'window': [15, 20, 30]
    }
    
    # Strategy parameters
    strategy_params = {
        'buy_threshold': [0.52, 0.53, 0.55, 0.57, 0.60],
        'sell_threshold': [0.40, 0.43, 0.45, 0.47, 0.48],
        'min_confidence': [0.05, 0.10, 0.15],
        'risk_per_trade': [0.005, 0.01, 0.015, 0.02],
        'stop_atr_multiplier': [1.5, 2.0, 2.5, 3.0],
        'risk_reward_ratio': [1.5, 2.0, 2.5, 3.0]
    }


class HyperparameterOptimizer:
    """
    Randomized hyperparameter search for TDA+NN trading strategy.
    
    Uses a weighted objective function to balance multiple metrics:
    - Sharpe ratio (40% weight)
    - Total return (30% weight)
    - Profit factor (20% weight)
    - Max drawdown penalty (-10% weight)
    """
    
    def __init__(
        self,
        n_iterations: int = 50,
        objective_weights: Dict[str, float] = None,
        results_dir: str = RESULTS_DIR,
        random_seed: int = 42
    ):
        """
        Initialize HyperparameterOptimizer.
        
        Args:
            n_iterations: Number of random combinations to try
            objective_weights: Dict of {metric: weight}
            results_dir: Directory to save results
            random_seed: Random seed for reproducibility
        """
        self.n_iterations = n_iterations
        self.results_dir = results_dir
        self.random_seed = random_seed
        
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        self.objective_weights = objective_weights or {
            'sharpe_ratio': 0.40,
            'total_return': 0.30,
            'profit_factor': 0.20,
            'max_drawdown': -0.10  # Negative weight = penalty
        }
        
        self.parameter_grid = ParameterGrid()
        self.results: List[Dict] = []
        self.best_params: Dict = {}
        self.best_score: float = float('-inf')
        
        # Initialize log file
        self.log_path = os.path.join(results_dir, 'optimization_log.csv')
        self._init_log_file()
        
        logger.info(f"HyperparameterOptimizer initialized: {n_iterations} iterations")
    
    def _init_log_file(self):
        """Initialize optimization log CSV."""
        os.makedirs(self.results_dir, exist_ok=True)
        
        with open(self.log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'iteration', 'timestamp', 'weighted_score', 'sharpe_ratio',
                'total_return', 'profit_factor', 'max_drawdown', 'num_trades',
                'sequence_length', 'lstm_units', 'learning_rate', 'batch_size',
                'epochs', 'buy_threshold', 'sell_threshold', 'risk_per_trade',
                'stop_atr_multiplier', 'risk_reward_ratio'
            ])
    
    def sample_parameters(self) -> Dict[str, Any]:
        """Sample random parameter combination from grid."""
        params = {}
        
        # LSTM parameters
        for key, values in self.parameter_grid.lstm_params.items():
            params[key] = random.choice(values)
        
        # TDA parameters
        for key, values in self.parameter_grid.tda_params.items():
            params[f'tda_{key}'] = random.choice(values)
        
        # Strategy parameters
        for key, values in self.parameter_grid.strategy_params.items():
            params[key] = random.choice(values)
        
        # Ensure sell_threshold < buy_threshold
        if params['sell_threshold'] >= params['buy_threshold']:
            params['sell_threshold'] = params['buy_threshold'] - 0.05
        
        return params
    
    def calculate_objective(self, metrics: Dict[str, float]) -> float:
        """
        Calculate weighted objective score.
        
        Formula:
        score = 0.4 * sharpe + 0.3 * return + 0.2 * profit_factor - 0.1 * max_drawdown
        
        Args:
            metrics: Dict with performance metrics
            
        Returns:
            Weighted objective score
        """
        score = 0.0
        
        # Sharpe ratio (normalize to ~1 scale)
        sharpe = metrics.get('sharpe_ratio', 0)
        score += self.objective_weights['sharpe_ratio'] * sharpe
        
        # Total return (scale by 10 to make comparable)
        ret = metrics.get('total_return', 0)
        score += self.objective_weights['total_return'] * ret * 10
        
        # Profit factor (cap at 5 to prevent outlier domination)
        pf = min(metrics.get('profit_factor', 1), 5)
        score += self.objective_weights['profit_factor'] * pf
        
        # Max drawdown penalty (scale by 10)
        dd = metrics.get('max_drawdown', 0)
        score += self.objective_weights['max_drawdown'] * dd * 10  # Negative weight
        
        return round(score, 6)
    
    def evaluate_parameters(
        self,
        params: Dict[str, Any],
        data_dict: Dict[str, pd.DataFrame],
        tickers: List[str],
        train_start: str,
        train_end: str,
        test_start: str,
        test_end: str,
        verbose: bool = False
    ) -> Tuple[float, Dict]:
        """
        Evaluate a single parameter combination.
        
        Args:
            params: Parameter combination to evaluate
            data_dict: Dict of {ticker: DataFrame}
            tickers: List of tickers to test
            train_start/end: Training date range
            test_start/end: Testing date range
            verbose: Print progress
            
        Returns:
            Tuple of (objective_score, metrics_dict)
        """
        try:
            import tensorflow as tf
            from src.tda_features import TDAFeatureGenerator
            from src.nn_predictor import NeuralNetPredictor, DataPreprocessor, train_model
            
            # Set TensorFlow seed
            tf.random.set_seed(self.random_seed)
            
            # Initialize TDA generator with params
            tda_gen = TDAFeatureGenerator(
                window=params.get('tda_window', 20),
                embedding_dim=3,
                feature_mode='v1.3'
            )
            
            # Initialize preprocessor
            preprocessor = DataPreprocessor(
                sequence_length=params['sequence_length'],
                use_extended_tda=True
            )
            
            # Build training dataset
            train_X_list = []
            train_y_list = []
            
            for ticker in tickers:
                df = data_dict.get(ticker)
                if df is None or df.empty:
                    continue
                
                train_df = df[(df.index >= train_start) & (df.index <= train_end)].copy()
                
                if len(train_df) < 50:
                    continue
                
                tda_features = tda_gen.generate_features(train_df)
                X, y = preprocessor.prepare_sequences(train_df, tda_features)
                
                if len(X) > 0:
                    train_X_list.append(X.astype(np.float32))
                    train_y_list.append(y.astype(np.float32))
            
            if not train_X_list:
                return float('-inf'), {'error': 'No training data'}
            
            train_X = np.concatenate(train_X_list, axis=0)
            train_y = np.concatenate(train_y_list, axis=0)
            
            # Determine n_features from actual data
            n_features = train_X.shape[2]
            
            # Create and train model
            model = NeuralNetPredictor(
                sequence_length=params['sequence_length'],
                n_features=n_features,
                lstm_units=params['lstm_units']
            )
            
            model.compile_model(
                learning_rate=params['learning_rate'],
                use_entropy_penalty=True,
                entropy_weight=0.05
            )
            
            # Build model
            _ = model(train_X[:1])
            
            # Train (silent mode)
            history = model.fit(
                train_X, train_y,
                batch_size=params['batch_size'],
                epochs=params['epochs'],
                validation_split=0.2,
                verbose=0
            )
            
            # Evaluate on test data
            test_results = []
            
            for ticker in tickers:
                df = data_dict.get(ticker)
                if df is None or df.empty:
                    continue
                
                test_df = df[(df.index >= test_start) & (df.index <= test_end)].copy()
                
                if len(test_df) < 30:
                    continue
                
                # Generate test predictions
                tda_features = tda_gen.generate_features(test_df)
                X_test, y_test = preprocessor.prepare_sequences(test_df, tda_features)
                
                if len(X_test) == 0:
                    continue
                
                predictions = model.predict(X_test.astype(np.float32), verbose=0).flatten()
                
                # Simulate trading with params
                signals = self._simulate_trading(
                    predictions=predictions,
                    prices=test_df['close'].values[-len(predictions):],
                    buy_threshold=params['buy_threshold'],
                    sell_threshold=params['sell_threshold']
                )
                
                test_results.append(signals)
            
            if not test_results:
                return float('-inf'), {'error': 'No test results'}
            
            # Aggregate metrics
            metrics = self._aggregate_simulation_results(test_results)
            score = self.calculate_objective(metrics)
            
            return score, metrics
            
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            return float('-inf'), {'error': str(e)}
    
    def _simulate_trading(
        self,
        predictions: np.ndarray,
        prices: np.ndarray,
        buy_threshold: float,
        sell_threshold: float
    ) -> Dict:
        """
        Simulate trading with given predictions and thresholds.
        
        Returns:
            Dict with trade results
        """
        position = 0  # 0 = flat, 1 = long
        entry_price = 0
        trades = []
        
        for i in range(1, len(predictions)):
            pred = predictions[i]
            price = prices[i]
            
            if position == 0 and pred > buy_threshold:
                # Enter long
                position = 1
                entry_price = price
            
            elif position == 1 and pred < sell_threshold:
                # Exit long
                pnl_pct = (price - entry_price) / entry_price
                trades.append({
                    'pnl_pct': pnl_pct,
                    'entry': entry_price,
                    'exit': price
                })
                position = 0
        
        # Calculate statistics
        if not trades:
            return {
                'num_trades': 0,
                'win_rate': 0,
                'avg_pnl': 0,
                'total_return': 0,
                'sharpe_estimate': 0,
                'max_drawdown': 0,
                'profit_factor': 0
            }
        
        pnls = [t['pnl_pct'] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [abs(p) for p in pnls if p < 0]
        
        win_rate = len(wins) / len(trades)
        avg_pnl = np.mean(pnls)
        total_return = np.sum(pnls)
        
        # Estimate Sharpe from trade returns
        if len(pnls) > 1:
            sharpe_estimate = np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(252 / max(1, len(pnls)))
        else:
            sharpe_estimate = 0
        
        # Profit factor
        gross_wins = sum(wins) if wins else 0
        gross_losses = sum(losses) if losses else 0.001
        profit_factor = gross_wins / gross_losses
        
        # Simple max drawdown estimate
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
        
        return {
            'num_trades': len(trades),
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'total_return': total_return,
            'sharpe_estimate': sharpe_estimate,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor
        }
    
    def _aggregate_simulation_results(self, results: List[Dict]) -> Dict:
        """Aggregate simulation results across tickers."""
        if not results:
            return {}
        
        total_trades = sum(r['num_trades'] for r in results)
        
        if total_trades == 0:
            return {
                'sharpe_ratio': 0,
                'total_return': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'num_trades': 0
            }
        
        # Weighted averages
        weights = [r['num_trades'] / total_trades for r in results]
        
        return {
            'sharpe_ratio': sum(r['sharpe_estimate'] * w for r, w in zip(results, weights)),
            'total_return': sum(r['total_return'] for r in results) / len(results),
            'profit_factor': sum(r['profit_factor'] * w for r, w in zip(results, weights)),
            'max_drawdown': max(r['max_drawdown'] for r in results),
            'num_trades': total_trades,
            'win_rate': sum(r['win_rate'] * w for r, w in zip(results, weights))
        }
    
    def run_optimization(
        self,
        data_dict: Dict[str, pd.DataFrame],
        tickers: List[str],
        train_start: str = '2015-01-01',
        train_end: str = '2022-12-31',
        test_start: str = '2023-01-01',
        test_end: str = '2025-12-31',
        verbose: bool = True
    ) -> Dict:
        """
        Run full hyperparameter optimization.
        
        Args:
            data_dict: Dict of {ticker: DataFrame}
            tickers: List of tickers
            train_start/end: Training period
            test_start/end: Testing period
            verbose: Print progress
            
        Returns:
            Dict with best parameters and results
        """
        print("\n" + "=" * 70)
        print("HYPERPARAMETER OPTIMIZATION")
        print("=" * 70)
        print(f"  Iterations: {self.n_iterations}")
        print(f"  Tickers: {tickers}")
        print(f"  Train: {train_start} to {train_end}")
        print(f"  Test: {test_start} to {test_end}")
        print("-" * 70)
        
        self.results = []
        self.best_params = {}
        self.best_score = float('-inf')
        
        for i in range(self.n_iterations):
            # Sample parameters
            params = self.sample_parameters()
            
            if verbose:
                print(f"\n  Iteration {i+1}/{self.n_iterations}:")
                print(f"    seq_len={params['sequence_length']}, "
                      f"units={params['lstm_units']}, "
                      f"lr={params['learning_rate']}, "
                      f"buy={params['buy_threshold']:.2f}, "
                      f"sell={params['sell_threshold']:.2f}")
            
            # Evaluate
            score, metrics = self.evaluate_parameters(
                params=params,
                data_dict=data_dict,
                tickers=tickers,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                verbose=False
            )
            
            # Record result
            result = {
                'iteration': i + 1,
                'params': params,
                'score': score,
                'metrics': metrics
            }
            self.results.append(result)
            
            # Log to CSV
            self._log_result(result)
            
            if verbose:
                print(f"    Score: {score:.4f} | Sharpe: {metrics.get('sharpe_ratio', 0):.3f} | "
                      f"Return: {metrics.get('total_return', 0)*100:.2f}% | "
                      f"Trades: {metrics.get('num_trades', 0)}")
            
            # Update best
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()
                self.best_metrics = metrics.copy()
                
                if verbose:
                    print(f"    ★ NEW BEST!")
        
        # Save results
        self._save_results()
        
        # Print summary
        self._print_summary()
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'best_metrics': self.best_metrics,
            'all_results': self.results
        }
    
    def _log_result(self, result: Dict):
        """Log a single result to CSV."""
        params = result['params']
        metrics = result['metrics']
        
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                result['iteration'],
                datetime.now().isoformat(),
                result['score'],
                metrics.get('sharpe_ratio', 0),
                metrics.get('total_return', 0),
                metrics.get('profit_factor', 0),
                metrics.get('max_drawdown', 0),
                metrics.get('num_trades', 0),
                params.get('sequence_length', ''),
                params.get('lstm_units', ''),
                params.get('learning_rate', ''),
                params.get('batch_size', ''),
                params.get('epochs', ''),
                params.get('buy_threshold', ''),
                params.get('sell_threshold', ''),
                params.get('risk_per_trade', ''),
                params.get('stop_atr_multiplier', ''),
                params.get('risk_reward_ratio', '')
            ])
    
    def _save_results(self):
        """Save optimization results to JSON."""
        output = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'n_iterations': self.n_iterations,
                'random_seed': self.random_seed,
                'objective_weights': self.objective_weights
            },
            'best_params': self.best_params,
            'best_score': self.best_score,
            'best_metrics': getattr(self, 'best_metrics', {}),
            'top_5_results': sorted(self.results, key=lambda x: x['score'], reverse=True)[:5]
        }
        
        output_path = os.path.join(self.results_dir, 'optimal_parameters.json')
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {output_path}")
    
    def _print_summary(self):
        """Print optimization summary."""
        print("\n" + "=" * 70)
        print("OPTIMIZATION SUMMARY")
        print("=" * 70)
        
        print(f"\n  Best Score: {self.best_score:.4f}")
        print(f"\n  Best Parameters:")
        
        # LSTM params
        print(f"    LSTM:")
        print(f"      sequence_length: {self.best_params.get('sequence_length')}")
        print(f"      lstm_units: {self.best_params.get('lstm_units')}")
        print(f"      learning_rate: {self.best_params.get('learning_rate')}")
        print(f"      batch_size: {self.best_params.get('batch_size')}")
        print(f"      epochs: {self.best_params.get('epochs')}")
        
        # Strategy params
        print(f"    Strategy:")
        print(f"      buy_threshold: {self.best_params.get('buy_threshold')}")
        print(f"      sell_threshold: {self.best_params.get('sell_threshold')}")
        print(f"      risk_per_trade: {self.best_params.get('risk_per_trade')}")
        print(f"      stop_atr_multiplier: {self.best_params.get('stop_atr_multiplier')}")
        print(f"      risk_reward_ratio: {self.best_params.get('risk_reward_ratio')}")
        
        print(f"\n  Best Metrics:")
        best_metrics = getattr(self, 'best_metrics', {})
        print(f"    Sharpe Ratio: {best_metrics.get('sharpe_ratio', 0):.4f}")
        print(f"    Total Return: {best_metrics.get('total_return', 0)*100:.2f}%")
        print(f"    Profit Factor: {best_metrics.get('profit_factor', 0):.2f}")
        print(f"    Max Drawdown: {best_metrics.get('max_drawdown', 0)*100:.2f}%")
        print(f"    Num Trades: {best_metrics.get('num_trades', 0)}")
        
        print("\n" + "=" * 70)
        print(f"  Full results saved to: {self.results_dir}/optimal_parameters.json")
        print(f"  Optimization log: {self.log_path}")
        print("=" * 70)


def main():
    """Run hyperparameter optimization."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperparameter Optimization')
    parser.add_argument('--iterations', type=int, default=50, help='Number of iterations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--train-start', type=str, default='2015-01-01', help='Training start date')
    parser.add_argument('--train-end', type=str, default='2022-12-31', help='Training end date')
    parser.add_argument('--test-start', type=str, default='2023-01-01', help='Testing start date')
    parser.add_argument('--test-end', type=str, default='2025-12-31', help='Testing end date')
    
    args = parser.parse_args()
    
    # Import data loading
    from src.data.data_provider import get_ohlcv_data
    
    # Load data
    tickers = ['SPY', 'QQQ', 'IWM', 'XLF', 'XLK']
    data_dict = {}
    
    print("\nLoading data...")
    for ticker in tickers:
        try:
            df = get_ohlcv_data(ticker, args.train_start, args.test_end, provider='yfinance')
            if not df.empty:
                data_dict[ticker] = df
                print(f"  {ticker}: {len(df)} bars")
        except Exception as e:
            print(f"  {ticker}: Error - {e}")
    
    if not data_dict:
        print("Error: No data loaded")
        return
    
    # Run optimization
    optimizer = HyperparameterOptimizer(
        n_iterations=args.iterations,
        random_seed=args.seed
    )
    
    results = optimizer.run_optimization(
        data_dict=data_dict,
        tickers=list(data_dict.keys()),
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        verbose=True
    )
    
    return results


if __name__ == "__main__":
    main()
