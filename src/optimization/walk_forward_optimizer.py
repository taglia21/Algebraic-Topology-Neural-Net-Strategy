"""
Walk-Forward Optimization Framework - V2.5 Elite Upgrade
=========================================================

Implements anchored and rolling walk-forward optimization
with adaptive window sizing for robust out-of-sample validation.

Key Features:
- Anchored WFO: Fixed start, expanding training window
- Rolling WFO: Fixed-size sliding window
- Adaptive WFO: Window size adjusts to market regime
- Out-of-sample performance tracking
- Overfitting detection metrics
- Parameter stability analysis
- Monte Carlo robustness validation

Research Basis:
- Walk-forward prevents lookahead bias
- Multiple optimization windows reveal parameter stability
- Out-of-sample degradation ratio detects overfitting
- Anchored vs rolling provides different bias-variance tradeoffs

Author: System V2.5
Date: 2025
"""

import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class WFOMode(Enum):
    """Walk-Forward Optimization modes."""
    ANCHORED = "anchored"  # Fixed start, expanding window
    ROLLING = "rolling"    # Fixed-size sliding window
    ADAPTIVE = "adaptive"  # Window adjusts to regime


@dataclass
class WFOConfig:
    """Configuration for Walk-Forward Optimization."""
    
    # Mode selection
    mode: WFOMode = WFOMode.ANCHORED
    
    # Window parameters
    min_train_size: int = 252  # 1 year minimum
    test_size: int = 63  # ~3 months test
    step_size: int = 21  # ~1 month step
    
    # Rolling mode specific
    train_size: int = 504  # 2 years for rolling
    
    # Adaptive mode specific
    min_train_ratio: float = 0.6  # Minimum train portion
    max_train_ratio: float = 0.9  # Maximum train portion
    
    # Validation settings
    n_splits: int = 10  # Number of walk-forward splits
    purge_gap: int = 5  # Gap between train/test to prevent leakage
    embargo_period: int = 5  # Embargo after test period
    
    # Overfitting detection
    overfit_threshold: float = 0.5  # IS/OOS ratio threshold
    min_oos_sharpe: float = 0.5  # Minimum acceptable OOS Sharpe
    
    # Robustness
    monte_carlo_runs: int = 100
    bootstrap_samples: int = 50
    
    # Parallelization
    n_jobs: int = -1


@dataclass
class WFOSplit:
    """Single walk-forward split."""
    split_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_size: int
    test_size: int


@dataclass
class WFOResult:
    """Result from a single walk-forward split."""
    split_id: int
    in_sample_sharpe: float
    out_of_sample_sharpe: float
    in_sample_return: float
    out_of_sample_return: float
    in_sample_drawdown: float
    out_of_sample_drawdown: float
    optimal_params: Dict[str, Any]
    oos_degradation: float  # How much worse OOS is vs IS
    is_overfit: bool


@dataclass
class WFOReport:
    """Complete walk-forward optimization report."""
    mode: WFOMode
    n_splits: int
    splits: List[WFOResult]
    
    # Aggregate metrics
    avg_is_sharpe: float
    avg_oos_sharpe: float
    avg_degradation: float
    overfit_count: int
    
    # Parameter stability
    param_stability: Dict[str, float]  # Variance of each param across splits
    
    # Robustness metrics
    oos_sharpe_std: float
    oos_return_consistency: float  # % of splits with positive OOS return
    
    # Final recommendation
    recommended_params: Dict[str, Any]
    confidence_score: float  # 0-1 confidence in recommendation


class WalkForwardOptimizer:
    """
    Walk-Forward Optimization framework for robust parameter tuning.
    
    Architecture:
    1. Split data into train/test windows
    2. Optimize parameters on each training window
    3. Validate on corresponding test window
    4. Analyze parameter stability and overfitting
    5. Recommend robust parameter set
    """
    
    def __init__(self, config: Optional[WFOConfig] = None):
        self.config = config or WFOConfig()
        self.splits: List[WFOSplit] = []
        self.results: List[WFOResult] = []
    
    def generate_splits(self, n_samples: int) -> List[WFOSplit]:
        """
        Generate walk-forward splits based on configuration.
        
        Args:
            n_samples: Total number of samples in dataset
            
        Returns:
            List of WFOSplit objects
        """
        splits = []
        
        if self.config.mode == WFOMode.ANCHORED:
            splits = self._generate_anchored_splits(n_samples)
        elif self.config.mode == WFOMode.ROLLING:
            splits = self._generate_rolling_splits(n_samples)
        elif self.config.mode == WFOMode.ADAPTIVE:
            splits = self._generate_adaptive_splits(n_samples)
        
        self.splits = splits
        logger.info(f"Generated {len(splits)} walk-forward splits ({self.config.mode.value})")
        
        return splits
    
    def _generate_anchored_splits(self, n_samples: int) -> List[WFOSplit]:
        """Generate anchored (expanding) walk-forward splits."""
        splits = []
        split_id = 0
        
        # Start with minimum training size
        train_end = self.config.min_train_size
        
        while train_end + self.config.purge_gap + self.config.test_size <= n_samples:
            test_start = train_end + self.config.purge_gap
            test_end = min(test_start + self.config.test_size, n_samples)
            
            splits.append(WFOSplit(
                split_id=split_id,
                train_start=0,  # Always start from beginning
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_size=train_end,
                test_size=test_end - test_start
            ))
            
            split_id += 1
            train_end += self.config.step_size
            
            if len(splits) >= self.config.n_splits:
                break
        
        return splits
    
    def _generate_rolling_splits(self, n_samples: int) -> List[WFOSplit]:
        """Generate rolling (fixed-size) walk-forward splits."""
        splits = []
        split_id = 0
        
        train_start = 0
        train_size = self.config.train_size
        
        while train_start + train_size + self.config.purge_gap + self.config.test_size <= n_samples:
            train_end = train_start + train_size
            test_start = train_end + self.config.purge_gap
            test_end = min(test_start + self.config.test_size, n_samples)
            
            splits.append(WFOSplit(
                split_id=split_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_size=train_size,
                test_size=test_end - test_start
            ))
            
            split_id += 1
            train_start += self.config.step_size
            
            if len(splits) >= self.config.n_splits:
                break
        
        return splits
    
    def _generate_adaptive_splits(self, n_samples: int) -> List[WFOSplit]:
        """Generate adaptive walk-forward splits with varying window sizes."""
        splits = []
        split_id = 0
        
        # Calculate split boundaries
        n_splits = self.config.n_splits
        total_test = n_splits * self.config.test_size
        available = n_samples - total_test - n_splits * self.config.purge_gap
        
        if available < self.config.min_train_size:
            logger.warning("Insufficient data for adaptive splits, using anchored")
            return self._generate_anchored_splits(n_samples)
        
        position = 0
        for i in range(n_splits):
            # Adaptive train size based on position
            progress = i / n_splits
            train_ratio = self.config.min_train_ratio + \
                         (self.config.max_train_ratio - self.config.min_train_ratio) * progress
            
            remaining = n_samples - position
            train_size = max(
                self.config.min_train_size,
                int(remaining * train_ratio) - self.config.test_size - self.config.purge_gap
            )
            
            train_end = position + train_size
            test_start = train_end + self.config.purge_gap
            test_end = min(test_start + self.config.test_size, n_samples)
            
            if test_end > n_samples:
                break
            
            splits.append(WFOSplit(
                split_id=split_id,
                train_start=position,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_size=train_size,
                test_size=test_end - test_start
            ))
            
            split_id += 1
            position = test_end + self.config.embargo_period
        
        return splits
    
    def optimize(
        self,
        data: pd.DataFrame,
        objective_func: Callable[[pd.DataFrame, Dict[str, Any]], Tuple[float, Dict[str, float]]],
        param_grid: Dict[str, List[Any]]
    ) -> WFOReport:
        """
        Run walk-forward optimization.
        
        Args:
            data: Full dataset (OHLCV + features)
            objective_func: Function that takes (data, params) and returns 
                          (sharpe_ratio, metrics_dict)
            param_grid: Dictionary of parameter names to lists of values to try
            
        Returns:
            WFOReport with complete analysis
        """
        start_time = time.perf_counter()
        
        if not self.splits:
            self.generate_splits(len(data))
        
        self.results = []
        all_params = []
        
        for split in self.splits:
            logger.debug(f"Processing split {split.split_id + 1}/{len(self.splits)}")
            
            # Extract train and test data
            train_data = data.iloc[split.train_start:split.train_end].copy()
            test_data = data.iloc[split.test_start:split.test_end].copy()
            
            # Find optimal parameters on training data
            best_params, best_is_sharpe, best_is_metrics = self._grid_search(
                train_data, objective_func, param_grid
            )
            
            # Evaluate on test data
            oos_sharpe, oos_metrics = objective_func(test_data, best_params)
            
            # Calculate degradation
            if best_is_sharpe > 0:
                degradation = 1 - (oos_sharpe / best_is_sharpe)
            else:
                degradation = 0 if oos_sharpe > 0 else 1
            
            # Detect overfitting
            is_overfit = (
                degradation > self.config.overfit_threshold or
                oos_sharpe < self.config.min_oos_sharpe
            )
            
            result = WFOResult(
                split_id=split.split_id,
                in_sample_sharpe=best_is_sharpe,
                out_of_sample_sharpe=oos_sharpe,
                in_sample_return=best_is_metrics.get('return', 0),
                out_of_sample_return=oos_metrics.get('return', 0),
                in_sample_drawdown=best_is_metrics.get('max_drawdown', 0),
                out_of_sample_drawdown=oos_metrics.get('max_drawdown', 0),
                optimal_params=best_params,
                oos_degradation=degradation,
                is_overfit=is_overfit
            )
            
            self.results.append(result)
            all_params.append(best_params)
        
        # Analyze results
        report = self._generate_report(all_params)
        
        elapsed = time.perf_counter() - start_time
        logger.info(f"Walk-forward optimization completed in {elapsed:.2f}s")
        
        return report
    
    def _grid_search(
        self,
        data: pd.DataFrame,
        objective_func: Callable,
        param_grid: Dict[str, List[Any]]
    ) -> Tuple[Dict[str, Any], float, Dict[str, float]]:
        """
        Simple grid search for optimal parameters.
        
        For more sophisticated optimization, use with BayesianTuner.
        """
        best_sharpe = float('-inf')
        best_params = {}
        best_metrics = {}
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        from itertools import product
        for values in product(*param_values):
            params = dict(zip(param_names, values))
            
            try:
                sharpe, metrics = objective_func(data, params)
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = params.copy()
                    best_metrics = metrics.copy()
            except Exception as e:
                logger.debug(f"Parameter combination failed: {e}")
                continue
        
        # Fallback if no valid params found
        if not best_params:
            best_params = {k: v[0] for k, v in param_grid.items()}
            best_sharpe = 0
            best_metrics = {}
        
        return best_params, best_sharpe, best_metrics
    
    def _generate_report(self, all_params: List[Dict[str, Any]]) -> WFOReport:
        """Generate comprehensive WFO report."""
        
        if not self.results:
            return WFOReport(
                mode=self.config.mode,
                n_splits=0,
                splits=[],
                avg_is_sharpe=0,
                avg_oos_sharpe=0,
                avg_degradation=0,
                overfit_count=0,
                param_stability={},
                oos_sharpe_std=0,
                oos_return_consistency=0,
                recommended_params={},
                confidence_score=0
            )
        
        # Calculate aggregate metrics
        avg_is_sharpe = np.mean([r.in_sample_sharpe for r in self.results])
        avg_oos_sharpe = np.mean([r.out_of_sample_sharpe for r in self.results])
        avg_degradation = np.mean([r.oos_degradation for r in self.results])
        overfit_count = sum(1 for r in self.results if r.is_overfit)
        
        oos_sharpes = [r.out_of_sample_sharpe for r in self.results]
        oos_sharpe_std = np.std(oos_sharpes)
        
        oos_returns = [r.out_of_sample_return for r in self.results]
        oos_return_consistency = np.mean([1 if r > 0 else 0 for r in oos_returns])
        
        # Analyze parameter stability
        param_stability = self._calculate_param_stability(all_params)
        
        # Determine recommended parameters
        recommended_params = self._select_robust_params(all_params, self.results)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence(
            avg_oos_sharpe, avg_degradation, overfit_count, 
            oos_return_consistency, param_stability
        )
        
        return WFOReport(
            mode=self.config.mode,
            n_splits=len(self.results),
            splits=self.results,
            avg_is_sharpe=avg_is_sharpe,
            avg_oos_sharpe=avg_oos_sharpe,
            avg_degradation=avg_degradation,
            overfit_count=overfit_count,
            param_stability=param_stability,
            oos_sharpe_std=oos_sharpe_std,
            oos_return_consistency=oos_return_consistency,
            recommended_params=recommended_params,
            confidence_score=confidence_score
        )
    
    def _calculate_param_stability(
        self,
        all_params: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate stability (inverse variance) of each parameter."""
        stability = {}
        
        if not all_params:
            return stability
        
        param_names = all_params[0].keys()
        
        for name in param_names:
            values = [p[name] for p in all_params if name in p]
            
            # Handle numeric parameters
            if all(isinstance(v, (int, float)) for v in values):
                if len(set(values)) == 1:
                    stability[name] = 1.0  # Perfect stability
                else:
                    # Coefficient of variation (lower = more stable)
                    cv = np.std(values) / (np.abs(np.mean(values)) + 1e-10)
                    stability[name] = max(0, 1 - cv)
            else:
                # Categorical: most common frequency
                from collections import Counter
                counter = Counter(values)
                most_common_freq = counter.most_common(1)[0][1] / len(values)
                stability[name] = most_common_freq
        
        return stability
    
    def _select_robust_params(
        self,
        all_params: List[Dict[str, Any]],
        results: List[WFOResult]
    ) -> Dict[str, Any]:
        """Select robust parameters based on OOS performance."""
        
        if not all_params or not results:
            return {}
        
        # Weight by OOS Sharpe (only positive)
        weights = []
        for r in results:
            if r.out_of_sample_sharpe > 0 and not r.is_overfit:
                weights.append(r.out_of_sample_sharpe)
            else:
                weights.append(0)
        
        total_weight = sum(weights)
        if total_weight == 0:
            weights = [1.0] * len(results)
            total_weight = len(results)
        
        weights = [w / total_weight for w in weights]
        
        recommended = {}
        param_names = all_params[0].keys()
        
        for name in param_names:
            values = [p[name] for p in all_params]
            
            if all(isinstance(v, (int, float)) for v in values):
                # Weighted average for numeric
                weighted_avg = sum(v * w for v, w in zip(values, weights))
                # Round to original type
                if all(isinstance(v, int) for v in values):
                    recommended[name] = int(round(weighted_avg))
                else:
                    recommended[name] = weighted_avg
            else:
                # Most common (weighted) for categorical
                from collections import defaultdict
                weighted_counts = defaultdict(float)
                for v, w in zip(values, weights):
                    weighted_counts[v] += w
                recommended[name] = max(weighted_counts.keys(), key=lambda k: weighted_counts[k])
        
        return recommended
    
    def _calculate_confidence(
        self,
        avg_oos_sharpe: float,
        avg_degradation: float,
        overfit_count: int,
        oos_consistency: float,
        param_stability: Dict[str, float]
    ) -> float:
        """Calculate confidence score for recommended parameters."""
        
        n_splits = len(self.results)
        if n_splits == 0:
            return 0
        
        # Component scores (0-1)
        sharpe_score = min(1.0, max(0, avg_oos_sharpe / 2))  # Cap at Sharpe 2
        degradation_score = max(0, 1 - avg_degradation)
        overfit_score = 1 - (overfit_count / n_splits)
        consistency_score = oos_consistency
        
        if param_stability:
            stability_score = np.mean(list(param_stability.values()))
        else:
            stability_score = 0.5
        
        # Weighted combination
        confidence = (
            0.3 * sharpe_score +
            0.25 * degradation_score +
            0.2 * overfit_score +
            0.15 * consistency_score +
            0.1 * stability_score
        )
        
        return min(1.0, max(0, confidence))
    
    def monte_carlo_validation(
        self,
        data: pd.DataFrame,
        objective_func: Callable,
        params: Dict[str, Any],
        n_runs: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo validation with random shuffling.
        
        Args:
            data: Full dataset
            objective_func: Objective function
            params: Parameters to validate
            n_runs: Number of MC runs (default from config)
            
        Returns:
            Dictionary with MC statistics
        """
        n_runs = n_runs or self.config.monte_carlo_runs
        
        results = []
        
        for i in range(n_runs):
            # Random subset (80% of data with replacement)
            sample_indices = np.random.choice(
                len(data), size=int(len(data) * 0.8), replace=True
            )
            sample_data = data.iloc[sample_indices].sort_index()
            
            try:
                sharpe, metrics = objective_func(sample_data, params)
                results.append({
                    'sharpe': sharpe,
                    'return': metrics.get('return', 0),
                    'max_drawdown': metrics.get('max_drawdown', 0)
                })
            except:
                continue
        
        if not results:
            return {'valid': False}
        
        sharpes = [r['sharpe'] for r in results]
        returns = [r['return'] for r in results]
        drawdowns = [r['max_drawdown'] for r in results]
        
        return {
            'valid': True,
            'n_runs': len(results),
            'sharpe_mean': np.mean(sharpes),
            'sharpe_std': np.std(sharpes),
            'sharpe_5th_percentile': np.percentile(sharpes, 5),
            'sharpe_95th_percentile': np.percentile(sharpes, 95),
            'return_mean': np.mean(returns),
            'return_std': np.std(returns),
            'max_drawdown_mean': np.mean(drawdowns),
            'positive_sharpe_pct': np.mean([1 if s > 0 else 0 for s in sharpes])
        }
    
    def get_split_visualization_data(self) -> pd.DataFrame:
        """Get data for visualizing splits."""
        if not self.splits:
            return pd.DataFrame()
        
        data = []
        for split in self.splits:
            data.append({
                'split_id': split.split_id,
                'train_start': split.train_start,
                'train_end': split.train_end,
                'test_start': split.test_start,
                'test_end': split.test_end,
                'train_size': split.train_size,
                'test_size': split.test_size
            })
        
        return pd.DataFrame(data)


def example_objective(data: pd.DataFrame, params: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """Example objective function for testing."""
    
    # Simulate strategy returns
    if 'close' in data.columns:
        returns = data['close'].pct_change().dropna()
    else:
        returns = pd.Series(np.random.randn(len(data)) * 0.01)
    
    # Apply some parameter effect
    threshold = params.get('threshold', 0.5)
    lookback = params.get('lookback', 10)
    
    # Simple momentum signal
    signal = (returns.rolling(lookback).mean() > 0).astype(float) * 2 - 1
    strategy_returns = signal.shift(1) * returns
    strategy_returns = strategy_returns.dropna()
    
    if len(strategy_returns) < 10:
        return 0, {}
    
    # Calculate metrics
    mean_return = strategy_returns.mean() * 252
    std_return = strategy_returns.std() * np.sqrt(252)
    sharpe = mean_return / (std_return + 1e-10)
    
    # Cumulative returns for drawdown
    cum_returns = (1 + strategy_returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdown = (cum_returns - rolling_max) / rolling_max
    max_drawdown = abs(drawdown.min())
    
    metrics = {
        'return': mean_return,
        'volatility': std_return,
        'max_drawdown': max_drawdown
    }
    
    return sharpe, metrics


# ============================================================
# SELF-TESTING
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Walk-Forward Optimizer")
    print("=" * 60)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    close = 100 * np.exp(np.cumsum(np.random.randn(n_samples) * 0.01 + 0.0002))
    
    data = pd.DataFrame({
        'close': close,
        'volume': np.random.randint(100000, 1000000, n_samples)
    }, index=dates)
    
    # Test different modes
    print("\n1. Testing Anchored WFO...")
    config_anchored = WFOConfig(
        mode=WFOMode.ANCHORED,
        n_splits=5,
        min_train_size=200,
        test_size=50,
        step_size=100
    )
    
    optimizer = WalkForwardOptimizer(config_anchored)
    splits = optimizer.generate_splits(len(data))
    print(f"   Generated {len(splits)} splits")
    
    if splits:
        print(f"   First split: train[{splits[0].train_start}:{splits[0].train_end}], "
              f"test[{splits[0].test_start}:{splits[0].test_end}]")
    
    # Test Rolling WFO
    print("\n2. Testing Rolling WFO...")
    config_rolling = WFOConfig(
        mode=WFOMode.ROLLING,
        n_splits=5,
        train_size=300,
        test_size=50,
        step_size=100
    )
    
    optimizer_rolling = WalkForwardOptimizer(config_rolling)
    splits_rolling = optimizer_rolling.generate_splits(len(data))
    print(f"   Generated {len(splits_rolling)} splits")
    
    # Test optimization
    print("\n3. Testing full optimization...")
    param_grid = {
        'threshold': [0.3, 0.5, 0.7],
        'lookback': [5, 10, 20]
    }
    
    config_test = WFOConfig(
        mode=WFOMode.ANCHORED,
        n_splits=3,
        min_train_size=200,
        test_size=50,
        step_size=150
    )
    
    optimizer_test = WalkForwardOptimizer(config_test)
    report = optimizer_test.optimize(data, example_objective, param_grid)
    
    print(f"   Avg IS Sharpe: {report.avg_is_sharpe:.4f}")
    print(f"   Avg OOS Sharpe: {report.avg_oos_sharpe:.4f}")
    print(f"   Avg Degradation: {report.avg_degradation:.2%}")
    print(f"   Overfit Count: {report.overfit_count}/{report.n_splits}")
    print(f"   Confidence: {report.confidence_score:.2%}")
    print(f"   Recommended params: {report.recommended_params}")
    
    # Test Monte Carlo
    print("\n4. Testing Monte Carlo validation...")
    mc_results = optimizer_test.monte_carlo_validation(
        data, example_objective, 
        report.recommended_params or {'threshold': 0.5, 'lookback': 10},
        n_runs=20
    )
    
    if mc_results.get('valid'):
        print(f"   MC Sharpe Mean: {mc_results['sharpe_mean']:.4f}")
        print(f"   MC Sharpe Std: {mc_results['sharpe_std']:.4f}")
        print(f"   Positive Sharpe %: {mc_results['positive_sharpe_pct']:.1%}")
    
    # Validation
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    
    results = []
    
    # Check splits generated
    if len(splits) >= 3:
        print("✅ Anchored splits generated")
        results.append(True)
    else:
        print("❌ Anchored splits failed")
        results.append(False)
    
    if len(splits_rolling) >= 3:
        print("✅ Rolling splits generated")
        results.append(True)
    else:
        print("❌ Rolling splits failed")
        results.append(False)
    
    # Check report generation
    if report.n_splits > 0:
        print("✅ WFO report generated")
        results.append(True)
    else:
        print("❌ WFO report failed")
        results.append(False)
    
    # Check Monte Carlo
    if mc_results.get('valid'):
        print("✅ Monte Carlo validation working")
        results.append(True)
    else:
        print("❌ Monte Carlo validation failed")
        results.append(False)
    
    print(f"\nPassed: {sum(results)}/{len(results)}")
