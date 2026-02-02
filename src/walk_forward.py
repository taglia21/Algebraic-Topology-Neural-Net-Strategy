"""
Walk-Forward Optimization Framework for Trading Strategy Validation.

This module implements rigorous walk-forward analysis to prevent overfitting
and validate strategy robustness through proper train/validate/test splits.

Walk-forward analysis is the gold standard for strategy validation because:
1. Parameters are optimized only on past data (no look-ahead bias)
2. Performance is measured on truly out-of-sample data
3. Multiple windows provide statistical significance

Academic References:
- Pardo, R. (2008). "The Evaluation and Optimization of Trading Strategies"
- Bailey, D.H. et al. (2017). "Stock Portfolio Design and Backtest Overfitting"
- López de Prado, M. (2018). "Advances in Financial Machine Learning"
- White, H. (2000). "A Reality Check for Data Snooping" - Econometrica

Author: Trading System
Version: 1.0.0
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any, Callable, Dict, Generic, Iterator, List, 
    Optional, Protocol, Tuple, TypeVar, Union
)
import numpy as np

try:
    from scipy import stats
    from scipy.optimize import minimize, differential_evolution
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not available - some optimization methods disabled")


# =============================================================================
# Type Definitions and Protocols
# =============================================================================

T = TypeVar('T')
ParamDict = Dict[str, Any]
Returns = np.ndarray


class OptimizationObjective(Enum):
    """Supported optimization objectives."""
    SHARPE = "sharpe"
    SORTINO = "sortino"
    CALMAR = "calmar"
    MAX_RETURN = "max_return"
    MIN_VOLATILITY = "min_volatility"
    MIN_DRAWDOWN = "min_drawdown"
    CUSTOM = "custom"


class StrategyProtocol(Protocol):
    """Protocol for strategy classes compatible with walk-forward."""
    
    def set_parameters(self, params: ParamDict) -> None:
        """Set strategy parameters."""
        ...
    
    def run(self, data: np.ndarray) -> Returns:
        """Run strategy on data and return period returns."""
        ...
    
    def get_parameters(self) -> ParamDict:
        """Get current parameter values."""
        ...


@dataclass
class WindowSpec:
    """Specification for a single walk-forward window."""
    window_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    
    # Optional validation period (for 3-way split)
    validate_start: Optional[int] = None
    validate_end: Optional[int] = None
    
    @property
    def train_size(self) -> int:
        return self.train_end - self.train_start
    
    @property
    def test_size(self) -> int:
        return self.test_end - self.test_start
    
    @property
    def validate_size(self) -> Optional[int]:
        if self.validate_start is not None and self.validate_end is not None:
            return self.validate_end - self.validate_start
        return None
    
    def has_validation(self) -> bool:
        return self.validate_start is not None


@dataclass
class WindowResult:
    """Results from a single walk-forward window."""
    window_id: int
    spec: WindowSpec
    
    # Optimized parameters
    optimal_params: ParamDict
    
    # Performance metrics
    train_returns: np.ndarray
    test_returns: np.ndarray
    validate_returns: Optional[np.ndarray] = None
    
    # Computed metrics
    train_sharpe: float = 0.0
    test_sharpe: float = 0.0
    validate_sharpe: Optional[float] = None
    
    train_total_return: float = 0.0
    test_total_return: float = 0.0
    
    # Optimization metadata
    optimization_iterations: int = 0
    optimization_time_seconds: float = 0.0
    
    @property
    def sharpe_degradation(self) -> float:
        """Ratio of test to train Sharpe (1.0 = no degradation)."""
        if self.train_sharpe == 0:
            return 0.0
        return self.test_sharpe / self.train_sharpe if self.train_sharpe != 0 else 0.0


@dataclass 
class WalkForwardSummary:
    """Summary of complete walk-forward analysis."""
    n_windows: int
    window_results: List[WindowResult]
    
    # Aggregate metrics
    mean_train_sharpe: float
    mean_test_sharpe: float
    std_test_sharpe: float
    
    mean_sharpe_degradation: float
    degradation_consistency: float  # Std of degradation ratios
    
    # Combined OOS performance
    combined_oos_returns: np.ndarray
    combined_oos_sharpe: float
    combined_oos_total_return: float
    
    # Statistical tests
    oos_t_statistic: float
    oos_p_value: float
    
    # Robustness assessment
    is_robust: bool
    robustness_score: float  # 0-100
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# Utility Functions
# =============================================================================

def _calculate_sharpe(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualization: float = 252.0
) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Parameters
    ----------
    returns : np.ndarray
        Period returns.
    risk_free_rate : float
        Annualized risk-free rate.
    annualization : float
        Annualization factor (252 for daily).
        
    Returns
    -------
    float
        Annualized Sharpe ratio.
    """
    if len(returns) == 0:
        return 0.0
    
    period_rf = risk_free_rate / annualization
    excess = returns - period_rf
    
    std = np.std(excess, ddof=1)
    if std == 0 or not np.isfinite(std):
        return 0.0
    
    return float((np.mean(excess) / std) * np.sqrt(annualization))


def _calculate_sortino(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualization: float = 252.0
) -> float:
    """Calculate annualized Sortino ratio."""
    if len(returns) == 0:
        return 0.0
    
    period_rf = risk_free_rate / annualization
    excess = returns - period_rf
    
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float('inf') if np.mean(excess) > 0 else 0.0
    
    downside_std = np.sqrt(np.mean(downside**2))
    if downside_std == 0:
        return float('inf') if np.mean(excess) > 0 else 0.0
    
    return float((np.mean(excess) / downside_std) * np.sqrt(annualization))


def _calculate_max_drawdown(returns: np.ndarray) -> float:
    """Calculate maximum drawdown from returns."""
    if len(returns) == 0:
        return 0.0
    
    cum_returns = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - running_max) / running_max
    
    return float(np.min(drawdowns))


def _calculate_calmar(
    returns: np.ndarray,
    annualization: float = 252.0
) -> float:
    """Calculate Calmar ratio (annualized return / max drawdown)."""
    if len(returns) == 0:
        return 0.0
    
    ann_return = (np.mean(returns) + 1) ** annualization - 1
    max_dd = abs(_calculate_max_drawdown(returns))
    
    if max_dd == 0:
        return float('inf') if ann_return > 0 else 0.0
    
    return float(ann_return / max_dd)


def _get_objective_function(
    objective: OptimizationObjective,
    custom_func: Optional[Callable[[np.ndarray], float]] = None
) -> Callable[[np.ndarray], float]:
    """Get objective function for optimization (to be maximized)."""
    
    objectives = {
        OptimizationObjective.SHARPE: _calculate_sharpe,
        OptimizationObjective.SORTINO: _calculate_sortino,
        OptimizationObjective.CALMAR: _calculate_calmar,
        OptimizationObjective.MAX_RETURN: lambda r: float(np.prod(1 + r) - 1),
        OptimizationObjective.MIN_VOLATILITY: lambda r: -float(np.std(r, ddof=1)),
        OptimizationObjective.MIN_DRAWDOWN: lambda r: -abs(_calculate_max_drawdown(r)),
    }
    
    if objective == OptimizationObjective.CUSTOM:
        if custom_func is None:
            raise ValueError("Custom objective requires custom_func parameter")
        return custom_func
    
    return objectives[objective]


def _verify_no_lookahead(windows: List[WindowSpec]) -> bool:
    """
    Verify that window specifications have no look-ahead bias.
    
    Checks:
    1. Train period ends before test period starts
    2. Test periods are non-overlapping
    3. Windows are in chronological order
    
    Parameters
    ----------
    windows : List[WindowSpec]
        List of window specifications.
        
    Returns
    -------
    bool
        True if no look-ahead bias detected.
        
    Raises
    ------
    ValueError
        If look-ahead bias is detected.
    """
    for w in windows:
        # Train must end before test starts
        if w.train_end > w.test_start:
            raise ValueError(
                f"Look-ahead bias in window {w.window_id}: "
                f"train_end ({w.train_end}) > test_start ({w.test_start})"
            )
        
        # Validation (if present) must be between train and test
        if w.has_validation():
            if w.validate_start < w.train_end:
                raise ValueError(
                    f"Look-ahead bias in window {w.window_id}: "
                    f"validate_start ({w.validate_start}) < train_end ({w.train_end})"
                )
            if w.validate_end > w.test_start:
                raise ValueError(
                    f"Look-ahead bias in window {w.window_id}: "
                    f"validate_end ({w.validate_end}) > test_start ({w.test_start})"
                )
    
    # Check test periods don't overlap
    test_ranges = [(w.test_start, w.test_end) for w in windows]
    for i, (s1, e1) in enumerate(test_ranges):
        for j, (s2, e2) in enumerate(test_ranges):
            if i < j:
                if not (e1 <= s2 or e2 <= s1):
                    raise ValueError(
                        f"Overlapping test periods: window {i} ({s1}-{e1}) "
                        f"and window {j} ({s2}-{e2})"
                    )
    
    return True


# =============================================================================
# WalkForwardEngine: Base Class
# =============================================================================

class WalkForwardEngine(ABC):
    """
    Abstract base class for walk-forward optimization engines.
    
    Walk-forward analysis divides historical data into multiple windows,
    each with a training period (for parameter optimization) and a test
    period (for out-of-sample evaluation).
    
    Attributes
    ----------
    data : np.ndarray
        Full dataset (typically price or return series).
    train_ratio : float
        Fraction of each window used for training.
    n_windows : int
        Number of walk-forward windows.
    objective : OptimizationObjective
        Metric to optimize.
    annualization : float
        Factor for annualizing metrics (252 for daily).
        
    References
    ----------
    Pardo, R. (2008). "The Evaluation and Optimization of Trading Strategies"
    Chapter 10: Walk-Forward Analysis
    """
    
    def __init__(
        self,
        data: np.ndarray,
        train_ratio: float = 0.7,
        n_windows: int = 5,
        objective: OptimizationObjective = OptimizationObjective.SHARPE,
        custom_objective: Optional[Callable[[np.ndarray], float]] = None,
        annualization: float = 252.0,
        purge_gap: int = 0,
        embargo_gap: int = 0
    ):
        """
        Initialize WalkForwardEngine.
        
        Parameters
        ----------
        data : np.ndarray
            Historical data series.
        train_ratio : float
            Fraction for training (0 < train_ratio < 1).
        n_windows : int
            Number of walk-forward windows.
        objective : OptimizationObjective
            Optimization objective.
        custom_objective : Optional[Callable]
            Custom objective function if objective=CUSTOM.
        annualization : float
            Annualization factor for metrics.
        purge_gap : int
            Gap between train and test to prevent leakage.
        embargo_gap : int
            Additional gap after test before next train.
        """
        self.data = np.asarray(data, dtype=np.float64)
        self.train_ratio = train_ratio
        self.n_windows = n_windows
        self.objective = objective
        self.custom_objective = custom_objective
        self.annualization = annualization
        self.purge_gap = purge_gap
        self.embargo_gap = embargo_gap
        
        self._objective_func = _get_objective_function(objective, custom_objective)
        self._windows: Optional[List[WindowSpec]] = None
        
        # Validate inputs
        if not 0 < train_ratio < 1:
            raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
        if n_windows < 2:
            raise ValueError(f"n_windows must be >= 2, got {n_windows}")
        if len(data) < 100:
            warnings.warn(f"Small dataset ({len(data)} points) - results may be unreliable")
    
    @abstractmethod
    def generate_windows(self) -> List[WindowSpec]:
        """Generate window specifications for walk-forward analysis."""
        pass
    
    def get_windows(self) -> List[WindowSpec]:
        """Get or generate window specifications."""
        if self._windows is None:
            self._windows = self.generate_windows()
            _verify_no_lookahead(self._windows)
        return self._windows
    
    def optimize_window(
        self,
        window: WindowSpec,
        strategy: StrategyProtocol,
        param_bounds: Dict[str, Tuple[float, float]],
        optimization_method: str = "grid",
        grid_points: int = 10,
        max_iterations: int = 100
    ) -> WindowResult:
        """
        Optimize strategy parameters for a single window.
        
        Parameters
        ----------
        window : WindowSpec
            Window specification.
        strategy : StrategyProtocol
            Strategy to optimize.
        param_bounds : Dict[str, Tuple[float, float]]
            Parameter name -> (min, max) bounds.
        optimization_method : str
            One of: 'grid', 'random', 'differential_evolution', 'nelder_mead'
        grid_points : int
            Points per dimension for grid search.
        max_iterations : int
            Max iterations for iterative methods.
            
        Returns
        -------
        WindowResult
            Optimization results for this window.
        """
        import time
        start_time = time.time()
        
        train_data = self.data[window.train_start:window.train_end]
        test_data = self.data[window.test_start:window.test_end]
        validate_data = None
        
        if window.has_validation():
            validate_data = self.data[window.validate_start:window.validate_end]
        
        # Run optimization
        if optimization_method == "grid":
            optimal_params, n_iters = self._grid_search(
                strategy, train_data, param_bounds, grid_points
            )
        elif optimization_method == "random":
            optimal_params, n_iters = self._random_search(
                strategy, train_data, param_bounds, max_iterations
            )
        elif optimization_method == "differential_evolution" and HAS_SCIPY:
            optimal_params, n_iters = self._differential_evolution(
                strategy, train_data, param_bounds, max_iterations
            )
        elif optimization_method == "nelder_mead" and HAS_SCIPY:
            optimal_params, n_iters = self._nelder_mead(
                strategy, train_data, param_bounds, max_iterations
            )
        else:
            raise ValueError(f"Unknown optimization method: {optimization_method}")
        
        # Apply optimal parameters and get returns
        strategy.set_parameters(optimal_params)
        train_returns = strategy.run(train_data)
        test_returns = strategy.run(test_data)
        
        validate_returns = None
        validate_sharpe = None
        if validate_data is not None:
            validate_returns = strategy.run(validate_data)
            validate_sharpe = _calculate_sharpe(validate_returns, annualization=self.annualization)
        
        elapsed = time.time() - start_time
        
        return WindowResult(
            window_id=window.window_id,
            spec=window,
            optimal_params=optimal_params,
            train_returns=train_returns,
            test_returns=test_returns,
            validate_returns=validate_returns,
            train_sharpe=_calculate_sharpe(train_returns, annualization=self.annualization),
            test_sharpe=_calculate_sharpe(test_returns, annualization=self.annualization),
            validate_sharpe=validate_sharpe,
            train_total_return=float(np.prod(1 + train_returns) - 1),
            test_total_return=float(np.prod(1 + test_returns) - 1),
            optimization_iterations=n_iters,
            optimization_time_seconds=elapsed
        )
    
    def _grid_search(
        self,
        strategy: StrategyProtocol,
        train_data: np.ndarray,
        param_bounds: Dict[str, Tuple[float, float]],
        grid_points: int
    ) -> Tuple[ParamDict, int]:
        """Exhaustive grid search over parameter space."""
        param_names = list(param_bounds.keys())
        
        # Create grid for each parameter
        grids = []
        for name in param_names:
            low, high = param_bounds[name]
            grids.append(np.linspace(low, high, grid_points))
        
        # Generate all combinations
        from itertools import product
        combinations = list(product(*grids))
        
        best_score = float('-inf')
        best_params = {name: (param_bounds[name][0] + param_bounds[name][1]) / 2 
                       for name in param_names}
        
        for combo in combinations:
            params = {name: val for name, val in zip(param_names, combo)}
            
            try:
                strategy.set_parameters(params)
                returns = strategy.run(train_data)
                score = self._objective_func(returns)
                
                if np.isfinite(score) and score > best_score:
                    best_score = score
                    best_params = params.copy()
            except Exception:
                continue
        
        return best_params, len(combinations)
    
    def _random_search(
        self,
        strategy: StrategyProtocol,
        train_data: np.ndarray,
        param_bounds: Dict[str, Tuple[float, float]],
        max_iterations: int
    ) -> Tuple[ParamDict, int]:
        """Random search over parameter space."""
        param_names = list(param_bounds.keys())
        
        best_score = float('-inf')
        best_params = {name: (param_bounds[name][0] + param_bounds[name][1]) / 2 
                       for name in param_names}
        
        rng = np.random.default_rng()
        
        for _ in range(max_iterations):
            params = {}
            for name in param_names:
                low, high = param_bounds[name]
                params[name] = rng.uniform(low, high)
            
            try:
                strategy.set_parameters(params)
                returns = strategy.run(train_data)
                score = self._objective_func(returns)
                
                if np.isfinite(score) and score > best_score:
                    best_score = score
                    best_params = params.copy()
            except Exception:
                continue
        
        return best_params, max_iterations
    
    def _differential_evolution(
        self,
        strategy: StrategyProtocol,
        train_data: np.ndarray,
        param_bounds: Dict[str, Tuple[float, float]],
        max_iterations: int
    ) -> Tuple[ParamDict, int]:
        """Differential evolution optimization."""
        param_names = list(param_bounds.keys())
        bounds = [param_bounds[name] for name in param_names]
        
        def objective(x):
            params = {name: val for name, val in zip(param_names, x)}
            try:
                strategy.set_parameters(params)
                returns = strategy.run(train_data)
                score = self._objective_func(returns)
                return -score if np.isfinite(score) else float('inf')
            except Exception:
                return float('inf')
        
        result = differential_evolution(
            objective,
            bounds,
            maxiter=max_iterations,
            seed=42,
            polish=False
        )
        
        best_params = {name: val for name, val in zip(param_names, result.x)}
        return best_params, result.nfev
    
    def _nelder_mead(
        self,
        strategy: StrategyProtocol,
        train_data: np.ndarray,
        param_bounds: Dict[str, Tuple[float, float]],
        max_iterations: int
    ) -> Tuple[ParamDict, int]:
        """Nelder-Mead simplex optimization."""
        param_names = list(param_bounds.keys())
        
        # Start from center of bounds
        x0 = [(param_bounds[name][0] + param_bounds[name][1]) / 2 for name in param_names]
        
        def objective(x):
            # Clip to bounds
            x_clipped = []
            for i, name in enumerate(param_names):
                low, high = param_bounds[name]
                x_clipped.append(np.clip(x[i], low, high))
            
            params = {name: val for name, val in zip(param_names, x_clipped)}
            try:
                strategy.set_parameters(params)
                returns = strategy.run(train_data)
                score = self._objective_func(returns)
                return -score if np.isfinite(score) else float('inf')
            except Exception:
                return float('inf')
        
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': max_iterations}
        )
        
        best_params = {name: np.clip(val, param_bounds[name][0], param_bounds[name][1])
                       for name, val in zip(param_names, result.x)}
        return best_params, result.nfev
    
    def run(
        self,
        strategy: StrategyProtocol,
        param_bounds: Dict[str, Tuple[float, float]],
        optimization_method: str = "grid",
        grid_points: int = 10,
        max_iterations: int = 100,
        parallel: bool = False,
        n_jobs: int = -1
    ) -> WalkForwardSummary:
        """
        Run complete walk-forward analysis.
        
        Parameters
        ----------
        strategy : StrategyProtocol
            Strategy to optimize.
        param_bounds : Dict[str, Tuple[float, float]]
            Parameter bounds.
        optimization_method : str
            Optimization method.
        grid_points : int
            Grid points for grid search.
        max_iterations : int
            Max iterations for iterative methods.
        parallel : bool
            Run windows in parallel (requires picklable strategy).
        n_jobs : int
            Number of parallel jobs (-1 for all CPUs).
            
        Returns
        -------
        WalkForwardSummary
            Complete walk-forward results.
        """
        windows = self.get_windows()
        results: List[WindowResult] = []
        
        if parallel and n_jobs != 1:
            # Parallel execution (strategy must be picklable)
            warnings.warn("Parallel walk-forward not fully tested - using sequential")
            parallel = False
        
        if not parallel:
            for window in windows:
                result = self.optimize_window(
                    window, strategy, param_bounds,
                    optimization_method, grid_points, max_iterations
                )
                results.append(result)
        
        return self._summarize_results(results)
    
    def _summarize_results(self, results: List[WindowResult]) -> WalkForwardSummary:
        """Generate summary from window results."""
        n_windows = len(results)
        
        train_sharpes = np.array([r.train_sharpe for r in results])
        test_sharpes = np.array([r.test_sharpe for r in results])
        
        # Sharpe degradation analysis
        degradations = []
        for r in results:
            if r.train_sharpe != 0:
                degradations.append(r.test_sharpe / r.train_sharpe)
            else:
                degradations.append(0.0)
        degradations = np.array(degradations)
        
        # Combine all OOS returns
        combined_oos = np.concatenate([r.test_returns for r in results])
        combined_oos_sharpe = _calculate_sharpe(combined_oos, annualization=self.annualization)
        combined_oos_return = float(np.prod(1 + combined_oos) - 1)
        
        # Statistical test on OOS returns
        if HAS_SCIPY and len(combined_oos) > 30:
            t_stat, p_value = stats.ttest_1samp(combined_oos, 0)
            # One-sided test (positive returns)
            p_value = p_value / 2 if t_stat > 0 else 1 - p_value / 2
        else:
            t_stat = np.mean(combined_oos) / (np.std(combined_oos, ddof=1) / np.sqrt(len(combined_oos)))
            p_value = 0.5  # Unable to compute exact p-value
        
        # Robustness assessment
        mean_degradation = float(np.mean(degradations))
        is_robust = (
            mean_degradation >= 0.8 and  # OOS within 20% of IS
            combined_oos_sharpe > 0.5 and  # Positive OOS Sharpe
            p_value < 0.10  # Statistically significant
        )
        
        # Calculate robustness score (0-100)
        degradation_score = min(100, max(0, mean_degradation * 100))
        sharpe_score = min(100, max(0, combined_oos_sharpe * 50))
        significance_score = max(0, (1 - p_value) * 100)
        robustness_score = (degradation_score * 0.4 + sharpe_score * 0.4 + significance_score * 0.2)
        
        # Generate warnings
        warnings_list = self._generate_warnings(
            mean_degradation, combined_oos_sharpe, p_value, n_windows
        )
        
        return WalkForwardSummary(
            n_windows=n_windows,
            window_results=results,
            mean_train_sharpe=float(np.mean(train_sharpes)),
            mean_test_sharpe=float(np.mean(test_sharpes)),
            std_test_sharpe=float(np.std(test_sharpes, ddof=1)),
            mean_sharpe_degradation=mean_degradation,
            degradation_consistency=float(np.std(degradations, ddof=1)),
            combined_oos_returns=combined_oos,
            combined_oos_sharpe=combined_oos_sharpe,
            combined_oos_total_return=combined_oos_return,
            oos_t_statistic=float(t_stat),
            oos_p_value=float(p_value),
            is_robust=is_robust,
            robustness_score=float(robustness_score),
            warnings=warnings_list
        )
    
    def _generate_warnings(
        self,
        mean_degradation: float,
        oos_sharpe: float,
        p_value: float,
        n_windows: int
    ) -> List[str]:
        """Generate human-readable warnings."""
        warnings_list = []
        
        if mean_degradation < 0.5:
            warnings_list.append(
                f"CRITICAL: Severe Sharpe degradation ({mean_degradation:.0%}) - "
                "strategy likely overfit to training data"
            )
        elif mean_degradation < 0.8:
            warnings_list.append(
                f"WARNING: Moderate Sharpe degradation ({mean_degradation:.0%}) - "
                "some overfitting may be present"
            )
        
        if oos_sharpe < 0:
            warnings_list.append(
                f"CRITICAL: Negative OOS Sharpe ({oos_sharpe:.2f}) - "
                "strategy loses money out-of-sample"
            )
        elif oos_sharpe < 0.5:
            warnings_list.append(
                f"WARNING: Low OOS Sharpe ({oos_sharpe:.2f}) - "
                "weak out-of-sample performance"
            )
        
        if p_value > 0.10:
            warnings_list.append(
                f"WARNING: OOS returns not significant (p={p_value:.3f}) - "
                "cannot rule out luck"
            )
        
        if n_windows < 5:
            warnings_list.append(
                f"CAUTION: Only {n_windows} windows - "
                "recommend 5+ for reliable inference"
            )
        
        return warnings_list


# =============================================================================
# AnchoredWalkForward: Expanding Window
# =============================================================================

class AnchoredWalkForward(WalkForwardEngine):
    """
    Anchored (expanding window) walk-forward analysis.
    
    In anchored walk-forward, the training window always starts from the
    beginning of the data and expands forward. This uses all available
    history for training, which may be beneficial for strategies that
    benefit from more data.
    
    Example with 5 windows over 1000 data points (train_ratio=0.7):
    
    Window 1: Train [0-140],    Test [140-200]
    Window 2: Train [0-340],    Test [340-400]
    Window 3: Train [0-540],    Test [540-600]
    Window 4: Train [0-740],    Test [740-800]
    Window 5: Train [0-940],    Test [940-1000]
    
    Pros:
    - Uses maximum available data for training
    - More stable parameter estimates
    
    Cons:
    - Early data may not be representative
    - Training time increases with each window
    
    Reference: Pardo (2008), Chapter 10.2
    """
    
    def generate_windows(self) -> List[WindowSpec]:
        """Generate anchored walk-forward windows."""
        n = len(self.data)
        windows = []
        
        # Calculate window size for test periods
        total_test = int(n * (1 - self.train_ratio))
        test_size = total_test // self.n_windows
        
        if test_size < 10:
            raise ValueError(
                f"Test window size ({test_size}) too small. "
                "Increase data or reduce n_windows."
            )
        
        for i in range(self.n_windows):
            # Test period
            test_start = int(n * self.train_ratio) + i * test_size
            test_end = min(test_start + test_size, n)
            
            if test_end > n:
                break
            
            # Training period (anchored at start)
            train_start = 0
            train_end = test_start - self.purge_gap
            
            if train_end <= train_start + 20:
                continue
            
            windows.append(WindowSpec(
                window_id=i,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end
            ))
        
        return windows


# =============================================================================
# RollingWalkForward: Fixed Window
# =============================================================================

class RollingWalkForward(WalkForwardEngine):
    """
    Rolling (fixed window) walk-forward analysis.
    
    In rolling walk-forward, both training and test windows move forward
    with a fixed size. This captures regime changes better as old data
    is eventually dropped.
    
    Example with 5 windows over 1000 data points (train_ratio=0.7):
    
    Window 1: Train [0-140],     Test [140-200]
    Window 2: Train [160-360],   Test [360-420]
    Window 3: Train [380-580],   Test [580-640]
    Window 4: Train [600-800],   Test [800-860]
    Window 5: Train [820-1020],  Test [would exceed data]
    
    Pros:
    - Adapts to regime changes
    - Consistent training size
    
    Cons:
    - Less data for training
    - May miss long-term patterns
    
    Reference: Pardo (2008), Chapter 10.3
    """
    
    def __init__(
        self,
        data: np.ndarray,
        train_size: Optional[int] = None,
        test_size: Optional[int] = None,
        step_size: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize RollingWalkForward.
        
        Parameters
        ----------
        data : np.ndarray
            Historical data.
        train_size : Optional[int]
            Fixed training window size. If None, calculated from train_ratio.
        test_size : Optional[int]
            Fixed test window size. If None, calculated from train_ratio.
        step_size : Optional[int]
            Step between windows. If None, equals test_size (non-overlapping).
        **kwargs
            Additional arguments passed to WalkForwardEngine.
        """
        super().__init__(data, **kwargs)
        
        n = len(data)
        
        if train_size is None:
            self.train_size = int(n * self.train_ratio / self.n_windows)
        else:
            self.train_size = train_size
        
        if test_size is None:
            self.test_size = int(n * (1 - self.train_ratio) / self.n_windows)
        else:
            self.test_size = test_size
        
        self.step_size = step_size if step_size is not None else self.test_size
    
    def generate_windows(self) -> List[WindowSpec]:
        """Generate rolling walk-forward windows."""
        n = len(self.data)
        windows = []
        
        window_id = 0
        offset = 0
        
        while True:
            train_start = offset
            train_end = train_start + self.train_size
            
            test_start = train_end + self.purge_gap
            test_end = test_start + self.test_size
            
            # Check if window fits
            if test_end > n:
                break
            
            windows.append(WindowSpec(
                window_id=window_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end
            ))
            
            window_id += 1
            offset += self.step_size + self.embargo_gap
            
            if window_id >= self.n_windows:
                break
        
        return windows


# =============================================================================
# DegradationAnalyzer: Assess Strategy Robustness
# =============================================================================

class DegradationAnalyzer:
    """
    Analyze performance degradation between in-sample and out-of-sample.
    
    Key insight: Overfit strategies show significant performance degradation
    from training to testing. This class quantifies that degradation and
    provides statistical tests for robustness.
    
    Metrics:
    - Sharpe degradation ratio: OOS_Sharpe / IS_Sharpe
    - Return degradation ratio: OOS_Return / IS_Return  
    - Rank correlation: Spearman correlation of IS vs OOS rankings
    - Degradation p-value: Statistical significance of degradation
    
    Reference: Bailey et al. (2017), "Stock Portfolio Design and Backtest Overfitting"
    """
    
    def __init__(self, annualization: float = 252.0):
        """
        Initialize DegradationAnalyzer.
        
        Parameters
        ----------
        annualization : float
            Annualization factor for metrics.
        """
        self.annualization = annualization
    
    def analyze(
        self,
        summary: WalkForwardSummary,
        significance_level: float = 0.05
    ) -> Dict[str, Any]:
        """
        Analyze degradation from walk-forward results.
        
        Parameters
        ----------
        summary : WalkForwardSummary
            Results from walk-forward analysis.
        significance_level : float
            Alpha for statistical tests.
            
        Returns
        -------
        Dict[str, Any]
            Comprehensive degradation analysis.
        """
        results = summary.window_results
        n_windows = len(results)
        
        # Extract metrics
        is_sharpes = np.array([r.train_sharpe for r in results])
        oos_sharpes = np.array([r.test_sharpe for r in results])
        is_returns = np.array([r.train_total_return for r in results])
        oos_returns = np.array([r.test_total_return for r in results])
        
        # Degradation ratios
        sharpe_degradation = oos_sharpes / np.where(is_sharpes != 0, is_sharpes, 1e-10)
        return_degradation = oos_returns / np.where(is_returns != 0, is_returns, 1e-10)
        
        # Statistical tests
        analysis = {
            "n_windows": n_windows,
            "sharpe_analysis": self._analyze_sharpe_degradation(is_sharpes, oos_sharpes),
            "return_analysis": self._analyze_return_degradation(is_returns, oos_returns),
            "consistency_analysis": self._analyze_consistency(results),
            "parameter_stability": self._analyze_parameter_stability(results),
            "overall_assessment": self._overall_assessment(summary, significance_level)
        }
        
        return analysis
    
    def _analyze_sharpe_degradation(
        self,
        is_sharpes: np.ndarray,
        oos_sharpes: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze Sharpe ratio degradation."""
        degradation_ratios = oos_sharpes / np.where(is_sharpes != 0, is_sharpes, 1e-10)
        
        # Paired t-test for degradation
        if HAS_SCIPY and len(is_sharpes) > 2:
            t_stat, p_value = stats.ttest_rel(oos_sharpes, is_sharpes)
        else:
            diff = oos_sharpes - is_sharpes
            t_stat = np.mean(diff) / (np.std(diff, ddof=1) / np.sqrt(len(diff)))
            p_value = 0.5
        
        # Rank correlation
        if HAS_SCIPY and len(is_sharpes) > 2:
            rank_corr, rank_p = stats.spearmanr(is_sharpes, oos_sharpes)
        else:
            rank_corr, rank_p = 0.0, 1.0
        
        return {
            "mean_is_sharpe": float(np.mean(is_sharpes)),
            "mean_oos_sharpe": float(np.mean(oos_sharpes)),
            "mean_degradation_ratio": float(np.mean(degradation_ratios)),
            "std_degradation_ratio": float(np.std(degradation_ratios, ddof=1)),
            "degradation_t_stat": float(t_stat),
            "degradation_p_value": float(p_value),
            "rank_correlation": float(rank_corr),
            "rank_correlation_p": float(rank_p),
            "is_significantly_degraded": p_value < 0.05 and np.mean(degradation_ratios) < 0.8
        }
    
    def _analyze_return_degradation(
        self,
        is_returns: np.ndarray,
        oos_returns: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze return degradation."""
        # Avoid division by zero
        safe_is = np.where(np.abs(is_returns) > 1e-10, is_returns, 1e-10)
        degradation_ratios = oos_returns / safe_is
        
        return {
            "mean_is_return": float(np.mean(is_returns)),
            "mean_oos_return": float(np.mean(oos_returns)),
            "mean_degradation_ratio": float(np.mean(degradation_ratios)),
            "oos_positive_pct": float(np.mean(oos_returns > 0) * 100),
            "sign_consistency": float(np.mean(np.sign(is_returns) == np.sign(oos_returns)) * 100)
        }
    
    def _analyze_consistency(
        self,
        results: List[WindowResult]
    ) -> Dict[str, Any]:
        """Analyze consistency across windows."""
        oos_sharpes = np.array([r.test_sharpe for r in results])
        
        return {
            "oos_sharpe_std": float(np.std(oos_sharpes, ddof=1)),
            "oos_sharpe_cv": float(np.std(oos_sharpes, ddof=1) / (np.mean(oos_sharpes) + 1e-10)),
            "min_oos_sharpe": float(np.min(oos_sharpes)),
            "max_oos_sharpe": float(np.max(oos_sharpes)),
            "pct_positive_windows": float(np.mean(oos_sharpes > 0) * 100),
            "pct_sharpe_above_1": float(np.mean(oos_sharpes > 1.0) * 100)
        }
    
    def _analyze_parameter_stability(
        self,
        results: List[WindowResult]
    ) -> Dict[str, Any]:
        """Analyze stability of optimal parameters across windows."""
        if len(results) < 2:
            return {"stability": "insufficient_windows"}
        
        # Extract parameter names from first result
        param_names = list(results[0].optimal_params.keys())
        
        stability = {}
        for name in param_names:
            values = [r.optimal_params.get(name, np.nan) for r in results]
            values = [v for v in values if np.isfinite(v)]
            
            if len(values) > 0:
                stability[name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)),
                    "cv": float(np.std(values, ddof=1) / (np.abs(np.mean(values)) + 1e-10)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values))
                }
        
        # Overall stability score
        cvs = [stability[name]["cv"] for name in stability if "cv" in stability[name]]
        mean_cv = np.mean(cvs) if cvs else 0
        
        stability["overall_stability_score"] = float(max(0, 100 * (1 - mean_cv)))
        
        return stability
    
    def _overall_assessment(
        self,
        summary: WalkForwardSummary,
        significance_level: float
    ) -> Dict[str, Any]:
        """Generate overall robustness assessment."""
        # Key metrics for assessment
        mean_degradation = summary.mean_sharpe_degradation
        oos_sharpe = summary.combined_oos_sharpe
        p_value = summary.oos_p_value
        consistency = 1 - summary.degradation_consistency
        
        # Score components (0-100)
        degradation_score = max(0, min(100, mean_degradation * 100))
        sharpe_score = max(0, min(100, oos_sharpe * 50))
        significance_score = max(0, min(100, (1 - p_value) * 100))
        consistency_score = max(0, min(100, consistency * 100))
        
        overall_score = (
            degradation_score * 0.30 +
            sharpe_score * 0.30 +
            significance_score * 0.20 +
            consistency_score * 0.20
        )
        
        # Determine grade
        if overall_score >= 80:
            grade = "A"
            recommendation = "Strategy appears robust - suitable for live trading"
        elif overall_score >= 65:
            grade = "B"
            recommendation = "Strategy shows promise - consider paper trading first"
        elif overall_score >= 50:
            grade = "C"
            recommendation = "Strategy has weaknesses - significant refinement needed"
        else:
            grade = "F"
            recommendation = "Strategy likely overfit - do not deploy"
        
        return {
            "overall_score": float(overall_score),
            "grade": grade,
            "recommendation": recommendation,
            "component_scores": {
                "degradation": float(degradation_score),
                "oos_sharpe": float(sharpe_score),
                "significance": float(significance_score),
                "consistency": float(consistency_score)
            },
            "is_deployment_ready": overall_score >= 65 and oos_sharpe > 0.5 and p_value < significance_level
        }
    
    def generate_report(
        self,
        summary: WalkForwardSummary,
        analysis: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate human-readable degradation report.
        
        Parameters
        ----------
        summary : WalkForwardSummary
            Walk-forward results.
        analysis : Optional[Dict[str, Any]]
            Pre-computed analysis (calls analyze() if None).
            
        Returns
        -------
        str
            Formatted report.
        """
        if analysis is None:
            analysis = self.analyze(summary)
        
        lines = [
            "=" * 60,
            "WALK-FORWARD DEGRADATION ANALYSIS REPORT",
            "=" * 60,
            "",
            f"Windows Analyzed: {analysis['n_windows']}",
            "",
            "-" * 40,
            "SHARPE RATIO ANALYSIS",
            "-" * 40,
            f"  Mean IS Sharpe:        {analysis['sharpe_analysis']['mean_is_sharpe']:>8.3f}",
            f"  Mean OOS Sharpe:       {analysis['sharpe_analysis']['mean_oos_sharpe']:>8.3f}",
            f"  Mean Degradation:      {analysis['sharpe_analysis']['mean_degradation_ratio']:>8.1%}",
            f"  IS-OOS Rank Corr:      {analysis['sharpe_analysis']['rank_correlation']:>8.3f}",
            "",
            "-" * 40,
            "RETURN ANALYSIS",
            "-" * 40,
            f"  Mean IS Return:        {analysis['return_analysis']['mean_is_return']:>8.2%}",
            f"  Mean OOS Return:       {analysis['return_analysis']['mean_oos_return']:>8.2%}",
            f"  OOS Positive Windows:  {analysis['return_analysis']['oos_positive_pct']:>8.1f}%",
            "",
            "-" * 40,
            "CONSISTENCY ANALYSIS",
            "-" * 40,
            f"  OOS Sharpe Std:        {analysis['consistency_analysis']['oos_sharpe_std']:>8.3f}",
            f"  % Windows Sharpe > 0:  {analysis['consistency_analysis']['pct_positive_windows']:>8.1f}%",
            f"  % Windows Sharpe > 1:  {analysis['consistency_analysis']['pct_sharpe_above_1']:>8.1f}%",
            "",
            "-" * 40,
            "OVERALL ASSESSMENT",
            "-" * 40,
            f"  Overall Score:         {analysis['overall_assessment']['overall_score']:>8.1f}/100",
            f"  Grade:                 {analysis['overall_assessment']['grade']:>8}",
            f"  Deployment Ready:      {'YES' if analysis['overall_assessment']['is_deployment_ready'] else 'NO':>8}",
            "",
            f"  Recommendation: {analysis['overall_assessment']['recommendation']}",
            "",
        ]
        
        if summary.warnings:
            lines.extend([
                "-" * 40,
                "WARNINGS",
                "-" * 40,
            ])
            for warning in summary.warnings:
                lines.append(f"  • {warning}")
            lines.append("")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_walk_forward(
    data: np.ndarray,
    strategy: StrategyProtocol,
    param_bounds: Dict[str, Tuple[float, float]],
    mode: str = "anchored",
    n_windows: int = 5,
    train_ratio: float = 0.7,
    objective: OptimizationObjective = OptimizationObjective.SHARPE,
    optimization_method: str = "grid",
    grid_points: int = 10
) -> Tuple[WalkForwardSummary, Dict[str, Any]]:
    """
    Quick walk-forward analysis with default settings.
    
    Parameters
    ----------
    data : np.ndarray
        Historical data.
    strategy : StrategyProtocol
        Strategy to optimize.
    param_bounds : Dict[str, Tuple[float, float]]
        Parameter bounds.
    mode : str
        'anchored' or 'rolling'.
    n_windows : int
        Number of windows.
    train_ratio : float
        Training fraction.
    objective : OptimizationObjective
        Optimization objective.
    optimization_method : str
        Optimization method.
    grid_points : int
        Grid points for grid search.
        
    Returns
    -------
    Tuple[WalkForwardSummary, Dict[str, Any]]
        (summary, degradation_analysis)
        
    Examples
    --------
    >>> class MyStrategy:
    ...     def set_parameters(self, params): self.threshold = params['threshold']
    ...     def run(self, data): return np.sign(data - self.threshold) * 0.01
    ...     def get_parameters(self): return {'threshold': self.threshold}
    >>> 
    >>> data = np.random.randn(1000).cumsum()
    >>> strategy = MyStrategy()
    >>> summary, analysis = quick_walk_forward(
    ...     data, strategy, {'threshold': (-1, 1)}, n_windows=5
    ... )
    >>> print(f"OOS Sharpe: {summary.combined_oos_sharpe:.2f}")
    """
    if mode == "anchored":
        engine = AnchoredWalkForward(
            data, train_ratio=train_ratio, n_windows=n_windows, objective=objective
        )
    elif mode == "rolling":
        engine = RollingWalkForward(
            data, train_ratio=train_ratio, n_windows=n_windows, objective=objective
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'anchored' or 'rolling'.")
    
    summary = engine.run(
        strategy, param_bounds,
        optimization_method=optimization_method,
        grid_points=grid_points
    )
    
    analyzer = DegradationAnalyzer()
    analysis = analyzer.analyze(summary)
    
    return summary, analysis


# =============================================================================
# Simple Strategy for Testing
# =============================================================================

class SimpleMovingAverageStrategy:
    """
    Simple moving average crossover strategy for testing.
    
    This is a minimal implementation for demonstrating walk-forward analysis.
    """
    
    def __init__(self):
        self.fast_period = 10
        self.slow_period = 50
    
    def set_parameters(self, params: ParamDict) -> None:
        self.fast_period = int(params.get('fast_period', 10))
        self.slow_period = int(params.get('slow_period', 50))
    
    def get_parameters(self) -> ParamDict:
        return {'fast_period': self.fast_period, 'slow_period': self.slow_period}
    
    def run(self, data: np.ndarray) -> Returns:
        """Generate returns from MA crossover signals."""
        if len(data) < self.slow_period + 1:
            return np.zeros(len(data))
        
        # Calculate moving averages
        fast_ma = np.convolve(data, np.ones(self.fast_period)/self.fast_period, mode='valid')
        slow_ma = np.convolve(data, np.ones(self.slow_period)/self.slow_period, mode='valid')
        
        # Align arrays
        offset = self.slow_period - self.fast_period
        fast_ma = fast_ma[offset:]
        
        min_len = min(len(fast_ma), len(slow_ma))
        fast_ma = fast_ma[:min_len]
        slow_ma = slow_ma[:min_len]
        
        # Generate signals
        signals = np.where(fast_ma > slow_ma, 1, -1)
        
        # Calculate returns (assume data is prices)
        price_returns = np.diff(data[-min_len-1:]) / data[-min_len-1:-1]
        
        # Strategy returns
        strategy_returns = signals[:-1] * price_returns[:-1] if len(price_returns) > 1 else np.array([0.0])
        
        return strategy_returns


if __name__ == "__main__":
    # Demo/test the module
    np.random.seed(42)
    
    print("Walk-Forward Analysis Demo")
    print("=" * 60)
    
    # Generate synthetic price data with trend
    n_points = 1000
    prices = 100 * np.exp(np.cumsum(np.random.randn(n_points) * 0.02 + 0.0001))
    
    # Create strategy
    strategy = SimpleMovingAverageStrategy()
    
    # Define parameter bounds
    param_bounds = {
        'fast_period': (5, 20),
        'slow_period': (20, 100)
    }
    
    # Run anchored walk-forward
    print("\n1. Anchored Walk-Forward Analysis")
    print("-" * 40)
    
    engine = AnchoredWalkForward(
        prices,
        train_ratio=0.7,
        n_windows=5,
        objective=OptimizationObjective.SHARPE
    )
    
    summary = engine.run(
        strategy,
        param_bounds,
        optimization_method="grid",
        grid_points=5
    )
    
    print(f"Windows: {summary.n_windows}")
    print(f"Mean IS Sharpe: {summary.mean_train_sharpe:.3f}")
    print(f"Mean OOS Sharpe: {summary.mean_test_sharpe:.3f}")
    print(f"Combined OOS Sharpe: {summary.combined_oos_sharpe:.3f}")
    print(f"Sharpe Degradation: {summary.mean_sharpe_degradation:.1%}")
    print(f"Is Robust: {summary.is_robust}")
    
    # Generate degradation report
    print("\n2. Degradation Analysis")
    print("-" * 40)
    
    analyzer = DegradationAnalyzer()
    analysis = analyzer.analyze(summary)
    
    print(f"Overall Score: {analysis['overall_assessment']['overall_score']:.1f}/100")
    print(f"Grade: {analysis['overall_assessment']['grade']}")
    print(f"Deployment Ready: {analysis['overall_assessment']['is_deployment_ready']}")
    
    # Print full report
    print("\n")
    print(analyzer.generate_report(summary, analysis))
