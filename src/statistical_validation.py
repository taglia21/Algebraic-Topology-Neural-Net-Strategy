"""
Statistical Validation Suite for Trading Strategy Alpha Verification.

This module provides rigorous statistical testing to validate that strategy
returns represent genuine alpha rather than luck, overfitting, or data-mining bias.

Academic References:
- Bailey, D.H. & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio"
- Harvey, C.R. & Liu, Y. (2015). "Backtesting" - Journal of Portfolio Management
- White, H. (2000). "A Reality Check for Data Snooping" - Econometrica
- Efron, B. & Tibshirani, R. (1993). "An Introduction to the Bootstrap"
- Holm, S. (1979). "A Simple Sequentially Rejective Multiple Test Procedure"

Author: Trading System
Version: 1.0.0
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple, List, Dict, Any
from enum import Enum
import numpy as np
from scipy import stats


# =============================================================================
# Constants and Configuration
# =============================================================================

TRADING_DAYS_PER_YEAR = 252
TRADING_HOURS_PER_DAY = 6.5
DEFAULT_RISK_FREE_RATE = 0.0
BOOTSTRAP_DEFAULT_ITERATIONS = 10000
CONFIDENCE_LEVEL_DEFAULT = 0.95


class ReturnFrequency(Enum):
    """Supported return frequencies with annualization factors."""
    DAILY = TRADING_DAYS_PER_YEAR
    HOURLY = TRADING_DAYS_PER_YEAR * TRADING_HOURS_PER_DAY
    MINUTE = TRADING_DAYS_PER_YEAR * TRADING_HOURS_PER_DAY * 60
    TICK = TRADING_DAYS_PER_YEAR * TRADING_HOURS_PER_DAY * 60 * 10  # ~10 ticks/min


@dataclass
class ValidationResult:
    """Container for statistical validation results."""
    statistic: float
    p_value: float
    is_significant: bool
    confidence_level: float
    method: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Complete set of performance metrics for reporting."""
    # Returns
    total_return: float
    annualized_return: float
    
    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Risk
    volatility: float
    downside_deviation: float
    max_drawdown: float
    max_drawdown_duration: int
    
    # Statistical
    t_statistic: float
    p_value: float
    skewness: float
    kurtosis: float
    
    # Confidence intervals
    sharpe_ci_lower: float
    sharpe_ci_upper: float
    
    # Metadata
    n_observations: int
    frequency: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None


# =============================================================================
# Helper Functions
# =============================================================================

def _validate_returns(returns: np.ndarray, min_observations: int = 30) -> np.ndarray:
    """
    Validate and clean return series.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of returns to validate.
    min_observations : int
        Minimum number of valid observations required.
        
    Returns
    -------
    np.ndarray
        Cleaned array with NaN/Inf removed.
        
    Raises
    ------
    ValueError
        If insufficient valid observations after cleaning.
    """
    returns = np.asarray(returns, dtype=np.float64)
    
    # Remove NaN and Inf values
    valid_mask = np.isfinite(returns)
    clean_returns = returns[valid_mask]
    
    n_removed = len(returns) - len(clean_returns)
    if n_removed > 0:
        warnings.warn(
            f"Removed {n_removed} NaN/Inf values from returns series.",
            UserWarning
        )
    
    if len(clean_returns) < min_observations:
        raise ValueError(
            f"Insufficient observations: {len(clean_returns)} < {min_observations}. "
            "Statistical tests require adequate sample size for reliable inference."
        )
    
    return clean_returns


def _annualization_factor(frequency: ReturnFrequency) -> float:
    """Get annualization factor for given return frequency."""
    return float(frequency.value)


# =============================================================================
# TStatCalculator: Statistical Significance Testing
# =============================================================================

class TStatCalculator:
    """
    Calculate t-statistics for strategy returns to test alpha significance.
    
    This class implements hypothesis testing for strategy returns using
    Student's t-distribution. The null hypothesis is that true mean return
    equals zero (or a specified benchmark).
    
    Methodology Reference:
    - Campbell, J.Y., Lo, A.W., & MacKinlay, A.C. (1997). 
      "The Econometrics of Financial Markets" - Princeton University Press
    
    Attributes
    ----------
    frequency : ReturnFrequency
        Frequency of return observations.
    risk_free_rate : float
        Annualized risk-free rate for excess return calculation.
        
    Examples
    --------
    >>> calc = TStatCalculator(frequency=ReturnFrequency.DAILY)
    >>> returns = np.random.randn(252) * 0.01 + 0.0003  # ~7.5% annual
    >>> result = calc.test_mean_return(returns)
    >>> print(f"t-stat: {result.statistic:.2f}, p-value: {result.p_value:.4f}")
    """
    
    def __init__(
        self,
        frequency: ReturnFrequency = ReturnFrequency.DAILY,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE
    ):
        """
        Initialize TStatCalculator.
        
        Parameters
        ----------
        frequency : ReturnFrequency
            Frequency of return observations (daily, hourly, minute).
        risk_free_rate : float
            Annualized risk-free rate (e.g., 0.02 for 2%).
        """
        self.frequency = frequency
        self.risk_free_rate = risk_free_rate
        self._period_rf = risk_free_rate / _annualization_factor(frequency)
    
    def test_mean_return(
        self,
        returns: np.ndarray,
        null_hypothesis: float = 0.0,
        alternative: Literal["two-sided", "greater", "less"] = "greater",
        confidence_level: float = CONFIDENCE_LEVEL_DEFAULT
    ) -> ValidationResult:
        """
        Test if mean return is significantly different from null hypothesis.
        
        Uses one-sample t-test matching scipy.stats.ttest_1samp behavior.
        
        Parameters
        ----------
        returns : np.ndarray
            Array of period returns.
        null_hypothesis : float
            Expected return under null (typically 0).
        alternative : str
            Alternative hypothesis: 'two-sided', 'greater', or 'less'.
        confidence_level : float
            Confidence level for significance testing.
            
        Returns
        -------
        ValidationResult
            Contains t-statistic, p-value, and significance determination.
            
        Notes
        -----
        For trading strategies, 'greater' is typically appropriate as we
        test H0: μ ≤ 0 vs H1: μ > 0 (strategy generates positive returns).
        
        Reference: Campbell et al. (1997), Chapter 4
        """
        clean_returns = _validate_returns(returns)
        
        # Use scipy's implementation for accuracy
        t_stat, p_value_two_sided = stats.ttest_1samp(clean_returns, null_hypothesis)
        
        # Adjust p-value for alternative hypothesis
        if alternative == "two-sided":
            p_value = p_value_two_sided
        elif alternative == "greater":
            p_value = p_value_two_sided / 2 if t_stat > 0 else 1 - p_value_two_sided / 2
        else:  # less
            p_value = p_value_two_sided / 2 if t_stat < 0 else 1 - p_value_two_sided / 2
        
        alpha = 1 - confidence_level
        is_significant = p_value < alpha
        
        # Calculate additional statistics
        mean_return = np.mean(clean_returns)
        std_error = stats.sem(clean_returns)
        
        return ValidationResult(
            statistic=float(t_stat),
            p_value=float(p_value),
            is_significant=is_significant,
            confidence_level=confidence_level,
            method="one-sample t-test",
            details={
                "mean_return": float(mean_return),
                "std_error": float(std_error),
                "degrees_of_freedom": len(clean_returns) - 1,
                "null_hypothesis": null_hypothesis,
                "alternative": alternative,
                "n_observations": len(clean_returns),
                "annualized_mean": float(mean_return * _annualization_factor(self.frequency))
            }
        )
    
    def test_excess_return(
        self,
        strategy_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        confidence_level: float = CONFIDENCE_LEVEL_DEFAULT
    ) -> ValidationResult:
        """
        Test if strategy excess returns over benchmark are significant.
        
        Implements paired t-test for excess returns (strategy - benchmark).
        
        Parameters
        ----------
        strategy_returns : np.ndarray
            Strategy period returns.
        benchmark_returns : np.ndarray
            Benchmark period returns (must match length).
        confidence_level : float
            Confidence level for testing.
            
        Returns
        -------
        ValidationResult
            Statistical test results for excess returns.
            
        Notes
        -----
        This tests whether alpha (Jensen's alpha) is significantly positive.
        Reference: Jensen, M.C. (1968). "The Performance of Mutual Funds"
        """
        strategy_clean = _validate_returns(strategy_returns)
        benchmark_clean = _validate_returns(benchmark_returns)
        
        min_len = min(len(strategy_clean), len(benchmark_clean))
        excess_returns = strategy_clean[:min_len] - benchmark_clean[:min_len]
        
        return self.test_mean_return(
            excess_returns,
            null_hypothesis=0.0,
            alternative="greater",
            confidence_level=confidence_level
        )
    
    def test_sharpe_ratio(
        self,
        returns: np.ndarray,
        confidence_level: float = CONFIDENCE_LEVEL_DEFAULT
    ) -> ValidationResult:
        """
        Test if Sharpe ratio is significantly different from zero.
        
        Uses Lo (2002) asymptotic distribution for Sharpe ratio standard error.
        
        Parameters
        ----------
        returns : np.ndarray
            Array of period returns.
        confidence_level : float
            Confidence level for testing.
            
        Returns
        -------
        ValidationResult
            t-test results for Sharpe ratio significance.
            
        Notes
        -----
        Reference: Lo, A.W. (2002). "The Statistics of Sharpe Ratios"
        Financial Analysts Journal, 58(4), 36-52
        """
        clean_returns = _validate_returns(returns)
        
        # Calculate Sharpe ratio components
        excess_returns = clean_returns - self._period_rf
        mean_excess = np.mean(excess_returns)
        std_dev = np.std(excess_returns, ddof=1)
        
        if std_dev == 0:
            return ValidationResult(
                statistic=float('nan'),
                p_value=1.0,
                is_significant=False,
                confidence_level=confidence_level,
                method="Lo (2002) Sharpe t-test",
                details={"error": "Zero volatility - cannot compute Sharpe ratio"}
            )
        
        sharpe = mean_excess / std_dev
        n = len(clean_returns)
        
        # Lo (2002) standard error formula
        skew = stats.skew(excess_returns)
        kurt = stats.kurtosis(excess_returns)
        
        # SE(SR) = sqrt((1 + 0.5*SR^2 - skew*SR + (kurt/4)*SR^2) / n)
        se_sharpe = np.sqrt(
            (1 + 0.5 * sharpe**2 - skew * sharpe + (kurt / 4) * sharpe**2) / n
        )
        
        t_stat = sharpe / se_sharpe if se_sharpe > 0 else float('nan')
        
        # Two-sided p-value from t-distribution
        if np.isfinite(t_stat):
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
        else:
            p_value = 1.0
        
        alpha = 1 - confidence_level
        
        # Annualize Sharpe ratio
        ann_factor = np.sqrt(_annualization_factor(self.frequency))
        annualized_sharpe = sharpe * ann_factor
        
        return ValidationResult(
            statistic=float(t_stat),
            p_value=float(p_value),
            is_significant=p_value < alpha,
            confidence_level=confidence_level,
            method="Lo (2002) Sharpe t-test",
            details={
                "sharpe_ratio": float(sharpe),
                "annualized_sharpe": float(annualized_sharpe),
                "standard_error": float(se_sharpe),
                "skewness": float(skew),
                "kurtosis": float(kurt),
                "n_observations": n
            }
        )


# =============================================================================
# BootstrapAnalyzer: Non-parametric Confidence Intervals
# =============================================================================

class BootstrapAnalyzer:
    """
    Bootstrap analysis for robust confidence interval estimation.
    
    Implements the stationary bootstrap of Politis & Romano (1994) for
    time-series data, and standard bootstrap for i.i.d. assumptions.
    
    Methodology References:
    - Efron, B. & Tibshirani, R. (1993). "An Introduction to the Bootstrap"
    - Politis, D.N. & Romano, J.P. (1994). "The Stationary Bootstrap"
    - Ledoit, O. & Wolf, M. (2008). "Robust Performance Hypothesis Testing"
    
    Attributes
    ----------
    n_iterations : int
        Number of bootstrap replications.
    random_state : Optional[int]
        Seed for reproducibility.
    block_size : Optional[int]
        Block size for stationary bootstrap (auto-selected if None).
    """
    
    def __init__(
        self,
        n_iterations: int = BOOTSTRAP_DEFAULT_ITERATIONS,
        random_state: Optional[int] = None,
        block_size: Optional[int] = None
    ):
        """
        Initialize BootstrapAnalyzer.
        
        Parameters
        ----------
        n_iterations : int
            Number of bootstrap samples to generate.
        random_state : Optional[int]
            Random seed for reproducibility.
        block_size : Optional[int]
            Block size for block bootstrap. If None, uses optimal
            selection rule of Politis & White (2004).
        """
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.block_size = block_size
        self._rng = np.random.default_rng(random_state)
    
    def _get_optimal_block_size(self, n: int) -> int:
        """
        Compute optimal block size using Politis & White (2004) rule.
        
        For AR(1) process, optimal block size ~ n^(1/3).
        """
        if self.block_size is not None:
            return self.block_size
        return max(1, int(np.ceil(n ** (1/3))))
    
    def _stationary_bootstrap_sample(
        self,
        data: np.ndarray,
        block_size: int
    ) -> np.ndarray:
        """
        Generate one stationary bootstrap sample.
        
        Uses geometric distribution for random block lengths,
        which produces stationary resamples.
        """
        n = len(data)
        sample = []
        p = 1.0 / block_size  # Probability of starting new block
        
        idx = self._rng.integers(0, n)
        
        while len(sample) < n:
            sample.append(data[idx])
            
            # With probability p, jump to random index; else continue
            if self._rng.random() < p:
                idx = self._rng.integers(0, n)
            else:
                idx = (idx + 1) % n
        
        return np.array(sample[:n])
    
    def sharpe_ratio_ci(
        self,
        returns: np.ndarray,
        frequency: ReturnFrequency = ReturnFrequency.DAILY,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
        confidence_level: float = CONFIDENCE_LEVEL_DEFAULT,
        use_stationary: bool = True
    ) -> Tuple[float, float, float]:
        """
        Compute bootstrap confidence interval for Sharpe ratio.
        
        Parameters
        ----------
        returns : np.ndarray
            Array of period returns.
        frequency : ReturnFrequency
            Return frequency for annualization.
        risk_free_rate : float
            Annualized risk-free rate.
        confidence_level : float
            Confidence level (e.g., 0.95 for 95% CI).
        use_stationary : bool
            Use stationary bootstrap (True) or standard bootstrap (False).
            
        Returns
        -------
        Tuple[float, float, float]
            (lower_bound, point_estimate, upper_bound) for annualized Sharpe.
            
        Notes
        -----
        For time-series returns, stationary bootstrap is preferred as it
        preserves autocorrelation structure. The percentile method is used
        for confidence intervals as it's robust to non-normality.
        
        Reference: Ledoit & Wolf (2008), Section 3.1
        """
        clean_returns = _validate_returns(returns)
        n = len(clean_returns)
        
        period_rf = risk_free_rate / _annualization_factor(frequency)
        ann_factor = np.sqrt(_annualization_factor(frequency))
        
        # Calculate point estimate
        excess = clean_returns - period_rf
        point_estimate = (np.mean(excess) / np.std(excess, ddof=1)) * ann_factor
        
        # Bootstrap distribution
        block_size = self._get_optimal_block_size(n)
        bootstrap_sharpes = np.zeros(self.n_iterations)
        
        for i in range(self.n_iterations):
            if use_stationary:
                sample = self._stationary_bootstrap_sample(clean_returns, block_size)
            else:
                indices = self._rng.integers(0, n, size=n)
                sample = clean_returns[indices]
            
            sample_excess = sample - period_rf
            std = np.std(sample_excess, ddof=1)
            
            if std > 0:
                bootstrap_sharpes[i] = (np.mean(sample_excess) / std) * ann_factor
            else:
                bootstrap_sharpes[i] = 0.0
        
        # Percentile confidence interval
        alpha = 1 - confidence_level
        lower = np.percentile(bootstrap_sharpes, 100 * alpha / 2)
        upper = np.percentile(bootstrap_sharpes, 100 * (1 - alpha / 2))
        
        return (float(lower), float(point_estimate), float(upper))
    
    def metric_ci(
        self,
        returns: np.ndarray,
        metric_func: callable,
        confidence_level: float = CONFIDENCE_LEVEL_DEFAULT,
        use_stationary: bool = True
    ) -> Tuple[float, float, float]:
        """
        Compute bootstrap CI for arbitrary metric function.
        
        Parameters
        ----------
        returns : np.ndarray
            Array of returns.
        metric_func : callable
            Function that takes returns array and returns scalar metric.
        confidence_level : float
            Confidence level for interval.
        use_stationary : bool
            Use stationary bootstrap for time-series.
            
        Returns
        -------
        Tuple[float, float, float]
            (lower, point_estimate, upper) confidence interval.
        """
        clean_returns = _validate_returns(returns)
        n = len(clean_returns)
        
        point_estimate = metric_func(clean_returns)
        block_size = self._get_optimal_block_size(n)
        
        bootstrap_metrics = np.zeros(self.n_iterations)
        
        for i in range(self.n_iterations):
            if use_stationary:
                sample = self._stationary_bootstrap_sample(clean_returns, block_size)
            else:
                indices = self._rng.integers(0, n, size=n)
                sample = clean_returns[indices]
            
            try:
                bootstrap_metrics[i] = metric_func(sample)
            except Exception:
                bootstrap_metrics[i] = np.nan
        
        # Remove failed iterations
        valid_metrics = bootstrap_metrics[np.isfinite(bootstrap_metrics)]
        
        if len(valid_metrics) < 100:
            warnings.warn("Few valid bootstrap samples - CI may be unreliable.")
        
        alpha = 1 - confidence_level
        lower = np.percentile(valid_metrics, 100 * alpha / 2)
        upper = np.percentile(valid_metrics, 100 * (1 - alpha / 2))
        
        return (float(lower), float(point_estimate), float(upper))
    
    def bootstrap_p_value(
        self,
        returns: np.ndarray,
        statistic_func: callable,
        null_value: float = 0.0,
        alternative: Literal["two-sided", "greater", "less"] = "two-sided"
    ) -> float:
        """
        Compute bootstrap p-value for hypothesis test.
        
        Parameters
        ----------
        returns : np.ndarray
            Observed returns.
        statistic_func : callable
            Function computing test statistic from returns.
        null_value : float
            Value of statistic under null hypothesis.
        alternative : str
            Type of alternative hypothesis.
            
        Returns
        -------
        float
            Bootstrap p-value.
            
        Notes
        -----
        This implements the percentile-t bootstrap p-value, which is
        more accurate than basic percentile method for hypothesis testing.
        
        Reference: Efron & Tibshirani (1993), Chapter 16
        """
        clean_returns = _validate_returns(returns)
        n = len(clean_returns)
        
        observed_stat = statistic_func(clean_returns)
        
        # Center returns under null hypothesis
        centered_returns = clean_returns - np.mean(clean_returns) + null_value
        
        block_size = self._get_optimal_block_size(n)
        bootstrap_stats = np.zeros(self.n_iterations)
        
        for i in range(self.n_iterations):
            sample = self._stationary_bootstrap_sample(centered_returns, block_size)
            bootstrap_stats[i] = statistic_func(sample)
        
        if alternative == "two-sided":
            p_value = np.mean(np.abs(bootstrap_stats) >= np.abs(observed_stat))
        elif alternative == "greater":
            p_value = np.mean(bootstrap_stats >= observed_stat)
        else:  # less
            p_value = np.mean(bootstrap_stats <= observed_stat)
        
        return float(p_value)


# =============================================================================
# MultipleTestingCorrector: Control Family-Wise Error Rate
# =============================================================================

class MultipleTestingCorrector:
    """
    Correct for multiple hypothesis testing to control false discovery.
    
    When testing many strategies or parameters, the probability of false
    positives increases dramatically. This class implements corrections
    to maintain statistical validity.
    
    Methodology References:
    - Bonferroni, C.E. (1936). "Teoria statistica delle classi e calcolo"
    - Holm, S. (1979). "A Simple Sequentially Rejective Multiple Test Procedure"
    - Benjamini, Y. & Hochberg, Y. (1995). "Controlling the False Discovery Rate"
    - Harvey, C.R., Liu, Y. & Zhu, H. (2016). "...and the Cross-Section of Expected Returns"
    
    Examples
    --------
    >>> corrector = MultipleTestingCorrector()
    >>> p_values = [0.01, 0.03, 0.08, 0.15, 0.45]
    >>> adjusted = corrector.holm_correction(p_values)
    >>> print(adjusted)
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize corrector.
        
        Parameters
        ----------
        alpha : float
            Family-wise error rate to control.
        """
        self.alpha = alpha
    
    def bonferroni_correction(
        self,
        p_values: List[float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Bonferroni correction for multiple comparisons.
        
        The simplest but most conservative correction. Divides alpha
        by the number of tests to ensure FWER ≤ alpha.
        
        Parameters
        ----------
        p_values : List[float]
            Original p-values from multiple tests.
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (adjusted_p_values, reject_null) where adjusted p-values are
            min(p * n_tests, 1.0) and reject_null is boolean array.
            
        Notes
        -----
        Very conservative - may miss true effects (low power).
        Use when false positives are very costly.
        
        Reference: Bonferroni (1936)
        """
        p_values = np.asarray(p_values, dtype=np.float64)
        n_tests = len(p_values)
        
        adjusted = np.minimum(p_values * n_tests, 1.0)
        reject = adjusted < self.alpha
        
        return adjusted, reject
    
    def holm_correction(
        self,
        p_values: List[float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Holm-Bonferroni step-down correction.
        
        More powerful than Bonferroni while still controlling FWER.
        Sorts p-values and applies sequentially decreasing correction.
        
        Parameters
        ----------
        p_values : List[float]
            Original p-values.
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (adjusted_p_values, reject_null) in original order.
            
        Notes
        -----
        Uniformly more powerful than Bonferroni. Recommended default
        for multiple testing correction in financial research.
        
        Reference: Holm (1979), Scandinavian Journal of Statistics
        """
        p_values = np.asarray(p_values, dtype=np.float64)
        n_tests = len(p_values)
        
        # Sort p-values and track original order
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        
        # Apply step-down correction
        adjusted_sorted = np.zeros(n_tests)
        for i in range(n_tests):
            multiplier = n_tests - i
            adjusted_sorted[i] = sorted_p[i] * multiplier
        
        # Enforce monotonicity (cumulative max)
        adjusted_sorted = np.maximum.accumulate(adjusted_sorted)
        adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
        
        # Restore original order
        adjusted = np.zeros(n_tests)
        adjusted[sorted_indices] = adjusted_sorted
        
        reject = adjusted < self.alpha
        
        return adjusted, reject
    
    def benjamini_hochberg(
        self,
        p_values: List[float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Benjamini-Hochberg FDR control procedure.
        
        Controls False Discovery Rate rather than FWER. More powerful
        but allows some proportion of false positives among rejections.
        
        Parameters
        ----------
        p_values : List[float]
            Original p-values.
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (adjusted_p_values, reject_null)
            
        Notes
        -----
        Use when testing many hypotheses and some false positives are
        acceptable. Controls E[# false positives / # rejections] ≤ alpha.
        
        Reference: Benjamini & Hochberg (1995), JRSS-B
        """
        p_values = np.asarray(p_values, dtype=np.float64)
        n_tests = len(p_values)
        
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        
        # BH adjustment: p_adj[i] = min(p[i] * n / (i+1), 1)
        adjusted_sorted = np.zeros(n_tests)
        for i in range(n_tests):
            adjusted_sorted[i] = sorted_p[i] * n_tests / (i + 1)
        
        # Enforce monotonicity (cumulative min from right)
        adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
        adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
        
        # Restore original order
        adjusted = np.zeros(n_tests)
        adjusted[sorted_indices] = adjusted_sorted
        
        reject = adjusted < self.alpha
        
        return adjusted, reject
    
    def haircut_sharpe_ratio(
        self,
        sharpe_ratios: List[float],
        n_strategies_tested: int,
        t_statistics: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Apply "haircut" to Sharpe ratios for multiple testing.
        
        Implements Harvey, Liu & Zhu (2016) adjustment for strategy
        Sharpe ratios to account for data snooping.
        
        Parameters
        ----------
        sharpe_ratios : List[float]
            Reported Sharpe ratios.
        n_strategies_tested : int
            Total number of strategies examined (including unreported).
        t_statistics : Optional[List[float]]
            t-statistics if available; computed from Sharpe if not.
            
        Returns
        -------
        np.ndarray
            Adjusted "haircutted" Sharpe ratios.
            
        Notes
        -----
        Key insight: If you test 100 strategies and report the best,
        a 2.0 Sharpe might only represent a 0.5 "true" Sharpe after
        accounting for selection bias.
        
        Reference: Harvey, Liu & Zhu (2016), Review of Financial Studies
        """
        sharpe_ratios = np.asarray(sharpe_ratios, dtype=np.float64)
        n_reported = len(sharpe_ratios)
        
        # Compute t-stat equivalent if not provided
        # Assuming ~5 years of monthly data (60 observations)
        if t_statistics is None:
            t_statistics = sharpe_ratios * np.sqrt(60)
        else:
            t_statistics = np.asarray(t_statistics, dtype=np.float64)
        
        # Harvey et al. suggest t > 3.0 threshold for believable Sharpe
        # Apply adjustment based on number of tests
        adjustment_factor = np.sqrt(
            2 * np.log(n_strategies_tested) - 
            np.log(np.log(n_strategies_tested)) - 
            np.log(4 * np.pi)
        )
        
        # Haircut = max(0, t - adjustment) / sqrt(60)
        adjusted_t = np.maximum(0, t_statistics - adjustment_factor)
        haircutted_sharpe = adjusted_t / np.sqrt(60)
        
        return haircutted_sharpe
    
    def summary(
        self,
        p_values: List[float],
        labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate summary of all correction methods.
        
        Parameters
        ----------
        p_values : List[float]
            Original p-values to correct.
        labels : Optional[List[str]]
            Labels for each test (e.g., strategy names).
            
        Returns
        -------
        Dict[str, Any]
            Summary with all correction methods compared.
        """
        n_tests = len(p_values)
        labels = labels or [f"Test_{i+1}" for i in range(n_tests)]
        
        bonf_adj, bonf_reject = self.bonferroni_correction(p_values)
        holm_adj, holm_reject = self.holm_correction(p_values)
        bh_adj, bh_reject = self.benjamini_hochberg(p_values)
        
        return {
            "n_tests": n_tests,
            "alpha": self.alpha,
            "labels": labels,
            "original_p_values": list(p_values),
            "bonferroni": {
                "adjusted": bonf_adj.tolist(),
                "n_rejected": int(np.sum(bonf_reject)),
                "rejected": [labels[i] for i in np.where(bonf_reject)[0]]
            },
            "holm": {
                "adjusted": holm_adj.tolist(),
                "n_rejected": int(np.sum(holm_reject)),
                "rejected": [labels[i] for i in np.where(holm_reject)[0]]
            },
            "benjamini_hochberg": {
                "adjusted": bh_adj.tolist(),
                "n_rejected": int(np.sum(bh_reject)),
                "rejected": [labels[i] for i in np.where(bh_reject)[0]]
            }
        }


# =============================================================================
# OverfitDetector: Cross-Validation for Strategy Robustness
# =============================================================================

class OverfitDetector:
    """
    Detect overfitting in trading strategies using cross-validation.
    
    Implements multiple methods to assess whether backtest performance
    is likely to persist out-of-sample.
    
    Methodology References:
    - Bailey, D.H. et al. (2015). "The Probability of Backtest Overfitting"
    - Prado, M.L. (2018). "Advances in Financial Machine Learning"
    - Bailey, D.H. & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio"
    
    Attributes
    ----------
    n_splits : int
        Number of cross-validation splits.
    random_state : Optional[int]
        Random seed for reproducibility.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        random_state: Optional[int] = None
    ):
        """
        Initialize OverfitDetector.
        
        Parameters
        ----------
        n_splits : int
            Number of time-series cross-validation folds.
        random_state : Optional[int]
            Random seed.
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
    
    def time_series_cv(
        self,
        returns: np.ndarray,
        min_train_size: Optional[int] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate time-series cross-validation splits.
        
        Uses expanding window: each split trains on all data up to that
        point and tests on the next fold.
        
        Parameters
        ----------
        returns : np.ndarray
            Full return series.
        min_train_size : Optional[int]
            Minimum training observations. Default is 2 * test_size.
            
        Returns
        -------
        List[Tuple[np.ndarray, np.ndarray]]
            List of (train_indices, test_indices) for each fold.
            
        Notes
        -----
        Time-series CV respects temporal ordering, unlike standard k-fold.
        Reference: Hyndman & Athanasopoulos (2018), Chapter 3.4
        """
        n = len(returns)
        test_size = n // (self.n_splits + 1)
        
        if min_train_size is None:
            min_train_size = 2 * test_size
        
        splits = []
        
        for i in range(self.n_splits):
            test_start = min_train_size + i * test_size
            test_end = test_start + test_size
            
            if test_end > n:
                break
            
            train_indices = np.arange(0, test_start)
            test_indices = np.arange(test_start, test_end)
            
            splits.append((train_indices, test_indices))
        
        return splits
    
    def combinatorial_purged_cv(
        self,
        returns: np.ndarray,
        n_test_splits: int = 2,
        purge_window: int = 5
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Combinatorial purged cross-validation (CPCV).
        
        Combines multiple test sets and purges observations near
        train/test boundaries to prevent leakage.
        
        Parameters
        ----------
        returns : np.ndarray
            Return series.
        n_test_splits : int
            Number of groups to use for testing in each iteration.
        purge_window : int
            Number of observations to purge at boundaries.
            
        Returns
        -------
        List[Tuple[np.ndarray, np.ndarray]]
            Train/test index pairs with purging applied.
            
        Notes
        -----
        CPCV provides more realistic out-of-sample estimates by
        preventing information leakage across time.
        
        Reference: Prado (2018), Chapter 12
        """
        n = len(returns)
        group_size = n // self.n_splits
        
        # Create group boundaries
        groups = []
        for i in range(self.n_splits):
            start = i * group_size
            end = start + group_size if i < self.n_splits - 1 else n
            groups.append(np.arange(start, end))
        
        # Generate all combinations of test groups
        from itertools import combinations
        
        splits = []
        for test_group_indices in combinations(range(self.n_splits), n_test_splits):
            test_indices = np.concatenate([groups[i] for i in test_group_indices])
            train_group_indices = [i for i in range(self.n_splits) if i not in test_group_indices]
            
            train_indices = []
            for gi in train_group_indices:
                group = groups[gi]
                
                # Purge observations near test boundaries
                purged_group = []
                for idx in group:
                    # Check distance to any test index
                    min_dist = min(abs(idx - ti) for ti in test_indices)
                    if min_dist > purge_window:
                        purged_group.append(idx)
                
                train_indices.extend(purged_group)
            
            if len(train_indices) > 0:
                splits.append((np.array(train_indices), test_indices))
        
        return splits
    
    def probability_of_backtest_overfitting(
        self,
        is_returns: np.ndarray,
        oos_returns: np.ndarray,
        n_strategies: int
    ) -> float:
        """
        Estimate probability that backtest is overfit (PBO).
        
        Compares in-sample vs out-of-sample rank to estimate probability
        that the "best" strategy is actually overfit.
        
        Parameters
        ----------
        is_returns : np.ndarray
            In-sample returns for each strategy (shape: n_strategies x n_is).
        oos_returns : np.ndarray
            Out-of-sample returns (shape: n_strategies x n_oos).
        n_strategies : int
            Number of strategies being compared.
            
        Returns
        -------
        float
            Probability of backtest overfitting (0 to 1).
            
        Notes
        -----
        PBO > 0.5 suggests overfitting is likely.
        PBO approaches 1.0 for severely overfit strategies.
        
        Reference: Bailey et al. (2015), Notices of the AMS
        """
        # Calculate IS and OOS Sharpe ratios
        is_sharpes = np.mean(is_returns, axis=1) / (np.std(is_returns, axis=1, ddof=1) + 1e-10)
        oos_sharpes = np.mean(oos_returns, axis=1) / (np.std(oos_returns, axis=1, ddof=1) + 1e-10)
        
        # Rank strategies (higher is better)
        is_ranks = stats.rankdata(is_sharpes)
        oos_ranks = stats.rankdata(oos_sharpes)
        
        # Find best IS strategy
        best_is_idx = np.argmax(is_sharpes)
        best_is_rank = is_ranks[best_is_idx]
        oos_rank_of_best = oos_ranks[best_is_idx]
        
        # PBO = probability that best IS strategy has below-median OOS rank
        # Estimate via logit of rank ratio
        rank_ratio = oos_rank_of_best / n_strategies
        pbo = 1 - rank_ratio  # Simple estimate
        
        # More sophisticated: compare IS vs OOS rank correlation
        rank_corr, _ = stats.spearmanr(is_ranks, oos_ranks)
        
        # Low or negative correlation suggests overfitting
        if rank_corr < 0:
            pbo = min(1.0, pbo + abs(rank_corr) / 2)
        
        return float(np.clip(pbo, 0, 1))
    
    def deflated_sharpe_ratio(
        self,
        sharpe_ratio: float,
        n_observations: int,
        n_strategies_tested: int,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
        var_sharpe_estimates: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Calculate deflated Sharpe ratio accounting for multiple testing.
        
        Adjusts Sharpe ratio for the number of strategies tested to
        estimate the "true" expected Sharpe.
        
        Parameters
        ----------
        sharpe_ratio : float
            Observed (annualized) Sharpe ratio.
        n_observations : int
            Number of return observations.
        n_strategies_tested : int
            Total strategies tested (including unreported).
        skewness : float
            Return skewness (0 for normal).
        kurtosis : float
            Return kurtosis (3 for normal).
        var_sharpe_estimates : Optional[float]
            Variance of Sharpe estimates if known.
            
        Returns
        -------
        Tuple[float, float]
            (deflated_sharpe, p_value) where p-value tests if deflated > 0.
            
        Notes
        -----
        DSR accounts for:
        1. Non-normal returns (skewness, kurtosis)
        2. Multiple testing (n_strategies_tested)
        3. Sample size (n_observations)
        
        Reference: Bailey & Lopez de Prado (2014), Journal of Portfolio Management
        """
        # Expected max Sharpe under null (all strategies have SR = 0)
        # Using approximation for max of n_strategies_tested standard normals
        if n_strategies_tested > 1:
            expected_max_sr_null = (
                (1 - np.euler_gamma) * stats.norm.ppf(1 - 1/n_strategies_tested) +
                np.euler_gamma * stats.norm.ppf(1 - 1/(n_strategies_tested * np.e))
            )
        else:
            expected_max_sr_null = 0
        
        # Standard error of Sharpe ratio (Lo 2002)
        se_sr = np.sqrt(
            (1 + 0.5 * sharpe_ratio**2 - skewness * sharpe_ratio + 
             ((kurtosis - 3) / 4) * sharpe_ratio**2) / n_observations
        )
        
        # Deflated Sharpe
        deflated_sr = sharpe_ratio - expected_max_sr_null
        
        # P-value: probability of observing SR >= observed under null
        z_score = (sharpe_ratio - expected_max_sr_null) / se_sr if se_sr > 0 else 0
        p_value = 1 - stats.norm.cdf(z_score)
        
        return (float(deflated_sr), float(p_value))
    
    def detect_overfitting(
        self,
        returns: np.ndarray,
        n_strategies_tested: int = 1,
        frequency: ReturnFrequency = ReturnFrequency.DAILY
    ) -> Dict[str, Any]:
        """
        Comprehensive overfitting detection analysis.
        
        Parameters
        ----------
        returns : np.ndarray
            Strategy return series.
        n_strategies_tested : int
            Number of strategies/parameters tested.
        frequency : ReturnFrequency
            Return frequency.
            
        Returns
        -------
        Dict[str, Any]
            Comprehensive overfitting analysis results.
        """
        clean_returns = _validate_returns(returns, min_observations=60)
        n = len(clean_returns)
        
        # Basic statistics
        mean_ret = np.mean(clean_returns)
        std_ret = np.std(clean_returns, ddof=1)
        sharpe = (mean_ret / std_ret) * np.sqrt(_annualization_factor(frequency))
        skew = float(stats.skew(clean_returns))
        kurt = float(stats.kurtosis(clean_returns) + 3)  # Convert excess to regular
        
        # Time-series CV performance
        cv_splits = self.time_series_cv(clean_returns)
        is_sharpes = []
        oos_sharpes = []
        
        for train_idx, test_idx in cv_splits:
            train_ret = clean_returns[train_idx]
            test_ret = clean_returns[test_idx]
            
            is_sr = np.mean(train_ret) / (np.std(train_ret, ddof=1) + 1e-10)
            oos_sr = np.mean(test_ret) / (np.std(test_ret, ddof=1) + 1e-10)
            
            is_sharpes.append(is_sr)
            oos_sharpes.append(oos_sr)
        
        is_sharpes = np.array(is_sharpes)
        oos_sharpes = np.array(oos_sharpes)
        
        # Sharpe ratio degradation (IS vs OOS)
        sr_degradation = np.mean(is_sharpes - oos_sharpes) / (np.std(is_sharpes) + 1e-10)
        
        # Deflated Sharpe
        dsr, dsr_pvalue = self.deflated_sharpe_ratio(
            sharpe, n, n_strategies_tested, skew, kurt
        )
        
        # Overall assessment
        is_overfit = (
            (dsr < 0.5) or 
            (dsr_pvalue > 0.1) or 
            (sr_degradation > 1.0)
        )
        
        return {
            "sharpe_ratio": float(sharpe),
            "deflated_sharpe_ratio": float(dsr),
            "dsr_p_value": float(dsr_pvalue),
            "is_likely_overfit": is_overfit,
            "cv_results": {
                "n_folds": len(cv_splits),
                "mean_is_sharpe": float(np.mean(is_sharpes)),
                "mean_oos_sharpe": float(np.mean(oos_sharpes)),
                "sharpe_degradation": float(sr_degradation),
                "is_oos_correlation": float(np.corrcoef(is_sharpes, oos_sharpes)[0, 1])
            },
            "return_statistics": {
                "n_observations": n,
                "skewness": skew,
                "kurtosis": kurt
            },
            "warnings": self._generate_warnings(dsr, dsr_pvalue, sr_degradation, n, n_strategies_tested)
        }
    
    def _generate_warnings(
        self,
        dsr: float,
        dsr_pvalue: float,
        sr_degradation: float,
        n_obs: int,
        n_strategies: int
    ) -> List[str]:
        """Generate human-readable warnings about potential overfitting."""
        warnings_list = []
        
        if dsr < 0:
            warnings_list.append(
                f"CRITICAL: Deflated Sharpe ({dsr:.2f}) is negative - "
                "performance may be entirely due to multiple testing."
            )
        elif dsr < 0.5:
            warnings_list.append(
                f"WARNING: Deflated Sharpe ({dsr:.2f}) < 0.5 - "
                "weak evidence of genuine alpha."
            )
        
        if dsr_pvalue > 0.1:
            warnings_list.append(
                f"WARNING: DSR p-value ({dsr_pvalue:.3f}) > 0.1 - "
                "cannot reject null of no skill."
            )
        
        if sr_degradation > 1.5:
            warnings_list.append(
                f"WARNING: Sharpe degrades {sr_degradation:.1f} std from IS to OOS - "
                "possible overfitting to training data."
            )
        
        if n_obs < 252:
            warnings_list.append(
                f"CAUTION: Only {n_obs} observations - "
                "insufficient for reliable inference (recommend 252+)."
            )
        
        if n_strategies > 20 and dsr_pvalue > 0.05:
            warnings_list.append(
                f"CAUTION: Tested {n_strategies} strategies - "
                "high probability of spurious discovery."
            )
        
        return warnings_list


# =============================================================================
# PerformanceReporter: Publication-Quality Reports
# =============================================================================

class PerformanceReporter:
    """
    Generate comprehensive, publication-quality performance reports.
    
    Produces all standard metrics used in academic finance papers
    with proper statistical significance testing.
    
    Report Standards:
    - Annualized metrics following CFA Institute standards
    - Statistical tests per Harvey et al. (2016) recommendations
    - Risk metrics per Basel III / SEC standards
    
    Examples
    --------
    >>> reporter = PerformanceReporter()
    >>> metrics = reporter.full_report(returns, frequency=ReturnFrequency.DAILY)
    >>> print(reporter.format_report(metrics))
    """
    
    def __init__(
        self,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
        confidence_level: float = CONFIDENCE_LEVEL_DEFAULT,
        bootstrap_iterations: int = BOOTSTRAP_DEFAULT_ITERATIONS
    ):
        """
        Initialize PerformanceReporter.
        
        Parameters
        ----------
        risk_free_rate : float
            Annualized risk-free rate.
        confidence_level : float
            Confidence level for intervals and tests.
        bootstrap_iterations : int
            Number of bootstrap samples for CI estimation.
        """
        self.risk_free_rate = risk_free_rate
        self.confidence_level = confidence_level
        self.bootstrap_iterations = bootstrap_iterations
        
        self._bootstrap = BootstrapAnalyzer(
            n_iterations=bootstrap_iterations,
            random_state=42
        )
    
    def _calculate_drawdowns(
        self,
        returns: np.ndarray
    ) -> Tuple[float, int, np.ndarray]:
        """
        Calculate maximum drawdown and duration.
        
        Returns
        -------
        Tuple[float, int, np.ndarray]
            (max_drawdown, max_dd_duration, drawdown_series)
        """
        # Cumulative returns (wealth index)
        cum_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cum_returns)
        
        # Drawdown series
        drawdowns = (cum_returns - running_max) / running_max
        max_dd = float(np.min(drawdowns))
        
        # Calculate duration of max drawdown
        underwater = drawdowns < 0
        dd_periods = []
        current_duration = 0
        
        for is_underwater in underwater:
            if is_underwater:
                current_duration += 1
            else:
                if current_duration > 0:
                    dd_periods.append(current_duration)
                current_duration = 0
        
        if current_duration > 0:
            dd_periods.append(current_duration)
        
        max_dd_duration = max(dd_periods) if dd_periods else 0
        
        return max_dd, max_dd_duration, drawdowns
    
    def _sortino_ratio(
        self,
        returns: np.ndarray,
        period_rf: float,
        ann_factor: float
    ) -> float:
        """
        Calculate Sortino ratio (downside risk-adjusted return).
        
        Reference: Sortino & van der Meer (1991)
        """
        excess = returns - period_rf
        mean_excess = np.mean(excess)
        
        # Downside deviation (only negative returns)
        downside = excess[excess < 0]
        if len(downside) == 0:
            return float('inf')
        
        downside_std = np.sqrt(np.mean(downside**2))
        
        if downside_std == 0:
            return float('inf')
        
        return (mean_excess / downside_std) * np.sqrt(ann_factor)
    
    def _calmar_ratio(
        self,
        annualized_return: float,
        max_drawdown: float
    ) -> float:
        """
        Calculate Calmar ratio (return / max drawdown).
        
        Reference: Young (1991)
        """
        if max_drawdown == 0:
            return float('inf') if annualized_return > 0 else 0.0
        
        return annualized_return / abs(max_drawdown)
    
    def full_report(
        self,
        returns: np.ndarray,
        frequency: ReturnFrequency = ReturnFrequency.DAILY,
        benchmark_returns: Optional[np.ndarray] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> PerformanceMetrics:
        """
        Generate comprehensive performance metrics.
        
        Parameters
        ----------
        returns : np.ndarray
            Strategy period returns.
        frequency : ReturnFrequency
            Return observation frequency.
        benchmark_returns : Optional[np.ndarray]
            Benchmark returns for relative metrics.
        start_date : Optional[str]
            Start date for reporting.
        end_date : Optional[str]
            End date for reporting.
            
        Returns
        -------
        PerformanceMetrics
            Complete set of performance statistics.
        """
        clean_returns = _validate_returns(returns, min_observations=20)
        n = len(clean_returns)
        
        ann_factor = _annualization_factor(frequency)
        period_rf = self.risk_free_rate / ann_factor
        
        # Basic return metrics
        total_return = float(np.prod(1 + clean_returns) - 1)
        mean_return = np.mean(clean_returns)
        annualized_return = float((1 + mean_return) ** ann_factor - 1)
        
        # Risk metrics
        volatility = float(np.std(clean_returns, ddof=1) * np.sqrt(ann_factor))
        
        excess = clean_returns - period_rf
        downside_returns = excess[excess < 0]
        downside_deviation = float(
            np.sqrt(np.mean(downside_returns**2)) * np.sqrt(ann_factor)
            if len(downside_returns) > 0 else 0
        )
        
        # Drawdown analysis
        max_dd, max_dd_duration, _ = self._calculate_drawdowns(clean_returns)
        
        # Risk-adjusted metrics
        sharpe = float(
            (np.mean(excess) / np.std(excess, ddof=1)) * np.sqrt(ann_factor)
            if np.std(excess, ddof=1) > 0 else 0
        )
        
        sortino = self._sortino_ratio(clean_returns, period_rf, ann_factor)
        calmar = self._calmar_ratio(annualized_return, max_dd)
        
        # Statistical tests
        t_calc = TStatCalculator(frequency, self.risk_free_rate)
        t_result = t_calc.test_mean_return(clean_returns, alternative="greater")
        
        # Higher moments
        skewness = float(stats.skew(clean_returns))
        kurtosis = float(stats.kurtosis(clean_returns))
        
        # Bootstrap confidence intervals for Sharpe
        sharpe_ci = self._bootstrap.sharpe_ratio_ci(
            clean_returns,
            frequency=frequency,
            risk_free_rate=self.risk_free_rate,
            confidence_level=self.confidence_level
        )
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe,
            sortino_ratio=float(sortino) if np.isfinite(sortino) else 999.0,
            calmar_ratio=float(calmar) if np.isfinite(calmar) else 999.0,
            volatility=volatility,
            downside_deviation=downside_deviation,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            t_statistic=t_result.statistic,
            p_value=t_result.p_value,
            skewness=skewness,
            kurtosis=kurtosis,
            sharpe_ci_lower=sharpe_ci[0],
            sharpe_ci_upper=sharpe_ci[2],
            n_observations=n,
            frequency=frequency.name,
            start_date=start_date,
            end_date=end_date
        )
    
    def format_report(
        self,
        metrics: PerformanceMetrics,
        include_interpretation: bool = True
    ) -> str:
        """
        Format metrics as publication-quality text report.
        
        Parameters
        ----------
        metrics : PerformanceMetrics
            Metrics from full_report().
        include_interpretation : bool
            Include qualitative interpretation of results.
            
        Returns
        -------
        str
            Formatted report string.
        """
        ci_pct = int(self.confidence_level * 100)
        
        lines = [
            "=" * 60,
            "STRATEGY PERFORMANCE REPORT",
            "=" * 60,
            "",
            f"Period: {metrics.start_date or 'N/A'} to {metrics.end_date or 'N/A'}",
            f"Frequency: {metrics.frequency}",
            f"Observations: {metrics.n_observations:,}",
            "",
            "-" * 40,
            "RETURN METRICS",
            "-" * 40,
            f"  Total Return:       {metrics.total_return:>10.2%}",
            f"  Annualized Return:  {metrics.annualized_return:>10.2%}",
            "",
            "-" * 40,
            "RISK METRICS",
            "-" * 40,
            f"  Volatility (ann.):  {metrics.volatility:>10.2%}",
            f"  Downside Dev.:      {metrics.downside_deviation:>10.2%}",
            f"  Max Drawdown:       {metrics.max_drawdown:>10.2%}",
            f"  Max DD Duration:    {metrics.max_drawdown_duration:>10} periods",
            "",
            "-" * 40,
            "RISK-ADJUSTED METRICS",
            "-" * 40,
            f"  Sharpe Ratio:       {metrics.sharpe_ratio:>10.3f}",
            f"    {ci_pct}% CI:          [{metrics.sharpe_ci_lower:.3f}, {metrics.sharpe_ci_upper:.3f}]",
            f"  Sortino Ratio:      {metrics.sortino_ratio:>10.3f}",
            f"  Calmar Ratio:       {metrics.calmar_ratio:>10.3f}",
            "",
            "-" * 40,
            "STATISTICAL SIGNIFICANCE",
            "-" * 40,
            f"  t-statistic:        {metrics.t_statistic:>10.3f}",
            f"  p-value:            {metrics.p_value:>10.4f}",
            "",
            "-" * 40,
            "DISTRIBUTION CHARACTERISTICS",
            "-" * 40,
            f"  Skewness:           {metrics.skewness:>10.3f}",
            f"  Excess Kurtosis:    {metrics.kurtosis:>10.3f}",
            "",
            "=" * 60,
        ]
        
        if include_interpretation:
            lines.extend(self._interpret_metrics(metrics))
        
        return "\n".join(lines)
    
    def _interpret_metrics(self, metrics: PerformanceMetrics) -> List[str]:
        """Generate qualitative interpretation of metrics."""
        interpretations = [
            "",
            "INTERPRETATION",
            "-" * 40,
        ]
        
        # Sharpe interpretation
        if metrics.sharpe_ratio > 2.0:
            interpretations.append("• Sharpe > 2.0: Exceptional risk-adjusted performance")
        elif metrics.sharpe_ratio > 1.0:
            interpretations.append("• Sharpe 1.0-2.0: Strong risk-adjusted performance")
        elif metrics.sharpe_ratio > 0.5:
            interpretations.append("• Sharpe 0.5-1.0: Moderate risk-adjusted performance")
        else:
            interpretations.append("• Sharpe < 0.5: Weak risk-adjusted performance")
        
        # Statistical significance
        if metrics.p_value < 0.01:
            interpretations.append("• p < 0.01: Highly significant (***)")
        elif metrics.p_value < 0.05:
            interpretations.append("• p < 0.05: Significant (**)") 
        elif metrics.p_value < 0.10:
            interpretations.append("• p < 0.10: Marginally significant (*)")
        else:
            interpretations.append("• p ≥ 0.10: NOT statistically significant")
        
        # Confidence interval
        if metrics.sharpe_ci_lower > 0:
            interpretations.append("• CI excludes zero: Evidence of genuine alpha")
        else:
            interpretations.append("• CI includes zero: Cannot rule out luck")
        
        # Distribution warnings
        if abs(metrics.skewness) > 1:
            interpretations.append(f"• High skewness ({metrics.skewness:.2f}): Non-normal returns")
        
        if metrics.kurtosis > 3:
            interpretations.append(f"• Fat tails (kurtosis={metrics.kurtosis:.2f}): Tail risk present")
        
        # Max drawdown warning
        if metrics.max_drawdown < -0.20:
            interpretations.append(f"• Large max DD ({metrics.max_drawdown:.1%}): Significant capital risk")
        
        interpretations.append("")
        
        return interpretations
    
    def compare_strategies(
        self,
        strategy_returns: Dict[str, np.ndarray],
        frequency: ReturnFrequency = ReturnFrequency.DAILY
    ) -> Dict[str, Any]:
        """
        Compare multiple strategies with proper multiple testing correction.
        
        Parameters
        ----------
        strategy_returns : Dict[str, np.ndarray]
            Dictionary mapping strategy names to return arrays.
        frequency : ReturnFrequency
            Return frequency.
            
        Returns
        -------
        Dict[str, Any]
            Comparison results with corrected p-values.
        """
        results = {}
        p_values = []
        strategy_names = list(strategy_returns.keys())
        
        for name, returns in strategy_returns.items():
            try:
                metrics = self.full_report(returns, frequency)
                results[name] = {
                    "sharpe": metrics.sharpe_ratio,
                    "sortino": metrics.sortino_ratio,
                    "max_dd": metrics.max_drawdown,
                    "p_value": metrics.p_value,
                    "t_stat": metrics.t_statistic
                }
                p_values.append(metrics.p_value)
            except ValueError as e:
                results[name] = {"error": str(e)}
                p_values.append(1.0)
        
        # Apply multiple testing correction
        corrector = MultipleTestingCorrector(alpha=1 - self.confidence_level)
        holm_adj, holm_reject = corrector.holm_correction(p_values)
        
        for i, name in enumerate(strategy_names):
            if "error" not in results[name]:
                results[name]["holm_adjusted_p"] = float(holm_adj[i])
                results[name]["significant_after_correction"] = bool(holm_reject[i])
        
        return {
            "strategies": results,
            "n_significant_raw": sum(1 for p in p_values if p < (1 - self.confidence_level)),
            "n_significant_corrected": int(np.sum(holm_reject)),
            "correction_method": "Holm-Bonferroni"
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def validate_strategy(
    returns: np.ndarray,
    frequency: ReturnFrequency = ReturnFrequency.DAILY,
    n_strategies_tested: int = 1,
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """
    One-stop validation of a trading strategy.
    
    Combines all validation methods into a single comprehensive analysis.
    
    Parameters
    ----------
    returns : np.ndarray
        Strategy returns.
    frequency : ReturnFrequency
        Return frequency.
    n_strategies_tested : int
        Number of strategies tested (for multiple testing adjustment).
    confidence_level : float
        Confidence level for tests.
        
    Returns
    -------
    Dict[str, Any]
        Complete validation results.
        
    Examples
    --------
    >>> returns = np.random.randn(252) * 0.01 + 0.0003
    >>> results = validate_strategy(returns, n_strategies_tested=10)
    >>> print(f"Valid alpha: {results['is_valid_alpha']}")
    """
    reporter = PerformanceReporter(confidence_level=confidence_level)
    detector = OverfitDetector()
    
    # Get full performance metrics
    metrics = reporter.full_report(returns, frequency)
    
    # Overfit detection
    overfit_analysis = detector.detect_overfitting(
        returns, 
        n_strategies_tested=n_strategies_tested,
        frequency=frequency
    )
    
    # Determine if alpha is valid
    is_valid_alpha = (
        metrics.p_value < (1 - confidence_level) and
        metrics.sharpe_ci_lower > 0 and
        not overfit_analysis["is_likely_overfit"]
    )
    
    return {
        "performance_metrics": metrics,
        "overfit_analysis": overfit_analysis,
        "is_valid_alpha": is_valid_alpha,
        "report": reporter.format_report(metrics)
    }


if __name__ == "__main__":
    # Demo/test the module
    np.random.seed(42)
    
    # Generate synthetic returns with positive alpha
    n_days = 504  # 2 years of daily data
    daily_returns = np.random.randn(n_days) * 0.015 + 0.0004  # ~10% ann. return
    
    print("Statistical Validation Suite Demo")
    print("=" * 60)
    
    # Run full validation
    results = validate_strategy(
        daily_returns,
        frequency=ReturnFrequency.DAILY,
        n_strategies_tested=5,
        confidence_level=0.95
    )
    
    print(results["report"])
    
    print("\nOverfitting Analysis:")
    print("-" * 40)
    oa = results["overfit_analysis"]
    print(f"  Deflated Sharpe Ratio: {oa['deflated_sharpe_ratio']:.3f}")
    print(f"  DSR p-value: {oa['dsr_p_value']:.4f}")
    print(f"  Is Likely Overfit: {oa['is_likely_overfit']}")
    
    if oa["warnings"]:
        print("\nWarnings:")
        for w in oa["warnings"]:
            print(f"  - {w}")
    
    print(f"\n{'='*60}")
    print(f"VERDICT: {'VALID ALPHA' if results['is_valid_alpha'] else 'INSUFFICIENT EVIDENCE'}")
    print(f"{'='*60}")
