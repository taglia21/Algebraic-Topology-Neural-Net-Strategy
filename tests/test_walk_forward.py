"""
Tests for Walk-Forward Optimization Framework.

This module tests the walk-forward analysis components:
- Window generation (anchored and rolling)
- Look-ahead bias detection
- Optimization correctness
- Degradation analysis

Run with: pytest tests/test_walk_forward.py -v
"""

import numpy as np
import pytest
from typing import Dict, Any

import sys
sys.path.insert(0, 'src')

from walk_forward import (
    WalkForwardEngine,
    AnchoredWalkForward,
    RollingWalkForward,
    DegradationAnalyzer,
    WindowSpec,
    WindowResult,
    WalkForwardSummary,
    OptimizationObjective,
    SimpleMovingAverageStrategy,
    _verify_no_lookahead,
    _calculate_sharpe,
    _calculate_sortino,
    _calculate_max_drawdown,
    quick_walk_forward
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_prices():
    """Generate sample price data."""
    np.random.seed(42)
    n = 1000
    return 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02 + 0.0001))


@pytest.fixture
def sample_returns():
    """Generate sample return data."""
    np.random.seed(42)
    return np.random.randn(500) * 0.015 + 0.0003


@pytest.fixture
def simple_strategy():
    """Create simple test strategy."""
    return SimpleMovingAverageStrategy()


@pytest.fixture
def param_bounds():
    """Standard parameter bounds for testing."""
    return {
        'fast_period': (5, 20),
        'slow_period': (20, 60)
    }


# =============================================================================
# Window Generation Tests
# =============================================================================

class TestWindowGeneration:
    """Test window generation for walk-forward analysis."""
    
    def test_anchored_window_count(self, sample_prices):
        """Anchored walk-forward should generate requested number of windows."""
        engine = AnchoredWalkForward(
            sample_prices,
            train_ratio=0.7,
            n_windows=5
        )
        windows = engine.get_windows()
        
        assert len(windows) == 5, f"Expected 5 windows, got {len(windows)}"
    
    def test_anchored_windows_start_at_zero(self, sample_prices):
        """Anchored windows should all start training at index 0."""
        engine = AnchoredWalkForward(sample_prices, n_windows=5)
        windows = engine.get_windows()
        
        for w in windows:
            assert w.train_start == 0, f"Window {w.window_id} train_start != 0"
    
    def test_rolling_windows_fixed_size(self, sample_prices):
        """Rolling windows should have consistent training size."""
        engine = RollingWalkForward(
            sample_prices,
            train_size=100,
            test_size=50,
            n_windows=5
        )
        windows = engine.get_windows()
        
        for w in windows:
            assert w.train_size == 100, f"Window {w.window_id} train_size != 100"
            assert w.test_size == 50, f"Window {w.window_id} test_size != 50"
    
    def test_windows_non_overlapping_test(self, sample_prices):
        """Test periods should not overlap."""
        engine = AnchoredWalkForward(sample_prices, n_windows=5)
        windows = engine.get_windows()
        
        for i in range(len(windows) - 1):
            assert windows[i].test_end <= windows[i+1].test_start, \
                f"Windows {i} and {i+1} have overlapping test periods"


# =============================================================================
# Look-Ahead Bias Tests
# =============================================================================

class TestLookAheadBias:
    """Test look-ahead bias detection."""
    
    def test_valid_windows_pass(self):
        """Valid windows should pass look-ahead check."""
        windows = [
            WindowSpec(0, 0, 100, 100, 150),
            WindowSpec(1, 0, 150, 150, 200),
            WindowSpec(2, 0, 200, 200, 250),
        ]
        assert _verify_no_lookahead(windows) is True
    
    def test_train_after_test_fails(self):
        """Training data after test start should fail."""
        windows = [
            WindowSpec(0, 0, 120, 100, 150),  # train_end > test_start!
        ]
        with pytest.raises(ValueError, match="Look-ahead bias"):
            _verify_no_lookahead(windows)
    
    def test_overlapping_test_fails(self):
        """Overlapping test periods should fail."""
        windows = [
            WindowSpec(0, 0, 100, 100, 160),
            WindowSpec(1, 0, 150, 150, 200),  # Overlaps with window 0
        ]
        with pytest.raises(ValueError, match="Overlapping test periods"):
            _verify_no_lookahead(windows)
    
    def test_engine_verifies_on_run(self, sample_prices, simple_strategy, param_bounds):
        """Engine should verify look-ahead bias when running."""
        engine = AnchoredWalkForward(sample_prices, n_windows=3)
        
        # This should not raise - valid windows
        summary = engine.run(simple_strategy, param_bounds, grid_points=3)
        assert summary.n_windows == 3


# =============================================================================
# Metric Calculation Tests
# =============================================================================

class TestMetricCalculations:
    """Test metric calculation functions."""
    
    def test_sharpe_positive_returns(self):
        """Sharpe should be positive for positive mean returns."""
        returns = np.array([0.01, 0.02, 0.01, 0.015, 0.01])
        sharpe = _calculate_sharpe(returns, annualization=252)
        assert sharpe > 0, "Sharpe should be positive for positive returns"
    
    def test_sharpe_zero_volatility(self):
        """Sharpe should be 0 for zero volatility."""
        returns = np.array([0.01, 0.01, 0.01, 0.01])
        sharpe = _calculate_sharpe(returns)
        assert sharpe == 0.0, "Sharpe should be 0 for zero volatility"
    
    def test_sortino_ignores_upside(self):
        """Sortino should only penalize downside volatility."""
        # All positive returns - infinite Sortino
        returns = np.array([0.01, 0.02, 0.03, 0.01])
        sortino = _calculate_sortino(returns)
        assert sortino == float('inf'), "Sortino should be inf for no downside"
    
    def test_max_drawdown_calculation(self):
        """Max drawdown should be calculated correctly."""
        # Simulate: 100 -> 110 -> 90 -> 100
        # Return sequence: +10%, -18.18%, +11.11%
        returns = np.array([0.10, -0.1818, 0.1111])
        max_dd = _calculate_max_drawdown(returns)
        
        # Max DD should be around -18.18% (from peak 110 to trough 90)
        assert max_dd < -0.15, f"Max DD should be significant, got {max_dd}"
    
    def test_empty_returns(self):
        """Functions should handle empty returns gracefully."""
        assert _calculate_sharpe(np.array([])) == 0.0
        assert _calculate_sortino(np.array([])) == 0.0
        assert _calculate_max_drawdown(np.array([])) == 0.0


# =============================================================================
# Optimization Tests
# =============================================================================

class TestOptimization:
    """Test parameter optimization."""
    
    def test_grid_search_finds_parameters(self, sample_prices, simple_strategy, param_bounds):
        """Grid search should find valid parameters."""
        engine = AnchoredWalkForward(sample_prices, n_windows=2)
        windows = engine.get_windows()
        
        result = engine.optimize_window(
            windows[0],
            simple_strategy,
            param_bounds,
            optimization_method="grid",
            grid_points=3
        )
        
        assert result.optimal_params is not None
        assert 'fast_period' in result.optimal_params
        assert 'slow_period' in result.optimal_params
    
    def test_random_search_finds_parameters(self, sample_prices, simple_strategy, param_bounds):
        """Random search should find valid parameters."""
        engine = AnchoredWalkForward(sample_prices, n_windows=2)
        windows = engine.get_windows()
        
        result = engine.optimize_window(
            windows[0],
            simple_strategy,
            param_bounds,
            optimization_method="random",
            max_iterations=20
        )
        
        assert result.optimal_params is not None
        bounds = param_bounds
        assert bounds['fast_period'][0] <= result.optimal_params['fast_period'] <= bounds['fast_period'][1]
    
    def test_parameters_within_bounds(self, sample_prices, simple_strategy, param_bounds):
        """Optimized parameters should be within bounds."""
        engine = AnchoredWalkForward(sample_prices, n_windows=3)
        summary = engine.run(simple_strategy, param_bounds, grid_points=3)
        
        for result in summary.window_results:
            for name, (low, high) in param_bounds.items():
                value = result.optimal_params[name]
                assert low <= value <= high, \
                    f"Parameter {name}={value} outside bounds [{low}, {high}]"


# =============================================================================
# Walk-Forward Summary Tests
# =============================================================================

class TestWalkForwardSummary:
    """Test walk-forward summary generation."""
    
    def test_summary_contains_all_windows(self, sample_prices, simple_strategy, param_bounds):
        """Summary should contain results for all windows."""
        n_windows = 4
        engine = AnchoredWalkForward(sample_prices, n_windows=n_windows)
        summary = engine.run(simple_strategy, param_bounds, grid_points=3)
        
        assert summary.n_windows == n_windows
        assert len(summary.window_results) == n_windows
    
    def test_combined_oos_returns(self, sample_prices, simple_strategy, param_bounds):
        """Combined OOS returns should concatenate all test returns."""
        engine = AnchoredWalkForward(sample_prices, n_windows=3)
        summary = engine.run(simple_strategy, param_bounds, grid_points=3)
        
        expected_len = sum(len(r.test_returns) for r in summary.window_results)
        assert len(summary.combined_oos_returns) == expected_len
    
    def test_degradation_ratio_calculation(self, sample_prices, simple_strategy, param_bounds):
        """Degradation ratio should be OOS/IS."""
        engine = AnchoredWalkForward(sample_prices, n_windows=3)
        summary = engine.run(simple_strategy, param_bounds, grid_points=3)
        
        # Calculate expected mean degradation
        degradations = []
        for r in summary.window_results:
            if r.train_sharpe != 0:
                degradations.append(r.test_sharpe / r.train_sharpe)
        
        if degradations:
            expected = np.mean(degradations)
            assert np.isclose(summary.mean_sharpe_degradation, expected, rtol=0.01)


# =============================================================================
# Degradation Analyzer Tests
# =============================================================================

class TestDegradationAnalyzer:
    """Test degradation analysis."""
    
    def test_analyzer_returns_all_components(self, sample_prices, simple_strategy, param_bounds):
        """Analyzer should return all analysis components."""
        engine = AnchoredWalkForward(sample_prices, n_windows=3)
        summary = engine.run(simple_strategy, param_bounds, grid_points=3)
        
        analyzer = DegradationAnalyzer()
        analysis = analyzer.analyze(summary)
        
        assert 'sharpe_analysis' in analysis
        assert 'return_analysis' in analysis
        assert 'consistency_analysis' in analysis
        assert 'parameter_stability' in analysis
        assert 'overall_assessment' in analysis
    
    def test_overall_score_range(self, sample_prices, simple_strategy, param_bounds):
        """Overall score should be in valid range."""
        engine = AnchoredWalkForward(sample_prices, n_windows=3)
        summary = engine.run(simple_strategy, param_bounds, grid_points=3)
        
        analyzer = DegradationAnalyzer()
        analysis = analyzer.analyze(summary)
        
        score = analysis['overall_assessment']['overall_score']
        assert 0 <= score <= 100, f"Score {score} outside [0, 100]"
    
    def test_grade_assignment(self, sample_prices, simple_strategy, param_bounds):
        """Grade should be valid letter grade."""
        engine = AnchoredWalkForward(sample_prices, n_windows=3)
        summary = engine.run(simple_strategy, param_bounds, grid_points=3)
        
        analyzer = DegradationAnalyzer()
        analysis = analyzer.analyze(summary)
        
        grade = analysis['overall_assessment']['grade']
        assert grade in ['A', 'B', 'C', 'F'], f"Invalid grade: {grade}"
    
    def test_report_generation(self, sample_prices, simple_strategy, param_bounds):
        """Report should be generated without errors."""
        engine = AnchoredWalkForward(sample_prices, n_windows=3)
        summary = engine.run(simple_strategy, param_bounds, grid_points=3)
        
        analyzer = DegradationAnalyzer()
        report = analyzer.generate_report(summary)
        
        assert len(report) > 0
        assert "WALK-FORWARD" in report
        assert "Overall Score" in report


# =============================================================================
# Robustness Criteria Tests
# =============================================================================

class TestRobustnessCriteria:
    """Test robustness assessment criteria."""
    
    def test_robustness_requires_positive_oos_sharpe(self, sample_prices, simple_strategy, param_bounds):
        """Strategy should not be robust with negative OOS Sharpe."""
        # Create engine and run
        engine = AnchoredWalkForward(sample_prices, n_windows=3)
        summary = engine.run(simple_strategy, param_bounds, grid_points=3)
        
        # If OOS Sharpe is negative, should not be robust
        if summary.combined_oos_sharpe < 0:
            assert not summary.is_robust
    
    def test_degradation_threshold(self, sample_prices, simple_strategy, param_bounds):
        """Severe degradation should trigger warning."""
        engine = AnchoredWalkForward(sample_prices, n_windows=3)
        summary = engine.run(simple_strategy, param_bounds, grid_points=3)
        
        # Check if degradation warnings are generated appropriately
        if summary.mean_sharpe_degradation < 0.5:
            assert any("degradation" in w.lower() for w in summary.warnings)


# =============================================================================
# Quick Walk-Forward Tests
# =============================================================================

class TestQuickWalkForward:
    """Test convenience function."""
    
    def test_quick_anchored(self, sample_prices, simple_strategy, param_bounds):
        """Quick walk-forward should work in anchored mode."""
        summary, analysis = quick_walk_forward(
            sample_prices,
            simple_strategy,
            param_bounds,
            mode="anchored",
            n_windows=3,
            grid_points=3
        )
        
        assert summary.n_windows == 3
        assert 'overall_assessment' in analysis
    
    def test_quick_rolling(self, sample_prices, simple_strategy, param_bounds):
        """Quick walk-forward should work in rolling mode."""
        summary, analysis = quick_walk_forward(
            sample_prices,
            simple_strategy,
            param_bounds,
            mode="rolling",
            n_windows=3,
            grid_points=3
        )
        
        assert summary.n_windows >= 1  # May have fewer due to data constraints
        assert 'overall_assessment' in analysis
    
    def test_invalid_mode_raises(self, sample_prices, simple_strategy, param_bounds):
        """Invalid mode should raise error."""
        with pytest.raises(ValueError, match="Unknown mode"):
            quick_walk_forward(
                sample_prices,
                simple_strategy,
                param_bounds,
                mode="invalid"
            )


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_small_dataset_warning(self):
        """Small dataset should trigger warning."""
        small_data = np.random.randn(50)
        
        with pytest.warns(UserWarning, match="Small dataset"):
            AnchoredWalkForward(small_data, n_windows=2)
    
    def test_invalid_train_ratio(self, sample_prices):
        """Invalid train ratio should raise error."""
        with pytest.raises(ValueError, match="train_ratio"):
            AnchoredWalkForward(sample_prices, train_ratio=1.5)
        
        with pytest.raises(ValueError, match="train_ratio"):
            AnchoredWalkForward(sample_prices, train_ratio=0)
    
    def test_too_few_windows(self, sample_prices):
        """Too few windows should raise error."""
        with pytest.raises(ValueError, match="n_windows"):
            AnchoredWalkForward(sample_prices, n_windows=1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
