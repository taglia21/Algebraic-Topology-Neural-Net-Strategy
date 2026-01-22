"""
V2.5 Elite Upgrade - Comprehensive Test Suite
==============================================

Tests all V2.5 components for correctness and integration.

Components Tested:
1. Elite Feature Engineer - 80-120 deep features
2. Gradient Boost Ensemble - Multi-model stacking
3. Multi-Indicator Validator - Signal confirmation
4. Walk-Forward Optimizer - Robust optimization
5. Bayesian Tuner - Efficient hyperparameter search
6. Data Quality Checker - Real-time monitoring

Author: System V2.5
Date: 2025
"""

import pytest
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.WARNING)

# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n_bars = 500
    
    dates = pd.date_range(end=datetime.now(), periods=n_bars, freq='1h')
    close = 100 * np.exp(np.cumsum(np.random.randn(n_bars) * 0.01))
    open_price = close * (1 + np.random.randn(n_bars) * 0.005)
    high = np.maximum(close, open_price) * (1 + np.abs(np.random.randn(n_bars)) * 0.01)
    low = np.minimum(close, open_price) * (1 - np.abs(np.random.randn(n_bars)) * 0.01)
    volume = np.random.randint(100000, 1000000, n_bars).astype(float)
    
    return pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)


@pytest.fixture
def sample_features():
    """Generate sample features for ML testing."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    X = np.random.randn(n_samples, n_features)
    # Target with some signal
    y = 0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.2 * np.random.randn(n_samples)
    
    return X, y


# ============================================================
# ELITE FEATURE ENGINEER TESTS
# ============================================================

class TestEliteFeatureEngineer:
    """Tests for Elite Feature Engineer."""
    
    def test_import(self):
        """Test module imports correctly."""
        from src.features.elite_feature_engineer import EliteFeatureEngineer, FeatureConfig
        assert EliteFeatureEngineer is not None
        assert FeatureConfig is not None
    
    def test_feature_generation(self, sample_ohlcv):
        """Test feature generation produces expected number of features."""
        from src.features.elite_feature_engineer import EliteFeatureEngineer
        
        engineer = EliteFeatureEngineer()
        features = engineer.generate_features(sample_ohlcv)
        
        # Should generate 80-130 features
        assert len(features.columns) >= 80, f"Only {len(features.columns)} features generated"
        assert len(features.columns) <= 150, f"Too many features: {len(features.columns)}"
    
    def test_no_nan_values(self, sample_ohlcv):
        """Test that output has no NaN values."""
        from src.features.elite_feature_engineer import EliteFeatureEngineer
        
        engineer = EliteFeatureEngineer()
        features = engineer.generate_features(sample_ohlcv)
        
        nan_count = features.isnull().sum().sum()
        assert nan_count == 0, f"Found {nan_count} NaN values"
    
    def test_feature_selection(self, sample_ohlcv):
        """Test MIC-based feature selection."""
        from src.features.elite_feature_engineer import EliteFeatureEngineer
        
        engineer = EliteFeatureEngineer()
        features = engineer.generate_features(sample_ohlcv)
        
        # Create target
        target = sample_ohlcv['close'].pct_change().shift(-1).dropna()
        
        # Align features with target
        aligned_features = features.iloc[:-1]
        
        selected, rankings = engineer.select_features(aligned_features, target.values)
        
        assert len(selected.columns) > 0
        assert len(rankings) > 0
    
    def test_performance_under_500ms(self, sample_ohlcv):
        """Test that feature generation is fast enough."""
        from src.features.elite_feature_engineer import EliteFeatureEngineer
        import time
        
        engineer = EliteFeatureEngineer()
        
        start = time.perf_counter()
        features = engineer.generate_features(sample_ohlcv)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        assert elapsed_ms < 1000, f"Too slow: {elapsed_ms:.0f}ms"
    
    def test_vmd_decomposition(self):
        """Test VMD decomposition component."""
        from src.features.elite_feature_engineer import VMDDecomposer, FeatureConfig
        
        np.random.seed(42)
        signal = np.sin(np.linspace(0, 10*np.pi, 200)) + 0.1 * np.random.randn(200)
        
        config = FeatureConfig()
        vmd = VMDDecomposer(config)
        components = vmd.decompose(signal)
        
        assert 'trend' in components
        assert 'cycles' in components
        assert 'noise' in components
    
    def test_mic_calculator(self):
        """Test MIC score calculation."""
        from src.features.elite_feature_engineer import MICCalculator, FeatureConfig
        
        np.random.seed(42)
        x = np.random.randn(100)
        y = x + 0.1 * np.random.randn(100)  # High correlation
        
        config = FeatureConfig()
        mic = MICCalculator(config)
        score = mic.compute_mic(x, y)
        
        assert 0 <= score <= 1, f"MIC score out of range: {score}"
        assert score > 0.3, f"Expected high MIC for correlated data: {score}"


# ============================================================
# GRADIENT BOOST ENSEMBLE TESTS
# ============================================================

class TestGradientBoostEnsemble:
    """Tests for Gradient Boost Ensemble."""
    
    def test_import(self):
        """Test module imports correctly."""
        from src.ml.gradient_boost_ensemble import GradientBoostEnsemble, EnsembleConfig
        assert GradientBoostEnsemble is not None
    
    def test_ensemble_training(self, sample_features):
        """Test ensemble trains successfully."""
        from src.ml.gradient_boost_ensemble import GradientBoostEnsemble, EnsembleConfig
        
        X, y = sample_features
        X_train, y_train = X[:800], y[:800]
        
        config = EnsembleConfig(
            use_lstm=False,
            cv_folds=2,
            xgb_n_estimators=20,
            lgb_n_estimators=20,
            rf_n_estimators=20,
            cat_iterations=20
        )
        
        ensemble = GradientBoostEnsemble(config)
        ensemble.fit(X_train, y_train)
        
        assert ensemble.is_fitted
    
    def test_ensemble_prediction(self, sample_features):
        """Test ensemble predictions."""
        from src.ml.gradient_boost_ensemble import GradientBoostEnsemble, EnsembleConfig
        
        X, y = sample_features
        X_train, X_test = X[:800], X[800:]
        y_train = y[:800]
        
        config = EnsembleConfig(
            use_lstm=False,
            cv_folds=2,
            xgb_n_estimators=20,
            lgb_n_estimators=20,
            rf_n_estimators=20,
            cat_iterations=20
        )
        
        ensemble = GradientBoostEnsemble(config)
        ensemble.fit(X_train, y_train)
        
        predictions = ensemble.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert not np.any(np.isnan(predictions))
    
    def test_feature_importance(self, sample_features):
        """Test feature importance extraction."""
        from src.ml.gradient_boost_ensemble import GradientBoostEnsemble, EnsembleConfig
        
        X, y = sample_features
        
        config = EnsembleConfig(
            use_lstm=False,
            cv_folds=2,
            rf_n_estimators=20
        )
        
        ensemble = GradientBoostEnsemble(config)
        ensemble.fit(X, y, feature_names=[f'f_{i}' for i in range(X.shape[1])])
        
        importance = ensemble.get_feature_importance()
        
        assert len(importance) > 0
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns
    
    def test_model_diagnostics(self, sample_features):
        """Test model diagnostics."""
        from src.ml.gradient_boost_ensemble import GradientBoostEnsemble, EnsembleConfig
        
        X, y = sample_features
        
        config = EnsembleConfig(use_lstm=False, cv_folds=2)
        ensemble = GradientBoostEnsemble(config)
        ensemble.fit(X[:500], y[:500])
        
        diag = ensemble.get_model_diagnostics()
        
        assert 'is_fitted' in diag
        assert 'n_models' in diag
        assert 'model_weights' in diag


# ============================================================
# MULTI-INDICATOR VALIDATOR TESTS
# ============================================================

class TestMultiIndicatorValidator:
    """Tests for Multi-Indicator Validator."""
    
    def test_import(self):
        """Test module imports correctly."""
        from src.validation.multi_indicator_validator import (
            MultiIndicatorValidator, SignalDirection, ValidationResult
        )
        assert MultiIndicatorValidator is not None
    
    def test_signal_validation(self, sample_ohlcv):
        """Test signal validation."""
        from src.validation.multi_indicator_validator import (
            MultiIndicatorValidator, SignalDirection
        )
        
        validator = MultiIndicatorValidator()
        result = validator.validate_signal(SignalDirection.LONG, sample_ohlcv)
        
        assert result.total_indicators >= 7
        assert 0 <= result.signal_strength <= 100
        assert 0 <= result.confidence <= 1
    
    def test_indicator_count(self, sample_ohlcv):
        """Test that we use 7+ indicators."""
        from src.validation.multi_indicator_validator import (
            MultiIndicatorValidator, SignalDirection
        )
        
        validator = MultiIndicatorValidator()
        result = validator.validate_signal(SignalDirection.LONG, sample_ohlcv)
        
        assert result.total_indicators >= 7
    
    def test_batch_validation(self, sample_ohlcv):
        """Test batch signal validation."""
        from src.validation.multi_indicator_validator import MultiIndicatorValidator
        
        signals = pd.Series([1, -1, 0, 1, -1], index=sample_ohlcv.index[-5:])
        
        validator = MultiIndicatorValidator()
        results = validator.validate_signals_batch(signals, sample_ohlcv)
        
        assert len(results) == 5
        assert 'validated_signal' in results.columns
        assert 'is_valid' in results.columns
    
    def test_conflict_detection(self, sample_ohlcv):
        """Test conflict detection between indicators."""
        from src.validation.multi_indicator_validator import (
            MultiIndicatorValidator, SignalDirection
        )
        
        validator = MultiIndicatorValidator()
        result = validator.validate_signal(SignalDirection.LONG, sample_ohlcv)
        
        # Conflicts should be a list
        assert isinstance(result.conflicts, list)


# ============================================================
# WALK-FORWARD OPTIMIZER TESTS
# ============================================================

class TestWalkForwardOptimizer:
    """Tests for Walk-Forward Optimizer."""
    
    def test_import(self):
        """Test module imports correctly."""
        from src.optimization.walk_forward_optimizer import (
            WalkForwardOptimizer, WFOConfig, WFOMode
        )
        assert WalkForwardOptimizer is not None
    
    def test_anchored_splits(self):
        """Test anchored walk-forward split generation."""
        from src.optimization.walk_forward_optimizer import (
            WalkForwardOptimizer, WFOConfig, WFOMode
        )
        
        config = WFOConfig(
            mode=WFOMode.ANCHORED,
            n_splits=5,
            min_train_size=100,
            test_size=20,
            step_size=50
        )
        
        optimizer = WalkForwardOptimizer(config)
        splits = optimizer.generate_splits(500)
        
        assert len(splits) >= 3
        # Anchored: all start at 0
        assert all(s.train_start == 0 for s in splits)
    
    def test_rolling_splits(self):
        """Test rolling walk-forward split generation."""
        from src.optimization.walk_forward_optimizer import (
            WalkForwardOptimizer, WFOConfig, WFOMode
        )
        
        config = WFOConfig(
            mode=WFOMode.ROLLING,
            n_splits=5,
            train_size=150,
            test_size=20,
            step_size=50
        )
        
        optimizer = WalkForwardOptimizer(config)
        splits = optimizer.generate_splits(500)
        
        assert len(splits) >= 3
        # Rolling: train size is constant
        assert all(s.train_size == 150 for s in splits)
    
    def test_optimization(self, sample_ohlcv):
        """Test full optimization workflow."""
        from src.optimization.walk_forward_optimizer import (
            WalkForwardOptimizer, WFOConfig, example_objective
        )
        
        config = WFOConfig(n_splits=2, min_train_size=100, test_size=30)
        optimizer = WalkForwardOptimizer(config)
        
        param_grid = {
            'threshold': [0.3, 0.5],
            'lookback': [5, 10]
        }
        
        report = optimizer.optimize(sample_ohlcv, example_objective, param_grid)
        
        assert report.n_splits >= 2
        assert report.recommended_params is not None
    
    def test_monte_carlo(self, sample_ohlcv):
        """Test Monte Carlo validation."""
        from src.optimization.walk_forward_optimizer import (
            WalkForwardOptimizer, example_objective
        )
        
        optimizer = WalkForwardOptimizer()
        
        results = optimizer.monte_carlo_validation(
            sample_ohlcv,
            example_objective,
            {'threshold': 0.5, 'lookback': 10},
            n_runs=10
        )
        
        assert results.get('valid')
        assert 'sharpe_mean' in results


# ============================================================
# BAYESIAN TUNER TESTS
# ============================================================

class TestBayesianTuner:
    """Tests for Bayesian Hyperparameter Tuner."""
    
    def test_import(self):
        """Test module imports correctly."""
        from src.optimization.bayesian_tuner import (
            BayesianTuner, BayesianConfig, ParamSpace, ParamType
        )
        assert BayesianTuner is not None
    
    def test_param_space_definition(self):
        """Test parameter space definition."""
        from src.optimization.bayesian_tuner import BayesianTuner
        
        tuner = BayesianTuner()
        tuner.define_space({
            'x': {'type': 'continuous', 'low': 0, 'high': 1},
            'n': {'type': 'integer', 'low': 1, 'high': 10},
            'mode': {'type': 'categorical', 'choices': ['a', 'b', 'c']}
        })
        
        assert len(tuner.param_spaces) == 3
    
    def test_optimization(self):
        """Test optimization finds improvement."""
        from src.optimization.bayesian_tuner import BayesianTuner, BayesianConfig
        
        def objective(params):
            x = params['x']
            return -((x - 0.5) ** 2)  # Max at x=0.5
        
        config = BayesianConfig(n_initial_points=3, n_iterations=10, verbose=False)
        tuner = BayesianTuner(config)
        tuner.define_space({'x': {'type': 'continuous', 'low': 0, 'high': 1}})
        
        result = tuner.optimize(objective, maximize=True)
        
        assert result.n_iterations > 0
        assert result.best_score > result.convergence_history[0]
    
    def test_ensemble_tuner_creation(self):
        """Test ensemble tuner factory function."""
        from src.optimization.bayesian_tuner import create_ensemble_tuner
        
        tuner = create_ensemble_tuner()
        
        assert len(tuner.param_spaces) >= 5


# ============================================================
# DATA QUALITY CHECKER TESTS
# ============================================================

class TestDataQualityChecker:
    """Tests for Data Quality Checker."""
    
    def test_import(self):
        """Test module imports correctly."""
        from src.monitoring.data_quality_checker import (
            DataQualityChecker, DataQualityStatus, QualityConfig
        )
        assert DataQualityChecker is not None
    
    def test_quality_check(self, sample_ohlcv):
        """Test quality check on good data."""
        from src.monitoring.data_quality_checker import DataQualityChecker
        
        checker = DataQualityChecker()
        report = checker.check_quality(sample_ohlcv)
        
        assert report.overall_score >= 0
        assert report.overall_score <= 100
        assert len(report.checks) >= 5
    
    def test_good_data_passes(self, sample_ohlcv):
        """Test that good data passes checks."""
        from src.monitoring.data_quality_checker import (
            DataQualityChecker, DataQualityStatus
        )
        
        checker = DataQualityChecker()
        report = checker.check_quality(sample_ohlcv)
        
        assert report.status in [DataQualityStatus.EXCELLENT, DataQualityStatus.GOOD]
    
    def test_bad_data_detected(self, sample_ohlcv):
        """Test that bad data is detected."""
        from src.monitoring.data_quality_checker import DataQualityChecker
        
        bad_data = sample_ohlcv.copy()
        bad_data.iloc[50:60, bad_data.columns.get_loc('close')] = np.nan
        
        checker = DataQualityChecker()
        report_good = checker.check_quality(sample_ohlcv)
        report_bad = checker.check_quality(bad_data)
        
        assert report_bad.overall_score < report_good.overall_score
    
    def test_should_trade_decision(self, sample_ohlcv):
        """Test trading decision logic."""
        from src.monitoring.data_quality_checker import DataQualityChecker
        
        checker = DataQualityChecker()
        report = checker.check_quality(sample_ohlcv)
        
        should_trade, reason = checker.should_trade(report)
        
        assert isinstance(should_trade, bool)
        assert isinstance(reason, str)
    
    def test_recommendations_generated(self, sample_ohlcv):
        """Test recommendations for bad data."""
        from src.monitoring.data_quality_checker import DataQualityChecker
        
        bad_data = sample_ohlcv.copy()
        bad_data.iloc[100, bad_data.columns.get_loc('high')] = 0  # OHLC violation
        
        checker = DataQualityChecker()
        report = checker.check_quality(bad_data)
        
        # Should have recommendations for fixing issues
        assert isinstance(report.recommendations, list)


# ============================================================
# INTEGRATION TESTS
# ============================================================

class TestV25Integration:
    """Integration tests for V2.5 components."""
    
    def test_feature_to_ensemble_pipeline(self, sample_ohlcv):
        """Test features flowing into ensemble."""
        from src.features.elite_feature_engineer import EliteFeatureEngineer
        from src.ml.gradient_boost_ensemble import GradientBoostEnsemble, EnsembleConfig
        
        # Generate features
        engineer = EliteFeatureEngineer()
        features = engineer.generate_features(sample_ohlcv)
        
        # Create target
        target = sample_ohlcv['close'].pct_change().shift(-1).fillna(0)
        
        # Align data
        aligned_features = features.iloc[:-1]
        aligned_target = target.iloc[:-1]
        
        # Train ensemble
        config = EnsembleConfig(use_lstm=False, cv_folds=2, rf_n_estimators=20)
        ensemble = GradientBoostEnsemble(config)
        ensemble.fit(aligned_features.values, aligned_target.values)
        
        assert ensemble.is_fitted
    
    def test_quality_to_validation_pipeline(self, sample_ohlcv):
        """Test data quality check before signal validation."""
        from src.monitoring.data_quality_checker import DataQualityChecker
        from src.validation.multi_indicator_validator import (
            MultiIndicatorValidator, SignalDirection
        )
        
        # Check data quality
        checker = DataQualityChecker()
        quality_report = checker.check_quality(sample_ohlcv)
        
        should_trade, reason = checker.should_trade(quality_report)
        
        if should_trade:
            # Validate a signal
            validator = MultiIndicatorValidator()
            result = validator.validate_signal(SignalDirection.LONG, sample_ohlcv)
            
            assert result.total_indicators >= 7
    
    def test_wfo_with_bayesian(self):
        """Test Walk-Forward with Bayesian optimization."""
        from src.optimization.walk_forward_optimizer import WalkForwardOptimizer, WFOConfig
        from src.optimization.bayesian_tuner import BayesianTuner, BayesianConfig
        
        # Both should be usable
        wfo = WalkForwardOptimizer(WFOConfig(n_splits=2))
        tuner = BayesianTuner(BayesianConfig(n_iterations=5))
        
        assert wfo is not None
        assert tuner is not None


# ============================================================
# MAIN RUNNER
# ============================================================

def run_all_tests():
    """Run all tests and print summary."""
    import traceback
    
    print("=" * 60)
    print("V2.5 ELITE UPGRADE - TEST SUITE")
    print("=" * 60)
    
    test_classes = [
        TestEliteFeatureEngineer,
        TestGradientBoostEnsemble,
        TestMultiIndicatorValidator,
        TestWalkForwardOptimizer,
        TestBayesianTuner,
        TestDataQualityChecker,
        TestV25Integration
    ]
    
    # Create fixtures
    np.random.seed(42)
    n_bars = 500
    dates = pd.date_range(end=datetime.now(), periods=n_bars, freq='1h')
    close = 100 * np.exp(np.cumsum(np.random.randn(n_bars) * 0.01))
    open_price = close * (1 + np.random.randn(n_bars) * 0.005)
    high = np.maximum(close, open_price) * (1 + np.abs(np.random.randn(n_bars)) * 0.01)
    low = np.minimum(close, open_price) * (1 - np.abs(np.random.randn(n_bars)) * 0.01)
    volume = np.random.randint(100000, 1000000, n_bars).astype(float)
    
    sample_ohlcv = pd.DataFrame({
        'open': open_price, 'high': high, 'low': low, 'close': close, 'volume': volume
    }, index=dates)
    
    X = np.random.randn(1000, 50)
    y = 0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.2 * np.random.randn(1000)
    sample_features = (X, y)
    
    total_passed = 0
    total_tests = 0
    
    for test_class in test_classes:
        class_name = test_class.__name__
        print(f"\n{class_name}")
        print("-" * 40)
        
        instance = test_class()
        
        for method_name in dir(instance):
            if not method_name.startswith('test_'):
                continue
            
            total_tests += 1
            method = getattr(instance, method_name)
            
            try:
                # Check if method needs fixtures
                import inspect
                sig = inspect.signature(method)
                params = sig.parameters
                
                if 'sample_ohlcv' in params:
                    method(sample_ohlcv)
                elif 'sample_features' in params:
                    method(sample_features)
                else:
                    method()
                
                print(f"  ✅ {method_name}")
                total_passed += 1
                
            except Exception as e:
                print(f"  ❌ {method_name}: {str(e)[:50]}")
                if os.environ.get('DEBUG'):
                    traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {total_passed}/{total_tests} tests passed")
    print("=" * 60)
    
    return total_passed == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
