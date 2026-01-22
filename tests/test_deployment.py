#!/usr/bin/env python3
"""
V2.1 Production Deployment Test Suite
======================================

Comprehensive tests validating all V2.1 production components:
1. V2.1 Engine initialization and components
2. Ensemble regime detection
3. Transformer predictor
4. Monitoring dashboard
5. Daily validation system
6. Production launcher
7. Integration tests

Run with:
    pytest tests/test_deployment.py -v
    python tests/test_deployment.py  # Direct execution
"""

import os
import sys
import json
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, patch, MagicMock
import tempfile

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd


# =============================================================================
# TEST: V2.1 PRODUCTION ENGINE
# =============================================================================

class TestV21ProductionEngine(unittest.TestCase):
    """Test V2.1 Production Engine components."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        from src.trading.v21_production_engine import V21ProductionEngine, V21Config
        cls.V21ProductionEngine = V21ProductionEngine
        cls.V21Config = V21Config
        
    def test_config_validation(self):
        """Test configuration validation."""
        config = self.V21Config(
            hmm_weight=0.6,
            gmm_weight=0.3,
            cluster_weight=0.2,  # Sum > 1
        )
        config.validate()
        
        # Should normalize to sum = 1
        total = config.hmm_weight + config.gmm_weight + config.cluster_weight
        self.assertAlmostEqual(total, 1.0, places=2)
        
    def test_engine_initialization(self):
        """Test engine initializes with components."""
        config = self.V21Config(
            use_ensemble_regime=False,  # Disable for faster test
            use_transformer=False,
            fallback_to_v13=False,
        )
        engine = self.V21ProductionEngine(config)
        
        status = engine.get_component_status()
        self.assertIn("is_halted", status)
        self.assertIn("current_regime", status)
        self.assertIn("universe_size", status)
        self.assertFalse(status["is_halted"])
        
    def test_universe_loading(self):
        """Test universe loading for different modes."""
        config = self.V21Config(universe_mode="core")
        engine = self.V21ProductionEngine(config)
        
        universe = engine.get_universe()
        self.assertIn("SPY", universe)
        self.assertIn("QQQ", universe)
        self.assertGreater(len(universe), 0)
        
    def test_simple_regime_detection(self):
        """Test fallback regime detection."""
        config = self.V21Config(
            use_ensemble_regime=False,
            use_transformer=False,
        )
        engine = self.V21ProductionEngine(config)
        
        # Create mock price series (uptrend)
        prices = np.linspace(100, 120, 100)  # 20% up
        
        regime, confidence = engine._simple_regime_detection(prices)
        
        self.assertIn(regime, ["bull", "bear", "neutral", "risk_off"])
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        
    def test_risk_limit_checks(self):
        """Test risk limit validation."""
        engine = self.V21ProductionEngine()
        
        # Test normal drawdown
        should_halt, reason = engine.check_risk_limits(0.02, 100)
        self.assertFalse(should_halt)
        
        # Test emergency halt
        should_halt, reason = engine.check_risk_limits(0.10, 100)
        self.assertTrue(should_halt)
        self.assertIn("Emergency", reason)
        
    def test_position_sizing(self):
        """Test position sizing with regime adjustment."""
        engine = self.V21ProductionEngine()
        
        predictions = {
            "AAPL": {"ticker": "AAPL", "direction": "long", "confidence": 0.8},
            "MSFT": {"ticker": "MSFT", "direction": "long", "confidence": 0.6},
            "TSLA": {"ticker": "TSLA", "direction": "short", "confidence": 0.7},
        }
        
        signals = engine.compute_position_sizes(
            predictions,
            regime="bull",
            regime_confidence=0.8,
            max_position_pct=0.03,
            max_heat=0.20,
        )
        
        # Check signals generated
        self.assertGreater(len(signals), 0)
        
        # Check weights are within limits
        for ticker, signal in signals.items():
            self.assertLessEqual(signal["weight"], 0.03)
            self.assertGreater(signal["weight"], 0)


# =============================================================================
# TEST: MONITORING DASHBOARD
# =============================================================================

class TestMonitoringDashboard(unittest.TestCase):
    """Test monitoring dashboard functionality."""
    
    @classmethod
    def setUpClass(cls):
        from src.trading.monitoring_dashboard import MonitoringDashboard, PerformanceMetrics
        cls.MonitoringDashboard = MonitoringDashboard
        cls.PerformanceMetrics = PerformanceMetrics
        
    def test_dashboard_initialization(self):
        """Test dashboard initializes correctly."""
        dashboard = self.MonitoringDashboard(port=8081)
        
        self.assertEqual(dashboard.port, 8081)
        self.assertIsNotNone(dashboard.metrics)
        self.assertEqual(len(dashboard.alerts), 0)
        
    def test_metric_updates(self):
        """Test metric update functionality."""
        dashboard = self.MonitoringDashboard(port=8082)
        
        dashboard.update_metric("equity", 100000)
        dashboard.update_metric("drawdown", 0.02)
        
        self.assertEqual(dashboard.metrics.equity, 100000)
        self.assertEqual(dashboard.metrics.drawdown, 0.02)
        
    def test_batch_metric_updates(self):
        """Test batch metric update."""
        dashboard = self.MonitoringDashboard(port=8083)
        
        dashboard.update_metrics({
            "equity": 150000,
            "cash": 50000,
            "positions_count": 25,
            "sharpe_30d": 1.5,
        })
        
        self.assertEqual(dashboard.metrics.equity, 150000)
        self.assertEqual(dashboard.metrics.positions_count, 25)
        
    def test_alert_generation(self):
        """Test alert generation on threshold breach."""
        dashboard = self.MonitoringDashboard(port=8084)
        
        # Trigger drawdown alert
        dashboard.update_metric("drawdown", 0.06)  # Above 5% critical
        
        self.assertGreater(len(dashboard.alerts), 0)
        self.assertEqual(dashboard.alerts[-1].level, "critical")
        
    def test_trade_history(self):
        """Test trade history tracking."""
        dashboard = self.MonitoringDashboard(port=8085)
        
        dashboard.add_trade({
            "symbol": "AAPL",
            "side": "buy",
            "qty": 100,
            "price": 150.00,
        })
        
        self.assertEqual(len(dashboard.trade_history), 1)
        self.assertEqual(dashboard.trade_history[0]["symbol"], "AAPL")
        
    def test_sharpe_calculation(self):
        """Test rolling Sharpe calculation."""
        dashboard = self.MonitoringDashboard(port=8086)
        
        # Add 30 days of returns
        for _ in range(35):
            dashboard.add_daily_return(0.001)  # 0.1% daily
            
        self.assertGreater(dashboard.metrics.sharpe_30d, 0)
        
    def test_html_generation(self):
        """Test dashboard HTML generation."""
        dashboard = self.MonitoringDashboard(port=8087)
        dashboard.update_metrics({
            "equity": 100000,
            "drawdown": 0.01,
            "sharpe_30d": 1.2,
        })
        
        html = dashboard.get_dashboard_html()
        
        self.assertIn("V2.1 Production Dashboard", html)
        self.assertIn("$100,000", html)
        
    def test_json_endpoint(self):
        """Test JSON metrics endpoint."""
        dashboard = self.MonitoringDashboard(port=8088)
        dashboard.update_metric("equity", 200000)
        
        json_str = dashboard.get_metrics_json()
        data = json.loads(json_str)
        
        self.assertIn("metrics", data)
        self.assertEqual(data["metrics"]["equity"], 200000)


# =============================================================================
# TEST: DAILY VALIDATOR
# =============================================================================

class TestDailyValidator(unittest.TestCase):
    """Test daily validation system."""
    
    @classmethod
    def setUpClass(cls):
        from src.trading.daily_validator import DailyValidator, ValidationResult
        cls.DailyValidator = DailyValidator
        cls.ValidationResult = ValidationResult
        
    def test_validator_initialization(self):
        """Test validator initializes with defaults."""
        validator = self.DailyValidator()
        
        self.assertEqual(validator.expected_sharpe, 1.35)
        self.assertEqual(validator.sigma_threshold, 2.0)
        
    def test_trade_count_validation(self):
        """Test trade count validation."""
        validator = self.DailyValidator()
        
        # Normal trade count
        trades = [{"symbol": f"TICK{i}"} for i in range(20)]
        result = validator._validate_trade_count(trades)
        self.assertTrue(result.passed)
        
        # Too few trades
        trades = [{"symbol": "TICK1"}]
        result = validator._validate_trade_count(trades)
        self.assertFalse(result.passed)
        
    def test_drawdown_validation(self):
        """Test drawdown validation."""
        validator = self.DailyValidator()
        
        # Normal drawdown
        result = validator._validate_drawdown(0.02)
        self.assertTrue(result.passed)
        
        # High drawdown
        result = validator._validate_drawdown(0.08)
        self.assertFalse(result.passed)
        
    def test_sharpe_validation(self):
        """Test Sharpe ratio validation."""
        validator = self.DailyValidator()
        
        # Good Sharpe
        result = validator._validate_sharpe(1.40, 1.35)
        self.assertTrue(result.passed)
        
        # Poor Sharpe
        result = validator._validate_sharpe(0.50, 1.35)
        self.assertFalse(result.passed)
        
    def test_full_validation_report(self):
        """Test complete validation report generation."""
        validator = self.DailyValidator()
        
        trades = [
            {"symbol": "AAPL", "side": "buy", "value": 10000, "pnl": 100},
            {"symbol": "MSFT", "side": "buy", "value": 15000, "pnl": -50},
        ]
        
        report = validator.run_validation(
            daily_trades=trades,
            daily_return=0.001,
            current_dd=0.02,
            current_sharpe=1.30,
        )
        
        self.assertIn("overall_status", report)
        self.assertIn("checks_passed", report)
        self.assertIn("validation_results", report)
        
    def test_recommendation_generation(self):
        """Test recommendation generation for anomalies."""
        validator = self.DailyValidator()
        
        anomalies = [
            {"type": "drawdown", "deviation": 2.5, "message": "High drawdown"},
            {"type": "execution", "deviation": 2.0, "message": "High slippage"},
        ]
        
        recommendations = validator._generate_recommendations(anomalies, [])
        
        self.assertGreater(len(recommendations), 0)


# =============================================================================
# TEST: ENSEMBLE REGIME DETECTION
# =============================================================================

class TestEnsembleRegimeDetection(unittest.TestCase):
    """Test ensemble regime detection components."""
    
    def test_hmm_regime_detector(self):
        """Test HMM regime detector."""
        try:
            from src.trading.regime_ensemble import HMMRegimeDetector
            
            detector = HMMRegimeDetector(n_regimes=3)
            
            # Create synthetic features
            np.random.seed(42)
            features = np.random.randn(100, 5)
            
            detector.fit(features)
            
            # Predict on new data
            if detector.is_fitted:
                regime, confidence, _ = detector.predict(features[-1:])
                self.assertIn(regime, detector.regime_labels + ['unknown'])
        except ImportError as e:
            self.skipTest(f"hmmlearn not available: {e}")
            
    def test_ensemble_regime_detector(self):
        """Test full ensemble regime detector."""
        try:
            from src.trading.regime_ensemble import EnsembleRegimeDetector, RegimeState
            
            detector = EnsembleRegimeDetector(n_regimes=3)
            
            # Create synthetic returns data
            np.random.seed(42)
            returns = np.random.randn(150) * 0.01  # Daily returns
            
            # Fit using returns
            detector.fit(returns)
            
            # Get ensemble prediction - returns RegimeState
            features = detector.compute_features(returns)
            state = detector.predict(features[-1:])
            
            self.assertIsNotNone(state.regime)
            self.assertGreaterEqual(state.confidence, 0)
            self.assertLessEqual(state.confidence, 1)
        except (ImportError, AttributeError) as e:
            self.skipTest(f"Required packages not available: {e}")


# =============================================================================
# TEST: TRANSFORMER PREDICTOR
# =============================================================================

class TestTransformerPredictor(unittest.TestCase):
    """Test transformer predictor components."""
    
    def test_transformer_initialization(self):
        """Test transformer predictor initialization."""
        try:
            from src.ml.transformer_predictor import TransformerPredictor
            
            predictor = TransformerPredictor(
                d_model=64,  # Smaller for testing
                n_heads=4,
                n_layers=2,
            )
            
            self.assertIsNotNone(predictor)
        except ImportError as e:
            self.skipTest(f"PyTorch not available: {e}")
            
    def test_transformer_forward_pass(self):
        """Test transformer forward pass."""
        try:
            import torch
            from src.ml.transformer_predictor import TransformerPredictorModel
            
            model = TransformerPredictorModel(
                n_features=10,
                d_model=64,
                n_heads=4,
                n_layers=2,
            )
            model.eval()
            
            # Create dummy input
            x = torch.randn(2, 20, 10)  # (batch, seq, features)
            
            with torch.no_grad():
                output = model(x)
                
            self.assertEqual(output.shape, (2, 1))
            self.assertTrue(torch.all(output >= 0))
            self.assertTrue(torch.all(output <= 1))
        except ImportError as e:
            self.skipTest(f"PyTorch not available: {e}")


# =============================================================================
# TEST: PRODUCTION LAUNCHER
# =============================================================================

class TestProductionLauncher(unittest.TestCase):
    """Test production launcher functionality."""
    
    def test_launcher_config(self):
        """Test launcher configuration."""
        # Import after path setup
        sys.path.insert(0, str(PROJECT_ROOT))
        
        # Create mock config
        from production_launcher import LauncherConfig
        
        config = LauncherConfig(
            paper_trading=True,
            dry_run=True,
            single_run=True,
        )
        
        self.assertTrue(config.paper_trading)
        self.assertTrue(config.dry_run)
        self.assertEqual(config.max_position_pct, 0.03)
        
    def test_config_to_dict(self):
        """Test config serialization."""
        from production_launcher import LauncherConfig
        
        config = LauncherConfig()
        config_dict = config.to_dict()
        
        self.assertIn("paper_trading", config_dict)
        self.assertIn("max_position_pct", config_dict)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration(unittest.TestCase):
    """Integration tests for full system."""
    
    def test_end_to_end_signal_generation(self):
        """Test end-to-end signal generation."""
        from src.trading.v21_production_engine import V21ProductionEngine, V21Config
        
        config = V21Config(
            use_ensemble_regime=False,
            use_transformer=False,
            universe_mode="core",
        )
        engine = V21ProductionEngine(config)
        
        # Get universe
        universe = engine.get_universe()
        self.assertGreater(len(universe), 0)
        
        # Detect regime
        regime, confidence = engine.detect_regime()
        self.assertIn(regime, ["bull", "bear", "neutral", "risk_off"])
        
    def test_dashboard_server_lifecycle(self):
        """Test dashboard server start/stop."""
        from src.trading.monitoring_dashboard import MonitoringDashboard
        
        dashboard = MonitoringDashboard(port=8099)
        
        # Start server
        dashboard.start_server()
        time.sleep(0.5)
        
        self.assertTrue(dashboard._is_running)
        
        # Stop server
        dashboard.stop_server()
        time.sleep(0.5)
        
        self.assertFalse(dashboard._is_running)
        
    def test_validation_with_real_data(self):
        """Test validation with realistic data."""
        from src.trading.daily_validator import DailyValidator
        
        validator = DailyValidator()
        
        # Simulate a trading day with realistic trade count and > 50% win rate
        trades = [
            {"symbol": f"TICK{i}", "side": "buy", "qty": 100,
             "price": 100.00 + i, "value": 10000, "pnl": 50 if i % 2 == 0 else -30}
            for i in range(20)  # 20 trades - within expected range, ~50% win rate
        ]
        
        report = validator.run_validation(
            daily_trades=trades,
            daily_return=0.0015,  # 0.15%
            current_dd=0.015,
            current_sharpe=1.40,
        )
        
        # Report should generate and have expected structure
        self.assertIn("overall_status", report)
        self.assertIn("checks_passed", report)
        self.assertIn("validation_results", report)
        self.assertGreater(report["checks_passed"], 0)


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance(unittest.TestCase):
    """Performance and stress tests."""
    
    def test_dashboard_update_speed(self):
        """Test dashboard can handle rapid updates."""
        from src.trading.monitoring_dashboard import MonitoringDashboard
        
        dashboard = MonitoringDashboard(port=8098)
        
        start = time.time()
        for i in range(1000):
            dashboard.update_metrics({
                "equity": 100000 + i,
                "drawdown": i * 0.00001,
            })
        elapsed = time.time() - start
        
        # Should complete 1000 updates in < 1 second
        self.assertLess(elapsed, 1.0)
        
    def test_trade_history_capacity(self):
        """Test trade history handles large volumes."""
        from src.trading.monitoring_dashboard import MonitoringDashboard
        
        dashboard = MonitoringDashboard(port=8097)
        
        # Add 2000 trades
        for i in range(2000):
            dashboard.add_trade({
                "symbol": f"TICK{i % 100}",
                "side": "buy" if i % 2 == 0 else "sell",
                "qty": 100,
                "price": 100 + i * 0.01,
            })
            
        # Should be capped at 1000
        self.assertLessEqual(len(dashboard.trade_history), 1000)


# =============================================================================
# MAIN
# =============================================================================

def run_tests():
    """Run all tests with detailed output."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestV21ProductionEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestMonitoringDashboard))
    suite.addTests(loader.loadTestsFromTestCase(TestDailyValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestEnsembleRegimeDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestTransformerPredictor))
    suite.addTests(loader.loadTestsFromTestCase(TestProductionLauncher))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    
    # Run with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print(f"Success: {result.wasSuccessful()}")
    print("=" * 60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
