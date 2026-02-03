"""
Unit Tests for Position Sizer Module
=====================================

Tests all position sizing functionality including:
- Kelly Criterion calculation
- Confidence score integration
- Volatility scaling
- Heat adjustment
- Portfolio constraints
- Performance tracking
"""

import unittest
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.position_sizer import (
    PositionSizer,
    SizingConfig,
    SizingMode,
    PerformanceMetrics,
    PositionSize
)


class TestPerformanceMetrics(unittest.TestCase):
    """Test PerformanceMetrics calculations."""
    
    def test_win_rate_calculation(self):
        """Test win rate calculation."""
        metrics = PerformanceMetrics(
            total_trades=100,
            winning_trades=60,
            losing_trades=40,
            total_profit=15000,
            total_loss=-10000
        )
        
        self.assertEqual(metrics.win_rate, 0.6)
    
    def test_avg_win_calculation(self):
        """Test average win calculation."""
        metrics = PerformanceMetrics(
            total_trades=100,
            winning_trades=50,
            losing_trades=50,
            total_profit=10000,
            total_loss=-5000
        )
        
        self.assertEqual(metrics.avg_win, 200.0)
    
    def test_avg_loss_calculation(self):
        """Test average loss calculation."""
        metrics = PerformanceMetrics(
            total_trades=100,
            winning_trades=50,
            losing_trades=50,
            total_profit=10000,
            total_loss=-5000
        )
        
        self.assertEqual(metrics.avg_loss, 100.0)
    
    def test_payoff_ratio(self):
        """Test payoff ratio calculation."""
        metrics = PerformanceMetrics(
            total_trades=100,
            winning_trades=40,
            losing_trades=60,
            total_profit=12000,
            total_loss=-6000
        )
        
        # avg_win = 12000/40 = 300
        # avg_loss = 6000/60 = 100
        # payoff = 300/100 = 3.0
        self.assertAlmostEqual(metrics.payoff_ratio, 3.0, places=2)
    
    def test_expectancy(self):
        """Test expectancy calculation."""
        metrics = PerformanceMetrics(
            total_trades=100,
            winning_trades=55,
            losing_trades=45,
            total_profit=11000,
            total_loss=-9000
        )
        
        # win_rate = 0.55
        # avg_win = 11000/55 = 200
        # avg_loss = 9000/45 = 200
        # expectancy = 0.55 * 200 - 0.45 * 200 = 110 - 90 = 20
        self.assertAlmostEqual(metrics.expectancy, 20.0, places=1)
    
    def test_zero_trades(self):
        """Test handling of zero trades."""
        metrics = PerformanceMetrics(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            total_profit=0,
            total_loss=0
        )
        
        self.assertEqual(metrics.win_rate, 0.0)
        self.assertEqual(metrics.avg_win, 0.0)
        self.assertEqual(metrics.avg_loss, 0.0)


class TestKellyCalculation(unittest.TestCase):
    """Test Kelly Criterion calculations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sizer = PositionSizer()
    
    def test_kelly_basic_calculation(self):
        """Test basic Kelly formula."""
        # 60% win rate, 2:1 payoff ratio
        # Kelly = (0.6 * 2 - 0.4) / 2 = 0.8 / 2 = 0.4
        # Half-Kelly = 0.2
        
        kelly = self.sizer.calculate_kelly_fraction(0.6, 200, 100)
        self.assertAlmostEqual(kelly, 0.2, places=2)
    
    def test_kelly_edge_case_high_win_rate(self):
        """Test Kelly with high win rate."""
        # 80% win rate, 1.5:1 payoff
        kelly = self.sizer.calculate_kelly_fraction(0.8, 150, 100)
        self.assertGreater(kelly, 0.0)
        self.assertLessEqual(kelly, self.sizer.config.max_kelly_fraction)
    
    def test_kelly_edge_case_low_win_rate(self):
        """Test Kelly with low win rate."""
        # 40% win rate, 3:1 payoff needed for positive Kelly
        kelly = self.sizer.calculate_kelly_fraction(0.4, 300, 100)
        self.assertGreater(kelly, 0.0)
    
    def test_kelly_negative_edge(self):
        """Test Kelly with negative edge (should return minimum)."""
        # 40% win rate, 1:1 payoff = negative edge
        kelly = self.sizer.calculate_kelly_fraction(0.4, 100, 100)
        self.assertEqual(kelly, self.sizer.config.min_kelly_fraction)
    
    def test_kelly_respects_min_bound(self):
        """Test Kelly respects minimum bound."""
        kelly = self.sizer.calculate_kelly_fraction(0.51, 100, 100)
        self.assertGreaterEqual(kelly, self.sizer.config.min_kelly_fraction)
    
    def test_kelly_respects_max_bound(self):
        """Test Kelly respects maximum bound."""
        # Unrealistically good metrics
        kelly = self.sizer.calculate_kelly_fraction(0.9, 1000, 100)
        self.assertLessEqual(kelly, self.sizer.config.max_kelly_fraction)
    
    def test_full_kelly_mode(self):
        """Test full Kelly mode (multiplier = 1.0)."""
        config = SizingConfig(
            sizing_mode=SizingMode.FULL_KELLY,
            kelly_multiplier=1.0
        )
        sizer = PositionSizer(config)
        
        kelly = sizer.calculate_kelly_fraction(0.6, 200, 100)
        # Full Kelly should be 2x half-Kelly
        half_kelly = self.sizer.calculate_kelly_fraction(0.6, 200, 100)
        self.assertAlmostEqual(kelly, half_kelly * 2, places=2)
    
    def test_quarter_kelly_mode(self):
        """Test quarter Kelly mode."""
        config = SizingConfig(
            sizing_mode=SizingMode.QUARTER_KELLY,
            kelly_multiplier=0.25
        )
        sizer = PositionSizer(config)
        
        kelly = sizer.calculate_kelly_fraction(0.6, 200, 100)
        self.assertLess(kelly, 0.15)  # Should be quite conservative


class TestVolatilityScaling(unittest.TestCase):
    """Test volatility-based position scaling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sizer = PositionSizer()
    
    def test_low_volatility_scales_up(self):
        """Test position scales up in low volatility."""
        historical = np.array([2.0, 2.5, 3.0, 3.5, 4.0] * 10)  # 50 data points
        current = 1.5  # Below 25th percentile
        
        scalar = self.sizer.calculate_volatility_scalar(current, historical)
        self.assertGreater(scalar, 1.0)
        self.assertLessEqual(scalar, 1.5)
    
    def test_high_volatility_scales_down(self):
        """Test position scales down in high volatility."""
        historical = np.array([2.0, 2.5, 3.0, 3.5, 4.0] * 10)
        current = 5.0  # Above 75th percentile
        
        scalar = self.sizer.calculate_volatility_scalar(current, historical)
        self.assertLess(scalar, 1.0)
        self.assertGreaterEqual(scalar, 0.5)
    
    def test_normal_volatility_no_adjustment(self):
        """Test no scaling for normal volatility."""
        historical = np.array([2.0, 2.5, 3.0, 3.5, 4.0] * 10)
        current = 3.0  # Median
        
        scalar = self.sizer.calculate_volatility_scalar(current, historical)
        self.assertAlmostEqual(scalar, 1.0, places=1)
    
    def test_volatility_scaling_disabled(self):
        """Test volatility scaling can be disabled."""
        config = SizingConfig(use_volatility_scaling=False)
        sizer = PositionSizer(config)
        
        historical = np.array([1.0, 2.0, 3.0])
        current = 10.0  # Very high
        
        scalar = sizer.calculate_volatility_scalar(current, historical)
        self.assertEqual(scalar, 1.0)
    
    def test_insufficient_data(self):
        """Test handling of insufficient volatility data."""
        historical = np.array([1.0, 2.0])  # Only 2 points
        current = 5.0
        
        scalar = self.sizer.calculate_volatility_scalar(current, historical)
        self.assertEqual(scalar, 1.0)


class TestConfidenceScaling(unittest.TestCase):
    """Test confidence score integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sizer = PositionSizer()
    
    def test_high_confidence_full_scale(self):
        """Test high confidence gives full scaling."""
        scalar = self.sizer.calculate_confidence_scalar(1.0)
        self.assertEqual(scalar, 1.0)
    
    def test_low_confidence_reduced_scale(self):
        """Test low confidence reduces scaling."""
        scalar = self.sizer.calculate_confidence_scalar(0.5)
        self.assertLess(scalar, 0.5)
        self.assertGreater(scalar, 0.0)
    
    def test_below_minimum_confidence_rejected(self):
        """Test confidence below minimum is rejected."""
        scalar = self.sizer.calculate_confidence_scalar(0.2)  # Below default 0.3
        self.assertEqual(scalar, 0.0)
    
    def test_confidence_power_effect(self):
        """Test confidence power function."""
        # With power > 1, should penalize moderate confidence
        scalar_60 = self.sizer.calculate_confidence_scalar(0.6)
        scalar_80 = self.sizer.calculate_confidence_scalar(0.8)
        
        # Ratio should be more than linear
        ratio = scalar_80 / scalar_60
        self.assertGreater(ratio, 1.3)


class TestHeatAdjustment(unittest.TestCase):
    """Test heat adjustment for losing streaks."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sizer = PositionSizer()
    
    def test_no_heat_initially(self):
        """Test no heat adjustment initially."""
        heat = self.sizer.calculate_heat_adjustment()
        self.assertEqual(heat, 1.0)
    
    def test_heat_after_max_losses(self):
        """Test heat kicks in after max consecutive losses."""
        # Record max losses
        for _ in range(self.sizer.config.max_consecutive_losses):
            self.sizer.record_trade_result(is_winner=False)
        
        heat = self.sizer.calculate_heat_adjustment()
        self.assertEqual(heat, self.sizer.config.heat_reduction_factor)
    
    def test_heat_resets_on_win(self):
        """Test heat resets after a win."""
        # Build up heat
        for _ in range(5):
            self.sizer.record_trade_result(is_winner=False)
        
        # Win should reset
        self.sizer.record_trade_result(is_winner=True)
        heat = self.sizer.calculate_heat_adjustment()
        self.assertEqual(heat, 1.0)
    
    def test_heat_disabled(self):
        """Test heat adjustment can be disabled."""
        config = SizingConfig(use_heat_adjustment=False)
        sizer = PositionSizer(config)
        
        for _ in range(10):
            sizer.record_trade_result(is_winner=False)
        
        heat = sizer.calculate_heat_adjustment()
        self.assertEqual(heat, 1.0)
    
    def test_manual_heat_reset(self):
        """Test manual heat reset."""
        for _ in range(5):
            self.sizer.record_trade_result(is_winner=False)
        
        self.sizer.reset_heat()
        self.assertEqual(self.sizer.consecutive_losses, 0)


class TestPositionSizing(unittest.TestCase):
    """Test complete position sizing logic."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sizer = PositionSizer()
        self.portfolio_value = 100000
        
        self.metrics = PerformanceMetrics(
            total_trades=100,
            winning_trades=60,
            losing_trades=40,
            total_profit=15000,
            total_loss=-10000
        )
    
    def test_basic_position_sizing(self):
        """Test basic position sizing."""
        position = self.sizer.size_position(
            self.portfolio_value,
            confidence=0.8,
            volatility_percentile=50.0,
            performance_metrics=self.metrics
        )
        
        self.assertTrue(position.is_valid)
        self.assertGreater(position.position_value, 0)
        self.assertLessEqual(position.position_pct, self.sizer.config.max_position_pct)
    
    def test_low_confidence_rejection(self):
        """Test low confidence rejects position."""
        position = self.sizer.size_position(
            self.portfolio_value,
            confidence=0.2,  # Below minimum
            performance_metrics=self.metrics
        )
        
        self.assertFalse(position.is_valid)
        self.assertIn("Confidence", position.rejection_reason)
    
    def test_minimum_position_value(self):
        """Test minimum position value enforcement."""
        small_portfolio = 1000
        position = self.sizer.size_position(
            small_portfolio,
            confidence=0.5,
            performance_metrics=self.metrics
        )
        
        # With small portfolio and moderate confidence, might hit minimum
        if not position.is_valid:
            self.assertIn("minimum", position.rejection_reason.lower())
    
    def test_maximum_position_percentage(self):
        """Test maximum position percentage cap."""
        position = self.sizer.size_position(
            self.portfolio_value,
            confidence=1.0,  # Maximum confidence
            volatility_percentile=10.0,  # Low volatility (scales up)
            performance_metrics=self.metrics
        )
        
        self.assertLessEqual(position.position_pct, self.sizer.config.max_position_pct)
    
    def test_high_volatility_reduces_size(self):
        """Test high volatility reduces position size."""
        normal_vol = self.sizer.size_position(
            self.portfolio_value,
            confidence=0.8,
            volatility_percentile=50.0,
            performance_metrics=self.metrics
        )
        
        high_vol = self.sizer.size_position(
            self.portfolio_value,
            confidence=0.8,
            volatility_percentile=90.0,
            performance_metrics=self.metrics
        )
        
        self.assertLess(high_vol.position_value, normal_vol.position_value)
    
    def test_sizing_factors_included(self):
        """Test sizing factors are included in result."""
        position = self.sizer.size_position(
            self.portfolio_value,
            confidence=0.8,
            performance_metrics=self.metrics
        )
        
        self.assertIn("kelly_fraction", position.sizing_factors)
        self.assertIn("confidence_scalar", position.sizing_factors)
        self.assertIn("win_rate", position.sizing_factors)
        self.assertIn("payoff_ratio", position.sizing_factors)
    
    def test_heat_reduces_size(self):
        """Test heat adjustment reduces position size."""
        # Size without heat
        normal = self.sizer.size_position(
            self.portfolio_value,
            confidence=0.8,
            performance_metrics=self.metrics
        )
        
        # Build up heat
        for _ in range(self.sizer.config.max_consecutive_losses):
            self.sizer.record_trade_result(is_winner=False)
        
        # Size with heat
        heated = self.sizer.size_position(
            self.portfolio_value,
            confidence=0.8,
            performance_metrics=self.metrics
        )
        
        self.assertLess(heated.position_value, normal.position_value)


class TestSizingConfig(unittest.TestCase):
    """Test SizingConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = SizingConfig()
        
        self.assertEqual(config.sizing_mode, SizingMode.HALF_KELLY)
        self.assertEqual(config.kelly_multiplier, 0.5)
        self.assertEqual(config.max_position_pct, 0.10)
        self.assertEqual(config.min_position_value, 100.0)
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = SizingConfig(
            sizing_mode=SizingMode.FULL_KELLY,
            kelly_multiplier=1.0,
            max_position_pct=0.20,
            use_heat_adjustment=False
        )
        
        self.assertEqual(config.sizing_mode, SizingMode.FULL_KELLY)
        self.assertEqual(config.kelly_multiplier, 1.0)
        self.assertEqual(config.max_position_pct, 0.20)
        self.assertFalse(config.use_heat_adjustment)


class TestPerformanceTracking(unittest.TestCase):
    """Test performance metrics updating."""
    
    def test_update_metrics(self):
        """Test updating performance metrics."""
        sizer = PositionSizer()
        
        new_metrics = PerformanceMetrics(
            total_trades=200,
            winning_trades=120,
            losing_trades=80,
            total_profit=30000,
            total_loss=-20000
        )
        
        sizer.update_performance_metrics(new_metrics)
        self.assertEqual(sizer.default_metrics, new_metrics)
    
    def test_trade_result_tracking(self):
        """Test trade result tracking."""
        sizer = PositionSizer()
        
        initial_trades = sizer.total_trades
        sizer.record_trade_result(is_winner=True)
        
        self.assertEqual(sizer.total_trades, initial_trades + 1)
        self.assertEqual(sizer.consecutive_losses, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
