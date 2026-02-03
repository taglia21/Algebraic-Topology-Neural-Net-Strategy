"""
Unit Tests for Multi-Timeframe Analyzer Module
===============================================

Tests all multi-timeframe analysis functionality including:
- Data fetching and caching
- Technical indicator calculations (EMA, RSI, MACD)
- Timeframe signal generation
- Alignment score calculation
- Trend determination
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.multi_timeframe_analyzer import (
    MultiTimeframeAnalyzer,
    AnalyzerConfig,
    Timeframe,
    TrendDirection,
    TimeframeSignals,
    TimeframeAnalysis
)


class TestTimeframeEnum(unittest.TestCase):
    """Test Timeframe enum."""
    
    def test_timeframe_properties(self):
        """Test timeframe enum properties."""
        tf = Timeframe.H1
        self.assertEqual(tf.interval, "1h")
        self.assertEqual(tf.description, "1 hour")
        self.assertEqual(tf.weight, 3)
    
    def test_timeframe_weights(self):
        """Test higher timeframes have higher weights."""
        self.assertLess(Timeframe.M5.weight, Timeframe.H1.weight)
        self.assertLess(Timeframe.H1.weight, Timeframe.D1.weight)


class TestTimeframeSignals(unittest.TestCase):
    """Test TimeframeSignals dataclass."""
    
    def test_bullish_detection(self):
        """Test bullish signal detection."""
        signals = TimeframeSignals(
            timeframe=Timeframe.H1,
            ema_cross=0.5,
            rsi_position=0.3,
            macd_histogram=0.4,
            trend_score=0.4
        )
        self.assertTrue(signals.is_bullish)
        self.assertFalse(signals.is_bearish)
    
    def test_bearish_detection(self):
        """Test bearish signal detection."""
        signals = TimeframeSignals(
            timeframe=Timeframe.H1,
            ema_cross=-0.5,
            rsi_position=-0.3,
            macd_histogram=-0.4,
            trend_score=-0.4
        )
        self.assertTrue(signals.is_bearish)
        self.assertFalse(signals.is_bullish)
    
    def test_neutral_detection(self):
        """Test neutral signal detection."""
        signals = TimeframeSignals(
            timeframe=Timeframe.H1,
            ema_cross=0.1,
            rsi_position=0.0,
            macd_histogram=-0.1,
            trend_score=0.0
        )
        self.assertFalse(signals.is_bullish)
        self.assertFalse(signals.is_bearish)


class TestTimeframeAnalysis(unittest.TestCase):
    """Test TimeframeAnalysis dataclass."""
    
    def test_alignment_detection(self):
        """Test strong alignment detection."""
        analysis = TimeframeAnalysis(
            symbol="TEST",
            timestamp=datetime.now(),
            signals={},
            alignment_score=75.0,
            dominant_trend=TrendDirection.BULLISH,
            bullish_timeframes=4,
            bearish_timeframes=0,
            neutral_timeframes=1
        )
        self.assertTrue(analysis.is_aligned)
        self.assertTrue(analysis.is_tradeable)
    
    def test_weak_alignment(self):
        """Test weak alignment detection."""
        analysis = TimeframeAnalysis(
            symbol="TEST",
            timestamp=datetime.now(),
            signals={},
            alignment_score=50.0,
            dominant_trend=TrendDirection.NEUTRAL,
            bullish_timeframes=2,
            bearish_timeframes=2,
            neutral_timeframes=1
        )
        self.assertFalse(analysis.is_aligned)
        self.assertFalse(analysis.is_tradeable)


class TestIndicatorCalculations(unittest.TestCase):
    """Test technical indicator calculations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = MultiTimeframeAnalyzer()
        
        # Create sample price data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        prices = 100 + np.cumsum(np.random.randn(100) * 2)
        
        self.sample_data = pd.DataFrame({
            'Open': prices,
            'High': prices + np.random.rand(100) * 2,
            'Low': prices - np.random.rand(100) * 2,
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
    
    def test_ema_calculation(self):
        """Test EMA calculation."""
        ema = self.analyzer._calculate_ema(self.sample_data, 8)
        
        self.assertEqual(len(ema), len(self.sample_data))
        self.assertFalse(ema.isna().all())
        
        # EMA should be smoother than price
        self.assertLess(ema.std(), self.sample_data['Close'].std())
    
    def test_rsi_calculation(self):
        """Test RSI calculation."""
        rsi = self.analyzer._calculate_rsi(self.sample_data, 14)
        
        self.assertEqual(len(rsi), len(self.sample_data))
        
        # RSI should be between 0 and 100 (after warm-up)
        rsi_valid = rsi.dropna()
        self.assertTrue((rsi_valid >= 0).all())
        self.assertTrue((rsi_valid <= 100).all())
    
    def test_macd_calculation(self):
        """Test MACD calculation."""
        macd_line, signal_line, histogram = self.analyzer._calculate_macd(self.sample_data)
        
        self.assertEqual(len(macd_line), len(self.sample_data))
        self.assertEqual(len(signal_line), len(self.sample_data))
        self.assertEqual(len(histogram), len(self.sample_data))
        
        # Histogram should equal MACD - Signal
        diff = macd_line - signal_line
        np.testing.assert_array_almost_equal(
            histogram.dropna().values,
            diff.dropna().values,
            decimal=6
        )
    
    def test_ema_crossover_logic(self):
        """Test EMA crossover generates correct signals."""
        # Fast EMA above slow = bullish
        ema_fast = self.analyzer._calculate_ema(self.sample_data, 8)
        ema_slow = self.analyzer._calculate_ema(self.sample_data, 21)
        
        # Both EMAs should exist
        self.assertFalse(ema_fast.isna().all())
        self.assertFalse(ema_slow.isna().all())


class TestTimeframeAnalysis(unittest.TestCase):
    """Test timeframe analysis logic."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = MultiTimeframeAnalyzer()
    
    def create_mock_data(self, trend: str = 'bullish') -> pd.DataFrame:
        """Create mock price data with specific trend."""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        
        if trend == 'bullish':
            prices = 100 + np.arange(100) * 0.5 + np.random.randn(100) * 0.5
        elif trend == 'bearish':
            prices = 150 - np.arange(100) * 0.5 + np.random.randn(100) * 0.5
        else:  # sideways
            prices = 100 + np.random.randn(100) * 2
        
        return pd.DataFrame({
            'Open': prices,
            'High': prices + np.abs(np.random.randn(100)),
            'Low': prices - np.abs(np.random.randn(100)),
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
    
    @patch('yfinance.Ticker')
    def test_bullish_trend_detection(self, mock_ticker):
        """Test detection of bullish trend."""
        # Mock yfinance to return bullish data
        mock_ticker.return_value.history.return_value = self.create_mock_data('bullish')
        
        signals = self.analyzer._analyze_timeframe('TEST', Timeframe.D1)
        
        if signals:
            # In bullish trend, expect positive trend score
            self.assertGreater(signals.trend_score, 0)
    
    @patch('yfinance.Ticker')
    def test_bearish_trend_detection(self, mock_ticker):
        """Test detection of bearish trend."""
        mock_ticker.return_value.history.return_value = self.create_mock_data('bearish')
        
        signals = self.analyzer._analyze_timeframe('TEST', Timeframe.D1)
        
        if signals:
            # In bearish trend, expect negative trend score
            self.assertLess(signals.trend_score, 0)
    
    @patch('yfinance.Ticker')
    def test_handles_insufficient_data(self, mock_ticker):
        """Test handling of insufficient data."""
        # Return very little data
        mock_ticker.return_value.history.return_value = pd.DataFrame()
        
        signals = self.analyzer._analyze_timeframe('TEST', Timeframe.D1)
        self.assertIsNone(signals)
    
    @patch('yfinance.Ticker')
    def test_handles_fetch_error(self, mock_ticker):
        """Test handling of data fetch errors."""
        mock_ticker.return_value.history.side_effect = Exception("Network error")
        
        signals = self.analyzer._analyze_timeframe('TEST', Timeframe.D1)
        self.assertIsNone(signals)


class TestAlignmentScoring(unittest.TestCase):
    """Test alignment score calculation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = MultiTimeframeAnalyzer()
    
    def test_perfect_bullish_alignment(self):
        """Test perfect bullish alignment scores high."""
        signals = {
            Timeframe.M5: TimeframeSignals(Timeframe.M5, 0.8, 0.6, 0.7, 0.7),
            Timeframe.M15: TimeframeSignals(Timeframe.M15, 0.7, 0.5, 0.6, 0.6),
            Timeframe.H1: TimeframeSignals(Timeframe.H1, 0.9, 0.7, 0.8, 0.8),
            Timeframe.H4: TimeframeSignals(Timeframe.H4, 0.8, 0.6, 0.7, 0.7),
            Timeframe.D1: TimeframeSignals(Timeframe.D1, 0.9, 0.8, 0.9, 0.85),
        }
        
        score = self.analyzer._calculate_alignment_score(signals)
        self.assertGreater(score, 70)  # Strong alignment threshold
    
    def test_perfect_bearish_alignment(self):
        """Test perfect bearish alignment scores low."""
        signals = {
            Timeframe.M5: TimeframeSignals(Timeframe.M5, -0.8, -0.6, -0.7, -0.7),
            Timeframe.M15: TimeframeSignals(Timeframe.M15, -0.7, -0.5, -0.6, -0.6),
            Timeframe.H1: TimeframeSignals(Timeframe.H1, -0.9, -0.7, -0.8, -0.8),
        }
        
        score = self.analyzer._calculate_alignment_score(signals)
        self.assertLess(score, 30)  # Should be bearish
    
    def test_mixed_signals(self):
        """Test mixed signals result in neutral score."""
        signals = {
            Timeframe.M5: TimeframeSignals(Timeframe.M5, 0.5, 0.3, 0.4, 0.4),
            Timeframe.H1: TimeframeSignals(Timeframe.H1, -0.5, -0.3, -0.4, -0.4),
            Timeframe.D1: TimeframeSignals(Timeframe.D1, 0.1, 0.0, 0.0, 0.05),
        }
        
        score = self.analyzer._calculate_alignment_score(signals)
        self.assertGreater(score, 40)
        self.assertLess(score, 60)  # Should be neutral
    
    def test_weighted_vs_unweighted(self):
        """Test weighted scoring gives more weight to higher timeframes."""
        signals = {
            Timeframe.M5: TimeframeSignals(Timeframe.M5, -0.8, -0.6, -0.7, -0.7),  # Bearish
            Timeframe.D1: TimeframeSignals(Timeframe.D1, 0.8, 0.6, 0.7, 0.7),  # Bullish
        }
        
        # With weighting, D1 should dominate
        weighted_score = self.analyzer._calculate_alignment_score(signals)
        
        # Without weighting
        config = AnalyzerConfig(use_weighted_scoring=False)
        unweighted_analyzer = MultiTimeframeAnalyzer(config)
        unweighted_score = unweighted_analyzer._calculate_alignment_score(signals)
        
        # Weighted should be more bullish (D1 has higher weight)
        self.assertGreater(weighted_score, unweighted_score)
    
    def test_empty_signals(self):
        """Test handling of empty signals."""
        score = self.analyzer._calculate_alignment_score({})
        self.assertEqual(score, 0.0)
    
    def test_score_bounds(self):
        """Test score is always between 0 and 100."""
        # Extreme bullish
        signals = {
            Timeframe.D1: TimeframeSignals(Timeframe.D1, 1.0, 1.0, 1.0, 1.0),
        }
        score = self.analyzer._calculate_alignment_score(signals)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)


class TestTrendDetermination(unittest.TestCase):
    """Test trend direction determination."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = MultiTimeframeAnalyzer()
    
    def test_strong_bullish_trend(self):
        """Test strong bullish trend detection."""
        signals = {
            Timeframe.D1: TimeframeSignals(Timeframe.D1, 0.8, 0.6, 0.7, 0.7),
        }
        
        trend = self.analyzer._determine_trend(80, signals)
        self.assertEqual(trend, TrendDirection.STRONG_BULLISH)
    
    def test_weak_bullish_trend(self):
        """Test weak bullish trend detection."""
        signals = {
            Timeframe.D1: TimeframeSignals(Timeframe.D1, 0.3, 0.2, 0.25, 0.25),
        }
        
        trend = self.analyzer._determine_trend(60, signals)
        self.assertEqual(trend, TrendDirection.BULLISH)
    
    def test_strong_bearish_trend(self):
        """Test strong bearish trend detection."""
        signals = {
            Timeframe.D1: TimeframeSignals(Timeframe.D1, -0.8, -0.6, -0.7, -0.7),
        }
        
        trend = self.analyzer._determine_trend(20, signals)
        self.assertEqual(trend, TrendDirection.STRONG_BEARISH)
    
    def test_neutral_trend(self):
        """Test neutral trend detection."""
        signals = {
            Timeframe.D1: TimeframeSignals(Timeframe.D1, 0.1, 0.0, -0.05, 0.02),
        }
        
        trend = self.analyzer._determine_trend(50, signals)
        self.assertEqual(trend, TrendDirection.NEUTRAL)


class TestCaching(unittest.TestCase):
    """Test caching functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = MultiTimeframeAnalyzer()
    
    @patch.object(MultiTimeframeAnalyzer, '_analyze_timeframe')
    def test_cache_hit(self, mock_analyze):
        """Test cache returns stored results."""
        # Create mock analysis
        mock_analysis = TimeframeAnalysis(
            symbol="TEST",
            timestamp=datetime.now(),
            signals={},
            alignment_score=75.0,
            dominant_trend=TrendDirection.BULLISH,
            bullish_timeframes=4,
            bearish_timeframes=0,
            neutral_timeframes=1
        )
        
        # Manually add to cache
        self.analyzer.cache["TEST"] = (mock_analysis, datetime.now())
        
        # Analyze should return cached result
        result = self.analyzer.analyze("TEST", use_cache=True)
        
        # Should not have called _analyze_timeframe
        mock_analyze.assert_not_called()
        self.assertEqual(result.symbol, "TEST")
    
    def test_cache_expiry(self):
        """Test cache expires after TTL."""
        mock_analysis = TimeframeAnalysis(
            symbol="TEST",
            timestamp=datetime.now(),
            signals={},
            alignment_score=75.0,
            dominant_trend=TrendDirection.BULLISH,
            bullish_timeframes=4,
            bearish_timeframes=0,
            neutral_timeframes=1
        )
        
        # Add old cache entry
        old_time = datetime.now() - timedelta(seconds=self.analyzer.config.cache_ttl_seconds + 10)
        self.analyzer.cache["TEST"] = (mock_analysis, old_time)
        
        # Should not use expired cache
        with patch.object(self.analyzer, '_analyze_timeframe', return_value=None):
            result = self.analyzer.analyze("TEST", use_cache=True)
    
    def test_clear_cache_specific(self):
        """Test clearing specific symbol from cache."""
        self.analyzer.cache["TEST1"] = (Mock(), datetime.now())
        self.analyzer.cache["TEST2"] = (Mock(), datetime.now())
        
        self.analyzer.clear_cache("TEST1")
        
        self.assertNotIn("TEST1", self.analyzer.cache)
        self.assertIn("TEST2", self.analyzer.cache)
    
    def test_clear_cache_all(self):
        """Test clearing entire cache."""
        self.analyzer.cache["TEST1"] = (Mock(), datetime.now())
        self.analyzer.cache["TEST2"] = (Mock(), datetime.now())
        
        self.analyzer.clear_cache()
        
        self.assertEqual(len(self.analyzer.cache), 0)


class TestAnalyzerConfig(unittest.TestCase):
    """Test AnalyzerConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = AnalyzerConfig()
        
        self.assertEqual(len(config.timeframes), 5)
        self.assertEqual(config.ema_fast, 8)
        self.assertEqual(config.ema_slow, 21)
        self.assertEqual(config.rsi_period, 14)
        self.assertEqual(config.cache_ttl_seconds, 60)
    
    def test_custom_config(self):
        """Test custom configuration."""
        custom_timeframes = [Timeframe.H1, Timeframe.D1]
        
        config = AnalyzerConfig(
            timeframes=custom_timeframes,
            ema_fast=10,
            cache_ttl_seconds=120
        )
        
        self.assertEqual(len(config.timeframes), 2)
        self.assertEqual(config.ema_fast, 10)
        self.assertEqual(config.cache_ttl_seconds, 120)


if __name__ == '__main__':
    unittest.main(verbosity=2)
