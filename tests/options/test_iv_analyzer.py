"""
Test IV Analyzer
================

Tests for IV rank, IV percentile, and regime detection.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

from src.options.iv_analyzer import IVAnalyzer, IVMetrics
from src.options.theta_decay_engine import IVRegime


class TestIVAnalyzer:
    """Test IV analysis functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.analyzer = IVAnalyzer(lookback_days=252)
        self.symbol = "SPY"
    
    def test_update_iv(self):
        """Test adding IV observations."""
        self.analyzer.update(self.symbol, 0.18)
        
        assert self.symbol in self.analyzer.history
        assert len(self.analyzer.history[self.symbol]) == 1
    
    def test_iv_rank_calculation(self):
        """Test IV rank calculation."""
        # Add 50 days of IV data (range: 0.10 to 0.30)
        for i in range(50):
            iv = 0.10 + (i / 50) * 0.20
            self.analyzer.update(self.symbol, iv, timestamp=datetime.now() - timedelta(days=50-i))
        
        # Current IV at midpoint (0.20) should have rank ~50
        iv_rank = self.analyzer.get_iv_rank(self.symbol, 0.20)
        
        assert 45 < iv_rank < 55
        
        # Current IV at maximum should have rank ~100
        iv_rank_high = self.analyzer.get_iv_rank(self.symbol, 0.30)
        assert iv_rank_high > 95
        
        # Current IV at minimum should have rank ~0
        iv_rank_low = self.analyzer.get_iv_rank(self.symbol, 0.10)
        assert iv_rank_low < 5
    
    def test_iv_percentile_calculation(self):
        """Test IV percentile calculation."""
        # Add data
        ivs = [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30]
        for iv in ivs:
            self.analyzer.update(self.symbol, iv)
        
        # Current IV of 0.22 should be ~57th percentile (4 below, 3 above)
        percentile = self.analyzer.get_iv_percentile(self.symbol, 0.22)
        
        assert 50 < percentile < 65
    
    def test_iv_regime_detection(self):
        """Test IV regime classification."""
        # Add historical data
        for i in range(100):
            self.analyzer.update(self.symbol, 0.15 + np.random.random() * 0.15)
        
        # Low IV
        regime_low = self.analyzer.detect_iv_regime(self.symbol, 0.10)
        assert regime_low == IVRegime.LOW
        
        # High IV
        regime_high = self.analyzer.detect_iv_regime(self.symbol, 0.40)
        assert regime_high in [IVRegime.HIGH, IVRegime.EXTREME]
        
        # Normal IV
        regime_normal = self.analyzer.detect_iv_regime(self.symbol, 0.20)
        assert regime_normal in [IVRegime.NORMAL, IVRegime.HIGH]
    
    def test_hv_iv_ratio(self):
        """Test HV/IV ratio calculation."""
        # IV overpriced (HV < IV)
        ratio = self.analyzer.calculate_hv_iv_ratio(
            historical_vol=0.15,
            implied_vol=0.20
        )
        
        assert ratio == 0.75  # HV/IV = 0.15/0.20
        assert ratio < 1.0  # IV is overpriced
        
        # IV underpriced (HV > IV)
        ratio_under = self.analyzer.calculate_hv_iv_ratio(
            historical_vol=0.25,
            implied_vol=0.20
        )
        
        assert ratio_under == 1.25
        assert ratio_under > 1.0  # IV is underpriced
    
    def test_should_sell_premium(self):
        """Test premium selling recommendation."""
        # Setup favorable environment (high IV)
        for i in range(100):
            iv = 0.10 + (i / 100) * 0.20
            self.analyzer.update(self.symbol, iv)
        
        # High IV should recommend selling
        should_sell, reason = self.analyzer.should_sell_premium(
            self.symbol,
            current_iv=0.28,
            min_iv_rank=50
        )
        
        assert should_sell is True
        assert "IV Rank" in reason
        
        # Low IV should not recommend selling
        should_sell_low, reason_low = self.analyzer.should_sell_premium(
            self.symbol,
            current_iv=0.12,
            min_iv_rank=50
        )
        
        assert should_sell_low is False
    
    def test_analyze_complete(self):
        """Test complete IV analysis."""
        # Add historical data
        for i in range(60):
            iv = 0.15 + np.random.random() * 0.10
            self.analyzer.update(
                self.symbol,
                iv,
                timestamp=datetime.now() - timedelta(days=60-i)
            )
        
        # Perform analysis
        metrics = self.analyzer.analyze(
            self.symbol,
            current_iv=0.22,
            historical_vol=0.18
        )
        
        # Check all metrics are populated
        assert metrics.symbol == self.symbol
        assert metrics.current_iv == 0.22
        assert 0 <= metrics.iv_rank <= 100
        assert 0 <= metrics.iv_percentile <= 100
        assert metrics.regime in list(IVRegime)
        assert metrics.hv_iv_ratio is not None
        assert metrics.min_iv_52w > 0
        assert metrics.max_iv_52w > metrics.min_iv_52w
        assert metrics.days_of_data > 0
    
    def test_insufficient_data(self):
        """Test behavior with insufficient data."""
        # Only add a few data points
        self.analyzer.update(self.symbol, 0.20)
        self.analyzer.update(self.symbol, 0.22)
        
        # Should return neutral values
        iv_rank = self.analyzer.get_iv_rank(self.symbol, 0.21)
        assert iv_rank == 50.0  # Default neutral
        
        percentile = self.analyzer.get_iv_percentile(self.symbol, 0.21)
        assert percentile == 50.0
    
    def test_persistence(self, tmp_path):
        """Test saving and loading IV history."""
        # Set custom state file
        state_file = tmp_path / "iv_history_test.json"
        analyzer = IVAnalyzer(state_file=state_file)
        
        # Add data
        analyzer.update("SPY", 0.18)
        analyzer.update("QQQ", 0.22)
        
        # Save
        analyzer.persist_history()
        
        # Load in new instance
        analyzer2 = IVAnalyzer(state_file=state_file)
        
        # Verify data loaded
        assert "SPY" in analyzer2.history
        assert "QQQ" in analyzer2.history
        assert len(analyzer2.history["SPY"]) == 1
    
    def test_cleanup_old_data(self):
        """Test removing old IV data."""
        # Add old and new data
        for i in range(400):  # More than lookback period
            iv = 0.15 + np.random.random() * 0.10
            days_ago = 400 - i
            self.analyzer.update(
                self.symbol,
                iv,
                timestamp=datetime.now() - timedelta(days=days_ago)
            )
        
        # Cleanup to 252 days
        removed = self.analyzer.cleanup_old_data(days_to_keep=252)
        
        # Should have removed ~148 observations
        assert removed > 100
        assert len(self.analyzer.history[self.symbol]) <= 252
    
    def test_statistics(self):
        """Test statistical summary."""
        # Add data
        ivs = [0.15, 0.18, 0.20, 0.22, 0.25, 0.28]
        for iv in ivs:
            self.analyzer.update(self.symbol, iv)
        
        stats = self.analyzer.get_statistics(self.symbol)
        
        assert stats['symbol'] == self.symbol
        assert stats['days_of_data'] == len(ivs)
        assert stats['min_iv'] == 0.15
        assert stats['max_iv'] == 0.28
        assert abs(stats['mean_iv'] - np.mean(ivs)) < 0.01
        assert 'percentiles' in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
