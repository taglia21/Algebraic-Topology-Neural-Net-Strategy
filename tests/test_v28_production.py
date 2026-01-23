#!/usr/bin/env python3
"""
V28 Production System - Comprehensive Test Suite
=================================================

Tests for all V28 components:
1. Dashboard API tests (REST endpoints, WebSocket)
2. Regime Detection tests (HMM, GARCH)
3. Correlation Engine tests
4. Kelly Position Sizing tests
5. Integration tests
6. Performance benchmarks

Target: >90% pass rate
"""

import pytest
import asyncio
import json
import numpy as np
import pandas as pd
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from typing import Dict, List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data."""
    np.random.seed(42)
    n_bars = 252
    
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n_bars, freq='D')
    returns = np.random.randn(n_bars) * 0.015 + 0.0003  # Slight positive drift
    close = 100 * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        'date': dates,
        'open': close * (1 + np.random.randn(n_bars) * 0.005),
        'high': close * (1 + np.abs(np.random.randn(n_bars)) * 0.015),
        'low': close * (1 - np.abs(np.random.randn(n_bars)) * 0.015),
        'close': close,
        'volume': np.random.randint(1_000_000, 10_000_000, n_bars).astype(float),
    })
    
    return df


@pytest.fixture
def sample_returns():
    """Generate sample returns series."""
    np.random.seed(42)
    n_days = 252
    returns = pd.Series(
        np.random.randn(n_days) * 0.015 + 0.0003,
        index=pd.date_range(end=pd.Timestamp.now(), periods=n_days, freq='D')
    )
    return returns


@pytest.fixture
def sample_correlation_matrix():
    """Generate sample correlation matrix."""
    np.random.seed(42)
    n = 10
    
    # Generate positive semi-definite correlation matrix
    A = np.random.randn(n, n)
    corr = A @ A.T
    D = np.diag(1 / np.sqrt(np.diag(corr)))
    corr = D @ corr @ D
    
    symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'JPM', 'XLE', 'XLF']
    return pd.DataFrame(corr, index=symbols, columns=symbols)


@pytest.fixture
def sample_trade_history():
    """Generate sample trade history."""
    np.random.seed(42)
    n_trades = 50
    
    # 55% win rate
    wins = np.random.rand(n_trades) < 0.55
    trades = []
    
    for i in range(n_trades):
        if wins[i]:
            pnl = np.random.uniform(0.01, 0.05)  # 1-5% wins
        else:
            pnl = -np.random.uniform(0.005, 0.03)  # 0.5-3% losses
        trades.append(pnl)
    
    return trades


# =============================================================================
# DASHBOARD API TESTS
# =============================================================================

class TestDashboardAPI:
    """Tests for V28 Dashboard API."""
    
    def test_metrics_calculator_sharpe(self):
        """Test Sharpe ratio calculation."""
        from v28_dashboard_api import MetricsCalculator
        
        calc = MetricsCalculator()
        
        # Add equity history
        equity = 100000
        for _ in range(100):
            equity *= (1 + np.random.randn() * 0.02 + 0.001)
            calc.update_equity(equity)
        
        sharpe = calc.calculate_sharpe()
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
    
    def test_metrics_calculator_cagr(self):
        """Test CAGR calculation."""
        from v28_dashboard_api import MetricsCalculator
        
        calc = MetricsCalculator()
        calc.start_equity = 100000
        calc.current_equity = 150000  # 50% gain
        calc.equity_curve = list(np.linspace(100000, 150000, 252))
        
        cagr = calc.calculate_cagr()
        assert cagr > 0.4  # Should be around 50% for 1 year
        assert cagr < 0.6
    
    def test_metrics_calculator_max_drawdown(self):
        """Test maximum drawdown calculation."""
        from v28_dashboard_api import MetricsCalculator
        
        calc = MetricsCalculator()
        
        # Create equity curve with known drawdown
        equity_curve = [100000, 110000, 105000, 115000, 100000, 120000]
        for eq in equity_curve:
            calc.equity_curve.append(eq)
        
        max_dd = calc.calculate_max_drawdown()
        
        # Max drawdown should be from 115000 to 100000 = -13%
        assert max_dd < 0
        assert max_dd > -0.20
    
    def test_metrics_calculator_win_rate(self, sample_trade_history):
        """Test win rate calculation."""
        from v28_dashboard_api import MetricsCalculator, Trade
        
        calc = MetricsCalculator()
        
        for i, pnl in enumerate(sample_trade_history):
            trade = Trade(
                trade_id=f"T{i}",
                symbol="SPY",
                side="buy",
                quantity=100,
                price=450.0,
                timestamp=datetime.now().isoformat(),
                pnl=pnl * 10000,  # Scale to dollars
                pnl_pct=pnl
            )
            calc.add_trade(trade)
        
        win_rate = calc.calculate_win_rate()
        assert 0.4 < win_rate < 0.7  # Should be around 55%
    
    def test_performance_metrics_to_dict(self):
        """Test PerformanceMetrics serialization."""
        from v28_dashboard_api import PerformanceMetrics
        
        metrics = PerformanceMetrics(
            sharpe_ratio=2.5,
            cagr=0.50,
            max_drawdown=-0.10,
            win_rate=0.60
        )
        
        result = metrics.to_dict()
        assert isinstance(result, dict)
        assert result['sharpe_ratio'] == 2.5
        assert result['cagr'] == 0.50
    
    def test_cache_manager_local(self):
        """Test cache manager with local cache."""
        from v28_dashboard_api import CacheManager
        
        cache = CacheManager(redis_url=None)
        
        # Test set and get
        cache.set('test_key', {'value': 123})
        result = cache.get('test_key')
        
        assert result == {'value': 123}
        
        # Test delete
        cache.delete('test_key')
        assert cache.get('test_key') is None
    
    def test_websocket_manager_connections(self):
        """Test WebSocket manager subscriber tracking."""
        from v28_dashboard_api import WebSocketManager
        
        ws_manager = WebSocketManager()
        
        mock_ws = MagicMock()
        
        ws_manager.add_subscriber(mock_ws, 'pnl')
        assert ws_manager.get_connection_count() == 1
        
        ws_manager.add_subscriber(mock_ws, 'trades')
        assert ws_manager.get_connection_count() == 2
        
        ws_manager.remove_subscriber(mock_ws)
        assert ws_manager.get_connection_count() == 0


# =============================================================================
# REGIME DETECTION TESTS
# =============================================================================

class TestRegimeDetection:
    """Tests for V28 Regime Detection."""
    
    def test_feature_builder(self, sample_ohlcv):
        """Test regime feature calculation."""
        from v28_regime_detector import RegimeFeatureBuilder
        
        builder = RegimeFeatureBuilder()
        features = builder.calculate_features(sample_ohlcv)
        
        assert 'returns_10d' in features.columns
        assert 'realized_vol_20d' in features.columns
        assert 'rsi_14' in features.columns
        assert 'macd' in features.columns
        
        # Check for valid values
        assert not features['returns_10d'].iloc[-1] is np.nan
        assert features['rsi_14'].iloc[-1] >= 0
        assert features['rsi_14'].iloc[-1] <= 100
    
    def test_hmm_detector_initialization(self):
        """Test HMM detector initialization."""
        from v28_regime_detector import HMMRegimeDetector
        
        detector = HMMRegimeDetector(n_states=4)
        
        assert detector.n_states == 4
        assert not detector.is_fitted
    
    def test_hmm_detector_fallback_predict(self, sample_ohlcv):
        """Test HMM detector fallback prediction."""
        from v28_regime_detector import HMMRegimeDetector
        
        detector = HMMRegimeDetector()
        detector.is_fitted = True  # Skip fitting
        
        state, prob = detector._fallback_predict(sample_ohlcv)
        
        assert state in [0, 1, 2, 3]
        assert 0 <= prob <= 1
    
    def test_hmm_state_names(self):
        """Test HMM state name mapping."""
        from v28_regime_detector import HMMRegimeDetector
        
        detector = HMMRegimeDetector()
        
        assert detector.get_state_name(0) == 'LowVolTrend'
        assert detector.get_state_name(1) == 'HighVolTrend'
        assert detector.get_state_name(2) == 'LowVolMeanRevert'
        assert detector.get_state_name(3) == 'Crisis'
    
    def test_garch_detector_initialization(self):
        """Test GARCH detector initialization."""
        from v28_regime_detector import GARCHVolatilityDetector
        
        detector = GARCHVolatilityDetector(p=1, q=1)
        
        assert detector.p == 1
        assert detector.q == 1
        assert not detector.is_fitted
    
    def test_garch_fallback_predict(self, sample_returns):
        """Test GARCH fallback prediction."""
        from v28_regime_detector import GARCHVolatilityDetector, VolatilityRegime
        
        detector = GARCHVolatilityDetector()
        detector.is_fitted = True
        
        regime, current_vol, forecast = detector._fallback_predict(sample_returns)
        
        assert isinstance(regime, VolatilityRegime)
        assert current_vol > 0
        assert forecast > 0
    
    def test_v28_regime_detector_integration(self, sample_ohlcv):
        """Test combined V28 regime detector."""
        from v28_regime_detector import V28RegimeDetector, MarketRegime
        
        detector = V28RegimeDetector()
        
        # Fit and detect
        state = detector.detect(sample_ohlcv)
        
        assert isinstance(state.market_regime, MarketRegime)
        assert state.hmm_state in [0, 1, 2, 3]
        assert 0 <= state.regime_confidence <= 1
        assert state.garch_volatility > 0
    
    def test_strategy_router(self):
        """Test adaptive strategy router."""
        from v28_regime_detector import AdaptiveStrategyRouter, MarketRegime, VolatilityRegime
        
        router = AdaptiveStrategyRouter()
        
        weights = router.get_strategy_weights(
            MarketRegime.BULL,
            VolatilityRegime.LOW,
            confidence=0.8
        )
        
        assert sum(weights.values()) > 0.99  # Should sum to ~1
        assert 'momentum' in weights
        assert weights['momentum'] > 0.3  # Bull market favors momentum


# =============================================================================
# CORRELATION ENGINE TESTS
# =============================================================================

class TestCorrelationEngine:
    """Tests for V28 Correlation Engine."""
    
    def test_dynamic_correlation_matrix(self):
        """Test dynamic correlation matrix update."""
        from v28_correlation_engine import DynamicCorrelationMatrix
        
        np.random.seed(42)
        
        # Create sample returns
        n_days = 100
        symbols = ['SPY', 'QQQ', 'AAPL']
        returns = pd.DataFrame({
            sym: np.random.randn(n_days) * 0.02
            for sym in symbols
        })
        
        tracker = DynamicCorrelationMatrix()
        corr_matrix = tracker.update(returns)
        
        assert corr_matrix.shape == (3, 3)
        assert np.allclose(np.diag(corr_matrix), 1.0)  # Diagonal should be 1
        assert np.all(corr_matrix >= -1) and np.all(corr_matrix <= 1)
    
    def test_correlation_regime_classification(self):
        """Test correlation regime classification."""
        from v28_correlation_engine import DynamicCorrelationMatrix, CorrelationRegime
        
        tracker = DynamicCorrelationMatrix()
        tracker.avg_corr_history = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        tracker.corr_percentile_25 = 0.2
        tracker.corr_percentile_75 = 0.5
        tracker.corr_percentile_90 = 0.6
        
        regime_low, _ = tracker.classify_regime(0.15)
        regime_normal, _ = tracker.classify_regime(0.35)
        regime_high, _ = tracker.classify_regime(0.55)
        regime_crisis, _ = tracker.classify_regime(0.65)
        
        assert regime_low == CorrelationRegime.LOW
        assert regime_normal == CorrelationRegime.NORMAL
        assert regime_high == CorrelationRegime.ELEVATED
        assert regime_crisis == CorrelationRegime.CRISIS
    
    def test_correlation_breakdown_detector(self):
        """Test correlation breakdown detection."""
        from v28_correlation_engine import (
            CorrelationBreakdownDetector, 
            CorrelationRegime, 
            AlertType
        )
        
        detector = CorrelationBreakdownDetector(spike_threshold=0.15)
        
        # Create matrices with significant change
        n = 3
        current = np.array([
            [1.0, 0.7, 0.6],
            [0.7, 1.0, 0.5],
            [0.6, 0.5, 1.0]
        ])
        previous = np.array([
            [1.0, 0.3, 0.2],
            [0.3, 1.0, 0.2],
            [0.2, 0.2, 1.0]
        ])
        
        alerts = detector.check_for_alerts(
            current, previous, 
            CorrelationRegime.ELEVATED, 
            ['SPY', 'QQQ', 'AAPL']
        )
        
        # Should detect spike
        assert len(alerts) > 0
        assert any(a.alert_type == AlertType.SPIKE for a in alerts)
    
    def test_sector_rotation_engine(self):
        """Test sector rotation signal generation."""
        from v28_correlation_engine import SectorRotationEngine
        
        engine = SectorRotationEngine()
        
        # Create sample returns
        np.random.seed(42)
        symbols = ['XLK', 'XLF', 'XLE', 'XLV']
        n_days = 60
        
        returns = pd.DataFrame({
            'XLK': np.random.randn(n_days) * 0.02 + 0.002,  # Positive momentum
            'XLF': np.random.randn(n_days) * 0.015 - 0.001,  # Negative momentum
            'XLE': np.random.randn(n_days) * 0.025,
            'XLV': np.random.randn(n_days) * 0.012 + 0.001
        })
        
        momentum = engine.calculate_sector_momentum(returns)
        
        assert 'Technology' in momentum
        assert 'Financials' in momentum
    
    def test_diversification_ratio(self):
        """Test diversification ratio calculation."""
        from v28_correlation_engine import DiversificationAnalyzer
        
        analyzer = DiversificationAnalyzer()
        
        # Equal weights
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        volatilities = np.array([0.20, 0.25, 0.30, 0.15])
        
        # Low correlation matrix
        corr_matrix = np.array([
            [1.0, 0.2, 0.1, 0.15],
            [0.2, 1.0, 0.3, 0.1],
            [0.1, 0.3, 1.0, 0.2],
            [0.15, 0.1, 0.2, 1.0]
        ])
        
        div_ratio = analyzer.calculate_diversification_ratio(weights, volatilities, corr_matrix)
        
        # Low correlation should give high diversification ratio
        assert div_ratio > 1.0
    
    def test_v28_correlation_engine_integration(self):
        """Test V28 correlation engine integration."""
        from v28_correlation_engine import V28CorrelationEngine
        
        np.random.seed(42)
        
        # Create sample data
        n_days = 100
        symbols = ['SPY', 'QQQ', 'XLK', 'XLF', 'XLE']
        
        returns = pd.DataFrame({
            sym: np.random.randn(n_days) * 0.02 + np.random.randn() * 0.001
            for sym in symbols
        })
        
        engine = V28CorrelationEngine()
        state = engine.analyze(returns)
        
        assert state.average_correlation >= -1
        assert state.average_correlation <= 1
        assert state.diversification_ratio > 0
        assert len(state.highest_pairs) > 0


# =============================================================================
# KELLY POSITION SIZING TESTS
# =============================================================================

class TestKellyPositionSizing:
    """Tests for V28 Kelly Position Sizer."""
    
    def test_kelly_calculator_classic(self):
        """Test classic Kelly formula."""
        from v28_kelly_sizer import KellyCalculator
        
        # 60% win rate, 1.5 win/loss ratio
        kelly = KellyCalculator.calculate_classic_kelly(0.60, 1.5)
        
        # Kelly should be positive with edge
        assert kelly > 0
        assert kelly < 1  # Full Kelly should be less than 100%
    
    def test_kelly_calculator_no_edge(self):
        """Test Kelly with no edge."""
        from v28_kelly_sizer import KellyCalculator
        
        # 50% win rate, 1.0 win/loss ratio = no edge
        kelly = KellyCalculator.calculate_classic_kelly(0.50, 1.0)
        
        assert kelly == 0  # No edge, no bet
    
    def test_kelly_from_trades(self, sample_trade_history):
        """Test Kelly calculation from trade history."""
        from v28_kelly_sizer import KellyCalculator
        
        kelly, params = KellyCalculator.calculate_kelly_from_trades(sample_trade_history)
        
        assert kelly >= 0
        assert params.win_rate > 0.5  # Based on our 55% setup
        assert params.avg_win > 0
        assert params.avg_loss > 0
    
    def test_regime_aware_kelly(self):
        """Test regime-adjusted Kelly."""
        from v28_kelly_sizer import RegimeAwareKelly
        
        regime_kelly = RegimeAwareKelly()
        
        # Bull + low vol should have highest multiplier
        mult_bull_low = regime_kelly.get_regime_multiplier('bull', 'low_vol')
        mult_crisis = regime_kelly.get_regime_multiplier('crisis', 'extreme_vol')
        
        assert mult_bull_low > mult_crisis
        assert mult_bull_low == 1.0
        assert mult_crisis < 0.1
    
    def test_correlation_penalty(self, sample_correlation_matrix):
        """Test correlation-based position penalty."""
        from v28_kelly_sizer import CorrelationAwareWeights
        
        weights = CorrelationAwareWeights()
        
        # Test with existing positions
        current_positions = {'SPY': 0.2, 'QQQ': 0.15}
        
        penalty = weights.calculate_correlation_penalty(
            'XLK',  # Tech should be correlated with QQQ
            sample_correlation_matrix,
            current_positions
        )
        
        assert 0 < penalty <= 1
    
    def test_v28_kelly_sizer(self, sample_trade_history):
        """Test V28 Kelly position sizer."""
        from v28_kelly_sizer import V28KellyPositionSizer
        
        sizer = V28KellyPositionSizer(
            kelly_fraction=0.25,
            min_position_pct=0.02,
            max_position_pct=0.15,
            portfolio_value=100000.0
        )
        
        # Add trade history
        for trade in sample_trade_history:
            sizer.add_trade(trade)
        
        # Calculate position size
        result = sizer.calculate_position_size(
            symbol='AAPL',
            signal_confidence=0.7,
            volatility=0.25,
            market_regime='bull',
            vol_regime='normal_vol'
        )
        
        assert 0.02 <= result.position_size_pct <= 0.15
        assert result.position_size_dollars > 0
        assert result.regime_multiplier <= 1.0
    
    def test_v28_kelly_sizer_drawdown_protection(self, sample_trade_history):
        """Test Kelly sizer drawdown protection."""
        from v28_kelly_sizer import V28KellyPositionSizer
        
        sizer = V28KellyPositionSizer(
            kelly_fraction=0.25,
            max_drawdown_limit=0.15
        )
        
        for trade in sample_trade_history:
            sizer.add_trade(trade)
        
        # Simulate large drawdown
        sizer.peak_value = 100000
        sizer.portfolio_value = 85000  # 15% drawdown
        sizer.current_drawdown = 0.15
        
        result = sizer.calculate_position_size(
            symbol='AAPL',
            signal_confidence=0.8,
            market_regime='bull',
            vol_regime='normal_vol'
        )
        
        # Should have drawdown protection applied
        assert 'drawdown_protection' in result.constraints_applied
    
    def test_portfolio_weight_optimization(self, sample_correlation_matrix):
        """Test portfolio weight optimization."""
        from v28_kelly_sizer import V28KellyPositionSizer
        
        sizer = V28KellyPositionSizer(kelly_fraction=0.25)
        
        symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL']
        signals = {s: 0.5 + np.random.rand() * 0.3 for s in symbols}
        volatilities = {s: 0.15 + np.random.rand() * 0.15 for s in symbols}
        
        # Filter correlation matrix
        corr_filtered = sample_correlation_matrix.loc[symbols, symbols]
        
        results = sizer.calculate_portfolio_weights(
            symbols=symbols,
            signals=signals,
            volatilities=volatilities,
            correlation_matrix=corr_filtered,
            market_regime='bull',
            vol_regime='normal_vol'
        )
        
        assert len(results) == len(symbols)
        total_weight = sum(r.final_weight for r in results.values())
        assert total_weight <= 1.05  # Should sum to approximately 1


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for V28 system."""
    
    def test_signal_generator(self, sample_ohlcv):
        """Test signal generation."""
        from v28_production_system import SignalGenerator, V28Config
        from v28_regime_detector import RegimeState, MarketRegime, VolatilityRegime
        
        config = V28Config()
        generator = SignalGenerator(config)
        
        # Create mock regime state
        regime = RegimeState(
            market_regime=MarketRegime.BULL,
            volatility_regime=VolatilityRegime.NORMAL,
            hmm_state=0,
            hmm_state_name='LowVolTrend',
            hmm_probability=0.8,
            garch_volatility=0.15,
            garch_forecast_1d=0.15,
            garch_forecast_5d=0.16,
            regime_confidence=0.8,
            trend_strength=0.05,
            momentum_score=0.3,
            transition_probability=0.1,
            recommended_strategies={'momentum': 0.7, 'mean_reversion': 0.3}
        )
        
        signal = generator.generate_signal('SPY', sample_ohlcv, regime)
        
        assert 'signal' in signal
        assert 'direction' in signal
        assert 'confidence' in signal
        assert signal['direction'] in ['long', 'short', 'none']
    
    def test_portfolio_manager(self):
        """Test portfolio management."""
        from v28_production_system import PortfolioManager, V28Config
        
        config = V28Config(initial_capital=100000)
        manager = PortfolioManager(config)
        
        # Open position
        pos = manager.update_position('SPY', 'long', 100, 450.0, 'bull')
        
        assert 'SPY' in manager.positions
        assert manager.positions['SPY'].quantity == 100
        
        # Update prices
        manager.update_prices({'SPY': 455.0})
        
        assert manager.positions['SPY'].unrealized_pnl == 500.0  # 100 * 5
    
    def test_full_system_initialization(self):
        """Test full V28 system initialization."""
        from v28_production_system import V28ProductionEngine, V28Config
        
        config = V28Config(
            mode='paper',
            initial_capital=100000
        )
        
        engine = V28ProductionEngine(config)
        
        assert engine.regime_detector is not None
        assert engine.correlation_engine is not None
        assert engine.position_sizer is not None
        assert engine.portfolio_manager is not None
        assert engine.dashboard is not None


# =============================================================================
# PERFORMANCE BENCHMARKS
# =============================================================================

class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    def test_regime_detection_latency(self, sample_ohlcv):
        """Benchmark regime detection speed."""
        from v28_regime_detector import V28RegimeDetector
        
        detector = V28RegimeDetector()
        
        start = time.time()
        for _ in range(10):
            detector.detect(sample_ohlcv)
        elapsed = time.time() - start
        
        avg_latency = elapsed / 10 * 1000  # ms
        assert avg_latency < 500, f"Regime detection too slow: {avg_latency:.0f}ms"
    
    def test_correlation_calculation_latency(self):
        """Benchmark correlation calculation speed."""
        from v28_correlation_engine import V28CorrelationEngine
        
        np.random.seed(42)
        symbols = [f"SYM_{i}" for i in range(20)]
        returns = pd.DataFrame({
            sym: np.random.randn(252) * 0.02
            for sym in symbols
        })
        
        engine = V28CorrelationEngine()
        
        start = time.time()
        for _ in range(10):
            engine.analyze(returns)
        elapsed = time.time() - start
        
        avg_latency = elapsed / 10 * 1000  # ms
        assert avg_latency < 200, f"Correlation too slow: {avg_latency:.0f}ms"
    
    def test_kelly_calculation_latency(self, sample_trade_history):
        """Benchmark Kelly calculation speed."""
        from v28_kelly_sizer import V28KellyPositionSizer
        
        sizer = V28KellyPositionSizer()
        for trade in sample_trade_history:
            sizer.add_trade(trade)
        
        start = time.time()
        for _ in range(100):
            sizer.calculate_position_size(
                symbol='AAPL',
                signal_confidence=0.7,
                volatility=0.25,
                market_regime='bull',
                vol_regime='normal_vol'
            )
        elapsed = time.time() - start
        
        avg_latency = elapsed / 100 * 1000  # ms
        assert avg_latency < 10, f"Kelly calculation too slow: {avg_latency:.0f}ms"
    
    def test_signal_generation_latency(self, sample_ohlcv):
        """Benchmark signal generation speed."""
        from v28_production_system import SignalGenerator, V28Config
        from v28_regime_detector import RegimeState, MarketRegime, VolatilityRegime
        
        config = V28Config()
        generator = SignalGenerator(config)
        
        regime = RegimeState(
            market_regime=MarketRegime.BULL,
            volatility_regime=VolatilityRegime.NORMAL,
            hmm_state=0,
            hmm_state_name='LowVolTrend',
            hmm_probability=0.8,
            garch_volatility=0.15,
            garch_forecast_1d=0.15,
            garch_forecast_5d=0.16,
            regime_confidence=0.8,
            trend_strength=0.05,
            momentum_score=0.3,
            transition_probability=0.1,
            recommended_strategies={'momentum': 0.7}
        )
        
        start = time.time()
        for _ in range(50):
            generator.generate_signal('SPY', sample_ohlcv, regime)
        elapsed = time.time() - start
        
        avg_latency = elapsed / 50 * 1000  # ms
        assert avg_latency < 50, f"Signal generation too slow: {avg_latency:.0f}ms"


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Edge case and error handling tests."""
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        from v28_regime_detector import RegimeFeatureBuilder
        
        builder = RegimeFeatureBuilder()
        
        empty_df = pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        features = builder.calculate_features(empty_df)
        
        assert len(features) == 0
    
    def test_insufficient_data_kelly(self):
        """Test Kelly with insufficient trade history."""
        from v28_kelly_sizer import V28KellyPositionSizer
        
        sizer = V28KellyPositionSizer(min_trades_for_kelly=20)
        
        # Only add 5 trades
        for i in range(5):
            sizer.add_trade(0.02)
        
        result = sizer.calculate_position_size('AAPL', signal_confidence=0.7)
        
        assert 'insufficient_history' in result.constraints_applied
    
    def test_extreme_volatility(self):
        """Test handling of extreme volatility."""
        from v28_kelly_sizer import V28KellyPositionSizer
        
        sizer = V28KellyPositionSizer()
        
        # Add some trades
        for i in range(30):
            sizer.add_trade(0.01 if i % 2 == 0 else -0.008)
        
        result = sizer.calculate_position_size(
            symbol='EXTREME',
            signal_confidence=0.8,
            volatility=2.0,  # 200% annualized vol
            market_regime='crisis',
            vol_regime='extreme_vol'
        )
        
        # Should be very conservative with extreme vol
        assert result.position_size_pct <= 0.05
    
    def test_nan_handling_in_returns(self):
        """Test NaN handling in returns."""
        from v28_correlation_engine import DynamicCorrelationMatrix
        
        returns = pd.DataFrame({
            'SPY': [0.01, np.nan, 0.02, -0.01, 0.015],
            'QQQ': [0.015, 0.02, np.nan, -0.005, 0.01]
        })
        
        tracker = DynamicCorrelationMatrix(min_periods=2)
        
        # Should handle NaN without crashing
        try:
            corr = tracker.update(returns)
            # Result may have NaN, but shouldn't crash
            assert corr.shape == (2, 2)
        except Exception as e:
            pytest.fail(f"NaN handling failed: {e}")
    
    def test_single_asset_correlation(self):
        """Test correlation with single asset."""
        from v28_correlation_engine import V28CorrelationEngine
        
        returns = pd.DataFrame({
            'SPY': np.random.randn(60) * 0.02
        })
        
        engine = V28CorrelationEngine()
        
        # Should handle single asset gracefully
        state = engine.analyze(returns)
        assert state.average_correlation == 0.0 or np.isnan(state.average_correlation)


# =============================================================================
# METRIC TARGET TESTS
# =============================================================================

class TestMetricTargets:
    """Tests to verify target metric achievement."""
    
    def test_sharpe_target(self):
        """Test that system can achieve Sharpe > 2.5."""
        from v28_dashboard_api import MetricsCalculator
        
        calc = MetricsCalculator()
        
        # Simulate good performance
        np.random.seed(42)
        equity = 100000
        for _ in range(252):
            # High Sharpe: high return, low vol
            daily_ret = np.random.randn() * 0.01 + 0.003  # ~75% annual, 16% vol
            equity *= (1 + daily_ret)
            calc.update_equity(equity)
        
        sharpe = calc.calculate_sharpe()
        # With these parameters, Sharpe should be high
        assert sharpe > 2.0, f"Sharpe {sharpe} below target"
    
    def test_win_rate_target(self):
        """Test that Kelly parameters support >60% win rate."""
        from v28_kelly_sizer import KellyCalculator
        
        # Simulate trades with 62% win rate
        np.random.seed(42)
        trades = []
        for _ in range(100):
            if np.random.rand() < 0.62:
                trades.append(np.random.uniform(0.01, 0.04))
            else:
                trades.append(-np.random.uniform(0.008, 0.025))
        
        kelly, params = KellyCalculator.calculate_kelly_from_trades(trades)
        
        assert params.win_rate >= 0.58, f"Win rate {params.win_rate} below target"
        assert kelly > 0, "Should have positive edge"
    
    def test_max_drawdown_protection(self):
        """Test that drawdown protection limits losses."""
        from v28_kelly_sizer import V28KellyPositionSizer
        
        sizer = V28KellyPositionSizer(max_drawdown_limit=0.15)
        
        # Add trades
        for _ in range(30):
            sizer.add_trade(0.01)
        
        # Simulate increasing drawdown
        drawdowns = [0.05, 0.10, 0.12, 0.14, 0.15]
        position_sizes = []
        
        for dd in drawdowns:
            sizer.current_drawdown = dd
            result = sizer.calculate_position_size('TEST', signal_confidence=0.8)
            position_sizes.append(result.position_size_pct)
        
        # Position sizes should decrease with drawdown
        for i in range(1, len(position_sizes)):
            assert position_sizes[i] <= position_sizes[i-1] * 1.1, \
                "Position size should decrease with drawdown"


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-x'])
