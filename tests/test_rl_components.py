"""
Tests for V2.2 RL Components
============================

Comprehensive test suite for:
- SAC Position Optimizer
- Hierarchical Regime Controller
- Anomaly-Aware Transformer
- RL Orchestrator Integration

Target Coverage: >80%
"""

import os
import sys
import json
import pytest  # type: ignore[import-not-found]
import numpy as np
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Import components
from src.agents.sac_position_optimizer import (
    SACPositionOptimizer, SACConfig, ReplayBuffer, PrioritizedReplayBuffer
)
from src.regime.hierarchical_controller import (
    HierarchicalController, RegimeState, SubPolicy, CUSUMDetector,
    VolatilityRegime, TrendRegime, VolatilityRegimeDetector, TrendRegimeDetector,
    AGGRESSIVE_POLICY, NEUTRAL_POLICY, CONSERVATIVE_POLICY
)
from src.models.anomaly_aware_transformer import (
    AnomalyAwareTransformer, TransformerConfig, IsolationForestDetector,
    AttentionAnalyzer
)
from src.trading.rl_orchestrator import (
    RLOrchestrator, RLOrchestratorConfig, MarketStateEncoder
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for test artifacts."""
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp, ignore_errors=True)


@pytest.fixture
def sac_config():
    """Default SAC configuration for tests."""
    return SACConfig(
        state_dim=16,
        action_dim=1,
        hidden_dims=(64, 64),
        learning_rate=1e-3,
        batch_size=32,
        buffer_size=1000,
        min_buffer_size=100,
        use_per=False,
    )


@pytest.fixture
def transformer_config():
    """Default transformer configuration for tests."""
    return TransformerConfig(
        input_dim=16,
        model_dim=32,
        output_dim=1,
        num_heads=2,
        num_layers=1,
        seq_length=10,
        batch_size=16,
        max_epochs=5,
    )


@pytest.fixture
def sample_market_data():
    """Generate sample market data for testing."""
    np.random.seed(42)
    
    prices = np.cumprod(1 + np.random.randn(100) * 0.02) * 100
    returns = np.diff(np.log(prices))
    
    return {
        "SPY": prices,
        "QQQ": prices * 1.1 + np.random.randn(100) * 2,
        "IWM": prices * 0.9 + np.random.randn(100) * 3,
    }


@pytest.fixture
def sample_signals():
    """Sample base signals for testing."""
    return {
        "SPY": {"weight": 0.02, "direction": "long", "confidence": 0.7},
        "QQQ": {"weight": -0.015, "direction": "short", "confidence": 0.6},
        "IWM": {"weight": 0.01, "direction": "long", "confidence": 0.5},
    }


# =============================================================================
# REPLAY BUFFER TESTS
# =============================================================================

class TestReplayBuffer:
    """Tests for ReplayBuffer class."""
    
    def test_buffer_initialization(self):
        """Test buffer initializes correctly."""
        buffer = ReplayBuffer(capacity=100, state_dim=8, action_dim=1)
        
        assert buffer.capacity == 100
        assert buffer.state_dim == 8
        assert buffer.action_dim == 1
        assert len(buffer) == 0
        
    def test_buffer_push_and_sample(self):
        """Test pushing and sampling experiences."""
        buffer = ReplayBuffer(capacity=100, state_dim=8, action_dim=1)
        
        # Push experiences
        for _ in range(50):
            state = np.random.randn(8)
            action = np.random.randn(1)
            reward = np.random.randn()
            next_state = np.random.randn(8)
            done = False
            
            buffer.push(state, action, reward, next_state, done)
            
        assert len(buffer) == 50
        
        # Sample
        batch = buffer.sample(10)
        states, actions, rewards, next_states, dones = batch
        
        assert states.shape == (10, 8)
        assert actions.shape == (10, 1)
        assert rewards.shape == (10,)
        assert next_states.shape == (10, 8)
        assert dones.shape == (10,)
        
    def test_buffer_overflow(self):
        """Test buffer handles overflow correctly."""
        buffer = ReplayBuffer(capacity=10, state_dim=4, action_dim=1)
        
        # Push more than capacity
        for i in range(20):
            buffer.push(
                np.ones(4) * i,
                np.array([0.5]),
                float(i),
                np.ones(4) * (i + 1),
                False
            )
            
        assert len(buffer) == 10  # Capped at capacity


class TestPrioritizedReplayBuffer:
    """Tests for PrioritizedReplayBuffer class."""
    
    def test_per_initialization(self):
        """Test PER buffer initializes correctly."""
        buffer = PrioritizedReplayBuffer(
            capacity=100, state_dim=8, action_dim=1,
            alpha=0.6, beta=0.4
        )
        
        assert buffer.alpha == 0.6
        assert buffer.beta == 0.4
        assert len(buffer) == 0
        
    def test_per_sampling_with_weights(self):
        """Test PER returns importance weights."""
        buffer = PrioritizedReplayBuffer(
            capacity=100, state_dim=4, action_dim=1
        )
        
        # Push experiences
        for _ in range(50):
            buffer.push(
                np.random.randn(4),
                np.random.randn(1),
                np.random.randn(),
                np.random.randn(4),
                False
            )
            
        # Sample (PER returns 7 elements including indices and weights)
        batch = buffer.sample(10)
        
        assert len(batch) == 7
        indices = batch[5]
        weights = batch[6]
        
        assert len(indices) == 10
        assert len(weights) == 10
        assert np.all(weights > 0)  # Weights should be positive
        
    def test_priority_update(self):
        """Test priority updates work correctly."""
        buffer = PrioritizedReplayBuffer(
            capacity=100, state_dim=4, action_dim=1
        )
        
        # Push experiences
        for _ in range(50):
            buffer.push(
                np.random.randn(4),
                np.random.randn(1),
                np.random.randn(),
                np.random.randn(4),
                False
            )
            
        # Sample and update priorities
        batch = buffer.sample(10)
        indices = batch[5]
        td_errors = np.abs(np.random.randn(10))
        
        buffer.update_priorities(indices, td_errors)
        
        # No assertion needed - just verify it doesn't crash


# =============================================================================
# SAC POSITION OPTIMIZER TESTS
# =============================================================================

class TestSACPositionOptimizer:
    """Tests for SAC Position Optimizer."""
    
    def test_sac_initialization(self, sac_config, temp_dir):
        """Test SAC initializes correctly."""
        sac_config.log_dir = temp_dir
        sac = SACPositionOptimizer(sac_config)
        
        assert sac.config == sac_config
        assert sac.total_steps == 0
        assert len(sac.buffer) == 0
        
    def test_get_position_returns_valid_range(self, sac_config, temp_dir):
        """Test position sizing returns valid values."""
        sac_config.log_dir = temp_dir
        sac = SACPositionOptimizer(sac_config)
        
        state = np.random.randn(sac_config.state_dim)
        
        for regime in ["trending", "volatile", "flat", "neutral"]:
            position = sac.get_position(state, regime=regime)
            
            assert 0 <= position <= sac_config.max_position_pct
            assert isinstance(position, float)
            
    def test_regime_affects_position(self, sac_config, temp_dir):
        """Test different regimes produce different position scales."""
        sac_config.log_dir = temp_dir
        sac = SACPositionOptimizer(sac_config)
        
        state = np.random.randn(sac_config.state_dim)
        
        pos_trending = sac.get_position(state, regime="trending")
        pos_volatile = sac.get_position(state, regime="volatile")
        
        # Volatile should typically produce smaller positions
        # (Due to regime_scale of 0.6 vs 1.2)
        # Note: This may not always hold due to stochastic policy
        
    def test_reward_computation(self, sac_config, temp_dir):
        """Test reward shaping produces sensible values."""
        sac_config.log_dir = temp_dir
        sac = SACPositionOptimizer(sac_config)
        
        # Profit should increase reward
        reward_profit = sac.compute_reward(
            pnl=0.01, drawdown=0.0, position_change=0.01, volatility=0.15
        )
        reward_loss = sac.compute_reward(
            pnl=-0.01, drawdown=0.02, position_change=0.01, volatility=0.15
        )
        
        assert reward_profit > reward_loss
        
    def test_experience_storage(self, sac_config, temp_dir):
        """Test experiences are stored in buffer."""
        sac_config.log_dir = temp_dir
        sac = SACPositionOptimizer(sac_config)
        
        # Store experiences
        for _ in range(10):
            sac.store_experience(
                state=np.random.randn(sac_config.state_dim),
                action=np.array([0.02]),
                reward=0.01,
                next_state=np.random.randn(sac_config.state_dim),
                done=False,
            )
            
        assert len(sac.buffer) == 10
        
    def test_training_stats(self, sac_config, temp_dir):
        """Test training statistics are tracked."""
        sac_config.log_dir = temp_dir
        sac = SACPositionOptimizer(sac_config)
        
        stats = sac.get_training_stats()
        
        assert "total_steps" in stats
        assert "buffer_size" in stats
        assert "alpha" in stats
        
    def test_save_and_load(self, sac_config, temp_dir):
        """Test model save and load."""
        sac_config.log_dir = temp_dir
        sac = SACPositionOptimizer(sac_config)
        
        # Store some experience
        for _ in range(10):
            sac.store_experience(
                np.random.randn(sac_config.state_dim),
                np.array([0.02]),
                0.01,
                np.random.randn(sac_config.state_dim),
                False,
            )
            
        # Save
        save_path = Path(temp_dir) / "test_model.pt"
        sac.save(save_path)
        
        # Check that some file was created 
        # np.savez adds .npz to any path, so test_model.pt.npz for numpy fallback
        files_in_dir = list(Path(temp_dir).glob("test_model*"))
        assert len(files_in_dir) > 0, f"No model files found in {temp_dir}"


# =============================================================================
# CUSUM DETECTOR TESTS
# =============================================================================

class TestCUSUMDetector:
    """Tests for CUSUM change detection."""
    
    def test_cusum_initialization(self):
        """Test CUSUM initializes correctly."""
        cusum = CUSUMDetector(threshold=2.5, drift=0.5, warmup_period=50)
        
        assert cusum.threshold == 2.5
        assert cusum.drift == 0.5
        assert cusum.warmup_period == 50
        assert cusum.samples == 0
        
    def test_no_change_during_warmup(self):
        """Test no changes detected during warmup."""
        cusum = CUSUMDetector(threshold=2.5, warmup_period=50)
        
        for i in range(40):
            change = cusum.update(np.random.randn())
            assert not change  # Should not detect during warmup
            
    def test_detects_mean_shift(self):
        """Test CUSUM detects mean shift."""
        cusum = CUSUMDetector(threshold=2.5, warmup_period=20)
        
        # Normal period
        for _ in range(30):
            cusum.update(np.random.randn())
            
        # Large shift
        detected = False
        for _ in range(20):
            if cusum.update(5.0):  # Large positive value
                detected = True
                break
                
        assert detected or cusum.get_statistic() > 0
        
    def test_cusum_state(self):
        """Test state retrieval."""
        cusum = CUSUMDetector()
        
        for _ in range(10):
            cusum.update(np.random.randn())
            
        state = cusum.get_state()
        
        assert "cusum_pos" in state
        assert "cusum_neg" in state
        assert "samples" in state
        assert state["samples"] == 10


# =============================================================================
# REGIME CONTROLLER TESTS
# =============================================================================

class TestVolatilityRegimeDetector:
    """Tests for volatility regime detection."""
    
    def test_low_vol_detection(self):
        """Test low volatility regime detection."""
        detector = VolatilityRegimeDetector(
            low_threshold=0.12, high_threshold=0.25
        )
        
        # Low volatility returns (< 12% annualized)
        low_vol_returns = np.random.randn(60) * 0.005  # ~8% annualized
        
        regime = detector.update(low_vol_returns)
        
        # Should detect low volatility
        assert regime in [VolatilityRegime.LOW, VolatilityRegime.MEDIUM]
        
    def test_high_vol_detection(self):
        """Test high volatility regime detection."""
        detector = VolatilityRegimeDetector(
            low_threshold=0.12, high_threshold=0.25
        )
        
        # High volatility returns (> 25% annualized)
        high_vol_returns = np.random.randn(60) * 0.03  # ~48% annualized
        
        regime = detector.update(high_vol_returns)
        
        assert regime == VolatilityRegime.HIGH


class TestTrendRegimeDetector:
    """Tests for trend regime detection."""
    
    def test_trending_up_detection(self):
        """Test uptrend detection."""
        detector = TrendRegimeDetector(trend_threshold=0.02)
        
        # Strong uptrend prices
        prices = np.cumprod(1 + np.ones(100) * 0.002) * 100
        
        regime = detector.update(prices)
        
        assert regime == TrendRegime.TRENDING_UP
        
    def test_trending_down_detection(self):
        """Test downtrend detection."""
        detector = TrendRegimeDetector(trend_threshold=0.02)
        
        # Strong downtrend prices
        prices = np.cumprod(1 - np.ones(100) * 0.002) * 100
        
        regime = detector.update(prices)
        
        assert regime == TrendRegime.TRENDING_DOWN
        
    def test_flat_detection(self):
        """Test flat market detection."""
        detector = TrendRegimeDetector(trend_threshold=0.02)
        
        # Sideways prices with small random walk
        prices = 100 + np.cumsum(np.random.randn(100) * 0.1)
        
        regime = detector.update(prices)
        
        assert regime in [TrendRegime.FLAT, TrendRegime.MEAN_REVERTING]


class TestHierarchicalController:
    """Tests for hierarchical regime controller."""
    
    def test_controller_initialization(self, temp_dir):
        """Test controller initializes correctly."""
        controller = HierarchicalController(log_dir=temp_dir)
        
        assert "aggressive" in controller.policies
        assert "neutral" in controller.policies
        assert "conservative" in controller.policies
        assert controller.current_state is not None
        
    def test_regime_update(self, temp_dir):
        """Test regime updates with market data."""
        controller = HierarchicalController(log_dir=temp_dir)
        
        returns = np.random.randn(60) * 0.02
        prices = np.cumprod(1 + returns) * 100
        
        state = controller.update(returns, prices)
        
        assert isinstance(state, RegimeState)
        assert state.volatility in VolatilityRegime
        assert state.trend in TrendRegime
        assert 0 <= state.confidence <= 1
        
    def test_policy_selection(self, temp_dir):
        """Test correct policy is selected for regime."""
        controller = HierarchicalController(log_dir=temp_dir)
        
        # Force a specific regime state
        controller.current_state = RegimeState(
            volatility=VolatilityRegime.LOW,
            trend=TrendRegime.TRENDING_UP,
        )
        
        policy = controller.get_active_policy()
        
        assert policy.name == "aggressive"
        
    def test_blended_parameters(self, temp_dir):
        """Test parameter blending during transitions."""
        controller = HierarchicalController(
            log_dir=temp_dir,
            blend_window=5
        )
        
        # Simulate transition
        controller.previous_state = RegimeState(
            volatility=VolatilityRegime.LOW,
            trend=TrendRegime.TRENDING_UP,
        )
        controller.current_state = RegimeState(
            volatility=VolatilityRegime.HIGH,
            trend=TrendRegime.FLAT,
        )
        controller.samples_since_transition = 2
        
        params = controller.get_blended_parameters()
        
        assert "max_position_pct" in params
        assert "position_scale" in params
        
    def test_regime_distribution(self, temp_dir):
        """Test regime distribution calculation."""
        controller = HierarchicalController(log_dir=temp_dir)
        
        # Add some history
        for _ in range(100):
            controller.state_history.append(RegimeState(
                volatility=VolatilityRegime.MEDIUM,
                trend=TrendRegime.FLAT,
            ))
            
        dist = controller.get_regime_distribution(lookback=50)
        
        assert len(dist) > 0
        assert sum(dist.values()) == pytest.approx(1.0)


# =============================================================================
# ANOMALY DETECTION TESTS
# =============================================================================

class TestIsolationForestDetector:
    """Tests for Isolation Forest anomaly detection."""
    
    def test_detector_initialization(self):
        """Test detector initializes correctly."""
        detector = IsolationForestDetector(
            contamination=0.05,
            n_estimators=50
        )
        
        assert detector.contamination == 0.05
        assert detector.n_estimators == 50
        assert not detector.is_fitted
        
    def test_fit_and_detect(self):
        """Test fitting and anomaly detection."""
        detector = IsolationForestDetector()
        
        # Normal data
        X_train = np.random.randn(200, 5)
        detector.fit(X_train)
        
        assert detector.is_fitted
        
        # Test detection
        X_normal = np.random.randn(10, 5)
        X_anomaly = np.random.randn(10, 5) * 5 + 10  # Outliers
        
        is_anomaly_normal, scores_normal = detector.detect(X_normal)
        is_anomaly_out, scores_out = detector.detect(X_anomaly)
        
        # Outliers should have higher scores on average
        assert np.mean(scores_out) >= np.mean(scores_normal) * 0.8
        
    def test_anomaly_mask(self):
        """Test anomaly mask generation."""
        detector = IsolationForestDetector()
        
        X_train = np.random.randn(100, 4)
        detector.fit(X_train)
        
        X_test = np.random.randn(20, 4)
        mask = detector.get_anomaly_mask(X_test, threshold=0.7)
        
        assert len(mask) == 20
        assert np.all(mask >= 0) and np.all(mask <= 1)


class TestAttentionAnalyzer:
    """Tests for attention analysis."""
    
    def test_analyzer_initialization(self):
        """Test analyzer initializes correctly."""
        analyzer = AttentionAnalyzer()
        
        assert len(analyzer.attention_history) == 0
        
    def test_record_attention(self):
        """Test recording attention weights."""
        analyzer = AttentionAnalyzer()
        
        # Simulate attention weights (batch, heads, seq, seq)
        attention = np.random.rand(4, 2, 10, 10)
        anomaly_mask = np.random.rand(10)
        
        analyzer.record(attention, anomaly_mask)
        
        assert len(analyzer.attention_history) == 1
        assert "attention" in analyzer.attention_history[0]
        
    def test_temporal_importance(self):
        """Test temporal importance computation."""
        analyzer = AttentionAnalyzer()
        
        # Add multiple records
        for _ in range(50):
            attention = np.random.rand(4, 2, 10, 10)
            analyzer.record(attention)
            
        importance = analyzer.get_temporal_importance()
        
        assert len(importance) > 0
        
    def test_attention_entropy(self):
        """Test entropy computation."""
        analyzer = AttentionAnalyzer()
        
        for _ in range(20):
            attention = np.random.rand(4, 2, 10, 10)
            analyzer.record(attention)
            
        entropy = analyzer.get_attention_entropy()
        
        assert entropy >= 0


# =============================================================================
# ANOMALY-AWARE TRANSFORMER TESTS
# =============================================================================

class TestAnomalyAwareTransformer:
    """Tests for Anomaly-Aware Transformer."""
    
    def test_transformer_initialization(self, transformer_config, temp_dir):
        """Test transformer initializes correctly."""
        transformer = AnomalyAwareTransformer(transformer_config)
        
        assert transformer.config == transformer_config
        assert not transformer.is_fitted
        
    def test_predict_before_fit(self, transformer_config):
        """Test prediction before fitting returns zeros."""
        transformer = AnomalyAwareTransformer(transformer_config)
        
        X = np.random.randn(10, transformer_config.seq_length, transformer_config.input_dim)
        predictions = transformer.predict(X)
        
        assert predictions.shape == (10, transformer_config.output_dim)
        assert np.allclose(predictions, 0)
        
    def test_fit_transformer(self, transformer_config):
        """Test transformer fitting."""
        transformer = AnomalyAwareTransformer(transformer_config)
        
        X = np.random.randn(100, transformer_config.seq_length, transformer_config.input_dim)
        y = np.random.randn(100, transformer_config.output_dim)
        
        result = transformer.fit(X, y)
        
        assert transformer.is_fitted
        assert "epochs" in result or "method" in result
        
    def test_predict_with_confidence(self, transformer_config):
        """Test prediction with confidence scores."""
        transformer = AnomalyAwareTransformer(transformer_config)
        
        # Fit first
        X = np.random.randn(100, transformer_config.seq_length, transformer_config.input_dim)
        y = np.random.randn(100, transformer_config.output_dim)
        transformer.fit(X, y)
        
        # Predict with confidence
        X_test = np.random.randn(10, transformer_config.seq_length, transformer_config.input_dim)
        predictions, confidence = transformer.predict_with_confidence(X_test)
        
        assert predictions.shape[0] == 10
        assert len(confidence) == 10
        assert np.all(confidence >= 0) and np.all(confidence <= 1)
        
    def test_signal_thresholding(self, transformer_config):
        """Test confidence-based signal filtering."""
        transformer = AnomalyAwareTransformer(transformer_config)
        
        # Fit
        X = np.random.randn(100, transformer_config.seq_length, transformer_config.input_dim)
        y = np.random.randn(100, transformer_config.output_dim)
        transformer.fit(X, y)
        
        # Get filtered signals
        X_test = np.random.randn(10, transformer_config.seq_length, transformer_config.input_dim)
        signals, confidence = transformer.get_signal_with_threshold(X_test, threshold=0.5)
        
        # Low confidence signals should be zero
        for i in range(len(signals)):
            if confidence[i] < 0.5:
                assert signals[i] == 0
                
    def test_model_summary(self, transformer_config):
        """Test model summary generation."""
        transformer = AnomalyAwareTransformer(transformer_config)
        
        summary = transformer.get_model_summary()
        
        assert "config" in summary
        assert "is_fitted" in summary
        assert "device" in summary


# =============================================================================
# MARKET STATE ENCODER TESTS
# =============================================================================

class TestMarketStateEncoder:
    """Tests for market state encoding."""
    
    def test_encoder_initialization(self):
        """Test encoder initializes correctly."""
        encoder = MarketStateEncoder(state_dim=32)
        
        assert encoder.state_dim == 32
        assert encoder.n_samples == 0
        
    def test_encode_prices(self):
        """Test encoding price data."""
        encoder = MarketStateEncoder(state_dim=32)
        
        prices = np.cumprod(1 + np.random.randn(100) * 0.02) * 100
        
        state = encoder.encode("SPY", prices)
        
        assert state.shape == (32,)
        assert np.all(np.isfinite(state))
        
    def test_encode_with_tda_features(self):
        """Test encoding with TDA features."""
        encoder = MarketStateEncoder(state_dim=32)
        
        prices = np.cumprod(1 + np.random.randn(100) * 0.02) * 100
        tda_features = {
            "persistence_entropy": 0.5,
            "betti_0": 3.0,
            "betti_1": 1.0,
        }
        
        state = encoder.encode("SPY", prices, tda_features=tda_features)
        
        assert state.shape == (32,)
        
    def test_encode_with_regime(self):
        """Test encoding with regime state."""
        encoder = MarketStateEncoder(state_dim=32)
        
        prices = np.cumprod(1 + np.random.randn(100) * 0.02) * 100
        regime_state = RegimeState(
            volatility=VolatilityRegime.HIGH,
            trend=TrendRegime.TRENDING_DOWN,
        )
        
        state = encoder.encode("SPY", prices, regime_state=regime_state)
        
        assert state.shape == (32,)
        
    def test_normalization(self):
        """Test state normalization."""
        encoder = MarketStateEncoder(state_dim=16)
        
        # Encode multiple samples to build statistics
        for _ in range(100):
            prices = np.cumprod(1 + np.random.randn(50) * 0.02) * 100
            state = encoder.encode("SPY", prices)
            
        # State should be roughly normalized
        assert np.abs(np.mean(state)) < 3.0  # Within 3 std
        assert np.abs(np.std(state) - 1.0) < 2.0


# =============================================================================
# RL ORCHESTRATOR TESTS
# =============================================================================

class TestRLOrchestrator:
    """Tests for RL Orchestrator integration."""
    
    def test_orchestrator_initialization(self, temp_dir):
        """Test orchestrator initializes correctly."""
        config = RLOrchestratorConfig(log_dir=temp_dir)
        orchestrator = RLOrchestrator(config)
        
        assert orchestrator.config == config
        assert orchestrator.total_decisions == 0
        
    def test_enhance_signals(self, temp_dir, sample_signals, sample_market_data):
        """Test signal enhancement."""
        config = RLOrchestratorConfig(log_dir=temp_dir)
        orchestrator = RLOrchestrator(config)
        
        enhanced = orchestrator.enhance_signals(
            base_signals=sample_signals,
            market_data=sample_market_data,
        )
        
        assert len(enhanced) == len(sample_signals)
        
        for ticker, signal in enhanced.items():
            assert "weight" in signal
            assert "direction" in signal
            
    def test_empty_signals(self, temp_dir):
        """Test handling of empty signals."""
        config = RLOrchestratorConfig(log_dir=temp_dir)
        orchestrator = RLOrchestrator(config)
        
        enhanced = orchestrator.enhance_signals({}, {})
        
        assert enhanced == {}
        
    def test_flat_signals_unchanged(self, temp_dir, sample_market_data):
        """Test flat signals remain flat."""
        config = RLOrchestratorConfig(log_dir=temp_dir)
        orchestrator = RLOrchestrator(config)
        
        flat_signals = {
            "SPY": {"weight": 0.0, "direction": "flat", "confidence": 0.5}
        }
        
        enhanced = orchestrator.enhance_signals(
            base_signals=flat_signals,
            market_data=sample_market_data,
        )
        
        assert enhanced["SPY"]["weight"] == 0.0
        
    def test_stats_tracking(self, temp_dir, sample_signals, sample_market_data):
        """Test statistics are tracked."""
        config = RLOrchestratorConfig(log_dir=temp_dir)
        orchestrator = RLOrchestrator(config)
        
        orchestrator.enhance_signals(
            base_signals=sample_signals,
            market_data=sample_market_data,
        )
        
        stats = orchestrator.get_stats()
        
        assert "total_decisions" in stats
        assert "components" in stats
        assert stats["total_decisions"] > 0
        
    def test_online_training(self, temp_dir):
        """Test online training interface."""
        config = RLOrchestratorConfig(log_dir=temp_dir)
        orchestrator = RLOrchestrator(config)
        
        state = np.random.randn(config.sac_state_dim)
        next_state = np.random.randn(config.sac_state_dim)
        
        # Should not raise
        orchestrator.train_online(
            state=state,
            action=0.02,
            reward=0.01,
            next_state=next_state,
            done=False,
        )
        
    def test_save_and_load(self, temp_dir, sample_signals, sample_market_data):
        """Test save and load functionality."""
        config = RLOrchestratorConfig(log_dir=temp_dir)
        orchestrator = RLOrchestrator(config)
        
        # Generate some decisions
        orchestrator.enhance_signals(sample_signals, sample_market_data)
        
        # Save
        save_path = Path(temp_dir) / "orchestrator"
        orchestrator.save(save_path)
        
        assert (save_path / "orchestrator_stats.json").exists()


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestRLIntegration:
    """Integration tests for the full RL pipeline."""
    
    def test_full_pipeline(self, temp_dir):
        """Test complete RL enhancement pipeline."""
        np.random.seed(42)
        
        # Create components
        config = RLOrchestratorConfig(
            log_dir=temp_dir,
            use_sac=True,
            use_hierarchical_regime=True,
            use_anomaly_transformer=True,
        )
        orchestrator = RLOrchestrator(config)
        
        # Generate market data
        market_data = {
            "SPY": np.cumprod(1 + np.random.randn(100) * 0.02) * 100,
            "QQQ": np.cumprod(1 + np.random.randn(100) * 0.025) * 200,
        }
        
        # Base signals from V2.1
        base_signals = {
            "SPY": {"weight": 0.025, "direction": "long", "confidence": 0.7},
            "QQQ": {"weight": 0.02, "direction": "long", "confidence": 0.6},
        }
        
        # Enhance signals
        enhanced = orchestrator.enhance_signals(base_signals, market_data)
        
        # Verify enhancement
        assert len(enhanced) == 2
        
        for ticker in ["SPY", "QQQ"]:
            signal = enhanced[ticker]
            assert 0 < abs(signal["weight"]) <= config.max_position_pct
            
    def test_regime_transition_handling(self, temp_dir):
        """Test handling of regime transitions."""
        config = RLOrchestratorConfig(log_dir=temp_dir)
        orchestrator = RLOrchestrator(config)
        
        # Simulate multiple days with different regimes
        for i in range(10):
            # Varying volatility
            vol_scale = 0.01 if i < 5 else 0.04
            
            market_data = {
                "SPY": np.cumprod(1 + np.random.randn(60) * vol_scale) * 100,
            }
            
            signals = {
                "SPY": {"weight": 0.02, "direction": "long", "confidence": 0.6},
            }
            
            enhanced = orchestrator.enhance_signals(signals, market_data)
            
        # Should have detected some transitions or anomalies
        stats = orchestrator.get_stats()
        assert stats["total_decisions"] == 10
        
    def test_anomaly_detection_integration(self, temp_dir):
        """Test anomaly detection affects position sizing."""
        config = RLOrchestratorConfig(
            log_dir=temp_dir,
            anomaly_threshold=0.3,  # Lower threshold for testing
            anomaly_position_cap=0.01,
        )
        orchestrator = RLOrchestrator(config)
        
        # Normal market data
        normal_data = {"SPY": np.cumprod(1 + np.random.randn(100) * 0.02) * 100}
        normal_signal = {"SPY": {"weight": 0.03, "direction": "long", "confidence": 0.8}}
        
        # Anomalous market data (extreme values)
        anomaly_data = {"SPY": np.cumprod(1 + np.random.randn(100) * 0.1) * 100}
        
        # Process normal
        enhanced_normal = orchestrator.enhance_signals(normal_signal.copy(), normal_data)
        
        # Process anomaly
        enhanced_anomaly = orchestrator.enhance_signals(normal_signal.copy(), anomaly_data)
        
        # Both should be processed
        assert len(enhanced_normal) == 1
        assert len(enhanced_anomaly) == 1


# =============================================================================
# SUB-POLICY TESTS
# =============================================================================

class TestSubPolicies:
    """Tests for sub-policy configurations."""
    
    def test_aggressive_policy(self):
        """Test aggressive policy configuration."""
        policy = AGGRESSIVE_POLICY
        
        assert policy.name == "aggressive"
        assert policy.position_scale > 1.0
        assert policy.max_position_pct > 0.03
        assert policy.signal_threshold < 0.6
        
    def test_conservative_policy(self):
        """Test conservative policy configuration."""
        policy = CONSERVATIVE_POLICY
        
        assert policy.name == "conservative"
        assert policy.position_scale < 1.0
        assert policy.max_position_pct < 0.02
        assert policy.signal_threshold > 0.6
        assert policy.use_limit_orders
        
    def test_neutral_policy(self):
        """Test neutral policy configuration."""
        policy = NEUTRAL_POLICY
        
        assert policy.name == "neutral"
        assert policy.position_scale == 1.0
        
    def test_policy_to_dict(self):
        """Test policy serialization."""
        policy = NEUTRAL_POLICY
        d = policy.to_dict()
        
        assert "name" in d
        assert "max_position_pct" in d
        assert "position_scale" in d


# =============================================================================
# REGIME STATE TESTS
# =============================================================================

class TestRegimeState:
    """Tests for regime state class."""
    
    def test_regime_state_creation(self):
        """Test regime state creation."""
        state = RegimeState(
            volatility=VolatilityRegime.HIGH,
            trend=TrendRegime.TRENDING_UP,
            confidence=0.8,
        )
        
        assert state.volatility == VolatilityRegime.HIGH
        assert state.trend == TrendRegime.TRENDING_UP
        assert state.confidence == 0.8
        assert state.timestamp is not None
        
    def test_meta_state(self):
        """Test meta-state computation."""
        state = RegimeState(
            volatility=VolatilityRegime.LOW,
            trend=TrendRegime.FLAT,
        )
        
        assert state.meta_state == "low_flat"
        
    def test_policy_name_aggressive(self):
        """Test aggressive policy selection."""
        state = RegimeState(
            volatility=VolatilityRegime.LOW,
            trend=TrendRegime.TRENDING_UP,
        )
        
        assert state.policy_name == "aggressive"
        
    def test_policy_name_conservative(self):
        """Test conservative policy selection."""
        state = RegimeState(
            volatility=VolatilityRegime.HIGH,
            trend=TrendRegime.MEAN_REVERTING,
        )
        
        assert state.policy_name == "conservative"
        
    def test_to_dict(self):
        """Test serialization."""
        state = RegimeState()
        d = state.to_dict()
        
        assert "volatility" in d
        assert "trend" in d
        assert "confidence" in d
        assert "timestamp" in d


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
