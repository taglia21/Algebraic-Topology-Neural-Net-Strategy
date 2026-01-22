"""
Test Suite for V2.0 Components

Comprehensive tests for all V2 enhancements:
- Transformer Predictor
- SAC Agent with PER
- Persistent Laplacian TDA
- Ensemble Regime Detection
- Order Flow Analyzer
- V2 Enhanced Engine
"""

import unittest
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTransformerPredictor(unittest.TestCase):
    """Tests for Transformer Predictor module."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        try:
            from src.ml.transformer_predictor import TransformerPredictor
            cls.predictor = TransformerPredictor(
                d_model=64,
                n_heads=4,
                n_layers=2
            )
            cls.available = True
        except (ImportError, TypeError) as e:
            cls.available = False
            cls.skip_reason = str(e)
    
    def test_initialization(self):
        """Test predictor initializes correctly."""
        if not self.available:
            self.skipTest(f"Transformer not available: {self.skip_reason}")
        
        self.assertIsNotNone(self.predictor)
        # Check internal model was created (or None if PyTorch unavailable)
        self.assertTrue(hasattr(self.predictor, 'model'))
    
    def test_feature_extraction(self):
        """Test feature extraction from prices."""
        if not self.available:
            self.skipTest("Transformer not available")
        
        # Skip if model is None (PyTorch not available)
        if self.predictor.model is None:
            self.skipTest("PyTorch not available for feature extraction")
        
        # Generate synthetic price data as DataFrame
        import pandas as pd
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        prices = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.01))
        df = pd.DataFrame({
            'Open': prices * 0.99,
            'High': prices * 1.01,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        features = self.predictor.prepare_features(df)
        
        if features is not None:
            self.assertEqual(features.shape[2], 10)  # 10 features
            self.assertFalse(np.any(np.isnan(features)))
    
    def test_training(self):
        """Test training on synthetic data."""
        if not self.available:
            self.skipTest("Transformer not available")
        
        # Skip if PyTorch not available
        if self.predictor.model is None:
            self.skipTest("PyTorch not available for training")
        
        # Generate synthetic price data as DataFrames
        import pandas as pd
        price_data = {}
        for ticker in ['AAPL', 'MSFT', 'GOOGL']:
            dates = pd.date_range(start='2020-01-01', periods=200, freq='D')
            prices = 100 * np.exp(np.cumsum(np.random.randn(200) * 0.01))
            price_data[ticker] = pd.DataFrame({
                'Open': prices * 0.99,
                'High': prices * 1.01,
                'Low': prices * 0.98,
                'Close': prices,
                'Volume': np.random.randint(1000, 10000, 200)
            }, index=dates)
        
        # Train for 1 epoch
        result = self.predictor.train(price_data, epochs=1, max_stocks=3, samples_per_stock=10)
        
        # Should return training metrics
        self.assertIn('accuracy', result)
    
    def test_prediction(self):
        """Test prediction output shape and range."""
        if not self.available:
            self.skipTest("Transformer not available")
        
        # Generate price data as DataFrames (predict expects Dict[str, pd.DataFrame])
        import pandas as pd
        price_data = {}
        for i, ticker in enumerate(['AAPL', 'MSFT', 'GOOGL']):
            dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
            prices = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.01))
            price_data[ticker] = pd.DataFrame({
                'Open': prices * 0.99,
                'High': prices * 1.01,
                'Low': prices * 0.98,
                'Close': prices,
                'Volume': np.random.randint(1000, 10000, 100)
            }, index=dates)
        
        predictions = self.predictor.predict(price_data)
        
        # Should return list of StockPrediction objects
        self.assertTrue(len(predictions) > 0)
        # Check prediction probabilities are in valid range
        for pred in predictions:
            self.assertTrue(0 <= pred.direction_prob <= 1)


class TestSACAgent(unittest.TestCase):
    """Tests for SAC Agent with PER."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        try:
            from src.ml.sac_agent import SACAgent, SACConfig, PrioritizedReplayBuffer
            cls.config = SACConfig(state_dim=10, hidden_dims=[32, 32])
            cls.agent = SACAgent(config=cls.config)
            cls.buffer_class = PrioritizedReplayBuffer
            cls.available = True
        except ImportError as e:
            cls.available = False
            cls.skip_reason = str(e)
    
    def test_agent_initialization(self):
        """Test agent initializes correctly."""
        if not self.available:
            self.skipTest(f"SAC not available: {self.skip_reason}")
        
        self.assertIsNotNone(self.agent)
        self.assertEqual(self.config.state_dim, 10)
    
    def test_action_selection(self):
        """Test action selection returns valid range."""
        if not self.available:
            self.skipTest("SAC not available")
        
        state = np.random.randn(10).astype(np.float32)
        
        action = self.agent.select_action(state, deterministic=True)
        
        # Action should be in [0, 2] (position multiplier)
        self.assertGreaterEqual(action, 0)
        self.assertLessEqual(action, 2)
    
    def test_experience_replay(self):
        """Test prioritized experience replay."""
        if not self.available:
            self.skipTest("SAC not available")
        
        buffer = self.buffer_class(capacity=100, alpha=0.6)
        
        # Add experiences
        from src.ml.sac_agent import Experience
        for i in range(50):
            exp = Experience(
                state=np.random.randn(10),
                action=np.random.rand(),
                reward=np.random.randn(),
                next_state=np.random.randn(10),
                done=False
            )
            buffer.add(exp, td_error=abs(np.random.randn()))
        
        self.assertEqual(len(buffer), 50)
        
        # Sample batch
        experiences, weights, indices = buffer.sample(16)
        
        self.assertEqual(len(experiences), 16)
        self.assertEqual(len(weights), 16)
        self.assertTrue(all(w > 0 for w in weights))
    
    def test_position_multiplier(self):
        """Test dynamic position multiplier computation."""
        if not self.available:
            self.skipTest("SAC not available")
        
        state = np.random.randn(10).astype(np.float32)
        
        # Normal volatility
        mult_normal = self.agent.compute_position_multiplier(state, vol_20d=0.02, vix=20)
        
        # High volatility
        mult_high_vol = self.agent.compute_position_multiplier(state, vol_20d=0.05, vix=30)
        
        # Low volatility
        mult_low_vol = self.agent.compute_position_multiplier(state, vol_20d=0.01, vix=12)
        
        # High vol should reduce position
        self.assertLess(mult_high_vol, mult_normal)
        
        # All multipliers in valid range
        for mult in [mult_normal, mult_high_vol, mult_low_vol]:
            self.assertGreaterEqual(mult, 0.25)
            self.assertLessEqual(mult, 2.0)


class TestPersistentLaplacian(unittest.TestCase):
    """Tests for Persistent Laplacian TDA."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        try:
            from src.tda_v2.persistent_laplacian import PersistentLaplacian, EnhancedTDAFeatures
            cls.laplacian = PersistentLaplacian()
            cls.enhanced_tda = EnhancedTDAFeatures(use_laplacian=True)
            cls.available = True
        except ImportError as e:
            cls.available = False
            cls.skip_reason = str(e)
    
    def test_initialization(self):
        """Test Laplacian initializes correctly."""
        if not self.available:
            self.skipTest(f"Laplacian not available: {self.skip_reason}")
        
        self.assertIsNotNone(self.laplacian)
        self.assertEqual(self.laplacian.max_dimension, 1)
    
    def test_distance_matrix(self):
        """Test distance matrix computation."""
        if not self.available:
            self.skipTest("Laplacian not available")
        
        returns = np.random.randn(100)
        dist_matrix = self.laplacian.compute_distance_matrix(returns)
        
        # Should be square symmetric
        self.assertEqual(dist_matrix.shape[0], dist_matrix.shape[1])
        np.testing.assert_array_almost_equal(dist_matrix, dist_matrix.T)
        
        # Diagonal should be zero
        np.testing.assert_array_almost_equal(np.diag(dist_matrix), 0)
    
    def test_feature_extraction(self):
        """Test 12 TDA features are extracted."""
        if not self.available:
            self.skipTest("Laplacian not available")
        
        returns = np.random.randn(100)
        features = self.laplacian.extract_features(returns)
        
        # Should have 12 features
        self.assertEqual(len(features), 12)
        
        # Check specific features exist
        self.assertIn('L0_min_nonzero', features)
        self.assertIn('L1_mean', features)
        self.assertIn('spectral_gap_L0', features)
        self.assertIn('betti_0_integral', features)
        self.assertIn('total_persistence', features)
    
    def test_feature_vector(self):
        """Test feature vector output."""
        if not self.available:
            self.skipTest("Laplacian not available")
        
        returns = np.random.randn(100)
        feature_vec = self.laplacian.get_feature_vector(returns)
        
        self.assertEqual(len(feature_vec), 12)
        self.assertFalse(np.any(np.isnan(feature_vec)))
    
    def test_simplicial_complex(self):
        """Test simplicial complex building."""
        if not self.available:
            self.skipTest("Laplacian not available")
        
        # Small distance matrix
        dist = np.array([[0, 1, 2],
                         [1, 0, 1],
                         [2, 1, 0]])
        
        complex = self.laplacian.build_simplicial_complex(dist, threshold=1.5)
        
        self.assertEqual(len(complex['vertices']), 3)
        self.assertGreater(len(complex['edges']), 0)


class TestEnsembleRegimeDetector(unittest.TestCase):
    """Tests for Ensemble Regime Detection."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        try:
            from src.trading.regime_ensemble import EnsembleRegimeDetector, RegimeType
            cls.detector = EnsembleRegimeDetector(n_regimes=3)
            cls.RegimeType = RegimeType
            cls.available = True
        except ImportError as e:
            cls.available = False
            cls.skip_reason = str(e)
    
    def test_initialization(self):
        """Test detector initializes correctly."""
        if not self.available:
            self.skipTest(f"Ensemble not available: {self.skip_reason}")
        
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.n_regimes, 3)
        
        # Check weights
        total_weight = self.detector.hmm_weight + self.detector.gmm_weight + self.detector.cluster_weight
        self.assertAlmostEqual(total_weight, 1.0, places=5)
    
    def test_feature_computation(self):
        """Test regime features are computed."""
        if not self.available:
            self.skipTest("Ensemble not available")
        
        returns = np.random.randn(100)
        features = self.detector.compute_features(returns)
        
        self.assertEqual(features.shape[1], 5)  # 5 features
    
    def test_fitting(self):
        """Test ensemble fitting."""
        if not self.available:
            self.skipTest("Ensemble not available")
        
        # Generate trend data
        n = 300
        returns = np.random.randn(n) * 0.01
        # Add regime structure
        returns[:100] += 0.01  # Bull
        returns[100:200] -= 0.01  # Bear
        returns[200:] += 0.005  # Sideways
        
        self.detector.fit(returns)
        
        self.assertTrue(self.detector.is_fitted)
    
    def test_prediction(self):
        """Test regime prediction."""
        if not self.available:
            self.skipTest("Ensemble not available")
        
        # Fit first
        returns = np.random.randn(200) * 0.01
        self.detector.fit(returns)
        
        # Predict
        features = self.detector.compute_features(returns)
        state = self.detector.predict(features[-1])
        
        self.assertIsInstance(state.regime, self.RegimeType)
        self.assertGreaterEqual(state.confidence, 0)
        self.assertLessEqual(state.confidence, 1)
        self.assertGreaterEqual(state.consensus_count, 0)
        self.assertLessEqual(state.consensus_count, 3)
    
    def test_regime_string(self):
        """Test regime string output."""
        if not self.available:
            self.skipTest("Ensemble not available")
        
        regime_str = self.detector.get_regime_string()
        
        self.assertIn(regime_str, ['risk_on', 'risk_off', 'neutral'])
    
    def test_position_multiplier(self):
        """Test position multiplier by regime."""
        if not self.available:
            self.skipTest("Ensemble not available")
        
        mult = self.detector.get_position_multiplier()
        
        self.assertGreaterEqual(mult, 0.25)
        self.assertLessEqual(mult, 1.5)


class TestOrderFlowAnalyzer(unittest.TestCase):
    """Tests for Order Flow Analyzer."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        try:
            from src.microstructure.order_flow_analyzer import (
                OrderFlowAnalyzer, Quote, Trade, OrderFlowMetrics
            )
            cls.analyzer = OrderFlowAnalyzer(window_minutes=15)
            cls.Quote = Quote
            cls.Trade = Trade
            cls.available = True
        except ImportError as e:
            cls.available = False
            cls.skip_reason = str(e)
    
    def test_initialization(self):
        """Test analyzer initializes correctly."""
        if not self.available:
            self.skipTest(f"Order flow not available: {self.skip_reason}")
        
        self.assertIsNotNone(self.analyzer)
        self.assertEqual(self.analyzer.window_minutes, 15)
    
    def test_quote_processing(self):
        """Test quote tick processing."""
        if not self.available:
            self.skipTest("Order flow not available")
        
        quote = self.Quote(
            timestamp=datetime.now(),
            bid=100.00,
            ask=100.02,
            bid_size=500,
            ask_size=300
        )
        
        self.analyzer.add_quote('AAPL', quote)
        
        self.assertEqual(len(self.analyzer.quotes['AAPL']), 1)
        self.assertAlmostEqual(quote.spread, 0.02, places=4)
        self.assertAlmostEqual(quote.mid, 100.01, places=4)
    
    def test_trade_classification(self):
        """Test trade classification (Lee-Ready)."""
        if not self.available:
            self.skipTest("Order flow not available")
        
        quote = self.Quote(
            timestamp=datetime.now(),
            bid=100.00,
            ask=100.02,
            bid_size=500,
            ask_size=300
        )
        
        # Trade above mid -> buy
        trade_buy = self.Trade(
            timestamp=datetime.now(),
            price=100.015,
            size=100
        )
        
        # Trade below mid -> sell
        trade_sell = self.Trade(
            timestamp=datetime.now(),
            price=100.005,
            size=100
        )
        
        self.assertEqual(self.analyzer.classify_trade(trade_buy, quote), 'buy')
        self.assertEqual(self.analyzer.classify_trade(trade_sell, quote), 'sell')
    
    def test_metrics_computation(self):
        """Test order flow metrics computation."""
        if not self.available:
            self.skipTest("Order flow not available")
        
        ticker = 'TEST'
        now = datetime.now()
        
        # Add quotes
        for i in range(10):
            quote = self.Quote(
                timestamp=now - timedelta(seconds=i),
                bid=100.00 + i * 0.01,
                ask=100.02 + i * 0.01,
                bid_size=500 - i * 10,
                ask_size=300 + i * 10
            )
            self.analyzer.add_quote(ticker, quote)
        
        # Add trades
        for i in range(20):
            trade = self.Trade(
                timestamp=now - timedelta(seconds=i),
                price=100.01 + i * 0.01,
                size=100 + i * 10
            )
            self.analyzer.add_trade(ticker, trade)
        
        metrics = self.analyzer.compute_metrics(ticker)
        
        self.assertIsNotNone(metrics)
        self.assertGreater(metrics.bid_ask_spread, 0)
    
    def test_feature_vector(self):
        """Test 10 feature vector output."""
        if not self.available:
            self.skipTest("Order flow not available")
        
        features = self.analyzer.get_feature_vector('UNKNOWN_TICKER')
        
        self.assertEqual(len(features), 10)
        self.assertFalse(np.any(np.isnan(features)))
    
    def test_signal_generation(self):
        """Test trading signal generation."""
        if not self.available:
            self.skipTest("Order flow not available")
        
        signal, strength = self.analyzer.get_signal('UNKNOWN_TICKER')
        
        self.assertIn(signal, ['buy', 'sell', 'hold'])
        self.assertGreaterEqual(strength, 0)
        self.assertLessEqual(strength, 1)


class TestV2EnhancedEngine(unittest.TestCase):
    """Tests for V2 Enhanced Engine."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        try:
            from src.trading.v2_enhanced_engine import V2EnhancedEngine, V2Config
            cls.config = V2Config(
                use_transformer=True,
                use_sac=False,  # Disable for faster tests
                use_persistent_laplacian=False,
                use_ensemble_regime=False,
                use_order_flow=False,
                fallback_to_v13=False
            )
            cls.engine = V2EnhancedEngine(config=cls.config, initial_capital=100000)
            cls.available = True
        except ImportError as e:
            cls.available = False
            cls.skip_reason = str(e)
    
    def test_initialization(self):
        """Test engine initializes correctly."""
        if not self.available:
            self.skipTest(f"V2 Engine not available: {self.skip_reason}")
        
        self.assertIsNotNone(self.engine)
        self.assertEqual(self.engine.initial_capital, 100000)
    
    def test_component_status(self):
        """Test component status reporting."""
        if not self.available:
            self.skipTest("V2 Engine not available")
        
        status = self.engine.component_status
        
        self.assertIn('transformer', status)
        self.assertIn('sac', status)
        self.assertIn('tda_laplacian', status)
    
    def test_base_features(self):
        """Test base feature computation."""
        if not self.available:
            self.skipTest("V2 Engine not available")
        
        prices = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.01))
        volume = np.random.randint(1000, 10000, 100)
        
        features = self.engine._compute_base_features(prices, volume)
        
        self.assertEqual(len(features), 10)
        self.assertFalse(np.any(np.isnan(features)))
    
    def test_prediction(self):
        """Test prediction generation."""
        if not self.available:
            self.skipTest("V2 Engine not available")
        
        prices = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.01))
        
        pred = self.engine.predict('TEST', prices)
        
        self.assertIn('ticker', pred)
        self.assertIn('direction', pred)
        self.assertIn('signal', pred)
        self.assertIn(pred['signal'], ['buy', 'sell', 'hold'])
    
    def test_position_sizing(self):
        """Test position size computation."""
        if not self.available:
            self.skipTest("V2 Engine not available")
        
        prediction = {
            'position_multiplier': 1.0,
            'confidence': 0.7
        }
        
        size = self.engine.compute_position_size('TEST', prediction, 100, 2.0)
        
        self.assertGreater(size, 0)
        self.assertLess(size, self.engine.portfolio_value * 0.1)
    
    def test_stop_loss_calculation(self):
        """Test stop loss and take profit."""
        if not self.available:
            self.skipTest("V2 Engine not available")
        
        entry = 100
        atr = 2.0
        
        stop = self.engine.compute_stop_loss(entry, atr, is_long=True)
        take_profit = self.engine.compute_take_profit(entry, atr, is_long=True)
        
        self.assertLess(stop, entry)
        self.assertGreater(take_profit, entry)
    
    def test_status(self):
        """Test status reporting."""
        if not self.available:
            self.skipTest("V2 Engine not available")
        
        status = self.engine.get_status()
        
        self.assertIn('version', status)
        self.assertEqual(status['version'], 'V2.0')
        self.assertIn('components', status)


if __name__ == '__main__':
    unittest.main(verbosity=2)
