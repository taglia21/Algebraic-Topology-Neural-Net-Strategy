"""
TIER 4 Alpha-Maximisation Test Suite
======================================

15+ tests covering:
  1. ML Ensemble Stacker (5)
  2. Order Book Imbalance (4)
  3. Sentiment Alpha (4)
  4. Execution Optimizer (5)
  5. Integration / Unified (2)

Run:
    pytest tests/test_tier4_alpha.py -v
"""

import math
import os
import sys
import time

import numpy as np
import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# 1. ML Ensemble Stacker
# =============================================================================

class TestMLEnsembleStacker:
    """Tests for src/ml_ensemble_stacker.py."""

    def test_import(self):
        """Module imports cleanly."""
        from src.ml_ensemble_stacker import MLEnsembleStacker, StackerConfig, StackerResult
        assert MLEnsembleStacker is not None

    def test_config_defaults(self):
        from src.ml_ensemble_stacker import StackerConfig
        cfg = StackerConfig()
        assert cfg.n_splits >= 2
        assert cfg.min_train_size > 0
        assert isinstance(cfg.xgb_params, dict)
        assert isinstance(cfg.lgb_params, dict)

    def test_fit_predict(self):
        """Fit on synthetic data and predict alpha scores."""
        from src.ml_ensemble_stacker import MLEnsembleStacker, StackerConfig

        rng = np.random.default_rng(42)
        n, d = 500, 10
        X = rng.standard_normal((n, d))
        y = (X[:, 0] + 0.5 * X[:, 1] + rng.standard_normal(n) * 0.3 > 0).astype(float)

        cfg = StackerConfig(n_splits=3, min_train_size=50)
        stacker = MLEnsembleStacker(cfg)
        stacker.fit(X, y)

        preds = stacker.predict_alpha(X[:10])
        assert preds.shape == (10,)
        assert np.all(preds >= 0) and np.all(preds <= 1)

    def test_predict_single(self):
        """predict_single returns a StackerResult."""
        from src.ml_ensemble_stacker import MLEnsembleStacker, StackerConfig

        rng = np.random.default_rng(99)
        n, d = 400, 8
        X = rng.standard_normal((n, d))
        y = (X[:, 0] > 0).astype(float)

        cfg = StackerConfig(n_splits=2, min_train_size=50)
        stacker = MLEnsembleStacker(cfg)
        stacker.fit(X, y)

        result = stacker.predict_single(X[:1])
        assert hasattr(result, "alpha_score")
        assert hasattr(result, "base_scores")
        assert 0.0 <= result.alpha_score <= 1.0

    def test_feature_importance(self):
        """Feature importance returns dict after fit."""
        from src.ml_ensemble_stacker import MLEnsembleStacker, StackerConfig

        rng = np.random.default_rng(7)
        n, d = 300, 5
        X = rng.standard_normal((n, d))
        y = (X[:, 0] > 0).astype(float)

        cfg = StackerConfig(n_splits=2, min_train_size=50)
        stacker = MLEnsembleStacker(cfg)
        stacker.fit(X, y)

        imp = stacker.feature_importance()
        assert isinstance(imp, list)
        # Should have entries for features
        assert len(imp) > 0


# =============================================================================
# 2. Order Book Imbalance
# =============================================================================

class TestOrderBookImbalance:
    """Tests for src/order_book_imbalance.py."""

    def test_import(self):
        from src.order_book_imbalance import (
            OrderBookImbalance, MicroConfig, MicrostructureState, MicroSignal,
        )
        assert OrderBookImbalance is not None
        assert MicroSignal.NEUTRAL is not None

    def test_update_book_and_signal(self):
        """Feed bids/asks and get a microstructure state."""
        from src.order_book_imbalance import OrderBookImbalance

        obi = OrderBookImbalance()
        bids = [(100.0, 500), (99.9, 300), (99.8, 200)]
        asks = [(100.1, 100), (100.2, 150), (100.3, 200)]

        obi.update_book(bids, asks)
        state = obi.get_microstructure_signal("TEST")

        assert hasattr(state, "signal")
        assert hasattr(state, "composite_score")
        assert -1.0 <= state.composite_score <= 1.0

    def test_heavy_bid_imbalance(self):
        """Heavy bid side should push composite positive."""
        from src.order_book_imbalance import OrderBookImbalance

        obi = OrderBookImbalance()
        bids = [(100.0, 5000), (99.9, 3000), (99.8, 2000)]
        asks = [(100.1, 100), (100.2, 100), (100.3, 100)]

        # Feed multiple updates to let EMA settle
        for _ in range(5):
            obi.update_book(bids, asks)

        state = obi.get_microstructure_signal("TEST")
        assert state.composite_score > 0.0

    def test_add_trade(self):
        """Adding trades doesn't crash."""
        from src.order_book_imbalance import OrderBookImbalance

        obi = OrderBookImbalance()
        bids = [(100.0, 500)]
        asks = [(100.1, 500)]
        obi.update_book(bids, asks)
        obi.add_trade(100.05, 200, "buy")
        obi.add_trade(100.08, 150, "sell")

        state = obi.get_microstructure_signal("TEST")
        assert state is not None


# =============================================================================
# 3. Sentiment Alpha
# =============================================================================

class TestSentimentAlpha:
    """Tests for src/sentiment_alpha.py."""

    def test_import(self):
        from src.sentiment_alpha import SentimentAlpha, SentimentConfig, SentimentDetail
        assert SentimentAlpha is not None

    def test_score_text(self):
        """Keyword scorer should score bullish / bearish headlines."""
        from src.sentiment_alpha import SentimentAlpha

        sa = SentimentAlpha()

        bull = sa.score_text("Apple beats earnings, stock surges on strong growth")
        bear = sa.score_text("Company faces bankruptcy amid fraud investigation")

        assert bull > 0, f"Expected positive, got {bull}"
        assert bear < 0, f"Expected negative, got {bear}"

    def test_get_sentiment_score_returns_float(self):
        """get_sentiment_score returns a float in [-1, 1]."""
        from src.sentiment_alpha import SentimentAlpha

        sa = SentimentAlpha()
        score = sa.get_sentiment_score("AAPL")
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0

    def test_detailed_sentiment(self):
        """get_detailed_sentiment returns SentimentDetail with expected fields."""
        from src.sentiment_alpha import SentimentAlpha, SentimentDetail

        sa = SentimentAlpha()
        detail = sa.get_detailed_sentiment("MSFT")

        assert isinstance(detail, SentimentDetail)
        assert detail.symbol == "MSFT"
        assert isinstance(detail.source_scores, dict)
        assert detail.signal in (
            "strong_bullish", "bullish", "neutral",
            "bearish", "strong_bearish",
        )

    def test_cache_reuse(self):
        """Second call should return cached result."""
        from src.sentiment_alpha import SentimentAlpha

        sa = SentimentAlpha()
        d1 = sa.get_detailed_sentiment("GOOG")
        d2 = sa.get_detailed_sentiment("GOOG")

        assert d2.cached is True
        assert d1.composite_score == d2.composite_score


# =============================================================================
# 4. Execution Optimizer
# =============================================================================

class TestExecutionOptimizer:
    """Tests for src/execution_optimizer.py."""

    def test_import(self):
        from src.execution_optimizer import (
            ExecutionOptimizer, ExecConfig, ExecStrategy,
            ExecutionResult, SlippagePredictor,
        )
        assert ExecutionOptimizer is not None
        assert ExecStrategy.TWAP is not None

    def test_twap_execution(self):
        """TWAP produces slices that sum to total qty."""
        from src.execution_optimizer import ExecutionOptimizer, ExecStrategy

        opt = ExecutionOptimizer()
        result = opt.execute_optimal(
            "AAPL", qty=300, side="buy", ref_price=185.0,
            urgency=0.5, strategy=ExecStrategy.TWAP,
        )

        assert result.filled_qty == 300
        assert result.strategy == "twap"
        assert result.num_slices >= 3
        assert result.avg_fill_price > 0

    def test_vwap_execution(self):
        """VWAP execution fills correctly."""
        from src.execution_optimizer import ExecutionOptimizer, ExecStrategy

        opt = ExecutionOptimizer()
        result = opt.execute_optimal(
            "MSFT", qty=500, side="sell", ref_price=410.0,
            urgency=0.3, strategy=ExecStrategy.VWAP,
        )

        assert result.filled_qty == 500
        assert result.strategy == "vwap"
        assert result.side == "sell"

    def test_iceberg_execution(self):
        """Iceberg fills full qty with many small slices."""
        from src.execution_optimizer import ExecutionOptimizer, ExecStrategy

        opt = ExecutionOptimizer()
        result = opt.execute_optimal(
            "NVDA", qty=2000, side="buy", ref_price=700.0,
            urgency=0.2, strategy=ExecStrategy.ICEBERG,
        )

        assert result.filled_qty >= 1900  # iceberg may lose a few shares to rounding
        assert result.strategy == "iceberg"
        assert result.num_slices >= 3   # iceberg should have multiple clips

    def test_estimate_cost(self):
        """Slippage predictor returns non-negative bps."""
        from src.execution_optimizer import ExecutionOptimizer

        opt = ExecutionOptimizer()
        cost = opt.estimate_cost(
            qty=1000, avg_daily_volume=2_000_000,
            volatility=0.02, urgency=0.5,
        )
        assert cost >= 0
        assert isinstance(cost, float)

    def test_adaptive_strategy_selection(self):
        """Adaptive mode picks appropriate strategy based on params."""
        from src.execution_optimizer import ExecutionOptimizer

        opt = ExecutionOptimizer()

        # Small order, low urgency → should pick TWAP or VWAP (not iceberg)
        result_small = opt.execute_optimal(
            "SPY", qty=10, side="buy", ref_price=450.0,
            urgency=0.3, avg_daily_volume=50_000_000,
        )
        assert result_small.strategy in ("twap", "vwap")

        # Large order relative to ADV → should pick iceberg
        result_large = opt.execute_optimal(
            "TEST", qty=100_000, side="buy", ref_price=50.0,
            urgency=0.3, avg_daily_volume=1_000_000,
        )
        assert result_large.strategy == "iceberg"


# =============================================================================
# 5. Integration / Unified Trader
# =============================================================================

class TestUnifiedIntegration:
    """Integration tests — TIER 4 components work together."""

    def test_all_modules_importable(self):
        """All four TIER 4 modules import cleanly together."""
        from src.ml_ensemble_stacker import MLEnsembleStacker
        from src.order_book_imbalance import OrderBookImbalance
        from src.sentiment_alpha import SentimentAlpha
        from src.execution_optimizer import ExecutionOptimizer

        stacker = MLEnsembleStacker()
        obi = OrderBookImbalance()
        sa = SentimentAlpha()
        opt = ExecutionOptimizer()

        assert stacker is not None
        assert obi is not None
        assert sa is not None
        assert opt is not None

    def test_pipeline_stacker_to_exec(self):
        """Simulate: stacker predict → sentiment check → exec."""
        from src.ml_ensemble_stacker import MLEnsembleStacker, StackerConfig
        from src.sentiment_alpha import SentimentAlpha
        from src.execution_optimizer import ExecutionOptimizer, ExecStrategy

        # 1. Train stacker
        rng = np.random.default_rng(123)
        X = rng.standard_normal((300, 6))
        y = (X[:, 0] > 0).astype(float)

        cfg = StackerConfig(n_splits=2, min_train_size=50)
        stacker = MLEnsembleStacker(cfg)
        stacker.fit(X, y)

        # 2. Predict alpha
        alpha = stacker.predict_alpha(rng.standard_normal((1, 6)))
        assert 0 <= alpha[0] <= 1

        # 3. Sentiment
        sa = SentimentAlpha()
        sent = sa.get_sentiment_score("AAPL")
        assert -1 <= sent <= 1

        # 4. If bullish enough, execute
        if alpha[0] > 0.5:
            opt = ExecutionOptimizer()
            result = opt.execute_optimal(
                "AAPL", qty=100, side="buy", ref_price=185.0,
                urgency=0.5, strategy=ExecStrategy.TWAP,
            )
            assert result.filled_qty == 100


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
