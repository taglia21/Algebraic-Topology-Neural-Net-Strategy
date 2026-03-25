"""Tests for MetaAllocator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ensemble.meta_allocator import AllocationResult, MetaAllocator


class TestMetaAllocator:
    """Tests for MetaAllocator."""

    @pytest.fixture()
    def allocator(self) -> MetaAllocator:
        return MetaAllocator(min_history=252, model_type="logistic")

    @pytest.fixture()
    def empty_signals(self) -> pd.DataFrame:
        return pd.DataFrame(columns=["ticker", "direction", "strength", "timestamp"])

    def test_default_allocation_normal(self, allocator, empty_signals):
        """Default mode NORMAL: 50/50 split."""
        result = allocator.allocate(
            empty_signals, empty_signals, {"regime": "NORMAL"}
        )
        assert isinstance(result, AllocationResult)
        assert result.tda_weight == pytest.approx(0.50)
        assert result.nn_weight == pytest.approx(0.50)

    def test_default_allocation_crash(self, allocator, empty_signals):
        """Default mode CRASH: 70/30 TDA/NN."""
        result = allocator.allocate(
            empty_signals, empty_signals, {"regime": "CRASH"}
        )
        assert result.tda_weight == pytest.approx(0.70)
        assert result.nn_weight == pytest.approx(0.30)

    def test_default_allocation_stressed(self, allocator, empty_signals):
        """Default mode STRESSED: 40/60 TDA/NN."""
        result = allocator.allocate(
            empty_signals, empty_signals, {"regime": "STRESSED"}
        )
        assert result.tda_weight == pytest.approx(0.40)
        assert result.nn_weight == pytest.approx(0.60)

    def test_weights_sum_to_one(self, allocator, empty_signals):
        """Weights must always sum to 1."""
        for regime in ["NORMAL", "STRESSED", "CRASH"]:
            result = allocator.allocate(
                empty_signals, empty_signals, {"regime": regime}
            )
            assert result.tda_weight + result.nn_weight == pytest.approx(1.0)

    def test_reasoning_populated(self, allocator, empty_signals):
        """Reasoning string should be non-empty."""
        result = allocator.allocate(
            empty_signals, empty_signals, {"regime": "NORMAL"}
        )
        assert len(result.reasoning) > 0

    def test_update_history(self, allocator):
        """update_history should accumulate returns."""
        allocator.update_history(0.01, 0.02)
        allocator.update_history(-0.005, 0.01)
        assert len(allocator._tda_returns) == 2
        assert len(allocator._nn_returns) == 2

    def test_not_trained_by_default(self, allocator):
        """Allocator should not be trained before training."""
        assert not allocator.is_trained

    def test_train_insufficient_history(self, allocator):
        """Training with insufficient data should not set trained flag."""
        n = 100
        tda = np.random.randn(n) * 0.01
        nn = np.random.randn(n) * 0.01
        regimes = ["NORMAL"] * n
        allocator.train(tda, nn, regimes)
        assert not allocator.is_trained

    def test_train_with_sufficient_history(self):
        """Training with enough data should produce a trained classifier."""
        allocator = MetaAllocator(min_history=50, model_type="logistic")
        n = 200
        np.random.seed(42)
        tda = np.random.randn(n) * 0.01
        nn = np.random.randn(n) * 0.01
        regimes = ["NORMAL"] * n
        allocator.train(tda, nn, regimes)
        assert allocator.is_trained

    def test_trained_allocator_weights(self):
        """Trained allocator should produce valid weights."""
        allocator = MetaAllocator(min_history=50, model_type="logistic")
        n = 200
        np.random.seed(42)
        tda = np.random.randn(n) * 0.01
        nn = np.random.randn(n) * 0.01
        regimes = ["NORMAL"] * n
        allocator.train(tda, nn, regimes)

        empty = pd.DataFrame(columns=["ticker", "direction", "strength", "timestamp"])
        result = allocator.allocate(empty, empty, {"regime": "NORMAL"})

        assert 0.0 <= result.tda_weight <= 1.0
        assert 0.0 <= result.nn_weight <= 1.0
        assert result.tda_weight + result.nn_weight == pytest.approx(1.0, abs=1e-4)

    def test_unknown_regime_defaults_to_normal(self, allocator, empty_signals):
        """Unknown regime falls back to NORMAL weights."""
        result = allocator.allocate(
            empty_signals, empty_signals, {"regime": "UNKNOWN"}
        )
        assert result.tda_weight == pytest.approx(0.50)
        assert result.nn_weight == pytest.approx(0.50)
