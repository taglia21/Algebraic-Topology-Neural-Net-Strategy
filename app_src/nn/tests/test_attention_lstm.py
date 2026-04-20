"""Tests for nn/models/attention_lstm.py."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from nn.models.attention_lstm import AttentionLSTMPredictor
from nn.models.lstm_predictor import BasePredictor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def default_model() -> AttentionLSTMPredictor:
    return AttentionLSTMPredictor(input_size=10)


@pytest.fixture()
def batch() -> torch.Tensor:
    """(batch=4, seq_len=20, features=10)."""
    return torch.randn(4, 20, 10)


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

class TestInstantiation:
    def test_default_config(self) -> None:
        model = AttentionLSTMPredictor(input_size=10)
        assert model.input_size == 10
        assert model.hidden_size == 128
        assert model.num_heads == 4
        assert model.num_classes == 3

    def test_custom_config(self) -> None:
        model = AttentionLSTMPredictor(
            input_size=16, hidden_size=64, num_heads=2,
            num_layers=3, dropout=0.1, num_classes=5,
        )
        assert model.hidden_size == 64
        assert model.num_heads == 2
        assert model.num_classes == 5

    def test_is_base_predictor(self) -> None:
        model = AttentionLSTMPredictor(input_size=10)
        assert isinstance(model, BasePredictor)


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

class TestForward:
    def test_output_shape(self, default_model: AttentionLSTMPredictor, batch: torch.Tensor) -> None:
        logits = default_model(batch)
        assert logits.shape == (4, 3)

    def test_with_lengths(self, default_model: AttentionLSTMPredictor) -> None:
        x = torch.randn(3, 20, 10)
        lengths = torch.tensor([20, 15, 10])
        logits = default_model(x, lengths=lengths)
        assert logits.shape == (3, 3)

    def test_single_timestep(self, default_model: AttentionLSTMPredictor) -> None:
        default_model.eval()  # BN needs eval mode for single-sample batches
        x = torch.randn(1, 1, 10)
        logits = default_model(x)
        assert logits.shape == (1, 3)


# ---------------------------------------------------------------------------
# Attention weights
# ---------------------------------------------------------------------------

class TestAttentionWeights:
    def test_weights_none_before_forward(self, default_model: AttentionLSTMPredictor) -> None:
        assert default_model.get_attention_weights() is None

    def test_weights_populated_after_forward(
        self, default_model: AttentionLSTMPredictor, batch: torch.Tensor,
    ) -> None:
        default_model(batch)
        weights = default_model.get_attention_weights()
        assert weights is not None

    def test_weights_shape(self, default_model: AttentionLSTMPredictor, batch: torch.Tensor) -> None:
        default_model(batch)
        weights = default_model.get_attention_weights()
        # (batch, seq_len, seq_len) — averaged across heads
        assert weights is not None
        assert weights.shape == (4, 20, 20)

    def test_weights_sum_to_one(self, default_model: AttentionLSTMPredictor, batch: torch.Tensor) -> None:
        default_model.eval()
        default_model(batch)
        weights = default_model.get_attention_weights()
        assert weights is not None
        # Each row should sum to ~1 (attention over keys)
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=0.15)


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

class TestPredict:
    def test_predict_interface(self, default_model: AttentionLSTMPredictor, batch: torch.Tensor) -> None:
        result = default_model.predict(batch)
        assert "direction" in result
        assert "probabilities" in result
        assert "confidence" in result
        assert result["direction"] in {0, 1, 2}
        assert 0.0 <= result["confidence"] <= 1.0


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

class TestGradients:
    def test_backward_runs(self, default_model: AttentionLSTMPredictor, batch: torch.Tensor) -> None:
        default_model.train()
        logits = default_model(batch)
        loss = logits.sum()
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in default_model.parameters()
        )
        assert has_grad


# ---------------------------------------------------------------------------
# Save / Load roundtrip
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_load_roundtrip(self, default_model: AttentionLSTMPredictor, batch: torch.Tensor) -> None:
        default_model.eval()
        with torch.no_grad():
            original_out = default_model(batch)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "model.pt")
            torch.save(default_model.state_dict(), path)

            loaded = AttentionLSTMPredictor(input_size=10)
            loaded.load_state_dict(torch.load(path, weights_only=True))
            loaded.eval()

            with torch.no_grad():
                loaded_out = loaded(batch)

        assert torch.allclose(original_out, loaded_out, atol=1e-6)


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

class TestFeatureImportance:
    def test_feature_importance_shape(self, default_model: AttentionLSTMPredictor) -> None:
        imp = default_model.get_feature_importance()
        assert isinstance(imp, np.ndarray)
        assert imp.shape == (10,)
