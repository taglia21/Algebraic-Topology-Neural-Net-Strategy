"""Tests for TDA and NN strategy signal generation."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch

from ensemble.strategy_nn import NNDirectionalStrategy
from ensemble.strategy_tda import TDADiffusionStrategy


# ---------------------------------------------------------------------------
# TDA Diffusion Strategy
# ---------------------------------------------------------------------------


class TestTDADiffusionStrategy:
    """Tests for TDADiffusionStrategy."""

    @pytest.fixture()
    def strategy(self) -> TDADiffusionStrategy:
        return TDADiffusionStrategy(residual_threshold=1.5)

    @pytest.fixture()
    def sample_residuals(self) -> pd.DataFrame:
        """Z-scored residuals: some stocks above/below threshold."""
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        data = {
            "AAPL": [2.0, -0.5, -2.5],   # day1: SHORT, day2: NEUTRAL, day3: LONG
            "MSFT": [-1.8, 1.6, 0.3],     # day1: LONG, day2: SHORT, day3: NEUTRAL
            "GOOG": [0.1, -0.2, 0.0],     # all NEUTRAL
        }
        return pd.DataFrame(data, index=dates)

    def test_signal_directions(self, strategy, sample_residuals):
        """Positive residual → SHORT, negative → LONG, within threshold → NEUTRAL."""
        signals = strategy.generate_signals(sample_residuals)

        # Check AAPL day 1: z=2.0 > 1.5 → SHORT
        aapl_d1 = signals[
            (signals["ticker"] == "AAPL")
            & (signals["timestamp"] == sample_residuals.index[0])
        ]
        assert aapl_d1.iloc[0]["direction"] == "SHORT"

        # Check AAPL day 3: z=-2.5 < -1.5 → LONG
        aapl_d3 = signals[
            (signals["ticker"] == "AAPL")
            & (signals["timestamp"] == sample_residuals.index[2])
        ]
        assert aapl_d3.iloc[0]["direction"] == "LONG"

        # Check GOOG all days: within threshold → NEUTRAL
        goog = signals[signals["ticker"] == "GOOG"]
        assert (goog["direction"] == "NEUTRAL").all()

    def test_signal_strength_range(self, strategy, sample_residuals):
        """All strengths should be in [0, 1]."""
        signals = strategy.generate_signals(sample_residuals)
        assert (signals["strength"] >= 0).all()
        assert (signals["strength"] <= 1).all()

    def test_regime_gating_crash(self, strategy):
        """CRASH regime should reduce signal strength by 50%."""
        dates = pd.date_range("2024-01-01", periods=1, freq="B")
        data = {"AAPL": [3.0], "regime": ["CRASH"]}
        df = pd.DataFrame(data, index=dates)

        signals = strategy.generate_signals(df)
        crash_strength = signals.iloc[0]["strength"]

        # Same signal without regime
        data_normal = {"AAPL": [3.0]}
        df_normal = pd.DataFrame(data_normal, index=dates)
        signals_normal = strategy.generate_signals(df_normal)
        normal_strength = signals_normal.iloc[0]["strength"]

        assert crash_strength == pytest.approx(normal_strength * 0.5, abs=1e-6)

    def test_regime_gating_stressed(self, strategy):
        """STRESSED regime should reduce signal strength by 25%."""
        dates = pd.date_range("2024-01-01", periods=1, freq="B")
        data = {"AAPL": [3.0], "regime": ["STRESSED"]}
        df = pd.DataFrame(data, index=dates)

        signals = strategy.generate_signals(df)
        stressed_strength = signals.iloc[0]["strength"]

        data_normal = {"AAPL": [3.0]}
        df_normal = pd.DataFrame(data_normal, index=dates)
        signals_normal = strategy.generate_signals(df_normal)
        normal_strength = signals_normal.iloc[0]["strength"]

        assert stressed_strength == pytest.approx(normal_strength * 0.75, abs=1e-6)

    def test_regime_numeric_encoding(self, strategy):
        """Should accept numeric regime encoding (0/1/2)."""
        dates = pd.date_range("2024-01-01", periods=1, freq="B")
        data = {"AAPL": [3.0], "regime": [2]}  # 2 = CRASH
        df = pd.DataFrame(data, index=dates)

        signals = strategy.generate_signals(df)
        assert signals.iloc[0]["regime"] == "CRASH"

    def test_get_top_signals(self, strategy, sample_residuals):
        """get_top_signals returns top N non-neutral signals."""
        strategy.generate_signals(sample_residuals)
        top = strategy.get_top_signals(n=2)
        assert len(top) <= 2
        assert (top["direction"] != "NEUTRAL").all()
        # Should be sorted by strength descending
        if len(top) > 1:
            assert top.iloc[0]["strength"] >= top.iloc[1]["strength"]

    def test_empty_input(self, strategy):
        """Empty DataFrame should return empty signals."""
        df = pd.DataFrame(columns=["AAPL", "MSFT"])
        signals = strategy.generate_signals(df)
        assert signals.empty or len(signals) == 0

    def test_output_columns(self, strategy, sample_residuals):
        """Output should have required columns."""
        signals = strategy.generate_signals(sample_residuals)
        required = {"ticker", "direction", "strength", "regime", "timestamp"}
        assert required.issubset(set(signals.columns))


# ---------------------------------------------------------------------------
# NN Directional Strategy
# ---------------------------------------------------------------------------


class TestNNDirectionalStrategy:
    """Tests for NNDirectionalStrategy."""

    @pytest.fixture()
    def strategy(self) -> NNDirectionalStrategy:
        return NNDirectionalStrategy(high_threshold=0.65, low_threshold=0.50)

    @pytest.fixture()
    def mock_model(self):
        """Mock NN model that returns configurable predictions."""
        model = MagicMock()
        return model

    @pytest.fixture()
    def sample_features(self) -> pd.DataFrame:
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        np.random.seed(42)
        return pd.DataFrame(
            np.random.randn(3, 10),
            index=dates,
            columns=[f"feat_{i}" for i in range(10)],
        )

    def test_strong_signal(self, strategy, mock_model, sample_features):
        """Confidence > high_threshold → direction preserved, high strength."""
        mock_model.predict.return_value = {
            "direction": 2,  # UP → LONG
            "probabilities": torch.tensor([[0.1, 0.2, 0.7]]),
            "confidence": 0.7,
        }
        signals = strategy.generate_signals(sample_features, mock_model)
        for _, row in signals.iterrows():
            assert row["direction"] == "LONG"
            assert row["strength"] >= 0.65

    def test_weak_signal(self, strategy, mock_model, sample_features):
        """Confidence between low/high → WEAK signal with reduced strength."""
        mock_model.predict.return_value = {
            "direction": 2,  # UP → LONG
            "probabilities": torch.tensor([[0.15, 0.25, 0.60]]),
            "confidence": 0.60,
        }
        signals = strategy.generate_signals(sample_features, mock_model)
        for _, row in signals.iterrows():
            assert row["direction"] == "LONG"
            # Weak signals capped at 0.5
            assert row["strength"] <= 0.5

    def test_neutral_below_threshold(self, strategy, mock_model, sample_features):
        """Confidence < low_threshold → NEUTRAL."""
        mock_model.predict.return_value = {
            "direction": 2,
            "probabilities": torch.tensor([[0.3, 0.3, 0.4]]),
            "confidence": 0.4,
        }
        signals = strategy.generate_signals(sample_features, mock_model)
        assert (signals["direction"] == "NEUTRAL").all()
        assert (signals["strength"] == 0.0).all()

    def test_crash_regime_overrides_long(self, strategy, mock_model, sample_features):
        """In CRASH regime, LONG signals are overridden to NEUTRAL."""
        mock_model.predict.return_value = {
            "direction": 2,  # LONG
            "probabilities": torch.tensor([[0.05, 0.15, 0.80]]),
            "confidence": 0.80,
        }
        signals = strategy.generate_signals(
            sample_features, mock_model, regime="CRASH"
        )
        assert (signals["direction"] == "NEUTRAL").all()

    def test_crash_regime_preserves_short(self, strategy, mock_model, sample_features):
        """In CRASH regime, SHORT signals are preserved."""
        mock_model.predict.return_value = {
            "direction": 0,  # DOWN → SHORT
            "probabilities": torch.tensor([[0.80, 0.15, 0.05]]),
            "confidence": 0.80,
        }
        signals = strategy.generate_signals(
            sample_features, mock_model, regime="CRASH"
        )
        assert (signals["direction"] == "SHORT").all()

    def test_output_columns(self, strategy, mock_model, sample_features):
        """Output should have required columns."""
        mock_model.predict.return_value = {
            "direction": 1,
            "probabilities": torch.tensor([[0.3, 0.4, 0.3]]),
            "confidence": 0.4,
        }
        signals = strategy.generate_signals(sample_features, mock_model)
        required = {"ticker", "direction", "strength", "prediction_probs", "timestamp"}
        assert required.issubset(set(signals.columns))

    def test_model_failure_skips_date(self, strategy, sample_features):
        """If model.predict raises, that date is skipped."""
        model = MagicMock()
        model.predict.side_effect = RuntimeError("model failed")

        signals = strategy.generate_signals(sample_features, model)
        assert len(signals) == 0
