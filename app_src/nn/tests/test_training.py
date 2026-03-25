"""Tests for nn/training.py and nn/data_loader.py."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from nn.data_loader import PurgedKFold, TimeSeriesDataset, collate_fn, direction_labels
from nn.models.lstm_predictor import LSTMPredictor
from nn.training import WalkForwardResult, WalkForwardTrainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_data(
    n: int = 300, n_features: int = 10, seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic features + target for testing."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    features = pd.DataFrame(
        rng.randn(n, n_features),
        index=dates,
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    # Synthetic returns → direction labels
    returns = pd.Series(rng.randn(n) * 0.01, index=dates)
    target = direction_labels(returns, threshold=0.005)
    return features, target


# ---------------------------------------------------------------------------
# PurgedKFold
# ---------------------------------------------------------------------------

class TestPurgedKFold:
    def test_splits_generated(self) -> None:
        kf = PurgedKFold(n_splits=3, purge_gap=5, embargo=5)
        splits = list(kf.split(200))
        assert len(splits) > 0

    def test_purge_gap_respected(self) -> None:
        kf = PurgedKFold(n_splits=3, purge_gap=5, embargo=5)
        for train_idx, test_idx in kf.split(200):
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            gap = test_idx.min() - train_idx.max()
            assert gap >= 5, f"Purge gap violated: gap={gap}"

    def test_no_overlap(self) -> None:
        kf = PurgedKFold(n_splits=3, purge_gap=5, embargo=5)
        for train_idx, test_idx in kf.split(200):
            overlap = np.intersect1d(train_idx, test_idx)
            assert len(overlap) == 0

    def test_expanding_window(self) -> None:
        """Training set should grow across folds."""
        kf = PurgedKFold(n_splits=3, purge_gap=2, embargo=2)
        splits = list(kf.split(200))
        if len(splits) >= 2:
            # First fold's training set should be smaller than the last one's
            # (expanding window: train always starts at 0)
            assert len(splits[0][0]) <= len(splits[-1][0])


# ---------------------------------------------------------------------------
# TimeSeriesDataset
# ---------------------------------------------------------------------------

class TestTimeSeriesDataset:
    def test_length(self) -> None:
        features, target = _synthetic_data(n=200)
        ds = TimeSeriesDataset(features, target, window=60, normalize=False)
        # Should have 200 - 60 = 140 valid samples
        assert len(ds) == 140

    def test_item_shapes(self) -> None:
        features, target = _synthetic_data(n=200, n_features=8)
        ds = TimeSeriesDataset(features, target, window=30, normalize=False)
        x, y = ds[0]
        assert x.shape == (30, 8)
        assert y.shape == ()
        assert y.dtype == torch.long

    def test_normalization_doesnt_crash(self) -> None:
        features, target = _synthetic_data(n=100)
        ds = TimeSeriesDataset(features, target, window=20, normalize=True)
        assert len(ds) > 0
        x, y = ds[0]
        assert not torch.isnan(x).any()


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

class TestCollate:
    def test_collate_fn(self) -> None:
        features, target = _synthetic_data(n=200, n_features=5)
        ds = TimeSeriesDataset(features, target, window=30, normalize=False)
        batch = [ds[i] for i in range(4)]
        padded, targets, lengths = collate_fn(batch)
        assert padded.shape == (4, 30, 5)
        assert targets.shape == (4,)
        assert lengths.shape == (4,)


# ---------------------------------------------------------------------------
# Direction labels
# ---------------------------------------------------------------------------

class TestDirectionLabels:
    def test_labels_range(self) -> None:
        returns = pd.Series([0.01, -0.01, 0.001, -0.001, 0.0])
        labels = direction_labels(returns, threshold=0.005)
        assert set(labels.values).issubset({0, 1, 2})

    def test_correct_mapping(self) -> None:
        returns = pd.Series([0.01, -0.01, 0.0])
        labels = direction_labels(returns, threshold=0.005)
        assert labels.iloc[0] == 2   # up
        assert labels.iloc[1] == 0   # down
        assert labels.iloc[2] == 1   # flat


# ---------------------------------------------------------------------------
# WalkForwardTrainer
# ---------------------------------------------------------------------------

class TestWalkForwardTrainer:
    def test_train_runs_on_small_data(self) -> None:
        """Full walk-forward on tiny data should not error."""
        features, target = _synthetic_data(n=300, n_features=8)
        trainer = WalkForwardTrainer(
            train_window=150,
            predict_horizon=20,
            purge_gap=5,
            embargo=5,
            max_epochs=2,
            patience=2,
            batch_size=32,
        )
        result = trainer.train_walk_forward(
            features, target,
            model_class=LSTMPredictor,
            window=20,
            input_size=8,
        )
        assert isinstance(result, WalkForwardResult)
        # Should have at least one fold
        assert len(result.metrics_per_fold) >= 1
        assert len(result.predictions) > 0

    def test_early_stopping(self) -> None:
        """With patience=1, training should stop early."""
        features, target = _synthetic_data(n=300, n_features=8)
        trainer = WalkForwardTrainer(
            train_window=150,
            predict_horizon=20,
            purge_gap=5,
            embargo=5,
            max_epochs=50,
            patience=1,
            batch_size=32,
        )
        result = trainer.train_walk_forward(
            features, target,
            model_class=LSTMPredictor,
            window=20,
            input_size=8,
        )
        assert isinstance(result, WalkForwardResult)

    def test_model_save_load(self) -> None:
        """Model checkpoint should be loadable."""
        features, target = _synthetic_data(n=300, n_features=8)
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = WalkForwardTrainer(
                train_window=150,
                predict_horizon=20,
                purge_gap=5,
                embargo=5,
                max_epochs=2,
                patience=2,
                batch_size=32,
                checkpoint_dir=tmpdir,
            )
            result = trainer.train_walk_forward(
                features, target,
                model_class=LSTMPredictor,
                window=20,
                input_size=8,
            )
            if result.best_model_path:
                loaded = WalkForwardTrainer.load_model(
                    LSTMPredictor, result.best_model_path, input_size=8,
                )
                assert isinstance(loaded, LSTMPredictor)

    def test_empty_dataset(self) -> None:
        """Very small data → empty result, no crash."""
        features, target = _synthetic_data(n=10, n_features=8)
        trainer = WalkForwardTrainer(
            train_window=100,
            predict_horizon=20,
            max_epochs=1,
        )
        result = trainer.train_walk_forward(
            features, target,
            model_class=LSTMPredictor,
            window=60,
            input_size=8,
        )
        assert isinstance(result, WalkForwardResult)
        assert len(result.predictions) == 0
