"""
nn/data_loader.py
=================
PyTorch Dataset and DataLoader utilities for financial time series.

Provides sliding-window datasets, purged k-fold cross-validation,
and z-score normalisation with expanding windows to prevent data leakage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Direction labels
_UP = 2
_FLAT = 1
_DOWN = 0


def direction_labels(
    returns: pd.Series,
    threshold: float = 0.005,
) -> pd.Series:
    """Convert returns to directional labels.

    Parameters
    ----------
    returns : pd.Series
        Simple returns series.
    threshold : float
        Threshold for up/down classification (default ±0.5%).

    Returns
    -------
    pd.Series
        Integer labels: 0=down, 1=flat, 2=up.
    """
    labels = pd.Series(_FLAT, index=returns.index, dtype=np.int64)
    labels[returns > threshold] = _UP
    labels[returns < -threshold] = _DOWN
    return labels


class TimeSeriesDataset(Dataset):
    """Sliding-window dataset for sequential financial features.

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix (T, F) with DatetimeIndex.
    target : pd.Series
        Target labels aligned with features index.
    window : int
        Number of past trading days per sample (default 60).
    normalize : bool
        Apply expanding-window z-score normalisation (default True).
    """

    def __init__(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        window: int = 60,
        normalize: bool = True,
    ) -> None:
        # Align features and target
        common_idx = features.index.intersection(target.index)
        features = features.loc[common_idx]
        target = target.loc[common_idx]

        self.window = window
        self.feature_names = list(features.columns)

        values = features.values.astype(np.float64)
        targets = target.values.astype(np.int64)

        # Expanding-window z-score (only uses past data)
        if normalize:
            values = self._expanding_zscore(values)

        self._features = values
        self._targets = targets

        # Valid sample indices (need at least `window` rows of history)
        self._valid_start = window
        self._length = max(0, len(self._features) - window)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx
        end = idx + self.window
        x = self._features[start:end]
        y = self._targets[end]  # next-day direction (t+1)
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )

    @staticmethod
    def _expanding_zscore(data: np.ndarray) -> np.ndarray:
        """Z-score normalise each feature using an expanding window.

        Only past observations are used (no future leakage).
        """
        result = np.zeros_like(data)
        for i in range(len(data)):
            window = data[: i + 1]
            mean = window.mean(axis=0)
            std = window.std(axis=0)
            std[std == 0] = 1.0
            result[i] = (data[i] - mean) / std
        return result


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate variable-length sequences with padding.

    Returns
    -------
    tuple
        (padded_features, targets, lengths) where padded_features is
        (batch, max_len, features) and lengths is (batch,).
    """
    sequences, targets = zip(*batch)
    lengths = torch.tensor([s.size(0) for s in sequences], dtype=torch.long)

    max_len = int(lengths.max().item())
    feat_dim = sequences[0].size(1)

    padded = torch.zeros(len(sequences), max_len, feat_dim)
    for i, seq in enumerate(sequences):
        padded[i, : seq.size(0)] = seq

    targets_tensor = torch.stack(targets)
    return padded, targets_tensor, lengths


@dataclass
class _FoldIndices:
    """Train / test index split for a single fold."""
    train: np.ndarray
    test: np.ndarray


class PurgedKFold:
    """Walk-forward cross-validation with purge gap and embargo.

    Prevents information leakage in time-series by maintaining a gap
    between training and test sets.

    Parameters
    ----------
    n_splits : int
        Number of folds (default 5).
    purge_gap : int
        Number of samples to drop between train and test (default 5).
    embargo : int
        Number of samples to drop after test set (default 5).
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_gap: int = 5,
        embargo: int = 5,
    ) -> None:
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo = embargo

    def split(self, n_samples: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test indices for each fold.

        Uses an expanding-window approach: each fold's test set is a
        contiguous block, and the training set is everything before
        the purge gap.

        Parameters
        ----------
        n_samples : int
            Total number of samples.

        Yields
        ------
        tuple[np.ndarray, np.ndarray]
            (train_indices, test_indices) for each fold.
        """
        fold_size = n_samples // (self.n_splits + 1)

        for k in range(self.n_splits):
            # Test set: block k+1
            test_start = (k + 1) * fold_size
            test_end = min(test_start + fold_size, n_samples)

            if test_start >= n_samples:
                break

            # Train end: purge_gap before test
            train_end = max(0, test_start - self.purge_gap)

            if train_end <= 0:
                continue

            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)

            yield train_idx, test_idx
