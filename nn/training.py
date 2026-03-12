"""
nn/training.py
==============
Walk-forward training pipeline for directional prediction models.

Rolling walk-forward with purge + embargo, early stopping, LR scheduling,
model persistence, and per-fold metrics tracking.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from nn.data_loader import TimeSeriesDataset, collate_fn
from nn.models.lstm_predictor import BasePredictor

logger = logging.getLogger(__name__)

# Defaults
_TRAIN_WINDOW = 756    # ~3 years of trading days
_PREDICT_HORIZON = 21  # predict next month then roll
_PURGE_GAP = 5
_EMBARGO = 5
_MAX_EPOCHS = 100
_PATIENCE = 10
_LR = 1e-3
_BATCH_SIZE = 64


@dataclass
class FoldMetrics:
    """Metrics for a single walk-forward fold."""

    fold: int
    accuracy: float
    precision_per_class: Dict[int, float]
    recall_per_class: Dict[int, float]
    f1_per_class: Dict[int, float]
    confusion_matrix: np.ndarray
    train_loss: float
    val_loss: float


@dataclass
class WalkForwardResult:
    """Aggregated results from a full walk-forward run.

    Attributes
    ----------
    predictions : np.ndarray
        Concatenated out-of-sample predictions.
    actuals : np.ndarray
        Concatenated out-of-sample true labels.
    metrics_per_fold : list[FoldMetrics]
        Per-fold evaluation metrics.
    best_model_path : str
        Path to the best model checkpoint.
    """

    predictions: np.ndarray
    actuals: np.ndarray
    metrics_per_fold: List[FoldMetrics] = field(default_factory=list)
    best_model_path: str = ""


class WalkForwardTrainer:
    """Rolling walk-forward training with purge/embargo.

    Parameters
    ----------
    train_window : int
        Training window in trading days (default 756 ≈ 3 years).
    predict_horizon : int
        Forward prediction period (default 21 days).
    purge_gap : int
        Gap between train and test to avoid leakage (default 5).
    embargo : int
        Embargo after test period (default 5).
    max_epochs : int
        Maximum training epochs per fold (default 100).
    patience : int
        Early-stopping patience (default 10).
    lr : float
        Learning rate for Adam (default 1e-3).
    batch_size : int
        Mini-batch size (default 64).
    device : str
        PyTorch device (default 'cpu').
    checkpoint_dir : str, optional
        Directory for saving model checkpoints.
    """

    def __init__(
        self,
        train_window: int = _TRAIN_WINDOW,
        predict_horizon: int = _PREDICT_HORIZON,
        purge_gap: int = _PURGE_GAP,
        embargo: int = _EMBARGO,
        max_epochs: int = _MAX_EPOCHS,
        patience: int = _PATIENCE,
        lr: float = _LR,
        batch_size: int = _BATCH_SIZE,
        device: str = "cpu",
        checkpoint_dir: Optional[str] = None,
    ) -> None:
        self.train_window = train_window
        self.predict_horizon = predict_horizon
        self.purge_gap = purge_gap
        self.embargo = embargo
        self.max_epochs = max_epochs
        self.patience = patience
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.checkpoint_dir = Path(
            checkpoint_dir or tempfile.mkdtemp(prefix="nn_ckpt_"),
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_walk_forward(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        model_class: Type[BasePredictor],
        window: int = 60,
        **model_kwargs: Any,
    ) -> WalkForwardResult:
        """Run rolling walk-forward training.

        Parameters
        ----------
        features_df : pd.DataFrame
            Full feature matrix (T, F).
        target : pd.Series
            Direction labels aligned with features.
        model_class : Type[BasePredictor]
            Model class (LSTMPredictor or AttentionLSTMPredictor).
        window : int
            Sequence length for the sliding-window dataset (default 60).
        **model_kwargs
            Passed to model_class constructor (must include input_size).

        Returns
        -------
        WalkForwardResult
        """
        dataset = TimeSeriesDataset(
            features_df, target, window=window, normalize=True,
        )
        n_samples = len(dataset)

        if n_samples == 0:
            logger.warning("Dataset is empty — nothing to train on.")
            return WalkForwardResult(
                predictions=np.array([]),
                actuals=np.array([]),
            )

        all_preds: List[np.ndarray] = []
        all_actuals: List[np.ndarray] = []
        fold_metrics: List[FoldMetrics] = []
        best_val_loss = float("inf")
        best_model_path = ""

        fold_idx = 0
        start = 0

        while start + self.train_window + self.purge_gap + self.predict_horizon <= n_samples:
            train_end = start + self.train_window
            test_start = train_end + self.purge_gap
            test_end = min(test_start + self.predict_horizon, n_samples)

            if test_start >= n_samples:
                break

            train_indices = list(range(start, train_end))
            test_indices = list(range(test_start, test_end))

            logger.info(
                "Fold %d: train [%d–%d], test [%d–%d]",
                fold_idx, start, train_end - 1, test_start, test_end - 1,
            )

            # Build subsets
            train_subset = Subset(dataset, train_indices)
            test_subset = Subset(dataset, test_indices)

            train_loader = DataLoader(
                train_subset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                drop_last=False,
            )
            test_loader = DataLoader(
                test_subset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                drop_last=False,
            )

            # Fresh model per fold
            model = model_class(**model_kwargs).to(self.device)
            train_loss, val_loss = self._train_fold(
                model, train_loader, test_loader,
            )

            # Evaluate
            preds, actuals = self._evaluate(model, test_loader)
            all_preds.append(preds)
            all_actuals.append(actuals)

            # Compute metrics
            fm = self._compute_fold_metrics(
                fold_idx, preds, actuals, train_loss, val_loss,
            )
            fold_metrics.append(fm)
            logger.info(
                "Fold %d — acc=%.4f, val_loss=%.4f",
                fold_idx, fm.accuracy, fm.val_loss,
            )

            # Save best model
            ckpt_path = str(self.checkpoint_dir / f"model_fold{fold_idx}.pt")
            torch.save(model.state_dict(), ckpt_path)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = ckpt_path

            # Roll forward
            start += self.predict_horizon
            fold_idx += 1

        return WalkForwardResult(
            predictions=np.concatenate(all_preds) if all_preds else np.array([]),
            actuals=np.concatenate(all_actuals) if all_actuals else np.array([]),
            metrics_per_fold=fold_metrics,
            best_model_path=best_model_path,
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def _train_fold(
        self,
        model: BasePredictor,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> tuple[float, float]:
        """Train a single fold with early stopping.

        Returns
        -------
        tuple[float, float]
            Final (train_loss, val_loss).
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=self.patience // 2, factor=0.5,
        )

        best_val = float("inf")
        epochs_no_improve = 0
        last_train_loss = 0.0
        last_val_loss = 0.0

        for epoch in range(self.max_epochs):
            # --- Train ---
            model.train()
            train_losses: List[float] = []
            for x_batch, y_batch, lengths in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                lengths = lengths.to(self.device)

                optimizer.zero_grad()
                logits = model(x_batch, lengths=lengths)
                loss = criterion(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(loss.item())

            last_train_loss = float(np.mean(train_losses))

            # --- Validate ---
            model.eval()
            val_losses: List[float] = []
            with torch.no_grad():
                for x_batch, y_batch, lengths in val_loader:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    lengths = lengths.to(self.device)
                    logits = model(x_batch, lengths=lengths)
                    loss = criterion(logits, y_batch)
                    val_losses.append(loss.item())

            last_val_loss = float(np.mean(val_losses)) if val_losses else 0.0
            scheduler.step(last_val_loss)

            # Early stopping
            if last_val_loss < best_val:
                best_val = last_val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

        return last_train_loss, last_val_loss

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        model: BasePredictor,
        loader: DataLoader,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run inference on a DataLoader.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (predictions, actuals) as 1-D integer arrays.
        """
        model.eval()
        all_preds: List[np.ndarray] = []
        all_targets: List[np.ndarray] = []

        with torch.no_grad():
            for x_batch, y_batch, lengths in loader:
                x_batch = x_batch.to(self.device)
                lengths = lengths.to(self.device)
                logits = model(x_batch, lengths=lengths)
                preds = logits.argmax(dim=-1).cpu().numpy()
                all_preds.append(preds)
                all_targets.append(y_batch.numpy())

        return np.concatenate(all_preds), np.concatenate(all_targets)

    @staticmethod
    def _compute_fold_metrics(
        fold: int,
        preds: np.ndarray,
        actuals: np.ndarray,
        train_loss: float,
        val_loss: float,
    ) -> FoldMetrics:
        """Compute classification metrics for a fold."""
        classes = [0, 1, 2]
        accuracy = float((preds == actuals).mean()) if len(preds) > 0 else 0.0

        precision: Dict[int, float] = {}
        recall: Dict[int, float] = {}
        f1: Dict[int, float] = {}

        for c in classes:
            tp = int(((preds == c) & (actuals == c)).sum())
            fp = int(((preds == c) & (actuals != c)).sum())
            fn = int(((preds != c) & (actuals == c)).sum())

            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

            precision[c] = p
            recall[c] = r
            f1[c] = f

        # Confusion matrix (3×3)
        cm = np.zeros((3, 3), dtype=np.int64)
        for pred_c, act_c in zip(preds, actuals):
            if 0 <= pred_c < 3 and 0 <= act_c < 3:
                cm[act_c, pred_c] += 1

        return FoldMetrics(
            fold=fold,
            accuracy=accuracy,
            precision_per_class=precision,
            recall_per_class=recall,
            f1_per_class=f1,
            confusion_matrix=cm,
            train_loss=train_loss,
            val_loss=val_loss,
        )

    # ------------------------------------------------------------------
    # Model persistence helpers
    # ------------------------------------------------------------------

    @staticmethod
    def save_model(model: BasePredictor, path: str) -> None:
        """Save model state dict to disk."""
        torch.save(model.state_dict(), path)
        logger.info("Model saved to %s", path)

    @staticmethod
    def load_model(
        model_class: Type[BasePredictor],
        path: str,
        **model_kwargs: Any,
    ) -> BasePredictor:
        """Load a saved model from disk.

        Parameters
        ----------
        model_class : Type[BasePredictor]
            The model class to instantiate.
        path : str
            Path to the state dict file.
        **model_kwargs
            Constructor arguments for the model.

        Returns
        -------
        BasePredictor
            The loaded model in eval mode.
        """
        model = model_class(**model_kwargs)
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
        logger.info("Model loaded from %s", path)
        return model
