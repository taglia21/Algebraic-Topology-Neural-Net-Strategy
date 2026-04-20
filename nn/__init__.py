"""
nn/
===
Neural network module for directional prediction.

Houses LSTM and Attention-LSTM models, feature engineering,
walk-forward training, and topology-aware feature integration
from the TDA module.
"""

from __future__ import annotations

from nn.data_loader import PurgedKFold, TimeSeriesDataset, collate_fn, direction_labels
from nn.features import NNFeatureEngine
from nn.models.attention_lstm import AttentionLSTMPredictor
from nn.models.lstm_predictor import BasePredictor, LSTMPredictor
from nn.training import WalkForwardResult, WalkForwardTrainer

__all__ = [
    "BasePredictor",
    "LSTMPredictor",
    "AttentionLSTMPredictor",
    "NNFeatureEngine",
    "TimeSeriesDataset",
    "PurgedKFold",
    "collate_fn",
    "direction_labels",
    "WalkForwardTrainer",
    "WalkForwardResult",
]
