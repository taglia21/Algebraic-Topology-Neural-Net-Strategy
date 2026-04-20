"""
nn/models/lstm_predictor.py
===========================
LSTM-based directional predictor for financial time series.

Provides a 2-layer stacked LSTM with batch normalisation, dropout,
and a 3-class softmax head (long / short / flat).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

logger = logging.getLogger(__name__)


class BasePredictor(nn.Module):
    """Abstract base class shared by all directional predictors.

    Subclasses must implement ``forward`` and may override ``predict``.
    """

    # Direction labels
    DIR_DOWN: int = 0
    DIR_FLAT: int = 1
    DIR_UP: int = 2

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_classes = num_classes

    def predict(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Run inference and return a structured prediction dict.

        Parameters
        ----------
        x : torch.Tensor
            (batch, seq_len, input_size) feature tensor.
        lengths : torch.Tensor, optional
            Actual sequence lengths for each sample.

        Returns
        -------
        dict
            Keys: 'direction' (int), 'probabilities' (Tensor), 'confidence' (float).
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, lengths=lengths)
            probs = torch.softmax(logits, dim=-1)

            # Take first sample in batch for scalar outputs
            probs_first = probs[0]
            confidence = float(probs_first.max().item())
            direction = int(probs_first.argmax().item())

        return {
            "direction": direction,
            "probabilities": probs,
            "confidence": confidence,
        }

    def get_feature_importance(self) -> np.ndarray:
        """Gradient-based feature importance estimation.

        Returns the L2-norm of the gradient of the output w.r.t. each input
        feature, averaged over input dimensions.  Requires a forward pass
        first (call ``forward`` with ``x.requires_grad_(True)``).

        Returns
        -------
        np.ndarray
            (input_size,) importance scores.
        """
        # Use the input batch-norm weight as a proxy when no gradient is available
        if hasattr(self, "bn") and self.bn.weight is not None:
            return self.bn.weight.detach().cpu().numpy()
        return np.ones(self.input_size)


class LSTMPredictor(BasePredictor):
    """Two-layer stacked LSTM for directional prediction.

    Parameters
    ----------
    input_size : int
        Number of input features per timestep.
    hidden_size : int
        LSTM hidden dimension (default 128).
    num_layers : int
        Number of stacked LSTM layers (default 2).
    dropout : float
        Dropout probability between LSTM layers (default 0.3).
    num_classes : int
        Number of output classes (default 3: down/flat/up).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 3,
    ) -> None:
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=num_classes,
        )

        self.bn = nn.BatchNorm1d(input_size)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            (batch, seq_len, input_size).
        lengths : torch.Tensor, optional
            Actual lengths per sample for packing.

        Returns
        -------
        torch.Tensor
            (batch, num_classes) logits.
        """
        batch, seq_len, feats = x.shape

        # Batch-norm across features (reshape to (batch*seq, feats))
        x_flat = x.reshape(-1, feats)
        x_flat = self.bn(x_flat)
        x = x_flat.reshape(batch, seq_len, feats)

        if lengths is not None:
            lengths_cpu = lengths.cpu().to(torch.int64)
            packed = pack_padded_sequence(
                x, lengths_cpu, batch_first=True, enforce_sorted=False,
            )
            lstm_out, (h_n, _) = self.lstm(packed)
        else:
            lstm_out, (h_n, _) = self.lstm(x)

        # Use final hidden state of top layer
        out = h_n[-1]  # (batch, hidden_size)
        out = self.drop(out)
        logits = self.fc(out)
        return logits

    def get_feature_importance(self) -> np.ndarray:
        """Return BN weight magnitudes as a feature importance proxy."""
        return torch.abs(self.bn.weight).detach().cpu().numpy()
