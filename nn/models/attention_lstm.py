"""
nn/models/attention_lstm.py
===========================
Attention-LSTM hybrid model inspired by the Momentum Transformer
(arXiv:2112.08534).

Adds multi-head self-attention on top of LSTM hidden states so the model
can attend to arbitrary past timesteps, learning both momentum and
mean-reversion patterns.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from nn.models.lstm_predictor import BasePredictor

logger = logging.getLogger(__name__)


class AttentionLSTMPredictor(BasePredictor):
    """LSTM encoder + multi-head self-attention for directional prediction.

    Parameters
    ----------
    input_size : int
        Number of input features per timestep.
    hidden_size : int
        LSTM hidden dimension (default 128).
    num_layers : int
        Number of stacked LSTM layers (default 2).
    dropout : float
        Dropout probability (default 0.3).
    num_classes : int
        Output classes (default 3: down/flat/up).
    num_heads : int
        Number of attention heads (default 4).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 3,
        num_heads: int = 4,
    ) -> None:
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=num_classes,
        )
        self.num_heads = num_heads

        self.bn = nn.BatchNorm1d(input_size)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

        # Store last attention weights for interpretability
        self._last_attn_weights: Optional[torch.Tensor] = None

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

        # Batch-norm across features
        x_flat = x.reshape(-1, feats)
        x_flat = self.bn(x_flat)
        x = x_flat.reshape(batch, seq_len, feats)

        if lengths is not None:
            lengths_cpu = lengths.cpu().to(torch.int64)
            packed = pack_padded_sequence(
                x, lengths_cpu, batch_first=True, enforce_sorted=False,
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(x)

        # Multi-head self-attention over LSTM hidden states
        attn_out, attn_weights = self.attention(
            lstm_out, lstm_out, lstm_out,
            need_weights=True,
            average_attn_weights=True,
        )
        self._last_attn_weights = attn_weights.detach()

        # Residual connection + layer norm
        attn_out = self.layer_norm(lstm_out + attn_out)

        # Use the last timestep representation
        if lengths is not None:
            # Gather last valid timestep for each sample
            idx = (lengths_cpu - 1).clamp(min=0).long()
            idx = idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, attn_out.size(2))
            out = attn_out.gather(1, idx).squeeze(1)
        else:
            out = attn_out[:, -1, :]

        out = self.drop(out)
        logits = self.fc(out)
        return logits

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Return attention weights from the most recent forward pass.

        Returns
        -------
        torch.Tensor or None
            (batch, seq_len, seq_len) attention weight matrix,
            or None if no forward pass has been run.
        """
        return self._last_attn_weights

    def get_feature_importance(self) -> np.ndarray:
        """Return BN weight magnitudes as a feature importance proxy."""
        return torch.abs(self.bn.weight).detach().cpu().numpy()
