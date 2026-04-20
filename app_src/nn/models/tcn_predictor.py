"""
nn/models/tcn_predictor.py
==========================
Temporal Convolutional Network for market regime prediction.

Why TCN instead of LSTM:
  - Parallelizable (no sequential dependency) → 5-10x faster training
  - Receptive field is explicit and tunable (no hidden state decay)
  - Less prone to vanishing gradients
  - Better at multi-scale patterns via dilated convolutions
  - Empirically: matches or outperforms LSTM on financial time series
    (Bai et al., 2018 "An Empirical Evaluation of Generic Convolutional
     and Recurrent Networks for Sequence Modeling")

Architecture: Temporal Convolutional Block stack with residual connections.
Each block uses causal (no future leakage) dilated convolutions.

Prediction target: REGIME (not direction)
  - 0: TRENDING_UP (momentum regime)
  - 1: TRENDING_DOWN (downtrend regime)
  - 2: MEAN_REVERTING (oscillating regime)
  - 3: VOLATILE / CHOPPY (high noise regime)

Using regime as the target is:
  - More stable (regimes last days-weeks, not bars)
  - More tractable (4-class vs binary with low base rate)
  - Directly actionable (each regime favors different strategy weights)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """Causal (left-padded) 1D convolution — no future leakage."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int = 1):
        super().__init__()
        self.pad = (kernel - 1) * dilation  # left padding only
        self.conv = nn.Conv1d(in_ch, out_ch, kernel,
                              dilation=dilation, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    """
    One TCN residual block:
      CausalConv → BN → ReLU → Dropout → CausalConv → BN → ReLU → Dropout
      + residual (1×1 conv if channels differ)
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel, dilation)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel, dilation)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.drop  = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.drop(out)
        return F.relu(out + self.residual(x))


class TCNPredictor(nn.Module):
    """
    Temporal Convolutional Network for 4-class regime prediction.

    Parameters
    ----------
    input_size : int
        Number of input features per timestep.
    num_channels : list[int]
        Number of channels in each TCN block (defines depth).
        Default [64, 64, 32] → 3-layer network.
    kernel_size : int
        Convolution kernel size (default 3).
    dropout : float
        Dropout rate (default 0.2).
    num_classes : int
        Number of output classes (default 4 for regimes).
    """

    def __init__(
        self,
        input_size: int,
        num_channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        num_classes: int = 4,
    ):
        super().__init__()

        if num_channels is None:
            num_channels = [64, 64, 32]

        layers = []
        in_ch = input_size
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i  # exponentially growing receptive field
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch

        self.tcn    = nn.Sequential(*layers)
        self.head   = nn.Linear(num_channels[-1], num_classes)
        self.dropout = nn.Dropout(dropout)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, seq_len, input_size)

        Returns
        -------
        torch.Tensor
            Shape (batch, num_classes) — raw logits
        """
        # TCN expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        x = self.tcn(x)
        # Take the last timestep's representation
        x = x[:, :, -1]
        x = self.dropout(x)
        return self.head(x)

    def predict_regime(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (predicted_class, confidence) for each item in batch.
        Confidence = max softmax probability.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs  = torch.softmax(logits, dim=-1)
            classes = probs.argmax(dim=-1)
            confs   = probs.max(dim=-1).values
        return classes, confs
