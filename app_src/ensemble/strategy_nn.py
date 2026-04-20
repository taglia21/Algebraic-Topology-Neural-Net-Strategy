"""
ensemble/strategy_nn.py
=======================
Neural Net Directional Strategy.

Wraps a trained LSTM / Attention-LSTM model and converts its predictions
into actionable trading signals with confidence-based gating.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


class NNDirectionalStrategy:
    """Generate trading signals from neural network predictions.

    Parameters
    ----------
    high_threshold : float
        Confidence above this → STRONG signal (default 0.65).
    low_threshold : float
        Confidence below this → NEUTRAL / no trade (default 0.50).
    """

    # Maps model output class indices to direction labels
    _DIR_MAP = {0: "SHORT", 1: "NEUTRAL", 2: "LONG"}

    def __init__(
        self,
        high_threshold: float = 0.65,
        low_threshold: float = 0.50,
    ) -> None:
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self._last_signals: Optional[pd.DataFrame] = None

    @staticmethod
    def load_model(path: str | Path) -> Any:
        """Load a pre-trained model from disk.

        Parameters
        ----------
        path : str or Path
            Path to a ``.pt`` or ``.pth`` saved model file.

        Returns
        -------
        torch.nn.Module
            The loaded model in eval mode.
        """
        model = torch.load(str(path), map_location="cpu", weights_only=False)
        model.eval()
        return model

    def generate_signals(
        self,
        features: pd.DataFrame,
        model: Any,
        regime: str = "NORMAL",
    ) -> pd.DataFrame:
        """Generate per-stock trading signals using the NN model.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix from ``NNFeatureEngine.build_features()``.
            Index is DatetimeIndex.  Each row is used for one prediction.
        model : torch.nn.Module
            Trained predictor with a ``predict(x)`` method returning a dict
            with keys ``'direction'``, ``'probabilities'``, ``'confidence'``.
        regime : str
            Current market regime (``'NORMAL'``, ``'STRESSED'``, ``'CRASH'``).
            In CRASH regime, all LONG signals are overridden to NEUTRAL.

        Returns
        -------
        pd.DataFrame
            Columns: ticker, direction, strength, prediction_probs, timestamp.
        """
        records = []
        feature_values = features.values.astype(np.float32)

        # The model expects (batch, seq_len, features) — we treat each row
        # as a single-timestep sequence for per-date prediction.
        for i, date in enumerate(features.index):
            x = torch.tensor(
                feature_values[i : i + 1].reshape(1, 1, -1),
                dtype=torch.float32,
            )

            try:
                prediction = model.predict(x)
            except Exception as exc:
                logger.warning("NN prediction failed at %s: %s", date, exc)
                continue

            raw_direction = self._DIR_MAP.get(prediction["direction"], "NEUTRAL")
            confidence = float(prediction["confidence"])
            probs = prediction["probabilities"]

            # Convert probs tensor to list for storage
            if isinstance(probs, torch.Tensor):
                probs_list = probs[0].tolist()
            else:
                probs_list = list(probs)

            # Confidence gating
            if confidence >= self.high_threshold:
                strength = min(1.0, confidence)
                direction = raw_direction
            elif confidence >= self.low_threshold:
                # Weak signal: scale strength linearly between thresholds
                frac = (confidence - self.low_threshold) / (
                    self.high_threshold - self.low_threshold
                )
                strength = round(frac * 0.5, 6)  # cap weak signals at 0.5
                direction = raw_direction
            else:
                strength = 0.0
                direction = "NEUTRAL"

            # Regime override: in CRASH, suppress all LONG signals
            if regime == "CRASH" and direction == "LONG":
                direction = "NEUTRAL"
                strength = 0.0

            # Ticker placeholder — caller should set actual ticker
            ticker = features.columns[0] if len(features.columns) > 0 else "UNKNOWN"

            records.append({
                "ticker": ticker,
                "direction": direction,
                "strength": round(strength, 6),
                "prediction_probs": probs_list,
                "timestamp": date,
            })

        result = pd.DataFrame(records)
        if result.empty:
            result = pd.DataFrame(
                columns=["ticker", "direction", "strength", "prediction_probs", "timestamp"]
            )
        self._last_signals = result
        return result
