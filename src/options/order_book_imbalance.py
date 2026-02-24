"""
Order Book Imbalance — Microstructure Alpha (Phase F, Item 1)
=============================================================

Compute real-time bid-ask volume imbalance ratio from Level 2 data.
Generates long signal when OFI > 0.65, short when OFI < 0.35.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["OrderFlowImbalance", "OFISignal"]

# Thresholds
LONG_OFI_THRESHOLD = 0.65
SHORT_OFI_THRESHOLD = 0.35


@dataclass
class OFISignal:
    """Order Flow Imbalance signal result."""
    symbol: str
    ofi_ratio: float
    direction: str  # "long", "short", or "neutral"
    confidence: float
    bid_volume: float
    ask_volume: float


class OrderFlowImbalance:
    """Compute real-time bid-ask volume imbalance from Level 2 data.

    OFI = bid_volume / (bid_volume + ask_volume)
      - OFI > 0.65  →  long signal  (buyers dominating)
      - OFI < 0.35  →  short signal (sellers dominating)
      - otherwise    →  neutral

    Parameters
    ----------
    long_threshold : float
        OFI above this triggers a long signal (default 0.65).
    short_threshold : float
        OFI below this triggers a short signal (default 0.35).
    depth_levels : int
        Number of book levels to aggregate (default 5).
    """

    def __init__(
        self,
        long_threshold: float = LONG_OFI_THRESHOLD,
        short_threshold: float = SHORT_OFI_THRESHOLD,
        depth_levels: int = 5,
    ):
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.depth_levels = depth_levels

    def order_flow_imbalance(
        self,
        bids: List[Dict],
        asks: List[Dict],
        symbol: str = "UNK",
    ) -> OFISignal:
        """Compute order flow imbalance from Level 2 book snapshot.

        Parameters
        ----------
        bids : list of dict
            Each dict has ``price`` and ``size`` keys.
        asks : list of dict
            Each dict has ``price`` and ``size`` keys.
        symbol : str
            Underlying symbol for logging.

        Returns
        -------
        OFISignal
        """
        if not bids or not asks:
            return OFISignal(
                symbol=symbol, ofi_ratio=0.5, direction="neutral",
                confidence=0.0, bid_volume=0.0, ask_volume=0.0,
            )

        # Aggregate top N levels
        bid_vol = sum(
            float(b.get("size", b.get("quantity", 0)))
            for b in bids[: self.depth_levels]
        )
        ask_vol = sum(
            float(a.get("size", a.get("quantity", 0)))
            for a in asks[: self.depth_levels]
        )

        total = bid_vol + ask_vol
        if total <= 0:
            return OFISignal(
                symbol=symbol, ofi_ratio=0.5, direction="neutral",
                confidence=0.0, bid_volume=bid_vol, ask_volume=ask_vol,
            )

        ofi = bid_vol / total

        if ofi > self.long_threshold:
            direction = "long"
            confidence = min(1.0, (ofi - self.long_threshold) / (1.0 - self.long_threshold))
        elif ofi < self.short_threshold:
            direction = "short"
            confidence = min(1.0, (self.short_threshold - ofi) / self.short_threshold)
        else:
            direction = "neutral"
            confidence = 0.0

        logger.info(
            "OFI %s: %.3f (bid_vol=%.0f ask_vol=%.0f) → %s conf=%.2f",
            symbol, ofi, bid_vol, ask_vol, direction, confidence,
        )

        return OFISignal(
            symbol=symbol,
            ofi_ratio=ofi,
            direction=direction,
            confidence=confidence,
            bid_volume=bid_vol,
            ask_volume=ask_vol,
        )
