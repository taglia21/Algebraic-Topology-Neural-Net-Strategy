"""
Execution Optimizer — VWAP Benchmark (Phase I, Item 11)
========================================================

Track execution price vs rolling VWAP; log slippage_bps per trade.
Alert Discord if slippage_bps > 5 on any single fill.
"""

import logging
import os
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["VWAPBenchmark", "VWAPRecord"]

SLIPPAGE_ALERT_BPS = 5.0


@dataclass
class VWAPRecord:
    """Single trade VWAP benchmark record."""
    symbol: str
    fill_price: float
    vwap_price: float
    slippage_bps: float
    quantity: int
    side: str
    alerted: bool = False


class VWAPBenchmark:
    """Track execution quality against VWAP benchmark.

    Parameters
    ----------
    alert_threshold_bps : float
        Trigger a Discord alert if slippage exceeds this (default 5 bps).
    discord_webhook : str
        Discord webhook URL for alerts.
    lookback : int
        Window size for rolling VWAP calculation (default 50 bars).
    """

    def __init__(
        self,
        alert_threshold_bps: float = SLIPPAGE_ALERT_BPS,
        discord_webhook: str = "",
        lookback: int = 50,
    ):
        self.alert_threshold_bps = alert_threshold_bps
        self.discord_webhook = discord_webhook or os.getenv("DISCORD_WEBHOOK_MARCUS", "")
        self.lookback = lookback
        self._records: List[VWAPRecord] = []
        # Per-symbol price × volume accumulators for rolling VWAP
        self._price_vol: Dict[str, deque] = {}

    def update_bar(self, symbol: str, price: float, volume: float) -> None:
        """Feed a new price bar for VWAP calculation."""
        if symbol not in self._price_vol:
            self._price_vol[symbol] = deque(maxlen=self.lookback)
        self._price_vol[symbol].append((price, volume))

    def get_vwap(self, symbol: str) -> float:
        """Compute current rolling VWAP for a symbol."""
        data = self._price_vol.get(symbol, [])
        if not data:
            return 0.0
        total_pv = sum(p * v for p, v in data)
        total_v = sum(v for _, v in data)
        return total_pv / total_v if total_v > 0 else 0.0

    def record_fill(
        self,
        symbol: str,
        fill_price: float,
        quantity: int,
        side: str = "buy",
        vwap_override: Optional[float] = None,
    ) -> VWAPRecord:
        """Record an execution fill and compute slippage vs VWAP.

        Parameters
        ----------
        symbol : str
        fill_price : float
        quantity : int
        side : str
        vwap_override : float or None
            Use this VWAP instead of computed rolling VWAP.

        Returns
        -------
        VWAPRecord
        """
        vwap = vwap_override if vwap_override is not None else self.get_vwap(symbol)
        if vwap <= 0:
            vwap = fill_price

        slippage = (fill_price - vwap) / vwap * 10_000
        if side == "sell":
            slippage = -slippage  # selling above VWAP is good

        alerted = False
        if abs(slippage) > self.alert_threshold_bps:
            alerted = True
            logger.warning(
                "SLIPPAGE ALERT %s: %.2f bps (fill=%.4f vwap=%.4f)",
                symbol, slippage, fill_price, vwap,
            )
            self._send_alert(symbol, slippage, fill_price, vwap)

        record = VWAPRecord(
            symbol=symbol, fill_price=fill_price, vwap_price=vwap,
            slippage_bps=slippage, quantity=quantity, side=side,
            alerted=alerted,
        )
        self._records.append(record)

        logger.info(
            "FILL %s %s %d @ %.4f vs VWAP %.4f → slip=%.2f bps",
            side, symbol, quantity, fill_price, vwap, slippage,
        )
        return record

    def avg_slippage_bps(self) -> float:
        """Average absolute slippage across all recorded fills."""
        if not self._records:
            return 0.0
        return float(np.mean([abs(r.slippage_bps) for r in self._records]))

    @property
    def records(self) -> List[VWAPRecord]:
        return list(self._records)

    def _send_alert(self, symbol, slippage, fill, vwap):
        """Send Discord alert for high slippage."""
        if not self.discord_webhook:
            return
        try:
            import requests
            requests.post(self.discord_webhook, json={
                "embeds": [{
                    "title": "⚠️ High Slippage Alert",
                    "description": (
                        f"**{symbol}**: {slippage:+.2f} bps\n"
                        f"Fill: ${fill:.4f} vs VWAP: ${vwap:.4f}"
                    ),
                    "color": 0xFF8C00,
                }]
            }, timeout=5)
        except Exception:
            pass
