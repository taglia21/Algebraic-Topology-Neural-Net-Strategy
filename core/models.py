"""
core/models.py
==============
Local dataclasses previously imported from equities.models.

These are the minimal models needed by core/ modules (kill_switch, reconciliation)
after the equities/ package was removed in v2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass
class Position:
    """Current holding in a single symbol.

    Attributes
    ----------
    symbol : str
        Ticker symbol.
    qty : int
        Net shares held. Positive = long, negative = short.
    avg_entry : float
        Volume-weighted average entry price.
    current_price : float
        Latest mark-to-market price.
    unrealized_pnl : float
        Mark-to-market P&L on the open position.
    sector : str
        GICS sector for sector-exposure tracking.
    strategy : str
        Strategy that originated this position.
    """

    symbol: str
    qty: int
    avg_entry: float
    current_price: float
    unrealized_pnl: float
    sector: str = "Unknown"
    strategy: str = ""

    @property
    def market_value(self) -> float:
        """Current market value of the position (positive = long)."""
        return self.qty * self.current_price

    @property
    def cost_basis(self) -> float:
        """Total cost basis of the position."""
        return self.qty * self.avg_entry
