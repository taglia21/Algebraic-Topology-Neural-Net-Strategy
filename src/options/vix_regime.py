"""
VIX Regime Overlay
===================

Fetches the current VIX level and maps it to one of four volatility
regimes, each with a position-size multiplier.  The CRISIS regime
(VIX > 30) halts all new entries.

VIX is cached for ``VIX_CACHE_SECONDS`` (default 300 s / 5 min) to
avoid excess yfinance API calls.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

# Import cache TTL from config (set in Phase 3, Change #4)
try:
    from .config import MONITORING_CONFIG
    VIX_CACHE_SECONDS = MONITORING_CONFIG.get("vix_cache_seconds", 300)
except Exception:
    VIX_CACHE_SECONDS = 300


# ============================================================================
# REGIME DEFINITIONS
# ============================================================================

class VIXRegime(Enum):
    LOW_VOL  = "low_vol"     # VIX < 15
    NORMAL   = "normal"      # 15 ≤ VIX < 20
    ELEVATED = "elevated"    # 20 ≤ VIX < 30
    CRISIS   = "crisis"      # VIX ≥ 30


# Multiplier applied to position size.  0.0 = halt all new entries.
REGIME_MULTIPLIERS = {
    VIXRegime.LOW_VOL:  1.2,
    VIXRegime.NORMAL:   1.0,
    VIXRegime.ELEVATED: 0.6,
    VIXRegime.CRISIS:   0.0,
}


@dataclass
class VIXSnapshot:
    """Point-in-time VIX reading."""
    level: float
    regime: VIXRegime
    multiplier: float
    timestamp: float  # time.time()


class VIXRegimeOverlay:
    """Fetch VIX, classify regime, provide position-size multiplier."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._cached: Optional[VIXSnapshot] = None

    # ------------------------------------------------------------------ #
    # PUBLIC API
    # ------------------------------------------------------------------ #

    def get_snapshot(self, force_refresh: bool = False) -> VIXSnapshot:
        """
        Return current VIX snapshot, using a time-based cache.

        Parameters
        ----------
        force_refresh : bool
            Bypass cache and fetch fresh data.

        Returns
        -------
        VIXSnapshot
        """
        now = time.time()
        if (
            not force_refresh
            and self._cached is not None
            and (now - self._cached.timestamp) < VIX_CACHE_SECONDS
        ):
            return self._cached

        vix_level = self._fetch_vix()
        regime = self._classify(vix_level)
        snap = VIXSnapshot(
            level=vix_level,
            regime=regime,
            multiplier=REGIME_MULTIPLIERS[regime],
            timestamp=now,
        )
        self._cached = snap
        self.logger.info(
            f"VIX={vix_level:.1f}  regime={regime.value}  "
            f"size_mult={snap.multiplier:.1f}x"
        )
        return snap

    # ------------------------------------------------------------------ #
    # INTERNAL
    # ------------------------------------------------------------------ #

    def _fetch_vix(self) -> float:
        """Fetch current VIX level from yfinance.  Returns 20.0 on failure."""
        try:
            import yfinance as yf
            data = yf.download("^VIX", period="5d", interval="1d", progress=False)
            if data is not None and len(data) > 0:
                import pandas as pd
                if isinstance(data.columns, pd.MultiIndex):
                    close = float(data["Close"].iloc[-1, 0])
                else:
                    close = float(data["Close"].iloc[-1])
                if close > 0:
                    return close
        except Exception as exc:
            self.logger.warning(f"VIX fetch failed: {exc}")

        # Conservative fallback — treat as NORMAL
        return 20.0

    @staticmethod
    def _classify(vix: float) -> VIXRegime:
        if vix < 15:
            return VIXRegime.LOW_VOL
        if vix < 20:
            return VIXRegime.NORMAL
        if vix < 30:
            return VIXRegime.ELEVATED
        return VIXRegime.CRISIS
