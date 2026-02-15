"""
Dynamic Symbol Universe Manager
================================
Maintains a filtered, scored universe of the top ~100 liquid US equities
plus sector ETFs.  Filters: avg volume > 1M, ATR% > 1%, price > $10.

Two operating modes:
  1. **Standalone** (no API key) — uses built-in pool of ~120 blue-chips +
     sector ETFs, filters them from bar data provided by the caller.
  2. **Polygon-backed** — if POLYGON_API_KEY_OTREP is set, can optionally
     fetch the full US ticker list and filter down.

Usage in unified_trader.py:
    from src.universe_manager import UniverseManager
    mgr = UniverseManager()
    symbols = mgr.get_active_universe()   # filtered list (up to 100)
    mgr.update_filters(bars_map)          # recalculate from latest bars
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta

import numpy as np

logger = logging.getLogger(__name__)


# ── Base universe pool (candidates before filtering) ─────────────────────
# Top ~120 liquid US equities + sector/thematic ETFs
BASE_POOL: List[str] = [
    # ── Mega-cap tech ────────────────────────────────
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "AVGO",
    "ORCL", "CRM", "ADBE", "AMD", "INTC", "CSCO", "QCOM", "NFLX",
    "NOW", "UBER", "SHOP", "SQ", "SNOW", "PLTR", "COIN",
    # ── Financials ───────────────────────────────────
    "JPM", "GS", "BAC", "MS", "WFC", "C", "V", "MA", "AXP",
    "SCHW", "BLK", "CME", "ICE", "KRE",
    # ── Healthcare ───────────────────────────────────
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT",
    "ISRG", "DXCM", "MRNA",
    # ── Energy ───────────────────────────────────────
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "OXY",
    # ── Consumer Discretionary ───────────────────────
    "HD", "MCD", "NKE", "SBUX", "TJX", "LULU", "BKNG",
    # ── Consumer Staples ─────────────────────────────
    "KO", "PG", "COST", "WMT", "PEP", "PM", "MO",
    # ── Industrials ──────────────────────────────────
    "CAT", "HON", "GE", "DE", "UPS", "LMT", "RTX", "BA",
    # ── Materials ────────────────────────────────────
    "LIN", "FCX", "NEM", "APD",
    # ── REITs ────────────────────────────────────────
    "AMT", "O", "PLD", "EQIX",
    # ── Utilities ────────────────────────────────────
    "NEE", "DUK", "SO",
    # ── Communications ───────────────────────────────
    "DIS", "CMCSA", "T", "VZ",
    # ── Sector & broad ETFs ──────────────────────────
    "SPY", "QQQ", "IWM", "DIA",
    "XLF", "XLK", "XLE", "XLV", "XLI", "XLP", "XLU", "XLY", "XLB",
    "ARKK", "SMH", "XBI", "GDX", "TLT", "HYG",
]

# Sector classification
SECTOR_CLASSIFICATION: Dict[str, str] = {
    "AAPL": "technology", "MSFT": "technology", "NVDA": "technology",
    "GOOGL": "technology", "META": "technology", "AMZN": "consumer",
    "TSLA": "consumer", "AVGO": "technology", "ORCL": "technology",
    "CRM": "technology", "ADBE": "technology", "AMD": "technology",
    "INTC": "technology", "CSCO": "technology", "QCOM": "technology",
    "NFLX": "consumer", "NOW": "technology", "UBER": "technology",
    "SHOP": "technology", "SQ": "financials", "SNOW": "technology",
    "PLTR": "technology", "COIN": "financials",
    "JPM": "financials", "GS": "financials", "BAC": "financials",
    "MS": "financials", "WFC": "financials", "C": "financials",
    "V": "financials", "MA": "financials", "AXP": "financials",
    "SCHW": "financials", "BLK": "financials", "CME": "financials",
    "ICE": "financials", "KRE": "financials",
    "UNH": "healthcare", "JNJ": "healthcare", "LLY": "healthcare",
    "ABBV": "healthcare", "MRK": "healthcare", "PFE": "healthcare",
    "TMO": "healthcare", "ABT": "healthcare", "ISRG": "healthcare",
    "DXCM": "healthcare", "MRNA": "healthcare",
    "XOM": "energy", "CVX": "energy", "COP": "energy",
    "SLB": "energy", "EOG": "energy", "MPC": "energy", "OXY": "energy",
    "HD": "consumer", "MCD": "consumer", "NKE": "consumer",
    "SBUX": "consumer", "TJX": "consumer", "LULU": "consumer",
    "BKNG": "consumer",
    "KO": "staples", "PG": "staples", "COST": "staples",
    "WMT": "staples", "PEP": "staples", "PM": "staples", "MO": "staples",
    "CAT": "industrials", "HON": "industrials", "GE": "industrials",
    "DE": "industrials", "UPS": "industrials", "LMT": "industrials",
    "RTX": "industrials", "BA": "industrials",
    "LIN": "materials", "FCX": "materials", "NEM": "materials", "APD": "materials",
    "AMT": "reits", "O": "reits", "PLD": "reits", "EQIX": "reits",
    "NEE": "utilities", "DUK": "utilities", "SO": "utilities",
    "DIS": "communications", "CMCSA": "communications",
    "T": "communications", "VZ": "communications",
    "SPY": "etf", "QQQ": "etf", "IWM": "etf", "DIA": "etf",
    "XLF": "etf", "XLK": "etf", "XLE": "etf", "XLV": "etf",
    "XLI": "etf", "XLP": "etf", "XLU": "etf", "XLY": "etf", "XLB": "etf",
    "ARKK": "etf", "SMH": "etf", "XBI": "etf", "GDX": "etf",
    "TLT": "etf", "HYG": "etf",
}


# ── Data classes ─────────────────────────────────────────────────────────

@dataclass
class SymbolMetrics:
    """Scored metrics for a single symbol."""
    symbol: str
    avg_volume: float = 0.0
    avg_dollar_volume: float = 0.0
    atr_pct: float = 0.0
    price: float = 0.0
    sector: str = "unknown"
    liquidity_score: float = 0.0   # 0-1 composite
    passes_filter: bool = False


@dataclass
class UniverseConfig:
    """Tunable filter thresholds."""
    min_avg_volume: float = 1_000_000     # 1M shares/day minimum
    min_atr_pct: float = 0.01             # 1% ATR minimum (need movement)
    max_atr_pct: float = 0.10             # 10% ATR max (avoid illiquid junk)
    min_price: float = 10.0               # No penny stocks
    max_symbols: int = 100                # Hard cap on universe size
    always_include: List[str] = field(
        default_factory=lambda: ["SPY", "QQQ", "IWM"]  # Always in universe
    )
    refresh_interval_hours: float = 4.0   # Refilter every 4 hours


# =========================================================================
# Main class
# =========================================================================

class UniverseManager:
    """
    Dynamic symbol universe management.

    Filters the BASE_POOL based on volume, ATR, and price criteria.
    Returns at most ``max_symbols`` (100) sorted by liquidity score.
    """

    def __init__(self, config: UniverseConfig = None):
        self.config = config or UniverseConfig()
        self._metrics: Dict[str, SymbolMetrics] = {}
        self._active_universe: List[str] = list(BASE_POOL[:self.config.max_symbols])
        self._last_refresh: Optional[datetime] = None
        self._sector_map: Dict[str, str] = dict(SECTOR_CLASSIFICATION)
        logger.info(
            f"UniverseManager initialized: pool={len(BASE_POOL)}, "
            f"vol>{self.config.min_avg_volume / 1e6:.0f}M, "
            f"ATR>{self.config.min_atr_pct:.0%}"
        )

    # ── Public API ───────────────────────────────────────────────────

    def get_active_universe(self) -> List[str]:
        """Return the currently active (filtered) symbol list."""
        return list(self._active_universe)

    def get_sector(self, symbol: str) -> str:
        """Get sector for a symbol."""
        return self._sector_map.get(symbol, "unknown")

    def get_sector_map(self) -> Dict[str, str]:
        """Return full sector classification map."""
        return dict(self._sector_map)

    def get_sector_etfs(self) -> Dict[str, str]:
        """Get sector ETF mapping for sector analysis."""
        return {
            "XLK": "Technology", "XLF": "Financials", "XLV": "Healthcare",
            "XLE": "Energy", "XLI": "Industrials", "XLP": "Consumer Staples",
            "XLY": "Consumer Discretionary", "XLU": "Utilities",
            "XLB": "Materials", "SMH": "Semiconductors",
        }

    def get_major_indices(self) -> List[str]:
        """Get major index ETFs for market analysis."""
        return ["SPY", "QQQ", "IWM", "DIA"]

    def needs_refresh(self) -> bool:
        """Check if universe filters need recalculating."""
        if self._last_refresh is None:
            return True
        elapsed = (datetime.now() - self._last_refresh).total_seconds() / 3600
        return elapsed >= self.config.refresh_interval_hours

    def update_filters(self, bars_map: Dict[str, List[dict]]) -> List[str]:
        """
        Recalculate universe from fresh bar data.

        Parameters
        ----------
        bars_map : dict
            ``{ symbol: [{"o","h","l","c","v"}, ...] }`` — at least 20 bars each.

        Returns
        -------
        list[str] — the new filtered universe (up to ``max_symbols``).
        """
        scored: List[SymbolMetrics] = []

        for sym, bars in bars_map.items():
            if len(bars) < 20:
                continue

            closes = np.array([float(b["c"]) for b in bars])
            highs = np.array([float(b["h"]) for b in bars])
            lows = np.array([float(b["l"]) for b in bars])
            volumes = np.array([float(b["v"]) for b in bars])

            price = closes[-1]
            avg_vol = float(np.mean(volumes[-20:]))
            avg_dollar_vol = avg_vol * price

            # ATR% (14-period)
            atr = self._compute_atr(highs, lows, closes, period=14)
            atr_pct = atr / price if price > 0 else 0.0

            sector = self._sector_map.get(sym, "unknown")

            # Liquidity score: weighted combo of volume and movement
            vol_score = min(avg_vol / 10_000_000, 1.0)   # cap at 10M
            atr_score = min(atr_pct / 0.03, 1.0)         # prefer 3%+ ATR
            liquidity = 0.6 * vol_score + 0.4 * atr_score

            m = SymbolMetrics(
                symbol=sym,
                avg_volume=avg_vol,
                avg_dollar_volume=avg_dollar_vol,
                atr_pct=atr_pct,
                price=price,
                sector=sector,
                liquidity_score=liquidity,
                passes_filter=(
                    avg_vol >= self.config.min_avg_volume
                    and self.config.min_atr_pct <= atr_pct <= self.config.max_atr_pct
                    and price >= self.config.min_price
                ),
            )
            self._metrics[sym] = m
            scored.append(m)

        # Build filtered universe
        passed = [m for m in scored if m.passes_filter]
        passed.sort(key=lambda m: m.liquidity_score, reverse=True)
        universe = [m.symbol for m in passed[: self.config.max_symbols]]

        # Always include core ETFs even if they didn't pass a filter
        for core in self.config.always_include:
            if core not in universe:
                universe.append(core)

        self._active_universe = universe
        self._last_refresh = datetime.now()

        logger.info(
            f"Universe refreshed: {len(passed)}/{len(scored)} passed filters → "
            f"{len(universe)} active symbols"
        )
        return list(universe)

    def get_metrics(self, symbol: str) -> Optional[SymbolMetrics]:
        """Get cached metrics for a symbol."""
        return self._metrics.get(symbol)

    def get_retraining_symbols(self, n: int = 10) -> List[str]:
        """
        Return the top-N most liquid symbols for ML retraining.
        Replaces hard-coded ``UNIVERSE[:10]``.
        """
        ranked = sorted(
            self._metrics.values(),
            key=lambda m: m.liquidity_score,
            reverse=True,
        )
        return [m.symbol for m in ranked[:n] if m.passes_filter] or self._active_universe[:n]

    def get_test_universe(self, size: int = 100) -> List[str]:
        """Return a deterministic sub-set for development/testing."""
        return list(BASE_POOL[:size])

    # ── Internals ────────────────────────────────────────────────────

    @staticmethod
    def _compute_atr(
        highs: np.ndarray, lows: np.ndarray,
        closes: np.ndarray, period: int = 14,
    ) -> float:
        """Average True Range (last ``period`` bars)."""
        if len(closes) < period + 1:
            return float(np.mean(highs[-period:] - lows[-period:]))
        tr = np.maximum(
            highs[-period:] - lows[-period:],
            np.maximum(
                np.abs(highs[-period:] - closes[-period - 1:-1]),
                np.abs(lows[-period:] - closes[-period - 1:-1]),
            ),
        )
        return float(np.mean(tr))


# ── Convenience ──────────────────────────────────────────────────────────

def get_base_pool() -> List[str]:
    """Return the full unfiltered candidate pool."""
    return list(BASE_POOL)
