"""
Order Book Imbalance — Real-Time Microstructure Signals (TIER 4)
=================================================================

Analyzes bid/ask depth, trade flow toxicity, and spread dynamics
to generate short-term directional microstructure signals.

Components:
1. **OrderBookImbalance** — Depth imbalance ratio across price levels
2. **VWAP deviation** — Detects institutional accumulation / distribution
3. **VPIN (trade flow toxicity)** — Volume-synchronised probability of informed trading
4. **Spread dynamics** — Bid-ask compression / expansion predictor
5. **get_microstructure_signal()** — Composite bullish / bearish / neutral

Usage:
    from src.order_book_imbalance import OrderBookImbalance

    obi = OrderBookImbalance()
    obi.update_book(bids=[(100.0, 500), (99.9, 700)],
                    asks=[(100.1, 400), (100.2, 600)])
    obi.add_trade(price=100.05, size=200, side="buy")
    signal = obi.get_microstructure_signal("AAPL")
"""

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS & CONFIG
# =============================================================================

class MicroSignal(Enum):
    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"


@dataclass
class MicroConfig:
    """Microstructure analyser configuration."""
    book_levels: int = 5                    # depth levels to analyse
    vpin_bucket_size: int = 50              # trades per VPIN bucket
    vpin_window: int = 50                   # buckets in rolling window
    vpin_toxic_threshold: float = 0.7       # above = toxic flow
    vwap_deviation_threshold: float = 0.003 # 30 bps
    spread_ema_alpha: float = 0.1           # EMA smoothing for spread
    imbalance_ema_alpha: float = 0.15
    signal_lookback: int = 100              # trades for rolling calcs
    strong_threshold: float = 0.65          # strong signal threshold


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class BookLevel:
    """Single price level."""
    price: float
    size: float


@dataclass
class TradeEvent:
    """Single trade tick."""
    timestamp: float          # epoch seconds
    price: float
    size: float
    side: str                 # "buy" or "sell"


@dataclass
class MicrostructureState:
    """Full microstructure snapshot."""
    symbol: str = ""
    timestamp: str = ""
    # Imbalance
    bid_depth: float = 0.0
    ask_depth: float = 0.0
    imbalance_ratio: float = 0.0      # (bid-ask)/(bid+ask), [-1, 1]
    imbalance_ema: float = 0.0
    # VWAP
    vwap: float = 0.0
    last_price: float = 0.0
    vwap_deviation: float = 0.0       # (price - vwap) / vwap
    # VPIN
    vpin: float = 0.0
    is_toxic: bool = False
    # Spread
    spread_bps: float = 0.0
    spread_ema_bps: float = 0.0
    spread_z_score: float = 0.0       # spread relative to recent history
    # Composite
    composite_score: float = 0.0      # [-1, 1]
    signal: str = "neutral"


# =============================================================================
# ORDER BOOK IMBALANCE ANALYSER
# =============================================================================

class OrderBookImbalance:
    """
    Real-time microstructure signal generator.

    Maintains internal state from streaming book updates and trade ticks.
    Call ``get_microstructure_signal(symbol)`` for a composite signal.
    """

    def __init__(self, config: Optional[MicroConfig] = None):
        self.config = config or MicroConfig()

        # Book state
        self._bids: List[BookLevel] = []
        self._asks: List[BookLevel] = []

        # Trade tape
        self._trades: deque = deque(maxlen=5000)

        # VPIN
        self._vpin_buy_vol: float = 0.0
        self._vpin_sell_vol: float = 0.0
        self._vpin_bucket_trades: int = 0
        self._vpin_buckets: deque = deque(maxlen=self.config.vpin_window)

        # EMA state
        self._imbalance_ema: float = 0.0
        self._spread_ema: float = 0.0
        self._spread_history: deque = deque(maxlen=500)

        # VWAP
        self._cum_pv: float = 0.0
        self._cum_vol: float = 0.0

        self._symbol: str = ""
        logger.info("OrderBookImbalance initialised (levels=%d)", self.config.book_levels)

    # ── Book updates ─────────────────────────────────────────────────────

    def update_book(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
    ) -> None:
        """
        Update order book snapshot.

        Parameters
        ----------
        bids : list of (price, size) sorted descending
        asks : list of (price, size) sorted ascending
        """
        n = self.config.book_levels
        self._bids = [BookLevel(p, s) for p, s in bids[:n]]
        self._asks = [BookLevel(p, s) for p, s in asks[:n]]

        # Imbalance
        bid_depth = sum(l.size for l in self._bids)
        ask_depth = sum(l.size for l in self._asks)
        total = bid_depth + ask_depth
        ratio = (bid_depth - ask_depth) / total if total > 0 else 0.0

        alpha = self.config.imbalance_ema_alpha
        self._imbalance_ema = alpha * ratio + (1 - alpha) * self._imbalance_ema

        # Spread
        if self._bids and self._asks:
            best_bid = self._bids[0].price
            best_ask = self._asks[0].price
            mid = (best_bid + best_ask) / 2
            spread_bps = (best_ask - best_bid) / mid * 10_000 if mid > 0 else 0.0
            self._spread_history.append(spread_bps)
            a = self.config.spread_ema_alpha
            self._spread_ema = a * spread_bps + (1 - a) * self._spread_ema

    # ── Trade ticks ──────────────────────────────────────────────────────

    def add_trade(self, price: float, size: float, side: str) -> None:
        """
        Record a single trade tick.

        Parameters
        ----------
        price : execution price
        size : number of shares
        side : "buy" or "sell"
        """
        ts = time.time()
        self._trades.append(TradeEvent(ts, price, size, side.lower()))

        # VWAP
        self._cum_pv += price * size
        self._cum_vol += size

        # VPIN bucket
        if side.lower() == "buy":
            self._vpin_buy_vol += size
        else:
            self._vpin_sell_vol += size
        self._vpin_bucket_trades += 1

        if self._vpin_bucket_trades >= self.config.vpin_bucket_size:
            total = self._vpin_buy_vol + self._vpin_sell_vol
            order_imb = abs(self._vpin_buy_vol - self._vpin_sell_vol) / total if total > 0 else 0
            self._vpin_buckets.append(order_imb)
            self._vpin_buy_vol = 0.0
            self._vpin_sell_vol = 0.0
            self._vpin_bucket_trades = 0

    # ── Signal generation ────────────────────────────────────────────────

    def get_microstructure_signal(self, symbol: str = "") -> MicrostructureState:
        """
        Compute composite microstructure signal.

        Returns
        -------
        MicrostructureState with composite_score ∈ [-1, 1] and
        signal ∈ {strong_bullish, bullish, neutral, bearish, strong_bearish}.
        """
        self._symbol = symbol or self._symbol
        state = MicrostructureState(
            symbol=self._symbol,
            timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        )

        # ── Imbalance ──
        bid_depth = sum(l.size for l in self._bids)
        ask_depth = sum(l.size for l in self._asks)
        total = bid_depth + ask_depth
        state.bid_depth = bid_depth
        state.ask_depth = ask_depth
        state.imbalance_ratio = (bid_depth - ask_depth) / total if total > 0 else 0.0
        state.imbalance_ema = self._imbalance_ema

        # ── VWAP ──
        state.vwap = self._cum_pv / self._cum_vol if self._cum_vol > 0 else 0.0
        if self._trades:
            state.last_price = self._trades[-1].price
        state.vwap_deviation = (
            (state.last_price - state.vwap) / state.vwap if state.vwap > 0 else 0.0
        )

        # ── VPIN ──
        if self._vpin_buckets:
            state.vpin = float(np.mean(self._vpin_buckets))
        state.is_toxic = state.vpin > self.config.vpin_toxic_threshold

        # ── Spread ──
        if self._bids and self._asks:
            best_bid = self._bids[0].price
            best_ask = self._asks[0].price
            mid = (best_bid + best_ask) / 2
            state.spread_bps = (best_ask - best_bid) / mid * 10_000 if mid > 0 else 0.0
        state.spread_ema_bps = self._spread_ema
        if len(self._spread_history) > 10:
            arr = np.array(self._spread_history)
            mu, sigma = float(np.mean(arr)), float(np.std(arr))
            state.spread_z_score = (state.spread_bps - mu) / sigma if sigma > 0 else 0.0

        # ── Composite score ──
        # Weighted combination: imbalance (40%), VWAP dev (30%), VPIN (20%), spread (10%)
        imb_signal = np.clip(state.imbalance_ema * 2, -1, 1)  # scale to [-1, 1]
        vwap_signal = np.clip(state.vwap_deviation / 0.01, -1, 1)
        vpin_signal = -1.0 if state.is_toxic else 0.0  # toxic => bearish
        spread_signal = np.clip(-state.spread_z_score / 3, -1, 1)  # wide spread => caution

        composite = (
            0.40 * imb_signal +
            0.30 * vwap_signal +
            0.20 * vpin_signal +
            0.10 * spread_signal
        )
        state.composite_score = float(np.clip(composite, -1, 1))

        # Map to signal
        t = self.config.strong_threshold
        if state.composite_score > t:
            state.signal = MicroSignal.STRONG_BULLISH.value
        elif state.composite_score > t * 0.4:
            state.signal = MicroSignal.BULLISH.value
        elif state.composite_score < -t:
            state.signal = MicroSignal.STRONG_BEARISH.value
        elif state.composite_score < -t * 0.4:
            state.signal = MicroSignal.BEARISH.value
        else:
            state.signal = MicroSignal.NEUTRAL.value

        return state

    # ── Accessors ────────────────────────────────────────────────────────

    @property
    def current_vpin(self) -> float:
        return float(np.mean(self._vpin_buckets)) if self._vpin_buckets else 0.0

    @property
    def current_vwap(self) -> float:
        return self._cum_pv / self._cum_vol if self._cum_vol > 0 else 0.0

    @property
    def trade_count(self) -> int:
        return len(self._trades)

    def reset(self) -> None:
        """Reset all internal state."""
        self._bids.clear()
        self._asks.clear()
        self._trades.clear()
        self._vpin_buckets.clear()
        self._vpin_buy_vol = 0.0
        self._vpin_sell_vol = 0.0
        self._vpin_bucket_trades = 0
        self._imbalance_ema = 0.0
        self._spread_ema = 0.0
        self._spread_history.clear()
        self._cum_pv = 0.0
        self._cum_vol = 0.0


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import random

    obi = OrderBookImbalance()

    for i in range(200):
        mid = 100.0 + random.gauss(0, 0.2)
        bids = [(mid - 0.01 * (j + 1), random.randint(100, 1000)) for j in range(5)]
        asks = [(mid + 0.01 * (j + 1), random.randint(100, 1000)) for j in range(5)]
        obi.update_book(bids, asks)

        side = "buy" if random.random() > 0.45 else "sell"
        obi.add_trade(mid + random.gauss(0, 0.01), random.randint(10, 300), side)

    sig = obi.get_microstructure_signal("AAPL")
    print(f"Signal: {sig.signal}")
    print(f"Composite: {sig.composite_score:.4f}")
    print(f"Imbalance: {sig.imbalance_ratio:.4f} (EMA: {sig.imbalance_ema:.4f})")
    print(f"VWAP: {sig.vwap:.2f}, deviation: {sig.vwap_deviation:.6f}")
    print(f"VPIN: {sig.vpin:.4f}, toxic: {sig.is_toxic}")
    print(f"Spread: {sig.spread_bps:.2f} bps (EMA: {sig.spread_ema_bps:.2f})")
