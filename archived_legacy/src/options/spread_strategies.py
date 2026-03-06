"""
Phase 5b — Advanced Credit Spread Strategies
=============================================

Three regime-aware spread strategies that integrate with the existing
SignalGenerator pipeline:

1. **IronCondorStrategy**
   - Sell when IV rank > 65, 30-45 DTE, delta 0.15-0.25, 5-pt wings
   - Exit at 50 % profit or 21 DTE remaining
   - Best in BULL_LOW_VOL / BULL_HIGH_VOL regimes

2. **BullPutSpreadStrategy**
   - Bullish regime (BULL_*) + IV rank > 50
   - Sell OTM put spread for credit
   - Delta 0.20-0.30 for short put

3. **BearCallSpreadStrategy**
   - Bearish regime (BEAR_*) + IV rank > 50
   - Sell OTM call spread for credit
   - Delta 0.20-0.30 for short call

All strategies produce ``Signal`` objects compatible with the existing
``SignalGenerator.generate_all_signals()`` pipeline.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from .config import RISK_CONFIG
from .iv_data_manager import IVDataManager
from .signal_generator import Signal, SignalSource, SignalType
from .universe import get_universe, is_strategy_allowed


logger = logging.getLogger(__name__)


# ── New signal-source enum values for spreads ───────────────────────
class SpreadSignalSource(str, Enum):
    """Extended signal sources for credit spread strategies."""
    IRON_CONDOR = "iron_condor_strategy"
    BULL_PUT_SPREAD = "bull_put_spread_strategy"
    BEAR_CALL_SPREAD = "bear_call_spread_strategy"


# ── Configuration ────────────────────────────────────────────────────

@dataclass
class SpreadConfig:
    """Tunable knobs for the spread strategies."""

    # Iron Condor
    ic_iv_rank_min: float = 65.0          # minimum IV rank to enter
    ic_dte_min: int = 30                  # minimum DTE at entry
    ic_dte_max: int = 45                  # maximum DTE at entry
    ic_delta_min: float = 0.15            # short-strike delta lower bound
    ic_delta_max: float = 0.25            # short-strike delta upper bound
    ic_wing_width: float = 5.0            # default wing width ($)
    ic_profit_target_pct: float = 0.50    # close at 50 % of max profit
    ic_dte_exit: int = 21                 # close if DTE falls below this
    ic_max_loss_pct: float = 2.0          # skip if max-loss > 2 × credit

    # Bull Put Spread
    bp_iv_rank_min: float = 50.0
    bp_dte_min: int = 25
    bp_dte_max: int = 45
    bp_delta_min: float = 0.20
    bp_delta_max: float = 0.30
    bp_wing_width: float = 5.0
    bp_profit_target_pct: float = 0.50

    # Bear Call Spread
    bc_iv_rank_min: float = 50.0
    bc_dte_min: int = 25
    bc_dte_max: int = 45
    bc_delta_min: float = 0.20
    bc_delta_max: float = 0.30
    bc_wing_width: float = 5.0
    bc_profit_target_pct: float = 0.50


# ── Regime helper ────────────────────────────────────────────────────

class _RegimeHint(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


def _infer_regime_hint(iv_rank: float) -> _RegimeHint:
    """
    Lightweight regime inference from IV rank alone
    (no HMM dependency) — used as fallback when the full
    RegimeDetector has not been injected.

    Low IV rank (<40) tends to accompany bullish drift;
    high rank (>70) suggests stress / bearish bias.
    """
    if iv_rank < 40:
        return _RegimeHint.BULLISH
    if iv_rank > 70:
        return _RegimeHint.BEARISH
    return _RegimeHint.NEUTRAL


# =====================================================================
#  1. Iron Condor Strategy
# =====================================================================

class IronCondorStrategy:
    """
    Sell iron condors when IV is elevated and the market is range-bound.

    Entry criteria:
      • IV rank ≥ ``ic_iv_rank_min`` (default 65)
      • Optimal DTE 30-45
      • Short-strike delta 0.15-0.25
      • Wing width 5 pts (configurable)

    Exit criteria (communicated via ``Signal.reason`` metadata):
      • 50 % of max profit reached
      • DTE < 21
      • Max-loss breached
    """

    def __init__(self, config: Optional[SpreadConfig] = None):
        self.cfg = config or SpreadConfig()
        self.risk = RISK_CONFIG
        self.iv_data = IVDataManager()
        self.logger = logging.getLogger(f"{__name__}.IronCondorStrategy")

    async def generate_signals(self, symbols: List[str]) -> List[Signal]:
        """Scan universe for iron-condor opportunities."""
        signals: List[Signal] = []
        for symbol in symbols:
            try:
                sig = await self._evaluate(symbol)
                if sig is not None:
                    signals.append(sig)
            except Exception as exc:
                self.logger.debug(f"IC scan error for {symbol}: {exc}")
        return signals

    async def _evaluate(self, symbol: str) -> Optional[Signal]:
        # Must be allowed in the universe
        if not is_strategy_allowed(symbol, "iron_condor"):
            return None

        iv_rank = self.iv_data.get_iv_rank(symbol)
        if iv_rank is None or iv_rank < self.cfg.ic_iv_rank_min:
            return None

        # Compute optimal DTE (midpoint of range)
        dte = (self.cfg.ic_dte_min + self.cfg.ic_dte_max) // 2

        # Confidence: scale from ic_iv_rank_min → 100  to 0.5 → 1.0
        raw_conf = (iv_rank - self.cfg.ic_iv_rank_min) / max(
            100 - self.cfg.ic_iv_rank_min, 1
        )
        confidence = round(0.50 + 0.50 * min(raw_conf, 1.0), 3)

        # Approximate PoP for a balanced IC with ~0.20 delta wings
        delta_mid = (self.cfg.ic_delta_min + self.cfg.ic_delta_max) / 2
        pop = round(1.0 - 2 * delta_mid, 2)  # ~0.60 for delta 0.20

        return Signal(
            symbol=symbol,
            signal_type=SignalType.SELL,
            signal_source=SignalSource.IV_RANK,  # reuse existing enum
            strategy="iron_condor",
            confidence=confidence,
            timestamp=datetime.now(),
            iv_rank=iv_rank,
            dte=dte,
            probability_of_profit=pop,
            reason=(
                f"IC: IV rank {iv_rank:.0f} ≥ {self.cfg.ic_iv_rank_min}, "
                f"DTE {dte}, delta {delta_mid:.2f}, "
                f"profit target {self.cfg.ic_profit_target_pct:.0%}, "
                f"exit if DTE < {self.cfg.ic_dte_exit}"
            ),
        )


# =====================================================================
#  2. Bull Put Spread Strategy
# =====================================================================

class BullPutSpreadStrategy:
    """
    Sell OTM put credit spread in bullish regimes.

    Entry:
      • Regime hint = BULLISH or IV rank < 40  (market not in distress)
      • IV rank ≥ 50  (enough premium to sell)
      • 25-45 DTE, delta 0.20-0.30

    The strategy profits from time decay + supportive price action.
    """

    def __init__(
        self,
        config: Optional[SpreadConfig] = None,
        regime_hint_fn=None,
    ):
        self.cfg = config or SpreadConfig()
        self.risk = RISK_CONFIG
        self.iv_data = IVDataManager()
        self._regime_hint = regime_hint_fn or _infer_regime_hint
        self.logger = logging.getLogger(f"{__name__}.BullPutSpreadStrategy")

    async def generate_signals(self, symbols: List[str]) -> List[Signal]:
        signals: List[Signal] = []
        for symbol in symbols:
            try:
                sig = await self._evaluate(symbol)
                if sig is not None:
                    signals.append(sig)
            except Exception as exc:
                self.logger.debug(f"Bull put scan error for {symbol}: {exc}")
        return signals

    async def _evaluate(self, symbol: str) -> Optional[Signal]:
        if not is_strategy_allowed(symbol, "credit_spread") and not is_strategy_allowed(symbol, "put_spread"):
            return None

        iv_rank = self.iv_data.get_iv_rank(symbol)
        if iv_rank is None or iv_rank < self.cfg.bp_iv_rank_min:
            return None

        # Regime gate: only enter in bullish conditions
        hint = self._regime_hint(iv_rank)
        if hint == _RegimeHint.BEARISH:
            return None

        dte = (self.cfg.bp_dte_min + self.cfg.bp_dte_max) // 2
        delta_mid = (self.cfg.bp_delta_min + self.cfg.bp_delta_max) / 2

        raw_conf = (iv_rank - self.cfg.bp_iv_rank_min) / max(
            100 - self.cfg.bp_iv_rank_min, 1
        )
        confidence = round(0.45 + 0.45 * min(raw_conf, 1.0), 3)

        pop = round(1.0 - delta_mid, 2)  # ~ 0.75 for delta 0.25

        return Signal(
            symbol=symbol,
            signal_type=SignalType.SELL,
            signal_source=SignalSource.MEAN_REVERSION,  # reuse existing
            strategy="put_spread",
            confidence=confidence,
            timestamp=datetime.now(),
            iv_rank=iv_rank,
            dte=dte,
            probability_of_profit=pop,
            reason=(
                f"Bull put spread: IV rank {iv_rank:.0f}, "
                f"regime={hint.value}, DTE {dte}, delta {delta_mid:.2f}"
            ),
        )


# =====================================================================
#  3. Bear Call Spread Strategy
# =====================================================================

class BearCallSpreadStrategy:
    """
    Sell OTM call credit spread in bearish / high-vol regimes.

    Entry:
      • Regime hint = BEARISH or IV rank > 70
      • IV rank ≥ 50
      • 25-45 DTE, delta 0.20-0.30

    Profits from downward / sideways price action + elevated IV.
    """

    def __init__(
        self,
        config: Optional[SpreadConfig] = None,
        regime_hint_fn=None,
    ):
        self.cfg = config or SpreadConfig()
        self.risk = RISK_CONFIG
        self.iv_data = IVDataManager()
        self._regime_hint = regime_hint_fn or _infer_regime_hint
        self.logger = logging.getLogger(f"{__name__}.BearCallSpreadStrategy")

    async def generate_signals(self, symbols: List[str]) -> List[Signal]:
        signals: List[Signal] = []
        for symbol in symbols:
            try:
                sig = await self._evaluate(symbol)
                if sig is not None:
                    signals.append(sig)
            except Exception as exc:
                self.logger.debug(f"Bear call scan error for {symbol}: {exc}")
        return signals

    async def _evaluate(self, symbol: str) -> Optional[Signal]:
        if not is_strategy_allowed(symbol, "credit_spread") and not is_strategy_allowed(symbol, "call_spread"):
            return None

        iv_rank = self.iv_data.get_iv_rank(symbol)
        if iv_rank is None or iv_rank < self.cfg.bc_iv_rank_min:
            return None

        # Regime gate: only enter in bearish / neutral conditions
        hint = self._regime_hint(iv_rank)
        if hint == _RegimeHint.BULLISH:
            return None

        dte = (self.cfg.bc_dte_min + self.cfg.bc_dte_max) // 2
        delta_mid = (self.cfg.bc_delta_min + self.cfg.bc_delta_max) / 2

        raw_conf = (iv_rank - self.cfg.bc_iv_rank_min) / max(
            100 - self.cfg.bc_iv_rank_min, 1
        )
        confidence = round(0.45 + 0.45 * min(raw_conf, 1.0), 3)

        pop = round(1.0 - delta_mid, 2)

        return Signal(
            symbol=symbol,
            signal_type=SignalType.SELL,
            signal_source=SignalSource.MEAN_REVERSION,
            strategy="call_spread",
            confidence=confidence,
            timestamp=datetime.now(),
            iv_rank=iv_rank,
            dte=dte,
            probability_of_profit=pop,
            reason=(
                f"Bear call spread: IV rank {iv_rank:.0f}, "
                f"regime={hint.value}, DTE {dte}, delta {delta_mid:.2f}"
            ),
        )


# =====================================================================
#  Spread Strategy Aggregator
# =====================================================================

class SpreadStrategyAggregator:
    """
    Convenience wrapper that runs all three spread strategies and
    deduplicates / ranks the output signals.

    Drop-in replacement for a single strategy when plugged into
    ``SignalGenerator.generate_all_signals``.
    """

    def __init__(
        self,
        config: Optional[SpreadConfig] = None,
        regime_hint_fn=None,
    ):
        self.cfg = config or SpreadConfig()
        self.ic = IronCondorStrategy(self.cfg)
        self.bp = BullPutSpreadStrategy(self.cfg, regime_hint_fn)
        self.bc = BearCallSpreadStrategy(self.cfg, regime_hint_fn)
        self.logger = logging.getLogger(f"{__name__}.Aggregator")

    async def generate_signals(self, symbols: List[str]) -> List[Signal]:
        """Run all three spread strategies in parallel and merge."""
        results = await asyncio.gather(
            self.ic.generate_signals(symbols),
            self.bp.generate_signals(symbols),
            self.bc.generate_signals(symbols),
            return_exceptions=True,
        )

        merged: List[Signal] = []
        for batch in results:
            if isinstance(batch, list):
                merged.extend(batch)
            else:
                self.logger.error(f"Spread sub-strategy failed: {batch}")

        # Deduplicate: keep highest-confidence signal per (symbol, strategy)
        best: Dict[tuple, Signal] = {}
        for sig in merged:
            key = (sig.symbol, sig.strategy)
            if key not in best or sig.confidence > best[key].confidence:
                best[key] = sig

        deduped = sorted(best.values(), key=lambda s: s.confidence, reverse=True)
        self.logger.info(
            f"Spread strategies produced {len(merged)} raw → {len(deduped)} deduped signals"
        )
        return deduped


# =====================================================================
# TIER 2 — Adaptive Spread Quoter (Phase I, Item 12)
# =====================================================================

class AdaptiveSpreadQuoter:
    """Price credit-spread limit orders at mid + 1-tick improvement.

    Retries up to ``max_retries`` times with 30-second intervals, widening
    the limit by ``tick_step`` each retry until filled or exhausted.

    Parameters
    ----------
    tick_step : float
        Per-retry price improvement step (default 0.01 = $0.01).
    retry_interval_seconds : int
        Seconds between retries (default 30).
    max_retries : int
        Maximum number of retry attempts (default 5).
    """

    def __init__(
        self,
        tick_step: float = 0.01,
        retry_interval_seconds: int = 30,
        max_retries: int = 5,
    ):
        self.tick_step = tick_step
        self.retry_interval_seconds = retry_interval_seconds
        self.max_retries = max_retries
        self.logger = logging.getLogger(f"{__name__}.AdaptiveSpreadQuoter")

    def compute_limit_price(
        self,
        bid: float,
        ask: float,
        retry: int = 0,
    ) -> float:
        """Compute limit price = mid + 1-tick improvement, plus retry widening.

        Args:
            bid: Best bid.
            ask: Best ask.
            retry: Current retry number (0-based).

        Returns:
            Limit price in USD.
        """
        mid = (bid + ask) / 2.0
        # Start at mid + 1 tick improvement; widen each retry
        limit = mid + self.tick_step + (self.tick_step * retry)
        return round(limit, 2)

    async def quote_and_fill(
        self,
        symbol: str,
        bid: float,
        ask: float,
        submit_fn=None,
    ) -> Optional[Dict]:
        """Attempt to fill a limit order with adaptive pricing.

        Args:
            symbol: OCC symbol or underlying.
            bid: Best bid.
            ask: Best ask.
            submit_fn: Async callable ``submit_fn(symbol, limit_price)``
                that returns ``{"filled": bool, ...}``.

        Returns:
            Fill result dict, or None if all retries exhausted.
        """
        for attempt in range(self.max_retries):
            limit = self.compute_limit_price(bid, ask, retry=attempt)
            self.logger.info(
                f"[{symbol}] Attempt {attempt+1}/{self.max_retries}: "
                f"limit=${limit:.2f} (bid={bid:.2f}, ask={ask:.2f})"
            )

            if submit_fn is not None:
                result = await submit_fn(symbol, limit)
                if result and result.get("filled"):
                    self.logger.info(f"[{symbol}] Filled at ${limit:.2f}")
                    return result

            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_interval_seconds)

        self.logger.warning(f"[{symbol}] Exhausted {self.max_retries} retries")
        return None
