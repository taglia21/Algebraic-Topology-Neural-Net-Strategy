"""
equities/signal_generator.py
==============================
Unified signal pipeline for the equities trading engine.

The :class:`SignalGenerator` orchestrates all equity strategy instances,
collects their raw signals, resolves conflicts between strategies, weights
signals by strategy allocation (regime-dependent), and returns a consolidated
signal list for the execution layer.

Regime-Dependent Strategy Allocations
--------------------------------------
::

    BULL:     stat_arb= 5%, momentum=50%, factor=30%, mean_rev=13%  (total 98% deployed)
    BEAR:     stat_arb=25%, momentum=15%, factor=25%, mean_rev=30%  (total 95% deployed)
    SIDEWAYS: stat_arb=25%, momentum=30%, factor=25%, mean_rev=18%  (total 98% deployed)
    UNKNOWN:  stat_arb=20%, momentum=30%, factor=25%, mean_rev=20%  (total 95% deployed)
    CRISIS:   stat_arb=25%, momentum= 5%, factor=15%, mean_rev=35%  (total 80% deployed)

Maximum capital deployment is critical — the old 75-90% allocations left
the system sitting in 10-25% cash, which is the single largest drag on
returns vs SPY.

Conflict Resolution
-------------------
When multiple strategies emit signals for the same symbol:
    - SAME direction: strength is averaged, weighted by allocation.
    - OPPOSITE directions: signals cancel; a ``close`` signal is emitted only
      if the combined weighted strength exceeds a cancellation threshold.
    - ``close`` signals always dominate: any ``close`` from any strategy
      immediately generates a ``close`` output.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional

import pandas as pd

from core.logger import TradeLogger, get_trade_logger
from core.regime_detector import Regime, RegimeState
from equities.models import Signal
from equities.strategies.factor_model import FactorModelStrategy
from equities.strategies.mean_reversion import MeanReversionStrategy
from equities.strategies.momentum import MomentumStrategy
from equities.strategies.stat_arb import StatArbStrategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regime → Strategy allocation tables
# ---------------------------------------------------------------------------

# Allocations by regime — deploy 95-100% of capital in all regimes.
# Holding excessive cash is the #1 drag on returns vs SPY.
_REGIME_ALLOCATIONS: Dict[str, Dict[str, float]] = {
    Regime.BULL.value: {
        "stat_arb":       0.05,
        "momentum":       0.50,
        "factor_model":   0.30,
        "mean_reversion": 0.13,
    },  # total 98% — heavy momentum + factor = long-biased
    Regime.BEAR.value: {
        "stat_arb":       0.25,
        "momentum":       0.15,
        "factor_model":   0.25,
        "mean_reversion": 0.30,
    },  # total 95%
    Regime.SIDEWAYS.value: {
        "stat_arb":       0.25,
        "momentum":       0.30,
        "factor_model":   0.25,
        "mean_reversion": 0.18,
    },  # total 98%
    Regime.UNKNOWN.value: {
        "stat_arb":       0.20,
        "momentum":       0.30,
        "factor_model":   0.25,
        "mean_reversion": 0.20,
    },  # total 95%
}

# Crisis override — still deploy 80% of capital.  Crisis is a regime, not a
# reason to sit in cash.  Mean reversion and stat-arb thrive in high-vol.
_CRISIS_ALLOCATIONS: Dict[str, float] = {
    "stat_arb":       0.25,
    "momentum":       0.05,
    "factor_model":   0.15,
    "mean_reversion": 0.35,
}  # total 80%

# Minimum weighted strength to keep a combined signal (avoids noise)
_MIN_COMBINED_STRENGTH: float = 0.02

# Cancellation threshold: if opposing signals are within this band, emit close
_CANCELLATION_THRESHOLD: float = 0.15

# Directional bias: softer penalties so capital actually deploys.
# Previous 0.40 short penalty in BULL killed stat-arb pair shorts entirely.
_REGIME_DIRECTION_BIAS: Dict[str, Dict[str, float]] = {
    Regime.BULL.value:     {"long": 1.3, "short": 0.60},
    Regime.BEAR.value:     {"long": 0.6, "short": 1.2},
    Regime.SIDEWAYS.value: {"long": 1.1, "short": 0.80},
    Regime.UNKNOWN.value:  {"long": 1.0, "short": 0.70},
}


# ---------------------------------------------------------------------------
# Base strategy protocol (duck-typed)
# ---------------------------------------------------------------------------

class _StrategyProtocol:
    """Duck-typing sentinel — any object with ``generate_signals`` qualifies."""
    STRATEGY_NAME: str = ""

    def generate_signals(
        self,
        price_data: pd.DataFrame,
        regime_state: RegimeState,
    ) -> List[Signal]:  # pragma: no cover
        ...


# ---------------------------------------------------------------------------
# SignalGenerator
# ---------------------------------------------------------------------------

class SignalGenerator:
    """Orchestrates all equity strategies and produces a unified signal list.

    The generator:
    1. Calls ``generate_signals()`` on each registered strategy.
    2. Scales each signal's strength by the strategy's regime allocation.
    3. Deduplicates signals for the same symbol.
    4. Resolves directional conflicts (long vs short from different strategies).
    5. Returns the consolidated list of signals.

    Parameters
    ----------
    strategies:
        List of strategy instances.  Each must have ``STRATEGY_NAME`` and
        ``generate_signals(price_data, regime_state) -> List[Signal]``.
    trade_logger:
        Audit logger.  If *None*, uses the process default.
    custom_allocations:
        Optional custom regime→strategy allocation override.  If provided,
        completely replaces the built-in table.

    Usage
    -----
    >>> gen = SignalGenerator(strategies=[
    ...     StatArbStrategy(),
    ...     MomentumStrategy(),
    ...     FactorModelStrategy(),
    ... ])
    >>> signals = gen.generate_all_signals(price_data, regime_state)
    """

    def __init__(
        self,
        strategies: Optional[List] = None,
        trade_logger: Optional[TradeLogger] = None,
        custom_allocations: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        self._strategies: List = strategies or []
        self._log = trade_logger or get_trade_logger()
        self._allocations = custom_allocations or dict(_REGIME_ALLOCATIONS)

        logger.info(
            f"SignalGenerator initialised with {len(self._strategies)} strategies: "
            + ", ".join(
                getattr(s, "STRATEGY_NAME", type(s).__name__)
                for s in self._strategies
            )
        )

    # ------------------------------------------------------------------
    # Allocation lookup
    # ------------------------------------------------------------------

    def get_allocations(self, regime_state: RegimeState) -> Dict[str, float]:
        """Return strategy allocations for the current regime.

        Parameters
        ----------
        regime_state:
            Current market regime.

        Returns
        -------
        Dict mapping strategy name → allocation fraction in [0, 1].
        """
        if regime_state.is_crisis:
            return dict(_CRISIS_ALLOCATIONS)
        regime_key = regime_state.regime.value
        return dict(self._allocations.get(regime_key, self._allocations[Regime.UNKNOWN.value]))

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def generate_all_signals(
        self,
        price_data: pd.DataFrame,
        regime_state: RegimeState,
        fundamental_data: Optional[pd.DataFrame] = None,
    ) -> List[Signal]:
        """Run all strategies and return a unified, deduplicated signal list.

        Parameters
        ----------
        price_data:
            Wide-format close price DataFrame.
        regime_state:
            Current market regime from the regime detector.
        fundamental_data:
            Optional fundamental data passed to the factor model strategy.

        Returns
        -------
        List[Signal]:
            Consolidated signals from all strategies, with strengths scaled
            by regime allocation and conflicts resolved.
        """
        allocations = self.get_allocations(regime_state)

        raw_signals: List[Signal] = []
        for strategy in self._strategies:
            strategy_name = getattr(strategy, "STRATEGY_NAME", type(strategy).__name__)
            alloc = allocations.get(strategy_name, 0.0)

            if alloc == 0.0:
                logger.debug(
                    f"SignalGenerator: skipping {strategy_name} "
                    f"(zero allocation in {regime_state.regime.value} regime)."
                )
                continue

            try:
                # Support strategies that accept fundamental_data
                if isinstance(strategy, FactorModelStrategy):
                    strat_signals = strategy.generate_signals(
                        price_data, fundamental_data, regime_state
                    )
                elif isinstance(strategy, MeanReversionStrategy):
                    strat_signals = strategy.generate_signals(price_data, regime_state)
                else:
                    strat_signals = strategy.generate_signals(price_data, regime_state)
            except Exception as exc:
                # Fail loud: log the error and re-raise (no silent swallowing)
                self._log.log_error(
                    f"Strategy {strategy_name} raised an error in generate_signals: {exc}",
                    exc_info=exc,
                )
                raise

            # Scale signal strengths by allocation
            scaled: List[Signal] = []
            for sig in strat_signals:
                new_strength = float(sig.strength * alloc)
                # Create a copy with scaled strength
                scaled.append(
                    Signal(
                        symbol=sig.symbol,
                        direction=sig.direction,
                        strength=max(new_strength, 0.001),
                        strategy=sig.strategy,
                        metadata={
                            **sig.metadata,
                            "regime_allocation": alloc,
                            "pre_scale_strength": sig.strength,
                        },
                        timestamp=sig.timestamp,
                    )
                )

            raw_signals.extend(scaled)
            logger.debug(
                f"SignalGenerator: {strategy_name} emitted {len(scaled)} signals "
                f"(alloc={alloc:.0%}, regime={regime_state.regime.value})."
            )

        # Apply regime directional bias before combination
        regime_key = regime_state.regime.value
        bias = _REGIME_DIRECTION_BIAS.get(regime_key, _REGIME_DIRECTION_BIAS[Regime.UNKNOWN.value])
        biased_signals: List[Signal] = []
        for sig in raw_signals:
            if sig.direction in ("long", "short"):
                multiplier = bias.get(sig.direction, 1.0)
                new_strength = float(sig.strength * multiplier)
                if new_strength >= 0.001:
                    biased_signals.append(
                        Signal(
                            symbol=sig.symbol,
                            direction=sig.direction,
                            strength=min(new_strength, 1.0),
                            strategy=sig.strategy,
                            metadata={**sig.metadata, "direction_bias": multiplier},
                            timestamp=sig.timestamp,
                        )
                    )
            else:
                biased_signals.append(sig)  # close signals pass through

        combined = self.combine_signals(biased_signals)

        self._log.log_info(
            f"SignalGenerator: {len(raw_signals)} raw → {len(combined)} combined signals",
            metadata={
                "regime": regime_state.regime.value,
                "n_strategies": len(self._strategies),
                "allocations": allocations,
            },
        )
        return combined

    # ------------------------------------------------------------------
    # Signal combination and conflict resolution
    # ------------------------------------------------------------------

    def combine_signals(self, signals: List[Signal]) -> List[Signal]:
        """Deduplicate and resolve conflicts in a list of raw signals.

        Grouping logic (per symbol):
        1. If any strategy emits ``close``, emit a single ``close``.
        2. If signals for the same symbol are all in the same direction,
           average their weighted strengths.
        3. If signals conflict (long and short), compute net direction from
           weighted strengths.  If the net is near zero (within cancellation
           threshold), emit a ``close``; otherwise emit the net direction.
        4. Drop any signal with combined strength < ``_MIN_COMBINED_STRENGTH``.

        Parameters
        ----------
        signals:
            Raw signal list (may contain duplicates and conflicts).

        Returns
        -------
        List[Signal]:
            Consolidated signals with at most one signal per symbol.
        """
        if not signals:
            return []

        # Group by symbol
        by_symbol: Dict[str, List[Signal]] = defaultdict(list)
        for sig in signals:
            by_symbol[sig.symbol].append(sig)

        result: List[Signal] = []

        for symbol, sym_signals in by_symbol.items():
            # 1. Close signals dominate
            close_sigs = [s for s in sym_signals if s.direction == "close"]
            if close_sigs:
                # Emit a single close signal with strength 1.0
                strongest = max(close_sigs, key=lambda s: s.strength)
                result.append(
                    Signal(
                        symbol=symbol,
                        direction="close",
                        strength=1.0,
                        strategy="combined",
                        metadata={
                            "source_strategies": [s.strategy for s in close_sigs],
                            "reason": strongest.metadata.get("action", "combined_close"),
                        },
                    )
                )
                continue

            long_sigs = [s for s in sym_signals if s.direction == "long"]
            short_sigs = [s for s in sym_signals if s.direction == "short"]

            long_strength = sum(s.strength for s in long_sigs)
            short_strength = sum(s.strength for s in short_sigs)

            # 2. No conflict
            if long_sigs and not short_sigs:
                avg_strength = long_strength / len(long_sigs)
                if avg_strength >= _MIN_COMBINED_STRENGTH:
                    result.append(self._merge_signals(symbol, "long", long_sigs, avg_strength))
                continue

            if short_sigs and not long_sigs:
                avg_strength = short_strength / len(short_sigs)
                if avg_strength >= _MIN_COMBINED_STRENGTH:
                    result.append(self._merge_signals(symbol, "short", short_sigs, avg_strength))
                continue

            # 3. Conflict: long and short present
            net_strength = long_strength - short_strength
            abs_net = abs(net_strength)

            if abs_net < _CANCELLATION_THRESHOLD:
                # Signals cancel → emit close
                result.append(
                    Signal(
                        symbol=symbol,
                        direction="close",
                        strength=1.0,
                        strategy="combined",
                        metadata={
                            "reason": "signal_conflict_cancelled",
                            "long_strength": long_strength,
                            "short_strength": short_strength,
                        },
                    )
                )
            else:
                # Net direction wins.  Strength = abs(net) which already
                # accounts for the partial offset from opposing signals.
                # Capping at 1.0 ensures we don't exceed the max.
                direction = "long" if net_strength > 0 else "short"
                dominant_sigs = long_sigs if direction == "long" else short_sigs
                net_signal_strength = min(abs_net, 1.0)

                if net_signal_strength >= _MIN_COMBINED_STRENGTH:
                    result.append(
                        self._merge_signals(symbol, direction, dominant_sigs, net_signal_strength)
                    )

        return result

    @staticmethod
    def _merge_signals(
        symbol: str,
        direction: str,
        signals: List[Signal],
        strength: float,
    ) -> Signal:
        """Merge multiple signals for the same symbol/direction into one.

        Parameters
        ----------
        symbol:
            Ticker symbol.
        direction:
            ``"long"`` or ``"short"``.
        signals:
            All contributing signals.
        strength:
            Pre-computed combined strength.

        Returns
        -------
        A single merged :class:`Signal`.
        """
        source_strategies = list({s.strategy for s in signals})
        all_metadata = {}
        for sig in signals:
            for k, v in sig.metadata.items():
                all_metadata[f"{sig.strategy}__{k}"] = v

        return Signal(
            symbol=symbol,
            direction=direction,
            strength=float(min(strength, 1.0)),
            strategy="combined",
            metadata={
                "source_strategies": source_strategies,
                "individual_strengths": {s.strategy: s.strength for s in signals},
                **all_metadata,
            },
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def add_strategy(self, strategy) -> None:
        """Register a new strategy with the generator.

        Parameters
        ----------
        strategy:
            Strategy instance with ``STRATEGY_NAME`` and
            ``generate_signals(price_data, regime_state) -> List[Signal]``.
        """
        name = getattr(strategy, "STRATEGY_NAME", type(strategy).__name__)
        logger.info(f"SignalGenerator: registering strategy '{name}'.")
        self._strategies.append(strategy)

    def set_allocations(
        self,
        regime: str,
        allocations: Dict[str, float],
    ) -> None:
        """Override strategy allocations for a specific regime.

        Parameters
        ----------
        regime:
            Regime string, e.g. ``"BULL"``, ``"BEAR"``, ``"SIDEWAYS"``,
            ``"UNKNOWN"``.
        allocations:
            Mapping of strategy name → allocation fraction.  Should sum to ≤ 1.0.
        """
        total = sum(allocations.values())
        if total > 1.001:
            raise ValueError(
                f"Strategy allocations for regime {regime!r} sum to {total:.3f} > 1.0. "
                "Reduce allocations so the total does not exceed 1.0."
            )
        self._allocations[regime] = dict(allocations)
        logger.info(
            f"SignalGenerator: updated {regime} allocations → {allocations}"
        )

    @property
    def strategies(self) -> List:
        """List of registered strategy instances."""
        return list(self._strategies)

    @property
    def n_strategies(self) -> int:
        """Number of registered strategies."""
        return len(self._strategies)
