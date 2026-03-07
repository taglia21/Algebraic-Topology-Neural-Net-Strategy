"""
equities/strategies/mean_reversion.py
=======================================
Short-Term Mean Reversion Strategy for the ATNN trading system.

Overview
--------
Implements a daily-bar mean reversion strategy grounded in the "overnight-
intraday reversal" and short-term contrarian literature (Lehmann 1990;
Lo & MacKinlay 1990).  The core insight is that stocks that have moved sharply
over a 5-day window tend to revert over the following 3–5 trading days.

The strategy is designed to be the **primary trade-frequency driver** in a
15-stock universe, targeting 2–8 signals per week (≈ 40–160 per year).

Signal Generation Pipeline
--------------------------
For each non-benchmark stock in the universe:

1. **5-Day Return Z-Score**
   Compute the 5-day arithmetic return, then z-score it against the stock's
   own trailing 60-day return distribution.

   - z < −``entry_z``  → "oversold"  (buy candidate)
   - z >  ``entry_z``  → "overbought" (sell candidate)

2. **RSI Filter** (5-period RSI per Wilder's definition)
   Only enter long when RSI < ``rsi_oversold``   (default 30).
   Only enter short when RSI > ``rsi_overbought`` (default 70).

3. **Volume Spike Filter**
   Require today's volume ≥ 1.2× the 20-day average volume.  This filters out
   low-conviction drift and focuses on flush / capitulation moves where mean
   reversion is more reliable.

4. **Volatility-Scaled Sizing** (Barroso & Santa-Clara 2015)
   Signal strength ∝ ``vol_target / realized_vol``, capped to [0.1, 1.0].

5. **Holding Period**
   Auto-exit after ``holding_days`` (default 5) trading bars have elapsed.
   Each bar call to :meth:`generate_signals` increments an internal bar
   counter per held position.

Exit Logic
----------
- **Mean-reversion complete**: z-score crosses back through ±``exit_z``.
- **Time stop**: position has been held for ``holding_days`` bars.
- **Hard stop**: z-score reaches ±``stop_z`` (loss exceeds expectation).
- **Crisis**: all open positions are closed when ``regime_state.is_crisis``.

Regime Overlay
--------------
- CRISIS  → close all existing positions; block all new entries.
- BEAR    → block new short entries (trending down may continue); longs only.
- BULL    → standard operation.
- UNKNOWN → standard operation (insufficient history guard in data layer).

References
----------
- Lehmann, B.N. (1990). Fads, Martingales, and Market Efficiency.
  *Quarterly Journal of Economics*, 105(1), 1–28.
- Lo, A.W. & MacKinlay, A.C. (1990). When are Contrarian Profits Due to
  Stock Market Overreaction? *Review of Financial Studies*, 3(2), 175–205.
- Barroso, P. & Santa-Clara, P. (2015). Momentum Has Its Moments.
  *Journal of Financial Economics*, 116(1), 111–120.

Usage
-----
>>> strategy = MeanReversionStrategy()
>>> signals = strategy.generate_signals(price_data, regime_state)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.config import MeanReversionConfig, get_config
from core.logger import TradeLogger, get_trade_logger
from core.regime_detector import Regime, RegimeState
from equities.models import Signal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Benchmark symbols excluded from signal generation
# ---------------------------------------------------------------------------
_BENCHMARKS: frozenset = frozenset({"SPY", "QQQ", "IWM"})

# Volume spike multiplier threshold
_VOLUME_SPIKE_THRESHOLD: float = 1.2

# Minimum bars of history required before any signal can be emitted.
# Must cover the 60-day z-score lookback + a small warm-up buffer.
_MIN_HISTORY_BARS: int = 65


# ---------------------------------------------------------------------------
# RSI computation (pure NumPy — no TA-Lib dependency)
# ---------------------------------------------------------------------------

def _compute_rsi(prices: pd.Series, period: int = 5) -> pd.Series:
    """Compute Wilder's RSI for a price series.

    Uses the exponential moving average (EMA) smoothing approach with
    ``alpha = 1 / period``, consistent with Wilder's original formulation.

    Parameters
    ----------
    prices:
        Close price series (daily, chronologically ordered).
    period:
        RSI lookback period (default 5 for short-term mean reversion).

    Returns
    -------
    pd.Series
        RSI values in [0, 100].  The first ``period`` values are NaN.
    """
    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder EMA smoothing: alpha = 1/period, adjust=False
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Where avg_loss == 0 and avg_gain > 0, RSI is exactly 100
    rsi = rsi.where(avg_loss > 0, other=100.0)
    # Where both are 0 (no movement), RSI is 50 (neutral)
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), other=50.0)

    return rsi


# ---------------------------------------------------------------------------
# Main strategy class
# ---------------------------------------------------------------------------

class MeanReversionStrategy:
    """Short-term mean reversion strategy based on 5-day return z-scores.

    Enters long positions in oversold stocks (z < −entry_z, RSI < 30,
    volume spike) and short positions in overbought stocks (z > entry_z,
    RSI > 70, volume spike) for a 15-stock daily-bar universe.

    This is the primary trade-frequency driver of the ATNN system, targeting
    2–8 signals per week in a 15-stock universe.

    Parameters
    ----------
    config:
        ``MeanReversionConfig`` from ``core.config``.  Defaults to the
        system configuration singleton when ``None``.
    trade_logger:
        ``TradeLogger`` for structured audit logging.  Defaults to the
        process-level logger singleton when ``None``.

    Attributes
    ----------
    STRATEGY_NAME:
        Strategy identifier used in :class:`~equities.models.Signal` objects.

    Internal State
    --------------
    ``_positions``
        Dict mapping symbol → position metadata dict with keys:

        - ``entry_bar``    (int)   — value of ``_bar_count`` when entered
        - ``entry_zscore`` (float) — z-score at entry
        - ``direction``    (str)   — ``"long"`` or ``"short"``

    ``_bar_count``
        Monotonically increasing counter incremented once per
        :meth:`generate_signals` call.  Used to compute holding durations.
    """

    STRATEGY_NAME: str = "mean_reversion"

    def __init__(
        self,
        config: Optional[MeanReversionConfig] = None,
        trade_logger: Optional[TradeLogger] = None,
    ) -> None:
        self._cfg: MeanReversionConfig = config or get_config().strategy.mean_reversion
        self._log: TradeLogger = trade_logger or get_trade_logger()

        # Active position book: symbol → {entry_bar, entry_zscore, direction}
        self._positions: Dict[str, Dict] = {}

        # Bar counter — incremented at the end of each generate_signals call
        self._bar_count: int = 0

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        price_data: pd.DataFrame,
        regime_state: RegimeState,
        volume_data: Optional[pd.DataFrame] = None,
    ) -> List[Signal]:
        """Generate mean reversion entry and exit signals for one bar.

        Parameters
        ----------
        price_data:
            Wide-format close price DataFrame with a sorted
            :class:`~pandas.DatetimeIndex` and one column per symbol.
            Must contain at least ``_MIN_HISTORY_BARS`` (65) rows.
        regime_state:
            Current market regime from :class:`~core.regime_detector.RegimeDetector`.
        volume_data:
            Optional wide-format volume DataFrame matching the shape and
            columns of ``price_data``.  When provided, volume spike
            filtering is applied.  When ``None``, the volume filter is
            skipped (all stocks pass volume gate).

        Returns
        -------
        List[Signal]
            A mixed list of entry signals (``direction`` = ``"long"`` or
            ``"short"``) and exit signals (``direction`` = ``"close"``).
            Exit signals are always emitted before entry signals so that
            the execution layer can free up capital first.

        Notes
        -----
        - CRISIS regime: closes all open positions, blocks new entries.
        - BEAR regime: blocks new short entries.
        - Signals are ordered: exits first, then entries.
        - ``_bar_count`` is incremented exactly once per call.
        """
        signals: List[Signal] = []

        # ----------------------------------------------------------------
        # Guard: insufficient history
        # ----------------------------------------------------------------
        if len(price_data) < _MIN_HISTORY_BARS:
            logger.warning(
                f"MeanReversionStrategy: only {len(price_data)} bars available "
                f"(need {_MIN_HISTORY_BARS}).  No signals emitted."
            )
            self._bar_count += 1
            return []

        # ----------------------------------------------------------------
        # Step 1: Compute 5-day returns and z-scores for all non-benchmark
        #         stocks with sufficient history.
        # ----------------------------------------------------------------
        tradeable = [
            col for col in price_data.columns
            if col not in _BENCHMARKS
        ]

        zscores: Dict[str, float] = {}
        rsi_values: Dict[str, float] = {}
        vol_values: Dict[str, float] = {}

        for sym in tradeable:
            col_data = price_data[sym].dropna()
            if len(col_data) < _MIN_HISTORY_BARS:
                continue

            z, rsi_val, rv = self._compute_indicators(col_data)

            if z is not None:
                zscores[sym] = z
            if rsi_val is not None:
                rsi_values[sym] = rsi_val
            if rv is not None:
                vol_values[sym] = rv

        # ----------------------------------------------------------------
        # Step 2: Volume spike filter
        # ----------------------------------------------------------------
        volume_ok: Dict[str, bool] = {}
        if volume_data is not None:
            for sym in tradeable:
                if sym not in volume_data.columns:
                    volume_ok[sym] = True   # pass if data unavailable
                    continue
                vol_series = volume_data[sym].dropna()
                if len(vol_series) < 21:
                    volume_ok[sym] = True
                    continue
                avg_vol = float(vol_series.iloc[-21:-1].mean())
                today_vol = float(vol_series.iloc[-1])
                if avg_vol <= 0:
                    volume_ok[sym] = True
                    continue
                volume_ok[sym] = (today_vol / avg_vol) >= _VOLUME_SPIKE_THRESHOLD
        else:
            # No volume data supplied — skip the filter
            for sym in tradeable:
                volume_ok[sym] = True

        # ----------------------------------------------------------------
        # Step 3: Generate EXIT signals
        #
        # Exit conditions (checked in priority order):
        #   a) CRISIS regime — close everything
        #   b) Time stop — holding_days elapsed
        #   c) Mean reversion complete — z-score crossed back through ±exit_z
        #   d) Hard stop — z-score hit ±stop_z (adverse move)
        # ----------------------------------------------------------------
        exit_signals = self._generate_exit_signals(zscores, regime_state)
        signals.extend(exit_signals)

        # ----------------------------------------------------------------
        # Step 4: Generate ENTRY signals (blocked during CRISIS)
        # ----------------------------------------------------------------
        if not regime_state.is_crisis:
            entry_signals = self._generate_entry_signals(
                zscores=zscores,
                rsi_values=rsi_values,
                vol_values=vol_values,
                volume_ok=volume_ok,
                regime_state=regime_state,
            )
            signals.extend(entry_signals)

        # ----------------------------------------------------------------
        # Step 5: Increment bar counter
        # ----------------------------------------------------------------
        self._bar_count += 1

        logger.info(
            f"MeanReversionStrategy.generate_signals: emitted {len(signals)} signals "
            f"({len(exit_signals)} exits, {len(signals) - len(exit_signals)} entries) "
            f"| regime={regime_state.regime.value} "
            f"| open_positions={len(self._positions)} "
            f"| bar={self._bar_count}"
        )

        return signals

    # ------------------------------------------------------------------
    # Indicator computation
    # ------------------------------------------------------------------

    def _compute_indicators(
        self,
        prices: pd.Series,
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Compute z-score, RSI, and realised volatility for one symbol.

        Parameters
        ----------
        prices:
            Clean (NaN-dropped) close price series, chronologically ordered.

        Returns
        -------
        z_score:
            5-day return z-scored against the trailing 60-day return
            distribution.  ``None`` if insufficient history.
        rsi:
            Latest 5-period RSI value.  ``None`` if insufficient history.
        realized_vol:
            Annualised 20-day realised volatility.  ``None`` if insufficient.
        """
        cfg = self._cfg
        lookback = 60         # z-score normalisation window
        rsi_period = cfg.rsi_period
        vol_lookback = cfg.vol_lookback

        # Need at least lookback + return_period bars
        if len(prices) < lookback + cfg.lookback_days + 1:
            return None, None, None

        # ------ 5-day arithmetic returns ------
        log_returns = np.log(prices / prices.shift(1)).dropna()

        # 5-day rolling sum (log return proxy for arithmetic return over 5 days)
        rolling_5d = log_returns.rolling(cfg.lookback_days, min_periods=cfg.lookback_days)
        returns_5d = rolling_5d.sum()

        # Drop NaN (first few bars)
        returns_5d_clean = returns_5d.dropna()
        if len(returns_5d_clean) < lookback:
            return None, None, None

        # Latest 5-day return (most recent bar)
        latest_5d = float(returns_5d_clean.iloc[-1])

        # Distribution of the last 60 5-day returns (includes today)
        dist_window = returns_5d_clean.iloc[-lookback:]
        mu = float(dist_window.mean())
        sigma = float(dist_window.std(ddof=1))

        if sigma < 1e-10:
            # Near-zero variance → stock hasn't moved at all
            z_score = 0.0
        else:
            z_score = (latest_5d - mu) / sigma

        # ------ RSI ------
        if len(prices) < rsi_period + 2:
            rsi_val = None
        else:
            rsi_series = _compute_rsi(prices, period=rsi_period)
            rsi_clean = rsi_series.dropna()
            rsi_val = float(rsi_clean.iloc[-1]) if len(rsi_clean) > 0 else None

        # ------ Realised vol (annualised) ------
        if len(log_returns) >= vol_lookback:
            rv_series = log_returns.rolling(vol_lookback, min_periods=vol_lookback).std()
            rv_clean = rv_series.dropna()
            realized_vol = float(rv_clean.iloc[-1]) * np.sqrt(252) if len(rv_clean) > 0 else None
        else:
            realized_vol = None

        return z_score, rsi_val, realized_vol

    # ------------------------------------------------------------------
    # Exit signal generation
    # ------------------------------------------------------------------

    def _generate_exit_signals(
        self,
        zscores: Dict[str, float],
        regime_state: RegimeState,
    ) -> List[Signal]:
        """Emit CLOSE signals for positions that have met an exit condition.

        Parameters
        ----------
        zscores:
            Latest z-score per symbol.
        regime_state:
            Current market regime.

        Returns
        -------
        List[Signal]:
            CLOSE signals with metadata describing the exit reason.
        """
        cfg = self._cfg
        exit_signals: List[Signal] = []
        symbols_to_close: List[str] = []

        for sym, pos in self._positions.items():
            bars_held = self._bar_count - pos["entry_bar"]
            z = zscores.get(sym)
            direction = pos["direction"]
            exit_reason = None

            # --- Priority 1: Crisis regime — close everything ---
            if regime_state.is_crisis:
                exit_reason = "crisis_regime"

            # --- Priority 2: Time stop ---
            elif bars_held >= cfg.holding_days:
                exit_reason = "time_stop"

            # --- Priority 3: Hard stop-loss ---
            elif z is not None:
                if direction == "long" and z <= -cfg.stop_z:
                    exit_reason = "stop_loss"
                elif direction == "short" and z >= cfg.stop_z:
                    exit_reason = "stop_loss"

                # --- Priority 4: Mean reversion complete ---
                elif direction == "long" and z >= -cfg.exit_z:
                    exit_reason = "mean_reversion_complete"
                elif direction == "short" and z <= cfg.exit_z:
                    exit_reason = "mean_reversion_complete"

            if exit_reason is not None:
                symbols_to_close.append(sym)
                strength = self._vol_scaled_strength(
                    abs(pos["entry_zscore"]) / max(cfg.entry_z, 1.0),
                    vol_target=cfg.vol_target,
                    realized_vol=pos.get("entry_vol", cfg.vol_target),
                )
                sig = Signal(
                    symbol=sym,
                    direction="close",
                    strength=strength,
                    strategy=self.STRATEGY_NAME,
                    metadata={
                        "exit_reason": exit_reason,
                        "bars_held": bars_held,
                        "current_z": float(z) if z is not None else None,
                        "entry_z": pos["entry_zscore"],
                        "entry_direction": direction,
                        "bar_count": self._bar_count,
                    },
                )
                exit_signals.append(sig)
                self._log.log_signal(
                    self.STRATEGY_NAME, sym, "CLOSE", strength,
                    {"exit_reason": exit_reason, "bars_held": bars_held, "z": z},
                )

        # Remove closed positions from internal state
        for sym in symbols_to_close:
            del self._positions[sym]

        return exit_signals

    # ------------------------------------------------------------------
    # Entry signal generation
    # ------------------------------------------------------------------

    def _generate_entry_signals(
        self,
        zscores: Dict[str, float],
        rsi_values: Dict[str, float],
        vol_values: Dict[str, float],
        volume_ok: Dict[str, bool],
        regime_state: RegimeState,
    ) -> List[Signal]:
        """Emit LONG/SHORT signals for stocks meeting all entry criteria.

        Entry criteria (all must be satisfied):
        1. |z-score| > ``entry_z``             (directional extreme)
        2. RSI < ``rsi_oversold``  (long) or RSI > ``rsi_overbought`` (short)
        3. Volume spike: today >= 1.2× 20-day average
        4. Not already holding a position in this symbol
        5. Direction is not blocked by regime overlay

        Parameters
        ----------
        zscores:
            z-score per eligible symbol.
        rsi_values:
            Latest 5-period RSI per symbol.
        vol_values:
            Annualised 20-day realised vol per symbol (used for sizing).
        volume_ok:
            Whether the volume spike filter passed per symbol.
        regime_state:
            Current market regime.

        Returns
        -------
        List[Signal]:
            LONG and SHORT entry signals.
        """
        cfg = self._cfg
        entry_signals: List[Signal] = []

        # In BEAR regime, block new short entries (downtrend may persist)
        block_shorts = (regime_state.regime == Regime.BEAR)

        for sym, z in zscores.items():
            # Skip if already in a position for this symbol
            if sym in self._positions:
                continue

            rsi = rsi_values.get(sym)
            rv = vol_values.get(sym, cfg.vol_target)
            vol_gate = volume_ok.get(sym, True)

            # Volume spike filter
            if not vol_gate:
                continue

            # RSI must be available for filtering
            if rsi is None:
                continue

            direction: Optional[str] = None

            # --- Oversold: buy candidate ---
            if z < -cfg.entry_z and rsi < cfg.rsi_oversold:
                direction = "long"

            # --- Overbought: sell candidate ---
            elif z > cfg.entry_z and rsi > cfg.rsi_overbought:
                if block_shorts:
                    logger.debug(
                        f"MeanReversionStrategy: short entry for {sym} blocked "
                        f"(BEAR regime)."
                    )
                    continue
                direction = "short"

            if direction is None:
                continue

            # --- Vol-scaled signal strength ---
            # |z| / entry_z gives a normalised "extremeness" score in [1, ...]
            # capped at stop_z for strength calculation purposes.
            extremeness = min(abs(z) / cfg.entry_z, cfg.stop_z / cfg.entry_z)
            extremeness_norm = (extremeness - 1.0) / max(
                (cfg.stop_z / cfg.entry_z) - 1.0, 1e-6
            )  # 0.0 at entry_z, 1.0 at stop_z
            extremeness_norm = float(np.clip(extremeness_norm, 0.0, 1.0))

            strength = self._vol_scaled_strength(
                extremeness_norm,
                vol_target=cfg.vol_target,
                realized_vol=rv if rv is not None else cfg.vol_target,
            )

            # Register position
            self._positions[sym] = {
                "entry_bar": self._bar_count,
                "entry_zscore": z,
                "direction": direction,
                "entry_vol": rv if rv is not None else cfg.vol_target,
            }

            sig = Signal(
                symbol=sym,
                direction=direction,
                strength=strength,
                strategy=self.STRATEGY_NAME,
                metadata={
                    "z_score": round(z, 4),
                    "rsi": round(rsi, 2) if rsi is not None else None,
                    "realized_vol": round(rv, 4) if rv is not None else None,
                    "volume_spike": vol_gate,
                    "entry_z_threshold": cfg.entry_z,
                    "rsi_threshold": cfg.rsi_oversold if direction == "long" else cfg.rsi_overbought,
                    "regime": regime_state.regime.value,
                    "bar_count": self._bar_count,
                },
            )
            entry_signals.append(sig)

            self._log.log_signal(
                self.STRATEGY_NAME,
                sym,
                "BUY" if direction == "long" else "SELL",
                strength,
                {
                    "z": round(z, 4),
                    "rsi": round(rsi, 2) if rsi is not None else None,
                    "rv": round(rv, 4) if rv is not None else None,
                    "regime": regime_state.regime.value,
                },
            )

        return entry_signals

    # ------------------------------------------------------------------
    # Vol-scaled sizing helper
    # ------------------------------------------------------------------

    @staticmethod
    def _vol_scaled_strength(
        base_strength: float,
        vol_target: float,
        realized_vol: float,
    ) -> float:
        """Compute volatility-scaled signal strength (Barroso approach).

        Scales ``base_strength`` by ``vol_target / realized_vol`` so that
        high-volatility stocks receive smaller position weights.

        Parameters
        ----------
        base_strength:
            Pre-scaling signal strength in [0, 1].
        vol_target:
            Annualised volatility target (e.g. 0.15 = 15%).
        realized_vol:
            Stock's annualised realised volatility.

        Returns
        -------
        float
            Vol-scaled strength in [0.1, 1.0].
        """
        vol_scale = vol_target / max(realized_vol, 0.01)
        vol_scale = float(np.clip(vol_scale, 0.1, 1.0))
        raw = base_strength * vol_scale
        return float(np.clip(raw, 0.1, 1.0))

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def open_positions(self) -> Dict[str, Dict]:
        """Return a shallow copy of the current open-position book.

        Returns
        -------
        dict
            Mapping symbol → position metadata.  Safe to inspect; mutating
            the returned dict does not affect internal state.
        """
        return dict(self._positions)

    @property
    def bar_count(self) -> int:
        """Number of bars processed since strategy instantiation."""
        return self._bar_count

    def reset(self) -> None:
        """Clear all internal state (positions and bar counter).

        Useful between walk-forward back-test windows so that stale
        positions from one window do not bleed into the next.
        """
        self._positions.clear()
        self._bar_count = 0
        logger.info("MeanReversionStrategy: state reset (positions cleared, bar_count=0).")

    def summary(self) -> Dict:
        """Return a summary of current strategy state for logging/diagnostics.

        Returns
        -------
        dict
            Keys: ``bar_count``, ``open_positions_count``, ``positions``.
        """
        return {
            "strategy": self.STRATEGY_NAME,
            "bar_count": self._bar_count,
            "open_positions_count": len(self._positions),
            "positions": {
                sym: {
                    "direction": pos["direction"],
                    "bars_held": self._bar_count - pos["entry_bar"],
                    "entry_zscore": round(pos["entry_zscore"], 4),
                }
                for sym, pos in self._positions.items()
            },
        }
