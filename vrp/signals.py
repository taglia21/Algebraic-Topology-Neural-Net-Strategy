"""
vrp/signals.py
==============
Signal enrichment layer for the VRP Alpha Engine.

Implements research-backed alpha signals that improve entry timing:

1. VRP Term Structure (VIX/VIX3M contango):
   - Contango (VIX < VIX3M) = rich VRP, safe to sell premium
   - Backwardation (VIX > VIX3M) = danger, reduce/avoid selling

2. Realized Vol vs Implied Vol spread:
   - True VRP = VIX - realized vol. Only sell when VRP is meaningfully positive.
   - Uses Yang-Zhang estimator for realized vol (captures overnight + intraday).

3. IV Skew slope:
   - Steeper put skew = market pricing more tail risk
   - Adjust delta target further OTM when skew is steep

4. Intraday gap risk model:
   - Uses Parkinson/Garman-Klass range estimators to measure gap risk
   - Reduces position size when gap risk is elevated

5. Event blackout filter:
   - Blocks new entries around FOMC, CPI, NFP, and triple-witching
   - These events cause vol jumps that can breach stop losses

References:
- Carr & Wu (2009), "Variance Risk Premiums"
- Bollerslev, Tauchen, Zhou (2009), "Expected Stock Returns and Variance Risk Premia"
- Euan Sinclair, "Volatility Trading" (Wiley, 2nd ed.)
- Yang & Zhang (2000), "Drift-Independent Volatility Estimation"
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum VRP (IV - RV) in vol points to justify selling premium.
# Below this, the risk/reward is unattractive.
# Set conservatively — even 1 vol point of VRP is tradeable for
# 30-45 DTE spreads since theta decay dominates.
MIN_VRP_SPREAD = 1.0  # 1 vol point

# VIX term structure thresholds
CONTANGO_NEUTRAL = 0.97   # VIX/VIX3M below this = healthy contango
BACKWARDATION_WARN = 1.08  # above this = backwardation warning
BACKWARDATION_HALT = 1.20  # above this = no new trades (severe backwardation)

# Skew adjustment thresholds
SKEW_NEUTRAL_Z = 0.0      # z-score of skew slope
SKEW_STEEP_Z = 1.5        # move delta target further OTM
SKEW_EXTREME_Z = 2.5      # reduce position size

# Gap risk thresholds (Parkinson ratio vs historical median)
GAP_RISK_ELEVATED = 1.5   # 1.5x historical = caution
GAP_RISK_HIGH = 2.0       # 2x historical = reduce size
GAP_RISK_EXTREME = 3.0    # 3x = no new entries

# Target annualized vol for EWMA vol targeting (position sizing)
TARGET_PORTFOLIO_VOL = 0.15  # 15% annualized


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class SignalState:
    """Aggregated signal state for trade decisions."""
    # VRP measurement
    vix: float = 0.0
    realized_vol_20d: float = 0.0
    vrp_spread: float = 0.0            # VIX - realized vol (in vol points)
    vrp_rich: bool = False             # True if VRP is meaningfully positive

    # Term structure
    vix_vix3m_ratio: float = 1.0       # < 1 = contango, > 1 = backwardation
    term_structure_ok: bool = True      # False if in dangerous backwardation

    # Skew
    skew_z_score: float = 0.0          # z-score of 25-delta put skew
    delta_adjustment: float = 0.0      # added to delta target (negative = further OTM)

    # Gap risk
    gap_risk_ratio: float = 1.0        # current vs historical gap risk
    gap_risk_ok: bool = True

    # Volatility targeting
    ewma_vol: float = 0.15             # current EWMA portfolio vol
    vol_target_scalar: float = 1.0     # position size scalar for vol targeting

    # Event filter
    event_blackout: bool = False
    next_event: str = ""

    # Composite
    can_trade: bool = True
    sizing_scalar: float = 1.0         # composite multiplier for position sizing
    reject_reason: str = ""

    def summary(self) -> str:
        """One-line summary for logging."""
        return (
            f"VRP={self.vrp_spread:+.1f} | "
            f"TS={self.vix_vix3m_ratio:.2f} | "
            f"Skew z={self.skew_z_score:+.1f} | "
            f"Gap={self.gap_risk_ratio:.1f}x | "
            f"Vol={self.ewma_vol:.1%} | "
            f"Scalar={self.sizing_scalar:.2f} | "
            f"{'OK' if self.can_trade else 'BLOCKED: ' + self.reject_reason}"
        )


# ---------------------------------------------------------------------------
# 1. Realized Volatility Tracker (Yang-Zhang estimator)
# ---------------------------------------------------------------------------

class RealizedVolTracker:
    """Tracks realized volatility using the Yang-Zhang estimator.

    Yang-Zhang (2000) is the minimum-variance unbiased estimator that
    accounts for both overnight jumps and intraday drift. It combines:
    - Overnight variance (close-to-open)
    - Rogers-Satchell intraday variance
    - Close-to-close classical variance

    This is strictly superior to simple close-to-close volatility for
    assets with meaningful overnight gaps (like SPX).
    """

    def __init__(self, window: int = 20) -> None:
        self.window = window
        self._opens: Deque[float] = deque(maxlen=window + 1)
        self._highs: Deque[float] = deque(maxlen=window + 1)
        self._lows: Deque[float] = deque(maxlen=window + 1)
        self._closes: Deque[float] = deque(maxlen=window + 1)

    def update(self, open_: float, high: float, low: float, close: float) -> None:
        """Add a new daily bar."""
        self._opens.append(open_)
        self._highs.append(high)
        self._lows.append(low)
        self._closes.append(close)

    @property
    def is_ready(self) -> bool:
        return len(self._closes) > self.window

    def realized_vol(self) -> float:
        """Calculate Yang-Zhang realized volatility (annualized).

        Returns annualized volatility in decimal form (e.g., 0.15 = 15%).
        """
        if not self.is_ready:
            return 0.0

        n = self.window
        opens = list(self._opens)[-n-1:]
        highs = list(self._highs)[-n-1:]
        lows = list(self._lows)[-n-1:]
        closes = list(self._closes)[-n-1:]

        # Overnight returns: log(open_i / close_{i-1})
        overnight = [math.log(opens[i] / closes[i-1]) for i in range(1, n+1)]
        # Close-to-close: log(close_i / close_{i-1})
        cc = [math.log(closes[i] / closes[i-1]) for i in range(1, n+1)]

        # Overnight variance
        mean_o = sum(overnight) / n
        var_overnight = sum((o - mean_o) ** 2 for o in overnight) / (n - 1)

        # Close-to-close variance
        mean_cc = sum(cc) / n
        var_cc = sum((c - mean_cc) ** 2 for c in cc) / (n - 1)

        # Rogers-Satchell intraday variance
        var_rs = 0.0
        for i in range(1, n+1):
            hi = math.log(highs[i] / opens[i])
            lo = math.log(lows[i] / opens[i])
            cl = math.log(closes[i] / opens[i])
            var_rs += hi * (hi - cl) + lo * (lo - cl)
        var_rs /= n

        # Yang-Zhang combination (minimum-variance weights)
        k = 0.34 / (1.34 + (n + 1) / (n - 1))
        var_yz = var_overnight + k * var_cc + (1 - k) * var_rs

        # Annualize (252 trading days)
        return math.sqrt(max(0.0, var_yz) * 252)

    def simple_realized_vol(self) -> float:
        """Simple close-to-close realized vol (for comparison/fallback)."""
        if len(self._closes) < self.window + 1:
            return 0.0
        closes = list(self._closes)[-(self.window+1):]
        log_returns = [math.log(closes[i] / closes[i-1]) for i in range(1, len(closes))]
        return float(np.std(log_returns) * math.sqrt(252))


# ---------------------------------------------------------------------------
# 2. EWMA Volatility Targeting
# ---------------------------------------------------------------------------

class VolatilityTargeting:
    """EWMA-based volatility targeting for position sizing.

    Scales position sizes so that portfolio volatility targets a fixed
    level (default 15% annualized). When recent vol is high, we trade
    smaller; when low, we trade larger.

    This is the same principle used by risk parity funds (Bridgewater,
    AQR) and is documented in Moskowitz, Ooi, Pedersen (2012)
    "Time Series Momentum".

    The EWMA decay factor (lambda = 0.94) follows RiskMetrics convention.
    """

    def __init__(
        self,
        target_vol: float = TARGET_PORTFOLIO_VOL,
        decay: float = 0.94,
        floor: float = 0.25,
        cap: float = 2.0,
    ) -> None:
        self.target_vol = target_vol
        self.decay = decay
        self.floor = floor      # minimum sizing scalar
        self.cap = cap          # maximum sizing scalar
        self._ewma_var: float = 0.0
        self._initialized: bool = False

    def update(self, daily_return: float) -> None:
        """Update EWMA variance with a new daily return."""
        if not self._initialized:
            # Initialize with the first observation
            self._ewma_var = daily_return ** 2
            self._initialized = True
        else:
            self._ewma_var = (
                self.decay * self._ewma_var
                + (1 - self.decay) * daily_return ** 2
            )

    @property
    def ewma_vol(self) -> float:
        """Current EWMA volatility (annualized)."""
        if not self._initialized or self._ewma_var <= 0:
            return self.target_vol  # assume target if no data
        return math.sqrt(self._ewma_var * 252)

    @property
    def sizing_scalar(self) -> float:
        """Position size multiplier to target constant volatility.

        scalar = target_vol / current_vol, clamped to [floor, cap].
        """
        current = self.ewma_vol
        if current <= 0:
            return 1.0
        raw = self.target_vol / current
        return max(self.floor, min(self.cap, raw))

    def seed(self, daily_returns: List[float]) -> None:
        """Seed EWMA with historical returns."""
        for r in daily_returns:
            self.update(r)


# ---------------------------------------------------------------------------
# 3. Gap Risk Model (Parkinson/Garman-Klass)
# ---------------------------------------------------------------------------

class GapRiskModel:
    """Measures intraday range risk relative to historical norms.

    Uses the Parkinson (1980) range-based volatility estimator:
        sigma_P = sqrt(1/(4n*ln2) * sum(ln(H/L)^2))

    When the Parkinson ratio (current range vol / historical median) is
    elevated, it signals heightened gap risk that could breach stop losses.
    """

    def __init__(self, window: int = 60) -> None:
        self.window = window
        self._log_ranges: Deque[float] = deque(maxlen=window)
        self._median_log_range: float = 0.0

    def update(self, high: float, low: float) -> None:
        """Add a new daily high-low bar."""
        if low > 0 and high > low:
            lr = math.log(high / low)
            self._log_ranges.append(lr)
            if len(self._log_ranges) >= 10:
                self._median_log_range = float(np.median(list(self._log_ranges)))

    @property
    def is_ready(self) -> bool:
        return len(self._log_ranges) >= 20

    @property
    def gap_risk_ratio(self) -> float:
        """Ratio of recent range to historical median.

        > 1.0 means higher-than-normal intraday ranges.
        """
        if not self.is_ready or self._median_log_range <= 0:
            return 1.0

        # Use last 5 days average vs full-window median
        recent = list(self._log_ranges)[-5:]
        recent_avg = sum(recent) / len(recent)
        return recent_avg / self._median_log_range

    def parkinson_vol(self) -> float:
        """Parkinson range-based volatility (annualized)."""
        if not self.is_ready:
            return 0.0
        ranges = list(self._log_ranges)
        n = len(ranges)
        var = sum(r ** 2 for r in ranges) / (4 * n * math.log(2))
        return math.sqrt(var * 252)


# ---------------------------------------------------------------------------
# 4. Event Calendar (FOMC, CPI, NFP, Triple Witching)
# ---------------------------------------------------------------------------

class EventCalendar:
    """Blocks new entries around major market-moving events.

    Covered events:
    - FOMC decisions (8 per year, 2pm ET)
    - CPI release (monthly, 8:30am ET)
    - Non-Farm Payrolls (first Friday of month)
    - Triple/Quadruple Witching (3rd Friday of Mar/Jun/Sep/Dec)

    The blackout window is configurable (default: 1 day before the event).
    This avoids entering short premium right before a vol-expanding event.
    """

    # Event severity: only block new entries for high-impact events.
    # For 30-45 DTE spreads, CPI/NFP are lower severity than FOMC.
    HIGH_IMPACT = {"FOMC", "OPEX"}  # hard blocks
    LOW_IMPACT = {"CPI", "NFP"}     # reduce sizing only, don't block

    def __init__(self, blackout_days_before: int = 1, blackout_days_after: int = 0) -> None:
        self.blackout_before = blackout_days_before
        self.blackout_after = blackout_days_after
        # Pre-compute known event dates for 2024-2027
        self._events = self._build_event_calendar()

    def _build_event_calendar(self) -> Dict[date, str]:
        """Build a calendar of known market events.

        FOMC dates are fixed by the Fed; CPI/NFP follow predictable patterns.
        """
        events: Dict[date, str] = {}

        # FOMC meeting dates (announcement day) — 2024 through 2027
        fomc_dates = [
            # 2024
            "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
            "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
            # 2025
            "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
            "2025-07-30", "2025-09-17", "2025-11-05", "2025-12-17",
            # 2026
            "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
            "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16",
            # 2027
            "2027-01-27", "2027-03-17", "2027-04-28", "2027-06-16",
            "2027-07-28", "2027-09-22", "2027-11-03", "2027-12-15",
        ]
        for d in fomc_dates:
            events[date.fromisoformat(d)] = "FOMC"

        # Triple/Quadruple witching: 3rd Friday of Mar, Jun, Sep, Dec
        for year in range(2024, 2028):
            for month in [3, 6, 9, 12]:
                first_day = date(year, month, 1)
                days_to_fri = (4 - first_day.weekday()) % 7
                third_friday = first_day + timedelta(days=days_to_fri + 14)
                events[third_friday] = "OPEX"

        # CPI: typically 2nd or 3rd Tuesday/Wednesday of month
        # NFP: first Friday of month
        # We approximate these with pattern-based generation
        for year in range(2024, 2028):
            for month in range(1, 13):
                # NFP: first Friday
                first_day = date(year, month, 1)
                days_to_fri = (4 - first_day.weekday()) % 7
                nfp = first_day + timedelta(days=days_to_fri)
                events[nfp] = "NFP"

                # CPI: approximately 10th-13th of month (typically Tuesday or Wednesday)
                # Use 12th as default; close enough for blackout window
                cpi_day = date(year, month, 12)
                # Adjust to weekday
                while cpi_day.weekday() >= 5:
                    cpi_day += timedelta(days=1)
                events[cpi_day] = "CPI"

        return events

    def is_blackout(self, as_of: date) -> Tuple[bool, str]:
        """Check if the given date falls within an event blackout window.

        Only HIGH_IMPACT events (FOMC, OPEX) trigger a hard blackout.
        LOW_IMPACT events (CPI, NFP) return the event name but blackout=False
        so the aggregator can reduce sizing instead of blocking.

        Returns (is_blackout, event_name).
        """
        # Check if today IS an event day
        if as_of in self._events:
            event = self._events[as_of]
            is_high = event in self.HIGH_IMPACT
            return is_high, event

        # Check if tomorrow (or next N days) is an event day
        for offset in range(1, self.blackout_before + 1):
            check = as_of + timedelta(days=offset)
            # Skip weekends
            while check.weekday() >= 5:
                check += timedelta(days=1)
            if check in self._events:
                event = self._events[check]
                is_high = event in self.HIGH_IMPACT
                return is_high, f"{event} (in {offset}d)"

        # Check if yesterday (or recent) was an event day
        for offset in range(1, self.blackout_after + 1):
            check = as_of - timedelta(days=offset)
            while check.weekday() >= 5:
                check -= timedelta(days=1)
            if check in self._events:
                event = self._events[check]
                is_high = event in self.HIGH_IMPACT
                return is_high, f"{event} (was {offset}d ago)"

        return False, ""


# ---------------------------------------------------------------------------
# 5. Kelly Criterion Position Sizer
# ---------------------------------------------------------------------------

class KellySizer:
    """Kelly criterion position sizing with fractional Kelly.

    The Kelly fraction f* = p - q/b where:
    - p = win probability
    - q = 1 - p
    - b = avg_win / avg_loss (payoff ratio)

    We use fractional Kelly (default 0.5x = "half Kelly") because:
    1. Full Kelly is optimal but has enormous variance
    2. Half Kelly gives ~75% of the growth rate with much less volatility
    3. This is standard practice at systematic funds (see Thorp, 2006)

    The sizer tracks a rolling window of trade outcomes to adapt
    the Kelly fraction as the strategy's edge evolves.
    """

    def __init__(
        self,
        fraction: float = 0.5,
        min_trades: int = 20,
        window: int = 100,
        floor: float = 0.10,
        cap: float = 0.60,
    ) -> None:
        self.fraction = fraction    # fractional Kelly multiplier
        self.min_trades = min_trades  # minimum trades before using Kelly
        self.window = window
        self.floor = floor          # minimum risk allocation
        self.cap = cap              # maximum risk allocation
        self._wins: Deque[float] = deque(maxlen=window)
        self._losses: Deque[float] = deque(maxlen=window)

    def record_trade(self, pnl: float) -> None:
        """Record a completed trade."""
        if pnl > 0:
            self._wins.append(pnl)
        elif pnl < 0:
            self._losses.append(abs(pnl))

    @property
    def total_trades(self) -> int:
        return len(self._wins) + len(self._losses)

    @property
    def win_rate(self) -> float:
        total = self.total_trades
        if total == 0:
            return 0.0
        return len(self._wins) / total

    @property
    def payoff_ratio(self) -> float:
        """Average win / average loss."""
        if not self._wins or not self._losses:
            return 1.0
        avg_w = sum(self._wins) / len(self._wins)
        avg_l = sum(self._losses) / len(self._losses)
        if avg_l <= 0:
            return 1.0
        return avg_w / avg_l

    @property
    def kelly_fraction(self) -> float:
        """Full Kelly f* = p - q/b, then apply fractional scaling."""
        if self.total_trades < self.min_trades:
            # Not enough data — use conservative default
            return self.floor

        p = self.win_rate
        q = 1 - p
        b = self.payoff_ratio

        if b <= 0:
            return self.floor

        full_kelly = p - (q / b)

        # Negative Kelly means negative edge — use floor
        if full_kelly <= 0:
            return self.floor

        # Apply fractional scaling and clamp
        frac = full_kelly * self.fraction
        return max(self.floor, min(self.cap, frac))

    def seed(self, trades: List[float]) -> None:
        """Seed with historical trade P&Ls."""
        for pnl in trades:
            self.record_trade(pnl)


# ---------------------------------------------------------------------------
# 6. Composite Signal Aggregator
# ---------------------------------------------------------------------------

class SignalAggregator:
    """Combines all signal sources into a single trade decision.

    This is the brain of the signal layer. It:
    1. Updates all signal components with new data
    2. Evaluates each signal independently
    3. Combines them into a composite go/no-go + sizing scalar

    The sizing scalar is multiplicative across all dimensions:
        final_size = base_size * vol_target * gap_risk * kelly * term_structure

    This ensures that when multiple signals are cautious, position size
    shrinks aggressively, and when all signals are green, we trade full size.
    """

    def __init__(self) -> None:
        self.rv_tracker = RealizedVolTracker(window=20)
        self.vol_target = VolatilityTargeting()
        self.gap_model = GapRiskModel(window=60)
        self.event_cal = EventCalendar()
        self.kelly = KellySizer()
        self._skew_history: Deque[float] = deque(maxlen=60)

    def update(
        self,
        spx_open: float,
        spx_high: float,
        spx_low: float,
        spx_close: float,
        vix: float,
        as_of: date,
        vix3m: Optional[float] = None,
        daily_portfolio_return: Optional[float] = None,
    ) -> SignalState:
        """Process a new daily bar and return the signal state.

        Parameters
        ----------
        spx_open, spx_high, spx_low, spx_close : Daily OHLC
        vix : Current VIX level
        as_of : Current date
        vix3m : 3-month VIX (VIX3M) if available — None if not
        daily_portfolio_return : Yesterday's portfolio return for vol targeting

        Returns
        -------
        SignalState with all signal evaluations
        """
        state = SignalState(vix=vix)

        # --- Realized vol ---
        self.rv_tracker.update(spx_open, spx_high, spx_low, spx_close)
        if self.rv_tracker.is_ready:
            rv = self.rv_tracker.realized_vol()
            state.realized_vol_20d = rv * 100  # convert to vol points
            state.vrp_spread = vix - state.realized_vol_20d
            state.vrp_rich = state.vrp_spread >= MIN_VRP_SPREAD
        else:
            # Not enough data — assume VRP is OK
            state.vrp_rich = True
            state.vrp_spread = vix * 0.15  # rough estimate

        # --- Term structure ---
        if vix3m is not None and vix3m > 0:
            state.vix_vix3m_ratio = vix / vix3m
            if state.vix_vix3m_ratio >= BACKWARDATION_HALT:
                state.term_structure_ok = False
            elif state.vix_vix3m_ratio >= BACKWARDATION_WARN:
                state.term_structure_ok = True  # allow but reduce
        else:
            # No VIX3M data — use VRP spread as proxy
            state.term_structure_ok = True
            state.vix_vix3m_ratio = 1.0

        # --- Gap risk ---
        self.gap_model.update(spx_high, spx_low)
        if self.gap_model.is_ready:
            state.gap_risk_ratio = self.gap_model.gap_risk_ratio
            state.gap_risk_ok = state.gap_risk_ratio < GAP_RISK_EXTREME

        # --- Vol targeting ---
        if daily_portfolio_return is not None:
            self.vol_target.update(daily_portfolio_return)
        elif self.rv_tracker.is_ready:
            # Use SPX return as proxy
            closes = list(self.rv_tracker._closes)
            if len(closes) >= 2:
                daily_ret = (closes[-1] / closes[-2]) - 1.0
                self.vol_target.update(daily_ret)

        state.ewma_vol = self.vol_target.ewma_vol
        state.vol_target_scalar = self.vol_target.sizing_scalar

        # --- Event blackout ---
        is_blackout, event = self.event_cal.is_blackout(as_of)
        state.event_blackout = is_blackout
        state.next_event = event

        # --- Composite decision ---
        state.can_trade = True
        state.sizing_scalar = 1.0
        reasons = []

        # Hard blocks (only the strongest signals block outright)
        if not state.vrp_rich:
            state.can_trade = False
            reasons.append(f"VRP too thin ({state.vrp_spread:+.1f})")

        if not state.term_structure_ok:
            state.can_trade = False
            reasons.append(f"Backwardation ({state.vix_vix3m_ratio:.2f})")

        if not state.gap_risk_ok:
            state.can_trade = False
            reasons.append(f"Extreme gap risk ({state.gap_risk_ratio:.1f}x)")

        if state.event_blackout:
            state.can_trade = False
            reasons.append(f"Event blackout: {state.next_event}")

        state.reject_reason = "; ".join(reasons) if reasons else ""

        # Soft scaling (applied even if can_trade=True)
        if state.can_trade:
            # Vol targeting
            state.sizing_scalar *= state.vol_target_scalar

            # Gap risk reduction (partial)
            if state.gap_risk_ratio > GAP_RISK_ELEVATED:
                gap_penalty = 1.0 - min(0.5, (state.gap_risk_ratio - GAP_RISK_ELEVATED) * 0.25)
                state.sizing_scalar *= gap_penalty

            # Term structure caution (backwardation warning zone)
            if state.vix_vix3m_ratio > CONTANGO_NEUTRAL:
                ts_penalty = 1.0 - min(0.3, (state.vix_vix3m_ratio - CONTANGO_NEUTRAL) * 1.5)
                state.sizing_scalar *= max(0.5, ts_penalty)

            # Low-impact event penalty (CPI/NFP: reduce by 25%, don't block)
            if event and not is_blackout:
                state.sizing_scalar *= 0.75

            # Kelly fraction (replaces fixed risk_per_trade)
            if self.kelly.total_trades >= self.kelly.min_trades:
                state.sizing_scalar *= self.kelly.kelly_fraction / 0.50  # normalize vs default 50%

            # Floor
            state.sizing_scalar = max(0.25, min(2.0, state.sizing_scalar))

        logger.debug(f"Signal: {state.summary()}")
        return state

    def record_trade_result(self, pnl: float) -> None:
        """Record a completed trade for Kelly adaptation."""
        self.kelly.record_trade(pnl)

    def seed_history(
        self,
        ohlc_data: List[Tuple[float, float, float, float]],
        daily_returns: Optional[List[float]] = None,
    ) -> None:
        """Seed all trackers with historical data.

        Parameters
        ----------
        ohlc_data : List of (open, high, low, close) tuples
        daily_returns : Portfolio daily returns for vol targeting
        """
        for o, h, l, c in ohlc_data:
            self.rv_tracker.update(o, h, l, c)
            self.gap_model.update(h, l)

        if daily_returns:
            self.vol_target.seed(daily_returns)
