"""
etf/runner.py
=============
Market-hours-aware scheduling for the ETF engine's live/paper loop (Phase 5).

The validated book rebalances on a low-frequency cadence (monthly by default),
so the live runner does NOT trade continuously. Its job is to wake up, decide
whether *today* is a rebalance day that falls inside a tradable execution window,
run exactly one rebalance cycle when so, and otherwise sleep — safely, across
restarts, and without ever trading when the market is closed.

Design
------
- **Pure decision core.** :func:`decide_action` takes the current time, a
  calendar adapter, and the persisted schedule state and returns a
  :class:`RunDecision`. It performs no I/O, so the gating logic (trading-day +
  execution-window + cadence) is fully unit-testable offline.
- **Calendar adapter.** :class:`MarketCalendar` wraps ``exchange_calendars``
  (XNYS) for accurate holiday/half-day handling, with a dependency-free
  weekday + 16:00-ET fallback so the runner still functions if the calendar
  package is unavailable.
- **Persisted cadence.** The last-rebalance date is stored in a small JSON file
  (atomic write) so a daily-triggered run only rebalances once per cadence even
  across process restarts.

This module is intentionally free of any IBKR/broker dependency; the async loop
in :mod:`etf.main` wires the decision to the existing fail-safe trade path.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("etf.runner")

try:  # zoneinfo is stdlib on 3.9+
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover - extremely unlikely
    _ET = timezone(timedelta(hours=-5))


# ---------------------------------------------------------------------------
# Calendar adapter
# ---------------------------------------------------------------------------
class MarketCalendar:
    """US equity (XNYS) trading-session adapter.

    Prefers ``exchange_calendars`` for correct holidays and early closes; falls
    back to a weekday + fixed 16:00-ET close if the package is missing. The
    fallback is conservative (it can only be *more* restrictive on real
    sessions, never trade on a weekend) and is logged once so the operator knows
    holiday precision is degraded.
    """

    def __init__(self, name: str = "XNYS") -> None:
        self._cal = None
        try:
            import exchange_calendars as xcals

            self._cal = xcals.get_calendar(name)
            logger.info("MarketCalendar using exchange_calendars[%s]", name)
        except Exception as exc:  # pragma: no cover - exercised via fallback tests
            logger.warning(
                "exchange_calendars unavailable (%s); using weekday fallback "
                "(holiday precision degraded).", exc,
            )

    def is_session(self, d: date) -> bool:
        """True if ``d`` is a trading session."""
        if self._cal is not None:
            try:
                import pandas as pd

                return bool(self._cal.is_session(pd.Timestamp(d)))
            except Exception:  # pragma: no cover - defensive
                pass
        return d.weekday() < 5  # Mon-Fri fallback

    def session_close_et(self, d: date) -> Optional[datetime]:
        """Timezone-aware (ET) close datetime for session ``d``, or None if not
        a session."""
        if not self.is_session(d):
            return None
        if self._cal is not None:
            try:
                import pandas as pd

                close_utc = self._cal.session_close(pd.Timestamp(d))
                # exchange_calendars returns a tz-aware (UTC) or tz-naive ts.
                ts = pd.Timestamp(close_utc)
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                return ts.tz_convert(_ET).to_pydatetime()
            except Exception:  # pragma: no cover - defensive
                pass
        # Fallback: regular session closes 16:00 ET.
        return datetime(d.year, d.month, d.day, 16, 0, tzinfo=_ET)

    def trading_days_between(self, start: date, end: date) -> int:
        """Number of trading sessions in the half-open interval ``(start, end]``.

        Used to measure cadence elapsed since the last rebalance. ``start`` is
        exclusive (the last rebalance day itself does not count); ``end`` is
        inclusive (today counts if it is a session).
        """
        if end <= start:
            return 0
        if self._cal is not None:
            try:
                import pandas as pd

                sessions = self._cal.sessions_in_range(
                    pd.Timestamp(start), pd.Timestamp(end)
                )
                # sessions_in_range is inclusive of both ends; drop ``start``.
                return int(sum(1 for s in sessions if s.date() > start))
            except Exception:  # pragma: no cover - defensive
                pass
        # Fallback: count weekdays in (start, end].
        n = 0
        cur = start + timedelta(days=1)
        while cur <= end:
            if cur.weekday() < 5:
                n += 1
            cur += timedelta(days=1)
        return n


# ---------------------------------------------------------------------------
# Persisted schedule state
# ---------------------------------------------------------------------------
@dataclass
class ScheduleState:
    """Persisted cadence memory for the runner.

    Attributes
    ----------
    last_rebalance_date :
        ISO date (YYYY-MM-DD) of the most recent *successful* rebalance, or None
        if the engine has never rebalanced (first deployment -> trade now).
    updated_at :
        ISO-8601 UTC timestamp of the last write.
    """

    last_rebalance_date: Optional[str] = None
    updated_at: str = ""


def load_schedule_state(path: str | Path) -> Optional[ScheduleState]:
    """Load persisted schedule state, or None if absent/unreadable (fail-safe)."""
    try:
        p = Path(path)
        if not p.exists():
            return None
        data = json.loads(p.read_text())
        return ScheduleState(
            last_rebalance_date=(data.get("last_rebalance_date") or None),
            updated_at=str(data.get("updated_at", "")),
        )
    except Exception as exc:
        logger.error("Failed to load schedule state from %s: %s", path, exc)
        return None


def save_schedule_state(
    state: ScheduleState, path: str | Path, *, now: Optional[datetime] = None
) -> None:
    """Persist schedule state atomically. Best-effort: never raises."""
    try:
        now = now or datetime.now(timezone.utc)
        state.updated_at = now.isoformat()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(asdict(state), fh, indent=2)
            os.replace(tmp, p)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    except Exception as exc:  # pragma: no cover - state I/O must never crash trading
        logger.error("Failed to save schedule state to %s: %s", path, exc)


# ---------------------------------------------------------------------------
# Pure decision core
# ---------------------------------------------------------------------------
@dataclass
class RunDecision:
    """Outcome of the per-wake scheduling decision."""

    should_trade: bool
    is_trading_day: bool
    in_execution_window: bool
    cadence_elapsed: bool
    minutes_to_close: Optional[float]
    sleep_seconds: int
    reasons: List[str] = field(default_factory=list)


def _parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def decide_action(
    now_et: datetime,
    cal: MarketCalendar,
    sched: Optional[ScheduleState],
    *,
    cadence_days: int,
    window_minutes: int = 30,
    force: bool = False,
    poll_seconds: int = 300,
    overnight_seconds: int = 3600,
) -> RunDecision:
    """Decide whether to run a rebalance cycle now.

    A cycle runs only when ALL hold:
      1. today is a trading session,
      2. the current time is inside the execution window (the last
         ``window_minutes`` before the close), and
      3. at least ``cadence_days`` trading sessions have elapsed since the last
         successful rebalance (or there has never been one).

    ``force`` bypasses the cadence gate (operator-initiated immediate rebalance)
    but NOT the market-open / execution-window gates — we never submit when the
    market is closed.

    Returns a :class:`RunDecision` including a recommended ``sleep_seconds`` so
    the loop backs off appropriately (short polls inside the window, long sleeps
    overnight / on non-trading days).
    """
    today = now_et.date()
    reasons: List[str] = []

    is_session = cal.is_session(today)
    close = cal.session_close_et(today) if is_session else None
    mins_to_close: Optional[float] = None
    in_window = False
    if close is not None:
        mins_to_close = (close - now_et).total_seconds() / 60.0
        in_window = 0.0 <= mins_to_close <= float(window_minutes)

    last_rb = _parse_date(sched.last_rebalance_date) if sched else None
    if last_rb is None:
        cadence_elapsed = True
        reasons.append("no prior rebalance on record (first deployment)")
    else:
        elapsed = cal.trading_days_between(last_rb, today)
        cadence_elapsed = elapsed >= cadence_days
        reasons.append(
            f"{elapsed} trading day(s) since last rebalance "
            f"({'>=' if cadence_elapsed else '<'} cadence {cadence_days})"
        )

    if not is_session:
        reasons.append("not a trading session today")
    elif mins_to_close is not None and mins_to_close < 0:
        reasons.append("market already closed for today")
    elif not in_window:
        reasons.append(
            f"outside execution window ({mins_to_close:.0f} min to close, "
            f"window is last {window_minutes} min)"
        )

    cadence_ok = cadence_elapsed or force
    should_trade = bool(is_session and in_window and cadence_ok)
    if force and not cadence_elapsed:
        reasons.append("cadence bypassed by --force")

    # --- Back-off schedule -------------------------------------------------
    if should_trade:
        sleep_seconds = poll_seconds
    elif not is_session or (mins_to_close is not None and mins_to_close < 0):
        # Non-trading day or after the close: sleep long (until next session-ish).
        sleep_seconds = overnight_seconds
    elif in_window:
        sleep_seconds = poll_seconds
    elif mins_to_close is not None and mins_to_close > window_minutes:
        # Before the window opens: sleep until ~window start, but poll at least
        # hourly so we never overshoot a short half-day session.
        until_window = (mins_to_close - window_minutes) * 60.0
        sleep_seconds = int(max(poll_seconds, min(overnight_seconds, until_window)))
    else:
        sleep_seconds = poll_seconds

    return RunDecision(
        should_trade=should_trade,
        is_trading_day=is_session,
        in_execution_window=in_window,
        cadence_elapsed=cadence_elapsed,
        minutes_to_close=mins_to_close,
        sleep_seconds=int(sleep_seconds),
        reasons=reasons,
    )


def now_et() -> datetime:
    """Current time in US/Eastern (the trading-clock reference)."""
    return datetime.now(_ET)
