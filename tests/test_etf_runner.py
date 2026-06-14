"""
tests/test_etf_runner.py
========================
Unit tests for the market-hours-aware live runner (etf/runner.py).

All tests are offline and deterministic. The calendar adapter is exercised both
through ``exchange_calendars`` (when installed) and via a tiny weekday-only stub
so the pure scheduling logic is verified independent of the calendar backend.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from etf.runner import (
    MarketCalendar,
    RunDecision,
    ScheduleState,
    decide_action,
    load_schedule_state,
    save_schedule_state,
)

try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover
    _ET = timezone(timedelta(hours=-5))


class _WeekdayCalendar:
    """Minimal calendar stub: Mon-Fri sessions, 16:00-ET close, no holidays.

    Lets us test the pure decision logic deterministically without depending on
    a specific holiday schedule.
    """

    def is_session(self, d: date) -> bool:
        return d.weekday() < 5

    def session_close_et(self, d: date):
        if not self.is_session(d):
            return None
        return datetime(d.year, d.month, d.day, 16, 0, tzinfo=_ET)

    def trading_days_between(self, start: date, end: date) -> int:
        if end <= start:
            return 0
        n, cur = 0, start + timedelta(days=1)
        while cur <= end:
            if cur.weekday() < 5:
                n += 1
            cur += timedelta(days=1)
        return n


@pytest.fixture
def cal():
    return _WeekdayCalendar()


def _et(y, m, d, h, mi=0):
    return datetime(y, m, d, h, mi, tzinfo=_ET)


# ---------------------------------------------------------------------------
# Execution-window gating
# ---------------------------------------------------------------------------
def test_trades_inside_window_when_cadence_elapsed(cal):
    # Wednesday 15:45 ET, 15 min to a 16:00 close -> inside a 30-min window.
    now = _et(2026, 6, 17, 15, 45)
    d = decide_action(now, cal, ScheduleState(last_rebalance_date=None),
                       cadence_days=21, window_minutes=30)
    assert d.should_trade
    assert d.is_trading_day and d.in_execution_window and d.cadence_elapsed
    assert d.minutes_to_close == pytest.approx(15.0, abs=1e-6)


def test_no_trade_before_window_opens(cal):
    # 10:00 ET -> 360 min to close, outside a 30-min window.
    now = _et(2026, 6, 17, 10, 0)
    d = decide_action(now, cal, None, cadence_days=21, window_minutes=30)
    assert not d.should_trade
    assert d.is_trading_day and not d.in_execution_window
    # Should sleep toward the window, but bounded by overnight cap.
    assert d.sleep_seconds > 0


def test_no_trade_after_close(cal):
    now = _et(2026, 6, 17, 16, 30)  # 30 min AFTER close
    d = decide_action(now, cal, None, cadence_days=21, window_minutes=30)
    assert not d.should_trade
    assert d.minutes_to_close is not None and d.minutes_to_close < 0


def test_no_trade_on_weekend(cal):
    now = _et(2026, 6, 20, 15, 45)  # Saturday
    d = decide_action(now, cal, None, cadence_days=21, window_minutes=30)
    assert not d.should_trade
    assert not d.is_trading_day
    assert d.minutes_to_close is None


# ---------------------------------------------------------------------------
# Cadence gating
# ---------------------------------------------------------------------------
def test_first_deployment_trades_immediately(cal):
    now = _et(2026, 6, 17, 15, 45)
    d = decide_action(now, cal, ScheduleState(last_rebalance_date=None),
                      cadence_days=21, window_minutes=30)
    assert d.cadence_elapsed and d.should_trade


def test_cadence_blocks_when_too_soon(cal):
    now = _et(2026, 6, 17, 15, 45)  # Wednesday
    # Rebalanced 5 calendar days ago (3 trading days) -> below a 21-day cadence.
    sched = ScheduleState(last_rebalance_date="2026-06-12")
    d = decide_action(now, cal, sched, cadence_days=21, window_minutes=30)
    assert not d.cadence_elapsed
    assert not d.should_trade


def test_cadence_elapsed_after_enough_sessions(cal):
    now = _et(2026, 6, 17, 15, 45)
    # ~7 weeks earlier -> well over 21 trading days.
    sched = ScheduleState(last_rebalance_date="2026-04-20")
    d = decide_action(now, cal, sched, cadence_days=21, window_minutes=30)
    assert d.cadence_elapsed and d.should_trade


def test_force_bypasses_cadence_but_not_market(cal):
    # Too soon by cadence, but --force => trade (in window).
    now = _et(2026, 6, 17, 15, 45)
    sched = ScheduleState(last_rebalance_date="2026-06-16")
    d = decide_action(now, cal, sched, cadence_days=21, window_minutes=30, force=True)
    assert d.should_trade
    # ...but force must NOT override a closed market.
    weekend = _et(2026, 6, 20, 15, 45)
    d2 = decide_action(weekend, cal, sched, cadence_days=21, window_minutes=30, force=True)
    assert not d2.should_trade


def test_anytime_window_allows_midday(cal):
    # A full-session window (1440 min) lets a midday session time trade.
    now = _et(2026, 6, 17, 11, 0)
    d = decide_action(now, cal, None, cadence_days=21, window_minutes=24 * 60)
    assert d.in_execution_window and d.should_trade


# ---------------------------------------------------------------------------
# Schedule-state persistence
# ---------------------------------------------------------------------------
def test_schedule_state_roundtrip(tmp_path):
    path = tmp_path / "sched.json"
    save_schedule_state(ScheduleState(last_rebalance_date="2026-06-17"), path)
    loaded = load_schedule_state(path)
    assert loaded is not None
    assert loaded.last_rebalance_date == "2026-06-17"


def test_schedule_state_missing_returns_none(tmp_path):
    assert load_schedule_state(tmp_path / "nope.json") is None


def test_schedule_state_corrupt_returns_none(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{ not json")
    assert load_schedule_state(p) is None


def test_save_schedule_failsafe_on_bad_path():
    save_schedule_state(ScheduleState(last_rebalance_date="2026-06-17"),
                        "/proc/cannot/write/here/sched.json")


# ---------------------------------------------------------------------------
# Real calendar adapter (exchange_calendars when available)
# ---------------------------------------------------------------------------
def test_real_calendar_knows_july_4_holiday():
    cal = MarketCalendar()
    # 2025-07-04 (Independence Day) is a market holiday; 2025-07-07 (Mon) is open.
    # The weekday fallback would (wrongly) call the Friday a session, so this also
    # documents the degraded-mode caveat. Only assert when the real calendar is
    # active to keep the test deterministic across environments.
    if cal._cal is not None:
        assert not cal.is_session(date(2025, 7, 4))
    assert cal.is_session(date(2025, 7, 7))
