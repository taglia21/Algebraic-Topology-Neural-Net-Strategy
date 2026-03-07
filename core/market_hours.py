"""
core/market_hours.py
====================
Market hours awareness and US equity holiday calendar.

Provides deterministic answers to:
- Is the US equity market open right now?
- When does it next open/close?
- Is today a trading day?

Uses exchange-calendars for authoritative NYSE schedule data, with a
lightweight built-in fallback if the package is unavailable.

Usage
-----
    from core.market_hours import MarketCalendar

    cal = MarketCalendar()
    if cal.is_market_open():
        print("Market is open — run trading cycle.")
    else:
        next_open = cal.next_open()
        print(f"Market closed. Next open: {next_open}")
"""

from __future__ import annotations

import logging
from datetime import datetime, time, timedelta, timezone
from typing import Optional

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

_ET_TZ = ZoneInfo("America/New_York")

# NYSE regular trading hours (Eastern Time)
_MARKET_OPEN = time(9, 30)
_MARKET_CLOSE = time(16, 0)

# NYSE observed holidays for 2024-2027
# (New Year's Day, MLK Day, Presidents' Day, Good Friday,
#  Memorial Day, Juneteenth, Independence Day, Labor Day,
#  Thanksgiving, Christmas)
_HOLIDAYS: set = {
    # 2024
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29",
    "2024-05-27", "2024-06-19", "2024-07-04", "2024-09-02",
    "2024-11-28", "2024-12-25",
    # 2025
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18",
    "2025-05-26", "2025-06-19", "2025-07-04", "2025-09-01",
    "2025-11-27", "2025-12-25",
    # 2026
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03",
    "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07",
    "2026-11-26", "2026-12-25",
    # 2027
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26",
    "2027-05-31", "2027-06-18", "2027-07-05", "2027-09-06",
    "2027-11-25", "2027-12-24",
}

# Early-close dates (1:00 PM ET): day before Independence Day, day after
# Thanksgiving, Christmas Eve (when not weekend)
_EARLY_CLOSE_DATES: set = {
    "2024-07-03", "2024-11-29", "2024-12-24",
    "2025-07-03", "2025-11-28", "2025-12-24",
    "2026-07-02", "2026-11-27", "2026-12-24",
    "2027-07-02", "2027-11-26", "2027-12-23",
}


def _now_et() -> datetime:
    """Current datetime in US Eastern Time (proper timezone-aware)."""
    return datetime.now(_ET_TZ)


class MarketCalendar:
    """NYSE market hours and holiday calendar.

    Uses exchange-calendars if available for authoritative data;
    otherwise falls back to the built-in holiday table.

    Parameters
    ----------
    exchange :
        Exchange code for exchange-calendars. Default ``"XNYS"`` (NYSE).
    """

    def __init__(self, exchange: str = "XNYS") -> None:
        self._xcal = None
        try:
            import exchange_calendars as xcals
            self._xcal = xcals.get_calendar(exchange)
            logger.info(f"MarketCalendar: using exchange-calendars for {exchange}")
        except ImportError:
            logger.info("MarketCalendar: exchange-calendars not installed; using built-in fallback")
        except Exception as exc:
            logger.warning(f"MarketCalendar: exchange-calendars init failed: {exc}; using fallback")

    def is_trading_day(self, dt: Optional[datetime] = None) -> bool:
        """Check if a given date is a regular NYSE trading day.

        Parameters
        ----------
        dt :
            Datetime to check. Defaults to now (Eastern Time).

        Returns
        -------
        bool
        """
        dt = dt or _now_et()
        date_str = dt.strftime("%Y-%m-%d")

        # Weekend check
        if dt.weekday() >= 5:  # Saturday=5, Sunday=6
            return False

        if self._xcal is not None:
            try:
                import pandas as pd
                ts = pd.Timestamp(date_str)
                return self._xcal.is_session(ts)
            except Exception:
                pass

        # Fallback: check holiday table
        return date_str not in _HOLIDAYS

    def is_market_open(self, dt: Optional[datetime] = None) -> bool:
        """Check if the NYSE is currently in regular trading hours.

        Parameters
        ----------
        dt :
            Datetime to check. Defaults to now (Eastern Time).

        Returns
        -------
        bool
        """
        dt = dt or _now_et()
        if not self.is_trading_day(dt):
            return False

        current_time = dt.time()
        close_time = self._close_time(dt)
        return _MARKET_OPEN <= current_time < close_time

    def minutes_until_close(self, dt: Optional[datetime] = None) -> float:
        """Minutes remaining until market close.

        Parameters
        ----------
        dt :
            Current datetime. Defaults to now (Eastern Time).

        Returns
        -------
        float
            Minutes until close (0.0 if market is closed).
        """
        dt = dt or _now_et()
        if not self.is_market_open(dt):
            return 0.0

        close_time = self._close_time(dt)
        close_dt = dt.replace(
            hour=close_time.hour, minute=close_time.minute, second=0, microsecond=0
        )
        delta = (close_dt - dt).total_seconds() / 60.0
        return max(delta, 0.0)

    def next_open(self, dt: Optional[datetime] = None) -> datetime:
        """Return the next market open datetime (Eastern Time).

        Parameters
        ----------
        dt :
            Reference datetime. Defaults to now.

        Returns
        -------
        datetime
        """
        dt = dt or _now_et()
        # If market is currently open, return current open time
        if self.is_market_open(dt):
            return dt.replace(
                hour=_MARKET_OPEN.hour, minute=_MARKET_OPEN.minute,
                second=0, microsecond=0,
            )

        # Search forward up to 10 days
        candidate = dt + timedelta(days=1)
        for _ in range(10):
            candidate = candidate.replace(
                hour=_MARKET_OPEN.hour, minute=_MARKET_OPEN.minute,
                second=0, microsecond=0,
            )
            if self.is_trading_day(candidate):
                return candidate
            candidate += timedelta(days=1)

        # Fallback: next weekday 9:30
        return candidate

    def _close_time(self, dt: datetime) -> time:
        """Return market close time for the given date (handles early closes)."""
        date_str = dt.strftime("%Y-%m-%d")
        if date_str in _EARLY_CLOSE_DATES:
            return time(13, 0)  # 1:00 PM ET
        return _MARKET_CLOSE

    def sleep_until_open(self) -> None:
        """Block until the market opens. Logs a message and sleeps."""
        import time as _time

        if self.is_market_open():
            return

        next_open_dt = self.next_open()
        now = _now_et()
        wait_seconds = max((next_open_dt - now).total_seconds(), 0)
        logger.info(
            f"MarketCalendar: market closed. Sleeping {wait_seconds / 60:.0f} "
            f"minutes until {next_open_dt.strftime('%Y-%m-%d %H:%M ET')}"
        )
        if wait_seconds > 0:
            _time.sleep(wait_seconds)
