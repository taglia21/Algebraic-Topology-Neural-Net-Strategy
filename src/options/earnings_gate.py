"""
Earnings Gate
==============

Prevents selling premium into earnings announcements where IV crush
and gap risk can cause catastrophic losses on short options.

Maintains approximate earnings dates for the options universe and
exports a simple blocking function used by the signal filter.
"""

from datetime import date, timedelta
from typing import Dict, Optional

# ============================================================================
# APPROXIMATE Q1-Q2 2026 EARNINGS DATES
# ============================================================================
# Update these quarterly.  Dates are best-guess based on historical cadence.
# The gate only needs to be roughly correct — blocking a few extra days is
# far cheaper than eating an earnings gap.

EARNINGS_CALENDAR: Dict[str, list[date]] = {
    "AAPL":  [date(2026, 1, 29), date(2026, 4, 30)],
    "MSFT":  [date(2026, 1, 27), date(2026, 4, 28)],
    "NVDA":  [date(2026, 2, 26), date(2026, 5, 28)],
    "AMZN":  [date(2026, 2, 5),  date(2026, 4, 30)],
    "META":  [date(2026, 1, 29), date(2026, 4, 29)],
    "GOOGL": [date(2026, 2, 4),  date(2026, 4, 29)],
    "SPY":   [],  # ETF — no earnings
    "QQQ":   [],  # ETF — no earnings
}

# How many days BEFORE earnings to block new positions
EARNINGS_BLACKOUT_DAYS = 3


def should_block_for_earnings(symbol: str, dte: int) -> bool:
    """
    Return True if *symbol* has an earnings announcement within *dte* days
    (i.e. the option would still be open when earnings hit).

    Parameters
    ----------
    symbol : str
        Underlying ticker (e.g. "AAPL").
    dte : int
        Days-to-expiration of the option being considered.

    Returns
    -------
    bool
        True  → block the trade (earnings inside the option window).
        False → safe to proceed.
    """
    dates = EARNINGS_CALENDAR.get(symbol.upper())
    if not dates:
        return False  # unknown symbol or ETF — allow

    today = date.today()
    option_expiry = today + timedelta(days=dte)

    for earn_date in dates:
        # Block if earnings falls between (today - blackout) and option expiry
        block_start = earn_date - timedelta(days=EARNINGS_BLACKOUT_DAYS)
        if block_start <= option_expiry and earn_date >= today:
            return True

    return False


def next_earnings_date(symbol: str) -> Optional[date]:
    """Return the next upcoming earnings date for *symbol*, or None."""
    dates = EARNINGS_CALENDAR.get(symbol.upper())
    if not dates:
        return None
    today = date.today()
    future = [d for d in dates if d >= today]
    return min(future) if future else None
