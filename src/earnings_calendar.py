"""
Earnings Calendar System
=========================

Multi-source earnings date fetching with SQLite caching.

Sources (in priority order):
  1. yfinance earnings_dates (most reliable, free)
  2. Alpaca corporate actions API (if available)
  3. Manual overrides from JSON file

Key features:
  - SQLite cache with 24h TTL to avoid rate limits
  - Batch fetching for universe of symbols
  - Pre-computed "days to earnings" for fast lookups
  - Thread-safe for concurrent access

Usage:
    from src.earnings_calendar import EarningsCalendar

    ec = EarningsCalendar()
    upcoming = ec.get_earnings_this_week(["AAPL", "MSFT", "GOOGL"])
    days = ec.get_days_to_earnings("AAPL")
    if ec.is_earnings_week("AAPL"):
        print("AAPL reports this week!")
"""

import json
import logging
import os
import sqlite3
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# ETF COMPONENT MAPPING (top holdings for options-relevant ETFs)
# ============================================================================

# Major SPY/QQQ/IWM component stocks whose earnings affect ETF IV
ETF_COMPONENTS: Dict[str, List[str]] = {
    "SPY": [
        "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK.B",
        "UNH", "JNJ", "JPM", "V", "XOM", "PG", "MA", "HD", "AVGO", "CVX",
        "MRK", "ABBV", "LLY", "PEP", "KO", "COST", "ADBE", "WMT", "CRM",
        "TMO", "CSCO", "MCD", "ACN", "ABT", "DHR", "NFLX", "AMD", "TXN",
        "LIN", "NEE", "PM", "UPS",
    ],
    "QQQ": [
        "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "AVGO",
        "ADBE", "COST", "CSCO", "AMD", "NFLX", "PEP", "INTC", "TXN",
        "INTU", "QCOM", "AMGN", "AMAT", "ISRG", "BKNG", "SBUX", "MDLZ",
        "ADI", "GILD", "LRCX", "REGN", "PYPL", "SNPS",
    ],
    "IWM": [
        # IWM is small-cap; individual earnings less impactful on ETF
        # but track the larger small-caps
        "PLUG", "APPS", "UPST", "BILL", "CROX", "AMC", "GME",
    ],
}


# ============================================================================
# EARNINGS ENTRY
# ============================================================================

class EarningsEntry:
    """Single earnings event for a symbol."""

    __slots__ = ("symbol", "earnings_date", "time_of_day", "source", "fetched_at")

    def __init__(
        self,
        symbol: str,
        earnings_date: date,
        time_of_day: str = "unknown",  # "BMO", "AMC", "unknown"
        source: str = "yfinance",
        fetched_at: Optional[datetime] = None,
    ):
        self.symbol = symbol
        self.earnings_date = earnings_date
        self.time_of_day = time_of_day
        self.source = source
        self.fetched_at = fetched_at or datetime.now()

    @property
    def days_until(self) -> int:
        """Days until this earnings event (negative = past)."""
        return (self.earnings_date - date.today()).days

    def __repr__(self) -> str:
        return (
            f"EarningsEntry({self.symbol}, {self.earnings_date}, "
            f"{self.time_of_day}, days={self.days_until})"
        )


# ============================================================================
# EARNINGS CALENDAR
# ============================================================================

class EarningsCalendar:
    """
    Multi-source earnings calendar with SQLite caching.

    Fetches and caches earnings dates for any symbol. Provides fast lookups
    for days-to-earnings, blackout periods, and weekly earnings lists.
    """

    # Cache TTL: 24 hours (earnings dates change rarely)
    CACHE_TTL_HOURS = 24

    # How far ahead to look for "upcoming" earnings
    LOOKAHEAD_DAYS = 30

    def __init__(
        self,
        db_path: str = "data/earnings_cache.db",
        overrides_path: str = "data/earnings_overrides.json",
    ):
        self._db_path = db_path
        self._overrides_path = overrides_path
        self._lock = threading.Lock()

        # Ensure data directory exists
        Path(os.path.dirname(db_path)).mkdir(parents=True, exist_ok=True)

        # Initialize SQLite
        self._init_db()

        # Load manual overrides if present
        self._overrides: Dict[str, str] = {}  # symbol -> "YYYY-MM-DD"
        self._load_overrides()

        logger.info(
            f"EarningsCalendar initialized (db={db_path}, "
            f"{len(self._overrides)} manual overrides)"
        )

    # ================================================================== #
    # PUBLIC API
    # ================================================================== #

    def get_days_to_earnings(self, symbol: str) -> Optional[int]:
        """
        Get the number of calendar days until the next earnings for a symbol.

        Returns:
            Positive int if earnings upcoming, 0 if today, negative if past
            (most recent), None if unknown.
        """
        entry = self._get_next_earnings(symbol)
        if entry is None:
            return None
        return entry.days_until

    def is_earnings_week(self, symbol: str) -> bool:
        """
        Check if a symbol reports earnings within the next 7 calendar days.
        """
        days = self.get_days_to_earnings(symbol)
        if days is None:
            return False
        return 0 <= days <= 7

    def is_in_blackout(self, symbol: str, blackout_days: int = 2) -> bool:
        """
        Check if a symbol is within the earnings blackout window.

        Blackout = earnings within next `blackout_days` calendar days.
        This means: no new positions should be opened.
        """
        days = self.get_days_to_earnings(symbol)
        if days is None:
            return False
        return 0 <= days <= blackout_days

    def is_pre_earnings_window(
        self, symbol: str, min_days: int = 3, max_days: int = 7
    ) -> bool:
        """
        Check if a symbol is in the "pre-earnings" premium selling window.

        This is the sweet spot: IV is elevated but there's still enough time
        for the trade to work before the event.

        Args:
            min_days: Minimum days before earnings (avoid blackout)
            max_days: Maximum days before earnings
        """
        days = self.get_days_to_earnings(symbol)
        if days is None:
            return False
        return min_days <= days <= max_days

    def is_post_earnings(self, symbol: str, within_days: int = 3) -> bool:
        """
        Check if a symbol reported earnings within the last N days.

        Post-earnings = IV crush occurred, possible mean reversion opportunity.
        """
        entry = self._get_most_recent_earnings(symbol)
        if entry is None:
            return False
        days_since = -entry.days_until  # Negative days_until = past
        return 0 <= days_since <= within_days

    def get_earnings_this_week(
        self, symbols: List[str]
    ) -> List[EarningsEntry]:
        """
        Get all symbols from the list that report earnings this week.

        Args:
            symbols: List of tickers to check

        Returns:
            List of EarningsEntry for symbols reporting this week
        """
        results = []
        for sym in symbols:
            entry = self._get_next_earnings(sym)
            if entry is not None and 0 <= entry.days_until <= 7:
                results.append(entry)
        results.sort(key=lambda e: e.earnings_date)
        return results

    def get_earnings_in_range(
        self, symbols: List[str], days_ahead: int = 14
    ) -> List[EarningsEntry]:
        """
        Get all upcoming earnings within N days for a list of symbols.
        """
        results = []
        for sym in symbols:
            entry = self._get_next_earnings(sym)
            if entry is not None and 0 <= entry.days_until <= days_ahead:
                results.append(entry)
        results.sort(key=lambda e: e.earnings_date)
        return results

    def get_etf_earnings_exposure(self, etf: str) -> Dict[str, int]:
        """
        For an ETF (SPY, QQQ, IWM), get the component stocks that report
        earnings within the next 7 days and how many days until each.

        This is critical for ETF options: if 5+ major components report
        in the same week, ETF IV will be elevated.

        Returns:
            Dict of {symbol: days_until_earnings} for upcoming reporters
        """
        components = ETF_COMPONENTS.get(etf.upper(), [])
        exposure = {}
        for sym in components:
            days = self.get_days_to_earnings(sym)
            if days is not None and 0 <= days <= 7:
                exposure[sym] = days
        return exposure

    def refresh(self, symbols: List[str], force: bool = False) -> int:
        """
        Refresh earnings data for a list of symbols.

        Args:
            symbols: Tickers to refresh
            force: If True, ignore cache TTL

        Returns:
            Number of symbols successfully refreshed
        """
        refreshed = 0
        for sym in symbols:
            try:
                if force or self._is_stale(sym):
                    entries = self._fetch_earnings(sym)
                    if entries:
                        self._cache_entries(sym, entries)
                        refreshed += 1
                    time.sleep(0.3)  # Rate limit protection
            except Exception as e:
                logger.warning(f"Earnings refresh failed for {sym}: {e}")
        logger.info(f"Refreshed earnings for {refreshed}/{len(symbols)} symbols")
        return refreshed

    # ================================================================== #
    # PRIVATE: Data fetching
    # ================================================================== #

    def _get_next_earnings(self, symbol: str) -> Optional[EarningsEntry]:
        """Get the next upcoming earnings for a symbol (cached, then fetch)."""
        # Check manual overrides first
        if symbol in self._overrides:
            try:
                override_date = date.fromisoformat(self._overrides[symbol])
                if override_date >= date.today():
                    return EarningsEntry(
                        symbol=symbol,
                        earnings_date=override_date,
                        source="manual_override",
                    )
            except ValueError:
                pass

        # Check cache
        cached = self._get_cached_next(symbol)
        if cached is not None:
            return cached

        # Fetch and cache
        if self._is_stale(symbol):
            entries = self._fetch_earnings(symbol)
            if entries:
                self._cache_entries(symbol, entries)
                # Return the first future entry
                today = date.today()
                for e in entries:
                    if e.earnings_date >= today:
                        return e

        return None

    def _get_most_recent_earnings(self, symbol: str) -> Optional[EarningsEntry]:
        """Get the most recent past earnings for a symbol."""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            try:
                cursor = conn.execute(
                    """
                    SELECT earnings_date, time_of_day, source
                    FROM earnings
                    WHERE symbol = ? AND earnings_date < ?
                    ORDER BY earnings_date DESC
                    LIMIT 1
                    """,
                    (symbol.upper(), date.today().isoformat()),
                )
                row = cursor.fetchone()
                if row:
                    return EarningsEntry(
                        symbol=symbol,
                        earnings_date=date.fromisoformat(row[0]),
                        time_of_day=row[1],
                        source=row[2],
                    )
            finally:
                conn.close()
        return None

    def _fetch_earnings(self, symbol: str) -> List[EarningsEntry]:
        """
        Fetch earnings dates from multiple sources.

        Priority:
          1. yfinance earnings_dates
          2. yfinance Ticker.calendar
          3. Fallback: empty list
        """
        entries = self._fetch_from_yfinance(symbol)
        if entries:
            return entries

        entries = self._fetch_from_yfinance_calendar(symbol)
        if entries:
            return entries

        return []

    def _fetch_from_yfinance(self, symbol: str) -> List[EarningsEntry]:
        """Fetch earnings dates via yfinance earnings_dates property."""
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)

            # earnings_dates returns a DataFrame indexed by date
            try:
                ed = ticker.earnings_dates
            except Exception:
                ed = None

            if ed is None or ed.empty:
                return []

            entries = []
            for idx in ed.index:
                try:
                    # Index is a Timestamp
                    earnings_dt = idx.date() if hasattr(idx, "date") else date.fromisoformat(str(idx)[:10])

                    # Try to determine BMO/AMC from the time
                    tod = "unknown"
                    if hasattr(idx, "hour"):
                        if idx.hour < 10:
                            tod = "BMO"
                        elif idx.hour >= 16:
                            tod = "AMC"

                    entries.append(
                        EarningsEntry(
                            symbol=symbol,
                            earnings_date=earnings_dt,
                            time_of_day=tod,
                            source="yfinance",
                        )
                    )
                except Exception:
                    continue

            logger.debug(f"yfinance: {len(entries)} earnings dates for {symbol}")
            return entries

        except Exception as e:
            logger.debug(f"yfinance earnings_dates failed for {symbol}: {e}")
            return []

    def _fetch_from_yfinance_calendar(self, symbol: str) -> List[EarningsEntry]:
        """Fetch next earnings from yfinance Ticker.calendar."""
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            cal = ticker.calendar

            if cal is None:
                return []

            # calendar can be a dict or DataFrame depending on yfinance version
            earnings_date_val = None
            if isinstance(cal, dict):
                earnings_date_val = cal.get("Earnings Date")
                if isinstance(earnings_date_val, list) and earnings_date_val:
                    earnings_date_val = earnings_date_val[0]
            else:
                # DataFrame format
                try:
                    if "Earnings Date" in cal.columns:
                        earnings_date_val = cal["Earnings Date"].iloc[0]
                    elif "Earnings Date" in cal.index:
                        earnings_date_val = cal.loc["Earnings Date"].iloc[0]
                except Exception:
                    pass

            if earnings_date_val is None:
                return []

            # Parse to date
            if hasattr(earnings_date_val, "date"):
                ed = earnings_date_val.date()
            else:
                ed = date.fromisoformat(str(earnings_date_val)[:10])

            return [
                EarningsEntry(
                    symbol=symbol,
                    earnings_date=ed,
                    source="yfinance_calendar",
                )
            ]

        except Exception as e:
            logger.debug(f"yfinance calendar failed for {symbol}: {e}")
            return []

    # ================================================================== #
    # PRIVATE: SQLite cache
    # ================================================================== #

    def _init_db(self):
        """Create SQLite tables if they don't exist."""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS earnings (
                        symbol TEXT NOT NULL,
                        earnings_date TEXT NOT NULL,
                        time_of_day TEXT DEFAULT 'unknown',
                        source TEXT DEFAULT 'yfinance',
                        fetched_at TEXT NOT NULL,
                        PRIMARY KEY (symbol, earnings_date)
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS fetch_log (
                        symbol TEXT PRIMARY KEY,
                        last_fetched TEXT NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_earnings_symbol
                    ON earnings(symbol)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_earnings_date
                    ON earnings(earnings_date)
                """)
                conn.commit()
            finally:
                conn.close()

    def _is_stale(self, symbol: str) -> bool:
        """Check if cached data is older than CACHE_TTL_HOURS."""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            try:
                cursor = conn.execute(
                    "SELECT last_fetched FROM fetch_log WHERE symbol = ?",
                    (symbol.upper(),),
                )
                row = cursor.fetchone()
                if row is None:
                    return True
                last_fetched = datetime.fromisoformat(row[0])
                age_hours = (datetime.now() - last_fetched).total_seconds() / 3600
                return age_hours > self.CACHE_TTL_HOURS
            finally:
                conn.close()

    def _get_cached_next(self, symbol: str) -> Optional[EarningsEntry]:
        """Get the next future earnings date from cache."""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            try:
                cursor = conn.execute(
                    """
                    SELECT earnings_date, time_of_day, source
                    FROM earnings
                    WHERE symbol = ? AND earnings_date >= ?
                    ORDER BY earnings_date ASC
                    LIMIT 1
                    """,
                    (symbol.upper(), date.today().isoformat()),
                )
                row = cursor.fetchone()
                if row:
                    return EarningsEntry(
                        symbol=symbol,
                        earnings_date=date.fromisoformat(row[0]),
                        time_of_day=row[1],
                        source=row[2],
                    )
            finally:
                conn.close()
        return None

    def _cache_entries(self, symbol: str, entries: List[EarningsEntry]):
        """Store earnings entries in SQLite cache."""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            try:
                now = datetime.now().isoformat()
                for entry in entries:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO earnings
                            (symbol, earnings_date, time_of_day, source, fetched_at)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            symbol.upper(),
                            entry.earnings_date.isoformat(),
                            entry.time_of_day,
                            entry.source,
                            now,
                        ),
                    )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO fetch_log (symbol, last_fetched)
                    VALUES (?, ?)
                    """,
                    (symbol.upper(), now),
                )
                conn.commit()
            finally:
                conn.close()

    # ================================================================== #
    # PRIVATE: Manual overrides
    # ================================================================== #

    def _load_overrides(self):
        """Load manual earnings date overrides from JSON file."""
        if os.path.exists(self._overrides_path):
            try:
                with open(self._overrides_path) as f:
                    self._overrides = json.load(f)
                logger.info(
                    f"Loaded {len(self._overrides)} earnings overrides "
                    f"from {self._overrides_path}"
                )
            except Exception as e:
                logger.warning(f"Failed to load earnings overrides: {e}")


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    ec = EarningsCalendar()
    print("EarningsCalendar initialized OK")

    # Test a few symbols
    for sym in ["AAPL", "MSFT", "NVDA", "SPY"]:
        days = ec.get_days_to_earnings(sym)
        in_week = ec.is_earnings_week(sym)
        blackout = ec.is_in_blackout(sym)
        pre_earn = ec.is_pre_earnings_window(sym)
        print(
            f"  {sym}: days_to_earnings={days}, "
            f"earnings_week={in_week}, blackout={blackout}, "
            f"pre_earnings={pre_earn}"
        )

    # Test ETF exposure
    for etf in ["SPY", "QQQ"]:
        exposure = ec.get_etf_earnings_exposure(etf)
        if exposure:
            print(f"  {etf} components reporting this week: {exposure}")
        else:
            print(f"  {etf}: no major components reporting this week")
