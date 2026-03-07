"""
data/data_manager.py
====================
Unified data abstraction layer for the ATNN Quant Powerhouse.

:class:`DataManager` is the **single point of contact** for all data needs
in the system.  Strategies, risk managers, and the ML pipeline all call this
class; they never interact with a :class:`~data.market_data.DataProvider`
directly.

Key responsibilities
--------------------
- Route data requests to the right provider (Alpaca primary / yfinance fallback).
- Cache results in-memory with TTL-based stale detection.
- Validate incoming data (NaN, negative prices, zero volume, out-of-hours ts).
- Handle missing data: fill-forward for up to 5 bars, then emit a warning.
- Serve data in **both** backtest mode (bar-by-bar iterator) and live mode
  (real-time API calls).  The same code path is used in both modes.

Usage — live / paper mode
--------------------------
    from data.data_manager import DataManager
    from datetime import datetime, timezone

    dm = DataManager()

    # Historical bars (UTC-aware MultiIndex DataFrame)
    bars = dm.get_historical_bars(
        ["AAPL", "MSFT"],
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 12, 31, tzinfo=timezone.utc),
    )

    # Latest N bars per symbol
    latest = dm.get_latest_bars(["AAPL"], limit=5)

    # Real-time snapshot
    snap = dm.get_market_snapshot(["AAPL", "MSFT"])

Usage — backtest mode
----------------------
    dm = DataManager()
    dm.load_backtest_data(
        symbols=["AAPL", "MSFT"],
        start=datetime(2022, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 12, 31, tzinfo=timezone.utc),
    )
    for bar_dt, bar_df in dm.iter_bars():
        # bar_df is a single-timestamp slice of the MultiIndex DataFrame
        process(bar_df)
"""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from typing import Dict, Generator, Iterator, List, Optional, Tuple

import pandas as pd

from core.config import get_config
from core.logger import get_trade_logger
from data.cache import DataCache, TTL_BARS
from data.market_data import (
    AlpacaDataProvider,
    DataProvider,
    YFinanceDataProvider,
    _OHLCV_COLS,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_FFILL_BARS: int = 5          # Fill-forward limit before issuing warning
_MARKET_OPEN_HOUR: int = 9        # 09:30 ET
_MARKET_OPEN_MINUTE: int = 30
_MARKET_CLOSE_HOUR: int = 16      # 16:00 ET
_ET_OFFSET_HOURS: int = -5        # Standard EST offset from UTC (approx)


# ---------------------------------------------------------------------------
# DataManager
# ---------------------------------------------------------------------------

class DataManager:
    """Unified data abstraction layer.

    Parameters
    ----------
    provider:
        Explicit :class:`~data.market_data.DataProvider` instance.  If not
        supplied, the manager auto-selects based on ``core.config``:
        ``AlpacaDataProvider`` when credentials are configured, otherwise
        ``YFinanceDataProvider`` (with a warning).
    cache:
        Explicit :class:`~data.cache.DataCache` instance.  Defaults to a
        fresh cache with ``max_entries=10_000``.
    mode:
        Override the system mode (``"backtest"`` / ``"paper"`` / ``"live"``).
        Defaults to ``get_config().system.mode``.

    Attributes
    ----------
    mode : str
        The operating mode (read-only after construction).

    Examples
    --------
    >>> dm = DataManager()
    >>> bars = dm.get_historical_bars(["AAPL"], start=..., end=...)
    """

    def __init__(
        self,
        provider: Optional[DataProvider] = None,
        cache: Optional[DataCache] = None,
        mode: Optional[str] = None,
    ) -> None:
        self._cfg = get_config()
        self._logger = get_trade_logger()

        self.mode: str = mode or self._cfg.system.mode

        # ------------------------------------------------------------------
        # Provider selection
        # ------------------------------------------------------------------
        if provider is not None:
            self._provider = provider
            self._fallback_provider: Optional[DataProvider] = None
        else:
            self._provider, self._fallback_provider = self._build_providers()

        # ------------------------------------------------------------------
        # Cache
        # ------------------------------------------------------------------
        self._cache = cache or DataCache(max_entries=10_000)

        # ------------------------------------------------------------------
        # Backtest state
        # ------------------------------------------------------------------
        self._backtest_data: Optional[pd.DataFrame] = None
        # Sorted list of unique datetimes in the backtest dataset
        self._backtest_datetimes: List[datetime] = []
        self._backtest_cursor: int = 0

        self._logger.log_info(
            "DataManager initialised",
            metadata={
                "mode": self.mode,
                "provider": type(self._provider).__name__,
                "fallback": type(self._fallback_provider).__name__
                if self._fallback_provider
                else None,
            },
        )

    # ------------------------------------------------------------------
    # Provider construction
    # ------------------------------------------------------------------

    def _build_providers(
        self,
    ) -> Tuple[DataProvider, Optional[DataProvider]]:
        """Select primary and fallback providers based on config.

        Returns
        -------
        (primary, fallback)
        """
        cfg_provider = self._cfg.data.provider.lower()
        alpaca_cfg = self._cfg.alpaca

        # Prefer Alpaca if credentials are available or if explicitly configured
        use_alpaca = (
            cfg_provider == "alpaca" or alpaca_cfg.is_configured()
        )

        if use_alpaca:
            try:
                primary: DataProvider = AlpacaDataProvider(
                    api_key=alpaca_cfg.api_key or None,
                    secret_key=alpaca_cfg.secret_key or None,
                    base_url=alpaca_cfg.base_url,
                )
                fallback: Optional[DataProvider] = (
                    YFinanceDataProvider()
                    if self._cfg.data.allow_yfinance_fallback
                    else None
                )
                return primary, fallback
            except Exception as exc:
                self._logger.log_error(
                    f"Failed to initialise AlpacaDataProvider: {exc}; "
                    "falling back to YFinance.",
                    exc_info=exc,
                )

        # Fall through to yfinance
        self._logger.log_info(
            "DataManager: using YFinanceDataProvider (FALLBACK)",
            metadata={"reason": "Alpaca not configured or failed"},
        )
        warnings.warn(
            "DataManager: AlpacaDataProvider unavailable — using yfinance "
            "as primary provider.  This is NOT suitable for live trading.",
            UserWarning,
            stacklevel=3,
        )
        return YFinanceDataProvider(), None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_historical_bars(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: str = "1D",
    ) -> pd.DataFrame:
        """Fetch historical OHLCV bars for multiple symbols.

        Results are cached with a 60-second TTL.  On a cache miss, the
        primary provider is queried; on failure the fallback provider
        (yfinance) is tried.

        Missing data handling:
        - Up to 5 consecutive NaN / missing rows are forward-filled per
          symbol.
        - If gaps exceed 5 bars a warning is emitted (but the data is still
          returned).

        Parameters
        ----------
        symbols:
            List of ticker symbols, e.g. ``["AAPL", "MSFT"]``.
        start:
            Inclusive start of the date range (UTC-aware).
        end:
            Inclusive end of the date range (UTC-aware).
        timeframe:
            Canonical timeframe string: ``"1D"``, ``"1Hour"``, ``"15Min"``.

        Returns
        -------
        pd.DataFrame
            MultiIndex ``(datetime, symbol)`` × columns
            ``[open, high, low, close, volume]``, sorted ascending.

        Raises
        ------
        RuntimeError
            If both the primary and fallback providers fail.
        """
        start = _ensure_utc(start)
        end = _ensure_utc(end)

        cache_key = _bar_cache_key(symbols, start, end, timeframe)
        cached = self._cache.get_bars(cache_key)
        if cached is not None:
            return cached

        df = self._fetch_bars_with_fallback(symbols, start, end, timeframe)
        df = self._validate_and_clean(df, symbols, timeframe)

        self._cache.set_bars(cache_key, df)
        return df

    def get_latest_bars(
        self,
        symbols: List[str],
        timeframe: str = "1D",
        limit: int = 1,
    ) -> pd.DataFrame:
        """Fetch the most recent *limit* bars per symbol.

        In backtest mode this reads from the loaded historical dataset (up to
        the current cursor position) rather than issuing a live API call.

        Parameters
        ----------
        symbols:
            Tickers to fetch.
        timeframe:
            Canonical timeframe string.
        limit:
            Number of bars per symbol.

        Returns
        -------
        pd.DataFrame
            MultiIndex ``(datetime, symbol)`` × ``[open, high, low, close,
            volume]``.
        """
        if self.mode == "backtest":
            return self._get_latest_bars_backtest(symbols, limit)

        cache_key = f"latest:{'_'.join(sorted(symbols))}:{timeframe}:{limit}"
        cached = self._cache.get_bars(cache_key)
        if cached is not None:
            return cached

        try:
            df = self._provider.get_latest(symbols, timeframe=timeframe, limit=limit)
        except Exception as exc:
            self._logger.log_error(
                f"Primary provider get_latest failed: {exc}; trying fallback",
                exc_info=exc,
            )
            df = self._try_fallback_latest(symbols, timeframe, limit)

        df = self._validate_and_clean(df, symbols, timeframe)
        self._cache.set(cache_key, df, TTL_BARS)
        return df

    def get_market_snapshot(self, symbols: List[str]) -> Dict[str, dict]:
        """Return the latest quote snapshot for each symbol.

        In backtest mode a synthetic snapshot is built from the most recent
        bar (mid = close; bid = close × 0.9995; ask = close × 1.0005).

        Parameters
        ----------
        symbols:
            Tickers to snapshot.

        Returns
        -------
        dict
            ``{symbol: {price, volume, bid, ask, timestamp}}``
        """
        if self.mode == "backtest":
            return self._get_snapshot_backtest(symbols)

        # Live/paper — check quote cache first (1-second TTL)
        missing = [s for s in symbols if self._cache.get_quote(s) is None]
        result: Dict[str, dict] = {
            s: self._cache.get_quote(s)
            for s in symbols
            if self._cache.get_quote(s) is not None
        }

        if missing:
            if hasattr(self._provider, "get_snapshot"):
                fresh = self._provider.get_snapshot(missing)  # type: ignore[attr-defined]
            else:
                fresh = self._synthetic_snapshot_from_latest(missing)
            for sym, snap in fresh.items():
                self._cache.set_quote(sym, snap)
                result[sym] = snap

        return result

    # ------------------------------------------------------------------
    # Backtest iterator interface
    # ------------------------------------------------------------------

    def load_backtest_data(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: str = "1D",
    ) -> None:
        """Pre-load the full historical dataset for backtesting.

        Must be called before :meth:`iter_bars` when in backtest mode.
        This downloads all data into memory so the bar-by-bar iteration
        is entirely in-process with no network calls.

        Parameters
        ----------
        symbols:
            Tickers to load.
        start:
            Start of the backtest window (UTC-aware).
        end:
            End of the backtest window (UTC-aware).
        timeframe:
            Canonical timeframe string.
        """
        self._logger.log_info(
            "DataManager: loading backtest data",
            metadata={
                "symbols": symbols,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "timeframe": timeframe,
            },
        )
        df = self.get_historical_bars(symbols, start=start, end=end, timeframe=timeframe)
        self._backtest_data = df
        self._backtest_datetimes = sorted(
            df.index.get_level_values("datetime").unique().tolist()
        )
        self._backtest_cursor = 0
        self._logger.log_info(
            "DataManager: backtest data loaded",
            metadata={
                "total_bars": len(self._backtest_datetimes),
                "symbols": symbols,
            },
        )

    def iter_bars(
        self,
    ) -> Generator[Tuple[datetime, pd.DataFrame], None, None]:
        """Yield ``(bar_datetime, bar_df)`` tuples, one timestamp at a time.

        Each yielded ``bar_df`` is a slice of the backtest dataset for the
        given timestamp, with index ``(datetime, symbol)`` and columns
        ``[open, high, low, close, volume]``.

        Yields
        ------
        (bar_datetime, bar_df) :
            The timestamp and a single-timestamp slice of the MultiIndex
            DataFrame.

        Raises
        ------
        RuntimeError
            If :meth:`load_backtest_data` has not been called first.
        """
        if self._backtest_data is None:
            raise RuntimeError(
                "load_backtest_data() must be called before iter_bars()."
            )

        n = len(self._backtest_datetimes)
        for i, bar_dt in enumerate(self._backtest_datetimes):
            self._backtest_cursor = i
            bar_df = self._backtest_data.xs(bar_dt, level="datetime", drop_level=False)
            yield bar_dt, bar_df

    def reset_backtest_cursor(self) -> None:
        """Reset the bar-by-bar cursor to the beginning of the dataset."""
        self._backtest_cursor = 0

    @property
    def backtest_cursor(self) -> int:
        """Current position in the backtest timeline (0-based bar index)."""
        return self._backtest_cursor

    @property
    def backtest_total_bars(self) -> int:
        """Total number of timestamps in the loaded backtest dataset."""
        return len(self._backtest_datetimes)

    # ------------------------------------------------------------------
    # Internal: fetch with fallback
    # ------------------------------------------------------------------

    def _fetch_bars_with_fallback(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: str,
    ) -> pd.DataFrame:
        """Try primary provider; fall back to yfinance on failure."""
        try:
            return self._provider.get_bars(symbols, start, end, timeframe)
        except Exception as exc:
            self._logger.log_error(
                f"Primary provider failed for {symbols}: {exc}; "
                "attempting fallback.",
                exc_info=exc,
            )

        if self._fallback_provider is not None:
            self._logger.log_info(
                "DataManager: using YFinance FALLBACK",
                metadata={"symbols": symbols},
            )
            return self._fallback_provider.get_bars(symbols, start, end, timeframe)

        raise RuntimeError(
            f"Both primary and fallback providers failed for symbols={symbols}, "
            f"start={start}, end={end}, timeframe={timeframe}."
        )

    def _try_fallback_latest(
        self,
        symbols: List[str],
        timeframe: str,
        limit: int,
    ) -> pd.DataFrame:
        """Attempt to get latest bars from the fallback provider."""
        if self._fallback_provider is None:
            return _empty_bars_df()
        try:
            return self._fallback_provider.get_latest(symbols, timeframe, limit)
        except Exception as exc:
            self._logger.log_error(
                f"Fallback provider get_latest failed: {exc}", exc_info=exc
            )
            return _empty_bars_df()

    # ------------------------------------------------------------------
    # Internal: backtest helpers
    # ------------------------------------------------------------------

    def _get_latest_bars_backtest(
        self,
        symbols: List[str],
        limit: int,
    ) -> pd.DataFrame:
        """Slice the most recent *limit* bars from loaded backtest data."""
        if self._backtest_data is None:
            return _empty_bars_df()

        # Bars up to and including the current cursor position
        dts = self._backtest_datetimes[: self._backtest_cursor + 1]
        if not dts:
            return _empty_bars_df()

        # Take the last `limit` datetimes
        recent_dts = dts[-limit:]
        mask = self._backtest_data.index.get_level_values("datetime").isin(recent_dts)
        df = self._backtest_data.loc[mask]

        # Filter to requested symbols
        sym_mask = df.index.get_level_values("symbol").isin(symbols)
        return df.loc[sym_mask]

    def _get_snapshot_backtest(self, symbols: List[str]) -> Dict[str, dict]:
        """Build a synthetic snapshot from the most recent bar at the cursor."""
        latest = self._get_latest_bars_backtest(symbols, limit=1)
        result: Dict[str, dict] = {}
        for sym in symbols:
            try:
                sym_df = latest.xs(sym, level="symbol")
                if sym_df.empty:
                    continue
                row = sym_df.iloc[-1]
                close = float(row.get("close", 0))
                result[sym] = {
                    "price": close,
                    "volume": int(row.get("volume", 0)),
                    "bid": round(close * 0.9995, 4),
                    "ask": round(close * 1.0005, 4),
                    "timestamp": sym_df.index[-1].isoformat()
                    if hasattr(sym_df.index[-1], "isoformat")
                    else str(sym_df.index[-1]),
                }
            except (KeyError, IndexError):
                continue
        return result

    def _synthetic_snapshot_from_latest(
        self, symbols: List[str]
    ) -> Dict[str, dict]:
        """Build a snapshot from the latest bar when provider lacks quote support."""
        latest = self.get_latest_bars(symbols, timeframe="1D", limit=1)
        result: Dict[str, dict] = {}
        for sym in symbols:
            try:
                sym_df = latest.xs(sym, level="symbol")
                if sym_df.empty:
                    continue
                row = sym_df.iloc[-1]
                close = float(row.get("close", 0))
                result[sym] = {
                    "price": close,
                    "volume": int(row.get("volume", 0)),
                    "bid": round(close * 0.9995, 4),
                    "ask": round(close * 1.0005, 4),
                    "timestamp": sym_df.index[-1].isoformat()
                    if hasattr(sym_df.index[-1], "isoformat")
                    else str(sym_df.index[-1]),
                }
            except (KeyError, IndexError):
                continue
        return result

    # ------------------------------------------------------------------
    # Data validation and cleaning
    # ------------------------------------------------------------------

    def _validate_and_clean(
        self,
        df: pd.DataFrame,
        symbols: List[str],
        timeframe: str,
    ) -> pd.DataFrame:
        """Validate and clean a raw OHLCV DataFrame.

        Checks performed per symbol:
        1. **NaN detection** — forward-fill up to ``_MAX_FFILL_BARS`` bars;
           warn if gaps are larger.
        2. **Negative prices** — rows with any negative OHLC value are
           dropped and a warning is emitted.
        3. **Zero volume** — rows with ``volume == 0`` are flagged (warning
           only; data is kept because some legitimate bars have zero volume on
           thin markets / non-trading days).
        4. **Out-of-hours timestamps** — for intraday timeframes, rows outside
           09:30–16:00 ET generate an INFO log but are not dropped (pre/post
           market data may be intentional).

        Parameters
        ----------
        df:
            Raw DataFrame with MultiIndex ``(datetime, symbol)``.
        symbols:
            Expected symbols.
        timeframe:
            Used to decide whether timestamp-range checks apply.

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame (same structure, potentially fewer rows).
        """
        if df.empty:
            self._logger.log_info(
                "DataManager: validate_and_clean received empty DataFrame",
                metadata={"symbols": symbols},
            )
            return df

        price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
        is_intraday = timeframe not in ("1D", "1Day")

        cleaned_parts: List[pd.DataFrame] = []

        for sym in df.index.get_level_values("symbol").unique():
            try:
                sym_df = df.xs(sym, level="symbol", drop_level=False)
            except KeyError:
                continue

            # ---- 1. NaN handling ------------------------------------------
            nan_counts = sym_df[price_cols].isna().sum()
            total_nans = int(nan_counts.sum())
            if total_nans > 0:
                # Count consecutive NaN runs per column
                max_gap = _max_consecutive_nans(sym_df[price_cols])
                if max_gap > _MAX_FFILL_BARS:
                    warnings.warn(
                        f"DataManager [{sym}]: NaN gap of {max_gap} bars exceeds "
                        f"fill limit of {_MAX_FFILL_BARS}.  Data may be unreliable.",
                        UserWarning,
                        stacklevel=2,
                    )
                    self._logger.log_info(
                        f"DataManager: large NaN gap in {sym}",
                        metadata={"max_gap": max_gap, "fill_limit": _MAX_FFILL_BARS},
                    )
                # Forward-fill within the limit
                sym_df = sym_df.copy()
                sym_df[price_cols] = (
                    sym_df[price_cols]
                    .ffill(limit=_MAX_FFILL_BARS)
                )

            # ---- 2. Negative prices ----------------------------------------
            if price_cols:
                neg_mask = (sym_df[price_cols] < 0).any(axis=1)
                if neg_mask.any():
                    n_neg = int(neg_mask.sum())
                    warnings.warn(
                        f"DataManager [{sym}]: dropping {n_neg} rows with "
                        "negative prices.",
                        UserWarning,
                        stacklevel=2,
                    )
                    self._logger.log_info(
                        f"DataManager: dropped {n_neg} negative-price rows for {sym}"
                    )
                    sym_df = sym_df.loc[~neg_mask]

            # ---- 3. Zero volume --------------------------------------------
            if "volume" in sym_df.columns:
                zero_vol = (sym_df["volume"] == 0).sum()
                if zero_vol > 0:
                    self._logger.log_info(
                        f"DataManager [{sym}]: {zero_vol} bars with zero volume "
                        "(kept; may be non-trading bars)"
                    )

            # ---- 4. Out-of-hours timestamps (intraday only) ----------------
            if is_intraday:
                ooh_count = _count_out_of_hours(sym_df)
                if ooh_count > 0:
                    self._logger.log_info(
                        f"DataManager [{sym}]: {ooh_count} out-of-hours bars "
                        "(pre/post market; kept)"
                    )

            cleaned_parts.append(sym_df)

        if not cleaned_parts:
            return _empty_bars_df()

        result = pd.concat(cleaned_parts).sort_index()
        return result

    # ------------------------------------------------------------------
    # Cache access passthrough (for external callers)
    # ------------------------------------------------------------------

    @property
    def cache(self) -> DataCache:
        """Direct access to the underlying :class:`~data.cache.DataCache`."""
        return self._cache

    def invalidate_cache(self, symbol: Optional[str] = None) -> None:
        """Invalidate cached data for *symbol* (or the entire cache).

        Parameters
        ----------
        symbol:
            If given, only entries whose key starts with ``symbol:`` are
            removed.  Otherwise, the entire cache is flushed.
        """
        if symbol is None:
            self._cache.clear()
            self._logger.log_info("DataManager: entire cache flushed")
        else:
            # The cache does not support prefix deletion natively; we use
            # delete on known key patterns.  Full-flush is the safe fallback.
            self._cache.clear()
            self._logger.log_info(
                f"DataManager: cache flushed (invalidate for {symbol})"
            )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _ensure_utc(dt: datetime) -> datetime:
    """Return *dt* as UTC-aware; attach UTC if naïve."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _empty_bars_df() -> pd.DataFrame:
    """Return an empty OHLCV DataFrame with the correct MultiIndex."""
    return pd.DataFrame(
        columns=_OHLCV_COLS,
        index=pd.MultiIndex.from_tuples([], names=["datetime", "symbol"]),
    )


def _bar_cache_key(
    symbols: List[str],
    start: datetime,
    end: datetime,
    timeframe: str,
) -> str:
    """Build a deterministic cache key for a bar request."""
    sym_part = "_".join(sorted(symbols))
    return (
        f"bars:{sym_part}:{timeframe}:"
        f"{start.strftime('%Y%m%dT%H%M%S')}:{end.strftime('%Y%m%dT%H%M%S')}"
    )


def _max_consecutive_nans(df: pd.DataFrame) -> int:
    """Return the maximum consecutive NaN run across all columns of *df*.

    Vectorised implementation using pandas groupby on non-NaN boundaries.
    """
    max_run = 0
    for col in df.columns:
        is_nan = df[col].isna()
        if not is_nan.any():
            continue
        # Group consecutive NaN sequences and find the longest
        groups = (~is_nan).cumsum()
        nan_groups = groups[is_nan]
        if len(nan_groups) == 0:
            continue
        longest = nan_groups.value_counts().max()
        max_run = max(max_run, int(longest))
    return max_run


def _count_out_of_hours(sym_df: pd.DataFrame) -> int:
    """Count bars whose timestamp falls outside 09:30–16:00 ET (approximate).

    Fully vectorised implementation using pandas DatetimeIndex operations.
    We convert UTC timestamps to approximate ET by subtracting 5 hours
    (standard time; daylight saving is not accounted for as this check
    is advisory only).
    """
    datetimes = sym_df.index.get_level_values("datetime")
    if len(datetimes) == 0:
        return 0

    try:
        # Vectorised: extract hour and minute as integer arrays
        et_hours = (datetimes.hour - _ET_OFFSET_HOURS) % 24
        et_minutes = datetimes.minute

        # Encode as fractional hours for simple comparison
        et_time = et_hours + et_minutes / 60.0
        market_open = _MARKET_OPEN_HOUR + _MARKET_OPEN_MINUTE / 60.0  # 9.5
        market_close = float(_MARKET_CLOSE_HOUR)                      # 16.0

        outside = (et_time < market_open) | (et_time > market_close)
        return int(outside.sum())
    except (AttributeError, TypeError):
        return 0
