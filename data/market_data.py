"""
data/market_data.py
===================
Concrete market-data providers for the ATNN Quant Powerhouse.

Providers
---------
AlpacaDataProvider
    Primary provider.  Uses the ``alpaca-py`` SDK to fetch historical OHLCV
    bars via :class:`~alpaca.data.historical.StockHistoricalDataClient` and
    real-time quotes via REST.  Respects the 200 req/min free-tier rate limit
    with automatic back-off.

YFinanceDataProvider
    FALLBACK ONLY.  Used for initial backtest data loading and gap-filling
    when Alpaca is unavailable.  NOT suitable for real-time production signals.
    Every call emits a WARNING log so accidental production use is visible.

Both providers implement the :class:`DataProvider` ABC.

Usage
-----
    from data.market_data import AlpacaDataProvider, YFinanceDataProvider
    from datetime import datetime, timezone

    provider = AlpacaDataProvider(api_key="...", secret_key="...")
    df = provider.get_bars(
        ["AAPL", "MSFT"],
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 12, 31, tzinfo=timezone.utc),
        timeframe="1D",
    )
"""

from __future__ import annotations

import time
import warnings
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd

from core.config import get_config
from core.logger import get_trade_logger

# ---------------------------------------------------------------------------
# Timeframe normalisation helpers
# ---------------------------------------------------------------------------

# Maps the system's canonical timeframe strings to alpaca-py TimeFrame values.
# alpaca-py exposes alpaca.data.timeframe.TimeFrame with class attributes.
_ALPACA_TF_MAP: Dict[str, str] = {
    "1D":    "Day",
    "1Day":  "Day",
    "1H":    "Hour",
    "1Hour": "Hour",
    "4H":    "Hour",     # approximated — caller must resample
    "15Min": "Minute",   # must pass amount=15
    "15M":   "Minute",
    "1Min":  "Minute",
    "1M":    "Minute",
    "5Min":  "Minute",
    "5M":    "Minute",
    "30Min": "Minute",
    "30M":   "Minute",
}

_ALPACA_TF_AMOUNT: Dict[str, int] = {
    "15Min": 15,
    "15M":   15,
    "5Min":  5,
    "5M":    5,
    "30Min": 30,
    "30M":   30,
    "4H":    4,
    "1Hour": 1,
    "1H":    1,
    "1D":    1,
    "1Day":  1,
    "1Min":  1,
    "1M":    1,
}

_YFINANCE_TF_MAP: Dict[str, str] = {
    "1D":    "1d",
    "1Day":  "1d",
    "1H":    "1h",
    "1Hour": "1h",
    "15Min": "15m",
    "15M":   "15m",
    "5Min":  "5m",
    "5M":    "5m",
    "30Min": "30m",
    "30M":   "30m",
    "1Min":  "1m",
    "1M":    "1m",
}

_OHLCV_COLS = ["open", "high", "low", "close", "volume"]


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class DataProvider(ABC):
    """Abstract interface that every concrete data provider must implement.

    Implementors return a :class:`pandas.DataFrame` with a two-level
    :class:`~pandas.MultiIndex` of ``(datetime, symbol)`` and columns
    ``[open, high, low, close, volume]``.  All datetime values are
    timezone-aware UTC.
    """

    @abstractmethod
    def get_bars(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: str = "1D",
    ) -> pd.DataFrame:
        """Fetch historical OHLCV bars.

        Parameters
        ----------
        symbols:
            List of ticker symbols, e.g. ``["AAPL", "MSFT"]``.
        start:
            Inclusive start of the date range (UTC-aware).
        end:
            Inclusive end of the date range (UTC-aware).
        timeframe:
            Canonical timeframe string: ``"1D"``, ``"1Hour"``, ``"15Min"``, …

        Returns
        -------
        pd.DataFrame
            MultiIndex ``(datetime, symbol)`` × columns ``[open, high, low,
            close, volume]``.  Both index levels are sorted ascending.
        """
        ...

    @abstractmethod
    def get_latest(
        self,
        symbols: List[str],
        timeframe: str = "1D",
        limit: int = 1,
    ) -> pd.DataFrame:
        """Fetch the most recent *limit* bars for each symbol.

        Parameters
        ----------
        symbols:
            List of ticker symbols.
        timeframe:
            Canonical timeframe string.
        limit:
            Number of bars to return per symbol.

        Returns
        -------
        pd.DataFrame
            Same structure as :meth:`get_bars`.
        """
        ...

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_utc(dt: datetime) -> datetime:
        """Return *dt* as a UTC-aware datetime, attaching UTC if naïve."""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @staticmethod
    def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Lower-case all column names."""
        df.columns = [c.lower() for c in df.columns]
        return df


# ---------------------------------------------------------------------------
# Alpaca provider
# ---------------------------------------------------------------------------

class AlpacaDataProvider(DataProvider):
    """Primary market-data provider backed by the Alpaca free-tier API.

    Uses ``alpaca-py`` (``pip install alpaca-py``) for historical bars and
    REST quotes.  Automatically enforces the 200 req/min free-tier rate limit
    with a simple token-bucket implementation.

    Parameters
    ----------
    api_key:
        Alpaca API key.  Falls back to ``core.config`` if not supplied.
    secret_key:
        Alpaca secret key.  Falls back to ``core.config`` if not supplied.
    base_url:
        Broker base URL.  Defaults to the paper-trading endpoint from config.
    requests_per_minute:
        Rate-limit cap.  Free-tier Alpaca is 200 req/min.

    Raises
    ------
    ImportError
        If ``alpaca-py`` is not installed.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url: Optional[str] = None,
        requests_per_minute: int = 200,
    ) -> None:
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.enums import DataFeed
            from alpaca.data.requests import (
                StockBarsRequest,
                StockLatestBarRequest,
                StockLatestQuoteRequest,
            )
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        except ImportError as exc:
            raise ImportError(
                "alpaca-py is required for AlpacaDataProvider. "
                "Install it with: pip install alpaca-py"
            ) from exc

        cfg = get_config()
        self._logger = get_trade_logger()

        self._api_key = api_key or cfg.alpaca.api_key
        self._secret_key = secret_key or cfg.alpaca.secret_key
        self._base_url = base_url or cfg.alpaca.base_url

        # Lazy-import references saved for use in methods
        self._StockHistoricalDataClient = StockHistoricalDataClient
        self._StockBarsRequest = StockBarsRequest
        self._StockLatestBarRequest = StockLatestBarRequest
        self._StockLatestQuoteRequest = StockLatestQuoteRequest
        self._DataFeed = DataFeed
        self._TimeFrame = TimeFrame
        self._TimeFrameUnit = TimeFrameUnit

        # Instantiate the data client.  Alpaca-py requires at least an api_key.
        # If credentials are missing we defer client creation until first use
        # so instantiation itself doesn't fail during config-less unit tests.
        self._client: Optional[object] = None
        if self._api_key and self._secret_key:
            self._client = StockHistoricalDataClient(
                api_key=self._api_key,
                secret_key=self._secret_key,
            )

        # Token-bucket rate limiter
        self._req_interval = 60.0 / requests_per_minute  # seconds per request
        self._last_request_time: float = 0.0
        self._rate_lock = __import__("threading").Lock()

        self._logger.log_info(
            "AlpacaDataProvider initialised",
            metadata={
                "base_url": self._base_url,
                "has_credentials": bool(self._api_key and self._secret_key),
            },
        )

    # ------------------------------------------------------------------
    # Client initialisation guard
    # ------------------------------------------------------------------

    def _get_client(self):  # type: ignore[return]
        """Return the alpaca-py client, initialising it if necessary.

        Raises
        ------
        RuntimeError
            If credentials are not configured.
        """
        if self._client is not None:
            return self._client
        if not (self._api_key and self._secret_key):
            raise RuntimeError(
                "AlpacaDataProvider requires api_key and secret_key. "
                "Set ALPACA_API_KEY / ALPACA_API_SECRET environment variables, "
                "or pass them to the constructor."
            )
        self._client = self._StockHistoricalDataClient(
            api_key=self._api_key,
            secret_key=self._secret_key,
        )
        return self._client

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _rate_limit(self) -> None:
        """Block until it is safe to issue the next API request."""
        with self._rate_lock:
            now = time.monotonic()
            elapsed = now - self._last_request_time
            wait = self._req_interval - elapsed
            if wait > 0:
                time.sleep(wait)
            self._last_request_time = time.monotonic()

    # ------------------------------------------------------------------
    # Timeframe helpers
    # ------------------------------------------------------------------

    def _build_timeframe(self, timeframe: str):  # type: ignore[return]
        """Translate a canonical timeframe string to an alpaca-py TimeFrame."""
        tf_unit_str = _ALPACA_TF_MAP.get(timeframe)
        if tf_unit_str is None:
            raise ValueError(
                f"Unsupported timeframe {timeframe!r}. "
                f"Supported: {sorted(_ALPACA_TF_MAP)}"
            )
        amount = _ALPACA_TF_AMOUNT.get(timeframe, 1)
        unit = getattr(self._TimeFrameUnit, tf_unit_str)
        return self._TimeFrame(amount=amount, unit=unit)

    # ------------------------------------------------------------------
    # DataProvider interface
    # ------------------------------------------------------------------

    def get_bars(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: str = "1D",
    ) -> pd.DataFrame:
        """Fetch historical OHLCV bars from Alpaca.

        Parameters
        ----------
        symbols:
            Tickers to fetch.
        start:
            Inclusive start (UTC-aware).
        end:
            Inclusive end (UTC-aware).
        timeframe:
            Canonical timeframe string (``"1D"``, ``"1Hour"``, ``"15Min"``…).

        Returns
        -------
        pd.DataFrame
            MultiIndex ``(datetime, symbol)`` × ``[open, high, low, close,
            volume]``.

        Raises
        ------
        RuntimeError
            If the Alpaca API call fails after logging the error.
        """
        start = self._ensure_utc(start)
        end = self._ensure_utc(end)
        tf = self._build_timeframe(timeframe)

        self._logger.log_info(
            "Alpaca: fetching bars",
            metadata={
                "symbols": symbols,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "timeframe": timeframe,
            },
        )

        try:
            self._rate_limit()
            request = self._StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=tf,
                start=start,
                end=end,
                feed=self._DataFeed.IEX,
            )
            client = self._get_client()
            bars = client.get_stock_bars(request)
            df = bars.df  # type: ignore[attr-defined]
        except Exception as exc:
            self._logger.log_error(
                f"Alpaca get_bars failed for {symbols}: {exc}", exc_info=exc
            )
            raise RuntimeError(
                f"AlpacaDataProvider.get_bars failed: {exc}"
            ) from exc

        return self._format_bars_df(df)

    def get_latest(
        self,
        symbols: List[str],
        timeframe: str = "1D",
        limit: int = 1,
    ) -> pd.DataFrame:
        """Fetch the most recent *limit* bars for each symbol.

        Parameters
        ----------
        symbols:
            Tickers to fetch.
        timeframe:
            Canonical timeframe string.
        limit:
            Number of bars to return per symbol (default 1).

        Returns
        -------
        pd.DataFrame
            Same structure as :meth:`get_bars`.
        """
        from datetime import timedelta

        now = datetime.now(timezone.utc)
        # For daily bars, look back limit + 10 extra days to account for
        # weekends/holidays.  For intraday, limit + 500 minutes is safe.
        if "D" in timeframe or "Day" in timeframe:
            delta = timedelta(days=limit + 10)
        else:
            delta = timedelta(minutes=limit * 60 + 500)

        start = now - delta
        try:
            df = self.get_bars(symbols, start=start, end=now, timeframe=timeframe)
        except RuntimeError:
            return pd.DataFrame(
                columns=_OHLCV_COLS,
                index=pd.MultiIndex.from_tuples([], names=["datetime", "symbol"]),
            )

        # Keep only the last `limit` bars per symbol
        if not df.empty:
            df = (
                df.groupby(level="symbol", group_keys=False)
                .tail(limit)
            )
        return df

    def get_snapshot(self, symbols: List[str]) -> Dict[str, dict]:
        """Fetch the latest quote snapshot for each symbol.

        Returns a dict of ``{symbol: {price, volume, bid, ask, timestamp}}``.

        Parameters
        ----------
        symbols:
            Tickers to query.

        Returns
        -------
        dict
            Mapping from symbol to snapshot dict.
        """
        result: Dict[str, dict] = {}
        try:
            self._rate_limit()
            request = self._StockLatestQuoteRequest(symbol_or_symbols=symbols)
            client = self._get_client()
            quotes = client.get_stock_latest_quote(request)
            for sym, q in quotes.items():
                result[sym] = {
                    "price": float((q.ask_price + q.bid_price) / 2)
                    if q.ask_price and q.bid_price
                    else float(q.ask_price or q.bid_price or 0),
                    "volume": 0,   # quote object doesn't carry volume
                    "bid": float(q.bid_price or 0),
                    "ask": float(q.ask_price or 0),
                    "timestamp": q.timestamp.isoformat()
                    if hasattr(q.timestamp, "isoformat")
                    else str(q.timestamp),
                }
        except Exception as exc:
            self._logger.log_error(
                f"Alpaca get_snapshot failed for {symbols}: {exc}", exc_info=exc
            )
            # Return empty dict; caller decides how to handle
        return result

    # ------------------------------------------------------------------
    # Private formatting helpers
    # ------------------------------------------------------------------

    def _format_bars_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalise an alpaca-py bars DataFrame to the system's standard format.

        Alpaca returns a MultiIndex ``(symbol, timestamp)`` DataFrame with
        columns ``[open, high, low, close, volume, trade_count, vwap]``.
        We reorder the index to ``(datetime, symbol)``, lower-case columns,
        and keep only OHLCV.

        Parameters
        ----------
        df:
            Raw DataFrame from alpaca-py ``bars.df``.

        Returns
        -------
        pd.DataFrame
            MultiIndex ``(datetime, symbol)`` × ``[open, high, low, close,
            volume]``.
        """
        if df.empty:
            return pd.DataFrame(
                columns=_OHLCV_COLS,
                index=pd.MultiIndex.from_tuples([], names=["datetime", "symbol"]),
            )

        df = self._normalise_columns(df)

        # alpaca-py returns index (symbol, timestamp) — swap to (datetime, symbol)
        idx_names = [n.lower() if n else n for n in df.index.names]
        if "symbol" in idx_names and "timestamp" in idx_names:
            df.index.names = ["symbol", "datetime"]
            df = df.swaplevel("symbol", "datetime")
        elif df.index.nlevels == 2:
            df.index.names = ["datetime", "symbol"]

        # Ensure UTC-aware datetimes
        dt_idx = df.index.get_level_values("datetime")
        if dt_idx.tzinfo is None:
            dt_idx = dt_idx.tz_localize("UTC")
        else:
            dt_idx = dt_idx.tz_convert("UTC")
        df.index = pd.MultiIndex.from_arrays(
            [dt_idx, df.index.get_level_values("symbol")],
            names=["datetime", "symbol"],
        )

        # Keep only OHLCV columns (drop vwap, trade_count, etc.)
        available = [c for c in _OHLCV_COLS if c in df.columns]
        df = df[available].sort_index()
        return df


# ---------------------------------------------------------------------------
# yfinance provider (FALLBACK ONLY)
# ---------------------------------------------------------------------------

class YFinanceDataProvider(DataProvider):
    """**FALLBACK ONLY** — historical bars via ``yfinance``.

    This provider is intentionally limited to backtest data loading and
    gap-filling.  Every public method emits a WARNING log so accidental
    production use is visible in the audit trail.

    Do NOT use this provider for real-time signals.

    Parameters
    ----------
    Raises
    ------
    ImportError
        If ``yfinance`` is not installed.
    """

    _FALLBACK_WARNING = (
        "YFinanceDataProvider is a FALLBACK — not suitable for "
        "real-time production signals.  Use AlpacaDataProvider in live/paper mode."
    )

    def __init__(self) -> None:
        try:
            import yfinance  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "yfinance is required for YFinanceDataProvider. "
                "Install it with: pip install yfinance"
            ) from exc

        self._logger = get_trade_logger()
        self._logger.log_info(
            "YFinanceDataProvider initialised (FALLBACK mode)",
            metadata={"warning": self._FALLBACK_WARNING},
        )

    def get_bars(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: str = "1D",
    ) -> pd.DataFrame:
        """Fetch historical OHLCV bars via yfinance.

        .. warning::
            FALLBACK ONLY.  Not for real-time production signals.

        Parameters
        ----------
        symbols:
            Tickers to fetch.
        start:
            Inclusive start date (UTC-aware or naïve).
        end:
            Inclusive end date (UTC-aware or naïve).
        timeframe:
            Canonical timeframe string (mapped to yfinance interval).

        Returns
        -------
        pd.DataFrame
            MultiIndex ``(datetime, symbol)`` × ``[open, high, low, close,
            volume]``.
        """
        import yfinance as yf

        self._logger.log_info(
            "YFinance (FALLBACK): fetching bars — WARNING: not for production",
            metadata={
                "symbols": symbols,
                "start": str(start),
                "end": str(end),
                "timeframe": timeframe,
            },
        )
        warnings.warn(self._FALLBACK_WARNING, UserWarning, stacklevel=3)

        yf_interval = _YFINANCE_TF_MAP.get(timeframe, "1d")
        start = self._ensure_utc(start)
        end = self._ensure_utc(end)

        frames: List[pd.DataFrame] = []
        for sym in symbols:
            try:
                ticker = yf.Ticker(sym)
                raw = ticker.history(
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    interval=yf_interval,
                    auto_adjust=True,
                )
                if raw.empty:
                    self._logger.log_info(
                        f"YFinance: no data returned for {sym}",
                        metadata={"symbol": sym},
                    )
                    continue
                raw = self._normalise_columns(raw)
                # Keep OHLCV only
                available = [c for c in _OHLCV_COLS if c in raw.columns]
                raw = raw[available].copy()
                # Ensure UTC tz-aware index
                if raw.index.tzinfo is None:
                    raw.index = raw.index.tz_localize("UTC")
                else:
                    raw.index = raw.index.tz_convert("UTC")
                raw.index.name = "datetime"
                raw["symbol"] = sym
                raw = raw.set_index("symbol", append=True)
                raw = raw.reorder_levels(["datetime", "symbol"])
                frames.append(raw)
            except Exception as exc:
                self._logger.log_error(
                    f"YFinance get_bars failed for {sym}: {exc}", exc_info=exc
                )
                # Continue with remaining symbols — partial data is better than none

        if not frames:
            return pd.DataFrame(
                columns=_OHLCV_COLS,
                index=pd.MultiIndex.from_tuples([], names=["datetime", "symbol"]),
            )

        df = pd.concat(frames).sort_index()
        return df

    def get_latest(
        self,
        symbols: List[str],
        timeframe: str = "1D",
        limit: int = 1,
    ) -> pd.DataFrame:
        """Fetch recent bars via yfinance.

        .. warning::
            FALLBACK ONLY.  Not for real-time production signals.

        Parameters
        ----------
        symbols:
            Tickers to fetch.
        timeframe:
            Canonical timeframe string.
        limit:
            Number of bars to return per symbol.

        Returns
        -------
        pd.DataFrame
            Same structure as :meth:`get_bars`.
        """
        from datetime import timedelta

        warnings.warn(self._FALLBACK_WARNING, UserWarning, stacklevel=2)

        now = datetime.now(timezone.utc)
        if "D" in timeframe or "Day" in timeframe:
            delta = timedelta(days=limit + 10)
        else:
            delta = timedelta(days=7)

        start = now - delta
        df = self.get_bars(symbols, start=start, end=now, timeframe=timeframe)
        if not df.empty:
            df = (
                df.groupby(level="symbol", group_keys=False)
                .tail(limit)
            )
        return df
