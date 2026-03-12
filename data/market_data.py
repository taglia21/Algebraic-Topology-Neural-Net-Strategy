"""
data/market_data.py
===================
Concrete market-data providers for the ATNN Quant Powerhouse.

Providers
---------
IBKRDataProvider
    Primary provider.  Uses ``ib_async`` to fetch historical OHLCV bars and
    real-time quotes from Interactive Brokers TWS / Gateway.

All providers implement the :class:`DataProvider` ABC.

Usage
-----
    from data.market_data import IBKRDataProvider
    from datetime import datetime, timezone

    provider = IBKRDataProvider(host="127.0.0.1", port=7497, client_id=1)
    df = provider.get_bars(
        ["AAPL", "MSFT"],
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 12, 31, tzinfo=timezone.utc),
        timeframe="1D",
    )
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd

from core.config import get_config
from core.logger import get_trade_logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OHLCV_COLS = ["open", "high", "low", "close", "volume"]

# Maps canonical timeframe strings to IBKR duration/bar-size pairs.
_IBKR_BAR_SIZE_MAP: Dict[str, str] = {
    "1D":    "1 day",
    "1Day":  "1 day",
    "1H":    "1 hour",
    "1Hour": "1 hour",
    "4H":    "4 hours",
    "15Min": "15 mins",
    "15M":   "15 mins",
    "5Min":  "5 mins",
    "5M":    "5 mins",
    "30Min": "30 mins",
    "30M":   "30 mins",
    "1Min":  "1 min",
    "1M":    "1 min",
}


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
# IBKR provider
# ---------------------------------------------------------------------------

class IBKRDataProvider(DataProvider):
    """Primary market-data provider backed by Interactive Brokers via ``ib_async``.

    This is a **stub** implementation.  The full provider will use the
    ``ib_async`` library to request historical bars and streaming quotes
    from TWS / IB Gateway.

    Parameters
    ----------
    host:
        TWS/Gateway host.  Defaults to config value.
    port:
        TWS/Gateway port.  Defaults to config value.
    client_id:
        IBKR client ID.  Defaults to config value.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        client_id: Optional[int] = None,
    ) -> None:
        cfg = get_config()
        self._logger = get_trade_logger()

        self._host = host or cfg.ibkr.host
        self._port = port or cfg.ibkr.port
        self._client_id = client_id or cfg.ibkr.client_id

        self._logger.log_info(
            "IBKRDataProvider initialised (stub)",
            metadata={
                "host": self._host,
                "port": self._port,
                "client_id": self._client_id,
            },
        )

    # ------------------------------------------------------------------
    # DataProvider interface (stub)
    # ------------------------------------------------------------------

    def get_bars(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: str = "1D",
    ) -> pd.DataFrame:
        """Fetch historical OHLCV bars from IBKR.

        .. note::
            Stub — returns an empty DataFrame.  Will be implemented when
            ``ib_async`` integration is complete.
        """
        bar_size = _IBKR_BAR_SIZE_MAP.get(timeframe)
        if bar_size is None:
            raise ValueError(
                f"Unsupported timeframe {timeframe!r}. "
                f"Supported: {sorted(_IBKR_BAR_SIZE_MAP)}"
            )

        self._logger.log_info(
            "IBKRDataProvider.get_bars (stub — returning empty DataFrame)",
            metadata={
                "symbols": symbols,
                "start": self._ensure_utc(start).isoformat(),
                "end": self._ensure_utc(end).isoformat(),
                "bar_size": bar_size,
            },
        )

        # TODO: implement via ib_async reqHistoricalData
        return pd.DataFrame(
            columns=_OHLCV_COLS,
            index=pd.MultiIndex.from_tuples([], names=["datetime", "symbol"]),
        )

    def get_latest(
        self,
        symbols: List[str],
        timeframe: str = "1D",
        limit: int = 1,
    ) -> pd.DataFrame:
        """Fetch the most recent *limit* bars for each symbol from IBKR.

        .. note::
            Stub — returns an empty DataFrame.
        """
        self._logger.log_info(
            "IBKRDataProvider.get_latest (stub — returning empty DataFrame)",
            metadata={"symbols": symbols, "timeframe": timeframe, "limit": limit},
        )

        # TODO: implement via ib_async reqHistoricalData with short duration
        return pd.DataFrame(
            columns=_OHLCV_COLS,
            index=pd.MultiIndex.from_tuples([], names=["datetime", "symbol"]),
        )
