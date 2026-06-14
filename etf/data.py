"""
etf/data.py
===========
Price-data loader for the ETF engine.

Uses yfinance for adjusted daily OHLCV and caches results to local parquet so
backtests are reproducible and runnable offline after the first pull. The
returned frames use *split/dividend-adjusted* close prices — essential for an
ETF total-return strategy (ignoring dividends materially understates bond/
dividend-ETF performance and biases momentum rankings).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger("etf.data")

_DEFAULT_CACHE_DIR = Path(os.environ.get("ETF_CACHE_DIR", ".etf_cache"))


class ETFDataError(RuntimeError):
    """Raised when price data cannot be obtained for the requested universe."""


def _cache_path(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "etf_prices.parquet"


def load_price_history(
    symbols: List[str],
    start: str,
    end: Optional[str] = None,
    *,
    use_cache: bool = True,
    refresh: bool = False,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load adjusted daily close prices for ``symbols``.

    Returns a wide DataFrame indexed by timezone-naive trading date with one
    column per symbol (adjusted close). Missing symbols raise ``ETFDataError``.

    Caching: a parquet snapshot keyed only by content is stored under
    ``cache_dir``. We re-download when the cache is missing, ``refresh`` is
    set, or the cache does not cover the requested symbols / date range.
    """
    cache_dir = cache_dir or _DEFAULT_CACHE_DIR
    path = _cache_path(cache_dir)
    end = end or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if use_cache and not refresh and path.exists():
        try:
            cached = pd.read_parquet(path)
            have = set(cached.columns)
            covers_symbols = set(symbols).issubset(have)
            covers_start = cached.index.min() <= pd.Timestamp(start)
            covers_end = cached.index.max() >= pd.Timestamp(end) - pd.Timedelta(days=7)
            if covers_symbols and covers_start and covers_end:
                logger.info("Loaded %d symbols from cache %s", len(symbols), path)
                return _slice(cached, symbols, start, end)
            logger.info("Cache miss (coverage); re-downloading.")
        except Exception as exc:  # pragma: no cover - corrupt cache fallback
            logger.warning("Failed to read cache (%s); re-downloading.", exc)

    prices = _download(symbols, start, end)

    if use_cache:
        try:
            to_store = prices
            if path.exists():
                existing = pd.read_parquet(path)
                to_store = existing.combine_first(prices)
                # prefer freshly downloaded values where they overlap
                to_store.update(prices)
            to_store.sort_index().to_parquet(path)
            logger.info("Cached %d symbols to %s", to_store.shape[1], path)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to write cache: %s", exc)

    return _slice(prices, symbols, start, end)


def _download(symbols: List[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices via yfinance."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ETFDataError("yfinance is required for backtest data") from exc

    logger.info("Downloading %d symbols %s -> %s via yfinance", len(symbols), start, end)
    raw = yf.download(
        tickers=symbols,
        start=start,
        end=end,
        auto_adjust=True,   # adjusted OHLC => total-return-consistent close
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw is None or len(raw) == 0:
        raise ETFDataError("yfinance returned no data (network or symbols issue)")

    # Normalise to a wide close-price frame regardless of single/multi ticker.
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise ETFDataError("yfinance response missing Close prices")
        close = raw["Close"].copy()
    else:
        # single ticker => flat columns
        close = raw[["Close"]].copy()
        close.columns = [symbols[0]]

    close.index = pd.to_datetime(close.index).tz_localize(None)
    close = close.sort_index().dropna(how="all")

    missing = [s for s in symbols if s not in close.columns]
    if missing:
        raise ETFDataError(f"No data returned for symbols: {missing}")

    return close


def _slice(prices: pd.DataFrame, symbols: List[str], start: str, end: str) -> pd.DataFrame:
    cols = [s for s in symbols if s in prices.columns]
    out = prices.loc[(prices.index >= pd.Timestamp(start)) & (prices.index <= pd.Timestamp(end)), cols]
    out = out.sort_index()
    # Forward-fill small gaps (holidays/half-days differ across asset classes),
    # then drop leading rows where the *core* universe is not yet listed.
    out = out.ffill()
    return out
