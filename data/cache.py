"""
data/cache.py
=============
Thread-safe in-memory data cache for the ATNN Quant Powerhouse.

Designed with a Redis-ready architecture: the storage backend is fully
abstracted so callers interact only with :class:`DataCache`.  Swapping to
Redis in production is a one-file change.

TTL presets
-----------
    - Quote cache   : 1 second   (real-time bid/ask)
    - Bar cache     : 60 seconds (OHLCV bars)
    - Feature cache : 300 seconds (derived ML features)

Usage
-----
    from data.cache import DataCache, TTL_BARS

    cache = DataCache(max_entries=5000)
    cache.set("SPY:bars:1D", df, ttl_seconds=TTL_BARS)
    df = cache.get("SPY:bars:1D")          # None if missing or stale

    # or use the all-in-one helper:
    df = cache.get_or_fetch("SPY:bars:1D", fetch_fn=lambda: provider.get_bars(...), ttl=TTL_BARS)
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from core.logger import get_trade_logger

# ---------------------------------------------------------------------------
# TTL constants (seconds)
# ---------------------------------------------------------------------------
TTL_QUOTE: int = 1       # real-time bid/ask snapshot
TTL_BARS: int = 60       # OHLCV bars
TTL_FEATURES: int = 300  # engineered ML features


# ---------------------------------------------------------------------------
# Internal cache entry
# ---------------------------------------------------------------------------

@dataclass
class _CacheEntry:
    """Single cached value with expiry metadata."""

    value: Any
    expires_at: float          # unix timestamp (time.monotonic)
    created_at: float = field(default_factory=time.monotonic)

    def is_expired(self) -> bool:
        """Return True if this entry has passed its TTL."""
        return time.monotonic() >= self.expires_at


# ---------------------------------------------------------------------------
# Cache backend (in-process, LRU)
# ---------------------------------------------------------------------------

class _InMemoryBackend:
    """LRU-evicting in-memory store.

    Uses :class:`collections.OrderedDict` for O(1) move-to-end on access,
    giving approximate LRU semantics.

    All public methods acquire the internal lock, making the backend
    safe to use from multiple threads.
    """

    def __init__(self, max_entries: int) -> None:
        self._max_entries = max(1, max_entries)
        self._store: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = threading.Lock()

        # Stats
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def set(self, key: str, value: Any, ttl_seconds: float) -> None:
        """Store *value* under *key* with the given TTL.

        If the cache is full, the least-recently-used entry is evicted
        before the new entry is inserted.

        Parameters
        ----------
        key:
            Cache key string.
        value:
            Any picklable Python object.
        ttl_seconds:
            Time-to-live in seconds (must be > 0).
        """
        if ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds must be positive; got {ttl_seconds!r}")

        expires_at = time.monotonic() + ttl_seconds
        entry = _CacheEntry(value=value, expires_at=expires_at)

        with self._lock:
            if key in self._store:
                # Update existing entry; move to end (most-recently-used)
                self._store.move_to_end(key)
                self._store[key] = entry
            else:
                # Evict LRU entry if at capacity
                if len(self._store) >= self._max_entries:
                    self._store.popitem(last=False)
                    self._evictions += 1
                self._store[key] = entry

    def get(self, key: str) -> Optional[Any]:
        """Return the cached value for *key*, or ``None`` if missing/stale.

        On a hit the entry is moved to the MRU position.  Expired entries
        are removed on access (lazy eviction).

        Parameters
        ----------
        key:
            Cache key string.

        Returns
        -------
        The stored value, or ``None``.
        """
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            if entry.is_expired():
                del self._store[key]
                self._misses += 1
                return None
            # Move to MRU position
            self._store.move_to_end(key)
            self._hits += 1
            return entry.value

    def delete(self, key: str) -> bool:
        """Remove *key* from the cache.

        Returns
        -------
        bool
            True if the key existed and was removed.
        """
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def clear(self) -> None:
        """Remove all entries from the cache."""
        with self._lock:
            self._store.clear()

    def purge_expired(self) -> int:
        """Remove all expired entries.

        Returns
        -------
        int
            Number of entries removed.
        """
        now = time.monotonic()
        removed = 0
        with self._lock:
            expired_keys = [
                k for k, v in self._store.items() if now >= v.expires_at
            ]
            for k in expired_keys:
                del self._store[k]
                removed += 1
        return removed

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Current number of entries in the cache."""
        with self._lock:
            return len(self._store)

    def get_stats(self) -> dict:
        """Return a snapshot of cache statistics.

        Returns
        -------
        dict with keys:
            - ``hits``          — total cache hits
            - ``misses``        — total cache misses
            - ``evictions``     — total LRU evictions
            - ``size``          — current entry count
            - ``max_entries``   — configured capacity
            - ``hit_rate``      — float in [0, 1] or NaN if no requests
            - ``miss_rate``     — float in [0, 1] or NaN if no requests
        """
        with self._lock:
            hits = self._hits
            misses = self._misses
            evictions = self._evictions
            size = len(self._store)

        total = hits + misses
        if total > 0:
            hit_rate = hits / total
            miss_rate = misses / total
        else:
            hit_rate = float("nan")
            miss_rate = float("nan")

        return {
            "hits": hits,
            "misses": misses,
            "evictions": evictions,
            "size": size,
            "max_entries": self._max_entries,
            "hit_rate": hit_rate,
            "miss_rate": miss_rate,
        }


# ---------------------------------------------------------------------------
# Public DataCache API
# ---------------------------------------------------------------------------

class DataCache:
    """High-level, Redis-ready data cache for market data.

    Wraps :class:`_InMemoryBackend` and exposes a clean interface aligned
    with the three TTL tiers used throughout the system:

    +---------------+----------+---------------------------------+
    | Tier          | TTL      | Typical keys                    |
    +===============+==========+=================================+
    | Quote         | 1 s      | ``<SYM>:quote``                 |
    | Bar           | 60 s     | ``<SYM>:bars:<TF>:<start>:<end>``|
    | Feature       | 300 s    | ``<SYM>:features:<date>``       |
    +---------------+----------+---------------------------------+

    The backend can be swapped to Redis later by replacing
    ``self._backend`` with a Redis-backed implementation that honours the
    same ``set`` / ``get`` / ``delete`` / ``clear`` contract.

    Parameters
    ----------
    max_entries:
        Maximum number of entries before LRU eviction kicks in.
        Defaults to 10 000.
    log_level:
        Logging level for cache diagnostics (INFO / DEBUG / …).

    Examples
    --------
    >>> cache = DataCache(max_entries=5000)
    >>> cache.set("SPY:quote", {"bid": 499.50, "ask": 499.51}, ttl_seconds=TTL_QUOTE)
    >>> snapshot = cache.get("SPY:quote")

    >>> df = cache.get_or_fetch(
    ...     "SPY:bars:1D:2024-01-01:2024-12-31",
    ...     fetch_fn=lambda: provider.get_bars(["SPY"], ...),
    ...     ttl=TTL_BARS,
    ... )
    """

    def __init__(
        self,
        max_entries: int = 10_000,
        log_level: str = "INFO",
    ) -> None:
        self._backend = _InMemoryBackend(max_entries=max_entries)
        self._logger = get_trade_logger()
        self._log_level = log_level

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def set(self, key: str, value: Any, ttl_seconds: float) -> None:
        """Store *value* under *key* for *ttl_seconds* seconds.

        Parameters
        ----------
        key:
            Unique cache key (suggested format: ``<symbol>:<type>:<…>``).
        value:
            Any Python object (DataFrame, dict, list, …).
        ttl_seconds:
            Positive TTL in seconds.  Use the :data:`TTL_QUOTE`,
            :data:`TTL_BARS`, or :data:`TTL_FEATURES` constants for
            consistency.
        """
        self._backend.set(key, value, ttl_seconds)

    def get(self, key: str) -> Optional[Any]:
        """Return the cached value for *key*, or ``None`` if stale/absent.

        Parameters
        ----------
        key:
            Cache key string.

        Returns
        -------
        Cached value or ``None``.
        """
        return self._backend.get(key)

    def delete(self, key: str) -> bool:
        """Invalidate *key*.

        Returns
        -------
        bool
            True if the key existed and was removed.
        """
        return self._backend.delete(key)

    def clear(self) -> None:
        """Flush the entire cache."""
        self._backend.clear()

    def purge_expired(self) -> int:
        """Proactively remove all expired entries.

        Call periodically from a maintenance thread to keep memory usage
        bounded.  Lazy eviction on :meth:`get` also handles this, but
        proactive purging prevents stale entries from occupying slots until
        they are next accessed.

        Returns
        -------
        int
            Number of entries removed.
        """
        removed = self._backend.purge_expired()
        if removed:
            self._logger.log_info(
                f"Cache: purged {removed} expired entries; "
                f"size now {self._backend.size}"
            )
        return removed

    # ------------------------------------------------------------------
    # Convenience helper
    # ------------------------------------------------------------------

    def get_or_fetch(
        self,
        key: str,
        fetch_fn: Callable[[], Any],
        ttl: float,
    ) -> Any:
        """Return cached value or call *fetch_fn* on a cache miss.

        On a miss, the result of *fetch_fn()* is stored under *key* with
        the given *ttl* before being returned.  If *fetch_fn* raises, the
        exception propagates unchanged — no partial results are cached.

        Parameters
        ----------
        key:
            Cache key string.
        fetch_fn:
            Zero-argument callable that returns fresh data.
        ttl:
            TTL in seconds for the newly fetched value.

        Returns
        -------
        Cached or freshly fetched value.

        Raises
        ------
        Any exception raised by *fetch_fn*.
        """
        cached = self._backend.get(key)
        if cached is not None:
            return cached

        # Cache miss — fetch fresh data
        value = fetch_fn()
        self._backend.set(key, value, ttl)
        return value

    # ------------------------------------------------------------------
    # Tier-aware helpers
    # ------------------------------------------------------------------

    def set_quote(self, symbol: str, snapshot: dict) -> None:
        """Cache a real-time quote snapshot (TTL = 1 s).

        Parameters
        ----------
        symbol:
            Ticker, e.g. ``"AAPL"``.
        snapshot:
            Dict with keys ``price``, ``bid``, ``ask``, ``volume``,
            ``timestamp``.
        """
        self.set(f"{symbol}:quote", snapshot, TTL_QUOTE)

    def get_quote(self, symbol: str) -> Optional[dict]:
        """Return the cached quote for *symbol*, or ``None``."""
        return self.get(f"{symbol}:quote")

    def set_bars(self, key: str, df: Any) -> None:
        """Cache an OHLCV bar DataFrame (TTL = 60 s).

        Parameters
        ----------
        key:
            Full bar cache key, e.g. ``"AAPL:bars:1D:2024-01-01:2024-12-31"``.
        df:
            ``pd.DataFrame`` with OHLCV data.
        """
        self.set(key, df, TTL_BARS)

    def get_bars(self, key: str) -> Optional[Any]:
        """Return a cached bar DataFrame, or ``None``."""
        return self.get(key)

    def set_features(self, key: str, features: Any) -> None:
        """Cache a feature matrix (TTL = 300 s).

        Parameters
        ----------
        key:
            Feature cache key, e.g. ``"AAPL:features:2024-12-31"``.
        features:
            Feature data (DataFrame, ndarray, dict, …).
        """
        self.set(key, features, TTL_FEATURES)

    def get_features(self, key: str) -> Optional[Any]:
        """Return cached feature data, or ``None``."""
        return self.get(key)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def hit_rate(self) -> float:
        """Cache hit rate in [0, 1] (NaN if no requests yet)."""
        return self._backend.get_stats()["hit_rate"]

    @property
    def miss_rate(self) -> float:
        """Cache miss rate in [0, 1] (NaN if no requests yet)."""
        return self._backend.get_stats()["miss_rate"]

    @property
    def eviction_count(self) -> int:
        """Total number of LRU evictions since instantiation."""
        return self._backend.get_stats()["evictions"]

    @property
    def size(self) -> int:
        """Current number of entries in the cache."""
        return self._backend.size

    def get_stats(self) -> dict:
        """Return a full statistics snapshot.

        Returns
        -------
        dict
            Same structure as :meth:`_InMemoryBackend.get_stats`.
        """
        return self._backend.get_stats()

    def log_stats(self) -> None:
        """Log current cache statistics at INFO level."""
        stats = self.get_stats()
        self._logger.log_info(
            "Cache stats",
            metadata={
                "hits": stats["hits"],
                "misses": stats["misses"],
                "hit_rate": round(stats["hit_rate"], 4)
                if stats["hit_rate"] == stats["hit_rate"]  # not NaN
                else "n/a",
                "evictions": stats["evictions"],
                "size": stats["size"],
                "max_entries": stats["max_entries"],
            },
        )
