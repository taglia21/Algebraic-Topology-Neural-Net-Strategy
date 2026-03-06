"""
data/
=====
Market-data infrastructure for the ATNN Quant Powerhouse.

Public surface
--------------
DataManager     — unified data abstraction layer (single point of contact)
DataCache       — thread-safe in-memory TTL cache
DataProvider    — abstract base class for concrete providers
AlpacaDataProvider  — primary provider (alpaca-py)
YFinanceDataProvider — fallback provider (yfinance, backtest only)

TTL constants: TTL_QUOTE, TTL_BARS, TTL_FEATURES
"""

from data.cache import DataCache, TTL_BARS, TTL_FEATURES, TTL_QUOTE
from data.data_manager import DataManager
from data.market_data import (
    AlpacaDataProvider,
    DataProvider,
    YFinanceDataProvider,
)

__all__ = [
    "DataManager",
    "DataCache",
    "DataProvider",
    "AlpacaDataProvider",
    "YFinanceDataProvider",
    "TTL_QUOTE",
    "TTL_BARS",
    "TTL_FEATURES",
]
