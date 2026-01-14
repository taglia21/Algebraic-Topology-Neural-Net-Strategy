"""Data layer for multi-provider OHLCV access.

V1.2-data: Polygon (Massive/OTREP) as primary, yfinance as fallback.
Designed for future intraday support.
"""

from .data_provider import get_ohlcv_data
from .polygon_client import PolygonClient, get_polygon_client

__all__ = ['get_ohlcv_data', 'PolygonClient', 'get_polygon_client']
