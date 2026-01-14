"""Polygon.io REST API client for OHLCV data.

V1.2-data: Uses Massive/OTREP key for authenticated access.
Supports daily and intraday (minute-based) timeframes.
"""

import os
import requests
import pandas as pd
from typing import Tuple, Optional


class PolygonClient:
    """
    Lightweight client for Massive/Polygon REST API to fetch OHLCV aggregates.
    
    Usage:
        client = PolygonClient(api_key_env="POLYGON_API_KEY_OTREP")
        df = client.get_ohlcv("SPY", "2024-01-01", "2024-12-31", timeframe="1d")
    """

    def __init__(self,
                 api_key_env: str = "POLYGON_API_KEY_OTREP",
                 base_url: str = "https://api.polygon.io"):
        """
        Initialize Polygon client.
        
        Args:
            api_key_env: Environment variable name containing the API key
            base_url: Polygon API base URL
        """
        self.api_key = os.getenv(api_key_env, "")
        if not self.api_key:
            raise ValueError(
                f"Polygon API key not found in environment variable '{api_key_env}'. "
                f"Please set it via: export {api_key_env}=your_key"
            )
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env

    def _map_timeframe(self, timeframe: str) -> Tuple[int, str]:
        """
        Map a timeframe string to (multiplier, timespan) for Polygon aggs.

        Supported examples:
          "1d"  -> (1, "day")
          "60m" -> (60, "minute")
          "30m" -> (30, "minute")
          "15m" -> (15, "minute")
          "5m"  -> (5, "minute")
          "1m"  -> (1, "minute")
        """
        tf = timeframe.lower().strip()
        
        if tf == "1d" or tf == "d" or tf == "day":
            return 1, "day"
        
        if tf.endswith("m"):
            try:
                m = int(tf[:-1])
                return m, "minute"
            except ValueError:
                pass
        
        if tf.endswith("h"):
            try:
                h = int(tf[:-1])
                return h, "hour"
            except ValueError:
                pass
        
        raise ValueError(
            f"Unsupported timeframe for Polygon: '{timeframe}'. "
            f"Examples: '1d', '60m', '30m', '15m', '5m', '1m'"
        )

    def get_ohlcv(self,
                  ticker: str,
                  start_date: str,
                  end_date: str,
                  timeframe: str = "1d") -> pd.DataFrame:
        """
        Fetch OHLCV bars for a ticker between start_date and end_date (inclusive).
        
        Args:
            ticker: Stock/ETF symbol (e.g., "SPY", "AAPL")
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            timeframe: Bar size ("1d", "60m", "30m", "15m", etc.)
        
        Returns:
            DataFrame with columns: open, high, low, close, volume,
            indexed by naive DatetimeIndex in ascending order.
            Empty DataFrame if no data available.
        """
        multiplier, timespan = self._map_timeframe(timeframe)
        
        url = (
            f"{self.base_url}/v2/aggs/ticker/{ticker.upper()}/range/"
            f"{multiplier}/{timespan}/{start_date}/{end_date}"
        )
        
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": self.api_key,
        }

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Polygon API request failed for {ticker}: {e}")
        
        data = resp.json()
        
        # Check for API errors
        if data.get("status") == "ERROR":
            error_msg = data.get("error", "Unknown error")
            raise RuntimeError(f"Polygon API error for {ticker}: {error_msg}")

        results = data.get("results", [])
        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        
        # Polygon fields: t (timestamp in ms), o, h, l, c, v, vw, n
        if "t" not in df.columns or "o" not in df.columns:
            return pd.DataFrame()

        # Convert timestamp (ms) to datetime
        df["datetime"] = pd.to_datetime(df["t"], unit="ms")
        df.set_index("datetime", inplace=True)
        
        # Remove timezone info for consistency with existing pipeline
        df.index = df.index.tz_localize(None)
        df = df.sort_index()

        # Rename to standard OHLCV columns
        df = df.rename(columns={
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        })

        # Select only the columns we need
        df = df[["open", "high", "low", "close", "volume"]]
        
        # Filter strictly within [start_date, end_date]
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        df = df[(df.index >= start_dt) & (df.index <= end_dt)]
        
        return df
    
    def test_connection(self) -> bool:
        """
        Test API connection with a simple request.
        
        Returns:
            True if connection successful, raises exception otherwise.
        """
        # Use a simple endpoint to test
        url = f"{self.base_url}/v1/marketstatus/now"
        params = {"apiKey": self.api_key}
        
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Polygon API connection test failed: {e}")


# Singleton pattern for client reuse
_client_singleton: Optional[PolygonClient] = None


def get_polygon_client(api_key_env: str = "POLYGON_API_KEY_OTREP") -> PolygonClient:
    """
    Get or create a singleton PolygonClient instance.
    
    Args:
        api_key_env: Environment variable name containing the API key
    
    Returns:
        PolygonClient instance
    """
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = PolygonClient(api_key_env=api_key_env)
    return _client_singleton


def reset_polygon_client():
    """Reset the singleton client (useful for testing)."""
    global _client_singleton
    _client_singleton = None


if __name__ == "__main__":
    # Quick test
    import sys
    
    print("Testing PolygonClient...")
    
    try:
        client = get_polygon_client()
        print(f"✓ Client initialized with key from {client.api_key_env}")
        
        # Test connection
        client.test_connection()
        print("✓ API connection successful")
        
        # Test data fetch
        df = client.get_ohlcv("SPY", "2024-01-01", "2024-01-31", timeframe="1d")
        if not df.empty:
            print(f"✓ Fetched {len(df)} bars for SPY (2024-01)")
            print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
            print(f"  Columns: {list(df.columns)}")
        else:
            print("⚠️ No data returned for SPY")
            
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)
    
    print("\nPolygonClient tests passed!")
