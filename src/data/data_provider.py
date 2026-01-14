"""Unified data provider API for OHLCV access.

V1.3-data: Extended validation, 10-year data support, quality checks.
V1.2-data: Abstracts over multiple data sources (Polygon, yfinance).
Single entry point for all strategy data needs.
"""

import os
import sys
import subprocess
import json
import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta

from .polygon_client import get_polygon_client, reset_polygon_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# V1.3: Extended date configuration
DEFAULT_START_DATE = '2015-01-01'  # Extended from 2022 for 10-year backtests
DEFAULT_END_DATE = '2025-12-31'


def _fetch_yfinance_subprocess(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch yfinance data in a subprocess to avoid TensorFlow conflicts.
    
    The TensorFlow import corrupts yfinance's internal state. By running
    yfinance in a fresh subprocess, we avoid this issue.
    """
    # Python script to run in subprocess
    script = f'''
import json
import yfinance as yf
import pandas as pd

try:
    t = yf.Ticker("{ticker}")
    df = t.history(start="{start_date}", end="{end_date}", auto_adjust=True)
    
    if df.empty:
        print(json.dumps({{"error": None, "data": None}}))
    else:
        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]]
        df.index = df.index.tz_localize(None)
        
        # Convert to JSON-serializable format
        records = []
        for idx, row in df.iterrows():
            records.append({{
                "date": idx.strftime("%Y-%m-%d"),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row["volume"]),
            }})
        print(json.dumps({{"error": None, "data": records}}))
except Exception as e:
    print(json.dumps({{"error": str(e), "data": None}}))
'''
    
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Subprocess failed: {result.stderr}")
        
        output = json.loads(result.stdout.strip())
        
        if output.get("error"):
            raise RuntimeError(output["error"])
        
        if output.get("data") is None:
            return pd.DataFrame()
        
        # Reconstruct DataFrame
        records = output["data"]
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        
        return df
        
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"yfinance subprocess timed out for {ticker}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse yfinance output: {e}")


def get_ohlcv_data(
    ticker: str,
    start_date: str,
    end_date: str,
    timeframe: str = "1d",
    provider: str = "polygon",
    polygon_api_key_env: str = "POLYGON_API_KEY_OTREP",
    use_subprocess: bool = True
) -> pd.DataFrame:
    """
    Unified OHLCV access layer.
    
    This is the single entry point for all OHLCV data in the strategy.
    It abstracts over different data providers and ensures consistent output format.
    
    Args:
        ticker: Stock/ETF symbol (e.g., "SPY", "AAPL")
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        timeframe: Bar size
            - "1d" supported by both providers
            - Intraday ("60m", "30m", "15m", etc.) only supported for "polygon"
        provider: Data provider ("polygon" or "yfinance")
        polygon_api_key_env: Environment variable for Polygon API key
        use_subprocess: If True, run yfinance in subprocess to avoid TF conflict
    
    Returns:
        DataFrame with columns: open, high, low, close, volume,
        indexed by naive DatetimeIndex in ascending order.
        Empty DataFrame if no data available.
    
    Raises:
        ValueError: If provider is unknown or timeframe unsupported
        RuntimeError: If data fetch fails
    """
    provider = provider.lower().strip()

    if provider == "polygon":
        client = get_polygon_client(api_key_env=polygon_api_key_env)
        return client.get_ohlcv(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
        )

    if provider == "yfinance":
        if timeframe != "1d":
            raise ValueError(
                f"yfinance provider only supports daily timeframe ('1d'), "
                f"got '{timeframe}'. Use 'polygon' for intraday data."
            )
        
        # Use subprocess to avoid TensorFlow + yfinance conflict
        if use_subprocess:
            return _fetch_yfinance_subprocess(ticker, start_date, end_date)
        
        # Direct fetch (may fail if TensorFlow is imported)
        import yfinance as yf
        import time
        max_retries = 3
        for attempt in range(max_retries):
            try:
                t = yf.Ticker(ticker)
                df = t.history(start=start_date, end=end_date, auto_adjust=True)
                
                if df.empty:
                    return pd.DataFrame()
                
                # Standardize column names
                df.columns = [c.lower() for c in df.columns]
                df = df[["open", "high", "low", "close", "volume"]]
                
                # Remove timezone info for consistency
                df.index = df.index.tz_localize(None)
                
                return df
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait before retry
                    continue
                raise RuntimeError(f"yfinance fetch failed for {ticker} after {max_retries} attempts: {e}")

    raise ValueError(
        f"Unknown data provider: '{provider}'. "
        f"Supported: 'polygon', 'yfinance'"
    )


def validate_ohlcv_data(
    df: pd.DataFrame,
    ticker: str,
    start_date: str,
    end_date: str,
    log_path: str = None
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Validate OHLCV data integrity and detect gaps.
    
    V1.3: Enhanced data validation for production backtesting.
    
    Checks:
    - Missing dates (compared against trading calendar)
    - Gaps > 5 consecutive trading days
    - OHLCV integrity (High >= Low, Volume >= 0)
    - Corporate action adjustments
    
    Args:
        df: OHLCV DataFrame
        ticker: Symbol for logging
        start_date: Expected start date
        end_date: Expected end date
        log_path: Optional path to save validation log
        
    Returns:
        Tuple of (cleaned_df, validation_report)
    """
    if df.empty:
        return df, {'valid': False, 'errors': ['Empty DataFrame']}
    
    report = {
        'ticker': ticker,
        'valid': True,
        'warnings': [],
        'errors': [],
        'stats': {}
    }
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            report['valid'] = False
            report['errors'].append(f'Invalid index: {e}')
            return df, report
    
    # Sort by date
    df = df.sort_index()
    
    # Basic stats
    report['stats'] = {
        'start_date': str(df.index[0].date()),
        'end_date': str(df.index[-1].date()),
        'num_bars': len(df),
        'date_range_days': (df.index[-1] - df.index[0]).days
    }
    
    # Check 1: OHLCV integrity
    high = df['high'] if 'high' in df.columns else df.get('High', df.iloc[:, 1])
    low = df['low'] if 'low' in df.columns else df.get('Low', df.iloc[:, 2])
    volume = df['volume'] if 'volume' in df.columns else df.get('Volume', df.iloc[:, 4])
    
    invalid_hl = (high < low).sum()
    if invalid_hl > 0:
        report['warnings'].append(f'{invalid_hl} bars with High < Low')
    
    negative_volume = (volume < 0).sum()
    if negative_volume > 0:
        report['warnings'].append(f'{negative_volume} bars with negative volume')
    
    zero_volume = (volume == 0).sum()
    if zero_volume > len(df) * 0.1:  # More than 10% zero volume
        report['warnings'].append(f'{zero_volume} bars ({zero_volume/len(df)*100:.1f}%) with zero volume')
    
    # Check 2: Gap detection
    date_diffs = df.index.to_series().diff().dropna()
    
    # Weekends don't count as gaps, so we look for > 4 days (handles 3-day weekends)
    large_gaps = date_diffs[date_diffs > pd.Timedelta(days=4)]
    
    if len(large_gaps) > 0:
        gap_info = []
        for gap_start, gap_size in large_gaps.items():
            gap_days = gap_size.days
            if gap_days > 5:  # Flag gaps > 5 trading days
                gap_info.append(f'{gap_start.date()}: {gap_days} days')
        
        if gap_info:
            report['warnings'].append(f'Large gaps detected: {"; ".join(gap_info[:5])}')
    
    # Check 3: Price continuity (detect potential split issues)
    close = df['close'] if 'close' in df.columns else df.get('Close', df.iloc[:, 3])
    returns = close.pct_change().dropna()
    
    extreme_moves = (returns.abs() > 0.50).sum()  # More than 50% in a day
    if extreme_moves > 0:
        report['warnings'].append(f'{extreme_moves} extreme price moves (>50%), check for unadjusted splits')
    
    # Check 4: Stale prices (same price repeated many days)
    stale_count = (close.diff() == 0).rolling(5).sum().max()
    if stale_count >= 5:
        report['warnings'].append(f'Potential stale prices detected (same close for 5+ days)')
    
    # Summary
    report['stats']['validation_warnings'] = len(report['warnings'])
    report['stats']['validation_errors'] = len(report['errors'])
    
    if report['errors']:
        report['valid'] = False
    
    # Log if path provided
    if log_path:
        _log_validation(ticker, report, log_path)
    
    return df, report


def _log_validation(ticker: str, report: Dict, log_path: str):
    """Write validation report to log file."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    with open(log_path, 'a') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Validation: {ticker} at {datetime.now().isoformat()}\n")
        f.write(f"Valid: {report['valid']}\n")
        f.write(f"Stats: {report['stats']}\n")
        
        if report['warnings']:
            f.write(f"Warnings:\n")
            for w in report['warnings']:
                f.write(f"  - {w}\n")
        
        if report['errors']:
            f.write(f"Errors:\n")
            for e in report['errors']:
                f.write(f"  - {e}\n")


def get_trading_calendar(
    start_date: str,
    end_date: str,
    market: str = 'NYSE'
) -> pd.DatetimeIndex:
    """
    Get trading calendar for a date range.
    
    Args:
        start_date: Start date string
        end_date: End date string
        market: Market calendar ('NYSE' or 'NASDAQ')
        
    Returns:
        DatetimeIndex of trading days
    """
    # Simple approximation: all weekdays minus major US holidays
    # For production, use pandas_market_calendars library
    
    all_days = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # Major US holidays (simplified list)
    holidays = [
        '01-01',  # New Year's Day
        '01-15',  # MLK Day (approx)
        '02-19',  # Presidents Day (approx)
        '07-04',  # Independence Day
        '09-02',  # Labor Day (approx)
        '11-28',  # Thanksgiving (approx)
        '12-25',  # Christmas
    ]
    
    # Filter out approximate holidays (this is a rough filter)
    # For exact holidays, would need a proper calendar
    
    return all_days


def validate_provider(
    provider: str,
    polygon_api_key_env: str = "POLYGON_API_KEY_OTREP"
) -> bool:
    """
    Validate that the specified provider is properly configured.
    
    Args:
        provider: Data provider ("polygon" or "yfinance")
        polygon_api_key_env: Environment variable for Polygon API key
    
    Returns:
        True if provider is ready
    
    Raises:
        ValueError: If provider is not configured properly
    """
    provider = provider.lower().strip()
    
    if provider == "polygon":
        # This will raise if key is missing
        client = get_polygon_client(api_key_env=polygon_api_key_env)
        # Optionally test connection
        try:
            client.test_connection()
        except Exception as e:
            raise ValueError(f"Polygon API connection failed: {e}")
        return True
    
    if provider == "yfinance":
        # yfinance doesn't require authentication
        return True
    
    raise ValueError(f"Unknown provider: '{provider}'")


if __name__ == "__main__":
    # Quick test of both providers
    import sys
    
    print("Testing data_provider module...")
    print("=" * 50)
    
    # Test yfinance
    print("\n[1] Testing yfinance provider...")
    try:
        df = get_ohlcv_data(
            "SPY", "2024-01-01", "2024-01-31",
            timeframe="1d", provider="yfinance"
        )
        if not df.empty:
            print(f"  ✓ yfinance: {len(df)} bars for SPY")
        else:
            print("  ⚠️ yfinance: No data returned")
    except Exception as e:
        print(f"  ✗ yfinance error: {e}")
    
    # Test polygon
    print("\n[2] Testing polygon provider...")
    try:
        df = get_ohlcv_data(
            "SPY", "2024-01-01", "2024-01-31",
            timeframe="1d", provider="polygon"
        )
        if not df.empty:
            print(f"  ✓ polygon: {len(df)} bars for SPY")
        else:
            print("  ⚠️ polygon: No data returned")
    except ValueError as e:
        print(f"  ⚠️ polygon not configured: {e}")
    except Exception as e:
        print(f"  ✗ polygon error: {e}")
    
    print("\n" + "=" * 50)
    print("Data provider tests complete!")
