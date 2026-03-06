"""
Hybrid Data Provider - Production-Grade with Fallback

Priority order:
1. Polygon.io (if API key configured) - Professional grade, clean data
2. yfinance (fallback) - Free but less reliable

This ensures the system works immediately while providing a clear upgrade path.
"""

import os
import logging
from typing import Dict, List, Optional
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridDataProvider:
    """
    Hybrid data provider with automatic fallback.
    
    Uses Polygon.io when available, falls back to yfinance otherwise.
    """
    
    def __init__(
        self,
        prefer_polygon: bool = True,
        polygon_api_key_env: str = "POLYGON_API_KEY_OTREP",
    ):
        self.prefer_polygon = prefer_polygon
        self.polygon_api_key_env = polygon_api_key_env
        
        self._polygon_provider = None
        self._yfinance_available = None
        
        # Track which provider is active
        self.active_provider = None
        self._initialize_provider()
    
    def _initialize_provider(self):
        """Initialize the best available provider."""
        # Try Polygon first
        if self.prefer_polygon:
            try:
                from src.data.polygon_provider import PolygonDataProvider
                self._polygon_provider = PolygonDataProvider(
                    api_key_env=self.polygon_api_key_env
                )
                
                if self._polygon_provider.is_api_available():
                    self.active_provider = "polygon"
                    logger.info("✅ Using Polygon.io as primary data provider")
                    return
                else:
                    logger.warning("⚠️ Polygon API key not set, checking fallback...")
            except ImportError as e:
                logger.warning(f"Polygon provider not available: {e}")
        
        # Try yfinance fallback
        try:
            import yfinance as yf
            self._yfinance_available = True
            self.active_provider = "yfinance"
            logger.warning("⚠️ Using yfinance as fallback provider (not production-grade)")
            logger.warning("   To enable Polygon: export POLYGON_API_KEY_OTREP=your_key")
        except ImportError:
            self._yfinance_available = False
            self.active_provider = None
            logger.error("❌ No data provider available!")
    
    def _normalize_columns(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """Normalize column names to lowercase standard format."""
        if df is None or df.empty:
            return df
        
        # Handle multi-level columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            # Take first level (Price columns) only
            df = df.droplevel(1, axis=1) if df.columns.nlevels > 1 else df
        
        # Map to standard lowercase names
        column_map = {}
        for col in df.columns:
            col_lower = str(col).lower()
            if col_lower in ['open', 'high', 'low', 'close', 'volume', 'adj close']:
                column_map[col] = col_lower.replace(' ', '_')
        
        df = df.rename(columns=column_map)
        
        # Ensure we have required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                # Try 'adj_close' for 'close'
                if col == 'close' and 'adj_close' in df.columns:
                    df['close'] = df['adj_close']
        
        # Select only standard columns
        cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
        return df[cols]
    
    def get_ohlcv(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for a single ticker.
        
        Returns DataFrame with lowercase columns: open, high, low, close, volume
        """
        ticker = ticker.upper()
        
        if self.active_provider == "polygon" and self._polygon_provider:
            df = self._polygon_provider.get_ohlcv(ticker, start_date, end_date)
            if df is not None:
                return df
            # Fall through to yfinance on failure
            logger.debug(f"Polygon fetch failed for {ticker}, trying yfinance")
        
        if self._yfinance_available:
            try:
                import yfinance as yf
                stock = yf.Ticker(ticker)
                df = stock.history(start=start_date, end=end_date)
                
                if df is not None and len(df) > 0:
                    return self._normalize_columns(df, "yfinance")
            except Exception as e:
                logger.debug(f"yfinance fetch failed for {ticker}: {e}")
        
        return None
    
    def fetch_batch_parallel(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        n_workers: int = 10,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple tickers.
        
        Returns dict of {ticker: DataFrame} with normalized columns.
        """
        if self.active_provider == "polygon" and self._polygon_provider:
            # Polygon provider handles parallelization internally
            return self._polygon_provider.fetch_batch_parallel(
                tickers, start_date, end_date
            )
        
        if self._yfinance_available:
            return self._fetch_yfinance_batch(tickers, start_date, end_date, n_workers)
        
        logger.error("No data provider available!")
        return {}
    
    def _fetch_yfinance_batch(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        n_workers: int,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch batch using yfinance with careful column handling."""
        import yfinance as yf
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        try:
            from tqdm import tqdm
            has_tqdm = True
        except ImportError:
            has_tqdm = False
        
        results = {}
        
        def fetch_single(ticker: str) -> tuple:
            try:
                stock = yf.Ticker(ticker.upper())
                df = stock.history(start=start_date, end=end_date)
                if df is not None and len(df) > 0:
                    df = self._normalize_columns(df, "yfinance")
                    return ticker.upper(), df
            except Exception as e:
                logger.debug(f"yfinance failed for {ticker}: {e}")
            return ticker.upper(), None
        
        if has_tqdm:
            pbar = tqdm(total=len(tickers), desc="Fetching from yfinance", unit="stocks")
        else:
            pbar = None
        
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(fetch_single, t): t for t in tickers}
            
            for future in as_completed(futures):
                ticker, df = future.result()
                if df is not None:
                    results[ticker] = df
                
                if pbar:
                    pbar.update(1)
        
        if pbar:
            pbar.close()
        
        logger.info(f"yfinance batch complete: {len(results)}/{len(tickers)} stocks")
        return results
    
    def get_sector(self, ticker: str) -> str:
        """Get sector classification for a ticker."""
        if self.active_provider == "polygon" and self._polygon_provider:
            details = self._polygon_provider.get_ticker_details(ticker)
            return details.get('sector', 'Diversified')
        
        # yfinance fallback
        if self._yfinance_available:
            try:
                import yfinance as yf
                info = yf.Ticker(ticker).info
                return info.get('sector', 'Diversified')
            except Exception:
                return 'Diversified'
        
        return 'Diversified'
    
    def print_status(self):
        """Print current provider status."""
        print("\n" + "="*50)
        print("HYBRID DATA PROVIDER STATUS")
        print("="*50)
        print(f"Active Provider: {self.active_provider or 'NONE'}")
        
        if self.active_provider == "polygon":
            print("✅ Polygon.io (production-grade)")
            print("   - Clean single-level columns")
            print("   - 99.9%+ reliability")
            print("   - Enterprise SLA")
        elif self.active_provider == "yfinance":
            print("⚠️ yfinance (development fallback)")
            print("   - May have multi-level column issues")
            print("   - ~98% reliability")
            print("   - No SLA, not for production")
            print("")
            print("To upgrade to Polygon.io:")
            print("   export POLYGON_API_KEY_OTREP=your_api_key")
        else:
            print("❌ No provider available!")
        print("="*50)


def test_hybrid_provider():
    """Test hybrid provider."""
    print("\n" + "="*60)
    print("Testing Hybrid Data Provider")
    print("="*60)
    
    provider = HybridDataProvider()
    provider.print_status()
    
    # Test single ticker
    print("\nFetching AAPL...")
    df = provider.get_ohlcv("AAPL", "2024-01-01", "2024-06-30")
    
    if df is not None:
        print(f"✅ AAPL: {len(df)} days")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Date range: {df.index[0]} to {df.index[-1]}")
        
        # Verify no multi-level columns
        if isinstance(df.columns, pd.MultiIndex):
            print("❌ WARNING: Multi-level columns detected!")
        else:
            print("✅ Clean single-level columns")
    else:
        print("❌ Failed to fetch AAPL")
    
    # Test batch
    print("\nFetching batch of 5 tickers...")
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    prices = provider.fetch_batch_parallel(tickers, "2024-01-01", "2024-06-30")
    
    print(f"✅ Fetched {len(prices)}/{len(tickers)} tickers")
    for t, df in prices.items():
        print(f"   {t}: {len(df)} days, columns: {list(df.columns)}")


if __name__ == "__main__":
    test_hybrid_provider()
