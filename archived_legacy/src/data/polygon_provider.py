"""
Polygon.io Data Provider - Production-Grade Data Infrastructure

This provider wraps the PolygonClient with:
- Parallel batch fetching with rate limiting
- Parquet-based caching for efficiency
- Graceful fallback when API key unavailable
- Sector data from Polygon ticker details
- Clean, single-level DataFrame output (no yfinance multi-level column issues)

Usage:
    provider = PolygonDataProvider(api_key_env="POLYGON_API_KEY_OTREP")
    prices = provider.fetch_batch_parallel(tickers, start_date, end_date)
"""

import os
import time
import logging
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import threading

import pandas as pd

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Import existing PolygonClient
from src.data.polygon_client import PolygonClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# SIC code to GICS sector mapping
SIC_TO_GICS = {
    # Technology
    '737': 'Technology',  # Computer Programming
    '7370': 'Technology',
    '7371': 'Technology',
    '7372': 'Technology',  # Prepackaged Software
    '7373': 'Technology',  # Computer Integrated Systems Design
    '7374': 'Technology',  # Computer Processing
    '7377': 'Technology',  # Computer Rental & Leasing
    '3571': 'Technology',  # Electronic Computers
    '3572': 'Technology',  # Computer Storage Devices
    '3674': 'Technology',  # Semiconductors
    '3677': 'Technology',  # Electronic Coils
    
    # Healthcare
    '2834': 'Healthcare',  # Pharmaceutical Preparations
    '2836': 'Healthcare',  # Biological Products
    '3841': 'Healthcare',  # Surgical & Medical Instruments
    '3842': 'Healthcare',  # Orthopedic, Prosthetic Devices
    '3843': 'Healthcare',  # Dental Equipment
    '3844': 'Healthcare',  # X-ray Apparatus
    '3845': 'Healthcare',  # Electromedical Apparatus
    '8011': 'Healthcare',  # Offices and Clinics of Doctors
    '8062': 'Healthcare',  # General Medical and Surgical Hospitals
    
    # Financials
    '6020': 'Financials',  # Commercial Banks
    '6021': 'Financials',
    '6022': 'Financials',
    '6029': 'Financials',
    '6035': 'Financials',  # Savings Institutions
    '6141': 'Financials',  # Personal Credit Institutions
    '6211': 'Financials',  # Security Brokers
    '6282': 'Financials',  # Investment Advice
    '6311': 'Financials',  # Life Insurance
    '6331': 'Financials',  # Fire, Marine Insurance
    
    # Consumer Discretionary
    '5311': 'Consumer Discretionary',  # Department Stores
    '5331': 'Consumer Discretionary',  # Variety Stores
    '5411': 'Consumer Discretionary',  # Grocery Stores
    '5812': 'Consumer Discretionary',  # Eating Places
    '7011': 'Consumer Discretionary',  # Hotels and Motels
    '7941': 'Consumer Discretionary',  # Professional Sports Clubs
    
    # Consumer Staples
    '2000': 'Consumer Staples',  # Food Products
    '2011': 'Consumer Staples',  # Meat Packing Plants
    '2020': 'Consumer Staples',  # Dairy Products
    '2030': 'Consumer Staples',  # Canned Foods
    '2080': 'Consumer Staples',  # Beverages
    '2111': 'Consumer Staples',  # Cigarettes
    
    # Energy
    '1311': 'Energy',  # Crude Petroleum & Natural Gas
    '1381': 'Energy',  # Drilling Oil & Gas Wells
    '2911': 'Energy',  # Petroleum Refining
    '4922': 'Energy',  # Natural Gas Transmission
    '4923': 'Energy',  # Natural Gas Distribution
    
    # Industrials
    '3711': 'Industrials',  # Motor Vehicles
    '3721': 'Industrials',  # Aircraft
    '3724': 'Industrials',  # Aircraft Engines
    '4011': 'Industrials',  # Railroads
    '4512': 'Industrials',  # Air Transportation
    '4513': 'Industrials',  # Courier Services
    
    # Materials
    '2800': 'Materials',  # Chemicals
    '2810': 'Materials',  # Industrial Inorganic Chemicals
    '2820': 'Materials',  # Plastics
    '2821': 'Materials',  # Plastic Materials
    '3312': 'Materials',  # Steel Works
    '3334': 'Materials',  # Primary Aluminum
    
    # Utilities
    '4911': 'Utilities',  # Electric Services
    '4922': 'Utilities',  # Natural Gas Transmission
    '4923': 'Utilities',  # Natural Gas Distribution
    '4931': 'Utilities',  # Electric and Other Services Combined
    
    # Real Estate
    '6500': 'Real Estate',  # Real Estate
    '6510': 'Real Estate',  # Real Estate Operators
    '6512': 'Real Estate',  # Operators of Nonresidential Buildings
    '6531': 'Real Estate',  # Real Estate Agents
    '6798': 'Real Estate',  # Real Estate Investment Trusts
    
    # Communication Services
    '4812': 'Communication Services',  # Radiotelephone Communications
    '4813': 'Communication Services',  # Telephone Communications
    '4833': 'Communication Services',  # Television Broadcasting
    '4841': 'Communication Services',  # Cable and Other Pay Television
    '7812': 'Communication Services',  # Motion Picture Production
}


class PolygonDataProvider:
    """
    Production-grade data provider using Polygon.io API.
    
    Features:
    - Clean DataFrame output with standard column names
    - Parallel batch fetching with rate limiting
    - Parquet-based caching
    - Sector classification from Polygon ticker details
    - Graceful degradation when API unavailable
    """
    
    def __init__(
        self,
        api_key_env: str = "POLYGON_API_KEY_OTREP",
        cache_dir: str = "data/polygon_cache",
        n_workers: int = 10,
        rate_limit_delay: float = 0.015,  # 15ms between calls (well under 100/sec limit)
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.n_workers = n_workers
        self.rate_limit_delay = rate_limit_delay
        self.api_key_env = api_key_env
        
        # Track API state
        self._client: Optional[PolygonClient] = None
        self._api_available = None  # None = not tested, True/False = tested
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Stats
        self.stats = {
            'cache_hits': 0,
            'api_fetches': 0,
            'api_failures': 0,
        }
        
        logger.info(f"Initialized PolygonDataProvider: cache={cache_dir}, workers={n_workers}")
    
    def _get_client(self) -> Optional[PolygonClient]:
        """Get or create Polygon client. Returns None if API key not available."""
        if self._client is not None:
            return self._client
        
        try:
            self._client = PolygonClient(api_key_env=self.api_key_env)
            self._api_available = True
            logger.info("Polygon API client initialized successfully")
            return self._client
        except ValueError as e:
            self._api_available = False
            logger.warning(f"Polygon API not available: {e}")
            return None
    
    def is_api_available(self) -> bool:
        """Check if Polygon API is available."""
        if self._api_available is None:
            self._get_client()
        return self._api_available or False
    
    def _get_cache_path(self, ticker: str, start_date: str, end_date: str) -> Path:
        """Get cache file path for a ticker and date range."""
        cache_key = f"{ticker}_{start_date}_{end_date}.parquet"
        return self.cache_dir / cache_key
    
    def _load_from_cache(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Load cached data if available and fresh."""
        cache_path = self._get_cache_path(ticker, start_date, end_date)
        
        if cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                self.stats['cache_hits'] += 1
                return df
            except Exception as e:
                logger.debug(f"Cache read failed for {ticker}: {e}")
        
        return None
    
    def _save_to_cache(self, ticker: str, start_date: str, end_date: str, df: pd.DataFrame):
        """Save data to cache."""
        try:
            cache_path = self._get_cache_path(ticker, start_date, end_date)
            df.to_parquet(cache_path)
        except Exception as e:
            logger.debug(f"Cache save failed for {ticker}: {e}")
    
    def get_ohlcv(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        use_cache: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for a single ticker.
        
        Args:
            ticker: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with columns: open, high, low, close, volume
            None if fetch failed
        """
        ticker = ticker.upper()
        
        # Check cache first
        if use_cache:
            cached = self._load_from_cache(ticker, start_date, end_date)
            if cached is not None:
                return cached
        
        # Fetch from API
        client = self._get_client()
        if client is None:
            return None
        
        try:
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            df = client.get_ohlcv(ticker, start_date, end_date)
            
            if df is not None and len(df) > 0:
                self.stats['api_fetches'] += 1
                
                # Save to cache
                if use_cache:
                    self._save_to_cache(ticker, start_date, end_date, df)
                
                return df
            
            return None
            
        except Exception as e:
            self.stats['api_failures'] += 1
            logger.debug(f"API fetch failed for {ticker}: {e}")
            return None
    
    def fetch_batch_parallel(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        use_cache: bool = True,
        force_download: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple tickers in parallel.
        
        Args:
            tickers: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_cache: Whether to use cached data
            force_download: Skip cache check
            
        Returns:
            Dict of {ticker: DataFrame}
        """
        start_time = time.time()
        results = {}
        to_fetch = []
        
        # First pass: check cache
        if use_cache and not force_download:
            for ticker in tickers:
                cached = self._load_from_cache(ticker.upper(), start_date, end_date)
                if cached is not None:
                    results[ticker.upper()] = cached
                else:
                    to_fetch.append(ticker.upper())
        else:
            to_fetch = [t.upper() for t in tickers]
        
        # Check if API is available
        if to_fetch and not self.is_api_available():
            logger.warning(f"Polygon API not available. Returning {len(results)} cached results.")
            return results
        
        # Fetch remaining from API
        if to_fetch:
            logger.info(f"Fetching {len(to_fetch)} tickers from Polygon API ({len(results)} from cache)")
            
            if HAS_TQDM:
                pbar = tqdm(total=len(to_fetch), desc="Fetching from Polygon", unit="stocks")
            else:
                pbar = None
            
            def fetch_single(ticker: str) -> Tuple[str, Optional[pd.DataFrame]]:
                df = self.get_ohlcv(ticker, start_date, end_date, use_cache=True)
                return ticker, df
            
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                futures = {executor.submit(fetch_single, t): t for t in to_fetch}
                
                for future in as_completed(futures):
                    ticker, df = future.result()
                    if df is not None and len(df) > 0:
                        with self._lock:
                            results[ticker] = df
                    
                    if pbar:
                        pbar.update(1)
            
            if pbar:
                pbar.close()
        
        elapsed = time.time() - start_time
        logger.info(f"Polygon fetch complete: {len(results)}/{len(tickers)} stocks in {elapsed:.1f}s")
        logger.info(f"Stats: {self.stats['cache_hits']} cache hits, {self.stats['api_fetches']} API fetches, {self.stats['api_failures']} failures")
        
        return results
    
    def get_ticker_details(self, ticker: str) -> Dict:
        """
        Get ticker details including sector classification.
        
        Returns:
            Dict with sector, market_cap, type, etc.
        """
        client = self._get_client()
        if client is None:
            return {'sector': 'Diversified', 'market_cap': 0, 'type': 'CS'}
        
        try:
            # Use Polygon REST client directly
            import requests
            
            url = f"https://api.polygon.io/v3/reference/tickers/{ticker.upper()}"
            params = {"apiKey": client.api_key}
            
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            
            data = resp.json()
            results = data.get('results', {})
            
            # Extract sector from SIC code
            sic_code = results.get('sic_code', '')
            sector = SIC_TO_GICS.get(str(sic_code), 'Diversified')
            
            # Also try description-based matching
            sic_desc = results.get('sic_description', '').lower()
            if sector == 'Diversified':
                for keyword, mapped_sector in [
                    ('software', 'Technology'),
                    ('computer', 'Technology'),
                    ('semiconductor', 'Technology'),
                    ('pharmaceutical', 'Healthcare'),
                    ('biotech', 'Healthcare'),
                    ('medical', 'Healthcare'),
                    ('bank', 'Financials'),
                    ('insurance', 'Financials'),
                    ('investment', 'Financials'),
                    ('oil', 'Energy'),
                    ('petroleum', 'Energy'),
                    ('retail', 'Consumer Discretionary'),
                    ('restaurant', 'Consumer Discretionary'),
                    ('food', 'Consumer Staples'),
                    ('beverage', 'Consumer Staples'),
                ]:
                    if keyword in sic_desc:
                        sector = mapped_sector
                        break
            
            return {
                'sector': sector,
                'market_cap': results.get('market_cap', 0),
                'type': results.get('type', 'CS'),
                'name': results.get('name', ticker),
                'sic_code': sic_code,
                'sic_description': results.get('sic_description', ''),
            }
            
        except Exception as e:
            logger.debug(f"Ticker details failed for {ticker}: {e}")
            return {'sector': 'Diversified', 'market_cap': 0, 'type': 'CS'}
    
    def get_batch_sectors(
        self,
        tickers: List[str],
    ) -> Dict[str, str]:
        """
        Get sector classification for multiple tickers.
        
        Returns:
            Dict of {ticker: sector_name}
        """
        results = {}
        
        for ticker in tickers:
            details = self.get_ticker_details(ticker)
            results[ticker.upper()] = details['sector']
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
        
        return results
    
    def print_stats(self):
        """Print provider statistics."""
        total = sum(self.stats.values())
        print("\n" + "="*50)
        print("POLYGON DATA PROVIDER STATS")
        print("="*50)
        print(f"Cache hits:   {self.stats['cache_hits']:>6}")
        print(f"API fetches:  {self.stats['api_fetches']:>6}")
        print(f"API failures: {self.stats['api_failures']:>6}")
        print(f"Total:        {total:>6}")
        if total > 0:
            success_rate = (self.stats['cache_hits'] + self.stats['api_fetches']) / total * 100
            print(f"Success rate: {success_rate:.1f}%")
        print("="*50)


def test_polygon_provider():
    """Test the Polygon data provider."""
    print("\n" + "="*60)
    print("Testing Polygon Data Provider")
    print("="*60)
    
    provider = PolygonDataProvider()
    
    if not provider.is_api_available():
        print("\n⚠️ Polygon API key not configured!")
        print("To enable Polygon.io data:")
        print("  export POLYGON_API_KEY_OTREP=your_api_key")
        print("\nFalling back to cache-only mode.")
        return
    
    # Test single ticker
    print("\nFetching AAPL...")
    df = provider.get_ohlcv("AAPL", "2024-01-01", "2024-06-30")
    
    if df is not None:
        print(f"✅ AAPL: {len(df)} days, columns: {list(df.columns)}")
        print(f"   Date range: {df.index[0]} to {df.index[-1]}")
        print(f"   Sample close prices: {df['close'].head(3).tolist()}")
    else:
        print("❌ Failed to fetch AAPL")
    
    # Test batch
    print("\nFetching batch of 5 tickers...")
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    prices = provider.fetch_batch_parallel(tickers, "2024-01-01", "2024-06-30")
    
    print(f"✅ Fetched {len(prices)}/{len(tickers)} tickers")
    for t, df in prices.items():
        print(f"   {t}: {len(df)} days")
    
    # Test sector
    print("\nFetching ticker details for AAPL...")
    details = provider.get_ticker_details("AAPL")
    print(f"   Sector: {details['sector']}")
    print(f"   Market Cap: ${details['market_cap']:,.0f}")
    print(f"   SIC: {details['sic_code']} - {details['sic_description']}")
    
    provider.print_stats()


if __name__ == "__main__":
    test_polygon_provider()
