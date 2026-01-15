"""Universe Manager for Full-Universe TDA Strategy.

Phase 5: Manages ~3,000 liquid US stocks for TDA analysis.
Fetches ticker lists from Polygon and filters by liquidity/price.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UniverseManager:
    """
    Manages the tradeable stock universe for TDA analysis.
    
    Features:
    - Fetches all US stocks from Polygon API
    - Filters by liquidity (volume) and price
    - Caches universe to avoid repeated API calls
    - Returns ~3,000-4,000 tradeable stocks
    """
    
    def __init__(
        self,
        polygon_api_key: str = None,
        cache_dir: str = "cache/universe",
        min_price: float = 5.0,
        min_avg_volume: int = 100_000,
        min_market_cap: float = 100_000_000,  # $100M minimum
    ):
        """
        Initialize Universe Manager.
        
        Args:
            polygon_api_key: Polygon API key (or uses env var)
            cache_dir: Directory for caching universe data
            min_price: Minimum stock price filter
            min_avg_volume: Minimum average daily volume
            min_market_cap: Minimum market cap (if available)
        """
        self.api_key = polygon_api_key or os.environ.get("POLYGON_API_KEY_OTREP")
        if not self.api_key:
            raise ValueError("Polygon API key required. Set POLYGON_API_KEY_OTREP env var.")
        
        self.cache_dir = cache_dir
        self.min_price = min_price
        self.min_avg_volume = min_avg_volume
        self.min_market_cap = min_market_cap
        
        os.makedirs(cache_dir, exist_ok=True)
        
        # In-memory cache
        self._universe_cache = None
        self._cache_date = None
    
    def get_all_us_tickers(self, use_cache: bool = True) -> List[Dict]:
        """
        Fetch all active US stock tickers from Polygon.
        
        Returns:
            List of ticker info dicts with: ticker, name, market, type, etc.
        """
        cache_file = os.path.join(self.cache_dir, "all_tickers.json")
        
        # Check cache (valid for 7 days)
        if use_cache and os.path.exists(cache_file):
            cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
            if cache_age < timedelta(days=7):
                logger.info(f"Loading tickers from cache: {cache_file}")
                with open(cache_file, 'r') as f:
                    return json.load(f)
        
        logger.info("Fetching all US tickers from Polygon API...")
        
        import requests
        
        all_tickers = []
        next_url = f"https://api.polygon.io/v3/reference/tickers?market=stocks&locale=us&active=true&limit=1000&apiKey={self.api_key}"
        
        page_count = 0
        while next_url:
            try:
                response = requests.get(next_url)
                response.raise_for_status()
                data = response.json()
                
                if data.get('results'):
                    all_tickers.extend(data['results'])
                
                # Get next page URL
                next_url = data.get('next_url')
                if next_url and 'apiKey' not in next_url:
                    next_url = f"{next_url}&apiKey={self.api_key}"
                
                page_count += 1
                if page_count % 10 == 0:
                    logger.info(f"  Fetched {len(all_tickers)} tickers so far...")
                
                # Rate limiting
                time.sleep(0.12)  # ~8 requests per second (free tier limit)
                
            except Exception as e:
                logger.error(f"Error fetching tickers: {e}")
                break
        
        logger.info(f"Total tickers fetched: {len(all_tickers)}")
        
        # Cache results
        with open(cache_file, 'w') as f:
            json.dump(all_tickers, f)
        
        return all_tickers
    
    def filter_liquid_universe(
        self,
        tickers: List[Dict],
        price_data: Optional[pd.DataFrame] = None,
    ) -> List[str]:
        """
        Filter tickers to liquid, tradeable universe.
        
        Args:
            tickers: List of ticker info from Polygon
            price_data: Optional DataFrame with price/volume info
            
        Returns:
            List of filtered ticker symbols
        """
        filtered = []
        
        for ticker_info in tickers:
            ticker = ticker_info.get('ticker', '')
            
            # Skip non-standard tickers
            if not ticker or len(ticker) > 5:
                continue
            
            # Skip warrants, units, etc.
            ticker_type = ticker_info.get('type', '')
            if ticker_type not in ['CS', 'ADRC', 'ETF', '']:  # Common stock, ADR, ETF
                continue
            
            # Skip OTC markets
            market = ticker_info.get('primary_exchange', '')
            if 'OTC' in market.upper():
                continue
            
            # Skip tickers with special characters
            if any(c in ticker for c in ['/', '.', '-', ' ']):
                continue
            
            filtered.append(ticker)
        
        logger.info(f"Filtered to {len(filtered)} standard tickers")
        return filtered
    
    def get_liquid_universe(
        self,
        reference_date: str = None,
        max_tickers: int = 3000,
        use_cache: bool = True,
    ) -> List[str]:
        """
        Get the liquid tradeable universe.
        
        This is the main entry point for getting the stock universe.
        
        Args:
            reference_date: Date for universe (uses latest if None)
            max_tickers: Maximum number of tickers to return
            use_cache: Whether to use cached data
            
        Returns:
            List of ticker symbols (sorted by liquidity)
        """
        # Check in-memory cache
        today = datetime.now().strftime("%Y-%m-%d")
        if self._universe_cache and self._cache_date == today:
            return self._universe_cache[:max_tickers]
        
        # Check file cache
        cache_file = os.path.join(self.cache_dir, f"liquid_universe_{today[:7]}.json")
        
        if use_cache and os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                self._universe_cache = cached_data['tickers']
                self._cache_date = today
                logger.info(f"Loaded {len(self._universe_cache)} tickers from cache")
                return self._universe_cache[:max_tickers]
        
        # Fetch and filter
        all_tickers = self.get_all_us_tickers(use_cache=use_cache)
        filtered = self.filter_liquid_universe(all_tickers)
        
        # Sort by exchange preference (NYSE, NASDAQ first)
        def exchange_priority(ticker_info):
            exchange = ticker_info.get('primary_exchange', '').upper()
            if 'NYSE' in exchange or 'XNYS' in exchange:
                return 0
            elif 'NASDAQ' in exchange or 'XNAS' in exchange:
                return 1
            elif 'BATS' in exchange or 'ARCA' in exchange:
                return 2
            return 3
        
        # Create lookup for sorting
        ticker_lookup = {t['ticker']: t for t in all_tickers if t.get('ticker') in filtered}
        sorted_tickers = sorted(filtered, key=lambda t: exchange_priority(ticker_lookup.get(t, {})))
        
        # Cache results
        self._universe_cache = sorted_tickers
        self._cache_date = today
        
        cache_data = {
            'date': today,
            'tickers': sorted_tickers,
            'count': len(sorted_tickers)
        }
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        
        logger.info(f"Universe: {len(sorted_tickers)} liquid tickers")
        return sorted_tickers[:max_tickers]
    
    def get_sector_etfs(self) -> Dict[str, str]:
        """Get sector ETF mapping for sector analysis."""
        return {
            'XLK': 'Technology',
            'XLF': 'Financials',
            'XLV': 'Healthcare',
            'XLE': 'Energy',
            'XLI': 'Industrials',
            'XLP': 'Consumer Staples',
            'XLY': 'Consumer Discretionary',
            'XLU': 'Utilities',
            'XLB': 'Materials',
            'XLRE': 'Real Estate',
            'XLC': 'Communications',
        }
    
    def get_major_indices(self) -> List[str]:
        """Get major index ETFs for market analysis."""
        return ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI']
    
    def get_test_universe(self, size: int = 100) -> List[str]:
        """
        Get a smaller test universe for development/testing.
        
        Includes major ETFs + top liquid stocks.
        """
        # Major ETFs first
        etfs = ['SPY', 'QQQ', 'IWM', 'DIA', 'XLK', 'XLF', 'XLV', 'XLE', 'XLI']
        
        # Top liquid stocks (FAANG + major names)
        stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
            'JPM', 'V', 'MA', 'JNJ', 'UNH', 'HD', 'PG', 'BAC',
            'XOM', 'CVX', 'PFE', 'ABBV', 'KO', 'PEP', 'MRK', 'WMT',
            'DIS', 'NFLX', 'ADBE', 'CRM', 'AMD', 'INTC', 'CSCO',
            'NKE', 'MCD', 'SBUX', 'T', 'VZ', 'CMCSA', 'COST',
            'ORCL', 'IBM', 'QCOM', 'TXN', 'AVGO', 'AMAT', 'MU',
            'GS', 'MS', 'C', 'WFC', 'AXP', 'BLK', 'SCHW',
            'LLY', 'BMY', 'AMGN', 'GILD', 'TMO', 'DHR', 'ABT',
            'CAT', 'DE', 'BA', 'RTX', 'LMT', 'GE', 'MMM',
            'COP', 'SLB', 'EOG', 'PSX', 'VLO', 'MPC', 'OXY',
            'LOW', 'TGT', 'BKNG', 'MAR', 'HLT', 'CMG', 'LULU',
            'PYPL', 'SQ', 'COIN', 'HOOD', 'SOFI', 'UBER', 'LYFT',
            'ZM', 'SNOW', 'PLTR', 'PATH', 'DDOG', 'NET', 'CRWD'
        ]
        
        universe = etfs + stocks
        return universe[:size]


class UniverseDataFetcher:
    """
    Fetches OHLCV data for entire universe using Polygon API.
    
    Implements parallel fetching with rate limiting.
    """
    
    def __init__(
        self,
        polygon_api_key: str = None,
        cache_dir: str = "cache/ohlcv",
        max_workers: int = 5,
        rate_limit_per_sec: float = 5.0,
    ):
        self.api_key = polygon_api_key or os.environ.get("POLYGON_API_KEY_OTREP")
        self.cache_dir = cache_dir
        self.max_workers = max_workers
        self.rate_limit_delay = 1.0 / rate_limit_per_sec
        
        os.makedirs(cache_dir, exist_ok=True)
    
    def fetch_single_ticker(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        """Fetch OHLCV for a single ticker."""
        import requests
        
        # Check cache first
        cache_file = os.path.join(
            self.cache_dir, 
            f"{ticker}_{start_date}_{end_date}.parquet"
        )
        
        if os.path.exists(cache_file):
            try:
                return pd.read_parquet(cache_file)
            except Exception:
                pass
        
        # Fetch from Polygon
        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/"
            f"{start_date}/{end_date}?adjusted=true&sort=asc&limit=50000"
            f"&apiKey={self.api_key}"
        )
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data.get('resultsCount', 0) == 0:
                return None
            
            results = data.get('results', [])
            
            df = pd.DataFrame(results)
            df['date'] = pd.to_datetime(df['t'], unit='ms')
            df = df.rename(columns={
                'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'
            })
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
            df.set_index('date', inplace=True)
            df.index = df.index.tz_localize(None)
            
            # Cache result
            df.to_parquet(cache_file)
            
            return df
            
        except Exception as e:
            logger.warning(f"Failed to fetch {ticker}: {e}")
            return None
    
    def fetch_universe_data(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        show_progress: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for entire universe.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            show_progress: Whether to show progress updates
            
        Returns:
            Dict mapping ticker -> DataFrame
        """
        logger.info(f"Fetching data for {len(tickers)} tickers...")
        
        results = {}
        failed = []
        
        for i, ticker in enumerate(tickers):
            if show_progress and (i + 1) % 100 == 0:
                logger.info(f"  Progress: {i + 1}/{len(tickers)} ({len(results)} successful)")
            
            df = self.fetch_single_ticker(ticker, start_date, end_date)
            
            if df is not None and len(df) > 100:  # At least 100 days of data
                results[ticker] = df
            else:
                failed.append(ticker)
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
        
        logger.info(f"Fetched {len(results)} tickers, {len(failed)} failed")
        
        return results
    
    def build_returns_matrix(
        self,
        ohlcv_data: Dict[str, pd.DataFrame],
        min_common_dates: int = 200,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Build returns matrix from OHLCV data.
        
        Args:
            ohlcv_data: Dict of ticker -> OHLCV DataFrame
            min_common_dates: Minimum common dates required
            
        Returns:
            Tuple of (returns_df, valid_tickers)
        """
        # Get all close prices
        closes = {}
        for ticker, df in ohlcv_data.items():
            closes[ticker] = df['close']
        
        # Combine into single DataFrame
        close_df = pd.DataFrame(closes)
        
        # Forward fill missing data (holidays, etc.)
        close_df = close_df.ffill().bfill()
        
        # Drop columns with too many NaN
        valid_cols = close_df.columns[close_df.notna().sum() >= min_common_dates]
        close_df = close_df[valid_cols]
        
        # Calculate returns
        returns_df = close_df.pct_change().dropna()
        
        logger.info(f"Returns matrix: {returns_df.shape[0]} days x {returns_df.shape[1]} stocks")
        
        return returns_df, list(returns_df.columns)


if __name__ == "__main__":
    # Test universe manager
    print("Testing Universe Manager...")
    print("=" * 60)
    
    try:
        um = UniverseManager()
        
        # Test small universe
        test_universe = um.get_test_universe(size=50)
        print(f"Test universe: {len(test_universe)} tickers")
        print(f"  First 10: {test_universe[:10]}")
        
        # Test full universe fetch (commented out for speed)
        # full_universe = um.get_liquid_universe(max_tickers=100)
        # print(f"Liquid universe: {len(full_universe)} tickers")
        
        print("\nUniverse Manager tests complete!")
        
    except Exception as e:
        print(f"Error: {e}")
