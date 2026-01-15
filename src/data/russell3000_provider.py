"""
Russell 3000 Data Provider - Phase 7 Scalable Infrastructure.

Provides efficient data fetching for large universes:
- Multi-threaded parallel downloads (20 workers)
- Smart caching with parquet format
- Progress tracking with tqdm
- Memory-efficient batch processing
- Robust error handling (skip failures, continue)

Target: Fetch 500-1000 liquid stocks in <5 minutes
"""

import os
import logging
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Liquid subset of Russell 3000 - ~500 most liquid stocks across sectors
# This is a curated list for MVP, representing major market cap tiers
LIQUID_UNIVERSE = [
    # Technology (100 stocks)
    "AAPL", "MSFT", "GOOGL", "GOOG", "META", "NVDA", "AVGO", "ORCL", "CSCO", "CRM",
    "AMD", "ADBE", "ACN", "INTC", "IBM", "TXN", "QCOM", "NOW", "INTU", "AMAT",
    "ADI", "LRCX", "MU", "PANW", "SNPS", "KLAC", "CDNS", "MRVL", "ROP", "FTNT",
    "ADSK", "NXPI", "MCHP", "APH", "CTSH", "IT", "KEYS", "ANSS", "FSLR", "HPQ",
    "ON", "ZBRA", "NTAP", "TYL", "EPAM", "CDW", "JNPR", "WDC", "AKAM", "SWKS",
    "FFIV", "ENPH", "QRVO", "GEN", "TRMB", "TER", "GDDY", "VRSN", "LOGI", "SMCI",
    "ANET", "CRWD", "DDOG", "ZS", "SNOW", "NET", "MDB", "TEAM", "OKTA", "ZM",
    "DOCU", "TWLO", "SPLK", "HUBS", "VEEV", "PAYC", "BILL", "DXCM", "MPWR", "SEDG",
    "WOLF", "ALGN", "MTCH", "RBLX", "U", "COIN", "PATH", "CFLT", "IOT", "AI",
    "PLTR", "DELL", "HPE", "ESTC", "PTC", "MANH", "BSY", "GLOB", "PEGA", "SSNC",
    
    # Financials (80 stocks)
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "SPGI",
    "PNC", "USB", "TFC", "COF", "BK", "CME", "ICE", "MCO", "CB", "MMC",
    "AON", "MET", "PRU", "AIG", "TRV", "ALL", "AFL", "PGR", "AJG", "HIG",
    "CINF", "WRB", "L", "GL", "RJF", "NTRS", "STT", "CFG", "FITB", "RF",
    "HBAN", "KEY", "MTB", "ZION", "CMA", "FHN", "SNV", "WAL", "SIVB", "SBNY",
    "FRC", "NDAQ", "CBOE", "MSCI", "FDS", "MKTX", "VIRT", "EVR", "HLI", "LAZ",
    "SF", "JEF", "IBKR", "SEIC", "LPLA", "RNR", "EG", "W", "RYAN", "BRO",
    "ERIE", "WTW", "AIZ", "UNM", "LNC", "VOYA", "PFG", "TROW", "BEN", "IVZ",
    
    # Healthcare (80 stocks)
    "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY",
    "AMGN", "GILD", "CVS", "ELV", "CI", "ISRG", "VRTX", "REGN", "MDT", "SYK",
    "BSX", "BDX", "EW", "ZBH", "IDXX", "IQV", "MTD", "A", "DXCM", "WST",
    "RMD", "HOLX", "BAX", "TECH", "TFX", "HSIC", "XRAY", "ALGN", "PODD", "NUVA",
    "MRNA", "BIIB", "ILMN", "ALNY", "SGEN", "BMRN", "INCY", "EXAS", "RARE", "IONS",
    "NBIX", "HZNP", "JAZZ", "UTHR", "SRPT", "BIO", "MEDP", "ICLR", "CRL", "PRGO",
    "CAH", "MCK", "ABC", "HCA", "CNC", "MOH", "HUM", "CERN", "VEEV", "ZTS",
    "IDEXX", "RVTY", "PKI", "WAT", "DGX", "LH", "DVA", "THC", "EHAB", "ACHC",
    
    # Consumer Discretionary (60 stocks)
    "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "CMG",
    "ORLY", "AZO", "ROST", "MAR", "HLT", "YUM", "DG", "DLTR", "ULTA", "BBY",
    "EBAY", "ETSY", "W", "CPRT", "POOL", "LKQ", "GPC", "AAP", "AN", "KMX",
    "TSCO", "WSM", "RH", "FIVE", "GRMN", "DRI", "LVS", "WYNN", "MGM", "CZR",
    "RCL", "CCL", "NCLH", "EXPE", "ABNB", "HAS", "MAT", "NWL", "LEG", "WHR",
    "F", "GM", "APTV", "BWA", "LEA", "VC", "DAN", "GNTX", "ALV", "LCII",
    
    # Industrials (60 stocks)
    "UNP", "UPS", "HON", "RTX", "CAT", "DE", "BA", "LMT", "GE", "MMM",
    "FDX", "EMR", "ITW", "ETN", "NSC", "CSX", "PCAR", "PH", "CMI", "ROK",
    "ODFL", "JBHT", "CHRW", "XPO", "EXPD", "LSTR", "SAIA", "KNX", "WERN", "SNDR",
    "WM", "RSG", "WCN", "CLH", "SRCL", "CTAS", "PAYX", "ADP", "CPAY", "BR",
    "VRSK", "INFO", "TRI", "GWW", "FAST", "WSO", "MSM", "WCC", "SITE", "SWK",
    "TT", "CARR", "LII", "IR", "DOV", "XYL", "IEX", "GNRC", "AME", "NDSN",
    
    # Consumer Staples (40 stocks)
    "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "CL", "EL", "MDLZ",
    "KMB", "GIS", "K", "CAG", "SJM", "HSY", "MKC", "CHD", "CLX", "CPB",
    "HRL", "TSN", "KHC", "ADM", "BG", "INGR", "DAR", "STZ", "BF.B", "TAP",
    "MNST", "CCEP", "KDP", "WBA", "KR", "SYY", "USFD", "PFGC", "CHEF", "LANC",
    
    # Energy (40 stocks)
    "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "PXD", "OXY",
    "HAL", "BKR", "FANG", "HES", "DVN", "MRO", "APA", "OVV", "CTRA", "EQT",
    "AR", "RRC", "SWN", "MTDR", "PR", "TRGP", "WMB", "KMI", "OKE", "ET",
    "EPD", "MPLX", "PAA", "CEQP", "ENLC", "DTM", "AM", "HESM", "USAC", "AROC",
    
    # Utilities (30 stocks)
    "NEE", "DUK", "SO", "D", "SRE", "AEP", "EXC", "XEL", "WEC", "ED",
    "PEG", "ES", "EIX", "DTE", "FE", "CMS", "CNP", "AEE", "PPL", "ATO",
    "NI", "EVRG", "OGE", "NRG", "VST", "CEG", "AWK", "WTR", "SJW", "WTRG",
    
    # Materials (30 stocks)
    "LIN", "APD", "SHW", "ECL", "FCX", "NEM", "NUE", "DD", "PPG", "VMC",
    "MLM", "DOW", "LYB", "CTVA", "ALB", "EMN", "CE", "WRK", "IP", "PKG",
    "AVY", "SON", "SEE", "BLL", "CCK", "AMCR", "SLVM", "RPM", "FUL", "CBT",
    
    # Real Estate (30 stocks)
    "PLD", "AMT", "CCI", "EQIX", "PSA", "O", "SPG", "WELL", "DLR", "AVB",
    "EQR", "VTR", "ARE", "ESS", "MAA", "UDR", "CPT", "AIV", "IRM", "WY",
    "EXR", "CUBE", "REXR", "STAG", "TRNO", "PEB", "RHP", "SHO", "HST", "PEAK",
    
    # Communication Services (30 stocks)
    "GOOG", "META", "NFLX", "DIS", "CMCSA", "VZ", "T", "TMUS", "CHTR", "ATVI",
    "EA", "TTWO", "WBD", "PARA", "FOX", "FOXA", "NWS", "NWSA", "OMC", "IPG",
    "LYV", "SPOT", "ROKU", "PINS", "SNAP", "MTCH", "ZG", "TRIP", "IAC", "ANGI",
]

# Remove duplicates and clean
LIQUID_UNIVERSE = list(dict.fromkeys([t for t in LIQUID_UNIVERSE if t and isinstance(t, str)]))


class Russell3000DataProvider:
    """
    Scalable data provider for Russell 3000 universe.
    
    Features:
    - Multi-threaded parallel downloads
    - Smart parquet caching
    - Progress tracking
    - Batch processing for memory efficiency
    - Robust error handling
    """
    
    def __init__(
        self,
        cache_dir: str = 'data/cache',
        refresh_days: int = 7,
        n_workers: int = 20,
        batch_size: int = 100,
    ):
        """
        Initialize data provider.
        
        Args:
            cache_dir: Directory for cached data
            refresh_days: Days before cache is considered stale
            n_workers: Number of parallel download workers
            batch_size: Stocks per batch for memory management
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.refresh_days = refresh_days
        self.n_workers = n_workers
        self.batch_size = batch_size
        
        self.failed_tickers: Set[str] = set()
        self.fetch_stats = {
            'cache_hits': 0,
            'downloads': 0,
            'failures': 0,
            'total_time': 0.0,
        }
        
        logger.info(f"Initialized Russell3000DataProvider: cache={self.cache_dir}, workers={n_workers}")
    
    def get_universe_list(self, size: str = 'liquid') -> List[str]:
        """
        Get list of tickers in universe.
        
        Args:
            size: 'liquid' (~500), 'medium' (~1000), 'full' (~3000)
            
        Returns:
            List of ticker symbols
        """
        if size == 'liquid':
            # Curated liquid subset
            return LIQUID_UNIVERSE[:500]
        elif size == 'medium':
            return LIQUID_UNIVERSE[:1000]
        elif size == 'full':
            # Would fetch from API in production
            return LIQUID_UNIVERSE
        else:
            return LIQUID_UNIVERSE[:100]  # Debug subset
    
    def _get_cache_path(self, ticker: str) -> Path:
        """Get cache file path for ticker."""
        return self.cache_dir / f"{ticker}.parquet"
    
    def _get_metadata_path(self) -> Path:
        """Get metadata file path."""
        return self.cache_dir / "_metadata.pkl"
    
    def _is_cache_fresh(self, ticker: str) -> bool:
        """Check if cached data is fresh enough."""
        cache_path = self._get_cache_path(ticker)
        if not cache_path.exists():
            return False
        
        # Check file modification time
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age_days = (datetime.now() - mtime).days
        
        return age_days < self.refresh_days
    
    def load_from_cache(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Load data from cache if available and fresh.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            DataFrame if cache hit, None otherwise
        """
        if not self._is_cache_fresh(ticker):
            return None
        
        try:
            cache_path = self._get_cache_path(ticker)
            df = pd.read_parquet(cache_path)
            self.fetch_stats['cache_hits'] += 1
            return df
        except Exception as e:
            logger.debug(f"Cache read failed for {ticker}: {e}")
            return None
    
    def save_to_cache(self, ticker: str, df: pd.DataFrame) -> bool:
        """
        Save data to parquet cache.
        
        Args:
            ticker: Stock symbol
            df: OHLCV DataFrame
            
        Returns:
            True if successful
        """
        try:
            cache_path = self._get_cache_path(ticker)
            df.to_parquet(cache_path, compression='snappy')
            return True
        except Exception as e:
            logger.debug(f"Cache write failed for {ticker}: {e}")
            return False
    
    def _fetch_single_ticker(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        force_download: bool = False,
    ) -> Tuple[str, Optional[pd.DataFrame]]:
        """
        Fetch data for single ticker with cache check.
        
        Args:
            ticker: Stock symbol
            start_date: Start date string
            end_date: End date string
            force_download: Skip cache
            
        Returns:
            Tuple of (ticker, DataFrame or None)
        """
        # Try cache first
        if not force_download:
            cached = self.load_from_cache(ticker)
            if cached is not None:
                # Verify date range
                if len(cached) > 0:
                    cached_start = cached.index.min()
                    cached_end = cached.index.max()
                    
                    # If cached data covers our range, use it
                    if (pd.Timestamp(start_date) >= cached_start - timedelta(days=5) and
                        pd.Timestamp(end_date) <= cached_end + timedelta(days=1)):
                        return (ticker, cached)
        
        # Download from yfinance
        if not HAS_YFINANCE:
            logger.error("yfinance not installed")
            return (ticker, None)
        
        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True,
            )
            
            if df is None or len(df) == 0:
                self.failed_tickers.add(ticker)
                self.fetch_stats['failures'] += 1
                return (ticker, None)
            
            # Standardize column names
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower() for c in df.columns]
            
            # Ensure required columns
            required = ['open', 'high', 'low', 'close', 'volume']
            if not all(c in df.columns for c in required):
                self.failed_tickers.add(ticker)
                self.fetch_stats['failures'] += 1
                return (ticker, None)
            
            # Save to cache
            self.save_to_cache(ticker, df)
            self.fetch_stats['downloads'] += 1
            
            return (ticker, df)
            
        except Exception as e:
            logger.debug(f"Download failed for {ticker}: {e}")
            self.failed_tickers.add(ticker)
            self.fetch_stats['failures'] += 1
            return (ticker, None)
    
    def fetch_batch_parallel(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        force_download: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple tickers in parallel.
        
        Args:
            tickers: List of stock symbols
            start_date: Start date string
            end_date: End date string
            force_download: Skip cache check
            
        Returns:
            Dict of {ticker: DataFrame}
        """
        import time
        start_time = time.time()
        
        results = {}
        
        # Use tqdm for progress if available
        if HAS_TQDM:
            pbar = tqdm(total=len(tickers), desc="Fetching data", unit="stocks")
        else:
            pbar = None
        
        # Process in batches to manage memory
        for batch_start in range(0, len(tickers), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(tickers))
            batch_tickers = tickers[batch_start:batch_end]
            
            if not pbar:
                logger.info(f"Fetching batch {batch_start//self.batch_size + 1}: "
                          f"tickers {batch_start+1}-{batch_end}")
            
            # Parallel fetch
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                futures = {
                    executor.submit(
                        self._fetch_single_ticker,
                        ticker,
                        start_date,
                        end_date,
                        force_download
                    ): ticker
                    for ticker in batch_tickers
                }
                
                for future in as_completed(futures):
                    ticker, df = future.result()
                    if df is not None and len(df) > 0:
                        results[ticker] = df
                    
                    if pbar:
                        pbar.update(1)
        
        if pbar:
            pbar.close()
        
        elapsed = time.time() - start_time
        self.fetch_stats['total_time'] = elapsed
        
        logger.info(f"Fetch complete: {len(results)}/{len(tickers)} stocks in {elapsed:.1f}s")
        logger.info(f"Stats: {self.fetch_stats['cache_hits']} cache hits, "
                   f"{self.fetch_stats['downloads']} downloads, "
                   f"{self.fetch_stats['failures']} failures")
        
        return results
    
    def fetch_universe(
        self,
        universe_size: str = 'liquid',
        start_date: str = '2021-01-01',
        end_date: str = '2024-12-31',
        force_download: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch full universe data.
        
        Args:
            universe_size: 'liquid', 'medium', or 'full'
            start_date: Backtest start date
            end_date: Backtest end date
            force_download: Force re-download
            
        Returns:
            Dict of {ticker: DataFrame}
        """
        tickers = self.get_universe_list(universe_size)
        
        # Add buffer for lookback periods (1 year)
        buffer_start = (pd.Timestamp(start_date) - timedelta(days=365)).strftime('%Y-%m-%d')
        
        logger.info(f"Fetching {len(tickers)} stocks from {buffer_start} to {end_date}")
        
        return self.fetch_batch_parallel(
            tickers,
            buffer_start,
            end_date,
            force_download,
        )
    
    def get_fetch_stats(self) -> Dict:
        """Get fetching statistics."""
        return {
            **self.fetch_stats,
            'failed_tickers': list(self.failed_tickers),
            'n_failed': len(self.failed_tickers),
        }
    
    def clear_cache(self, older_than_days: int = None):
        """
        Clear cache files.
        
        Args:
            older_than_days: Only clear files older than N days (None=all)
        """
        count = 0
        for f in self.cache_dir.glob("*.parquet"):
            if older_than_days:
                mtime = datetime.fromtimestamp(f.stat().st_mtime)
                age = (datetime.now() - mtime).days
                if age < older_than_days:
                    continue
            f.unlink()
            count += 1
        
        logger.info(f"Cleared {count} cache files")


def print_provider_test():
    """Test the data provider."""
    print("\n" + "="*60)
    print("Phase 7: Russell 3000 Data Provider Test")
    print("="*60)
    
    provider = Russell3000DataProvider(
        cache_dir='data/cache',
        n_workers=10,
        batch_size=50,
    )
    
    # Test with small subset
    test_tickers = provider.get_universe_list('liquid')[:20]
    print(f"\nTest fetching {len(test_tickers)} stocks...")
    
    data = provider.fetch_batch_parallel(
        test_tickers,
        start_date='2023-01-01',
        end_date='2024-12-31',
    )
    
    print(f"\nSuccessfully fetched: {len(data)} stocks")
    
    # Print sample
    if data:
        sample_ticker = list(data.keys())[0]
        sample_df = data[sample_ticker]
        print(f"\nSample ({sample_ticker}):")
        print(f"  Date range: {sample_df.index.min()} to {sample_df.index.max()}")
        print(f"  Rows: {len(sample_df)}")
        print(f"  Columns: {list(sample_df.columns)}")
    
    # Print stats
    stats = provider.get_fetch_stats()
    print(f"\nFetch Statistics:")
    print(f"  Cache hits: {stats['cache_hits']}")
    print(f"  Downloads: {stats['downloads']}")
    print(f"  Failures: {stats['failures']}")
    print(f"  Total time: {stats['total_time']:.2f}s")
    
    if stats['failed_tickers']:
        print(f"  Failed tickers: {stats['failed_tickers'][:10]}...")


if __name__ == "__main__":
    print_provider_test()
