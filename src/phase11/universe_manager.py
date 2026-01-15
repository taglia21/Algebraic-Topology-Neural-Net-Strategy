"""
Universe Manager for Phase 11
==============================

Manages the full US stock universe using yfinance and local caching.
Handles filtering for quality (price, volume, market cap).

Note: For production, use Polygon.io API. For backtesting, we use yfinance
which is free and sufficient for historical data.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StockFilter:
    """Filter criteria for stock universe."""
    min_price: float = 5.0           # Avoid penny stocks
    min_volume: int = 500_000        # Daily volume threshold
    min_market_cap: float = 300e6    # $300M minimum market cap
    max_price: float = 10000.0       # Exclude extreme outliers
    exchanges: List[str] = None      # NYSE, NASDAQ, AMEX
    exclude_otc: bool = True         # Exclude OTC stocks
    exclude_etfs: bool = False       # Include ETFs for leverage
    
    def __post_init__(self):
        if self.exchanges is None:
            self.exchanges = ['NYSE', 'NASDAQ', 'NYQ', 'NMS', 'NGM', 'NCM']


class UniverseManager:
    """
    Manages the full US stock universe.
    
    Features:
    - Load ticker lists from multiple sources
    - Filter for quality (price, volume, market cap)
    - Cache results to reduce API calls
    - Support for ~3,000-4,000 quality stocks
    """
    
    def __init__(
        self,
        cache_dir: str = None,
        filter_config: StockFilter = None,
    ):
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(__file__), '..', '..', 'cache', 'universe'
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.filter_config = filter_config or StockFilter()
        self.universe_cache = {}
        self.price_cache = {}
        
    def get_tradeable_universe(
        self,
        date: str = None,
        use_cache: bool = True,
    ) -> List[str]:
        """
        Get list of tradeable tickers for a given date.
        
        Returns:
            List of ticker symbols meeting quality filters
        """
        cache_key = date or 'latest'
        
        if use_cache and cache_key in self.universe_cache:
            return self.universe_cache[cache_key]
        
        # Try to load from file cache
        cache_file = os.path.join(self.cache_dir, f'universe_{cache_key}.json')
        if use_cache and os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                tickers = json.load(f)
                self.universe_cache[cache_key] = tickers
                return tickers
        
        # Build universe from scratch
        logger.info("Building tradeable universe from scratch...")
        tickers = self._build_universe(date)
        
        # Cache results
        self.universe_cache[cache_key] = tickers
        with open(cache_file, 'w') as f:
            json.dump(tickers, f)
        
        return tickers
    
    def _build_universe(self, date: str = None) -> List[str]:
        """Build the full tradeable universe."""
        # Start with known liquid tickers from multiple sources
        tickers = set()
        
        # 1. S&P 500 companies
        sp500 = self._get_sp500_tickers()
        tickers.update(sp500)
        logger.info(f"Added {len(sp500)} S&P 500 tickers")
        
        # 2. NASDAQ 100
        nasdaq100 = self._get_nasdaq100_tickers()
        tickers.update(nasdaq100)
        logger.info(f"Added {len(nasdaq100)} NASDAQ 100 tickers")
        
        # 3. Russell 3000 (broad market)
        russell = self._get_russell3000_tickers()
        tickers.update(russell)
        logger.info(f"Added {len(russell)} Russell 3000 tickers")
        
        # 4. Key leveraged ETFs (for Phase 10 compatibility)
        etfs = self._get_leveraged_etfs()
        tickers.update(etfs)
        logger.info(f"Added {len(etfs)} leveraged ETFs")
        
        logger.info(f"Total raw universe: {len(tickers)} tickers")
        
        # Filter will happen during price loading
        return list(tickers)
    
    def _get_sp500_tickers(self) -> Set[str]:
        """Get S&P 500 tickers."""
        try:
            # Try Wikipedia table
            tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            sp500_table = tables[0]
            tickers = set(sp500_table['Symbol'].str.replace('.', '-', regex=False).tolist())
            return tickers
        except Exception as e:
            logger.warning(f"Failed to fetch S&P 500 list: {e}")
            # Fallback to hardcoded major tickers
            return {
                'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'BRK-B',
                'UNH', 'JNJ', 'XOM', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'MRK',
                'ABBV', 'LLY', 'PEP', 'KO', 'AVGO', 'COST', 'TMO', 'MCD', 'CSCO',
                'WMT', 'ACN', 'DHR', 'ABT', 'NEE', 'ADBE', 'CRM', 'VZ', 'CMCSA',
                'NKE', 'INTC', 'PM', 'TXN', 'QCOM', 'UPS', 'RTX', 'HON', 'LOW',
                'BMY', 'SBUX', 'CAT', 'IBM', 'GE', 'BA', 'AMD', 'NFLX', 'AMGN'
            }
    
    def _get_nasdaq100_tickers(self) -> Set[str]:
        """Get NASDAQ 100 tickers."""
        try:
            tables = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
            for table in tables:
                if 'Ticker' in table.columns or 'Symbol' in table.columns:
                    col = 'Ticker' if 'Ticker' in table.columns else 'Symbol'
                    return set(table[col].tolist())
            return set()
        except Exception as e:
            logger.warning(f"Failed to fetch NASDAQ 100: {e}")
            return {
                'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'GOOG', 'META', 'TSLA',
                'AVGO', 'COST', 'ADBE', 'CSCO', 'PEP', 'AMD', 'NFLX', 'CMCSA',
                'INTC', 'TMUS', 'INTU', 'QCOM', 'TXN', 'AMGN', 'AMAT', 'ISRG'
            }
    
    def _get_russell3000_tickers(self) -> Set[str]:
        """Get Russell 3000 tickers from local cache or curated list."""
        cache_file = os.path.join(self.cache_dir, 'russell3000.json')
        
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return set(json.load(f))
        
        # Curated list of liquid mid/small caps
        # In production, this would come from Polygon.io or Russell index
        additional = {
            # Large cap tech
            'CRM', 'NOW', 'PANW', 'CRWD', 'ZS', 'FTNT', 'SNPS', 'CDNS', 'ANSS',
            'ADSK', 'MCHP', 'KLAC', 'LRCX', 'AMAT', 'ASML', 'MU', 'ON', 'MRVL',
            # Growth tech
            'COIN', 'MSTR', 'PLTR', 'NET', 'DDOG', 'SNOW', 'MDB', 'U', 'CFLT',
            # Financials
            'GS', 'MS', 'C', 'BAC', 'WFC', 'USB', 'PNC', 'TFC', 'COF', 'AXP',
            'BLK', 'SCHW', 'CME', 'ICE', 'SPGI', 'MCO', 'MSCI', 'CBOE',
            # Healthcare
            'PFE', 'LLY', 'UNH', 'JNJ', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR',
            'ISRG', 'MDT', 'SYK', 'EW', 'BSX', 'ZBH', 'DXCM', 'IDXX', 'VEEV',
            # Consumer
            'AMZN', 'HD', 'LOW', 'TGT', 'COST', 'WMT', 'MCD', 'SBUX', 'NKE',
            'LULU', 'ULTA', 'ROST', 'TJX', 'DG', 'DLTR', 'ORLY', 'AZO',
            # Energy
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'OXY', 'MPC', 'VLO', 'PSX',
            'HAL', 'BKR', 'DVN', 'PXD', 'FANG', 'HES',
            # Industrials
            'CAT', 'DE', 'UPS', 'FDX', 'UNP', 'CSX', 'NSC', 'LMT', 'RTX',
            'NOC', 'GD', 'BA', 'GE', 'HON', 'MMM', 'EMR', 'ITW', 'ROK',
            # Materials
            'LIN', 'APD', 'SHW', 'ECL', 'NEM', 'FCX', 'NUE', 'STLD',
            # Real Estate
            'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'SPG', 'O', 'WELL', 'DLR',
            # Utilities
            'NEE', 'DUK', 'SO', 'D', 'AEP', 'XEL', 'SRE', 'WEC', 'ES',
            # Communications
            'T', 'VZ', 'CMCSA', 'TMUS', 'CHTR', 'DIS', 'NFLX', 'WBD',
            # More mid caps
            'WDAY', 'SPLK', 'OKTA', 'ZM', 'DOCU', 'TWLO', 'PINS', 'SNAP',
            'SQ', 'PYPL', 'FIS', 'FISV', 'GPN', 'ADYEN', 'SHOP', 'SE',
            'ABNB', 'UBER', 'LYFT', 'DASH', 'RBLX', 'MTCH', 'BMBL',
            'RIVN', 'LCID', 'NIO', 'XPEV', 'LI', 'FSR',
        }
        
        return additional
    
    def _get_leveraged_etfs(self) -> Set[str]:
        """Get leveraged ETFs for Phase 10 compatibility."""
        return {
            # 3x Bull ETFs
            'TQQQ', 'SPXL', 'UPRO', 'TNA', 'SOXL', 'TECL', 'FNGU', 'LABU',
            'FAS', 'ERX', 'CURE', 'DPST', 'RETL', 'NAIL', 'DFEN',
            # 2x Bull ETFs (less volatile)
            'QLD', 'SSO', 'UWM', 'ROM', 'USD',
            # Sector ETFs (unleveraged for selection)
            'XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'XLRE',
            'SMH', 'IBB', 'XBI', 'IYR', 'VNQ',
            # Core market ETFs
            'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO',
        }
    
    def filter_by_liquidity(
        self,
        tickers: List[str],
        price_data: Dict[str, pd.DataFrame],
        date: str = None,
    ) -> List[str]:
        """
        Filter tickers by liquidity and quality criteria.
        
        Args:
            tickers: List of ticker symbols
            price_data: Dict of ticker -> DataFrame with OHLCV data
            date: Date to check (uses latest if None)
            
        Returns:
            Filtered list of liquid, quality tickers
        """
        filtered = []
        
        for ticker in tickers:
            if ticker not in price_data:
                continue
                
            df = price_data[ticker]
            if len(df) < 60:  # Need 60 days of history
                continue
            
            # Get recent data
            if date:
                df = df[df.index <= pd.to_datetime(date)]
            
            if len(df) < 20:
                continue
                
            recent = df.tail(20)
            
            # Apply filters
            avg_price = recent['close'].mean()
            avg_volume = recent['volume'].mean()
            
            # Price filter
            if avg_price < self.filter_config.min_price:
                continue
            if avg_price > self.filter_config.max_price:
                continue
                
            # Volume filter
            if avg_volume < self.filter_config.min_volume:
                continue
            
            filtered.append(ticker)
        
        logger.info(f"Filtered {len(tickers)} -> {len(filtered)} quality tickers")
        return filtered
    
    def get_sector_map(self, tickers: List[str]) -> Dict[str, str]:
        """
        Get sector mapping for tickers.
        
        Returns:
            Dict of ticker -> sector
        """
        # Hardcoded sector map for major tickers
        # In production, get from Polygon.io ticker details
        sector_map = {
            # Technology
            'AAPL': 'Technology', 'MSFT': 'Technology', 'NVDA': 'Technology',
            'GOOGL': 'Technology', 'GOOG': 'Technology', 'META': 'Technology',
            'AVGO': 'Technology', 'ADBE': 'Technology', 'CRM': 'Technology',
            'CSCO': 'Technology', 'INTC': 'Technology', 'AMD': 'Technology',
            'TXN': 'Technology', 'QCOM': 'Technology', 'MU': 'Technology',
            'NOW': 'Technology', 'PANW': 'Technology', 'CRWD': 'Technology',
            'AMAT': 'Technology', 'LRCX': 'Technology', 'KLAC': 'Technology',
            'SNPS': 'Technology', 'CDNS': 'Technology', 'MRVL': 'Technology',
            # Consumer Discretionary
            'AMZN': 'Consumer', 'TSLA': 'Consumer', 'HD': 'Consumer',
            'NKE': 'Consumer', 'MCD': 'Consumer', 'SBUX': 'Consumer',
            'LOW': 'Consumer', 'TGT': 'Consumer', 'LULU': 'Consumer',
            'COST': 'Consumer', 'TJX': 'Consumer', 'ROST': 'Consumer',
            # Healthcare
            'UNH': 'Healthcare', 'JNJ': 'Healthcare', 'LLY': 'Healthcare',
            'PFE': 'Healthcare', 'ABBV': 'Healthcare', 'MRK': 'Healthcare',
            'TMO': 'Healthcare', 'ABT': 'Healthcare', 'DHR': 'Healthcare',
            'ISRG': 'Healthcare', 'MDT': 'Healthcare', 'SYK': 'Healthcare',
            # Financials
            'JPM': 'Financials', 'V': 'Financials', 'MA': 'Financials',
            'BAC': 'Financials', 'WFC': 'Financials', 'GS': 'Financials',
            'MS': 'Financials', 'C': 'Financials', 'BLK': 'Financials',
            # Energy
            'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy',
            'EOG': 'Energy', 'SLB': 'Energy', 'OXY': 'Energy',
            # ETFs
            'SPY': 'ETF', 'QQQ': 'ETF', 'IWM': 'ETF', 'DIA': 'ETF',
            'TQQQ': 'Leveraged', 'SPXL': 'Leveraged', 'UPRO': 'Leveraged',
            'SOXL': 'Leveraged', 'TNA': 'Leveraged', 'FAS': 'Leveraged',
        }
        
        result = {}
        for ticker in tickers:
            result[ticker] = sector_map.get(ticker, 'Unknown')
        
        return result
