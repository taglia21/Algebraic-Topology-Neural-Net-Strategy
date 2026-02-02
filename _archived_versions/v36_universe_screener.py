#!/usr/bin/env python3
"""
V36 Universe Screener
=====================
Daily S&P 500 screening system with momentum-based ranking.

Features:
- Screens S&P 500 stocks using yfinance data
- Filters: market_cap > $10B, avg_volume > 1M, price > $10
- Ranks by: 12-month momentum, 3-month momentum, volatility-adjusted returns
- Returns top 50 stocks as dynamic trading universe
- 24-hour TTL cache using pickle for efficiency

Usage:
    screener = UniverseScreener()
    universe = screener.get_universe()
    print(f"Top 50 stocks: {universe}")
"""

import logging
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('V36_Screener')


# S&P 500 symbols (representative sample)
SP500_SYMBOLS = [
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'GOOG', 'TSLA', 'BRK-B', 'UNH',
    'XOM', 'JNJ', 'JPM', 'V', 'PG', 'MA', 'AVGO', 'HD', 'CVX', 'MRK',
    'ABBV', 'LLY', 'PEP', 'KO', 'COST', 'ADBE', 'WMT', 'MCD', 'CSCO', 'CRM',
    'BAC', 'PFE', 'TMO', 'ACN', 'NFLX', 'AMD', 'ABT', 'DHR', 'DIS', 'LIN',
    'CMCSA', 'VZ', 'INTC', 'NKE', 'PM', 'WFC', 'TXN', 'NEE', 'RTX', 'UPS',
    'QCOM', 'BMY', 'COP', 'HON', 'LOW', 'ORCL', 'UNP', 'SPGI', 'IBM', 'CAT',
    'GE', 'BA', 'INTU', 'AMAT', 'AMGN', 'GS', 'SBUX', 'BLK', 'DE', 'ELV',
    'ISRG', 'MDLZ', 'ADP', 'GILD', 'ADI', 'BKNG', 'VRTX', 'TJX', 'PLD', 'MMC',
    'SYK', 'MS', 'CVS', 'LMT', 'REGN', 'CI', 'TMUS', 'CB', 'SCHW', 'ZTS',
    'ETN', 'MO', 'SO', 'BDX', 'EOG', 'DUK', 'AMT', 'BSX', 'LRCX', 'NOC',
    'PYPL', 'AON', 'CME', 'ICE', 'ITW', 'WM', 'SLB', 'APD', 'CSX', 'CL',
    'PNC', 'TGT', 'FCX', 'MCK', 'EMR', 'MPC', 'USB', 'SHW', 'SNPS', 'NSC',
    'FDX', 'CDNS', 'GD', 'ORLY', 'PSX', 'AZO', 'OXY', 'TFC', 'AJG', 'KLAC',
    'MCO', 'ROP', 'HUM', 'MCHP', 'PCAR', 'VLO', 'MAR', 'AEP', 'MET', 'KMB',
    'CTAS', 'AFL', 'MSCI', 'D', 'AIG', 'TRV', 'CCI', 'GIS', 'PSA', 'JCI',
    'HCA', 'APH', 'WELL', 'CMG', 'DXCM', 'F', 'GM', 'TEL', 'CARR', 'NUE',
    'ADM', 'SRE', 'CHTR', 'WMB', 'STZ', 'HES', 'DVN', 'KHC', 'A', 'IDXX',
    'BIIB', 'EW', 'DHI', 'LHX', 'HAL', 'AMP', 'EXC', 'DOW', 'PAYX', 'MNST',
    'ROK', 'PRU', 'MTD', 'ODFL', 'FTNT', 'SPG', 'XEL', 'ED', 'ROST', 'OTIS',
    'AME', 'BK', 'CTSH', 'GWW', 'DD', 'CMI', 'CPRT', 'EA', 'IQV', 'PEG',
]


@dataclass
class ScreenerConfig:
    """Configuration for the universe screener."""
    min_market_cap: float = 10e9  # $10 billion
    min_avg_volume: float = 1e6   # 1 million shares
    min_price: float = 10.0       # $10 minimum price
    top_n: int = 50               # Return top 50 stocks
    cache_ttl_hours: int = 24     # Cache for 24 hours
    cache_dir: Path = Path("cache/universe")
    momentum_12m_weight: float = 0.4
    momentum_3m_weight: float = 0.4
    vol_adj_return_weight: float = 0.2


@dataclass
class StockMetrics:
    """Metrics for a single stock."""
    symbol: str
    price: float
    market_cap: float
    avg_volume: float
    momentum_12m: float
    momentum_3m: float
    volatility: float
    vol_adj_return: float
    composite_score: float = 0.0


class UniverseScreener:
    """
    S&P 500 universe screener with momentum-based ranking.
    
    Screens stocks daily based on liquidity and market cap filters,
    then ranks by momentum and volatility-adjusted returns.
    
    Args:
        config: Screener configuration settings
    
    Example:
        screener = UniverseScreener()
        universe = screener.get_universe()  # Returns List[str] of top 50 symbols
    """

    def __init__(self, config: Optional[ScreenerConfig] = None):
        if not YF_AVAILABLE:
            raise ImportError("yfinance is required. Install with: pip install yfinance")
        
        self.config = config or ScreenerConfig()
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = self.config.cache_dir / "universe_cache.pkl"

    def _is_cache_valid(self) -> bool:
        """Check if cache exists and is within TTL."""
        if not self._cache_file.exists():
            return False
        
        cache_mtime = datetime.fromtimestamp(self._cache_file.stat().st_mtime)
        cache_age = datetime.now() - cache_mtime
        return cache_age < timedelta(hours=self.config.cache_ttl_hours)

    def _load_cache(self) -> Optional[List[str]]:
        """Load universe from cache."""
        try:
            with open(self._cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                logger.info(f"Loaded cached universe: {len(cached_data)} symbols")
                return cached_data
        except (pickle.PickleError, IOError) as e:
            logger.warning(f"Cache load failed: {e}")
            return None

    def _save_cache(self, universe: List[str]) -> None:
        """Save universe to cache."""
        try:
            with open(self._cache_file, 'wb') as f:
                pickle.dump(universe, f)
            logger.info(f"Cached universe: {len(universe)} symbols")
        except (pickle.PickleError, IOError) as e:
            logger.warning(f"Cache save failed: {e}")

    def _fetch_stock_data(self, symbols: List[str]) -> pd.DataFrame:
        """
        Fetch historical price data for all symbols.
        
        Args:
            symbols: List of stock symbols
        
        Returns:
            DataFrame with adjusted close prices
        """
        logger.info(f"Fetching data for {len(symbols)} symbols...")
        try:
            data = yf.download(
                symbols,
                period="1y",
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=True
            )
            if isinstance(data.columns, pd.MultiIndex):
                return data['Close']
            return data
        except Exception as e:
            logger.error(f"Data fetch error: {e}")
            return pd.DataFrame()

    def _get_stock_info(self, symbol: str) -> Tuple[float, float]:
        """
        Get market cap and average volume for a symbol.
        
        Returns:
            Tuple of (market_cap, avg_volume)
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            market_cap = info.get('marketCap', 0) or 0
            avg_volume = info.get('averageVolume', 0) or 0
            return float(market_cap), float(avg_volume)
        except Exception:
            return 0.0, 0.0

    def _calculate_metrics(
        self, symbol: str, prices: pd.Series, market_cap: float, avg_volume: float
    ) -> Optional[StockMetrics]:
        """
        Calculate momentum and volatility metrics for a stock.
        
        Args:
            symbol: Stock symbol
            prices: Price series
            market_cap: Market capitalization
            avg_volume: Average daily volume
        
        Returns:
            StockMetrics or None if calculation fails
        """
        try:
            prices = prices.dropna()
            if len(prices) < 63:  # Need at least 3 months of data
                return None
            
            current_price = prices.iloc[-1]
            
            # Apply filters
            if current_price < self.config.min_price:
                return None
            if market_cap < self.config.min_market_cap:
                return None
            if avg_volume < self.config.min_avg_volume:
                return None
            
            # Calculate returns
            returns = prices.pct_change().dropna()
            
            # 12-month momentum (or max available)
            momentum_12m = (prices.iloc[-1] / prices.iloc[0] - 1) if len(prices) >= 252 else \
                           (prices.iloc[-1] / prices.iloc[0] - 1)
            
            # 3-month momentum
            momentum_3m = (prices.iloc[-1] / prices.iloc[-63] - 1) if len(prices) >= 63 else 0.0
            
            # Volatility (annualized)
            volatility = returns.std() * np.sqrt(252)
            
            # Volatility-adjusted return (Sharpe-like)
            vol_adj_return = momentum_12m / volatility if volatility > 0 else 0.0
            
            return StockMetrics(
                symbol=symbol,
                price=current_price,
                market_cap=market_cap,
                avg_volume=avg_volume,
                momentum_12m=momentum_12m,
                momentum_3m=momentum_3m,
                volatility=volatility,
                vol_adj_return=vol_adj_return
            )
        except Exception as e:
            logger.debug(f"Metrics calculation failed for {symbol}: {e}")
            return None

    def _rank_stocks(self, metrics_list: List[StockMetrics]) -> List[StockMetrics]:
        """
        Rank stocks by composite score.
        
        Composite score = weighted sum of:
        - 12-month momentum rank (40%)
        - 3-month momentum rank (40%)
        - Volatility-adjusted return rank (20%)
        """
        if not metrics_list:
            return []
        
        n = len(metrics_list)
        
        # Sort and rank by each factor (higher = better rank)
        mom_12m_sorted = sorted(metrics_list, key=lambda x: x.momentum_12m)
        mom_3m_sorted = sorted(metrics_list, key=lambda x: x.momentum_3m)
        vol_adj_sorted = sorted(metrics_list, key=lambda x: x.vol_adj_return)
        
        ranks_12m = {m.symbol: i / n for i, m in enumerate(mom_12m_sorted)}
        ranks_3m = {m.symbol: i / n for i, m in enumerate(mom_3m_sorted)}
        ranks_vol = {m.symbol: i / n for i, m in enumerate(vol_adj_sorted)}
        
        # Calculate composite scores
        for m in metrics_list:
            m.composite_score = (
                self.config.momentum_12m_weight * ranks_12m[m.symbol] +
                self.config.momentum_3m_weight * ranks_3m[m.symbol] +
                self.config.vol_adj_return_weight * ranks_vol[m.symbol]
            )
        
        # Sort by composite score (descending)
        return sorted(metrics_list, key=lambda x: x.composite_score, reverse=True)

    def screen(self, force_refresh: bool = False) -> List[StockMetrics]:
        """
        Screen and rank all stocks.
        
        Args:
            force_refresh: Bypass cache if True
        
        Returns:
            List of StockMetrics for top stocks, ranked by composite score
        """
        logger.info("Starting universe screening...")
        
        # Fetch price data
        prices_df = self._fetch_stock_data(SP500_SYMBOLS)
        if prices_df.empty:
            logger.error("Failed to fetch price data")
            return []
        
        # Calculate metrics for each stock
        metrics_list: List[StockMetrics] = []
        for symbol in prices_df.columns:
            if symbol not in prices_df.columns:
                continue
            
            market_cap, avg_volume = self._get_stock_info(symbol)
            metrics = self._calculate_metrics(
                symbol, prices_df[symbol], market_cap, avg_volume
            )
            if metrics:
                metrics_list.append(metrics)
        
        logger.info(f"Calculated metrics for {len(metrics_list)} stocks")
        
        # Rank and return top N
        ranked = self._rank_stocks(metrics_list)
        top_n = ranked[:self.config.top_n]
        
        logger.info(f"Top {len(top_n)} stocks selected")
        return top_n

    def get_universe(self, force_refresh: bool = False) -> List[str]:
        """
        Get the current trading universe.
        
        Returns cached results if available and within TTL,
        otherwise performs fresh screening.
        
        Args:
            force_refresh: Bypass cache if True
        
        Returns:
            List of stock symbols (top 50 by composite ranking)
        """
        # Check cache
        if not force_refresh and self._is_cache_valid():
            cached = self._load_cache()
            if cached:
                return cached
        
        # Perform fresh screening
        ranked_metrics = self.screen(force_refresh=force_refresh)
        universe = [m.symbol for m in ranked_metrics]
        
        # Save to cache
        self._save_cache(universe)
        
        return universe

    def get_universe_with_metrics(self, force_refresh: bool = False) -> List[StockMetrics]:
        """
        Get universe with full metrics for each stock.
        
        Args:
            force_refresh: Bypass cache if True
        
        Returns:
            List of StockMetrics for top 50 stocks
        """
        return self.screen(force_refresh=force_refresh)


def main() -> None:
    """Example usage of UniverseScreener."""
    screener = UniverseScreener()
    
    # Get universe (uses cache if available)
    universe = screener.get_universe()
    print(f"\n{'='*60}")
    print(f"DYNAMIC UNIVERSE ({len(universe)} stocks)")
    print(f"{'='*60}")
    print(", ".join(universe[:10]), "...")
    
    # Get with metrics
    metrics = screener.get_universe_with_metrics()
    if metrics:
        print(f"\nTop 5 by composite score:")
        for m in metrics[:5]:
            print(f"  {m.symbol}: score={m.composite_score:.3f}, "
                  f"12m={m.momentum_12m:+.1%}, 3m={m.momentum_3m:+.1%}")


if __name__ == "__main__":
    main()
