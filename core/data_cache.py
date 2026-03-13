"""
core/data_cache.py
==================
Cache historical market data to parquet for NN training and backtesting.
"""
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path("data/market_cache")


class MarketDataCache:
    def __init__(self, cache_dir: Path = _DEFAULT_CACHE_DIR):
        self._dir = cache_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def save_bars(self, market_data: Dict[str, pd.DataFrame], date_str: Optional[str] = None) -> None:
        """Save daily bar data to parquet files, one per symbol."""
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")

        for symbol, df in market_data.items():
            if df is None or df.empty:
                continue
            symbol_dir = self._dir / symbol
            symbol_dir.mkdir(parents=True, exist_ok=True)
            path = symbol_dir / f"{date_str}.parquet"
            try:
                df.to_parquet(path)
            except Exception as e:
                logger.warning("Failed to cache %s data: %s", symbol, e)

        logger.info("Cached market data for %d symbols", len(market_data))

    def load_symbol(self, symbol: str, min_rows: int = 60) -> Optional[pd.DataFrame]:
        """Load all cached data for a symbol, concatenated."""
        symbol_dir = self._dir / symbol
        if not symbol_dir.exists():
            return None

        files = sorted(symbol_dir.glob("*.parquet"))
        if not files:
            return None

        dfs = []
        for f in files:
            try:
                dfs.append(pd.read_parquet(f))
            except Exception as e:
                logger.warning("Failed to read %s: %s", f, e)

        if not dfs:
            return None

        combined = pd.concat(dfs, ignore_index=False)
        combined = combined[~combined.index.duplicated(keep='last')]
        combined = combined.sort_index()

        if len(combined) < min_rows:
            logger.info("%s has only %d rows (need %d)", symbol, len(combined), min_rows)
            return None

        return combined

    def load_all(self, symbols: list, min_rows: int = 60) -> Dict[str, pd.DataFrame]:
        """Load cached data for all symbols."""
        result = {}
        for sym in symbols:
            df = self.load_symbol(sym, min_rows=min_rows)
            if df is not None:
                result[sym] = df
        return result

    def save_combined(self, price_df: pd.DataFrame, volume_df: Optional[pd.DataFrame] = None) -> None:
        """Save combined price/volume DataFrames to a single parquet each."""
        try:
            price_path = self._dir / "combined_prices.parquet"
            price_df.to_parquet(price_path)
            if volume_df is not None:
                vol_path = self._dir / "combined_volumes.parquet"
                volume_df.to_parquet(vol_path)
            logger.info("Saved combined price data: %d rows x %d symbols", len(price_df), len(price_df.columns))
        except Exception as e:
            logger.warning("Failed to save combined data: %s", e)

    def load_combined(self) -> tuple:
        """Load combined price/volume DataFrames."""
        price_path = self._dir / "combined_prices.parquet"
        vol_path = self._dir / "combined_volumes.parquet"

        prices = None
        volumes = None

        if price_path.exists():
            try:
                prices = pd.read_parquet(price_path)
            except Exception as e:
                logger.warning("Failed to load combined prices: %s", e)

        if vol_path.exists():
            try:
                volumes = pd.read_parquet(vol_path)
            except Exception as e:
                logger.warning("Failed to load combined volumes: %s", e)

        return prices, volumes
