"""Data Caching Infrastructure for Phase 6.

Provides HDF5-based caching for OHLCV data and pickle caching for TDA features.
Designed for efficient handling of 3000+ stock universe.
"""

import os
import pickle
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

import pandas as pd
import numpy as np

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    
logger = logging.getLogger(__name__)


class DataCache:
    """
    High-performance data caching for large stock universe.
    
    Features:
    - HDF5-based storage for OHLCV data (fast read/write)
    - Pickle caching for computed TDA features
    - Automatic cache invalidation based on dates
    - Memory-efficient batch operations
    """
    
    def __init__(
        self,
        cache_dir: str = './cache',
        ohlcv_file: str = 'ohlcv_cache.h5',
        tda_dir: str = 'tda_features',
        max_age_days: int = 1,  # Cache staleness threshold
    ):
        """
        Initialize data cache.
        
        Args:
            cache_dir: Base directory for cache files
            ohlcv_file: HDF5 file for OHLCV data
            tda_dir: Subdirectory for TDA pickle files
            max_age_days: Days before cache is considered stale
        """
        self.cache_dir = Path(cache_dir)
        self.ohlcv_path = self.cache_dir / ohlcv_file
        self.tda_dir = self.cache_dir / tda_dir
        self.max_age_days = max_age_days
        
        # Create directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.tda_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata for tracking cache state
        self.metadata_path = self.cache_dir / 'cache_metadata.pkl'
        self._load_metadata()
        
        logger.info(f"DataCache initialized: {self.cache_dir}")
        
    def _load_metadata(self):
        """Load cache metadata from disk."""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
                self.metadata = {}
        else:
            self.metadata = {}
            
    def _save_metadata(self):
        """Save cache metadata to disk."""
        try:
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
        except Exception as e:
            logger.warning(f"Failed to save cache metadata: {e}")
            
    def _get_cache_key(self, ticker: str, start_date: str, end_date: str) -> str:
        """Generate unique cache key for ticker + date range."""
        return f"{ticker}_{start_date}_{end_date}"
    
    # ==================== OHLCV CACHING ====================
    
    def has_ohlcv_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> bool:
        """Check if OHLCV data exists in cache and is fresh."""
        cache_key = self._get_cache_key(ticker, start_date, end_date)
        
        # Check metadata
        if cache_key not in self.metadata:
            return False
        
        meta = self.metadata[cache_key]
        
        # Check age
        cached_time = meta.get('cached_at')
        if cached_time:
            age = datetime.now() - cached_time
            if age.days > self.max_age_days:
                return False
        
        # Check HDF5 file exists
        if not self.ohlcv_path.exists():
            return False
        
        # Check data exists in HDF5
        if HAS_H5PY:
            try:
                with h5py.File(self.ohlcv_path, 'r') as f:
                    return ticker in f
            except Exception:
                return False
        else:
            # Fallback to pickle
            pkl_path = self.cache_dir / f"ohlcv_{ticker}.pkl"
            return pkl_path.exists()
    
    def save_ohlcv_data(
        self,
        ticker: str,
        data: pd.DataFrame,
        start_date: str,
        end_date: str,
    ) -> bool:
        """
        Save OHLCV data to cache.
        
        Args:
            ticker: Stock symbol
            data: OHLCV DataFrame
            start_date: Data start date
            end_date: Data end date
            
        Returns:
            True if saved successfully
        """
        if data.empty:
            return False
        
        cache_key = self._get_cache_key(ticker, start_date, end_date)
        
        try:
            if HAS_H5PY:
                # Save to HDF5
                with h5py.File(self.ohlcv_path, 'a') as f:
                    # Delete existing dataset if present
                    if ticker in f:
                        del f[ticker]
                    
                    # Create group for ticker
                    grp = f.create_group(ticker)
                    
                    # Save columns
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        if col in data.columns:
                            grp.create_dataset(col, data=data[col].values)
                    
                    # Save index as string dates
                    dates = data.index.strftime('%Y-%m-%d').values.astype('S10')
                    grp.create_dataset('dates', data=dates)
            else:
                # Fallback to pickle
                pkl_path = self.cache_dir / f"ohlcv_{ticker}.pkl"
                data.to_pickle(pkl_path)
            
            # Update metadata
            self.metadata[cache_key] = {
                'ticker': ticker,
                'start_date': start_date,
                'end_date': end_date,
                'n_bars': len(data),
                'cached_at': datetime.now(),
            }
            self._save_metadata()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache OHLCV for {ticker}: {e}")
            return False
    
    def get_ohlcv_data(
        self,
        ticker: str,
        start_date: str = None,
        end_date: str = None,
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve OHLCV data from cache.
        
        Args:
            ticker: Stock symbol
            start_date: Optional filter start
            end_date: Optional filter end
            
        Returns:
            OHLCV DataFrame or None if not cached
        """
        try:
            if HAS_H5PY and self.ohlcv_path.exists():
                with h5py.File(self.ohlcv_path, 'r') as f:
                    if ticker not in f:
                        return None
                    
                    grp = f[ticker]
                    
                    # Read data
                    dates = pd.to_datetime([d.decode() for d in grp['dates'][:]])
                    
                    data = {
                        'open': grp['open'][:],
                        'high': grp['high'][:],
                        'low': grp['low'][:],
                        'close': grp['close'][:],
                        'volume': grp['volume'][:],
                    }
                    
                    df = pd.DataFrame(data, index=dates)
                    
                    # Filter by date range if specified
                    if start_date:
                        df = df[df.index >= start_date]
                    if end_date:
                        df = df[df.index <= end_date]
                    
                    return df
            else:
                # Fallback to pickle
                pkl_path = self.cache_dir / f"ohlcv_{ticker}.pkl"
                if pkl_path.exists():
                    df = pd.read_pickle(pkl_path)
                    if start_date:
                        df = df[df.index >= start_date]
                    if end_date:
                        df = df[df.index <= end_date]
                    return df
                    
            return None
            
        except Exception as e:
            logger.error(f"Failed to read OHLCV cache for {ticker}: {e}")
            return None
    
    def get_ohlcv_batch(
        self,
        tickers: List[str],
        start_date: str = None,
        end_date: str = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Retrieve OHLCV data for multiple tickers.
        
        Args:
            tickers: List of stock symbols
            start_date: Optional filter start
            end_date: Optional filter end
            
        Returns:
            Dict mapping ticker to DataFrame
        """
        result = {}
        
        if HAS_H5PY and self.ohlcv_path.exists():
            try:
                with h5py.File(self.ohlcv_path, 'r') as f:
                    for ticker in tickers:
                        if ticker in f:
                            grp = f[ticker]
                            dates = pd.to_datetime([d.decode() for d in grp['dates'][:]])
                            
                            data = {
                                'open': grp['open'][:],
                                'high': grp['high'][:],
                                'low': grp['low'][:],
                                'close': grp['close'][:],
                                'volume': grp['volume'][:],
                            }
                            
                            df = pd.DataFrame(data, index=dates)
                            
                            if start_date:
                                df = df[df.index >= start_date]
                            if end_date:
                                df = df[df.index <= end_date]
                            
                            if not df.empty:
                                result[ticker] = df
            except Exception as e:
                logger.error(f"Batch OHLCV read failed: {e}")
        else:
            # Fallback to individual pickle reads
            for ticker in tickers:
                df = self.get_ohlcv_data(ticker, start_date, end_date)
                if df is not None and not df.empty:
                    result[ticker] = df
        
        return result
    
    # ==================== TDA FEATURE CACHING ====================
    
    def _get_tda_path(self, ticker: str) -> Path:
        """Get path for TDA feature pickle file."""
        return self.tda_dir / f"{ticker}_tda.pkl"
    
    def has_tda_features(self, ticker: str) -> bool:
        """Check if TDA features exist in cache."""
        return self._get_tda_path(ticker).exists()
    
    def save_tda_features(
        self,
        ticker: str,
        features: Dict[str, Any],
    ) -> bool:
        """
        Save computed TDA features to cache.
        
        Args:
            ticker: Stock symbol
            features: Dict of TDA features
            
        Returns:
            True if saved successfully
        """
        try:
            path = self._get_tda_path(ticker)
            
            # Add metadata
            features['_cached_at'] = datetime.now()
            features['_ticker'] = ticker
            
            with open(path, 'wb') as f:
                pickle.dump(features, f)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache TDA features for {ticker}: {e}")
            return False
    
    def get_tda_features(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve TDA features from cache.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dict of TDA features or None
        """
        try:
            path = self._get_tda_path(ticker)
            
            if not path.exists():
                return None
            
            with open(path, 'rb') as f:
                features = pickle.load(f)
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to read TDA cache for {ticker}: {e}")
            return None
    
    def get_tda_batch(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """Retrieve TDA features for multiple tickers."""
        result = {}
        for ticker in tickers:
            features = self.get_tda_features(ticker)
            if features is not None:
                result[ticker] = features
        return result
    
    # ==================== CACHE MANAGEMENT ====================
    
    def invalidate_cache(
        self,
        ticker: str = None,
        cache_type: str = 'all',
    ):
        """
        Clear cache entries.
        
        Args:
            ticker: Specific ticker or None for all
            cache_type: 'ohlcv', 'tda', or 'all'
        """
        if cache_type in ('ohlcv', 'all'):
            if ticker:
                # Remove specific ticker from HDF5
                if HAS_H5PY and self.ohlcv_path.exists():
                    try:
                        with h5py.File(self.ohlcv_path, 'a') as f:
                            if ticker in f:
                                del f[ticker]
                    except Exception as e:
                        logger.error(f"Failed to invalidate OHLCV cache: {e}")
                
                # Remove pickle fallback
                pkl_path = self.cache_dir / f"ohlcv_{ticker}.pkl"
                if pkl_path.exists():
                    pkl_path.unlink()
            else:
                # Remove all OHLCV
                if self.ohlcv_path.exists():
                    self.ohlcv_path.unlink()
                for pkl in self.cache_dir.glob("ohlcv_*.pkl"):
                    pkl.unlink()
        
        if cache_type in ('tda', 'all'):
            if ticker:
                path = self._get_tda_path(ticker)
                if path.exists():
                    path.unlink()
            else:
                # Remove all TDA
                for pkl in self.tda_dir.glob("*_tda.pkl"):
                    pkl.unlink()
        
        # Update metadata
        if ticker:
            keys_to_remove = [k for k in self.metadata if k.startswith(ticker)]
            for k in keys_to_remove:
                del self.metadata[k]
        else:
            self.metadata = {}
        
        self._save_metadata()
        logger.info(f"Cache invalidated: ticker={ticker}, type={cache_type}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            'cache_dir': str(self.cache_dir),
            'ohlcv_cached': 0,
            'tda_cached': 0,
            'total_size_mb': 0,
        }
        
        # Count OHLCV
        if HAS_H5PY and self.ohlcv_path.exists():
            try:
                with h5py.File(self.ohlcv_path, 'r') as f:
                    stats['ohlcv_cached'] = len(f.keys())
                stats['total_size_mb'] += self.ohlcv_path.stat().st_size / (1024 * 1024)
            except Exception:
                pass
        
        # Count TDA
        tda_files = list(self.tda_dir.glob("*_tda.pkl"))
        stats['tda_cached'] = len(tda_files)
        for f in tda_files:
            stats['total_size_mb'] += f.stat().st_size / (1024 * 1024)
        
        stats['total_size_mb'] = round(stats['total_size_mb'], 2)
        
        return stats


# Singleton instance for global access
_cache_instance: Optional[DataCache] = None


def get_data_cache(cache_dir: str = './cache') -> DataCache:
    """Get or create singleton DataCache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = DataCache(cache_dir=cache_dir)
    return _cache_instance


if __name__ == "__main__":
    # Test cache
    cache = DataCache(cache_dir='./cache_test')
    
    # Create test data
    test_df = pd.DataFrame({
        'open': [100.0, 101.0, 102.0],
        'high': [102.0, 103.0, 104.0],
        'low': [99.0, 100.0, 101.0],
        'close': [101.0, 102.0, 103.0],
        'volume': [1000000, 1100000, 1200000],
    }, index=pd.date_range('2024-01-01', periods=3))
    
    # Test OHLCV caching
    print("Testing OHLCV caching...")
    cache.save_ohlcv_data('TEST', test_df, '2024-01-01', '2024-01-03')
    retrieved = cache.get_ohlcv_data('TEST')
    print(f"  Saved and retrieved: {len(retrieved)} bars")
    
    # Test TDA caching
    print("Testing TDA caching...")
    test_tda = {'betti_0': 5, 'persistence': [0.1, 0.2, 0.3]}
    cache.save_tda_features('TEST', test_tda)
    retrieved_tda = cache.get_tda_features('TEST')
    print(f"  Saved and retrieved TDA: {retrieved_tda}")
    
    # Get stats
    print(f"Cache stats: {cache.get_cache_stats()}")
    
    # Cleanup
    cache.invalidate_cache()
    print("Cache tests complete!")
