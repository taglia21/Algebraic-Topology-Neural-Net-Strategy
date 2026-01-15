"""
Parallel TDA Engine - Phase 7 Scalable TDA Computation.

Provides batched parallel TDA computation for large universes:
- ProcessPoolExecutor for true parallelism (bypasses GIL)
- Batch processing to manage memory
- Robust error handling (skip failures, continue)
- Progress tracking and timing
- Caching of computed features

Target: Compute TDA for 200-300 stocks in <10 minutes
"""

import os
import logging
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_tda_single_stock(
    ticker: str,
    ohlcv_data: Dict[str, np.ndarray],  # Serializable format
    window: int = 30,
    embedding_dim: int = 3,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Compute TDA features for a single stock.
    
    This function is designed to run in a separate process.
    Must be at module level for pickle serialization.
    
    Args:
        ticker: Stock symbol
        ohlcv_data: Dict with 'close', 'high', 'low', 'volume' arrays
        window: TDA window size
        embedding_dim: Takens embedding dimension
        
    Returns:
        Tuple of (ticker, features_dict or None on failure)
    """
    try:
        # Import inside function for process isolation
        from ripser import ripser
        
        close = ohlcv_data.get('close')
        if close is None or len(close) < window + embedding_dim:
            return (ticker, None)
        
        # Takens embedding
        def takens_embed(series, dim, delay=1):
            n = len(series)
            m = n - (dim - 1) * delay
            if m <= 0:
                return np.array([]).reshape(0, dim)
            embedded = np.zeros((m, dim))
            for i in range(dim):
                embedded[:, i] = series[i * delay : i * delay + m]
            return embedded
        
        # Compute TDA features for last window
        recent_close = close[-window:]
        point_cloud = takens_embed(recent_close, embedding_dim)
        
        if len(point_cloud) < 3:
            return (ticker, None)
        
        # Run Ripser
        result = ripser(point_cloud, maxdim=1)
        diagrams = result['dgms']
        
        # Extract features
        features = {}
        
        for dim in [0, 1]:
            if dim < len(diagrams):
                dgm = diagrams[dim]
                finite_mask = np.isfinite(dgm[:, 1])
                finite_pairs = dgm[finite_mask]
                
                if len(finite_pairs) > 0:
                    lifetimes = finite_pairs[:, 1] - finite_pairs[:, 0]
                    lifetimes = lifetimes[lifetimes > 0]
                    
                    features[f'persistence_l{dim}'] = float(np.linalg.norm(lifetimes))
                    features[f'betti_{dim}'] = len(finite_pairs)
                    features[f'max_lifetime_l{dim}'] = float(np.max(lifetimes)) if len(lifetimes) > 0 else 0
                    features[f'sum_lifetime_l{dim}'] = float(np.sum(lifetimes))
                    
                    # Entropy
                    if len(lifetimes) > 0 and np.sum(lifetimes) > 0:
                        probs = lifetimes / np.sum(lifetimes)
                        entropy = -np.sum(probs * np.log(probs + 1e-10))
                        features[f'entropy_l{dim}'] = float(entropy)
                    else:
                        features[f'entropy_l{dim}'] = 0.0
                else:
                    features[f'persistence_l{dim}'] = 0.0
                    features[f'betti_{dim}'] = 0
                    features[f'max_lifetime_l{dim}'] = 0.0
                    features[f'sum_lifetime_l{dim}'] = 0.0
                    features[f'entropy_l{dim}'] = 0.0
            else:
                features[f'persistence_l{dim}'] = 0.0
                features[f'betti_{dim}'] = 0
                features[f'max_lifetime_l{dim}'] = 0.0
                features[f'sum_lifetime_l{dim}'] = 0.0
                features[f'entropy_l{dim}'] = 0.0
        
        # Composite score
        features['tda_score'] = (
            features['persistence_l0'] * 0.4 +
            features['persistence_l1'] * 0.3 +
            features['max_lifetime_l1'] * 0.3
        )
        
        return (ticker, features)
        
    except Exception as e:
        logger.debug(f"TDA computation failed for {ticker}: {e}")
        return (ticker, None)


def compute_tda_rolling(
    ticker: str,
    ohlcv_data: Dict[str, np.ndarray],
    window: int = 30,
    embedding_dim: int = 3,
    step: int = 5,  # Compute every N days
) -> Tuple[str, Optional[pd.DataFrame]]:
    """
    Compute rolling TDA features for a stock.
    
    Args:
        ticker: Stock symbol
        ohlcv_data: Dict with price arrays
        window: TDA window
        embedding_dim: Takens dimension
        step: Days between computations (for efficiency)
        
    Returns:
        Tuple of (ticker, DataFrame of features or None)
    """
    try:
        from ripser import ripser
        
        close = ohlcv_data.get('close')
        dates = ohlcv_data.get('dates')
        
        if close is None or len(close) < window + embedding_dim + 10:
            return (ticker, None)
        
        def takens_embed(series, dim, delay=1):
            n = len(series)
            m = n - (dim - 1) * delay
            if m <= 0:
                return np.array([]).reshape(0, dim)
            embedded = np.zeros((m, dim))
            for i in range(dim):
                embedded[:, i] = series[i * delay : i * delay + m]
            return embedded
        
        results = []
        start_idx = window + embedding_dim
        
        for i in range(start_idx, len(close), step):
            window_data = close[i-window:i]
            point_cloud = takens_embed(window_data, embedding_dim)
            
            if len(point_cloud) < 3:
                continue
            
            try:
                result = ripser(point_cloud, maxdim=1)
                diagrams = result['dgms']
                
                row = {'idx': i}
                if dates is not None and i < len(dates):
                    row['date'] = dates[i]
                
                for dim in [0, 1]:
                    if dim < len(diagrams):
                        dgm = diagrams[dim]
                        finite_mask = np.isfinite(dgm[:, 1])
                        finite_pairs = dgm[finite_mask]
                        
                        if len(finite_pairs) > 0:
                            lifetimes = finite_pairs[:, 1] - finite_pairs[:, 0]
                            lifetimes = lifetimes[lifetimes > 0]
                            
                            row[f'persistence_l{dim}'] = float(np.linalg.norm(lifetimes))
                            row[f'betti_{dim}'] = len(finite_pairs)
                        else:
                            row[f'persistence_l{dim}'] = 0.0
                            row[f'betti_{dim}'] = 0
                    else:
                        row[f'persistence_l{dim}'] = 0.0
                        row[f'betti_{dim}'] = 0
                
                results.append(row)
                
            except Exception:
                continue
        
        if not results:
            return (ticker, None)
        
        df = pd.DataFrame(results)
        return (ticker, df)
        
    except Exception as e:
        logger.debug(f"Rolling TDA failed for {ticker}: {e}")
        return (ticker, None)


class ParallelTDAEngine:
    """
    Parallel TDA computation engine for large stock universes.
    
    Uses ProcessPoolExecutor for true parallelism.
    """
    
    def __init__(
        self,
        n_workers: int = 4,
        batch_size: int = 50,
        window: int = 30,
        embedding_dim: int = 3,
        cache_dir: str = 'data/tda_cache',
        timeout_seconds: int = 60,
    ):
        """
        Initialize parallel TDA engine.
        
        Args:
            n_workers: Number of parallel processes
            batch_size: Stocks per batch
            window: TDA window size
            embedding_dim: Takens embedding dimension
            cache_dir: Directory for TDA cache
            timeout_seconds: Timeout per stock computation
        """
        self.n_workers = n_workers
        self.batch_size = batch_size
        self.window = window
        self.embedding_dim = embedding_dim
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = timeout_seconds
        
        self.compute_stats = {
            'success': 0,
            'failures': 0,
            'cache_hits': 0,
            'total_time': 0.0,
        }
        self.failed_tickers: List[str] = []
    
    def _prepare_data_for_process(
        self,
        df: pd.DataFrame,
    ) -> Dict[str, np.ndarray]:
        """
        Convert DataFrame to serializable dict for multiprocessing.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Dict with numpy arrays
        """
        data = {}
        
        for col in ['close', 'high', 'low', 'volume', 'open']:
            if col in df.columns:
                data[col] = df[col].values.astype(np.float64)
        
        # Include dates as strings
        if hasattr(df.index, 'strftime'):
            data['dates'] = df.index.strftime('%Y-%m-%d').values
        
        return data
    
    def _get_cache_path(self, ticker: str) -> Path:
        """Get cache file path for ticker."""
        return self.cache_dir / f"{ticker}_tda.pkl"
    
    def load_from_cache(self, ticker: str) -> Optional[Dict]:
        """Load TDA features from cache."""
        cache_path = self._get_cache_path(ticker)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                self.compute_stats['cache_hits'] += 1
                return data
            except Exception:
                pass
        return None
    
    def save_to_cache(self, ticker: str, features: Dict):
        """Save TDA features to cache."""
        try:
            cache_path = self._get_cache_path(ticker)
            with open(cache_path, 'wb') as f:
                pickle.dump(features, f)
        except Exception as e:
            logger.debug(f"Cache save failed for {ticker}: {e}")
    
    def compute_batch_tda(
        self,
        ohlcv_dict: Dict[str, pd.DataFrame],
        use_cache: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute TDA features for multiple stocks in parallel.
        
        Args:
            ohlcv_dict: Dict of {ticker: OHLCV DataFrame}
            use_cache: Whether to use cached results
            
        Returns:
            Dict of {ticker: features_dict}
        """
        start_time = time.time()
        results = {}
        to_compute = []
        
        # Check cache first
        for ticker, df in ohlcv_dict.items():
            if use_cache:
                cached = self.load_from_cache(ticker)
                if cached is not None:
                    results[ticker] = cached
                    continue
            
            # Prepare data for processing
            data = self._prepare_data_for_process(df)
            to_compute.append((ticker, data))
        
        if not to_compute:
            logger.info("All TDA features loaded from cache")
            return results
        
        logger.info(f"Computing TDA for {len(to_compute)} stocks ({len(results)} from cache)")
        
        # Progress bar
        if HAS_TQDM:
            pbar = tqdm(total=len(to_compute), desc="Computing TDA", unit="stocks")
        else:
            pbar = None
        
        # Process in batches
        for batch_start in range(0, len(to_compute), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(to_compute))
            batch = to_compute[batch_start:batch_end]
            
            if not pbar:
                logger.info(f"TDA batch {batch_start//self.batch_size + 1}: "
                          f"stocks {batch_start+1}-{batch_end}")
            
            # Use ProcessPoolExecutor for true parallelism
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = {
                    executor.submit(
                        compute_tda_single_stock,
                        ticker,
                        data,
                        self.window,
                        self.embedding_dim,
                    ): ticker
                    for ticker, data in batch
                }
                
                for future in as_completed(futures, timeout=self.timeout_seconds * len(batch)):
                    try:
                        ticker, features = future.result(timeout=self.timeout_seconds)
                        
                        if features is not None:
                            results[ticker] = features
                            self.save_to_cache(ticker, features)
                            self.compute_stats['success'] += 1
                        else:
                            self.failed_tickers.append(ticker)
                            self.compute_stats['failures'] += 1
                            
                    except TimeoutError:
                        ticker = futures[future]
                        logger.debug(f"TDA timeout for {ticker}")
                        self.failed_tickers.append(ticker)
                        self.compute_stats['failures'] += 1
                    except Exception as e:
                        ticker = futures[future]
                        logger.debug(f"TDA error for {ticker}: {e}")
                        self.failed_tickers.append(ticker)
                        self.compute_stats['failures'] += 1
                    
                    if pbar:
                        pbar.update(1)
        
        if pbar:
            pbar.close()
        
        elapsed = time.time() - start_time
        self.compute_stats['total_time'] = elapsed
        
        logger.info(f"TDA computation complete: {len(results)} success, "
                   f"{len(self.failed_tickers)} failures in {elapsed:.1f}s")
        
        return results
    
    def compute_tda_scores(
        self,
        ohlcv_dict: Dict[str, pd.DataFrame],
    ) -> pd.Series:
        """
        Compute TDA scores for ranking.
        
        Args:
            ohlcv_dict: Dict of {ticker: OHLCV DataFrame}
            
        Returns:
            Series of {ticker: tda_score}
        """
        features = self.compute_batch_tda(ohlcv_dict)
        
        scores = {}
        for ticker, feat in features.items():
            scores[ticker] = feat.get('tda_score', 0)
        
        return pd.Series(scores).sort_values(ascending=False)
    
    def get_compute_stats(self) -> Dict:
        """Get computation statistics."""
        return {
            **self.compute_stats,
            'failed_tickers': self.failed_tickers,
            'n_failed': len(self.failed_tickers),
            'success_rate': (
                self.compute_stats['success'] / 
                max(1, self.compute_stats['success'] + self.compute_stats['failures'])
            ),
        }
    
    def clear_cache(self):
        """Clear TDA cache."""
        count = 0
        for f in self.cache_dir.glob("*_tda.pkl"):
            f.unlink()
            count += 1
        logger.info(f"Cleared {count} TDA cache files")


def test_parallel_tda():
    """Test parallel TDA engine."""
    print("\n" + "="*60)
    print("Phase 7: Parallel TDA Engine Test")
    print("="*60)
    
    # Generate test data
    np.random.seed(42)
    
    test_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "BAC", "WFC"]
    
    ohlcv_dict = {}
    for ticker in test_tickers:
        n = 200
        dates = pd.date_range('2023-01-01', periods=n, freq='D')
        returns = np.random.randn(n) * 0.02
        prices = 100 * np.cumprod(1 + returns)
        
        ohlcv_dict[ticker] = pd.DataFrame({
            'open': prices * 0.99,
            'high': prices * 1.01,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.uniform(1e6, 1e8, n),
        }, index=dates)
    
    # Test engine
    engine = ParallelTDAEngine(
        n_workers=2,
        batch_size=5,
        window=30,
        embedding_dim=3,
    )
    
    print(f"\nComputing TDA for {len(test_tickers)} stocks...")
    
    features = engine.compute_batch_tda(ohlcv_dict, use_cache=False)
    
    print(f"\nResults: {len(features)} stocks computed")
    
    # Print sample features
    if features:
        sample = list(features.items())[0]
        print(f"\nSample features ({sample[0]}):")
        for k, v in sample[1].items():
            print(f"  {k}: {v:.4f}")
    
    # Print stats
    stats = engine.get_compute_stats()
    print(f"\nComputation Stats:")
    print(f"  Success: {stats['success']}")
    print(f"  Failures: {stats['failures']}")
    print(f"  Cache hits: {stats['cache_hits']}")
    print(f"  Total time: {stats['total_time']:.2f}s")
    print(f"  Success rate: {stats['success_rate']:.1%}")
    
    # Test scores
    scores = engine.compute_tda_scores(ohlcv_dict)
    print(f"\nTDA Scores (top 5):")
    for ticker, score in scores.head().items():
        print(f"  {ticker}: {score:.4f}")


if __name__ == "__main__":
    test_parallel_tda()
