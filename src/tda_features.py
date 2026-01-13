"""TDA Feature Generator using persistent homology for market regime detection."""

import numpy as np
import pandas as pd
from ripser import ripser


class TDAFeatureGenerator:
    """Generates topological features from OHLCV data using persistent homology."""

    def __init__(self, window: int = 30, embedding_dim: int = 3):
        """Initialize with rolling window size and Takens embedding dimension."""
        self.window = window
        self.embedding_dim = embedding_dim

    def takens_embedding(self, series: np.ndarray, delay: int = 1) -> np.ndarray:
        """Convert 1D time series to point cloud via Takens delay embedding."""
        n = len(series)
        m = n - (self.embedding_dim - 1) * delay
        
        if m <= 0:
            return np.array([]).reshape(0, self.embedding_dim)
        
        embedded = np.zeros((m, self.embedding_dim))
        for i in range(self.embedding_dim):
            embedded[:, i] = series[i * delay : i * delay + m]
        
        return embedded

    def compute_persistence_features(self, series: np.ndarray) -> dict:
        """Compute persistent homology features from 1D log-returns."""
        result = {
            'persistence_l0': 0.0,
            'persistence_l1': 0.0,
            'betti_0': 0,
            'betti_1': 0
        }
        
        if len(series) < self.embedding_dim + 2:
            return result
        
        point_cloud = self.takens_embedding(series)
        
        if point_cloud.shape[0] < 3:
            return result
        
        diagrams = ripser(point_cloud, maxdim=1)['dgms']
        
        result['persistence_l0'] = self._compute_lifetime_norm(diagrams[0])
        result['persistence_l1'] = self._compute_lifetime_norm(diagrams[1])
        result['betti_0'] = self._count_features(diagrams[0])
        result['betti_1'] = self._count_features(diagrams[1])
        
        return result

    def _compute_lifetime_norm(self, diagram: np.ndarray) -> float:
        """Compute L2-norm of finite lifetimes in persistence diagram."""
        if len(diagram) == 0:
            return 0.0
        
        finite_mask = np.isfinite(diagram[:, 1])
        finite_pairs = diagram[finite_mask]
        
        if len(finite_pairs) == 0:
            return 0.0
        
        lifetimes = finite_pairs[:, 1] - finite_pairs[:, 0]
        return float(np.linalg.norm(lifetimes))

    def _count_features(self, diagram: np.ndarray) -> int:
        """Count number of finite features in persistence diagram."""
        if len(diagram) == 0:
            return 0
        
        finite_mask = np.isfinite(diagram[:, 1])
        return int(np.sum(finite_mask))

    def generate_features(self, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        """Generate rolling TDA features from OHLCV DataFrame."""
        close = ohlcv_df['close'].values if 'close' in ohlcv_df.columns else ohlcv_df['Close'].values
        
        log_prices = np.log(close + 1e-10)
        returns = np.diff(log_prices)
        
        features_list = []
        
        for i in range(self.window, len(returns) + 1):
            window_returns = returns[i - self.window : i]
            features = self.compute_persistence_features(window_returns)
            features_list.append(features)
        
        return pd.DataFrame(features_list)

    def compute_turbulence_index(self, features_df: pd.DataFrame) -> np.ndarray:
        """Compute turbulence index from persistence features (0-1 scale)."""
        l0 = features_df['persistence_l0'].values
        l1 = features_df['persistence_l1'].values
        
        combined = np.sqrt(l0**2 + l1**2)
        
        if combined.max() > combined.min():
            normalized = (combined - combined.min()) / (combined.max() - combined.min())
        else:
            normalized = np.zeros_like(combined)
        
        return normalized


def test():
    """Test TDA feature generation on synthetic OHLCV data."""
    np.random.seed(42)
    n_bars = 100
    
    base_price = 100 + np.cumsum(np.random.randn(n_bars) * 0.5)
    ohlcv = pd.DataFrame({
        'open': base_price + np.random.randn(n_bars) * 0.1,
        'high': base_price + np.abs(np.random.randn(n_bars) * 0.5),
        'low': base_price - np.abs(np.random.randn(n_bars) * 0.5),
        'close': base_price,
        'volume': np.random.randint(1000, 10000, n_bars)
    })
    
    generator = TDAFeatureGenerator(window=30, embedding_dim=3)
    
    series = np.random.randn(100)
    embedded = generator.takens_embedding(series)
    assert embedded.shape == (98, 3), f"Takens embedding failed: {embedded.shape}"
    
    features = generator.compute_persistence_features(series)
    assert all(k in features for k in ['persistence_l0', 'persistence_l1', 'betti_0', 'betti_1'])
    assert all(not np.isnan(v) for v in features.values())
    
    result_df = generator.generate_features(ohlcv)
    expected_rows = n_bars - 1 - generator.window + 1
    assert len(result_df) == expected_rows, f"Expected {expected_rows} rows, got {len(result_df)}"
    
    turbulence = generator.compute_turbulence_index(result_df)
    assert len(turbulence) == len(result_df)
    assert turbulence.min() >= 0 and turbulence.max() <= 1
    
    return True


if __name__ == "__main__":
    success = test()
    if success:
        import sys
        sys.stdout.write("TDA Feature Generator: All tests passed\n")
