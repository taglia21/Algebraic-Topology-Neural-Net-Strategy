"""TDA Feature Generator using persistent homology for market regime detection.

V1.3: Enriched TDA descriptors (top_k lifetimes, count_above_thresh, wasserstein approx).
V1.2: Extended with richer TDA descriptors (entropy, multi-scale summaries).
"""

import numpy as np
import pandas as pd
from ripser import ripser


class TDAFeatureGenerator:
    """Generates topological features from OHLCV data using persistent homology.
    
    V1.3 Features (Enriched):
    - All V1.2 features (10 total)
    - top_k_lifetimes_l0/l1: Top 3 longest lifetimes (6 additional)
    - count_large_l0/l1: Count of features above 75th percentile threshold (2 additional)
    - wasserstein_approx_l0/l1: Approximate Wasserstein distance proxy (2 additional)
    Total V1.3: 20 features
    
    V1.2 Features:
    - persistence_l0, persistence_l1: L2-norm of lifetimes (original)
    - betti_0, betti_1: Feature counts (original)
    - entropy_l0, entropy_l1: Shannon entropy over lifetime distribution
    - max_lifetime_l0, max_lifetime_l1: Maximum lifetime (multi-scale)
    - sum_lifetime_l0, sum_lifetime_l1: Sum of lifetimes
    """

    # Feature set versions
    FEATURE_MODES = {
        'v1.1': 4,    # persistence_l0/l1, betti_0/1
        'v1.2': 10,   # V1.1 + entropy, max/sum lifetime
        'v1.3': 20,   # V1.2 + top_k, count_large, wasserstein_approx
    }

    def __init__(self, window: int = 30, embedding_dim: int = 3, 
                 feature_mode: str = 'v1.3', extended_features: bool = None):
        """Initialize with rolling window size and Takens embedding dimension.
        
        Args:
            window: Rolling window size for TDA computation
            embedding_dim: Dimension for Takens delay embedding
            feature_mode: 'v1.1' (4 features), 'v1.2' (10 features), 'v1.3' (20 features)
            extended_features: Deprecated - use feature_mode instead. 
                             If provided, True maps to 'v1.2', False to 'v1.1'.
        """
        self.window = window
        self.embedding_dim = embedding_dim
        
        # Handle backward compatibility with extended_features
        if extended_features is not None:
            self.feature_mode = 'v1.2' if extended_features else 'v1.1'
        else:
            self.feature_mode = feature_mode
        
        # For backward compatibility
        self.extended_features = self.feature_mode in ('v1.2', 'v1.3')

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

    def _compute_lifetime_entropy(self, diagram: np.ndarray) -> float:
        """Compute Shannon entropy over the lifetime distribution."""
        if len(diagram) == 0:
            return 0.0
        
        finite_mask = np.isfinite(diagram[:, 1])
        finite_pairs = diagram[finite_mask]
        
        if len(finite_pairs) == 0:
            return 0.0
        
        lifetimes = finite_pairs[:, 1] - finite_pairs[:, 0]
        lifetimes = lifetimes[lifetimes > 0]  # Only positive lifetimes
        
        if len(lifetimes) == 0:
            return 0.0
        
        # Normalize to probabilities
        total = np.sum(lifetimes)
        if total == 0:
            return 0.0
        
        probs = lifetimes / total
        
        # Shannon entropy (with small epsilon to avoid log(0))
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        return float(entropy)

    def _compute_max_lifetime(self, diagram: np.ndarray) -> float:
        """Compute maximum lifetime in persistence diagram."""
        if len(diagram) == 0:
            return 0.0
        
        finite_mask = np.isfinite(diagram[:, 1])
        finite_pairs = diagram[finite_mask]
        
        if len(finite_pairs) == 0:
            return 0.0
        
        lifetimes = finite_pairs[:, 1] - finite_pairs[:, 0]
        return float(np.max(lifetimes)) if len(lifetimes) > 0 else 0.0

    def _compute_sum_lifetime(self, diagram: np.ndarray) -> float:
        """Compute sum of lifetimes in persistence diagram."""
        if len(diagram) == 0:
            return 0.0
        
        finite_mask = np.isfinite(diagram[:, 1])
        finite_pairs = diagram[finite_mask]
        
        if len(finite_pairs) == 0:
            return 0.0
        
        lifetimes = finite_pairs[:, 1] - finite_pairs[:, 0]
        return float(np.sum(lifetimes))

    def _compute_top_k_lifetimes(self, diagram: np.ndarray, k: int = 3) -> list:
        """Compute top k longest lifetimes, zero-padded if fewer exist.
        
        V1.3: Returns sorted descending list of top k lifetimes.
        """
        if len(diagram) == 0:
            return [0.0] * k
        
        finite_mask = np.isfinite(diagram[:, 1])
        finite_pairs = diagram[finite_mask]
        
        if len(finite_pairs) == 0:
            return [0.0] * k
        
        lifetimes = finite_pairs[:, 1] - finite_pairs[:, 0]
        lifetimes = lifetimes[lifetimes > 0]
        
        if len(lifetimes) == 0:
            return [0.0] * k
        
        # Sort descending and take top k
        sorted_lifetimes = np.sort(lifetimes)[::-1]
        top_k = list(sorted_lifetimes[:k])
        
        # Zero-pad if fewer than k
        while len(top_k) < k:
            top_k.append(0.0)
        
        return [float(x) for x in top_k]

    def _compute_count_above_threshold(self, diagram: np.ndarray, quantile: float = 0.75) -> int:
        """Count persistence features with lifetime above the quantile threshold.
        
        V1.3: Uses rolling window's 75th percentile as threshold.
        """
        if len(diagram) == 0:
            return 0
        
        finite_mask = np.isfinite(diagram[:, 1])
        finite_pairs = diagram[finite_mask]
        
        if len(finite_pairs) == 0:
            return 0
        
        lifetimes = finite_pairs[:, 1] - finite_pairs[:, 0]
        lifetimes = lifetimes[lifetimes > 0]
        
        if len(lifetimes) == 0:
            return 0
        
        threshold = np.quantile(lifetimes, quantile)
        return int(np.sum(lifetimes >= threshold))

    def _compute_wasserstein_approx(self, diagram: np.ndarray) -> float:
        """Compute approximate 1-Wasserstein distance from the diagonal.
        
        V1.3: Sum of (death - birth)/2 for all finite points.
        This approximates W_1 distance to the trivial diagram.
        """
        if len(diagram) == 0:
            return 0.0
        
        finite_mask = np.isfinite(diagram[:, 1])
        finite_pairs = diagram[finite_mask]
        
        if len(finite_pairs) == 0:
            return 0.0
        
        # For each point (b, d), distance to diagonal is (d-b)/sqrt(2)
        # Sum of these approximates W_1
        lifetimes = finite_pairs[:, 1] - finite_pairs[:, 0]
        return float(np.sum(lifetimes) / np.sqrt(2))

    def compute_persistence_features(self, series: np.ndarray) -> dict:
        """Compute persistent homology features from 1D log-returns.
        
        V1.3: Enriched feature set with top_k, count_large, wasserstein_approx.
        V1.2: Extended feature set when feature_mode='v1.2'.
        V1.1: Base features only when feature_mode='v1.1'.
        """
        # Base features (V1.1 compatible - always present)
        result = {
            'persistence_l0': 0.0,
            'persistence_l1': 0.0,
            'betti_0': 0,
            'betti_1': 0
        }
        
        # Extended features (V1.2+)
        if self.feature_mode in ('v1.2', 'v1.3'):
            result.update({
                'entropy_l0': 0.0,
                'entropy_l1': 0.0,
                'max_lifetime_l0': 0.0,
                'max_lifetime_l1': 0.0,
                'sum_lifetime_l0': 0.0,
                'sum_lifetime_l1': 0.0,
            })
        
        # Enriched features (V1.3)
        if self.feature_mode == 'v1.3':
            result.update({
                # Top 3 lifetimes for H0
                'top1_lifetime_l0': 0.0,
                'top2_lifetime_l0': 0.0,
                'top3_lifetime_l0': 0.0,
                # Top 3 lifetimes for H1
                'top1_lifetime_l1': 0.0,
                'top2_lifetime_l1': 0.0,
                'top3_lifetime_l1': 0.0,
                # Count above 75th percentile threshold
                'count_large_l0': 0,
                'count_large_l1': 0,
                # Wasserstein approximation
                'wasserstein_approx_l0': 0.0,
                'wasserstein_approx_l1': 0.0,
            })
        
        if len(series) < self.embedding_dim + 2:
            return result
        
        point_cloud = self.takens_embedding(series)
        
        if point_cloud.shape[0] < 3:
            return result
        
        diagrams = ripser(point_cloud, maxdim=1)['dgms']
        
        # Original features (V1.1)
        result['persistence_l0'] = self._compute_lifetime_norm(diagrams[0])
        result['persistence_l1'] = self._compute_lifetime_norm(diagrams[1])
        result['betti_0'] = self._count_features(diagrams[0])
        result['betti_1'] = self._count_features(diagrams[1])
        
        # Extended features (V1.2+)
        if self.feature_mode in ('v1.2', 'v1.3'):
            result['entropy_l0'] = self._compute_lifetime_entropy(diagrams[0])
            result['entropy_l1'] = self._compute_lifetime_entropy(diagrams[1])
            result['max_lifetime_l0'] = self._compute_max_lifetime(diagrams[0])
            result['max_lifetime_l1'] = self._compute_max_lifetime(diagrams[1])
            result['sum_lifetime_l0'] = self._compute_sum_lifetime(diagrams[0])
            result['sum_lifetime_l1'] = self._compute_sum_lifetime(diagrams[1])
        
        # Enriched features (V1.3)
        if self.feature_mode == 'v1.3':
            # Top k lifetimes
            top_k_l0 = self._compute_top_k_lifetimes(diagrams[0], k=3)
            top_k_l1 = self._compute_top_k_lifetimes(diagrams[1], k=3)
            
            result['top1_lifetime_l0'] = top_k_l0[0]
            result['top2_lifetime_l0'] = top_k_l0[1]
            result['top3_lifetime_l0'] = top_k_l0[2]
            result['top1_lifetime_l1'] = top_k_l1[0]
            result['top2_lifetime_l1'] = top_k_l1[1]
            result['top3_lifetime_l1'] = top_k_l1[2]
            
            # Count above threshold
            result['count_large_l0'] = self._compute_count_above_threshold(diagrams[0])
            result['count_large_l1'] = self._compute_count_above_threshold(diagrams[1])
            
            # Wasserstein approximation
            result['wasserstein_approx_l0'] = self._compute_wasserstein_approx(diagrams[0])
            result['wasserstein_approx_l1'] = self._compute_wasserstein_approx(diagrams[1])
        
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
        """Compute turbulence index from persistence features (0-1 scale).
        
        V1.2: Uses entropy in addition to lifetime norms when available.
        """
        l0 = features_df['persistence_l0'].values
        l1 = features_df['persistence_l1'].values
        
        # Base turbulence from lifetime norms
        combined = np.sqrt(l0**2 + l1**2)
        
        # V1.2: Incorporate entropy if available (higher entropy = more chaotic)
        if 'entropy_l0' in features_df.columns and 'entropy_l1' in features_df.columns:
            e0 = features_df['entropy_l0'].values
            e1 = features_df['entropy_l1'].values
            entropy_combined = np.sqrt(e0**2 + e1**2)
            
            # Blend lifetime and entropy (equal weight)
            if entropy_combined.max() > entropy_combined.min():
                entropy_norm = (entropy_combined - entropy_combined.min()) / (entropy_combined.max() - entropy_combined.min())
            else:
                entropy_norm = np.zeros_like(entropy_combined)
            
            if combined.max() > combined.min():
                lifetime_norm = (combined - combined.min()) / (combined.max() - combined.min())
            else:
                lifetime_norm = np.zeros_like(combined)
            
            normalized = 0.5 * lifetime_norm + 0.5 * entropy_norm
        else:
            if combined.max() > combined.min():
                normalized = (combined - combined.min()) / (combined.max() - combined.min())
            else:
                normalized = np.zeros_like(combined)
        
        return normalized

    def get_feature_names(self) -> list:
        """Return list of feature names produced by this generator."""
        base_names = ['persistence_l0', 'persistence_l1', 'betti_0', 'betti_1']
        
        if self.feature_mode == 'v1.1':
            return base_names
        
        extended_names = ['entropy_l0', 'entropy_l1', 'max_lifetime_l0', 
                        'max_lifetime_l1', 'sum_lifetime_l0', 'sum_lifetime_l1']
        
        if self.feature_mode == 'v1.2':
            return base_names + extended_names
        
        # V1.3: Add enriched features
        enriched_names = [
            'top1_lifetime_l0', 'top2_lifetime_l0', 'top3_lifetime_l0',
            'top1_lifetime_l1', 'top2_lifetime_l1', 'top3_lifetime_l1',
            'count_large_l0', 'count_large_l1',
            'wasserstein_approx_l0', 'wasserstein_approx_l1',
        ]
        return base_names + extended_names + enriched_names

    def get_n_features(self) -> int:
        """Return number of TDA features produced."""
        return self.FEATURE_MODES.get(self.feature_mode, 20)
    
    def print_feature_summary(self, features_df: pd.DataFrame, ticker: str = ""):
        """Print debug summary of TDA feature statistics.
        
        V1.3: Useful for diagnostics and ablation experiments.
        """
        print(f"\n  TDA Feature Summary{f' ({ticker})' if ticker else ''}:")
        print(f"    Feature mode: {self.feature_mode}")
        print(f"    Number of features: {self.get_n_features()}")
        print(f"    Sample size: {len(features_df)}")
        print(f"    Feature statistics (mean ± std):")
        for col in features_df.columns:
            mean = features_df[col].mean()
            std = features_df[col].std()
            print(f"      {col}: {mean:.4f} ± {std:.4f}")


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
    
    # Test V1.3 enriched features
    generator_v13 = TDAFeatureGenerator(window=30, embedding_dim=3, feature_mode='v1.3')
    
    series = np.random.randn(100)
    embedded = generator_v13.takens_embedding(series)
    assert embedded.shape == (98, 3), f"Takens embedding failed: {embedded.shape}"
    
    features = generator_v13.compute_persistence_features(series)
    
    # Check all V1.3 features present
    expected_keys_v13 = [
        'persistence_l0', 'persistence_l1', 'betti_0', 'betti_1',
        'entropy_l0', 'entropy_l1', 'max_lifetime_l0', 'max_lifetime_l1',
        'sum_lifetime_l0', 'sum_lifetime_l1',
        'top1_lifetime_l0', 'top2_lifetime_l0', 'top3_lifetime_l0',
        'top1_lifetime_l1', 'top2_lifetime_l1', 'top3_lifetime_l1',
        'count_large_l0', 'count_large_l1',
        'wasserstein_approx_l0', 'wasserstein_approx_l1',
    ]
    assert all(k in features for k in expected_keys_v13), f"Missing V1.3 keys: {set(expected_keys_v13) - set(features.keys())}"
    assert all(not np.isnan(v) for v in features.values())
    
    result_v13 = generator_v13.generate_features(ohlcv)
    expected_rows = n_bars - 1 - generator_v13.window + 1
    assert len(result_v13) == expected_rows, f"Expected {expected_rows} rows, got {len(result_v13)}"
    assert len(result_v13.columns) == 20, f"Expected 20 columns (V1.3), got {len(result_v13.columns)}"
    assert generator_v13.get_n_features() == 20
    assert len(generator_v13.get_feature_names()) == 20
    
    # Test V1.2 extended features (backward compatibility)
    generator_v12 = TDAFeatureGenerator(window=30, embedding_dim=3, feature_mode='v1.2')
    result_v12 = generator_v12.generate_features(ohlcv)
    assert len(result_v12.columns) == 10, f"V1.2 mode should have 10 columns, got {len(result_v12.columns)}"
    assert generator_v12.get_n_features() == 10
    
    # Test V1.1 compatibility mode
    generator_v11 = TDAFeatureGenerator(window=30, embedding_dim=3, feature_mode='v1.1')
    result_v11 = generator_v11.generate_features(ohlcv)
    assert len(result_v11.columns) == 4, f"V1.1 mode should have 4 columns, got {len(result_v11.columns)}"
    assert generator_v11.get_n_features() == 4
    
    # Test backward compatibility with extended_features parameter
    generator_compat = TDAFeatureGenerator(window=30, embedding_dim=3, extended_features=True)
    assert generator_compat.feature_mode == 'v1.2'
    
    generator_compat_false = TDAFeatureGenerator(window=30, embedding_dim=3, extended_features=False)
    assert generator_compat_false.feature_mode == 'v1.1'
    
    # Test turbulence index
    turbulence = generator_v13.compute_turbulence_index(result_v13)
    assert len(turbulence) == len(result_v13)
    assert turbulence.min() >= 0 and turbulence.max() <= 1
    
    print("All V1.3 TDA tests passed!")
    return True


if __name__ == "__main__":
    success = test()
    if success:
        import sys
        sys.stdout.write("TDA Feature Generator V1.3: All tests passed\n")
