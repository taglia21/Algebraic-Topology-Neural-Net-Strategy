"""TDA Engine for Full-Universe Topological Analysis.

Phase 5: Computes persistent homology across thousands of stocks.
Uses correlation matrices to build simplicial complexes.
Extracts Betti numbers, persistence diagrams, and turbulence indices.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from scipy.spatial.distance import squareform, pdist
from scipy.cluster.hierarchy import linkage, fcluster
import warnings

warnings.filterwarnings('ignore')

# Import TDA libraries
try:
    from ripser import ripser
    from persim import PersistenceImager, plot_diagrams
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False
    print("Warning: ripser not available. Install with: pip install ripser persim")

try:
    import networkx as nx
    from networkx.algorithms.community import louvain_communities
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PersistenceFeatures:
    """Container for persistence homology features."""
    betti_0: int  # Connected components
    betti_1: int  # Loops/cycles
    betti_2: int  # Voids (if computed)
    
    # Persistence statistics
    total_persistence_h0: float  # Sum of H0 lifetimes
    total_persistence_h1: float  # Sum of H1 lifetimes
    max_persistence_h0: float    # Longest H0 lifetime
    max_persistence_h1: float    # Longest H1 lifetime
    
    # Entropy measures
    entropy_h0: float  # Persistence entropy for H0
    entropy_h1: float  # Persistence entropy for H1
    
    # Derived metrics
    fragmentation: float  # How fragmented is the market (high β₀)
    cyclicity: float      # How many cycles (high β₁)
    stability: float      # Persistence stability measure


@dataclass
class TDAFeatures:
    """Daily TDA features for the entire market."""
    date: str
    persistence: PersistenceFeatures
    turbulence_index: float  # 0-100 turbulence score
    regime_signal: str       # RISK_ON, RISK_OFF, TRANSITION
    n_clusters: int          # Number of detected clusters
    cluster_sizes: List[int] # Sizes of each cluster


class TDAEngine:
    """
    Topological Data Analysis engine for market-wide analysis.
    
    This is the core computational engine that:
    1. Computes correlation matrices across all stocks
    2. Builds distance matrices for TDA
    3. Runs persistent homology using Ripser
    4. Extracts Betti numbers and persistence features
    5. Detects market clusters using graph methods
    """
    
    def __init__(
        self,
        correlation_window: int = 30,
        min_stocks: int = 50,
        max_dimension: int = 1,  # H0 and H1 (H2 is expensive)
        distance_threshold: float = 0.7,  # For Betti number computation
    ):
        """
        Initialize TDA Engine.
        
        Args:
            correlation_window: Rolling window for correlation
            min_stocks: Minimum stocks required for analysis
            max_dimension: Maximum homology dimension (0, 1, or 2)
            distance_threshold: Threshold for persistent features
        """
        self.correlation_window = correlation_window
        self.min_stocks = min_stocks
        self.max_dimension = max_dimension
        self.distance_threshold = distance_threshold
        
        if not RIPSER_AVAILABLE:
            logger.warning("Ripser not available - TDA features will be limited")
    
    def compute_correlation_matrix(
        self,
        returns_df: pd.DataFrame,
        method: str = 'pearson'
    ) -> np.ndarray:
        """
        Compute correlation matrix from returns.
        
        Args:
            returns_df: DataFrame of daily returns (rows=dates, cols=stocks)
            method: Correlation method ('pearson' or 'spearman')
            
        Returns:
            Correlation matrix (n_stocks x n_stocks)
        """
        if len(returns_df) < self.correlation_window:
            raise ValueError(f"Need at least {self.correlation_window} days of data")
        
        # Use most recent window
        window_data = returns_df.iloc[-self.correlation_window:]
        
        # Compute correlation
        corr_matrix = window_data.corr(method=method).values
        
        # Handle NaN
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        
        # Ensure symmetry
        corr_matrix = (corr_matrix + corr_matrix.T) / 2
        
        # Ensure diagonal is 1
        np.fill_diagonal(corr_matrix, 1.0)
        
        return corr_matrix
    
    def correlation_to_distance(
        self,
        corr_matrix: np.ndarray,
        method: str = 'angular'
    ) -> np.ndarray:
        """
        Convert correlation matrix to distance matrix.
        
        Args:
            corr_matrix: Correlation matrix
            method: Distance method
                - 'angular': d = sqrt(2 * (1 - corr)) - standard for TDA
                - 'absolute': d = 1 - |corr|
                
        Returns:
            Distance matrix (n_stocks x n_stocks)
        """
        if method == 'angular':
            # Angular distance: accounts for both positive and negative correlation
            distance = np.sqrt(2 * (1 - corr_matrix))
        elif method == 'absolute':
            # Absolute: treats negative correlation as similar
            distance = 1 - np.abs(corr_matrix)
        else:
            raise ValueError(f"Unknown distance method: {method}")
        
        # Ensure non-negative
        distance = np.clip(distance, 0, 2)
        
        # Ensure diagonal is 0
        np.fill_diagonal(distance, 0)
        
        return distance
    
    def compute_persistence_diagrams(
        self,
        distance_matrix: np.ndarray,
    ) -> Dict:
        """
        Compute persistent homology using Ripser.
        
        Args:
            distance_matrix: Distance matrix (n x n)
            
        Returns:
            Dict with persistence diagrams for each dimension
        """
        if not RIPSER_AVAILABLE:
            return self._fallback_persistence(distance_matrix)
        
        # Run Ripser
        result = ripser(
            distance_matrix,
            maxdim=self.max_dimension,
            distance_matrix=True,  # Input is already distance matrix
        )
        
        return {
            'dgms': result['dgms'],
            'num_points': distance_matrix.shape[0],
        }
    
    def _fallback_persistence(self, distance_matrix: np.ndarray) -> Dict:
        """Fallback when Ripser is not available."""
        n = distance_matrix.shape[0]
        
        # Simple approximation using clustering
        # H0: Number of connected components at threshold
        threshold = np.median(distance_matrix[distance_matrix > 0])
        adjacency = (distance_matrix < threshold).astype(int)
        np.fill_diagonal(adjacency, 0)
        
        # Count components via simple connected components
        # This is a rough approximation
        from scipy.sparse.csgraph import connected_components
        from scipy.sparse import csr_matrix
        
        n_components, labels = connected_components(
            csr_matrix(adjacency),
            directed=False
        )
        
        # Create fake persistence diagrams
        h0_dgm = np.array([[0, threshold]] * n_components)
        h1_dgm = np.array([]).reshape(0, 2)
        
        return {
            'dgms': [h0_dgm, h1_dgm],
            'num_points': n,
            'fallback': True
        }
    
    def extract_persistence_features(
        self,
        persistence_result: Dict,
    ) -> PersistenceFeatures:
        """
        Extract numerical features from persistence diagrams.
        
        Args:
            persistence_result: Result from compute_persistence_diagrams
            
        Returns:
            PersistenceFeatures dataclass
        """
        dgms = persistence_result['dgms']
        
        # H0 (connected components)
        h0_dgm = dgms[0]
        h0_finite = h0_dgm[np.isfinite(h0_dgm[:, 1])] if len(h0_dgm) > 0 else np.array([]).reshape(0, 2)
        
        if len(h0_finite) > 0:
            h0_lifetimes = h0_finite[:, 1] - h0_finite[:, 0]
            h0_lifetimes = h0_lifetimes[h0_lifetimes > 0]
        else:
            h0_lifetimes = np.array([0])
        
        # H1 (loops)
        h1_dgm = dgms[1] if len(dgms) > 1 else np.array([]).reshape(0, 2)
        h1_finite = h1_dgm[np.isfinite(h1_dgm[:, 1])] if len(h1_dgm) > 0 else np.array([]).reshape(0, 2)
        
        if len(h1_finite) > 0:
            h1_lifetimes = h1_finite[:, 1] - h1_finite[:, 0]
            h1_lifetimes = h1_lifetimes[h1_lifetimes > 0]
        else:
            h1_lifetimes = np.array([0])
        
        # Betti numbers (count of significant features)
        betti_0 = len(h0_lifetimes[h0_lifetimes > 0.1]) if len(h0_lifetimes) > 0 else 0
        betti_1 = len(h1_lifetimes[h1_lifetimes > 0.1]) if len(h1_lifetimes) > 0 else 0
        
        # Persistence statistics
        total_h0 = np.sum(h0_lifetimes) if len(h0_lifetimes) > 0 else 0
        total_h1 = np.sum(h1_lifetimes) if len(h1_lifetimes) > 0 else 0
        max_h0 = np.max(h0_lifetimes) if len(h0_lifetimes) > 0 else 0
        max_h1 = np.max(h1_lifetimes) if len(h1_lifetimes) > 0 else 0
        
        # Entropy
        entropy_h0 = self._compute_entropy(h0_lifetimes)
        entropy_h1 = self._compute_entropy(h1_lifetimes)
        
        # Derived metrics
        n_stocks = persistence_result['num_points']
        fragmentation = betti_0 / max(n_stocks, 1)  # Normalized by universe size
        cyclicity = betti_1 / max(n_stocks, 1)
        stability = max_h0 + max_h1  # Higher = more stable structures
        
        return PersistenceFeatures(
            betti_0=betti_0,
            betti_1=betti_1,
            betti_2=0,  # Not computed by default
            total_persistence_h0=float(total_h0),
            total_persistence_h1=float(total_h1),
            max_persistence_h0=float(max_h0),
            max_persistence_h1=float(max_h1),
            entropy_h0=float(entropy_h0),
            entropy_h1=float(entropy_h1),
            fragmentation=float(fragmentation),
            cyclicity=float(cyclicity),
            stability=float(stability),
        )
    
    def _compute_entropy(self, lifetimes: np.ndarray) -> float:
        """Compute Shannon entropy of lifetime distribution."""
        if len(lifetimes) == 0 or np.sum(lifetimes) == 0:
            return 0.0
        
        # Normalize to probabilities
        probs = lifetimes / np.sum(lifetimes)
        probs = probs[probs > 0]  # Avoid log(0)
        
        # Shannon entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        return float(entropy)
    
    def detect_clusters(
        self,
        corr_matrix: np.ndarray,
        tickers: List[str],
        method: str = 'louvain',
        min_cluster_size: int = 5,
    ) -> Tuple[Dict[str, int], Dict[int, List[str]]]:
        """
        Detect market clusters using graph community detection.
        
        Args:
            corr_matrix: Correlation matrix
            tickers: List of ticker symbols
            method: Clustering method ('louvain', 'hierarchical')
            min_cluster_size: Minimum cluster size
            
        Returns:
            Tuple of (ticker_to_cluster, cluster_to_tickers)
        """
        n = len(tickers)
        
        if method == 'louvain' and NETWORKX_AVAILABLE:
            return self._cluster_louvain(corr_matrix, tickers, min_cluster_size)
        else:
            return self._cluster_hierarchical(corr_matrix, tickers, min_cluster_size)
    
    def _cluster_louvain(
        self,
        corr_matrix: np.ndarray,
        tickers: List[str],
        min_cluster_size: int,
    ) -> Tuple[Dict[str, int], Dict[int, List[str]]]:
        """Cluster using Louvain community detection."""
        # Build graph from correlation matrix
        # Use positive correlations as edges with weight = correlation
        G = nx.Graph()
        
        for i, ticker_i in enumerate(tickers):
            G.add_node(ticker_i)
        
        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                weight = corr_matrix[i, j]
                if weight > 0.3:  # Only add edges for significant correlations
                    G.add_edge(tickers[i], tickers[j], weight=weight)
        
        # Run Louvain
        communities = louvain_communities(G, weight='weight', seed=42)
        
        # Convert to mappings
        ticker_to_cluster = {}
        cluster_to_tickers = {}
        
        for cluster_id, community in enumerate(communities):
            if len(community) >= min_cluster_size:
                cluster_to_tickers[cluster_id] = list(community)
                for ticker in community:
                    ticker_to_cluster[ticker] = cluster_id
        
        # Assign unclustered to cluster -1
        for ticker in tickers:
            if ticker not in ticker_to_cluster:
                ticker_to_cluster[ticker] = -1
        
        return ticker_to_cluster, cluster_to_tickers
    
    def _cluster_hierarchical(
        self,
        corr_matrix: np.ndarray,
        tickers: List[str],
        min_cluster_size: int,
    ) -> Tuple[Dict[str, int], Dict[int, List[str]]]:
        """Cluster using hierarchical clustering."""
        # Convert correlation to distance
        distance = 1 - corr_matrix
        np.fill_diagonal(distance, 0)
        
        # Condensed distance matrix
        condensed = squareform(distance)
        
        # Hierarchical clustering
        Z = linkage(condensed, method='ward')
        
        # Cut tree to get clusters
        n_clusters = max(5, len(tickers) // 50)  # Aim for ~50 stocks per cluster
        labels = fcluster(Z, t=n_clusters, criterion='maxclust')
        
        # Build mappings
        ticker_to_cluster = {}
        cluster_to_tickers = {}
        
        for ticker, label in zip(tickers, labels):
            ticker_to_cluster[ticker] = label
            if label not in cluster_to_tickers:
                cluster_to_tickers[label] = []
            cluster_to_tickers[label].append(ticker)
        
        # Filter small clusters
        cluster_to_tickers = {
            k: v for k, v in cluster_to_tickers.items() 
            if len(v) >= min_cluster_size
        }
        
        return ticker_to_cluster, cluster_to_tickers
    
    def compute_turbulence_index(
        self,
        current_features: PersistenceFeatures,
        history: List[PersistenceFeatures],
        lookback: int = 20,
    ) -> float:
        """
        Compute market turbulence index from TDA features.
        
        Turbulence signals:
        - Rapidly changing Betti numbers
        - Decreasing persistence (unstable topology)
        - Increasing fragmentation
        - High entropy (chaotic structure)
        
        Args:
            current_features: Current day's TDA features
            history: Historical TDA features
            lookback: Days of history to consider
            
        Returns:
            Turbulence index 0-100
        """
        if len(history) < 5:
            return 30.0  # Neutral-low if not enough history
        
        recent = history[-lookback:] if len(history) >= lookback else history
        
        # 1. Betti number volatility (0-25 points)
        # Only count as turbulent if Betti numbers are changing rapidly
        betti_0_series = [f.betti_0 for f in recent]
        betti_1_series = [f.betti_1 for f in recent]
        
        betti_0_mean = np.mean(betti_0_series) if betti_0_series else 1
        betti_1_mean = np.mean(betti_1_series) if betti_1_series else 1
        
        # Relative volatility (normalized by mean)
        betti_0_cv = np.std(betti_0_series) / max(betti_0_mean, 1)
        betti_1_cv = np.std(betti_1_series) / max(betti_1_mean, 1)
        
        betti_vol = (betti_0_cv + betti_1_cv) / 2
        betti_score = min(25, betti_vol * 50)  # CV > 0.5 = high turbulence
        
        # 2. Persistence trend (0-25 points)
        # Declining persistence = increasing turbulence
        persistence_series = [f.total_persistence_h0 + f.total_persistence_h1 for f in recent]
        if len(persistence_series) > 5:
            persistence_mean = np.mean(persistence_series)
            persistence_slope = np.polyfit(range(len(persistence_series)), persistence_series, 1)[0]
            # Normalize by mean persistence
            relative_slope = persistence_slope / max(persistence_mean, 0.1)
            if relative_slope < -0.05:  # Declining by more than 5% per period
                persistence_score = min(25, abs(relative_slope) * 100)
            else:
                persistence_score = 0
        else:
            persistence_score = 0
        
        # 3. Current fragmentation vs historical (0-25 points)
        # Compare to historical baseline
        historical_frag = [f.fragmentation for f in recent]
        frag_baseline = np.mean(historical_frag) if historical_frag else 0.2
        frag_excess = (current_features.fragmentation - frag_baseline) / max(frag_baseline, 0.1)
        fragmentation_score = min(25, max(0, frag_excess * 25))  # Only penalize above average
        
        # 4. Entropy level vs baseline (0-25 points)
        historical_entropy = [(f.entropy_h0 + f.entropy_h1) / 2 for f in recent]
        entropy_baseline = np.mean(historical_entropy) if historical_entropy else 1.0
        current_entropy = (current_features.entropy_h0 + current_features.entropy_h1) / 2
        entropy_excess = (current_entropy - entropy_baseline) / max(entropy_baseline, 0.1)
        entropy_score = min(25, max(0, entropy_excess * 25))  # Only penalize above average
        
        # Total turbulence
        turbulence = betti_score + persistence_score + fragmentation_score + entropy_score
        turbulence = np.clip(turbulence, 0, 100)
        
        return float(turbulence)
    
    def analyze_market(
        self,
        returns_df: pd.DataFrame,
        tickers: List[str],
        date: str,
        history: List[TDAFeatures] = None,
    ) -> TDAFeatures:
        """
        Complete TDA analysis for a single day.
        
        Args:
            returns_df: Returns DataFrame up to and including target date
            tickers: List of ticker symbols (matching returns_df columns)
            date: Target date string
            history: Previous TDA features for turbulence calculation
            
        Returns:
            TDAFeatures for the day
        """
        history = history or []
        
        # Step 1: Correlation matrix
        corr_matrix = self.compute_correlation_matrix(returns_df)
        
        # Step 2: Distance matrix
        distance_matrix = self.correlation_to_distance(corr_matrix)
        
        # Step 3: Persistent homology
        persistence = self.compute_persistence_diagrams(distance_matrix)
        
        # Step 4: Extract features
        features = self.extract_persistence_features(persistence)
        
        # Step 5: Detect clusters
        ticker_to_cluster, cluster_to_tickers = self.detect_clusters(
            corr_matrix, tickers
        )
        
        # Step 6: Compute turbulence
        history_features = [t.persistence for t in history] if history else []
        turbulence = self.compute_turbulence_index(features, history_features)
        
        # Step 7: Determine regime
        if turbulence < 30:
            regime = "RISK_ON"
        elif turbulence < 60:
            regime = "NEUTRAL"
        elif turbulence < 80:
            regime = "RISK_OFF"
        else:
            regime = "CRISIS"
        
        # Build result
        cluster_sizes = [len(v) for v in cluster_to_tickers.values()]
        
        return TDAFeatures(
            date=date,
            persistence=features,
            turbulence_index=turbulence,
            regime_signal=regime,
            n_clusters=len(cluster_to_tickers),
            cluster_sizes=sorted(cluster_sizes, reverse=True)[:10],
        )


if __name__ == "__main__":
    print("Testing TDA Engine...")
    print("=" * 60)
    
    # Create synthetic test data
    np.random.seed(42)
    n_stocks = 100
    n_days = 60
    
    # Generate correlated returns (simulate market structure)
    base_returns = np.random.randn(n_days, 1) * 0.01  # Market factor
    stock_returns = base_returns + np.random.randn(n_days, n_stocks) * 0.02
    
    tickers = [f"STOCK{i:03d}" for i in range(n_stocks)]
    returns_df = pd.DataFrame(stock_returns, columns=tickers)
    
    # Test TDA engine
    engine = TDAEngine(correlation_window=30)
    
    print(f"\nTest data: {n_stocks} stocks, {n_days} days")
    
    # Compute correlation
    corr = engine.compute_correlation_matrix(returns_df)
    print(f"Correlation matrix shape: {corr.shape}")
    print(f"Avg correlation: {np.mean(corr[np.triu_indices_from(corr, k=1)]):.4f}")
    
    # Compute distance
    dist = engine.correlation_to_distance(corr)
    print(f"Distance matrix range: [{dist.min():.4f}, {dist.max():.4f}]")
    
    # Compute persistence
    persistence = engine.compute_persistence_diagrams(dist)
    print(f"Persistence diagrams computed")
    
    # Extract features
    features = engine.extract_persistence_features(persistence)
    print(f"\nTDA Features:")
    print(f"  Betti-0 (components): {features.betti_0}")
    print(f"  Betti-1 (loops): {features.betti_1}")
    print(f"  Total persistence H0: {features.total_persistence_h0:.4f}")
    print(f"  Total persistence H1: {features.total_persistence_h1:.4f}")
    print(f"  Entropy H0: {features.entropy_h0:.4f}")
    print(f"  Fragmentation: {features.fragmentation:.4f}")
    
    # Detect clusters
    ticker_to_cluster, cluster_to_tickers = engine.detect_clusters(corr, tickers)
    print(f"\nClusters detected: {len(cluster_to_tickers)}")
    for cid, members in list(cluster_to_tickers.items())[:3]:
        print(f"  Cluster {cid}: {len(members)} stocks")
    
    # Full analysis
    tda_result = engine.analyze_market(returns_df, tickers, "2024-01-01")
    print(f"\nFull Analysis Result:")
    print(f"  Turbulence Index: {tda_result.turbulence_index:.1f}")
    print(f"  Regime: {tda_result.regime_signal}")
    print(f"  Clusters: {tda_result.n_clusters}")
    
    print("\nTDA Engine tests complete!")
