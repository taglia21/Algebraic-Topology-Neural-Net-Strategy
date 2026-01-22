"""
Persistent Laplacian for Enhanced TDA Features

Extends standard persistent homology with spectral information from
the combinatorial Laplacian, capturing richer topological structure.

Key Features:
- Persistent Laplacian eigenvalues across filtrations
- Betti curves from persistent homology
- Spectral gap analysis for topological stability
- 12 new topological features for regime detection

Reference: Wang, Wei, et al. "Persistent spectral graph" (2020)
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    from scipy import sparse
    from scipy.sparse.linalg import eigsh, ArpackNoConvergence
    from scipy.spatial.distance import pdist, squareform
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from ripser import ripser
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False

try:
    from persim import wasserstein, bottleneck
    PERSIM_AVAILABLE = True
except ImportError:
    PERSIM_AVAILABLE = False

logger = logging.getLogger(__name__)


class PersistentLaplacian:
    """
    Compute persistent Laplacian features from financial time series.
    
    The persistent Laplacian extends persistent homology by computing
    the combinatorial Laplacian at each filtration level, providing
    spectral information about topological features.
    
    For dimension 0 (connected components):
        L_0 = B_1 @ B_1.T  (boundary matrix of 1-simplices)
    
    For dimension 1 (loops):
        L_1 = B_1.T @ B_1 + B_2 @ B_2.T
    
    Key spectral features:
    - Smallest non-zero eigenvalue: "algebraic connectivity"
    - Spectral gap: Stability of topological features
    - Eigenvalue distribution: Shape of feature landscape
    """
    
    def __init__(self, max_dimension: int = 1, 
                 n_filtrations: int = 20,
                 max_edge_length: float = 2.0,
                 n_eigenvalues: int = 10):
        """
        Args:
            max_dimension: Maximum homology dimension to compute (0 or 1)
            n_filtrations: Number of filtration steps
            max_edge_length: Maximum edge length in filtration
            n_eigenvalues: Number of eigenvalues to compute per step
        """
        self.max_dimension = min(max_dimension, 1)  # Limit to H_0 and H_1
        self.n_filtrations = n_filtrations
        self.max_edge_length = max_edge_length
        self.n_eigenvalues = n_eigenvalues
        
        if not SCIPY_AVAILABLE:
            logger.warning("scipy not available - Laplacian computation disabled")
    
    def compute_distance_matrix(self, returns: np.ndarray, 
                                 method: str = 'correlation') -> np.ndarray:
        """
        Compute pairwise distance matrix from return series.
        
        Args:
            returns: (n_samples, n_assets) return matrix
            method: Distance method ('correlation', 'euclidean', 'dtw')
        
        Returns:
            (n_assets, n_assets) distance matrix
        """
        if returns.ndim == 1:
            # Single asset: use sliding windows
            window = min(20, len(returns) // 5)
            n_windows = len(returns) - window + 1
            if n_windows < 5:
                return np.zeros((5, 5))
            
            windows = np.array([returns[i:i+window] for i in range(n_windows)])
            returns = windows[:min(50, n_windows)]  # Limit for efficiency
        
        if method == 'correlation':
            # Correlation distance: d = sqrt(2 * (1 - corr))
            corr = np.corrcoef(returns.T)
            corr = np.clip(corr, -1, 1)
            dist = np.sqrt(2 * (1 - corr))
        else:
            # Euclidean distance
            dist = squareform(pdist(returns.T, metric='euclidean'))
        
        # Normalize to [0, max_edge_length]
        if dist.max() > 0:
            dist = dist / dist.max() * self.max_edge_length
        
        return dist
    
    def build_simplicial_complex(self, dist_matrix: np.ndarray, 
                                  threshold: float) -> Dict:
        """
        Build simplicial complex at given filtration threshold.
        
        Returns:
            Dictionary with 'vertices', 'edges', 'triangles'
        """
        n = dist_matrix.shape[0]
        
        # Vertices (0-simplices)
        vertices = list(range(n))
        
        # Edges (1-simplices) where distance <= threshold
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if dist_matrix[i, j] <= threshold:
                    edges.append((i, j))
        
        # Triangles (2-simplices) - only if computing H_1
        triangles = []
        if self.max_dimension >= 1:
            edge_set = set(edges)
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        if ((i, j) in edge_set and 
                            (i, k) in edge_set and 
                            (j, k) in edge_set):
                            triangles.append((i, j, k))
        
        return {
            'vertices': vertices,
            'edges': edges,
            'triangles': triangles
        }
    
    def compute_boundary_matrices(self, complex: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute boundary matrices B_1 (edges->vertices) and B_2 (triangles->edges).
        
        B_1[v, e] = ±1 if vertex v is endpoint of edge e
        B_2[e, t] = ±1 if edge e is face of triangle t
        """
        n_vertices = len(complex['vertices'])
        n_edges = len(complex['edges'])
        n_triangles = len(complex['triangles'])
        
        # B_1: vertices x edges
        if n_edges > 0:
            B1 = np.zeros((n_vertices, n_edges))
            for e_idx, (i, j) in enumerate(complex['edges']):
                B1[i, e_idx] = -1
                B1[j, e_idx] = 1
        else:
            B1 = np.zeros((n_vertices, 1))
        
        # B_2: edges x triangles
        if n_triangles > 0 and n_edges > 0:
            edge_to_idx = {e: idx for idx, e in enumerate(complex['edges'])}
            B2 = np.zeros((n_edges, n_triangles))
            
            for t_idx, (i, j, k) in enumerate(complex['triangles']):
                # Triangle edges with orientation
                edges = [(i, j), (i, k), (j, k)]
                signs = [1, -1, 1]
                
                for edge, sign in zip(edges, signs):
                    if edge in edge_to_idx:
                        B2[edge_to_idx[edge], t_idx] = sign
                    elif (edge[1], edge[0]) in edge_to_idx:
                        B2[edge_to_idx[(edge[1], edge[0])], t_idx] = -sign
        else:
            B2 = np.zeros((max(1, n_edges), 1))
        
        return B1, B2
    
    def compute_laplacian_eigenvalues(self, L: np.ndarray, k: int = None) -> np.ndarray:
        """
        Compute smallest k eigenvalues of Laplacian.
        
        Args:
            L: Laplacian matrix
            k: Number of eigenvalues (default: self.n_eigenvalues)
        
        Returns:
            Array of eigenvalues (sorted ascending)
        """
        k = k or self.n_eigenvalues
        n = L.shape[0]
        
        if n <= 1:
            return np.zeros(k)
        
        # Limit k to matrix size
        k = min(k, n - 1)
        
        if k <= 0:
            return np.zeros(self.n_eigenvalues)
        
        try:
            if n < 50:
                # Full eigendecomposition for small matrices
                eigenvalues = np.linalg.eigvalsh(L)
                eigenvalues = np.sort(eigenvalues)[:k]
            else:
                # Sparse solver for large matrices
                L_sparse = sparse.csr_matrix(L)
                eigenvalues, _ = eigsh(L_sparse, k=k, which='SM')
                eigenvalues = np.sort(eigenvalues)
        except (ArpackNoConvergence, np.linalg.LinAlgError):
            eigenvalues = np.zeros(k)
        
        # Pad to expected size
        result = np.zeros(self.n_eigenvalues)
        result[:len(eigenvalues)] = eigenvalues[:self.n_eigenvalues]
        
        return result
    
    def compute_persistent_laplacian(self, dist_matrix: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute persistent Laplacian eigenvalues across filtration.
        
        Args:
            dist_matrix: Pairwise distance matrix
        
        Returns:
            Dictionary with L0 and L1 eigenvalues at each filtration step
        """
        thresholds = np.linspace(0, self.max_edge_length, self.n_filtrations)
        
        L0_eigenvalues = []
        L1_eigenvalues = []
        betti_0 = []
        betti_1 = []
        
        for threshold in thresholds:
            # Build complex at this threshold
            complex = self.build_simplicial_complex(dist_matrix, threshold)
            
            # Boundary matrices
            B1, B2 = self.compute_boundary_matrices(complex)
            
            # L_0 = B_1 @ B_1.T (graph Laplacian)
            L0 = B1 @ B1.T
            eigs_0 = self.compute_laplacian_eigenvalues(L0)
            L0_eigenvalues.append(eigs_0)
            
            # Betti_0 = nullity of L_0 = number of connected components
            betti_0.append(np.sum(eigs_0 < 1e-10))
            
            if self.max_dimension >= 1:
                # L_1 = B_1.T @ B_1 + B_2 @ B_2.T
                L1 = B1.T @ B1 + B2 @ B2.T
                
                if L1.shape[0] > 1:
                    eigs_1 = self.compute_laplacian_eigenvalues(L1)
                else:
                    eigs_1 = np.zeros(self.n_eigenvalues)
                
                L1_eigenvalues.append(eigs_1)
                
                # Betti_1 = nullity of L_1 = number of independent loops
                betti_1.append(np.sum(eigs_1 < 1e-10))
            else:
                L1_eigenvalues.append(np.zeros(self.n_eigenvalues))
                betti_1.append(0)
        
        return {
            'L0_eigenvalues': np.array(L0_eigenvalues),
            'L1_eigenvalues': np.array(L1_eigenvalues),
            'betti_0': np.array(betti_0),
            'betti_1': np.array(betti_1),
            'thresholds': thresholds
        }
    
    def compute_persistence_diagram(self, dist_matrix: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute persistent homology using ripser.
        
        Args:
            dist_matrix: Pairwise distance matrix
        
        Returns:
            Dictionary with H0 and H1 persistence diagrams
        """
        if not RIPSER_AVAILABLE:
            # Fallback: compute from Laplacian
            result = self.compute_persistent_laplacian(dist_matrix)
            
            # Approximate persistence from Betti numbers
            betti_0 = result['betti_0']
            betti_1 = result['betti_1']
            thresholds = result['thresholds']
            
            # Convert Betti curve to approximate persistence pairs
            h0_dgm = []
            h1_dgm = []
            
            # H0: Features born at 0, die when component merges
            for i in range(1, len(betti_0)):
                if betti_0[i] < betti_0[i-1]:
                    # Component merged
                    for _ in range(int(betti_0[i-1] - betti_0[i])):
                        h0_dgm.append([0, thresholds[i]])
            
            # Add essential features (never die)
            for _ in range(int(betti_0[-1])):
                h0_dgm.append([0, np.inf])
            
            # H1: Features born when loop forms, die when filled
            for i in range(1, len(betti_1)):
                if betti_1[i] > betti_1[i-1]:
                    # Loop formed
                    for _ in range(int(betti_1[i] - betti_1[i-1])):
                        h1_dgm.append([thresholds[i], np.inf])
                elif betti_1[i] < betti_1[i-1]:
                    # Loop filled
                    for _ in range(int(betti_1[i-1] - betti_1[i])):
                        if h1_dgm:
                            # Update death time of latest loop
                            for j in range(len(h1_dgm)-1, -1, -1):
                                if h1_dgm[j][1] == np.inf:
                                    h1_dgm[j][1] = thresholds[i]
                                    break
            
            return {
                'H0': np.array(h0_dgm) if h0_dgm else np.zeros((0, 2)),
                'H1': np.array(h1_dgm) if h1_dgm else np.zeros((0, 2))
            }
        
        # Use ripser for efficient computation
        result = ripser(dist_matrix, maxdim=self.max_dimension, 
                       distance_matrix=True, thresh=self.max_edge_length)
        
        return {
            'H0': result['dgms'][0],
            'H1': result['dgms'][1] if self.max_dimension >= 1 else np.zeros((0, 2))
        }
    
    def extract_features(self, returns: np.ndarray, 
                         include_persistence: bool = True) -> Dict[str, float]:
        """
        Extract 12 topological features from return series.
        
        Features:
        1-3: L0 eigenvalue statistics (min nonzero, mean, std)
        4-6: L1 eigenvalue statistics (min nonzero, mean, std)
        7-8: Spectral gaps (L0, L1)
        9-10: Betti curve integrals (area under curve)
        11-12: Persistence statistics (total persistence, max persistence)
        
        Args:
            returns: Return series (1D or 2D)
            include_persistence: Whether to compute persistence diagram
        
        Returns:
            Dictionary of 12 topological features
        """
        features = {
            'L0_min_nonzero': 0.0,
            'L0_mean': 0.0,
            'L0_std': 0.0,
            'L1_min_nonzero': 0.0,
            'L1_mean': 0.0,
            'L1_std': 0.0,
            'spectral_gap_L0': 0.0,
            'spectral_gap_L1': 0.0,
            'betti_0_integral': 0.0,
            'betti_1_integral': 0.0,
            'total_persistence': 0.0,
            'max_persistence': 0.0,
        }
        
        if not SCIPY_AVAILABLE or len(returns) < 10:
            return features
        
        try:
            # Compute distance matrix
            dist_matrix = self.compute_distance_matrix(returns)
            
            if dist_matrix.shape[0] < 3:
                return features
            
            # Compute persistent Laplacian
            pl_result = self.compute_persistent_laplacian(dist_matrix)
            
            # L0 eigenvalue statistics (across all filtration steps)
            L0_eigs = pl_result['L0_eigenvalues'].flatten()
            L0_nonzero = L0_eigs[L0_eigs > 1e-10]
            
            if len(L0_nonzero) > 0:
                features['L0_min_nonzero'] = float(np.min(L0_nonzero))
                features['L0_mean'] = float(np.mean(L0_nonzero))
                features['L0_std'] = float(np.std(L0_nonzero))
            
            # L1 eigenvalue statistics
            L1_eigs = pl_result['L1_eigenvalues'].flatten()
            L1_nonzero = L1_eigs[L1_eigs > 1e-10]
            
            if len(L1_nonzero) > 0:
                features['L1_min_nonzero'] = float(np.min(L1_nonzero))
                features['L1_mean'] = float(np.mean(L1_nonzero))
                features['L1_std'] = float(np.std(L1_nonzero))
            
            # Spectral gaps
            for t in range(len(pl_result['thresholds'])):
                eigs_0 = np.sort(pl_result['L0_eigenvalues'][t])
                eigs_1 = np.sort(pl_result['L1_eigenvalues'][t])
                
                # Gap = smallest nonzero eigenvalue (algebraic connectivity)
                nonzero_0 = eigs_0[eigs_0 > 1e-10]
                nonzero_1 = eigs_1[eigs_1 > 1e-10]
                
                if len(nonzero_0) > 0:
                    features['spectral_gap_L0'] = max(features['spectral_gap_L0'], 
                                                      float(nonzero_0[0]))
                if len(nonzero_1) > 0:
                    features['spectral_gap_L1'] = max(features['spectral_gap_L1'], 
                                                      float(nonzero_1[0]))
            
            # Betti curve integrals (area under Betti number vs threshold)
            dt = pl_result['thresholds'][1] - pl_result['thresholds'][0]
            # Use scipy.integrate.trapezoid or np.trapezoid (numpy 2.0+) or fallback
            try:
                from scipy.integrate import trapezoid
                features['betti_0_integral'] = float(trapezoid(pl_result['betti_0'], dx=dt))
                features['betti_1_integral'] = float(trapezoid(pl_result['betti_1'], dx=dt))
            except ImportError:
                # Fallback: simple sum approximation
                features['betti_0_integral'] = float(np.sum(pl_result['betti_0']) * dt)
                features['betti_1_integral'] = float(np.sum(pl_result['betti_1']) * dt)
            
            # Persistence statistics
            if include_persistence:
                dgm = self.compute_persistence_diagram(dist_matrix)
                
                # Filter out infinite deaths
                h0_finite = dgm['H0'][np.isfinite(dgm['H0'][:, 1])] if len(dgm['H0']) > 0 else np.zeros((0, 2))
                h1_finite = dgm['H1'][np.isfinite(dgm['H1'][:, 1])] if len(dgm['H1']) > 0 else np.zeros((0, 2))
                
                # Total persistence = sum of (death - birth)
                if len(h0_finite) > 0:
                    pers_0 = h0_finite[:, 1] - h0_finite[:, 0]
                else:
                    pers_0 = np.array([0.0])
                
                if len(h1_finite) > 0:
                    pers_1 = h1_finite[:, 1] - h1_finite[:, 0]
                else:
                    pers_1 = np.array([0.0])
                
                all_pers = np.concatenate([pers_0, pers_1])
                features['total_persistence'] = float(np.sum(all_pers))
                features['max_persistence'] = float(np.max(all_pers)) if len(all_pers) > 0 else 0.0
        
        except Exception as e:
            logger.warning(f"Error computing persistent Laplacian features: {e}")
        
        return features
    
    def compute_market_topology(self, returns_matrix: np.ndarray) -> Dict[str, float]:
        """
        Compute market-wide topological features from cross-sectional returns.
        
        Args:
            returns_matrix: (n_samples, n_assets) return matrix
        
        Returns:
            Dictionary of market topology features
        """
        features = self.extract_features(returns_matrix)
        
        # Add market-specific interpretations
        market_features = {
            # Connectivity: lower spectral gap = more interconnected market = higher systemic risk
            'systemic_risk': 1.0 / (features['spectral_gap_L0'] + 0.1),
            
            # Clustering: higher Betti_0 integral = more fragmented market
            'market_fragmentation': features['betti_0_integral'],
            
            # Cycles: higher Betti_1 integral = more circular correlations = potential instability
            'correlation_cycles': features['betti_1_integral'],
            
            # Overall topological complexity
            'topological_complexity': features['total_persistence'],
        }
        
        # Merge all features
        features.update(market_features)
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names in order."""
        return [
            'L0_min_nonzero', 'L0_mean', 'L0_std',
            'L1_min_nonzero', 'L1_mean', 'L1_std',
            'spectral_gap_L0', 'spectral_gap_L1',
            'betti_0_integral', 'betti_1_integral',
            'total_persistence', 'max_persistence'
        ]
    
    def get_feature_vector(self, returns: np.ndarray) -> np.ndarray:
        """
        Get feature vector suitable for ML input.
        
        Args:
            returns: Return series
        
        Returns:
            numpy array of 12 features
        """
        features = self.extract_features(returns)
        return np.array([features[name] for name in self.get_feature_names()])


# =============================================================================
# INTEGRATION WITH V1.3 TDA
# =============================================================================

class EnhancedTDAFeatures:
    """
    Combined V1.3 TDA + V2.0 Persistent Laplacian features.
    
    Total features: 22 (10 base + 12 persistent Laplacian)
    """
    
    def __init__(self, use_laplacian: bool = True):
        """
        Args:
            use_laplacian: Whether to include persistent Laplacian features
        """
        self.use_laplacian = use_laplacian
        self.laplacian = PersistentLaplacian() if use_laplacian else None
        
        # V1.3 base TDA features (placeholder for compatibility)
        self.base_features = [
            'betti_0', 'betti_1', 'betti_2',
            'persistence_entropy', 'wasserstein_0', 'wasserstein_1',
            'landscape_mean', 'landscape_max', 'silhouette_mean', 'silhouette_max'
        ]
    
    def compute_all_features(self, returns: np.ndarray, 
                              reference_returns: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compute all TDA features (V1.3 + V2.0).
        
        Args:
            returns: Current return series
            reference_returns: Reference period for comparison (optional)
        
        Returns:
            Dictionary of 22 TDA features
        """
        features = {}
        
        # V1.3 base features (placeholder values if not available)
        for name in self.base_features:
            features[name] = 0.0
        
        # V2.0 persistent Laplacian features
        if self.laplacian:
            pl_features = self.laplacian.extract_features(returns)
            features.update(pl_features)
        
        return features
    
    def get_feature_vector(self, returns: np.ndarray) -> np.ndarray:
        """Get combined feature vector."""
        features = self.compute_all_features(returns)
        return np.array(list(features.values()))


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_rolling_laplacian_features(prices: np.ndarray, 
                                        window: int = 60,
                                        step: int = 5) -> np.ndarray:
    """
    Compute rolling persistent Laplacian features over time.
    
    Args:
        prices: Price time series
        window: Rolling window size
        step: Step size between windows
    
    Returns:
        (n_windows, 12) array of features
    """
    pl = PersistentLaplacian()
    
    n_windows = (len(prices) - window) // step + 1
    features = np.zeros((n_windows, 12))
    
    for i in range(n_windows):
        start = i * step
        end = start + window
        
        window_prices = prices[start:end]
        returns = np.diff(np.log(window_prices + 1e-10))
        
        feat_vec = pl.get_feature_vector(returns)
        features[i] = feat_vec
    
    return features


def detect_topological_regime_change(features_history: np.ndarray,
                                     threshold: float = 2.0) -> bool:
    """
    Detect regime change from topological feature changes.
    
    Args:
        features_history: (n_windows, 12) feature history
        threshold: Z-score threshold for anomaly detection
    
    Returns:
        True if regime change detected
    """
    if len(features_history) < 10:
        return False
    
    # Use last 10 windows for baseline
    baseline = features_history[-10:-1]
    current = features_history[-1]
    
    # Z-score of current features
    mean = np.mean(baseline, axis=0)
    std = np.std(baseline, axis=0) + 1e-10
    z_scores = np.abs((current - mean) / std)
    
    # Regime change if any feature exceeds threshold
    return bool(np.any(z_scores > threshold))
