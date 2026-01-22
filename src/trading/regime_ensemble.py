"""
Ensemble Regime Detection

Combines multiple regime detection methods with weighted consensus:
- HMM (Hidden Markov Model): 50% weight
- GMM (Gaussian Mixture Model): 30% weight  
- Agglomerative Clustering: 20% weight

Requires 2/3 model agreement for regime change confirmation.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

try:
    from sklearn.mixture import GaussianMixture
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

logger = logging.getLogger(__name__)


class RegimeType(Enum):
    """Market regime types."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOL = "high_volatility"
    LOW_VOL = "low_volatility"
    CRISIS = "crisis"
    RECOVERY = "recovery"
    UNKNOWN = "unknown"


@dataclass
class RegimeState:
    """Current regime state with confidence."""
    regime: RegimeType
    confidence: float
    hmm_regime: Optional[str] = None
    gmm_regime: Optional[str] = None
    cluster_regime: Optional[str] = None
    consensus_count: int = 0
    transition_probability: float = 0.0


# =============================================================================
# HIDDEN MARKOV MODEL REGIME DETECTOR
# =============================================================================

class HMMRegimeDetector:
    """
    Hidden Markov Model for regime detection.
    
    Uses Gaussian emissions with covariance type 'diag' for efficiency.
    Default 3 states: bull, bear, sideways.
    """
    
    def __init__(self, n_regimes: int = 3, n_features: int = 5,
                 n_iter: int = 100, random_state: int = 42):
        """
        Args:
            n_regimes: Number of hidden states
            n_features: Number of input features
            n_iter: Max EM iterations
            random_state: Random seed
        """
        self.n_regimes = n_regimes
        self.n_features = n_features
        self.n_iter = n_iter
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.is_fitted = False
        
        # Regime labels (ordered by expected return)
        self.regime_labels = ['bear', 'sideways', 'bull'][:n_regimes]
    
    def _build_model(self):
        """Build HMM model."""
        if not HMM_AVAILABLE:
            logger.warning("hmmlearn not available - HMM disabled")
            return None
        
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type='diag',
            n_iter=self.n_iter,
            random_state=self.random_state
        )
        return self.model
    
    def fit(self, features: np.ndarray) -> 'HMMRegimeDetector':
        """
        Fit HMM to feature matrix.
        
        Args:
            features: (n_samples, n_features) feature matrix
        """
        if features.shape[0] < 50:
            logger.warning("Insufficient data for HMM fitting")
            return self
        
        if self.model is None:
            self._build_model()
        
        if self.model is None:
            return self
        
        try:
            # Standardize features
            features_scaled = self.scaler.fit_transform(features)
            
            # Fit HMM
            self.model.fit(features_scaled)
            
            # Order states by mean return (assumed first feature)
            means = self.model.means_[:, 0]
            state_order = np.argsort(means)
            
            # Reorder regime labels
            self.regime_labels = [self.regime_labels[min(i, len(self.regime_labels)-1)] 
                                  for i in state_order]
            
            self.is_fitted = True
            logger.info(f"HMM fitted with {self.n_regimes} regimes")
            
        except Exception as e:
            logger.warning(f"HMM fitting failed: {e}")
        
        return self
    
    def predict(self, features: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        Predict regime for features.
        
        Args:
            features: (n_samples, n_features) or (n_features,) feature array
        
        Returns:
            regime: Predicted regime label
            confidence: Prediction confidence
            probs: State probabilities
        """
        if not self.is_fitted or self.model is None:
            return 'unknown', 0.0, np.ones(self.n_regimes) / self.n_regimes
        
        try:
            features = np.atleast_2d(features)
            features_scaled = self.scaler.transform(features)
            
            # Get most likely state
            state = self.model.predict(features_scaled)[-1]
            
            # Get state probabilities
            probs = self.model.predict_proba(features_scaled)[-1]
            confidence = float(probs[state])
            
            regime = self.regime_labels[state]
            
            return regime, confidence, probs
            
        except Exception as e:
            logger.warning(f"HMM prediction failed: {e}")
            return 'unknown', 0.0, np.ones(self.n_regimes) / self.n_regimes
    
    def get_transition_matrix(self) -> np.ndarray:
        """Get transition probability matrix."""
        if self.model is None:
            return np.eye(self.n_regimes)
        return self.model.transmat_


# =============================================================================
# GAUSSIAN MIXTURE MODEL REGIME DETECTOR
# =============================================================================

class GMMRegimeDetector:
    """
    Gaussian Mixture Model for regime detection.
    
    Clusters feature space into regime components.
    """
    
    def __init__(self, n_regimes: int = 3, n_features: int = 5,
                 covariance_type: str = 'full', random_state: int = 42):
        """
        Args:
            n_regimes: Number of mixture components
            n_features: Number of input features
            covariance_type: 'full', 'tied', 'diag', 'spherical'
            random_state: Random seed
        """
        self.n_regimes = n_regimes
        self.n_features = n_features
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.is_fitted = False
        
        self.regime_labels = ['low_vol', 'normal', 'high_vol'][:n_regimes]
    
    def _build_model(self):
        """Build GMM model."""
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn not available - GMM disabled")
            return None
        
        self.model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            max_iter=200,
            n_init=3
        )
        return self.model
    
    def fit(self, features: np.ndarray) -> 'GMMRegimeDetector':
        """Fit GMM to feature matrix."""
        if features.shape[0] < 50:
            logger.warning("Insufficient data for GMM fitting")
            return self
        
        if self.model is None:
            self._build_model()
        
        if self.model is None:
            return self
        
        try:
            features_scaled = self.scaler.fit_transform(features)
            self.model.fit(features_scaled)
            
            # Order by volatility (assumed second feature)
            if features.shape[1] >= 2:
                vol_order = np.argsort(self.model.means_[:, 1])
                self.regime_labels = [self.regime_labels[min(i, len(self.regime_labels)-1)] 
                                     for i in vol_order]
            
            self.is_fitted = True
            logger.info(f"GMM fitted with {self.n_regimes} components")
            
        except Exception as e:
            logger.warning(f"GMM fitting failed: {e}")
        
        return self
    
    def predict(self, features: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """Predict regime for features."""
        if not self.is_fitted or self.model is None:
            return 'unknown', 0.0, np.ones(self.n_regimes) / self.n_regimes
        
        try:
            features = np.atleast_2d(features)
            features_scaled = self.scaler.transform(features)
            
            probs = self.model.predict_proba(features_scaled)[-1]
            state = np.argmax(probs)
            confidence = float(probs[state])
            
            regime = self.regime_labels[state]
            
            return regime, confidence, probs
            
        except Exception as e:
            logger.warning(f"GMM prediction failed: {e}")
            return 'unknown', 0.0, np.ones(self.n_regimes) / self.n_regimes


# =============================================================================
# AGGLOMERATIVE CLUSTERING REGIME DETECTOR
# =============================================================================

class ClusterRegimeDetector:
    """
    Agglomerative Clustering for regime detection.
    
    Uses hierarchical clustering with Ward linkage.
    """
    
    def __init__(self, n_regimes: int = 3, n_features: int = 5):
        """
        Args:
            n_regimes: Number of clusters
            n_features: Number of input features
        """
        self.n_regimes = n_regimes
        self.n_features = n_features
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.is_fitted = False
        
        # Cluster centers for prediction
        self.cluster_centers = None
        self.regime_labels = ['regime_0', 'regime_1', 'regime_2'][:n_regimes]
    
    def _build_model(self):
        """Build clustering model."""
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn not available - Clustering disabled")
            return None
        
        self.model = AgglomerativeClustering(
            n_clusters=self.n_regimes,
            linkage='ward'
        )
        return self.model
    
    def fit(self, features: np.ndarray) -> 'ClusterRegimeDetector':
        """Fit clustering to feature matrix."""
        if features.shape[0] < 50:
            logger.warning("Insufficient data for clustering")
            return self
        
        if self.model is None:
            self._build_model()
        
        if self.model is None:
            return self
        
        try:
            features_scaled = self.scaler.fit_transform(features)
            
            # Fit and get labels
            labels = self.model.fit_predict(features_scaled)
            
            # Compute cluster centers
            self.cluster_centers = np.array([
                features_scaled[labels == i].mean(axis=0) 
                for i in range(self.n_regimes)
            ])
            
            # Order by first feature (expected return)
            ret_order = np.argsort(self.cluster_centers[:, 0])
            self.regime_labels = ['bear', 'sideways', 'bull']
            self.regime_labels = [self.regime_labels[min(i, 2)] for i in ret_order]
            
            # Reorder cluster centers
            self.cluster_centers = self.cluster_centers[ret_order]
            
            self.is_fitted = True
            logger.info(f"Clustering fitted with {self.n_regimes} clusters")
            
        except Exception as e:
            logger.warning(f"Clustering failed: {e}")
        
        return self
    
    def predict(self, features: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """Predict regime by nearest cluster center."""
        if not self.is_fitted or self.cluster_centers is None:
            return 'unknown', 0.0, np.ones(self.n_regimes) / self.n_regimes
        
        try:
            features = np.atleast_2d(features)
            features_scaled = self.scaler.transform(features)[-1]
            
            # Distance to each cluster center
            distances = np.linalg.norm(self.cluster_centers - features_scaled, axis=1)
            
            # Softmax for probabilities
            neg_distances = -distances
            probs = np.exp(neg_distances - neg_distances.max())
            probs = probs / probs.sum()
            
            state = np.argmin(distances)
            confidence = float(probs[state])
            regime = self.regime_labels[state]
            
            return regime, confidence, probs
            
        except Exception as e:
            logger.warning(f"Cluster prediction failed: {e}")
            return 'unknown', 0.0, np.ones(self.n_regimes) / self.n_regimes


# =============================================================================
# ENSEMBLE REGIME DETECTOR
# =============================================================================

class EnsembleRegimeDetector:
    """
    Ensemble regime detection with weighted consensus.
    
    Combines HMM, GMM, and Agglomerative Clustering:
    - HMM: 50% weight (captures temporal dynamics)
    - GMM: 30% weight (flexible distributions)
    - Clustering: 20% weight (robust to outliers)
    
    Requires 2/3 agreement for regime change confirmation.
    """
    
    def __init__(self, n_regimes: int = 3, n_features: int = 5,
                 hmm_weight: float = 0.5,
                 gmm_weight: float = 0.3,
                 cluster_weight: float = 0.2,
                 consensus_threshold: int = 2):
        """
        Args:
            n_regimes: Number of regimes
            n_features: Number of input features
            hmm_weight: Weight for HMM prediction
            gmm_weight: Weight for GMM prediction
            cluster_weight: Weight for clustering prediction
            consensus_threshold: Minimum models agreeing for regime change
        """
        self.n_regimes = n_regimes
        self.n_features = n_features
        
        # Weights (normalized)
        total = hmm_weight + gmm_weight + cluster_weight
        self.hmm_weight = hmm_weight / total
        self.gmm_weight = gmm_weight / total
        self.cluster_weight = cluster_weight / total
        
        self.consensus_threshold = consensus_threshold
        
        # Individual detectors
        self.hmm = HMMRegimeDetector(n_regimes, n_features)
        self.gmm = GMMRegimeDetector(n_regimes, n_features)
        self.cluster = ClusterRegimeDetector(n_regimes, n_features)
        
        # State tracking
        self.current_regime = RegimeType.UNKNOWN
        self.regime_history = []
        self.is_fitted = False
        
        # Regime mapping (standardize labels)
        self.regime_map = {
            'bull': RegimeType.BULL,
            'bear': RegimeType.BEAR,
            'sideways': RegimeType.SIDEWAYS,
            'high_vol': RegimeType.HIGH_VOL,
            'low_vol': RegimeType.LOW_VOL,
            'normal': RegimeType.SIDEWAYS,
            'regime_0': RegimeType.BEAR,
            'regime_1': RegimeType.SIDEWAYS,
            'regime_2': RegimeType.BULL,
            'unknown': RegimeType.UNKNOWN
        }
    
    def compute_features(self, returns: np.ndarray, 
                         volatility: np.ndarray = None,
                         volume: np.ndarray = None) -> np.ndarray:
        """
        Compute regime detection features from market data.
        
        Default features:
        1. Rolling return (20-day)
        2. Rolling volatility (20-day)
        3. Return momentum (5-day vs 20-day)
        4. Volatility momentum
        5. Volume trend (if available)
        
        Args:
            returns: Daily return series
            volatility: Daily volatility (optional)
            volume: Daily volume (optional)
        
        Returns:
            (n_samples, n_features) feature matrix
        """
        n = len(returns)
        features = np.zeros((n, self.n_features))
        
        if n < 20:
            return features
        
        # Rolling return (20-day)
        for i in range(20, n):
            features[i, 0] = np.mean(returns[i-20:i])
        
        # Rolling volatility
        if volatility is not None:
            for i in range(20, n):
                features[i, 1] = np.mean(volatility[i-20:i])
        else:
            for i in range(20, n):
                features[i, 1] = np.std(returns[i-20:i])
        
        # Return momentum (5d vs 20d)
        for i in range(20, n):
            ret_5d = np.mean(returns[max(0, i-5):i])
            ret_20d = np.mean(returns[i-20:i])
            features[i, 2] = ret_5d - ret_20d
        
        # Volatility momentum
        for i in range(40, n):
            vol_20d = features[i, 1]
            vol_20d_prev = features[i-20, 1] if i >= 40 else vol_20d
            features[i, 3] = vol_20d - vol_20d_prev
        
        # Volume trend (if available)
        if volume is not None:
            for i in range(20, n):
                vol_sma = np.mean(volume[max(0, i-20):i])
                features[i, 4] = (volume[i] / vol_sma - 1) if vol_sma > 0 else 0
        
        return features
    
    def fit(self, returns: np.ndarray, 
            volatility: np.ndarray = None,
            volume: np.ndarray = None) -> 'EnsembleRegimeDetector':
        """
        Fit all ensemble models to historical data.
        
        Args:
            returns: Daily return series
            volatility: Daily volatility (optional)
            volume: Daily volume (optional)
        """
        features = self.compute_features(returns, volatility, volume)
        
        # Remove warmup period
        valid_features = features[40:]
        
        if len(valid_features) < 50:
            logger.warning("Insufficient data for ensemble fitting")
            return self
        
        # Fit each model
        self.hmm.fit(valid_features)
        self.gmm.fit(valid_features)
        self.cluster.fit(valid_features)
        
        self.is_fitted = True
        logger.info("Ensemble regime detector fitted")
        
        return self
    
    def predict(self, features: np.ndarray) -> RegimeState:
        """
        Predict regime with ensemble consensus.
        
        Args:
            features: Current feature vector
        
        Returns:
            RegimeState with regime, confidence, and component predictions
        """
        if not self.is_fitted:
            return RegimeState(
                regime=RegimeType.UNKNOWN,
                confidence=0.0,
                consensus_count=0
            )
        
        # Get predictions from each model
        hmm_regime, hmm_conf, hmm_probs = self.hmm.predict(features)
        gmm_regime, gmm_conf, gmm_probs = self.gmm.predict(features)
        cluster_regime, cluster_conf, cluster_probs = self.cluster.predict(features)
        
        # Map to RegimeType
        hmm_type = self.regime_map.get(hmm_regime, RegimeType.UNKNOWN)
        gmm_type = self.regime_map.get(gmm_regime, RegimeType.UNKNOWN)
        cluster_type = self.regime_map.get(cluster_regime, RegimeType.UNKNOWN)
        
        # Weighted vote
        regime_votes = {}
        for regime, weight, conf in [
            (hmm_type, self.hmm_weight, hmm_conf),
            (gmm_type, self.gmm_weight, gmm_conf),
            (cluster_type, self.cluster_weight, cluster_conf)
        ]:
            if regime not in regime_votes:
                regime_votes[regime] = 0.0
            regime_votes[regime] += weight * conf
        
        # Winner
        final_regime = max(regime_votes.keys(), key=lambda k: regime_votes[k])
        final_confidence = regime_votes[final_regime]
        
        # Consensus count
        predictions = [hmm_type, gmm_type, cluster_type]
        consensus_count = sum(1 for p in predictions if p == final_regime)
        
        # Transition probability from HMM
        transition_prob = 0.0
        if self.current_regime != RegimeType.UNKNOWN and hmm_probs is not None:
            # Probability of staying in current regime
            transition_prob = 1.0 - hmm_probs[np.argmax(hmm_probs)]
        
        # Only confirm regime change if consensus met
        if consensus_count >= self.consensus_threshold:
            if final_regime != self.current_regime:
                logger.info(f"Regime change: {self.current_regime.value} -> {final_regime.value} "
                           f"(consensus: {consensus_count}/3, confidence: {final_confidence:.2f})")
            self.current_regime = final_regime
        else:
            # Keep current regime
            final_regime = self.current_regime
            final_confidence = min(final_confidence, 0.5)  # Lower confidence
        
        state = RegimeState(
            regime=final_regime,
            confidence=final_confidence,
            hmm_regime=hmm_regime,
            gmm_regime=gmm_regime,
            cluster_regime=cluster_regime,
            consensus_count=consensus_count,
            transition_probability=transition_prob
        )
        
        self.regime_history.append(state)
        
        return state
    
    def get_current_regime(self) -> RegimeType:
        """Get current detected regime."""
        return self.current_regime
    
    def get_regime_string(self) -> str:
        """Get current regime as string for compatibility."""
        regime_to_string = {
            RegimeType.BULL: 'risk_on',
            RegimeType.BEAR: 'risk_off',
            RegimeType.SIDEWAYS: 'neutral',
            RegimeType.HIGH_VOL: 'risk_off',
            RegimeType.LOW_VOL: 'risk_on',
            RegimeType.CRISIS: 'risk_off',
            RegimeType.RECOVERY: 'neutral',
            RegimeType.UNKNOWN: 'neutral'
        }
        return regime_to_string.get(self.current_regime, 'neutral')
    
    def get_position_multiplier(self) -> float:
        """
        Get position sizing multiplier based on regime.
        
        Returns:
            Multiplier between 0.25 and 1.5
        """
        multipliers = {
            RegimeType.BULL: 1.25,
            RegimeType.BEAR: 0.5,
            RegimeType.SIDEWAYS: 1.0,
            RegimeType.HIGH_VOL: 0.5,
            RegimeType.LOW_VOL: 1.25,
            RegimeType.CRISIS: 0.25,
            RegimeType.RECOVERY: 0.75,
            RegimeType.UNKNOWN: 0.75
        }
        return multipliers.get(self.current_regime, 1.0)
    
    def get_stats(self) -> Dict:
        """Get ensemble statistics."""
        regime_counts = {}
        for state in self.regime_history:
            regime = state.regime.value
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        return {
            'current_regime': self.current_regime.value,
            'is_fitted': self.is_fitted,
            'history_length': len(self.regime_history),
            'regime_distribution': regime_counts,
            'hmm_weight': self.hmm_weight,
            'gmm_weight': self.gmm_weight,
            'cluster_weight': self.cluster_weight
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_regime_detector(n_regimes: int = 3,
                           use_hmm: bool = True,
                           use_gmm: bool = True,
                           use_clustering: bool = True) -> EnsembleRegimeDetector:
    """
    Create ensemble regime detector with specified components.
    
    Args:
        n_regimes: Number of regimes to detect
        use_hmm: Include HMM
        use_gmm: Include GMM
        use_clustering: Include Agglomerative Clustering
    
    Returns:
        Configured EnsembleRegimeDetector
    """
    # Adjust weights based on enabled models
    hmm_w = 0.5 if use_hmm else 0.0
    gmm_w = 0.3 if use_gmm else 0.0
    cluster_w = 0.2 if use_clustering else 0.0
    
    return EnsembleRegimeDetector(
        n_regimes=n_regimes,
        hmm_weight=hmm_w,
        gmm_weight=gmm_w,
        cluster_weight=cluster_w
    )


def detect_regime_from_returns(returns: np.ndarray,
                                window: int = 60) -> Tuple[str, float]:
    """
    Quick regime detection from return series.
    
    Simple heuristic without full model fitting.
    
    Args:
        returns: Return series
        window: Lookback window
    
    Returns:
        regime: 'risk_on', 'risk_off', or 'neutral'
        confidence: Confidence score
    """
    if len(returns) < window:
        return 'neutral', 0.5
    
    recent_returns = returns[-window:]
    
    # Compute metrics
    mean_ret = np.mean(recent_returns)
    vol = np.std(recent_returns)
    sharpe = mean_ret / (vol + 1e-10) * np.sqrt(252)
    
    # Recent momentum
    momentum = np.mean(returns[-5:]) - np.mean(returns[-20:])
    
    # Classify
    if sharpe > 1.0 and momentum > 0:
        return 'risk_on', min(0.9, 0.5 + sharpe * 0.2)
    elif sharpe < -0.5 or (vol > 0.02 and momentum < 0):
        return 'risk_off', min(0.9, 0.5 + abs(sharpe) * 0.2)
    else:
        return 'neutral', 0.6
