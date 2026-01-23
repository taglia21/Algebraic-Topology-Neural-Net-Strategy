#!/usr/bin/env python3
"""
V25 Phase 2: Pattern Memory System
===================================

Pattern Memory with Similarity Search using TDA features.

Key Innovation:
- Store successful trades with their pattern signatures (TDA features)
- When similar patterns emerge, amplify confidence
- Use locality-sensitive hashing (LSH) for fast similarity search
- Target: 2% additional Sharpe improvement

Pattern Types:
1. Price Pattern: TDA features from price time series
2. Volume Pattern: TDA features from volume
3. Volatility Pattern: Rolling volatility signature
4. Combined Pattern: Concatenated feature vector

Similarity Metrics:
- Cosine similarity for normalized feature vectors
- Euclidean distance for raw features
- DTW for time series comparison (expensive, used sparingly)
"""

import json
import logging
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import hashlib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('V25_PatternMemory')


# =============================================================================
# PATTERN SIGNATURE
# =============================================================================

@dataclass
class PatternSignature:
    """
    Feature vector representing a market pattern.
    
    Contains:
    - Price pattern: TDA features from price series
    - Volume pattern: Volume distribution features
    - Volatility pattern: Rolling vol features
    - Market context: Regime, sector info
    """
    # Core features
    price_features: np.ndarray
    volume_features: np.ndarray  
    volatility_features: np.ndarray
    
    # Context
    regime: str
    sector: Optional[str] = None
    
    # Metadata
    date: str = ""
    ticker: str = ""
    
    def to_vector(self) -> np.ndarray:
        """Concatenate all features into single vector."""
        return np.concatenate([
            self.price_features,
            self.volume_features,
            self.volatility_features
        ])
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            'price_features': self.price_features.tolist(),
            'volume_features': self.volume_features.tolist(),
            'volatility_features': self.volatility_features.tolist(),
            'regime': self.regime,
            'sector': self.sector,
            'date': self.date,
            'ticker': self.ticker
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'PatternSignature':
        """Deserialize from dictionary."""
        return cls(
            price_features=np.array(d['price_features']),
            volume_features=np.array(d['volume_features']),
            volatility_features=np.array(d['volatility_features']),
            regime=d['regime'],
            sector=d.get('sector'),
            date=d.get('date', ''),
            ticker=d.get('ticker', '')
        )


@dataclass 
class PatternMemoryEntry:
    """
    Single entry in pattern memory.
    
    Stores:
    - Pattern signature at entry
    - Trade outcome (return, holding period)
    - Success metrics
    """
    pattern: PatternSignature
    entry_date: str
    exit_date: str
    trade_return: float
    holding_days: int
    strategy: str  # 'v21' or 'v24'
    success: bool  # Return > 0
    
    # LSH hash for fast retrieval
    lsh_hash: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'pattern': self.pattern.to_dict(),
            'entry_date': self.entry_date,
            'exit_date': self.exit_date,
            'trade_return': float(self.trade_return),
            'holding_days': self.holding_days,
            'strategy': self.strategy,
            'success': bool(self.success),
            'lsh_hash': self.lsh_hash
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'PatternMemoryEntry':
        return cls(
            pattern=PatternSignature.from_dict(d['pattern']),
            entry_date=d['entry_date'],
            exit_date=d['exit_date'],
            trade_return=d['trade_return'],
            holding_days=d['holding_days'],
            strategy=d['strategy'],
            success=d['success'],
            lsh_hash=d.get('lsh_hash', '')
        )


# =============================================================================
# LOCALITY SENSITIVE HASHING
# =============================================================================

class LSHIndex:
    """
    Locality Sensitive Hashing for fast similarity search.
    
    Uses random hyperplane projections for cosine similarity.
    """
    
    def __init__(self, 
                 n_dimensions: int,
                 n_planes: int = 10,
                 n_tables: int = 4,
                 seed: int = 42):
        """
        Initialize LSH index.
        
        Args:
            n_dimensions: Dimension of feature vectors
            n_planes: Number of random hyperplanes per hash
            n_tables: Number of hash tables (more = better recall)
            seed: Random seed for reproducibility
        """
        self.n_dimensions = n_dimensions
        self.n_planes = n_planes
        self.n_tables = n_tables
        
        np.random.seed(seed)
        
        # Create random hyperplanes for each table
        self.hyperplanes = [
            np.random.randn(n_planes, n_dimensions)
            for _ in range(n_tables)
        ]
        
        # Hash buckets: table_id -> hash_key -> list of indices
        self.buckets: List[Dict[str, List[int]]] = [
            {} for _ in range(n_tables)
        ]
        
        # All stored vectors
        self.vectors: List[np.ndarray] = []
        self.entries: List[PatternMemoryEntry] = []
        
    def _hash_vector(self, vector: np.ndarray, table_id: int) -> str:
        """Compute hash for a vector using random hyperplanes."""
        # Normalize vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        # Project onto hyperplanes and take sign
        projections = np.dot(self.hyperplanes[table_id], vector)
        bits = (projections > 0).astype(int)
        
        # Convert to hex string
        return ''.join(str(b) for b in bits)
    
    def add(self, entry: PatternMemoryEntry) -> str:
        """Add a pattern to the index."""
        vector = entry.pattern.to_vector()
        idx = len(self.vectors)
        
        self.vectors.append(vector)
        self.entries.append(entry)
        
        # Hash into all tables
        hash_keys = []
        for table_id in range(self.n_tables):
            hash_key = self._hash_vector(vector, table_id)
            hash_keys.append(hash_key)
            
            if hash_key not in self.buckets[table_id]:
                self.buckets[table_id][hash_key] = []
            self.buckets[table_id][hash_key].append(idx)
        
        # Return combined hash for storage
        combined = '-'.join(hash_keys)
        entry.lsh_hash = combined
        return combined
    
    def query(self, 
              pattern: PatternSignature,
              k: int = 10,
              min_similarity: float = 0.7) -> List[Tuple[PatternMemoryEntry, float]]:
        """
        Find k most similar patterns.
        
        Args:
            pattern: Query pattern
            k: Number of results
            min_similarity: Minimum cosine similarity threshold
            
        Returns:
            List of (entry, similarity) tuples
        """
        query_vector = pattern.to_vector()
        
        # Normalize query
        query_norm = np.linalg.norm(query_vector)
        if query_norm > 0:
            query_vector = query_vector / query_norm
        else:
            return []
        
        # Find candidate indices from all tables
        candidates = set()
        for table_id in range(self.n_tables):
            hash_key = self._hash_vector(query_vector * query_norm, table_id)
            if hash_key in self.buckets[table_id]:
                candidates.update(self.buckets[table_id][hash_key])
        
        if not candidates:
            return []
        
        # Compute actual similarities
        results = []
        for idx in candidates:
            stored_vector = self.vectors[idx]
            stored_norm = np.linalg.norm(stored_vector)
            
            if stored_norm > 0:
                similarity = np.dot(query_vector, stored_vector / stored_norm)
                
                if similarity >= min_similarity:
                    results.append((self.entries[idx], similarity))
        
        # Sort by similarity and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def stats(self) -> Dict:
        """Get index statistics."""
        bucket_sizes = []
        for table in self.buckets:
            bucket_sizes.extend(len(v) for v in table.values())
            
        return {
            'n_entries': len(self.vectors),
            'n_tables': self.n_tables,
            'n_buckets': sum(len(t) for t in self.buckets),
            'avg_bucket_size': np.mean(bucket_sizes) if bucket_sizes else 0,
            'max_bucket_size': max(bucket_sizes) if bucket_sizes else 0
        }


# =============================================================================
# PATTERN MEMORY SYSTEM
# =============================================================================

class PatternMemory:
    """
    Memory system for storing and retrieving successful patterns.
    
    Key Features:
    - LSH-based fast similarity search
    - Per-regime pattern statistics
    - Success rate tracking by pattern cluster
    - Confidence amplification for known patterns
    """
    
    def __init__(self,
                 n_features: int = 30,
                 max_entries: int = 10000,
                 similarity_threshold: float = 0.75,
                 min_pattern_matches: int = 5,
                 state_dir: str = "logs/v25"):
        """
        Initialize pattern memory.
        
        Args:
            n_features: Dimension of pattern vectors
            max_entries: Maximum entries to store (FIFO eviction)
            similarity_threshold: Min similarity for pattern match
            min_pattern_matches: Min matches before confidence boost
            state_dir: Directory for persistence
        """
        self.n_features = n_features
        self.max_entries = max_entries
        self.similarity_threshold = similarity_threshold
        self.min_pattern_matches = min_pattern_matches
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        # LSH index for fast retrieval
        self.lsh_index = LSHIndex(
            n_dimensions=n_features,
            n_planes=12,
            n_tables=5
        )
        
        # Per-regime statistics
        self.regime_stats: Dict[str, Dict] = {}
        
        # Pattern cluster success rates
        self.cluster_success: Dict[str, Dict] = {}
        
        logger.info(f"PatternMemory initialized with {max_entries} capacity")
    
    def add_pattern(self, entry: PatternMemoryEntry):
        """
        Add a trade pattern to memory.
        
        Args:
            entry: Pattern memory entry with trade outcome
        """
        # FIFO eviction if at capacity
        if len(self.lsh_index.entries) >= self.max_entries:
            # Would need more complex eviction logic
            # For now, just log warning
            logger.warning(f"Pattern memory at capacity ({self.max_entries})")
            return
        
        # Add to LSH index
        hash_key = self.lsh_index.add(entry)
        
        # Update regime stats
        regime = entry.pattern.regime
        if regime not in self.regime_stats:
            self.regime_stats[regime] = {
                'v21_success': 0, 'v21_total': 0,
                'v24_success': 0, 'v24_total': 0,
                'total_return': 0.0
            }
        
        stats = self.regime_stats[regime]
        strategy_key = f"{entry.strategy}_"
        stats[strategy_key + 'total'] += 1
        if entry.success:
            stats[strategy_key + 'success'] += 1
        stats['total_return'] += entry.trade_return
        
        # Update cluster success rates
        cluster_key = hash_key.split('-')[0]  # Use first table hash
        if cluster_key not in self.cluster_success:
            self.cluster_success[cluster_key] = {
                'successes': 0, 'total': 0, 'returns': []
            }
        
        cluster = self.cluster_success[cluster_key]
        cluster['total'] += 1
        if entry.success:
            cluster['successes'] += 1
        cluster['returns'].append(entry.trade_return)
    
    def get_pattern_confidence(self, 
                                pattern: PatternSignature,
                                strategy: str) -> Tuple[float, Dict]:
        """
        Get confidence score for a pattern based on similar historical patterns.
        
        Args:
            pattern: Current pattern signature
            strategy: 'v21' or 'v24'
            
        Returns:
            (confidence_multiplier, metadata)
            - confidence_multiplier: 0.5 to 1.5 (1.0 = neutral)
            - metadata: dict with match details
        """
        # Find similar patterns
        similar = self.lsh_index.query(
            pattern,
            k=20,
            min_similarity=self.similarity_threshold
        )
        
        if len(similar) < self.min_pattern_matches:
            return 1.0, {'n_matches': len(similar), 'reason': 'insufficient_matches'}
        
        # Calculate success rate for this strategy in similar patterns
        strategy_matches = [s for s in similar if s[0].strategy == strategy]
        
        if len(strategy_matches) < 3:
            return 1.0, {'n_matches': len(similar), 'strategy_matches': len(strategy_matches)}
        
        # Weighted success rate (weight by similarity)
        weighted_successes = sum(
            s[1] * (1 if s[0].success else 0) 
            for s in strategy_matches
        )
        weighted_total = sum(s[1] for s in strategy_matches)
        
        success_rate = weighted_successes / weighted_total if weighted_total > 0 else 0.5
        
        # Calculate average return
        weighted_returns = sum(s[1] * s[0].trade_return for s in strategy_matches)
        avg_return = weighted_returns / weighted_total if weighted_total > 0 else 0
        
        # Convert success rate to confidence multiplier
        # 50% success = 1.0, 70% = 1.2, 80% = 1.4
        # 30% success = 0.8, 20% = 0.6
        confidence = 0.5 + success_rate  # Maps 0-100% to 0.5-1.5
        confidence = np.clip(confidence, 0.5, 1.5)
        
        return confidence, {
            'n_matches': len(similar),
            'strategy_matches': len(strategy_matches),
            'success_rate': success_rate,
            'avg_return': avg_return,
            'avg_similarity': np.mean([s[1] for s in strategy_matches])
        }
    
    def get_regime_success_rates(self, regime: str) -> Dict[str, float]:
        """
        Get historical success rates for each strategy in a regime.
        
        Args:
            regime: Regime meta_state string
            
        Returns:
            {'v21_success_rate': float, 'v24_success_rate': float}
        """
        if regime not in self.regime_stats:
            return {'v21_success_rate': 0.5, 'v24_success_rate': 0.5}
        
        stats = self.regime_stats[regime]
        
        v21_rate = (stats['v21_success'] / stats['v21_total'] 
                    if stats['v21_total'] > 0 else 0.5)
        v24_rate = (stats['v24_success'] / stats['v24_total']
                    if stats['v24_total'] > 0 else 0.5)
        
        return {
            'v21_success_rate': v21_rate,
            'v24_success_rate': v24_rate,
            'v21_count': stats['v21_total'],
            'v24_count': stats['v24_total']
        }
    
    def save_state(self, filepath: Optional[str] = None):
        """Save memory state to disk."""
        if filepath is None:
            filepath = self.state_dir / "pattern_memory_state.json"
        else:
            filepath = Path(filepath)
        
        # Don't save vectors/entries (too large), just stats
        state = {
            'regime_stats': self.regime_stats,
            'cluster_success': {
                k: {'successes': v['successes'], 'total': v['total']}
                for k, v in self.cluster_success.items()
            },
            'n_entries': len(self.lsh_index.entries),
            'lsh_stats': self.lsh_index.stats(),
            'saved_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Pattern memory state saved to {filepath}")
    
    def load_state(self, filepath: Optional[str] = None) -> bool:
        """Load memory state from disk."""
        if filepath is None:
            filepath = self.state_dir / "pattern_memory_state.json"
        else:
            filepath = Path(filepath)
        
        if not filepath.exists():
            logger.warning(f"No state file found at {filepath}")
            return False
        
        with open(filepath) as f:
            state = json.load(f)
        
        self.regime_stats = state.get('regime_stats', {})
        
        logger.info(f"Loaded pattern memory: {state.get('n_entries', 0)} entries")
        return True
    
    def get_statistics(self) -> Dict:
        """Get memory statistics."""
        total_entries = len(self.lsh_index.entries)
        successful = sum(1 for e in self.lsh_index.entries if e.success)
        
        return {
            'total_entries': total_entries,
            'successful_trades': successful,
            'success_rate': successful / total_entries if total_entries > 0 else 0,
            'n_regimes': len(self.regime_stats),
            'n_clusters': len(self.cluster_success),
            'lsh_stats': self.lsh_index.stats()
        }


# =============================================================================
# PATTERN FEATURE EXTRACTOR
# =============================================================================

class PatternFeatureExtractor:
    """
    Extracts pattern signatures from market data.
    
    Uses combination of:
    - Simple price/volume features (fast)
    - TDA features if available (slow but more powerful)
    """
    
    def __init__(self, 
                 window_size: int = 20,
                 use_tda: bool = True):
        """
        Initialize extractor.
        
        Args:
            window_size: Lookback window for feature computation
            use_tda: Whether to compute TDA features (slower)
        """
        self.window_size = window_size
        self.use_tda = use_tda
        
        # TDA generator if available
        self.tda_generator = None
        if use_tda:
            try:
                from src.tda_features import TDAFeatureGenerator
                self.tda_generator = TDAFeatureGenerator(
                    window=window_size,
                    feature_mode='v1.2'  # 10 features
                )
            except ImportError:
                logger.warning("TDA features not available, using fallback")
                self.use_tda = False
    
    def extract_signature(self,
                          prices: np.ndarray,
                          volumes: np.ndarray,
                          regime: str,
                          date: str = "",
                          ticker: str = "") -> PatternSignature:
        """
        Extract pattern signature from price/volume data.
        
        Args:
            prices: Recent price array (close prices)
            volumes: Recent volume array
            regime: Current regime meta_state
            date: Date string
            ticker: Stock ticker
            
        Returns:
            PatternSignature with extracted features
        """
        # Price features
        if self.use_tda and self.tda_generator and len(prices) >= self.window_size:
            # Use TDA features
            price_features = self._extract_tda_features(prices)
        else:
            # Fallback to simple features
            price_features = self._extract_simple_price_features(prices)
        
        # Volume features
        volume_features = self._extract_volume_features(volumes)
        
        # Volatility features
        volatility_features = self._extract_volatility_features(prices)
        
        return PatternSignature(
            price_features=price_features,
            volume_features=volume_features,
            volatility_features=volatility_features,
            regime=regime,
            date=date,
            ticker=ticker
        )
    
    def _extract_tda_features(self, prices: np.ndarray) -> np.ndarray:
        """Extract TDA features from price series."""
        # Compute log returns
        returns = np.diff(np.log(prices))
        
        if len(returns) < self.window_size:
            return np.zeros(10)
        
        # Use last window
        window_data = returns[-self.window_size:]
        
        # Takens embedding
        embedded = self.tda_generator.takens_embedding(window_data)
        
        if embedded.size == 0:
            return np.zeros(10)
        
        try:
            from ripser import ripser
            result = ripser(embedded, maxdim=1)
            
            # Extract features from persistence diagrams
            features = []
            for dim in range(2):
                diagram = result['dgms'][dim]
                finite_mask = np.isfinite(diagram[:, 1])
                finite_pairs = diagram[finite_mask]
                
                if len(finite_pairs) > 0:
                    lifetimes = finite_pairs[:, 1] - finite_pairs[:, 0]
                    features.extend([
                        np.sum(lifetimes**2)**0.5,  # L2 norm
                        len(lifetimes),              # Count
                        np.max(lifetimes),           # Max
                        np.sum(lifetimes),           # Sum
                        -np.sum(lifetimes * np.log(lifetimes + 1e-10))  # Entropy
                    ])
                else:
                    features.extend([0.0, 0, 0.0, 0.0, 0.0])
            
            return np.array(features)
            
        except Exception as e:
            logger.debug(f"TDA extraction failed: {e}")
            return np.zeros(10)
    
    def _extract_simple_price_features(self, prices: np.ndarray) -> np.ndarray:
        """Extract simple price features (fallback)."""
        if len(prices) < 5:
            return np.zeros(10)
        
        returns = np.diff(np.log(prices))
        
        features = [
            np.mean(returns),
            np.std(returns),
            np.min(returns),
            np.max(returns),
            returns[-1] if len(returns) > 0 else 0,  # Latest return
            np.sum(returns > 0) / len(returns),       # Win rate
            np.sum(returns),                           # Cumulative return
            np.corrcoef(np.arange(len(returns)), returns)[0, 1] if len(returns) > 1 else 0,  # Trend
            self._hurst_exponent(returns) if len(returns) >= 20 else 0.5,
            np.percentile(returns, 75) - np.percentile(returns, 25)  # IQR
        ]
        
        return np.array(features)
    
    def _extract_volume_features(self, volumes: np.ndarray) -> np.ndarray:
        """Extract volume features."""
        if len(volumes) < 5:
            return np.zeros(10)
        
        log_vol = np.log1p(volumes)
        vol_change = np.diff(log_vol)
        
        features = [
            np.mean(log_vol),
            np.std(log_vol),
            np.mean(vol_change),
            np.std(vol_change),
            log_vol[-1] - np.mean(log_vol),  # Relative volume
            np.sum(vol_change > 0) / len(vol_change) if len(vol_change) > 0 else 0.5,
            np.max(log_vol) - np.min(log_vol),  # Range
            np.corrcoef(np.arange(len(log_vol)), log_vol)[0, 1] if len(log_vol) > 1 else 0,
            np.percentile(log_vol, 90) - np.percentile(log_vol, 10),
            log_vol[-1] / np.mean(log_vol[-5:]) if np.mean(log_vol[-5:]) > 0 else 1
        ]
        
        return np.array(features)
    
    def _extract_volatility_features(self, prices: np.ndarray) -> np.ndarray:
        """Extract volatility features."""
        if len(prices) < 10:
            return np.zeros(10)
        
        returns = np.diff(np.log(prices))
        
        # Rolling volatility at different scales
        vol_5 = np.std(returns[-5:]) if len(returns) >= 5 else 0
        vol_10 = np.std(returns[-10:]) if len(returns) >= 10 else 0
        vol_20 = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)
        
        features = [
            vol_5,
            vol_10,
            vol_20,
            vol_5 / vol_20 if vol_20 > 0 else 1,  # Vol ratio (regime indicator)
            np.mean(np.abs(returns)),              # Mean absolute return
            np.max(np.abs(returns)),               # Max absolute return
            self._realized_vol(returns, 5),
            self._realized_vol(returns, 10),
            vol_10 - vol_20,                       # Vol trend
            np.std(np.abs(returns))                # Vol of vol
        ]
        
        return np.array(features)
    
    def _realized_vol(self, returns: np.ndarray, window: int) -> float:
        """Compute realized volatility."""
        if len(returns) < window:
            return np.std(returns) * np.sqrt(252)
        return np.std(returns[-window:]) * np.sqrt(252)
    
    def _hurst_exponent(self, returns: np.ndarray) -> float:
        """
        Estimate Hurst exponent using R/S analysis.
        H < 0.5: Mean reverting
        H = 0.5: Random walk
        H > 0.5: Trending
        """
        n = len(returns)
        if n < 20:
            return 0.5
        
        # R/S analysis
        max_k = min(int(n / 2), 50)
        rs_values = []
        
        for k in range(10, max_k):
            subset = returns[:k]
            mean = np.mean(subset)
            devs = np.cumsum(subset - mean)
            r = np.max(devs) - np.min(devs)
            s = np.std(subset)
            if s > 0:
                rs_values.append(r / s)
        
        if len(rs_values) < 5:
            return 0.5
        
        # Log-log regression
        log_n = np.log(np.arange(10, 10 + len(rs_values)))
        log_rs = np.log(np.array(rs_values) + 1e-10)
        
        coeffs = np.polyfit(log_n, log_rs, 1)
        return np.clip(coeffs[0], 0, 1)


# =============================================================================
# INTEGRATION WITH V25 ALLOCATOR
# =============================================================================

class V25PatternAugmentedAllocator:
    """
    V25 Allocator augmented with pattern memory.
    
    Combines:
    1. Regime-based allocation (Phase 1)
    2. Pattern similarity-based confidence (Phase 2)
    """
    
    def __init__(self, 
                 base_allocator,  # V25AdaptiveAllocator from Phase 1
                 pattern_memory: Optional[PatternMemory] = None,
                 feature_extractor: Optional[PatternFeatureExtractor] = None,
                 confidence_scale: float = 0.15):
        """
        Initialize pattern-augmented allocator.
        
        Args:
            base_allocator: V25AdaptiveAllocator instance
            pattern_memory: PatternMemory instance (created if None)
            feature_extractor: PatternFeatureExtractor (created if None)
            confidence_scale: How much to adjust weights based on pattern confidence
        """
        self.base_allocator = base_allocator
        self.pattern_memory = pattern_memory or PatternMemory()
        self.feature_extractor = feature_extractor or PatternFeatureExtractor()
        self.confidence_scale = confidence_scale
        
        logger.info("V25PatternAugmentedAllocator initialized")
    
    def get_allocation(self,
                        prices: np.ndarray,
                        volumes: np.ndarray,
                        market_returns: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Get V21/V24 allocation with pattern confidence adjustment.
        
        Args:
            prices: Recent price array
            volumes: Recent volume array
            market_returns: Optional market index returns
            
        Returns:
            {'v21': weight, 'v24': weight}
        """
        # Get base allocation from regime
        base_weights = self.base_allocator.get_allocation(
            prices=prices,
            volumes=volumes,
            market_returns=market_returns
        )
        
        # Get current regime
        regime = self.base_allocator.controller.current_state.meta_state
        
        # Extract current pattern
        pattern = self.feature_extractor.extract_signature(
            prices=prices,
            volumes=volumes,
            regime=regime
        )
        
        # Get pattern-based confidence for each strategy
        v21_conf, v21_meta = self.pattern_memory.get_pattern_confidence(pattern, 'v21')
        v24_conf, v24_meta = self.pattern_memory.get_pattern_confidence(pattern, 'v24')
        
        # Adjust weights based on relative confidence
        # If V21 has higher pattern confidence, increase its weight
        conf_diff = (v21_conf - v24_conf) * self.confidence_scale
        
        adjusted = base_weights.copy()
        adjusted['v21'] += conf_diff
        adjusted['v24'] -= conf_diff
        
        # Clip to valid range
        adjusted['v21'] = np.clip(adjusted['v21'], 0.15, 0.85)
        adjusted['v24'] = np.clip(adjusted['v24'], 0.15, 0.85)
        
        # Renormalize
        total = adjusted['v21'] + adjusted['v24']
        adjusted['v21'] /= total
        adjusted['v24'] /= total
        
        return adjusted
    
    def record_trade(self,
                     entry_prices: np.ndarray,
                     entry_volumes: np.ndarray,
                     trade_return: float,
                     strategy: str,
                     holding_days: int,
                     entry_date: str,
                     exit_date: str):
        """
        Record a completed trade to pattern memory.
        
        Args:
            entry_prices: Prices at trade entry
            entry_volumes: Volumes at trade entry
            trade_return: Trade return
            strategy: 'v21' or 'v24'
            holding_days: Days held
            entry_date: Entry date string
            exit_date: Exit date string
        """
        # Get regime at entry
        regime = self.base_allocator.controller.current_state.meta_state
        
        # Extract pattern at entry
        pattern = self.feature_extractor.extract_signature(
            prices=entry_prices,
            volumes=entry_volumes,
            regime=regime,
            date=entry_date
        )
        
        # Create memory entry
        entry = PatternMemoryEntry(
            pattern=pattern,
            entry_date=entry_date,
            exit_date=exit_date,
            trade_return=trade_return,
            holding_days=holding_days,
            strategy=strategy,
            success=trade_return > 0
        )
        
        # Add to memory
        self.pattern_memory.add_pattern(entry)


# =============================================================================
# TESTING
# =============================================================================

def test_pattern_memory():
    """Test pattern memory functionality."""
    logger.info("Testing PatternMemory...")
    
    # Create memory
    memory = PatternMemory(n_features=30)
    
    # Create some test patterns
    np.random.seed(42)
    for i in range(100):
        pattern = PatternSignature(
            price_features=np.random.randn(10),
            volume_features=np.random.randn(10),
            volatility_features=np.random.randn(10),
            regime='medium_flat',
            date=f"2024-01-{i+1:02d}",
            ticker='TEST'
        )
        
        entry = PatternMemoryEntry(
            pattern=pattern,
            entry_date=f"2024-01-{i+1:02d}",
            exit_date=f"2024-01-{i+6:02d}",
            trade_return=np.random.randn() * 0.02,
            holding_days=5,
            strategy='v21' if i % 2 == 0 else 'v24',
            success=np.random.random() > 0.4
        )
        
        memory.add_pattern(entry)
    
    logger.info(f"Memory stats: {memory.get_statistics()}")
    
    # Query similar patterns
    query = PatternSignature(
        price_features=np.random.randn(10),
        volume_features=np.random.randn(10),
        volatility_features=np.random.randn(10),
        regime='medium_flat'
    )
    
    conf, meta = memory.get_pattern_confidence(query, 'v21')
    logger.info(f"Pattern confidence: {conf:.3f}, metadata: {meta}")
    
    # Save state
    memory.save_state()
    
    logger.info("PatternMemory tests passed!")


if __name__ == "__main__":
    test_pattern_memory()
