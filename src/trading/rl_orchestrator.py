"""
V2.2 RL Orchestrator
====================

Orchestrates the RL-enhanced position sizing pipeline:
1. Hierarchical regime detection → policy selection
2. SAC position optimization → risk-aware sizing
3. Anomaly-aware attention → confidence scoring

Integration with V2.1 Production Engine:
- Wraps existing signal generation
- Adds RL-based position scaling
- Provides uncertainty-adjusted execution

Usage:
    orchestrator = RLOrchestrator(config)
    enhanced_signals = orchestrator.enhance_signals(base_signals, market_data)
"""

import os
import sys
import json
import logging
import numpy as np
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from collections import deque

logger = logging.getLogger(__name__)

# Import RL components
try:
    from src.agents.sac_position_optimizer import (
        SACPositionOptimizer, SACConfig, ReplayBuffer
    )
    from src.regime.hierarchical_controller import (
        HierarchicalController, RegimeState, SubPolicy
    )
    from src.models.anomaly_aware_transformer import (
        AnomalyAwareTransformer, TransformerConfig, IsolationForestDetector
    )
    RL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"RL components not fully available: {e}")
    RL_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class RLOrchestratorConfig:
    """Configuration for RL orchestrator."""
    
    # Component toggles
    use_sac: bool = True
    use_hierarchical_regime: bool = True
    use_anomaly_transformer: bool = True
    
    # SAC parameters
    sac_state_dim: int = 32
    sac_learning_rate: float = 3e-4
    sac_batch_size: int = 256
    sac_tau: float = 0.005
    sac_gamma: float = 0.99
    
    # Position sizing
    base_position_pct: float = 0.02  # 2% base position
    max_position_pct: float = 0.03   # 3% max position
    min_position_pct: float = 0.005  # 0.5% min position
    
    # Risk adjustments
    high_vol_scale: float = 0.6
    transition_scale: float = 0.7
    low_confidence_scale: float = 0.5
    
    # Anomaly handling
    anomaly_threshold: float = 0.7
    anomaly_position_cap: float = 0.01  # 1% max during anomalies
    
    # Logging
    log_dir: str = "logs"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# MARKET STATE ENCODER
# =============================================================================

class MarketStateEncoder:
    """
    Encodes market data into state vectors for RL agents.
    
    Features extracted:
    - Price momentum (multiple timeframes)
    - Volatility measures (realized, EWMA)
    - TDA features (persistence, entropy)
    - Cross-sectional ranks
    - Regime indicators
    """
    
    def __init__(self, state_dim: int = 32):
        self.state_dim = state_dim
        self.price_buffers: Dict[str, deque] = {}
        self.return_buffers: Dict[str, deque] = {}
        
        # Normalization stats
        self.feature_means = np.zeros(state_dim)
        self.feature_stds = np.ones(state_dim)
        self.n_samples = 0
        
    def encode(self, 
               ticker: str,
               prices: np.ndarray,
               tda_features: Optional[Dict[str, float]] = None,
               regime_state: Optional[RegimeState] = None) -> np.ndarray:
        """
        Encode market data into state vector.
        
        Args:
            ticker: Asset symbol
            prices: Recent price history
            tda_features: Optional TDA features dict
            regime_state: Optional regime state
            
        Returns:
            Normalized state vector (state_dim,)
        """
        features = []
        
        # 1. Price momentum features (10 dims)
        returns = np.diff(np.log(prices + 1e-8))
        
        if len(returns) >= 5:
            features.append(returns[-5:].mean())   # 5-day momentum
        else:
            features.append(0.0)
            
        if len(returns) >= 20:
            features.append(returns[-20:].mean())  # 20-day momentum
        else:
            features.append(0.0)
            
        if len(returns) >= 60:
            features.append(returns[-60:].mean())  # 60-day momentum
        else:
            features.append(0.0)
            
        # Momentum rank (relative strength)
        if len(returns) >= 20:
            features.append(np.percentile(returns, 75) - np.percentile(returns, 25))
        else:
            features.append(0.0)
            
        # Recent returns
        for i in [1, 2, 3, 5, 10]:
            if len(returns) >= i:
                features.append(returns[-i])
            else:
                features.append(0.0)
                
        # 2. Volatility features (6 dims)
        if len(returns) >= 5:
            features.append(np.std(returns[-5:]) * np.sqrt(252))  # 5-day vol
        else:
            features.append(0.15)
            
        if len(returns) >= 20:
            features.append(np.std(returns[-20:]) * np.sqrt(252))  # 20-day vol
        else:
            features.append(0.15)
            
        if len(returns) >= 60:
            features.append(np.std(returns[-60:]) * np.sqrt(252))  # 60-day vol
        else:
            features.append(0.15)
            
        # Vol of vol
        if len(returns) >= 20:
            rolling_vol = [np.std(returns[i:i+5]) for i in range(len(returns) - 5)]
            features.append(np.std(rolling_vol) if rolling_vol else 0.0)
        else:
            features.append(0.0)
            
        # Skewness and kurtosis
        if len(returns) >= 20:
            from scipy.stats import skew, kurtosis
            features.append(skew(returns[-20:]))
            features.append(kurtosis(returns[-20:]))
        else:
            features.extend([0.0, 0.0])
            
        # 3. TDA features (6 dims)
        if tda_features:
            features.append(tda_features.get("persistence_entropy", 0.0))
            features.append(tda_features.get("betti_0", 0.0))
            features.append(tda_features.get("betti_1", 0.0))
            features.append(tda_features.get("wasserstein_1", 0.0))
            features.append(tda_features.get("landscape_norm", 0.0))
            features.append(tda_features.get("silhouette", 0.0))
        else:
            features.extend([0.0] * 6)
            
        # 4. Regime features (5 dims)
        if regime_state:
            # Volatility regime one-hot
            vol_regimes = {"low": 0, "medium": 1, "high": 2}
            vol_idx = vol_regimes.get(regime_state.volatility.value, 1)
            vol_onehot = [1.0 if i == vol_idx else 0.0 for i in range(3)]
            features.extend(vol_onehot)
            
            features.append(regime_state.confidence)
            features.append(regime_state.transition_prob)
        else:
            features.extend([0.0, 1.0, 0.0, 0.5, 0.1])  # Default neutral
            
        # 5. Pad/truncate to state_dim
        features = features[:self.state_dim]
        while len(features) < self.state_dim:
            features.append(0.0)
            
        state = np.array(features, dtype=np.float32)
        
        # Update normalization stats
        self.n_samples += 1
        delta = state - self.feature_means
        self.feature_means += delta / self.n_samples
        self.feature_stds = np.sqrt(
            (self.feature_stds ** 2) * (self.n_samples - 1) / self.n_samples +
            delta ** 2 / self.n_samples
        )
        
        # Normalize
        normalized = (state - self.feature_means) / (self.feature_stds + 1e-8)
        return np.clip(normalized, -5, 5)  # Clip extreme values


# =============================================================================
# RL ORCHESTRATOR
# =============================================================================

class RLOrchestrator:
    """
    Orchestrates RL-enhanced position sizing.
    
    Pipeline:
    1. Market state encoding
    2. Hierarchical regime detection → policy selection
    3. SAC position optimization → base sizing
    4. Anomaly detection → confidence adjustment
    5. Final position scaling
    
    Logging:
    - All decisions logged to rl_decisions.jsonl
    - Regime transitions logged separately
    - Metrics for analysis
    """
    
    def __init__(self, config: Optional[RLOrchestratorConfig] = None):
        """
        Initialize RL orchestrator.
        
        Args:
            config: Orchestrator configuration
        """
        self.config = config or RLOrchestratorConfig()
        
        # Logging setup
        self.log_dir = Path(self.config.log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.decisions_log = self.log_dir / "rl_decisions.jsonl"
        
        # State encoder
        self.state_encoder = MarketStateEncoder(self.config.sac_state_dim)
        
        # Initialize components
        self._init_components()
        
        # Statistics tracking
        self.total_decisions = 0
        self.anomaly_detections = 0
        self.regime_transitions = 0
        
        logger.info(f"RLOrchestrator initialized (RL available: {RL_AVAILABLE})")
        
    def _init_components(self):
        """Initialize RL components."""
        
        # 1. SAC Position Optimizer
        if self.config.use_sac and RL_AVAILABLE:
            sac_config = SACConfig(
                state_dim=self.config.sac_state_dim,
                learning_rate=self.config.sac_learning_rate,
                batch_size=self.config.sac_batch_size,
                tau=self.config.sac_tau,
                gamma=self.config.sac_gamma,
                max_position_pct=self.config.max_position_pct,
                log_dir=self.config.log_dir,
            )
            self.sac = SACPositionOptimizer(sac_config)
            logger.info("✅ SAC Position Optimizer initialized")
        else:
            self.sac = None
            logger.info("ℹ️  SAC disabled or unavailable")
            
        # 2. Hierarchical Regime Controller
        if self.config.use_hierarchical_regime and RL_AVAILABLE:
            self.regime_controller = HierarchicalController(
                log_dir=self.config.log_dir
            )
            logger.info("✅ Hierarchical Regime Controller initialized")
        else:
            self.regime_controller = None
            logger.info("ℹ️  Hierarchical regime disabled or unavailable")
            
        # 3. Anomaly Detector (using IsolationForest directly)
        if self.config.use_anomaly_transformer and RL_AVAILABLE:
            self.anomaly_detector = IsolationForestDetector(
                contamination=0.05,
                n_estimators=100,
            )
            logger.info("✅ Anomaly Detector initialized")
        else:
            self.anomaly_detector = None
            logger.info("ℹ️  Anomaly detection disabled or unavailable")
            
    def enhance_signals(self,
                        base_signals: Dict[str, Dict[str, Any]],
                        market_data: Dict[str, np.ndarray],
                        tda_features: Optional[Dict[str, Dict]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Enhance base signals with RL-based position sizing.
        
        Args:
            base_signals: Dict mapping ticker -> signal info from V2.1
            market_data: Dict mapping ticker -> price array
            tda_features: Optional TDA features per ticker
            
        Returns:
            Enhanced signals with RL position sizing
        """
        if not base_signals:
            return {}
            
        enhanced = {}
        
        # 1. Update regime state
        regime_state = self._update_regime(market_data)
        regime_string = self._get_regime_string(regime_state)
        
        # 2. Get current policy from regime
        policy = self._get_active_policy()
        
        # 3. Detect anomalies in market data
        anomaly_scores = self._detect_anomalies(market_data)
        
        # 4. Process each signal
        for ticker, signal in base_signals.items():
            try:
                enhanced_signal = self._enhance_single_signal(
                    ticker=ticker,
                    signal=signal,
                    prices=market_data.get(ticker, np.array([])),
                    tda_features=tda_features.get(ticker) if tda_features else None,
                    regime_state=regime_state,
                    regime_string=regime_string,
                    policy=policy,
                    anomaly_score=anomaly_scores.get(ticker, 0.0),
                )
                enhanced[ticker] = enhanced_signal
                
            except Exception as e:
                logger.warning(f"Failed to enhance signal for {ticker}: {e}")
                enhanced[ticker] = signal  # Fallback to base signal
                
        # 5. Log summary
        self._log_decision_summary(enhanced, regime_state, anomaly_scores)
        
        return enhanced
    
    def _enhance_single_signal(self,
                               ticker: str,
                               signal: Dict[str, Any],
                               prices: np.ndarray,
                               tda_features: Optional[Dict],
                               regime_state: Optional[RegimeState],
                               regime_string: str,
                               policy: Optional[SubPolicy],
                               anomaly_score: float) -> Dict[str, Any]:
        """
        Enhance a single signal with RL position sizing.
        """
        enhanced = signal.copy()
        base_weight = signal.get("weight", 0.0)
        base_direction = signal.get("direction", "flat")
        
        if base_direction == "flat" or abs(base_weight) < 0.001:
            return enhanced
            
        # 1. Encode market state
        if len(prices) > 0:
            state = self.state_encoder.encode(
                ticker, prices, tda_features, regime_state
            )
        else:
            state = np.zeros(self.config.sac_state_dim)
            
        # 2. Get SAC position recommendation
        if self.sac:
            sac_position = self.sac.get_position(
                state, 
                regime=regime_string,
                deterministic=True,
            )
        else:
            sac_position = self.config.base_position_pct
            
        # 3. Apply regime-based scaling
        regime_scale = 1.0
        if policy:
            regime_scale = policy.position_scale
            
        # Reduce position during regime transitions
        if regime_state and regime_state.transition_prob > 0.5:
            regime_scale *= self.config.transition_scale
            
        # 4. Apply anomaly-based scaling
        anomaly_scale = 1.0
        if anomaly_score > self.config.anomaly_threshold:
            anomaly_scale = self.config.low_confidence_scale
            self.anomaly_detections += 1
            
        # 5. Compute final position size
        # Blend SAC recommendation with base signal
        blend_weight = 0.6  # 60% SAC, 40% base signal
        final_position = (
            blend_weight * sac_position +
            (1 - blend_weight) * abs(base_weight)
        )
        
        # Apply scalings
        final_position *= regime_scale * anomaly_scale
        
        # Apply constraints
        final_position = np.clip(
            final_position,
            self.config.min_position_pct,
            self.config.max_position_pct,
        )
        
        # Cap position if anomaly detected
        if anomaly_score > self.config.anomaly_threshold:
            final_position = min(final_position, self.config.anomaly_position_cap)
            
        # 6. Update enhanced signal
        enhanced["weight"] = final_position if base_direction == "long" else -final_position
        enhanced["rl_enhanced"] = True
        enhanced["sac_position"] = sac_position
        enhanced["regime_scale"] = regime_scale
        enhanced["anomaly_scale"] = anomaly_scale
        enhanced["anomaly_score"] = anomaly_score
        enhanced["regime"] = regime_string
        enhanced["confidence"] = signal.get("confidence", 0.5) * (1 - 0.3 * anomaly_score)
        
        self.total_decisions += 1
        
        return enhanced
    
    def _update_regime(self, 
                       market_data: Dict[str, np.ndarray]) -> Optional[RegimeState]:
        """
        Update regime state from market data.
        """
        if not self.regime_controller:
            return None
            
        # Compute aggregate returns and prices
        all_returns = []
        all_prices = []
        
        for ticker, prices in market_data.items():
            if len(prices) >= 2:
                returns = np.diff(np.log(prices + 1e-8))
                all_returns.extend(returns)
                all_prices.extend(prices)
                
        if not all_returns:
            return None
            
        returns_arr = np.array(all_returns[-60:])  # Last 60 days
        prices_arr = np.array(all_prices[-100:])   # Last 100 days
        
        # Update controller
        prev_state = self.regime_controller.current_state
        new_state = self.regime_controller.update(returns_arr, prices_arr)
        
        # Check for transition
        if prev_state and new_state.meta_state != prev_state.meta_state:
            self.regime_transitions += 1
            logger.info(f"Regime transition: {prev_state.meta_state} → {new_state.meta_state}")
            
        return new_state
    
    def _get_regime_string(self, regime_state: Optional[RegimeState]) -> str:
        """Get simple regime string for SAC."""
        if not regime_state:
            return "neutral"
            
        if self.regime_controller:
            return self.regime_controller.get_regime_string()
        return "neutral"
    
    def _get_active_policy(self) -> Optional[SubPolicy]:
        """Get active sub-policy from regime controller."""
        if not self.regime_controller:
            return None
        return self.regime_controller.get_active_policy()
    
    def _detect_anomalies(self, 
                          market_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Detect anomalies in market data.
        
        Returns:
            Dict mapping ticker -> anomaly score [0, 1]
        """
        anomaly_scores = {}
        
        if not self.anomaly_detector:
            return anomaly_scores
            
        # Prepare features
        for ticker, prices in market_data.items():
            if len(prices) < 20:
                anomaly_scores[ticker] = 0.0
                continue
                
            # Compute features for anomaly detection
            returns = np.diff(np.log(prices + 1e-8))
            
            features = np.array([
                np.mean(returns[-5:]),
                np.std(returns[-5:]),
                np.mean(returns[-20:]),
                np.std(returns[-20:]),
                np.max(np.abs(returns[-5:])),
            ]).reshape(1, -1)
            
            # Detect
            if not self.anomaly_detector.is_fitted:
                # Fit on this batch
                all_features = []
                for t, p in market_data.items():
                    if len(p) >= 20:
                        r = np.diff(np.log(p + 1e-8))
                        f = [np.mean(r[-5:]), np.std(r[-5:]), np.mean(r[-20:]),
                             np.std(r[-20:]), np.max(np.abs(r[-5:]))]
                        all_features.append(f)
                        
                if all_features:
                    self.anomaly_detector.fit(np.array(all_features))
                    
            _, scores = self.anomaly_detector.detect(features)
            anomaly_scores[ticker] = float(scores[0])
            
        return anomaly_scores
    
    def _log_decision_summary(self,
                              signals: Dict[str, Dict],
                              regime_state: Optional[RegimeState],
                              anomaly_scores: Dict[str, float]):
        """Log decision summary to JSONL."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_signals": len(signals),
            "rl_enhanced_count": sum(1 for s in signals.values() if s.get("rl_enhanced")),
            "regime": regime_state.to_dict() if regime_state else None,
            "mean_anomaly_score": np.mean(list(anomaly_scores.values())) if anomaly_scores else 0.0,
            "high_anomaly_count": sum(1 for s in anomaly_scores.values() if s > self.config.anomaly_threshold),
            "total_decisions": self.total_decisions,
            "total_anomaly_detections": self.anomaly_detections,
            "total_regime_transitions": self.regime_transitions,
        }
        
        try:
            with open(self.decisions_log, 'a') as f:
                f.write(json.dumps(summary) + "\n")
        except Exception as e:
            logger.debug(f"Failed to log decision summary: {e}")
            
    def train_online(self,
                     state: np.ndarray,
                     action: float,
                     reward: float,
                     next_state: np.ndarray,
                     done: bool = False):
        """
        Online training for SAC from experience.
        
        Call this after each trading day with actual PnL.
        """
        if not self.sac:
            return
            
        self.sac.store_experience(
            state,
            np.array([action]),
            reward,
            next_state,
            done,
        )
        
        # Update if enough experience
        metrics = self.sac.update()
        
        if metrics:
            logger.debug(f"SAC update: {metrics}")
            
    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        stats = {
            "total_decisions": self.total_decisions,
            "anomaly_detections": self.anomaly_detections,
            "regime_transitions": self.regime_transitions,
            "rl_available": RL_AVAILABLE,
            "components": {
                "sac_enabled": self.sac is not None,
                "regime_enabled": self.regime_controller is not None,
                "anomaly_enabled": self.anomaly_detector is not None,
            },
        }
        
        if self.sac:
            stats["sac_stats"] = self.sac.get_training_stats()
            
        if self.regime_controller:
            stats["regime_state"] = self.regime_controller.get_state_summary()
            
        return stats
    
    def save(self, path: Path):
        """Save all component states."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        if self.sac:
            self.sac.save(path / "sac_model.pt")
            
        # Save stats
        with open(path / "orchestrator_stats.json", 'w') as f:
            json.dump(self.get_stats(), f, indent=2, default=str)
            
        logger.info(f"RLOrchestrator saved to {path}")
        
    def load(self, path: Path):
        """Load component states."""
        path = Path(path)
        
        sac_path = path / "sac_model.pt"
        if sac_path.exists() and self.sac:
            self.sac.load(sac_path)
            
        logger.info(f"RLOrchestrator loaded from {path}")
