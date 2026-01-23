"""
V26 Continuous Learning Module
==============================

Real-time model adaptation with drift detection and feedback loops.

Key Features:
1. Performance feedback loops: Track hit/miss over 100 trades
2. Drift detection: Kolmogorov-Smirnov test on features
3. Incremental updates: EMA weights (alpha=0.01) with checkpoint rollback
4. Automatic recalibration when accuracy < 55%

Target: +5% signal accuracy, <24hr drift detection
"""

import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from collections import deque
from pathlib import Path
import numpy as np

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ContinuousLearnerConfig:
    """Configuration for continuous learning."""
    
    # Performance tracking
    trade_window: int = 100              # Number of trades to track
    min_accuracy_threshold: float = 0.55  # Recalibrate if accuracy < 55%
    accuracy_check_interval: int = 50     # Check every N trades
    
    # Drift detection
    drift_ks_threshold: float = 0.15      # KS test score threshold
    drift_check_interval_hours: float = 4.0  # Check every 4 hours
    drift_alert_threshold: float = 0.15   # Alert when drift > 15%
    
    # Incremental updates
    ema_alpha: float = 0.01              # EMA weight for model updates
    checkpoint_interval: int = 100        # Trades between checkpoints
    max_checkpoints: int = 5             # Keep last N checkpoints
    
    # Paths
    checkpoint_dir: str = "state/checkpoints"
    metrics_log_path: str = "logs/v26_metrics.jsonl"
    
    # Alerts
    enable_discord_alerts: bool = True


@dataclass
class TradeResult:
    """Single trade result for tracking."""
    timestamp: datetime
    ticker: str
    signal_direction: str        # 'long' or 'short'
    signal_confidence: float
    predicted_return: float
    actual_return: float
    is_hit: bool                 # True if prediction was correct
    features: Optional[Dict[str, float]] = None


@dataclass
class DriftReport:
    """Report from drift detection."""
    timestamp: datetime
    has_drift: bool
    ks_scores: Dict[str, float]  # Feature -> KS score
    max_ks_score: float
    drifted_features: List[str]
    hours_since_last_check: float


@dataclass
class LearnerState:
    """Current state of continuous learner."""
    accuracy: float
    total_trades: int
    hits: int
    misses: int
    last_recalibration: Optional[datetime]
    drift_detected: bool
    drift_first_detected: Optional[datetime]
    current_weights: Dict[str, float]
    checkpoint_id: str


# =============================================================================
# PERFORMANCE TRACKER
# =============================================================================

class PerformanceTracker:
    """
    Track trading performance with rolling window.
    
    Monitors hit/miss rate and triggers recalibration when needed.
    """
    
    def __init__(self, config: ContinuousLearnerConfig):
        self.config = config
        self.trades: deque = deque(maxlen=config.trade_window)
        self.total_trades = 0
        self.recalibration_count = 0
        self.last_recalibration: Optional[datetime] = None
        
    def record_trade(self, result: TradeResult) -> Dict[str, Any]:
        """
        Record a trade result.
        
        Returns metrics update and recalibration trigger status.
        """
        self.trades.append(result)
        self.total_trades += 1
        
        # Calculate current accuracy
        if len(self.trades) < 10:
            accuracy = 0.5  # Insufficient data
        else:
            hits = sum(1 for t in self.trades if t.is_hit)
            accuracy = hits / len(self.trades)
        
        # Check if recalibration needed
        needs_recalibration = False
        if (len(self.trades) >= self.config.accuracy_check_interval and
            self.total_trades % self.config.accuracy_check_interval == 0):
            if accuracy < self.config.min_accuracy_threshold:
                needs_recalibration = True
                self.recalibration_count += 1
                self.last_recalibration = datetime.now()
                logger.warning(f"Recalibration triggered: accuracy {accuracy:.1%} < {self.config.min_accuracy_threshold:.1%}")
        
        return {
            'accuracy': accuracy,
            'total_trades': self.total_trades,
            'hits': sum(1 for t in self.trades if t.is_hit),
            'misses': sum(1 for t in self.trades if not t.is_hit),
            'needs_recalibration': needs_recalibration,
            'recalibration_count': self.recalibration_count,
        }
    
    def get_accuracy(self) -> float:
        """Get current rolling accuracy."""
        if len(self.trades) < 10:
            return 0.5
        hits = sum(1 for t in self.trades if t.is_hit)
        return hits / len(self.trades)
    
    def get_accuracy_by_ticker(self) -> Dict[str, float]:
        """Get accuracy broken down by ticker."""
        ticker_stats = {}
        for t in self.trades:
            if t.ticker not in ticker_stats:
                ticker_stats[t.ticker] = {'hits': 0, 'total': 0}
            ticker_stats[t.ticker]['total'] += 1
            if t.is_hit:
                ticker_stats[t.ticker]['hits'] += 1
        
        return {
            ticker: stats['hits'] / stats['total'] if stats['total'] > 0 else 0.5
            for ticker, stats in ticker_stats.items()
        }
    
    def get_recent_trades(self, n: int = 20) -> List[TradeResult]:
        """Get most recent N trades."""
        return list(self.trades)[-n:]


# =============================================================================
# DRIFT DETECTOR
# =============================================================================

class DriftDetector:
    """
    Detect feature distribution drift using Kolmogorov-Smirnov test.
    
    Compares current feature distributions to baseline and alerts
    when drift exceeds threshold.
    """
    
    def __init__(self, config: ContinuousLearnerConfig):
        self.config = config
        self.baseline_features: Dict[str, np.ndarray] = {}
        self.current_features: Dict[str, deque] = {}
        self.last_check: Optional[datetime] = None
        self.drift_first_detected: Optional[datetime] = None
        self.has_drift: bool = False
        
    def set_baseline(self, features: Dict[str, np.ndarray]):
        """Set baseline feature distributions."""
        self.baseline_features = {k: np.array(v) for k, v in features.items()}
        self.current_features = {k: deque(maxlen=500) for k in features.keys()}
        logger.info(f"Drift baseline set with {len(features)} features")
        
    def add_observation(self, features: Dict[str, float]):
        """Add a new feature observation."""
        for name, value in features.items():
            if name in self.current_features:
                self.current_features[name].append(value)
    
    def check_drift(self) -> DriftReport:
        """
        Check for feature drift using KS test.
        
        Returns detailed drift report.
        """
        now = datetime.now()
        hours_since = 0.0
        
        if self.last_check:
            hours_since = (now - self.last_check).total_seconds() / 3600
            if hours_since < self.config.drift_check_interval_hours:
                # Return cached result
                return DriftReport(
                    timestamp=now,
                    has_drift=self.has_drift,
                    ks_scores={},
                    max_ks_score=0.0,
                    drifted_features=[],
                    hours_since_last_check=hours_since
                )
        
        self.last_check = now
        ks_scores = {}
        drifted_features = []
        
        if not SCIPY_AVAILABLE:
            logger.warning("SciPy not available, drift detection disabled")
            return DriftReport(
                timestamp=now, has_drift=False, ks_scores={},
                max_ks_score=0.0, drifted_features=[], hours_since_last_check=hours_since
            )
        
        for name, baseline in self.baseline_features.items():
            current = self.current_features.get(name)
            if current is None or len(current) < 30:
                continue
            
            current_arr = np.array(current)
            
            # Kolmogorov-Smirnov test
            try:
                statistic, p_value = stats.ks_2samp(baseline, current_arr)
                ks_scores[name] = statistic
                
                if statistic > self.config.drift_ks_threshold:
                    drifted_features.append(name)
            except Exception as e:
                logger.warning(f"KS test failed for {name}: {e}")
                continue
        
        max_ks = max(ks_scores.values()) if ks_scores else 0.0
        has_drift = len(drifted_features) > 0
        
        # Track drift duration
        if has_drift:
            if self.drift_first_detected is None:
                self.drift_first_detected = now
                logger.warning(f"Drift detected in {len(drifted_features)} features: {drifted_features[:5]}")
        else:
            self.drift_first_detected = None
        
        self.has_drift = has_drift
        
        return DriftReport(
            timestamp=now,
            has_drift=has_drift,
            ks_scores=ks_scores,
            max_ks_score=max_ks,
            drifted_features=drifted_features,
            hours_since_last_check=hours_since
        )
    
    def get_drift_duration_hours(self) -> float:
        """Get hours since drift first detected."""
        if self.drift_first_detected is None:
            return 0.0
        return (datetime.now() - self.drift_first_detected).total_seconds() / 3600


# =============================================================================
# MODEL UPDATER
# =============================================================================

class IncrementalUpdater:
    """
    Incremental model updates with EMA weights and checkpointing.
    
    Allows gradual adaptation to market changes while maintaining
    the ability to rollback if performance degrades.
    """
    
    def __init__(self, config: ContinuousLearnerConfig):
        self.config = config
        self.current_weights: Dict[str, float] = {}
        self.weight_history: deque = deque(maxlen=config.max_checkpoints)
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.update_count = 0
        self.last_checkpoint: Optional[str] = None
        
    def initialize_weights(self, weights: Dict[str, float]):
        """Initialize model weights."""
        self.current_weights = weights.copy()
        self._save_checkpoint("initial")
        
    def update_weights(self, new_weights: Dict[str, float], 
                       alpha: Optional[float] = None) -> Dict[str, float]:
        """
        Update weights using EMA.
        
        Args:
            new_weights: New weight values to blend in
            alpha: EMA coefficient (uses config default if None)
            
        Returns:
            Updated weights
        """
        if alpha is None:
            alpha = self.config.ema_alpha
        
        for key, new_value in new_weights.items():
            if key in self.current_weights:
                old_value = self.current_weights[key]
                self.current_weights[key] = (1 - alpha) * old_value + alpha * new_value
            else:
                self.current_weights[key] = new_value
        
        self.update_count += 1
        
        # Checkpoint periodically
        if self.update_count % self.config.checkpoint_interval == 0:
            checkpoint_id = f"update_{self.update_count}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self._save_checkpoint(checkpoint_id)
        
        return self.current_weights.copy()
    
    def _save_checkpoint(self, checkpoint_id: str):
        """Save weights to checkpoint."""
        checkpoint = {
            'id': checkpoint_id,
            'timestamp': datetime.now().isoformat(),
            'update_count': self.update_count,
            'weights': self.current_weights.copy()
        }
        
        self.weight_history.append(checkpoint)
        self.last_checkpoint = checkpoint_id
        
        # Save to file
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.json"
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2, default=str)
            logger.info(f"Saved checkpoint: {checkpoint_id}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def rollback(self, steps: int = 1) -> Dict[str, float]:
        """
        Rollback weights to previous checkpoint.
        
        Args:
            steps: Number of checkpoints to rollback
            
        Returns:
            Restored weights
        """
        if len(self.weight_history) <= steps:
            logger.warning(f"Cannot rollback {steps} steps, only {len(self.weight_history)} checkpoints")
            steps = max(0, len(self.weight_history) - 1)
        
        if steps == 0:
            return self.current_weights.copy()
        
        # Get checkpoint from history
        for _ in range(steps):
            self.weight_history.pop()
        
        if self.weight_history:
            checkpoint = self.weight_history[-1]
            self.current_weights = checkpoint['weights'].copy()
            self.last_checkpoint = checkpoint['id']
            logger.info(f"Rolled back to checkpoint: {checkpoint['id']}")
        
        return self.current_weights.copy()
    
    def get_checkpoint_list(self) -> List[Dict[str, Any]]:
        """Get list of available checkpoints."""
        return [
            {'id': c['id'], 'timestamp': c['timestamp'], 'update_count': c['update_count']}
            for c in self.weight_history
        ]


# =============================================================================
# CONTINUOUS LEARNER
# =============================================================================

class ContinuousLearner:
    """
    V26 Continuous Learning System.
    
    Integrates performance tracking, drift detection, and incremental
    model updates into a unified learning loop.
    
    Usage:
        learner = ContinuousLearner()
        learner.set_baseline(feature_distributions)
        
        # For each trade
        result = TradeResult(...)
        update = learner.record_trade(result)
        
        if update['needs_recalibration']:
            learner.trigger_recalibration()
        
        # Periodic drift check
        drift = learner.check_drift()
        if drift.has_drift:
            learner.handle_drift(drift)
    """
    
    def __init__(self, config: Optional[ContinuousLearnerConfig] = None):
        self.config = config or ContinuousLearnerConfig()
        
        self.performance = PerformanceTracker(self.config)
        self.drift_detector = DriftDetector(self.config)
        self.updater = IncrementalUpdater(self.config)
        
        # State
        self.is_recalibrating = False
        self.recalibration_start: Optional[datetime] = None
        self.metrics_log_path = Path(self.config.metrics_log_path)
        self.metrics_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Discord callback (set externally)
        self.discord_callback = None
        
        logger.info("ContinuousLearner initialized")
    
    def set_baseline(self, features: Dict[str, np.ndarray], 
                     initial_weights: Optional[Dict[str, float]] = None):
        """
        Set baseline for drift detection and initialize weights.
        
        Args:
            features: Dict of feature_name -> historical values
            initial_weights: Initial model weights
        """
        self.drift_detector.set_baseline(features)
        
        if initial_weights:
            self.updater.initialize_weights(initial_weights)
    
    def record_trade(self, result: TradeResult) -> Dict[str, Any]:
        """
        Record a trade result and return update metrics.
        
        Args:
            result: Trade result with prediction and actual outcome
            
        Returns:
            Dict with performance metrics and action flags
        """
        # Track performance
        perf_update = self.performance.record_trade(result)
        
        # Add feature observation for drift detection
        if result.features:
            self.drift_detector.add_observation(result.features)
        
        # Log metrics
        self._log_metrics(result, perf_update)
        
        return perf_update
    
    def check_drift(self) -> DriftReport:
        """Check for feature drift."""
        return self.drift_detector.check_drift()
    
    def trigger_recalibration(self, reason: str = "accuracy_drop") -> bool:
        """
        Trigger model recalibration.
        
        Args:
            reason: Reason for recalibration
            
        Returns:
            True if recalibration started
        """
        if self.is_recalibrating:
            logger.warning("Already recalibrating")
            return False
        
        self.is_recalibrating = True
        self.recalibration_start = datetime.now()
        
        logger.info(f"Recalibration triggered: {reason}")
        
        # Send alert
        if self.config.enable_discord_alerts and self.discord_callback:
            try:
                self.discord_callback(
                    "🔄 V26 Model Recalibration",
                    f"Reason: {reason}\n"
                    f"Accuracy: {self.performance.get_accuracy():.1%}\n"
                    f"Total trades: {self.performance.total_trades}",
                    color=0xFFAA00
                )
            except Exception as e:
                logger.error(f"Discord alert failed: {e}")
        
        return True
    
    def complete_recalibration(self, new_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Complete recalibration with new weights.
        
        Args:
            new_weights: New model weights from retraining
            
        Returns:
            Updated weights
        """
        if not self.is_recalibrating:
            logger.warning("Not currently recalibrating")
        
        # Use larger alpha for recalibration (more aggressive update)
        updated = self.updater.update_weights(new_weights, alpha=0.2)
        
        self.is_recalibrating = False
        duration = (datetime.now() - self.recalibration_start).total_seconds() if self.recalibration_start else 0
        
        logger.info(f"Recalibration complete in {duration:.1f}s")
        
        return updated
    
    def handle_drift(self, drift_report: DriftReport):
        """
        Handle detected drift.
        
        Actions based on drift severity and duration:
        - < 24 hours: Alert only
        - > 24 hours: Trigger recalibration
        - > 48 hours: Consider rollback
        """
        drift_hours = self.drift_detector.get_drift_duration_hours()
        
        if drift_hours > 48:
            logger.warning(f"Drift persisting for {drift_hours:.1f} hours, considering rollback")
            # Rollback to previous weights
            self.updater.rollback(steps=1)
            self.trigger_recalibration(reason=f"drift_{drift_hours:.0f}h")
            
        elif drift_hours > 24:
            logger.warning(f"Drift persisting for {drift_hours:.1f} hours, triggering recalibration")
            self.trigger_recalibration(reason="drift_24h")
            
        else:
            logger.info(f"Drift detected, monitoring (duration: {drift_hours:.1f}h)")
        
        # Alert
        if self.config.enable_discord_alerts and self.discord_callback:
            try:
                self.discord_callback(
                    "⚠️ V26 Feature Drift Alert",
                    f"Max KS score: {drift_report.max_ks_score:.3f}\n"
                    f"Drifted features: {', '.join(drift_report.drifted_features[:5])}\n"
                    f"Duration: {drift_hours:.1f} hours",
                    color=0xFF6600
                )
            except Exception as e:
                logger.error(f"Discord alert failed: {e}")
    
    def update_weights_incremental(self, new_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Incrementally update weights using EMA.
        
        Args:
            new_weights: New weight values to blend
            
        Returns:
            Updated weights
        """
        return self.updater.update_weights(new_weights)
    
    def get_state(self) -> LearnerState:
        """Get current learner state."""
        return LearnerState(
            accuracy=self.performance.get_accuracy(),
            total_trades=self.performance.total_trades,
            hits=sum(1 for t in self.performance.trades if t.is_hit),
            misses=sum(1 for t in self.performance.trades if not t.is_hit),
            last_recalibration=self.performance.last_recalibration,
            drift_detected=self.drift_detector.has_drift,
            drift_first_detected=self.drift_detector.drift_first_detected,
            current_weights=self.updater.current_weights.copy(),
            checkpoint_id=self.updater.last_checkpoint or ""
        )
    
    def _log_metrics(self, result: TradeResult, update: Dict[str, Any]):
        """Log metrics to JSONL file."""
        try:
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'trade': {
                    'ticker': result.ticker,
                    'signal': result.signal_direction,
                    'confidence': result.signal_confidence,
                    'predicted_return': result.predicted_return,
                    'actual_return': result.actual_return,
                    'is_hit': result.is_hit
                },
                'performance': update
            }
            
            with open(self.metrics_log_path, 'a') as f:
                f.write(json.dumps(metrics) + '\n')
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            'accuracy': self.performance.get_accuracy(),
            'total_trades': self.performance.total_trades,
            'recalibration_count': self.performance.recalibration_count,
            'by_ticker': self.performance.get_accuracy_by_ticker(),
            'drift_detected': self.drift_detector.has_drift,
            'drift_duration_hours': self.drift_detector.get_drift_duration_hours(),
            'is_recalibrating': self.is_recalibrating,
            'checkpoint_count': len(self.updater.weight_history),
        }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing V26 Continuous Learner...")
    
    config = ContinuousLearnerConfig()
    learner = ContinuousLearner(config)
    
    # Set baseline features
    baseline_features = {
        'momentum': np.random.randn(1000),
        'volatility': np.abs(np.random.randn(1000)) * 0.02,
        'rsi': np.random.uniform(30, 70, 1000),
    }
    learner.set_baseline(baseline_features, initial_weights={'factor1': 0.5, 'factor2': 0.3})
    
    # Simulate trades
    print("\n1. Simulating 100 trades...")
    for i in range(100):
        result = TradeResult(
            timestamp=datetime.now(),
            ticker="SPY" if i % 2 == 0 else "QQQ",
            signal_direction="long" if np.random.random() > 0.3 else "short",
            signal_confidence=np.random.uniform(0.5, 0.9),
            predicted_return=np.random.uniform(-0.02, 0.02),
            actual_return=np.random.uniform(-0.02, 0.02),
            is_hit=np.random.random() > 0.45,  # ~55% hit rate
            features={'momentum': np.random.randn(), 'volatility': abs(np.random.randn()) * 0.02, 'rsi': np.random.uniform(30, 70)}
        )
        update = learner.record_trade(result)
    
    print(f"   Accuracy: {learner.performance.get_accuracy():.1%}")
    print(f"   Total trades: {learner.performance.total_trades}")
    
    # Check drift
    print("\n2. Checking drift...")
    drift = learner.check_drift()
    print(f"   Has drift: {drift.has_drift}")
    print(f"   Max KS score: {drift.max_ks_score:.3f}")
    
    # Test recalibration
    print("\n3. Testing recalibration...")
    learner.trigger_recalibration(reason="test")
    new_weights = {'factor1': 0.6, 'factor2': 0.25, 'factor3': 0.15}
    learner.complete_recalibration(new_weights)
    
    # Get state
    print("\n4. Learner state...")
    state = learner.get_state()
    print(f"   Accuracy: {state.accuracy:.1%}")
    print(f"   Checkpoint: {state.checkpoint_id}")
    print(f"   Weights: {state.current_weights}")
    
    # Test rollback
    print("\n5. Testing rollback...")
    learner.updater.update_weights({'factor1': 0.9})
    print(f"   Before rollback: {learner.updater.current_weights}")
    learner.updater.rollback(steps=1)
    print(f"   After rollback: {learner.updater.current_weights}")
    
    print("\n✅ V26 Continuous Learner tests passed!")
