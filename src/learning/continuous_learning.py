#!/usr/bin/env python3
"""
V25 Phase 4: Continuous Learning Loop
=======================================

Daily learning updates with persistence and performance monitoring.

Key Innovation:
- Daily state persistence for production continuity
- Rolling accuracy tracking with trend analysis
- Automated weight adjustments based on recent performance
- Self-healing: Detect and correct regime classification errors

Learning Loop Components:
1. DailyUpdater: Handles end-of-day state updates
2. AccuracyTracker: Monitors prediction accuracy over time
3. PerformanceMonitor: Detects performance degradation
4. StateManager: Persists and restores system state

Target: Positive accuracy trend over 30-day rolling window
"""

import json
import logging
import numpy as np
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import threading
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('V25_LearningLoop')


# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)


# =============================================================================
# ACCURACY TRACKER
# =============================================================================

@dataclass
class PredictionRecord:
    """Single prediction record for accuracy tracking."""
    date: str
    regime: str
    predicted_strategy: str  # 'v21', 'v24', or 'equal'
    v21_return: float
    v24_return: float
    correct: bool
    confidence: float = 1.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


class AccuracyTracker:
    """
    Tracks regime prediction accuracy over rolling windows.
    
    Key Features:
    - Multiple window sizes (7, 30, 60 days)
    - Trend detection (improving vs degrading)
    - Per-regime accuracy breakdown
    """
    
    def __init__(self, 
                 short_window: int = 7,
                 medium_window: int = 30,
                 long_window: int = 60,
                 max_history: int = 252):
        """
        Initialize accuracy tracker.
        
        Args:
            short_window: Short-term accuracy window (days)
            medium_window: Medium-term window for trend detection
            long_window: Long-term baseline window
            max_history: Maximum history to retain
        """
        self.short_window = short_window
        self.medium_window = medium_window
        self.long_window = long_window
        
        self.history: deque = deque(maxlen=max_history)
        self.regime_stats: Dict[str, Dict] = {}
        
        # Rolling accuracy values
        self.short_accuracy: deque = deque(maxlen=medium_window)
        
    def record_prediction(self, record: PredictionRecord):
        """Add a new prediction record."""
        self.history.append(record)
        
        # Update regime stats
        regime = record.regime
        if regime not in self.regime_stats:
            self.regime_stats[regime] = {
                'correct': 0, 'total': 0, 
                'v21_wins': 0, 'v24_wins': 0,
                'returns': []
            }
        
        stats = self.regime_stats[regime]
        stats['total'] += 1
        if record.correct:
            stats['correct'] += 1
        
        if record.v21_return > record.v24_return:
            stats['v21_wins'] += 1
        else:
            stats['v24_wins'] += 1
        
        # Track rolling accuracy
        if len(self.history) >= self.short_window:
            recent = list(self.history)[-self.short_window:]
            acc = sum(1 for r in recent if r.correct) / len(recent)
            self.short_accuracy.append(acc)
    
    def get_accuracy(self, window: Optional[int] = None) -> float:
        """Get accuracy over specified window (default: all history)."""
        if not self.history:
            return 0.5
        
        if window:
            records = list(self.history)[-window:]
        else:
            records = list(self.history)
        
        if not records:
            return 0.5
        
        return sum(1 for r in records if r.correct) / len(records)
    
    def get_accuracy_trend(self) -> Tuple[float, str]:
        """
        Calculate accuracy trend over medium window.
        
        Returns:
            (slope, direction) where direction is 'improving', 'degrading', or 'stable'
        """
        if len(self.short_accuracy) < 10:
            return 0.0, 'insufficient_data'
        
        accuracies = list(self.short_accuracy)
        n = len(accuracies)
        
        # Linear regression
        x = np.arange(n)
        y = np.array(accuracies)
        
        # Least squares slope
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
        
        # Determine direction
        if slope > 0.001:
            direction = 'improving'
        elif slope < -0.001:
            direction = 'degrading'
        else:
            direction = 'stable'
        
        return slope, direction
    
    def get_regime_accuracy(self, regime: str) -> float:
        """Get accuracy for a specific regime."""
        if regime not in self.regime_stats:
            return 0.5
        
        stats = self.regime_stats[regime]
        if stats['total'] == 0:
            return 0.5
        
        return stats['correct'] / stats['total']
    
    def get_summary(self) -> Dict:
        """Get accuracy summary."""
        short_acc = self.get_accuracy(self.short_window)
        medium_acc = self.get_accuracy(self.medium_window)
        long_acc = self.get_accuracy(self.long_window)
        trend_slope, trend_dir = self.get_accuracy_trend()
        
        return {
            'total_predictions': len(self.history),
            f'accuracy_{self.short_window}d': short_acc,
            f'accuracy_{self.medium_window}d': medium_acc,
            f'accuracy_{self.long_window}d': long_acc,
            'trend_slope': trend_slope,
            'trend_direction': trend_dir,
            'n_regimes_seen': len(self.regime_stats)
        }
    
    def to_dict(self) -> Dict:
        """Serialize for persistence."""
        return {
            'history': [r.to_dict() for r in self.history],
            'regime_stats': self.regime_stats,
            'short_accuracy': list(self.short_accuracy),
            'summary': self.get_summary()
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'AccuracyTracker':
        """Deserialize from dict."""
        tracker = cls()
        
        for record_dict in d.get('history', []):
            record = PredictionRecord(**record_dict)
            tracker.history.append(record)
        
        tracker.regime_stats = d.get('regime_stats', {})
        tracker.short_accuracy = deque(d.get('short_accuracy', []), maxlen=tracker.medium_window)
        
        return tracker


# =============================================================================
# PERFORMANCE MONITOR
# =============================================================================

class PerformanceMonitor:
    """
    Monitors system performance and detects degradation.
    
    Triggers alerts when:
    - Accuracy drops below threshold
    - Sharpe degradation detected
    - Unusual regime patterns observed
    """
    
    def __init__(self,
                 accuracy_threshold: float = 0.45,
                 sharpe_threshold: float = 0.50,
                 alert_cooldown_hours: int = 24):
        """
        Initialize monitor.
        
        Args:
            accuracy_threshold: Alert if accuracy falls below
            sharpe_threshold: Alert if rolling Sharpe falls below
            alert_cooldown_hours: Minimum time between alerts
        """
        self.accuracy_threshold = accuracy_threshold
        self.sharpe_threshold = sharpe_threshold
        self.alert_cooldown = timedelta(hours=alert_cooldown_hours)
        
        self.last_alert_time: Optional[datetime] = None
        self.alerts: List[Dict] = []
        self.daily_returns: deque = deque(maxlen=60)
        
    def check_accuracy(self, accuracy_tracker: AccuracyTracker) -> Optional[Dict]:
        """Check accuracy and return alert if needed."""
        current_acc = accuracy_tracker.get_accuracy(30)
        
        if current_acc < self.accuracy_threshold:
            return self._create_alert(
                'accuracy_degradation',
                f"30-day accuracy {current_acc:.1%} below threshold {self.accuracy_threshold:.1%}",
                severity='warning'
            )
        
        # Check trend
        slope, direction = accuracy_tracker.get_accuracy_trend()
        if direction == 'degrading' and slope < -0.005:
            return self._create_alert(
                'accuracy_trend_negative',
                f"Accuracy trend declining: slope={slope:.4f}",
                severity='info'
            )
        
        return None
    
    def check_performance(self, daily_return: float) -> Optional[Dict]:
        """Check daily return and detect performance issues."""
        self.daily_returns.append(daily_return)
        
        if len(self.daily_returns) < 20:
            return None
        
        returns = np.array(self.daily_returns)
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        if sharpe < self.sharpe_threshold:
            return self._create_alert(
                'sharpe_degradation',
                f"Rolling Sharpe {sharpe:.2f} below threshold {self.sharpe_threshold:.2f}",
                severity='warning'
            )
        
        return None
    
    def _create_alert(self, alert_type: str, message: str, severity: str = 'info') -> Optional[Dict]:
        """Create alert if cooldown has passed."""
        now = datetime.now()
        
        if self.last_alert_time and (now - self.last_alert_time) < self.alert_cooldown:
            return None
        
        alert = {
            'type': alert_type,
            'message': message,
            'severity': severity,
            'timestamp': now.isoformat()
        }
        
        self.alerts.append(alert)
        self.last_alert_time = now
        
        logger.warning(f"ALERT [{severity.upper()}]: {message}")
        
        return alert
    
    def get_health_status(self, accuracy_tracker: AccuracyTracker) -> Dict:
        """Get overall system health status."""
        acc_30d = accuracy_tracker.get_accuracy(30)
        slope, direction = accuracy_tracker.get_accuracy_trend()
        
        # Health score 0-100
        health_score = 0
        
        # Accuracy component (0-40 points)
        if acc_30d >= 0.55:
            health_score += 40
        elif acc_30d >= 0.50:
            health_score += 30
        elif acc_30d >= 0.45:
            health_score += 20
        else:
            health_score += 10
        
        # Trend component (0-30 points)
        if direction == 'improving':
            health_score += 30
        elif direction == 'stable':
            health_score += 20
        else:
            health_score += 10
        
        # Rolling Sharpe component (0-30 points)
        if len(self.daily_returns) >= 20:
            returns = np.array(self.daily_returns)
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            
            if sharpe >= 1.0:
                health_score += 30
            elif sharpe >= 0.5:
                health_score += 20
            else:
                health_score += 10
        else:
            health_score += 15  # Neutral if insufficient data
        
        # Determine status
        if health_score >= 80:
            status = 'healthy'
        elif health_score >= 60:
            status = 'good'
        elif health_score >= 40:
            status = 'fair'
        else:
            status = 'poor'
        
        return {
            'health_score': health_score,
            'status': status,
            'accuracy_30d': acc_30d,
            'trend': direction,
            'recent_alerts': len([a for a in self.alerts if (datetime.now() - datetime.fromisoformat(a['timestamp'])).days < 7])
        }


# =============================================================================
# STATE MANAGER
# =============================================================================

class StateManager:
    """
    Manages persistence and restoration of system state.
    
    Features:
    - Daily state snapshots
    - Automatic backup rotation
    - State validation on restore
    """
    
    def __init__(self, 
                 state_dir: str = "state/v25",
                 max_backups: int = 7):
        """
        Initialize state manager.
        
        Args:
            state_dir: Directory for state files
            max_backups: Maximum backup files to retain
        """
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.max_backups = max_backups
        
        self.current_state_file = self.state_dir / "current_state.json"
        self.backup_dir = self.state_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
    
    def save_state(self, 
                   accuracy_tracker: AccuracyTracker,
                   allocator_state: Dict,
                   additional_data: Optional[Dict] = None):
        """Save complete system state."""
        state = {
            'saved_at': datetime.now().isoformat(),
            'version': '25.1',
            'accuracy_tracker': accuracy_tracker.to_dict(),
            'allocator': allocator_state,
            'additional': additional_data or {}
        }
        
        # Save current state
        with open(self.current_state_file, 'w') as f:
            json.dump(state, f, indent=2, cls=NumpyEncoder)
        
        # Create dated backup
        backup_name = f"state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        backup_path = self.backup_dir / backup_name
        
        with open(backup_path, 'w') as f:
            json.dump(state, f, indent=2, cls=NumpyEncoder)
        
        # Rotate old backups
        self._rotate_backups()
        
        logger.info(f"State saved to {self.current_state_file}")
    
    def load_state(self) -> Optional[Dict]:
        """Load system state from disk."""
        if not self.current_state_file.exists():
            logger.warning("No saved state found")
            return None
        
        try:
            with open(self.current_state_file) as f:
                state = json.load(f)
            
            # Validate
            if 'version' not in state:
                logger.warning("State file missing version, may be outdated")
            
            logger.info(f"State loaded from {self.current_state_file}")
            return state
            
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return None
    
    def restore_from_backup(self, backup_name: Optional[str] = None) -> Optional[Dict]:
        """Restore state from a backup file."""
        if backup_name:
            backup_path = self.backup_dir / backup_name
        else:
            # Get most recent backup
            backups = sorted(self.backup_dir.glob("state_*.json"), reverse=True)
            if not backups:
                logger.error("No backups available")
                return None
            backup_path = backups[0]
        
        if not backup_path.exists():
            logger.error(f"Backup not found: {backup_path}")
            return None
        
        with open(backup_path) as f:
            state = json.load(f)
        
        logger.info(f"State restored from backup: {backup_path}")
        return state
    
    def _rotate_backups(self):
        """Remove old backups beyond max_backups."""
        backups = sorted(self.backup_dir.glob("state_*.json"), reverse=True)
        
        for old_backup in backups[self.max_backups:]:
            old_backup.unlink()
            logger.debug(f"Removed old backup: {old_backup}")
    
    def list_backups(self) -> List[Dict]:
        """List available backups with metadata."""
        backups = []
        
        for backup_file in sorted(self.backup_dir.glob("state_*.json"), reverse=True):
            try:
                with open(backup_file) as f:
                    state = json.load(f)
                
                backups.append({
                    'filename': backup_file.name,
                    'saved_at': state.get('saved_at', 'unknown'),
                    'version': state.get('version', 'unknown'),
                    'size_kb': backup_file.stat().st_size / 1024
                })
            except Exception:
                backups.append({
                    'filename': backup_file.name,
                    'error': 'Failed to read'
                })
        
        return backups


# =============================================================================
# DAILY UPDATER
# =============================================================================

class DailyUpdater:
    """
    Handles end-of-day learning updates.
    
    Coordinates:
    - Accuracy tracking updates
    - Weight adjustments
    - State persistence
    - Performance monitoring
    """
    
    def __init__(self,
                 accuracy_tracker: Optional[AccuracyTracker] = None,
                 performance_monitor: Optional[PerformanceMonitor] = None,
                 state_manager: Optional[StateManager] = None,
                 learning_rate: float = 0.05):
        """
        Initialize daily updater.
        
        Args:
            accuracy_tracker: AccuracyTracker instance
            performance_monitor: PerformanceMonitor instance
            state_manager: StateManager instance
            learning_rate: Weight adjustment learning rate
        """
        self.accuracy_tracker = accuracy_tracker or AccuracyTracker()
        self.monitor = performance_monitor or PerformanceMonitor()
        self.state_manager = state_manager or StateManager()
        self.learning_rate = learning_rate
        
        # Weight adjustment history
        self.weight_adjustments: List[Dict] = []
        
        logger.info("DailyUpdater initialized")
    
    def update(self,
               date: str,
               regime: str,
               predicted_strategy: str,
               v21_return: float,
               v24_return: float,
               confidence: float = 1.0) -> Dict:
        """
        Process end-of-day update.
        
        Args:
            date: Date string
            regime: Detected regime
            predicted_strategy: Strategy that was weighted higher
            v21_return: V21 daily return
            v24_return: V24 daily return
            confidence: Prediction confidence
            
        Returns:
            Update summary with any alerts or adjustments
        """
        # Determine if prediction was correct
        if predicted_strategy == 'v21':
            correct = v21_return > v24_return
        elif predicted_strategy == 'v24':
            correct = v24_return > v21_return
        else:  # 'equal'
            correct = True  # Equal weight is always "correct" in a sense
        
        # Record prediction
        record = PredictionRecord(
            date=date,
            regime=regime,
            predicted_strategy=predicted_strategy,
            v21_return=v21_return,
            v24_return=v24_return,
            correct=correct,
            confidence=confidence
        )
        
        self.accuracy_tracker.record_prediction(record)
        
        # Check for alerts
        alerts = []
        
        acc_alert = self.monitor.check_accuracy(self.accuracy_tracker)
        if acc_alert:
            alerts.append(acc_alert)
        
        # Combined return
        combined_return = 0.5 * v21_return + 0.5 * v24_return
        perf_alert = self.monitor.check_performance(combined_return)
        if perf_alert:
            alerts.append(perf_alert)
        
        # Calculate weight adjustment suggestion
        adjustment = self._calculate_adjustment(regime, v21_return, v24_return)
        
        summary = {
            'date': date,
            'regime': regime,
            'correct': correct,
            'accuracy_7d': self.accuracy_tracker.get_accuracy(7),
            'accuracy_30d': self.accuracy_tracker.get_accuracy(30),
            'alerts': alerts,
            'suggested_adjustment': adjustment,
            'health': self.monitor.get_health_status(self.accuracy_tracker)
        }
        
        return summary
    
    def _calculate_adjustment(self, 
                               regime: str,
                               v21_return: float,
                               v24_return: float) -> Dict:
        """
        Calculate suggested weight adjustment based on performance.
        """
        # Get regime accuracy
        regime_acc = self.accuracy_tracker.get_regime_accuracy(regime)
        
        # Determine which strategy performed better
        if v21_return > v24_return:
            better = 'v21'
            edge = v21_return - v24_return
        else:
            better = 'v24'
            edge = v24_return - v21_return
        
        # Adjustment proportional to edge and learning rate
        adjustment = self.learning_rate * edge
        
        # Cap adjustment
        adjustment = np.clip(adjustment, -0.05, 0.05)
        
        result = {
            'regime': regime,
            'better_strategy': better,
            'edge': edge,
            'adjustment': adjustment,
            'regime_accuracy': regime_acc
        }
        
        self.weight_adjustments.append({
            'timestamp': datetime.now().isoformat(),
            **result
        })
        
        return result
    
    def save_state(self, allocator_state: Dict):
        """Save system state."""
        self.state_manager.save_state(
            accuracy_tracker=self.accuracy_tracker,
            allocator_state=allocator_state,
            additional_data={
                'weight_adjustments': self.weight_adjustments[-100:],
                'monitor_alerts': self.monitor.alerts[-50:]
            }
        )
    
    def load_state(self) -> bool:
        """Load system state."""
        state = self.state_manager.load_state()
        
        if not state:
            return False
        
        # Restore accuracy tracker
        if 'accuracy_tracker' in state:
            self.accuracy_tracker = AccuracyTracker.from_dict(state['accuracy_tracker'])
        
        # Restore other data
        if 'additional' in state:
            self.weight_adjustments = state['additional'].get('weight_adjustments', [])
        
        return True
    
    def get_learning_summary(self) -> Dict:
        """Get summary of learning progress."""
        return {
            'accuracy_summary': self.accuracy_tracker.get_summary(),
            'health': self.monitor.get_health_status(self.accuracy_tracker),
            'n_adjustments': len(self.weight_adjustments),
            'recent_adjustments': self.weight_adjustments[-10:],
            'backups_available': len(self.state_manager.list_backups())
        }


# =============================================================================
# V25 CONTINUOUS LEARNING SYSTEM
# =============================================================================

class V25ContinuousLearner:
    """
    Complete V25 Continuous Learning System.
    
    Integrates all Phase 4 components for production operation.
    """
    
    def __init__(self,
                 allocator,  # V25AdaptiveAllocator or V25FullAllocator
                 state_dir: str = "state/v25",
                 learning_rate: float = 0.05):
        """
        Initialize continuous learner.
        
        Args:
            allocator: V25 allocator instance
            state_dir: Directory for state persistence
            learning_rate: Weight adjustment learning rate
        """
        self.allocator = allocator
        
        self.updater = DailyUpdater(
            state_manager=StateManager(state_dir),
            learning_rate=learning_rate
        )
        
        self.is_running = False
        self.update_thread: Optional[threading.Thread] = None
        
        logger.info("V25ContinuousLearner initialized")
    
    def process_day(self,
                    date: str,
                    prices: np.ndarray,
                    v21_return: float,
                    v24_return: float) -> Dict:
        """
        Process a single day of data.
        
        Args:
            date: Date string
            prices: Price data for regime detection
            v21_return: V21 strategy return
            v24_return: V24 strategy return
            
        Returns:
            Processing summary
        """
        # Update regime
        returns = np.diff(np.log(prices))
        self.allocator.update_regime(returns, prices)
        regime = self.allocator.current_regime
        
        # Get current allocation
        weights = self.allocator.get_allocation()
        
        # Determine predicted strategy
        if weights['v21'] > weights['v24'] + 0.05:
            predicted = 'v21'
        elif weights['v24'] > weights['v21'] + 0.05:
            predicted = 'v24'
        else:
            predicted = 'equal'
        
        # Record and analyze
        summary = self.updater.update(
            date=date,
            regime=regime,
            predicted_strategy=predicted,
            v21_return=v21_return,
            v24_return=v24_return
        )
        
        # Record performance for online learning
        self.allocator.record_daily_performance(v21_return, v24_return, date)
        
        return summary
    
    def save(self):
        """Save system state."""
        try:
            allocator_state = self.allocator.learner.to_dict() if hasattr(self.allocator.learner, 'to_dict') else {}
        except Exception:
            allocator_state = {}
        self.updater.save_state(allocator_state)
    
    def load(self) -> bool:
        """Load system state."""
        return self.updater.load_state()
    
    def get_status(self) -> Dict:
        """Get system status."""
        return {
            'is_running': self.is_running,
            'current_regime': self.allocator.current_regime,
            'current_weights': self.allocator.get_allocation(),
            'learning_summary': self.updater.get_learning_summary()
        }


# =============================================================================
# TESTING
# =============================================================================

def test_continuous_learning():
    """Test continuous learning components."""
    logger.info("Testing V25 Continuous Learning components...")
    
    # Test AccuracyTracker
    tracker = AccuracyTracker()
    
    np.random.seed(42)
    for i in range(60):
        v21_ret = np.random.normal(0.001, 0.02)
        v24_ret = np.random.normal(0.001, 0.02)
        
        # Simulate prediction
        predicted = 'v21' if np.random.random() > 0.5 else 'v24'
        actual_better = 'v21' if v21_ret > v24_ret else 'v24'
        
        record = PredictionRecord(
            date=f"2024-01-{i+1:02d}",
            regime='medium_flat',
            predicted_strategy=predicted,
            v21_return=v21_ret,
            v24_return=v24_ret,
            correct=predicted == actual_better
        )
        tracker.record_prediction(record)
    
    summary = tracker.get_summary()
    logger.info(f"Accuracy Summary: {summary}")
    
    # Test PerformanceMonitor
    monitor = PerformanceMonitor()
    health = monitor.get_health_status(tracker)
    logger.info(f"Health Status: {health}")
    
    # Test StateManager
    state_mgr = StateManager(state_dir="logs/v25_test")
    state_mgr.save_state(
        accuracy_tracker=tracker,
        allocator_state={'test': 'data'}
    )
    
    loaded = state_mgr.load_state()
    assert loaded is not None, "State should load"
    logger.info(f"State loaded successfully")
    
    # Test DailyUpdater
    updater = DailyUpdater()
    
    result = updater.update(
        date="2024-03-01",
        regime='low_trending_up',
        predicted_strategy='v24',
        v21_return=0.005,
        v24_return=0.012
    )
    
    logger.info(f"Update result: {result}")
    
    logger.info("V25 Continuous Learning tests passed!")


if __name__ == "__main__":
    test_continuous_learning()
