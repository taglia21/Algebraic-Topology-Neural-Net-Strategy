"""
V26 Model Health Monitor
========================

Real-time model monitoring with auto-retrain triggers.

Key Features:
1. Real-time metrics: Sharpe, win rate, profit factor every 5 min
2. Drift alerts to Discord: feature shift, accuracy drop
3. Auto-retrain triggers: >10% accuracy drop, drift >48hr, weekly Saturday

Target: Detect degradation <24hr
"""

import logging
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict, field
from collections import deque
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ModelHealthConfig:
    """Configuration for model health monitoring."""
    
    # Metric calculation intervals
    metric_interval_minutes: float = 5.0     # Calculate metrics every 5 min
    
    # Accuracy thresholds
    baseline_accuracy: float = 0.55           # Expected accuracy
    accuracy_drop_threshold: float = 0.10     # Alert if accuracy drops >10%
    critical_accuracy_drop: float = 0.15      # Trigger retrain if >15% drop
    
    # Sharpe monitoring
    baseline_sharpe: float = 0.77            # V25 baseline Sharpe
    sharpe_warning_threshold: float = 0.50   # Alert if Sharpe < 0.5
    sharpe_critical_threshold: float = 0.20  # Retrain if Sharpe < 0.2
    
    # Win rate
    min_win_rate: float = 0.45               # Alert if win rate < 45%
    
    # Profit factor
    min_profit_factor: float = 1.0           # Alert if PF < 1.0
    
    # Drift monitoring
    drift_duration_warning_hours: float = 24.0   # Alert after 24hr drift
    drift_duration_retrain_hours: float = 48.0   # Retrain after 48hr drift
    
    # Auto-retrain schedule
    weekly_retrain_day: int = 5              # Saturday (0=Monday)
    weekly_retrain_hour: int = 2             # 2 AM UTC
    
    # Alert settings
    enable_discord_alerts: bool = True
    alert_cooldown_minutes: float = 30.0     # Min time between same alerts
    
    # Logging
    metrics_log_path: str = "logs/v26_model_health.jsonl"
    
    # Lookback windows
    sharpe_lookback_days: int = 30
    accuracy_lookback_trades: int = 100


@dataclass
class HealthMetrics:
    """Current model health metrics."""
    timestamp: datetime
    accuracy: float
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    avg_return: float
    max_drawdown: float
    trades_count: int
    is_healthy: bool
    health_score: float          # 0-100 composite score
    warnings: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'accuracy': round(self.accuracy, 4),
            'sharpe_ratio': round(self.sharpe_ratio, 4),
            'win_rate': round(self.win_rate, 4),
            'profit_factor': round(self.profit_factor, 4),
            'avg_return': round(self.avg_return, 6),
            'max_drawdown': round(self.max_drawdown, 4),
            'trades_count': self.trades_count,
            'is_healthy': self.is_healthy,
            'health_score': round(self.health_score, 1),
            'warnings': self.warnings
        }


class RetrainTrigger:
    """Reasons for model retraining."""
    ACCURACY_DROP = "accuracy_drop"
    SHARPE_DROP = "sharpe_drop"
    DRIFT_48H = "drift_48h"
    WEEKLY_SCHEDULED = "weekly_scheduled"
    MANUAL = "manual"


# =============================================================================
# METRIC CALCULATOR
# =============================================================================

class MetricCalculator:
    """Calculate trading performance metrics."""
    
    def __init__(self, config: ModelHealthConfig):
        self.config = config
        
        # Trade tracking
        self.trades: deque = deque(maxlen=1000)
        self.daily_returns: deque = deque(maxlen=252)  # 1 year
        self.predictions: deque = deque(maxlen=config.accuracy_lookback_trades)
        
        # Running stats
        self.peak_equity = 1.0
        self.current_equity = 1.0
        
    def record_trade(self, predicted_direction: str, actual_return: float,
                     confidence: float, timestamp: Optional[datetime] = None):
        """
        Record a trade for metric calculation.
        
        Args:
            predicted_direction: 'long' or 'short'
            actual_return: Actual return achieved
            confidence: Prediction confidence
            timestamp: Trade timestamp
        """
        ts = timestamp or datetime.now()
        
        # Determine if prediction was correct
        is_correct = (predicted_direction == 'long' and actual_return > 0) or \
                     (predicted_direction == 'short' and actual_return < 0)
        
        trade = {
            'timestamp': ts,
            'direction': predicted_direction,
            'return': actual_return,
            'confidence': confidence,
            'is_correct': is_correct
        }
        
        self.trades.append(trade)
        self.predictions.append(is_correct)
        
        # Update equity curve
        self.current_equity *= (1 + actual_return)
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity
    
    def record_daily_return(self, daily_return: float):
        """Record end-of-day return."""
        self.daily_returns.append(daily_return)
    
    def calculate_accuracy(self) -> float:
        """Calculate prediction accuracy."""
        if len(self.predictions) < 10:
            return 0.5  # Insufficient data
        
        return sum(self.predictions) / len(self.predictions)
    
    def calculate_sharpe(self) -> float:
        """Calculate Sharpe ratio from daily returns."""
        returns = list(self.daily_returns)[-self.config.sharpe_lookback_days:]
        
        if len(returns) < 10:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return < 1e-6:
            return 0.0
        
        # Annualized Sharpe
        return (mean_return * 252) / (std_return * np.sqrt(252))
    
    def calculate_win_rate(self) -> float:
        """Calculate win rate."""
        if len(self.trades) < 10:
            return 0.5
        
        wins = sum(1 for t in self.trades if t['return'] > 0)
        return wins / len(self.trades)
    
    def calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        if len(self.trades) < 10:
            return 1.0
        
        gross_profit = sum(t['return'] for t in self.trades if t['return'] > 0)
        gross_loss = abs(sum(t['return'] for t in self.trades if t['return'] < 0))
        
        if gross_loss < 1e-6:
            return float('inf') if gross_profit > 0 else 1.0
        
        return gross_profit / gross_loss
    
    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if self.peak_equity <= 0:
            return 0.0
        
        return (self.peak_equity - self.current_equity) / self.peak_equity
    
    def get_metrics(self) -> HealthMetrics:
        """Calculate all health metrics."""
        accuracy = self.calculate_accuracy()
        sharpe = self.calculate_sharpe()
        win_rate = self.calculate_win_rate()
        profit_factor = self.calculate_profit_factor()
        max_dd = self.calculate_max_drawdown()
        
        # Calculate average return
        if self.trades:
            avg_return = np.mean([t['return'] for t in self.trades])
        else:
            avg_return = 0.0
        
        # Determine warnings
        warnings = []
        
        if accuracy < self.config.baseline_accuracy * (1 - self.config.accuracy_drop_threshold):
            warnings.append(f"Accuracy drop: {accuracy:.1%}")
        
        if sharpe < self.config.sharpe_warning_threshold:
            warnings.append(f"Low Sharpe: {sharpe:.2f}")
        
        if win_rate < self.config.min_win_rate:
            warnings.append(f"Low win rate: {win_rate:.1%}")
        
        if profit_factor < self.config.min_profit_factor:
            warnings.append(f"Profit factor < 1: {profit_factor:.2f}")
        
        # Calculate health score (0-100)
        health_score = self._calculate_health_score(accuracy, sharpe, win_rate, profit_factor, max_dd)
        
        return HealthMetrics(
            timestamp=datetime.now(),
            accuracy=accuracy,
            sharpe_ratio=sharpe,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_return=avg_return,
            max_drawdown=max_dd,
            trades_count=len(self.trades),
            is_healthy=len(warnings) == 0,
            health_score=health_score,
            warnings=warnings
        )
    
    def _calculate_health_score(self, accuracy: float, sharpe: float,
                                win_rate: float, profit_factor: float,
                                max_dd: float) -> float:
        """Calculate composite health score (0-100)."""
        scores = []
        
        # Accuracy score (0-25)
        acc_score = min(25, (accuracy - 0.45) / 0.15 * 25)
        scores.append(max(0, acc_score))
        
        # Sharpe score (0-25)
        sharpe_score = min(25, (sharpe - 0.2) / 0.6 * 25)
        scores.append(max(0, sharpe_score))
        
        # Win rate score (0-25)
        wr_score = min(25, (win_rate - 0.40) / 0.20 * 25)
        scores.append(max(0, wr_score))
        
        # Drawdown score (0-25, inverse)
        dd_score = max(0, 25 - max_dd * 100)
        scores.append(dd_score)
        
        return sum(scores)


# =============================================================================
# MODEL HEALTH MONITOR
# =============================================================================

class ModelHealthMonitor:
    """
    V26 Model Health Monitor.
    
    Continuously monitors model performance and triggers alerts/retraining
    when degradation is detected.
    
    Usage:
        monitor = ModelHealthMonitor()
        monitor.set_discord_callback(discord_func)
        
        # Record trades
        monitor.record_trade('long', 0.02, 0.75)
        
        # Periodic check
        metrics = monitor.check_health()
        if metrics.needs_retrain:
            trigger_retrain(metrics.retrain_reason)
    """
    
    def __init__(self, config: Optional[ModelHealthConfig] = None):
        self.config = config or ModelHealthConfig()
        
        self.calculator = MetricCalculator(self.config)
        
        # Alert tracking
        self.last_alerts: Dict[str, datetime] = {}
        
        # Drift tracking
        self.drift_detected: bool = False
        self.drift_start: Optional[datetime] = None
        
        # Retrain tracking
        self.last_retrain: Optional[datetime] = None
        self.retrain_count: int = 0
        
        # Metric history
        self.metric_history: deque = deque(maxlen=1000)
        
        # Callbacks
        self.discord_callback: Optional[Callable] = None
        self.retrain_callback: Optional[Callable] = None
        
        # Logging
        self.log_path = Path(self.config.metrics_log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Background thread for periodic checks
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        
        logger.info("ModelHealthMonitor initialized")
    
    def set_discord_callback(self, callback: Callable):
        """Set Discord alert callback."""
        self.discord_callback = callback
    
    def set_retrain_callback(self, callback: Callable):
        """Set retrain trigger callback."""
        self.retrain_callback = callback
    
    def record_trade(self, predicted_direction: str, actual_return: float,
                     confidence: float = 0.5, timestamp: Optional[datetime] = None):
        """Record a trade for monitoring."""
        self.calculator.record_trade(predicted_direction, actual_return, confidence, timestamp)
    
    def record_daily_return(self, daily_return: float):
        """Record daily portfolio return."""
        self.calculator.record_daily_return(daily_return)
    
    def set_drift_status(self, has_drift: bool, drift_duration_hours: float = 0):
        """Update drift status from continuous learner."""
        if has_drift and not self.drift_detected:
            self.drift_detected = True
            self.drift_start = datetime.now() - timedelta(hours=drift_duration_hours)
        elif not has_drift:
            self.drift_detected = False
            self.drift_start = None
    
    def check_health(self) -> Dict[str, Any]:
        """
        Check model health and determine if action needed.
        
        Returns:
            Dict with metrics, needs_retrain, and retrain_reason
        """
        metrics = self.calculator.get_metrics()
        
        # Log metrics
        self._log_metrics(metrics)
        self.metric_history.append(metrics)
        
        # Check for retrain triggers
        needs_retrain = False
        retrain_reason = None
        
        # 1. Accuracy drop
        accuracy_drop = self.config.baseline_accuracy - metrics.accuracy
        if accuracy_drop > self.config.critical_accuracy_drop:
            needs_retrain = True
            retrain_reason = RetrainTrigger.ACCURACY_DROP
            self._send_alert("accuracy_critical", 
                           f"Critical accuracy drop: {metrics.accuracy:.1%} (baseline: {self.config.baseline_accuracy:.1%})")
        elif accuracy_drop > self.config.accuracy_drop_threshold:
            self._send_alert("accuracy_warning",
                           f"Accuracy drop: {metrics.accuracy:.1%}")
        
        # 2. Sharpe drop
        if metrics.sharpe_ratio < self.config.sharpe_critical_threshold:
            needs_retrain = True
            retrain_reason = RetrainTrigger.SHARPE_DROP
            self._send_alert("sharpe_critical",
                           f"Critical Sharpe drop: {metrics.sharpe_ratio:.2f}")
        elif metrics.sharpe_ratio < self.config.sharpe_warning_threshold:
            self._send_alert("sharpe_warning",
                           f"Low Sharpe: {metrics.sharpe_ratio:.2f}")
        
        # 3. Drift duration
        if self.drift_detected and self.drift_start:
            drift_hours = (datetime.now() - self.drift_start).total_seconds() / 3600
            
            if drift_hours > self.config.drift_duration_retrain_hours:
                needs_retrain = True
                retrain_reason = RetrainTrigger.DRIFT_48H
                self._send_alert("drift_critical",
                               f"Drift persisting {drift_hours:.1f} hours")
            elif drift_hours > self.config.drift_duration_warning_hours:
                self._send_alert("drift_warning",
                               f"Drift detected for {drift_hours:.1f} hours")
        
        # 4. Weekly scheduled retrain
        if self._is_weekly_retrain_time():
            needs_retrain = True
            retrain_reason = RetrainTrigger.WEEKLY_SCHEDULED
        
        # Trigger retrain if needed
        if needs_retrain and self.retrain_callback:
            try:
                self.retrain_callback(retrain_reason)
                self.last_retrain = datetime.now()
                self.retrain_count += 1
            except Exception as e:
                logger.error(f"Retrain callback failed: {e}")
        
        return {
            'metrics': metrics.to_dict(),
            'needs_retrain': needs_retrain,
            'retrain_reason': retrain_reason,
            'drift_detected': self.drift_detected,
            'drift_hours': (datetime.now() - self.drift_start).total_seconds() / 3600 if self.drift_start else 0
        }
    
    def _is_weekly_retrain_time(self) -> bool:
        """Check if it's time for weekly scheduled retrain."""
        now = datetime.now()
        
        # Check day and hour
        if now.weekday() != self.config.weekly_retrain_day:
            return False
        if now.hour != self.config.weekly_retrain_hour:
            return False
        
        # Check if already retrained this week
        if self.last_retrain:
            days_since = (now - self.last_retrain).days
            if days_since < 6:  # Less than 6 days since last retrain
                return False
        
        return True
    
    def _send_alert(self, alert_type: str, message: str):
        """Send alert if not in cooldown."""
        now = datetime.now()
        
        # Check cooldown
        if alert_type in self.last_alerts:
            elapsed = (now - self.last_alerts[alert_type]).total_seconds() / 60
            if elapsed < self.config.alert_cooldown_minutes:
                return
        
        self.last_alerts[alert_type] = now
        
        logger.warning(f"Alert [{alert_type}]: {message}")
        
        # Discord alert
        if self.config.enable_discord_alerts and self.discord_callback:
            colors = {
                'accuracy_warning': 0xFFAA00,
                'accuracy_critical': 0xFF0000,
                'sharpe_warning': 0xFFAA00,
                'sharpe_critical': 0xFF0000,
                'drift_warning': 0xFF6600,
                'drift_critical': 0xFF0000,
            }
            
            try:
                self.discord_callback(
                    f"⚠️ V26 Model Health Alert",
                    f"Type: {alert_type}\n{message}",
                    color=colors.get(alert_type, 0xFFAA00)
                )
            except Exception as e:
                logger.error(f"Discord alert failed: {e}")
    
    def _log_metrics(self, metrics: HealthMetrics):
        """Log metrics to file."""
        try:
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(metrics.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
    
    def start_background_monitoring(self, interval_seconds: float = 300):
        """Start background monitoring thread."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            logger.warning("Monitoring already running")
            return
        
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._monitor_thread.start()
        logger.info(f"Started background monitoring (interval: {interval_seconds}s)")
    
    def stop_background_monitoring(self):
        """Stop background monitoring thread."""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Stopped background monitoring")
    
    def _monitoring_loop(self, interval: float):
        """Background monitoring loop."""
        while not self._stop_event.is_set():
            try:
                self.check_health()
            except Exception as e:
                logger.error(f"Monitoring check failed: {e}")
            
            self._stop_event.wait(interval)
    
    def get_current_metrics(self) -> HealthMetrics:
        """Get current metrics without triggering actions."""
        return self.calculator.get_metrics()
    
    def get_metric_history(self, hours: float = 24) -> List[Dict]:
        """Get metric history for last N hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [
            m.to_dict() for m in self.metric_history
            if m.timestamp > cutoff
        ]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get monitor summary."""
        metrics = self.get_current_metrics()
        
        return {
            'health_score': metrics.health_score,
            'is_healthy': metrics.is_healthy,
            'accuracy': metrics.accuracy,
            'sharpe': metrics.sharpe_ratio,
            'win_rate': metrics.win_rate,
            'profit_factor': metrics.profit_factor,
            'max_drawdown': metrics.max_drawdown,
            'trades_count': metrics.trades_count,
            'warnings': metrics.warnings,
            'drift_detected': self.drift_detected,
            'drift_hours': (datetime.now() - self.drift_start).total_seconds() / 3600 if self.drift_start else 0,
            'retrain_count': self.retrain_count,
            'last_retrain': self.last_retrain.isoformat() if self.last_retrain else None
        }
    
    def trigger_manual_retrain(self):
        """Manually trigger a retrain."""
        logger.info("Manual retrain triggered")
        
        if self.retrain_callback:
            try:
                self.retrain_callback(RetrainTrigger.MANUAL)
                self.last_retrain = datetime.now()
                self.retrain_count += 1
            except Exception as e:
                logger.error(f"Retrain callback failed: {e}")


# =============================================================================
# FACTORY
# =============================================================================

def create_model_health_monitor() -> ModelHealthMonitor:
    """Factory function for V26 model health monitor."""
    config = ModelHealthConfig()
    return ModelHealthMonitor(config)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing V26 Model Health Monitor...")
    
    config = ModelHealthConfig()
    monitor = ModelHealthMonitor(config)
    
    # Simulate trades
    print("\n1. Recording simulated trades...")
    for i in range(100):
        direction = 'long' if np.random.random() > 0.4 else 'short'
        ret = np.random.normal(0.001, 0.02)  # Slight positive edge
        confidence = np.random.uniform(0.5, 0.9)
        monitor.record_trade(direction, ret, confidence)
    
    # Record daily returns
    print("\n2. Recording daily returns...")
    for _ in range(30):
        daily_ret = np.random.normal(0.001, 0.015)
        monitor.record_daily_return(daily_ret)
    
    # Check health
    print("\n3. Checking health...")
    result = monitor.check_health()
    print(f"   Metrics: {json.dumps(result['metrics'], indent=2)}")
    print(f"   Needs retrain: {result['needs_retrain']}")
    
    # Get summary
    print("\n4. Summary...")
    summary = monitor.get_summary()
    print(f"   Health score: {summary['health_score']:.1f}")
    print(f"   Is healthy: {summary['is_healthy']}")
    print(f"   Warnings: {summary['warnings']}")
    
    # Test drift status
    print("\n5. Testing drift status...")
    monitor.set_drift_status(True, drift_duration_hours=25)
    result = monitor.check_health()
    print(f"   Drift detected: {result['drift_detected']}")
    print(f"   Drift hours: {result['drift_hours']:.1f}")
    
    # Test with degraded performance
    print("\n6. Testing degraded performance...")
    for _ in range(50):
        # Negative edge
        ret = np.random.normal(-0.002, 0.02)
        monitor.record_trade('long', ret, 0.5)
    
    result = monitor.check_health()
    print(f"   Updated accuracy: {result['metrics']['accuracy']:.1%}")
    print(f"   Needs retrain: {result['needs_retrain']}")
    
    print("\n✅ V26 Model Health Monitor tests passed!")
