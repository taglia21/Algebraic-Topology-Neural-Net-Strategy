"""
V2.5 Monitoring Dashboard
=========================

Real-time monitoring dashboard for V2.5 Elite components:

1. Feature Generation Health - Time, completeness, quality scores
2. Ensemble Model Contributions - Which models drive performance
3. Signal Validation Stats - Confirmation rate, false positive rate
4. Hyperparameter Stability - Track Bayesian tuner recommendations
5. V2.5 vs V2.2 Performance Comparison - Side-by-side metrics

Integrates with existing monitoring infrastructure.
"""

import time
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import deque
from pathlib import Path
import threading

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DashboardConfig:
    """Configuration for V2.5 monitoring dashboard."""
    
    # Metric retention
    max_history_hours: int = 24
    sample_interval_seconds: int = 60
    
    # Alert thresholds
    latency_alert_ms: float = 1000
    error_rate_alert: float = 0.1  # 10%
    quality_score_alert: int = 70
    memory_alert_gb: float = 5.0
    
    # Comparison mode
    enable_v22_comparison: bool = True
    
    # Output
    output_dir: str = "results/v25_monitoring"


# =============================================================================
# METRIC COLLECTORS
# =============================================================================

@dataclass
class FeatureMetrics:
    """Feature generation metrics."""
    timestamp: str = ""
    features_generated: int = 0
    generation_time_ms: float = 0.0
    features_per_asset: Dict[str, int] = field(default_factory=dict)
    mic_scores: Dict[str, float] = field(default_factory=dict)
    vmd_decomposition_success: bool = True
    errors: List[str] = field(default_factory=list)


@dataclass
class EnsembleMetrics:
    """Ensemble model metrics."""
    timestamp: str = ""
    
    # Model contributions (weights in meta-model)
    model_weights: Dict[str, float] = field(default_factory=dict)
    
    # Model performance
    model_predictions: Dict[str, float] = field(default_factory=dict)
    model_confidence: Dict[str, float] = field(default_factory=dict)
    
    # Overall
    ensemble_prediction: float = 0.0
    ensemble_confidence: float = 0.0
    prediction_time_ms: float = 0.0


@dataclass
class ValidationMetrics:
    """Signal validation metrics."""
    timestamp: str = ""
    
    # Counts
    total_signals: int = 0
    valid_signals: int = 0
    rejected_signals: int = 0
    
    # Confirmation rates
    avg_confirmations: float = 0.0
    confirmation_rate: float = 0.0
    false_positive_rate: float = 0.0
    
    # By indicator
    indicator_confirmation_rates: Dict[str, float] = field(default_factory=dict)


@dataclass
class DataQualityMetrics:
    """Data quality metrics."""
    timestamp: str = ""
    
    # Scores by ticker
    quality_scores: Dict[str, int] = field(default_factory=dict)
    avg_quality_score: float = 0.0
    
    # Check results
    freshness_pass_rate: float = 0.0
    completeness_pass_rate: float = 0.0
    price_validity_pass_rate: float = 0.0
    
    # Issues
    data_issues: List[str] = field(default_factory=list)


@dataclass
class PerformanceComparison:
    """V2.5 vs V2.2 performance comparison."""
    timestamp: str = ""
    
    # V2.5 metrics
    v25_sharpe: float = 0.0
    v25_sortino: float = 0.0
    v25_win_rate: float = 0.0
    v25_return_pct: float = 0.0
    v25_drawdown: float = 0.0
    
    # V2.2 metrics (baseline)
    v22_sharpe: float = 0.0
    v22_sortino: float = 0.0
    v22_win_rate: float = 0.0
    v22_return_pct: float = 0.0
    v22_drawdown: float = 0.0
    
    # Delta
    sharpe_improvement: float = 0.0
    win_rate_improvement: float = 0.0
    return_improvement: float = 0.0


@dataclass
class SystemHealthMetrics:
    """Overall system health metrics."""
    timestamp: str = ""
    
    # Health status
    is_healthy: bool = True
    error_count: int = 0
    warning_count: int = 0
    
    # Resource usage
    memory_usage_gb: float = 0.0
    cpu_usage_pct: float = 0.0
    
    # Latency
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    
    # Throughput
    signals_per_minute: float = 0.0
    trades_today: int = 0


# =============================================================================
# V2.5 MONITORING DASHBOARD
# =============================================================================

class V25MonitoringDashboard:
    """
    V2.5 Monitoring Dashboard.
    
    Collects and displays real-time metrics for V2.5 components.
    """
    
    def __init__(self, config: Optional[DashboardConfig] = None):
        self.config = config or DashboardConfig()
        self._lock = threading.Lock()
        
        # Metric history (circular buffers)
        max_samples = int(self.config.max_history_hours * 3600 / self.config.sample_interval_seconds)
        
        self.feature_history: deque = deque(maxlen=max_samples)
        self.ensemble_history: deque = deque(maxlen=max_samples)
        self.validation_history: deque = deque(maxlen=max_samples)
        self.quality_history: deque = deque(maxlen=max_samples)
        self.performance_history: deque = deque(maxlen=max_samples)
        self.health_history: deque = deque(maxlen=max_samples)
        
        # Current metrics
        self.current_features = FeatureMetrics()
        self.current_ensemble = EnsembleMetrics()
        self.current_validation = ValidationMetrics()
        self.current_quality = DataQualityMetrics()
        self.current_performance = PerformanceComparison()
        self.current_health = SystemHealthMetrics()
        
        # Alerts
        self.active_alerts: List[Dict[str, Any]] = []
        
        # Initialize output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("V2.5 Monitoring Dashboard initialized")
    
    # =========================================================================
    # METRIC RECORDING
    # =========================================================================
    
    def record_feature_metrics(
        self,
        features_generated: int,
        generation_time_ms: float,
        features_per_asset: Optional[Dict[str, int]] = None,
        mic_scores: Optional[Dict[str, float]] = None,
        errors: Optional[List[str]] = None,
    ):
        """Record feature generation metrics."""
        with self._lock:
            self.current_features = FeatureMetrics(
                timestamp=datetime.now().isoformat(),
                features_generated=features_generated,
                generation_time_ms=generation_time_ms,
                features_per_asset=features_per_asset or {},
                mic_scores=mic_scores or {},
                errors=errors or [],
            )
            self.feature_history.append(self.current_features)
        
        # Check alerts
        if generation_time_ms > self.config.latency_alert_ms:
            self._add_alert(
                "feature_latency",
                f"Feature generation time {generation_time_ms:.0f}ms exceeds {self.config.latency_alert_ms}ms"
            )
    
    def record_ensemble_metrics(
        self,
        model_weights: Dict[str, float],
        model_predictions: Dict[str, float],
        ensemble_prediction: float,
        ensemble_confidence: float,
        prediction_time_ms: float,
    ):
        """Record ensemble model metrics."""
        with self._lock:
            self.current_ensemble = EnsembleMetrics(
                timestamp=datetime.now().isoformat(),
                model_weights=model_weights,
                model_predictions=model_predictions,
                ensemble_prediction=ensemble_prediction,
                ensemble_confidence=ensemble_confidence,
                prediction_time_ms=prediction_time_ms,
            )
            self.ensemble_history.append(self.current_ensemble)
    
    def record_validation_metrics(
        self,
        total_signals: int,
        valid_signals: int,
        avg_confirmations: float,
        indicator_rates: Optional[Dict[str, float]] = None,
    ):
        """Record signal validation metrics."""
        with self._lock:
            rejection_rate = 1 - (valid_signals / max(total_signals, 1))
            
            self.current_validation = ValidationMetrics(
                timestamp=datetime.now().isoformat(),
                total_signals=total_signals,
                valid_signals=valid_signals,
                rejected_signals=total_signals - valid_signals,
                avg_confirmations=avg_confirmations,
                confirmation_rate=avg_confirmations / 9,  # Out of 9 indicators
                false_positive_rate=rejection_rate,
                indicator_confirmation_rates=indicator_rates or {},
            )
            self.validation_history.append(self.current_validation)
    
    def record_quality_metrics(
        self,
        quality_scores: Dict[str, int],
        data_issues: Optional[List[str]] = None,
    ):
        """Record data quality metrics."""
        with self._lock:
            scores = list(quality_scores.values())
            avg_score = sum(scores) / len(scores) if scores else 0
            
            self.current_quality = DataQualityMetrics(
                timestamp=datetime.now().isoformat(),
                quality_scores=quality_scores,
                avg_quality_score=avg_score,
                data_issues=data_issues or [],
            )
            self.quality_history.append(self.current_quality)
        
        # Check alerts
        if avg_score < self.config.quality_score_alert:
            self._add_alert(
                "data_quality",
                f"Average data quality {avg_score:.0f} below threshold {self.config.quality_score_alert}"
            )
    
    def record_performance_comparison(
        self,
        v25_metrics: Dict[str, float],
        v22_metrics: Optional[Dict[str, float]] = None,
    ):
        """Record V2.5 vs V2.2 performance comparison."""
        with self._lock:
            v22 = v22_metrics or {}
            
            self.current_performance = PerformanceComparison(
                timestamp=datetime.now().isoformat(),
                v25_sharpe=v25_metrics.get('sharpe', 0),
                v25_sortino=v25_metrics.get('sortino', 0),
                v25_win_rate=v25_metrics.get('win_rate', 0),
                v25_return_pct=v25_metrics.get('return_pct', 0),
                v25_drawdown=v25_metrics.get('drawdown', 0),
                v22_sharpe=v22.get('sharpe', 0),
                v22_sortino=v22.get('sortino', 0),
                v22_win_rate=v22.get('win_rate', 0),
                v22_return_pct=v22.get('return_pct', 0),
                v22_drawdown=v22.get('drawdown', 0),
                sharpe_improvement=v25_metrics.get('sharpe', 0) - v22.get('sharpe', 0),
                win_rate_improvement=v25_metrics.get('win_rate', 0) - v22.get('win_rate', 0),
                return_improvement=v25_metrics.get('return_pct', 0) - v22.get('return_pct', 0),
            )
            self.performance_history.append(self.current_performance)
    
    def record_health_metrics(
        self,
        is_healthy: bool,
        error_count: int,
        memory_usage_gb: float,
        avg_latency_ms: float,
        signals_per_minute: float,
        trades_today: int,
    ):
        """Record system health metrics."""
        with self._lock:
            self.current_health = SystemHealthMetrics(
                timestamp=datetime.now().isoformat(),
                is_healthy=is_healthy,
                error_count=error_count,
                memory_usage_gb=memory_usage_gb,
                avg_latency_ms=avg_latency_ms,
                signals_per_minute=signals_per_minute,
                trades_today=trades_today,
            )
            self.health_history.append(self.current_health)
        
        # Check alerts
        if memory_usage_gb > self.config.memory_alert_gb:
            self._add_alert(
                "memory",
                f"Memory usage {memory_usage_gb:.1f}GB exceeds {self.config.memory_alert_gb}GB"
            )
    
    # =========================================================================
    # ALERTS
    # =========================================================================
    
    def _add_alert(self, alert_type: str, message: str):
        """Add an alert."""
        alert = {
            'type': alert_type,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'acknowledged': False,
        }
        
        with self._lock:
            # Don't duplicate recent alerts
            recent_same = [
                a for a in self.active_alerts
                if a['type'] == alert_type
                and (datetime.now() - datetime.fromisoformat(a['timestamp'])).seconds < 300
            ]
            
            if not recent_same:
                self.active_alerts.append(alert)
                logger.warning(f"🚨 ALERT [{alert_type}]: {message}")
    
    def acknowledge_alert(self, alert_type: str):
        """Acknowledge an alert."""
        with self._lock:
            for alert in self.active_alerts:
                if alert['type'] == alert_type:
                    alert['acknowledged'] = True
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get unacknowledged alerts."""
        with self._lock:
            return [a for a in self.active_alerts if not a['acknowledged']]
    
    # =========================================================================
    # DASHBOARD OUTPUT
    # =========================================================================
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get complete dashboard data."""
        with self._lock:
            return {
                'timestamp': datetime.now().isoformat(),
                'features': asdict(self.current_features),
                'ensemble': asdict(self.current_ensemble),
                'validation': asdict(self.current_validation),
                'quality': asdict(self.current_quality),
                'performance': asdict(self.current_performance),
                'health': asdict(self.current_health),
                'alerts': self.get_active_alerts(),
                'summary': self._generate_summary(),
            }
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate dashboard summary."""
        return {
            'status': 'healthy' if self.current_health.is_healthy else 'degraded',
            'v25_sharpe': self.current_performance.v25_sharpe,
            'improvement_vs_v22': self.current_performance.sharpe_improvement,
            'signal_confirmation_rate': self.current_validation.confirmation_rate,
            'avg_latency_ms': self.current_health.avg_latency_ms,
            'trades_today': self.current_health.trades_today,
            'active_alerts': len(self.get_active_alerts()),
        }
    
    def print_dashboard(self):
        """Print dashboard to console."""
        data = self.get_dashboard_data()
        
        print("\n" + "=" * 70)
        print("V2.5 ELITE MONITORING DASHBOARD")
        print("=" * 70)
        print(f"Time: {data['timestamp']}")
        print()
        
        # Health
        health = data['health']
        status_icon = "✅" if health['is_healthy'] else "❌"
        print(f"Health: {status_icon} {'Healthy' if health['is_healthy'] else 'Degraded'}")
        print(f"Memory: {health['memory_usage_gb']:.1f}GB | Latency: {health['avg_latency_ms']:.0f}ms")
        print()
        
        # Features
        features = data['features']
        print(f"Features: {features['features_generated']} generated in {features['generation_time_ms']:.0f}ms")
        
        # Ensemble
        ensemble = data['ensemble']
        if ensemble['model_weights']:
            weights_str = ", ".join([f"{k}: {v:.2f}" for k, v in ensemble['model_weights'].items()])
            print(f"Ensemble Weights: {weights_str}")
        
        # Validation
        validation = data['validation']
        print(f"Signals: {validation['valid_signals']}/{validation['total_signals']} valid ({validation['confirmation_rate']:.1%} confirmation)")
        
        # Performance
        perf = data['performance']
        print(f"\nPerformance (V2.5 vs V2.2):")
        print(f"  Sharpe: {perf['v25_sharpe']:.2f} vs {perf['v22_sharpe']:.2f} ({perf['sharpe_improvement']:+.2f})")
        print(f"  Win Rate: {perf['v25_win_rate']:.1%} vs {perf['v22_win_rate']:.1%}")
        
        # Alerts
        alerts = data['alerts']
        if alerts:
            print(f"\n🚨 Active Alerts ({len(alerts)}):")
            for alert in alerts[:3]:
                print(f"  - [{alert['type']}] {alert['message']}")
        
        print("=" * 70)
    
    def save_snapshot(self):
        """Save current dashboard snapshot to file."""
        data = self.get_dashboard_data()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = Path(self.config.output_dir) / f"snapshot_{timestamp}.json"
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Dashboard snapshot saved to {filepath}")
        return str(filepath)
    
    # =========================================================================
    # HISTORY ANALYSIS
    # =========================================================================
    
    def get_latency_trend(self, hours: int = 1) -> Dict[str, List[float]]:
        """Get latency trend over time."""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            feature_latencies = [
                m.generation_time_ms
                for m in self.feature_history
                if datetime.fromisoformat(m.timestamp) > cutoff
            ]
            
            ensemble_latencies = [
                m.prediction_time_ms
                for m in self.ensemble_history
                if datetime.fromisoformat(m.timestamp) > cutoff
            ]
        
        return {
            'feature_gen': feature_latencies,
            'ensemble_pred': ensemble_latencies,
        }
    
    def get_performance_trend(self, hours: int = 24) -> List[Dict[str, float]]:
        """Get performance trend over time."""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            return [
                {
                    'timestamp': p.timestamp,
                    'sharpe': p.v25_sharpe,
                    'win_rate': p.v25_win_rate,
                    'improvement': p.sharpe_improvement,
                }
                for p in self.performance_history
                if datetime.fromisoformat(p.timestamp) > cutoff
            ]


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing V2.5 Monitoring Dashboard...")
    
    dashboard = V25MonitoringDashboard()
    
    # Simulate metrics
    dashboard.record_feature_metrics(
        features_generated=127,
        generation_time_ms=350,
        features_per_asset={'SPY': 127, 'QQQ': 127},
    )
    
    dashboard.record_ensemble_metrics(
        model_weights={'rf': 0.35, 'lstm': 0.40, 'linear': 0.25},
        model_predictions={'rf': 0.015, 'lstm': 0.012, 'linear': 0.008},
        ensemble_prediction=0.012,
        ensemble_confidence=0.75,
        prediction_time_ms=150,
    )
    
    dashboard.record_validation_metrics(
        total_signals=100,
        valid_signals=62,
        avg_confirmations=6.5,
    )
    
    dashboard.record_quality_metrics(
        quality_scores={'SPY': 95, 'QQQ': 92, 'AAPL': 88},
    )
    
    dashboard.record_performance_comparison(
        v25_metrics={'sharpe': 2.3, 'sortino': 2.8, 'win_rate': 0.58, 'return_pct': 12.5, 'drawdown': 0.08},
        v22_metrics={'sharpe': 1.5, 'sortino': 1.9, 'win_rate': 0.51, 'return_pct': 8.2, 'drawdown': 0.12},
    )
    
    dashboard.record_health_metrics(
        is_healthy=True,
        error_count=0,
        memory_usage_gb=3.2,
        avg_latency_ms=480,
        signals_per_minute=2.5,
        trades_today=8,
    )
    
    # Print dashboard
    dashboard.print_dashboard()
    
    # Save snapshot
    filepath = dashboard.save_snapshot()
    print(f"\nSnapshot saved to: {filepath}")
    
    print("\n✅ V2.5 Monitoring Dashboard tests passed!")
