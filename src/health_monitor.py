#!/usr/bin/env python3
"""System Health Monitor - Production-grade health monitoring and alerting.

This module implements heartbeat mechanisms, system component monitoring,
automatic failsafes, and alerting for a trading bot system. Ensures system
reliability and provides early warning of failures.

Author: Agent 2 (System Health Specialist)
Created: 2026-02-02
"""

import os
import sys
import time
import logging
import threading
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import requests
from collections import deque

# Configure logging
logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """Types of system components to monitor."""
    API_CONNECTION = "api_connection"
    DATA_FEED = "data_feed"
    POSITION_TRACKER = "position_tracker"
    MODEL_INFERENCE = "model_inference"
    RISK_MANAGER = "risk_manager"
    DATABASE = "database"
    CUSTOM = "custom"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    component: str
    component_type: ComponentType
    status: HealthStatus
    timestamp: datetime
    message: str
    latency_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'component': self.component,
            'component_type': self.component_type.value,
            'status': self.status.value,
            'timestamp': self.timestamp.isoformat(),
            'message': self.message,
            'latency_ms': self.latency_ms,
            'metadata': self.metadata
        }


class HeartbeatService:
    """Heartbeat service that periodically signals system is alive.
    
    Sends regular heartbeat signals and tracks missed heartbeats
    to detect system freezes or failures.
    """
    
    def __init__(self, interval_seconds: int = 60, alert_callback: Optional[Callable] = None):
        """Initialize heartbeat service.
        
        Args:
            interval_seconds: Seconds between heartbeats
            alert_callback: Function to call on missed heartbeats
        """
        self.interval_seconds = interval_seconds
        self.alert_callback = alert_callback
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_beat: Optional[datetime] = None
        self._missed_beats = 0
        self._beat_count = 0
        logger.info(f"HeartbeatService initialized (interval: {interval_seconds}s)")
    
    def start(self):
        """Start the heartbeat service."""
        if self._running:
            logger.warning("Heartbeat already running")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._thread.start()
        logger.info("Heartbeat service started")
    
    def stop(self):
        """Stop the heartbeat service."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=self.interval_seconds + 5)
        logger.info("Heartbeat service stopped")
    
    def _heartbeat_loop(self):
        """Main heartbeat loop."""
        while self._running:
            try:
                self._beat()
                time.sleep(self.interval_seconds)
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                if self.alert_callback:
                    self.alert_callback("heartbeat_error", str(e))
    
    def _beat(self):
        """Perform a single heartbeat."""
        now = datetime.now()
        self._last_beat = now
        self._beat_count += 1
        logger.debug(f"Heartbeat #{self._beat_count} at {now.strftime('%H:%M:%S')}")
    
    def is_alive(self, max_age_seconds: int = None) -> bool:
        """Check if system is showing signs of life.
        
        Args:
            max_age_seconds: Maximum age of last heartbeat (default: 2x interval)
            
        Returns:
            True if recent heartbeat detected
        """
        if not self._last_beat:
            return False
        
        max_age = max_age_seconds or (self.interval_seconds * 2)
        age = (datetime.now() - self._last_beat).total_seconds()
        return age <= max_age
    
    def get_stats(self) -> Dict[str, Any]:
        """Get heartbeat statistics."""
        return {
            'running': self._running,
            'beat_count': self._beat_count,
            'last_beat': self._last_beat.isoformat() if self._last_beat else None,
            'missed_beats': self._missed_beats,
            'is_alive': self.is_alive()
        }


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance.
    
    Prevents cascading failures by opening the circuit after
    consecutive failures, giving the system time to recover.
    """
    
    def __init__(self, failure_threshold: int = 3, reset_timeout_seconds: int = 60):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of consecutive failures before opening
            reset_timeout_seconds: Seconds before attempting reset
        """
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout_seconds
        self.failure_count = 0
        self.is_open = False
        self.last_failure_time: Optional[datetime] = None
        logger.info(f"CircuitBreaker initialized (threshold: {failure_threshold})")
    
    def record_success(self):
        """Record a successful operation."""
        self.failure_count = 0
        self.is_open = False
        logger.debug("Circuit breaker: Success recorded, circuit closed")
    
    def record_failure(self) -> bool:
        """Record a failed operation.
        
        Returns:
            True if circuit is now open
        """
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.is_open = True
            logger.warning(
                f"Circuit breaker OPENED after {self.failure_count} consecutive failures"
            )
            return True
        
        logger.debug(f"Circuit breaker: Failure {self.failure_count}/{self.failure_threshold}")
        return False
    
    def can_attempt(self) -> bool:
        """Check if operation can be attempted.
        
        Returns:
            True if circuit is closed or reset timeout has passed
        """
        if not self.is_open:
            return True
        
        # Check if reset timeout has passed
        if self.last_failure_time:
            time_since_failure = (datetime.now() - self.last_failure_time).total_seconds()
            if time_since_failure >= self.reset_timeout:
                logger.info("Circuit breaker: Attempting reset after timeout")
                self.is_open = False
                self.failure_count = 0
                return True
        
        return False
    
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state."""
        return {
            'is_open': self.is_open,
            'failure_count': self.failure_count,
            'can_attempt': self.can_attempt(),
            'last_failure': self.last_failure_time.isoformat() if self.last_failure_time else None
        }


class AlertManager:
    """Manages alerts and notifications.
    
    Sends alerts via multiple channels (Discord webhook, logging, etc.)
    when system health issues are detected.
    """
    
    def __init__(self, discord_webhook_url: Optional[str] = None):
        """Initialize alert manager.
        
        Args:
            discord_webhook_url: Discord webhook URL for alerts
        """
        self.discord_webhook_url = discord_webhook_url or os.getenv('DISCORD_WEBHOOK_URL')
        self.alert_history: deque = deque(maxlen=100)
        logger.info("AlertManager initialized")
    
    def send_alert(self, severity: str, component: str, message: str, metadata: Dict[str, Any] = None):
        """Send an alert.
        
        Args:
            severity: Alert severity (info, warning, error, critical)
            component: Component that triggered alert
            message: Alert message
            metadata: Additional metadata
        """
        alert = {
            'timestamp': datetime.now().isoformat(),
            'severity': severity,
            'component': component,
            'message': message,
            'metadata': metadata or {}
        }
        
        self.alert_history.append(alert)
        
        # Log the alert
        log_func = getattr(logger, severity.lower(), logger.info)
        log_func(f"[ALERT] {component}: {message}")
        
        # Send to Discord if configured
        if self.discord_webhook_url and severity in ['error', 'critical']:
            self._send_discord_alert(alert)
    
    def _send_discord_alert(self, alert: Dict[str, Any]):
        """Send alert to Discord webhook.
        
        Args:
            alert: Alert dictionary
        """
        try:
            emoji_map = {
                'info': '\U00002139',  # ℹ️
                'warning': '\U000026A0',  # ⚠️
                'error': '\U0000274C',  # ❌
                'critical': '\U0001F6A8'  # 🚨
            }
            
            emoji = emoji_map.get(alert['severity'], '\U0001F4E2')
            
            payload = {
                'content': f"{emoji} **{alert['severity'].upper()}** - {alert['component']}",
                'embeds': [{
                    'title': 'Health Monitor Alert',
                    'description': alert['message'],
                    'color': self._get_color_for_severity(alert['severity']),
                    'timestamp': alert['timestamp'],
                    'fields': [
                        {'name': 'Component', 'value': alert['component'], 'inline': True},
                        {'name': 'Severity', 'value': alert['severity'].upper(), 'inline': True}
                    ]
                }]
            }
            
            response = requests.post(
                self.discord_webhook_url,
                json=payload,
                timeout=5
            )
            
            if response.status_code != 204:
                logger.warning(f"Discord webhook returned {response.status_code}")
        
        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")
    
    def _get_color_for_severity(self, severity: str) -> int:
        """Get Discord embed color for severity."""
        colors = {
            'info': 3447003,      # Blue
            'warning': 16776960,  # Yellow
            'error': 15158332,    # Red
            'critical': 10038562  # Dark Red
        }
        return colors.get(severity, 8421504)  # Gray default
    
    def get_recent_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent alerts.
        
        Args:
            limit: Maximum number of alerts to return
            
        Returns:
            List of recent alerts
        """
        return list(self.alert_history)[-limit:]


class HealthMonitor:
    """Main health monitoring system.
    
    Coordinates heartbeat, component monitoring, circuit breakers,
    and alerting to provide comprehensive system health oversight.
    """
    
    def __init__(self,
                 heartbeat_interval: int = 60,
                 check_interval: int = 30,
                 discord_webhook: Optional[str] = None):
        """Initialize health monitor.
        
        Args:
            heartbeat_interval: Seconds between heartbeats
            check_interval: Seconds between health checks
            discord_webhook: Discord webhook URL
        """
        self.heartbeat = HeartbeatService(
            interval_seconds=heartbeat_interval,
            alert_callback=self._on_heartbeat_error
        )
        self.alert_manager = AlertManager(discord_webhook)
        self.check_interval = check_interval
        
        # Component tracking
        self.components: Dict[str, Callable] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.last_check_results: Dict[str, HealthCheckResult] = {}
        
        # Monitoring thread
        self._monitor_running = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        logger.info("HealthMonitor initialized")
    
    def register_component(self,
                          name: str,
                          check_function: Callable[[], HealthCheckResult],
                          component_type: ComponentType = ComponentType.CUSTOM,
                          enable_circuit_breaker: bool = True):
        """Register a component for monitoring.
        
        Args:
            name: Component name
            check_function: Function that performs health check
            component_type: Type of component
            enable_circuit_breaker: Whether to use circuit breaker
        """
        self.components[name] = check_function
        
        if enable_circuit_breaker:
            self.circuit_breakers[name] = CircuitBreaker()
        
        logger.info(f"Registered component: {name} ({component_type.value})")
    
    def start(self):
        """Start the health monitoring system."""
        logger.info("Starting health monitoring system...")
        
        # Start heartbeat
        self.heartbeat.start()
        
        # Start component monitoring
        self._monitor_running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        logger.info("Health monitoring system started")
    
    def stop(self):
        """Stop the health monitoring system."""
        logger.info("Stopping health monitoring system...")
        
        self._monitor_running = False
        self.heartbeat.stop()
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=self.check_interval + 5)
        
        logger.info("Health monitoring system stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._monitor_running:
            try:
                self._check_all_components()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}\n{traceback.format_exc()}")
                self.alert_manager.send_alert('error', 'monitor_loop', str(e))
    
    def _check_all_components(self):
        """Check health of all registered components."""
        for name, check_func in self.components.items():
            # Check circuit breaker
            if name in self.circuit_breakers:
                if not self.circuit_breakers[name].can_attempt():
                    logger.debug(f"Skipping {name} - circuit breaker open")
                    continue
            
            # Perform health check
            try:
                result = check_func()
                self.last_check_results[name] = result
                
                # Update circuit breaker
                if name in self.circuit_breakers:
                    if result.status == HealthStatus.HEALTHY:
                        self.circuit_breakers[name].record_success()
                    else:
                        breaker_opened = self.circuit_breakers[name].record_failure()
                        if breaker_opened:
                            self.alert_manager.send_alert(
                                'critical',
                                name,
                                f"Circuit breaker opened for {name}"
                            )
                
                # Send alerts for unhealthy components
                if result.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                    severity = 'critical' if result.status == HealthStatus.CRITICAL else 'error'
                    self.alert_manager.send_alert(
                        severity,
                        name,
                        result.message,
                        result.metadata
                    )
            
            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                if name in self.circuit_breakers:
                    self.circuit_breakers[name].record_failure()
    
    def _on_heartbeat_error(self, error_type: str, message: str):
        """Callback for heartbeat errors."""
        self.alert_manager.send_alert('critical', 'heartbeat', f"{error_type}: {message}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status.
        
        Returns:
            Dictionary with system health information
        """
        component_statuses = {}
        worst_status = HealthStatus.HEALTHY
        
        for name, result in self.last_check_results.items():
            component_statuses[name] = result.to_dict()
            
            # Track worst status
            status_priority = {
                HealthStatus.HEALTHY: 0,
                HealthStatus.DEGRADED: 1,
                HealthStatus.UNHEALTHY: 2,
                HealthStatus.CRITICAL: 3,
                HealthStatus.UNKNOWN: 4
            }
            
            if status_priority[result.status] > status_priority[worst_status]:
                worst_status = result.status
        
        return {
            'overall_status': worst_status.value,
            'heartbeat': self.heartbeat.get_stats(),
            'components': component_statuses,
            'circuit_breakers': {name: cb.get_state() for name, cb in self.circuit_breakers.items()},
            'recent_alerts': self.alert_manager.get_recent_alerts(5)
        }


# Convenience function
def initialize_health_monitor(heartbeat_interval: int = 60,
                             check_interval: int = 30,
                             discord_webhook: Optional[str] = None) -> HealthMonitor:
    """Initialize health monitoring system.
    
    Args:
        heartbeat_interval: Seconds between heartbeats
        check_interval: Seconds between health checks
        discord_webhook: Discord webhook URL
        
    Returns:
        Configured HealthMonitor instance
    """
    return HealthMonitor(
        heartbeat_interval=heartbeat_interval,
        check_interval=check_interval,
        discord_webhook=discord_webhook
    )


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Health Monitor...")
    
    # Initialize monitor
    monitor = initialize_health_monitor(heartbeat_interval=5, check_interval=10)
    
    # Register a test component
    def check_test_component() -> HealthCheckResult:
        """Example health check."""
        import random
        status = random.choice([HealthStatus.HEALTHY, HealthStatus.DEGRADED])
        return HealthCheckResult(
            component="test_api",
            component_type=ComponentType.API_CONNECTION,
            status=status,
            timestamp=datetime.now(),
            message=f"API is {status.value}",
            latency_ms=random.uniform(10, 100)
        )
    
    monitor.register_component(
        "test_api",
        check_test_component,
        ComponentType.API_CONNECTION
    )
    
    # Start monitoring
    monitor.start()
    
    print("\nMonitoring for 30 seconds...")
    time.sleep(30)
    
    # Check system health
    health = monitor.get_system_health()
    print(f"\nSystem Health:")
    print(f"  Overall: {health['overall_status']}")
    print(f"  Heartbeat running: {health['heartbeat']['running']}")
    print(f"  Heartbeat count: {health['heartbeat']['beat_count']}")
    
    # Stop monitoring
    monitor.stop()
    
    print("\nHealth Monitor test complete!")
