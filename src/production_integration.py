"""
Production Integration Module - Orchestrates All Trading System Components.

This module provides a unified API to integrate:
1. Survivorship Bias Handler - Handles delisted stocks
2. Health Monitor - System health monitoring with heartbeat
3. Statistical Validation - Rigorous alpha verification
4. Walk-Forward Optimization - OOS performance validation

Usage:
    from src.production_integration import ProductionSystem
    
    system = ProductionSystem(config)
    system.initialize()
    system.run_backtest(strategy, data)
    system.validate_strategy(returns)
    system.start_live_trading()

Author: Trading System
Version: 1.0.0
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import json

import numpy as np

# =============================================================================
# Feature Flags - Enable/Disable Modules
# =============================================================================

FEATURE_FLAGS = {
    "SURVIVORSHIP_HANDLER": os.environ.get("ENABLE_SURVIVORSHIP", "true").lower() == "true",
    "HEALTH_MONITOR": os.environ.get("ENABLE_HEALTH_MONITOR", "true").lower() == "true",
    "STATISTICAL_VALIDATION": os.environ.get("ENABLE_STATS_VALIDATION", "true").lower() == "true",
    "WALK_FORWARD": os.environ.get("ENABLE_WALK_FORWARD", "true").lower() == "true",
    "DISCORD_NOTIFICATIONS": os.environ.get("ENABLE_DISCORD", "false").lower() == "true",
}


# =============================================================================
# Conditional Imports with Graceful Fallbacks
# =============================================================================

# Survivorship Handler
if FEATURE_FLAGS["SURVIVORSHIP_HANDLER"]:
    try:
        from src.survivorship_handler import (
            DelisterTracker,
            PointInTimeUniverse,
            BiasValidator,
            DelistingReason,
            DelistedSecurity,
            initialize_survivorship_handler
        )
        HAS_SURVIVORSHIP = True
    except ImportError as e:
        warnings.warn(f"Survivorship handler not available: {e}")
        HAS_SURVIVORSHIP = False
else:
    HAS_SURVIVORSHIP = False

# Health Monitor
if FEATURE_FLAGS["HEALTH_MONITOR"]:
    try:
        from src.health_monitor import (
            HealthMonitor,
            HealthStatus,
            HealthCheckResult,
            HeartbeatService,
            CircuitBreaker,
            AlertManager,
            ComponentType,
            initialize_health_monitor
        )
        HAS_HEALTH_MONITOR = True
    except ImportError as e:
        warnings.warn(f"Health monitor not available: {e}")
        HAS_HEALTH_MONITOR = False
else:
    HAS_HEALTH_MONITOR = False

# Statistical Validation
if FEATURE_FLAGS["STATISTICAL_VALIDATION"]:
    try:
        from src.statistical_validation import (
            TStatCalculator,
            BootstrapAnalyzer,
            MultipleTestingCorrector,
            OverfitDetector,
            PerformanceReporter,
            ReturnFrequency,
            ValidationResult,
            PerformanceMetrics,
            validate_strategy as validate_strategy_func
        )
        HAS_STATS_VALIDATION = True
    except ImportError as e:
        warnings.warn(f"Statistical validation not available: {e}")
        HAS_STATS_VALIDATION = False
else:
    HAS_STATS_VALIDATION = False

# Walk-Forward Optimization
if FEATURE_FLAGS["WALK_FORWARD"]:
    try:
        from src.walk_forward import (
            WalkForwardEngine,
            AnchoredWalkForward,
            RollingWalkForward,
            DegradationAnalyzer,
            OptimizationObjective,
            WalkForwardSummary,
            quick_walk_forward
        )
        HAS_WALK_FORWARD = True
    except ImportError as e:
        warnings.warn(f"Walk-forward optimization not available: {e}")
        HAS_WALK_FORWARD = False
else:
    HAS_WALK_FORWARD = False


# =============================================================================
# Logging Configuration
# =============================================================================

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Configure logging for production system.
    
    Parameters
    ----------
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    log_file : Optional[str]
        Path to log file. If None, logs to console only.
    log_format : Optional[str]
        Custom log format string.
        
    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    if log_format is None:
        log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    
    logger = logging.getLogger("production_system")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
    
    return logger


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ProductionConfig:
    """Configuration for production trading system."""
    
    # General
    system_name: str = "AlgebraicTopologyTradingSystem"
    environment: str = "development"  # development, staging, production
    
    # Survivorship handling
    delisted_data_path: Optional[str] = None
    check_survivorship: bool = True
    
    # Health monitoring
    health_check_interval: int = 60  # seconds
    discord_webhook_url: Optional[str] = None
    alert_on_degradation: bool = True
    
    # Statistical validation
    min_sharpe_threshold: float = 0.5
    min_t_stat_threshold: float = 2.0
    confidence_level: float = 0.95
    bootstrap_iterations: int = 10000
    
    # Walk-forward
    walk_forward_mode: str = "anchored"  # anchored or rolling
    n_walk_forward_windows: int = 5
    train_ratio: float = 0.7
    max_sharpe_degradation: float = 0.2  # 20% max degradation allowed
    
    # Deployment gates
    require_statistical_validation: bool = True
    require_walk_forward_pass: bool = True
    fail_on_weak_stats: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = "logs/production.log"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }


class DeploymentGate(Enum):
    """Deployment gates that must pass before live trading."""
    STATISTICAL_SIGNIFICANCE = "statistical_significance"
    WALK_FORWARD_ROBUSTNESS = "walk_forward_robustness"
    HEALTH_CHECK = "health_check"
    SURVIVORSHIP_CHECK = "survivorship_check"


@dataclass
class DeploymentStatus:
    """Status of deployment gates."""
    gate: DeploymentGate
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# =============================================================================
# Production System - Main Integration Class
# =============================================================================

class ProductionSystem:
    """
    Unified production system integrating all trading components.
    
    This class orchestrates:
    - Survivorship bias handling for universe construction
    - Health monitoring with Discord notifications
    - Statistical validation for alpha verification
    - Walk-forward optimization for robustness testing
    
    Attributes
    ----------
    config : ProductionConfig
        System configuration.
    logger : logging.Logger
        System logger.
    health_monitor : Optional[HealthMonitor]
        Health monitoring instance.
    survivorship_handler : Optional[SurvivorshipHandler]
        Survivorship bias handler.
        
    Examples
    --------
    >>> config = ProductionConfig(
    ...     discord_webhook_url="https://discord.com/api/webhooks/...",
    ...     min_sharpe_threshold=1.0
    ... )
    >>> system = ProductionSystem(config)
    >>> system.initialize()
    >>> 
    >>> # Run validation
    >>> result = system.validate_strategy(returns, n_strategies_tested=10)
    >>> if result.is_valid:
    ...     system.start_live_trading()
    """
    
    def __init__(self, config: Optional[ProductionConfig] = None):
        """
        Initialize production system.
        
        Parameters
        ----------
        config : Optional[ProductionConfig]
            System configuration. Uses defaults if None.
        """
        self.config = config or ProductionConfig()
        
        # Setup logging
        log_dir = Path(self.config.log_file).parent if self.config.log_file else None
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logging(
            log_level=self.config.log_level,
            log_file=self.config.log_file
        )
        
        # Module instances (initialized lazily)
        self._health_monitor: Optional[Any] = None
        self._delister_tracker: Optional[Any] = None
        self._point_in_time_universe: Optional[Any] = None
        self._bias_validator: Optional[Any] = None
        self._performance_reporter: Optional[Any] = None
        self._overfit_detector: Optional[Any] = None
        self._degradation_analyzer: Optional[Any] = None
        
        # State
        self._initialized = False
        self._running = False
        self._shutdown_requested = False
        self._deployment_gates: List[DeploymentStatus] = []
        
        self.logger.info(f"ProductionSystem created: {self.config.system_name}")
        self._log_feature_flags()
    
    def _log_feature_flags(self) -> None:
        """Log enabled feature flags."""
        self.logger.info("Feature Flags:")
        for flag, enabled in FEATURE_FLAGS.items():
            status = "✓ ENABLED" if enabled else "✗ DISABLED"
            self.logger.info(f"  {flag}: {status}")
    
    # =========================================================================
    # Initialization
    # =========================================================================
    
    def initialize(self) -> None:
        """
        Initialize all enabled modules.
        
        Must be called before using the system.
        """
        self.logger.info("Initializing production system...")
        
        # Initialize survivorship handler
        if HAS_SURVIVORSHIP and self.config.check_survivorship:
            self._init_survivorship()
        
        # Initialize health monitor
        if HAS_HEALTH_MONITOR:
            self._init_health_monitor()
        
        # Initialize statistical validation
        if HAS_STATS_VALIDATION:
            self._init_statistical_validation()
        
        # Initialize walk-forward components
        if HAS_WALK_FORWARD:
            self._init_walk_forward()
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        self._initialized = True
        self.logger.info("Production system initialized successfully")
    
    def _init_survivorship(self) -> None:
        """Initialize survivorship bias handler."""
        self.logger.info("Initializing survivorship handler...")
        
        try:
            self._delister_tracker = DelisterTracker()
            self._point_in_time_universe = PointInTimeUniverse(self._delister_tracker)
            self._bias_validator = BiasValidator(self._delister_tracker)
            self.logger.info("Survivorship handler initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize survivorship handler: {e}")
            raise
    
    def _init_health_monitor(self) -> None:
        """Initialize health monitoring."""
        self.logger.info("Initializing health monitor...")
        
        try:
            webhook = self.config.discord_webhook_url if FEATURE_FLAGS["DISCORD_NOTIFICATIONS"] else None
            self._health_monitor = HealthMonitor(
                heartbeat_interval=self.config.health_check_interval,
                check_interval=self.config.health_check_interval,
                discord_webhook=webhook
            )
            self.logger.info("Health monitor initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize health monitor: {e}")
            raise
    
    def _init_statistical_validation(self) -> None:
        """Initialize statistical validation components."""
        self.logger.info("Initializing statistical validation...")
        
        try:
            self._performance_reporter = PerformanceReporter(
                confidence_level=self.config.confidence_level,
                bootstrap_iterations=self.config.bootstrap_iterations
            )
            self._overfit_detector = OverfitDetector()
            self.logger.info("Statistical validation initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize statistical validation: {e}")
            raise
    
    def _init_walk_forward(self) -> None:
        """Initialize walk-forward components."""
        self.logger.info("Initializing walk-forward optimization...")
        
        try:
            self._degradation_analyzer = DegradationAnalyzer()
            self.logger.info("Walk-forward optimization initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize walk-forward: {e}")
            raise
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def shutdown_handler(signum, frame):
            self.logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
            self._shutdown_requested = True
            self.shutdown()
        
        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)
    
    # =========================================================================
    # Survivorship Integration
    # =========================================================================
    
    def filter_universe(
        self,
        symbols: List[str],
        as_of_date: Optional[datetime] = None
    ) -> List[str]:
        """
        Filter trading universe to exclude delisted stocks.
        
        Parameters
        ----------
        symbols : List[str]
            List of symbols to filter.
        as_of_date : Optional[datetime]
            Point-in-time date for universe reconstruction.
            
        Returns
        -------
        List[str]
            Filtered list of active symbols.
        """
        if not HAS_SURVIVORSHIP or not self.config.check_survivorship:
            return symbols
        
        if self._delister_tracker is None:
            self.logger.warning("Survivorship handler not initialized")
            return symbols
        
        as_of_date = as_of_date or datetime.now()
        
        # Get point-in-time universe
        try:
            symbol_set = set(symbols)
            filtered_set = self._delister_tracker.get_active_symbols(as_of_date, symbol_set)
            filtered = [s for s in symbols if s in filtered_set]
            
            removed = symbol_set - filtered_set
            if removed:
                self.logger.info(f"Filtered {len(removed)} delisted symbols: {list(removed)[:5]}...")
            
            return filtered
        except Exception as e:
            self.logger.error(f"Error filtering universe: {e}")
            return symbols
    
    def check_delist_status(self, symbol: str, as_of_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Check if a symbol has been delisted.
        
        Parameters
        ----------
        symbol : str
            Symbol to check.
        as_of_date : Optional[datetime]
            Date to check status at.
            
        Returns
        -------
        Dict[str, Any]
            Delist status information.
        """
        if not HAS_SURVIVORSHIP or self._delister_tracker is None:
            return {"is_delisted": False, "reason": None}
        
        as_of_date = as_of_date or datetime.now()
        
        try:
            is_delisted = self._delister_tracker.is_delisted(symbol, as_of_date)
            delist_return = self._delister_tracker.get_delisting_return(symbol)
            return {
                "is_delisted": is_delisted,
                "delisting_return": delist_return
            }
        except Exception as e:
            self.logger.error(f"Error checking delist status for {symbol}: {e}")
            return {"is_delisted": False, "error": str(e)}
    
    # =========================================================================
    # Health Monitoring Integration
    # =========================================================================
    
    def start_health_monitoring(self) -> None:
        """Start background health monitoring."""
        if not HAS_HEALTH_MONITOR or self._health_monitor is None:
            self.logger.warning("Health monitor not available")
            return
        
        try:
            self._health_monitor.start()
            self.logger.info("Health monitoring started")
        except Exception as e:
            self.logger.error(f"Failed to start health monitoring: {e}")
    
    def stop_health_monitoring(self) -> None:
        """Stop health monitoring."""
        if self._health_monitor is not None:
            try:
                self._health_monitor.stop()
                self.logger.info("Health monitoring stopped")
            except Exception as e:
                self.logger.error(f"Error stopping health monitor: {e}")
    
    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: str = "gauge"
    ) -> None:
        """
        Record a metric for health monitoring.
        
        Parameters
        ----------
        name : str
            Metric name.
        value : float
            Metric value.
        metric_type : str
            Type: 'gauge', 'counter', 'histogram'.
        """
        # Health monitor tracks component status, not arbitrary metrics
        # Log the metric instead
        self.logger.debug(f"Metric recorded: {name}={value} ({metric_type})")
    
    def send_alert(
        self,
        message: str,
        severity: str = "warning",
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Send alert via health monitor (Discord, email, etc.).
        
        Parameters
        ----------
        message : str
            Alert message.
        severity : str
            Alert severity: 'info', 'warning', 'error', 'critical'.
        context : Optional[Dict[str, Any]]
            Additional context for the alert.
        """
        if not HAS_HEALTH_MONITOR or self._health_monitor is None:
            self.logger.warning(f"Alert (no monitor): [{severity}] {message}")
            return
        
        try:
            self._health_monitor.alert_manager.send_alert(severity, "system", message, context)
        except Exception as e:
            self.logger.error(f"Failed to send alert: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get current system health status.
        
        Returns
        -------
        Dict[str, Any]
            Health status summary.
        """
        if not HAS_HEALTH_MONITOR or self._health_monitor is None:
            return {"status": "unknown", "reason": "Health monitor not available"}
        
        try:
            return self._health_monitor.get_system_health()
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    # =========================================================================
    # Statistical Validation Integration
    # =========================================================================
    
    def validate_strategy(
        self,
        returns: np.ndarray,
        frequency: str = "daily",
        n_strategies_tested: int = 1,
        strategy_name: str = "Strategy"
    ) -> Dict[str, Any]:
        """
        Validate strategy returns with rigorous statistical testing.
        
        Parameters
        ----------
        returns : np.ndarray
            Strategy returns.
        frequency : str
            Return frequency: 'daily', 'hourly', 'minute'.
        n_strategies_tested : int
            Number of strategies tested (for multiple testing correction).
        strategy_name : str
            Name for logging and reporting.
            
        Returns
        -------
        Dict[str, Any]
            Validation results including metrics, tests, and pass/fail status.
        """
        if not HAS_STATS_VALIDATION:
            self.logger.warning("Statistical validation not available")
            return {
                "is_valid": True,
                "warning": "Validation skipped - module not available"
            }
        
        self.logger.info(f"Validating strategy: {strategy_name}")
        
        # Map frequency string to enum
        freq_map = {
            "daily": ReturnFrequency.DAILY,
            "hourly": ReturnFrequency.HOURLY,
            "minute": ReturnFrequency.MINUTE
        }
        freq = freq_map.get(frequency.lower(), ReturnFrequency.DAILY)
        
        try:
            # Run full validation
            result = validate_strategy_func(
                returns,
                frequency=freq,
                n_strategies_tested=n_strategies_tested,
                confidence_level=self.config.confidence_level
            )
            
            metrics = result["performance_metrics"]
            overfit = result["overfit_analysis"]
            
            # Check against thresholds
            passes_sharpe = metrics.sharpe_ratio >= self.config.min_sharpe_threshold
            passes_tstat = metrics.t_statistic >= self.config.min_t_stat_threshold
            passes_overfit = not overfit["is_likely_overfit"]
            
            is_valid = passes_sharpe and passes_tstat and passes_overfit
            
            # Log results
            self.logger.info(f"Validation Results for {strategy_name}:")
            self.logger.info(f"  Sharpe Ratio: {metrics.sharpe_ratio:.3f} (threshold: {self.config.min_sharpe_threshold})")
            self.logger.info(f"  t-statistic: {metrics.t_statistic:.3f} (threshold: {self.config.min_t_stat_threshold})")
            self.logger.info(f"  p-value: {metrics.p_value:.4f}")
            self.logger.info(f"  Deflated Sharpe: {overfit['deflated_sharpe_ratio']:.3f}")
            self.logger.info(f"  Is Valid: {is_valid}")
            
            # Record deployment gate
            self._deployment_gates.append(DeploymentStatus(
                gate=DeploymentGate.STATISTICAL_SIGNIFICANCE,
                passed=is_valid,
                message=f"Sharpe={metrics.sharpe_ratio:.2f}, t={metrics.t_statistic:.2f}",
                details={
                    "sharpe_ratio": metrics.sharpe_ratio,
                    "t_statistic": metrics.t_statistic,
                    "p_value": metrics.p_value,
                    "passes_sharpe": passes_sharpe,
                    "passes_tstat": passes_tstat,
                    "passes_overfit": passes_overfit
                }
            ))
            
            # Alert on failure if configured
            if not is_valid and self.config.alert_on_degradation:
                self.send_alert(
                    f"Strategy {strategy_name} failed validation: Sharpe={metrics.sharpe_ratio:.2f}",
                    severity="warning",
                    context={"sharpe": metrics.sharpe_ratio, "t_stat": metrics.t_statistic}
                )
            
            return {
                "is_valid": is_valid,
                "metrics": metrics,
                "overfit_analysis": overfit,
                "report": result["report"],
                "thresholds": {
                    "sharpe": self.config.min_sharpe_threshold,
                    "t_stat": self.config.min_t_stat_threshold
                }
            }
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return {"is_valid": False, "error": str(e)}
    
    def generate_performance_report(
        self,
        returns: np.ndarray,
        frequency: str = "daily"
    ) -> str:
        """
        Generate publication-quality performance report.
        
        Parameters
        ----------
        returns : np.ndarray
            Strategy returns.
        frequency : str
            Return frequency.
            
        Returns
        -------
        str
            Formatted performance report.
        """
        if not HAS_STATS_VALIDATION or self._performance_reporter is None:
            return "Performance reporting not available"
        
        freq_map = {
            "daily": ReturnFrequency.DAILY,
            "hourly": ReturnFrequency.HOURLY,
            "minute": ReturnFrequency.MINUTE
        }
        freq = freq_map.get(frequency.lower(), ReturnFrequency.DAILY)
        
        try:
            metrics = self._performance_reporter.full_report(returns, freq)
            return self._performance_reporter.format_report(metrics)
        except Exception as e:
            return f"Error generating report: {e}"
    
    # =========================================================================
    # Walk-Forward Integration
    # =========================================================================
    
    def run_walk_forward(
        self,
        data: np.ndarray,
        strategy: Any,
        param_bounds: Dict[str, Tuple[float, float]],
        optimization_method: str = "grid",
        grid_points: int = 10
    ) -> Dict[str, Any]:
        """
        Run walk-forward analysis on strategy.
        
        Parameters
        ----------
        data : np.ndarray
            Historical price or return data.
        strategy : Any
            Strategy object with set_parameters() and run() methods.
        param_bounds : Dict[str, Tuple[float, float]]
            Parameter bounds for optimization.
        optimization_method : str
            Optimization method: 'grid', 'random', etc.
        grid_points : int
            Grid points for grid search.
            
        Returns
        -------
        Dict[str, Any]
            Walk-forward results including summary and degradation analysis.
        """
        if not HAS_WALK_FORWARD:
            self.logger.warning("Walk-forward optimization not available")
            return {
                "is_robust": True,
                "warning": "Walk-forward skipped - module not available"
            }
        
        self.logger.info(f"Running walk-forward analysis ({self.config.walk_forward_mode} mode)...")
        
        try:
            # Create walk-forward engine
            if self.config.walk_forward_mode == "anchored":
                engine = AnchoredWalkForward(
                    data,
                    train_ratio=self.config.train_ratio,
                    n_windows=self.config.n_walk_forward_windows,
                    objective=OptimizationObjective.SHARPE
                )
            else:
                engine = RollingWalkForward(
                    data,
                    train_ratio=self.config.train_ratio,
                    n_windows=self.config.n_walk_forward_windows,
                    objective=OptimizationObjective.SHARPE
                )
            
            # Run walk-forward
            summary = engine.run(
                strategy,
                param_bounds,
                optimization_method=optimization_method,
                grid_points=grid_points
            )
            
            # Analyze degradation
            analysis = self._degradation_analyzer.analyze(summary)
            report = self._degradation_analyzer.generate_report(summary, analysis)
            
            # Check robustness criteria
            degradation_ok = summary.mean_sharpe_degradation >= (1 - self.config.max_sharpe_degradation)
            oos_sharpe_ok = summary.combined_oos_sharpe >= self.config.min_sharpe_threshold
            is_robust = degradation_ok and oos_sharpe_ok
            
            # Log results
            self.logger.info(f"Walk-Forward Results:")
            self.logger.info(f"  Windows: {summary.n_windows}")
            self.logger.info(f"  Mean IS Sharpe: {summary.mean_train_sharpe:.3f}")
            self.logger.info(f"  Mean OOS Sharpe: {summary.mean_test_sharpe:.3f}")
            self.logger.info(f"  Combined OOS Sharpe: {summary.combined_oos_sharpe:.3f}")
            self.logger.info(f"  Degradation: {summary.mean_sharpe_degradation:.1%}")
            self.logger.info(f"  Is Robust: {is_robust}")
            
            # Record deployment gate
            self._deployment_gates.append(DeploymentStatus(
                gate=DeploymentGate.WALK_FORWARD_ROBUSTNESS,
                passed=is_robust,
                message=f"OOS Sharpe={summary.combined_oos_sharpe:.2f}, Degradation={summary.mean_sharpe_degradation:.0%}",
                details={
                    "mean_train_sharpe": summary.mean_train_sharpe,
                    "mean_test_sharpe": summary.mean_test_sharpe,
                    "combined_oos_sharpe": summary.combined_oos_sharpe,
                    "degradation": summary.mean_sharpe_degradation,
                    "n_windows": summary.n_windows
                }
            ))
            
            # Alert on failure
            if not is_robust and self.config.alert_on_degradation:
                self.send_alert(
                    f"Walk-forward failed: OOS Sharpe={summary.combined_oos_sharpe:.2f}, "
                    f"Degradation={summary.mean_sharpe_degradation:.0%}",
                    severity="warning"
                )
            
            return {
                "is_robust": is_robust,
                "summary": summary,
                "analysis": analysis,
                "report": report,
                "combined_oos_sharpe": summary.combined_oos_sharpe,
                "degradation": summary.mean_sharpe_degradation
            }
            
        except Exception as e:
            self.logger.error(f"Walk-forward failed: {e}")
            return {"is_robust": False, "error": str(e)}
    
    # =========================================================================
    # Deployment Gates
    # =========================================================================
    
    def check_deployment_gates(self) -> Tuple[bool, List[DeploymentStatus]]:
        """
        Check all deployment gates.
        
        Returns
        -------
        Tuple[bool, List[DeploymentStatus]]
            (all_passed, list_of_gate_statuses)
        """
        all_passed = all(gate.passed for gate in self._deployment_gates)
        return all_passed, self._deployment_gates
    
    def can_deploy(self) -> bool:
        """
        Check if system can be deployed to production.
        
        Returns
        -------
        bool
            True if all required gates passed.
        """
        all_passed, gates = self.check_deployment_gates()
        
        if not self.config.require_statistical_validation:
            # Skip stats gate check
            gates = [g for g in gates if g.gate != DeploymentGate.STATISTICAL_SIGNIFICANCE]
        
        if not self.config.require_walk_forward_pass:
            # Skip walk-forward gate check
            gates = [g for g in gates if g.gate != DeploymentGate.WALK_FORWARD_ROBUSTNESS]
        
        return all(g.passed for g in gates)
    
    def get_deployment_report(self) -> str:
        """Generate deployment readiness report."""
        lines = [
            "=" * 60,
            "DEPLOYMENT READINESS REPORT",
            "=" * 60,
            f"System: {self.config.system_name}",
            f"Environment: {self.config.environment}",
            f"Timestamp: {datetime.now().isoformat()}",
            "",
            "-" * 40,
            "DEPLOYMENT GATES",
            "-" * 40,
        ]
        
        for gate in self._deployment_gates:
            status = "✓ PASSED" if gate.passed else "✗ FAILED"
            lines.append(f"  {gate.gate.value}: {status}")
            lines.append(f"    {gate.message}")
        
        lines.extend([
            "",
            "-" * 40,
            f"DEPLOYMENT READY: {'YES' if self.can_deploy() else 'NO'}",
            "=" * 60,
        ])
        
        return "\n".join(lines)
    
    # =========================================================================
    # Lifecycle Management
    # =========================================================================
    
    def start_live_trading(self) -> bool:
        """
        Start live trading if all gates pass.
        
        Returns
        -------
        bool
            True if trading started successfully.
        """
        if not self._initialized:
            self.logger.error("System not initialized. Call initialize() first.")
            return False
        
        if not self.can_deploy():
            self.logger.error("Deployment gates not passed. Cannot start live trading.")
            self.logger.info(self.get_deployment_report())
            return False
        
        self.logger.info("Starting live trading...")
        self._running = True
        
        # Start health monitoring
        self.start_health_monitoring()
        
        # Send startup notification
        self.send_alert(
            f"🚀 {self.config.system_name} started live trading",
            severity="info"
        )
        
        return True
    
    def shutdown(self) -> None:
        """Graceful shutdown of all components."""
        self.logger.info("Shutting down production system...")
        
        self._running = False
        
        # Stop health monitoring
        self.stop_health_monitoring()
        
        # Send shutdown notification
        self.send_alert(
            f"🛑 {self.config.system_name} shutting down",
            severity="info"
        )
        
        self.logger.info("Production system shutdown complete")
    
    def __enter__(self) -> "ProductionSystem":
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.shutdown()


# =============================================================================
# Quick Start Functions
# =============================================================================

def create_production_system(
    discord_webhook: Optional[str] = None,
    log_level: str = "INFO",
    environment: str = "development"
) -> ProductionSystem:
    """
    Create a configured production system instance.
    
    Parameters
    ----------
    discord_webhook : Optional[str]
        Discord webhook URL for notifications.
    log_level : str
        Logging level.
    environment : str
        Environment name.
        
    Returns
    -------
    ProductionSystem
        Configured and initialized system.
    """
    config = ProductionConfig(
        environment=environment,
        discord_webhook_url=discord_webhook,
        log_level=log_level
    )
    
    system = ProductionSystem(config)
    system.initialize()
    
    return system


def quick_validate(
    returns: np.ndarray,
    n_strategies: int = 1
) -> Tuple[bool, str]:
    """
    Quick validation of strategy returns.
    
    Parameters
    ----------
    returns : np.ndarray
        Strategy returns.
    n_strategies : int
        Number of strategies tested.
        
    Returns
    -------
    Tuple[bool, str]
        (is_valid, report)
    """
    config = ProductionConfig(log_level="WARNING")
    system = ProductionSystem(config)
    system.initialize()
    
    result = system.validate_strategy(returns, n_strategies_tested=n_strategies)
    
    return result.get("is_valid", False), result.get("report", "")


# =============================================================================
# Module Status Check
# =============================================================================

def check_module_status() -> Dict[str, bool]:
    """
    Check availability of all production modules.
    
    Returns
    -------
    Dict[str, bool]
        Module name -> availability status.
    """
    return {
        "survivorship_handler": HAS_SURVIVORSHIP,
        "health_monitor": HAS_HEALTH_MONITOR,
        "statistical_validation": HAS_STATS_VALIDATION,
        "walk_forward": HAS_WALK_FORWARD
    }


if __name__ == "__main__":
    # Demo/test the integration
    print("Production Integration Module Demo")
    print("=" * 60)
    
    # Check module status
    print("\nModule Status:")
    for module, available in check_module_status().items():
        status = "✓ Available" if available else "✗ Not Available"
        print(f"  {module}: {status}")
    
    # Create system
    print("\n" + "-" * 40)
    print("Creating Production System...")
    
    config = ProductionConfig(
        system_name="DemoSystem",
        environment="development",
        log_level="INFO"
    )
    
    with ProductionSystem(config) as system:
        # Test statistical validation
        print("\n" + "-" * 40)
        print("Testing Statistical Validation...")
        
        np.random.seed(42)
        test_returns = np.random.randn(252) * 0.015 + 0.0004
        
        result = system.validate_strategy(
            test_returns,
            strategy_name="TestStrategy",
            n_strategies_tested=5
        )
        
        print(f"\nValidation Result: {'PASS' if result.get('is_valid') else 'FAIL'}")
        
        # Test walk-forward (using simple strategy)
        if HAS_WALK_FORWARD:
            print("\n" + "-" * 40)
            print("Testing Walk-Forward Analysis...")
            
            from src.walk_forward import SimpleMovingAverageStrategy
            
            prices = 100 * np.exp(np.cumsum(np.random.randn(500) * 0.02 + 0.0002))
            strategy = SimpleMovingAverageStrategy()
            param_bounds = {'fast_period': (5, 15), 'slow_period': (20, 40)}
            
            wf_result = system.run_walk_forward(
                prices,
                strategy,
                param_bounds,
                grid_points=3
            )
            
            print(f"\nWalk-Forward Result: {'ROBUST' if wf_result.get('is_robust') else 'NOT ROBUST'}")
        
        # Deployment report
        print("\n" + "-" * 40)
        print(system.get_deployment_report())
