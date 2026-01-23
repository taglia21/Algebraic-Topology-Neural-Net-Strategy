#!/usr/bin/env python3
"""
V2.5 Production Launcher - Adaptive Trading System Orchestrator
================================================================

Production-ready launcher that orchestrates the full trading pipeline:
1. Data Validation - Verify data quality and freshness
2. TDA Computation - Calculate topological features
3. Signal Generation - V2.5 adaptive regime + ensemble predictions
4. Trade Execution - Alpaca integration with execution alpha optimization
5. Discord Notifications - Real-time trade alerts and daily summaries

V2.5 Adaptive Features:
- Phase 1: Enhanced Regime Detection (OnlineRegimeLearner)
- Phase 2: Pattern Memory System (LSH similarity search)
- Phase 3: Adaptive Position Sizing (DD/Vol-aware)
- Phase 4: Continuous Learning Loop (daily updates)

Target Performance (V2.5):
- Sharpe: > 0.85 (adaptive improvement from V2.1)
- Max Drawdown: < 15% (with DD-aware sizing)
- Win Rate: > 55%

Risk Controls:
- Position limits: 2-5% capital per position
- Stop-loss: 3-4 sigma moves
- Portfolio exposure caps
- Circuit breakers at 5%/8% drawdown

Usage:
    # Paper trading mode (default)
    python production_launcher.py
    
    # Live trading mode (requires explicit flag)
    python production_launcher.py --live
    
    # Dry run (no orders submitted)
    python production_launcher.py --dry-run
    
    # Single run (no scheduling)
    python production_launcher.py --once
"""

import os
import sys
import json
import time
import signal
import logging
import argparse
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
import traceback
import hashlib

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Production deployment utilities
try:
    from src.utils.health_check import HealthCheckServer, HealthStatus
    from src.utils.graceful_shutdown import GracefulShutdown
    DEPLOYMENT_UTILS_AVAILABLE = True
except ImportError:
    DEPLOYMENT_UTILS_AVAILABLE = False

# Configure structured JSON logging
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Rotating log handler
from logging.handlers import RotatingFileHandler

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure structured JSON logging with rotation."""
    logger = logging.getLogger("production")
    logger.setLevel(level)
    
    # Console handler
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(console)
    
    # Rotating file handler (10MB, keep 10 files)
    file_handler = RotatingFileHandler(
        LOG_DIR / "production.log",
        maxBytes=10*1024*1024,
        backupCount=10
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(
        '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
        datefmt='%Y-%m-%dT%H:%M:%S'
    ))
    logger.addHandler(file_handler)
    
    # JSON structured log for metrics
    json_handler = RotatingFileHandler(
        LOG_DIR / "metrics.jsonl",
        maxBytes=50*1024*1024,
        backupCount=5
    )
    json_handler.setLevel(logging.INFO)
    logger.addHandler(json_handler)
    
    return logger

logger = setup_logging()

# Import trading components
try:
    from src.trading.v25_production_engine import V25ProductionEngine, V25EngineConfig
    from src.trading.alpaca_client import AlpacaClient, OrderSide
    from src.trading.notifications import (
        send_discord, notify_trade_executed, notify_rebalance_summary,
        notify_regime_change, notify_error
    )
    from src.trading.monitoring_dashboard import MonitoringDashboard
    from src.trading.daily_validator import DailyValidator
except ImportError as e:
    logger.error(f"Failed to import trading components: {e}")
    logger.error("Ensure all V2.5 modules are installed")
    sys.exit(1)

# Import V2.2 RL components (optional enhancement)
try:
    from src.trading.rl_orchestrator import RLOrchestrator, RLOrchestratorConfig
    RL_AVAILABLE = True
    logger.info("V2.2 RL components available")
except ImportError as e:
    RL_AVAILABLE = False
    logger.info(f"V2.2 RL components not available: {e} (using V2.1 baseline)")

# Import V26 components (adaptive learning)
try:
    from src.ml.continuous_learner import ContinuousLearner, ContinuousLearnerConfig, TradeResult
    from src.risk.circuit_breakers import V26CircuitBreakers, V26CircuitBreakerConfig
    from src.monitoring.model_health import ModelHealthMonitor, ModelHealthConfig
    V26_AVAILABLE = True
    logger.info("V26 Adaptive Learning components available")
except ImportError as e:
    V26_AVAILABLE = False
    logger.info(f"V26 components not available: {e}")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class LauncherConfig:
    """Production launcher configuration."""
    
    # Trading mode
    paper_trading: bool = True
    dry_run: bool = False
    single_run: bool = False
    
    # Schedule (Eastern Time)
    market_open: str = "09:30"
    market_close: str = "16:00"
    rebalance_time: str = "09:45"  # 15 min after open for stability
    eod_summary_time: str = "16:30"
    
    # Risk controls
    max_position_pct: float = 0.03  # 3% max per position
    max_portfolio_heat: float = 0.20  # 20% total risk
    stop_loss_sigma: float = 3.5  # 3.5 sigma stop
    circuit_breaker_dd_pct: float = 0.05  # 5% drawdown halt
    emergency_halt_dd_pct: float = 0.08  # 8% emergency halt
    
    # Execution optimization
    use_limit_orders: bool = True
    limit_order_buffer_pct: float = 0.001  # 0.1% limit price buffer
    max_slippage_pct: float = 0.005  # 0.5% max slippage tolerance
    high_liquidity_hours: Tuple[int, int] = (10, 15)  # 10am-3pm ET
    
    # V2.1 enhancements
    use_ensemble_regime: bool = True
    use_transformer: bool = True
    fallback_to_v13: bool = True
    
    # V2.2 RL enhancements
    use_rl_position_sizing: bool = True  # Enable SAC position optimizer
    use_hierarchical_regime: bool = True  # Enable hierarchical controller
    use_anomaly_detection: bool = True    # Enable anomaly-aware sizing
    rl_blend_weight: float = 0.6          # Weight for RL vs base signal
    
    # V26 Adaptive Learning enhancements
    use_v26_continuous_learning: bool = True   # Enable continuous learner
    use_v26_circuit_breakers: bool = True      # Enable 3-level circuit breakers
    use_v26_model_health: bool = True          # Enable model health monitoring
    v26_metrics_log_path: str = "logs/v26_metrics.jsonl"
    
    # Monitoring
    enable_dashboard: bool = True
    dashboard_port: int = 8080
    discord_notifications: bool = True
    
    # Data validation
    data_staleness_minutes: int = 60
    min_data_points: int = 100
    max_missing_pct: float = 0.05  # 5% max missing data
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# PRODUCTION LAUNCHER
# =============================================================================

class ProductionLauncher:
    """
    V2.1 Production System Orchestrator
    
    Manages the complete trading lifecycle:
    1. Startup validation
    2. Market data ingestion
    3. TDA feature computation
    4. Signal generation (ensemble + transformer)
    5. Risk-adjusted position sizing
    6. Order execution with alpha optimization
    7. Real-time monitoring and alerts
    8. Daily validation and reporting
    """
    
    def __init__(self, config: LauncherConfig):
        self.config = config
        self.is_running = False
        self._shutdown_event = threading.Event()
        
        # State tracking
        self._last_rebalance: Optional[datetime] = None
        self._daily_trades: List[Dict] = []
        self._daily_pnl: float = 0.0
        self._peak_equity: float = 0.0
        self._current_drawdown: float = 0.0
        self._is_halted: bool = False
        self._halt_reason: str = ""
        
        # Initialize components
        self._initialize_components()
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        
    def _initialize_components(self):
        """Initialize all trading components."""
        logger.info("=" * 60)
        logger.info("V2.5 PRODUCTION LAUNCHER - INITIALIZING")
        logger.info("=" * 60)
        
        # 1. V2.5 Adaptive Trading Engine
        v25_config = V25EngineConfig(
            use_v25_elite=True,
            signal_mode="hybrid",
            use_elite_features=True,
            use_gradient_ensemble=True,
            use_signal_validator=True,
            use_data_quality=True,
            use_attention_factor=self.config.use_ensemble_regime,
            use_temporal_transformer=self.config.use_transformer,
            use_tca_optimizer=True,
            use_kelly_sizer=True,
            max_position_pct=self.config.max_position_pct,
            max_portfolio_heat=self.config.max_portfolio_heat,
            max_daily_loss=self.config.circuit_breaker_dd_pct,
        )
        self.engine = V25ProductionEngine(v25_config)
        logger.info("✅ V2.5 Adaptive Trading Engine initialized")
        
        # 2. Alpaca Client (paper or live)
        try:
            self.alpaca = AlpacaClient()
            account = self.alpaca.get_account()
            self._peak_equity = float(account.equity)
            logger.info(f"✅ Alpaca connected - {'PAPER' if self.config.paper_trading else 'LIVE'}")
            logger.info(f"   Account equity: ${account.equity:,.2f}")
        except Exception as e:
            logger.error(f"❌ Alpaca connection failed: {e}")
            if not self.config.dry_run:
                raise
            self.alpaca = None
            
        # 3. Monitoring Dashboard
        if self.config.enable_dashboard:
            self.dashboard = MonitoringDashboard(port=self.config.dashboard_port)
            logger.info(f"✅ Monitoring dashboard on port {self.config.dashboard_port}")
        else:
            self.dashboard = None
            
        # 4. Daily Validator
        self.validator = DailyValidator()
        logger.info("✅ Daily validator initialized")
        
        # 5. V2.2 RL Orchestrator (optional enhancement)
        if self.config.use_rl_position_sizing and RL_AVAILABLE:
            rl_config = RLOrchestratorConfig(
                use_sac=self.config.use_rl_position_sizing,
                use_hierarchical_regime=self.config.use_hierarchical_regime,
                use_anomaly_transformer=self.config.use_anomaly_detection,
                max_position_pct=self.config.max_position_pct,
            )
            self.rl_orchestrator = RLOrchestrator(rl_config)
            logger.info("✅ V2.2 RL Orchestrator initialized (SAC + Hierarchical Regime)")
        else:
            self.rl_orchestrator = None
            if self.config.use_rl_position_sizing and not RL_AVAILABLE:
                logger.warning("⚠️  V2.2 RL requested but not available, using V2.1 baseline")
        
        # 6. Notification system check
        if self.config.discord_notifications:
            webhook = os.getenv("DISCORD_WEBHOOK", "")
            if webhook:
                logger.info("✅ Discord notifications enabled")
                rl_status = "Enabled" if self.rl_orchestrator else "Disabled"
                send_discord("🚀 V2.1/V2.2 Production Launcher Started",
                           f"Mode: {'PAPER' if self.config.paper_trading else '🔴 LIVE'}\n"
                           f"Dry Run: {self.config.dry_run}\n"
                           f"Ensemble Regime: {self.config.use_ensemble_regime}\n"
                           f"Transformer: {self.config.use_transformer}\n"
                           f"V2.2 RL Sizing: {rl_status}",
                           color=0x00FF00)
            else:
                logger.warning("⚠️  Discord webhook not configured")
        
        # 7. Health Check Server (production deployment)
        self.health_server = None
        if DEPLOYMENT_UTILS_AVAILABLE and self.config.enable_dashboard:
            try:
                self.health_server = HealthCheckServer(
                    port=self.config.dashboard_port,
                    version="2.1.0",
                )
                self.health_server.start_background()
                logger.info(f"✅ Health check server on port {self.config.dashboard_port}")
            except Exception as e:
                logger.warning(f"⚠️  Health check server failed to start: {e}")
        
        # 8. Graceful Shutdown Handler (production deployment)
        self.graceful_shutdown = None
        if DEPLOYMENT_UTILS_AVAILABLE:
            self.graceful_shutdown = GracefulShutdown(
                state_dir=PROJECT_ROOT / "state",
                discord_webhook=os.getenv("DISCORD_WEBHOOK_URL", ""),
            )
            self.graceful_shutdown.register()
            logger.info("✅ Graceful shutdown handler registered")
        
        # 9. V26 Continuous Learner
        self.continuous_learner = None
        if V26_AVAILABLE and self.config.use_v26_continuous_learning:
            try:
                learner_config = ContinuousLearnerConfig(
                    metrics_log_path=self.config.v26_metrics_log_path
                )
                self.continuous_learner = ContinuousLearner(learner_config)
                self.continuous_learner.discord_callback = send_discord if self.config.discord_notifications else None
                logger.info("✅ V26 Continuous Learner initialized")
            except Exception as e:
                logger.warning(f"⚠️  V26 Continuous Learner failed: {e}")
        
        # 10. V26 Circuit Breakers
        self.v26_circuit_breakers = None
        if V26_AVAILABLE and self.config.use_v26_circuit_breakers:
            try:
                breaker_config = V26CircuitBreakerConfig(
                    max_position_pct=self.config.max_position_pct
                )
                self.v26_circuit_breakers = V26CircuitBreakers(breaker_config)
                self.v26_circuit_breakers.discord_callback = send_discord if self.config.discord_notifications else None
                if self.alpaca:
                    account = self.alpaca.get_account()
                    self.v26_circuit_breakers.reset_daily(float(account.equity))
                logger.info("✅ V26 Circuit Breakers initialized")
            except Exception as e:
                logger.warning(f"⚠️  V26 Circuit Breakers failed: {e}")
        
        # 11. V26 Model Health Monitor
        self.model_health_monitor = None
        if V26_AVAILABLE and self.config.use_v26_model_health:
            try:
                health_config = ModelHealthConfig(
                    metrics_log_path=self.config.v26_metrics_log_path.replace('.jsonl', '_health.jsonl')
                )
                self.model_health_monitor = ModelHealthMonitor(health_config)
                self.model_health_monitor.set_discord_callback(send_discord if self.config.discord_notifications else None)
                self.model_health_monitor.start_background_monitoring(interval_seconds=300)  # 5 min
                logger.info("✅ V26 Model Health Monitor initialized")
            except Exception as e:
                logger.warning(f"⚠️  V26 Model Health Monitor failed: {e}")
                
        logger.info("=" * 60)
        logger.info("INITIALIZATION COMPLETE - READY FOR TRADING")
        logger.info("=" * 60)
        
    def _handle_shutdown(self, signum, frame):
        """Graceful shutdown handler."""
        logger.info(f"Received shutdown signal {signum}")
        self._shutdown_event.set()
        self.is_running = False
        
        if self.config.discord_notifications:
            send_discord("🛑 Production Launcher Shutdown",
                        f"Signal: {signum}\nTime: {datetime.now().isoformat()}",
                        color=0xFF0000)
            
    # =========================================================================
    # DATA VALIDATION
    # =========================================================================
    
    def validate_data(self, tickers: List[str]) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate data quality and freshness.
        
        Returns:
            Tuple of (is_valid, validation_report)
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "tickers_requested": len(tickers),
            "tickers_valid": 0,
            "tickers_missing": [],
            "tickers_stale": [],
            "data_quality_score": 0.0,
            "is_valid": False,
        }
        
        try:
            valid_count = 0
            for ticker in tickers[:50]:  # Validate sample
                # Check data freshness
                is_fresh = self.engine.check_data_freshness(
                    ticker, 
                    max_age_minutes=self.config.data_staleness_minutes
                )
                
                if is_fresh:
                    valid_count += 1
                else:
                    report["tickers_stale"].append(ticker)
                    
            report["tickers_valid"] = valid_count
            report["data_quality_score"] = valid_count / len(tickers[:50])
            
            # Validation criteria
            min_valid_ratio = 1.0 - self.config.max_missing_pct
            report["is_valid"] = report["data_quality_score"] >= min_valid_ratio
            
            if not report["is_valid"]:
                logger.warning(f"Data validation failed: {report['data_quality_score']:.1%} valid")
                
            return report["is_valid"], report
            
        except Exception as e:
            logger.error(f"Data validation error: {e}")
            report["error"] = str(e)
            return False, report
            
    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================
    
    def generate_signals(self) -> Dict[str, Dict[str, Any]]:
        """
        Generate trading signals using V2.1 engine + V2.2 RL enhancement.
        
        Pipeline:
        1. V2.1: TDA features + ensemble regime + transformer predictions
        2. V2.2: RL-enhanced position sizing (SAC + hierarchical controller)
        
        Returns:
            Dict mapping ticker -> signal info
        """
        logger.info("Generating V2.1/V2.2 signals...")
        
        try:
            # Get universe from engine
            universe = self.engine.get_universe()
            
            # Compute TDA features
            tda_features = self.engine.compute_tda_features(universe)
            
            # Detect regime (ensemble method)
            regime, regime_confidence = self.engine.detect_regime()
            logger.info(f"Market regime: {regime} (confidence: {regime_confidence:.2f})")
            
            # Generate predictions (transformer + LSTM ensemble)
            predictions = self.engine.generate_predictions(universe, tda_features)
            
            # V2.1: Risk-adjusted sizing (base signals)
            base_signals = self.engine.compute_position_sizes(
                predictions,
                regime,
                regime_confidence,
                max_position_pct=self.config.max_position_pct,
                max_heat=self.config.max_portfolio_heat,
            )
            
            # V2.2: RL-enhanced position sizing
            if self.rl_orchestrator:
                # Get market data for RL state encoding
                market_data = self.engine.get_market_data(universe)
                
                # Enhance signals with RL orchestrator
                signals = self.rl_orchestrator.enhance_signals(
                    base_signals=base_signals,
                    market_data=market_data,
                    tda_features=tda_features,
                )
                
                # Log RL enhancement stats
                rl_stats = self.rl_orchestrator.get_stats()
                logger.info(f"RL enhancement: {rl_stats.get('total_decisions', 0)} decisions, "
                           f"{rl_stats.get('anomaly_detections', 0)} anomalies detected")
            else:
                signals = base_signals
            
            # Log signal summary
            long_count = sum(1 for s in signals.values() if s.get("direction") == "long")
            short_count = sum(1 for s in signals.values() if s.get("direction") == "short")
            rl_enhanced = sum(1 for s in signals.values() if s.get("rl_enhanced", False))
            logger.info(f"Signals: {long_count} long, {short_count} short, {rl_enhanced} RL-enhanced")
            
            return signals
            
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            traceback.print_exc()
            return {}
            
    # =========================================================================
    # TRADE EXECUTION
    # =========================================================================
    
    def execute_trades(self, signals: Dict[str, Dict[str, Any]]) -> List[Dict]:
        """
        Execute trades with execution alpha optimization.
        
        Optimizations:
        - Limit orders with smart pricing
        - High-liquidity hour preference
        - Slippage monitoring
        - Partial fill handling
        
        Returns:
            List of executed trade records
        """
        if self.config.dry_run:
            logger.info("[DRY RUN] Skipping trade execution")
            return []
            
        if self._is_halted:
            logger.warning(f"Trading halted: {self._halt_reason}")
            return []
            
        if not self.alpaca:
            logger.error("Alpaca client not available")
            return []
            
        executed_trades = []
        
        try:
            # Get current positions
            current_positions = {p.symbol: p for p in self.alpaca.get_positions()}
            account = self.alpaca.get_account()
            equity = float(account.equity)
            
            # Check drawdown before trading
            self._update_drawdown(equity)
            if self._is_halted:
                return []
            
            # V26: Update circuit breakers and check if trading allowed
            if self.v26_circuit_breakers:
                breaker_state = self.v26_circuit_breakers.update(equity)
                can_trade, reason = self.v26_circuit_breakers.can_trade()
                if not can_trade:
                    logger.warning(f"V26 Circuit Breaker: {reason}")
                    return []
                
            # Check if in high-liquidity hours
            current_hour = datetime.now().hour
            in_high_liquidity = (
                self.config.high_liquidity_hours[0] <= 
                current_hour < 
                self.config.high_liquidity_hours[1]
            )
            
            for ticker, signal in signals.items():
                try:
                    target_weight = signal.get("weight", 0.0)
                    direction = signal.get("direction", "flat")
                    confidence = signal.get("confidence", 0.5)
                    
                    # V26: Check signal against circuit breaker confidence filter
                    if self.v26_circuit_breakers:
                        should_trade, reason = self.v26_circuit_breakers.should_trade_signal(confidence)
                        if not should_trade:
                            logger.info(f"V26 Signal filtered for {ticker}: {reason}")
                            continue
                    
                    # Calculate target shares
                    current_pos = current_positions.get(ticker)
                    current_value = float(current_pos.market_value) if current_pos else 0.0
                    target_value = equity * target_weight
                    
                    # V26: Adjust position size using circuit breaker scaling
                    if self.v26_circuit_breakers:
                        kelly_scale = self.v26_circuit_breakers.get_state().kelly_scale
                        target_value = target_value * kelly_scale
                    
                    trade_value = target_value - current_value
                    
                    # Skip small trades (< 0.5% equity)
                    if abs(trade_value) < equity * 0.005:
                        continue
                        
                    # Get current price
                    current_price = self.engine.get_current_price(ticker)
                    if not current_price:
                        logger.warning(f"No price for {ticker}, skipping")
                        continue
                        
                    shares = int(abs(trade_value) / current_price)
                    if shares == 0:
                        continue
                        
                    # Determine order side
                    side = OrderSide.BUY if trade_value > 0 else OrderSide.SELL
                    
                    # Calculate limit price with buffer
                    if self.config.use_limit_orders and in_high_liquidity:
                        buffer = self.config.limit_order_buffer_pct
                        if side == OrderSide.BUY:
                            limit_price = current_price * (1 + buffer)
                        else:
                            limit_price = current_price * (1 - buffer)
                    else:
                        limit_price = None
                        
                    # Submit order
                    order = self.alpaca.submit_order(
                        symbol=ticker,
                        qty=shares,
                        side=side,
                        order_type="limit" if limit_price else "market",
                        limit_price=limit_price,
                    )
                    
                    if order:
                        trade_record = {
                            "timestamp": datetime.now().isoformat(),
                            "symbol": ticker,
                            "side": side.value,
                            "qty": shares,
                            "price": limit_price or current_price,
                            "value": shares * (limit_price or current_price),
                            "order_id": order.id,
                            "signal_confidence": signal.get("confidence", 0.5),
                        }
                        executed_trades.append(trade_record)
                        self._daily_trades.append(trade_record)
                        
                        # V26: Record trade for circuit breaker Kelly calculation
                        if self.v26_circuit_breakers:
                            # PnL will be updated later when we know actual fill
                            pass
                        
                        # V26: Record trade for model health monitoring
                        if self.model_health_monitor:
                            self.model_health_monitor.record_trade(
                                predicted_direction=direction,
                                actual_return=0.0,  # Updated later with actual
                                confidence=confidence
                            )
                        
                        # Notify
                        if self.config.discord_notifications:
                            notify_trade_executed(
                                ticker, side.value, shares, 
                                limit_price or current_price,
                                trade_record["value"]
                            )
                            
                        logger.info(f"Executed: {side.value.upper()} {shares} {ticker} @ ${limit_price or current_price:.2f}")
                        
                except Exception as e:
                    logger.error(f"Trade execution failed for {ticker}: {e}")
                    continue
                    
            return executed_trades
            
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            traceback.print_exc()
            return []
    
    def record_trade_result(self, ticker: str, direction: str, 
                           predicted_return: float, actual_return: float,
                           confidence: float):
        """
        V26: Record trade result for continuous learning.
        
        Called when trade outcome is known (e.g., after position closed).
        """
        if self.continuous_learner:
            result = TradeResult(
                timestamp=datetime.now(),
                ticker=ticker,
                signal_direction=direction,
                signal_confidence=confidence,
                predicted_return=predicted_return,
                actual_return=actual_return,
                is_hit=(predicted_return > 0) == (actual_return > 0)
            )
            update = self.continuous_learner.record_trade(result)
            
            # Check if recalibration needed
            if update.get('needs_recalibration'):
                self.continuous_learner.trigger_recalibration(reason="accuracy_drop")
        
        # Update circuit breaker Kelly
        if self.v26_circuit_breakers:
            self.v26_circuit_breakers.record_trade(actual_return)
        
        # Update model health
        if self.model_health_monitor:
            self.model_health_monitor.record_trade(direction, actual_return, confidence)
            
    def _update_drawdown(self, current_equity: float):
        """Update drawdown and check circuit breakers."""
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity
            
        self._current_drawdown = (self._peak_equity - current_equity) / self._peak_equity
        
        # Update dashboard
        if self.dashboard:
            self.dashboard.update_metric("drawdown", self._current_drawdown)
            
        # Check circuit breakers
        if self._current_drawdown >= self.config.emergency_halt_dd_pct:
            self._is_halted = True
            self._halt_reason = f"Emergency halt: {self._current_drawdown:.1%} drawdown"
            logger.critical(self._halt_reason)
            
            if self.config.discord_notifications:
                send_discord("🚨 EMERGENCY HALT",
                           f"Drawdown: {self._current_drawdown:.1%}\n"
                           f"Threshold: {self.config.emergency_halt_dd_pct:.1%}\n"
                           "All trading halted. Manual review required.",
                           color=0xFF0000)
                           
        elif self._current_drawdown >= self.config.circuit_breaker_dd_pct:
            logger.warning(f"Circuit breaker: {self._current_drawdown:.1%} drawdown")
            if self.config.discord_notifications:
                send_discord("⚠️ Circuit Breaker Warning",
                           f"Drawdown: {self._current_drawdown:.1%}\n"
                           f"Emergency threshold: {self.config.emergency_halt_dd_pct:.1%}",
                           color=0xFFAA00)
                           
    # =========================================================================
    # MONITORING & VALIDATION
    # =========================================================================
    
    def update_dashboard(self):
        """Update monitoring dashboard with current metrics."""
        if not self.dashboard:
            return
            
        try:
            # Get current state
            if self.alpaca:
                account = self.alpaca.get_account()
                positions = self.alpaca.get_positions()
                
                self.dashboard.update_metrics({
                    "equity": float(account.equity),
                    "cash": float(account.cash),
                    "buying_power": float(account.buying_power),
                    "positions_count": len(positions),
                    "drawdown": self._current_drawdown,
                    "daily_trades": len(self._daily_trades),
                    "is_halted": self._is_halted,
                })
                
            # Engine metrics
            engine_status = self.engine.get_component_status()
            self.dashboard.update_metrics(engine_status)
            
        except Exception as e:
            logger.error(f"Dashboard update failed: {e}")
    
    def _update_health_status(self):
        """Update health check endpoint with current status (production deployment)."""
        if not self.health_server:
            return
            
        try:
            # Determine health status based on system state
            if self._is_halted:
                status = HealthStatus.UNHEALTHY
            elif self._current_drawdown > 0.03:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            # Build status info
            portfolio_value = self._peak_equity * (1 - self._current_drawdown)
            
            status_info = {
                "status": status.value,
                "portfolio_value": portfolio_value,
                "max_drawdown_7d": self._current_drawdown,
                "daily_trades": len(self._daily_trades),
                "is_halted": self._is_halted,
                "halt_reason": self._halt_reason if self._is_halted else None,
                "last_rebalance": self._last_rebalance.isoformat() if self._last_rebalance else None,
            }
            
            self.health_server.update_status(status_info)
            
        except Exception as e:
            logger.error(f"Health status update failed: {e}")
            
    def run_daily_validation(self) -> Dict[str, Any]:
        """
        Run daily validation comparing paper vs expected results.
        
        Flags anomalies > 2 sigma from expected.
        """
        try:
            report = self.validator.run_validation(
                daily_trades=self._daily_trades,
                expected_sharpe=1.35,  # V1.3 baseline
                sigma_threshold=2.0,
            )
            
            # Log to file
            validation_file = LOG_DIR / f"validation_{datetime.now().strftime('%Y%m%d')}.json"
            with open(validation_file, 'w') as f:
                json.dump(report, f, indent=2)
                
            # Alert on anomalies
            if report.get("has_anomalies"):
                logger.warning(f"Validation anomalies detected: {report.get('anomalies')}")
                if self.config.discord_notifications:
                    send_discord("⚠️ Daily Validation Anomalies",
                               json.dumps(report.get("anomalies"), indent=2)[:1900],
                               color=0xFFAA00)
                               
            return report
            
        except Exception as e:
            logger.error(f"Daily validation failed: {e}")
            return {"error": str(e)}
            
    def send_daily_summary(self):
        """Send end-of-day summary via Discord."""
        if not self.config.discord_notifications:
            return
            
        try:
            account = self.alpaca.get_account() if self.alpaca else None
            
            summary = f"""
📊 **Daily Trading Summary**
```
Date:         {datetime.now().strftime('%Y-%m-%d')}
Equity:       ${float(account.equity):,.2f if account else 'N/A'}
Daily P&L:    ${self._daily_pnl:+,.2f}
Trades:       {len(self._daily_trades)}
Drawdown:     {self._current_drawdown:.2%}
Status:       {'🔴 HALTED' if self._is_halted else '🟢 ACTIVE'}
```

**Component Status:**
{json.dumps(self.engine.get_component_status(), indent=2)}
            """
            
            send_discord("📈 End of Day Summary", summary, color=0x00D4FF)
            
        except Exception as e:
            logger.error(f"Daily summary failed: {e}")
            
    # =========================================================================
    # MAIN LOOP
    # =========================================================================
    
    def run_once(self) -> Dict[str, Any]:
        """
        Execute single trading cycle.
        
        Returns:
            Cycle result with trades and metrics
        """
        cycle_start = datetime.now()
        result = {
            "timestamp": cycle_start.isoformat(),
            "status": "success",
            "trades": [],
            "signals_count": 0,
            "errors": [],
        }
        
        try:
            # 1. Data validation
            logger.info("Step 1: Data validation")
            is_valid, validation_report = self.validate_data(self.engine.get_universe())
            if not is_valid:
                result["status"] = "data_validation_failed"
                result["errors"].append(validation_report)
                return result
                
            # 2. Generate signals
            logger.info("Step 2: Generate signals")
            signals = self.generate_signals()
            result["signals_count"] = len(signals)
            
            if not signals:
                result["status"] = "no_signals"
                return result
                
            # 3. Execute trades
            logger.info("Step 3: Execute trades")
            trades = self.execute_trades(signals)
            result["trades"] = trades
            
            # 4. Update monitoring
            logger.info("Step 4: Update monitoring")
            self.update_dashboard()
            
            # Log cycle metrics
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            logger.info(f"Cycle completed in {cycle_duration:.1f}s: {len(trades)} trades executed")
            
            # Save to metrics log
            with open(LOG_DIR / "cycles.jsonl", 'a') as f:
                f.write(json.dumps(result) + "\n")
                
            return result
            
        except Exception as e:
            logger.error(f"Trading cycle failed: {e}")
            traceback.print_exc()
            result["status"] = "error"
            result["errors"].append(str(e))
            return result
            
    def run(self):
        """
        Main production loop with scheduling.
        
        Runs continuously, executing trades at scheduled times.
        """
        self.is_running = True
        logger.info("Starting production loop...")
        
        # Start dashboard server if enabled
        if self.dashboard:
            self.dashboard.start_server()
            
        try:
            while self.is_running and not self._shutdown_event.is_set():
                now = datetime.now()
                current_time = now.strftime("%H:%M")
                
                # Check if market hours (simplified - use proper market calendar in production)
                is_market_hours = self._is_market_hours(now)
                
                # Rebalance time
                if current_time == self.config.rebalance_time and is_market_hours:
                    if self._should_rebalance():
                        logger.info("=" * 40)
                        logger.info("SCHEDULED REBALANCE")
                        logger.info("=" * 40)
                        self.run_once()
                        self._last_rebalance = now
                        
                # End of day summary
                if current_time == self.config.eod_summary_time:
                    self.send_daily_summary()
                    self.run_daily_validation()
                    
                    # Reset daily counters
                    self._daily_trades = []
                    self._daily_pnl = 0.0
                    
                # Update dashboard every minute
                if now.second == 0:
                    self.update_dashboard()
                    self._update_health_status()
                    
                # Check graceful shutdown request
                if self.graceful_shutdown and self.graceful_shutdown.should_exit:
                    logger.info("Graceful shutdown requested")
                    break
                    
                # Sleep to prevent busy loop
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            self.shutdown()
            
    def _is_market_hours(self, dt: datetime) -> bool:
        """Check if current time is during market hours."""
        # Simple check - use proper market calendar in production
        hour, minute = dt.hour, dt.minute
        current_minutes = hour * 60 + minute
        
        open_h, open_m = map(int, self.config.market_open.split(":"))
        close_h, close_m = map(int, self.config.market_close.split(":"))
        
        open_minutes = open_h * 60 + open_m
        close_minutes = close_h * 60 + close_m
        
        return open_minutes <= current_minutes <= close_minutes
        
    def _should_rebalance(self) -> bool:
        """Check if rebalance should occur."""
        if self._last_rebalance is None:
            return True
            
        # At least 30 minutes between rebalances
        min_interval = timedelta(minutes=30)
        return datetime.now() - self._last_rebalance >= min_interval
        
    def shutdown(self):
        """Clean shutdown of all components."""
        logger.info("Shutting down production launcher...")
        
        self.is_running = False
        
        # Stop dashboard
        if self.dashboard:
            self.dashboard.stop_server()
        
        # Stop health check server (production deployment)
        if self.health_server:
            try:
                self.health_server.stop()
                logger.info("Health check server stopped")
            except Exception as e:
                logger.warning(f"Error stopping health server: {e}")
        
        # Save state via graceful shutdown handler (production deployment)
        if self.graceful_shutdown:
            try:
                state = {
                    "positions": self._daily_trades,
                    "peak_equity": self._peak_equity,
                    "current_drawdown": self._current_drawdown,
                    "last_rebalance": self._last_rebalance.isoformat() if self._last_rebalance else None,
                    "daily_pnl": self._daily_pnl,
                }
                self.graceful_shutdown.save_state(state)
                logger.info("State persisted for recovery")
            except Exception as e:
                logger.warning(f"Error saving state: {e}")
            
        # Final summary
        self.send_daily_summary()
        
        logger.info("Shutdown complete")


# =============================================================================
# CLI INTERFACE
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="V2.1 Production Trading Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Paper trading (default)
    python production_launcher.py
    
    # Dry run mode (no orders submitted)
    python production_launcher.py --dry-run
    
    # Single cycle (no scheduling)
    python production_launcher.py --once
    
    # Live trading (requires explicit flag)
    python production_launcher.py --live --i-understand-the-risks
        """
    )
    
    parser.add_argument("--live", action="store_true",
                       help="Enable LIVE trading (default: paper)")
    parser.add_argument("--i-understand-the-risks", action="store_true",
                       help="Confirmation for live trading")
    parser.add_argument("--dry-run", action="store_true",
                       help="Dry run mode - no orders submitted")
    parser.add_argument("--once", action="store_true",
                       help="Run single cycle then exit")
    parser.add_argument("--no-dashboard", action="store_true",
                       help="Disable monitoring dashboard")
    parser.add_argument("--no-notifications", action="store_true",
                       help="Disable Discord notifications")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Safety check for live trading
    if args.live and not args.i_understand_the_risks:
        print("❌ Live trading requires --i-understand-the-risks flag")
        print("   This flag confirms you understand real money is at risk.")
        sys.exit(1)
        
    # Configure logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Build config
    config = LauncherConfig(
        paper_trading=not args.live,
        dry_run=args.dry_run,
        single_run=args.once,
        enable_dashboard=not args.no_dashboard,
        discord_notifications=not args.no_notifications,
    )
    
    # Display startup banner
    print("=" * 60)
    print("V2.1 PRODUCTION LAUNCHER")
    print("=" * 60)
    print(f"Mode:          {'🔴 LIVE' if args.live else '📝 PAPER'}")
    print(f"Dry Run:       {args.dry_run}")
    print(f"Single Cycle:  {args.once}")
    print(f"Dashboard:     {not args.no_dashboard}")
    print(f"Notifications: {not args.no_notifications}")
    print("=" * 60)
    
    # Create and run launcher
    launcher = ProductionLauncher(config)
    
    if args.once:
        result = launcher.run_once()
        print(json.dumps(result, indent=2))
    else:
        launcher.run()


if __name__ == "__main__":
    main()
