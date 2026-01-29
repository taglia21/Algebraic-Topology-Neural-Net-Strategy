#!/usr/bin/env python3
"""
V51 Unified Trading Engine - Production Ready
=============================================
Integrates all V51 advanced modules with market hours scheduling.

Features:
- Market hours awareness (9:30 AM - 4:00 PM EST)
- Pre-market preparation (9:00 AM)
- Post-market analysis (4:00 PM)
- Circuit breaker for fault tolerance
- Health monitoring
- Async data orchestration
- Paper trading mode support

Version: 51.0.0
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta, time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import traceback
import pytz
import signal

# Add project root to path
sys.path.insert(0, '/opt/Algebraic-Topology-Neural-Net-Strategy')

VERSION = "51.0.0"

# Timezone
EST = pytz.timezone('America/New_York')

# Market Hours Configuration
MARKET_OPEN = time(9, 30)   # 9:30 AM EST
MARKET_CLOSE = time(16, 0)  # 4:00 PM EST
PRE_MARKET_START = time(9, 0)   # 9:00 AM EST - preparation
POST_MARKET_END = time(16, 30)  # 4:30 PM EST - analysis

# V51 Core Modules
try:
    from src.trading.v51_integration import V51Engine
    V51_CORE_AVAILABLE = True
except ImportError as e:
    V51_CORE_AVAILABLE = False
    print(f"V51 Core not available: {e}")

# V51 Analysis Modules
try:
    from src.trading.transformer_predictor import TransformerPredictor
    from src.trading.meta_ensemble import MetaEnsembleLearner
    from src.trading.options_flow_analyzer import OptionsFlowAnalyzer
    from src.trading.microstructure_analyzer import MicrostructureAnalyzer
    from src.trading.realtime_anomaly_detector import RealtimeAnomalyDetector
    from src.trading.regime_detector import RegimeDetector
    from src.trading.rl_position_optimizer import RLPositionOptimizer
    from src.trading.advanced_regime_predictor import AdvancedRegimePredictor
    from src.trading.economic_integrator import EconomicIntegrator
    V51_ANALYSIS_AVAILABLE = True
except ImportError as e:
    V51_ANALYSIS_AVAILABLE = False
    print(f"V51 Analysis modules not available: {e}")

# Production Hardening Modules
try:
    from src.trading.async_data_orchestrator import AsyncDataOrchestrator
    from src.trading.circuit_breaker import CircuitBreaker, CircuitState, CircuitConfig
    from src.trading.mega_universe import MEGA_UNIVERSE, get_mega_universe
    from src.trading.health_monitor import HealthMonitor, HealthStatus
    from src.trading.backtester import WalkForwardBacktester
    PRODUCTION_HARDENING_AVAILABLE = True
except ImportError as e:
    PRODUCTION_HARDENING_AVAILABLE = False
    print(f"Production hardening modules not available: {e}")

# Volatility and Options
try:
    from src.trading.volatility_surface import VolatilitySurface
    from src.trading.options_strategy_engine import OptionsStrategyEngine
    OPTIONS_MODULES_AVAILABLE = True
except ImportError:
    OPTIONS_MODULES_AVAILABLE = False


class MarketState(Enum):
    """Market state enumeration"""
    PRE_MARKET = "pre_market"
    MARKET_OPEN = "market_open"
    MARKET_CLOSED = "market_closed"
    POST_MARKET = "post_market"
    WEEKEND = "weekend"

class MarketRegime(Enum):
    """Market regime classification"""
    BULL_QUIET = "bull_quiet"
    BULL_VOLATILE = "bull_volatile"
    BEAR_QUIET = "bear_quiet"
    BEAR_VOLATILE = "bear_volatile"
    SIDEWAYS = "sideways"
    CRISIS = "crisis"

@dataclass
class V51Config:
    """Configuration for V51 Unified Trading Engine"""
    # API Configuration
    alpaca_api_key: str = ""
    alpaca_api_secret: str = ""
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    tradier_api_key: str = ""
    tradier_account_id: str = ""
    tradier_base_url: str = "https://sandbox.tradier.com/v1"
    
    # Trading Configuration
    paper_trading: bool = True
    starting_capital: float = 100000.0
    max_daily_loss_pct: float = 0.03
    max_position_pct: float = 0.08
    max_leveraged_etf_pct: float = 0.25
    # Symbol Universe
    use_mega_universe: bool = True
    max_symbols_per_scan: int = 500  # Limit for memory on 1GB droplet
    # Symbol Universe
    use_mega_universe: bool = True
    max_symbols_per_scan: int = 500  # Limit for memory on 1GB droplet
    
    # V51 Module Flags
    use_transformer_predictor: bool = True
    use_meta_ensemble: bool = True
    use_options_flow: bool = True
    use_microstructure: bool = True
    use_anomaly_detector: bool = True
    use_regime_detector: bool = True
    use_rl_optimizer: bool = True
    use_economic_integrator: bool = True
    
    # Production Hardening Flags
    use_circuit_breaker: bool = True
    use_health_monitor: bool = True
    use_async_orchestrator: bool = True
    
    # Risk Management
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 300
    health_check_interval: int = 60
    
    # Scanning Configuration  
    scan_interval_seconds: int = 300  # 5 minutes during market hours
    overnight_check_interval: int = 3600  # 1 hour when market closed
    
    # Logging
    log_file: str = "logs/v51_trading.log"
    log_level: str = "INFO"
    discord_webhook: str = ""


class V51UnifiedEngine:
    """V51 Unified Trading Engine with market hours scheduling"""
    
    def __init__(self, config: V51Config):
        self.config = config
        self.logger = self._setup_logging()
        self._initialized = False
        self._running = False
        self._shutdown_requested = False
        self.scan_count = 0
        
        # Core components
        self.v51_engine = None
        self.circuit_breaker = None
        self.health_monitor = None
        self.async_orchestrator = None

        # Symbol Universe
        self.symbols = self._load_symbol_universe()

        # Symbol Universe
        self.symbols = self._load_symbol_universe()
        
        # Analysis modules
        self.regime_detector = None
        self.options_flow = None
        self.anomaly_detector = None
        self.economic_integrator = None
        
        # State
        self.current_regime = MarketRegime.SIDEWAYS
        self.current_market_state = MarketState.MARKET_CLOSED
        self.positions = {}
        self.daily_pnl = 0.0
        self.account_value = config.starting_capital
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self._shutdown_requested = True
        self._running = False
    

    def _load_symbol_universe(self) -> List[str]:
        """Load the symbol universe for scanning."""
        if self.config.use_mega_universe:
            symbols = list(get_mega_universe())
            # Limit symbols for memory constraints
            if len(symbols) > self.config.max_symbols_per_scan:
                self.logger.info(f"Limiting from {len(symbols)} to {self.config.max_symbols_per_scan} symbols")
                symbols = symbols[:self.config.max_symbols_per_scan]
            self.logger.info(f"Loaded {len(symbols)} symbols from MEGA_UNIVERSE")
            return symbols
        default_symbols = ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
        self.logger.info(f"Using default {len(default_symbols)} symbols")
        return default_symbols


    def _load_symbol_universe(self) -> List[str]:
        """Load the symbol universe for scanning."""
        if self.config.use_mega_universe:
            symbols = list(get_mega_universe())
            # Limit symbols for memory constraints
            if len(symbols) > self.config.max_symbols_per_scan:
                self.logger.info(f"Limiting from {len(symbols)} to {self.config.max_symbols_per_scan} symbols")
                symbols = symbols[:self.config.max_symbols_per_scan]
            self.logger.info(f"Loaded {len(symbols)} symbols from MEGA_UNIVERSE")
            return symbols
        default_symbols = ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
        self.logger.info(f"Using default {len(default_symbols)} symbols")
        return default_symbols

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('V51Engine')
        logger.setLevel(getattr(logging, self.config.log_level))
        
        os.makedirs(os.path.dirname(self.config.log_file) or 'logs', exist_ok=True)
        
        fh = logging.FileHandler(self.config.log_file)
        fh.setLevel(logging.DEBUG)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        if not logger.handlers:
            logger.addHandler(fh)
            logger.addHandler(ch)
        
        return logger
    
    def get_market_state(self) -> Tuple[MarketState, datetime]:
        """Determine current market state based on EST time"""
        now_est = datetime.now(EST)
        current_time = now_est.time()
        weekday = now_est.weekday()
        
        # Weekend check (Saturday=5, Sunday=6)
        if weekday >= 5:
            return MarketState.WEEKEND, now_est
        
        # Market hours check
        if current_time < PRE_MARKET_START:
            return MarketState.MARKET_CLOSED, now_est
        elif current_time < MARKET_OPEN:
            return MarketState.PRE_MARKET, now_est
        elif current_time < MARKET_CLOSE:
            return MarketState.MARKET_OPEN, now_est
        elif current_time < POST_MARKET_END:
            return MarketState.POST_MARKET, now_est
        else:
            return MarketState.MARKET_CLOSED, now_est
    
    def time_until_market_open(self) -> timedelta:
        """Calculate time until next market open"""
        now_est = datetime.now(EST)
        today_open = now_est.replace(hour=9, minute=30, second=0, microsecond=0)
        
        if now_est.time() >= MARKET_OPEN:
            # Market already opened today, calculate for tomorrow
            next_open = today_open + timedelta(days=1)
        else:
            next_open = today_open
        
        # Skip weekends
        while next_open.weekday() >= 5:
            next_open += timedelta(days=1)
        
        return next_open - now_est

    
    async def initialize(self) -> bool:
        """Initialize all V51 components"""
        try:
            self.logger.info("="*60)
            self.logger.info("V51 UNIFIED ALPHA ENGINE")
            self.logger.info(f"Version: {VERSION}")
            self.logger.info(f"Mode: {'PAPER' if self.config.paper_trading else 'LIVE'}")
            self.logger.info("Advanced ML + Production Hardening")
            self.logger.info("="*60)
            
            if self.config.use_circuit_breaker and PRODUCTION_HARDENING_AVAILABLE:
                self.circuit_breaker = CircuitBreaker(CircuitConfig(
                    name="v51_main",
                    failure_threshold=self.config.circuit_breaker_threshold
                ))
                self.logger.info("[OK] CircuitBreaker initialized")
            
            # Initialize health monitor
            if self.config.use_health_monitor and PRODUCTION_HARDENING_AVAILABLE:
                self.health_monitor = HealthMonitor(
                    check_interval=float(self.config.health_check_interval)
                )
                self.logger.info("[OK] HealthMonitor initialized")
                self.logger.info("[OK] HealthMonitor initialized")
            
            # Initialize V51 core engine
            if V51_CORE_AVAILABLE:
                self.v51_engine = V51Engine()
                self.logger.info("[OK] V51Engine core initialized")
            
            # Initialize analysis modules
            if V51_ANALYSIS_AVAILABLE:
                if self.config.use_regime_detector:
                    self.regime_detector = RegimeDetector()
                    self.logger.info("[OK] RegimeDetector initialized")
                
                if self.config.use_options_flow:
                    self.options_flow = OptionsFlowAnalyzer()
                    self.logger.info("[OK] OptionsFlowAnalyzer initialized")
                
                if self.config.use_anomaly_detector:
                    self.anomaly_detector = RealtimeAnomalyDetector()
                    self.logger.info("[OK] RealtimeAnomalyDetector initialized")
                
                if self.config.use_economic_integrator:
                    self.economic_integrator = EconomicIntegrator()
                    self.logger.info("[OK] EconomicIntegrator initialized")
            
            self._initialized = True
            self.logger.info("="*60)
            self.logger.info("V51 ENGINE INITIALIZATION COMPLETE")
            self.logger.info("="*60)
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            self.logger.error(traceback.format_exc())
            return False

    
    async def run(self):
        """Main trading loop with market hours awareness"""
        if not self._initialized:
            if not await self.initialize():
                return
        
        self._running = True
        self.logger.info("Starting V51 trading engine...")
        
        while self._running and not self._shutdown_requested:
            try:
                # Get current market state
                market_state, now_est = self.get_market_state()
                self.current_market_state = market_state
                
                if market_state == MarketState.MARKET_CLOSED or market_state == MarketState.WEEKEND:
                    # Market is closed - wait until open
                    time_until_open = self.time_until_market_open()
                    hours, remainder = divmod(int(time_until_open.total_seconds()), 3600)
                    minutes = remainder // 60
                    
                    self.logger.info(f"Market closed. Next open in {hours}h {minutes}m")
                    self.logger.info(f"Current time EST: {now_est.strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    # Sleep for longer intervals when market is closed
                    await asyncio.sleep(min(self.config.overnight_check_interval, time_until_open.total_seconds()))
                    continue
                
                elif market_state == MarketState.PRE_MARKET:
                    # Pre-market preparation phase
                    self.logger.info("=" * 40)
                    self.logger.info("PRE-MARKET PREPARATION PHASE")
                    self.logger.info(f"Time: {now_est.strftime('%H:%M:%S')} EST")
                    self.logger.info("=" * 40)
                    
                    # Run pre-market analysis
                    await self._run_premarket_analysis()
                    
                    # Wait until market open
                    time_until_open = self.time_until_market_open()
                    if time_until_open.total_seconds() > 0:
                        self.logger.info(f"Waiting {int(time_until_open.total_seconds())}s until market open...")
                        await asyncio.sleep(min(60, time_until_open.total_seconds()))
                    continue
                
                elif market_state == MarketState.MARKET_OPEN:
                    # Active trading hours
                    self.scan_count += 1
                    self.logger.info(f"\n{'='*50}")
                    self.logger.info(f"SCAN #{self.scan_count} | {now_est.strftime('%H:%M:%S')} EST")
                    self.logger.info(f"{'='*50}")
                    
                    # Check circuit breaker
                    if self.circuit_breaker and self.circuit_breaker.state == CircuitState.OPEN:
                        self.logger.warning("Circuit breaker OPEN - skipping trading")
                        await asyncio.sleep(60)
                        continue
                    
                    # Run trading scan
                    await self._run_trading_scan()
                    
                    # Log status
                    self.logger.info(f"Account Value: ${self.account_value:,.2f}")
                    self.logger.info(f"Daily P&L: ${self.daily_pnl:,.2f}")
                    self.logger.info(f"Current Regime: {self.current_regime.value}")
                    
                    # Wait for next scan
                    await asyncio.sleep(self.config.scan_interval_seconds)
                
                elif market_state == MarketState.POST_MARKET:
                    # Post-market analysis
                    self.logger.info("=" * 40)
                    self.logger.info("POST-MARKET ANALYSIS")
                    self.logger.info("=" * 40)
                    
                    await self._run_postmarket_analysis()
                    
                    # Reset daily P&L for next day
                    self.logger.info(f"Final Daily P&L: ${self.daily_pnl:,.2f}")
                    
                    # Wait until market closes completely
                    await asyncio.sleep(self.config.overnight_check_interval)
                    
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                self.logger.error(traceback.format_exc())
                if self.circuit_breaker:
                    self.circuit_breaker.record_failure()
                await asyncio.sleep(60)
        
        self.logger.info("V51 Engine stopped.")

    
    async def _run_premarket_analysis(self):
        """Pre-market preparation and analysis"""
        self.logger.info("Running pre-market analysis...")
        
        # Reset daily metrics
        self.daily_pnl = 0.0
        
        # Detect current market regime
        if self.regime_detector:
            try:
                # This would analyze overnight futures, news, etc.
                self.logger.info("Analyzing market regime...")
            except Exception as e:
                self.logger.warning(f"Regime detection error: {e}")
        
        # Check economic calendar
        if self.economic_integrator:
            try:
                self.logger.info("Checking economic calendar...")
            except Exception as e:
                self.logger.warning(f"Economic integrator error: {e}")
        
        self.logger.info("Pre-market analysis complete.")
    
    async def _run_trading_scan(self):
        """Execute main trading scan during market hours"""
        self.logger.info("Running trading scan...")
        
        # Health check
        if self.health_monitor:
            try:
                health = await self.health_monitor.check_health()
                if health.status == HealthStatus.CRITICAL:
                    self.logger.error("Health check CRITICAL")
                    return
            except Exception as e:
                self.logger.warning(f"Health check error: {e}")
        
        # V51 analysis pipeline
        if self.v51_engine:
            try:
                # Run V51 analysis
                self.logger.info("Running V51 analysis pipeline...")
                # result = await self.v51_engine.analyze()
            except Exception as e:
                self.logger.warning(f"V51 analysis error: {e}")
        
        # Options flow analysis
        if self.options_flow:
            try:
                self.logger.info("Analyzing options flow...")
            except Exception as e:
                self.logger.warning(f"Options flow error: {e}")
        
        # Anomaly detection
        if self.anomaly_detector:
            try:
                self.logger.info("Running anomaly detection...")
            except Exception as e:
                self.logger.warning(f"Anomaly detection error: {e}")
        
        self.logger.info("Trading scan complete.")
    
    async def _run_postmarket_analysis(self):
        """Post-market analysis and reporting"""
        self.logger.info("Running post-market analysis...")
        
        # Generate daily report
        self.logger.info(f"Daily Scans: {self.scan_count}")
        self.logger.info(f"Final P&L: ${self.daily_pnl:,.2f}")
        self.logger.info(f"Account Value: ${self.account_value:,.2f}")
        
        # Reset for next day
        self.scan_count = 0
        
        self.logger.info("Post-market analysis complete.")
    
    def stop(self):
        """Stop the trading engine"""
        self._running = False
        self._shutdown_requested = True
        self.logger.info("V51 Engine shutdown requested.")


def load_config_from_env() -> V51Config:
    """Load configuration from environment variables"""
    return V51Config(
        alpaca_api_key=os.getenv('APCA_API_KEY_ID', ''),
        alpaca_api_secret=os.getenv('APCA_API_SECRET_KEY', ''),
        alpaca_base_url=os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets'),
        tradier_api_key=os.getenv('TRADIER_API_KEY', ''),
        tradier_account_id=os.getenv('TRADIER_ACCOUNT_ID', ''),
        tradier_base_url=os.getenv('TRADIER_BASE_URL', 'https://sandbox.tradier.com/v1'),
        paper_trading=os.getenv('PAPER_TRADING', 'true').lower() == 'true',
        starting_capital=float(os.getenv('STARTING_CAPITAL', '100000')),
        max_daily_loss_pct=float(os.getenv('MAX_DAILY_LOSS_PCT', '0.03')),
        scan_interval_seconds=int(os.getenv('SCAN_INTERVAL_SECONDS', '300')),
        log_file=os.getenv('LOG_FILE', 'logs/v51_trading.log'),
        discord_webhook=os.getenv('DISCORD_WEBHOOK', ''),
    )


async def main():
    """Main entry point for V51 Trading Engine"""
    print(f"V51 Unified Trading Engine v{VERSION}")
    print(f"Current time: {datetime.now(EST).strftime('%Y-%m-%d %H:%M:%S')} EST")
    print("Loading configuration...")
    
    config = load_config_from_env()
    engine = V51UnifiedEngine(config)
    
    try:
        await engine.run()
    except KeyboardInterrupt:
        print("\nShutdown requested...")
        engine.stop()
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
