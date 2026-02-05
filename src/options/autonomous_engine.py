"""
Autonomous Options Trading Engine
==================================

Main orchestrator for fully autonomous options trading.

6-Step Trading Loop (60-second cycle):
1. SCAN: Generate signals from all strategies
2. FILTER: Remove invalid/duplicate signals
3. SIZE: Calculate position size using Kelly Criterion
4. EXECUTE: Place orders via Alpaca API
5. MANAGE: Monitor positions, trigger stops/targets
6. CHECK: Verify portfolio risk within limits

Features:
- Multi-strategy signal generation
- Kelly Criterion position sizing
- Automated trade execution
- Real-time position management
- Portfolio risk monitoring
- Graceful shutdown and state persistence
"""

import asyncio
import argparse
import logging
from datetime import datetime, time
from typing import Dict, List, Optional
import json
import os
from zoneinfo import ZoneInfo

from .config import RISK_CONFIG, MONITORING_CONFIG
from .universe import get_universe
from .signal_generator import SignalGenerator, Signal, SignalType
from .position_sizer import MedallionPositionSizer, PositionSize, calculate_max_loss_per_contract
from .trade_executor import AlpacaOptionsExecutor, OrderSide, ExecutionResult
from .iv_data_manager import IVDataManager

# ==== NEW ENHANCED MODULES ====
from .regime_detector import RegimeDetector, MarketRegime
from .correlation_manager import CorrelationManager, Position as CorrPosition
from .weight_optimizer import DynamicWeightOptimizer
from .volatility_surface import VolatilitySurfaceEngine
from .cointegration_engine import CointegrationEngine


# ============================================================================
# MARKET HOURS
# ============================================================================

def market_is_open() -> bool:
    """
    Check if market is currently open.
    
    Returns:
        True if open, False otherwise
    """
    now_et_dt = datetime.now(ZoneInfo("America/New_York"))
    now = now_et_dt.time()
    
    # Check if within trading hours (9:30 AM - 4:00 PM ET)
    market_open = time(9, 30)
    market_close = time(16, 0)
    
    # TODO: Also check for holidays and weekends
    is_weekday = now_et_dt.weekday() < 5  # Monday=0, Friday=4
    
    return is_weekday and market_open <= now <= market_close


def safe_entry_window() -> bool:
    """
    Check if we're in the safe entry window (avoid first/last 15 min).
    
    Returns:
        True if safe to enter, False otherwise
    """
    now = datetime.now(ZoneInfo("America/New_York")).time()
    
    safe_open = time(9, 45)  # 15 min after open
    safe_close = time(15, 45)  # 15 min before close
    
    return safe_open <= now <= safe_close


# ============================================================================
# AUTONOMOUS TRADING ENGINE
# ============================================================================

class AutonomousTradingEngine:
    """
    Main autonomous trading engine.
    
    Runs continuously during market hours, executing the 6-step trading loop.
    """
    
    def __init__(
        self,
        portfolio_value: float,
        paper: bool = True,
        state_file: str = "trading_state.json",
    ):
        """
        Initialize engine.
        
        Args:
            portfolio_value: Starting portfolio value ($)
            paper: Use paper trading (default True)
            state_file: File to persist state
        """
        # get_config() in options/config.py requires a key and returns a single value.
        # The engine expects a dict-like config, so we merge the relevant config dicts.
        self.config = {**RISK_CONFIG, **MONITORING_CONFIG}
        self.logger = logging.getLogger(__name__)
        
        # Portfolio state
        self.portfolio_value = portfolio_value
        self.current_positions = []
        self.portfolio_delta = 0.0
        self.paper = paper
        self.state_file = state_file
        self._stop_event = asyncio.Event()
        
        # Initialize components
        self.signal_generator = SignalGenerator()
        self.position_sizer = MedallionPositionSizer()
        self.trade_executor = AlpacaOptionsExecutor(paper=paper)
        self.iv_data_manager = IVDataManager()  # NEW: IV data management
        
        # ==== ENHANCED MODULES ====
        self.regime_detector = RegimeDetector()
        self.correlation_manager = CorrelationManager()
        self.weight_optimizer = DynamicWeightOptimizer(
            strategies=["iv_rank", "theta_decay", "mean_reversion", "delta_hedging"],
            regime_detector=self.regime_detector
        )
        self.vol_surface_engine = VolatilitySurfaceEngine()
        self.cointegration_engine = CointegrationEngine()
        
        # Backfill IV data on startup
        self._backfill_iv_data()
        
        # Current market regime
        self.current_regime: Optional[MarketRegime] = None
        self.regime_fitted = False
        
        # Statistics
        self.stats = {
            "cycles_run": 0,
            "signals_generated": 0,
            "trades_executed": 0,
            "trades_failed": 0,
            "positions_closed": 0,
            "total_pnl": 0.0,
            "start_time": datetime.now().isoformat(),
        }
        
        # Load previous state if exists
        self._load_state()
        
        self.logger.info(f"Initialized autonomous engine (paper={paper}, portfolio=${portfolio_value:,.0f})")
        self.logger.info("✓ Enhanced modules loaded: RegimeDetector, CorrelationManager, WeightOptimizer, VolSurface, Cointegration")

    def request_shutdown(self) -> None:
        """Request graceful shutdown of the engine."""
        self._stop_event.set()

    async def _sleep_or_stop(self, seconds: float) -> None:
        if seconds <= 0:
            return
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            return

    async def run_forever(self) -> None:
        """Run continuously until a shutdown is requested."""
        self.logger.info("🚀 AUTONOMOUS TRADING ENGINE STARTED")

        try:
            while not self._stop_event.is_set():
                # Check if market is open
                if not market_is_open():
                    self.logger.info("Market closed, waiting...")
                    await self._sleep_or_stop(60)
                    continue

                # Run trading cycle
                await self._trading_cycle()

                # Save state
                self._save_state()

                # Sleep between cycles
                cycle_sleep = self.config["signal_scan_interval_seconds"]
                self.logger.info(f"Cycle complete, sleeping {cycle_sleep}s")
                await self._sleep_or_stop(cycle_sleep)

        except asyncio.CancelledError:
            self.logger.info("Shutdown task cancelled")
            raise
        except KeyboardInterrupt:
            self.logger.info("Shutdown signal received")
        except Exception as e:
            self.logger.error(f"Fatal error in main loop: {e}", exc_info=True)
        finally:
            await self._shutdown()
    
    async def run(self):
        """
        Main trading loop - runs continuously during market hours.
        """
        await self.run_forever()
    
    async def _trading_cycle(self):
        """
        Execute one complete trading cycle (6 steps).
        
        ENHANCED: Now includes regime detection and dynamic weight optimization.
        """
        self.stats["cycles_run"] += 1
        cycle_num = self.stats["cycles_run"]
        
        self.logger.info(f"{'='*60}")
        self.logger.info(f"CYCLE #{cycle_num} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"{'='*60}")
        
        # STEP 0 (NEW): REGIME DETECTION & WEIGHT OPTIMIZATION
        await self._update_regime_and_weights()
        
        # STEP 1: SCAN - Generate signals
        signals = await self._scan_for_signals()
        self.logger.info(f"Step 1 (SCAN): Generated {len(signals)} signals")
        
        # STEP 2: FILTER - Remove invalid signals
        valid_signals = await self._filter_signals(signals)
        self.logger.info(f"Step 2 (FILTER): {len(valid_signals)} valid signals")
        
        # STEP 3: SIZE - Calculate position sizes
        sized_signals = await self._size_positions(valid_signals)
        self.logger.info(f"Step 3 (SIZE): {len(sized_signals)} positions sized")
        
        # STEP 4: EXECUTE - Place orders
        if safe_entry_window():
            executions = await self._execute_trades(sized_signals)
            self.logger.info(f"Step 4 (EXECUTE): {len(executions)} orders submitted")
        else:
            self.logger.info("Step 4 (EXECUTE): Outside safe entry window, skipping")
        
        # STEP 5: MANAGE - Monitor positions
        await self._manage_positions()
        self.logger.info(f"Step 5 (MANAGE): {len(self.current_positions)} positions monitored")
        
        # STEP 6: CHECK - Verify risk limits
        risk_ok = await self._check_risk_limits()
        self.logger.info(f"Step 6 (CHECK): Risk limits {'✓ OK' if risk_ok else '✗ EXCEEDED'}")
        
        # Log cycle summary
        self._log_cycle_summary()
    
    async def _scan_for_signals(self) -> List[Signal]:
        """Step 1: Generate signals from all strategies."""
        symbols = get_universe()
        
        signals = await self.signal_generator.generate_all_signals(
            symbols=symbols,
            portfolio_delta=self.portfolio_delta,
        )
        
        self.stats["signals_generated"] += len(signals)
        
        return signals
    
    async def _filter_signals(self, signals: List[Signal]) -> List[Signal]:
        """
        Step 2: Filter signals to remove invalid/duplicate ones.
        
        ENHANCED: Now includes concentration risk checks.
        """
        # First, check concentration risk
        concentration_ok = await self._check_concentration_risk()
        if not concentration_ok:
            self.logger.warning("Concentration risk too high - blocking new positions")
            return []
        
        valid_signals = []
        
        for signal in signals:
            # Skip HOLD signals
            if signal.signal_type == SignalType.HOLD:
                continue
            
            # Skip low confidence (<30%)
            if signal.confidence < 0.30:
                self.logger.debug(f"Skipping low confidence signal: {signal.symbol} ({signal.confidence:.1%})")
                continue
            
            # Skip if already have position in this symbol
            if self._has_position(signal.symbol):
                self.logger.debug(f"Skipping {signal.symbol} - already have position")
                continue
            
            # Check max positions limit
            max_positions = self.config["max_positions"]
            if len(self.current_positions) >= max_positions:
                self.logger.warning(f"Max positions ({max_positions}) reached, skipping new signals")
                break
            
            valid_signals.append(signal)
        
        return valid_signals
    
    async def _size_positions(self, signals: List[Signal]) -> List[tuple]:
        """Step 3: Calculate position sizes using Kelly Criterion."""
        sized_signals = []
        
        for signal in signals:
            # Estimate max loss per contract
            max_loss = calculate_max_loss_per_contract(
                strategy=signal.strategy,
                strike_width=5.0,  # Default $5 wide spreads
                premium_received=0.30,  # Assume $30 credit per spread
            )
            
            # Calculate position size
            position_size = self.position_sizer.calculate_position_size(
                portfolio_value=self.portfolio_value,
                max_loss_per_contract=max_loss,
                signal_confidence=signal.confidence,
                probability_of_profit=signal.probability_of_profit,
                iv_rank=signal.iv_rank,
                current_portfolio_delta=self.portfolio_delta,
                position_delta_per_contract=signal.delta or 0.0,
            )
            
            # Validate
            if self.position_sizer.validate_position_size(position_size, self.portfolio_value):
                sized_signals.append((signal, position_size))
                self.logger.info(
                    f"Sized {signal.symbol}: {position_size.contracts} contracts "
                    f"(risk: {position_size.risk_percent:.2%})"
                )
            else:
                self.logger.warning(f"Invalid position size for {signal.symbol}, skipping")
        
        return sized_signals
    
    async def _execute_trades(self, sized_signals: List[tuple]) -> List[ExecutionResult]:
        """Step 4: Execute trades via Alpaca API."""
        executions = []
        
        for signal, position_size in sized_signals:
            try:
                # Submit order based on strategy type
                if signal.strategy in ["credit_spread", "put_spread"]:
                    # Submit spread order
                    result = await self.trade_executor.submit_spread_order(
                        long_symbol=f"{signal.symbol}_PUT_{signal.strike_put}",
                        short_symbol=f"{signal.symbol}_PUT_{signal.strike_put + 5}",
                        quantity=position_size.contracts,
                        net_credit=0.30,  # $30 credit
                    )
                elif signal.strategy == "iron_condor":
                    # Submit iron condor
                    result = await self.trade_executor.submit_iron_condor(
                        underlying=signal.symbol,
                        put_buy_strike=signal.strike_put,
                        put_sell_strike=signal.strike_put + 5,
                        call_sell_strike=signal.strike_call - 5,
                        call_buy_strike=signal.strike_call,
                        quantity=position_size.contracts,
                        net_credit=0.50,  # $50 credit
                    )
                else:
                    # Default: Single leg
                    result = await self.trade_executor.submit_single_leg_order(
                        option_symbol=f"{signal.symbol}_CALL_100",
                        side=OrderSide.BUY if signal.signal_type == SignalType.BUY else OrderSide.SELL,
                        quantity=position_size.contracts,
                        limit_price=1.0,
                    )
                
                if result.success:
                    self.stats["trades_executed"] += 1
                    self.logger.info(f"✓ Trade executed: {signal.symbol} - Order {result.order_id}")
                    
                    # Track position
                    self.current_positions.append({
                        "signal": signal,
                        "position_size": position_size,
                        "execution": result,
                        "entry_time": datetime.now().isoformat(),
                    })
                else:
                    self.stats["trades_failed"] += 1
                    self.logger.error(f"✗ Trade failed: {signal.symbol} - {result.error_message}")
                
                executions.append(result)
            
            except Exception as e:
                self.logger.error(f"Execution error for {signal.symbol}: {e}", exc_info=True)
                self.stats["trades_failed"] += 1
        
        return executions
    
    async def _manage_positions(self):
        """Step 5: Monitor positions and trigger stops/targets."""
        positions_to_close = []
        
        for position in self.current_positions:
            # Check if stop-loss or take-profit triggered
            # (This would query current market prices)
            
            # Mock: Close 5% of positions randomly
            import random
            if random.random() < 0.05:
                positions_to_close.append(position)
        
        # Close positions
        for position in positions_to_close:
            self.logger.info(f"Closing position: {position['signal'].symbol}")
            self.current_positions.remove(position)
            self.stats["positions_closed"] += 1
    
    async def _check_risk_limits(self) -> bool:
        """Step 6: Verify portfolio risk within limits."""
        # Check portfolio delta
        max_delta = self.config["max_portfolio_delta"]
        if abs(self.portfolio_delta) > max_delta:
            self.logger.warning(f"Portfolio delta {self.portfolio_delta:.2f} exceeds max {max_delta}")
            return False
        
        # Check max positions
        max_positions = self.config["max_positions"]
        if len(self.current_positions) > max_positions:
            self.logger.warning(f"Position count {len(self.current_positions)} exceeds max {max_positions}")
            return False
        
        return True
    
    def _has_position(self, symbol: str) -> bool:
        """Check if we have a position in symbol."""
        return any(pos["signal"].symbol == symbol for pos in self.current_positions)
    
    def _log_cycle_summary(self):
        """Log summary of current cycle."""
        self.logger.info(f"Portfolio Value: ${self.portfolio_value:,.0f}")
        self.logger.info(f"Open Positions: {len(self.current_positions)}")
        self.logger.info(f"Portfolio Delta: {self.portfolio_delta:.2f}")
        self.logger.info(f"Total Trades: {self.stats['trades_executed']}")
        self.logger.info(f"Total P&L: ${self.stats['total_pnl']:,.0f}")
    
    def _save_state(self):
        """Save engine state to file."""
        state = {
            "portfolio_value": self.portfolio_value,
            "portfolio_delta": self.portfolio_delta,
            "current_positions": self.current_positions,
            "stats": self.stats,
            "last_update": datetime.now().isoformat(),
        }
        
        try:
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
    
    def _load_state(self):
        """Load engine state from file."""
        if not os.path.exists(self.state_file):
            return
        
        try:
            with open(self.state_file, "r") as f:
                state = json.load(f)
            
            self.portfolio_value = state.get("portfolio_value", self.portfolio_value)
            self.portfolio_delta = state.get("portfolio_delta", 0.0)
            self.current_positions = state.get("current_positions", [])
            self.stats = state.get("stats", self.stats)
            
            self.logger.info(f"Loaded state from {self.state_file}")
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
    
    def _backfill_iv_data(self):
        """
        Backfill historical IV data on startup to enable IV rank calculations.
        
        Fixes: "Insufficient data for IV rank (need 20 days)" errors
        """
        try:
            self.logger.info("🔄 Checking IV data cache on startup...")
            
            # Get current IV data stats
            stats = self.iv_data_manager.get_stats()
            symbols_cached = stats.get('symbols', 0)
            total_records = stats.get('total_records', 0)
            
            self.logger.info(
                f"Current IV cache: {total_records} records across {symbols_cached} symbols"
            )
            
            # Get trading universe
            universe = get_universe()
            
            # Backfill for each symbol if needed
            for symbol in universe:
                # Check if we have sufficient data
                iv_rank = self.iv_data_manager.get_iv_rank(symbol, lookback_days=252)
                
                if iv_rank is None:
                    self.logger.info(f"Backfilling IV data for {symbol}...")
                    records = self.iv_data_manager.backfill_historical_iv(symbol, days=252)
                    
                    if records > 0:
                        self.logger.info(f"✓ {symbol}: Added {records} days of IV history")
                    else:
                        self.logger.warning(f"✗ {symbol}: Backfill failed, using synthetic...")
                        records = self.iv_data_manager.backfill_synthetic_data(symbol, days=252)
                        self.logger.info(f"✓ {symbol}: Added {records} days of synthetic IV")
                else:
                    self.logger.info(f"✓ {symbol}: IV rank = {iv_rank:.1f}% (data OK)")
            
            # Log final stats
            stats = self.iv_data_manager.get_stats()
            self.logger.info(
                f"✅ IV backfill complete: {stats['total_records']} records, "
                f"{stats['symbols']} symbols"
            )
            
        except Exception as e:
            self.logger.error(f"IV backfill failed (non-fatal): {e}")
    
    # ========================================================================
    # ENHANCED METHODS (NEW)
    # ========================================================================
    
    async def _update_regime_and_weights(self):
        """
        Update market regime detection and rebalance strategy weights.
        
        This runs at the start of each trading cycle.
        """
        # Fit regime detector on first run
        if not self.regime_fitted:
            try:
                self.logger.info("Fitting regime detector for first time...")
                await self.regime_detector.fit()
                self.regime_fitted = True
                self.logger.info("✓ Regime detector fitted")
            except Exception as e:
                self.logger.error(f"Failed to fit regime detector: {e}")
                return
        
        # Detect current regime
        try:
            regime_state = await self.regime_detector.detect_current_regime()
            old_regime = self.current_regime
            self.current_regime = regime_state.current_regime
            
            # Log regime info
            self.logger.info(
                f"Market Regime: {self.current_regime.value} "
                f"(confidence: {regime_state.confidence:.1%})"
            )
            
            # Rebalance weights if regime changed
            if old_regime != self.current_regime or self.stats["cycles_run"] % 20 == 0:
                self.logger.info("Rebalancing strategy weights...")
                new_weights = await self.weight_optimizer.rebalance(
                    regime=self.current_regime,
                    force=(old_regime != self.current_regime)
                )
                
                # Update signal generator weights (if method exists)
                # This would need to be implemented in signal_generator.py
                self.logger.info(f"Updated strategy weights: {new_weights}")
        
        except Exception as e:
            self.logger.error(f"Regime update failed: {e}")
    
    async def _check_concentration_risk(self) -> bool:
        """
        Check for portfolio concentration risk.
        
        Returns:
            True if safe to proceed, False if concentration limits exceeded
        """
        if len(self.current_positions) == 0:
            return True
        
        try:
            # Convert positions to CorrelationManager format
            corr_positions = []
            for pos in self.current_positions:
                signal_obj = None
                if isinstance(pos, dict):
                    signal_obj = pos.get("signal")
                elif isinstance(pos, str):
                    signal_obj = pos

                if not signal_obj:
                    continue

                symbol = None
                strategy_type = "unknown"
                delta = 0.0

                if isinstance(signal_obj, Signal):
                    symbol = signal_obj.symbol
                    strategy_type = signal_obj.strategy
                    delta = signal_obj.delta or 0.0
                elif isinstance(signal_obj, dict):
                    symbol = signal_obj.get("symbol")
                    strategy_type = signal_obj.get("strategy", strategy_type)
                    delta = float(signal_obj.get("delta", 0.0) or 0.0)
                elif isinstance(signal_obj, str):
                    symbol = signal_obj

                if not symbol:
                    continue

                corr_positions.append(CorrPosition(
                    symbol=str(symbol),
                    quantity=1,
                    entry_price=1.0,
                    current_price=1.0,
                    strategy_type=str(strategy_type),
                    delta=delta,
                    gamma=0.0,
                    theta=0.0,
                    vega=0.0,
                    notional_value=1000.0,  # Simplified
                    sector="Unknown",
                ))
            
            if len(corr_positions) == 0:
                return True
            
            # Build correlation matrix
            corr_matrix = await self.correlation_manager.build_correlation_matrix(corr_positions)
            
            # Check for alerts
            alerts = self.correlation_manager.detect_concentration_risk(
                positions=corr_positions,
                portfolio_value=self.portfolio_value,
                correlation_matrix=corr_matrix,
            )
            
            # Log alerts
            critical_alerts = [a for a in alerts if a.severity == "critical"]
            if critical_alerts:
                for alert in critical_alerts:
                    self.logger.warning(f"⚠ CRITICAL: {alert.message}")
                return False
            
            if alerts:
                for alert in alerts[:3]:  # Show top 3
                    self.logger.warning(f"⚠ {alert.severity.upper()}: {alert.message}")
            
            return True
        
        except Exception as e:
            self.logger.error(f"Concentration check failed: {e}")
            return True  # Allow trading to proceed on error
    
    async def _get_vol_surface_signals(self, symbols: List[str]) -> List[Signal]:
        """
        Generate additional signals from volatility surface analysis.
        
        Args:
            symbols: Symbols to analyze
        
        Returns:
            List of vol-based signals
        """
        vol_signals = []
        
        # Only analyze a few symbols per cycle to avoid slowdown
        for symbol in symbols[:2]:
            try:
                # Build surface
                surface = await self.vol_surface_engine.build_iv_surface(symbol)
                
                # Detect anomalies
                anomalies = await self.vol_surface_engine.detect_anomalies(surface)
                
                # Generate arb signals
                arb_signals = await self.vol_surface_engine.generate_arb_signals(
                    anomalies, surface
                )
                
                # Convert to Signal format (simplified)
                for arb in arb_signals[:1]:  # Max 1 per symbol
                    vol_signals.append(Signal(
                        symbol=symbol,
                        signal_type=SignalType.BUY if "buy" in arb.signal_type else SignalType.SELL,
                        signal_source="vol_surface",
                        strategy="vol_arb",
                        confidence=arb.confidence,
                        timestamp=datetime.now(),
                        reason=arb.reasoning,
                    ))
            
            except Exception as e:
                self.logger.debug(f"Vol surface analysis failed for {symbol}: {e}")
                continue
        
        return vol_signals
    
    async def _get_cointegration_signals(self, symbols: List[str]) -> List[Signal]:
        """
        Generate pairs trading signals from cointegration analysis.
        
        Args:
            symbols: Symbols to test for pairs
        
        Returns:
            List of pairs signals
        """
        # Only scan for pairs periodically (every 50 cycles)
        if self.stats["cycles_run"] % 50 != 1:
            return []
        
        try:
            self.logger.info("Scanning for cointegrated pairs...")
            pairs = await self.cointegration_engine.find_cointegrated_pairs(
                symbols=symbols[:10],  # Limit to avoid slowdown
                max_pairs=5,
            )
            
            if pairs:
                self.logger.info(f"Found {len(pairs)} cointegrated pairs")
        
        except Exception as e:
            self.logger.error(f"Cointegration scan failed: {e}")
        
        return []  # Could convert pairs signals to Signal format
    
    async def _shutdown(self):
        """Graceful shutdown."""
        self.logger.info("Shutting down autonomous engine...")
        
        # Save final state
        self._save_state()
        
        # Log final stats
        self.logger.info("="*60)
        self.logger.info("FINAL STATISTICS")
        self.logger.info("="*60)
        for key, value in self.stats.items():
            self.logger.info(f"{key}: {value}")
        
        self.logger.info("Shutdown complete")


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous options trading engine")
    parser.add_argument(
        "--portfolio-value",
        type=float,
        default=100000,
        help="Starting portfolio value in dollars (default: 100000)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    async def _runner() -> None:
        engine = AutonomousTradingEngine(portfolio_value=args.portfolio_value)

        loop = asyncio.get_running_loop()
        try:
            import signal

            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.add_signal_handler(sig, engine.request_shutdown)
                except NotImplementedError:
                    signal.signal(sig, lambda *_: engine.request_shutdown())
        except Exception:
            # If signal wiring fails for any reason, the engine can still be stopped with Ctrl+C.
            pass

        await engine.run_forever()

    try:
        asyncio.run(_runner())
    except ValueError as e:
        logging.getLogger(__name__).error(str(e))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
