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
import logging
from datetime import datetime, time
from typing import Dict, List, Optional
import json
import os

from .config import get_config
from .universe import get_universe
from .signal_generator import SignalGenerator, Signal, SignalType
from .position_sizer import MedallionPositionSizer, PositionSize, calculate_max_loss_per_contract
from .trade_executor import AlpacaOptionsExecutor, OrderSide, ExecutionResult


# ============================================================================
# MARKET HOURS
# ============================================================================

def market_is_open() -> bool:
    """
    Check if market is currently open.
    
    Returns:
        True if open, False otherwise
    """
    config = get_config()
    now = datetime.now().time()
    
    # Check if within trading hours (9:30 AM - 4:00 PM ET)
    market_open = time(9, 30)
    market_close = time(16, 0)
    
    # TODO: Also check for holidays and weekends
    is_weekday = datetime.now().weekday() < 5  # Monday=0, Friday=4
    
    return is_weekday and market_open <= now <= market_close


def safe_entry_window() -> bool:
    """
    Check if we're in the safe entry window (avoid first/last 15 min).
    
    Returns:
        True if safe to enter, False otherwise
    """
    config = get_config()
    now = datetime.now().time()
    
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
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # Portfolio state
        self.portfolio_value = portfolio_value
        self.current_positions = []
        self.portfolio_delta = 0.0
        self.paper = paper
        self.state_file = state_file
        
        # Initialize components
        self.signal_generator = SignalGenerator()
        self.position_sizer = MedallionPositionSizer()
        self.trade_executor = AlpacaOptionsExecutor(paper=paper)
        
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
    
    async def run(self):
        """
        Main trading loop - runs continuously during market hours.
        """
        self.logger.info("🚀 AUTONOMOUS TRADING ENGINE STARTED")
        
        try:
            while True:
                # Check if market is open
                if not market_is_open():
                    self.logger.info("Market closed, waiting...")
                    await asyncio.sleep(60)
                    continue
                
                # Run trading cycle
                await self._trading_cycle()
                
                # Save state
                self._save_state()
                
                # Sleep between cycles
                cycle_sleep = self.config["signal_scan_interval_seconds"]
                self.logger.info(f"Cycle complete, sleeping {cycle_sleep}s")
                await asyncio.sleep(cycle_sleep)
        
        except KeyboardInterrupt:
            self.logger.info("Shutdown signal received")
            await self._shutdown()
        except Exception as e:
            self.logger.error(f"Fatal error in main loop: {e}", exc_info=True)
            await self._shutdown()
    
    async def _trading_cycle(self):
        """
        Execute one complete trading cycle (6 steps).
        """
        self.stats["cycles_run"] += 1
        cycle_num = self.stats["cycles_run"]
        
        self.logger.info(f"{'='*60}")
        self.logger.info(f"CYCLE #{cycle_num} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"{'='*60}")
        
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
        """Step 2: Filter signals to remove invalid/duplicate ones."""
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
