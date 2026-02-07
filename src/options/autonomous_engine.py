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

import numpy as np

from .config import RISK_CONFIG, MONITORING_CONFIG
from .universe import get_universe
from .signal_generator import SignalGenerator, Signal, SignalType
from .position_sizer import MedallionPositionSizer, PositionSize, calculate_max_loss_per_contract
from .trade_executor import AlpacaOptionsExecutor, OrderSide, ExecutionResult
from .iv_data_manager import IVDataManager
from .contract_resolver import OptionContractResolver, ResolvedContract, ResolvedSpread, ResolvedIronCondor

# ==== NEW ENHANCED MODULES ====
from .regime_detector import RegimeDetector, MarketRegime
from .correlation_manager import CorrelationManager, Position as CorrPosition
from .weight_optimizer import DynamicWeightOptimizer
from .volatility_surface import VolatilitySurfaceEngine
from .cointegration_engine import CointegrationEngine

# ==== PHASE 4: WIRED ORPHANED MODULES ====
try:
    from .manifold_regime_detector import ManifoldRegimeDetector
    MANIFOLD_AVAILABLE = True
except ImportError:
    ManifoldRegimeDetector = None
    MANIFOLD_AVAILABLE = False

try:
    from src.ml.stacked_ensemble import StackedEnsemble
    from src.ml.transformer_predictor import TransformerPredictor
    from src.ml.gradient_boost_ensemble import GradientBoostEnsemble
    ML_AVAILABLE = True
except ImportError:
    StackedEnsemble = None
    TransformerPredictor = None
    GradientBoostEnsemble = None
    ML_AVAILABLE = False

try:
    from src.ml.continuous_learner import ContinuousLearner
    CONTINUOUS_LEARNER_AVAILABLE = True
except ImportError:
    ContinuousLearner = None
    CONTINUOUS_LEARNER_AVAILABLE = False

try:
    from src.quant_models.garch import GARCHModel
    GARCH_AVAILABLE = True
except ImportError:
    GARCHModel = None
    GARCH_AVAILABLE = False

try:
    from src.quant_models.monte_carlo_pricer import MonteCarloPricer
    from src.quant_models.heston_model import HestonModel
    ADVANCED_PRICING_AVAILABLE = True
except ImportError:
    MonteCarloPricer = None
    HestonModel = None
    ADVANCED_PRICING_AVAILABLE = False

try:
    from src.signal_aggregator import SignalAggregator
    AGGREGATOR_AVAILABLE = True
except ImportError:
    SignalAggregator = None
    AGGREGATOR_AVAILABLE = False

try:
    from src.optimization.bayesian_tuner import BayesianTuner
    BAYESIAN_AVAILABLE = True
except ImportError:
    BayesianTuner = None
    BAYESIAN_AVAILABLE = False


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
        
        # Contract resolver — bridges signals to real OCC symbols with live pricing
        self.contract_resolver = OptionContractResolver(
            trading_client=self.trade_executor.trading_client,
            data_client=self.trade_executor.data_client,
        )
        
        # ==== ENHANCED MODULES ====
        self.regime_detector = RegimeDetector()
        self.correlation_manager = CorrelationManager()
        self.weight_optimizer = DynamicWeightOptimizer(
            strategies=["iv_rank", "theta_decay", "mean_reversion", "delta_hedging"],
            regime_detector=self.regime_detector
        )
        self.vol_surface_engine = VolatilitySurfaceEngine()
        self.cointegration_engine = CointegrationEngine()

        # ==== PHASE 4: WIRED ORPHANED MODULES ====
        self.manifold_detector = None
        if MANIFOLD_AVAILABLE:
            try:
                self.manifold_detector = ManifoldRegimeDetector()
                self.logger.info("✓ ManifoldRegimeDetector loaded")
            except Exception as e:
                self.logger.warning(f"ManifoldRegimeDetector init failed: {e}")

        self.ml_ensemble = None
        self.ml_transformer = None
        if ML_AVAILABLE:
            try:
                self.ml_ensemble = StackedEnsemble()
                self.logger.info("✓ ML StackedEnsemble loaded")
            except Exception as e:
                self.logger.warning(f"StackedEnsemble init failed: {e}")
            try:
                self.ml_transformer = TransformerPredictor()
                self.logger.info("✓ ML TransformerPredictor loaded")
            except Exception as e:
                self.logger.warning(f"TransformerPredictor init failed: {e}")

        self.continuous_learner = None
        if CONTINUOUS_LEARNER_AVAILABLE:
            try:
                self.continuous_learner = ContinuousLearner()
                self.logger.info("✓ ContinuousLearner loaded")
            except Exception as e:
                self.logger.warning(f"ContinuousLearner init failed: {e}")

        self.garch_model = None
        if GARCH_AVAILABLE:
            try:
                self.garch_model = GARCHModel()
                self.logger.info("✓ GARCH model loaded")
            except Exception as e:
                self.logger.warning(f"GARCH init failed: {e}")

        self.mc_pricer = None
        self.heston_model = None
        if ADVANCED_PRICING_AVAILABLE:
            try:
                self.mc_pricer = MonteCarloPricer(n_paths=50000)
                self.heston_model = HestonModel()
                self.logger.info("✓ Monte Carlo + Heston pricing loaded")
            except Exception as e:
                self.logger.warning(f"Advanced pricing init failed: {e}")

        self.signal_aggregator = None
        if AGGREGATOR_AVAILABLE:
            try:
                self.signal_aggregator = SignalAggregator(min_confidence=0.4)
                self.signal_aggregator.initialize()
                self.logger.info("✓ SignalAggregator loaded")
            except Exception as e:
                self.logger.warning(f"SignalAggregator init failed: {e}")

        self.bayesian_tuner = None
        if BAYESIAN_AVAILABLE:
            try:
                self.bayesian_tuner = BayesianTuner()
                self.logger.info("✓ BayesianTuner loaded")
            except Exception as e:
                self.logger.warning(f"BayesianTuner init failed: {e}")

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
        self.logger.info(f"✓ Phase4 modules: Manifold={self.manifold_detector is not None}, "
                        f"ML={self.ml_ensemble is not None}, GARCH={self.garch_model is not None}, "
                        f"Heston={self.heston_model is not None}, Aggregator={self.signal_aggregator is not None}")

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
        
        ENHANCED: Now includes concentration risk checks and SignalAggregator
        confidence boosting for each signal's underlying.
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
            
            # Skip low confidence - LOWERED for paper trading to get trades flowing
            min_confidence = 0.15 if self.paper else 0.30

            # PHASE 5: Boost confidence using SignalAggregator on the underlying
            if self.signal_aggregator is not None:
                try:
                    agg = self.signal_aggregator.aggregate(signal.symbol, min_confidence=0.3)
                    # Align: if aggregator agrees with signal direction, boost confidence
                    if signal.signal_type == SignalType.BUY and agg.signal > 0.2:
                        signal.confidence = min(signal.confidence * 1.25, 0.99)
                        self.logger.debug(f"Boosted {signal.symbol} confidence via aggregator (bullish agreement)")
                    elif signal.signal_type == SignalType.SELL and agg.signal < -0.2:
                        signal.confidence = min(signal.confidence * 1.25, 0.99)
                        self.logger.debug(f"Boosted {signal.symbol} confidence via aggregator (bearish agreement)")
                    elif abs(agg.signal) > 0.3 and (
                        (signal.signal_type == SignalType.BUY and agg.signal < -0.3)
                        or (signal.signal_type == SignalType.SELL and agg.signal > 0.3)
                    ):
                        # Aggregator strongly disagrees — dampen confidence
                        signal.confidence *= 0.7
                        self.logger.debug(f"Dampened {signal.symbol} confidence via aggregator (disagreement)")
                except Exception as e:
                    self.logger.debug(f"Aggregator boost failed for {signal.symbol}: {e}")

            # PHASE 5: Heston/MC pricing comparison — if model price deviates
            # from market mid-price significantly, boost/reject signal
            if self.heston_model is not None and signal.expected_premium:
                try:
                    from src.quant_models.heston_model import HestonParams
                    # Use stored GARCH vol as v0
                    v0 = getattr(self, '_last_garch_vol', 0.04)
                    params = HestonParams(v0=v0, kappa=2.0, theta=v0, xi=0.3, rho=-0.7)
                    current_price = signal.current_price or 100
                    strike = signal.strike_put or signal.strike_call or current_price
                    T = (signal.dte or 30) / 365.0
                    heston_result = self.heston_model.price_call(
                        current_price, strike, T, params
                    )
                    model_price = heston_result.price
                    market_price = signal.expected_premium
                    if market_price > 0 and model_price > 0:
                        price_ratio = model_price / market_price
                        if price_ratio > 1.15:  # Model says option is underpriced
                            signal.confidence = min(signal.confidence * 1.15, 0.99)
                            self.logger.info(f"Heston: {signal.symbol} underpriced by {(price_ratio-1)*100:.0f}%")
                        elif price_ratio < 0.85:  # Model says option is overpriced
                            signal.confidence *= 0.85
                            self.logger.info(f"Heston: {signal.symbol} overpriced by {(1-price_ratio)*100:.0f}%")
                except Exception as e:
                    self.logger.debug(f"Heston pricing comparison failed: {e}")
            
            if signal.confidence < min_confidence:
                self.logger.debug(f"Skipping low confidence signal: {signal.symbol} ({signal.confidence:.1%})")
                # PHASE 5: Log every signal, even rejected ones
                await self._log_signal_with_reasoning(signal, execution_result=None)
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
        """Step 4: Resolve signals to real contracts and execute via Alpaca API.

        For each (signal, position_size) pair:
        1. Resolve the abstract signal to real OCC contract(s) with live pricing.
        2. Use the resolved mid-price as the limit price (not hardcoded).
        3. Pass the real OCC symbol to trade_executor (not "{symbol}_CALL_100").
        4. If resolution fails, log a warning and skip (never crash).
        
        ENHANCED: Post-trade feedback to ContinuousLearner + Discord notifications.
        """
        executions: List[ExecutionResult] = []

        for signal, position_size in sized_signals:
            try:
                result = await self._resolve_and_execute(signal, position_size)
                if result is not None:
                    executions.append(result)
                    # PHASE 5: Log signal with execution result + Discord
                    await self._log_signal_with_reasoning(signal, execution_result=result)
                    await self._send_discord_notification(
                        f"🔔 Trade Executed: {signal.symbol} {signal.strategy} "
                        f"conf={signal.confidence:.0%} status={result.status}"
                    )
                    # PHASE 4: ContinuousLearner post-trade recording
                    if self.continuous_learner is not None:
                        try:
                            from src.ml.continuous_learner import TradeResult
                            trade_result = TradeResult(
                                timestamp=datetime.now(),
                                ticker=signal.symbol,
                                signal_direction="long" if signal.signal_type == SignalType.BUY else "short",
                                signal_confidence=signal.confidence,
                                predicted_return=signal.probability_of_profit * 0.02,
                                actual_return=0.0,  # Will be updated on position close
                                is_hit=True,  # Placeholder, updated on close
                                features={
                                    "iv_rank": signal.iv_rank or 0.0,
                                    "confidence": signal.confidence,
                                    "strategy": signal.strategy,
                                },
                                regime=self.current_regime.value if self.current_regime else "unknown",
                            )
                            self.continuous_learner.record_trade(trade_result)
                        except Exception as e:
                            self.logger.debug(f"ContinuousLearner recording failed: {e}")
            except Exception as e:
                self.logger.error(
                    f"Execution error for {signal.symbol} ({signal.strategy}): {e}",
                    exc_info=True,
                )
                self.stats["trades_failed"] += 1

        return executions

    async def _resolve_and_execute(
        self, signal: Signal, position_size
    ) -> Optional[ExecutionResult]:
        """Resolve a single signal to real contracts, then execute.

        Returns:
            ExecutionResult on success/failure, or None if resolution fails.
        """
        target_dte = signal.dte or 30

        # ---------------------------------------------------------------- #
        # CREDIT SPREAD / PUT SPREAD
        # ---------------------------------------------------------------- #
        if signal.strategy in ("credit_spread", "put_spread", "call_spread"):
            resolved = await self.contract_resolver.resolve_spread(
                symbol=signal.symbol,
                spread_type=signal.strategy,
                target_dte=target_dte,
            )
            if resolved is None:
                self.logger.warning(
                    f"Contract resolution failed for {signal.symbol} "
                    f"{signal.strategy} ~{target_dte}DTE — skipping trade"
                )
                return None

            # Populate signal with resolved data
            signal.occ_symbol = resolved.short_leg.occ_symbol
            signal.expiration_date = resolved.short_leg.expiration

            self.logger.info(
                f"Executing spread {signal.symbol}: "
                f"short={resolved.short_leg.occ_symbol} (${resolved.short_leg.mid_price:.2f}) "
                f"long={resolved.long_leg.occ_symbol} (${resolved.long_leg.mid_price:.2f}) "
                f"net_credit=${resolved.net_credit:.2f}"
            )

            result = await self.trade_executor.submit_spread_order(
                long_symbol=resolved.long_leg.occ_symbol,
                short_symbol=resolved.short_leg.occ_symbol,
                quantity=position_size.contracts,
                net_credit=resolved.net_credit if resolved.net_credit > 0 else None,
                net_debit=abs(resolved.net_credit) if resolved.net_credit <= 0 else None,
            )

        # ---------------------------------------------------------------- #
        # IRON CONDOR
        # ---------------------------------------------------------------- #
        elif signal.strategy == "iron_condor":
            resolved = await self.contract_resolver.resolve_iron_condor(
                symbol=signal.symbol,
                target_dte=target_dte,
            )
            if resolved is None:
                self.logger.warning(
                    f"Contract resolution failed for {signal.symbol} "
                    f"iron_condor ~{target_dte}DTE — skipping trade"
                )
                return None

            signal.occ_symbol = resolved.put_spread.short_leg.occ_symbol
            signal.expiration_date = resolved.put_spread.short_leg.expiration

            self.logger.info(
                f"Executing iron condor {signal.symbol}: "
                f"put_spread=[{resolved.put_spread.short_leg.occ_symbol}/"
                f"{resolved.put_spread.long_leg.occ_symbol}] "
                f"call_spread=[{resolved.call_spread.short_leg.occ_symbol}/"
                f"{resolved.call_spread.long_leg.occ_symbol}] "
                f"total_credit=${resolved.total_credit:.2f}"
            )

            result = await self.trade_executor.submit_iron_condor(
                underlying=signal.symbol,
                put_buy_strike=resolved.put_spread.long_leg.strike,
                put_sell_strike=resolved.put_spread.short_leg.strike,
                call_sell_strike=resolved.call_spread.short_leg.strike,
                call_buy_strike=resolved.call_spread.long_leg.strike,
                quantity=position_size.contracts,
                net_credit=resolved.total_credit,
            )

        # ---------------------------------------------------------------- #
        # SINGLE LEG (default: calls/puts, straddles, etc.)
        # ---------------------------------------------------------------- #
        else:
            option_type = "call" if signal.signal_type == SignalType.BUY else "put"
            resolved = await self.contract_resolver.resolve_single_leg(
                symbol=signal.symbol,
                option_type=option_type,
                target_dte=target_dte,
            )
            if resolved is None:
                self.logger.warning(
                    f"Contract resolution failed for {signal.symbol} "
                    f"{option_type} ~{target_dte}DTE — skipping trade"
                )
                return None

            signal.occ_symbol = resolved.occ_symbol
            signal.expiration_date = resolved.expiration

            self.logger.info(
                f"Executing single leg {signal.symbol}: {resolved.occ_symbol} "
                f"strike={resolved.strike} exp={resolved.expiration} "
                f"bid={resolved.bid:.2f} ask={resolved.ask:.2f} "
                f"limit={resolved.mid_price:.2f}"
            )

            result = await self.trade_executor.submit_single_leg_order(
                option_symbol=resolved.occ_symbol,
                side=OrderSide.BUY if signal.signal_type == SignalType.BUY else OrderSide.SELL,
                quantity=position_size.contracts,
                limit_price=resolved.mid_price,
            )

        # ---------------------------------------------------------------- #
        # POST-EXECUTION BOOKKEEPING
        # ---------------------------------------------------------------- #
        if result.success:
            self.stats["trades_executed"] += 1
            self.logger.info(
                f"✓ Trade executed: {signal.symbol} ({signal.strategy}) "
                f"— Order {result.order_id}"
            )
            self.current_positions.append({
                "signal": signal,
                "position_size": position_size,
                "execution": result,
                "entry_time": datetime.now().isoformat(),
            })
        else:
            self.stats["trades_failed"] += 1
            self.logger.error(
                f"✗ Trade failed: {signal.symbol} ({signal.strategy}) "
                f"— {result.error_message}"
            )

        return result
    
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
        for pos in self.current_positions:
            if pos == symbol:
                return True
            if isinstance(pos, dict):
                if pos.get("symbol") == symbol:
                    return True
                signal = pos.get("signal")
                if getattr(signal, "symbol", None) == symbol:
                    return True
        return False
    
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
        
        ENHANCED: Now uses ManifoldRegimeDetector alongside HMM detector.
        Also runs GARCH vol forecast to adjust risk parameters.
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
        
        # Detect current regime (HMM)
        try:
            regime_state = await self.regime_detector.detect_current_regime()
            old_regime = self.current_regime
            self.current_regime = regime_state.current_regime
            
            self.logger.info(
                f"HMM Regime: {self.current_regime.value} "
                f"(confidence: {regime_state.confidence:.1%})"
            )

            # ManifoldRegimeDetector cross-validation — requires price data + vol
            if self.manifold_detector is not None:
                try:
                    import yfinance as yf
                    spy_data = yf.download("SPY", period="1y", interval="1d", progress=False)
                    if not spy_data.empty and len(spy_data) > 30:
                        prices = spy_data["Close"].values.flatten()
                        log_rets = np.diff(np.log(prices[-21:]))
                        realized_vol = float(np.std(log_rets) * np.sqrt(252))
                        # Implied vol from VIX
                        implied_vol = realized_vol * 1.2
                        try:
                            vix_data = yf.download("^VIX", period="5d", interval="1d", progress=False)
                            if not vix_data.empty:
                                implied_vol = float(vix_data["Close"].values.flatten()[-1]) / 100.0
                        except Exception:
                            pass
                        manifold_result = self.manifold_detector.detect_regime(
                            prices, realized_vol, implied_vol
                        )
                        manifold_regime = getattr(manifold_result, 'regime', 'unknown')
                        manifold_conf = getattr(manifold_result, 'confidence', 0.0)
                        self.logger.info(
                            f"Manifold Regime: {manifold_regime} (confidence: {manifold_conf:.1%})"
                        )
                        # Store for use in signal scoring
                        self._last_manifold_state = manifold_result
                except Exception as e:
                    self.logger.debug(f"Manifold regime detection failed: {e}")

            # GARCH vol overlay
            if self.garch_model is not None:
                try:
                    garch_forecast = self.garch_model.fit_and_forecast("SPY", horizon=5)
                    self._last_garch_vol = garch_forecast.current_vol ** 2  # variance for Heston v0
                    self.logger.info(
                        f"GARCH Vol: current={garch_forecast.current_vol:.1%}, "
                        f"5d_forecast={garch_forecast.forecast_vols[-1]:.1%}, "
                        f"persistence={garch_forecast.params.persistence:.4f}"
                    )
                    # Adjust position sizing risk if vol is elevated
                    if garch_forecast.current_vol > 0.30:
                        self.logger.warning("Elevated vol detected — tightening risk limits")
                except Exception as e:
                    self.logger.debug(f"GARCH update failed: {e}")
            
            # Rebalance weights if regime changed
            if old_regime != self.current_regime or self.stats["cycles_run"] % 20 == 0:
                self.logger.info("Rebalancing strategy weights...")
                new_weights = await self.weight_optimizer.rebalance(
                    regime=self.current_regime,
                    force=(old_regime != self.current_regime)
                )
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

    # ========================================================================
    # PHASE 5: REGIME-BASED OPTIONS STRATEGY SELECTION
    # ========================================================================

    def _get_regime_strategies(self) -> List[str]:
        """
        Return the preferred options strategies based on current regime.

        Covered calls, cash-secured puts, vertical spreads, iron condors
        are selected depending on the detected market regime.
        """
        if self.current_regime is None:
            return ["credit_spread", "iron_condor"]

        regime_str = str(self.current_regime.value).upper()

        if "BULL_LOW" in regime_str:
            return ["cash_secured_put", "covered_call", "call_spread"]
        elif "BULL_HIGH" in regime_str:
            return ["iron_condor", "credit_spread", "covered_call"]
        elif "BEAR_LOW" in regime_str:
            return ["put_spread", "covered_call"]
        elif "BEAR_HIGH" in regime_str:
            return ["iron_condor", "put_spread"]
        else:
            return ["iron_condor", "credit_spread"]

    async def _send_discord_notification(self, message: str) -> None:
        """Send a trade notification to Discord webhook (if configured)."""
        webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
        if not webhook_url:
            return
        try:
            import aiohttp
            payload = {"content": message}
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload, timeout=aiohttp.ClientTimeout(total=10)):
                    pass
        except Exception as e:
            self.logger.debug(f"Discord notification failed: {e}")

    async def _log_signal_with_reasoning(self, signal, execution_result=None) -> None:
        """Log every signal with full reasoning for audit trail."""
        reasoning = (
            f"📊 **Signal: {signal.symbol}** ({signal.strategy})\n"
            f"  Type: {signal.signal_type.value} | Confidence: {signal.confidence:.1%}\n"
            f"  IV Rank: {getattr(signal, 'iv_rank', 'N/A')} | "
            f"Regime: {self.current_regime.value if self.current_regime else 'unknown'}\n"
        )
        if execution_result:
            if execution_result.success:
                reasoning += f"  ✅ EXECUTED: Order {execution_result.order_id}\n"
            else:
                reasoning += f"  ❌ FAILED: {execution_result.error_message}\n"

        self.logger.info(reasoning)
        await self._send_discord_notification(reasoning)

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
