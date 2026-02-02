#!/usr/bin/env python3
"""
V38 Alpha Trader - Trading Execution Engine (Module 2/3)
=========================================================
Trading execution and management for the Ultimate Alpha Generation System.

Components:
- AlphaTrader: Main trading orchestrator
- IntraDayLoop: 5-minute signal generation and execution
- ExecutionEngine: TWAP and order management
- BacktestEngine: Historical strategy testing

Author: V38 Alpha Team
Version: 1.0.0
"""

from dotenv import load_dotenv
load_dotenv()

import os
import sys
import time
import argparse
import logging
import threading
import signal as sig
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

import numpy as np
import pandas as pd

# Import core components
from v38_alpha_core import (
    ExpandedUniverse,
    RegimeDetector,
    MLEnsemble,
    AggressivePositionSizer,
    Signal,
    SignalType,
    RegimeState,
    RegimeInfo,
    AssetClass,
    get_alpaca_api,
    validate_environment,
)

try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.rest import REST, TimeFrame
    from alpaca_trade_api.stream import Stream
    HAS_ALPACA = True
except ImportError:
    HAS_ALPACA = False

try:
    import pytz
    HAS_PYTZ = True
    ET = pytz.timezone('US/Eastern')
    UTC = pytz.UTC
except ImportError:
    HAS_PYTZ = False
    ET = None
    UTC = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/v38_trader.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Ensure logs directory exists
Path('logs').mkdir(exist_ok=True)
Path('cache').mkdir(exist_ok=True)
Path('results').mkdir(exist_ok=True)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TradeRecord:
    """Record of an executed trade."""
    symbol: str
    side: str  # 'buy' or 'sell'
    qty: float
    price: float
    timestamp: datetime
    order_id: str
    strategy: str
    signal_strength: float
    regime: str
    pnl: float = 0.0
    commission: float = 0.0


@dataclass 
class PositionInfo:
    """Current position information."""
    symbol: str
    qty: float
    market_value: float
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    side: str


@dataclass
class PerformanceMetrics:
    """Performance tracking metrics."""
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    daily_returns: List[float] = field(default_factory=list)


# =============================================================================
# EXECUTION ENGINE
# =============================================================================

class ExecutionEngine:
    """
    Handles order execution with TWAP and smart order routing.
    """
    
    def __init__(self, api: Any, universe: ExpandedUniverse):
        """
        Initialize execution engine.
        
        Args:
            api: Alpaca API instance
            universe: Trading universe
        """
        self.api = api
        self.universe = universe
        self.pending_orders: Dict[str, Any] = {}
        self.executed_trades: List[TradeRecord] = []
        
        # Execution parameters
        self.twap_slices = 5
        self.twap_interval_seconds = 30
        self.max_market_impact_pct = 0.005  # 0.5%
        
        logger.info("Initialized ExecutionEngine")
    
    def execute_market_order(self,
                              symbol: str,
                              qty: float,
                              side: str,
                              strategy: str = "alpha",
                              signal_strength: float = 0.5) -> Optional[TradeRecord]:
        """
        Execute a market order.
        
        Args:
            symbol: Trading symbol
            qty: Quantity (positive)
            side: 'buy' or 'sell'
            strategy: Strategy name
            signal_strength: Signal confidence
            
        Returns:
            TradeRecord or None if failed
        """
        if qty <= 0:
            return None
        
        try:
            # Check if crypto
            is_crypto = symbol.endswith('USD') and len(symbol) > 6
            
            # Submit order
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty if not is_crypto else None,
                notional=qty * 100 if is_crypto else None,  # Use notional for crypto
                side=side,
                type='market',
                time_in_force='day' if not is_crypto else 'gtc'
            )
            
            logger.info(f"Submitted {side} order: {qty} {symbol} @ market")
            
            # Wait for fill (with timeout)
            filled_order = self._wait_for_fill(order.id, timeout=30)
            
            if filled_order and filled_order.status == 'filled':
                trade = TradeRecord(
                    symbol=symbol,
                    side=side,
                    qty=float(filled_order.filled_qty),
                    price=float(filled_order.filled_avg_price),
                    timestamp=datetime.now(),
                    order_id=order.id,
                    strategy=strategy,
                    signal_strength=signal_strength,
                    regime=str(RegimeState.BULL)  # Will be updated
                )
                self.executed_trades.append(trade)
                logger.info(f"Filled: {trade.qty} {symbol} @ ${trade.price:.2f}")
                return trade
            else:
                logger.warning(f"Order not filled: {order.id}")
                return None
                
        except Exception as e:
            logger.error(f"Order execution failed for {symbol}: {e}")
            return None
    
    def execute_twap(self,
                      symbol: str,
                      total_qty: float,
                      side: str,
                      duration_minutes: int = 5,
                      strategy: str = "alpha") -> List[TradeRecord]:
        """
        Execute order using TWAP algorithm.
        
        Args:
            symbol: Trading symbol
            total_qty: Total quantity to execute
            side: 'buy' or 'sell'
            duration_minutes: TWAP duration
            strategy: Strategy name
            
        Returns:
            List of executed trades
        """
        trades = []
        slice_qty = total_qty / self.twap_slices
        interval = (duration_minutes * 60) / self.twap_slices
        
        logger.info(f"Starting TWAP: {total_qty} {symbol} in {self.twap_slices} slices")
        
        for i in range(self.twap_slices):
            trade = self.execute_market_order(
                symbol=symbol,
                qty=slice_qty,
                side=side,
                strategy=f"{strategy}_twap",
                signal_strength=0.5
            )
            
            if trade:
                trades.append(trade)
            
            if i < self.twap_slices - 1:
                time.sleep(interval)
        
        total_filled = sum(t.qty for t in trades)
        avg_price = (
            sum(t.qty * t.price for t in trades) / total_filled 
            if total_filled > 0 else 0
        )
        
        logger.info(f"TWAP complete: {total_filled}/{total_qty} filled @ avg ${avg_price:.2f}")
        return trades
    
    def _wait_for_fill(self, order_id: str, timeout: int = 30) -> Optional[Any]:
        """Wait for order to fill."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                order = self.api.get_order(order_id)
                if order.status in ['filled', 'partially_filled']:
                    return order
                elif order.status in ['canceled', 'expired', 'rejected']:
                    return order
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Error checking order {order_id}: {e}")
                break
        return None
    
    def cancel_all_orders(self) -> int:
        """Cancel all open orders."""
        try:
            self.api.cancel_all_orders()
            logger.info("Cancelled all open orders")
            return 0
        except Exception as e:
            logger.error(f"Failed to cancel orders: {e}")
            return -1
    
    def get_current_positions(self) -> Dict[str, PositionInfo]:
        """Get all current positions."""
        positions = {}
        try:
            for pos in self.api.list_positions():
                positions[pos.symbol] = PositionInfo(
                    symbol=pos.symbol,
                    qty=float(pos.qty),
                    market_value=float(pos.market_value),
                    avg_entry_price=float(pos.avg_entry_price),
                    current_price=float(pos.current_price),
                    unrealized_pnl=float(pos.unrealized_pl),
                    unrealized_pnl_pct=float(pos.unrealized_plpc) * 100,
                    side='long' if float(pos.qty) > 0 else 'short'
                )
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
        return positions


# =============================================================================
# SIGNAL GENERATOR
# =============================================================================

class SignalGenerator:
    """
    Generates trading signals by combining multiple strategies.
    """
    
    def __init__(self,
                 ml_ensemble: MLEnsemble,
                 regime_detector: RegimeDetector,
                 universe: ExpandedUniverse):
        """
        Initialize signal generator.
        
        Args:
            ml_ensemble: ML model ensemble
            regime_detector: Regime detector
            universe: Trading universe
        """
        self.ml_ensemble = ml_ensemble
        self.regime_detector = regime_detector
        self.universe = universe
        
        # Signal thresholds
        self.strong_signal_threshold = 0.7
        self.signal_threshold = 0.4
        
        logger.info("Initialized SignalGenerator")
    
    def generate_ml_signals(self, 
                            price_data: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        """
        Generate signals from ML ensemble.
        
        Args:
            price_data: Dictionary of symbol to OHLCV DataFrame
            
        Returns:
            Dictionary of symbol to Signal
        """
        signals = {}
        
        for symbol, df in price_data.items():
            if len(df) < 60:  # Need enough data
                continue
            
            try:
                predictions, probabilities = self.ml_ensemble.predict(df)
                
                if len(predictions) == 0:
                    continue
                
                # Get latest prediction
                latest_pred = predictions[-1]
                latest_probs = probabilities[-1]
                
                # Calculate signal strength (-1 to 1)
                strength = latest_probs[2] - latest_probs[0]  # P(up) - P(down)
                confidence = max(latest_probs)
                
                # Determine signal type
                if latest_pred == 2 and strength > self.strong_signal_threshold:
                    signal_type = SignalType.STRONG_BUY
                elif latest_pred == 2 and strength > self.signal_threshold:
                    signal_type = SignalType.BUY
                elif latest_pred == 0 and strength < -self.strong_signal_threshold:
                    signal_type = SignalType.STRONG_SELL
                elif latest_pred == 0 and strength < -self.signal_threshold:
                    signal_type = SignalType.SELL
                else:
                    signal_type = SignalType.HOLD
                
                signals[symbol] = Signal(
                    symbol=symbol,
                    signal_type=signal_type,
                    strength=strength,
                    timestamp=datetime.now(),
                    strategy="ml_ensemble",
                    confidence=confidence
                )
                
            except Exception as e:
                logger.error(f"ML signal generation failed for {symbol}: {e}")
        
        return signals
    
    def generate_momentum_signals(self,
                                   price_data: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        """
        Generate momentum-based signals.
        
        Args:
            price_data: Price data dictionary
            
        Returns:
            Dictionary of signals
        """
        signals = {}
        momentum_scores = {}
        
        # Calculate momentum for all assets
        for symbol, df in price_data.items():
            if len(df) < 21:
                continue
            
            close = df['close']
            
            # Multi-period momentum
            mom_5 = close.pct_change(5).iloc[-1]
            mom_10 = close.pct_change(10).iloc[-1]
            mom_21 = close.pct_change(21).iloc[-1]
            
            # Weighted momentum score
            score = 0.5 * mom_5 + 0.3 * mom_10 + 0.2 * mom_21
            
            if not np.isnan(score):
                momentum_scores[symbol] = score
        
        if not momentum_scores:
            return signals
        
        # Rank by momentum
        sorted_symbols = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        n_symbols = len(sorted_symbols)
        
        # Top decile = buy, bottom decile = sell
        top_n = max(1, n_symbols // 10)
        bottom_n = max(1, n_symbols // 10)
        
        for i, (symbol, score) in enumerate(sorted_symbols):
            if i < top_n:
                # Top momentum - buy
                strength = min(1.0, score * 10)  # Scale to 0-1
                signals[symbol] = Signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY if strength < 0.7 else SignalType.STRONG_BUY,
                    strength=strength,
                    timestamp=datetime.now(),
                    strategy="momentum",
                    confidence=0.6
                )
            elif i >= n_symbols - bottom_n:
                # Bottom momentum - sell (if shortable)
                asset = self.universe.get_asset(symbol)
                if asset and asset.is_shortable:
                    strength = max(-1.0, score * 10)
                    signals[symbol] = Signal(
                        symbol=symbol,
                        signal_type=SignalType.SELL if strength > -0.7 else SignalType.STRONG_SELL,
                        strength=strength,
                        timestamp=datetime.now(),
                        strategy="momentum",
                        confidence=0.6
                    )
        
        return signals
    
    def generate_mean_reversion_signals(self,
                                         price_data: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        """
        Generate mean-reversion signals based on Bollinger Bands and RSI.
        
        Args:
            price_data: Price data dictionary
            
        Returns:
            Dictionary of signals
        """
        signals = {}
        
        for symbol, df in price_data.items():
            if len(df) < 20:
                continue
            
            try:
                close = df['close']
                
                # Bollinger Bands
                ma_20 = close.rolling(20).mean()
                std_20 = close.rolling(20).std()
                upper = ma_20 + 2 * std_20
                lower = ma_20 - 2 * std_20
                
                current = close.iloc[-1]
                bb_pctb = (current - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1] + 1e-10)
                
                # RSI
                delta = close.diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / (loss + 1e-10)
                rsi = 100 - (100 / (1 + rs))
                current_rsi = rsi.iloc[-1]
                
                # Generate signals
                signal_type = SignalType.HOLD
                strength = 0.0
                
                # Oversold conditions - buy
                if bb_pctb < 0.05 and current_rsi < 30:
                    signal_type = SignalType.STRONG_BUY
                    strength = 0.8
                elif bb_pctb < 0.20 and current_rsi < 40:
                    signal_type = SignalType.BUY
                    strength = 0.5
                
                # Overbought conditions - sell
                elif bb_pctb > 0.95 and current_rsi > 70:
                    signal_type = SignalType.STRONG_SELL
                    strength = -0.8
                elif bb_pctb > 0.80 and current_rsi > 60:
                    signal_type = SignalType.SELL
                    strength = -0.5
                
                if signal_type != SignalType.HOLD:
                    signals[symbol] = Signal(
                        symbol=symbol,
                        signal_type=signal_type,
                        strength=strength,
                        timestamp=datetime.now(),
                        strategy="mean_reversion",
                        confidence=0.55,
                        features={'rsi': current_rsi, 'bb_pctb': bb_pctb}
                    )
                    
            except Exception as e:
                logger.error(f"Mean reversion signal failed for {symbol}: {e}")
        
        return signals
    
    def generate_trend_signals(self,
                                price_data: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        """
        Generate trend-following signals based on moving average crossovers.
        
        Args:
            price_data: Price data dictionary
            
        Returns:
            Dictionary of signals
        """
        signals = {}
        
        for symbol, df in price_data.items():
            if len(df) < 50:
                continue
            
            try:
                close = df['close']
                
                # Moving averages
                ma_10 = close.rolling(10).mean()
                ma_20 = close.rolling(20).mean()
                ma_50 = close.rolling(50).mean()
                
                current = close.iloc[-1]
                
                # Trend conditions
                above_10 = current > ma_10.iloc[-1]
                above_20 = current > ma_20.iloc[-1]
                above_50 = current > ma_50.iloc[-1]
                ma_10_above_20 = ma_10.iloc[-1] > ma_20.iloc[-1]
                ma_20_above_50 = ma_20.iloc[-1] > ma_50.iloc[-1]
                
                # Golden cross detection
                golden_cross = (
                    ma_10.iloc[-1] > ma_20.iloc[-1] and 
                    ma_10.iloc[-2] <= ma_20.iloc[-2]
                )
                
                # Death cross detection
                death_cross = (
                    ma_10.iloc[-1] < ma_20.iloc[-1] and 
                    ma_10.iloc[-2] >= ma_20.iloc[-2]
                )
                
                signal_type = SignalType.HOLD
                strength = 0.0
                
                if golden_cross:
                    signal_type = SignalType.STRONG_BUY
                    strength = 0.75
                elif death_cross:
                    signal_type = SignalType.STRONG_SELL
                    strength = -0.75
                elif above_10 and above_20 and above_50 and ma_10_above_20 and ma_20_above_50:
                    signal_type = SignalType.BUY
                    strength = 0.5
                elif not above_10 and not above_20 and not above_50:
                    signal_type = SignalType.SELL
                    strength = -0.5
                
                if signal_type != SignalType.HOLD:
                    signals[symbol] = Signal(
                        symbol=symbol,
                        signal_type=signal_type,
                        strength=strength,
                        timestamp=datetime.now(),
                        strategy="trend_following",
                        confidence=0.5
                    )
                    
            except Exception as e:
                logger.error(f"Trend signal failed for {symbol}: {e}")
        
        return signals
    
    def combine_signals(self,
                        price_data: Dict[str, pd.DataFrame],
                        regime_info: RegimeInfo) -> Dict[str, Signal]:
        """
        Combine all strategy signals weighted by regime.
        
        Args:
            price_data: Price data dictionary
            regime_info: Current regime information
            
        Returns:
            Combined signals dictionary
        """
        # Get strategy weights from regime
        weights = regime_info.strategy_weights
        
        # Generate signals from each strategy
        ml_signals = {}
        if self.ml_ensemble.is_trained:
            ml_signals = self.generate_ml_signals(price_data)
        
        momentum_signals = self.generate_momentum_signals(price_data)
        mean_rev_signals = self.generate_mean_reversion_signals(price_data)
        trend_signals = self.generate_trend_signals(price_data)
        
        # Combine signals
        all_symbols = set(
            list(ml_signals.keys()) + 
            list(momentum_signals.keys()) + 
            list(mean_rev_signals.keys()) +
            list(trend_signals.keys())
        )
        
        combined_signals = {}
        
        for symbol in all_symbols:
            combined_strength = 0.0
            combined_confidence = 0.0
            total_weight = 0.0
            contributing_strategies = []
            
            # ML signal (gets extra weight)
            if symbol in ml_signals:
                ml_weight = 0.4  # ML gets 40% base weight
                combined_strength += ml_signals[symbol].strength * ml_weight
                combined_confidence += ml_signals[symbol].confidence * ml_weight
                total_weight += ml_weight
                contributing_strategies.append("ml")
            
            # Momentum signal
            if symbol in momentum_signals:
                mom_weight = weights.get("momentum", 0.2)
                combined_strength += momentum_signals[symbol].strength * mom_weight
                combined_confidence += momentum_signals[symbol].confidence * mom_weight
                total_weight += mom_weight
                contributing_strategies.append("momentum")
            
            # Mean reversion signal
            if symbol in mean_rev_signals:
                mr_weight = weights.get("mean_reversion", 0.2)
                combined_strength += mean_rev_signals[symbol].strength * mr_weight
                combined_confidence += mean_rev_signals[symbol].confidence * mr_weight
                total_weight += mr_weight
                contributing_strategies.append("mean_reversion")
            
            # Trend signal
            if symbol in trend_signals:
                trend_weight = weights.get("trend_following", 0.2)
                combined_strength += trend_signals[symbol].strength * trend_weight
                combined_confidence += trend_signals[symbol].confidence * trend_weight
                total_weight += trend_weight
                contributing_strategies.append("trend")
            
            if total_weight > 0:
                final_strength = combined_strength / total_weight
                final_confidence = combined_confidence / total_weight
                
                # Determine signal type
                if final_strength > 0.6:
                    signal_type = SignalType.STRONG_BUY
                elif final_strength > 0.3:
                    signal_type = SignalType.BUY
                elif final_strength < -0.6:
                    signal_type = SignalType.STRONG_SELL
                elif final_strength < -0.3:
                    signal_type = SignalType.SELL
                else:
                    signal_type = SignalType.HOLD
                
                combined_signals[symbol] = Signal(
                    symbol=symbol,
                    signal_type=signal_type,
                    strength=final_strength,
                    timestamp=datetime.now(),
                    strategy="+".join(contributing_strategies),
                    confidence=final_confidence,
                    regime=regime_info.hmm_state
                )
        
        # Filter for actionable signals only
        actionable = {
            k: v for k, v in combined_signals.items() 
            if v.signal_type != SignalType.HOLD
        }
        
        logger.info(f"Generated {len(actionable)} actionable signals from {len(combined_signals)} total")
        return actionable


# =============================================================================
# RISK MANAGER
# =============================================================================

class RiskManager:
    """
    Manages risk limits and circuit breakers.
    """
    
    def __init__(self,
                 max_daily_drawdown: float = 0.05,
                 max_position_pct: float = 0.20,
                 max_sector_pct: float = 0.40,
                 stop_loss_pct: float = 0.03,
                 hard_stop_pct: float = 0.05):
        """
        Initialize risk manager.
        
        Args:
            max_daily_drawdown: Max daily drawdown before circuit breaker
            max_position_pct: Max single position size
            max_sector_pct: Max sector exposure
            stop_loss_pct: Trailing stop loss
            hard_stop_pct: Hard stop loss
        """
        self.max_daily_drawdown = max_daily_drawdown
        self.max_position_pct = max_position_pct
        self.max_sector_pct = max_sector_pct
        self.stop_loss_pct = stop_loss_pct
        self.hard_stop_pct = hard_stop_pct
        
        # State
        self.daily_high_watermark = 0.0
        self.circuit_breaker_triggered = False
        self.position_stops: Dict[str, float] = {}
        
        logger.info("Initialized RiskManager")
    
    def check_circuit_breaker(self, 
                               current_value: float,
                               starting_value: float) -> bool:
        """
        Check if circuit breaker should be triggered.
        
        Args:
            current_value: Current portfolio value
            starting_value: Day start portfolio value
            
        Returns:
            True if circuit breaker triggered
        """
        # Update high water mark
        if current_value > self.daily_high_watermark:
            self.daily_high_watermark = current_value
        
        # Check drawdown from start
        if starting_value > 0:
            drawdown = (starting_value - current_value) / starting_value
            if drawdown > self.max_daily_drawdown:
                logger.warning(f"CIRCUIT BREAKER: Daily drawdown {drawdown:.2%} exceeds {self.max_daily_drawdown:.2%}")
                self.circuit_breaker_triggered = True
                return True
        
        return False
    
    def check_position_stop(self,
                             symbol: str,
                             entry_price: float,
                             current_price: float) -> Tuple[bool, str]:
        """
        Check if position should be stopped out.
        
        Args:
            symbol: Position symbol
            entry_price: Entry price
            current_price: Current price
            
        Returns:
            Tuple of (should_stop, reason)
        """
        if entry_price <= 0:
            return False, ""
        
        pnl_pct = (current_price - entry_price) / entry_price
        
        # Update trailing stop
        if symbol in self.position_stops:
            # Trailing stop: max high - stop_loss_pct
            if current_price > self.position_stops[symbol]:
                self.position_stops[symbol] = current_price
            
            trail_stop = self.position_stops[symbol] * (1 - self.stop_loss_pct)
            if current_price < trail_stop:
                return True, f"trailing_stop ({self.stop_loss_pct:.1%})"
        else:
            self.position_stops[symbol] = current_price
        
        # Hard stop
        if pnl_pct < -self.hard_stop_pct:
            return True, f"hard_stop ({self.hard_stop_pct:.1%})"
        
        return False, ""
    
    def validate_trade(self,
                        symbol: str,
                        qty: float,
                        side: str,
                        price: float,
                        portfolio_value: float,
                        current_positions: Dict[str, PositionInfo],
                        universe: ExpandedUniverse) -> Tuple[bool, str]:
        """
        Validate trade against risk limits.
        
        Args:
            symbol: Trade symbol
            qty: Trade quantity
            side: 'buy' or 'sell'
            price: Current price
            portfolio_value: Portfolio value
            current_positions: Current positions
            universe: Trading universe
            
        Returns:
            Tuple of (is_valid, reason)
        """
        if self.circuit_breaker_triggered:
            return False, "circuit_breaker_active"
        
        trade_value = abs(qty * price)
        
        # Check position size
        if trade_value / portfolio_value > self.max_position_pct:
            return False, f"exceeds_max_position ({self.max_position_pct:.0%})"
        
        # Check sector exposure
        asset = universe.get_asset(symbol)
        if asset and asset.sector:
            sector_exposure = sum(
                abs(p.market_value) for s, p in current_positions.items()
                if universe.get_asset(s) and universe.get_asset(s).sector == asset.sector
            )
            new_exposure = sector_exposure + trade_value
            if new_exposure / portfolio_value > self.max_sector_pct:
                return False, f"exceeds_sector_limit ({self.max_sector_pct:.0%})"
        
        return True, "valid"
    
    def get_take_profit_levels(self, entry_price: float) -> List[Tuple[float, float]]:
        """
        Get take profit levels for scaling out.
        
        Args:
            entry_price: Entry price
            
        Returns:
            List of (price_level, fraction_to_sell)
        """
        return [
            (entry_price * 1.02, 0.25),  # 2% gain: sell 25%
            (entry_price * 1.05, 0.25),  # 5% gain: sell 25%
            (entry_price * 1.10, 0.25),  # 10% gain: sell 25%
        ]


# =============================================================================
# ALPHA TRADER
# =============================================================================

class AlphaTrader:
    """
    Main trading orchestrator for V38 Alpha System.
    """
    
    def __init__(self, paper_trading: bool = True):
        """
        Initialize Alpha Trader.
        
        Args:
            paper_trading: Use paper trading API
        """
        self.paper_trading = paper_trading
        self.running = False
        self.shutdown_event = threading.Event()
        
        # Initialize API
        self.api = get_alpaca_api()
        if not self.api:
            raise RuntimeError("Failed to connect to Alpaca API")
        
        # Get account info
        self.account = self.api.get_account()
        self.starting_equity = float(self.account.equity)
        logger.info(f"Account equity: ${self.starting_equity:,.2f}")
        
        # Initialize components
        self.universe = ExpandedUniverse(self.api)
        self.regime_detector = RegimeDetector()
        self.ml_ensemble = MLEnsemble()
        self.position_sizer = AggressivePositionSizer()
        self.execution_engine = ExecutionEngine(self.api, self.universe)
        self.risk_manager = RiskManager()
        self.signal_generator = SignalGenerator(
            self.ml_ensemble,
            self.regime_detector,
            self.universe
        )
        
        # Performance tracking
        self.performance = PerformanceMetrics()
        self.daily_start_equity = self.starting_equity
        
        # Trading state
        self.last_signal_time: Optional[datetime] = None
        self.signal_interval = timedelta(minutes=5)
        
        # Watchlist (subset for active trading)
        self.active_watchlist: List[str] = []
        self._build_watchlist()
        
        logger.info(f"AlphaTrader initialized with {len(self.active_watchlist)} active symbols")
    
    def _build_watchlist(self) -> None:
        """Build active trading watchlist."""
        # Start with leveraged ETFs for maximum alpha
        self.active_watchlist = self.universe.get_leveraged_bull()[:15]
        
        # Add sector ETFs
        sector_etfs = ["XLK", "XLF", "XLE", "XLV", "SMH", "XBI"]
        self.active_watchlist.extend(sector_etfs)
        
        # Add major indices
        indices = ["SPY", "QQQ", "IWM"]
        self.active_watchlist.extend(indices)
        
        # Add crypto
        crypto = ["BTCUSD", "ETHUSD"]
        self.active_watchlist.extend(crypto)
        
        # Add some inverse for hedging
        inverse = ["SQQQ", "SPXS", "TZA"]
        self.active_watchlist.extend(inverse)
        
        # Remove duplicates
        self.active_watchlist = list(set(self.active_watchlist))
        
        logger.info(f"Active watchlist: {len(self.active_watchlist)} symbols")
    
    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        try:
            clock = self.api.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Failed to check market hours: {e}")
            return False
    
    def get_next_market_open(self) -> Optional[datetime]:
        """Get next market open time."""
        try:
            clock = self.api.get_clock()
            return clock.next_open
        except Exception:
            return None
    
    def fetch_price_data(self, 
                          symbols: List[str],
                          timeframe: str = "5Min",
                          bars: int = 100) -> Dict[str, pd.DataFrame]:
        """
        Fetch price data for symbols.
        
        Args:
            symbols: List of symbols
            timeframe: Bar timeframe
            bars: Number of bars
            
        Returns:
            Dictionary of symbol to DataFrame
        """
        price_data = {}
        
        # Separate crypto from stocks
        crypto_symbols = [s for s in symbols if s.endswith('USD')]
        stock_symbols = [s for s in symbols if not s.endswith('USD')]
        
        # Fetch stock data
        if stock_symbols:
            try:
                tf = TimeFrame.Minute if "Min" in timeframe else TimeFrame.Day
                end = datetime.now()
                start = end - timedelta(days=7)  # Get enough data
                
                stock_bars = self.api.get_bars(
                    stock_symbols,
                    tf,
                    start=start.isoformat(),
                    end=end.isoformat(),
                    limit=bars * len(stock_symbols)
                )
                
                for bar in stock_bars:
                    symbol = bar.S
                    if symbol not in price_data:
                        price_data[symbol] = []
                    price_data[symbol].append({
                        'timestamp': bar.t,
                        'open': bar.o,
                        'high': bar.h,
                        'low': bar.l,
                        'close': bar.c,
                        'volume': bar.v
                    })
                    
            except Exception as e:
                logger.error(f"Failed to fetch stock data: {e}")
        
        # Fetch crypto data
        if crypto_symbols:
            try:
                for symbol in crypto_symbols:
                    try:
                        crypto_bars = self.api.get_crypto_bars(
                            symbol,
                            TimeFrame.Minute,
                            limit=bars
                        )
                        
                        price_data[symbol] = []
                        for bar in crypto_bars[symbol]:
                            price_data[symbol].append({
                                'timestamp': bar.t,
                                'open': bar.o,
                                'high': bar.h,
                                'low': bar.l,
                                'close': bar.c,
                                'volume': bar.v
                            })
                    except Exception as e:
                        logger.warning(f"Failed to fetch crypto data for {symbol}: {e}")
                        
            except Exception as e:
                logger.error(f"Failed to fetch crypto data: {e}")
        
        # Convert to DataFrames
        result = {}
        for symbol, data_list in price_data.items():
            if data_list:
                df = pd.DataFrame(data_list)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                result[symbol] = df.tail(bars)
        
        return result
    
    def get_vix_value(self) -> Optional[float]:
        """Get current VIX value (approximation)."""
        try:
            # Use VIXY as proxy
            bars = self.api.get_bars(
                "VIXY",
                TimeFrame.Minute,
                limit=1
            )
            for bar in bars:
                # VIXY roughly tracks VIX
                return bar.c * 2  # Approximate VIX
        except Exception:
            return 20.0  # Default to normal
    
    def run_trading_loop(self) -> None:
        """Run the main trading loop."""
        logger.info("Starting trading loop...")
        self.running = True
        
        # Setup signal handler
        sig.signal(sig.SIGINT, self._signal_handler)
        sig.signal(sig.SIGTERM, self._signal_handler)
        
        while self.running and not self.shutdown_event.is_set():
            try:
                # Check market hours
                if not self.is_market_open():
                    next_open = self.get_next_market_open()
                    logger.info(f"Market closed. Next open: {next_open}")
                    time.sleep(60)
                    continue
                
                # Check if time for new signals
                now = datetime.now()
                if self.last_signal_time and (now - self.last_signal_time) < self.signal_interval:
                    time.sleep(10)
                    continue
                
                logger.info("=" * 50)
                logger.info("Starting signal generation cycle...")
                
                # Fetch price data
                price_data = self.fetch_price_data(self.active_watchlist)
                logger.info(f"Fetched data for {len(price_data)} symbols")
                
                if not price_data:
                    logger.warning("No price data available")
                    time.sleep(60)
                    continue
                
                # Get SPY data for regime detection
                spy_data = price_data.get("SPY")
                if spy_data is None and "QQQ" in price_data:
                    spy_data = price_data["QQQ"]
                
                # Detect regime
                vix_value = self.get_vix_value()
                if spy_data is not None:
                    regime_info = self.regime_detector.detect_regime(spy_data, vix_value)
                    logger.info(f"Regime: {regime_info.hmm_state.value}, VIX: {regime_info.vix_regime.value}")
                else:
                    regime_info = RegimeInfo(
                        hmm_state=RegimeState.BULL,
                        vix_regime=self.regime_detector.get_vix_regime(vix_value or 20),
                        volatility_regime="normal",
                        trend_regime="uptrend",
                        composite_score=0.5,
                        confidence=0.5,
                        strategy_weights=self.regime_detector.get_regime_weights()
                    )
                
                # Generate signals
                signals = self.signal_generator.combine_signals(price_data, regime_info)
                logger.info(f"Generated {len(signals)} actionable signals")
                
                # Get current positions
                positions = self.execution_engine.get_current_positions()
                portfolio_value = float(self.api.get_account().equity)
                
                # Check circuit breaker
                if self.risk_manager.check_circuit_breaker(portfolio_value, self.daily_start_equity):
                    logger.warning("Circuit breaker triggered! Stopping trading.")
                    self.running = False
                    break
                
                # Calculate target positions
                signal_list = list(signals.values())
                regime_mult = self.regime_detector.get_risk_multiplier()
                
                target_positions = self.position_sizer.size_portfolio(
                    signal_list,
                    portfolio_value,
                    {s: p.market_value for s, p in positions.items()},
                    regime_mult
                )
                
                # Calculate trades needed
                current_values = {s: p.market_value for s, p in positions.items()}
                trades = self.position_sizer.rebalance_positions(
                    target_positions,
                    current_values,
                    portfolio_value
                )
                
                logger.info(f"Calculated {len(trades)} trades")
                
                # Execute trades
                for symbol, trade_value in trades.items():
                    try:
                        # Get current price
                        if symbol in price_data and len(price_data[symbol]) > 0:
                            current_price = price_data[symbol]['close'].iloc[-1]
                        else:
                            continue
                        
                        qty = abs(trade_value) / current_price
                        side = 'buy' if trade_value > 0 else 'sell'
                        
                        # Validate trade
                        is_valid, reason = self.risk_manager.validate_trade(
                            symbol, qty, side, current_price,
                            portfolio_value, positions, self.universe
                        )
                        
                        if not is_valid:
                            logger.warning(f"Trade rejected for {symbol}: {reason}")
                            continue
                        
                        # Get signal for logging
                        signal = signals.get(symbol)
                        signal_strength = signal.strength if signal else 0.5
                        
                        # Execute
                        if abs(trade_value) > portfolio_value * 0.10:
                            # Large order - use TWAP
                            self.execution_engine.execute_twap(
                                symbol, qty, side,
                                duration_minutes=2,
                                strategy="alpha"
                            )
                        else:
                            # Small order - market
                            self.execution_engine.execute_market_order(
                                symbol, qty, side,
                                strategy="alpha",
                                signal_strength=signal_strength
                            )
                        
                    except Exception as e:
                        logger.error(f"Trade execution failed for {symbol}: {e}")
                
                self.last_signal_time = now
                
                # Update performance
                self._update_performance()
                
                # Log status
                self._log_status()
                
                # Sleep until next cycle
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                time.sleep(60)
        
        logger.info("Trading loop stopped")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("Shutdown signal received")
        self.running = False
        self.shutdown_event.set()
    
    def _update_performance(self) -> None:
        """Update performance metrics."""
        try:
            account = self.api.get_account()
            current_equity = float(account.equity)
            
            self.performance.unrealized_pnl = float(account.unrealized_pl)
            self.performance.total_pnl = current_equity - self.starting_equity
            
            # Calculate daily return
            if self.daily_start_equity > 0:
                daily_return = (current_equity - self.daily_start_equity) / self.daily_start_equity
                self.performance.daily_returns.append(daily_return)
            
            # Update win/loss from trades
            trades = self.execution_engine.executed_trades
            if trades:
                self.performance.total_trades = len(trades)
                # (Would need to match entry/exit trades to calculate actual wins/losses)
                
        except Exception as e:
            logger.error(f"Failed to update performance: {e}")
    
    def _log_status(self) -> None:
        """Log current trading status."""
        try:
            account = self.api.get_account()
            positions = self.execution_engine.get_current_positions()
            
            logger.info("-" * 40)
            logger.info(f"Portfolio Value: ${float(account.equity):,.2f}")
            logger.info(f"Cash: ${float(account.cash):,.2f}")
            logger.info(f"Buying Power: ${float(account.buying_power):,.2f}")
            logger.info(f"Positions: {len(positions)}")
            logger.info(f"Total P&L: ${self.performance.total_pnl:,.2f}")
            logger.info(f"Unrealized P&L: ${self.performance.unrealized_pnl:,.2f}")
            logger.info("-" * 40)
            
        except Exception as e:
            logger.error(f"Failed to log status: {e}")
    
    def train_models(self, days: int = 365) -> Dict[str, float]:
        """
        Train ML models on historical data.
        
        Args:
            days: Days of historical data
            
        Returns:
            Training metrics
        """
        logger.info(f"Training models on {days} days of data...")
        
        # Fetch historical data for training
        end = datetime.now()
        start = end - timedelta(days=days)
        
        # Use SPY as reference for training
        try:
            bars = self.api.get_bars(
                "SPY",
                TimeFrame.Day,
                start=start.isoformat(),
                end=end.isoformat()
            )
            
            data = []
            for bar in bars:
                data.append({
                    'timestamp': bar.t,
                    'open': bar.o,
                    'high': bar.h,
                    'low': bar.l,
                    'close': bar.c,
                    'volume': bar.v
                })
            
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Training on {len(df)} bars")
            
            # Train ML ensemble
            metrics = self.ml_ensemble.train(df)
            
            # Train regime detector
            self.regime_detector.train_hmm(df)
            
            # Save models
            model_path = Path("cache/v38_models.pkl")
            self.ml_ensemble.save(str(model_path))
            logger.info(f"Models saved to {model_path}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return {}
    
    def get_status(self) -> Dict[str, Any]:
        """Get current trading status."""
        try:
            account = self.api.get_account()
            positions = self.execution_engine.get_current_positions()
            
            return {
                "account": {
                    "equity": float(account.equity),
                    "cash": float(account.cash),
                    "buying_power": float(account.buying_power),
                    "pattern_day_trader": account.pattern_day_trader,
                },
                "positions": {
                    s: {
                        "qty": p.qty,
                        "market_value": p.market_value,
                        "unrealized_pnl": p.unrealized_pnl,
                        "unrealized_pnl_pct": p.unrealized_pnl_pct,
                    }
                    for s, p in positions.items()
                },
                "performance": {
                    "total_pnl": self.performance.total_pnl,
                    "total_trades": self.performance.total_trades,
                    "win_rate": self.performance.win_rate,
                },
                "regime": {
                    "current": self.regime_detector.current_regime.hmm_state.value if self.regime_detector.current_regime else "unknown",
                },
                "ml_trained": self.ml_ensemble.is_trained,
            }
            
        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            return {"error": str(e)}
    
    def close_all_positions(self) -> None:
        """Close all open positions."""
        logger.info("Closing all positions...")
        try:
            self.api.close_all_positions()
            logger.info("All positions closed")
        except Exception as e:
            logger.error(f"Failed to close positions: {e}")


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

class BacktestEngine:
    """
    Historical backtesting for V38 Alpha System.
    """
    
    def __init__(self, api: Any):
        """
        Initialize backtest engine.
        
        Args:
            api: Alpaca API instance
        """
        self.api = api
        self.universe = ExpandedUniverse()
        self.regime_detector = RegimeDetector()
        self.ml_ensemble = MLEnsemble()
        self.position_sizer = AggressivePositionSizer()
        
        logger.info("Initialized BacktestEngine")
    
    def run_backtest(self,
                      start_date: str,
                      end_date: str,
                      initial_capital: float = 100000) -> Dict[str, Any]:
        """
        Run historical backtest.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_capital: Starting capital
            
        Returns:
            Backtest results
        """
        logger.info(f"Running backtest from {start_date} to {end_date}")
        logger.info(f"Initial capital: ${initial_capital:,.2f}")
        
        # Fetch historical data
        symbols = ["SPY", "QQQ", "TQQQ", "SOXL", "XLK", "XLF"]
        
        all_data = {}
        for symbol in symbols:
            try:
                bars = self.api.get_bars(
                    symbol,
                    TimeFrame.Day,
                    start=start_date,
                    end=end_date
                )
                
                data = []
                for bar in bars:
                    data.append({
                        'timestamp': bar.t,
                        'open': bar.o,
                        'high': bar.h,
                        'low': bar.l,
                        'close': bar.c,
                        'volume': bar.v
                    })
                
                if data:
                    df = pd.DataFrame(data)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                    all_data[symbol] = df
                    
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")
        
        if not all_data:
            return {"error": "No data fetched"}
        
        # Train on first half
        spy_data = all_data.get("SPY")
        if spy_data is not None:
            half = len(spy_data) // 2
            train_data = spy_data.iloc[:half]
            self.ml_ensemble.train(train_data)
            self.regime_detector.train_hmm(train_data)
        
        # Simulate trading
        portfolio_value = initial_capital
        cash = initial_capital
        positions: Dict[str, float] = {}  # symbol -> qty
        
        equity_curve = []
        trades = []
        
        # Get common dates
        dates = spy_data.index[half:] if spy_data is not None else []
        
        for date in dates:
            try:
                # Get prices for this date
                prices = {}
                for symbol, df in all_data.items():
                    if date in df.index:
                        prices[symbol] = df.loc[date, 'close']
                
                # Update portfolio value
                portfolio_value = cash + sum(
                    qty * prices.get(symbol, 0) 
                    for symbol, qty in positions.items()
                )
                
                equity_curve.append({
                    'date': date,
                    'equity': portfolio_value
                })
                
                # Generate signals (simplified for backtest)
                # ... (would implement full signal generation here)
                
            except Exception as e:
                logger.error(f"Backtest error on {date}: {e}")
        
        # Calculate metrics
        equity_df = pd.DataFrame(equity_curve)
        if len(equity_df) > 0:
            returns = equity_df['equity'].pct_change().dropna()
            
            total_return = (portfolio_value - initial_capital) / initial_capital
            sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            max_dd = (equity_df['equity'].cummax() - equity_df['equity']).max() / equity_df['equity'].cummax().max()
            
            results = {
                "start_date": start_date,
                "end_date": end_date,
                "initial_capital": initial_capital,
                "final_value": portfolio_value,
                "total_return": total_return,
                "sharpe_ratio": sharpe,
                "max_drawdown": max_dd,
                "total_trades": len(trades),
                "equity_curve": equity_df.to_dict('records'),
            }
        else:
            results = {"error": "No data for backtest period"}
        
        logger.info(f"Backtest complete. Return: {results.get('total_return', 0):.2%}")
        return results


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="V38 Alpha Trading Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python v38_alpha_trader.py --trade           Start live paper trading
  python v38_alpha_trader.py --train           Train ML models
  python v38_alpha_trader.py --status          Show current positions
  python v38_alpha_trader.py --backtest --start 2024-01-01 --end 2024-12-31
        """
    )
    
    parser.add_argument('--trade', action='store_true', help='Start live paper trading')
    parser.add_argument('--train', action='store_true', help='Train ML models')
    parser.add_argument('--status', action='store_true', help='Show positions and performance')
    parser.add_argument('--backtest', action='store_true', help='Run historical backtest')
    parser.add_argument('--start', type=str, help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='Backtest end date (YYYY-MM-DD)')
    parser.add_argument('--close-all', action='store_true', help='Close all positions')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("V38 ALPHA TRADING ENGINE")
    print("=" * 60)
    
    # Validate environment
    env_status = validate_environment()
    print("\nEnvironment check:")
    for component, available in env_status.items():
        status = "✓" if available else "✗"
        print(f"  {status} {component}")
    
    if not env_status.get("alpaca_api"):
        print("\n❌ Alpaca API not configured. Set ALPACA_API_KEY and ALPACA_SECRET_KEY")
        sys.exit(1)
    
    try:
        if args.trade:
            print("\n🚀 Starting live paper trading...")
            trader = AlphaTrader(paper_trading=True)
            
            # Try to load saved models
            model_path = Path("cache/v38_models.pkl")
            if model_path.exists():
                trader.ml_ensemble.load(str(model_path))
                print("✓ Loaded trained models")
            else:
                print("⚠ No trained models found. Run --train first for best results.")
            
            trader.run_trading_loop()
            
        elif args.train:
            print("\n🧠 Training ML models...")
            trader = AlphaTrader(paper_trading=True)
            metrics = trader.train_models(days=365)
            
            print("\nTraining Results:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")
            
        elif args.status:
            print("\n📊 Current Status:")
            trader = AlphaTrader(paper_trading=True)
            status = trader.get_status()
            
            print(f"\nAccount:")
            print(f"  Equity: ${status['account']['equity']:,.2f}")
            print(f"  Cash: ${status['account']['cash']:,.2f}")
            print(f"  Buying Power: ${status['account']['buying_power']:,.2f}")
            
            print(f"\nPositions ({len(status['positions'])}):")
            for symbol, pos in status['positions'].items():
                print(f"  {symbol}: ${pos['market_value']:,.2f} ({pos['unrealized_pnl_pct']:.2f}%)")
            
            print(f"\nPerformance:")
            print(f"  Total P&L: ${status['performance']['total_pnl']:,.2f}")
            print(f"  Total Trades: {status['performance']['total_trades']}")
            
            print(f"\nSystem:")
            print(f"  ML Trained: {status['ml_trained']}")
            print(f"  Current Regime: {status['regime']['current']}")
            
        elif args.backtest:
            if not args.start or not args.end:
                print("❌ --backtest requires --start and --end dates")
                sys.exit(1)
            
            print(f"\n📈 Running backtest from {args.start} to {args.end}...")
            api = get_alpaca_api()
            engine = BacktestEngine(api)
            results = engine.run_backtest(args.start, args.end)
            
            print("\nBacktest Results:")
            print(f"  Total Return: {results.get('total_return', 0):.2%}")
            print(f"  Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
            print(f"  Max Drawdown: {results.get('max_drawdown', 0):.2%}")
            print(f"  Final Value: ${results.get('final_value', 0):,.2f}")
            
        elif args.close_all:
            print("\n🛑 Closing all positions...")
            trader = AlphaTrader(paper_trading=True)
            trader.close_all_positions()
            print("✓ All positions closed")
            
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\n\nShutdown requested...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
