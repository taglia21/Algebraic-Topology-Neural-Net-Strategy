#!/usr/bin/env python3
"""
V28 Production Trading System
==============================
Integrated production trading system with all V28 features.

Components:
- Real-time Dashboard API with WebSocket streaming
- Advanced HMM + GARCH regime detection
- Cross-asset correlation tracking
- Enhanced fractional Kelly position sizing
- Adaptive strategy routing

Target Metrics:
- Sharpe Ratio > 2.5
- CAGR > 50%
- Maximum Drawdown < 15%
- Win Rate > 60%

Usage:
    python v28_production_system.py --mode=paper
    python v28_production_system.py --mode=live
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import threading

import numpy as np
import pandas as pd

# V28 Components
from v28_dashboard_api import (
    DashboardAPIServer, 
    PerformanceMetrics, 
    Position, 
    Trade, 
    RegimeInfo
)
from v28_regime_detector import (
    V28RegimeDetector, 
    RegimeState, 
    MarketRegime, 
    VolatilityRegime
)
from v28_correlation_engine import (
    V28CorrelationEngine, 
    CorrelationState, 
    CorrelationAlert
)
from v28_kelly_sizer import (
    V28KellyPositionSizer, 
    PositionSizeResult,
    PortfolioWeightResult
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/v28_production.log')
    ]
)
logger = logging.getLogger('V28_Production')


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class V28Config:
    """V28 Production system configuration."""
    # Mode
    mode: str = 'paper'  # 'paper', 'live', 'backtest'
    
    # API
    api_host: str = '0.0.0.0'
    api_port: int = 8080
    
    # Redis
    redis_url: str = 'redis://localhost:6379/0'
    
    # Trading
    initial_capital: float = 100000.0
    max_positions: int = 10
    max_position_pct: float = 0.15
    max_portfolio_risk: float = 0.25
    
    # Kelly Sizing
    kelly_fraction: float = 0.25
    min_position_pct: float = 0.02
    target_volatility: float = 0.15
    max_drawdown_limit: float = 0.15
    
    # Regime Detection
    hmm_states: int = 4
    garch_p: int = 1
    garch_q: int = 1
    
    # Correlation
    correlation_lookback: int = 60
    max_correlation_exposure: float = 0.40
    
    # Universe
    universe: List[str] = field(default_factory=lambda: [
        'SPY', 'QQQ', 'IWM',  # Indices
        'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'AMZN', 'TSLA',  # Mega Tech
        'JPM', 'BAC', 'GS',  # Financials
        'XLE', 'XLF', 'XLV', 'XLK', 'XLI', 'XLY', 'XLP', 'XLU'  # Sectors
    ])
    
    # Intervals
    trading_interval_seconds: int = 60
    regime_update_seconds: int = 300
    correlation_update_seconds: int = 600
    
    # Logging
    log_level: str = 'INFO'
    log_trades: bool = True
    log_signals: bool = True


# =============================================================================
# SIGNAL GENERATOR
# =============================================================================

class SignalGenerator:
    """
    Generate trading signals based on regime and technical analysis.
    """
    
    def __init__(self, config: V28Config):
        self.config = config
        self.signal_cache: Dict[str, Dict[str, Any]] = {}
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for signal generation."""
        df = df.copy()
        
        close = df['close']
        high = df.get('high', close)
        low = df.get('low', close)
        volume = df.get('volume', pd.Series([1e8] * len(df), index=close.index))
        
        # Moving averages
        df['sma_20'] = close.rolling(20).mean()
        df['sma_50'] = close.rolling(50).mean()
        df['sma_200'] = close.rolling(200).mean()
        df['ema_12'] = close.ewm(span=12).mean()
        df['ema_26'] = close.ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_mid'] = close.rolling(20).mean()
        df['bb_std'] = close.rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
        df['bb_pct'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        
        # ATR
        tr = pd.DataFrame({
            'hl': high - low,
            'hc': abs(high - close.shift()),
            'lc': abs(low - close.shift())
        }).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        # Volatility
        df['returns'] = close.pct_change()
        df['volatility'] = df['returns'].rolling(20).std() * np.sqrt(252)
        
        # Momentum
        df['momentum_10'] = close.pct_change(10)
        df['momentum_20'] = close.pct_change(20)
        
        # Volume ratio
        df['volume_ratio'] = volume / volume.rolling(20).mean()
        
        return df
    
    def generate_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        regime: RegimeState
    ) -> Dict[str, Any]:
        """
        Generate trading signal for a symbol.
        
        Returns:
            Dictionary with signal details
        """
        if len(df) < 50:
            return {'signal': 0, 'confidence': 0, 'reason': 'insufficient_data'}
        
        # Calculate indicators
        df = self.calculate_technical_indicators(df)
        last = df.iloc[-1]
        
        # Initialize signal components
        signals = []
        
        # Trend signals
        if last['close'] > last['sma_50'] > last['sma_200']:
            signals.append(('trend_up', 0.3))
        elif last['close'] < last['sma_50'] < last['sma_200']:
            signals.append(('trend_down', -0.3))
        
        # MACD signals
        if last['macd_hist'] > 0 and df['macd_hist'].iloc[-2] <= 0:
            signals.append(('macd_bullish', 0.2))
        elif last['macd_hist'] < 0 and df['macd_hist'].iloc[-2] >= 0:
            signals.append(('macd_bearish', -0.2))
        
        # RSI signals
        if last['rsi'] < 30:
            signals.append(('rsi_oversold', 0.25))
        elif last['rsi'] > 70:
            signals.append(('rsi_overbought', -0.25))
        elif 40 < last['rsi'] < 60:
            signals.append(('rsi_neutral', 0))
        
        # Bollinger Band signals
        if last['bb_pct'] < 0.1:
            signals.append(('bb_oversold', 0.2))
        elif last['bb_pct'] > 0.9:
            signals.append(('bb_overbought', -0.2))
        
        # Momentum signals
        if last['momentum_10'] > 0.02:
            signals.append(('momentum_positive', 0.15))
        elif last['momentum_10'] < -0.02:
            signals.append(('momentum_negative', -0.15))
        
        # Aggregate signal
        total_signal = sum(s[1] for s in signals)
        
        # Apply regime adjustment
        regime_mult = self._get_regime_signal_multiplier(regime)
        adjusted_signal = total_signal * regime_mult
        
        # Normalize to -1 to 1
        normalized_signal = np.clip(adjusted_signal, -1, 1)
        
        # Calculate confidence
        n_confirming = sum(1 for s in signals if 
                          (normalized_signal > 0 and s[1] > 0) or 
                          (normalized_signal < 0 and s[1] < 0))
        confidence = min(1.0, n_confirming / max(len(signals), 1) + 0.2)
        
        # Determine direction
        if abs(normalized_signal) < 0.1:
            direction = 'none'
        elif normalized_signal > 0:
            direction = 'long'
        else:
            direction = 'short'
        
        result = {
            'symbol': symbol,
            'signal': normalized_signal,
            'direction': direction,
            'confidence': confidence,
            'regime_multiplier': regime_mult,
            'signals': signals,
            'indicators': {
                'rsi': last['rsi'],
                'macd_hist': last['macd_hist'],
                'bb_pct': last['bb_pct'],
                'volatility': last['volatility'],
                'momentum_10': last['momentum_10']
            },
            'timestamp': datetime.now().isoformat()
        }
        
        self.signal_cache[symbol] = result
        return result
    
    def _get_regime_signal_multiplier(self, regime: RegimeState) -> float:
        """Get signal multiplier based on regime."""
        multipliers = {
            MarketRegime.BULL: 1.2,
            MarketRegime.BEAR: 0.6,
            MarketRegime.SIDEWAYS: 0.8,
            MarketRegime.CRISIS: 0.3
        }
        
        base = multipliers.get(regime.market_regime, 1.0)
        
        # Adjust for volatility
        if regime.volatility_regime == VolatilityRegime.EXTREME:
            base *= 0.5
        elif regime.volatility_regime == VolatilityRegime.HIGH:
            base *= 0.7
        
        return base


# =============================================================================
# PORTFOLIO MANAGER
# =============================================================================

class PortfolioManager:
    """
    Manage portfolio positions and risk.
    """
    
    def __init__(self, config: V28Config):
        self.config = config
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity = config.initial_capital
        self.peak_equity = config.initial_capital
        self.trade_counter = 0
    
    def update_position(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        regime: str = ''
    ) -> Position:
        """Update or create a position."""
        if symbol in self.positions:
            pos = self.positions[symbol]
            # Update existing position
            if pos.side == side:
                # Adding to position
                total_qty = pos.quantity + quantity
                avg_price = (pos.entry_price * pos.quantity + price * quantity) / total_qty
                pos.quantity = total_qty
                pos.entry_price = avg_price
            else:
                # Reducing or closing position
                if quantity >= pos.quantity:
                    # Close position
                    pnl = self._calculate_pnl(pos, price, quantity)
                    self._record_trade(pos, price, pnl)
                    del self.positions[symbol]
                    return None
                else:
                    pos.quantity -= quantity
        else:
            # New position
            pos = Position(
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=price,
                current_price=price,
                unrealized_pnl=0.0,
                unrealized_pnl_pct=0.0,
                entry_time=datetime.now().isoformat(),
                holding_period_hours=0.0,
                regime_at_entry=regime
            )
            self.positions[symbol] = pos
        
        return pos
    
    def update_prices(self, prices: Dict[str, float]):
        """Update current prices for all positions."""
        for symbol, pos in self.positions.items():
            if symbol in prices:
                pos.current_price = prices[symbol]
                
                if pos.side == 'long':
                    pos.unrealized_pnl = (pos.current_price - pos.entry_price) * pos.quantity
                else:
                    pos.unrealized_pnl = (pos.entry_price - pos.current_price) * pos.quantity
                
                pos.unrealized_pnl_pct = pos.unrealized_pnl / (pos.entry_price * pos.quantity)
                
                # Update holding period
                entry_time = datetime.fromisoformat(pos.entry_time)
                pos.holding_period_hours = (datetime.now() - entry_time).total_seconds() / 3600
    
    def _calculate_pnl(self, pos: Position, exit_price: float, quantity: float) -> float:
        """Calculate P&L for a trade."""
        if pos.side == 'long':
            return (exit_price - pos.entry_price) * quantity
        else:
            return (pos.entry_price - exit_price) * quantity
    
    def _record_trade(self, pos: Position, exit_price: float, pnl: float):
        """Record a completed trade."""
        self.trade_counter += 1
        
        trade = Trade(
            trade_id=f"T{self.trade_counter:06d}",
            symbol=pos.symbol,
            side=pos.side,
            quantity=pos.quantity,
            price=exit_price,
            timestamp=datetime.now().isoformat(),
            pnl=pnl,
            pnl_pct=pnl / (pos.entry_price * pos.quantity),
            regime=pos.regime_at_entry
        )
        
        self.trades.append(trade)
        self.equity += pnl
        self.peak_equity = max(self.peak_equity, self.equity)
    
    def get_current_exposure(self) -> float:
        """Get current portfolio exposure."""
        total_value = sum(
            pos.current_price * pos.quantity 
            for pos in self.positions.values()
        )
        return total_value / self.equity if self.equity > 0 else 0
    
    def get_drawdown(self) -> float:
        """Get current drawdown from peak."""
        return (self.peak_equity - self.equity) / self.peak_equity


# =============================================================================
# V28 PRODUCTION ENGINE
# =============================================================================

class V28ProductionEngine:
    """
    Main production trading engine.
    
    Integrates all V28 components:
    - Regime detection
    - Correlation analysis
    - Signal generation
    - Position sizing
    - Portfolio management
    - Dashboard API
    """
    
    def __init__(self, config: V28Config = None):
        self.config = config or V28Config()
        
        # Initialize components
        logger.info("🚀 Initializing V28 Production Engine...")
        
        # Core components
        self.regime_detector = V28RegimeDetector()
        self.correlation_engine = V28CorrelationEngine(
            lookback=self.config.correlation_lookback
        )
        self.position_sizer = V28KellyPositionSizer(
            kelly_fraction=self.config.kelly_fraction,
            min_position_pct=self.config.min_position_pct,
            max_position_pct=self.config.max_position_pct,
            target_volatility=self.config.target_volatility,
            max_drawdown_limit=self.config.max_drawdown_limit,
            portfolio_value=self.config.initial_capital
        )
        self.signal_generator = SignalGenerator(self.config)
        self.portfolio_manager = PortfolioManager(self.config)
        
        # Dashboard API
        self.dashboard = DashboardAPIServer(
            host=self.config.api_host,
            port=self.config.api_port,
            redis_url=self.config.redis_url
        )
        
        # State
        self.current_regime: Optional[RegimeState] = None
        self.current_correlation: Optional[CorrelationState] = None
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.latest_prices: Dict[str, float] = {}
        
        # Control
        self._running = False
        self._shutdown_event = threading.Event()
        
        # Register correlation alerts
        self.correlation_engine.register_alert_callback(self._handle_correlation_alert)
        
        logger.info("✅ V28 Production Engine initialized")
    
    def _handle_correlation_alert(self, alert: CorrelationAlert):
        """Handle correlation alerts."""
        logger.warning(f"🔔 Correlation Alert: {alert.message}")
        # Could send to dashboard, email, etc.
    
    async def start(self):
        """Start the production engine."""
        logger.info("🚀 Starting V28 Production Engine...")
        
        self._running = True
        
        # Start dashboard API
        await self.dashboard.start()
        
        # Start main trading loop
        asyncio.create_task(self._trading_loop())
        asyncio.create_task(self._regime_update_loop())
        asyncio.create_task(self._correlation_update_loop())
        asyncio.create_task(self._pnl_broadcast_loop())
        
        logger.info("✅ V28 Production Engine running")
        
        # Keep running until shutdown
        while self._running and not self._shutdown_event.is_set():
            await asyncio.sleep(1)
    
    async def stop(self):
        """Stop the production engine."""
        logger.info("🛑 Stopping V28 Production Engine...")
        self._running = False
        self._shutdown_event.set()
        self.dashboard.stop()
        logger.info("✅ V28 Production Engine stopped")
    
    async def _trading_loop(self):
        """Main trading loop."""
        while self._running:
            try:
                await self._execute_trading_cycle()
            except Exception as e:
                logger.error(f"Trading cycle error: {e}")
            
            await asyncio.sleep(self.config.trading_interval_seconds)
    
    async def _regime_update_loop(self):
        """Regime detection update loop."""
        while self._running:
            try:
                await self._update_regime()
            except Exception as e:
                logger.error(f"Regime update error: {e}")
            
            await asyncio.sleep(self.config.regime_update_seconds)
    
    async def _correlation_update_loop(self):
        """Correlation analysis update loop."""
        while self._running:
            try:
                await self._update_correlation()
            except Exception as e:
                logger.error(f"Correlation update error: {e}")
            
            await asyncio.sleep(self.config.correlation_update_seconds)
    
    async def _pnl_broadcast_loop(self):
        """Broadcast P&L updates via WebSocket."""
        while self._running:
            try:
                pnl_data = {
                    'equity': self.portfolio_manager.equity,
                    'daily_pnl': self._calculate_daily_pnl(),
                    'positions': len(self.portfolio_manager.positions),
                    'drawdown': self.portfolio_manager.get_drawdown(),
                    'exposure': self.portfolio_manager.get_current_exposure(),
                    'timestamp': datetime.now().isoformat()
                }
                await self.dashboard.ws_manager.broadcast_pnl(pnl_data)
            except Exception as e:
                logger.error(f"P&L broadcast error: {e}")
            
            await asyncio.sleep(1)
    
    def _calculate_daily_pnl(self) -> float:
        """Calculate today's P&L."""
        today = datetime.now().date()
        today_trades = [
            t for t in self.portfolio_manager.trades
            if datetime.fromisoformat(t.timestamp).date() == today
        ]
        return sum(t.pnl for t in today_trades)
    
    async def _execute_trading_cycle(self):
        """Execute one trading cycle."""
        if not self.current_regime:
            logger.debug("Waiting for regime detection...")
            return
        
        # Update market data (placeholder - in production, use real data feed)
        await self._update_market_data()
        
        # Generate signals for universe
        signals = {}
        for symbol in self.config.universe:
            if symbol in self.market_data:
                signal = self.signal_generator.generate_signal(
                    symbol,
                    self.market_data[symbol],
                    self.current_regime
                )
                signals[symbol] = signal
        
        # Filter actionable signals
        actionable = [
            (sym, sig) for sym, sig in signals.items()
            if abs(sig['signal']) > 0.2 and sig['confidence'] > 0.5
        ]
        
        if actionable and self.config.log_signals:
            logger.info(f"📊 Actionable signals: {len(actionable)}")
            for sym, sig in actionable[:5]:
                logger.info(f"   {sym}: {sig['direction']} "
                           f"(signal: {sig['signal']:.2f}, conf: {sig['confidence']:.2f})")
        
        # Calculate position sizes
        volatilities = {
            sym: self.market_data[sym]['returns'].std() * np.sqrt(252)
            for sym in signals.keys() if sym in self.market_data
        }
        
        # Get correlation matrix
        if self.current_correlation:
            corr_matrix = pd.DataFrame(
                self.current_correlation.correlation_matrix,
                index=list(signals.keys())[:len(self.current_correlation.correlation_matrix)],
                columns=list(signals.keys())[:len(self.current_correlation.correlation_matrix)]
            )
        else:
            corr_matrix = pd.DataFrame()
        
        # Calculate portfolio weights
        portfolio_weights = self.position_sizer.calculate_portfolio_weights(
            symbols=[sym for sym, _ in actionable],
            signals={sym: sig['signal'] for sym, sig in actionable},
            volatilities=volatilities,
            correlation_matrix=corr_matrix,
            market_regime=self.current_regime.market_regime.value,
            vol_regime=self.current_regime.volatility_regime.value
        )
        
        # Update dashboard
        self.dashboard.metrics_calc.update_equity(self.portfolio_manager.equity)
        
        # Update regime info on dashboard
        if self.current_regime:
            regime_info = RegimeInfo(
                regime=self.current_regime.market_regime.value,
                volatility=self.current_regime.volatility_regime.value,
                hmm_state=self.current_regime.hmm_state,
                hmm_state_name=self.current_regime.hmm_state_name,
                garch_forecast=self.current_regime.garch_forecast_1d,
                confidence=self.current_regime.regime_confidence,
                strategy_allocation=self.current_regime.recommended_strategies,
                timestamp=datetime.now().isoformat()
            )
            self.dashboard.update_regime(regime_info)
        
        # Update positions on dashboard
        for pos in self.portfolio_manager.positions.values():
            self.dashboard.update_position(pos)
    
    async def _update_market_data(self):
        """Update market data for universe (placeholder)."""
        # In production, this would connect to real data feed
        # For now, generate synthetic data
        for symbol in self.config.universe:
            if symbol not in self.market_data:
                # Initialize with synthetic data
                n_bars = 252
                dates = pd.date_range(end=datetime.now(), periods=n_bars, freq='D')
                close = 100 * np.cumprod(1 + np.random.randn(n_bars) * 0.02)
                
                self.market_data[symbol] = pd.DataFrame({
                    'date': dates,
                    'open': close * (1 + np.random.randn(n_bars) * 0.01),
                    'high': close * (1 + np.abs(np.random.randn(n_bars)) * 0.02),
                    'low': close * (1 - np.abs(np.random.randn(n_bars)) * 0.02),
                    'close': close,
                    'volume': np.random.randint(1_000_000, 10_000_000, n_bars)
                })
            else:
                # Add new bar (simulate)
                df = self.market_data[symbol]
                last_close = df['close'].iloc[-1]
                new_close = last_close * (1 + np.random.randn() * 0.015)
                
                new_bar = pd.DataFrame({
                    'date': [datetime.now()],
                    'open': [last_close],
                    'high': [max(last_close, new_close) * 1.005],
                    'low': [min(last_close, new_close) * 0.995],
                    'close': [new_close],
                    'volume': [np.random.randint(1_000_000, 10_000_000)]
                })
                
                self.market_data[symbol] = pd.concat([df, new_bar], ignore_index=True).iloc[-252:]
            
            self.latest_prices[symbol] = self.market_data[symbol]['close'].iloc[-1]
        
        # Update position prices
        self.portfolio_manager.update_prices(self.latest_prices)
    
    async def _update_regime(self):
        """Update market regime detection."""
        # Use SPY for regime detection
        if 'SPY' in self.market_data:
            spy_data = self.market_data['SPY']
            self.current_regime = self.regime_detector.detect(spy_data)
            
            # Update position sizer with regime
            self.position_sizer.update_regime(
                self.current_regime.market_regime.value,
                self.current_regime.volatility_regime.value
            )
            
            logger.info(
                f"📈 Regime: {self.current_regime.market_regime.value.upper()} | "
                f"Vol: {self.current_regime.volatility_regime.value} | "
                f"HMM: {self.current_regime.hmm_state_name} ({self.current_regime.hmm_probability:.0%})"
            )
    
    async def _update_correlation(self):
        """Update correlation analysis."""
        if len(self.market_data) < 3:
            return
        
        # Build returns DataFrame
        returns_data = {}
        for symbol, df in self.market_data.items():
            if len(df) >= 60:
                returns_data[symbol] = df['close'].pct_change().iloc[-60:]
        
        if len(returns_data) >= 3:
            returns_df = pd.DataFrame(returns_data)
            self.current_correlation = self.correlation_engine.analyze(returns_df)
            
            logger.info(
                f"📊 Correlation: {self.current_correlation.regime.value.upper()} | "
                f"Avg: {self.current_correlation.average_correlation:.2f} | "
                f"Div Ratio: {self.current_correlation.diversification_ratio:.2f}"
            )
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        return self.dashboard.metrics_calc.get_metrics()


# =============================================================================
# CLI RUNNER
# =============================================================================

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='V28 Production Trading System')
    parser.add_argument('--mode', choices=['paper', 'live', 'backtest'], 
                       default='paper', help='Trading mode')
    parser.add_argument('--host', default='0.0.0.0', help='API host')
    parser.add_argument('--port', type=int, default=8080, help='API port')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    
    args = parser.parse_args()
    
    # Create config
    config = V28Config(
        mode=args.mode,
        api_host=args.host,
        api_port=args.port,
        initial_capital=args.capital
    )
    
    # Ensure logs directory exists
    Path('logs').mkdir(exist_ok=True)
    Path('state').mkdir(exist_ok=True)
    
    # Create engine
    engine = V28ProductionEngine(config)
    
    # Handle shutdown signals
    loop = asyncio.get_event_loop()
    
    def shutdown_handler():
        logger.info("Shutdown signal received")
        asyncio.create_task(engine.stop())
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown_handler)
    
    # Run
    logger.info("=" * 60)
    logger.info("V28 PRODUCTION TRADING SYSTEM")
    logger.info(f"Mode: {config.mode.upper()}")
    logger.info(f"API: http://{config.api_host}:{config.api_port}")
    logger.info(f"Capital: ${config.initial_capital:,.0f}")
    logger.info("=" * 60)
    
    try:
        await engine.start()
    except KeyboardInterrupt:
        await engine.stop()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        await engine.stop()
        raise


if __name__ == '__main__':
    asyncio.run(main())
