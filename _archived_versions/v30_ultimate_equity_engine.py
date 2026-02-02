#!/usr/bin/env python3
"""
V30 ULTIMATE EQUITY ENGINE - Maximum Profit NYSE Trading System
===============================================================
Production-ready algorithmic trading system for NYSE via Alpaca API

Features:
- Multi-factor alpha generation (momentum, value, quality, volatility)
- Machine learning ensemble for signal confirmation
- Adaptive position sizing based on conviction and volatility
- Sector rotation with macro regime awareness
- Smart order execution (TWAP/VWAP targeting)
- Real-time drawdown protection with dynamic hedging

Usage:
    python v30_ultimate_equity_engine.py --mode [backtest|live|paper]
    python v30_ultimate_equity_engine.py --status
    python v30_ultimate_equity_engine.py --analyze TICKER

Author: Algebraic Topology Neural Net Strategy
Version: 30.0.0
"""

import os
import sys
import json
import logging
import argparse
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.rest import REST, TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("Warning: alpaca-trade-api not installed")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class V30Config:
    """Ultimate configuration for V30 Equity Engine"""
    
    # API Configuration
    alpaca_api_key: str = field(default_factory=lambda: os.getenv('ALPACA_API_KEY', ''))
    alpaca_secret_key: str = field(default_factory=lambda: os.getenv('ALPACA_SECRET_KEY', ''))
    alpaca_base_url: str = field(default_factory=lambda: os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets'))
    
    # Portfolio Configuration
    max_portfolio_value: float = 100000.0
    max_positions: int = 25
    max_position_size: float = 0.08  # 8% max per position
    min_position_size: float = 0.02  # 2% min per position
    
    # Sector Limits
    max_sector_exposure: float = 0.35  # 35% max per sector
    
    # Risk Management
    max_drawdown_threshold: float = 0.10  # 10% max drawdown
    drawdown_reduction_factor: float = 0.5  # Reduce exposure by 50% at threshold
    stop_loss_percent: float = 0.07  # 7% stop loss
    trailing_stop_percent: float = 0.05  # 5% trailing stop
    
    # Factor Weights (must sum to 1.0)
    momentum_weight: float = 0.30
    value_weight: float = 0.25
    quality_weight: float = 0.25
    volatility_weight: float = 0.20
    
    # ML Configuration
    ml_confidence_threshold: float = 0.65
    ml_lookback_days: int = 252
    ml_retrain_frequency: int = 5  # Retrain every 5 days
    
    # Execution Configuration
    use_twap: bool = True
    twap_intervals: int = 5
    min_volume_filter: int = 500000
    max_spread_percent: float = 0.01
    
    # Logging
    log_level: str = 'INFO'
    log_file: str = 'v30_engine.log'


# =============================================================================
# MARKET REGIME DETECTION
# =============================================================================

class MarketRegime(Enum):
    """Market regime classification"""
    BULL_HIGH_VOL = "bull_high_vol"
    BULL_LOW_VOL = "bull_low_vol"
    BEAR_HIGH_VOL = "bear_high_vol"
    BEAR_LOW_VOL = "bear_low_vol"
    SIDEWAYS = "sideways"
    CRISIS = "crisis"

class RegimeDetector:
    """Detect current market regime using multiple indicators"""
    
    def __init__(self, lookback: int = 60):
        self.lookback = lookback
        self.vix_threshold_high = 25
        self.vix_threshold_crisis = 35
        
    def detect(self, spy_data: pd.DataFrame, vix_data: Optional[pd.DataFrame] = None) -> MarketRegime:
        """Detect current market regime"""
        if len(spy_data) < self.lookback:
            return MarketRegime.SIDEWAYS
            
        returns = spy_data['close'].pct_change().dropna()
        recent_return = returns.tail(20).sum()
        volatility = returns.tail(20).std() * np.sqrt(252)
        
        # Trend detection using SMA crossover
        sma_20 = spy_data['close'].rolling(20).mean().iloc[-1]
        sma_50 = spy_data['close'].rolling(50).mean().iloc[-1]
        current_price = spy_data['close'].iloc[-1]
        
        is_uptrend = current_price > sma_20 > sma_50
        is_downtrend = current_price < sma_20 < sma_50
        
        # Volatility classification
        if vix_data is not None and len(vix_data) > 0:
            current_vix = vix_data['close'].iloc[-1]
            is_high_vol = current_vix > self.vix_threshold_high
            is_crisis = current_vix > self.vix_threshold_crisis
        else:
            is_high_vol = volatility > 0.20
            is_crisis = volatility > 0.35
            
        # Regime classification
        if is_crisis:
            return MarketRegime.CRISIS
        elif is_uptrend:
            return MarketRegime.BULL_HIGH_VOL if is_high_vol else MarketRegime.BULL_LOW_VOL
        elif is_downtrend:
            return MarketRegime.BEAR_HIGH_VOL if is_high_vol else MarketRegime.BEAR_LOW_VOL
        else:
            return MarketRegime.SIDEWAYS
            
    def get_regime_multiplier(self, regime: MarketRegime) -> float:
        """Get position sizing multiplier based on regime"""
        multipliers = {
            MarketRegime.BULL_LOW_VOL: 1.2,
            MarketRegime.BULL_HIGH_VOL: 0.9,
            MarketRegime.SIDEWAYS: 0.8,
            MarketRegime.BEAR_LOW_VOL: 0.6,
            MarketRegime.BEAR_HIGH_VOL: 0.4,
            MarketRegime.CRISIS: 0.2
        }
        return multipliers.get(regime, 0.5)

# =============================================================================
# MULTI-FACTOR ALPHA GENERATOR
# =============================================================================

class AlphaGenerator:
    """Generate alpha signals from multiple factors"""
    
    def __init__(self, config: V30Config):
        self.config = config
        self.logger = logging.getLogger('AlphaGenerator')
        
    def calculate_momentum_score(self, df: pd.DataFrame) -> float:
        """Calculate momentum factor score"""
        if len(df) < 252:
            return 0.0
            
        # Multi-period momentum
        ret_5d = df['close'].pct_change(5).iloc[-1]
        ret_20d = df['close'].pct_change(20).iloc[-1]
        ret_60d = df['close'].pct_change(60).iloc[-1]
        ret_252d = df['close'].pct_change(252).iloc[-1]
        
        # Momentum with skip (exclude most recent month)
        ret_12_1 = (df['close'].iloc[-21] / df['close'].iloc[-252] - 1) if len(df) >= 252 else 0
        
        # Weight recent momentum more heavily
        momentum = (0.1 * ret_5d + 0.2 * ret_20d + 0.3 * ret_60d + 0.4 * ret_12_1)
        
        return np.clip(momentum, -1, 1)
        
    def calculate_value_score(self, fundamentals: Dict) -> float:
        """Calculate value factor score from fundamentals"""
        scores = []
        
        # P/E ratio (lower is better)
        pe = fundamentals.get('pe_ratio')
        if pe and pe > 0:
            pe_score = 1 - np.clip(pe / 50, 0, 1)
            scores.append(pe_score)
            
        # P/B ratio (lower is better)
        pb = fundamentals.get('pb_ratio')
        if pb and pb > 0:
            pb_score = 1 - np.clip(pb / 10, 0, 1)
            scores.append(pb_score)
            
        # Dividend yield (higher is better)
        div_yield = fundamentals.get('dividend_yield', 0)
        if div_yield:
            div_score = np.clip(div_yield / 0.05, 0, 1)
            scores.append(div_score)
            
        return np.mean(scores) if scores else 0.5
        
    def calculate_quality_score(self, fundamentals: Dict) -> float:
        """Calculate quality factor score"""
        scores = []
        
        # ROE (higher is better)
        roe = fundamentals.get('roe')
        if roe:
            roe_score = np.clip(roe / 0.30, 0, 1)
            scores.append(roe_score)
            
        # Profit margin (higher is better)
        margin = fundamentals.get('profit_margin')
        if margin:
            margin_score = np.clip(margin / 0.25, 0, 1)
            scores.append(margin_score)
            
        # Debt to equity (lower is better)
        de = fundamentals.get('debt_to_equity')
        if de is not None:
            de_score = 1 - np.clip(de / 2, 0, 1)
            scores.append(de_score)
            
        return np.mean(scores) if scores else 0.5
        
    def calculate_volatility_score(self, df: pd.DataFrame) -> float:
        """Calculate low volatility factor score (lower vol = higher score)"""
        if len(df) < 60:
            return 0.5
            
        returns = df['close'].pct_change().dropna()
        vol_60d = returns.tail(60).std() * np.sqrt(252)
        
        # Lower volatility gets higher score
        vol_score = 1 - np.clip(vol_60d / 0.50, 0, 1)
        
        return vol_score
        
    def generate_composite_alpha(self, df: pd.DataFrame, fundamentals: Dict) -> Tuple[float, Dict]:
        """Generate composite alpha score from all factors"""
        momentum = self.calculate_momentum_score(df)
        value = self.calculate_value_score(fundamentals)
        quality = self.calculate_quality_score(fundamentals)
        volatility = self.calculate_volatility_score(df)
        
        # Weighted composite
        composite = (
            self.config.momentum_weight * momentum +
            self.config.value_weight * value +
            self.config.quality_weight * quality +
            self.config.volatility_weight * volatility
        )
        
        factor_breakdown = {
            'momentum': momentum,
            'value': value,
            'quality': quality,
            'volatility': volatility,
            'composite': composite
        }
        
        return composite, factor_breakdown


# =============================================================================
# ML ENSEMBLE FOR SIGNAL CONFIRMATION
# =============================================================================

class MLEnsemble:
    """Machine Learning ensemble for trade signal confirmation"""
    
    def __init__(self, config: V30Config):
        self.config = config
        self.logger = logging.getLogger('MLEnsemble')
        self.scaler = StandardScaler()
        self.models = {
            'rf': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
        }
        self.is_trained = False
        self.last_train_date = None
        
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare feature matrix from price data"""
        features = pd.DataFrame(index=df.index)
        
        # Price-based features
        features['ret_1d'] = df['close'].pct_change(1)
        features['ret_5d'] = df['close'].pct_change(5)
        features['ret_20d'] = df['close'].pct_change(20)
        features['ret_60d'] = df['close'].pct_change(60)
        
        # Volatility features
        features['vol_5d'] = features['ret_1d'].rolling(5).std()
        features['vol_20d'] = features['ret_1d'].rolling(20).std()
        
        # Momentum features
        features['rsi'] = self._calculate_rsi(df['close'], 14)
        features['macd'] = self._calculate_macd(df['close'])
        
        # Mean reversion features
        features['bb_position'] = self._calculate_bollinger_position(df['close'])
        features['dist_from_sma20'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).mean()
        features['dist_from_sma50'] = (df['close'] - df['close'].rolling(50).mean()) / df['close'].rolling(50).mean()
        
        # Volume features
        if 'volume' in df.columns:
            features['vol_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
            
        return features.dropna()
        
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def _calculate_macd(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        return exp1 - exp2
        
    def _calculate_bollinger_position(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """Calculate position within Bollinger Bands"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        return (prices - lower) / (upper - lower)
        
    def train(self, df: pd.DataFrame) -> bool:
        """Train the ML ensemble on historical data"""
        try:
            features = self.prepare_features(df)
            if len(features) < 100:
                self.logger.warning("Insufficient data for training")
                return False
                
            # Create labels: 1 if next 5-day return > 0, else 0
            future_returns = df['close'].pct_change(5).shift(-5)
            labels = (future_returns > 0).astype(int)
            
            # Align features and labels
            common_idx = features.index.intersection(labels.dropna().index)
            X = features.loc[common_idx].values
            y = labels.loc[common_idx].values
            
            # Use only training data (exclude last 20 days)
            X_train = X[:-20]
            y_train = y[:-20]
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X_train)
            
            # Train models
            for name, model in self.models.items():
                model.fit(X_scaled, y_train)
                self.logger.info(f"Trained {name} model")
                
            self.is_trained = True
            self.last_train_date = datetime.now()
            return True
            
        except Exception as e:
            self.logger.error(f"Error training ML ensemble: {e}")
            return False
            
    def predict(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Predict signal probability and confidence"""
        if not self.is_trained:
            return 0.5, 0.0
            
        try:
            features = self.prepare_features(df)
            if len(features) < 1:
                return 0.5, 0.0
                
            X = features.iloc[-1:].values
            X_scaled = self.scaler.transform(X)
            
            # Get predictions from all models
            probabilities = []
            for name, model in self.models.items():
                prob = model.predict_proba(X_scaled)[0][1]
                probabilities.append(prob)
                
            # Ensemble average
            avg_prob = np.mean(probabilities)
            
            # Confidence is based on agreement between models
            confidence = 1 - np.std(probabilities)
            
            return avg_prob, confidence
            
        except Exception as e:
            self.logger.error(f"Error in ML prediction: {e}")
            return 0.5, 0.0

# =============================================================================
# POSITION SIZER
# =============================================================================

class AdaptivePositionSizer:
    """Adaptive position sizing based on conviction, volatility, and regime"""
    
    def __init__(self, config: V30Config):
        self.config = config
        self.logger = logging.getLogger('PositionSizer')
        
    def calculate_position_size(
        self,
        portfolio_value: float,
        alpha_score: float,
        ml_confidence: float,
        volatility: float,
        regime_multiplier: float,
        current_drawdown: float = 0.0
    ) -> float:
        """Calculate optimal position size"""
        
        # Base position size from alpha strength
        base_size = self.config.min_position_size + \
                    (self.config.max_position_size - self.config.min_position_size) * \
                    np.clip(alpha_score, 0, 1)
        
        # Adjust for ML confidence
        confidence_multiplier = 0.5 + 0.5 * ml_confidence
        
        # Adjust for volatility (reduce size in high vol)
        vol_multiplier = np.clip(0.20 / max(volatility, 0.10), 0.5, 1.5)
        
        # Apply regime multiplier
        size = base_size * confidence_multiplier * vol_multiplier * regime_multiplier
        
        # Reduce size if in drawdown
        if current_drawdown > self.config.max_drawdown_threshold * 0.5:
            drawdown_factor = 1 - (current_drawdown / self.config.max_drawdown_threshold)
            size *= max(drawdown_factor, 0.3)
            
        # Enforce limits
        size = np.clip(size, self.config.min_position_size, self.config.max_position_size)
        
        return size * portfolio_value


# =============================================================================
# SMART ORDER EXECUTION (TWAP/VWAP)
# =============================================================================

class SmartExecutor:
    """Smart order execution with TWAP/VWAP targeting"""
    
    def __init__(self, config: V30Config, api: Any = None):
        self.config = config
        self.api = api
        self.logger = logging.getLogger('SmartExecutor')
        self.execution_stats = {'twap': [], 'vwap': [], 'market': []}
        
    async def execute_vwap(self, symbol: str, side: str, quantity: int,
                           duration_minutes: int = 30, participation_rate: float = 0.10) -> List[Dict]:
        """Execute order using Volume-Weighted Average Price strategy"""
        if not self.api:
            self.logger.warning("No API connection for VWAP execution")
            return []
        
        executions = []
        # Get historical volume profile (hourly pattern)
        # U-shaped volume: high at open/close, low midday
        intervals = 6
        volume_weights = [0.25, 0.12, 0.08, 0.10, 0.15, 0.30]  # Open to close
        interval_seconds = (duration_minutes * 60) // intervals
        
        for i, weight in enumerate(volume_weights):
            slice_qty = int(quantity * weight)
            if slice_qty <= 0:
                continue
            
            # Add randomization to avoid detection
            jitter = np.random.uniform(0.9, 1.1)
            slice_qty = int(slice_qty * jitter)
            
            try:
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=max(1, slice_qty),
                    side=side,
                    type='market',
                    time_in_force='day'
                )
                executions.append({
                    'order_id': order.id,
                    'symbol': symbol,
                    'qty': slice_qty,
                    'side': side,
                    'slice': i + 1,
                    'algo': 'VWAP',
                    'weight': weight
                })
                self.logger.info(f"VWAP slice {i+1}/{intervals}: {side} {slice_qty} {symbol} (weight={weight:.0%})")
                
                if i < intervals - 1:
                    await asyncio.sleep(interval_seconds)
                    
            except Exception as e:
                self.logger.error(f"VWAP execution error: {e}")
        
        self.execution_stats['vwap'].append({'symbol': symbol, 'qty': quantity, 'slices': len(executions)})
        return executions
    
    async def execute_adaptive(self, symbol: str, side: str, quantity: int,
                               urgency: float = 0.5) -> List[Dict]:
        """Adaptive execution based on urgency level (0=patient, 1=aggressive)"""
        if urgency >= 0.8:
            # Urgent - single market order
            self.logger.info(f"Adaptive execution (URGENT): {side} {quantity} {symbol}")
            if self.api:
                try:
                    order = self.api.submit_order(
                        symbol=symbol, qty=quantity, side=side,
                        type='market', time_in_force='day'
                    )
                    return [{'order_id': order.id, 'symbol': symbol, 'qty': quantity, 'algo': 'MARKET'}]
                except Exception as e:
                    self.logger.error(f"Market order error: {e}")
            return []
        elif urgency >= 0.5:
            # Moderate - short TWAP
            duration = int(15 * (1 - urgency) + 5)  # 5-15 minutes
            return await self.execute_twap(symbol, side, quantity, duration)
        else:
            # Patient - full VWAP
            duration = int(30 * (1 - urgency) + 15)  # 15-45 minutes
            return await self.execute_vwap(symbol, side, quantity, duration)
        
    async def execute_twap(self, symbol: str, side: str, quantity: int, 
                          duration_minutes: int = 30) -> List[Dict]:
        """Execute order using Time-Weighted Average Price strategy"""
        if not self.api:
            self.logger.warning("No API connection for execution")
            return []
            
        executions = []
        intervals = self.config.twap_intervals
        slice_qty = quantity // intervals
        remainder = quantity % intervals
        interval_seconds = (duration_minutes * 60) // intervals
        
        for i in range(intervals):
            qty = slice_qty + (1 if i < remainder else 0)
            if qty <= 0:
                continue
                
            try:
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    type='market',
                    time_in_force='day'
                )
                executions.append({
                    'order_id': order.id,
                    'symbol': symbol,
                    'qty': qty,
                    'side': side,
                    'slice': i + 1
                })
                self.logger.info(f"TWAP slice {i+1}/{intervals}: {side} {qty} {symbol}")
                
                if i < intervals - 1:
                    await asyncio.sleep(interval_seconds)
                    
            except Exception as e:
                self.logger.error(f"TWAP execution error: {e}")
        
        self.execution_stats['twap'].append({'symbol': symbol, 'qty': quantity, 'slices': len(executions)})
        return executions


# =============================================================================
# DRAWDOWN PROTECTOR WITH DYNAMIC HEDGING
# =============================================================================

class DrawdownProtector:
    """Real-time drawdown protection with dynamic hedging"""
    
    # Hedge instruments for tail risk protection
    HEDGE_INSTRUMENTS = {
        'inverse_etf': ['SH', 'PSQ', 'DOG'],  # 1x inverse S&P, QQQ, DOW
        'leveraged_inverse': ['SQQQ', 'SPXS', 'SDOW'],  # 3x inverse
        'volatility': ['VXX', 'UVXY', 'VIXY'],  # VIX-based
        'put_protection': ['SPY', 'QQQ']  # For put options
    }
    
    def __init__(self, config: V30Config, api: Any = None):
        self.config = config
        self.api = api
        self.logger = logging.getLogger('DrawdownProtector')
        self.peak_equity = config.max_portfolio_value
        self.current_drawdown = 0.0
        self.hedge_positions = {}
        self.is_halted = False
        self.halt_time = None
        self.drawdown_history = []
        
    def update_equity(self, current_equity: float) -> Tuple[float, bool, str]:
        """
        Update equity and check drawdown status.
        Returns: (drawdown_pct, should_reduce_exposure, action_message)
        """
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
        self.drawdown_history.append((datetime.now(), self.current_drawdown))
        
        # Keep only last 1000 entries
        if len(self.drawdown_history) > 1000:
            self.drawdown_history = self.drawdown_history[-1000:]
        
        action_msg = "NORMAL"
        should_reduce = False
        
        # Hard stop - halt trading
        if self.current_drawdown >= self.config.max_drawdown_threshold:
            if not self.is_halted:
                self.is_halted = True
                self.halt_time = datetime.now()
                self.logger.critical(f"TRADING HALTED: Drawdown {self.current_drawdown:.1%} >= {self.config.max_drawdown_threshold:.1%}")
            action_msg = "HALT"
            should_reduce = True
        # Soft threshold - reduce exposure
        elif self.current_drawdown >= self.config.max_drawdown_threshold * 0.5:
            action_msg = "REDUCE"
            should_reduce = True
            self.logger.warning(f"Drawdown warning: {self.current_drawdown:.1%} - reducing exposure")
        # Recovery check
        elif self.is_halted and self.current_drawdown < self.config.max_drawdown_threshold * 0.3:
            self.is_halted = False
            self.halt_time = None
            action_msg = "RESUME"
            self.logger.info(f"Trading resumed - drawdown recovered to {self.current_drawdown:.1%}")
        
        return self.current_drawdown, should_reduce, action_msg
    
    def calculate_hedge_allocation(self, market_regime: MarketRegime, vix_level: float = None) -> float:
        """Calculate target hedge allocation based on conditions"""
        # Base allocation from drawdown
        dd_hedge = 0.0
        if self.current_drawdown > 0.03:
            dd_hedge = min(0.15, self.current_drawdown * 0.8)
        
        # Regime-based hedge
        regime_hedge = {
            MarketRegime.BULL_LOW_VOL: 0.0,
            MarketRegime.BULL_HIGH_VOL: 0.02,
            MarketRegime.SIDEWAYS: 0.03,
            MarketRegime.BEAR_LOW_VOL: 0.05,
            MarketRegime.BEAR_HIGH_VOL: 0.08,
            MarketRegime.CRISIS: 0.15
        }.get(market_regime, 0.03)
        
        # VIX-based override
        vix_hedge = 0.0
        if vix_level:
            if vix_level > 35:
                vix_hedge = 0.15
            elif vix_level > 30:
                vix_hedge = 0.10
            elif vix_level > 25:
                vix_hedge = 0.06
            elif vix_level > 20:
                vix_hedge = 0.03
        
        # Take maximum
        return min(0.20, max(dd_hedge, regime_hedge, vix_hedge))  # Cap at 20%
    
    def get_hedge_orders(self, portfolio_value: float, current_hedge_value: float,
                         market_regime: MarketRegime, vix_level: float = None) -> List[Dict]:
        """Generate hedge orders based on target allocation"""
        target_allocation = self.calculate_hedge_allocation(market_regime, vix_level)
        target_value = portfolio_value * target_allocation
        
        orders = []
        
        # Calculate adjustment needed
        adjustment = target_value - current_hedge_value
        
        if abs(adjustment) < 500:  # Skip small adjustments
            return orders
        
        # Choose hedge instrument based on urgency
        if self.current_drawdown > 0.10 or market_regime == MarketRegime.CRISIS:
            # Use leveraged inverse for crisis
            hedge_symbol = 'SQQQ'  # 3x inverse QQQ
        elif self.current_drawdown > 0.05 or market_regime in [MarketRegime.BEAR_HIGH_VOL, MarketRegime.BEAR_LOW_VOL]:
            # Use 1x inverse
            hedge_symbol = 'SH'  # 1x inverse SPY
        else:
            # Use VIX-based protection
            hedge_symbol = 'VIXY'
        
        if adjustment > 0:
            orders.append({
                'symbol': hedge_symbol,
                'side': 'buy',
                'value': adjustment,
                'type': 'hedge',
                'reason': f'DD={self.current_drawdown:.1%}, Regime={market_regime.value}'
            })
        elif adjustment < 0:
            orders.append({
                'symbol': hedge_symbol,
                'side': 'sell',
                'value': abs(adjustment),
                'type': 'close_hedge',
                'reason': 'Reducing hedge allocation'
            })
        
        return orders
    
    def get_position_scaling_factor(self) -> float:
        """Get position scaling factor based on drawdown"""
        if self.is_halted:
            return 0.0
        
        if self.current_drawdown < self.config.max_drawdown_threshold * 0.3:
            return 1.0
        elif self.current_drawdown < self.config.max_drawdown_threshold * 0.5:
            return 0.8
        elif self.current_drawdown < self.config.max_drawdown_threshold * 0.7:
            return 0.5
        else:
            return 0.3


# =============================================================================
# SECTOR ROTATOR
# =============================================================================

class SectorRotator:
    """Sector rotation based on momentum and macro regime"""
    
    SECTORS = {
        'Technology': ['XLK', 'AAPL', 'MSFT', 'NVDA', 'GOOGL'],
        'Healthcare': ['XLV', 'JNJ', 'UNH', 'PFE', 'ABBV'],
        'Financial': ['XLF', 'JPM', 'BAC', 'GS', 'WFC'],
        'Energy': ['XLE', 'XOM', 'CVX', 'COP', 'SLB'],
        'Consumer': ['XLY', 'AMZN', 'TSLA', 'HD', 'NKE'],
        'Staples': ['XLP', 'PG', 'KO', 'PEP', 'WMT'],
        'Industrial': ['XLI', 'CAT', 'BA', 'UNP', 'HON'],
        'Utilities': ['XLU', 'NEE', 'DUK', 'SO', 'AEP'],
        'Materials': ['XLB', 'LIN', 'APD', 'FCX', 'NEM'],
        'RealEstate': ['XLRE', 'AMT', 'PLD', 'SPG', 'EQIX']
    }
    
    # Regime-sector preferences
    REGIME_TILTS = {
        MarketRegime.BULL_LOW_VOL: {'Technology': 0.25, 'Consumer': 0.20, 'Financial': 0.15},
        MarketRegime.BULL_HIGH_VOL: {'Technology': 0.15, 'Healthcare': 0.15, 'Staples': 0.10},
        MarketRegime.SIDEWAYS: {'Financial': 0.15, 'Healthcare': 0.15, 'Industrial': 0.10},
        MarketRegime.BEAR_LOW_VOL: {'Staples': 0.25, 'Utilities': 0.20, 'Healthcare': 0.15},
        MarketRegime.BEAR_HIGH_VOL: {'Utilities': 0.25, 'Staples': 0.20, 'Healthcare': 0.15},
        MarketRegime.CRISIS: {'Staples': 0.30, 'Utilities': 0.25, 'Healthcare': 0.20}
    }
    
    def __init__(self, config: V30Config):
        self.config = config
        self.logger = logging.getLogger('SectorRotator')
        self.sector_momentum = {}
        self.sector_weights = {}
        
    def calculate_sector_momentum(self, sector_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Calculate momentum score for each sector"""
        momentum_scores = {}
        
        for sector, symbols in self.SECTORS.items():
            scores = []
            for symbol in symbols:
                if symbol in sector_data and len(sector_data[symbol]) >= 60:
                    df = sector_data[symbol]
                    mom_20 = df['close'].iloc[-1] / df['close'].iloc[-20] - 1
                    mom_60 = df['close'].iloc[-1] / df['close'].iloc[-60] - 1
                    scores.append(0.6 * mom_20 + 0.4 * mom_60)
            
            momentum_scores[sector] = np.mean(scores) if scores else 0.0
        
        self.sector_momentum = momentum_scores
        return momentum_scores
    
    def get_sector_weights(self, regime: MarketRegime) -> Dict[str, float]:
        """Get target sector weights based on regime and momentum"""
        # Rank sectors by momentum
        sorted_sectors = sorted(self.sector_momentum.items(), key=lambda x: x[1], reverse=True)
        
        weights = {}
        n_sectors = len(sorted_sectors)
        
        for i, (sector, mom_score) in enumerate(sorted_sectors):
            # Base weight from momentum rank
            rank_weight = max(0, (n_sectors - i) / n_sectors) * 0.15  # Up to 15%
            
            # Regime tilt
            regime_tilt = self.REGIME_TILTS.get(regime, {}).get(sector, 0.05)
            
            # Combine: 60% momentum, 40% regime
            weights[sector] = 0.6 * rank_weight + 0.4 * regime_tilt
        
        # Normalize to sum to 1
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        # Apply max sector constraint
        max_sector = self.config.max_sector_exposure
        for sector in weights:
            weights[sector] = min(weights[sector], max_sector)
        
        # Re-normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        self.sector_weights = weights
        return weights
    
    def check_sector_constraints(self, positions: Dict[str, float], 
                                  total_equity: float) -> Dict[str, float]:
        """Check and return any sector overweights"""
        sector_values = {sector: 0.0 for sector in self.SECTORS}
        
        for symbol, value in positions.items():
            for sector, symbols in self.SECTORS.items():
                if symbol in symbols:
                    sector_values[sector] += value
                    break
        
        overweights = {}
        for sector, value in sector_values.items():
            allocation = value / total_equity if total_equity > 0 else 0
            if allocation > self.config.max_sector_exposure:
                overweights[sector] = allocation - self.config.max_sector_exposure
        
        return overweights


# =============================================================================
# V30 ULTIMATE EQUITY ENGINE - MAIN CLASS
# =============================================================================

class V30UltimateEquityEngine:
    """Ultimate production-ready equity trading engine"""
    
    # Sector mappings for NYSE stocks
    SECTOR_MAP = {
        'XLK': 'Technology', 'AAPL': 'Technology', 'MSFT': 'Technology', 'NVDA': 'Technology',
        'XLF': 'Financial', 'JPM': 'Financial', 'BAC': 'Financial', 'GS': 'Financial',
        'XLV': 'Healthcare', 'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'PFE': 'Healthcare',
        'XLE': 'Energy', 'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy',
        'XLY': 'Consumer', 'AMZN': 'Consumer', 'TSLA': 'Consumer', 'HD': 'Consumer',
        'XLP': 'Staples', 'PG': 'Staples', 'KO': 'Staples', 'WMT': 'Staples',
        'XLI': 'Industrial', 'CAT': 'Industrial', 'BA': 'Industrial', 'GE': 'Industrial',
        'XLU': 'Utilities', 'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities',
        'XLRE': 'RealEstate', 'SPG': 'RealEstate', 'AMT': 'RealEstate',
        'XLB': 'Materials', 'LIN': 'Materials', 'APD': 'Materials',
        'XLC': 'Communication', 'GOOGL': 'Communication', 'META': 'Communication'
    }
    
    def __init__(self, config: V30Config = None):
        self.config = config or V30Config()
        self.logger = self._setup_logging()
        
        # Initialize components
        self.regime_detector = RegimeDetector()
        self.alpha_generator = AlphaGenerator(self.config)
        self.ml_ensemble = MLEnsemble(self.config)
        self.position_sizer = AdaptivePositionSizer(self.config)
        
        # API connection
        self.api = None
        if ALPACA_AVAILABLE and self.config.alpaca_api_key:
            self.api = REST(
                self.config.alpaca_api_key,
                self.config.alpaca_secret_key,
                self.config.alpaca_base_url
            )
            self.executor = SmartExecutor(self.config, self.api)
        else:
            self.executor = SmartExecutor(self.config)
            
        # Additional components
        self.drawdown_protector = DrawdownProtector(self.config, self.api)
        self.sector_rotator = SectorRotator(self.config)
        
        # State tracking
        self.positions = {}
        self.trade_history = []
        self.daily_pnl = 0.0
        self.peak_value = 0.0
        self.current_drawdown = 0.0
        self.hedge_positions = {}
        
        self.logger.info("V30 Ultimate Equity Engine initialized with all components")
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('V30Engine')
        logger.setLevel(getattr(logging, self.config.log_level))
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File handler
        fh = logging.FileHandler(self.config.log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        return logger
        
    def get_portfolio_value(self) -> float:
        """Get current portfolio value from Alpaca"""
        if self.api:
            try:
                account = self.api.get_account()
                return float(account.portfolio_value)
            except Exception as e:
                self.logger.error(f"Error getting portfolio value: {e}")
        return self.config.max_portfolio_value
        
    def get_positions(self) -> Dict:
        """Get current positions from Alpaca"""
        if self.api:
            try:
                positions = self.api.list_positions()
                return {p.symbol: {
                    'qty': int(p.qty),
                    'market_value': float(p.market_value),
                    'unrealized_pl': float(p.unrealized_pl),
                    'avg_entry_price': float(p.avg_entry_price),
                    'current_price': float(p.current_price)
                } for p in positions}
            except Exception as e:
                self.logger.error(f"Error getting positions: {e}")
        return {}
        
    def get_historical_data(self, symbol: str, days: int = 252) -> pd.DataFrame:
        """Get historical price data"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 30)  # Extra buffer
        
        if self.api:
            try:
                bars = self.api.get_bars(
                    symbol,
                    TimeFrame.Day,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d')
                ).df
                
                if len(bars) > 0:
                    bars.index = pd.to_datetime(bars.index).tz_localize(None)
                    return bars
            except Exception as e:
                self.logger.warning(f"Alpaca data error for {symbol}: {e}")
                
        # Fallback to yfinance
        if YFINANCE_AVAILABLE:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=f"{days}d")
                df.columns = [c.lower() for c in df.columns]
                return df
            except Exception as e:
                self.logger.warning(f"yfinance error for {symbol}: {e}")
                
        return pd.DataFrame()
        
    def get_fundamentals(self, symbol: str) -> Dict:
        """Get fundamental data for a symbol"""
        fundamentals = {
            'pe_ratio': None,
            'pb_ratio': None,
            'dividend_yield': None,
            'roe': None,
            'profit_margin': None,
            'debt_to_equity': None
        }
        
        if YFINANCE_AVAILABLE:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                fundamentals['pe_ratio'] = info.get('trailingPE')
                fundamentals['pb_ratio'] = info.get('priceToBook')
                fundamentals['dividend_yield'] = info.get('dividendYield')
                fundamentals['roe'] = info.get('returnOnEquity')
                fundamentals['profit_margin'] = info.get('profitMargins')
                fundamentals['debt_to_equity'] = info.get('debtToEquity')
                
            except Exception as e:
                self.logger.warning(f"Error getting fundamentals for {symbol}: {e}")
                
        return fundamentals


    def analyze_symbol(self, symbol: str) -> Dict:
        """Comprehensive analysis of a single symbol"""
        self.logger.info(f"Analyzing {symbol}...")
        
        # Get data
        df = self.get_historical_data(symbol)
        if len(df) < 60:
            return {'symbol': symbol, 'error': 'Insufficient data'}
            
        fundamentals = self.get_fundamentals(symbol)
        
        # Generate alpha
        alpha_score, factor_breakdown = self.alpha_generator.generate_composite_alpha(df, fundamentals)
        
        # ML prediction
        if not self.ml_ensemble.is_trained:
            spy_data = self.get_historical_data('SPY')
            self.ml_ensemble.train(spy_data)
        ml_prob, ml_confidence = self.ml_ensemble.predict(df)
        
        # Get regime
        spy_data = self.get_historical_data('SPY', 100)
        regime = self.regime_detector.detect(spy_data)
        regime_mult = self.regime_detector.get_regime_multiplier(regime)
        
        # Calculate volatility
        returns = df['close'].pct_change().dropna()
        volatility = returns.tail(20).std() * np.sqrt(252)
        
        # Calculate position size
        portfolio_value = self.get_portfolio_value()
        position_size = self.position_sizer.calculate_position_size(
            portfolio_value, alpha_score, ml_confidence, volatility, regime_mult
        )
        
        return {
            'symbol': symbol,
            'alpha_score': alpha_score,
            'factor_breakdown': factor_breakdown,
            'ml_probability': ml_prob,
            'ml_confidence': ml_confidence,
            'regime': regime.value,
            'regime_multiplier': regime_mult,
            'volatility': volatility,
            'suggested_position_size': position_size,
            'signal': 'BUY' if alpha_score > 0.3 and ml_prob > 0.55 else ('SELL' if alpha_score < -0.2 else 'HOLD')
        }
        
    def generate_signals(self, universe: List[str]) -> List[Dict]:
        """Generate trading signals for entire universe"""
        signals = []
        
        for symbol in universe:
            try:
                analysis = self.analyze_symbol(symbol)
                if 'error' not in analysis:
                    signals.append(analysis)
            except Exception as e:
                self.logger.error(f"Error analyzing {symbol}: {e}")
                
        # Sort by alpha score
        signals.sort(key=lambda x: x.get('alpha_score', 0), reverse=True)
        
        return signals
        
    def get_status(self) -> Dict:
        """Get current engine status"""
        portfolio_value = self.get_portfolio_value()
        positions = self.get_positions()
        
        # Calculate metrics
        total_unrealized_pl = sum(p['unrealized_pl'] for p in positions.values())
        
        # Update drawdown tracking
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
        self.current_drawdown = (self.peak_value - portfolio_value) / self.peak_value if self.peak_value > 0 else 0
        
        # Get regime
        spy_data = self.get_historical_data('SPY', 100)
        regime = self.regime_detector.detect(spy_data)
        
        # Get sector weights
        sector_allocation = {}
        for symbol, pos_data in positions.items():
            for sector, symbols in self.SECTOR_MAP.items():
                if symbol in symbols or symbol == sector:
                    sector_allocation[sector] = sector_allocation.get(sector, 0) + pos_data['market_value']
        
        # Update drawdown protector
        dd_pct, should_reduce, action = self.drawdown_protector.update_equity(portfolio_value)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'portfolio_value': portfolio_value,
            'peak_value': self.drawdown_protector.peak_equity,
            'current_drawdown': dd_pct,
            'drawdown_action': action,
            'trading_halted': self.drawdown_protector.is_halted,
            'position_scaling': self.drawdown_protector.get_position_scaling_factor(),
            'num_positions': len(positions),
            'positions': positions,
            'sector_allocation': sector_allocation,
            'sector_weights': self.sector_rotator.sector_weights,
            'total_unrealized_pl': total_unrealized_pl,
            'market_regime': regime.value,
            'regime_multiplier': self.regime_detector.get_regime_multiplier(regime),
            'hedge_allocation': self.drawdown_protector.calculate_hedge_allocation(regime),
            'ml_trained': self.ml_ensemble.is_trained
        }
        
    async def run_trading_cycle(self, universe: List[str]) -> Dict:
        """Run a complete trading cycle with drawdown protection and sector rotation"""
        self.logger.info("="*60)
        self.logger.info("Starting V30 trading cycle...")
        
        # Check if market is open
        if self.api:
            try:
                clock = self.api.get_clock()
                if not clock.is_open:
                    self.logger.info("Market is closed")
                    return {'status': 'market_closed'}
            except Exception as e:
                self.logger.error(f"Error checking market status: {e}")
        
        # Get current state
        portfolio_value = self.get_portfolio_value()
        current_positions = self.get_positions()
        
        # Update drawdown protection
        dd_pct, should_reduce, action = self.drawdown_protector.update_equity(portfolio_value)
        position_scale = self.drawdown_protector.get_position_scaling_factor()
        
        # Check if trading halted
        if self.drawdown_protector.is_halted:
            self.logger.critical(f"Trading HALTED - Drawdown {dd_pct:.1%}")
            # Execute hedge orders only
            regime = self.regime_detector.detect(self.get_historical_data('SPY', 100))
            hedge_orders = self.drawdown_protector.get_hedge_orders(
                portfolio_value, 0, regime
            )
            hedges_executed = []
            for order in hedge_orders:
                if order['side'] == 'buy':
                    qty = int(order['value'] / 30)  # Approximate price
                    if qty > 0:
                        execs = await self.executor.execute_adaptive(
                            order['symbol'], 'buy', qty, urgency=0.9
                        )
                        hedges_executed.extend(execs)
            return {
                'status': 'halted',
                'drawdown': dd_pct,
                'hedges_executed': hedges_executed
            }
        
        # Get market regime
        spy_data = self.get_historical_data('SPY', 100)
        regime = self.regime_detector.detect(spy_data)
        regime_mult = self.regime_detector.get_regime_multiplier(regime)
        self.logger.info(f"Market regime: {regime.value}, multiplier: {regime_mult:.2f}")
        self.logger.info(f"Drawdown: {dd_pct:.2%}, Position scale: {position_scale:.2f}")
        
        # Calculate sector momentum and weights
        sector_data = {}
        all_sector_symbols = [s for symbols in self.sector_rotator.SECTORS.values() for s in symbols]
        for symbol in all_sector_symbols:
            df = self.get_historical_data(symbol, 100)
            if len(df) > 0:
                sector_data[symbol] = df
        
        self.sector_rotator.calculate_sector_momentum(sector_data)
        sector_weights = self.sector_rotator.get_sector_weights(regime)
        self.logger.info(f"Top sectors: {sorted(sector_weights.items(), key=lambda x: x[1], reverse=True)[:3]}")
        
        # Check sector constraints
        current_values = {s: p['market_value'] for s, p in current_positions.items()}
        overweights = self.sector_rotator.check_sector_constraints(current_values, portfolio_value)
        if overweights:
            self.logger.warning(f"Sector overweights: {overweights}")
                
        # Generate signals
        signals = self.generate_signals(universe)
        buy_signals = [s for s in signals if s['signal'] == 'BUY']
        sell_signals = [s for s in signals if s['signal'] == 'SELL']
        
        # Execute sells first
        sells_executed = []
        for signal in sell_signals:
            symbol = signal['symbol']
            if symbol in current_positions:
                qty = current_positions[symbol]['qty']
                # Use adaptive execution based on urgency
                urgency = 0.7 if should_reduce else 0.5
                executions = await self.executor.execute_adaptive(symbol, 'sell', qty, urgency)
                sells_executed.append({'symbol': symbol, 'qty': qty, 'executions': executions})
                self.logger.info(f"SELL executed: {symbol} x {qty}")
                
        # Execute buys (with position scaling)
        buys_executed = []
        available_slots = self.config.max_positions - len(current_positions)
        
        for signal in buy_signals[:available_slots]:
            symbol = signal['symbol']
            if symbol not in current_positions:
                # Apply position scaling from drawdown protector
                base_size = signal['suggested_position_size']
                scaled_size = base_size * position_scale * regime_mult
                
                df = self.get_historical_data(symbol, 5)
                if len(df) > 0:
                    price = df['close'].iloc[-1]
                    qty = int(scaled_size / price)
                    if qty > 0:
                        # Use VWAP for larger orders, TWAP for smaller
                        if scaled_size > 5000:
                            executions = await self.executor.execute_vwap(symbol, 'buy', qty)
                        else:
                            executions = await self.executor.execute_twap(symbol, 'buy', qty)
                        buys_executed.append({
                            'symbol': symbol, 
                            'qty': qty, 
                            'value': scaled_size,
                            'executions': executions
                        })
                        self.logger.info(f"BUY executed: {symbol} x {qty} @ ${price:.2f}")
        
        # Execute hedge orders if needed
        hedges_executed = []
        current_hedge_value = sum(
            p['market_value'] for s, p in current_positions.items() 
            if s in ['SH', 'SQQQ', 'VIXY', 'VXX', 'UVXY']
        )
        hedge_orders = self.drawdown_protector.get_hedge_orders(
            portfolio_value, current_hedge_value, regime
        )
        
        for order in hedge_orders:
            if order['value'] > 500:  # Min hedge trade
                qty = int(order['value'] / 30)  # Approximate
                if qty > 0:
                    execs = await self.executor.execute_adaptive(
                        order['symbol'], order['side'], qty, urgency=0.6
                    )
                    hedges_executed.append({
                        'symbol': order['symbol'],
                        'side': order['side'],
                        'value': order['value'],
                        'executions': execs
                    })
                    self.logger.info(f"HEDGE: {order['side']} {order['symbol']} (${order['value']:.0f})")
                        
        return {
            'status': 'completed',
            'regime': regime.value,
            'drawdown': dd_pct,
            'position_scale': position_scale,
            'signals_generated': len(signals),
            'buy_signals': len(buy_signals),
            'sell_signals': len(sell_signals),
            'sells_executed': sells_executed,
            'buys_executed': buys_executed,
            'hedges_executed': hedges_executed,
            'sector_weights': sector_weights
        }

# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='V30 Ultimate Equity Engine')
    parser.add_argument('--mode', choices=['backtest', 'live', 'paper'], default='paper',
                        help='Trading mode')
    parser.add_argument('--status', action='store_true', help='Show engine status')
    parser.add_argument('--analyze', type=str, help='Analyze a specific symbol')
    parser.add_argument('--signals', action='store_true', help='Generate signals for universe')
    args = parser.parse_args()
    
    # Initialize engine
    config = V30Config()
    if args.mode == 'live':
        config.alpaca_base_url = 'https://api.alpaca.markets'
    
    engine = V30UltimateEquityEngine(config)
    
    # Default universe
    universe = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'V', 'UNH',
                'JNJ', 'WMT', 'PG', 'XOM', 'HD', 'CVX', 'MRK', 'ABBV', 'KO', 'PEP',
                'COST', 'TMO', 'AVGO', 'MCD', 'CSCO', 'ACN', 'ABT', 'DHR', 'NEE', 'LIN']
    
    if args.status:
        status = engine.get_status()
        print("\n" + "="*60)
        print("V30 ULTIMATE EQUITY ENGINE STATUS")
        print("="*60)
        print(f"Portfolio Value: ${status['portfolio_value']:,.2f}")
        print(f"Peak Value: ${status['peak_value']:,.2f}")
        print(f"Current Drawdown: {status['current_drawdown']*100:.2f}%")
        print(f"Number of Positions: {status['num_positions']}")
        print(f"Total Unrealized P&L: ${status['total_unrealized_pl']:,.2f}")
        print(f"Market Regime: {status['market_regime']}")
        print(f"Regime Multiplier: {status['regime_multiplier']:.2f}")
        print(f"ML Model Trained: {status['ml_trained']}")
        print("="*60 + "\n")
        
    elif args.analyze:
        analysis = engine.analyze_symbol(args.analyze)
        print(f"\nAnalysis for {args.analyze}:")
        print(json.dumps(analysis, indent=2, default=str))
        
    elif args.signals:
        signals = engine.generate_signals(universe)
        print(f"\nGenerated {len(signals)} signals:")
        for s in signals[:10]:
            print(f"  {s['symbol']}: {s['signal']} (alpha={s['alpha_score']:.3f}, ml_prob={s['ml_probability']:.3f})")
            
    else:
        # Run trading cycle
        result = asyncio.run(engine.run_trading_cycle(universe))
        print(f"\nTrading cycle result:")
        print(json.dumps(result, indent=2, default=str))

if __name__ == '__main__':
    main()
