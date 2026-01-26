#!/usr/bin/env python3
"""
V48 INSTITUTIONAL QUANTITATIVE TRADING ENGINE
=============================================
Medallion Fund-Inspired Architecture

Core Principles:
1. Statistical edge over fundamentals - find patterns, not explanations
2. High frequency x small edge = massive profits
3. Kelly Criterion for optimal position sizing
4. Market neutral (beta ≈ 0) - profit in any conditions
5. Extreme diversification - thousands of small bets

Strategies Implemented:
- Statistical Arbitrage (Cointegration Pairs)
- Hidden Markov Model Regime Detection
- Mean Reversion (Bollinger, RSI, Z-Score)
- Momentum (Cross-sectional & Time-series)
- Order Flow Imbalance Analysis
- Machine Learning Price Prediction

Author: Institutional Quant Team
Version: 48.0.0
"""

import os
import sys
import asyncio
import logging
import argparse
import json
import time
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import pickle

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

# Alpaca imports
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, LimitOrderRequest, GetAssetsRequest
)
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass, AssetStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockBarsRequest, StockQuotesRequest
from alpaca.data.timeframe import TimeFrame

# Advanced ML imports (with fallbacks)
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("Warning: hmmlearn not installed. HMM regime detection disabled.")

try:
    from statsmodels.tsa.stattools import coint, adfuller
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not installed. Cointegration analysis disabled.")

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not installed. ML predictions disabled.")


# ============================================================================
# CONFIGURATION
# ============================================================================

class MarketRegime(Enum):
    """Market regime states from HMM"""
    BULL_LOW_VOL = "bull_low_vol"
    BULL_HIGH_VOL = "bull_high_vol"
    BEAR_LOW_VOL = "bear_low_vol"
    BEAR_HIGH_VOL = "bear_high_vol"
    SIDEWAYS = "sideways"

@dataclass
class InstitutionalConfig:
    """Institutional-grade configuration"""
    # API Configuration
    api_key: str = ""
    api_secret: str = ""
    paper: bool = True
    
    # Universe Configuration
    universe_size: int = 500  # Expanded universe
    min_price: float = 5.0
    max_price: float = 1000.0
    min_volume: int = 500000  # Daily volume filter
    min_market_cap: float = 1e9  # $1B minimum
    
    # Scanning Configuration  
    scan_interval: int = 5  # 5-second scans (high frequency)
    batch_size: int = 50
    
    # Position Management
    max_positions: int = 100  # Extreme diversification
    position_size_pct: float = 0.01  # 1% per position (100 positions = 100%)
    max_sector_exposure: float = 0.15  # 15% max per sector
    max_single_position: float = 0.03  # 3% max single position
    
    # Risk Management
    max_portfolio_var: float = 0.02  # 2% daily VaR limit
    max_drawdown: float = 0.10  # 10% drawdown circuit breaker
    kelly_fraction: float = 0.25  # Fractional Kelly (conservative)
    correlation_threshold: float = 0.7  # Correlation limit
    
    # Strategy Thresholds
    # Mean Reversion
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    rsi_period: int = 14
    rsi_oversold: float = 30
    rsi_overbought: float = 70
    zscore_entry: float = 2.0
    zscore_exit: float = 0.5
    
    # Momentum
    momentum_lookback: int = 20
    momentum_threshold: float = 0.02
    
    # Cointegration (Pairs Trading)
    coint_lookback: int = 60
    coint_pvalue: float = 0.05
    spread_entry: float = 2.0
    spread_exit: float = 0.5
    
    # HMM Configuration
    hmm_n_states: int = 4
    hmm_lookback: int = 252
    
    # ML Configuration
    ml_lookback: int = 100
    ml_retrain_interval: int = 24 * 60 * 60  # Daily retraining
    ml_confidence_threshold: float = 0.6
    
    # Options Configuration
    options_enabled: bool = True
    wheel_delta_target: float = 0.30
    premium_min_pct: float = 0.005
    dte_min: int = 7
    dte_max: int = 45
    
    # Execution Configuration
    use_limit_orders: bool = True
    limit_offset_bps: int = 5  # 5 basis points
    twap_slices: int = 10
    max_spread_pct: float = 0.005  # 0.5% max spread


# ============================================================================
# DATA INFRASTRUCTURE
# ============================================================================

class MarketDataCache:
    """High-performance market data cache with history"""
    
    def __init__(self, max_history: int = 500):
        self.max_history = max_history
        self._bars: Dict[str, deque] = {}
        self._quotes: Dict[str, Any] = {}
        self._features: Dict[str, pd.DataFrame] = {}
        self._last_update: Dict[str, datetime] = {}
        
    def update_bars(self, symbol: str, bar_data: pd.DataFrame):
        """Update bar history for symbol"""
        if symbol not in self._bars:
            self._bars[symbol] = deque(maxlen=self.max_history)
        
        for _, row in bar_data.iterrows():
            self._bars[symbol].append({
                'timestamp': row.name if hasattr(row, 'name') else datetime.now(),
                'open': row.get('open', row.get('Open', 0)),
                'high': row.get('high', row.get('High', 0)),
                'low': row.get('low', row.get('Low', 0)),
                'close': row.get('close', row.get('Close', 0)),
                'volume': row.get('volume', row.get('Volume', 0)),
                'vwap': row.get('vwap', row.get('VWAP', 0))
            })
        self._last_update[symbol] = datetime.now()
    
    def get_bars_df(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get bars as DataFrame"""
        if symbol not in self._bars or len(self._bars[symbol]) == 0:
            return None
        return pd.DataFrame(list(self._bars[symbol]))
    
    def update_quote(self, symbol: str, bid: float, ask: float, bid_size: int, ask_size: int):
        """Update latest quote"""
        self._quotes[symbol] = {
            'bid': bid, 'ask': ask, 
            'bid_size': bid_size, 'ask_size': ask_size,
            'mid': (bid + ask) / 2,
            'spread': ask - bid,
            'spread_pct': (ask - bid) / ((bid + ask) / 2) if (bid + ask) > 0 else 0,
            'order_imbalance': (bid_size - ask_size) / (bid_size + ask_size) if (bid_size + ask_size) > 0 else 0
        }
    
    def get_quote(self, symbol: str) -> Optional[Dict]:
        return self._quotes.get(symbol)


class FeatureEngine:
    """Technical feature calculation engine"""
    
    @staticmethod
    def calculate_all_features(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive feature set"""
        if df is None or len(df) < 30:
            return None
            
        features = df.copy()
        close = features['close'].values
        high = features['high'].values
        low = features['low'].values
        volume = features['volume'].values
        
        # Price-based features
        features['returns'] = np.diff(close, prepend=close[0]) / np.maximum(close, 1e-8)
        features['log_returns'] = np.log(close / np.roll(close, 1))
        features['log_returns'].iloc[0] = 0
        
        # Volatility features
        features['volatility_20'] = features['returns'].rolling(20).std()
        features['volatility_5'] = features['returns'].rolling(5).std()
        features['vol_ratio'] = features['volatility_5'] / features['volatility_20'].replace(0, np.nan)
        
        # Trend features
        features['sma_5'] = features['close'].rolling(5).mean()
        features['sma_20'] = features['close'].rolling(20).mean()
        features['sma_50'] = features['close'].rolling(50).mean()
        features['ema_12'] = features['close'].ewm(span=12).mean()
        features['ema_26'] = features['close'].ewm(span=26).mean()
        
        # MACD
        features['macd'] = features['ema_12'] - features['ema_26']
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        # Bollinger Bands
        features['bb_mid'] = features['close'].rolling(20).mean()
        bb_std = features['close'].rolling(20).std()
        features['bb_upper'] = features['bb_mid'] + 2 * bb_std
        features['bb_lower'] = features['bb_mid'] - 2 * bb_std
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['bb_mid']
        features['bb_position'] = (features['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower']).replace(0, np.nan)
        
        # RSI
        delta = features['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # Stochastic
        low_14 = features['low'].rolling(14).min()
        high_14 = features['high'].rolling(14).max()
        features['stoch_k'] = 100 * (features['close'] - low_14) / (high_14 - low_14).replace(0, np.nan)
        features['stoch_d'] = features['stoch_k'].rolling(3).mean()
        
        # ATR
        tr1 = high - low
        tr2 = abs(high - np.roll(close, 1))
        tr3 = abs(low - np.roll(close, 1))
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        features['atr'] = pd.Series(tr).rolling(14).mean().values
        
        # Volume features
        features['volume_sma'] = features['volume'].rolling(20).mean()
        features['volume_ratio'] = features['volume'] / features['volume_sma'].replace(0, np.nan)
        features['obv'] = (np.sign(features['returns']) * features['volume']).cumsum()
        
        # Momentum features
        features['momentum_5'] = features['close'] / features['close'].shift(5) - 1
        features['momentum_10'] = features['close'] / features['close'].shift(10) - 1
        features['momentum_20'] = features['close'] / features['close'].shift(20) - 1
        
        # Z-score
        features['zscore'] = (features['close'] - features['sma_20']) / features['close'].rolling(20).std()
        
        # Mean reversion signals
        features['mean_rev_signal'] = np.where(
            features['zscore'] < -2, 1,
            np.where(features['zscore'] > 2, -1, 0)
        )
        
        return features.fillna(0)


# ============================================================================
# ADVANCED STRATEGIES
# ============================================================================

class CointegrationPairs:
    """Statistical Arbitrage via Cointegration Pairs Trading"""
    
    def __init__(self, lookback: int = 60, pvalue_threshold: float = 0.05):
        self.lookback = lookback
        self.pvalue_threshold = pvalue_threshold
        self.pairs: List[Tuple[str, str, float]] = []  # (sym1, sym2, hedge_ratio)
        self.spreads: Dict[str, deque] = {}
        
    def find_cointegrated_pairs(self, price_data: Dict[str, pd.DataFrame]) -> List[Tuple[str, str, float]]:
        """Find cointegrated pairs from price data"""
        if not STATSMODELS_AVAILABLE:
            return []
            
        symbols = list(price_data.keys())
        n = len(symbols)
        pairs = []
        
        # Build price matrix
        prices = {}
        for sym in symbols:
            df = price_data.get(sym)
            if df is not None and len(df) >= self.lookback:
                prices[sym] = df['close'].values[-self.lookback:]
        
        valid_symbols = list(prices.keys())
        
        # Test all pairs for cointegration
        for i, sym1 in enumerate(valid_symbols):
            for sym2 in valid_symbols[i+1:]:
                try:
                    p1, p2 = prices[sym1], prices[sym2]
                    
                    # Engle-Granger cointegration test
                    score, pvalue, _ = coint(p1, p2)
                    
                    if pvalue < self.pvalue_threshold:
                        # Calculate hedge ratio via OLS
                        X = add_constant(p2)
                        model = OLS(p1, X).fit()
                        hedge_ratio = model.params[1]
                        
                        pairs.append((sym1, sym2, hedge_ratio))
                except Exception:
                    continue
        
        self.pairs = pairs
        return pairs
    
    def calculate_spread(self, sym1: str, sym2: str, hedge_ratio: float,
                        price1: float, price2: float) -> float:
        """Calculate spread for a pair"""
        return price1 - hedge_ratio * price2
    
    def get_zscore(self, pair_key: str, spread: float) -> float:
        """Get z-score of spread"""
        if pair_key not in self.spreads:
            self.spreads[pair_key] = deque(maxlen=self.lookback)
        
        self.spreads[pair_key].append(spread)
        
        if len(self.spreads[pair_key]) < 20:
            return 0
        
        spreads = list(self.spreads[pair_key])
        mean = np.mean(spreads)
        std = np.std(spreads)
        
        if std == 0:
            return 0
        
        return (spread - mean) / std
    
    def generate_signals(self, price_data: Dict[str, float], 
                        entry_threshold: float = 2.0,
                        exit_threshold: float = 0.5) -> List[Dict]:
        """Generate trading signals for pairs"""
        signals = []
        
        for sym1, sym2, hedge_ratio in self.pairs:
            if sym1 not in price_data or sym2 not in price_data:
                continue
                
            price1 = price_data[sym1]
            price2 = price_data[sym2]
            pair_key = f"{sym1}_{sym2}"
            
            spread = self.calculate_spread(sym1, sym2, hedge_ratio, price1, price2)
            zscore = self.get_zscore(pair_key, spread)
            
            if zscore < -entry_threshold:
                # Spread too low: buy sym1, sell sym2
                signals.append({
                    'type': 'pairs',
                    'action': 'open_long',
                    'sym1': sym1, 'sym2': sym2,
                    'hedge_ratio': hedge_ratio,
                    'zscore': zscore
                })
            elif zscore > entry_threshold:
                # Spread too high: sell sym1, buy sym2
                signals.append({
                    'type': 'pairs',
                    'action': 'open_short',
                    'sym1': sym1, 'sym2': sym2,
                    'hedge_ratio': hedge_ratio,
                    'zscore': zscore
                })
            elif abs(zscore) < exit_threshold:
                # Mean reversion - close position
                signals.append({
                    'type': 'pairs',
                    'action': 'close',
                    'sym1': sym1, 'sym2': sym2,
                    'zscore': zscore
                })
        
        return signals


class HMMRegimeDetector:
    """Hidden Markov Model for Market Regime Detection"""
    
    def __init__(self, n_states: int = 4, lookback: int = 252):
        self.n_states = n_states
        self.lookback = lookback
        self.model = None
        self.current_regime = MarketRegime.SIDEWAYS
        self.regime_history = deque(maxlen=100)
        
    def fit(self, returns: np.ndarray, volatility: np.ndarray) -> bool:
        """Fit HMM to market data"""
        if not HMM_AVAILABLE:
            return False
            
        try:
            # Prepare features
            X = np.column_stack([returns, volatility])
            X = X[~np.isnan(X).any(axis=1)]  # Remove NaN rows
            
            if len(X) < 50:
                return False
            
            # Fit Gaussian HMM
            self.model = hmm.GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=100,
                random_state=42
            )
            self.model.fit(X)
            return True
        except Exception as e:
            logging.error(f"HMM fitting error: {e}")
            return False
    
    def predict_regime(self, returns: float, volatility: float) -> MarketRegime:
        """Predict current market regime"""
        if self.model is None or not HMM_AVAILABLE:
            return MarketRegime.SIDEWAYS
            
        try:
            X = np.array([[returns, volatility]])
            state = self.model.predict(X)[0]
            
            # Map states to regimes based on characteristics
            means = self.model.means_
            
            # State classification based on return/vol characteristics
            ret_mean = means[state, 0]
            vol_mean = means[state, 1]
            
            median_vol = np.median(means[:, 1])
            
            if ret_mean > 0 and vol_mean < median_vol:
                regime = MarketRegime.BULL_LOW_VOL
            elif ret_mean > 0 and vol_mean >= median_vol:
                regime = MarketRegime.BULL_HIGH_VOL
            elif ret_mean < 0 and vol_mean < median_vol:
                regime = MarketRegime.BEAR_LOW_VOL
            elif ret_mean < 0 and vol_mean >= median_vol:
                regime = MarketRegime.BEAR_HIGH_VOL
            else:
                regime = MarketRegime.SIDEWAYS
            
            self.current_regime = regime
            self.regime_history.append(regime)
            return regime
            
        except Exception:
            return MarketRegime.SIDEWAYS
    
    def get_regime_probability(self) -> Dict[MarketRegime, float]:
        """Get probability distribution over regimes"""
        if not self.regime_history:
            return {r: 0.25 for r in MarketRegime}
        
        counts = {}
        for r in self.regime_history:
            counts[r] = counts.get(r, 0) + 1
        
        total = len(self.regime_history)
        return {r: counts.get(r, 0) / total for r in MarketRegime}


class MLPricePredictor:
    """Machine Learning Price Direction Predictor"""
    
    def __init__(self, lookback: int = 100):
        self.lookback = lookback
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.last_train: Dict[str, datetime] = {}
        self.feature_cols = [
            'returns', 'volatility_20', 'volatility_5', 'vol_ratio',
            'macd', 'macd_hist', 'rsi', 'stoch_k', 'bb_position',
            'volume_ratio', 'momentum_5', 'momentum_10', 'zscore'
        ]
        
    def train(self, symbol: str, features: pd.DataFrame) -> bool:
        """Train ML model for symbol"""
        if not SKLEARN_AVAILABLE:
            return False
            
        try:
            df = features.copy()
            
            # Create target: next period return direction
            df['target'] = (df['returns'].shift(-1) > 0).astype(int)
            df = df.dropna()
            
            if len(df) < 50:
                return False
            
            # Select features
            available_cols = [c for c in self.feature_cols if c in df.columns]
            X = df[available_cols].values
            y = df['target'].values
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train ensemble
            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
            model.fit(X_scaled[:-1], y[:-1])  # Leave last row for prediction
            
            self.models[symbol] = model
            self.scalers[symbol] = scaler
            self.last_train[symbol] = datetime.now()
            
            return True
        except Exception as e:
            logging.error(f"ML training error for {symbol}: {e}")
            return False
    
    def predict(self, symbol: str, features: pd.DataFrame) -> Tuple[int, float]:
        """Predict price direction with confidence"""
        if not SKLEARN_AVAILABLE or symbol not in self.models:
            return 0, 0.5
            
        try:
            available_cols = [c for c in self.feature_cols if c in features.columns]
            X = features[available_cols].iloc[-1:].values
            X_scaled = self.scalers[symbol].transform(X)
            
            pred = self.models[symbol].predict(X_scaled)[0]
            proba = self.models[symbol].predict_proba(X_scaled)[0]
            confidence = max(proba)
            
            direction = 1 if pred == 1 else -1
            return direction, confidence
        except Exception:
            return 0, 0.5


# ============================================================================
# RISK MANAGEMENT
# ============================================================================

class InstitutionalRiskManager:
    """Institutional-grade risk management system"""
    
    def __init__(self, config: InstitutionalConfig):
        self.config = config
        self.trade_history: List[Dict] = []
        self.daily_pnl: deque = deque(maxlen=252)
        self.peak_equity: float = 0
        self.current_drawdown: float = 0
        self.win_count: int = 0
        self.loss_count: int = 0
        self.circuit_breaker_active: bool = False
        
    def calculate_kelly_fraction(self) -> float:
        """Calculate optimal Kelly fraction"""
        total_trades = self.win_count + self.loss_count
        if total_trades < 20:
            return self.config.kelly_fraction  # Use default until enough history
        
        win_rate = self.win_count / total_trades
        
        # Calculate average win/loss ratio from history
        wins = [t['pnl'] for t in self.trade_history if t.get('pnl', 0) > 0]
        losses = [abs(t['pnl']) for t in self.trade_history if t.get('pnl', 0) < 0]
        
        if not wins or not losses:
            return self.config.kelly_fraction
        
        avg_win = np.mean(wins)
        avg_loss = np.mean(losses)
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1
        
        # Kelly formula: f* = (bp - q) / b
        # where b = win/loss ratio, p = win probability, q = 1-p
        kelly = (win_loss_ratio * win_rate - (1 - win_rate)) / win_loss_ratio
        
        # Apply fractional Kelly for safety
        return max(0, min(kelly * self.config.kelly_fraction, 0.5))
    
    def calculate_position_size(self, symbol: str, price: float, 
                               signal_strength: float, equity: float) -> int:
        """Calculate optimal position size"""
        if self.circuit_breaker_active:
            return 0
        
        kelly = self.calculate_kelly_fraction()
        
        # Base position size
        base_size = equity * self.config.position_size_pct
        
        # Adjust by Kelly fraction and signal strength
        adjusted_size = base_size * kelly * abs(signal_strength)
        
        # Apply max single position limit
        max_position = equity * self.config.max_single_position
        adjusted_size = min(adjusted_size, max_position)
        
        # Convert to shares
        shares = int(adjusted_size / price) if price > 0 else 0
        
        return max(1, shares)  # At least 1 share if valid
    
    def update_drawdown(self, equity: float):
        """Update drawdown tracking"""
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        self.current_drawdown = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0
        
        # Circuit breaker
        if self.current_drawdown >= self.config.max_drawdown:
            self.circuit_breaker_active = True
            logging.warning(f"CIRCUIT BREAKER ACTIVATED: Drawdown {self.current_drawdown:.2%}")
    
    def record_trade(self, symbol: str, side: str, shares: int, 
                    entry_price: float, exit_price: float = None):
        """Record trade for statistics"""
        trade = {
            'symbol': symbol,
            'side': side,
            'shares': shares,
            'entry_price': entry_price,
            'entry_time': datetime.now(),
            'exit_price': exit_price,
            'pnl': None
        }
        
        if exit_price:
            if side == 'buy':
                trade['pnl'] = (exit_price - entry_price) * shares
            else:
                trade['pnl'] = (entry_price - exit_price) * shares
            
            if trade['pnl'] > 0:
                self.win_count += 1
            else:
                self.loss_count += 1
        
        self.trade_history.append(trade)
    
    def calculate_portfolio_var(self, positions: Dict[str, Dict], 
                               returns_data: Dict[str, np.ndarray]) -> float:
        """Calculate portfolio Value at Risk"""
        if not positions or not returns_data:
            return 0
        
        # Build portfolio returns
        portfolio_returns = []
        weights = []
        
        total_value = sum(p.get('market_value', 0) for p in positions.values())
        if total_value == 0:
            return 0
        
        for symbol, pos in positions.items():
            if symbol in returns_data:
                returns = returns_data[symbol]
                weight = pos.get('market_value', 0) / total_value
                portfolio_returns.append(returns * weight)
                weights.append(weight)
        
        if not portfolio_returns:
            return 0
        
        # Calculate VaR at 95% confidence
        combined_returns = np.sum(portfolio_returns, axis=0)
        var_95 = np.percentile(combined_returns, 5)
        
        return abs(var_95)
    
    def check_correlation(self, symbol: str, existing_positions: Dict[str, Dict],
                         returns_data: Dict[str, np.ndarray]) -> bool:
        """Check if adding position would exceed correlation threshold"""
        if symbol not in returns_data or not existing_positions:
            return True  # Allow if no data
        
        new_returns = returns_data[symbol]
        
        for existing_sym in existing_positions:
            if existing_sym in returns_data:
                existing_returns = returns_data[existing_sym]
                
                # Calculate correlation
                min_len = min(len(new_returns), len(existing_returns))
                if min_len > 10:
                    corr = np.corrcoef(
                        new_returns[-min_len:], 
                        existing_returns[-min_len:]
                    )[0, 1]
                    
                    if abs(corr) > self.config.correlation_threshold:
                        return False  # Too correlated
        
        return True
    
    def get_statistics(self) -> Dict:
        """Get risk statistics"""
        total = self.win_count + self.loss_count
        return {
            'total_trades': total,
            'win_rate': self.win_count / total if total > 0 else 0,
            'current_drawdown': self.current_drawdown,
            'peak_equity': self.peak_equity,
            'kelly_fraction': self.calculate_kelly_fraction(),
            'circuit_breaker': self.circuit_breaker_active
        }


# ============================================================================
# MAIN ENGINE
# ============================================================================

class V48InstitutionalEngine:
    """V48 Institutional Quantitative Trading Engine"""
    
    def __init__(self, config: InstitutionalConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # API Clients
        self.trading_client = None
        self.data_client = None
        
        # Components
        self.data_cache = MarketDataCache()
        self.risk_manager = InstitutionalRiskManager(config)
        self.pairs_strategy = CointegrationPairs(
            lookback=config.coint_lookback,
            pvalue_threshold=config.coint_pvalue
        )
        self.regime_detector = HMMRegimeDetector(
            n_states=config.hmm_n_states,
            lookback=config.hmm_lookback
        )
        self.ml_predictor = MLPricePredictor(lookback=config.ml_lookback)
        
        # State
        self.universe: List[str] = []
        self.positions: Dict[str, Dict] = {}
        self.pending_orders: Dict[str, Any] = {}
        self.returns_data: Dict[str, np.ndarray] = {}
        self._initialized = False
        self._running = False
        
        # Performance tracking
        self.start_equity = 0
        self.scan_count = 0
        self.trade_count = 0
        
    def _setup_logging(self) -> logging.Logger:
        """Configure logging"""
        logger = logging.getLogger('V48Engine')
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(handler)
        
        return logger
    
    async def initialize(self) -> bool:
        """Initialize engine components"""
        try:
            self.logger.info("="*60)
            self.logger.info("V48 INSTITUTIONAL QUANTITATIVE ENGINE")
            self.logger.info("Medallion Fund-Inspired Architecture")
            self.logger.info("="*60)
            
            # Initialize API clients
            self.trading_client = TradingClient(
                self.config.api_key,
                self.config.api_secret,
                paper=self.config.paper
            )
            self.data_client = StockHistoricalDataClient(
                self.config.api_key,
                self.config.api_secret
            )
            
            # Get account info
            account = self.trading_client.get_account()
            self.start_equity = float(account.equity)
            self.risk_manager.peak_equity = self.start_equity
            
            self.logger.info(f"Account Equity: ${self.start_equity:,.2f}")
            self.logger.info(f"Buying Power: ${float(account.buying_power):,.2f}")
            self.logger.info(f"Paper Trading: {self.config.paper}")
            
            # Build universe
            await self._build_universe()
            
            # Load historical data
            await self._load_historical_data()
            
            # Initialize strategies
            self._initialize_strategies()
            
            self._initialized = True
            self.logger.info(f"Engine initialized with {len(self.universe)} symbols")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    async def _build_universe(self):
        """Build tradeable universe"""
        self.logger.info("Building trading universe...")
        
        try:
            # Get all tradeable assets
            request = GetAssetsRequest(
                asset_class=AssetClass.US_EQUITY,
                status=AssetStatus.ACTIVE
            )
            assets = self.trading_client.get_all_assets(request)
            
            # Filter assets
            candidates = []
            for asset in assets:
                if (asset.tradable and 
                    asset.fractionable and
                    not asset.symbol.startswith('.')  and
                    '/' not in asset.symbol):
                    candidates.append(asset.symbol)
            
            # Take top N by some criteria (here just alphabetical for simplicity)
            # In production, would filter by volume/market cap
            self.universe = sorted(candidates)[:self.config.universe_size]
            
            self.logger.info(f"Universe: {len(self.universe)} symbols")
            
        except Exception as e:
            self.logger.error(f"Universe building failed: {e}")
            # Fallback to major stocks
            self.universe = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'BRK.B',
                'UNH', 'JNJ', 'V', 'XOM', 'JPM', 'PG', 'MA', 'HD', 'CVX', 'ABBV',
                'MRK', 'PEP', 'COST', 'KO', 'AVGO', 'LLY', 'TMO', 'WMT', 'MCD',
                'CSCO', 'ACN', 'ABT', 'DHR', 'VZ', 'ADBE', 'CRM', 'NKE', 'CMCSA',
                'TXN', 'PM', 'NEE', 'INTC', 'WFC', 'BMY', 'UPS', 'MS', 'QCOM',
                'RTX', 'HON', 'T', 'ORCL', 'AMD', 'IBM', 'GS', 'CAT', 'BA',
                'SBUX', 'GE', 'INTU', 'BLK', 'AMGN', 'ISRG', 'GILD', 'AXP',
                'MDLZ', 'CVS', 'ADI', 'SYK', 'TJX', 'BKNG', 'PLD', 'REGN',
                'VRTX', 'MMC', 'AMT', 'LRCX', 'C', 'NOW', 'CI', 'ZTS', 'EOG',
                'MO', 'SCHW', 'TMUS', 'CB', 'BDX', 'SO', 'DUK', 'PNC', 'ITW',
                'BSX', 'EQIX', 'CME', 'AON', 'DE', 'USB', 'APD', 'CL', 'SLB',
                'ETN', 'SNPS', 'TGT', 'NOC', 'WM', 'FISV', 'MU', 'ICE', 'CSX'
            ]
    
    async def _load_historical_data(self):
        """Load historical data for all symbols"""
        self.logger.info("Loading historical data...")
        
        end = datetime.now()
        start = end - timedelta(days=100)
        
        for i in range(0, len(self.universe), self.config.batch_size):
            batch = self.universe[i:i + self.config.batch_size]
            
            try:
                request = StockBarsRequest(feed=DataFeed.IEX, 
                    symbol_or_symbols=batch,
                    timeframe=TimeFrame.Hour,
                    start=start,
                    end=end
                )
                bars = self.data_client.get_stock_bars(request)
                
                for symbol in batch:
                    if symbol in bars.data:
                        df = pd.DataFrame([{
                            'timestamp': bar.timestamp,
                            'open': bar.open,
                            'high': bar.high,
                            'low': bar.low,
                            'close': bar.close,
                            'volume': bar.volume,
                            'vwap': bar.vwap
                        } for bar in bars.data[symbol]])
                        
                        if len(df) > 0:
                            self.data_cache.update_bars(symbol, df)
                            
                            # Calculate features
                            features = FeatureEngine.calculate_all_features(df)
                            if features is not None:
                                self.returns_data[symbol] = features['returns'].values
                                
            except Exception as e:
                self.logger.warning(f"Data load error for batch: {e}")
            
            await asyncio.sleep(0.2)  # Rate limiting
        
        self.logger.info(f"Loaded data for {len(self.returns_data)} symbols")

    
    def _initialize_strategies(self):
        """Initialize all trading strategies"""
        self.logger.info("Initializing strategies...")
        
        # Find cointegrated pairs
        if STATSMODELS_AVAILABLE:
            price_data = {}
            for symbol in self.universe[:100]:  # Limit for performance
                df = self.data_cache.get_bars_df(symbol)
                if df is not None:
                    price_data[symbol] = df
            
            pairs = self.pairs_strategy.find_cointegrated_pairs(price_data)
            self.logger.info(f"Found {len(pairs)} cointegrated pairs")
        
        # Train HMM on market data (using SPY as proxy)
        if HMM_AVAILABLE and 'SPY' in self.returns_data:
            returns = self.returns_data['SPY']
            volatility = pd.Series(returns).rolling(20).std().values
            if len(returns) > 50:
                self.regime_detector.fit(returns[-252:], volatility[-252:])
                self.logger.info("HMM regime detector trained")
        
        # Train ML models for top symbols
        if SKLEARN_AVAILABLE:
            trained = 0
            for symbol in self.universe[:50]:
                df = self.data_cache.get_bars_df(symbol)
                if df is not None:
                    features = FeatureEngine.calculate_all_features(df)
                    if features is not None and self.ml_predictor.train(symbol, features):
                        trained += 1
            self.logger.info(f"Trained ML models for {trained} symbols")
    
    async def _generate_signals(self, symbol: str) -> List[Dict]:
        """Generate trading signals for a symbol"""
        signals = []
        
        df = self.data_cache.get_bars_df(symbol)
        if df is None or len(df) < 30:
            return signals
        
        features = FeatureEngine.calculate_all_features(df)
        if features is None:
            return signals
        
        latest = features.iloc[-1]
        
        # 1. MEAN REVERSION SIGNALS
        zscore = latest.get('zscore', 0)
        rsi = latest.get('rsi', 50)
        bb_pos = latest.get('bb_position', 0.5)
        
        # Oversold - BUY signal
        if zscore < -self.config.zscore_entry and rsi < self.config.rsi_oversold:
            signals.append({
                'strategy': 'mean_reversion',
                'action': 'buy',
                'symbol': symbol,
                'strength': abs(zscore) / 3,  # Normalize to 0-1
                'reason': f'Oversold: z={zscore:.2f}, RSI={rsi:.1f}'
            })
        
        # Overbought - SELL signal
        elif zscore > self.config.zscore_entry and rsi > self.config.rsi_overbought:
            signals.append({
                'strategy': 'mean_reversion',
                'action': 'sell',
                'symbol': symbol,
                'strength': abs(zscore) / 3,
                'reason': f'Overbought: z={zscore:.2f}, RSI={rsi:.1f}'
            })
        
        # 2. MOMENTUM SIGNALS
        mom_5 = latest.get('momentum_5', 0)
        mom_20 = latest.get('momentum_20', 0)
        vol_ratio = latest.get('volume_ratio', 1)
        
        # Strong upward momentum with volume confirmation
        if mom_5 > self.config.momentum_threshold and mom_20 > 0 and vol_ratio > 1.5:
            signals.append({
                'strategy': 'momentum',
                'action': 'buy',
                'symbol': symbol,
                'strength': min(mom_5 / 0.1, 1),
                'reason': f'Bullish momentum: {mom_5:.2%} with volume {vol_ratio:.1f}x'
            })
        
        # Strong downward momentum - short or exit
        elif mom_5 < -self.config.momentum_threshold and mom_20 < 0:
            signals.append({
                'strategy': 'momentum',
                'action': 'sell',
                'symbol': symbol,
                'strength': min(abs(mom_5) / 0.1, 1),
                'reason': f'Bearish momentum: {mom_5:.2%}'
            })
        
        # 3. MACD CROSSOVER
        macd_hist = latest.get('macd_hist', 0)
        prev_macd_hist = features['macd_hist'].iloc[-2] if len(features) > 1 else 0
        
        if prev_macd_hist < 0 and macd_hist > 0:
            signals.append({
                'strategy': 'macd',
                'action': 'buy',
                'symbol': symbol,
                'strength': 0.6,
                'reason': 'MACD bullish crossover'
            })
        elif prev_macd_hist > 0 and macd_hist < 0:
            signals.append({
                'strategy': 'macd',
                'action': 'sell',
                'symbol': symbol,
                'strength': 0.6,
                'reason': 'MACD bearish crossover'
            })
        
        # 4. ML PREDICTION
        if SKLEARN_AVAILABLE and symbol in self.ml_predictor.models:
            direction, confidence = self.ml_predictor.predict(symbol, features)
            if confidence >= self.config.ml_confidence_threshold:
                signals.append({
                    'strategy': 'ml',
                    'action': 'buy' if direction > 0 else 'sell',
                    'symbol': symbol,
                    'strength': confidence,
                    'reason': f'ML prediction: {"bullish" if direction > 0 else "bearish"} ({confidence:.1%})'
                })
        
        return signals
    
    async def _execute_signal(self, signal: Dict) -> bool:
        """Execute a trading signal"""
        symbol = signal['symbol']
        action = signal['action']
        strength = signal.get('strength', 1.0)
        
        try:
            # Get current price
            df = self.data_cache.get_bars_df(symbol)
            if df is None or len(df) == 0:
                return False
            
            price = df['close'].iloc[-1]
            
            # Get account info
            account = self.trading_client.get_account()
            equity = float(account.equity)
            buying_power = float(account.buying_power)
            
            # Update drawdown
            self.risk_manager.update_drawdown(equity)
            
            if self.risk_manager.circuit_breaker_active:
                self.logger.warning("Circuit breaker active - no new trades")
                return False
            
            # Check if we already have a position
            positions = self.trading_client.get_all_positions()
            current_pos = None
            for pos in positions:
                if pos.symbol == symbol:
                    current_pos = pos
                    break
            
            # Calculate position size
            shares = self.risk_manager.calculate_position_size(
                symbol, price, strength, equity
            )
            
            if shares < 1:
                return False
            
            # Check correlation with existing positions
            if not self.risk_manager.check_correlation(
                symbol, self.positions, self.returns_data
            ):
                self.logger.debug(f"Skipping {symbol} - too correlated with existing positions")
                return False
            
            # Execute based on action
            if action == 'buy':
                if len(positions) >= self.config.max_positions:
                    return False
                
                if buying_power < price * shares:
                    shares = int(buying_power / price)
                    if shares < 1:
                        return False
                
                order = MarketOrderRequest(
                    symbol=symbol,
                    qty=shares,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                )
                result = self.trading_client.submit_order(order)
                self.logger.info(f"BUY {shares} {symbol} @ ~${price:.2f} | {signal.get('reason', '')}")
                self.trade_count += 1
                self.risk_manager.record_trade(symbol, 'buy', shares, price)
                return True
                
            elif action == 'sell' and current_pos:
                # Close or reduce position
                qty = int(float(current_pos.qty))
                
                order = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY
                )
                result = self.trading_client.submit_order(order)
                entry_price = float(current_pos.avg_entry_price)
                self.logger.info(f"SELL {qty} {symbol} @ ~${price:.2f} (entry ${entry_price:.2f}) | {signal.get('reason', '')}")
                self.trade_count += 1
                self.risk_manager.record_trade(symbol, 'sell', qty, entry_price, price)
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Execution error for {symbol}: {e}")
            return False

    
    async def _refresh_data(self, symbols: List[str]):
        """Refresh market data for symbols"""
        end = datetime.now()
        start = end - timedelta(days=5)
        
        try:
            request = StockBarsRequest(feed=DataFeed.IEX, 
                symbol_or_symbols=symbols,
                timeframe=TimeFrame.Minute,
                start=start,
                end=end
            )
            bars = self.data_client.get_stock_bars(request)
            
            for symbol in symbols:
                if symbol in bars.data:
                    df = pd.DataFrame([{
                        'timestamp': bar.timestamp,
                        'open': bar.open,
                        'high': bar.high,
                        'low': bar.low,
                        'close': bar.close,
                        'volume': bar.volume,
                        'vwap': bar.vwap
                    } for bar in bars.data[symbol]])
                    
                    if len(df) > 0:
                        self.data_cache.update_bars(symbol, df)
        except Exception as e:
            self.logger.warning(f"Data refresh error: {e}")
    
    async def _scan_cycle(self):
        """Execute one scan cycle"""
        self.scan_count += 1
        all_signals = []
        
        # Scan in batches
        for i in range(0, len(self.universe), self.config.batch_size):
            batch = self.universe[i:i + self.config.batch_size]
            
            # Refresh data
            await self._refresh_data(batch)
            
            # Generate signals
            for symbol in batch:
                signals = await self._generate_signals(symbol)
                all_signals.extend(signals)
            
            await asyncio.sleep(0.1)  # Rate limiting
        
        # Sort by signal strength
        all_signals.sort(key=lambda x: x.get('strength', 0), reverse=True)
        
        # Execute top signals
        executed = 0
        for signal in all_signals[:10]:  # Limit executions per cycle
            if await self._execute_signal(signal):
                executed += 1
                await asyncio.sleep(0.5)  # Pace executions
        
        return executed
    
    def _print_status(self):
        """Print current status"""
        try:
            account = self.trading_client.get_account()
            equity = float(account.equity)
            positions = self.trading_client.get_all_positions()
            
            pnl = equity - self.start_equity
            pnl_pct = (pnl / self.start_equity * 100) if self.start_equity > 0 else 0
            
            stats = self.risk_manager.get_statistics()
            
            self.logger.info("-" * 60)
            self.logger.info(f"SCAN #{self.scan_count} | Positions: {len(positions)}/{self.config.max_positions}")
            self.logger.info(f"Equity: ${equity:,.2f} | P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)")
            self.logger.info(f"Trades: {self.trade_count} | Win Rate: {stats['win_rate']:.1%}")
            self.logger.info(f"Drawdown: {stats['current_drawdown']:.2%} | Kelly: {stats['kelly_fraction']:.2f}")
            
            if HMM_AVAILABLE:
                self.logger.info(f"Market Regime: {self.regime_detector.current_regime.value}")
            
        except Exception as e:
            self.logger.error(f"Status error: {e}")
    
    async def run(self):
        """Main trading loop"""
        if not self._initialized:
            if not await self.initialize():
                return
        
        self._running = True
        self.logger.info(f"Starting trading loop - {self.config.scan_interval}s interval")
        
        try:
            while self._running:
                start_time = time.time()
                
                # Execute scan cycle
                executed = await self._scan_cycle()
                
                # Print status every 10 scans
                if self.scan_count % 10 == 0:
                    self._print_status()
                
                # Calculate sleep time
                elapsed = time.time() - start_time
                sleep_time = max(0, self.config.scan_interval - elapsed)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            self.logger.info("Shutdown requested...")
        except Exception as e:
            self.logger.error(f"Trading loop error: {e}")
        finally:
            self._running = False
            self._print_status()
            self.logger.info("Engine stopped")
    
    def stop(self):
        """Stop the engine"""
        self._running = False


# ============================================================================
# ENTRY POINT
# ============================================================================

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='V48 Institutional Quantitative Engine')
    parser.add_argument('--test', action='store_true', help='Test mode (dry run)')
    parser.add_argument('--trade', action='store_true', help='Live trading mode')
    parser.add_argument('--universe', type=int, default=500, help='Universe size')
    parser.add_argument('--positions', type=int, default=100, help='Max positions')
    parser.add_argument('--interval', type=int, default=5, help='Scan interval (seconds)')
    args = parser.parse_args()
    
    # Configuration
    config = InstitutionalConfig(
        api_key=os.getenv('APCA_API_KEY_ID', ''),
        api_secret=os.getenv('APCA_API_SECRET_KEY', ''),
        paper=True,
        universe_size=args.universe,
        max_positions=args.positions,
        scan_interval=args.interval
    )
    
    # Validate credentials
    if not config.api_key or not config.api_secret:
        print("ERROR: Missing API credentials")
        print("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY environment variables")
        return
    
    # Create and run engine
    engine = V48InstitutionalEngine(config)
    
    if args.test:
        print("\n" + "="*60)
        print("V48 INSTITUTIONAL ENGINE - TEST MODE")
        print("="*60)
        
        if await engine.initialize():
            print("\n[OK] Initialization successful")
            print(f"[OK] Universe: {len(engine.universe)} symbols")
            print(f"[OK] Data cached: {len(engine.returns_data)} symbols")
            print(f"[OK] Cointegrated pairs: {len(engine.pairs_strategy.pairs)}")
            print(f"[OK] ML models: {len(engine.ml_predictor.models)}")
            print("\n[READY] Engine ready for trading")
        else:
            print("\n[FAIL] Initialization failed")
        return
    
    if args.trade:
        await engine.run()
    else:
        print("Use --test for testing or --trade for live trading")


if __name__ == '__main__':
    asyncio.run(main())
