#!/usr/bin/env python3
"""
V37 ML Multi-Asset Alpha Engine - Aggressive Trading
=====================================================
Machine learning-powered multi-asset trading system with aggressive capital deployment.

Architecture:
- MLSignalGenerator: XGBoost + LightGBM ensemble for signal generation
- MultiAssetUniverse: Stocks + Leveraged ETFs + Sector ETFs + Crypto
- AggressivePositionSizer: 90% capital deployment with Kelly criterion
- IntradayTrader: 15-minute signal generation with WebSocket streaming

Features:
- ML ensemble (XGBoost momentum + LightGBM mean-reversion)
- 70+ tradeable assets across multiple asset classes
- Intraday signal generation every 15 minutes
- TWAP execution for large orders
- Stop-loss (2%) and trailing stop (1.5%) management
- Weekly model retraining on rolling 252-day window

Usage:
    # Train models
    python v37_ml_alpha_engine.py --train

    # Generate predictions
    python v37_ml_alpha_engine.py --predict

    # Live trading
    python v37_ml_alpha_engine.py --trade --live

    # Backtest
    python v37_ml_alpha_engine.py --backtest --start 2024-01-01 --end 2024-12-31

Author: V37 Alpha Engine
Version: 1.0.0
"""

import argparse
import asyncio
import json
import logging
import os
import pickle
import signal
import sys
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time as dt_time
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy import stats

# ML imports
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    warnings.warn("XGBoost not installed. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    warnings.warn("LightGBM not installed. Install with: pip install lightgbm")

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False
    warnings.warn("yfinance not installed. Install with: pip install yfinance")

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    warnings.warn("aiohttp not installed. Install with: pip install aiohttp")

try:
    import websockets
    WS_AVAILABLE = True
except ImportError:
    WS_AVAILABLE = False
    warnings.warn("websockets not installed. Install with: pip install websockets")

from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('V37_Alpha')

EST = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


# =============================================================================
# CONFIGURATION CLASSES
# =============================================================================

@dataclass
class MLConfig:
    """Configuration for ML models."""
    # XGBoost parameters
    xgb_n_estimators: int = 200
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    
    # LightGBM parameters
    lgb_n_estimators: int = 200
    lgb_max_depth: int = 6
    lgb_learning_rate: float = 0.05
    lgb_num_leaves: int = 31
    lgb_subsample: float = 0.8
    
    # Ensemble weights
    xgb_weight: float = 0.5
    lgb_weight: float = 0.5
    
    # Training parameters
    rolling_window: int = 252  # 1 year of trading days
    retrain_frequency_days: int = 7  # Weekly retraining
    min_training_samples: int = 100
    validation_split: float = 0.2
    
    # Feature parameters
    momentum_windows: List[int] = field(default_factory=lambda: [5, 20, 60])
    volatility_window: int = 20
    rsi_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_window: int = 20
    bb_std: float = 2.0


@dataclass
class UniverseConfig:
    """Configuration for multi-asset universe."""
    # Equity settings
    min_market_cap: float = 5e9  # $5 billion (lower for more stocks)
    min_avg_volume: float = 500_000  # 500k shares
    min_price: float = 5.0  # $5 minimum
    top_n_momentum: int = 50  # Top 50 momentum stocks
    
    # Universe composition
    include_leveraged_etfs: bool = True
    include_sector_etfs: bool = True
    include_crypto: bool = True
    
    # Cache settings
    cache_dir: Path = Path("cache/v37")
    cache_ttl_hours: int = 6  # Refresh more frequently


@dataclass
class PositionSizerConfig:
    """Configuration for position sizing."""
    # Capital deployment
    target_deployment: float = 0.90  # 90% capital deployment
    min_cash_buffer: float = 0.05  # 5% minimum cash
    
    # Position limits
    max_position_pct: float = 0.15  # 15% max per position
    min_position_pct: float = 0.02  # 2% minimum per position
    max_positions: int = 25  # Maximum simultaneous positions
    
    # Kelly criterion
    kelly_fraction: float = 0.5  # Half-Kelly for safety
    min_win_rate: float = 0.52  # Minimum win rate to trade
    
    # Asset class limits
    max_leveraged_pct: float = 0.30  # 30% max in leveraged ETFs
    max_crypto_pct: float = 0.10  # 10% max in crypto
    max_sector_concentration: float = 0.25  # 25% max per sector


@dataclass
class TradingConfig:
    """Configuration for trading execution."""
    # Alpaca API
    api_key: str = field(default_factory=lambda: os.getenv('ALPACA_API_KEY', ''))
    secret_key: str = field(default_factory=lambda: os.getenv('ALPACA_SECRET_KEY', ''))
    base_url: str = field(
        default_factory=lambda: os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    )
    data_url: str = field(
        default_factory=lambda: os.getenv('ALPACA_DATA_URL', 'https://data.alpaca.markets')
    )
    
    # Trading parameters
    signal_interval_minutes: int = 15
    confidence_threshold: float = 0.70  # 70% confidence to trade
    min_trades_per_day: int = 5
    max_trades_per_day: int = 50
    
    # Risk management
    stop_loss_pct: float = 0.02  # 2% stop loss
    trailing_stop_pct: float = 0.015  # 1.5% trailing stop
    
    # Execution
    twap_threshold: float = 10_000  # Use TWAP for orders > $10k
    twap_slices: int = 5
    twap_duration_minutes: int = 15
    limit_order_improvement_bps: int = 5  # Cross spread by 5 bps
    
    # Market hours (EST)
    market_open: dt_time = field(default_factory=lambda: dt_time(9, 30))
    market_close: dt_time = field(default_factory=lambda: dt_time(16, 0))


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class SignalType(Enum):
    """Trading signal types."""
    STRONG_BUY = 2
    BUY = 1
    HOLD = 0
    SELL = -1
    STRONG_SELL = -2


class AssetClass(Enum):
    """Asset class classification."""
    EQUITY = "equity"
    LEVERAGED_ETF = "leveraged_etf"
    SECTOR_ETF = "sector_etf"
    CRYPTO = "crypto"


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class Asset:
    """Represents a tradeable asset."""
    symbol: str
    asset_class: AssetClass
    sector: Optional[str] = None
    leverage: float = 1.0
    is_crypto: bool = False


@dataclass
class MLSignal:
    """ML-generated trading signal."""
    symbol: str
    signal_type: SignalType
    probability: float  # Probability of positive return
    magnitude: float  # Predicted return magnitude
    confidence: float  # Model confidence (0-1)
    xgb_score: float
    lgb_score: float
    features: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(EST))


@dataclass
class Position:
    """Current position in an asset."""
    symbol: str
    shares: float
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    entry_time: datetime
    highest_price: float  # For trailing stop
    stop_loss_price: float
    trailing_stop_price: float


@dataclass
class Trade:
    """Executed trade record."""
    symbol: str
    side: OrderSide
    shares: float
    price: float
    value: float
    order_id: str
    timestamp: datetime
    reason: str
    signal: Optional[MLSignal] = None


@dataclass
class PortfolioState:
    """Current portfolio state."""
    cash: float
    equity: float
    total_value: float
    positions: Dict[str, Position]
    deployment_pct: float
    open_orders: int
    daily_trades: int
    daily_pnl: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(EST))


# =============================================================================
# LEVERAGED & SECTOR ETF DEFINITIONS
# =============================================================================

LEVERAGED_ETFS = [
    Asset("TQQQ", AssetClass.LEVERAGED_ETF, "technology", 3.0),  # 3x Nasdaq
    Asset("SQQQ", AssetClass.LEVERAGED_ETF, "technology", -3.0),  # -3x Nasdaq
    Asset("SPXL", AssetClass.LEVERAGED_ETF, "broad_market", 3.0),  # 3x S&P 500
    Asset("SPXS", AssetClass.LEVERAGED_ETF, "broad_market", -3.0),  # -3x S&P 500
    Asset("SOXL", AssetClass.LEVERAGED_ETF, "semiconductors", 3.0),  # 3x Semiconductors
    Asset("SOXS", AssetClass.LEVERAGED_ETF, "semiconductors", -3.0),  # -3x Semiconductors
    Asset("UPRO", AssetClass.LEVERAGED_ETF, "broad_market", 3.0),  # 3x S&P 500
    Asset("TNA", AssetClass.LEVERAGED_ETF, "small_cap", 3.0),  # 3x Russell 2000
    Asset("TZA", AssetClass.LEVERAGED_ETF, "small_cap", -3.0),  # -3x Russell 2000
    Asset("LABU", AssetClass.LEVERAGED_ETF, "biotech", 3.0),  # 3x Biotech
    Asset("LABD", AssetClass.LEVERAGED_ETF, "biotech", -3.0),  # -3x Biotech
    Asset("FAS", AssetClass.LEVERAGED_ETF, "financials", 3.0),  # 3x Financials
    Asset("FAZ", AssetClass.LEVERAGED_ETF, "financials", -3.0),  # -3x Financials
    Asset("TECL", AssetClass.LEVERAGED_ETF, "technology", 3.0),  # 3x Technology
    Asset("TECS", AssetClass.LEVERAGED_ETF, "technology", -3.0),  # -3x Technology
]

SECTOR_ETFS = [
    Asset("XLK", AssetClass.SECTOR_ETF, "technology"),  # Technology
    Asset("XLF", AssetClass.SECTOR_ETF, "financials"),  # Financials
    Asset("XLE", AssetClass.SECTOR_ETF, "energy"),  # Energy
    Asset("XLV", AssetClass.SECTOR_ETF, "healthcare"),  # Healthcare
    Asset("XLI", AssetClass.SECTOR_ETF, "industrials"),  # Industrials
    Asset("XLY", AssetClass.SECTOR_ETF, "consumer_discretionary"),  # Consumer Discretionary
    Asset("XLP", AssetClass.SECTOR_ETF, "consumer_staples"),  # Consumer Staples
    Asset("XLB", AssetClass.SECTOR_ETF, "materials"),  # Materials
    Asset("XLU", AssetClass.SECTOR_ETF, "utilities"),  # Utilities
    Asset("XLRE", AssetClass.SECTOR_ETF, "real_estate"),  # Real Estate
    Asset("XLC", AssetClass.SECTOR_ETF, "communication"),  # Communication Services
    Asset("SMH", AssetClass.SECTOR_ETF, "semiconductors"),  # Semiconductors
    Asset("IBB", AssetClass.SECTOR_ETF, "biotech"),  # Biotech
    Asset("XBI", AssetClass.SECTOR_ETF, "biotech"),  # Biotech (equal weight)
    Asset("XHB", AssetClass.SECTOR_ETF, "homebuilders"),  # Homebuilders
]

CRYPTO_ASSETS = [
    Asset("BTCUSD", AssetClass.CRYPTO, "crypto", is_crypto=True),  # Bitcoin
    Asset("ETHUSD", AssetClass.CRYPTO, "crypto", is_crypto=True),  # Ethereum
]

# S&P 500 representative symbols for momentum screening
SP500_SYMBOLS = [
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'GOOG', 'TSLA', 'BRK-B', 'UNH',
    'XOM', 'JNJ', 'JPM', 'V', 'PG', 'MA', 'AVGO', 'HD', 'CVX', 'MRK',
    'ABBV', 'LLY', 'PEP', 'KO', 'COST', 'ADBE', 'WMT', 'MCD', 'CSCO', 'CRM',
    'BAC', 'PFE', 'TMO', 'ACN', 'NFLX', 'AMD', 'ABT', 'DHR', 'DIS', 'LIN',
    'CMCSA', 'VZ', 'INTC', 'NKE', 'PM', 'WFC', 'TXN', 'NEE', 'RTX', 'UPS',
    'QCOM', 'BMY', 'COP', 'HON', 'LOW', 'ORCL', 'UNP', 'SPGI', 'IBM', 'CAT',
    'GE', 'BA', 'INTU', 'AMAT', 'AMGN', 'GS', 'SBUX', 'BLK', 'DE', 'ELV',
    'ISRG', 'MDLZ', 'ADP', 'GILD', 'ADI', 'BKNG', 'VRTX', 'TJX', 'PLD', 'MMC',
    'SYK', 'MS', 'CVS', 'LMT', 'REGN', 'CI', 'TMUS', 'CB', 'SCHW', 'ZTS',
    'ETN', 'MO', 'SO', 'BDX', 'EOG', 'DUK', 'AMT', 'BSX', 'LRCX', 'NOC',
    'PYPL', 'AON', 'CME', 'ICE', 'ITW', 'WM', 'SLB', 'APD', 'CSX', 'CL',
    'PNC', 'TGT', 'FCX', 'MCK', 'EMR', 'MPC', 'USB', 'SHW', 'SNPS', 'NSC',
    'FDX', 'CDNS', 'GD', 'ORLY', 'PSX', 'AZO', 'OXY', 'TFC', 'AJG', 'KLAC',
    'MCO', 'ROP', 'HUM', 'MCHP', 'PCAR', 'VLO', 'MAR', 'AEP', 'MET', 'KMB',
    'CTAS', 'AFL', 'MSCI', 'D', 'AIG', 'TRV', 'CCI', 'GIS', 'PSA', 'JCI',
    'HCA', 'APH', 'WELL', 'CMG', 'DXCM', 'F', 'GM', 'TEL', 'CARR', 'NUE',
    'ADM', 'SRE', 'CHTR', 'WMB', 'STZ', 'HES', 'DVN', 'KHC', 'A', 'IDXX',
    'BIIB', 'EW', 'DHI', 'LHX', 'HAL', 'AMP', 'EXC', 'DOW', 'PAYX', 'MNST',
    'ROK', 'PRU', 'MTD', 'ODFL', 'FTNT', 'SPG', 'XEL', 'ED', 'ROST', 'OTIS',
    'AME', 'BK', 'CTSH', 'GWW', 'DD', 'CMI', 'CPRT', 'EA', 'IQV', 'PEG',
]


# =============================================================================
# ML SIGNAL GENERATOR
# =============================================================================

class MLSignalGenerator:
    """
    ML-powered signal generator using XGBoost and LightGBM ensemble.
    
    XGBoost focuses on momentum signals while LightGBM handles mean-reversion.
    The ensemble combines both for robust predictions.
    
    Args:
        config: ML configuration parameters
    
    Example:
        generator = MLSignalGenerator()
        generator.train(historical_data)
        signals = generator.generate_signals(current_data)
    """

    def __init__(self, config: Optional[MLConfig] = None):
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost required. Install with: pip install xgboost")
        if not LGB_AVAILABLE:
            raise ImportError("LightGBM required. Install with: pip install lightgbm")
        
        self.config = config or MLConfig()
        
        # Models
        self.xgb_classifier: Optional[xgb.XGBClassifier] = None
        self.xgb_regressor: Optional[xgb.XGBRegressor] = None
        self.lgb_classifier: Optional[lgb.LGBMClassifier] = None
        self.lgb_regressor: Optional[lgb.LGBMRegressor] = None
        
        # Model metadata
        self._is_fitted = False
        self._last_train_date: Optional[datetime] = None
        self._feature_names: List[str] = []
        self._train_history: List[Dict] = []
        
        # Cache directory
        self.model_dir = Path("models/v37")
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(
        self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD, Signal Line, and Histogram."""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def _calculate_bollinger_bands(
        self, prices: pd.Series, window: int = 20, num_std: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        middle = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        return upper, middle, lower

    def _calculate_features(self, prices: pd.DataFrame, volumes: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all features for ML models.
        
        Args:
            prices: DataFrame of adjusted close prices (columns = symbols)
            volumes: DataFrame of volumes (columns = symbols)
        
        Returns:
            DataFrame with all features
        """
        features_list = []
        
        for symbol in prices.columns:
            if symbol not in volumes.columns:
                continue
                
            price = prices[symbol].dropna()
            volume = volumes[symbol].dropna()
            
            if len(price) < max(self.config.momentum_windows) + 10:
                continue
            
            # Align price and volume
            common_idx = price.index.intersection(volume.index)
            price = price.loc[common_idx]
            volume = volume.loc[common_idx]
            
            # Returns
            returns = price.pct_change()
            log_returns = np.log(price / price.shift(1))
            
            # Momentum features
            momentum_5d = price.pct_change(5)
            momentum_20d = price.pct_change(20)
            momentum_60d = price.pct_change(60)
            
            # Volatility
            volatility_20d = returns.rolling(20).std() * np.sqrt(252)
            volatility_5d = returns.rolling(5).std() * np.sqrt(252)
            
            # RSI
            rsi = self._calculate_rsi(price, self.config.rsi_window)
            
            # MACD
            macd_line, signal_line, macd_hist = self._calculate_macd(
                price, self.config.macd_fast, self.config.macd_slow, self.config.macd_signal
            )
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(
                price, self.config.bb_window, self.config.bb_std
            )
            bb_position = (price - bb_lower) / (bb_upper - bb_lower + 1e-10)
            bb_width = (bb_upper - bb_lower) / bb_middle
            
            # Z-score (mean reversion signal)
            zscore_20d = (price - price.rolling(20).mean()) / (price.rolling(20).std() + 1e-10)
            zscore_60d = (price - price.rolling(60).mean()) / (price.rolling(60).std() + 1e-10)
            
            # Volume features
            volume_ma = volume.rolling(20).mean()
            volume_ratio = volume / (volume_ma + 1e-10)
            
            # Price acceleration
            momentum_diff = momentum_5d.diff(5)
            
            # Create feature DataFrame
            symbol_features = pd.DataFrame({
                'symbol': symbol,
                'price': price,
                'returns': returns,
                'log_returns': log_returns,
                
                # Momentum features (XGBoost focus)
                'momentum_5d': momentum_5d,
                'momentum_20d': momentum_20d,
                'momentum_60d': momentum_60d,
                'momentum_acceleration': momentum_diff,
                
                # Volatility
                'volatility_5d': volatility_5d,
                'volatility_20d': volatility_20d,
                'vol_ratio': volatility_5d / (volatility_20d + 1e-10),
                
                # Mean reversion features (LightGBM focus)
                'rsi': rsi,
                'rsi_oversold': (rsi < 30).astype(float),
                'rsi_overbought': (rsi > 70).astype(float),
                
                'bb_position': bb_position,
                'bb_width': bb_width,
                'price_vs_bb_upper': (price - bb_upper) / price,
                'price_vs_bb_lower': (price - bb_lower) / price,
                
                'zscore_20d': zscore_20d,
                'zscore_60d': zscore_60d,
                
                # MACD
                'macd': macd_line,
                'macd_signal': signal_line,
                'macd_hist': macd_hist,
                'macd_crossover': (macd_line > signal_line).astype(float),
                
                # Volume
                'volume_ratio': volume_ratio,
                'volume_trend': volume.pct_change(5),
                
                # Target: next-day return (for training)
                'target_return': returns.shift(-1),
                'target_direction': (returns.shift(-1) > 0).astype(int),
            }, index=price.index)
            
            features_list.append(symbol_features)
        
        if not features_list:
            return pd.DataFrame()
        
        all_features = pd.concat(features_list, ignore_index=True)
        return all_features

    def _get_feature_columns(self) -> List[str]:
        """Get list of feature columns for ML models."""
        return [
            'momentum_5d', 'momentum_20d', 'momentum_60d', 'momentum_acceleration',
            'volatility_5d', 'volatility_20d', 'vol_ratio',
            'rsi', 'rsi_oversold', 'rsi_overbought',
            'bb_position', 'bb_width', 'price_vs_bb_upper', 'price_vs_bb_lower',
            'zscore_20d', 'zscore_60d',
            'macd', 'macd_signal', 'macd_hist', 'macd_crossover',
            'volume_ratio', 'volume_trend',
        ]

    def train(self, prices: pd.DataFrame, volumes: pd.DataFrame) -> Dict[str, float]:
        """
        Train ML models on historical data.
        
        Args:
            prices: DataFrame of adjusted close prices
            volumes: DataFrame of volumes
        
        Returns:
            Dictionary of training metrics
        """
        logger.info("Starting ML model training...")
        
        # Calculate features
        features = self._calculate_features(prices, volumes)
        if features.empty:
            raise ValueError("No valid features calculated")
        
        # Clean data
        feature_cols = self._get_feature_columns()
        self._feature_names = feature_cols
        
        clean_features = features.dropna(subset=feature_cols + ['target_direction', 'target_return'])
        
        if len(clean_features) < self.config.min_training_samples:
            raise ValueError(f"Insufficient training samples: {len(clean_features)}")
        
        X = clean_features[feature_cols].values
        y_class = clean_features['target_direction'].values
        y_reg = clean_features['target_return'].values
        
        # Split data
        split_idx = int(len(X) * (1 - self.config.validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_class_train, y_class_val = y_class[:split_idx], y_class[split_idx:]
        y_reg_train, y_reg_val = y_reg[:split_idx], y_reg[split_idx:]
        
        logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        # Train XGBoost Classifier (momentum-focused)
        self.xgb_classifier = xgb.XGBClassifier(
            n_estimators=self.config.xgb_n_estimators,
            max_depth=self.config.xgb_max_depth,
            learning_rate=self.config.xgb_learning_rate,
            subsample=self.config.xgb_subsample,
            colsample_bytree=self.config.xgb_colsample_bytree,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        )
        self.xgb_classifier.fit(X_train, y_class_train)
        xgb_class_acc = self.xgb_classifier.score(X_val, y_class_val)
        
        # Train XGBoost Regressor
        self.xgb_regressor = xgb.XGBRegressor(
            n_estimators=self.config.xgb_n_estimators,
            max_depth=self.config.xgb_max_depth,
            learning_rate=self.config.xgb_learning_rate,
            subsample=self.config.xgb_subsample,
            colsample_bytree=self.config.xgb_colsample_bytree,
            random_state=42,
            n_jobs=-1
        )
        self.xgb_regressor.fit(X_train, y_reg_train)
        xgb_reg_r2 = self.xgb_regressor.score(X_val, y_reg_val)
        
        # Train LightGBM Classifier (mean-reversion focused)
        self.lgb_classifier = lgb.LGBMClassifier(
            n_estimators=self.config.lgb_n_estimators,
            max_depth=self.config.lgb_max_depth,
            learning_rate=self.config.lgb_learning_rate,
            num_leaves=self.config.lgb_num_leaves,
            subsample=self.config.lgb_subsample,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        self.lgb_classifier.fit(X_train, y_class_train)
        lgb_class_acc = self.lgb_classifier.score(X_val, y_class_val)
        
        # Train LightGBM Regressor
        self.lgb_regressor = lgb.LGBMRegressor(
            n_estimators=self.config.lgb_n_estimators,
            max_depth=self.config.lgb_max_depth,
            learning_rate=self.config.lgb_learning_rate,
            num_leaves=self.config.lgb_num_leaves,
            subsample=self.config.lgb_subsample,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        self.lgb_regressor.fit(X_train, y_reg_train)
        lgb_reg_r2 = self.lgb_regressor.score(X_val, y_reg_val)
        
        self._is_fitted = True
        self._last_train_date = datetime.now(EST)
        
        metrics = {
            'xgb_classifier_accuracy': xgb_class_acc,
            'xgb_regressor_r2': xgb_reg_r2,
            'lgb_classifier_accuracy': lgb_class_acc,
            'lgb_regressor_r2': lgb_reg_r2,
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'train_date': self._last_train_date.isoformat(),
        }
        
        self._train_history.append(metrics)
        logger.info(f"Training complete. XGB Acc: {xgb_class_acc:.4f}, LGB Acc: {lgb_class_acc:.4f}")
        
        return metrics

    def generate_signals(
        self, prices: pd.DataFrame, volumes: pd.DataFrame
    ) -> Dict[str, MLSignal]:
        """
        Generate trading signals for all symbols.
        
        Args:
            prices: DataFrame of adjusted close prices
            volumes: DataFrame of volumes
        
        Returns:
            Dictionary mapping symbol to MLSignal
        """
        if not self._is_fitted:
            raise RuntimeError("Models not trained. Call train() first.")
        
        # Calculate features
        features = self._calculate_features(prices, volumes)
        if features.empty:
            return {}
        
        feature_cols = self._feature_names
        signals = {}
        
        # Group by symbol and get latest features
        for symbol in features['symbol'].unique():
            symbol_data = features[features['symbol'] == symbol].iloc[-1]
            
            # Check for valid features
            if symbol_data[feature_cols].isna().any():
                continue
            
            X = symbol_data[feature_cols].values.reshape(1, -1)
            
            # XGBoost predictions
            xgb_prob = self.xgb_classifier.predict_proba(X)[0, 1]
            xgb_return = self.xgb_regressor.predict(X)[0]
            
            # LightGBM predictions
            lgb_prob = self.lgb_classifier.predict_proba(X)[0, 1]
            lgb_return = self.lgb_regressor.predict(X)[0]
            
            # Ensemble
            ensemble_prob = (
                self.config.xgb_weight * xgb_prob +
                self.config.lgb_weight * lgb_prob
            )
            ensemble_return = (
                self.config.xgb_weight * xgb_return +
                self.config.lgb_weight * lgb_return
            )
            
            # Determine signal type based on probability
            if ensemble_prob >= 0.70:
                signal_type = SignalType.STRONG_BUY
            elif ensemble_prob >= 0.55:
                signal_type = SignalType.BUY
            elif ensemble_prob <= 0.30:
                signal_type = SignalType.STRONG_SELL
            elif ensemble_prob <= 0.45:
                signal_type = SignalType.SELL
            else:
                signal_type = SignalType.HOLD
            
            # Calculate confidence as agreement between models
            prob_diff = abs(xgb_prob - lgb_prob)
            confidence = 1 - prob_diff  # Higher agreement = higher confidence
            
            signals[symbol] = MLSignal(
                symbol=symbol,
                signal_type=signal_type,
                probability=ensemble_prob,
                magnitude=ensemble_return,
                confidence=confidence,
                xgb_score=xgb_prob,
                lgb_score=lgb_prob,
                features={col: float(symbol_data[col]) for col in feature_cols[:5]},  # Top features
            )
        
        logger.info(f"Generated signals for {len(signals)} symbols")
        return signals

    def save_models(self, path: Optional[Path] = None) -> None:
        """Save trained models to disk."""
        save_path = path or self.model_dir / "ml_models.pkl"
        
        model_data = {
            'xgb_classifier': self.xgb_classifier,
            'xgb_regressor': self.xgb_regressor,
            'lgb_classifier': self.lgb_classifier,
            'lgb_regressor': self.lgb_regressor,
            'feature_names': self._feature_names,
            'last_train_date': self._last_train_date,
            'config': self.config,
            'train_history': self._train_history,
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Models saved to {save_path}")

    def load_models(self, path: Optional[Path] = None) -> bool:
        """Load trained models from disk."""
        load_path = path or self.model_dir / "ml_models.pkl"
        
        if not load_path.exists():
            logger.warning(f"Model file not found: {load_path}")
            return False
        
        try:
            with open(load_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.xgb_classifier = model_data['xgb_classifier']
            self.xgb_regressor = model_data['xgb_regressor']
            self.lgb_classifier = model_data['lgb_classifier']
            self.lgb_regressor = model_data['lgb_regressor']
            self._feature_names = model_data['feature_names']
            self._last_train_date = model_data['last_train_date']
            self._train_history = model_data.get('train_history', [])
            self._is_fitted = True
            
            logger.info(f"Models loaded from {load_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False

    def needs_retrain(self) -> bool:
        """Check if models need retraining based on frequency."""
        if not self._is_fitted or self._last_train_date is None:
            return True
        
        days_since_train = (datetime.now(EST) - self._last_train_date).days
        return days_since_train >= self.config.retrain_frequency_days


# =============================================================================
# MULTI-ASSET UNIVERSE
# =============================================================================

class MultiAssetUniverse:
    """
    Multi-asset universe manager for stocks, ETFs, and crypto.
    
    Dynamically screens top momentum stocks and combines with
    leveraged ETFs, sector ETFs, and crypto assets.
    
    Args:
        config: Universe configuration parameters
    
    Example:
        universe = MultiAssetUniverse()
        assets = universe.get_tradeable_assets()
        prices = universe.get_prices(lookback_days=60)
    """

    def __init__(self, config: Optional[UniverseConfig] = None):
        if not YF_AVAILABLE:
            raise ImportError("yfinance required. Install with: pip install yfinance")
        
        self.config = config or UniverseConfig()
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._cache_file = self.config.cache_dir / "universe_cache.pkl"
        self._price_cache: Dict[str, pd.DataFrame] = {}
        self._tradeable_assets: List[Asset] = []

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        if not self._cache_file.exists():
            return False
        
        mtime = datetime.fromtimestamp(self._cache_file.stat().st_mtime)
        age = datetime.now() - mtime
        return age < timedelta(hours=self.config.cache_ttl_hours)

    def _load_cache(self) -> Optional[List[Asset]]:
        """Load cached universe."""
        try:
            with open(self._cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None

    def _save_cache(self, assets: List[Asset]) -> None:
        """Save universe to cache."""
        try:
            with open(self._cache_file, 'wb') as f:
                pickle.dump(assets, f)
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")

    def _screen_momentum_stocks(self) -> List[Asset]:
        """Screen S&P 500 for top momentum stocks."""
        logger.info("Screening momentum stocks...")
        
        try:
            # Download price data
            data = yf.download(
                SP500_SYMBOLS,
                period="6mo",
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=True
            )
            
            if data.empty:
                logger.warning("Failed to download stock data")
                return []
            
            prices = data['Close'] if 'Close' in data.columns else data
            
            # Calculate momentum scores
            momentum_scores = {}
            for symbol in prices.columns:
                try:
                    price_series = prices[symbol].dropna()
                    if len(price_series) < 60:
                        continue
                    
                    # Calculate 3-month and 1-month momentum
                    mom_3m = (price_series.iloc[-1] / price_series.iloc[-63]) - 1 if len(price_series) >= 63 else 0
                    mom_1m = (price_series.iloc[-1] / price_series.iloc[-21]) - 1 if len(price_series) >= 21 else 0
                    
                    # Volatility-adjusted momentum
                    volatility = price_series.pct_change().std() * np.sqrt(252)
                    vol_adj_mom = mom_3m / (volatility + 0.01)
                    
                    # Composite score
                    score = 0.4 * mom_3m + 0.3 * mom_1m + 0.3 * vol_adj_mom
                    momentum_scores[symbol] = score
                except Exception:
                    continue
            
            # Sort and select top N
            sorted_symbols = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
            top_symbols = sorted_symbols[:self.config.top_n_momentum]
            
            assets = [
                Asset(symbol=sym, asset_class=AssetClass.EQUITY)
                for sym, score in top_symbols
            ]
            
            logger.info(f"Selected {len(assets)} momentum stocks")
            return assets
            
        except Exception as e:
            logger.error(f"Momentum screening failed: {e}")
            return []

    def get_tradeable_assets(self, force_refresh: bool = False) -> List[Asset]:
        """
        Get full list of tradeable assets.
        
        Args:
            force_refresh: Force refresh even if cache is valid
        
        Returns:
            List of Asset objects
        """
        # Check cache
        if not force_refresh and self._is_cache_valid():
            cached = self._load_cache()
            if cached:
                self._tradeable_assets = cached
                return cached
        
        assets = []
        
        # Add momentum stocks
        momentum_stocks = self._screen_momentum_stocks()
        assets.extend(momentum_stocks)
        
        # Add leveraged ETFs
        if self.config.include_leveraged_etfs:
            assets.extend(LEVERAGED_ETFS)
            logger.info(f"Added {len(LEVERAGED_ETFS)} leveraged ETFs")
        
        # Add sector ETFs
        if self.config.include_sector_etfs:
            assets.extend(SECTOR_ETFS)
            logger.info(f"Added {len(SECTOR_ETFS)} sector ETFs")
        
        # Add crypto
        if self.config.include_crypto:
            assets.extend(CRYPTO_ASSETS)
            logger.info(f"Added {len(CRYPTO_ASSETS)} crypto assets")
        
        self._tradeable_assets = assets
        self._save_cache(assets)
        
        logger.info(f"Total tradeable universe: {len(assets)} assets")
        return assets

    def get_symbols(self, asset_class: Optional[AssetClass] = None) -> List[str]:
        """Get list of symbols, optionally filtered by asset class."""
        if not self._tradeable_assets:
            self.get_tradeable_assets()
        
        if asset_class:
            return [a.symbol for a in self._tradeable_assets if a.asset_class == asset_class]
        
        return [a.symbol for a in self._tradeable_assets]

    def get_prices(self, lookback_days: int = 252) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get historical prices and volumes for all assets.
        
        Args:
            lookback_days: Number of days of history
        
        Returns:
            Tuple of (prices DataFrame, volumes DataFrame)
        """
        symbols = self.get_symbols()
        
        # Separate crypto symbols (different API)
        stock_symbols = [s for s in symbols if not s.endswith('USD')]
        crypto_symbols = [s for s in symbols if s.endswith('USD')]
        
        prices_list = []
        volumes_list = []
        
        # Download stock/ETF data
        if stock_symbols:
            try:
                data = yf.download(
                    stock_symbols,
                    period=f"{lookback_days}d",
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    threads=True
                )
                
                if not data.empty:
                    if 'Close' in data.columns:
                        prices_list.append(data['Close'])
                        volumes_list.append(data['Volume'])
                    else:
                        prices_list.append(data)
            except Exception as e:
                logger.error(f"Stock data download failed: {e}")
        
        # Download crypto data (if supported)
        # Note: Alpaca Crypto API would be used in live trading
        # For now, we use yfinance tickers like BTC-USD
        if crypto_symbols:
            try:
                crypto_tickers = [s.replace('USD', '-USD') for s in crypto_symbols]
                data = yf.download(
                    crypto_tickers,
                    period=f"{lookback_days}d",
                    interval="1d",
                    auto_adjust=True,
                    progress=False
                )
                
                if not data.empty:
                    # Rename columns back to original format
                    if isinstance(data.columns, pd.MultiIndex):
                        prices = data['Close'].copy()
                        volumes = data['Volume'].copy()
                        prices.columns = crypto_symbols
                        volumes.columns = crypto_symbols
                        prices_list.append(prices)
                        volumes_list.append(volumes)
            except Exception as e:
                logger.warning(f"Crypto data download failed: {e}")
        
        # Combine all data
        if prices_list:
            all_prices = pd.concat(prices_list, axis=1)
            all_volumes = pd.concat(volumes_list, axis=1) if volumes_list else pd.DataFrame()
            return all_prices, all_volumes
        
        return pd.DataFrame(), pd.DataFrame()


# =============================================================================
# AGGRESSIVE POSITION SIZER
# =============================================================================

class AggressivePositionSizer:
    """
    Aggressive position sizing with 90% capital deployment and Kelly criterion.
    
    Features:
    - Target 90% capital deployment
    - Kelly criterion with fractional Kelly for safety
    - Position limits (2-15% per position)
    - Asset class concentration limits
    
    Args:
        config: Position sizing configuration
    
    Example:
        sizer = AggressivePositionSizer()
        allocations = sizer.calculate_allocations(signals, portfolio, prices)
    """

    def __init__(self, config: Optional[PositionSizerConfig] = None):
        self.config = config or PositionSizerConfig()

    def _kelly_fraction(self, win_rate: float, win_loss_ratio: float) -> float:
        """
        Calculate Kelly criterion fraction.
        
        Kelly % = W - [(1-W) / R]
        Where W = win rate, R = win/loss ratio
        """
        if win_rate <= 0 or win_loss_ratio <= 0:
            return 0.0
        
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # Apply fractional Kelly
        kelly = kelly * self.config.kelly_fraction
        
        # Clamp to reasonable range
        return max(0, min(kelly, 0.25))

    def _get_asset_info(self, symbol: str, universe: MultiAssetUniverse) -> Optional[Asset]:
        """Get Asset info for a symbol."""
        for asset in universe._tradeable_assets:
            if asset.symbol == symbol:
                return asset
        return None

    def calculate_allocations(
        self,
        signals: Dict[str, MLSignal],
        portfolio: PortfolioState,
        universe: MultiAssetUniverse,
        prices: pd.DataFrame
    ) -> Dict[str, Tuple[float, str]]:
        """
        Calculate target allocations for all assets.
        
        Args:
            signals: ML signals for each symbol
            portfolio: Current portfolio state
            universe: Asset universe
            prices: Current prices
        
        Returns:
            Dictionary of symbol -> (target_allocation, action)
            where allocation is dollar amount and action is 'buy', 'sell', or 'hold'
        """
        total_value = portfolio.total_value
        target_equity = total_value * self.config.target_deployment
        min_cash = total_value * self.config.min_cash_buffer
        
        allocations: Dict[str, Tuple[float, str]] = {}
        
        # Filter signals by confidence threshold
        high_confidence_signals = {
            sym: sig for sym, sig in signals.items()
            if sig.confidence >= 0.5 and sig.probability >= 0.50
        }
        
        if not high_confidence_signals:
            logger.warning("No high-confidence signals available")
            return {}
        
        # Calculate raw Kelly allocations
        kelly_allocations = {}
        for symbol, signal in high_confidence_signals.items():
            if signal.signal_type in [SignalType.STRONG_BUY, SignalType.BUY]:
                # Estimate win rate from probability
                win_rate = signal.probability
                
                # Estimate win/loss ratio from magnitude
                # Assuming average win is 2x average loss (conservative)
                win_loss_ratio = 2.0 + abs(signal.magnitude) * 10
                
                kelly = self._kelly_fraction(win_rate, win_loss_ratio)
                kelly_allocations[symbol] = kelly
            
            elif signal.signal_type in [SignalType.STRONG_SELL, SignalType.SELL]:
                # For sell signals, mark for position reduction
                kelly_allocations[symbol] = -0.1  # Negative indicates sell
        
        if not kelly_allocations:
            return {}
        
        # Normalize positive allocations to target deployment
        positive_allocs = {s: k for s, k in kelly_allocations.items() if k > 0}
        total_kelly = sum(positive_allocs.values())
        
        if total_kelly <= 0:
            return {}
        
        # Scale allocations
        for symbol, kelly in positive_allocs.items():
            raw_alloc = (kelly / total_kelly) * target_equity
            
            # Apply position limits
            max_position = total_value * self.config.max_position_pct
            min_position = total_value * self.config.min_position_pct
            
            # Get asset info for leverage adjustment
            asset = self._get_asset_info(symbol, universe)
            if asset and asset.asset_class == AssetClass.LEVERAGED_ETF:
                # Reduce allocation for leveraged products
                raw_alloc = raw_alloc * 0.5
            
            # Clamp allocation
            target_alloc = max(min_position, min(raw_alloc, max_position))
            
            # Determine action
            current_position = portfolio.positions.get(symbol)
            if current_position:
                current_value = current_position.market_value
                diff = target_alloc - current_value
                if abs(diff) > min_position * 0.2:  # 20% threshold to rebalance
                    action = 'buy' if diff > 0 else 'sell'
                else:
                    action = 'hold'
            else:
                action = 'buy'
            
            allocations[symbol] = (target_alloc, action)
        
        # Handle sell signals
        for symbol, kelly in kelly_allocations.items():
            if kelly < 0 and symbol in portfolio.positions:
                allocations[symbol] = (0.0, 'sell')
        
        # Apply asset class limits
        allocations = self._apply_asset_class_limits(allocations, universe, total_value)
        
        # Limit to max positions
        if len(allocations) > self.config.max_positions:
            # Sort by allocation size and keep top N
            sorted_allocs = sorted(
                allocations.items(),
                key=lambda x: x[1][0],
                reverse=True
            )[:self.config.max_positions]
            allocations = dict(sorted_allocs)
        
        logger.info(f"Calculated allocations for {len(allocations)} positions")
        return allocations

    def _apply_asset_class_limits(
        self,
        allocations: Dict[str, Tuple[float, str]],
        universe: MultiAssetUniverse,
        total_value: float
    ) -> Dict[str, Tuple[float, str]]:
        """Apply asset class concentration limits."""
        max_leveraged = total_value * self.config.max_leveraged_pct
        max_crypto = total_value * self.config.max_crypto_pct
        
        leveraged_total = 0.0
        crypto_total = 0.0
        adjusted = {}
        
        for symbol, (alloc, action) in allocations.items():
            asset = self._get_asset_info(symbol, universe)
            if not asset:
                adjusted[symbol] = (alloc, action)
                continue
            
            if asset.asset_class == AssetClass.LEVERAGED_ETF:
                if leveraged_total + alloc > max_leveraged:
                    alloc = max(0, max_leveraged - leveraged_total)
                leveraged_total += alloc
            
            elif asset.asset_class == AssetClass.CRYPTO:
                if crypto_total + alloc > max_crypto:
                    alloc = max(0, max_crypto - crypto_total)
                crypto_total += alloc
            
            if alloc > 0 or action == 'sell':
                adjusted[symbol] = (alloc, action)
        
        return adjusted

    def calculate_order_size(
        self,
        symbol: str,
        target_allocation: float,
        current_position: Optional[Position],
        current_price: float,
        portfolio_value: float
    ) -> Tuple[int, OrderSide]:
        """
        Calculate exact order size in shares.
        
        Returns:
            Tuple of (shares, side)
        """
        current_value = current_position.market_value if current_position else 0.0
        target_shares = int(target_allocation / current_price)
        current_shares = int(current_position.shares) if current_position else 0
        
        shares_diff = target_shares - current_shares
        
        if shares_diff > 0:
            return shares_diff, OrderSide.BUY
        elif shares_diff < 0:
            return abs(shares_diff), OrderSide.SELL
        else:
            return 0, OrderSide.BUY


# =============================================================================
# ALPACA API CLIENT
# =============================================================================

class AlpacaClient:
    """Async Alpaca API client for trading and data."""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure HTTP session exists."""
        if self.session is None or self.session.closed:
            headers = {
                'APCA-API-KEY-ID': self.config.api_key,
                'APCA-API-SECRET-KEY': self.config.secret_key,
            }
            self.session = aiohttp.ClientSession(headers=headers)
        return self.session

    async def close(self) -> None:
        """Close session."""
        if self.session and not self.session.closed:
            await self.session.close()

    async def get_account(self) -> Optional[Dict[str, Any]]:
        """Get account information."""
        session = await self._ensure_session()
        url = f"{self.config.base_url}/v2/account"
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    return await resp.json()
                logger.error(f"Account fetch failed: {resp.status}")
        except aiohttp.ClientError as e:
            logger.error(f"Account fetch error: {e}")
        return None

    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get all positions."""
        session = await self._ensure_session()
        url = f"{self.config.base_url}/v2/positions"
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    return await resp.json()
        except aiohttp.ClientError as e:
            logger.error(f"Positions fetch error: {e}")
        return []

    async def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest quote for a symbol."""
        session = await self._ensure_session()
        url = f"{self.config.data_url}/v2/stocks/{symbol}/quotes/latest"
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get('quote', {})
        except aiohttp.ClientError as e:
            logger.error(f"Quote fetch error for {symbol}: {e}")
        return None

    async def get_bars(
        self, symbol: str, timeframe: str = "15Min", limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get historical bars."""
        session = await self._ensure_session()
        url = f"{self.config.data_url}/v2/stocks/{symbol}/bars"
        params = {'timeframe': timeframe, 'limit': limit}
        try:
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get('bars', [])
        except aiohttp.ClientError as e:
            logger.error(f"Bars fetch error for {symbol}: {e}")
        return []

    async def submit_order(
        self,
        symbol: str,
        shares: int,
        side: OrderSide,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Submit an order."""
        session = await self._ensure_session()
        url = f"{self.config.base_url}/v2/orders"
        
        payload = {
            'symbol': symbol,
            'qty': str(abs(shares)),
            'side': side.value,
            'type': order_type,
            'time_in_force': 'day'
        }
        
        if order_type == "limit" and limit_price:
            payload['limit_price'] = str(round(limit_price, 2))
        
        # Add bracket order for stop-loss
        if stop_loss or take_profit:
            payload['order_class'] = 'bracket'
            if stop_loss:
                payload['stop_loss'] = {'stop_price': str(round(stop_loss, 2))}
            if take_profit:
                payload['take_profit'] = {'limit_price': str(round(take_profit, 2))}
        
        try:
            async with session.post(url, json=payload) as resp:
                if resp.status in (200, 201):
                    return await resp.json()
                else:
                    error = await resp.text()
                    logger.error(f"Order submit failed: {resp.status} - {error}")
        except aiohttp.ClientError as e:
            logger.error(f"Order submit error: {e}")
        return None

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        session = await self._ensure_session()
        url = f"{self.config.base_url}/v2/orders/{order_id}"
        try:
            async with session.delete(url) as resp:
                return resp.status in (200, 204)
        except aiohttp.ClientError:
            return False

    async def get_open_orders(self) -> List[Dict[str, Any]]:
        """Get all open orders."""
        session = await self._ensure_session()
        url = f"{self.config.base_url}/v2/orders"
        params = {'status': 'open'}
        try:
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
        except aiohttp.ClientError:
            pass
        return []


# =============================================================================
# INTRADAY TRADER
# =============================================================================

class IntradayTrader:
    """
    Intraday trading engine with 15-minute signal generation.
    
    Features:
    - 15-minute signal updates during market hours
    - WebSocket streaming for real-time data
    - Stop-loss and trailing stop management
    - TWAP execution for large orders
    
    Args:
        ml_generator: MLSignalGenerator instance
        universe: MultiAssetUniverse instance
        sizer: AggressivePositionSizer instance
        config: Trading configuration
    
    Example:
        trader = IntradayTrader(generator, universe, sizer)
        await trader.run()
    """

    def __init__(
        self,
        ml_generator: MLSignalGenerator,
        universe: MultiAssetUniverse,
        sizer: AggressivePositionSizer,
        config: Optional[TradingConfig] = None
    ):
        self.ml_generator = ml_generator
        self.universe = universe
        self.sizer = sizer
        self.config = config or TradingConfig()
        
        self.client = AlpacaClient(self.config)
        
        # State
        self._portfolio: Optional[PortfolioState] = None
        self._signals: Dict[str, MLSignal] = {}
        self._trades: List[Trade] = []
        self._daily_trades: int = 0
        self._running = False
        self._shutdown_event = asyncio.Event()
        
        # Price cache for trailing stops
        self._highest_prices: Dict[str, float] = {}

    def _is_market_hours(self) -> bool:
        """Check if currently during market hours."""
        now = datetime.now(EST)
        current_time = now.time()
        
        # Weekends
        if now.weekday() >= 5:
            return False
        
        return self.config.market_open <= current_time <= self.config.market_close

    async def _fetch_portfolio_state(self) -> PortfolioState:
        """Fetch current portfolio state from Alpaca."""
        account = await self.client.get_account()
        positions_data = await self.client.get_positions()
        
        if not account:
            raise RuntimeError("Failed to fetch account data")
        
        cash = float(account.get('cash', 0))
        equity = float(account.get('equity', 0))
        
        positions = {}
        for pos in positions_data:
            symbol = pos['symbol']
            shares = float(pos['qty'])
            avg_cost = float(pos['avg_entry_price'])
            current_price = float(pos['current_price'])
            market_value = float(pos['market_value'])
            unrealized_pnl = float(pos['unrealized_pl'])
            unrealized_pnl_pct = float(pos['unrealized_plpc'])
            
            # Track highest price for trailing stop
            if symbol not in self._highest_prices:
                self._highest_prices[symbol] = current_price
            else:
                self._highest_prices[symbol] = max(
                    self._highest_prices[symbol], current_price
                )
            
            positions[symbol] = Position(
                symbol=symbol,
                shares=shares,
                avg_cost=avg_cost,
                current_price=current_price,
                market_value=market_value,
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_pct=unrealized_pnl_pct,
                entry_time=datetime.now(EST),  # Would need to track separately
                highest_price=self._highest_prices[symbol],
                stop_loss_price=avg_cost * (1 - self.config.stop_loss_pct),
                trailing_stop_price=self._highest_prices[symbol] * (1 - self.config.trailing_stop_pct),
            )
        
        deployment_pct = 1 - (cash / equity) if equity > 0 else 0
        
        return PortfolioState(
            cash=cash,
            equity=equity,
            total_value=equity,
            positions=positions,
            deployment_pct=deployment_pct,
            open_orders=len(await self.client.get_open_orders()),
            daily_trades=self._daily_trades,
            daily_pnl=sum(p.unrealized_pnl for p in positions.values()),
        )

    async def _check_stop_losses(self) -> List[Trade]:
        """Check and execute stop-loss orders."""
        trades = []
        
        if not self._portfolio:
            return trades
        
        for symbol, position in self._portfolio.positions.items():
            triggered = False
            reason = ""
            
            # Check stop-loss
            if position.current_price <= position.stop_loss_price:
                triggered = True
                reason = f"Stop-loss triggered at {position.current_price:.2f}"
            
            # Check trailing stop
            elif position.current_price <= position.trailing_stop_price:
                triggered = True
                reason = f"Trailing stop triggered at {position.current_price:.2f}"
            
            if triggered:
                logger.warning(f"{symbol}: {reason}")
                
                result = await self.client.submit_order(
                    symbol=symbol,
                    shares=int(position.shares),
                    side=OrderSide.SELL,
                    order_type="market"
                )
                
                if result:
                    trade = Trade(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        shares=position.shares,
                        price=position.current_price,
                        value=position.market_value,
                        order_id=result['id'],
                        timestamp=datetime.now(EST),
                        reason=reason,
                    )
                    trades.append(trade)
                    self._daily_trades += 1
        
        return trades

    async def _execute_signals(self) -> List[Trade]:
        """Execute trades based on current signals."""
        trades = []
        
        if not self._portfolio or not self._signals:
            return trades
        
        # Check daily trade limit
        if self._daily_trades >= self.config.max_trades_per_day:
            logger.info("Daily trade limit reached")
            return trades
        
        # Get current prices
        prices, volumes = self.universe.get_prices(lookback_days=5)
        
        # Calculate target allocations
        allocations = self.sizer.calculate_allocations(
            self._signals,
            self._portfolio,
            self.universe,
            prices
        )
        
        for symbol, (target_alloc, action) in allocations.items():
            if action == 'hold':
                continue
            
            # Get current price
            quote = await self.client.get_quote(symbol)
            if not quote:
                continue
            
            current_price = (quote.get('ap', 0) + quote.get('bp', 0)) / 2
            if current_price <= 0:
                continue
            
            # Calculate order size
            current_position = self._portfolio.positions.get(symbol)
            shares, side = self.sizer.calculate_order_size(
                symbol, target_alloc, current_position, current_price, self._portfolio.total_value
            )
            
            if shares <= 0:
                continue
            
            order_value = shares * current_price
            
            # Use TWAP for large orders
            if order_value > self.config.twap_threshold:
                trades.extend(await self._execute_twap(symbol, shares, side, current_price))
            else:
                trade = await self._execute_single_order(symbol, shares, side, current_price)
                if trade:
                    trades.append(trade)
        
        return trades

    async def _execute_twap(
        self, symbol: str, total_shares: int, side: OrderSide, price: float
    ) -> List[Trade]:
        """Execute order using TWAP strategy."""
        trades = []
        slices = self.config.twap_slices
        shares_per_slice = total_shares // slices
        remainder = total_shares % slices
        
        interval = self.config.twap_duration_minutes / slices * 60  # seconds
        
        logger.info(f"TWAP: {side.value} {total_shares} {symbol} in {slices} slices")
        
        for i in range(slices):
            slice_shares = shares_per_slice + (1 if i < remainder else 0)
            if slice_shares <= 0:
                continue
            
            # Get fresh quote
            quote = await self.client.get_quote(symbol)
            if quote:
                if side == OrderSide.BUY:
                    # Cross the spread slightly for urgency
                    limit_price = quote.get('ap', price) * (1 + self.config.limit_order_improvement_bps / 10000)
                else:
                    limit_price = quote.get('bp', price) * (1 - self.config.limit_order_improvement_bps / 10000)
            else:
                limit_price = price
            
            result = await self.client.submit_order(
                symbol=symbol,
                shares=slice_shares,
                side=side,
                order_type="limit",
                limit_price=limit_price
            )
            
            if result:
                trade = Trade(
                    symbol=symbol,
                    side=side,
                    shares=slice_shares,
                    price=limit_price,
                    value=slice_shares * limit_price,
                    order_id=result['id'],
                    timestamp=datetime.now(EST),
                    reason=f"TWAP slice {i+1}/{slices}",
                    signal=self._signals.get(symbol),
                )
                trades.append(trade)
                self._daily_trades += 1
            
            # Wait before next slice
            if i < slices - 1:
                await asyncio.sleep(interval)
        
        return trades

    async def _execute_single_order(
        self, symbol: str, shares: int, side: OrderSide, price: float
    ) -> Optional[Trade]:
        """Execute a single order."""
        # Calculate stop-loss for new positions
        stop_loss = None
        if side == OrderSide.BUY:
            stop_loss = price * (1 - self.config.stop_loss_pct)
        
        # Get limit price with improvement
        quote = await self.client.get_quote(symbol)
        if quote:
            if side == OrderSide.BUY:
                limit_price = quote.get('ap', price) * (1 + self.config.limit_order_improvement_bps / 10000)
            else:
                limit_price = quote.get('bp', price) * (1 - self.config.limit_order_improvement_bps / 10000)
        else:
            limit_price = price
        
        result = await self.client.submit_order(
            symbol=symbol,
            shares=shares,
            side=side,
            order_type="limit",
            limit_price=limit_price,
            stop_loss=stop_loss
        )
        
        if result:
            self._daily_trades += 1
            return Trade(
                symbol=symbol,
                side=side,
                shares=shares,
                price=limit_price,
                value=shares * limit_price,
                order_id=result['id'],
                timestamp=datetime.now(EST),
                reason="ML signal",
                signal=self._signals.get(symbol),
            )
        
        return None

    async def _update_signals(self) -> None:
        """Update ML signals for all assets."""
        logger.info("Updating ML signals...")
        
        # Get recent price data
        prices, volumes = self.universe.get_prices(lookback_days=self.ml_generator.config.rolling_window)
        
        if prices.empty:
            logger.warning("No price data available for signal generation")
            return
        
        # Generate signals
        self._signals = self.ml_generator.generate_signals(prices, volumes)
        
        # Log signal summary
        buy_signals = sum(1 for s in self._signals.values() if s.signal_type in [SignalType.BUY, SignalType.STRONG_BUY])
        sell_signals = sum(1 for s in self._signals.values() if s.signal_type in [SignalType.SELL, SignalType.STRONG_SELL])
        
        logger.info(f"Signals: {buy_signals} buy, {sell_signals} sell, {len(self._signals) - buy_signals - sell_signals} hold")

    def _setup_signal_handlers(self) -> None:
        """Setup graceful shutdown handlers."""
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._handle_shutdown)

    def _handle_shutdown(self) -> None:
        """Handle shutdown signal."""
        logger.info("Shutdown signal received, stopping trader...")
        self._shutdown_event.set()

    async def run(self) -> None:
        """Main trading loop."""
        self._running = True
        self._setup_signal_handlers()
        
        logger.info("=" * 60)
        logger.info("V37 ML Alpha Engine - Intraday Trader Started")
        logger.info("=" * 60)
        
        try:
            while not self._shutdown_event.is_set():
                if self._is_market_hours():
                    try:
                        # Update portfolio state
                        self._portfolio = await self._fetch_portfolio_state()
                        logger.info(
                            f"Portfolio: ${self._portfolio.total_value:,.2f} | "
                            f"Deployed: {self._portfolio.deployment_pct:.1%} | "
                            f"Positions: {len(self._portfolio.positions)} | "
                            f"Trades today: {self._daily_trades}"
                        )
                        
                        # Check stop-losses
                        stop_trades = await self._check_stop_losses()
                        self._trades.extend(stop_trades)
                        
                        # Update signals
                        await self._update_signals()
                        
                        # Execute signals
                        signal_trades = await self._execute_signals()
                        self._trades.extend(signal_trades)
                        
                        if signal_trades:
                            logger.info(f"Executed {len(signal_trades)} trades")
                        
                    except Exception as e:
                        logger.error(f"Trading loop error: {e}")
                    
                    # Wait for next signal interval
                    await asyncio.sleep(self.config.signal_interval_minutes * 60)
                else:
                    # Reset daily counters at start of day
                    now = datetime.now(EST)
                    if now.time() < self.config.market_open:
                        self._daily_trades = 0
                        self._trades = []
                    
                    logger.info("Market closed. Waiting...")
                    await asyncio.sleep(60)  # Check every minute
        
        finally:
            await self.client.close()
            self._running = False
            logger.info("Trader stopped")

    async def run_once(self) -> List[Trade]:
        """Run a single trading iteration (for testing/backtesting)."""
        self._portfolio = await self._fetch_portfolio_state()
        await self._update_signals()
        
        stop_trades = await self._check_stop_losses()
        signal_trades = await self._execute_signals()
        
        all_trades = stop_trades + signal_trades
        self._trades.extend(all_trades)
        
        return all_trades


# =============================================================================
# BACKTESTER
# =============================================================================

class Backtester:
    """
    Simple backtester for the V37 strategy.
    
    Features:
    - Historical simulation with ML signals
    - Transaction cost modeling
    - Performance metrics calculation
    
    Args:
        ml_generator: Trained MLSignalGenerator
        universe: MultiAssetUniverse
        sizer: AggressivePositionSizer
        initial_capital: Starting capital
    """

    def __init__(
        self,
        ml_generator: MLSignalGenerator,
        universe: MultiAssetUniverse,
        sizer: AggressivePositionSizer,
        initial_capital: float = 100_000
    ):
        self.ml_generator = ml_generator
        self.universe = universe
        self.sizer = sizer
        self.initial_capital = initial_capital
        
        # Results
        self.equity_curve: List[float] = []
        self.trades: List[Trade] = []
        self.daily_returns: List[float] = []

    def run(
        self,
        start_date: str,
        end_date: str,
        transaction_cost_bps: float = 5.0
    ) -> Dict[str, float]:
        """
        Run backtest simulation.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            transaction_cost_bps: Transaction cost in basis points
        
        Returns:
            Dictionary of performance metrics
        """
        logger.info(f"Running backtest from {start_date} to {end_date}")
        
        # Get historical data
        prices, volumes = self.universe.get_prices(lookback_days=500)
        
        if prices.empty:
            raise ValueError("No price data available")
        
        # Filter date range
        prices = prices[start_date:end_date]
        volumes = volumes[start_date:end_date]
        
        if len(prices) < 60:
            raise ValueError("Insufficient data for backtest")
        
        # Initialize
        cash = self.initial_capital
        positions: Dict[str, float] = {}  # symbol -> shares
        self.equity_curve = [self.initial_capital]
        prev_equity = self.initial_capital
        
        # Training window
        train_window = self.ml_generator.config.rolling_window
        
        for i in range(train_window, len(prices)):
            date = prices.index[i]
            
            # Get training data
            train_prices = prices.iloc[i-train_window:i]
            train_volumes = volumes.iloc[i-train_window:i]
            
            # Retrain weekly
            if i % 5 == 0:
                try:
                    self.ml_generator.train(train_prices, train_volumes)
                except Exception:
                    continue
            
            # Generate signals
            try:
                signals = self.ml_generator.generate_signals(train_prices, train_volumes)
            except Exception:
                continue
            
            current_prices = prices.iloc[i]
            
            # Calculate current equity
            equity = cash
            for sym, shares in positions.items():
                if sym in current_prices:
                    equity += shares * current_prices[sym]
            
            # Create mock portfolio state
            mock_positions = {}
            for sym, shares in positions.items():
                if sym in current_prices:
                    price = current_prices[sym]
                    mock_positions[sym] = Position(
                        symbol=sym,
                        shares=shares,
                        avg_cost=price,
                        current_price=price,
                        market_value=shares * price,
                        unrealized_pnl=0,
                        unrealized_pnl_pct=0,
                        entry_time=datetime.now(EST),
                        highest_price=price,
                        stop_loss_price=price * 0.98,
                        trailing_stop_price=price * 0.985,
                    )
            
            portfolio = PortfolioState(
                cash=cash,
                equity=equity,
                total_value=equity,
                positions=mock_positions,
                deployment_pct=1 - (cash / equity) if equity > 0 else 0,
                open_orders=0,
                daily_trades=0,
                daily_pnl=0,
            )
            
            # Get allocations
            allocations = self.sizer.calculate_allocations(
                signals, portfolio, self.universe, train_prices
            )
            
            # Execute trades (simplified)
            for symbol, (target_alloc, action) in allocations.items():
                if symbol not in current_prices or pd.isna(current_prices[symbol]):
                    continue
                
                price = current_prices[symbol]
                current_shares = positions.get(symbol, 0)
                target_shares = int(target_alloc / price) if price > 0 else 0
                
                shares_diff = target_shares - current_shares
                
                if shares_diff != 0:
                    trade_value = abs(shares_diff * price)
                    cost = trade_value * (transaction_cost_bps / 10000)
                    
                    if shares_diff > 0 and cash >= trade_value + cost:
                        # Buy
                        cash -= trade_value + cost
                        positions[symbol] = current_shares + shares_diff
                    elif shares_diff < 0:
                        # Sell
                        cash += trade_value - cost
                        positions[symbol] = max(0, current_shares + shares_diff)
                        if positions[symbol] == 0:
                            del positions[symbol]
            
            # Calculate end-of-day equity
            equity = cash
            for sym, shares in positions.items():
                if sym in current_prices:
                    equity += shares * current_prices[sym]
            
            self.equity_curve.append(equity)
            
            daily_return = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0
            self.daily_returns.append(daily_return)
            prev_equity = equity
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        logger.info(f"Backtest complete. Final equity: ${equity:,.2f}")
        return metrics

    def _calculate_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics."""
        returns = np.array(self.daily_returns)
        equity = np.array(self.equity_curve)
        
        if len(returns) == 0:
            return {}
        
        # Total return
        total_return = (equity[-1] / equity[0]) - 1
        
        # Annualized return
        days = len(returns)
        annual_return = ((1 + total_return) ** (252 / max(days, 1))) - 1
        
        # Volatility
        volatility = np.std(returns) * np.sqrt(252)
        
        # Sharpe ratio
        sharpe = annual_return / volatility if volatility > 0 else 0
        
        # Max drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdown = np.max(drawdown)
        
        # Win rate
        positive_days = np.sum(returns > 0)
        win_rate = positive_days / len(returns) if len(returns) > 0 else 0
        
        # Calmar ratio
        calmar = annual_return / max_drawdown if max_drawdown > 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'calmar_ratio': calmar,
            'num_days': days,
            'final_equity': equity[-1],
        }


# =============================================================================
# CLI INTERFACE
# =============================================================================

async def cmd_train(args):
    """Train ML models."""
    logger.info("Training ML models...")
    
    universe = MultiAssetUniverse()
    universe.get_tradeable_assets()
    
    prices, volumes = universe.get_prices(lookback_days=300)
    
    if prices.empty:
        logger.error("No data available for training")
        return
    
    generator = MLSignalGenerator()
    metrics = generator.train(prices, volumes)
    
    print("\n" + "=" * 50)
    print("TRAINING RESULTS")
    print("=" * 50)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    generator.save_models()
    print(f"\nModels saved to {generator.model_dir}")


async def cmd_predict(args):
    """Generate predictions."""
    logger.info("Generating predictions...")
    
    generator = MLSignalGenerator()
    if not generator.load_models():
        logger.error("No trained models found. Run --train first.")
        return
    
    universe = MultiAssetUniverse()
    universe.get_tradeable_assets()
    
    prices, volumes = universe.get_prices(lookback_days=100)
    
    signals = generator.generate_signals(prices, volumes)
    
    print("\n" + "=" * 60)
    print("ML SIGNALS")
    print("=" * 60)
    
    # Sort by probability
    sorted_signals = sorted(
        signals.items(),
        key=lambda x: x[1].probability,
        reverse=True
    )
    
    print("\nTOP BUY SIGNALS:")
    print("-" * 60)
    for symbol, signal in sorted_signals[:10]:
        if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            print(
                f"  {symbol:8} | {signal.signal_type.name:12} | "
                f"Prob: {signal.probability:.1%} | Conf: {signal.confidence:.1%}"
            )
    
    print("\nTOP SELL SIGNALS:")
    print("-" * 60)
    for symbol, signal in reversed(sorted_signals[-10:]):
        if signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
            print(
                f"  {symbol:8} | {signal.signal_type.name:12} | "
                f"Prob: {signal.probability:.1%} | Conf: {signal.confidence:.1%}"
            )


async def cmd_trade(args):
    """Run live trading."""
    logger.info("Starting live trading...")
    
    generator = MLSignalGenerator()
    if not generator.load_models():
        logger.error("No trained models found. Run --train first.")
        return
    
    universe = MultiAssetUniverse()
    sizer = AggressivePositionSizer()
    
    config = TradingConfig()
    if not args.live:
        config.base_url = 'https://paper-api.alpaca.markets'
        logger.info("Running in PAPER trading mode")
    else:
        logger.info("Running in LIVE trading mode")
    
    trader = IntradayTrader(generator, universe, sizer, config)
    await trader.run()


async def cmd_backtest(args):
    """Run backtest."""
    logger.info(f"Running backtest from {args.start} to {args.end}")
    
    generator = MLSignalGenerator()
    universe = MultiAssetUniverse()
    universe.get_tradeable_assets()
    
    sizer = AggressivePositionSizer()
    
    backtester = Backtester(
        generator, universe, sizer,
        initial_capital=100_000
    )
    
    metrics = backtester.run(args.start, args.end)
    
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Period: {args.start} to {args.end}")
    print(f"  Total Return: {metrics.get('total_return', 0):.2%}")
    print(f"  Annual Return: {metrics.get('annual_return', 0):.2%}")
    print(f"  Volatility: {metrics.get('volatility', 0):.2%}")
    print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
    print(f"  Win Rate: {metrics.get('win_rate', 0):.1%}")
    print(f"  Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}")
    print(f"  Final Equity: ${metrics.get('final_equity', 0):,.2f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="V37 ML Multi-Asset Alpha Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train ML models
    python v37_ml_alpha_engine.py --train
    
    # Generate predictions
    python v37_ml_alpha_engine.py --predict
    
    # Paper trading
    python v37_ml_alpha_engine.py --trade
    
    # Live trading
    python v37_ml_alpha_engine.py --trade --live
    
    # Backtest
    python v37_ml_alpha_engine.py --backtest --start 2024-01-01 --end 2024-12-31
        """
    )
    
    parser.add_argument('--train', action='store_true', help='Train ML models')
    parser.add_argument('--predict', action='store_true', help='Generate predictions')
    parser.add_argument('--trade', action='store_true', help='Start trading')
    parser.add_argument('--live', action='store_true', help='Use live trading (default: paper)')
    parser.add_argument('--backtest', action='store_true', help='Run backtest')
    parser.add_argument('--start', type=str, default='2024-01-01', help='Backtest start date')
    parser.add_argument('--end', type=str, default='2024-12-31', help='Backtest end date')
    
    args = parser.parse_args()
    
    if args.train:
        asyncio.run(cmd_train(args))
    elif args.predict:
        asyncio.run(cmd_predict(args))
    elif args.trade:
        asyncio.run(cmd_trade(args))
    elif args.backtest:
        asyncio.run(cmd_backtest(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
