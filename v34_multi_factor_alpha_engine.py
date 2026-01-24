#!/usr/bin/env python3
"""
v34_multi_factor_alpha_engine.py - Multi-Factor Quantitative Trading Engine

A high-performance trading system combining:
1. Multi-Factor Model (Momentum, Value, Quality, Low Vol, Mean Reversion)
2. Cointegration-Based Pairs Trading
3. Regime Detection (Bull/Bear/Volatile)
4. Kelly Criterion Position Sizing

Target: 30%+ annual return, Sharpe > 1.5

Author: Quantitative Trading System
Date: January 2026
"""

import argparse
import logging
import os
import sys
import json
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Suppress warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """System configuration parameters."""
    # Universe
    UNIVERSE: List[str] = field(default_factory=lambda: [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'BAC', 'WFC',
        'V', 'MA', 'JNJ', 'PFE', 'UNH', 'XOM', 'CVX', 'COP', 'HD', 'LOW',
        'KO', 'PEP', 'PG', 'WMT', 'COST', 'DIS', 'NFLX', 'CMCSA', 'T', 'VZ'
    ])
    
    # Pairs for cointegration trading
    PAIRS: List[Tuple[str, str]] = field(default_factory=lambda: [
        ('XOM', 'CVX'), ('JPM', 'BAC'), ('KO', 'PEP'), ('V', 'MA'), ('HD', 'LOW')
    ])
    
    # Position limits
    MAX_POSITION_PCT: float = 0.08  # 8% max per position
    MAX_POSITIONS: int = 20
    HALF_KELLY: float = 0.5  # Half-Kelly for safety
    
    # Technical parameters
    MOMENTUM_LONG: int = 252  # 12 months
    MOMENTUM_SHORT: int = 21  # 1 month
    VOLATILITY_WINDOW: int = 60
    RSI_PERIOD: int = 14
    BB_PERIOD: int = 20
    BB_STD: float = 2.0
    ATR_PERIOD: int = 14
    ATR_MULTIPLIER: float = 2.0
    
    # Regime thresholds
    SMA_PERIOD: int = 200
    VIX_BULL: float = 20.0
    VIX_BEAR: float = 25.0
    VIX_VOLATILE: float = 30.0
    
    # Pairs trading
    ZSCORE_ENTRY: float = 2.0
    ZSCORE_EXIT: float = 0.5
    COINT_PVALUE: float = 0.05
    LOOKBACK_COINT: int = 252
    
    # API
    ALPACA_KEY: str = os.environ.get('APCA_API_KEY_ID', '')
    ALPACA_SECRET: str = os.environ.get('APCA_API_SECRET_KEY', '')
    ALPACA_BASE_URL: str = os.environ.get('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure logging with both file and console handlers."""
    logger = logging.getLogger('v34_alpha_engine')
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_fmt = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', '%H:%M:%S')
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)
    
    # File handler
    os.makedirs('logs', exist_ok=True)
    file_handler = logging.FileHandler(f'logs/v34_engine_{datetime.now():%Y%m%d}.log')
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter('%(asctime)s | %(levelname)-8s | %(funcName)s | %(message)s')
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)
    
    return logger


logger = setup_logging()


# =============================================================================
# DATA FETCHER
# =============================================================================

class DataFetcher:
    """Fetches and caches market data from yfinance."""
    
    def __init__(self, config: Config):
        self.config = config
        self.cache: Dict[str, pd.DataFrame] = {}
        self.info_cache: Dict[str, Dict] = {}
    
    def get_prices(self, symbols: List[str], period: str = '2y', 
                   interval: str = '1d') -> pd.DataFrame:
        """
        Fetch OHLCV data for multiple symbols.
        
        Args:
            symbols: List of ticker symbols
            period: Data period (e.g., '1y', '2y')
            interval: Data interval (e.g., '1d', '1h')
            
        Returns:
            DataFrame with MultiIndex columns (symbol, OHLCV)
        """
        cache_key = f"{','.join(sorted(symbols))}_{period}_{interval}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        logger.info(f"Fetching price data for {len(symbols)} symbols...")
        
        try:
            data = yf.download(
                symbols, 
                period=period, 
                interval=interval, 
                progress=False,
                group_by='ticker',
                auto_adjust=True
            )
            
            if data.empty:
                logger.error("No data returned from yfinance")
                return pd.DataFrame()
            
            self.cache[cache_key] = data
            logger.info(f"Fetched {len(data)} rows of price data")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching prices: {e}")
            return pd.DataFrame()
    
    def get_single_ticker(self, symbol: str, period: str = '2y') -> pd.DataFrame:
        """Fetch data for a single ticker."""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, auto_adjust=True)
            return data
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()
    
    def get_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch fundamental data for a symbol.
        
        Returns dict with: P/E, P/B, ROE, debt_to_equity
        """
        if symbol in self.info_cache:
            return self.info_cache[symbol]
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            fundamentals = {
                'pe_ratio': info.get('forwardPE') or info.get('trailingPE'),
                'pb_ratio': info.get('priceToBook'),
                'roe': info.get('returnOnEquity'),
                'debt_to_equity': info.get('debtToEquity'),
                'market_cap': info.get('marketCap'),
                'sector': info.get('sector', 'Unknown')
            }
            
            self.info_cache[symbol] = fundamentals
            return fundamentals
            
        except Exception as e:
            logger.warning(f"Error fetching fundamentals for {symbol}: {e}")
            return {}
    
    def get_vix(self, period: str = '2y') -> pd.Series:
        """Fetch VIX data."""
        try:
            vix = yf.Ticker('^VIX')
            data = vix.history(period=period, auto_adjust=True)
            return data['Close']
        except Exception as e:
            logger.error(f"Error fetching VIX: {e}")
            return pd.Series()
    
    def get_spy(self, period: str = '2y') -> pd.DataFrame:
        """Fetch SPY data for regime detection."""
        return self.get_single_ticker('SPY', period)


# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

class TechnicalIndicators:
    """Calculate technical indicators for trading signals."""
    
    @staticmethod
    def momentum(prices: pd.Series, long_period: int = 252, 
                 short_period: int = 21) -> float:
        """
        Calculate 12-1 momentum (12-month return minus 1-month return).
        
        This avoids short-term reversal while capturing medium-term momentum.
        """
        if len(prices) < long_period:
            return np.nan
        
        long_return = (prices.iloc[-1] / prices.iloc[-long_period]) - 1
        short_return = (prices.iloc[-1] / prices.iloc[-short_period]) - 1
        
        return long_return - short_return
    
    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return 50.0
        
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        
        rs = gain / loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
    
    @staticmethod
    def bollinger_bands(prices: pd.Series, period: int = 20, 
                        std_dev: float = 2.0) -> Tuple[float, float, float]:
        """
        Calculate Bollinger Bands.
        
        Returns: (lower, middle, upper)
        """
        if len(prices) < period:
            return (np.nan, np.nan, np.nan)
        
        middle = prices.rolling(period).mean().iloc[-1]
        std = prices.rolling(period).std().iloc[-1]
        
        return (middle - std_dev * std, middle, middle + std_dev * std)
    
    @staticmethod
    def bb_deviation(prices: pd.Series, period: int = 20, 
                     std_dev: float = 2.0) -> float:
        """
        Calculate normalized deviation from Bollinger middle band.
        
        Returns value in [-1, 1] range, where:
        -1 = at lower band, +1 = at upper band
        """
        lower, middle, upper = TechnicalIndicators.bollinger_bands(prices, period, std_dev)
        
        if pd.isna(middle) or upper == lower:
            return 0.0
        
        current = prices.iloc[-1]
        return (current - middle) / ((upper - lower) / 2)
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, 
            period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(close) < period + 1:
            return np.nan
        
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        
        return tr.rolling(period).mean().iloc[-1]
    
    @staticmethod
    def volatility(returns: pd.Series, period: int = 60) -> float:
        """Calculate annualized volatility."""
        if len(returns) < period:
            return np.nan
        
        return returns.iloc[-period:].std() * np.sqrt(252)
    
    @staticmethod
    def sma(prices: pd.Series, period: int) -> float:
        """Simple Moving Average."""
        if len(prices) < period:
            return np.nan
        return prices.rolling(period).mean().iloc[-1]


# =============================================================================
# MULTI-FACTOR MODEL
# =============================================================================

class MultiFactorModel:
    """
    Combines multiple alpha factors into a composite score.
    
    Factors:
    1. Momentum (12-1 month)
    2. Value (inverse P/E, inverse P/B)
    3. Quality (ROE, inverse debt/equity)
    4. Low Volatility (inverse 60-day vol)
    5. Mean Reversion (RSI extremes, BB deviation)
    """
    
    def __init__(self, config: Config, data_fetcher: DataFetcher):
        self.config = config
        self.fetcher = data_fetcher
        self.scaler = StandardScaler()
        self.factor_weights = {
            'momentum': 0.25,
            'value': 0.20,
            'quality': 0.20,
            'low_vol': 0.15,
            'mean_reversion': 0.20
        }
    
    def calculate_momentum_factor(self, prices_df: pd.DataFrame, 
                                   symbol: str) -> float:
        """Calculate momentum factor for a symbol."""
        try:
            if symbol not in prices_df.columns.get_level_values(0):
                return np.nan
            
            prices = prices_df[symbol]['Close'].dropna()
            return TechnicalIndicators.momentum(
                prices, 
                self.config.MOMENTUM_LONG, 
                self.config.MOMENTUM_SHORT
            )
        except Exception:
            return np.nan
    
    def calculate_value_factor(self, symbol: str) -> float:
        """
        Calculate value factor from fundamentals.
        
        Uses inverse P/E and inverse P/B (lower = better value).
        """
        fundamentals = self.fetcher.get_fundamentals(symbol)
        
        pe = fundamentals.get('pe_ratio')
        pb = fundamentals.get('pb_ratio')
        
        scores = []
        if pe and pe > 0:
            scores.append(1 / pe)  # Inverse P/E
        if pb and pb > 0:
            scores.append(1 / pb)  # Inverse P/B
        
        return np.mean(scores) if scores else np.nan
    
    def calculate_quality_factor(self, symbol: str) -> float:
        """
        Calculate quality factor.
        
        Uses ROE (higher = better) and inverse debt/equity (lower debt = better).
        """
        fundamentals = self.fetcher.get_fundamentals(symbol)
        
        roe = fundamentals.get('roe')
        debt_equity = fundamentals.get('debt_to_equity')
        
        scores = []
        if roe is not None:
            scores.append(roe)
        if debt_equity is not None and debt_equity > 0:
            scores.append(1 / (1 + debt_equity))  # Inverse, bounded
        
        return np.mean(scores) if scores else np.nan
    
    def calculate_low_vol_factor(self, prices_df: pd.DataFrame, 
                                  symbol: str) -> float:
        """
        Calculate low volatility factor.
        
        Uses inverse of 60-day volatility (lower vol = higher score).
        """
        try:
            if symbol not in prices_df.columns.get_level_values(0):
                return np.nan
            
            prices = prices_df[symbol]['Close'].dropna()
            returns = prices.pct_change().dropna()
            
            vol = TechnicalIndicators.volatility(returns, self.config.VOLATILITY_WINDOW)
            
            if pd.isna(vol) or vol <= 0:
                return np.nan
            
            return 1 / vol  # Inverse volatility
            
        except Exception:
            return np.nan
    
    def calculate_mean_reversion_factor(self, prices_df: pd.DataFrame, 
                                         symbol: str) -> float:
        """
        Calculate mean reversion factor.
        
        Combines RSI extremes and Bollinger Band deviation.
        Oversold = positive score (expect reversion up)
        """
        try:
            if symbol not in prices_df.columns.get_level_values(0):
                return np.nan
            
            prices = prices_df[symbol]['Close'].dropna()
            
            # RSI-based score: oversold (RSI < 30) = positive, overbought (RSI > 70) = negative
            rsi = TechnicalIndicators.rsi(prices, self.config.RSI_PERIOD)
            rsi_score = (50 - rsi) / 50  # Normalized: -1 to 1
            
            # BB deviation: below middle = positive (oversold)
            bb_dev = TechnicalIndicators.bb_deviation(
                prices, self.config.BB_PERIOD, self.config.BB_STD
            )
            bb_score = -bb_dev  # Inverse: oversold = positive
            
            return (rsi_score + bb_score) / 2
            
        except Exception:
            return np.nan
    
    def calculate_composite_scores(self, symbols: List[str]) -> pd.DataFrame:
        """
        Calculate composite factor scores for all symbols.
        
        Returns DataFrame with factor scores and composite ranking.
        """
        logger.info("Calculating multi-factor scores...")
        
        # Fetch price data
        prices_df = self.fetcher.get_prices(symbols)
        
        if prices_df.empty:
            logger.error("No price data available")
            return pd.DataFrame()
        
        # Calculate raw factor scores
        factor_data = []
        
        for symbol in symbols:
            row = {
                'symbol': symbol,
                'momentum': self.calculate_momentum_factor(prices_df, symbol),
                'value': self.calculate_value_factor(symbol),
                'quality': self.calculate_quality_factor(symbol),
                'low_vol': self.calculate_low_vol_factor(prices_df, symbol),
                'mean_reversion': self.calculate_mean_reversion_factor(prices_df, symbol)
            }
            factor_data.append(row)
        
        df = pd.DataFrame(factor_data).set_index('symbol')
        
        # Z-score normalize each factor
        for col in ['momentum', 'value', 'quality', 'low_vol', 'mean_reversion']:
            valid = df[col].dropna()
            if len(valid) > 1:
                mean, std = valid.mean(), valid.std()
                if std > 0:
                    df[f'{col}_z'] = (df[col] - mean) / std
                else:
                    df[f'{col}_z'] = 0
            else:
                df[f'{col}_z'] = 0
        
        # Calculate weighted composite score
        df['composite'] = sum(
            df[f'{factor}_z'].fillna(0) * weight
            for factor, weight in self.factor_weights.items()
        )
        
        # Rank by composite score
        df['rank'] = df['composite'].rank(ascending=False)
        
        logger.info(f"Factor scores calculated for {len(df)} symbols")
        return df.sort_values('composite', ascending=False)


# =============================================================================
# COINTEGRATION PAIRS TRADING
# =============================================================================

class PairsTrader:
    """
    Cointegration-based pairs trading strategy.
    
    Uses ADF test to find cointegrated pairs, OLS for hedge ratios,
    and z-score for entry/exit signals.
    """
    
    def __init__(self, config: Config, data_fetcher: DataFetcher):
        self.config = config
        self.fetcher = data_fetcher
        self.hedge_ratios: Dict[Tuple[str, str], float] = {}
        self.spread_stats: Dict[Tuple[str, str], Tuple[float, float]] = {}
    
    def adf_test(self, series: pd.Series) -> Tuple[float, float]:
        """
        Perform Augmented Dickey-Fuller test for stationarity.
        
        Returns: (test_statistic, p_value)
        """
        from scipy.stats import norm
        
        # Simple ADF implementation
        n = len(series)
        if n < 20:
            return (0, 1.0)
        
        # First difference
        diff = series.diff().dropna()
        
        # Lag of original series
        lag = series.shift(1).dropna()
        
        # Align
        min_len = min(len(diff), len(lag))
        diff = diff.iloc[-min_len:]
        lag = lag.iloc[-min_len:]
        
        # OLS regression: diff = alpha + beta * lag + epsilon
        X = np.column_stack([np.ones(len(lag)), lag.values])
        y = diff.values
        
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            residuals = y - X @ beta
            se = np.sqrt(np.sum(residuals**2) / (len(y) - 2) / np.sum((lag - lag.mean())**2))
            t_stat = beta[1] / se
            
            # Approximate p-value (simplified)
            # Critical values: -3.43 (1%), -2.86 (5%), -2.57 (10%)
            if t_stat < -3.43:
                p_value = 0.01
            elif t_stat < -2.86:
                p_value = 0.05
            elif t_stat < -2.57:
                p_value = 0.10
            else:
                p_value = 0.5
            
            return (t_stat, p_value)
            
        except Exception:
            return (0, 1.0)
    
    def calculate_hedge_ratio(self, y: pd.Series, x: pd.Series) -> float:
        """Calculate hedge ratio using OLS regression."""
        if len(y) != len(x) or len(y) < 30:
            return 1.0
        
        model = LinearRegression()
        model.fit(x.values.reshape(-1, 1), y.values)
        
        return model.coef_[0]
    
    def calculate_spread(self, y: pd.Series, x: pd.Series, 
                         hedge_ratio: float) -> pd.Series:
        """Calculate the spread between two assets."""
        return y - hedge_ratio * x
    
    def is_cointegrated(self, pair: Tuple[str, str], 
                        prices_df: pd.DataFrame) -> bool:
        """
        Test if a pair is cointegrated.
        
        Returns True if ADF test p-value < threshold.
        """
        sym1, sym2 = pair
        
        try:
            y = prices_df[sym1]['Close'].dropna()
            x = prices_df[sym2]['Close'].dropna()
            
            # Align series
            common_idx = y.index.intersection(x.index)
            y = y.loc[common_idx].iloc[-self.config.LOOKBACK_COINT:]
            x = x.loc[common_idx].iloc[-self.config.LOOKBACK_COINT:]
            
            if len(y) < 100:
                return False
            
            # Calculate hedge ratio and spread
            hedge_ratio = self.calculate_hedge_ratio(y, x)
            spread = self.calculate_spread(y, x, hedge_ratio)
            
            # ADF test on spread
            _, p_value = self.adf_test(spread)
            
            if p_value < self.config.COINT_PVALUE:
                self.hedge_ratios[pair] = hedge_ratio
                self.spread_stats[pair] = (spread.mean(), spread.std())
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error testing cointegration for {pair}: {e}")
            return False
    
    def get_zscore(self, pair: Tuple[str, str], 
                   prices_df: pd.DataFrame) -> Optional[float]:
        """Calculate current z-score for a pair."""
        if pair not in self.hedge_ratios:
            return None
        
        sym1, sym2 = pair
        
        try:
            y = prices_df[sym1]['Close'].iloc[-1]
            x = prices_df[sym2]['Close'].iloc[-1]
            
            hedge_ratio = self.hedge_ratios[pair]
            spread = y - hedge_ratio * x
            
            mean, std = self.spread_stats[pair]
            
            if std == 0:
                return 0.0
            
            return (spread - mean) / std
            
        except Exception:
            return None
    
    def generate_signals(self, prices_df: pd.DataFrame) -> Dict[Tuple[str, str], Dict]:
        """
        Generate trading signals for all pairs.
        
        Returns dict with pair -> signal info:
        - action: 'long_spread', 'short_spread', 'close', 'none'
        - zscore: current z-score
        - hedge_ratio: hedge ratio
        """
        signals = {}
        
        for pair in self.config.PAIRS:
            # Test cointegration
            if pair not in self.hedge_ratios:
                if not self.is_cointegrated(pair, prices_df):
                    signals[pair] = {'action': 'none', 'reason': 'not_cointegrated'}
                    continue
            
            zscore = self.get_zscore(pair, prices_df)
            
            if zscore is None:
                signals[pair] = {'action': 'none', 'reason': 'no_zscore'}
                continue
            
            signal = {
                'zscore': zscore,
                'hedge_ratio': self.hedge_ratios[pair],
                'spread_mean': self.spread_stats[pair][0],
                'spread_std': self.spread_stats[pair][1]
            }
            
            # Entry signals
            if zscore > self.config.ZSCORE_ENTRY:
                signal['action'] = 'short_spread'  # Short sym1, long sym2
                signal['reason'] = f'zscore={zscore:.2f} > {self.config.ZSCORE_ENTRY}'
            elif zscore < -self.config.ZSCORE_ENTRY:
                signal['action'] = 'long_spread'  # Long sym1, short sym2
                signal['reason'] = f'zscore={zscore:.2f} < -{self.config.ZSCORE_ENTRY}'
            # Exit signals
            elif abs(zscore) < self.config.ZSCORE_EXIT:
                signal['action'] = 'close'
                signal['reason'] = f'zscore={zscore:.2f} near mean'
            else:
                signal['action'] = 'hold'
                signal['reason'] = f'zscore={zscore:.2f} in dead zone'
            
            signals[pair] = signal
        
        return signals


# =============================================================================
# REGIME DETECTOR
# =============================================================================

class RegimeDetector:
    """
    Detects market regime based on SPY trend and VIX levels.
    
    Regimes:
    - BULL: SPY > 200 SMA, VIX < 20
    - BEAR: SPY < 200 SMA, VIX > 25
    - VOLATILE: VIX > 30 (overrides trend)
    - NEUTRAL: Everything else
    """
    
    def __init__(self, config: Config, data_fetcher: DataFetcher):
        self.config = config
        self.fetcher = data_fetcher
        self.current_regime: str = 'NEUTRAL'
        self.regime_history: List[Tuple[datetime, str]] = []
    
    def detect_regime(self) -> str:
        """
        Detect current market regime.
        
        Returns: 'BULL', 'BEAR', 'VOLATILE', or 'NEUTRAL'
        """
        try:
            # Fetch SPY data
            spy_data = self.fetcher.get_spy()
            if spy_data.empty:
                logger.warning("Could not fetch SPY data, defaulting to NEUTRAL")
                return 'NEUTRAL'
            
            spy_close = spy_data['Close']
            spy_current = spy_close.iloc[-1]
            spy_sma = TechnicalIndicators.sma(spy_close, self.config.SMA_PERIOD)
            
            # Fetch VIX data
            vix_data = self.fetcher.get_vix()
            if vix_data.empty:
                logger.warning("Could not fetch VIX data, defaulting to NEUTRAL")
                return 'NEUTRAL'
            
            vix_current = vix_data.iloc[-1]
            
            logger.info(f"SPY: {spy_current:.2f}, 200SMA: {spy_sma:.2f}, VIX: {vix_current:.2f}")
            
            # Determine regime
            if vix_current > self.config.VIX_VOLATILE:
                regime = 'VOLATILE'
            elif spy_current > spy_sma and vix_current < self.config.VIX_BULL:
                regime = 'BULL'
            elif spy_current < spy_sma and vix_current > self.config.VIX_BEAR:
                regime = 'BEAR'
            else:
                regime = 'NEUTRAL'
            
            self.current_regime = regime
            self.regime_history.append((datetime.now(), regime))
            
            return regime
            
        except Exception as e:
            logger.error(f"Error detecting regime: {e}")
            return 'NEUTRAL'
    
    def get_position_multiplier(self) -> float:
        """
        Get position size multiplier based on regime.
        
        BULL: 1.0 (full size)
        NEUTRAL: 0.75
        BEAR: 0.5
        VOLATILE: 0.25
        """
        multipliers = {
            'BULL': 1.0,
            'NEUTRAL': 0.75,
            'BEAR': 0.5,
            'VOLATILE': 0.25
        }
        return multipliers.get(self.current_regime, 0.75)


# =============================================================================
# POSITION SIZER
# =============================================================================

class PositionSizer:
    """
    Calculates optimal position sizes using Kelly Criterion.
    
    Features:
    - Half-Kelly for safety
    - ATR-based stop losses
    - Maximum position limits
    """
    
    def __init__(self, config: Config):
        self.config = config
    
    def kelly_fraction(self, win_rate: float, win_loss_ratio: float) -> float:
        """
        Calculate Kelly fraction: f* = (p * b - q) / b
        
        Where:
        - p = win rate
        - q = 1 - p (loss rate)
        - b = win/loss ratio
        
        Returns fraction of capital to risk (0 to 1).
        """
        if win_loss_ratio <= 0:
            return 0.0
        
        q = 1 - win_rate
        kelly = (win_rate * win_loss_ratio - q) / win_loss_ratio
        
        # Apply half-Kelly and cap
        kelly = max(0, kelly) * self.config.HALF_KELLY
        kelly = min(kelly, self.config.MAX_POSITION_PCT)
        
        return kelly
    
    def calculate_position_size(self, 
                                capital: float,
                                price: float,
                                atr: float,
                                win_rate: float = 0.55,
                                win_loss_ratio: float = 1.5,
                                regime_multiplier: float = 1.0) -> Dict[str, float]:
        """
        Calculate position size with Kelly Criterion and ATR stop.
        
        Args:
            capital: Available capital
            price: Current asset price
            atr: Average True Range
            win_rate: Estimated win rate
            win_loss_ratio: Estimated win/loss ratio
            regime_multiplier: Regime-based adjustment
            
        Returns:
            Dict with shares, position_value, stop_loss, risk_amount
        """
        # Kelly fraction
        kelly = self.kelly_fraction(win_rate, win_loss_ratio)
        
        # Apply regime adjustment
        kelly *= regime_multiplier
        
        # Position value
        position_value = capital * kelly
        position_value = min(position_value, capital * self.config.MAX_POSITION_PCT)
        
        # Calculate shares
        shares = int(position_value / price)
        
        if shares == 0:
            return {
                'shares': 0,
                'position_value': 0,
                'stop_loss': 0,
                'risk_amount': 0
            }
        
        actual_value = shares * price
        
        # ATR-based stop loss
        stop_loss = price - (atr * self.config.ATR_MULTIPLIER)
        risk_per_share = price - stop_loss
        risk_amount = shares * risk_per_share
        
        return {
            'shares': shares,
            'position_value': actual_value,
            'stop_loss': stop_loss,
            'risk_amount': risk_amount,
            'kelly_fraction': kelly
        }


# =============================================================================
# ALPACA BROKER INTERFACE
# =============================================================================

class AlpacaBroker:
    """Interface to Alpaca Trading API."""
    
    def __init__(self, config: Config):
        self.config = config
        self.api = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Alpaca API connection."""
        if not self.config.ALPACA_KEY or not self.config.ALPACA_SECRET:
            logger.warning("Alpaca API credentials not found in environment")
            return
        
        try:
            import alpaca_trade_api as tradeapi
            self.api = tradeapi.REST(
                self.config.ALPACA_KEY,
                self.config.ALPACA_SECRET,
                self.config.ALPACA_BASE_URL,
                api_version='v2'
            )
            logger.info("Alpaca API initialized successfully")
        except ImportError:
            logger.error("alpaca-trade-api package not installed")
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca API: {e}")
    
    def get_account(self) -> Optional[Dict]:
        """Get account information."""
        if not self.api:
            return None
        
        try:
            account = self.api.get_account()
            return {
                'equity': float(account.equity),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value),
                'day_trade_count': account.daytrade_count,
                'status': account.status
            }
        except Exception as e:
            logger.error(f"Error getting account: {e}")
            return None
    
    def get_positions(self) -> List[Dict]:
        """Get all current positions."""
        if not self.api:
            return []
        
        try:
            positions = self.api.list_positions()
            return [
                {
                    'symbol': p.symbol,
                    'qty': int(p.qty),
                    'side': 'long' if int(p.qty) > 0 else 'short',
                    'market_value': float(p.market_value),
                    'cost_basis': float(p.cost_basis),
                    'unrealized_pl': float(p.unrealized_pl),
                    'unrealized_plpc': float(p.unrealized_plpc),
                    'current_price': float(p.current_price)
                }
                for p in positions
            ]
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def submit_order(self, symbol: str, qty: int, side: str, 
                     order_type: str = 'market', 
                     stop_price: Optional[float] = None) -> Optional[Dict]:
        """Submit an order."""
        if not self.api:
            logger.error("Alpaca API not initialized")
            return None
        
        try:
            order_params = {
                'symbol': symbol,
                'qty': abs(qty),
                'side': side,
                'type': order_type,
                'time_in_force': 'day'
            }
            
            if stop_price and order_type == 'stop':
                order_params['stop_price'] = stop_price
            
            order = self.api.submit_order(**order_params)
            
            logger.info(f"Order submitted: {side} {qty} {symbol}")
            
            return {
                'id': order.id,
                'symbol': order.symbol,
                'qty': order.qty,
                'side': order.side,
                'status': order.status
            }
            
        except Exception as e:
            logger.error(f"Error submitting order: {e}")
            return None
    
    def close_position(self, symbol: str) -> bool:
        """Close a position."""
        if not self.api:
            return False
        
        try:
            self.api.close_position(symbol)
            logger.info(f"Closed position: {symbol}")
            return True
        except Exception as e:
            logger.error(f"Error closing position {symbol}: {e}")
            return False


# =============================================================================
# BACKTESTER
# =============================================================================

class Backtester:
    """Simple backtesting engine for strategy validation."""
    
    def __init__(self, config: Config):
        self.config = config
        self.fetcher = DataFetcher(config)
        self.factor_model = MultiFactorModel(config, self.fetcher)
        self.pairs_trader = PairsTrader(config, self.fetcher)
        self.regime_detector = RegimeDetector(config, self.fetcher)
        self.position_sizer = PositionSizer(config)
    
    def run(self, start_date: str = '2024-01-01', 
            initial_capital: float = 100000) -> Dict:
        """
        Run backtest on historical data.
        
        Returns dict with performance metrics.
        """
        logger.info(f"Starting backtest from {start_date} with ${initial_capital:,.0f}")
        
        # Fetch historical data
        all_symbols = list(set(
            self.config.UNIVERSE + 
            [s for pair in self.config.PAIRS for s in pair]
        ))
        
        prices_df = self.fetcher.get_prices(all_symbols, period='2y')
        
        if prices_df.empty:
            logger.error("No price data for backtest")
            return {}
        
        # Get multi-factor scores
        factor_scores = self.factor_model.calculate_composite_scores(self.config.UNIVERSE)
        
        if factor_scores.empty:
            logger.error("No factor scores calculated")
            return {}
        
        # Select top N stocks by composite score
        top_n = min(10, len(factor_scores))
        selected = factor_scores.head(top_n).index.tolist()
        
        logger.info(f"Selected stocks: {selected}")
        
        # Simulate performance
        results = self._simulate_portfolio(prices_df, selected, initial_capital)
        
        return results
    
    def _simulate_portfolio(self, prices_df: pd.DataFrame, 
                            symbols: List[str], 
                            initial_capital: float) -> Dict:
        """Simulate portfolio performance."""
        
        # Get common date range
        returns_data = {}
        for sym in symbols:
            try:
                if sym in prices_df.columns.get_level_values(0):
                    prices = prices_df[sym]['Close'].dropna()
                    returns_data[sym] = prices.pct_change().dropna()
            except Exception:
                continue
        
        if not returns_data:
            return {}
        
        # Align all returns
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()
        
        if len(returns_df) < 30:
            logger.warning("Insufficient data for backtest")
            return {}
        
        # Equal weight portfolio
        n_stocks = len(returns_df.columns)
        weights = np.ones(n_stocks) / n_stocks
        
        # Calculate portfolio returns
        portfolio_returns = (returns_df * weights).sum(axis=1)
        
        # Calculate metrics
        total_return = (1 + portfolio_returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe = annual_return / volatility if volatility > 0 else 0
        
        # Max drawdown
        cum_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cum_returns.cummax()
        drawdown = (cum_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Final value
        final_value = initial_capital * (1 + total_return)
        
        results = {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'n_stocks': n_stocks,
            'n_days': len(portfolio_returns)
        }
        
        logger.info("=" * 60)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"Initial Capital:  ${initial_capital:,.0f}")
        logger.info(f"Final Value:      ${final_value:,.0f}")
        logger.info(f"Total Return:     {total_return:.1%}")
        logger.info(f"Annual Return:    {annual_return:.1%}")
        logger.info(f"Volatility:       {volatility:.1%}")
        logger.info(f"Sharpe Ratio:     {sharpe:.2f}")
        logger.info(f"Max Drawdown:     {max_drawdown:.1%}")
        logger.info("=" * 60)
        
        return results


# =============================================================================
# MAIN ENGINE
# =============================================================================

class MultiFactorAlphaEngine:
    """
    Main trading engine orchestrating all components.
    
    Combines:
    - Multi-factor stock selection
    - Pairs trading
    - Regime detection
    - Position sizing
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.fetcher = DataFetcher(self.config)
        self.factor_model = MultiFactorModel(self.config, self.fetcher)
        self.pairs_trader = PairsTrader(self.config, self.fetcher)
        self.regime_detector = RegimeDetector(self.config, self.fetcher)
        self.position_sizer = PositionSizer(self.config)
        self.broker = AlpacaBroker(self.config)
    
    def get_status(self) -> Dict:
        """Get current system status."""
        logger.info("=" * 60)
        logger.info("V34 MULTI-FACTOR ALPHA ENGINE - STATUS")
        logger.info("=" * 60)
        
        status = {
            'timestamp': datetime.now().isoformat(),
            'regime': 'UNKNOWN',
            'account': None,
            'positions': [],
            'signals': {},
            'top_stocks': []
        }
        
        # Account status
        account = self.broker.get_account()
        if account:
            status['account'] = account
            logger.info(f"Portfolio Value: ${account['portfolio_value']:,.2f}")
            logger.info(f"Cash:            ${account['cash']:,.2f}")
            logger.info(f"Buying Power:    ${account['buying_power']:,.2f}")
        else:
            logger.warning("Could not fetch account status")
        
        # Current positions
        positions = self.broker.get_positions()
        status['positions'] = positions
        logger.info(f"Open Positions:  {len(positions)}")
        
        for pos in positions:
            logger.info(f"  {pos['symbol']:5s}: {pos['qty']:+4d} shares, "
                       f"P&L: ${pos['unrealized_pl']:+,.2f} ({pos['unrealized_plpc']:+.1%})")
        
        # Regime
        regime = self.regime_detector.detect_regime()
        status['regime'] = regime
        logger.info(f"Market Regime:   {regime}")
        logger.info(f"Position Mult:   {self.regime_detector.get_position_multiplier():.0%}")
        
        # Top factor stocks
        try:
            factor_scores = self.factor_model.calculate_composite_scores(self.config.UNIVERSE)
            top_5 = factor_scores.head(5)
            status['top_stocks'] = top_5.index.tolist()
            
            logger.info("\nTop 5 Stocks by Factor Score:")
            for sym in top_5.index:
                score = factor_scores.loc[sym, 'composite']
                logger.info(f"  {sym:5s}: score={score:+.3f}")
        except Exception as e:
            logger.warning(f"Could not calculate factor scores: {e}")
        
        # Pairs signals
        try:
            all_pair_symbols = list(set(s for pair in self.config.PAIRS for s in pair))
            prices_df = self.fetcher.get_prices(all_pair_symbols)
            
            if not prices_df.empty:
                pairs_signals = self.pairs_trader.generate_signals(prices_df)
                status['signals']['pairs'] = pairs_signals
                
                logger.info("\nPairs Trading Signals:")
                for pair, signal in pairs_signals.items():
                    if signal.get('action') != 'none':
                        logger.info(f"  {pair[0]}/{pair[1]}: {signal.get('action', 'N/A')} "
                                   f"(z={signal.get('zscore', 0):.2f})")
        except Exception as e:
            logger.warning(f"Could not generate pairs signals: {e}")
        
        logger.info("=" * 60)
        
        return status
    
    def run(self, dry_run: bool = True) -> Dict:
        """
        Execute trading logic.
        
        Args:
            dry_run: If True, don't submit actual orders
            
        Returns:
            Dict with actions taken
        """
        logger.info("=" * 60)
        logger.info("V34 MULTI-FACTOR ALPHA ENGINE - RUN")
        logger.info(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
        logger.info("=" * 60)
        
        actions = {
            'timestamp': datetime.now().isoformat(),
            'mode': 'dry_run' if dry_run else 'live',
            'orders': [],
            'regime': None
        }
        
        # Check account
        account = self.broker.get_account()
        if not account:
            logger.error("Cannot run without account access")
            return actions
        
        capital = account['portfolio_value']
        cash = account['cash']
        
        # Get regime
        regime = self.regime_detector.detect_regime()
        actions['regime'] = regime
        regime_mult = self.regime_detector.get_position_multiplier()
        
        logger.info(f"Capital: ${capital:,.2f}, Cash: ${cash:,.2f}")
        logger.info(f"Regime: {regime}, Position Multiplier: {regime_mult:.0%}")
        
        # Current positions
        positions = self.broker.get_positions()
        position_symbols = [p['symbol'] for p in positions]
        
        # Calculate factor scores
        factor_scores = self.factor_model.calculate_composite_scores(self.config.UNIVERSE)
        
        if factor_scores.empty:
            logger.error("No factor scores available")
            return actions
        
        # Get target positions (top N by score)
        max_positions = min(self.config.MAX_POSITIONS, len(factor_scores))
        target_symbols = factor_scores.head(max_positions).index.tolist()
        
        logger.info(f"Target positions: {target_symbols[:10]}...")
        
        # Fetch price data for position sizing
        prices_df = self.fetcher.get_prices(target_symbols)
        
        # SELL: Close positions not in target
        for pos in positions:
            if pos['symbol'] not in target_symbols:
                logger.info(f"SELL {pos['symbol']} (not in target universe)")
                
                if not dry_run:
                    self.broker.close_position(pos['symbol'])
                
                actions['orders'].append({
                    'action': 'sell',
                    'symbol': pos['symbol'],
                    'qty': pos['qty'],
                    'reason': 'not_in_target'
                })
        
        # BUY: Enter positions for target symbols not held
        for symbol in target_symbols:
            if symbol in position_symbols:
                continue
            
            if len([o for o in actions['orders'] if o['action'] == 'buy']) >= 5:
                logger.info("Limiting new positions to 5 per run")
                break
            
            try:
                # Get price data
                if symbol not in prices_df.columns.get_level_values(0):
                    continue
                
                sym_data = prices_df[symbol]
                current_price = sym_data['Close'].iloc[-1]
                
                # Calculate ATR for stop loss
                atr = TechnicalIndicators.atr(
                    sym_data['High'], 
                    sym_data['Low'], 
                    sym_data['Close'],
                    self.config.ATR_PERIOD
                )
                
                if pd.isna(atr):
                    atr = current_price * 0.02  # Default 2% if ATR unavailable
                
                # Position sizing
                sizing = self.position_sizer.calculate_position_size(
                    capital=cash / 2,  # Use half of available cash
                    price=current_price,
                    atr=atr,
                    regime_multiplier=regime_mult
                )
                
                shares = sizing['shares']
                
                if shares > 0:
                    score = factor_scores.loc[symbol, 'composite']
                    logger.info(f"BUY {symbol}: {shares} shares @ ${current_price:.2f} "
                               f"(score={score:.3f}, stop=${sizing['stop_loss']:.2f})")
                    
                    if not dry_run:
                        self.broker.submit_order(symbol, shares, 'buy')
                    
                    actions['orders'].append({
                        'action': 'buy',
                        'symbol': symbol,
                        'qty': shares,
                        'price': current_price,
                        'stop_loss': sizing['stop_loss'],
                        'factor_score': score
                    })
                    
            except Exception as e:
                logger.warning(f"Error processing {symbol}: {e}")
        
        # PAIRS TRADING
        try:
            all_pair_symbols = list(set(s for pair in self.config.PAIRS for s in pair))
            pair_prices = self.fetcher.get_prices(all_pair_symbols)
            
            if not pair_prices.empty:
                pairs_signals = self.pairs_trader.generate_signals(pair_prices)
                
                for pair, signal in pairs_signals.items():
                    action = signal.get('action')
                    
                    if action in ['long_spread', 'short_spread']:
                        sym1, sym2 = pair
                        hedge_ratio = signal.get('hedge_ratio', 1.0)
                        
                        # Calculate position sizes for pair
                        pair_capital = capital * 0.05  # 5% per pair
                        
                        if action == 'long_spread':
                            # Long sym1, short sym2
                            side1, side2 = 'buy', 'sell'
                        else:
                            # Short sym1, long sym2
                            side1, side2 = 'sell', 'buy'
                        
                        logger.info(f"PAIRS: {side1.upper()} {sym1}, {side2.upper()} {sym2} "
                                   f"(z={signal['zscore']:.2f})")
                        
                        actions['orders'].append({
                            'action': 'pairs_trade',
                            'pair': pair,
                            'signal': action,
                            'zscore': signal['zscore'],
                            'hedge_ratio': hedge_ratio
                        })
                        
        except Exception as e:
            logger.warning(f"Error in pairs trading: {e}")
        
        # Summary
        n_buys = len([o for o in actions['orders'] if o['action'] == 'buy'])
        n_sells = len([o for o in actions['orders'] if o['action'] == 'sell'])
        n_pairs = len([o for o in actions['orders'] if o['action'] == 'pairs_trade'])
        
        logger.info("=" * 60)
        logger.info(f"SUMMARY: {n_buys} buys, {n_sells} sells, {n_pairs} pairs trades")
        logger.info("=" * 60)
        
        return actions


# =============================================================================
# CLI
# =============================================================================

def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='V34 Multi-Factor Alpha Engine - Quantitative Trading System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python v34_multi_factor_alpha_engine.py --status
  python v34_multi_factor_alpha_engine.py --run --dry-run
  python v34_multi_factor_alpha_engine.py --backtest
  python v34_multi_factor_alpha_engine.py --run --live
        """
    )
    
    parser.add_argument('--status', action='store_true',
                        help='Show current system status')
    parser.add_argument('--run', action='store_true',
                        help='Execute trading logic')
    parser.add_argument('--backtest', action='store_true',
                        help='Run backtest on historical data')
    parser.add_argument('--dry-run', action='store_true', default=True,
                        help='Simulate trades without execution (default)')
    parser.add_argument('--live', action='store_true',
                        help='Execute real trades (use with caution!)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
    
    # Initialize engine
    config = Config()
    engine = MultiFactorAlphaEngine(config)
    
    # Execute requested action
    if args.backtest:
        backtester = Backtester(config)
        results = backtester.run()
        
        if results:
            print(f"\n{'='*60}")
            print("BACKTEST COMPLETE")
            print(f"{'='*60}")
            print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
            print(f"Annual Return: {results.get('annual_return', 0):.1%}")
            print(f"Max Drawdown: {results.get('max_drawdown', 0):.1%}")
            
    elif args.run:
        dry_run = not args.live
        
        if args.live:
            print("\n" + "="*60)
            print("⚠️  WARNING: LIVE TRADING MODE")
            print("="*60)
            print("This will execute real trades with real money!")
            confirm = input("Type 'CONFIRM' to proceed: ")
            if confirm != 'CONFIRM':
                print("Aborted.")
                return
        
        actions = engine.run(dry_run=dry_run)
        
        print(f"\nCompleted: {len(actions.get('orders', []))} orders generated")
        
    elif args.status:
        engine.get_status()
        
    else:
        parser.print_help()
        print("\n💡 Quick start: python v34_multi_factor_alpha_engine.py --status")


if __name__ == '__main__':
    main()
