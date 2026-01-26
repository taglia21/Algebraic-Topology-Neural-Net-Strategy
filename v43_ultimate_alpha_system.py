#!/usr/bin/env python3
"""
v43_ultimate_alpha_system.py - Comprehensive Multi-Strategy Trading System

Components:
1. IEX Data Feed with yfinance fallback
2. Sentiment Analysis (VADER + news scraping)
3. Statistical Arbitrage (Pairs Trading)
4. Volatility Arbitrage (IV vs RV)
5. Neural Network Forecaster (LSTM)
6. Kelly Criterion Position Sizing
7. Weighted Signal Orchestration

Author: Trading System v43
Date: 2026-01-26
"""

import os
import sys
import json
import logging
import argparse
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path
import time
import hashlib

import numpy as np
import pandas as pd

# Suppress warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SystemConfig:
    """System configuration parameters."""
    # API Settings
    paper_trading: bool = True
    feed: str = 'iex'  # Always use IEX feed
    
    # Universe
    symbols: List[str] = field(default_factory=lambda: [
        'SPY', 'QQQ', 'IWM', 'DIA', 'XLF', 'XLK', 'XLE', 'XLV', 
        'GLD', 'GDX', 'TLT', 'HYG', 'VXX', 'AAPL', 'MSFT', 'GOOGL',
        'AMZN', 'META', 'NVDA', 'TSLA'
    ])
    
    # Pairs for stat arb
    pairs: List[Tuple[str, str]] = field(default_factory=lambda: [
        ('SPY', 'QQQ'), ('XLF', 'XLK'), ('GLD', 'GDX'), 
        ('TLT', 'HYG'), ('AAPL', 'MSFT'), ('GOOGL', 'META')
    ])
    
    # Risk Parameters
    max_portfolio_pct: float = 0.10  # Max 10% per position
    max_drawdown_pct: float = 0.25   # 25% drawdown stop
    half_kelly: bool = True          # Use half-Kelly
    
    # Signal Weights
    sentiment_weight: float = 0.15
    pairs_weight: float = 0.25
    vol_arb_weight: float = 0.20
    lstm_weight: float = 0.25
    momentum_weight: float = 0.15
    
    # Thresholds
    sentiment_extreme: float = 0.6   # Sentiment threshold
    zscore_entry: float = 2.0        # Pairs entry z-score
    zscore_exit: float = 0.5         # Pairs exit z-score
    vol_spread_threshold: float = 0.05  # 5% IV-RV spread
    lstm_confidence: float = 0.7     # LSTM confidence threshold
    
    # Data Settings
    lookback_days: int = 252         # 1 year for IV rank
    lstm_sequence_length: int = 20   # LSTM input sequence
    
    # State file
    state_file: str = 'v43_state.json'
    
    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure logging with timestamps and formatting."""
    logger = logging.getLogger('v43_alpha')
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

logger = setup_logging()


# =============================================================================
# DATA FEED MODULE (IEX + yfinance fallback)
# =============================================================================

class DataFeed:
    """
    Data feed manager with IEX primary and yfinance fallback.
    Always uses feed='iex' for Alpaca requests.
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.alpaca_client = None
        self.stock_client = None
        self._init_alpaca()
        
    def _init_alpaca(self) -> None:
        """Initialize Alpaca clients."""
        try:
            from alpaca.data import StockHistoricalDataClient
            from alpaca.trading.client import TradingClient
            
            api_key = os.environ.get('APCA_API_KEY_ID')
            api_secret = os.environ.get('APCA_API_SECRET_KEY')
            
            if api_key and api_secret:
                self.stock_client = StockHistoricalDataClient(api_key, api_secret)
                self.alpaca_client = TradingClient(
                    api_key, api_secret, 
                    paper=self.config.paper_trading
                )
                logger.info("✓ Alpaca clients initialized (IEX feed)")
            else:
                logger.warning("⚠ Alpaca credentials not found, using yfinance only")
        except ImportError:
            logger.warning("⚠ alpaca-py not installed, using yfinance only")
        except Exception as e:
            logger.error(f"✗ Alpaca init error: {e}")
    
    def get_historical_bars(
        self, 
        symbols: List[str], 
        days: int = 252,
        timeframe: str = '1Day'
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical bars using IEX feed with yfinance fallback.
        """
        data = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 10)  # Buffer for weekends
        
        # Try Alpaca IEX first
        if self.stock_client:
            try:
                from alpaca.data.requests import StockBarsRequest
                from alpaca.data.timeframe import TimeFrame
                
                tf = TimeFrame.Day if timeframe == '1Day' else TimeFrame.Hour
                
                request = StockBarsRequest(
                    symbol_or_symbols=symbols,
                    start=start_date,
                    end=end_date,
                    timeframe=tf,
                    feed='iex'  # Always use IEX feed
                )
                
                bars = self.stock_client.get_stock_bars(request)
                
                for symbol in symbols:
                    if symbol in bars.data:
                        df = pd.DataFrame([{
                            'open': b.open,
                            'high': b.high,
                            'low': b.low,
                            'close': b.close,
                            'volume': b.volume,
                            'timestamp': b.timestamp
                        } for b in bars.data[symbol]])
                        
                        if not df.empty:
                            df['timestamp'] = pd.to_datetime(df['timestamp'])
                            df.set_index('timestamp', inplace=True)
                            df = df.tail(days)
                            data[symbol] = df
                            
                logger.info(f"✓ Fetched {len(data)} symbols from Alpaca IEX")
                
            except Exception as e:
                logger.warning(f"⚠ Alpaca IEX error: {e}, falling back to yfinance")
        
        # Fallback to yfinance for missing symbols
        missing_symbols = [s for s in symbols if s not in data]
        if missing_symbols:
            data.update(self._fetch_yfinance(missing_symbols, days))
        
        return data
    
    def _fetch_yfinance(
        self, 
        symbols: List[str], 
        days: int
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data from yfinance as fallback."""
        data = {}
        try:
            import yfinance as yf
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(period=f'{days}d')
                    
                    if not df.empty:
                        df.columns = df.columns.str.lower()
                        df = df[['open', 'high', 'low', 'close', 'volume']]
                        data[symbol] = df
                        
                except Exception as e:
                    logger.debug(f"yfinance error for {symbol}: {e}")
                    
            logger.info(f"✓ Fetched {len(data)} symbols from yfinance")
            
        except ImportError:
            logger.error("✗ yfinance not installed")
            
        return data
    
    def get_latest_quote(self, symbol: str) -> Optional[float]:
        """Get latest quote for a symbol."""
        if self.stock_client:
            try:
                from alpaca.data.requests import StockLatestQuoteRequest
                
                request = StockLatestQuoteRequest(
                    symbol_or_symbols=[symbol],
                    feed='iex'
                )
                quotes = self.stock_client.get_stock_latest_quote(request)
                
                if symbol in quotes:
                    return float(quotes[symbol].ask_price)
                    
            except Exception as e:
                logger.debug(f"Quote error for {symbol}: {e}")
        
        # Fallback to yfinance
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            return ticker.info.get('regularMarketPrice')
        except:
            pass
            
        return None
    
    def get_account(self) -> Optional[Dict]:
        """Get account information."""
        if self.alpaca_client:
            try:
                account = self.alpaca_client.get_account()
                return {
                    'equity': float(account.equity),
                    'cash': float(account.cash),
                    'buying_power': float(account.buying_power),
                    'portfolio_value': float(account.portfolio_value)
                }
            except Exception as e:
                logger.error(f"Account fetch error: {e}")
        return None
    
    def get_positions(self) -> List[Dict]:
        """Get current positions."""
        if self.alpaca_client:
            try:
                positions = self.alpaca_client.get_all_positions()
                return [{
                    'symbol': p.symbol,
                    'qty': float(p.qty),
                    'market_value': float(p.market_value),
                    'unrealized_pl': float(p.unrealized_pl),
                    'unrealized_plpc': float(p.unrealized_plpc)
                } for p in positions]
            except Exception as e:
                logger.error(f"Positions fetch error: {e}")
        return []


# =============================================================================
# SENTIMENT ANALYSIS MODULE
# =============================================================================

class SentimentAnalyzer:
    """
    Sentiment analysis using VADER with news scraping.
    Score: -1 (bearish) to +1 (bullish)
    """
    
    def __init__(self):
        self.vader = None
        self._init_vader()
        self.cache: Dict[str, Tuple[float, datetime]] = {}
        self.cache_ttl = timedelta(hours=1)
        
    def _init_vader(self) -> None:
        """Initialize VADER sentiment analyzer."""
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            import nltk
            
            # Download vader lexicon if needed
            try:
                nltk.data.find('sentiment/vader_lexicon.zip')
            except LookupError:
                nltk.download('vader_lexicon', quiet=True)
            
            self.vader = SentimentIntensityAnalyzer()
            logger.info("✓ VADER sentiment analyzer initialized")
            
        except ImportError:
            logger.warning("⚠ NLTK not installed, sentiment analysis disabled")
        except Exception as e:
            logger.warning(f"⚠ VADER init error: {e}")
    
    def scrape_headlines(self, symbol: str) -> List[str]:
        """Scrape news headlines for a symbol."""
        headlines = []
        
        # Try Yahoo Finance RSS
        try:
            import feedparser
            
            url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}"
            feed = feedparser.parse(url)
            
            for entry in feed.entries[:10]:
                headlines.append(entry.title)
                
        except ImportError:
            logger.debug("feedparser not installed")
        except Exception as e:
            logger.debug(f"RSS scrape error: {e}")
        
        # Try yfinance news
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            for item in news[:10]:
                if 'title' in item:
                    headlines.append(item['title'])
                    
        except Exception as e:
            logger.debug(f"yfinance news error: {e}")
        
        return headlines
    
    def analyze_text(self, text: str) -> float:
        """Analyze sentiment of text, returns -1 to +1."""
        if not self.vader:
            return 0.0
            
        try:
            scores = self.vader.polarity_scores(text)
            return scores['compound']
        except Exception:
            return 0.0
    
    def get_sentiment(self, symbol: str) -> float:
        """
        Get aggregate sentiment for a symbol.
        Returns: -1 (very bearish) to +1 (very bullish)
        """
        # Check cache
        if symbol in self.cache:
            score, timestamp = self.cache[symbol]
            if datetime.now() - timestamp < self.cache_ttl:
                return score
        
        headlines = self.scrape_headlines(symbol)
        
        if not headlines:
            return 0.0
        
        scores = [self.analyze_text(h) for h in headlines]
        avg_score = np.mean(scores) if scores else 0.0
        
        # Cache result
        self.cache[symbol] = (avg_score, datetime.now())
        
        return float(avg_score)
    
    def get_bulk_sentiment(self, symbols: List[str]) -> Dict[str, float]:
        """Get sentiment for multiple symbols."""
        return {symbol: self.get_sentiment(symbol) for symbol in symbols}
    
    def generate_signals(
        self, 
        symbols: List[str], 
        threshold: float = 0.6
    ) -> Dict[str, int]:
        """
        Generate trading signals based on sentiment.
        Returns: {symbol: signal} where signal is -1, 0, or 1
        """
        signals = {}
        sentiments = self.get_bulk_sentiment(symbols)
        
        for symbol, score in sentiments.items():
            if score >= threshold:
                signals[symbol] = 1  # Bullish
            elif score <= -threshold:
                signals[symbol] = -1  # Bearish
            else:
                signals[symbol] = 0  # Neutral
                
        return signals


# =============================================================================
# STATISTICAL ARBITRAGE (PAIRS TRADING)
# =============================================================================

class PairsTrader:
    """
    Statistical arbitrage using cointegrated pairs.
    Entry: z-score > 2, Exit: z-score < 0.5
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.pairs = config.pairs
        self.hedge_ratios: Dict[Tuple[str, str], float] = {}
        self.spread_stats: Dict[Tuple[str, str], Dict] = {}
        
    def test_cointegration(
        self, 
        series1: pd.Series, 
        series2: pd.Series
    ) -> Tuple[bool, float, float]:
        """
        Test for cointegration using ADF test.
        Returns: (is_cointegrated, p_value, hedge_ratio)
        """
        try:
            from statsmodels.tsa.stattools import adfuller
            from statsmodels.regression.linear_model import OLS
            
            # Calculate hedge ratio via OLS
            model = OLS(series1, series2).fit()
            hedge_ratio = model.params[0]
            
            # Calculate spread
            spread = series1 - hedge_ratio * series2
            
            # ADF test on spread
            adf_result = adfuller(spread.dropna())
            p_value = adf_result[1]
            
            is_cointegrated = p_value < 0.05
            
            return is_cointegrated, p_value, hedge_ratio
            
        except ImportError:
            logger.warning("⚠ statsmodels not installed")
            return False, 1.0, 1.0
        except Exception as e:
            logger.debug(f"Cointegration test error: {e}")
            return False, 1.0, 1.0
    
    def calculate_zscore(
        self, 
        series1: pd.Series, 
        series2: pd.Series, 
        hedge_ratio: float,
        window: int = 20
    ) -> pd.Series:
        """Calculate rolling z-score of spread."""
        spread = series1 - hedge_ratio * series2
        zscore = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()
        return zscore
    
    def find_cointegrated_pairs(
        self, 
        data: Dict[str, pd.DataFrame]
    ) -> List[Tuple[str, str, float, float]]:
        """
        Find cointegrated pairs from data.
        Returns: List of (symbol1, symbol2, p_value, hedge_ratio)
        """
        cointegrated = []
        
        for sym1, sym2 in self.pairs:
            if sym1 not in data or sym2 not in data:
                continue
                
            series1 = data[sym1]['close']
            series2 = data[sym2]['close']
            
            # Align series
            combined = pd.concat([series1, series2], axis=1).dropna()
            if len(combined) < 60:
                continue
                
            is_coint, p_val, hedge = self.test_cointegration(
                combined.iloc[:, 0], 
                combined.iloc[:, 1]
            )
            
            if is_coint:
                cointegrated.append((sym1, sym2, p_val, hedge))
                self.hedge_ratios[(sym1, sym2)] = hedge
                logger.info(f"✓ Cointegrated pair: {sym1}/{sym2} (p={p_val:.4f})")
                
        return cointegrated
    
    def generate_signals(
        self, 
        data: Dict[str, pd.DataFrame]
    ) -> Dict[Tuple[str, str], Dict]:
        """
        Generate pairs trading signals.
        Returns: {(sym1, sym2): {'signal': int, 'zscore': float, 'hedge': float}}
        """
        signals = {}
        
        for sym1, sym2 in self.pairs:
            if sym1 not in data or sym2 not in data:
                continue
            
            series1 = data[sym1]['close']
            series2 = data[sym2]['close']
            
            # Get or calculate hedge ratio
            if (sym1, sym2) not in self.hedge_ratios:
                is_coint, _, hedge = self.test_cointegration(series1, series2)
                if not is_coint:
                    continue
                self.hedge_ratios[(sym1, sym2)] = hedge
            
            hedge = self.hedge_ratios[(sym1, sym2)]
            zscore = self.calculate_zscore(series1, series2, hedge)
            
            if zscore.empty:
                continue
                
            current_z = zscore.iloc[-1]
            
            # Generate signal
            if current_z > self.config.zscore_entry:
                signal = -1  # Short spread (short sym1, long sym2)
            elif current_z < -self.config.zscore_entry:
                signal = 1   # Long spread (long sym1, short sym2)
            elif abs(current_z) < self.config.zscore_exit:
                signal = 0   # Exit
            else:
                signal = 0   # Hold
            
            signals[(sym1, sym2)] = {
                'signal': signal,
                'zscore': float(current_z),
                'hedge_ratio': hedge
            }
            
        return signals


# =============================================================================
# VOLATILITY ARBITRAGE
# =============================================================================

class VolatilityArbitrage:
    """
    Volatility arbitrage comparing implied vs realized volatility.
    Uses VIX as market IV proxy.
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        
    def calculate_realized_vol(
        self, 
        prices: pd.Series, 
        window: int = 20
    ) -> pd.Series:
        """Calculate realized volatility (annualized)."""
        returns = np.log(prices / prices.shift(1))
        realized = returns.rolling(window).std() * np.sqrt(252) * 100
        return realized
    
    def calculate_iv_rank(
        self, 
        iv_series: pd.Series, 
        lookback: int = 252
    ) -> float:
        """
        Calculate IV Rank over lookback period.
        IV Rank = (Current IV - Min IV) / (Max IV - Min IV)
        """
        if len(iv_series) < lookback:
            lookback = len(iv_series)
            
        recent = iv_series.tail(lookback)
        current = recent.iloc[-1]
        
        min_iv = recent.min()
        max_iv = recent.max()
        
        if max_iv == min_iv:
            return 50.0
            
        iv_rank = (current - min_iv) / (max_iv - min_iv) * 100
        return float(iv_rank)
    
    def get_vix_data(self, data_feed: DataFeed) -> Optional[pd.DataFrame]:
        """Fetch VIX data as IV proxy."""
        vix_data = data_feed.get_historical_bars(['^VIX', 'VIX', 'VIXY'], days=252)
        
        for symbol in ['^VIX', 'VIX', 'VIXY']:
            if symbol in vix_data and not vix_data[symbol].empty:
                return vix_data[symbol]
                
        return None
    
    def analyze_vol_spread(
        self, 
        symbol: str, 
        price_data: pd.DataFrame,
        vix_data: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Analyze IV vs RV spread for a symbol.
        Returns analysis dict with signal.
        """
        if price_data.empty:
            return {'signal': 0, 'spread': 0}
        
        # Calculate realized vol
        rv = self.calculate_realized_vol(price_data['close'])
        current_rv = rv.iloc[-1] if not rv.empty else 20.0
        
        # Use VIX as IV proxy or estimate from historical
        if vix_data is not None and not vix_data.empty:
            current_iv = vix_data['close'].iloc[-1]
            iv_rank = self.calculate_iv_rank(vix_data['close'])
        else:
            # Estimate IV from historical vol with premium
            hist_vol = self.calculate_realized_vol(price_data['close'], window=60)
            current_iv = hist_vol.iloc[-1] * 1.15 if not hist_vol.empty else 25.0
            iv_rank = 50.0
        
        # Calculate spread
        spread = (current_iv - current_rv) / 100  # As decimal
        
        # Generate signal
        threshold = self.config.vol_spread_threshold
        
        if spread > threshold:
            signal = -1  # Sell vol (IV too high)
        elif spread < -threshold:
            signal = 1   # Buy vol (IV too low)
        else:
            signal = 0   # Neutral
        
        return {
            'signal': signal,
            'implied_vol': float(current_iv),
            'realized_vol': float(current_rv),
            'spread': float(spread),
            'iv_rank': float(iv_rank)
        }
    
    def generate_signals(
        self, 
        data: Dict[str, pd.DataFrame],
        vix_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Dict]:
        """Generate volatility arbitrage signals for all symbols."""
        signals = {}
        
        for symbol, df in data.items():
            analysis = self.analyze_vol_spread(symbol, df, vix_data)
            signals[symbol] = analysis
            
        return signals


# =============================================================================
# NEURAL NETWORK FORECASTER (LSTM)
# =============================================================================

class LSTMForecaster:
    """
    LSTM neural network for price direction prediction.
    Features: OHLCV + technical indicators
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.is_available = False
        self._check_dependencies()
        
    def _check_dependencies(self) -> None:
        """Check if TensorFlow/Keras is available."""
        try:
            import tensorflow as tf
            tf.get_logger().setLevel('ERROR')
            self.is_available = True
            logger.info("✓ TensorFlow available for LSTM")
        except ImportError:
            logger.warning("⚠ TensorFlow not installed, LSTM disabled")
            self.is_available = False
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicator features."""
        features = df.copy()
        
        # Price features
        features['returns'] = features['close'].pct_change()
        features['log_returns'] = np.log(features['close'] / features['close'].shift(1))
        
        # Moving averages
        features['sma_5'] = features['close'].rolling(5).mean()
        features['sma_20'] = features['close'].rolling(20).mean()
        features['sma_ratio'] = features['sma_5'] / features['sma_20']
        
        # RSI
        delta = features['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = features['close'].ewm(span=12, adjust=False).mean()
        exp2 = features['close'].ewm(span=26, adjust=False).mean()
        features['macd'] = exp1 - exp2
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        # Bollinger Bands
        sma = features['close'].rolling(20).mean()
        std = features['close'].rolling(20).std()
        features['bb_upper'] = sma + 2 * std
        features['bb_lower'] = sma - 2 * std
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / sma
        features['bb_position'] = (features['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        # Volume features
        features['volume_sma'] = features['volume'].rolling(20).mean()
        features['volume_ratio'] = features['volume'] / features['volume_sma']
        
        # Volatility
        features['volatility'] = features['returns'].rolling(20).std() * np.sqrt(252)
        
        # Target: next day direction
        features['target'] = (features['close'].shift(-1) > features['close']).astype(int)
        
        return features.dropna()
    
    def prepare_sequences(
        self, 
        features: pd.DataFrame, 
        seq_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM input."""
        feature_cols = [
            'returns', 'log_returns', 'sma_ratio', 'rsi', 'macd_hist',
            'bb_width', 'bb_position', 'volume_ratio', 'volatility'
        ]
        
        # Filter available columns
        available_cols = [c for c in feature_cols if c in features.columns]
        
        data = features[available_cols].values
        targets = features['target'].values
        
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(targets[i + seq_length - 1])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape: Tuple[int, int]) -> Any:
        """Build LSTM model architecture."""
        if not self.is_available:
            return None
            
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
            from tensorflow.keras.optimizers import Adam
            
            model = Sequential([
                LSTM(64, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                BatchNormalization(),
                LSTM(32, return_sequences=False),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            return model
            
        except Exception as e:
            logger.error(f"Model build error: {e}")
            return None
    
    def train(
        self, 
        symbol: str, 
        data: pd.DataFrame,
        epochs: int = 50,
        verbose: int = 0
    ) -> bool:
        """Train LSTM model for a symbol."""
        if not self.is_available:
            return False
            
        try:
            from sklearn.preprocessing import StandardScaler
            
            features = self.calculate_features(data)
            if len(features) < self.config.lstm_sequence_length + 50:
                return False
            
            X, y = self.prepare_sequences(features, self.config.lstm_sequence_length)
            
            # Scale features
            scaler = StandardScaler()
            X_flat = X.reshape(-1, X.shape[-1])
            X_scaled = scaler.fit_transform(X_flat).reshape(X.shape)
            
            # Train/test split
            split = int(len(X_scaled) * 0.8)
            X_train, X_test = X_scaled[:split], X_scaled[split:]
            y_train, y_test = y[:split], y[split:]
            
            # Build and train
            model = self.build_model((X.shape[1], X.shape[2]))
            if model is None:
                return False
            
            model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=32,
                verbose=verbose
            )
            
            # Evaluate
            accuracy = model.evaluate(X_test, y_test, verbose=0)[1]
            logger.info(f"✓ LSTM trained for {symbol}: accuracy={accuracy:.2%}")
            
            self.models[symbol] = model
            self.scalers[symbol] = scaler
            
            return True
            
        except Exception as e:
            logger.error(f"LSTM training error for {symbol}: {e}")
            return False
    
    def predict(self, symbol: str, data: pd.DataFrame) -> Tuple[int, float]:
        """
        Predict next-day direction.
        Returns: (signal, confidence) where signal is -1, 0, or 1
        """
        if not self.is_available or symbol not in self.models:
            return 0, 0.5
            
        try:
            features = self.calculate_features(data)
            if len(features) < self.config.lstm_sequence_length:
                return 0, 0.5
            
            # Prepare last sequence
            X, _ = self.prepare_sequences(features.tail(self.config.lstm_sequence_length + 1), 
                                          self.config.lstm_sequence_length)
            
            if len(X) == 0:
                return 0, 0.5
            
            # Scale and predict
            scaler = self.scalers[symbol]
            X_flat = X[-1:].reshape(-1, X.shape[-1])
            X_scaled = scaler.transform(X_flat).reshape(1, X.shape[1], X.shape[2])
            
            prob = self.models[symbol].predict(X_scaled, verbose=0)[0][0]
            
            # Generate signal based on confidence
            if prob > self.config.lstm_confidence:
                signal = 1  # Bullish
            elif prob < (1 - self.config.lstm_confidence):
                signal = -1  # Bearish
            else:
                signal = 0  # Low confidence
            
            confidence = max(prob, 1 - prob)
            
            return signal, float(confidence)
            
        except Exception as e:
            logger.debug(f"LSTM prediction error for {symbol}: {e}")
            return 0, 0.5
    
    def generate_signals(
        self, 
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict]:
        """Generate LSTM signals for all symbols."""
        signals = {}
        
        for symbol, df in data.items():
            if len(df) < 100:
                continue
                
            # Train if needed
            if symbol not in self.models:
                self.train(symbol, df)
            
            signal, confidence = self.predict(symbol, df)
            signals[symbol] = {
                'signal': signal,
                'confidence': confidence,
                'direction': 'up' if signal > 0 else ('down' if signal < 0 else 'neutral')
            }
            
        return signals


# =============================================================================
# KELLY CRITERION POSITION SIZING
# =============================================================================

class KellySizer:
    """
    Kelly Criterion position sizing with half-Kelly option.
    Dynamically sizes based on strategy confidence.
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.trade_history: List[Dict] = []
        
    def calculate_kelly_fraction(
        self, 
        win_rate: float, 
        avg_win: float, 
        avg_loss: float
    ) -> float:
        """
        Calculate Kelly fraction.
        K = (W * R - L) / R
        where W = win_rate, L = 1 - win_rate, R = avg_win / avg_loss
        """
        if avg_loss == 0 or avg_win == 0:
            return 0.0
            
        loss_rate = 1 - win_rate
        win_loss_ratio = avg_win / abs(avg_loss)
        
        kelly = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio
        
        # Apply half-Kelly for safety
        if self.config.half_kelly:
            kelly *= 0.5
        
        # Clamp to reasonable bounds
        kelly = max(0, min(kelly, self.config.max_portfolio_pct))
        
        return kelly
    
    def calculate_position_size(
        self, 
        portfolio_value: float,
        confidence: float,
        win_rate: float = 0.55,
        avg_win: float = 0.02,
        avg_loss: float = 0.015
    ) -> float:
        """
        Calculate position size in dollars.
        Adjusts Kelly fraction by confidence.
        """
        kelly = self.calculate_kelly_fraction(win_rate, avg_win, avg_loss)
        
        # Adjust by confidence
        adjusted_kelly = kelly * confidence
        
        # Calculate position
        position_size = portfolio_value * adjusted_kelly
        max_position = portfolio_value * self.config.max_portfolio_pct
        
        return min(position_size, max_position)
    
    def update_history(self, trade: Dict) -> None:
        """Update trade history for dynamic sizing."""
        self.trade_history.append(trade)
        
        # Keep last 100 trades
        if len(self.trade_history) > 100:
            self.trade_history = self.trade_history[-100:]
    
    def get_dynamic_stats(self) -> Tuple[float, float, float]:
        """Calculate win rate and avg win/loss from history."""
        if len(self.trade_history) < 10:
            return 0.55, 0.02, 0.015  # Default values
            
        wins = [t for t in self.trade_history if t.get('pnl', 0) > 0]
        losses = [t for t in self.trade_history if t.get('pnl', 0) < 0]
        
        win_rate = len(wins) / len(self.trade_history)
        
        avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0.02
        avg_loss = np.mean([abs(t['pnl']) for t in losses]) if losses else 0.015
        
        return win_rate, avg_win, avg_loss


# =============================================================================
# SIGNAL ORCHESTRATOR
# =============================================================================

class SignalOrchestrator:
    """
    Combines signals from all strategies with weighted voting.
    Implements risk management and drawdown protection.
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.peak_value: float = 0
        self.current_drawdown: float = 0
        self.is_halted: bool = False
        
    def calculate_weighted_signal(
        self, 
        symbol: str,
        sentiment_signal: int,
        pairs_signal: Optional[int],
        vol_signal: int,
        lstm_signal: int,
        lstm_confidence: float,
        momentum_signal: int = 0
    ) -> Tuple[float, str]:
        """
        Calculate weighted aggregate signal.
        Returns: (signal_strength, rationale)
        """
        weights = {
            'sentiment': self.config.sentiment_weight,
            'pairs': self.config.pairs_weight,
            'volatility': self.config.vol_arb_weight,
            'lstm': self.config.lstm_weight,
            'momentum': self.config.momentum_weight
        }
        
        # Calculate weighted sum
        total = 0
        total_weight = 0
        rationale_parts = []
        
        if sentiment_signal != 0:
            total += sentiment_signal * weights['sentiment']
            total_weight += weights['sentiment']
            rationale_parts.append(f"sentiment={sentiment_signal:+d}")
        
        if pairs_signal is not None and pairs_signal != 0:
            total += pairs_signal * weights['pairs']
            total_weight += weights['pairs']
            rationale_parts.append(f"pairs={pairs_signal:+d}")
        
        if vol_signal != 0:
            total += vol_signal * weights['volatility']
            total_weight += weights['volatility']
            rationale_parts.append(f"vol_arb={vol_signal:+d}")
        
        if lstm_signal != 0:
            # Weight by confidence
            lstm_weighted = lstm_signal * weights['lstm'] * lstm_confidence
            total += lstm_weighted
            total_weight += weights['lstm'] * lstm_confidence
            rationale_parts.append(f"lstm={lstm_signal:+d}@{lstm_confidence:.0%}")
        
        if momentum_signal != 0:
            total += momentum_signal * weights['momentum']
            total_weight += weights['momentum']
            rationale_parts.append(f"momentum={momentum_signal:+d}")
        
        # Normalize
        signal_strength = total / total_weight if total_weight > 0 else 0
        rationale = ", ".join(rationale_parts) if rationale_parts else "no signals"
        
        return signal_strength, rationale
    
    def calculate_momentum_signal(self, data: pd.DataFrame) -> int:
        """Simple momentum signal based on moving averages."""
        if len(data) < 50:
            return 0
            
        close = data['close']
        sma_10 = close.rolling(10).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1]
        current = close.iloc[-1]
        
        if current > sma_10 > sma_50:
            return 1  # Bullish momentum
        elif current < sma_10 < sma_50:
            return -1  # Bearish momentum
        return 0
    
    def check_drawdown(self, portfolio_value: float) -> bool:
        """
        Check if drawdown limit is exceeded.
        Returns True if trading should continue.
        """
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
            self.current_drawdown = 0
        else:
            self.current_drawdown = (self.peak_value - portfolio_value) / self.peak_value
        
        if self.current_drawdown >= self.config.max_drawdown_pct:
            if not self.is_halted:
                logger.warning(f"⚠ DRAWDOWN LIMIT: {self.current_drawdown:.1%} >= {self.config.max_drawdown_pct:.0%}")
                self.is_halted = True
            return False
        
        if self.is_halted and self.current_drawdown < self.config.max_drawdown_pct * 0.5:
            logger.info("✓ Resuming trading: drawdown recovered")
            self.is_halted = False
        
        return not self.is_halted
    
    def generate_orders(
        self,
        signals: Dict[str, float],
        rationales: Dict[str, str],
        kelly_sizer: KellySizer,
        portfolio_value: float,
        current_positions: Dict[str, float]
    ) -> List[Dict]:
        """
        Generate orders based on signals and position sizing.
        """
        orders = []
        
        for symbol, signal_strength in signals.items():
            if abs(signal_strength) < 0.2:  # Ignore weak signals
                continue
            
            # Calculate position size
            confidence = min(abs(signal_strength), 1.0)
            win_rate, avg_win, avg_loss = kelly_sizer.get_dynamic_stats()
            
            target_size = kelly_sizer.calculate_position_size(
                portfolio_value, confidence, win_rate, avg_win, avg_loss
            )
            
            current_size = current_positions.get(symbol, 0)
            
            # Determine order
            if signal_strength > 0.2:
                # Long signal
                if current_size < 0:
                    # Close short first
                    orders.append({
                        'symbol': symbol,
                        'side': 'buy',
                        'notional': abs(current_size),
                        'action': 'close_short',
                        'rationale': rationales.get(symbol, '')
                    })
                    current_size = 0
                
                if current_size < target_size:
                    orders.append({
                        'symbol': symbol,
                        'side': 'buy',
                        'notional': target_size - current_size,
                        'action': 'open_long',
                        'rationale': rationales.get(symbol, '')
                    })
                    
            elif signal_strength < -0.2:
                # Short signal (if allowed)
                if current_size > 0:
                    orders.append({
                        'symbol': symbol,
                        'side': 'sell',
                        'notional': current_size,
                        'action': 'close_long',
                        'rationale': rationales.get(symbol, '')
                    })
        
        return orders


# =============================================================================
# STATE PERSISTENCE
# =============================================================================

class StateManager:
    """JSON state persistence for the trading system."""
    
    def __init__(self, filepath: str = 'v43_state.json'):
        self.filepath = Path(filepath)
        
    def save(self, state: Dict) -> bool:
        """Save state to JSON file."""
        try:
            state['timestamp'] = datetime.now().isoformat()
            with open(self.filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            return True
        except Exception as e:
            logger.error(f"State save error: {e}")
            return False
    
    def load(self) -> Dict:
        """Load state from JSON file."""
        try:
            if self.filepath.exists():
                with open(self.filepath, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"State load error: {e}")
        return {}
    
    def update(self, key: str, value: Any) -> None:
        """Update a single key in state."""
        state = self.load()
        state[key] = value
        self.save(state)


# =============================================================================
# MAIN TRADING SYSTEM
# =============================================================================

class UltimateAlphaSystem:
    """
    Main trading system orchestrating all components.
    """
    
    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        
        # Initialize components
        logger.info("=" * 60)
        logger.info("INITIALIZING V43 ULTIMATE ALPHA SYSTEM")
        logger.info("=" * 60)
        
        self.data_feed = DataFeed(self.config)
        self.sentiment = SentimentAnalyzer()
        self.pairs_trader = PairsTrader(self.config)
        self.vol_arb = VolatilityArbitrage(self.config)
        self.lstm = LSTMForecaster(self.config)
        self.kelly = KellySizer(self.config)
        self.orchestrator = SignalOrchestrator(self.config)
        self.state = StateManager(self.config.state_file)
        
        logger.info("=" * 60)
        logger.info("SYSTEM INITIALIZED")
        logger.info("=" * 60)
    
    def run_analysis(self) -> Dict:
        """Run full analysis across all strategies."""
        logger.info("\n📊 Running Analysis...")
        
        # Fetch data
        data = self.data_feed.get_historical_bars(
            self.config.symbols, 
            days=self.config.lookback_days
        )
        
        if not data:
            logger.error("✗ No data fetched")
            return {}
        
        logger.info(f"✓ Data fetched for {len(data)} symbols")
        
        # Get VIX data for vol arb
        vix_data = self.vol_arb.get_vix_data(self.data_feed)
        
        # Run strategies
        results = {
            'timestamp': datetime.now().isoformat(),
            'symbols_analyzed': len(data)
        }
        
        # 1. Sentiment Analysis
        logger.info("🔍 Analyzing sentiment...")
        sentiment_signals = self.sentiment.generate_signals(
            list(data.keys()), 
            self.config.sentiment_extreme
        )
        results['sentiment'] = sentiment_signals
        
        # 2. Pairs Trading
        logger.info("🔗 Analyzing pairs...")
        pairs_signals = self.pairs_trader.generate_signals(data)
        results['pairs'] = {f"{k[0]}/{k[1]}": v for k, v in pairs_signals.items()}
        
        # 3. Volatility Arbitrage
        logger.info("📈 Analyzing volatility...")
        vol_signals = self.vol_arb.generate_signals(data, vix_data)
        results['volatility'] = vol_signals
        
        # 4. LSTM Predictions
        logger.info("🧠 Running LSTM predictions...")
        lstm_signals = self.lstm.generate_signals(data)
        results['lstm'] = lstm_signals
        
        # 5. Generate aggregate signals
        logger.info("⚖️ Computing aggregate signals...")
        aggregate_signals = {}
        rationales = {}
        
        for symbol in data.keys():
            sent_sig = sentiment_signals.get(symbol, 0)
            vol_sig = vol_signals.get(symbol, {}).get('signal', 0)
            lstm_result = lstm_signals.get(symbol, {'signal': 0, 'confidence': 0.5})
            lstm_sig = lstm_result['signal']
            lstm_conf = lstm_result['confidence']
            momentum_sig = self.orchestrator.calculate_momentum_signal(data[symbol])
            
            # Check for pairs signal
            pairs_sig = None
            for (s1, s2), pdata in pairs_signals.items():
                if symbol == s1:
                    pairs_sig = pdata['signal']
                    break
                elif symbol == s2:
                    pairs_sig = -pdata['signal']  # Opposite side
                    break
            
            signal, rationale = self.orchestrator.calculate_weighted_signal(
                symbol, sent_sig, pairs_sig, vol_sig, lstm_sig, lstm_conf, momentum_sig
            )
            
            aggregate_signals[symbol] = signal
            rationales[symbol] = rationale
        
        results['aggregate_signals'] = aggregate_signals
        results['rationales'] = rationales
        
        # Save state
        self.state.save(results)
        
        return results
    
    def execute_trades(self) -> Dict:
        """Execute trades based on analysis."""
        # Check trading status
        account = self.data_feed.get_account()
        if not account:
            logger.error("✗ Cannot fetch account")
            return {'error': 'Account fetch failed'}
        
        portfolio_value = account['portfolio_value']
        
        # Check drawdown
        if not self.orchestrator.check_drawdown(portfolio_value):
            logger.warning("⚠ Trading halted due to drawdown")
            return {'halted': True, 'drawdown': self.orchestrator.current_drawdown}
        
        # Run analysis
        analysis = self.run_analysis()
        if not analysis:
            return {'error': 'Analysis failed'}
        
        # Get current positions
        positions = self.data_feed.get_positions()
        current_positions = {p['symbol']: p['market_value'] for p in positions}
        
        # Generate orders
        orders = self.orchestrator.generate_orders(
            analysis['aggregate_signals'],
            analysis['rationales'],
            self.kelly,
            portfolio_value,
            current_positions
        )
        
        logger.info(f"\n📝 Generated {len(orders)} orders")
        
        # Execute orders
        executed = []
        for order in orders:
            logger.info(
                f"  → {order['action']}: {order['side']} ${order['notional']:.2f} {order['symbol']}"
            )
            logger.info(f"    Rationale: {order['rationale']}")
            
            # Execute via Alpaca
            if self.data_feed.alpaca_client:
                try:
                    from alpaca.trading.requests import MarketOrderRequest
                    from alpaca.trading.enums import OrderSide, TimeInForce
                    
                    side = OrderSide.BUY if order['side'] == 'buy' else OrderSide.SELL
                    
                    request = MarketOrderRequest(
                        symbol=order['symbol'],
                        notional=order['notional'],
                        side=side,
                        time_in_force=TimeInForce.DAY
                    )
                    
                    result = self.data_feed.alpaca_client.submit_order(request)
                    executed.append({
                        'symbol': order['symbol'],
                        'order_id': result.id,
                        'status': result.status
                    })
                    logger.info(f"    ✓ Order submitted: {result.id}")
                    
                except Exception as e:
                    logger.error(f"    ✗ Order failed: {e}")
        
        return {
            'orders_generated': len(orders),
            'orders_executed': len(executed),
            'executed': executed,
            'portfolio_value': portfolio_value
        }
    
    def get_status(self) -> Dict:
        """Get current system status."""
        status = {
            'timestamp': datetime.now().isoformat(),
            'system': 'v43_ultimate_alpha'
        }
        
        # Account info
        account = self.data_feed.get_account()
        if account:
            status['account'] = account
        
        # Positions
        positions = self.data_feed.get_positions()
        status['positions'] = positions
        status['position_count'] = len(positions)
        
        # Drawdown
        if account:
            self.orchestrator.check_drawdown(account['portfolio_value'])
        status['drawdown'] = f"{self.orchestrator.current_drawdown:.2%}"
        status['trading_halted'] = self.orchestrator.is_halted
        
        # Last analysis
        state = self.state.load()
        if state:
            status['last_analysis'] = state.get('timestamp')
            status['symbols_analyzed'] = state.get('symbols_analyzed', 0)
        
        return status
    
    def run_backtest(
        self, 
        start_date: str = None,
        end_date: str = None,
        initial_capital: float = 100000
    ) -> Dict:
        """Run backtesting simulation."""
        logger.info("\n🔬 Running Backtest...")
        
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Initial Capital: ${initial_capital:,.2f}")
        
        # Fetch data
        data = self.data_feed.get_historical_bars(
            self.config.symbols[:5],  # Limit for speed
            days=365
        )
        
        if not data:
            return {'error': 'No data for backtest'}
        
        # Simple backtest simulation
        capital = initial_capital
        trades = []
        equity_curve = [capital]
        
        # Get signals for each day
        for symbol, df in data.items():
            if len(df) < 60:
                continue
            
            # Calculate features and signals
            for i in range(60, len(df)):
                window = df.iloc[:i]
                
                # Simple momentum signal
                sma_10 = window['close'].tail(10).mean()
                sma_50 = window['close'].tail(50).mean()
                current = window['close'].iloc[-1]
                
                if current > sma_10 > sma_50:
                    signal = 1
                elif current < sma_10 < sma_50:
                    signal = -1
                else:
                    signal = 0
                
                if signal != 0:
                    # Simulate trade
                    entry = current
                    next_price = df['close'].iloc[i] if i < len(df) else current
                    
                    pnl = (next_price - entry) / entry * signal
                    position_size = capital * 0.05  # 5% position
                    trade_pnl = position_size * pnl
                    
                    capital += trade_pnl
                    equity_curve.append(capital)
                    
                    trades.append({
                        'symbol': symbol,
                        'signal': signal,
                        'entry': entry,
                        'exit': next_price,
                        'pnl': trade_pnl
                    })
        
        # Calculate metrics
        returns = pd.Series(equity_curve).pct_change().dropna()
        
        total_return = (capital - initial_capital) / initial_capital
        win_trades = [t for t in trades if t['pnl'] > 0]
        win_rate = len(win_trades) / len(trades) if trades else 0
        
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        max_dd = 0
        peak = equity_curve[0]
        for val in equity_curve:
            if val > peak:
                peak = val
            dd = (peak - val) / peak
            max_dd = max(max_dd, dd)
        
        results = {
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_return': f"{total_return:.2%}",
            'total_trades': len(trades),
            'win_rate': f"{win_rate:.2%}",
            'sharpe_ratio': round(sharpe, 2),
            'max_drawdown': f"{max_dd:.2%}",
            'equity_curve_length': len(equity_curve)
        }
        
        logger.info("\n📊 Backtest Results:")
        for key, value in results.items():
            logger.info(f"  {key}: {value}")
        
        return results
    
    def print_status(self) -> None:
        """Print formatted system status."""
        status = self.get_status()
        
        print("\n" + "=" * 60)
        print("V43 ULTIMATE ALPHA SYSTEM STATUS")
        print("=" * 60)
        print(f"Timestamp: {status['timestamp']}")
        print()
        
        if 'account' in status:
            acc = status['account']
            print("📊 ACCOUNT:")
            print(f"  Equity:       ${acc['equity']:>12,.2f}")
            print(f"  Cash:         ${acc['cash']:>12,.2f}")
            print(f"  Buying Power: ${acc['buying_power']:>12,.2f}")
        
        print()
        print(f"📈 POSITIONS: {status['position_count']}")
        for pos in status.get('positions', []):
            pnl_str = f"${pos['unrealized_pl']:+,.2f}"
            print(f"  {pos['symbol']:6} {pos['qty']:>8.2f} shares | {pnl_str:>12} ({pos['unrealized_plpc']:+.2%})")
        
        print()
        print(f"⚠️  RISK:")
        print(f"  Drawdown: {status['drawdown']}")
        print(f"  Trading Halted: {status['trading_halted']}")
        
        if status.get('last_analysis'):
            print()
            print(f"🔍 Last Analysis: {status['last_analysis']}")
            print(f"   Symbols: {status.get('symbols_analyzed', 0)}")
        
        print("=" * 60)


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Main entry point with CLI interface."""
    parser = argparse.ArgumentParser(
        description='V43 Ultimate Alpha Trading System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python v43_ultimate_alpha_system.py --status
  python v43_ultimate_alpha_system.py --test
  python v43_ultimate_alpha_system.py --trade
  python v43_ultimate_alpha_system.py --backtest
  python v43_ultimate_alpha_system.py --analyze
        """
    )
    
    parser.add_argument(
        '--trade', 
        action='store_true',
        help='Execute live trading (paper mode by default)'
    )
    parser.add_argument(
        '--test', 
        action='store_true',
        help='Run analysis without trading'
    )
    parser.add_argument(
        '--status', 
        action='store_true',
        help='Show current system status'
    )
    parser.add_argument(
        '--backtest', 
        action='store_true',
        help='Run backtesting simulation'
    )
    parser.add_argument(
        '--analyze', 
        action='store_true',
        help='Run full analysis and show signals'
    )
    parser.add_argument(
        '--live', 
        action='store_true',
        help='Enable live trading (not paper)'
    )
    parser.add_argument(
        '--symbols',
        type=str,
        help='Comma-separated list of symbols'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        setup_logging(logging.DEBUG)
    
    # Build config
    config = SystemConfig()
    
    if args.live:
        config.paper_trading = False
        logger.warning("⚠️  LIVE TRADING ENABLED")
    
    if args.symbols:
        config.symbols = [s.strip().upper() for s in args.symbols.split(',')]
    
    # Initialize system
    system = UltimateAlphaSystem(config)
    
    # Execute command
    if args.status:
        system.print_status()
        
    elif args.test or args.analyze:
        results = system.run_analysis()
        
        print("\n" + "=" * 60)
        print("ANALYSIS RESULTS")
        print("=" * 60)
        
        # Show top signals
        signals = results.get('aggregate_signals', {})
        rationales = results.get('rationales', {})
        
        sorted_signals = sorted(
            signals.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        
        print("\n🎯 TOP SIGNALS:")
        for symbol, strength in sorted_signals[:10]:
            direction = "📈 LONG" if strength > 0 else "📉 SHORT" if strength < 0 else "➖ FLAT"
            print(f"  {symbol:6} {direction} ({strength:+.3f})")
            if symbol in rationales:
                print(f"         → {rationales[symbol]}")
        
        # Show sentiment
        print("\n🎭 SENTIMENT:")
        for symbol, sig in list(results.get('sentiment', {}).items())[:5]:
            emoji = "🟢" if sig > 0 else "🔴" if sig < 0 else "⚪"
            print(f"  {emoji} {symbol}: {sig:+d}")
        
        # Show pairs
        print("\n🔗 PAIRS:")
        for pair, data in results.get('pairs', {}).items():
            z = data.get('zscore', 0)
            sig = data.get('signal', 0)
            print(f"  {pair}: z={z:.2f}, signal={sig:+d}")
        
        # Show volatility
        print("\n📊 VOLATILITY (sample):")
        for symbol, data in list(results.get('volatility', {}).items())[:5]:
            iv = data.get('implied_vol', 0)
            rv = data.get('realized_vol', 0)
            spread = data.get('spread', 0)
            print(f"  {symbol}: IV={iv:.1f}%, RV={rv:.1f}%, spread={spread:+.2f}")
        
        print("=" * 60)
        
    elif args.backtest:
        results = system.run_backtest()
        
    elif args.trade:
        if not args.live:
            logger.info("📝 Paper trading mode")
        
        results = system.execute_trades()
        
        print("\n" + "=" * 60)
        print("TRADING RESULTS")
        print("=" * 60)
        print(f"Orders Generated: {results.get('orders_generated', 0)}")
        print(f"Orders Executed:  {results.get('orders_executed', 0)}")
        
        if results.get('halted'):
            print(f"\n⚠️  Trading HALTED - Drawdown: {results.get('drawdown', 0):.1%}")
        
        print("=" * 60)
        
    else:
        # Default: show status
        system.print_status()
        print("\nUse --help for available commands")


if __name__ == '__main__':
    main()
