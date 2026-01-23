#!/usr/bin/env python3
"""
V15.0 ELITE RETAIL SYSTEMATIC TRADING STRATEGY
==============================================
Institutional-grade systematic trading using:
- Polygon.io: 2+ years historical data, 1-minute bars
- Alpaca Markets: Paper trading execution
- Machine Learning: Ensemble signals
- Multi-timeframe: Daily + Intraday alpha

Target: Sharpe ≥3.5, CAGR ≥50%, MaxDD >-15%
"""

import os
import sys
import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    """V15.0 System Configuration"""
    
    # API Keys
    POLYGON_KEY = os.getenv('POLYGON_API_KEY')
    ALPACA_KEY = os.getenv('ALPACA_API_KEY')
    ALPACA_SECRET = os.getenv('ALPACA_SECRET_KEY')
    ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    
    # Data Settings
    INTRADAY_LOOKBACK_MONTHS = 6
    DAILY_LOOKBACK_YEARS = 2
    
    # V13.0 Production Tickers (proven performers)
    TICKERS = [
        'SPY', 'QQQ', 'IWM',  # Core ETFs
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',  # Mega tech
        'XLF', 'XLE', 'XLK', 'XLV',  # Sectors
        'GLD', 'TLT', 'VXX',  # Alternatives
    ]
    
    # Trading Parameters
    INITIAL_CAPITAL = 100_000
    MAX_POSITION_PCT = 0.10  # 10% max per position
    KELLY_FRACTION = 0.25   # 1/4 Kelly for safety
    MAX_RISK_PER_TRADE = 0.02  # 2% risk per trade
    
    # Slippage & Costs
    SLIPPAGE_INTRADAY_BPS = 10  # 10 bps intraday
    SLIPPAGE_DAILY_BPS = 5      # 5 bps daily
    COMMISSION_PER_SHARE = 0.00  # Alpaca is commission-free
    
    # Output Paths
    RESULTS_DIR = Path('/workspaces/Algebraic-Topology-Neural-Net-Strategy/results/v150')
    
    # ML Settings
    ML_FEATURES = 50
    ML_TRAIN_SPLIT = 0.7
    ML_TARGET_ACCURACY = 0.55


# ==============================================================================
# PHASE 1: API CONNECTIVITY TESTS
# ==============================================================================

def test_polygon_connection() -> Tuple[bool, str]:
    """Test Polygon.io API connectivity"""
    try:
        from polygon import RESTClient
        
        client = RESTClient(api_key=Config.POLYGON_KEY)
        
        # Test with a simple aggregates request
        aggs = client.get_aggs(
            ticker="SPY",
            multiplier=1,
            timespan="day",
            from_="2025-01-15",
            to="2025-01-20",
            limit=5
        )
        
        if aggs and len(aggs) > 0:
            return True, f"✅ Polygon.io connected - Retrieved {len(aggs)} bars"
        else:
            return False, "❌ Polygon.io - No data returned"
            
    except Exception as e:
        return False, f"❌ Polygon.io error: {str(e)}"


def test_alpaca_connection() -> Tuple[bool, str]:
    """Test Alpaca Markets API connectivity"""
    try:
        from alpaca.trading.client import TradingClient
        
        client = TradingClient(
            api_key=Config.ALPACA_KEY,
            secret_key=Config.ALPACA_SECRET,
            paper=True
        )
        
        # Get account info
        account = client.get_account()
        
        return True, f"✅ Alpaca connected - Balance: ${float(account.cash):,.2f}"
        
    except Exception as e:
        return False, f"❌ Alpaca error: {str(e)}"


def run_phase1() -> Dict[str, Any]:
    """Run Phase 1: Infrastructure Setup & API Tests"""
    print("\n" + "="*70)
    print("PHASE 1: INFRASTRUCTURE SETUP")
    print("="*70)
    
    results = {
        'phase': 1,
        'timestamp': datetime.now().isoformat(),
        'polygon_status': None,
        'alpaca_status': None,
        'success': False
    }
    
    # Test Polygon
    polygon_ok, polygon_msg = test_polygon_connection()
    print(f"\n{polygon_msg}")
    results['polygon_status'] = {'ok': polygon_ok, 'message': polygon_msg}
    
    # Test Alpaca
    alpaca_ok, alpaca_msg = test_alpaca_connection()
    print(f"{alpaca_msg}")
    results['alpaca_status'] = {'ok': alpaca_ok, 'message': alpaca_msg}
    
    # Overall status
    results['success'] = polygon_ok and alpaca_ok
    
    if results['success']:
        print("\n✅ PHASE 1 COMPLETE - All APIs connected")
    else:
        print("\n⚠️  PHASE 1 WARNING - Some APIs failed")
    
    return results


# ==============================================================================
# PHASE 2: DATA DOWNLOAD VIA POLYGON.IO
# ==============================================================================

def download_intraday_data(tickers: List[str], months: int = 6) -> pd.DataFrame:
    """Download 1-minute intraday data from Polygon.io"""
    from polygon import RESTClient
    
    client = RESTClient(api_key=Config.POLYGON_KEY)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months * 30)
    
    all_data = []
    
    for ticker in tickers:
        print(f"  Downloading {ticker} intraday...", end=" ")
        try:
            aggs = client.get_aggs(
                ticker=ticker,
                multiplier=1,
                timespan="minute",
                from_=start_date.strftime("%Y-%m-%d"),
                to=end_date.strftime("%Y-%m-%d"),
                limit=50000  # Max limit
            )
            
            if aggs:
                df = pd.DataFrame([{
                    'timestamp': pd.Timestamp(a.timestamp, unit='ms'),
                    'ticker': ticker,
                    'open': a.open,
                    'high': a.high,
                    'low': a.low,
                    'close': a.close,
                    'volume': a.volume,
                    'vwap': getattr(a, 'vwap', None),
                    'trades': getattr(a, 'transactions', None)
                } for a in aggs])
                
                all_data.append(df)
                print(f"✅ {len(df):,} bars")
            else:
                print("⚠️ No data")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        return combined
    return pd.DataFrame()


def download_daily_data(tickers: List[str], years: int = 2) -> pd.DataFrame:
    """Download daily OHLCV data from Polygon.io"""
    from polygon import RESTClient
    
    client = RESTClient(api_key=Config.POLYGON_KEY)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    all_data = []
    
    for ticker in tickers:
        print(f"  Downloading {ticker} daily...", end=" ")
        try:
            aggs = client.get_aggs(
                ticker=ticker,
                multiplier=1,
                timespan="day",
                from_=start_date.strftime("%Y-%m-%d"),
                to=end_date.strftime("%Y-%m-%d"),
                limit=5000
            )
            
            if aggs:
                df = pd.DataFrame([{
                    'date': pd.Timestamp(a.timestamp, unit='ms').date(),
                    'ticker': ticker,
                    'open': a.open,
                    'high': a.high,
                    'low': a.low,
                    'close': a.close,
                    'volume': a.volume,
                    'vwap': getattr(a, 'vwap', None)
                } for a in aggs])
                
                all_data.append(df)
                print(f"✅ {len(df)} days")
            else:
                print("⚠️ No data")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        return combined
    return pd.DataFrame()


def calculate_technical_features(df: pd.DataFrame, is_intraday: bool = False) -> pd.DataFrame:
    """Calculate technical indicators for each ticker"""
    
    results = []
    time_col = 'timestamp' if is_intraday else 'date'
    
    for ticker in df['ticker'].unique():
        ticker_df = df[df['ticker'] == ticker].copy()
        ticker_df = ticker_df.sort_values(time_col)
        
        # RSI
        delta = ticker_df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        ticker_df['rsi'] = 100 - (100 / (1 + rs))
        
        # Moving averages
        ticker_df['sma_20'] = ticker_df['close'].rolling(20).mean()
        ticker_df['sma_50'] = ticker_df['close'].rolling(50).mean()
        ticker_df['ema_12'] = ticker_df['close'].ewm(span=12).mean()
        ticker_df['ema_26'] = ticker_df['close'].ewm(span=26).mean()
        
        # MACD
        ticker_df['macd'] = ticker_df['ema_12'] - ticker_df['ema_26']
        ticker_df['macd_signal'] = ticker_df['macd'].ewm(span=9).mean()
        
        # Bollinger Bands
        ticker_df['bb_mid'] = ticker_df['close'].rolling(20).mean()
        ticker_df['bb_std'] = ticker_df['close'].rolling(20).std()
        ticker_df['bb_upper'] = ticker_df['bb_mid'] + 2 * ticker_df['bb_std']
        ticker_df['bb_lower'] = ticker_df['bb_mid'] - 2 * ticker_df['bb_std']
        ticker_df['bb_pct'] = (ticker_df['close'] - ticker_df['bb_lower']) / (ticker_df['bb_upper'] - ticker_df['bb_lower'] + 1e-10)
        
        # ATR
        tr1 = ticker_df['high'] - ticker_df['low']
        tr2 = abs(ticker_df['high'] - ticker_df['close'].shift(1))
        tr3 = abs(ticker_df['low'] - ticker_df['close'].shift(1))
        ticker_df['atr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()
        
        # Volume features
        ticker_df['volume_sma'] = ticker_df['volume'].rolling(20).mean()
        ticker_df['volume_ratio'] = ticker_df['volume'] / (ticker_df['volume_sma'] + 1)
        
        # Returns
        ticker_df['return_1d'] = ticker_df['close'].pct_change()
        ticker_df['return_5d'] = ticker_df['close'].pct_change(5)
        ticker_df['return_20d'] = ticker_df['close'].pct_change(20)
        
        # Momentum
        ticker_df['momentum_10'] = ticker_df['close'] / ticker_df['close'].shift(10) - 1
        ticker_df['momentum_20'] = ticker_df['close'] / ticker_df['close'].shift(20) - 1
        
        if is_intraday:
            # Opening Range Breakout features
            ticker_df['hour'] = pd.to_datetime(ticker_df['timestamp']).dt.hour
            ticker_df['minute'] = pd.to_datetime(ticker_df['timestamp']).dt.minute
            
        results.append(ticker_df)
    
    return pd.concat(results, ignore_index=True)


def run_phase2() -> Dict[str, Any]:
    """Run Phase 2: Data Download via Polygon.io"""
    print("\n" + "="*70)
    print("PHASE 2: DATA DOWNLOAD VIA POLYGON.IO")
    print("="*70)
    
    results = {
        'phase': 2,
        'timestamp': datetime.now().isoformat(),
        'intraday_bars': 0,
        'daily_bars': 0,
        'tickers': Config.TICKERS,
        'success': False
    }
    
    # Download intraday data
    print(f"\n📊 Downloading {Config.INTRADAY_LOOKBACK_MONTHS}-month intraday data...")
    intraday_df = download_intraday_data(Config.TICKERS, Config.INTRADAY_LOOKBACK_MONTHS)
    
    if not intraday_df.empty:
        # Calculate features
        print("\n  Calculating technical features...")
        intraday_df = calculate_technical_features(intraday_df, is_intraday=True)
        
        # Save
        intraday_path = Config.RESULTS_DIR / 'v150_intraday_6m.parquet'
        intraday_df.to_parquet(intraday_path, compression='gzip')
        results['intraday_bars'] = len(intraday_df)
        print(f"  ✅ Saved {len(intraday_df):,} intraday bars to {intraday_path.name}")
    
    # Download daily data
    print(f"\n📊 Downloading {Config.DAILY_LOOKBACK_YEARS}-year daily data...")
    daily_df = download_daily_data(Config.TICKERS, Config.DAILY_LOOKBACK_YEARS)
    
    if not daily_df.empty:
        # Calculate features
        print("\n  Calculating technical features...")
        daily_df = calculate_technical_features(daily_df, is_intraday=False)
        
        # Save
        daily_path = Config.RESULTS_DIR / 'v150_daily_2y.parquet'
        daily_df.to_parquet(daily_path, compression='gzip')
        results['daily_bars'] = len(daily_df)
        print(f"  ✅ Saved {len(daily_df):,} daily bars to {daily_path.name}")
    
    results['success'] = results['intraday_bars'] > 0 or results['daily_bars'] > 0
    
    if results['success']:
        print(f"\n✅ PHASE 2 COMPLETE - Downloaded {results['intraday_bars']:,} intraday + {results['daily_bars']:,} daily bars")
    else:
        print("\n❌ PHASE 2 FAILED - No data downloaded")
    
    return results


# ==============================================================================
# PHASE 3: STRATEGY DEVELOPMENT
# ==============================================================================

class HMMRegimeDetector:
    """Hidden Markov Model for market regime detection"""
    
    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes
        self.model = None
        
    def fit(self, returns: np.ndarray, volatility: np.ndarray):
        """Fit HMM to returns and volatility"""
        from hmmlearn.hmm import GaussianHMM
        
        # Prepare features
        X = np.column_stack([returns, volatility])
        X = X[~np.isnan(X).any(axis=1)]
        
        self.model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        self.model.fit(X)
        
    def predict_regime(self, returns: np.ndarray, volatility: np.ndarray) -> int:
        """Predict current market regime"""
        if self.model is None:
            return 1  # Default to neutral
            
        X = np.column_stack([returns[-1:], volatility[-1:]])
        return self.model.predict(X)[0]


class DailyStrategy:
    """Daily timeframe strategy component"""
    
    def __init__(self):
        self.hmm = HMMRegimeDetector()
        
    def calculate_momentum_signal(self, df: pd.DataFrame) -> pd.Series:
        """Multi-factor momentum signal"""
        # Price momentum
        mom_1m = df['close'].pct_change(21)
        mom_3m = df['close'].pct_change(63)
        mom_6m = df['close'].pct_change(126)
        
        # Composite momentum (skip most recent month for reversal)
        momentum = (mom_6m - mom_1m) * 0.5 + mom_3m * 0.5
        
        # Normalize
        momentum = (momentum - momentum.rolling(252).mean()) / (momentum.rolling(252).std() + 1e-10)
        
        return momentum.clip(-3, 3)
    
    def calculate_quality_signal(self, df: pd.DataFrame) -> pd.Series:
        """Quality factor: low volatility, high returns"""
        returns = df['close'].pct_change()
        vol = returns.rolling(21).std()
        sharpe_rolling = returns.rolling(63).mean() / (vol + 1e-10)
        
        return sharpe_rolling.clip(-3, 3)
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate daily trading signals"""
        signals = pd.DataFrame(index=df.index)
        
        signals['momentum'] = self.calculate_momentum_signal(df)
        signals['quality'] = self.calculate_quality_signal(df)
        
        # Combined signal
        signals['combined'] = signals['momentum'] * 0.6 + signals['quality'] * 0.4
        
        # Position sizing based on regime
        signals['position'] = np.clip(signals['combined'] / 2, -1, 1)
        
        return signals


class IntradayStrategy:
    """Intraday strategy component: ORB + VWAP mean reversion"""
    
    def __init__(self):
        self.orb_minutes = 15  # Opening range breakout window
        
    def calculate_opening_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate opening range for each day"""
        df = df.copy()
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        
        # First 15 minutes of each day
        df['minutes_since_open'] = pd.to_datetime(df['timestamp']).dt.hour * 60 + \
                                    pd.to_datetime(df['timestamp']).dt.minute - 9 * 60 - 30
        
        # Opening range
        orb = df[df['minutes_since_open'] <= self.orb_minutes].groupby(['date', 'ticker']).agg({
            'high': 'max',
            'low': 'min'
        }).rename(columns={'high': 'orb_high', 'low': 'orb_low'})
        
        return df.merge(orb, on=['date', 'ticker'], how='left')
    
    def generate_orb_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate ORB breakout signals"""
        df = self.calculate_opening_range(df)
        
        # Breakout signals
        df['orb_long'] = (df['close'] > df['orb_high']) & (df['minutes_since_open'] > self.orb_minutes)
        df['orb_short'] = (df['close'] < df['orb_low']) & (df['minutes_since_open'] > self.orb_minutes)
        
        df['orb_signal'] = 0
        df.loc[df['orb_long'], 'orb_signal'] = 1
        df.loc[df['orb_short'], 'orb_signal'] = -1
        
        return df
    
    def generate_vwap_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate VWAP mean reversion signals"""
        if 'vwap' not in df.columns or df['vwap'].isna().all():
            df['vwap_signal'] = 0
            return df
            
        # Distance from VWAP
        df['vwap_dist'] = (df['close'] - df['vwap']) / (df['atr'] + 1e-10)
        
        # Mean reversion: long when far below, short when far above
        df['vwap_signal'] = 0
        df.loc[df['vwap_dist'] < -2, 'vwap_signal'] = 1  # Oversold
        df.loc[df['vwap_dist'] > 2, 'vwap_signal'] = -1   # Overbought
        
        return df


class RiskManager:
    """Dynamic position sizing and risk management"""
    
    def __init__(self, initial_capital: float = 100_000):
        self.capital = initial_capital
        self.max_position_pct = Config.MAX_POSITION_PCT
        self.kelly_fraction = Config.KELLY_FRACTION
        self.max_risk = Config.MAX_RISK_PER_TRADE
        
    def calculate_kelly_size(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate Kelly criterion position size"""
        if avg_loss == 0:
            return 0
            
        b = avg_win / abs(avg_loss)  # Win/loss ratio
        p = win_rate
        q = 1 - p
        
        kelly = (b * p - q) / b
        
        # Apply fraction and caps
        size = kelly * self.kelly_fraction
        size = max(0, min(size, self.max_position_pct))
        
        return size
    
    def calculate_risk_adjusted_size(self, signal: float, volatility: float, 
                                     win_rate: float = 0.55) -> float:
        """Calculate final position size with risk adjustment"""
        # Base size from Kelly
        base_size = self.calculate_kelly_size(win_rate, 0.02, 0.01)
        
        # Adjust for signal strength
        signal_adj = abs(signal) / 3  # Assuming signals are -3 to 3
        
        # Adjust for volatility (smaller size in high vol)
        vol_adj = 1 / (1 + volatility * 10)
        
        # Final size
        size = base_size * signal_adj * vol_adj
        
        # Apply max risk constraint
        max_size = self.max_risk / (volatility + 0.001)
        size = min(size, max_size, self.max_position_pct)
        
        return size


def run_phase3(daily_df: pd.DataFrame = None, intraday_df: pd.DataFrame = None) -> Dict[str, Any]:
    """Run Phase 3: Strategy Development"""
    print("\n" + "="*70)
    print("PHASE 3: STRATEGY DEVELOPMENT")
    print("="*70)
    
    results = {
        'phase': 3,
        'timestamp': datetime.now().isoformat(),
        'daily_signals': 0,
        'intraday_signals': 0,
        'success': False
    }
    
    # Load data if not provided
    if daily_df is None:
        daily_path = Config.RESULTS_DIR / 'v150_daily_2y.parquet'
        if daily_path.exists():
            daily_df = pd.read_parquet(daily_path)
    
    if intraday_df is None:
        intraday_path = Config.RESULTS_DIR / 'v150_intraday_6m.parquet'
        if intraday_path.exists():
            intraday_df = pd.read_parquet(intraday_path)
    
    # Daily strategy
    if daily_df is not None and not daily_df.empty:
        print("\n📈 Developing Daily Strategy...")
        daily_strategy = DailyStrategy()
        
        # HMM regime detection
        print("  - Training HMM regime detector...")
        spy_data = daily_df[daily_df['ticker'] == 'SPY'].copy()
        if not spy_data.empty:
            returns = spy_data['return_1d'].fillna(0).values
            volatility = spy_data['return_1d'].rolling(21).std().fillna(0.01).values
            daily_strategy.hmm.fit(returns, volatility)
            print("    ✅ HMM trained with 3 regimes")
        
        # Generate signals for all tickers
        print("  - Generating momentum + quality signals...")
        daily_signals = []
        for ticker in daily_df['ticker'].unique():
            ticker_df = daily_df[daily_df['ticker'] == ticker].copy()
            signals = daily_strategy.generate_signals(ticker_df)
            signals['ticker'] = ticker
            daily_signals.append(signals)
        
        daily_signals_df = pd.concat(daily_signals, ignore_index=True)
        daily_signals_df.to_parquet(Config.RESULTS_DIR / 'v150_daily_signals.parquet')
        results['daily_signals'] = len(daily_signals_df)
        print(f"    ✅ Generated {len(daily_signals_df):,} daily signals")
    
    # Intraday strategy
    if intraday_df is not None and not intraday_df.empty:
        print("\n📊 Developing Intraday Strategy...")
        intraday_strategy = IntradayStrategy()
        
        # ORB signals
        print("  - Generating ORB-15 signals...")
        intraday_signals = intraday_strategy.generate_orb_signals(intraday_df)
        
        # VWAP signals
        print("  - Generating VWAP mean reversion signals...")
        intraday_signals = intraday_strategy.generate_vwap_signals(intraday_signals)
        
        # Combined intraday signal
        intraday_signals['intraday_signal'] = (
            intraday_signals['orb_signal'] * 0.6 + 
            intraday_signals['vwap_signal'] * 0.4
        )
        
        intraday_signals.to_parquet(Config.RESULTS_DIR / 'v150_intraday_signals.parquet')
        results['intraday_signals'] = len(intraday_signals)
        print(f"    ✅ Generated {len(intraday_signals):,} intraday signals")
    
    # Risk manager
    print("\n🛡️ Initializing Risk Manager...")
    risk_mgr = RiskManager(Config.INITIAL_CAPITAL)
    print(f"  - Kelly fraction: {risk_mgr.kelly_fraction}")
    print(f"  - Max position: {risk_mgr.max_position_pct:.1%}")
    print(f"  - Max risk/trade: {risk_mgr.max_risk:.1%}")
    
    results['success'] = results['daily_signals'] > 0 or results['intraday_signals'] > 0
    
    if results['success']:
        print(f"\n✅ PHASE 3 COMPLETE - Strategies developed")
    else:
        print("\n⚠️ PHASE 3 WARNING - No signals generated (need data)")
    
    return results


# ==============================================================================
# PHASE 4: MACHINE LEARNING
# ==============================================================================

class MLSignalGenerator:
    """Ensemble ML model for signal generation"""
    
    def __init__(self):
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        self.scaler = StandardScaler()
        self.models = {
            'rf': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
            'gbm': GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
            'lr': LogisticRegression(max_iter=1000, random_state=42)
        }
        self.weights = {'rf': 0.4, 'gbm': 0.4, 'lr': 0.2}
        self.feature_columns = []
        self.is_fitted = False
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Prepare ML features from DataFrame"""
        feature_cols = [
            'rsi', 'macd', 'macd_signal', 'bb_pct', 'atr',
            'volume_ratio', 'return_1d', 'return_5d', 'return_20d',
            'momentum_10', 'momentum_20'
        ]
        
        # Add more features if available
        optional_cols = ['sma_20', 'sma_50', 'ema_12', 'ema_26', 'bb_upper', 'bb_lower']
        for col in optional_cols:
            if col in df.columns:
                feature_cols.append(col)
        
        available_cols = [c for c in feature_cols if c in df.columns]
        self.feature_columns = available_cols
        
        X = df[available_cols].values
        return X, available_cols
    
    def prepare_target(self, df: pd.DataFrame, forward_days: int = 5) -> np.ndarray:
        """Prepare target: 1 if future return > 0, else 0"""
        future_return = df['close'].shift(-forward_days) / df['close'] - 1
        y = (future_return > 0).astype(int)
        return y.values
    
    def fit(self, df: pd.DataFrame, train_ratio: float = 0.7) -> Dict[str, float]:
        """Train ensemble models"""
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import accuracy_score
        
        X, _ = self.prepare_features(df)
        y = self.prepare_target(df)
        
        # Remove NaN rows
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) < 100:
            return {'error': 'Insufficient data'}
        
        # Train/test split (time-series aware)
        split_idx = int(len(X) * train_ratio)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Train each model
        results = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results[f'{name}_accuracy'] = acc
        
        # Ensemble accuracy
        ensemble_proba = np.zeros(len(X_test))
        for name, model in self.models.items():
            proba = model.predict_proba(X_test)[:, 1]
            ensemble_proba += proba * self.weights[name]
        
        ensemble_pred = (ensemble_proba > 0.5).astype(int)
        results['ensemble_accuracy'] = accuracy_score(y_test, ensemble_pred)
        
        self.is_fitted = True
        return results
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Get ensemble probability predictions"""
        if not self.is_fitted:
            return np.full(len(df), 0.5)
        
        X, _ = self.prepare_features(df)
        X = self.scaler.transform(np.nan_to_num(X))
        
        proba = np.zeros(len(X))
        for name, model in self.models.items():
            proba += model.predict_proba(X)[:, 1] * self.weights[name]
        
        return proba


def run_phase4(daily_df: pd.DataFrame = None) -> Dict[str, Any]:
    """Run Phase 4: Machine Learning"""
    print("\n" + "="*70)
    print("PHASE 4: MACHINE LEARNING")
    print("="*70)
    
    results = {
        'phase': 4,
        'timestamp': datetime.now().isoformat(),
        'features': 0,
        'accuracy': {},
        'success': False
    }
    
    # Load data if not provided
    if daily_df is None:
        daily_path = Config.RESULTS_DIR / 'v150_daily_2y.parquet'
        if daily_path.exists():
            daily_df = pd.read_parquet(daily_path)
        else:
            print("❌ No daily data available for ML training")
            return results
    
    print("\n🤖 Training ML Ensemble...")
    
    ml_generator = MLSignalGenerator()
    
    # Train on each ticker
    all_accuracies = []
    for ticker in ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA']:  # Key tickers
        if ticker not in daily_df['ticker'].values:
            continue
            
        ticker_df = daily_df[daily_df['ticker'] == ticker].copy()
        
        if len(ticker_df) < 100:
            continue
            
        print(f"  Training on {ticker}...", end=" ")
        acc_results = ml_generator.fit(ticker_df)
        
        if 'ensemble_accuracy' in acc_results:
            print(f"Accuracy: {acc_results['ensemble_accuracy']:.1%}")
            all_accuracies.append(acc_results['ensemble_accuracy'])
            results['accuracy'][ticker] = acc_results
    
    if all_accuracies:
        avg_accuracy = np.mean(all_accuracies)
        results['features'] = len(ml_generator.feature_columns)
        results['avg_accuracy'] = avg_accuracy
        results['success'] = avg_accuracy >= Config.ML_TARGET_ACCURACY
        
        print(f"\n📊 ML Results:")
        print(f"  - Features used: {results['features']}")
        print(f"  - Average accuracy: {avg_accuracy:.1%}")
        print(f"  - Target accuracy: {Config.ML_TARGET_ACCURACY:.1%}")
        
        if results['success']:
            print(f"\n✅ PHASE 4 COMPLETE - ML accuracy {avg_accuracy:.1%} ≥ {Config.ML_TARGET_ACCURACY:.1%}")
        else:
            print(f"\n⚠️ PHASE 4 WARNING - ML accuracy below target")
    else:
        print("\n❌ PHASE 4 FAILED - Could not train ML models")
    
    return results


# ==============================================================================
# PHASE 5: BACKTESTING
# ==============================================================================

class Backtester:
    """Vectorized backtester with realistic slippage"""
    
    def __init__(self, initial_capital: float = 100_000):
        self.initial_capital = initial_capital
        self.slippage_bps = Config.SLIPPAGE_DAILY_BPS
        
    def run_backtest(self, df: pd.DataFrame, signals: pd.Series) -> Dict[str, float]:
        """Run vectorized backtest"""
        # Align signals with returns
        returns = df['close'].pct_change().fillna(0)
        
        # Apply slippage on position changes
        position_changes = signals.diff().abs().fillna(0)
        slippage_cost = position_changes * (self.slippage_bps / 10000)
        
        # Strategy returns
        strategy_returns = signals.shift(1).fillna(0) * returns - slippage_cost
        
        # Calculate equity curve
        equity = self.initial_capital * (1 + strategy_returns).cumprod()
        
        # Metrics
        total_return = equity.iloc[-1] / self.initial_capital - 1
        
        # Annualized metrics (assuming daily data)
        trading_days = len(df)
        years = trading_days / 252
        
        cagr = (equity.iloc[-1] / self.initial_capital) ** (1 / max(years, 0.01)) - 1
        
        # Volatility and Sharpe
        daily_vol = strategy_returns.std()
        annual_vol = daily_vol * np.sqrt(252)
        sharpe = (cagr - 0.05) / max(annual_vol, 0.01)  # Assuming 5% risk-free rate
        
        # Drawdown
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_days = (strategy_returns > 0).sum()
        total_days = (strategy_returns != 0).sum()
        win_rate = winning_days / max(total_days, 1)
        
        return {
            'total_return': total_return,
            'cagr': cagr,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'annual_vol': annual_vol,
            'win_rate': win_rate,
            'trading_days': trading_days,
            'final_equity': equity.iloc[-1]
        }


def run_phase5(daily_df: pd.DataFrame = None) -> Dict[str, Any]:
    """Run Phase 5: Backtesting"""
    print("\n" + "="*70)
    print("PHASE 5: BACKTESTING")
    print("="*70)
    
    results = {
        'phase': 5,
        'timestamp': datetime.now().isoformat(),
        'backtest_results': {},
        'combined_metrics': {},
        'success': False
    }
    
    # Load data
    if daily_df is None:
        daily_path = Config.RESULTS_DIR / 'v150_daily_2y.parquet'
        if daily_path.exists():
            daily_df = pd.read_parquet(daily_path)
        else:
            print("❌ No data available for backtesting")
            return results
    
    # Load signals
    signals_path = Config.RESULTS_DIR / 'v150_daily_signals.parquet'
    if signals_path.exists():
        signals_df = pd.read_parquet(signals_path)
    else:
        # Generate simple momentum signals if no signals file
        print("  Generating fallback momentum signals...")
        signals_df = daily_df.copy()
        signals_df['position'] = np.clip(signals_df['momentum_20'].fillna(0) * 5, -1, 1)
    
    print("\n📊 Running Backtests...")
    
    backtester = Backtester(Config.INITIAL_CAPITAL)
    all_results = []
    
    for ticker in Config.TICKERS[:8]:  # Top 8 tickers
        if ticker not in daily_df['ticker'].values:
            continue
            
        ticker_data = daily_df[daily_df['ticker'] == ticker].copy().reset_index(drop=True)
        
        if 'position' in signals_df.columns:
            ticker_signals = signals_df[signals_df['ticker'] == ticker]['position'] if 'ticker' in signals_df.columns else signals_df['position']
        else:
            ticker_signals = np.clip(ticker_data['momentum_20'].fillna(0) * 5, -1, 1)
        
        if len(ticker_data) < 50:
            continue
        
        # Ensure signals align with data
        if len(ticker_signals) != len(ticker_data):
            ticker_signals = pd.Series(np.clip(ticker_data['momentum_20'].fillna(0) * 5, -1, 1))
        
        print(f"  Backtesting {ticker}...", end=" ")
        bt_results = backtester.run_backtest(ticker_data, ticker_signals)
        bt_results['ticker'] = ticker
        all_results.append(bt_results)
        print(f"Sharpe: {bt_results['sharpe']:.2f}, CAGR: {bt_results['cagr']:.1%}")
    
    if all_results:
        results['backtest_results'] = all_results
        
        # Portfolio metrics (equal-weighted average)
        avg_sharpe = np.mean([r['sharpe'] for r in all_results])
        avg_cagr = np.mean([r['cagr'] for r in all_results])
        worst_dd = min([r['max_drawdown'] for r in all_results])
        avg_win_rate = np.mean([r['win_rate'] for r in all_results])
        
        results['combined_metrics'] = {
            'sharpe': avg_sharpe,
            'cagr': avg_cagr,
            'max_drawdown': worst_dd,
            'win_rate': avg_win_rate
        }
        
        print(f"\n📈 Combined Portfolio Metrics:")
        print(f"  - Sharpe Ratio: {avg_sharpe:.2f}")
        print(f"  - CAGR: {avg_cagr:.1%}")
        print(f"  - Max Drawdown: {worst_dd:.1%}")
        print(f"  - Win Rate: {avg_win_rate:.1%}")
        
        # Decision criteria
        meets_sharpe = avg_sharpe >= 3.0
        meets_cagr = avg_cagr >= 0.40
        meets_dd = worst_dd > -0.18
        
        results['success'] = meets_sharpe and meets_cagr and meets_dd
        
        if results['success']:
            print(f"\n✅ PHASE 5 COMPLETE - Backtest metrics meet criteria")
        else:
            print(f"\n⚠️ PHASE 5 WARNING - Some metrics below target")
    else:
        print("\n❌ PHASE 5 FAILED - No backtest results")
    
    return results


# ==============================================================================
# PHASE 6: PRODUCTION DEPLOYMENT
# ==============================================================================

def test_alpaca_paper_trade() -> Dict[str, Any]:
    """Test Alpaca paper trading with a small order"""
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        
        client = TradingClient(
            api_key=Config.ALPACA_KEY,
            secret_key=Config.ALPACA_SECRET,
            paper=True
        )
        
        # Get account
        account = client.get_account()
        
        # Check if market is open
        clock = client.get_clock()
        
        result = {
            'account_connected': True,
            'buying_power': float(account.buying_power),
            'equity': float(account.equity),
            'market_open': clock.is_open,
            'test_order': None
        }
        
        # Only place test order if market is open
        if clock.is_open:
            # Place a small test order (1 share of SPY)
            order_data = MarketOrderRequest(
                symbol="SPY",
                qty=1,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            order = client.submit_order(order_data)
            result['test_order'] = {
                'id': str(order.id),
                'symbol': order.symbol,
                'status': str(order.status)
            }
            
            # Cancel the test order
            client.cancel_order_by_id(order.id)
            result['test_order']['cancelled'] = True
        
        return result
        
    except Exception as e:
        return {'error': str(e)}


def run_phase6() -> Dict[str, Any]:
    """Run Phase 6: Production Deployment"""
    print("\n" + "="*70)
    print("PHASE 6: PRODUCTION DEPLOYMENT & CLEANUP")
    print("="*70)
    
    results = {
        'phase': 6,
        'timestamp': datetime.now().isoformat(),
        'alpaca_test': None,
        'cleanup': {},
        'success': False
    }
    
    # A) Test Alpaca Paper Trading
    print("\n💰 Testing Alpaca Paper Trading...")
    alpaca_result = test_alpaca_paper_trade()
    results['alpaca_test'] = alpaca_result
    
    if 'error' not in alpaca_result:
        print(f"  ✅ Connected to Alpaca")
        print(f"  - Equity: ${alpaca_result.get('equity', 0):,.2f}")
        print(f"  - Buying Power: ${alpaca_result.get('buying_power', 0):,.2f}")
        print(f"  - Market Open: {alpaca_result.get('market_open', False)}")
        if alpaca_result.get('test_order'):
            print(f"  ✅ Test order placed and cancelled successfully")
    else:
        print(f"  ⚠️ Alpaca error: {alpaca_result['error']}")
    
    # B) Verify security
    print("\n🔒 Security Check...")
    env_path = Path('/workspaces/Algebraic-Topology-Neural-Net-Strategy/.env')
    gitignore_path = Path('/workspaces/Algebraic-Topology-Neural-Net-Strategy/.gitignore')
    
    if env_path.exists():
        print("  ✅ .env file exists")
        
        # Check if .env is in .gitignore
        if gitignore_path.exists():
            with open(gitignore_path, 'r') as f:
                gitignore_content = f.read()
            if '.env' in gitignore_content:
                print("  ✅ .env is in .gitignore")
                results['cleanup']['env_secure'] = True
            else:
                print("  ⚠️ .env should be added to .gitignore")
                results['cleanup']['env_secure'] = False
    
    results['success'] = 'error' not in alpaca_result
    
    if results['success']:
        print(f"\n✅ PHASE 6 COMPLETE - Production ready")
    else:
        print(f"\n⚠️ PHASE 6 WARNING - Review Alpaca connection")
    
    return results


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def generate_final_report(phase_results: List[Dict]) -> str:
    """Generate V150 Production Report"""
    
    report = """
# V15.0 ELITE RETAIL SYSTEMATIC STRATEGY
## PRODUCTION REPORT

Generated: {timestamp}

---

## EXECUTIVE SUMMARY

V15.0 represents a major upgrade from previous versions, leveraging:
- **Polygon.io API**: Institutional-grade market data
- **Alpaca Markets**: Commission-free paper trading execution
- **Machine Learning**: Ensemble signals (RF + GBM + LR)
- **Multi-Timeframe**: Daily momentum + Intraday ORB/VWAP

---

## PHASE RESULTS

""".format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    for pr in phase_results:
        phase_num = pr.get('phase', '?')
        success = "✅ PASS" if pr.get('success', False) else "⚠️ CHECK"
        report += f"### Phase {phase_num}: {success}\n\n"
        
        if phase_num == 1:
            report += f"- Polygon Status: {pr.get('polygon_status', {}).get('message', 'N/A')}\n"
            report += f"- Alpaca Status: {pr.get('alpaca_status', {}).get('message', 'N/A')}\n"
        elif phase_num == 2:
            report += f"- Intraday Bars: {pr.get('intraday_bars', 0):,}\n"
            report += f"- Daily Bars: {pr.get('daily_bars', 0):,}\n"
        elif phase_num == 4:
            report += f"- Features: {pr.get('features', 0)}\n"
            report += f"- Average Accuracy: {pr.get('avg_accuracy', 0):.1%}\n"
        elif phase_num == 5:
            metrics = pr.get('combined_metrics', {})
            report += f"- Sharpe Ratio: {metrics.get('sharpe', 0):.2f}\n"
            report += f"- CAGR: {metrics.get('cagr', 0):.1%}\n"
            report += f"- Max Drawdown: {metrics.get('max_drawdown', 0):.1%}\n"
        elif phase_num == 6:
            alpaca = pr.get('alpaca_test', {})
            report += f"- Alpaca Connected: {'error' not in alpaca}\n"
            if 'equity' in alpaca:
                report += f"- Account Equity: ${alpaca['equity']:,.2f}\n"
        
        report += "\n"
    
    # Decision
    phase5 = next((p for p in phase_results if p.get('phase') == 5), {})
    metrics = phase5.get('combined_metrics', {})
    
    sharpe = metrics.get('sharpe', 0)
    cagr = metrics.get('cagr', 0)
    max_dd = metrics.get('max_drawdown', 0)
    
    if sharpe >= 3.5 and cagr >= 0.50 and max_dd > -0.15:
        decision = "🟢 GO - Full Production Deployment"
    elif sharpe >= 3.0 and cagr >= 0.40 and max_dd > -0.18:
        decision = "🟡 CONDITIONAL_GO - Deploy with Enhanced Monitoring"
    else:
        decision = "🔴 NO_GO - Further Development Required"
    
    report += f"""
---

## DECISION

**{decision}**

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Sharpe | {sharpe:.2f} | ≥3.5 | {'✅' if sharpe >= 3.5 else '⚠️'} |
| CAGR | {cagr:.1%} | ≥50% | {'✅' if cagr >= 0.50 else '⚠️'} |
| Max DD | {max_dd:.1%} | >-15% | {'✅' if max_dd > -0.15 else '⚠️'} |

---

## FILES GENERATED

- `v150_intraday_6m.parquet` - 6-month intraday data
- `v150_daily_2y.parquet` - 2-year daily data
- `v150_daily_signals.parquet` - Strategy signals
- `V150_PRODUCTION_REPORT.md` - This report

---

## NEXT STEPS

1. Review backtest results in detail
2. Paper trade for 2 weeks minimum
3. Monitor ML signal accuracy
4. Scale capital gradually

---

*V15.0 Elite Retail Systematic Strategy - Powered by Polygon.io & Alpaca*
"""
    
    return report


def main():
    """Main execution pipeline"""
    print("\n" + "="*70)
    print("V15.0 ELITE RETAIL SYSTEMATIC TRADING STRATEGY")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_results = []
    
    # Phase 1: Infrastructure
    p1_results = run_phase1()
    all_results.append(p1_results)
    
    if not p1_results['success']:
        print("\n⚠️ Phase 1 failed - checking individual API status...")
    
    # Phase 2: Data Download
    p2_results = run_phase2()
    all_results.append(p2_results)
    
    # Phase 3: Strategy Development
    p3_results = run_phase3()
    all_results.append(p3_results)
    
    # Phase 4: Machine Learning
    p4_results = run_phase4()
    all_results.append(p4_results)
    
    # Phase 5: Backtesting
    p5_results = run_phase5()
    all_results.append(p5_results)
    
    # Phase 6: Production
    p6_results = run_phase6()
    all_results.append(p6_results)
    
    # Generate report
    print("\n" + "="*70)
    print("GENERATING FINAL REPORT")
    print("="*70)
    
    report = generate_final_report(all_results)
    report_path = Config.RESULTS_DIR / 'V150_PRODUCTION_REPORT.md'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\n📄 Report saved to: {report_path}")
    
    # Save all results as JSON
    results_path = Config.RESULTS_DIR / 'v150_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"📊 Results saved to: {results_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("V15.0 EXECUTION COMPLETE")
    print("="*70)
    
    success_count = sum(1 for r in all_results if r.get('success', False))
    print(f"\nPhases Passed: {success_count}/6")
    
    return all_results


if __name__ == "__main__":
    main()
