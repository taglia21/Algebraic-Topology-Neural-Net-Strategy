"""
TDA Universe Paper Trading Engine
==================================

Full-universe trading engine using Topological Data Analysis and 
multi-factor stock selection for hedge fund deployment.

Core Strategy:
1. Multi-Factor Stock Selection (40-50 stocks from S&P 500 + NASDAQ 100)
2. TDA-based Market Regime Detection (Betti numbers, persistence)
3. Neural Network Predictions for direction
4. Trend-Following Leveraged ETF Overlay
5. Multi-Layer Risk Management (VIX, Drawdown, Regime)

Target Performance:
- CAGR: 25-35%
- Max Drawdown: < 22%
- Sharpe: > 1.0
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import time
import pickle

import numpy as np
import pandas as pd
import yfinance as yf

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.trading.alpaca_client import AlpacaClient, OrderSide, Position, Account
from src.trading.notifications import notify_rebalance_summary, notify_trade_executed, notify_regime_change, notify_error

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

class MarketRegime(Enum):
    RISK_ON = "risk_on"      # Strong bull, full allocation
    BULL = "bull"            # Bull market, high allocation
    NEUTRAL = "neutral"      # Mixed signals, reduced allocation
    BEAR = "bear"            # Bear market, minimal/inverse
    RISK_OFF = "risk_off"    # Crisis, cash/inverse


class TrendState(Enum):
    STRONG_UP = "strong_up"   # Price > SMA20 > SMA50 > SMA200
    UP = "up"                  # Price > SMA50 > SMA200
    NEUTRAL = "neutral"        # Mixed
    DOWN = "down"              # Price < SMA50
    STRONG_DOWN = "strong_down"  # Price < SMA20 < SMA50 < SMA200


@dataclass
class TDASignals:
    """TDA-derived market signals."""
    betti_0: int = 0           # Connected components
    betti_1: int = 0           # Cycles/loops
    turbulence_index: float = 0.0
    fragmentation: float = 0.0
    regime: MarketRegime = MarketRegime.NEUTRAL
    confidence: float = 0.5


@dataclass
class FactorScores:
    """Multi-factor scores for a stock."""
    ticker: str
    momentum_12m: float = 0.0    # 12-month momentum, skip last month
    momentum_3m: float = 0.0     # 3-month momentum
    volatility_adj: float = 0.0  # Return / volatility
    relative_strength: float = 0.0  # vs SPY
    volume_score: float = 0.0    # Liquidity
    composite: float = 0.0       # Weighted average


@dataclass
class PortfolioState:
    """Current portfolio state."""
    timestamp: str
    equity: float
    cash: float
    positions: Dict[str, float]  # ticker -> market value
    weights: Dict[str, float]    # ticker -> weight
    regime: MarketRegime
    trend: TrendState
    vix_level: float
    drawdown: float
    leverage_multiplier: float


@dataclass
class TradeRecord:
    """Record of executed trade."""
    timestamp: str
    symbol: str
    side: str
    qty: float
    price: float
    value: float
    order_id: str
    regime: str
    reason: str


# =============================================================================
# UNIVERSE CONFIGURATION - FULL 700+ STOCK UNIVERSE
# =============================================================================

# Import MEGA universe for MAXIMUM opportunities - PRINT CASH!
try:
    from src.trading.mega_universe import (
        MEGA_UNIVERSE, SP500, ETFS, UNIVERSE_STATS,
        get_mega_universe, get_leveraged_bull_etfs, get_leveraged_bear_etfs
    )
    STOCK_UNIVERSE = MEGA_UNIVERSE
    logger.info(f"🚀 MEGA UNIVERSE LOADED: {len(STOCK_UNIVERSE)} tradeable symbols!")
    logger.info(f"   PRINTING CASH LIKE RENAISSANCE TECHNOLOGIES!")
except ImportError:
    # Fallback to old full_universe
    try:
        from src.trading.full_universe import (
            FULL_UNIVERSE, SP500_TICKERS, ETFS, get_leveraged_bull_etfs, get_leveraged_bear_etfs
        )
        STOCK_UNIVERSE = FULL_UNIVERSE
        logger.info(f"Loaded full universe with {len(STOCK_UNIVERSE)} symbols")
    except ImportError:
        logger.warning("No universe module found, using minimal universe")
        STOCK_UNIVERSE = [
            'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'AMZN', 'TSLA', 'AMD', 'AVGO', 'CRM',
            'JPM', 'V', 'MA', 'BAC', 'GS', 'UNH', 'JNJ', 'LLY', 'PFE', 'ABBV',
            'XOM', 'CVX', 'COP', 'CAT', 'HON', 'UPS', 'HD', 'MCD', 'COST', 'WMT',
        ]

SECTOR_MAP = {}

# Import Adaptive Learning Engine
try:
    from src.trading.adaptive_learning_engine import AdaptiveLearningEngine
    ADAPTIVE_LEARNING_AVAILABLE = True
    logger.info("🧠 Adaptive Learning Engine LOADED - ML/RL enabled!")
except ImportError as e:
    logger.warning(f"Adaptive Learning Engine not available: {e}")
    ADAPTIVE_LEARNING_AVAILABLE = False

# Leveraged ETFs for amplification (3x)
LEVERAGED_ETFS = {
    'TQQQ': 0.35,  # Nasdaq 3x
    'SPXL': 0.25,  # S&P 500 3x
    'SOXL': 0.20,  # Semiconductors 3x
    'TECL': 0.10,  # Technology 3x
    'FNGU': 0.10,  # FAANG 3x
}

INVERSE_ETFS = {
    'SQQQ': 0.35,  # Nasdaq -3x
    'SPXU': 0.25,  # S&P 500 -3x
    'SOXS': 0.20,  # Semiconductors -3x
    'TECS': 0.10,  # Technology -3x
    'FNGD': 0.10,  # FAANG -3x
}


# =============================================================================
# TDA ENGINE (Simplified for production)
# =============================================================================

class TDAMarketAnalyzer:
    """
    Simplified TDA market analysis for production.
    
    Uses correlation-based distance matrices to compute:
    - Market fragmentation (Betti-0 proxy)
    - Cyclicity (correlation clustering)
    - Turbulence index
    """
    
    def __init__(self, lookback: int = 30):
        self.lookback = lookback
        self.history = []
    
    def analyze(self, returns_df: pd.DataFrame) -> TDASignals:
        """
        Analyze market topology from returns.
        
        Args:
            returns_df: DataFrame of daily returns (columns = tickers)
            
        Returns:
            TDASignals with market regime classification
        """
        if len(returns_df) < self.lookback:
            return TDASignals(regime=MarketRegime.NEUTRAL)
        
        # Use last N days
        recent = returns_df.iloc[-self.lookback:]
        
        # Compute correlation matrix
        corr = recent.corr()
        
        # Convert to distance (1 - correlation)
        dist = 1 - corr.abs()
        
        # Compute fragmentation (number of weakly correlated clusters)
        threshold = 0.5  # Correlation < 0.5 considered disconnected
        n_disconnected = (corr.abs() < threshold).sum().sum() / 2
        fragmentation = n_disconnected / (len(corr) ** 2) * 2
        
        # Compute turbulence (Mahalanobis-style)
        mean_ret = recent.mean()
        cov = recent.cov()
        try:
            inv_cov = np.linalg.pinv(cov.values)
            today_ret = returns_df.iloc[-1].values
            diff = today_ret - mean_ret.values
            turbulence = float(np.sqrt(diff @ inv_cov @ diff))
        except:
            turbulence = 0.0
        
        # Normalize turbulence to 0-100 scale
        turbulence_index = min(100, turbulence * 10)
        
        # Estimate Betti numbers (simplified)
        betti_0 = int(fragmentation * 10) + 1  # Higher = more fragmented
        betti_1 = max(0, int((1 - fragmentation) * 5))  # Higher = more connected
        
        # Classify regime
        if turbulence_index > 90:
            regime = MarketRegime.RISK_OFF
            confidence = 0.8
        elif turbulence_index > 70:
            regime = MarketRegime.BEAR
            confidence = 0.6
        elif fragmentation > 0.7:
            regime = MarketRegime.NEUTRAL
            confidence = 0.5
        elif fragmentation < 0.5 and turbulence_index < 40:
            regime = MarketRegime.RISK_ON
            confidence = 0.8
        else:
            regime = MarketRegime.BULL
            confidence = 0.7
        
        return TDASignals(
            betti_0=betti_0,
            betti_1=betti_1,
            turbulence_index=turbulence_index,
            fragmentation=fragmentation,
            regime=regime,
            confidence=confidence,
        )


# =============================================================================
# MULTI-FACTOR ENGINE
# =============================================================================

class MultiFactorEngine:
    """
    Multi-factor stock ranking engine.
    
    Factors:
    - Momentum (40%): 12-month return, skip last month
    - Quality/Vol-Adjusted (25%): Return / volatility
    - Relative Strength (20%): vs SPY
    - Liquidity (15%): Average dollar volume
    """
    
    WEIGHTS = {
        'momentum': 0.40,
        'vol_adjusted': 0.25,
        'relative_strength': 0.20,
        'liquidity': 0.15,
    }
    
    def __init__(self, min_history: int = 252):
        self.min_history = min_history
    
    def compute_factor_scores(
        self,
        price_data: Dict[str, pd.DataFrame],
        spy_data: pd.DataFrame,
    ) -> List[FactorScores]:
        """
        Compute factor scores for all stocks.
        
        Args:
            price_data: Dict of ticker -> DataFrame with 'Close' column
            spy_data: SPY price data for relative strength
            
        Returns:
            List of FactorScores, sorted by composite score
        """
        scores = []
        
        for ticker, df in price_data.items():
            if len(df) < self.min_history:
                continue
            
            try:
                close = df['Close']
                
                # Momentum (12 month, skip last month)
                if len(close) >= 252:
                    momentum_12m = close.iloc[-21] / close.iloc[-252] - 1
                else:
                    momentum_12m = close.iloc[-21] / close.iloc[0] - 1
                
                # 3-month momentum
                if len(close) >= 63:
                    momentum_3m = close.iloc[-1] / close.iloc[-63] - 1
                else:
                    momentum_3m = 0.0
                
                # Volatility-adjusted return
                returns = close.pct_change().dropna()
                vol = returns.iloc[-60:].std() * np.sqrt(252)
                annual_return = (close.iloc[-1] / close.iloc[-252] - 1) if len(close) >= 252 else momentum_12m
                vol_adjusted = annual_return / vol if vol > 0 else 0
                
                # Relative strength vs SPY
                spy_close = spy_data['Close']
                if len(spy_close) >= 252 and len(close) >= 252:
                    stock_ret = close.iloc[-1] / close.iloc[-252] - 1
                    spy_ret = spy_close.iloc[-1] / spy_close.iloc[-252] - 1
                    relative_strength = stock_ret - spy_ret
                else:
                    relative_strength = 0.0
                
                # Liquidity (log of avg dollar volume)
                if 'Volume' in df.columns:
                    dollar_vol = (df['Close'] * df['Volume']).iloc[-20:].mean()
                    volume_score = np.log10(max(dollar_vol, 1))
                else:
                    volume_score = 8.0  # Default assumption
                
                scores.append(FactorScores(
                    ticker=ticker,
                    momentum_12m=momentum_12m,
                    momentum_3m=momentum_3m,
                    volatility_adj=vol_adjusted,
                    relative_strength=relative_strength,
                    volume_score=volume_score,
                    composite=0.0,  # Computed after z-scoring
                ))
            except Exception as e:
                logger.warning(f"Failed to compute factors for {ticker}: {e}")
                continue
        
        if not scores:
            return []
        
        # Z-score normalize each factor
        df = pd.DataFrame([asdict(s) for s in scores])
        
        for col in ['momentum_12m', 'volatility_adj', 'relative_strength', 'volume_score']:
            if df[col].std() > 0:
                df[col + '_z'] = (df[col] - df[col].mean()) / df[col].std()
            else:
                df[col + '_z'] = 0
        
        # Composite score
        df['composite'] = (
            self.WEIGHTS['momentum'] * df['momentum_12m_z'] +
            self.WEIGHTS['vol_adjusted'] * df['volatility_adj_z'] +
            self.WEIGHTS['relative_strength'] * df['relative_strength_z'] +
            self.WEIGHTS['liquidity'] * df['volume_score_z']
        )
        
        # Sort by composite
        df = df.sort_values('composite', ascending=False)
        
        # Convert back to FactorScores
        result = []
        for _, row in df.iterrows():
            result.append(FactorScores(
                ticker=row['ticker'],
                momentum_12m=row['momentum_12m'],
                momentum_3m=row['momentum_3m'],
                volatility_adj=row['volatility_adj'],
                relative_strength=row['relative_strength'],
                volume_score=row['volume_score'],
                composite=row['composite'],
            ))
        
        return result


# =============================================================================
# TREND ANALYZER
# =============================================================================

class TrendAnalyzer:
    """Analyze market trend using SPY moving averages."""
    
    def __init__(self):
        self.sma_periods = [20, 50, 200]
    
    def analyze_trend(self, spy_prices: pd.Series) -> Tuple[TrendState, Dict[str, float]]:
        """
        Analyze SPY trend.
        
        Returns:
            Tuple of (TrendState, dict with SMA values)
        """
        if len(spy_prices) < 200:
            return TrendState.NEUTRAL, {}
        
        current = spy_prices.iloc[-1]
        sma20 = spy_prices.rolling(20).mean().iloc[-1]
        sma50 = spy_prices.rolling(50).mean().iloc[-1]
        sma200 = spy_prices.rolling(200).mean().iloc[-1]
        
        smas = {
            'current': current,
            'sma20': sma20,
            'sma50': sma50,
            'sma200': sma200,
        }
        
        # Classify trend
        if current > sma20 > sma50 > sma200:
            return TrendState.STRONG_UP, smas
        elif current > sma50 > sma200:
            return TrendState.UP, smas
        elif current < sma20 < sma50 < sma200:
            return TrendState.STRONG_DOWN, smas
        elif current < sma50:
            return TrendState.DOWN, smas
        else:
            return TrendState.NEUTRAL, smas


# =============================================================================
# RISK MANAGER
# =============================================================================

class RiskManager:
    """Multi-layer risk management."""
    
    def __init__(self):
        self.max_drawdown = 0.0
        self.peak_equity = 0.0
        self.current_drawdown = 0.0
    
    def update_drawdown(self, current_equity: float) -> float:
        """Update drawdown tracking."""
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        if self.peak_equity > 0:
            self.current_drawdown = 1 - (current_equity / self.peak_equity)
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        
        return self.current_drawdown
    
    def get_trend_multiplier(self, trend: TrendState) -> float:
        """Get allocation multiplier based on trend."""
        multipliers = {
            TrendState.STRONG_UP: 1.0,
            TrendState.UP: 0.8,
            TrendState.NEUTRAL: 0.5,
            TrendState.DOWN: 0.25,
            TrendState.STRONG_DOWN: 0.10,
        }
        return multipliers.get(trend, 0.5)
    
    def get_vix_multiplier(self, vix: float) -> float:
        """Get allocation multiplier based on VIX."""
        if vix < 15:
            return 1.0
        elif vix < 20:
            return 0.9
        elif vix < 25:
            return 0.7
        elif vix < 30:
            return 0.5
        elif vix < 40:
            return 0.3
        else:
            return 0.15  # Crisis mode
    
    def get_drawdown_multiplier(self, drawdown: float) -> float:
        """Get allocation multiplier based on drawdown."""
        if drawdown < 0.03:
            return 1.0
        elif drawdown < 0.06:
            return 0.9
        elif drawdown < 0.10:
            return 0.75
        elif drawdown < 0.15:
            return 0.50
        elif drawdown < 0.20:
            return 0.30
        else:
            return 0.15  # Emergency mode
    
    def get_regime_multiplier(self, regime: MarketRegime) -> float:
        """Get multiplier based on TDA regime."""
        multipliers = {
            MarketRegime.RISK_ON: 1.0,
            MarketRegime.BULL: 0.95,
            MarketRegime.NEUTRAL: 0.85,
            MarketRegime.BEAR: 0.65,
            MarketRegime.RISK_OFF: 0.25,
        }
        return multipliers.get(regime, 0.5)
    
    def compute_total_multiplier(
        self,
        trend: TrendState,
        vix: float,
        drawdown: float,
        regime: MarketRegime,
    ) -> float:
        """
        Compute combined risk multiplier.
        
        All multipliers compound to create conservative positioning.
        """
        trend_mult = self.get_trend_multiplier(trend)
        vix_mult = self.get_vix_multiplier(vix)
        dd_mult = self.get_drawdown_multiplier(drawdown)
        regime_mult = self.get_regime_multiplier(regime)
        
        # Compound all multipliers (conservative)
        total = trend_mult * vix_mult * dd_mult * regime_mult
        
        # Floor at 10%, cap at 100%
        return max(0.10, min(1.0, total))


# =============================================================================
# MAIN ENGINE
# =============================================================================

class TDAPaperTradingEngine:
    """
    Full TDA Universe Paper Trading Engine.
    
    Orchestrates:
    - Data fetching
    - TDA market analysis
    - Multi-factor stock selection
    - Risk management
    - Portfolio construction
    - Trade execution via Alpaca
    """
    
    def __init__(self):
        """Initialize engine."""
        self.client = AlpacaClient()
        self.tda_analyzer = TDAMarketAnalyzer()
        self.factor_engine = MultiFactorEngine(min_history=200)
        self.trend_analyzer = TrendAnalyzer()
        self.risk_manager = RiskManager()
        
        # Initialize Adaptive Learning Engine (ML/RL)
        if ADAPTIVE_LEARNING_AVAILABLE:
            self.adaptive_engine = AdaptiveLearningEngine(models_dir="models")
            self.use_adaptive = True
            logger.info("🧠 Adaptive Learning Engine initialized!")
        else:
            self.adaptive_engine = None
            self.use_adaptive = False
            logger.warning("⚠️ Using static factor model (no ML/RL)")
        
        # Configuration - MEGA UNIVERSE MODE - 3000+ STOCKS = MAX ALPHA
        self.n_stocks = 100  # Hold 100 stocks for maximum diversification with 3000+ universe
        self.max_position_weight = 0.04  # 4% max per stock (more diversified)
        self.leveraged_etf_weight = 0.40  # 40% to leveraged ETFs - AGGRESSIVE!
        self.stock_weight = 0.58  # 58% to individual stocks (scanning 3000+ for alpha)
        self.cash_buffer = 0.02  # 2% cash buffer - FULLY DEPLOYED
        
        # State
        self.starting_capital = float(os.getenv("STARTING_CAPITAL", 100000))
        self.trade_log: List[TradeRecord] = []
        self.portfolio_history: List[PortfolioState] = []
        
        # Cache
        self.price_cache: Dict[str, pd.DataFrame] = {}
        self.cache_date: str = ""
        
        # Log universe size
        logger.info("=" * 60)
        logger.info("🚀 TDA TRADING ENGINE - RENAISSANCE STYLE - MEGA UNIVERSE 🚀")
        logger.info("=" * 60)
        logger.info(f"MEGA UNIVERSE: {len(STOCK_UNIVERSE)} tradeable symbols!")
        logger.info("PRINTING CASH LIKE THE MEDALLION FUND!")
        logger.info(f"Leveraged ETFs: {len(LEVERAGED_ETFS)} bull + {len(INVERSE_ETFS)} bear")
        logger.info(f"Target Holdings: {self.n_stocks} stocks + ETFs")
        if self.use_adaptive:
            logger.info("🧠 ADAPTIVE LEARNING: Neural Net + RL + Risk Parity ENABLED")
            status = self.adaptive_engine.get_status()
            logger.info(f"   NN Available: {status['nn_available']}")
            logger.info(f"   RL States: {status['rl_states']}")
            logger.info(f"   Factor Weights: {status['factor_weights']}")
        logger.info("=" * 60)
    
    def fetch_market_data(self) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, float]:
        """
        Fetch all required market data using efficient batch processing.
        
        For 700+ stock universe, uses yfinance batch download for efficiency.
        
        Returns:
            Tuple of (stock_data, spy_data, vix_level)
        """
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Check cache
        if self.cache_date == today and self.price_cache:
            logger.info("Using cached price data")
            spy_data = self.price_cache.get('SPY', pd.DataFrame())
            vix = self._get_vix()
            return self.price_cache, spy_data, vix
        
        universe_size = len(STOCK_UNIVERSE)
        logger.info(f"Fetching market data for {universe_size} stocks (FULL UNIVERSE)...")
        
        # All tickers to fetch
        all_tickers = list(set(STOCK_UNIVERSE + list(LEVERAGED_ETFS.keys()) + list(INVERSE_ETFS.keys()) + ['SPY', '^VIX']))
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=400)  # ~1.5 years for 252 day lookback
        
        price_data = {}
        
        # Batch download in chunks (yfinance can handle ~50-100 tickers efficiently)
        batch_size = 50
        total_batches = (len(all_tickers) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(all_tickers), batch_size):
            batch = all_tickers[batch_idx:batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1
            logger.info(f"Fetching batch {batch_num}/{total_batches}: {len(batch)} tickers")
            
            try:
                # Batch download (much faster than individual)
                batch_data = yf.download(
                    batch, 
                    start=start_date, 
                    end=end_date, 
                    progress=False,
                    group_by='ticker',
                    threads=True
                )
                
                # Extract individual ticker data
                if len(batch) == 1:
                    # Single ticker returns differently
                    ticker = batch[0]
                    if len(batch_data) >= 100:
                        price_data[ticker] = batch_data
                else:
                    for ticker in batch:
                        try:
                            if ticker in batch_data.columns.get_level_values(0):
                                df = batch_data[ticker].dropna()
                                if len(df) >= 100:
                                    price_data[ticker] = df
                        except Exception as e:
                            logger.debug(f"Skipping {ticker}: {e}")
                            
            except Exception as e:
                logger.warning(f"Batch download failed: {e}")
                # Fallback: individual downloads for this batch
                for ticker in batch:
                    try:
                        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                        if len(df) >= 100:
                            price_data[ticker] = df
                    except:
                        pass
            
            # Small delay between batches to avoid rate limits
            time.sleep(0.5)
        
        logger.info(f"Successfully fetched data for {len(price_data)} / {len(all_tickers)} tickers")
        
        # Update cache
        self.price_cache = price_data
        self.cache_date = today
        
        spy_data = price_data.get('SPY', pd.DataFrame())
        vix = self._get_vix()
        
        return price_data, spy_data, vix
    
    def _get_vix(self) -> float:
        """Get current VIX level."""
        try:
            vix_data = self.price_cache.get('^VIX')
            if vix_data is not None and len(vix_data) > 0:
                return float(vix_data['Close'].iloc[-1])
        except:
            pass
        return 20.0  # Default
    
    def compute_target_portfolio(self) -> Dict[str, float]:
        """
        Compute target portfolio weights.
        
        Uses Adaptive Learning Engine when available:
        - Neural Network predictions for direction
        - RL-based position sizing
        - Online learning for factor weights
        - Risk parity allocation
        
        Returns:
            Dict of ticker -> target weight
        """
        # Fetch data
        price_data, spy_data, vix = self.fetch_market_data()
        
        if spy_data.empty:
            logger.error("No SPY data available")
            return {}
        
        # 1. Compute TDA signals
        stock_returns = pd.DataFrame()
        for ticker in STOCK_UNIVERSE:
            if ticker in price_data:
                stock_returns[ticker] = price_data[ticker]['Close'].pct_change()
        
        tda_signals = self.tda_analyzer.analyze(stock_returns)
        logger.info(f"TDA Regime: {tda_signals.regime.value}, Turbulence: {tda_signals.turbulence_index:.1f}")
        
        # 2. Analyze trend
        trend, smas = self.trend_analyzer.analyze_trend(spy_data['Close'])
        logger.info(f"Trend: {trend.value}")
        
        # 3. Get account info and update drawdown
        account = self.client.get_account()
        drawdown = self.risk_manager.update_drawdown(account.equity)
        logger.info(f"Drawdown: {drawdown:.2%}")
        
        # 4. Compute risk multiplier
        risk_mult = self.risk_manager.compute_total_multiplier(
            trend=trend,
            vix=vix,
            drawdown=drawdown,
            regime=tda_signals.regime,
        )
        logger.info(f"Risk Multiplier: {risk_mult:.2%}")
        
        # 5. Build target portfolio
        target = {}
        
        # =========================================================
        # ADAPTIVE LEARNING PATH (ML/RL enabled)
        # =========================================================
        if self.use_adaptive and self.adaptive_engine is not None:
            logger.info("🧠 Using ADAPTIVE LEARNING for stock selection")
            
            # Get current positions and prices for learning
            current_positions = self.client.get_positions()
            current_prices = {t: price_data[t]['Close'].iloc[-1] 
                            for t in price_data if len(price_data[t]) > 0}
            current_pos_dict = {p.symbol: p.market_value for p in current_positions}
            
            # Learn from previous trades
            self.adaptive_engine.learn_from_trades(current_prices, current_pos_dict)
            
            # Train models if needed (only first run)
            if not self.adaptive_engine.nn_predictor.is_trained:
                logger.info("🧠 Training neural network on historical data...")
                self.adaptive_engine.train_models(price_data)
            
            # Get adaptive stock weights (combines NN + RL + Risk Parity)
            stock_weights, factor_scores = self.adaptive_engine.compute_adaptive_scores(
                {t: price_data[t] for t in STOCK_UNIVERSE if t in price_data},
                spy_data,
                n_stocks=self.n_stocks
            )
            
            # Apply risk multiplier
            stock_allocation = max(0.70, self.stock_weight * risk_mult)  # Min 70%
            for ticker, weight in stock_weights.items():
                adjusted_weight = weight * (stock_allocation / 0.58)  # Scale to allocation
                adjusted_weight = min(adjusted_weight, self.max_position_weight)
                target[ticker] = adjusted_weight
            
            # Log adaptive engine status
            status = self.adaptive_engine.get_status()
            logger.info(f"🧠 Adaptive Status: NN={status['nn_trained']}, RL_states={status['rl_states']}")
            logger.info(f"🧠 Factor Weights: {status['factor_weights']}")
            
        else:
            # =========================================================
            # STATIC FACTOR PATH (fallback)
            # =========================================================
            logger.info("📊 Using STATIC factor model for stock selection")
            
            # Compute factor scores
            factor_scores = self.factor_engine.compute_factor_scores(
                {t: price_data[t] for t in STOCK_UNIVERSE if t in price_data},
                spy_data,
            )
            
            # Select top stocks by factor score
            selected_stocks = [
                s for s in factor_scores[:self.n_stocks * 2]
                if s.momentum_12m > 0.05
            ][:self.n_stocks]
            
            logger.info(f"Selected {len(selected_stocks)} stocks by factor score")
            
            # Allocate to stocks
            stock_allocation = max(0.70, self.stock_weight * risk_mult)
            if selected_stocks:
                total_score = sum(max(0.1, s.composite) for s in selected_stocks)
                for stock in selected_stocks:
                    weight = (max(0.1, stock.composite) / total_score) * stock_allocation
                    weight = min(weight, self.max_position_weight)
                    target[stock.ticker] = weight
        
        # =========================================================
        # LEVERAGED ETF ALLOCATION (always apply)
        # =========================================================
        # Force ETF allocation regardless of regime (Renaissance style - always deployed!)
        if True:  # Always allocate ETFs
            etf_allocation = self.leveraged_etf_weight  # Full 40%
            for etf, base_weight in LEVERAGED_ETFS.items():
                if etf in price_data:
                    target[etf] = base_weight * etf_allocation
            logger.info(f"ETF allocation: {etf_allocation:.1%} to leveraged bull ETFs")
        
        # C. Remaining to cash
        total_allocated = sum(target.values())
        target['$CASH'] = max(self.cash_buffer, 1.0 - total_allocated)
        
        # Log target
        logger.info(f"Target portfolio: {len([k for k in target if k != '$CASH'])} positions")
        logger.info(f"Total allocation: {total_allocated:.1%}, Cash: {target['$CASH']:.1%}")
        
        # Store state
        self.portfolio_history.append(PortfolioState(
            timestamp=datetime.now().isoformat(),
            equity=account.equity,
            cash=account.cash,
            positions={},  # Current positions
            weights=target,  # Target weights
            regime=tda_signals.regime,
            trend=trend,
            vix_level=vix,
            drawdown=drawdown,
            leverage_multiplier=risk_mult,
        ))
        
        return target
    
    def rebalance(self) -> Dict[str, any]:
        """
        Execute portfolio rebalance.
        
        Returns:
            Dict with rebalance results
        """
        logger.info("=" * 60)
        logger.info("STARTING REBALANCE")
        logger.info("=" * 60)
        
        # Check if market is open
        if not self.client.is_market_open():
            logger.warning("Market is closed, skipping rebalance")
            return {"status": "skipped", "reason": "market_closed"}
        
        # Get target portfolio
        target_weights = self.compute_target_portfolio()
        
        if not target_weights:
            return {"status": "skipped", "reason": "no_targets"}
        
        # Get account info
        account = self.client.get_account()
        portfolio_value = account.equity
        
        # Get current positions
        current_positions = self.client.get_positions()
        current_symbols = {p.symbol for p in current_positions}
        
        # Compute target values
        target_values = {
            ticker: weight * portfolio_value
            for ticker, weight in target_weights.items()
            if ticker != '$CASH' and weight > 0.005  # Min 0.5% position
        }
        
        # Compute current values
        current_values = {p.symbol: p.market_value for p in current_positions}
        
        # Determine trades needed
        trades_to_execute = []
        
        # Sells first (to free up cash)
        for symbol, current_value in current_values.items():
            target_value = target_values.get(symbol, 0)
            diff = target_value - current_value
            
            if diff < -100:  # Sell if need to reduce by > $100
                # Find current position
                pos = next((p for p in current_positions if p.symbol == symbol), None)
                if pos:
                    shares_to_sell = min(int(abs(diff) / pos.current_price), int(pos.qty))
                    if shares_to_sell > 0:
                        trades_to_execute.append({
                            'symbol': symbol,
                            'side': 'sell',
                            'qty': shares_to_sell,
                            'reason': 'rebalance_reduce',
                        })
        
        # Buys
        for symbol, target_value in target_values.items():
            current_value = current_values.get(symbol, 0)
            diff = target_value - current_value
            
            if diff > 100:  # Buy if need to increase by > $100
                # Get current price
                try:
                    price = self.price_cache.get(symbol, pd.DataFrame())['Close'].iloc[-1]
                    shares_to_buy = int(diff / price)
                    if shares_to_buy > 0:
                        trades_to_execute.append({
                            'symbol': symbol,
                            'side': 'buy',
                            'qty': shares_to_buy,
                            'reason': 'rebalance_increase',
                        })
                except:
                    logger.warning(f"Could not get price for {symbol}")
        
        # Execute trades
        executed = []
        for trade in trades_to_execute:
            try:
                if trade['side'] == 'buy':
                    order = self.client.submit_order(
                        symbol=trade['symbol'],
                        qty=trade['qty'],
                        side=OrderSide.BUY,
                    )
                else:
                    order = self.client.submit_order(
                        symbol=trade['symbol'],
                        qty=trade['qty'],
                        side=OrderSide.SELL,
                    )
                
                if order:
                    executed.append({
                        'symbol': trade['symbol'],
                        'side': trade['side'],
                        'qty': trade['qty'],
                        'order_id': order.id,
                        'value': trade['qty'] * self.price_cache.get(trade['symbol'], pd.DataFrame()).get('Close', pd.Series([0])).iloc[-1] if trade['symbol'] in self.price_cache else 0,
                    })
                    logger.info(f"Executed: {trade['side'].upper()} {trade['qty']} {trade['symbol']}")
                    
                    # Rate limit
                    time.sleep(0.5)
            except Exception as e:
                logger.error(f"Failed to execute {trade}: {e}")
        
        logger.info(f"Rebalance complete: {len(executed)} trades executed")
        
        # Send notification if trades were executed
        if executed:
            try:
                account = self.client.get_account()
                positions = self.client.get_positions()
                regime_str = self.last_regime.value if hasattr(self, 'last_regime') else 'NEUTRAL'
                notify_rebalance_summary(
                    trades=executed,
                    regime=regime_str.upper(),
                    equity=float(account.equity),
                    cash=float(account.cash),
                    positions=len(positions)
                )
            except Exception as e:
                logger.error(f"Failed to send rebalance notification: {e}")
        
        return {
            "status": "success",
            "trades_planned": len(trades_to_execute),
            "trades_executed": len(executed),
            "executed": executed,
        }
    
    def get_status(self) -> Dict[str, any]:
        """Get current system status."""
        try:
            account = self.client.get_account()
            positions = self.client.get_positions()
            
            # Get current regime
            try:
                price_data, spy_data, vix = self.fetch_market_data()
                stock_returns = pd.DataFrame()
                for ticker in STOCK_UNIVERSE[:20]:  # Use subset for quick analysis
                    if ticker in price_data:
                        stock_returns[ticker] = price_data[ticker]['Close'].pct_change()
                tda_signals = self.tda_analyzer.analyze(stock_returns)
                trend, _ = self.trend_analyzer.analyze_trend(spy_data['Close'])
                regime = tda_signals.regime.value
                trend_state = trend.value
            except:
                regime = "unknown"
                trend_state = "unknown"
                vix = 0
            
            drawdown = self.risk_manager.update_drawdown(account.equity)
            
            return {
                "status": "healthy",
                "account": {
                    "equity": account.equity,
                    "cash": account.cash,
                    "portfolio_value": account.portfolio_value,
                },
                "positions": [
                    {
                        "symbol": p.symbol,
                        "qty": p.qty,
                        "market_value": p.market_value,
                        "unrealized_pl": p.unrealized_pl,
                    }
                    for p in positions
                ],
                "regime": regime,
                "trend": trend_state,
                "vix": vix,
                "drawdown": drawdown,
                "trades_executed": len(self.trade_log),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def health_check(self) -> bool:
        """Check if system is healthy."""
        try:
            health = self.client.health_check()
            return health.get("status") == "healthy"
        except:
            return False


# =============================================================================
# CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TDA Universe Paper Trading Engine")
    parser.add_argument("command", choices=["test", "status", "rebalance", "start"])
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    
    engine = TDAPaperTradingEngine()
    
    if args.command == "test":
        print("\n" + "=" * 60)
        print("TESTING TDA PAPER TRADING ENGINE")
        print("=" * 60)
        
        if engine.health_check():
            print("✅ Connection successful")
            status = engine.get_status()
            print(f"\nAccount Equity: ${status['account']['equity']:,.2f}")
            print(f"Positions: {len(status['positions'])}")
            print(f"Regime: {status['regime']}")
            print(f"Trend: {status['trend']}")
        else:
            print("❌ Connection failed")
    
    elif args.command == "status":
        status = engine.get_status()
        print(json.dumps(status, indent=2, default=str))
    
    elif args.command == "rebalance":
        result = engine.rebalance()
        print(json.dumps(result, indent=2, default=str))
    
    elif args.command == "start":
        import schedule
        
        rebalance_time = os.getenv("REBALANCE_TIME", "15:50")
        
        print(f"Starting TDA Paper Trading Engine")
        print(f"Rebalance scheduled at {rebalance_time} daily")
        print(f"Universe: {len(STOCK_UNIVERSE)} stocks")
        
        schedule.every().day.at(rebalance_time).do(engine.rebalance)
        
        while True:
            schedule.run_pending()
            time.sleep(60)
