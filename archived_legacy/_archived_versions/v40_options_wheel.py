#!/usr/bin/env python3
"""
v40_options_wheel.py - Options Wheel Premium Harvesting Engine

A comprehensive implementation of the Options Wheel strategy for premium harvesting:
1. Sell cash-secured puts on high-quality underlyings
2. If assigned, write covered calls on the position
3. Collect premium throughout the cycle

Author: Trading System v40
Version: 1.0.0
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from logging.handlers import RotatingFileHandler
from typing import Any, Optional

import numpy as np
import pandas as pd

# Alpaca imports
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        GetAssetsRequest,
        GetOrdersRequest,
        LimitOrderRequest,
        MarketOrderRequest,
    )
    from alpaca.trading.enums import (
        AssetClass,
        AssetStatus,
        OrderSide,
        OrderStatus,
        OrderType,
        TimeInForce,
    )
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("Warning: alpaca-py not installed. Install with: pip install alpaca-py")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class WheelConfig:
    """Configuration for the Options Wheel Strategy."""
    
    # Stock Selection Criteria
    min_iv_rank: float = 30.0  # Minimum IV Rank percentage
    min_liquidity: int = 1000  # Minimum contracts/day
    min_market_cap: float = 10e9  # $10 billion minimum
    require_positive_earnings: bool = True
    
    # Technical Filters
    rsi_lower: float = 30.0
    rsi_upper: float = 70.0
    require_above_200sma: bool = True
    
    # Put Selling Parameters
    put_delta_min: float = 0.20
    put_delta_max: float = 0.30
    target_dte_min: int = 30
    target_dte_max: int = 45
    min_premium_pct: float = 1.0  # 1% monthly minimum
    
    # Covered Call Parameters
    call_delta_min: float = 0.30
    call_delta_max: float = 0.40
    
    # Position Sizing
    max_position_pct: float = 5.0  # Max 5% per underlying
    max_wheel_allocation: float = 20.0  # Max 20% in wheel positions
    max_per_sector: int = 3
    
    # Risk Management
    stop_loss_pct: float = 20.0  # Close if down 20%
    max_portfolio_delta: float = 0.50
    
    # Execution
    order_timeout_seconds: int = 60
    paper_trading: bool = True


class PositionState(Enum):
    """State of a wheel position."""
    NO_POSITION = "no_position"
    SHORT_PUT = "short_put"
    LONG_STOCK = "long_stock"
    COVERED_CALL = "covered_call"


@dataclass
class WheelPosition:
    """Represents a position in the wheel strategy."""
    symbol: str
    state: PositionState
    entry_date: datetime
    cost_basis: float = 0.0
    shares: int = 0
    option_symbol: Optional[str] = None
    option_strike: Optional[float] = None
    option_expiry: Optional[datetime] = None
    option_premium: float = 0.0
    total_premium_collected: float = 0.0
    sector: str = "Unknown"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "state": self.state.value,
            "entry_date": self.entry_date.isoformat(),
            "cost_basis": self.cost_basis,
            "shares": self.shares,
            "option_symbol": self.option_symbol,
            "option_strike": self.option_strike,
            "option_expiry": self.option_expiry.isoformat() if self.option_expiry else None,
            "option_premium": self.option_premium,
            "total_premium_collected": self.total_premium_collected,
            "sector": self.sector,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "WheelPosition":
        """Create from dictionary."""
        return cls(
            symbol=data["symbol"],
            state=PositionState(data["state"]),
            entry_date=datetime.fromisoformat(data["entry_date"]),
            cost_basis=data.get("cost_basis", 0.0),
            shares=data.get("shares", 0),
            option_symbol=data.get("option_symbol"),
            option_strike=data.get("option_strike"),
            option_expiry=datetime.fromisoformat(data["option_expiry"]) if data.get("option_expiry") else None,
            option_premium=data.get("option_premium", 0.0),
            total_premium_collected=data.get("total_premium_collected", 0.0),
            sector=data.get("sector", "Unknown"),
        )


@dataclass
class OptionQuote:
    """Represents an option quote."""
    symbol: str
    underlying: str
    option_type: str  # "put" or "call"
    strike: float
    expiry: datetime
    bid: float
    ask: float
    delta: float
    theta: float
    gamma: float
    vega: float
    iv: float
    volume: int
    open_interest: int
    
    @property
    def mid_price(self) -> float:
        """Calculate mid price."""
        return (self.bid + self.ask) / 2
    
    @property
    def spread_pct(self) -> float:
        """Calculate bid-ask spread percentage."""
        if self.mid_price == 0:
            return float('inf')
        return (self.ask - self.bid) / self.mid_price * 100


@dataclass
class WheelCandidate:
    """A candidate stock for the wheel strategy."""
    symbol: str
    price: float
    market_cap: float
    sector: str
    iv_rank: float
    avg_option_volume: int
    rsi: float
    above_200sma: bool
    earnings_positive: bool
    score: float = 0.0
    
    def meets_criteria(self, config: WheelConfig) -> bool:
        """Check if candidate meets all criteria."""
        return (
            self.iv_rank >= config.min_iv_rank
            and self.avg_option_volume >= config.min_liquidity
            and self.market_cap >= config.min_market_cap
            and (not config.require_positive_earnings or self.earnings_positive)
            and config.rsi_lower <= self.rsi <= config.rsi_upper
            and (not config.require_above_200sma or self.above_200sma)
        )


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_file: str = "wheel_strategy.log") -> logging.Logger:
    """Setup logging with rotation."""
    logger = logging.getLogger("WheelStrategy")
    logger.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_format)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


# ============================================================================
# S&P 500 UNIVERSE
# ============================================================================

# Top S&P 500 stocks by market cap (commonly used for wheel strategy)
SP500_TOP_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "BRK.B", "UNH", "XOM", "JNJ",
    "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "LLY", "PEP",
    "KO", "COST", "AVGO", "TMO", "MCD", "WMT", "CSCO", "ACN", "ABT", "DHR",
    "NEE", "VZ", "ADBE", "CRM", "NKE", "CMCSA", "TXN", "PM", "UPS", "HON",
    "INTC", "ORCL", "AMD", "QCOM", "IBM", "CAT", "BA", "GE", "MMM", "RTX",
    "LOW", "AMGN", "SBUX", "DE", "GS", "BLK", "ISRG", "MDLZ", "AXP", "GILD",
    "BKNG", "CVS", "TJX", "SYK", "SPGI", "ADP", "LMT", "SCHW", "MMC", "PLD",
    "C", "MO", "ZTS", "CB", "DUK", "CI", "SO", "CL", "ICE", "BDX",
    "EOG", "REGN", "ITW", "WM", "NOC", "EMR", "SLB", "PNC", "USB", "TGT",
    "FDX", "APD", "MCO", "NSC", "CCI", "EW", "GM", "F", "FCX", "PSX"
]

SECTOR_MAP = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology", "AMZN": "Consumer",
    "NVDA": "Technology", "META": "Technology", "BRK.B": "Financials", "UNH": "Healthcare",
    "XOM": "Energy", "JNJ": "Healthcare", "JPM": "Financials", "V": "Financials",
    "PG": "Consumer", "MA": "Financials", "HD": "Consumer", "CVX": "Energy",
    "MRK": "Healthcare", "ABBV": "Healthcare", "LLY": "Healthcare", "PEP": "Consumer",
    "KO": "Consumer", "COST": "Consumer", "AVGO": "Technology", "TMO": "Healthcare",
    "MCD": "Consumer", "WMT": "Consumer", "CSCO": "Technology", "ACN": "Technology",
    "ABT": "Healthcare", "DHR": "Healthcare", "NEE": "Utilities", "VZ": "Telecom",
    "ADBE": "Technology", "CRM": "Technology", "NKE": "Consumer", "CMCSA": "Telecom",
    "TXN": "Technology", "PM": "Consumer", "UPS": "Industrials", "HON": "Industrials",
    "INTC": "Technology", "ORCL": "Technology", "AMD": "Technology", "QCOM": "Technology",
    "IBM": "Technology", "CAT": "Industrials", "BA": "Industrials", "GE": "Industrials",
    "MMM": "Industrials", "RTX": "Industrials", "LOW": "Consumer", "AMGN": "Healthcare",
    "SBUX": "Consumer", "DE": "Industrials", "GS": "Financials", "BLK": "Financials",
    "ISRG": "Healthcare", "MDLZ": "Consumer", "AXP": "Financials", "GILD": "Healthcare",
    "BKNG": "Consumer", "CVS": "Healthcare", "TJX": "Consumer", "SYK": "Healthcare",
    "SPGI": "Financials", "ADP": "Technology", "LMT": "Industrials", "SCHW": "Financials",
    "MMC": "Financials", "PLD": "Real Estate", "C": "Financials", "MO": "Consumer",
    "ZTS": "Healthcare", "CB": "Financials", "DUK": "Utilities", "CI": "Healthcare",
    "SO": "Utilities", "CL": "Consumer", "ICE": "Financials", "BDX": "Healthcare",
    "EOG": "Energy", "REGN": "Healthcare", "ITW": "Industrials", "WM": "Industrials",
    "NOC": "Industrials", "EMR": "Industrials", "SLB": "Energy", "PNC": "Financials",
    "USB": "Financials", "TGT": "Consumer", "FDX": "Industrials", "APD": "Materials",
    "MCO": "Financials", "NSC": "Industrials", "CCI": "Real Estate", "EW": "Healthcare",
    "GM": "Consumer", "F": "Consumer", "FCX": "Materials", "PSX": "Energy",
}


# ============================================================================
# MAIN ENGINE
# ============================================================================

class OptionsWheelEngine:
    """
    Options Wheel Premium Harvesting Engine.
    
    Implements the wheel strategy:
    1. Sell cash-secured puts on quality stocks
    2. If assigned, write covered calls
    3. Repeat to harvest premium
    """
    
    def __init__(
        self,
        config: Optional[WheelConfig] = None,
        paper_trading: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Options Wheel Engine.
        
        Args:
            config: Strategy configuration
            paper_trading: If True, use paper trading endpoint
            logger: Logger instance
        """
        self.config = config or WheelConfig()
        self.config.paper_trading = paper_trading
        self.logger = logger or setup_logging()
        
        # Load API credentials from environment
        self.api_key = os.environ.get("APCA_API_KEY_ID", "")
        self.api_secret = os.environ.get("APCA_API_SECRET_KEY", "")
        
        if not self.api_key or not self.api_secret:
            self.logger.warning("Alpaca API credentials not found in environment")
        
        # Initialize clients
        self.trading_client: Optional[TradingClient] = None
        self.data_client: Optional[StockHistoricalDataClient] = None
        self._init_clients()
        
        # Position tracking
        self.positions: dict[str, WheelPosition] = {}
        self.positions_file = "wheel_positions.json"
        self._load_positions()
        
        # Performance tracking
        self.total_premium_collected: float = 0.0
        self.total_capital_used: float = 0.0
        self.trades_executed: int = 0
        
        self.logger.info(
            f"OptionsWheelEngine initialized | Paper Trading: {paper_trading}"
        )
    
    def _init_clients(self) -> None:
        """Initialize Alpaca API clients."""
        if not ALPACA_AVAILABLE:
            self.logger.warning("Alpaca SDK not available")
            return
        
        if not self.api_key or not self.api_secret:
            return
        
        try:
            self.trading_client = TradingClient(
                api_key=self.api_key,
                secret_key=self.api_secret,
                paper=self.config.paper_trading
            )
            
            self.data_client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.api_secret
            )
            
            # Verify connection
            account = self.trading_client.get_account()
            self.logger.info(
                f"Connected to Alpaca | Account: {account.account_number} | "
                f"Equity: ${float(account.equity):,.2f}"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Alpaca clients: {e}")
            self.trading_client = None
            self.data_client = None
    
    def _load_positions(self) -> None:
        """Load positions from file."""
        try:
            if os.path.exists(self.positions_file):
                with open(self.positions_file, "r") as f:
                    data = json.load(f)
                    self.positions = {
                        symbol: WheelPosition.from_dict(pos_data)
                        for symbol, pos_data in data.get("positions", {}).items()
                    }
                    self.total_premium_collected = data.get("total_premium_collected", 0.0)
                    self.total_capital_used = data.get("total_capital_used", 0.0)
                    self.trades_executed = data.get("trades_executed", 0)
                self.logger.info(f"Loaded {len(self.positions)} positions from file")
        except Exception as e:
            self.logger.error(f"Error loading positions: {e}")
            self.positions = {}
    
    def _save_positions(self) -> None:
        """Save positions to file."""
        try:
            data = {
                "positions": {
                    symbol: pos.to_dict() for symbol, pos in self.positions.items()
                },
                "total_premium_collected": self.total_premium_collected,
                "total_capital_used": self.total_capital_used,
                "trades_executed": self.trades_executed,
                "last_updated": datetime.now().isoformat(),
            }
            with open(self.positions_file, "w") as f:
                json.dump(data, f, indent=2)
            self.logger.debug("Positions saved to file")
        except Exception as e:
            self.logger.error(f"Error saving positions: {e}")
    
    # ========================================================================
    # MARKET DATA METHODS
    # ========================================================================
    
    def get_stock_price(self, symbol: str) -> Optional[float]:
        """
        Get current stock price.
        
        Args:
            symbol: Stock ticker
            
        Returns:
            Current price or None if unavailable
        """
        if not self.data_client:
            self.logger.warning("Data client not available")
            return None
        
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self.data_client.get_stock_latest_quote(request)
            
            if symbol in quotes:
                quote = quotes[symbol]
                mid_price = (quote.bid_price + quote.ask_price) / 2
                return float(mid_price)
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting price for {symbol}: {e}")
            return None
    
    def get_historical_data(
        self,
        symbol: str,
        days: int = 250
    ) -> Optional[pd.DataFrame]:
        """
        Get historical OHLCV data.
        
        Args:
            symbol: Stock ticker
            days: Number of days of history
            
        Returns:
            DataFrame with OHLCV data
        """
        if not self.data_client:
            return None
        
        try:
            end = datetime.now()
            start = end - timedelta(days=days)
            
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start,
                end=end
            )
            
            bars = self.data_client.get_stock_bars(request)
            
            if symbol in bars:
                df = bars[symbol].df
                return df
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol}: {e}")
            return None
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """
        Calculate RSI indicator.
        
        Args:
            prices: Price series
            period: RSI period
            
        Returns:
            RSI value (0-100)
        """
        if len(prices) < period + 1:
            return 50.0  # Default neutral
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
    
    def calculate_sma(self, prices: pd.Series, period: int = 200) -> float:
        """Calculate Simple Moving Average."""
        if len(prices) < period:
            return prices.mean()
        return float(prices.rolling(window=period).mean().iloc[-1])
    
    def estimate_iv_rank(self, symbol: str) -> float:
        """
        Estimate IV Rank based on historical volatility.
        
        Note: For production, integrate with options data provider for actual IV.
        
        Args:
            symbol: Stock ticker
            
        Returns:
            Estimated IV Rank (0-100)
        """
        df = self.get_historical_data(symbol, days=365)
        if df is None or len(df) < 30:
            return 50.0  # Default
        
        # Calculate historical volatility
        returns = df["close"].pct_change().dropna()
        
        # Current 20-day HV
        current_hv = returns.tail(20).std() * np.sqrt(252) * 100
        
        # Calculate rolling 20-day HV for the year
        rolling_hv = returns.rolling(20).std() * np.sqrt(252) * 100
        rolling_hv = rolling_hv.dropna()
        
        if len(rolling_hv) < 10:
            return 50.0
        
        # IV Rank = percentile of current vs historical range
        hv_min = rolling_hv.min()
        hv_max = rolling_hv.max()
        
        if hv_max == hv_min:
            return 50.0
        
        iv_rank = (current_hv - hv_min) / (hv_max - hv_min) * 100
        return float(np.clip(iv_rank, 0, 100))
    
    # ========================================================================
    # STOCK SCREENING METHODS
    # ========================================================================
    
    def get_wheel_candidates(
        self,
        max_candidates: int = 20
    ) -> list[WheelCandidate]:
        """
        Screen and rank optionable stocks for the wheel strategy.
        
        Args:
            max_candidates: Maximum number of candidates to return
            
        Returns:
            List of WheelCandidate objects, ranked by score
        """
        self.logger.info("Screening for wheel candidates...")
        candidates: list[WheelCandidate] = []
        
        for symbol in SP500_TOP_SYMBOLS:
            try:
                # Get price data
                df = self.get_historical_data(symbol, days=250)
                if df is None or len(df) < 200:
                    continue
                
                current_price = float(df["close"].iloc[-1])
                
                # Calculate technical indicators
                rsi = self.calculate_rsi(df["close"])
                sma_200 = self.calculate_sma(df["close"], 200)
                above_200sma = current_price > sma_200
                
                # Estimate IV rank
                iv_rank = self.estimate_iv_rank(symbol)
                
                # Get sector
                sector = SECTOR_MAP.get(symbol, "Unknown")
                
                # Create candidate (using estimated values for some metrics)
                candidate = WheelCandidate(
                    symbol=symbol,
                    price=current_price,
                    market_cap=50e9,  # Assume large cap for S&P 500
                    sector=sector,
                    iv_rank=iv_rank,
                    avg_option_volume=5000,  # Assume liquid for S&P 500
                    rsi=rsi,
                    above_200sma=above_200sma,
                    earnings_positive=True,  # Assume for S&P 500
                )
                
                # Check if meets criteria
                if candidate.meets_criteria(self.config):
                    # Calculate score (higher is better)
                    candidate.score = self._score_candidate(candidate)
                    candidates.append(candidate)
                    
            except Exception as e:
                self.logger.debug(f"Error screening {symbol}: {e}")
                continue
        
        # Sort by score (descending)
        candidates.sort(key=lambda x: x.score, reverse=True)
        
        self.logger.info(
            f"Found {len(candidates)} candidates meeting criteria"
        )
        
        return candidates[:max_candidates]
    
    def _score_candidate(self, candidate: WheelCandidate) -> float:
        """
        Score a wheel candidate (higher is better).
        
        Scoring factors:
        - IV Rank (higher = more premium)
        - RSI near 50 (neutral = less directional risk)
        - Above 200 SMA (bullish trend)
        """
        score = 0.0
        
        # IV Rank contribution (0-40 points)
        score += min(candidate.iv_rank, 80) * 0.5
        
        # RSI contribution (closer to 50 is better, max 30 points)
        rsi_distance = abs(candidate.rsi - 50)
        score += max(0, 30 - rsi_distance)
        
        # Trend contribution (20 points if above 200 SMA)
        if candidate.above_200sma:
            score += 20
        
        # Liquidity bonus (assume all S&P 500 are liquid)
        score += 10
        
        return score
    
    # ========================================================================
    # OPTIONS ANALYSIS METHODS
    # ========================================================================
    
    def analyze_options_chain(
        self,
        symbol: str,
        option_type: str = "put"
    ) -> list[OptionQuote]:
        """
        Analyze options chain to find optimal strikes/expirations.
        
        Note: For production, integrate with options data provider.
        This implementation uses estimated values.
        
        Args:
            symbol: Underlying stock ticker
            option_type: "put" or "call"
            
        Returns:
            List of OptionQuote objects meeting criteria
        """
        self.logger.info(f"Analyzing {option_type} options for {symbol}")
        
        current_price = self.get_stock_price(symbol)
        if current_price is None:
            self.logger.warning(f"Could not get price for {symbol}")
            return []
        
        options: list[OptionQuote] = []
        
        # Generate synthetic options for demonstration
        # In production, fetch from options data provider
        now = datetime.now()
        
        for days_to_exp in [30, 37, 44]:
            expiry = now + timedelta(days=days_to_exp)
            
            # Generate strikes around current price
            if option_type == "put":
                # OTM puts: strikes below current price
                strike_range = np.arange(
                    current_price * 0.85,
                    current_price * 0.98,
                    current_price * 0.02
                )
            else:
                # OTM calls: strikes above current price
                strike_range = np.arange(
                    current_price * 1.02,
                    current_price * 1.15,
                    current_price * 0.02
                )
            
            for strike in strike_range:
                strike = round(strike, 0)  # Round to nearest dollar
                
                # Estimate option Greeks (simplified Black-Scholes approximation)
                moneyness = strike / current_price
                dte = days_to_exp
                
                # Estimate delta based on moneyness
                if option_type == "put":
                    delta = -0.5 * (1 - moneyness) * 2  # Simplified
                    delta = max(-0.50, min(-0.10, delta))
                else:
                    delta = 0.5 * (moneyness - 1) * 2 + 0.5  # Simplified
                    delta = max(0.10, min(0.50, delta))
                
                # Estimate premium (simplified)
                iv = 0.30  # Assume 30% IV
                time_factor = np.sqrt(dte / 365)
                premium = current_price * iv * time_factor * 0.4
                
                if option_type == "put":
                    premium *= (1 - moneyness + 0.1)
                else:
                    premium *= (moneyness - 1 + 0.1)
                
                premium = max(0.10, premium)
                
                # Create option quote
                option = OptionQuote(
                    symbol=f"{symbol}{expiry.strftime('%y%m%d')}{option_type[0].upper()}{int(strike*1000):08d}",
                    underlying=symbol,
                    option_type=option_type,
                    strike=strike,
                    expiry=expiry,
                    bid=premium * 0.95,
                    ask=premium * 1.05,
                    delta=delta,
                    theta=-premium / dte * 0.7,  # Approximate
                    gamma=0.01,
                    vega=premium * 0.1,
                    iv=iv,
                    volume=1000,
                    open_interest=5000,
                )
                
                options.append(option)
        
        # Filter by delta range
        if option_type == "put":
            delta_min = -self.config.put_delta_max
            delta_max = -self.config.put_delta_min
        else:
            delta_min = self.config.call_delta_min
            delta_max = self.config.call_delta_max
        
        filtered = [
            opt for opt in options
            if delta_min <= opt.delta <= delta_max
            and self.config.target_dte_min <= (opt.expiry - now).days <= self.config.target_dte_max
        ]
        
        # Sort by premium/risk ratio
        filtered.sort(key=lambda x: x.mid_price / abs(x.delta), reverse=True)
        
        self.logger.info(f"Found {len(filtered)} options meeting criteria")
        return filtered
    
    def select_optimal_option(
        self,
        options: list[OptionQuote],
        underlying_price: float
    ) -> Optional[OptionQuote]:
        """
        Select the optimal option from a list of candidates.
        
        Args:
            options: List of option quotes
            underlying_price: Current underlying price
            
        Returns:
            Best option or None
        """
        if not options:
            return None
        
        best_option = None
        best_score = 0.0
        
        for opt in options:
            # Calculate premium as percentage of strike (for puts) or underlying
            if opt.option_type == "put":
                premium_pct = (opt.mid_price / opt.strike) * 100
                # Annualized return
                dte = (opt.expiry - datetime.now()).days
                annual_return = premium_pct * (365 / dte)
            else:
                premium_pct = (opt.mid_price / underlying_price) * 100
                dte = (opt.expiry - datetime.now()).days
                annual_return = premium_pct * (365 / dte)
            
            # Score based on:
            # - Premium yield (higher is better)
            # - Delta (closer to target range is better)
            # - Spread (tighter is better)
            
            score = annual_return
            
            # Penalize wide spreads
            if opt.spread_pct > 5:
                score *= 0.9
            
            if score > best_score and premium_pct >= self.config.min_premium_pct:
                best_score = score
                best_option = opt
        
        return best_option
    
    # ========================================================================
    # TRADING METHODS
    # ========================================================================
    
    def get_account_info(self) -> dict[str, Any]:
        """Get account information."""
        if not self.trading_client:
            return {"error": "Trading client not available"}
        
        try:
            account = self.trading_client.get_account()
            return {
                "buying_power": float(account.buying_power),
                "cash": float(account.cash),
                "equity": float(account.equity),
                "portfolio_value": float(account.portfolio_value),
                "day_trading_buying_power": float(account.daytrading_buying_power),
                "pattern_day_trader": account.pattern_day_trader,
            }
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return {"error": str(e)}
    
    def calculate_position_size(
        self,
        strike: float,
        symbol: str
    ) -> int:
        """
        Calculate position size (number of contracts) based on risk limits.
        
        Args:
            strike: Option strike price
            symbol: Underlying symbol
            
        Returns:
            Number of contracts to trade
        """
        account = self.get_account_info()
        if "error" in account:
            return 0
        
        portfolio_value = account.get("portfolio_value", 0)
        buying_power = account.get("buying_power", 0)
        
        # Maximum capital per position
        max_position_capital = portfolio_value * (self.config.max_position_pct / 100)
        
        # Check sector concentration
        sector = SECTOR_MAP.get(symbol, "Unknown")
        sector_count = sum(
            1 for pos in self.positions.values()
            if pos.sector == sector and pos.state != PositionState.NO_POSITION
        )
        
        if sector_count >= self.config.max_per_sector:
            self.logger.warning(f"Sector limit reached for {sector}")
            return 0
        
        # Check total wheel allocation
        current_wheel_capital = sum(
            pos.cost_basis * pos.shares / 100
            for pos in self.positions.values()
            if pos.state != PositionState.NO_POSITION
        )
        
        remaining_allocation = (
            portfolio_value * (self.config.max_wheel_allocation / 100)
            - current_wheel_capital
        )
        
        if remaining_allocation <= 0:
            self.logger.warning("Maximum wheel allocation reached")
            return 0
        
        # Capital required per contract (100 shares at strike price)
        capital_per_contract = strike * 100
        
        # Maximum contracts based on limits
        max_by_position = int(max_position_capital / capital_per_contract)
        max_by_allocation = int(remaining_allocation / capital_per_contract)
        max_by_buying_power = int(buying_power / capital_per_contract)
        
        contracts = min(max_by_position, max_by_allocation, max_by_buying_power)
        
        # At least 1 contract if we can afford it
        return max(0, min(contracts, 10))  # Cap at 10 contracts per position
    
    def sell_cash_secured_put(
        self,
        symbol: str,
        strike: float,
        expiry: datetime,
        contracts: int = 1,
        limit_price: Optional[float] = None
    ) -> Optional[str]:
        """
        Execute a cash-secured put sell order.
        
        Args:
            symbol: Underlying stock ticker
            strike: Put strike price
            expiry: Option expiration date
            contracts: Number of contracts to sell
            limit_price: Limit price per contract (optional)
            
        Returns:
            Order ID if successful, None otherwise
        """
        self.logger.info(
            f"Selling {contracts} CSP on {symbol} "
            f"@ ${strike:.2f} strike, exp {expiry.strftime('%Y-%m-%d')}"
        )
        
        if not self.trading_client:
            self.logger.error("Trading client not available")
            return None
        
        # Build option symbol (OCC format)
        option_symbol = self._build_option_symbol(symbol, expiry, "P", strike)
        
        try:
            # For now, log the trade (actual options trading requires options-enabled account)
            self.logger.info(
                f"[SIMULATED] Sell to Open: {contracts}x {option_symbol} "
                f"@ ${limit_price:.2f}" if limit_price else f"@ Market"
            )
            
            # Track position
            position = WheelPosition(
                symbol=symbol,
                state=PositionState.SHORT_PUT,
                entry_date=datetime.now(),
                option_symbol=option_symbol,
                option_strike=strike,
                option_expiry=expiry,
                option_premium=limit_price or 0,
                total_premium_collected=limit_price * contracts * 100 if limit_price else 0,
                sector=SECTOR_MAP.get(symbol, "Unknown"),
            )
            
            self.positions[symbol] = position
            self.total_premium_collected += position.total_premium_collected
            self.total_capital_used += strike * contracts * 100
            self.trades_executed += 1
            self._save_positions()
            
            self.logger.info(
                f"CSP position opened | Premium: ${position.total_premium_collected:.2f}"
            )
            
            return option_symbol
            
        except Exception as e:
            self.logger.error(f"Error selling CSP: {e}")
            return None
    
    def write_covered_call(
        self,
        symbol: str,
        strike: float,
        expiry: datetime,
        contracts: int = 1,
        limit_price: Optional[float] = None
    ) -> Optional[str]:
        """
        Execute a covered call write order.
        
        Args:
            symbol: Underlying stock ticker
            strike: Call strike price
            expiry: Option expiration date
            contracts: Number of contracts to write
            limit_price: Limit price per contract (optional)
            
        Returns:
            Order ID if successful, None otherwise
        """
        self.logger.info(
            f"Writing {contracts} CC on {symbol} "
            f"@ ${strike:.2f} strike, exp {expiry.strftime('%Y-%m-%d')}"
        )
        
        if symbol not in self.positions:
            self.logger.error(f"No position in {symbol} to cover")
            return None
        
        position = self.positions[symbol]
        
        if position.state != PositionState.LONG_STOCK:
            self.logger.error(f"Position in {symbol} is not long stock")
            return None
        
        # Check we have enough shares
        shares_needed = contracts * 100
        if position.shares < shares_needed:
            self.logger.error(
                f"Not enough shares: have {position.shares}, need {shares_needed}"
            )
            return None
        
        # Build option symbol
        option_symbol = self._build_option_symbol(symbol, expiry, "C", strike)
        
        try:
            self.logger.info(
                f"[SIMULATED] Sell to Open: {contracts}x {option_symbol} "
                f"@ ${limit_price:.2f}" if limit_price else f"@ Market"
            )
            
            # Update position
            position.state = PositionState.COVERED_CALL
            position.option_symbol = option_symbol
            position.option_strike = strike
            position.option_expiry = expiry
            position.option_premium = limit_price or 0
            position.total_premium_collected += limit_price * contracts * 100 if limit_price else 0
            
            self.total_premium_collected += limit_price * contracts * 100 if limit_price else 0
            self.trades_executed += 1
            self._save_positions()
            
            self.logger.info(
                f"CC position opened | Total Premium: ${position.total_premium_collected:.2f}"
            )
            
            return option_symbol
            
        except Exception as e:
            self.logger.error(f"Error writing covered call: {e}")
            return None
    
    def _build_option_symbol(
        self,
        underlying: str,
        expiry: datetime,
        option_type: str,
        strike: float
    ) -> str:
        """Build OCC option symbol."""
        # Format: SYMBOL + YYMMDD + C/P + Strike*1000 (8 digits)
        exp_str = expiry.strftime("%y%m%d")
        strike_str = f"{int(strike * 1000):08d}"
        return f"{underlying}{exp_str}{option_type}{strike_str}"
    
    # ========================================================================
    # POSITION MANAGEMENT
    # ========================================================================
    
    def manage_positions(self) -> None:
        """
        Manage existing wheel positions.
        
        Handles:
        - Rolling options if challenged
        - Closing positions at stop-loss
        - Transitioning from short put to long stock (assignment)
        - Writing covered calls on assigned positions
        """
        self.logger.info("Managing wheel positions...")
        
        for symbol, position in list(self.positions.items()):
            try:
                self._manage_single_position(symbol, position)
            except Exception as e:
                self.logger.error(f"Error managing {symbol}: {e}")
    
    def _manage_single_position(
        self,
        symbol: str,
        position: WheelPosition
    ) -> None:
        """Manage a single wheel position."""
        
        if position.state == PositionState.NO_POSITION:
            return
        
        current_price = self.get_stock_price(symbol)
        if current_price is None:
            self.logger.warning(f"Could not get price for {symbol}")
            return
        
        # Check stop-loss
        if position.state == PositionState.LONG_STOCK:
            if position.cost_basis > 0:
                loss_pct = (position.cost_basis - current_price) / position.cost_basis * 100
                if loss_pct >= self.config.stop_loss_pct:
                    self.logger.warning(
                        f"Stop-loss triggered for {symbol}: "
                        f"down {loss_pct:.1f}% from cost basis"
                    )
                    self._close_stock_position(symbol, position)
                    return
        
        # Check option expiration
        if position.option_expiry:
            days_to_expiry = (position.option_expiry - datetime.now()).days
            
            if days_to_expiry <= 0:
                # Option expired - check assignment
                self._handle_expiration(symbol, position, current_price)
            
            elif days_to_expiry <= 5:
                # Close to expiration - consider rolling
                self._consider_roll(symbol, position, current_price)
    
    def _handle_expiration(
        self,
        symbol: str,
        position: WheelPosition,
        current_price: float
    ) -> None:
        """Handle option expiration."""
        
        if position.state == PositionState.SHORT_PUT:
            # Check if put was assigned
            if position.option_strike and current_price < position.option_strike:
                self.logger.info(
                    f"{symbol} put assigned at ${position.option_strike:.2f}"
                )
                # Transition to long stock
                position.state = PositionState.LONG_STOCK
                position.cost_basis = position.option_strike
                position.shares = 100  # 1 contract = 100 shares
                position.option_symbol = None
                position.option_expiry = None
                self._save_positions()
                
                # Immediately write covered call
                self._write_call_on_assigned(symbol, position)
            else:
                self.logger.info(f"{symbol} put expired worthless - profit!")
                position.state = PositionState.NO_POSITION
                self._save_positions()
        
        elif position.state == PositionState.COVERED_CALL:
            # Check if call was assigned
            if position.option_strike and current_price > position.option_strike:
                self.logger.info(
                    f"{symbol} call assigned at ${position.option_strike:.2f}"
                )
                # Stock called away
                profit = (position.option_strike - position.cost_basis) * position.shares
                profit += position.total_premium_collected
                
                self.logger.info(
                    f"Wheel cycle complete for {symbol} | "
                    f"Total Profit: ${profit:.2f}"
                )
                
                position.state = PositionState.NO_POSITION
                position.shares = 0
                self._save_positions()
            else:
                self.logger.info(f"{symbol} call expired worthless")
                # Can write another call
                position.state = PositionState.LONG_STOCK
                position.option_symbol = None
                position.option_expiry = None
                self._save_positions()
                
                # Write new covered call
                self._write_call_on_assigned(symbol, position)
    
    def _consider_roll(
        self,
        symbol: str,
        position: WheelPosition,
        current_price: float
    ) -> None:
        """Consider rolling an option position."""
        
        if position.state == PositionState.SHORT_PUT:
            # Check if put is ITM
            if position.option_strike and current_price < position.option_strike:
                self.logger.info(
                    f"{symbol} put is ITM - consider rolling out"
                )
                # In production, would close current and open new position
        
        elif position.state == PositionState.COVERED_CALL:
            # Check if call is ITM
            if position.option_strike and current_price > position.option_strike:
                self.logger.info(
                    f"{symbol} call is ITM - consider rolling up and out"
                )
                # In production, would close current and open new position
    
    def _write_call_on_assigned(
        self,
        symbol: str,
        position: WheelPosition
    ) -> None:
        """Write a covered call on an assigned position."""
        
        # Find optimal call
        options = self.analyze_options_chain(symbol, "call")
        current_price = self.get_stock_price(symbol)
        
        if not options or not current_price:
            return
        
        # Select strike above cost basis if possible
        valid_options = [
            opt for opt in options
            if opt.strike > position.cost_basis
        ]
        
        if not valid_options:
            valid_options = options  # Use any if none above cost basis
        
        best_option = self.select_optimal_option(valid_options, current_price)
        
        if best_option:
            self.write_covered_call(
                symbol=symbol,
                strike=best_option.strike,
                expiry=best_option.expiry,
                contracts=position.shares // 100,
                limit_price=best_option.mid_price
            )
    
    def _close_stock_position(
        self,
        symbol: str,
        position: WheelPosition
    ) -> None:
        """Close a stock position at stop-loss."""
        
        self.logger.warning(f"Closing {symbol} position at stop-loss")
        
        if not self.trading_client:
            return
        
        try:
            # In production, would submit market sell order
            self.logger.info(
                f"[SIMULATED] Sell {position.shares} shares of {symbol}"
            )
            
            position.state = PositionState.NO_POSITION
            position.shares = 0
            self._save_positions()
            
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
    
    # ========================================================================
    # REPORTING METHODS
    # ========================================================================
    
    def calculate_returns(self) -> dict[str, Any]:
        """
        Calculate and report strategy returns.
        
        Returns:
            Dictionary with return metrics
        """
        # Calculate total premium vs capital
        if self.total_capital_used > 0:
            total_return_pct = (
                self.total_premium_collected / self.total_capital_used * 100
            )
        else:
            total_return_pct = 0.0
        
        # Position-level returns
        position_returns = []
        for symbol, pos in self.positions.items():
            if pos.state != PositionState.NO_POSITION:
                position_returns.append({
                    "symbol": symbol,
                    "state": pos.state.value,
                    "premium_collected": pos.total_premium_collected,
                    "cost_basis": pos.cost_basis,
                    "shares": pos.shares,
                })
        
        return {
            "total_premium_collected": self.total_premium_collected,
            "total_capital_used": self.total_capital_used,
            "return_pct": total_return_pct,
            "trades_executed": self.trades_executed,
            "active_positions": len([
                p for p in self.positions.values()
                if p.state != PositionState.NO_POSITION
            ]),
            "positions": position_returns,
        }
    
    def get_status(self) -> str:
        """Get formatted status report."""
        account = self.get_account_info()
        returns = self.calculate_returns()
        
        lines = [
            "=" * 60,
            "OPTIONS WHEEL STRATEGY STATUS",
            "=" * 60,
            "",
            "ACCOUNT:",
            f"  Portfolio Value: ${account.get('portfolio_value', 0):,.2f}",
            f"  Buying Power:    ${account.get('buying_power', 0):,.2f}",
            f"  Cash:            ${account.get('cash', 0):,.2f}",
            "",
            "PERFORMANCE:",
            f"  Total Premium:   ${returns['total_premium_collected']:,.2f}",
            f"  Capital Used:    ${returns['total_capital_used']:,.2f}",
            f"  Return:          {returns['return_pct']:.2f}%",
            f"  Trades:          {returns['trades_executed']}",
            "",
            "POSITIONS:",
        ]
        
        for pos in returns['positions']:
            lines.append(
                f"  {pos['symbol']:6s} | {pos['state']:15s} | "
                f"Premium: ${pos['premium_collected']:,.2f}"
            )
        
        if not returns['positions']:
            lines.append("  No active positions")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    # ========================================================================
    # MAIN ORCHESTRATION
    # ========================================================================
    
    def run_wheel_cycle(self) -> None:
        """
        Main orchestration loop for the wheel strategy.
        
        Steps:
        1. Manage existing positions
        2. Screen for new candidates
        3. Open new positions if capacity available
        """
        self.logger.info("=" * 60)
        self.logger.info("STARTING WHEEL CYCLE")
        self.logger.info("=" * 60)
        
        # Step 1: Manage existing positions
        self.manage_positions()
        
        # Step 2: Check capacity for new positions
        account = self.get_account_info()
        if "error" in account:
            self.logger.error("Could not get account info")
            return
        
        portfolio_value = account.get("portfolio_value", 0)
        current_allocation = sum(
            pos.cost_basis * pos.shares / 100
            for pos in self.positions.values()
            if pos.state != PositionState.NO_POSITION
        )
        
        allocation_pct = (current_allocation / portfolio_value * 100) if portfolio_value > 0 else 0
        remaining_capacity = self.config.max_wheel_allocation - allocation_pct
        
        self.logger.info(
            f"Current allocation: {allocation_pct:.1f}% | "
            f"Remaining capacity: {remaining_capacity:.1f}%"
        )
        
        if remaining_capacity <= self.config.max_position_pct:
            self.logger.info("At maximum wheel allocation - no new positions")
            return
        
        # Step 3: Screen for new candidates
        candidates = self.get_wheel_candidates(max_candidates=10)
        
        if not candidates:
            self.logger.info("No suitable candidates found")
            return
        
        # Step 4: Open new positions
        for candidate in candidates:
            if candidate.symbol in self.positions:
                if self.positions[candidate.symbol].state != PositionState.NO_POSITION:
                    continue  # Already have position
            
            # Check sector limit
            sector_count = sum(
                1 for pos in self.positions.values()
                if pos.sector == candidate.sector
                and pos.state != PositionState.NO_POSITION
            )
            
            if sector_count >= self.config.max_per_sector:
                continue
            
            # Analyze put options
            put_options = self.analyze_options_chain(candidate.symbol, "put")
            best_put = self.select_optimal_option(put_options, candidate.price)
            
            if not best_put:
                continue
            
            # Calculate position size
            contracts = self.calculate_position_size(best_put.strike, candidate.symbol)
            
            if contracts <= 0:
                continue
            
            # Execute trade
            order_id = self.sell_cash_secured_put(
                symbol=candidate.symbol,
                strike=best_put.strike,
                expiry=best_put.expiry,
                contracts=contracts,
                limit_price=best_put.mid_price
            )
            
            if order_id:
                self.logger.info(
                    f"Opened CSP position on {candidate.symbol} | "
                    f"Strike: ${best_put.strike:.2f} | "
                    f"Contracts: {contracts}"
                )
                
                # Check if we've hit allocation limit
                remaining_capacity -= self.config.max_position_pct
                if remaining_capacity <= 0:
                    break
        
        # Print status
        print(self.get_status())
        
        self.logger.info("WHEEL CYCLE COMPLETE")
        self.logger.info("=" * 60)


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Main entry point with CLI interface."""
    parser = argparse.ArgumentParser(
        description="Options Wheel Premium Harvesting Strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python v40_options_wheel.py --test      # Run in test mode (no trades)
  python v40_options_wheel.py --status    # Show current status
  python v40_options_wheel.py --trade     # Run trading cycle
  python v40_options_wheel.py --candidates # Screen for candidates
        """
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with simulated data"
    )
    
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current strategy status"
    )
    
    parser.add_argument(
        "--trade",
        action="store_true",
        help="Run a trading cycle"
    )
    
    parser.add_argument(
        "--candidates",
        action="store_true",
        help="Screen and display wheel candidates"
    )
    
    parser.add_argument(
        "--paper",
        action="store_true",
        default=True,
        help="Use paper trading (default: True)"
    )
    
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use live trading (requires confirmation)"
    )
    
    args = parser.parse_args()
    
    # Determine paper trading mode
    paper_trading = not args.live
    
    if args.live:
        confirm = input(
            "WARNING: Live trading mode selected. "
            "Are you sure? (type 'YES' to confirm): "
        )
        if confirm != "YES":
            print("Live trading cancelled.")
            sys.exit(0)
    
    # Initialize engine
    print("\n" + "=" * 60)
    print("OPTIONS WHEEL PREMIUM HARVESTING ENGINE")
    print("=" * 60 + "\n")
    
    config = WheelConfig(paper_trading=paper_trading)
    engine = OptionsWheelEngine(config=config, paper_trading=paper_trading)
    
    # Execute requested action
    if args.test:
        print("Running in TEST mode...\n")
        
        # Test stock screening
        print("Testing stock screening...")
        candidates = engine.get_wheel_candidates(max_candidates=5)
        
        if candidates:
            print(f"\nTop {len(candidates)} Wheel Candidates:")
            print("-" * 60)
            for i, c in enumerate(candidates, 1):
                print(
                    f"{i}. {c.symbol:6s} | "
                    f"Price: ${c.price:>8.2f} | "
                    f"IV Rank: {c.iv_rank:>5.1f}% | "
                    f"RSI: {c.rsi:>5.1f} | "
                    f"Score: {c.score:>5.1f}"
                )
        
        # Test options analysis
        if candidates:
            print(f"\nAnalyzing put options for {candidates[0].symbol}...")
            options = engine.analyze_options_chain(candidates[0].symbol, "put")
            
            if options:
                print(f"\nTop 5 Put Options:")
                print("-" * 60)
                for opt in options[:5]:
                    print(
                        f"{opt.symbol} | "
                        f"Strike: ${opt.strike:>7.2f} | "
                        f"Delta: {opt.delta:>6.2f} | "
                        f"Premium: ${opt.mid_price:>6.2f} | "
                        f"DTE: {(opt.expiry - datetime.now()).days:>3d}"
                    )
        
        print("\nTest complete!")
    
    elif args.status:
        print(engine.get_status())
    
    elif args.trade:
        print("Running trading cycle...\n")
        engine.run_wheel_cycle()
    
    elif args.candidates:
        print("Screening for wheel candidates...\n")
        candidates = engine.get_wheel_candidates(max_candidates=20)
        
        if candidates:
            print(f"Found {len(candidates)} candidates:\n")
            print(f"{'#':>2} | {'Symbol':6s} | {'Price':>10s} | {'IV Rank':>8s} | "
                  f"{'RSI':>6s} | {'Sector':12s} | {'Score':>6s}")
            print("-" * 70)
            
            for i, c in enumerate(candidates, 1):
                print(
                    f"{i:>2} | {c.symbol:6s} | ${c.price:>8.2f} | "
                    f"{c.iv_rank:>7.1f}% | {c.rsi:>6.1f} | "
                    f"{c.sector:12s} | {c.score:>6.1f}"
                )
        else:
            print("No candidates found meeting criteria.")
    
    else:
        # Default: show status
        parser.print_help()


if __name__ == "__main__":
    main()
