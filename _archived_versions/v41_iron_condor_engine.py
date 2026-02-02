#!/usr/bin/env python3
"""
v41_iron_condor_engine.py - Iron Condor Premium Strategy Engine

A comprehensive implementation of the Iron Condor strategy for premium harvesting:
- Focus on high-liquidity ETFs: SPY, QQQ, IWM
- Entry when IV Rank > 50, ideally > 70
- Target 20-25 delta for balanced POP (75-80%)
- 30-45 DTE optimal entry
- Close at 50% profit or 7 DTE

Author: Trading System v41
Version: 1.0.0
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from logging.handlers import RotatingFileHandler
from typing import Any, Optional, Tuple

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
class IronCondorConfig:
    """Configuration for the Iron Condor Strategy."""
    
    # Underlying Selection
    target_underlyings: list[str] = field(default_factory=lambda: ["SPY", "QQQ", "IWM"])
    
    # IV Rank Thresholds
    min_iv_rank: float = 50.0  # Minimum IV Rank to enter
    ideal_iv_rank: float = 70.0  # Ideal IV Rank for best entries
    
    # Strike Selection - Delta Targeting
    delta_profile: str = "balanced"  # "conservative", "balanced", "aggressive"
    delta_targets: dict = field(default_factory=lambda: {
        "conservative": {"min": 0.10, "max": 0.16, "pop": 0.85},
        "balanced": {"min": 0.20, "max": 0.25, "pop": 0.77},
        "aggressive": {"min": 0.30, "max": 0.35, "pop": 0.67},
    })
    
    # DTE Parameters
    target_dte_min: int = 30
    target_dte_max: int = 45
    close_dte: int = 7  # Close if DTE <= this
    
    # Wing Width
    wing_width_dollars: float = 5.0  # Width of wings in dollars
    min_credit_pct: float = 25.0  # Min credit as % of wing width
    max_credit_pct: float = 40.0  # Max credit (avoid too aggressive)
    
    # Exit Rules
    profit_target_pct: float = 50.0  # Close at 50% profit
    max_loss_multiplier: float = 2.0  # Max loss = 2x premium collected
    
    # Position Sizing
    max_positions: int = 5
    max_position_pct: float = 5.0  # Max 5% per position
    max_allocation_pct: float = 25.0  # Max 25% in condors
    
    # Risk Management
    max_portfolio_delta: float = 0.10  # Keep delta-neutral
    max_portfolio_vega: float = 500.0  # Vega exposure limit
    
    # Execution
    paper_trading: bool = True
    order_timeout_seconds: int = 60
    min_bid_ask_spread_ratio: float = 0.10  # Max 10% spread/mid


class CondorState(Enum):
    """State of an Iron Condor position."""
    OPEN = "open"
    PROFIT_TARGET = "profit_target"
    ROLLED = "rolled"
    CLOSED_DTE = "closed_dte"
    STOPPED_OUT = "stopped_out"
    EXPIRED = "expired"


@dataclass
class CondorLeg:
    """Represents a single leg of the Iron Condor."""
    option_type: str  # "put" or "call"
    strike: float
    side: str  # "short" or "long"
    delta: float
    premium: float
    iv: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "option_type": self.option_type,
            "strike": self.strike,
            "side": self.side,
            "delta": self.delta,
            "premium": self.premium,
            "iv": self.iv,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "CondorLeg":
        """Create from dictionary."""
        return cls(
            option_type=data["option_type"],
            strike=data["strike"],
            side=data["side"],
            delta=data["delta"],
            premium=data["premium"],
            iv=data["iv"],
        )


@dataclass
class IronCondorPosition:
    """Represents an Iron Condor position."""
    symbol: str
    entry_date: datetime
    expiry: datetime
    
    # Legs
    put_long: CondorLeg
    put_short: CondorLeg
    call_short: CondorLeg
    call_long: CondorLeg
    
    # Position details
    contracts: int
    credit_received: float
    max_loss: float
    current_value: float = 0.0
    
    # Greeks
    delta: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    gamma: float = 0.0
    
    # State
    state: CondorState = CondorState.OPEN
    iv_rank_at_entry: float = 0.0
    
    @property
    def dte(self) -> int:
        """Days to expiration."""
        return (self.expiry - datetime.now()).days
    
    @property
    def pnl(self) -> float:
        """Current P&L."""
        return self.credit_received - self.current_value
    
    @property
    def pnl_pct(self) -> float:
        """P&L as percentage of max profit."""
        if self.credit_received == 0:
            return 0.0
        return (self.pnl / self.credit_received) * 100
    
    @property
    def wing_width(self) -> float:
        """Width of the wings."""
        return self.put_short.strike - self.put_long.strike
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "entry_date": self.entry_date.isoformat(),
            "expiry": self.expiry.isoformat(),
            "put_long": self.put_long.to_dict(),
            "put_short": self.put_short.to_dict(),
            "call_short": self.call_short.to_dict(),
            "call_long": self.call_long.to_dict(),
            "contracts": self.contracts,
            "credit_received": self.credit_received,
            "max_loss": self.max_loss,
            "current_value": self.current_value,
            "delta": self.delta,
            "theta": self.theta,
            "vega": self.vega,
            "gamma": self.gamma,
            "state": self.state.value,
            "iv_rank_at_entry": self.iv_rank_at_entry,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "IronCondorPosition":
        """Create from dictionary."""
        return cls(
            symbol=data["symbol"],
            entry_date=datetime.fromisoformat(data["entry_date"]),
            expiry=datetime.fromisoformat(data["expiry"]),
            put_long=CondorLeg.from_dict(data["put_long"]),
            put_short=CondorLeg.from_dict(data["put_short"]),
            call_short=CondorLeg.from_dict(data["call_short"]),
            call_long=CondorLeg.from_dict(data["call_long"]),
            contracts=data["contracts"],
            credit_received=data["credit_received"],
            max_loss=data["max_loss"],
            current_value=data.get("current_value", 0.0),
            delta=data.get("delta", 0.0),
            theta=data.get("theta", 0.0),
            vega=data.get("vega", 0.0),
            gamma=data.get("gamma", 0.0),
            state=CondorState(data.get("state", "open")),
            iv_rank_at_entry=data.get("iv_rank_at_entry", 0.0),
        )


@dataclass
class CondorCandidate:
    """A candidate Iron Condor trade."""
    symbol: str
    price: float
    iv_rank: float
    iv_percentile: float
    hv_30: float  # 30-day historical volatility
    expiry: datetime
    
    # Strikes
    put_long_strike: float
    put_short_strike: float
    call_short_strike: float
    call_long_strike: float
    
    # Premiums
    put_spread_credit: float
    call_spread_credit: float
    total_credit: float
    
    # Greeks
    delta: float
    theta: float
    vega: float
    
    # Metrics
    pop: float  # Probability of profit
    credit_to_width_ratio: float
    score: float = 0.0


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_file: str = "iron_condor.log") -> logging.Logger:
    """Setup logging with rotation."""
    logger = logging.getLogger("IronCondorEngine")
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers.clear()
    
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
# IRON CONDOR ENGINE
# ============================================================================

class IronCondorEngine:
    """
    Iron Condor Premium Strategy Engine.
    
    Implements iron condor strategy on liquid ETFs:
    - Sell OTM put spread (bull put)
    - Sell OTM call spread (bear call)
    - Collect premium with defined risk
    """
    
    def __init__(
        self,
        config: Optional[IronCondorConfig] = None,
        paper_trading: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Iron Condor Engine.
        
        Args:
            config: Strategy configuration
            paper_trading: If True, use paper trading endpoint
            logger: Logger instance
        """
        self.config = config or IronCondorConfig()
        self.config.paper_trading = paper_trading
        self.logger = logger or setup_logging()
        
        # Load API credentials
        self.api_key = os.environ.get("APCA_API_KEY_ID", "")
        self.api_secret = os.environ.get("APCA_API_SECRET_KEY", "")
        
        if not self.api_key or not self.api_secret:
            self.logger.warning("Alpaca API credentials not found in environment")
        
        # Initialize clients
        self.trading_client: Optional[TradingClient] = None
        self.data_client: Optional[StockHistoricalDataClient] = None
        self._init_clients()
        
        # Position tracking
        self.positions: dict[str, IronCondorPosition] = {}
        self.positions_file = "iron_condor_positions.json"
        self._load_positions()
        
        # Performance tracking
        self.stats = {
            "total_premium_collected": 0.0,
            "total_premium_returned": 0.0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_trades": 0,
        }
        
        self.logger.info(
            f"IronCondorEngine initialized | Paper Trading: {paper_trading} | "
            f"Delta Profile: {self.config.delta_profile}"
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
                        key: IronCondorPosition.from_dict(pos_data)
                        for key, pos_data in data.get("positions", {}).items()
                    }
                    self.stats = data.get("stats", self.stats)
                self.logger.info(f"Loaded {len(self.positions)} positions from file")
        except Exception as e:
            self.logger.error(f"Error loading positions: {e}")
            self.positions = {}
    
    def _save_positions(self) -> None:
        """Save positions to file."""
        try:
            data = {
                "positions": {
                    key: pos.to_dict() for key, pos in self.positions.items()
                },
                "stats": self.stats,
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
        days: int = 365
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
    
    def calculate_iv_rank(self, symbol: str) -> Tuple[float, float, float]:
        """
        Calculate IV Rank, IV Percentile, and current HV.
        
        IV Rank = (Current IV - 52-week Low IV) / (52-week High IV - 52-week Low IV)
        IV Percentile = % of days IV was lower than current
        
        Args:
            symbol: Stock ticker
            
        Returns:
            Tuple of (iv_rank, iv_percentile, hv_30)
        """
        df = self.get_historical_data(symbol, days=365)
        if df is None or len(df) < 30:
            return 50.0, 50.0, 0.20
        
        # Calculate historical volatility
        returns = df["close"].pct_change().dropna()
        
        # Current 30-day HV (annualized)
        current_hv = returns.tail(30).std() * np.sqrt(252)
        
        # Calculate rolling 30-day HV for the year
        rolling_hv = returns.rolling(30).std() * np.sqrt(252)
        rolling_hv = rolling_hv.dropna()
        
        if len(rolling_hv) < 10:
            return 50.0, 50.0, float(current_hv)
        
        # IV Rank calculation
        hv_min = rolling_hv.min()
        hv_max = rolling_hv.max()
        
        if hv_max == hv_min:
            iv_rank = 50.0
        else:
            iv_rank = (current_hv - hv_min) / (hv_max - hv_min) * 100
            iv_rank = float(np.clip(iv_rank, 0, 100))
        
        # IV Percentile calculation
        iv_percentile = (rolling_hv < current_hv).sum() / len(rolling_hv) * 100
        iv_percentile = float(np.clip(iv_percentile, 0, 100))
        
        self.logger.debug(
            f"{symbol} - IV Rank: {iv_rank:.1f}%, IV Percentile: {iv_percentile:.1f}%, "
            f"HV30: {current_hv*100:.1f}%"
        )
        
        return iv_rank, iv_percentile, float(current_hv)
    
    # ========================================================================
    # CANDIDATE SCREENING
    # ========================================================================
    
    def find_candidates(self) -> list[CondorCandidate]:
        """
        Find Iron Condor candidates from target underlyings.
        
        Returns:
            List of CondorCandidate objects meeting entry criteria
        """
        self.logger.info("Screening for Iron Condor candidates...")
        candidates: list[CondorCandidate] = []
        
        for symbol in self.config.target_underlyings:
            # Skip if already have position
            if symbol in self.positions:
                self.logger.debug(f"Skipping {symbol} - already have position")
                continue
            
            try:
                # Get current price
                price = self.get_stock_price(symbol)
                if price is None:
                    continue
                
                # Calculate IV metrics
                iv_rank, iv_percentile, hv_30 = self.calculate_iv_rank(symbol)
                
                # Check IV Rank threshold
                if iv_rank < self.config.min_iv_rank:
                    self.logger.info(
                        f"{symbol} - IV Rank {iv_rank:.1f}% below threshold "
                        f"({self.config.min_iv_rank}%)"
                    )
                    continue
                
                # Generate condor structures
                condor_candidates = self._generate_condor_structures(
                    symbol, price, iv_rank, iv_percentile, hv_30
                )
                
                candidates.extend(condor_candidates)
                
            except Exception as e:
                self.logger.error(f"Error screening {symbol}: {e}")
                continue
        
        # Sort by score (descending)
        candidates.sort(key=lambda x: x.score, reverse=True)
        
        self.logger.info(f"Found {len(candidates)} condor candidates")
        return candidates
    
    def _generate_condor_structures(
        self,
        symbol: str,
        price: float,
        iv_rank: float,
        iv_percentile: float,
        hv_30: float
    ) -> list[CondorCandidate]:
        """Generate potential condor structures for a symbol."""
        candidates: list[CondorCandidate] = []
        
        # Get delta targets based on profile
        delta_config = self.config.delta_targets[self.config.delta_profile]
        target_delta = (delta_config["min"] + delta_config["max"]) / 2
        target_pop = delta_config["pop"]
        
        # Estimate IV for premium calculation (use HV as proxy)
        iv = max(hv_30, 0.15)  # Minimum 15% IV
        
        # Generate expirations in target range
        now = datetime.now()
        for dte in [30, 35, 40, 45]:
            if not (self.config.target_dte_min <= dte <= self.config.target_dte_max):
                continue
            
            expiry = now + timedelta(days=dte)
            
            # Calculate strikes based on delta targeting
            # Using simplified delta approximation
            time_factor = np.sqrt(dte / 365)
            std_dev = price * iv * time_factor
            
            # Put strikes (OTM)
            put_short_strike = round(price - target_delta * std_dev * 3, 0)
            put_long_strike = put_short_strike - self.config.wing_width_dollars
            
            # Call strikes (OTM)
            call_short_strike = round(price + target_delta * std_dev * 3, 0)
            call_long_strike = call_short_strike + self.config.wing_width_dollars
            
            # Estimate premiums (simplified Black-Scholes approximation)
            put_spread_credit = self._estimate_spread_credit(
                price, put_short_strike, put_long_strike, iv, dte, "put"
            )
            call_spread_credit = self._estimate_spread_credit(
                price, call_short_strike, call_long_strike, iv, dte, "call"
            )
            
            total_credit = put_spread_credit + call_spread_credit
            wing_width = self.config.wing_width_dollars
            credit_ratio = (total_credit / wing_width) * 100
            
            # Check credit ratio threshold
            if credit_ratio < self.config.min_credit_pct:
                continue
            
            # Estimate Greeks
            delta = self._estimate_condor_delta(
                price, put_short_strike, call_short_strike, iv, dte
            )
            theta = total_credit / dte * 0.5  # Simplified theta
            vega = total_credit * 0.1  # Simplified vega
            
            # Calculate probability of profit
            pop = self._estimate_pop(
                price, put_short_strike, call_short_strike, iv, dte
            )
            
            # Score the candidate
            score = self._score_candidate(
                iv_rank, credit_ratio, pop, dte, abs(delta)
            )
            
            candidate = CondorCandidate(
                symbol=symbol,
                price=price,
                iv_rank=iv_rank,
                iv_percentile=iv_percentile,
                hv_30=hv_30,
                expiry=expiry,
                put_long_strike=put_long_strike,
                put_short_strike=put_short_strike,
                call_short_strike=call_short_strike,
                call_long_strike=call_long_strike,
                put_spread_credit=put_spread_credit,
                call_spread_credit=call_spread_credit,
                total_credit=total_credit,
                delta=delta,
                theta=theta,
                vega=vega,
                pop=pop,
                credit_to_width_ratio=credit_ratio,
                score=score,
            )
            
            candidates.append(candidate)
        
        return candidates
    
    def _estimate_spread_credit(
        self,
        price: float,
        short_strike: float,
        long_strike: float,
        iv: float,
        dte: int,
        option_type: str
    ) -> float:
        """Estimate credit spread premium."""
        time_factor = np.sqrt(dte / 365)
        
        if option_type == "put":
            # OTM put spread
            short_moneyness = (price - short_strike) / price
            long_moneyness = (price - long_strike) / price
        else:
            # OTM call spread
            short_moneyness = (short_strike - price) / price
            long_moneyness = (long_strike - price) / price
        
        # Simplified premium estimation
        short_premium = price * iv * time_factor * 0.4 * np.exp(-short_moneyness * 5)
        long_premium = price * iv * time_factor * 0.4 * np.exp(-long_moneyness * 5)
        
        credit = max(0, short_premium - long_premium)
        return round(credit, 2)
    
    def _estimate_condor_delta(
        self,
        price: float,
        put_short_strike: float,
        call_short_strike: float,
        iv: float,
        dte: int
    ) -> float:
        """Estimate net delta of condor."""
        # Simplified: condor delta based on proximity to strikes
        put_delta = -0.25 * np.exp(-((price - put_short_strike) / price) * 10)
        call_delta = 0.25 * np.exp(-((call_short_strike - price) / price) * 10)
        
        return round(put_delta + call_delta, 3)
    
    def _estimate_pop(
        self,
        price: float,
        put_short_strike: float,
        call_short_strike: float,
        iv: float,
        dte: int
    ) -> float:
        """Estimate probability of profit."""
        # Simplified: based on distance from strikes relative to expected move
        expected_move = price * iv * np.sqrt(dte / 365)
        
        put_distance = (price - put_short_strike) / expected_move
        call_distance = (call_short_strike - price) / expected_move
        
        # Approximate POP using standard normal
        from scipy.stats import norm
        put_prob_safe = norm.cdf(put_distance)
        call_prob_safe = norm.cdf(call_distance)
        
        pop = put_prob_safe * call_prob_safe
        return round(pop * 100, 1)
    
    def _score_candidate(
        self,
        iv_rank: float,
        credit_ratio: float,
        pop: float,
        dte: int,
        abs_delta: float
    ) -> float:
        """Score a condor candidate (higher is better)."""
        score = 0.0
        
        # IV Rank contribution (0-30 points, higher is better)
        score += min(iv_rank, 100) * 0.3
        
        # Credit ratio contribution (0-25 points)
        score += min(credit_ratio, 40) * 0.625
        
        # POP contribution (0-25 points)
        score += min(pop, 90) * 0.28
        
        # DTE in sweet spot (35-40 days is ideal)
        if 35 <= dte <= 40:
            score += 10
        elif 30 <= dte <= 45:
            score += 5
        
        # Delta neutrality bonus (0-10 points)
        score += max(0, 10 - abs_delta * 100)
        
        return round(score, 2)
    
    # ========================================================================
    # STRIKE SELECTION
    # ========================================================================
    
    def select_strikes(
        self,
        symbol: str,
        price: float,
        iv: float,
        dte: int
    ) -> Optional[dict]:
        """
        Select optimal strikes for an Iron Condor.
        
        Uses delta targeting based on configuration profile.
        
        Args:
            symbol: Underlying symbol
            price: Current underlying price
            iv: Implied volatility
            dte: Days to expiration
            
        Returns:
            Dictionary with strike selection details or None
        """
        delta_config = self.config.delta_targets[self.config.delta_profile]
        target_delta = (delta_config["min"] + delta_config["max"]) / 2
        
        # Calculate expected move
        expected_move = price * iv * np.sqrt(dte / 365)
        
        # Use delta-based strike calculation
        # For ~25 delta, strikes are typically ~1 std dev OTM
        std_multiplier = 1.0 / target_delta  # Approximate
        
        # Put side (OTM below price)
        put_short_strike = round(price - expected_move * std_multiplier * 0.5, 0)
        put_long_strike = put_short_strike - self.config.wing_width_dollars
        
        # Call side (OTM above price)
        call_short_strike = round(price + expected_move * std_multiplier * 0.5, 0)
        call_long_strike = call_short_strike + self.config.wing_width_dollars
        
        # Validate structure
        if put_short_strike >= price or call_short_strike <= price:
            self.logger.warning(f"Invalid strike structure for {symbol}")
            return None
        
        # Calculate estimated premiums
        put_spread = self._estimate_spread_credit(
            price, put_short_strike, put_long_strike, iv, dte, "put"
        )
        call_spread = self._estimate_spread_credit(
            price, call_short_strike, call_long_strike, iv, dte, "call"
        )
        
        total_credit = put_spread + call_spread
        wing_width = self.config.wing_width_dollars
        credit_ratio = (total_credit / wing_width) * 100
        
        if credit_ratio < self.config.min_credit_pct:
            self.logger.warning(
                f"Credit ratio {credit_ratio:.1f}% below threshold "
                f"({self.config.min_credit_pct}%)"
            )
            return None
        
        return {
            "symbol": symbol,
            "price": price,
            "dte": dte,
            "put_long_strike": put_long_strike,
            "put_short_strike": put_short_strike,
            "call_short_strike": call_short_strike,
            "call_long_strike": call_long_strike,
            "put_spread_credit": put_spread,
            "call_spread_credit": call_spread,
            "total_credit": total_credit,
            "wing_width": wing_width,
            "credit_ratio": credit_ratio,
            "max_loss": (wing_width - total_credit) * 100,  # Per contract
        }
    
    # ========================================================================
    # EXECUTION
    # ========================================================================
    
    def place_condor(
        self,
        candidate: CondorCandidate,
        contracts: int = 1
    ) -> Optional[str]:
        """
        Place an Iron Condor order.
        
        Args:
            candidate: CondorCandidate with trade details
            contracts: Number of contracts
            
        Returns:
            Position ID if successful, None otherwise
        """
        self.logger.info(
            f"Placing Iron Condor on {candidate.symbol}: "
            f"P{candidate.put_long_strike}/{candidate.put_short_strike} | "
            f"C{candidate.call_short_strike}/{candidate.call_long_strike} | "
            f"Credit: ${candidate.total_credit:.2f} | "
            f"Contracts: {contracts}"
        )
        
        # Create position object
        position = IronCondorPosition(
            symbol=candidate.symbol,
            entry_date=datetime.now(),
            expiry=candidate.expiry,
            put_long=CondorLeg(
                option_type="put",
                strike=candidate.put_long_strike,
                side="long",
                delta=-0.10,
                premium=candidate.put_spread_credit * 0.3,
                iv=candidate.hv_30,
            ),
            put_short=CondorLeg(
                option_type="put",
                strike=candidate.put_short_strike,
                side="short",
                delta=-0.25,
                premium=candidate.put_spread_credit * 0.7,
                iv=candidate.hv_30,
            ),
            call_short=CondorLeg(
                option_type="call",
                strike=candidate.call_short_strike,
                side="short",
                delta=0.25,
                premium=candidate.call_spread_credit * 0.7,
                iv=candidate.hv_30,
            ),
            call_long=CondorLeg(
                option_type="call",
                strike=candidate.call_long_strike,
                side="long",
                delta=0.10,
                premium=candidate.call_spread_credit * 0.3,
                iv=candidate.hv_30,
            ),
            contracts=contracts,
            credit_received=candidate.total_credit * contracts * 100,
            max_loss=(candidate.put_short_strike - candidate.put_long_strike - 
                     candidate.total_credit) * contracts * 100,
            current_value=candidate.total_credit * contracts * 100,
            delta=candidate.delta * contracts,
            theta=candidate.theta * contracts,
            vega=candidate.vega * contracts,
            state=CondorState.OPEN,
            iv_rank_at_entry=candidate.iv_rank,
        )
        
        # Store position
        position_id = f"{candidate.symbol}_{candidate.expiry.strftime('%Y%m%d')}"
        self.positions[position_id] = position
        
        # Update stats
        self.stats["total_premium_collected"] += position.credit_received
        self.stats["total_trades"] += 1
        
        self._save_positions()
        
        self.logger.info(
            f"[SIMULATED] Iron Condor opened | Position ID: {position_id} | "
            f"Credit: ${position.credit_received:.2f} | Max Loss: ${position.max_loss:.2f}"
        )
        
        return position_id
    
    # ========================================================================
    # POSITION MANAGEMENT
    # ========================================================================
    
    def manage_positions(self) -> None:
        """
        Manage existing Iron Condor positions.
        
        Checks:
        - Profit target (50% of max profit)
        - DTE threshold (close at 7 DTE)
        - Stop loss (2x premium)
        - Roll opportunities
        """
        self.logger.info("Managing Iron Condor positions...")
        
        for position_id, position in list(self.positions.items()):
            if position.state != CondorState.OPEN:
                continue
            
            try:
                self._manage_single_position(position_id, position)
            except Exception as e:
                self.logger.error(f"Error managing {position_id}: {e}")
    
    def _manage_single_position(
        self,
        position_id: str,
        position: IronCondorPosition
    ) -> None:
        """Manage a single Iron Condor position."""
        
        # Update current price
        current_price = self.get_stock_price(position.symbol)
        if current_price is None:
            return
        
        # Estimate current position value (simplified)
        dte = position.dte
        if dte <= 0:
            dte = 1
        
        # Time decay: assume position decays towards zero
        time_decay_factor = dte / 45  # Original 45 DTE
        current_value = position.credit_received * time_decay_factor
        
        # Adjust for price movement
        if current_price < position.put_short.strike:
            # Put side threatened
            loss = (position.put_short.strike - current_price) * position.contracts * 100
            current_value = position.credit_received + loss
        elif current_price > position.call_short.strike:
            # Call side threatened
            loss = (current_price - position.call_short.strike) * position.contracts * 100
            current_value = position.credit_received + loss
        
        position.current_value = current_value
        pnl_pct = position.pnl_pct
        
        self.logger.debug(
            f"{position_id}: Price=${current_price:.2f} | DTE={dte} | "
            f"P&L={pnl_pct:.1f}%"
        )
        
        # Check exit conditions
        
        # 1. Profit target (50%)
        if pnl_pct >= self.config.profit_target_pct:
            self.logger.info(
                f"Profit target reached for {position_id}: {pnl_pct:.1f}%"
            )
            self._close_position(position_id, position, CondorState.PROFIT_TARGET)
            return
        
        # 2. DTE threshold
        if dte <= self.config.close_dte:
            self.logger.info(
                f"DTE threshold reached for {position_id}: {dte} days"
            )
            self._close_position(position_id, position, CondorState.CLOSED_DTE)
            return
        
        # 3. Stop loss
        if pnl_pct <= -self.config.max_loss_multiplier * 100:
            self.logger.warning(
                f"Stop loss triggered for {position_id}: {pnl_pct:.1f}%"
            )
            self._close_position(position_id, position, CondorState.STOPPED_OUT)
            return
        
        # 4. Check for roll opportunity (untested side)
        self._check_roll_opportunity(position_id, position, current_price)
        
        self._save_positions()
    
    def _close_position(
        self,
        position_id: str,
        position: IronCondorPosition,
        reason: CondorState
    ) -> None:
        """Close an Iron Condor position."""
        
        position.state = reason
        pnl = position.pnl
        
        # Update stats
        self.stats["total_premium_returned"] += position.current_value
        if pnl > 0:
            self.stats["winning_trades"] += 1
        else:
            self.stats["losing_trades"] += 1
        
        self.logger.info(
            f"[SIMULATED] Closed {position_id} | Reason: {reason.value} | "
            f"P&L: ${pnl:.2f} ({position.pnl_pct:.1f}%)"
        )
        
        self._save_positions()
    
    def _check_roll_opportunity(
        self,
        position_id: str,
        position: IronCondorPosition,
        current_price: float
    ) -> None:
        """Check and execute roll on untested side."""
        
        # Check if one side is being tested
        put_distance = (current_price - position.put_short.strike) / current_price
        call_distance = (position.call_short.strike - current_price) / current_price
        
        roll_threshold = 0.02  # 2% from short strike
        
        if put_distance < roll_threshold and call_distance > 0.05:
            # Put side tested, call side safe - consider rolling call down
            self.logger.info(
                f"{position_id}: Put side tested, consider rolling call down"
            )
        elif call_distance < roll_threshold and put_distance > 0.05:
            # Call side tested, put side safe - consider rolling put up
            self.logger.info(
                f"{position_id}: Call side tested, consider rolling put up"
            )
    
    # ========================================================================
    # REPORTING
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
            }
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return {"error": str(e)}
    
    def get_status(self) -> str:
        """Get formatted status report."""
        account = self.get_account_info()
        
        lines = [
            "=" * 70,
            "IRON CONDOR ENGINE STATUS",
            "=" * 70,
            "",
            "ACCOUNT:",
            f"  Portfolio Value: ${account.get('portfolio_value', 0):,.2f}",
            f"  Buying Power:    ${account.get('buying_power', 0):,.2f}",
            "",
            "STRATEGY PERFORMANCE:",
            f"  Total Premium Collected: ${self.stats['total_premium_collected']:,.2f}",
            f"  Total Premium Returned:  ${self.stats['total_premium_returned']:,.2f}",
            f"  Net P&L:                 ${self.stats['total_premium_collected'] - self.stats['total_premium_returned']:,.2f}",
            f"  Winning Trades:          {self.stats['winning_trades']}",
            f"  Losing Trades:           {self.stats['losing_trades']}",
            f"  Total Trades:            {self.stats['total_trades']}",
            "",
            "OPEN POSITIONS:",
        ]
        
        open_positions = [
            (pid, pos) for pid, pos in self.positions.items()
            if pos.state == CondorState.OPEN
        ]
        
        if open_positions:
            lines.append(
                f"  {'Symbol':<8} {'Expiry':<12} {'Put Spread':<16} "
                f"{'Call Spread':<16} {'Credit':>10} {'P&L':>10} {'DTE':>5}"
            )
            lines.append("  " + "-" * 80)
            
            for pid, pos in open_positions:
                put_spread = f"{pos.put_long.strike:.0f}/{pos.put_short.strike:.0f}"
                call_spread = f"{pos.call_short.strike:.0f}/{pos.call_long.strike:.0f}"
                
                lines.append(
                    f"  {pos.symbol:<8} {pos.expiry.strftime('%Y-%m-%d'):<12} "
                    f"{put_spread:<16} {call_spread:<16} "
                    f"${pos.credit_received:>8.2f} ${pos.pnl:>8.2f} {pos.dte:>5d}"
                )
        else:
            lines.append("  No open positions")
        
        lines.append("")
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    # ========================================================================
    # MAIN ORCHESTRATION
    # ========================================================================
    
    def run_cycle(self) -> None:
        """
        Main orchestration loop for Iron Condor strategy.
        
        Steps:
        1. Manage existing positions
        2. Screen for new candidates
        3. Open new positions if capacity available
        """
        self.logger.info("=" * 60)
        self.logger.info("STARTING IRON CONDOR CYCLE")
        self.logger.info("=" * 60)
        
        # Step 1: Manage existing positions
        self.manage_positions()
        
        # Step 2: Check capacity
        open_count = sum(
            1 for pos in self.positions.values()
            if pos.state == CondorState.OPEN
        )
        
        if open_count >= self.config.max_positions:
            self.logger.info(f"At max positions ({open_count}). No new trades.")
            print(self.get_status())
            return
        
        # Step 3: Find candidates
        candidates = self.find_candidates()
        
        if not candidates:
            self.logger.info("No suitable candidates found")
            print(self.get_status())
            return
        
        # Step 4: Open new positions
        for candidate in candidates:
            if open_count >= self.config.max_positions:
                break
            
            # Calculate position size
            account = self.get_account_info()
            portfolio_value = account.get("portfolio_value", 100000)
            
            max_risk = portfolio_value * (self.config.max_position_pct / 100)
            max_loss_per_contract = (
                candidate.put_short_strike - candidate.put_long_strike - 
                candidate.total_credit
            ) * 100
            
            contracts = max(1, int(max_risk / max_loss_per_contract))
            contracts = min(contracts, 5)  # Cap at 5 contracts
            
            # Place the condor
            position_id = self.place_condor(candidate, contracts)
            
            if position_id:
                open_count += 1
        
        # Print status
        print(self.get_status())
        
        self.logger.info("IRON CONDOR CYCLE COMPLETE")


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Main entry point with CLI interface."""
    parser = argparse.ArgumentParser(
        description="Iron Condor Premium Strategy Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python v41_iron_condor_engine.py --test       # Run in test mode
  python v41_iron_condor_engine.py --status     # Show current status
  python v41_iron_condor_engine.py --trade      # Run trading cycle
  python v41_iron_condor_engine.py --candidates # Show candidates
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
        help="Show Iron Condor candidates"
    )
    
    parser.add_argument(
        "--profile",
        choices=["conservative", "balanced", "aggressive"],
        default="balanced",
        help="Delta profile (default: balanced)"
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
        help="Use live trading"
    )
    
    args = parser.parse_args()
    
    # Determine paper trading mode
    paper_trading = not args.live
    
    if args.live:
        confirm = input(
            "WARNING: Live trading mode. Type 'YES' to confirm: "
        )
        if confirm != "YES":
            print("Live trading cancelled.")
            sys.exit(0)
    
    # Initialize engine
    print("\n" + "=" * 60)
    print("IRON CONDOR PREMIUM STRATEGY ENGINE")
    print("=" * 60 + "\n")
    
    config = IronCondorConfig(
        paper_trading=paper_trading,
        delta_profile=args.profile,
    )
    engine = IronCondorEngine(config=config, paper_trading=paper_trading)
    
    # Execute requested action
    if args.test:
        print("Running in TEST mode...\n")
        
        # Test IV rank calculation
        print("Testing IV Rank calculation...")
        for symbol in ["SPY", "QQQ", "IWM"]:
            iv_rank, iv_pct, hv = engine.calculate_iv_rank(symbol)
            print(
                f"  {symbol}: IV Rank={iv_rank:.1f}%, "
                f"IV Percentile={iv_pct:.1f}%, HV30={hv*100:.1f}%"
            )
        
        # Test candidate finding
        print("\nFinding condor candidates...")
        candidates = engine.find_candidates()
        
        if candidates:
            print(f"\nTop {min(5, len(candidates))} Candidates:")
            print("-" * 80)
            for c in candidates[:5]:
                print(
                    f"  {c.symbol} | P{c.put_long_strike:.0f}/{c.put_short_strike:.0f} "
                    f"C{c.call_short_strike:.0f}/{c.call_long_strike:.0f} | "
                    f"Credit: ${c.total_credit:.2f} | POP: {c.pop:.1f}% | "
                    f"Score: {c.score:.1f}"
                )
        
        print("\nTest complete!")
    
    elif args.status:
        print(engine.get_status())
    
    elif args.trade:
        print("Running trading cycle...\n")
        engine.run_cycle()
    
    elif args.candidates:
        print("Screening for Iron Condor candidates...\n")
        candidates = engine.find_candidates()
        
        if candidates:
            print(f"Found {len(candidates)} candidates:\n")
            print(
                f"  {'Symbol':<8} {'IV Rank':>8} {'Put Spread':<14} "
                f"{'Call Spread':<14} {'Credit':>8} {'POP':>6} {'Score':>6}"
            )
            print("  " + "-" * 70)
            
            for c in candidates:
                put_spread = f"{c.put_long_strike:.0f}/{c.put_short_strike:.0f}"
                call_spread = f"{c.call_short_strike:.0f}/{c.call_long_strike:.0f}"
                print(
                    f"  {c.symbol:<8} {c.iv_rank:>7.1f}% "
                    f"{put_spread:<14} {call_spread:<14} "
                    f"${c.total_credit:>6.2f} {c.pop:>5.1f}% {c.score:>6.1f}"
                )
        else:
            print("No candidates found meeting criteria.")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    # Add scipy to path check
    try:
        from scipy.stats import norm
    except ImportError:
        print("Warning: scipy not installed. Install with: pip install scipy")
        print("Some probability calculations will be estimated.")
    
    main()
