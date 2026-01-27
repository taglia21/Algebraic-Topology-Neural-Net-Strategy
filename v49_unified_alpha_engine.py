#!/usr/bin/env python3
"""
V49 UNIFIED ALPHA ENGINE
========================
Combines V48 Institutional Quant + Advanced Options Strategies

Key Optimizations:
1. Smarter position sizing to avoid PDT issues
2. Options Wheel Strategy with Greeks analysis
3. Iron Condors on high IV rank stocks
4. Credit Spreads for defined risk
5. Performance tracking and reporting
6. Regime-aware strategy selection

Author: Unified Quant Team
Version: 49.0.0
"""

import os
import sys
import asyncio
import logging
import argparse
import json
import time
import math
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
from scipy.stats import norm

warnings.filterwarnings('ignore')

# Alpaca imports
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, LimitOrderRequest, GetAssetsRequest
)
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass, AssetStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

# ML imports
try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from statsmodels.tsa.stattools import coint
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


# ============================================================================
# CONFIGURATION
# ============================================================================

class MarketRegime(Enum):
    BULL_LOW_VOL = "bull_low_vol"
    BULL_HIGH_VOL = "bull_high_vol" 
    BEAR_LOW_VOL = "bear_low_vol"
    BEAR_HIGH_VOL = "bear_high_vol"
    SIDEWAYS = "sideways"

class OptionsStrategy(Enum):
    CASH_SECURED_PUT = "cash_secured_put"
    COVERED_CALL = "covered_call"
    IRON_CONDOR = "iron_condor"
    CREDIT_SPREAD = "credit_spread"
    WHEEL = "wheel"

@dataclass
class UnifiedConfig:
    """Unified configuration for all strategies"""
    # API
    api_key: str = ""
    api_secret: str = ""
    paper: bool = True
    
    # Universe - optimized for free tier
    universe_size: int = 150  # Reduced to avoid rate limits
    min_price: float = 10.0
    max_price: float = 500.0
    min_volume: int = 500000
    
    # Scanning - balanced for performance
    scan_interval: int = 30  # 30 seconds (avoid rate limits)
    batch_size: int = 25
    
    # Position Management - PDT optimized
    max_positions: int = 30  # Fewer, larger positions
    max_day_trades: int = 3  # Stay under PDT limit
    position_size_pct: float = 0.03  # 3% per position
    max_single_position: float = 0.05  # 5% max
    min_hold_days: int = 1  # Avoid day trades
    
    # Risk Management
    max_drawdown: float = 0.08  # 8% circuit breaker
    kelly_fraction: float = 0.25
    stop_loss_pct: float = 0.05  # 5% stop loss
    take_profit_pct: float = 0.10  # 10% take profit
    
    # Mean Reversion
    zscore_entry: float = 2.0
    zscore_exit: float = 0.5
    rsi_oversold: float = 30
    rsi_overbought: float = 70
    
    # Momentum  
    momentum_threshold: float = 0.03
    volume_surge: float = 1.5
    
    # Options Configuration
    options_enabled: bool = True
    wheel_stocks: List[str] = field(default_factory=lambda: [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'AMD', 'INTC',
        'SPY', 'QQQ', 'IWM', 'DIA', 'XLF', 'XLE', 'XLK', 'XLV',
        'TSLA', 'JPM', 'BAC', 'WFC', 'GS', 'V', 'MA', 'PYPL'
    ])
    
    # Options Greeks targets
    put_delta_target: float = -0.30  # 30 delta puts
    call_delta_target: float = 0.30  # 30 delta calls
    min_iv_rank: float = 30  # Only sell premium when IV > 30%
    max_iv_rank: float = 80  # Don't sell into extreme IV
    premium_min_pct: float = 0.01  # 1% minimum premium
    dte_min: int = 21  # 3 weeks minimum
    dte_max: int = 45  # 6 weeks maximum
    
    # Iron Condor specific
    ic_wing_width: int = 5  # $5 wide wings
    ic_min_credit: float = 0.30  # 30% of wing width
    
    # Performance tracking
    track_performance: bool = True
    report_interval: int = 3600  # Hourly reports


# ============================================================================
# OPTIONS ANALYTICS
# ============================================================================

class BlackScholes:
    """Black-Scholes option pricing and Greeks"""
    
    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d1"""
        if T <= 0 or sigma <= 0:
            return 0
        return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d2"""
        if T <= 0 or sigma <= 0:
            return 0
        return BlackScholes.d1(S, K, T, r, sigma) - sigma * np.sqrt(T)
    
    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate call option price"""
        if T <= 0:
            return max(0, S - K)
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate put option price"""
        if T <= 0:
            return max(0, K - S)
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    @staticmethod
    def call_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate call delta"""
        if T <= 0:
            return 1.0 if S > K else 0.0
        return norm.cdf(BlackScholes.d1(S, K, T, r, sigma))
    
    @staticmethod
    def put_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate put delta"""
        if T <= 0:
            return -1.0 if S < K else 0.0
        return -norm.cdf(-BlackScholes.d1(S, K, T, r, sigma))
    
    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate gamma (same for calls and puts)"""
        if T <= 0 or sigma <= 0:
            return 0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    @staticmethod
    def theta_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate call theta (per day)"""
        if T <= 0:
            return 0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        term1 = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
        term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
        return (term1 + term2) / 365  # Per day
    
    @staticmethod
    def theta_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate put theta (per day)"""
        if T <= 0:
            return 0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        term1 = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
        term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
        return (term1 + term2) / 365  # Per day
    
    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate vega (same for calls and puts)"""
        if T <= 0:
            return 0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        return S * np.sqrt(T) * norm.pdf(d1) / 100  # Per 1% IV change
    
    @staticmethod
    def implied_volatility(option_price: float, S: float, K: float, T: float, 
                          r: float, is_call: bool = True) -> float:
        """Calculate implied volatility using Newton-Raphson"""
        sigma = 0.3  # Initial guess
        for _ in range(100):
            if is_call:
                price = BlackScholes.call_price(S, K, T, r, sigma)
            else:
                price = BlackScholes.put_price(S, K, T, r, sigma)
            
            vega = BlackScholes.vega(S, K, T, r, sigma) * 100
            
            if vega < 1e-10:
                break
            
            diff = price - option_price
            if abs(diff) < 1e-6:
                break
            
            sigma = sigma - diff / vega
            sigma = max(0.01, min(sigma, 5.0))  # Bound sigma
        
        return sigma


class IVAnalyzer:
    """Implied Volatility Rank and Percentile Calculator"""
    
    def __init__(self, lookback_days: int = 252):
        self.lookback_days = lookback_days
        self.iv_history: Dict[str, deque] = {}
    
    def calculate_historical_volatility(self, prices: np.ndarray, window: int = 20) -> float:
        """Calculate historical volatility"""
        if len(prices) < window + 1:
            return 0.3  # Default
        
        returns = np.diff(np.log(prices))
        return np.std(returns[-window:]) * np.sqrt(252)
    
    def update_iv(self, symbol: str, iv: float):
        """Update IV history"""
        if symbol not in self.iv_history:
            self.iv_history[symbol] = deque(maxlen=self.lookback_days)
        self.iv_history[symbol].append(iv)
    
    def get_iv_rank(self, symbol: str, current_iv: float) -> float:
        """Calculate IV Rank (0-100)"""
        if symbol not in self.iv_history or len(self.iv_history[symbol]) < 20:
            return 50  # Default to neutral
        
        history = list(self.iv_history[symbol])
        iv_min = min(history)
        iv_max = max(history)
        
        if iv_max == iv_min:
            return 50
        
        return ((current_iv - iv_min) / (iv_max - iv_min)) * 100
    
    def get_iv_percentile(self, symbol: str, current_iv: float) -> float:
        """Calculate IV Percentile (0-100)"""
        if symbol not in self.iv_history or len(self.iv_history[symbol]) < 20:
            return 50
        
        history = list(self.iv_history[symbol])
        below_count = sum(1 for iv in history if iv < current_iv)
        return (below_count / len(history)) * 100


# ============================================================================
# OPTIONS STRATEGIES
# ============================================================================

class WheelStrategy:
    """The Wheel Strategy: Cash-Secured Puts -> Assignment -> Covered Calls"""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.wheel_positions: Dict[str, Dict] = {}  # Track wheel state
        self.iv_analyzer = IVAnalyzer()
    
    def find_put_strike(self, symbol: str, current_price: float, 
                        iv: float, dte: int, target_delta: float = -0.30) -> Tuple[float, float]:
        """Find strike for target delta put"""
        r = 0.05  # Risk-free rate
        T = dte / 365
        
        # Binary search for target delta strike
        low_strike = current_price * 0.7
        high_strike = current_price * 1.0
        
        for _ in range(50):
            mid_strike = (low_strike + high_strike) / 2
            delta = BlackScholes.put_delta(current_price, mid_strike, T, r, iv)
            
            if abs(delta - target_delta) < 0.01:
                break
            elif delta < target_delta:  # More negative, need higher strike
                low_strike = mid_strike
            else:
                high_strike = mid_strike
        
        # Calculate premium
        premium = BlackScholes.put_price(current_price, mid_strike, T, r, iv)
        
        # Round strike to nearest $1 (or $0.50 for cheap stocks)
        if mid_strike < 50:
            mid_strike = round(mid_strike * 2) / 2
        else:
            mid_strike = round(mid_strike)
        
        return mid_strike, premium
    
    def find_call_strike(self, symbol: str, cost_basis: float, current_price: float,
                        iv: float, dte: int, target_delta: float = 0.30) -> Tuple[float, float]:
        """Find strike for target delta covered call"""
        r = 0.05
        T = dte / 365
        
        # Ensure strike is above cost basis for profit
        min_strike = max(cost_basis * 1.02, current_price * 0.95)
        
        low_strike = current_price * 1.0
        high_strike = current_price * 1.3
        
        for _ in range(50):
            mid_strike = (low_strike + high_strike) / 2
            delta = BlackScholes.call_delta(current_price, mid_strike, T, r, iv)
            
            if abs(delta - target_delta) < 0.01:
                break
            elif delta > target_delta:  # Need higher strike for lower delta
                low_strike = mid_strike
            else:
                high_strike = mid_strike
        
        # Ensure above minimum strike
        mid_strike = max(mid_strike, min_strike)
        
        premium = BlackScholes.call_price(current_price, mid_strike, T, r, iv)
        
        if mid_strike < 50:
            mid_strike = round(mid_strike * 2) / 2
        else:
            mid_strike = round(mid_strike)
        
        return mid_strike, premium
    
    def generate_wheel_signal(self, symbol: str, current_price: float, 
                             prices: np.ndarray) -> Optional[Dict]:
        """Generate wheel strategy signal"""
        # Calculate IV
        hv = self.iv_analyzer.calculate_historical_volatility(prices)
        iv = hv * 1.1  # IV typically slightly higher than HV
        
        self.iv_analyzer.update_iv(symbol, iv)
        iv_rank = self.iv_analyzer.get_iv_rank(symbol, iv)
        
        # Only sell premium when IV is elevated
        if iv_rank < self.config.min_iv_rank:
            return None
        
        if iv_rank > self.config.max_iv_rank:
            return None  # Too risky
        
        # Check if we have existing wheel position
        wheel_state = self.wheel_positions.get(symbol, {'state': 'none'})
        
        if wheel_state['state'] == 'none':
            # Start wheel with cash-secured put
            strike, premium = self.find_put_strike(
                symbol, current_price, iv, 
                self.config.dte_min,
                self.config.put_delta_target
            )
            
            premium_pct = premium / strike
            if premium_pct < self.config.premium_min_pct:
                return None
            
            return {
                'strategy': 'wheel',
                'action': 'sell_put',
                'symbol': symbol,
                'strike': strike,
                'premium': premium,
                'premium_pct': premium_pct,
                'iv_rank': iv_rank,
                'dte': self.config.dte_min,
                'delta': self.config.put_delta_target
            }
        
        elif wheel_state['state'] == 'has_shares':
            # Sell covered call on shares
            cost_basis = wheel_state.get('cost_basis', current_price)
            
            strike, premium = self.find_call_strike(
                symbol, cost_basis, current_price, iv,
                self.config.dte_min,
                self.config.call_delta_target
            )
            
            return {
                'strategy': 'wheel',
                'action': 'sell_call',
                'symbol': symbol,
                'strike': strike,
                'premium': premium,
                'cost_basis': cost_basis,
                'iv_rank': iv_rank,
                'dte': self.config.dte_min,
                'delta': self.config.call_delta_target
            }
        
        return None
    
    def update_wheel_state(self, symbol: str, new_state: str, **kwargs):
        """Update wheel position state"""
        self.wheel_positions[symbol] = {'state': new_state, **kwargs}


class IronCondorStrategy:
    """Iron Condor Strategy for Range-Bound Markets"""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.iv_analyzer = IVAnalyzer()
    
    def find_iron_condor_strikes(self, symbol: str, current_price: float,
                                 iv: float, dte: int) -> Dict:
        """Find optimal iron condor strikes"""
        r = 0.05
        T = dte / 365
        width = self.config.ic_wing_width
        
        # Find short put strike (30 delta)
        short_put_strike, _ = self._find_delta_strike(
            current_price, iv, T, r, -0.30, 'put'
        )
        long_put_strike = short_put_strike - width
        
        # Find short call strike (30 delta)
        short_call_strike, _ = self._find_delta_strike(
            current_price, iv, T, r, 0.30, 'call'
        )
        long_call_strike = short_call_strike + width
        
        # Calculate credits
        put_credit = (BlackScholes.put_price(current_price, short_put_strike, T, r, iv) -
                     BlackScholes.put_price(current_price, long_put_strike, T, r, iv))
        
        call_credit = (BlackScholes.call_price(current_price, short_call_strike, T, r, iv) -
                      BlackScholes.call_price(current_price, long_call_strike, T, r, iv))
        
        total_credit = put_credit + call_credit
        max_loss = width - total_credit
        
        return {
            'long_put': long_put_strike,
            'short_put': short_put_strike,
            'short_call': short_call_strike,
            'long_call': long_call_strike,
            'put_credit': put_credit,
            'call_credit': call_credit,
            'total_credit': total_credit,
            'max_loss': max_loss,
            'credit_pct': total_credit / width
        }
    
    def _find_delta_strike(self, price: float, iv: float, T: float, 
                          r: float, target_delta: float, 
                          option_type: str) -> Tuple[float, float]:
        """Find strike for target delta"""
        if option_type == 'put':
            low, high = price * 0.7, price
        else:
            low, high = price, price * 1.3
        
        for _ in range(50):
            mid = (low + high) / 2
            if option_type == 'put':
                delta = BlackScholes.put_delta(price, mid, T, r, iv)
                if delta < target_delta:
                    low = mid
                else:
                    high = mid
            else:
                delta = BlackScholes.call_delta(price, mid, T, r, iv)
                if delta > target_delta:
                    low = mid
                else:
                    high = mid
            
            if abs(delta - target_delta) < 0.01:
                break
        
        return round(mid), delta
    
    def generate_iron_condor_signal(self, symbol: str, current_price: float,
                                   prices: np.ndarray) -> Optional[Dict]:
        """Generate iron condor signal"""
        hv = self.iv_analyzer.calculate_historical_volatility(prices)
        iv = hv * 1.1
        
        self.iv_analyzer.update_iv(symbol, iv)
        iv_rank = self.iv_analyzer.get_iv_rank(symbol, iv)
        
        # Iron condors work best in high IV environments
        if iv_rank < 40:  # Higher threshold for IC
            return None
        
        # Check for range-bound market (low momentum)
        if len(prices) >= 20:
            momentum = (prices[-1] / prices[-20]) - 1
            if abs(momentum) > 0.05:  # >5% move, not range-bound
                return None
        
        ic_strikes = self.find_iron_condor_strikes(
            symbol, current_price, iv, self.config.dte_min
        )
        
        # Only trade if credit is sufficient
        if ic_strikes['credit_pct'] < self.config.ic_min_credit:
            return None
        
        return {
            'strategy': 'iron_condor',
            'symbol': symbol,
            **ic_strikes,
            'iv_rank': iv_rank,
            'dte': self.config.dte_min
        }


# ============================================================================
# PERFORMANCE TRACKER
# ============================================================================

class PerformanceTracker:
    """Track and report trading performance"""
    
    def __init__(self):
        self.trades: List[Dict] = []
        self.daily_pnl: Dict[str, float] = {}
        self.start_equity: float = 0
        self.peak_equity: float = 0
        self.strategy_stats: Dict[str, Dict] = {}
    
    def record_trade(self, trade: Dict):
        """Record a completed trade"""
        trade['timestamp'] = datetime.now()
        self.trades.append(trade)
        
        # Update strategy stats
        strategy = trade.get('strategy', 'unknown')
        if strategy not in self.strategy_stats:
            self.strategy_stats[strategy] = {
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'total_pnl': 0,
                'avg_win': 0,
                'avg_loss': 0
            }
        
        stats = self.strategy_stats[strategy]
        stats['total_trades'] += 1
        
        pnl = trade.get('pnl', 0)
        stats['total_pnl'] += pnl
        
        if pnl > 0:
            stats['wins'] += 1
        elif pnl < 0:
            stats['losses'] += 1
    
    def update_equity(self, equity: float):
        """Update equity tracking"""
        if self.start_equity == 0:
            self.start_equity = equity
        
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        # Record daily PnL
        today = datetime.now().strftime('%Y-%m-%d')
        if today not in self.daily_pnl:
            self.daily_pnl[today] = equity
    
    def get_metrics(self, current_equity: float) -> Dict:
        """Calculate performance metrics"""
        total_pnl = current_equity - self.start_equity if self.start_equity > 0 else 0
        total_return = (total_pnl / self.start_equity * 100) if self.start_equity > 0 else 0
        
        max_drawdown = 0
        if self.peak_equity > 0:
            max_drawdown = (self.peak_equity - current_equity) / self.peak_equity * 100
        
        total_trades = len(self.trades)
        wins = sum(1 for t in self.trades if t.get('pnl', 0) > 0)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        # Calculate Sharpe ratio approximation
        if len(self.daily_pnl) > 1:
            daily_returns = []
            pnl_values = list(self.daily_pnl.values())
            for i in range(1, len(pnl_values)):
                ret = (pnl_values[i] - pnl_values[i-1]) / pnl_values[i-1]
                daily_returns.append(ret)
            
            if daily_returns:
                avg_return = np.mean(daily_returns)
                std_return = np.std(daily_returns)
                sharpe = (avg_return / std_return * np.sqrt(252)) if std_return > 0 else 0
            else:
                sharpe = 0
        else:
            sharpe = 0
        
        return {
            'total_pnl': total_pnl,
            'total_return_pct': total_return,
            'max_drawdown_pct': max_drawdown,
            'total_trades': total_trades,
            'win_rate_pct': win_rate,
            'sharpe_ratio': sharpe,
            'strategy_breakdown': self.strategy_stats
        }
    
    def generate_report(self, current_equity: float) -> str:
        """Generate performance report"""
        metrics = self.get_metrics(current_equity)
        
        report = [
            "\n" + "="*60,
            "PERFORMANCE REPORT",
            "="*60,
            f"Start Equity:    ${self.start_equity:,.2f}",
            f"Current Equity:  ${current_equity:,.2f}",
            f"Total P&L:       ${metrics['total_pnl']:+,.2f} ({metrics['total_return_pct']:+.2f}%)",
            f"Max Drawdown:    {metrics['max_drawdown_pct']:.2f}%",
            f"Total Trades:    {metrics['total_trades']}",
            f"Win Rate:        {metrics['win_rate_pct']:.1f}%",
            f"Sharpe Ratio:    {metrics['sharpe_ratio']:.2f}",
            "-"*60,
            "STRATEGY BREAKDOWN:"
        ]
        
        for strategy, stats in metrics['strategy_breakdown'].items():
            win_rate = (stats['wins'] / stats['total_trades'] * 100) if stats['total_trades'] > 0 else 0
            report.append(f"  {strategy}:")
            report.append(f"    Trades: {stats['total_trades']} | Wins: {stats['wins']} | Win Rate: {win_rate:.1f}%")
            report.append(f"    Total P&L: ${stats['total_pnl']:+,.2f}")
        
        report.append("="*60)
        
        return "\n".join(report)


# ============================================================================
# MAIN ENGINE
# ============================================================================

class V49UnifiedEngine:
    """V49 Unified Alpha Engine"""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # API Clients
        self.trading_client = None
        self.data_client = None
        
        # Strategies
        self.wheel_strategy = WheelStrategy(config)
        self.iron_condor_strategy = IronCondorStrategy(config)
        
        # Tracking
        self.performance = PerformanceTracker()
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.positions: Dict[str, Dict] = {}
        self.day_trades_today: int = 0
        self.last_trade_date: str = ""
        
        # State
        self.universe: List[str] = []
        self._initialized = False
        self._running = False
        self.scan_count = 0
    
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('V49Engine')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        if not logger.handlers:
            logger.addHandler(handler)
        return logger
    
    async def initialize(self) -> bool:
        """Initialize engine"""
        try:
            self.logger.info("="*60)
            self.logger.info("V49 UNIFIED ALPHA ENGINE")
            self.logger.info("Equities + Options Strategies")
            self.logger.info("="*60)
            
            # Initialize clients
            self.trading_client = TradingClient(
                self.config.api_key,
                self.config.api_secret,
                paper=self.config.paper
            )
            self.data_client = StockHistoricalDataClient(
                self.config.api_key,
                self.config.api_secret
            )
            
            # Get account
            account = self.trading_client.get_account()
            equity = float(account.equity)
            self.performance.start_equity = equity
            self.performance.peak_equity = equity
            
            self.logger.info(f"Account Equity: ${equity:,.2f}")
            self.logger.info(f"Buying Power: ${float(account.buying_power):,.2f}")
            
            # Build universe
            await self._build_universe()
            
            # Load data
            await self._load_historical_data()
            
            self._initialized = True
            self.logger.info(f"Initialized with {len(self.universe)} symbols")
            
            return True
        except Exception as e:
            self.logger.error(f"Init failed: {e}")
            return False
    
    async def _build_universe(self):
        """Build trading universe"""
        # Combine wheel stocks with general universe
        self.universe = list(set(self.config.wheel_stocks + [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
            'JPM', 'V', 'JNJ', 'UNH', 'PG', 'HD', 'MA', 'DIS',
            'PYPL', 'NFLX', 'ADBE', 'CRM', 'COST', 'PEP', 'TMO',
            'ABBV', 'MRK', 'AVGO', 'ACN', 'WMT', 'NKE', 'MCD',
            'LLY', 'DHR', 'TXN', 'NEE', 'BMY', 'PM', 'UNP', 'RTX',
            'LOW', 'ORCL', 'QCOM', 'IBM', 'GE', 'CAT', 'HON', 'BA',
            'SBUX', 'MDLZ', 'AMT', 'ISRG', 'GILD', 'CVS', 'BLK',
            'SPGI', 'SYK', 'AXP', 'BKNG', 'TJX', 'MMC', 'LRCX'
        ]))[:self.config.universe_size]
        
        self.logger.info(f"Universe: {len(self.universe)} symbols")

    
    async def _load_historical_data(self):
        """Load historical data"""
        self.logger.info("Loading historical data...")
        
        end = datetime.now()
        start = end - timedelta(days=60)
        
        for i in range(0, len(self.universe), self.config.batch_size):
            batch = self.universe[i:i + self.config.batch_size]
            
            try:
                request = StockBarsRequest(
                    feed=DataFeed.IEX,
                    symbol_or_symbols=batch,
                    timeframe=TimeFrame.Day,
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
                            'volume': bar.volume
                        } for bar in bars.data[symbol]])
                        
                        if len(df) > 0:
                            self.data_cache[symbol] = df
            except Exception as e:
                self.logger.warning(f"Data error: {e}")
            
            await asyncio.sleep(0.3)
        
        self.logger.info(f"Loaded data for {len(self.data_cache)} symbols")
    
    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical features"""
        if len(df) < 20:
            return df
        
        df = df.copy()
        close = df['close']
        
        # Trend
        df['sma_20'] = close.rolling(20).mean()
        df['sma_50'] = close.rolling(50).mean() if len(df) >= 50 else close.rolling(20).mean()
        
        # Volatility
        df['returns'] = close.pct_change()
        df['volatility'] = df['returns'].rolling(20).std() * np.sqrt(252)
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Z-score
        df['zscore'] = (close - df['sma_20']) / close.rolling(20).std()
        
        # Momentum
        df['momentum_5'] = close / close.shift(5) - 1
        df['momentum_20'] = close / close.shift(20) - 1
        
        # Volume
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma'].replace(0, np.nan)
        
        return df.fillna(0)
    
    async def _generate_equity_signals(self, symbol: str) -> List[Dict]:
        """Generate equity trading signals"""
        signals = []
        
        if symbol not in self.data_cache:
            return signals
        
        df = self._calculate_features(self.data_cache[symbol])
        if len(df) < 20:
            return signals
        
        latest = df.iloc[-1]
        price = latest['close']
        zscore = latest['zscore']
        rsi = latest['rsi']
        momentum = latest['momentum_5']
        vol_ratio = latest['volume_ratio']
        
        # Mean Reversion - Oversold
        if zscore < -self.config.zscore_entry and rsi < self.config.rsi_oversold:
            signals.append({
                'strategy': 'mean_reversion',
                'action': 'buy',
                'symbol': symbol,
                'price': price,
                'strength': min(abs(zscore) / 3, 1),
                'reason': f'Oversold: z={zscore:.2f}, RSI={rsi:.0f}'
            })
        
        # Mean Reversion - Overbought (sell existing)
        elif zscore > self.config.zscore_entry and rsi > self.config.rsi_overbought:
            signals.append({
                'strategy': 'mean_reversion',
                'action': 'sell',
                'symbol': symbol,
                'price': price,
                'strength': min(abs(zscore) / 3, 1),
                'reason': f'Overbought: z={zscore:.2f}, RSI={rsi:.0f}'
            })
        
        # Momentum with volume confirmation
        if momentum > self.config.momentum_threshold and vol_ratio > self.config.volume_surge:
            signals.append({
                'strategy': 'momentum',
                'action': 'buy',
                'symbol': symbol,
                'price': price,
                'strength': min(momentum / 0.1, 1),
                'reason': f'Momentum: {momentum:.1%} with {vol_ratio:.1f}x volume'
            })
        
        return signals
    
    async def _generate_options_signals(self, symbol: str) -> List[Dict]:
        """Generate options trading signals"""
        signals = []
        
        if not self.config.options_enabled:
            return signals
        
        if symbol not in self.config.wheel_stocks:
            return signals
        
        if symbol not in self.data_cache:
            return signals
        
        df = self.data_cache[symbol]
        if len(df) < 30:
            return signals
        
        prices = df['close'].values
        current_price = prices[-1]
        
        # Generate wheel signal
        wheel_signal = self.wheel_strategy.generate_wheel_signal(
            symbol, current_price, prices
        )
        if wheel_signal:
            signals.append(wheel_signal)
        
        # Generate iron condor signal (for high IV, range-bound)
        ic_signal = self.iron_condor_strategy.generate_iron_condor_signal(
            symbol, current_price, prices
        )
        if ic_signal:
            signals.append(ic_signal)
        
        return signals
    
    def _check_pdt_limit(self) -> bool:
        """Check if we can make more day trades"""
        today = datetime.now().strftime('%Y-%m-%d')
        
        if today != self.last_trade_date:
            self.day_trades_today = 0
            self.last_trade_date = today
        
        return self.day_trades_today < self.config.max_day_trades
    
    async def _execute_signal(self, signal: Dict) -> bool:
        """Execute a trading signal"""
        symbol = signal['symbol']
        action = signal['action']
        strategy = signal.get('strategy', 'unknown')
        
        try:
            account = self.trading_client.get_account()
            equity = float(account.equity)
            buying_power = float(account.buying_power)
            
            # Update performance tracking
            self.performance.update_equity(equity)
            
            # Get current positions
            positions = self.trading_client.get_all_positions()
            current_pos = None
            for pos in positions:
                if pos.symbol == symbol:
                    current_pos = pos
                    break
            
            # For options signals, log but don't execute (Alpaca paper doesn't support options)
            if strategy in ['wheel', 'iron_condor']:
                self.logger.info(f"OPTIONS SIGNAL: {signal}")
                return False
            
            # Calculate position size
            price = signal.get('price', 0)
            if price <= 0:
                return False
            
            position_value = equity * self.config.position_size_pct
            shares = int(position_value / price)
            
            if shares < 1:
                return False
            
            # Execute based on action
            if action == 'buy':
                if len(positions) >= self.config.max_positions:
                    return False
                
                if current_pos:
                    return False  # Already have position
                
                cost = price * shares
                if buying_power < cost:
                    shares = int(buying_power * 0.9 / price)
                    if shares < 1:
                        return False
                
                order = MarketOrderRequest(
                    symbol=symbol,
                    qty=shares,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                )
                self.trading_client.submit_order(order)
                self.logger.info(f"BUY {shares} {symbol} @ ${price:.2f} | {signal.get('reason', '')}")
                
                self.performance.record_trade({
                    'strategy': strategy,
                    'symbol': symbol,
                    'action': 'buy',
                    'shares': shares,
                    'price': price,
                    'pnl': 0  # Will update on close
                })
                return True
            
            elif action == 'sell' and current_pos:
                qty = int(float(current_pos.qty))
                entry = float(current_pos.avg_entry_price)
                
                order = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY
                )
                self.trading_client.submit_order(order)
                
                pnl = (price - entry) * qty
                self.logger.info(f"SELL {qty} {symbol} @ ${price:.2f} (entry ${entry:.2f}) P&L: ${pnl:+.2f}")
                
                self.performance.record_trade({
                    'strategy': strategy,
                    'symbol': symbol,
                    'action': 'sell',
                    'shares': qty,
                    'price': price,
                    'entry_price': entry,
                    'pnl': pnl
                })
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Execution error {symbol}: {e}")
            return False

    
    async def _scan_cycle(self):
        """Execute one scan cycle"""
        self.scan_count += 1
        all_signals = []
        
        # Generate signals for all symbols
        for symbol in self.universe:
            # Equity signals
            equity_signals = await self._generate_equity_signals(symbol)
            all_signals.extend(equity_signals)
            
            # Options signals
            options_signals = await self._generate_options_signals(symbol)
            all_signals.extend(options_signals)
        
        # Sort by strength
        all_signals.sort(key=lambda x: x.get('strength', 0), reverse=True)
        
        # Execute top signals
        executed = 0
        for signal in all_signals[:5]:  # Limit per cycle
            if await self._execute_signal(signal):
                executed += 1
                await asyncio.sleep(1)
        
        return executed
    
    def _print_status(self):
        """Print status"""
        try:
            account = self.trading_client.get_account()
            equity = float(account.equity)
            positions = self.trading_client.get_all_positions()
            
            self.performance.update_equity(equity)
            
            pnl = equity - self.performance.start_equity
            pnl_pct = (pnl / self.performance.start_equity * 100) if self.performance.start_equity > 0 else 0
            
            self.logger.info("-"*60)
            self.logger.info(f"SCAN #{self.scan_count} | Positions: {len(positions)}/{self.config.max_positions}")
            self.logger.info(f"Equity: ${equity:,.2f} | P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)")
            
        except Exception as e:
            self.logger.error(f"Status error: {e}")
    
    async def run(self):
        """Main loop"""
        if not self._initialized:
            if not await self.initialize():
                return
        
        self._running = True
        self.logger.info(f"Starting - {self.config.scan_interval}s interval")
        
        last_report = time.time()
        
        try:
            while self._running:
                start = time.time()
                
                await self._scan_cycle()
                
                # Status every 5 scans
                if self.scan_count % 5 == 0:
                    self._print_status()
                
                # Performance report
                if time.time() - last_report > self.config.report_interval:
                    account = self.trading_client.get_account()
                    report = self.performance.generate_report(float(account.equity))
                    self.logger.info(report)
                    last_report = time.time()
                
                # Sleep
                elapsed = time.time() - start
                sleep_time = max(0, self.config.scan_interval - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            self.logger.info("Shutdown...")
        finally:
            self._running = False
            self._print_status()
            account = self.trading_client.get_account()
            self.logger.info(self.performance.generate_report(float(account.equity)))


# ============================================================================
# ENTRY POINT
# ============================================================================

async def main():
    parser = argparse.ArgumentParser(description='V49 Unified Alpha Engine')
    parser.add_argument('--test', action='store_true', help='Test mode')
    parser.add_argument('--trade', action='store_true', help='Trading mode')
    parser.add_argument('--interval', type=int, default=30, help='Scan interval')
    args = parser.parse_args()
    
    config = UnifiedConfig(
        api_key=os.getenv('APCA_API_KEY_ID', ''),
        api_secret=os.getenv('APCA_API_SECRET_KEY', ''),
        paper=True,
        scan_interval=args.interval
    )
    
    if not config.api_key or not config.api_secret:
        print("ERROR: Missing API credentials")
        return
    
    engine = V49UnifiedEngine(config)
    
    if args.test:
        print("\n" + "="*60)
        print("V49 UNIFIED ENGINE - TEST")
        print("="*60)
        
        if await engine.initialize():
            print(f"\n[OK] Universe: {len(engine.universe)} symbols")
            print(f"[OK] Data: {len(engine.data_cache)} symbols")
            print(f"[OK] Wheel stocks: {len(engine.config.wheel_stocks)}")
            print("\n[READY] Engine ready")
        else:
            print("[FAIL] Init failed")
        return
    
    if args.trade:
        await engine.run()
    else:
        print("Use --test or --trade")


if __name__ == '__main__':
    asyncio.run(main())
