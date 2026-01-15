"""
Options Pricing Module for Phase 13
====================================

Black-Scholes model and Greeks calculation for options overlay.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)


class OptionType(Enum):
    """Option type."""
    CALL = "call"
    PUT = "put"


@dataclass
class OptionContract:
    """Represents an option contract."""
    underlying: str
    option_type: OptionType
    strike: float
    expiry_days: int
    underlying_price: float
    implied_vol: float = 0.25
    risk_free_rate: float = 0.05
    dividend_yield: float = 0.0


@dataclass
class OptionsGreeks:
    """Option Greeks."""
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


@dataclass
class OptionPrice:
    """Option pricing result."""
    price: float
    intrinsic: float
    time_value: float
    greeks: OptionsGreeks
    moneyness: str  # 'ITM', 'ATM', 'OTM'


class BlackScholes:
    """
    Black-Scholes option pricing model.
    
    Calculates theoretical option prices and Greeks.
    """
    
    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
        """Calculate d1 parameter."""
        if T <= 0 or sigma <= 0:
            return 0
        return (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
        """Calculate d2 parameter."""
        if T <= 0 or sigma <= 0:
            return 0
        return BlackScholes.d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)
    
    @classmethod
    def call_price(
        cls,
        S: float,  # Underlying price
        K: float,  # Strike price
        T: float,  # Time to expiry (years)
        r: float,  # Risk-free rate
        sigma: float,  # Implied volatility
        q: float = 0,  # Dividend yield
    ) -> float:
        """Calculate call option price."""
        if T <= 0:
            return max(0, S - K)
        
        d1 = cls.d1(S, K, T, r, sigma, q)
        d2 = cls.d2(S, K, T, r, sigma, q)
        
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return max(0, price)
    
    @classmethod
    def put_price(
        cls,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0,
    ) -> float:
        """Calculate put option price."""
        if T <= 0:
            return max(0, K - S)
        
        d1 = cls.d1(S, K, T, r, sigma, q)
        d2 = cls.d2(S, K, T, r, sigma, q)
        
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        return max(0, price)
    
    @classmethod
    def price(
        cls,
        option_type: OptionType,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0,
    ) -> float:
        """Calculate option price based on type."""
        if option_type == OptionType.CALL:
            return cls.call_price(S, K, T, r, sigma, q)
        else:
            return cls.put_price(S, K, T, r, sigma, q)
    
    @classmethod
    def delta(
        cls,
        option_type: OptionType,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0,
    ) -> float:
        """Calculate delta (rate of change vs underlying)."""
        if T <= 0:
            if option_type == OptionType.CALL:
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0
        
        d1 = cls.d1(S, K, T, r, sigma, q)
        
        if option_type == OptionType.CALL:
            return np.exp(-q * T) * norm.cdf(d1)
        else:
            return np.exp(-q * T) * (norm.cdf(d1) - 1)
    
    @classmethod
    def gamma(
        cls,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0,
    ) -> float:
        """Calculate gamma (rate of change of delta)."""
        if T <= 0 or sigma <= 0:
            return 0
        
        d1 = cls.d1(S, K, T, r, sigma, q)
        return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    @classmethod
    def theta(
        cls,
        option_type: OptionType,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0,
    ) -> float:
        """Calculate theta (time decay) per day."""
        if T <= 0 or sigma <= 0:
            return 0
        
        d1 = cls.d1(S, K, T, r, sigma, q)
        d2 = cls.d2(S, K, T, r, sigma, q)
        
        term1 = -np.exp(-q * T) * S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
        
        if option_type == OptionType.CALL:
            term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
            term3 = q * S * np.exp(-q * T) * norm.cdf(d1)
        else:
            term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
            term3 = -q * S * np.exp(-q * T) * norm.cdf(-d1)
        
        theta_annual = term1 + term2 + term3
        return theta_annual / 365  # Daily theta
    
    @classmethod
    def vega(
        cls,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0,
    ) -> float:
        """Calculate vega (sensitivity to volatility) per 1% vol change."""
        if T <= 0:
            return 0
        
        d1 = cls.d1(S, K, T, r, sigma, q)
        return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100
    
    @classmethod
    def rho(
        cls,
        option_type: OptionType,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0,
    ) -> float:
        """Calculate rho (sensitivity to interest rate) per 1% rate change."""
        if T <= 0:
            return 0
        
        d2 = cls.d2(S, K, T, r, sigma, q)
        
        if option_type == OptionType.CALL:
            return K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    
    @classmethod
    def calculate_greeks(
        cls,
        option_type: OptionType,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0,
    ) -> OptionsGreeks:
        """Calculate all Greeks for an option."""
        return OptionsGreeks(
            delta=cls.delta(option_type, S, K, T, r, sigma, q),
            gamma=cls.gamma(S, K, T, r, sigma, q),
            theta=cls.theta(option_type, S, K, T, r, sigma, q),
            vega=cls.vega(S, K, T, r, sigma, q),
            rho=cls.rho(option_type, S, K, T, r, sigma, q),
        )
    
    @classmethod
    def price_option(cls, contract: OptionContract) -> OptionPrice:
        """
        Price an option contract with full details.
        """
        T = contract.expiry_days / 365
        
        # Calculate price
        price = cls.price(
            contract.option_type,
            contract.underlying_price,
            contract.strike,
            T,
            contract.risk_free_rate,
            contract.implied_vol,
            contract.dividend_yield,
        )
        
        # Calculate intrinsic value
        if contract.option_type == OptionType.CALL:
            intrinsic = max(0, contract.underlying_price - contract.strike)
        else:
            intrinsic = max(0, contract.strike - contract.underlying_price)
        
        time_value = price - intrinsic
        
        # Calculate Greeks
        greeks = cls.calculate_greeks(
            contract.option_type,
            contract.underlying_price,
            contract.strike,
            T,
            contract.risk_free_rate,
            contract.implied_vol,
            contract.dividend_yield,
        )
        
        # Determine moneyness
        if contract.option_type == OptionType.CALL:
            if contract.underlying_price > contract.strike * 1.02:
                moneyness = 'ITM'
            elif contract.underlying_price < contract.strike * 0.98:
                moneyness = 'OTM'
            else:
                moneyness = 'ATM'
        else:
            if contract.underlying_price < contract.strike * 0.98:
                moneyness = 'ITM'
            elif contract.underlying_price > contract.strike * 1.02:
                moneyness = 'OTM'
            else:
                moneyness = 'ATM'
        
        return OptionPrice(
            price=price,
            intrinsic=intrinsic,
            time_value=time_value,
            greeks=greeks,
            moneyness=moneyness,
        )


def estimate_implied_vol(underlying_ticker: str, vix_level: float) -> float:
    """
    Estimate implied volatility for an underlying based on VIX.
    
    Simple approximation - in production would use options chain data.
    """
    # Base IV multipliers by ticker
    iv_multipliers = {
        'SPY': 1.0,
        'QQQ': 1.15,  # Tech more volatile
        'IWM': 1.20,
        'TQQQ': 3.0,  # 3x leverage
        'SPXL': 3.0,
        'SOXL': 3.5,
        'SQQQ': 3.0,
        'SPXU': 3.0,
        'SOXS': 3.5,
    }
    
    multiplier = iv_multipliers.get(underlying_ticker, 1.0)
    
    # Convert VIX to decimal IV and apply multiplier
    base_iv = vix_level / 100
    return base_iv * multiplier


def get_atm_strike(price: float, strike_interval: float = 5.0) -> float:
    """Get nearest ATM strike price."""
    return round(price / strike_interval) * strike_interval


def get_otm_call_strike(price: float, pct_otm: float = 0.05, strike_interval: float = 5.0) -> float:
    """Get OTM call strike (above current price)."""
    target = price * (1 + pct_otm)
    return np.ceil(target / strike_interval) * strike_interval


def get_otm_put_strike(price: float, pct_otm: float = 0.05, strike_interval: float = 5.0) -> float:
    """Get OTM put strike (below current price)."""
    target = price * (1 - pct_otm)
    return np.floor(target / strike_interval) * strike_interval
