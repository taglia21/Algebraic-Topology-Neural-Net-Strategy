"""
Black-Scholes Option Pricing Model
==================================

Production-grade implementation with Greeks calculation.
Extracted from V50 engine with improvements.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from enum import Enum
from typing import Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class OptionType(Enum):
    """Type of option contract."""
    CALL = "call"
    PUT = "put"


@dataclass
class Greeks:
    """Container for option Greeks."""
    delta: float
    gamma: float
    theta: float  # Per day
    vega: float  # Per 1% IV change
    rho: float  # Per 1% rate change
    
    def __mul__(self, scalar: float) -> 'Greeks':
        """Scale Greeks by number of contracts."""
        return Greeks(
            delta=self.delta * scalar,
            gamma=self.gamma * scalar,
            theta=self.theta * scalar,
            vega=self.vega * scalar,
            rho=self.rho * scalar
        )
    
    def __add__(self, other: 'Greeks') -> 'Greeks':
        """Add two Greeks objects (for portfolio aggregation)."""
        return Greeks(
            delta=self.delta + other.delta,
            gamma=self.gamma + other.gamma,
            theta=self.theta + other.theta,
            vega=self.vega + other.vega,
            rho=self.rho + other.rho
        )


class BlackScholes:
    """
    Black-Scholes-Merton option pricing model.
    
    Assumptions:
    - European-style options
    - No dividends (can be extended)
    - Constant volatility
    - Lognormal distribution of returns
    """
    
    def __init__(self, risk_free_rate: float = 0.05):
        """
        Initialize Black-Scholes calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate (default 5%)
        """
        self.risk_free_rate = risk_free_rate
    
    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate d1 parameter.
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate (annualized)
            sigma: Implied volatility (annualized)
            
        Returns:
            d1 value
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        
        return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d2 parameter."""
        if T <= 0 or sigma <= 0:
            return 0.0
        
        return BlackScholes.d1(S, K, T, r, sigma) - sigma * np.sqrt(T)
    
    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate call option price.
        
        Returns:
            Call option value
        """
        # Handle expiration edge case
        if T <= 0:
            return max(0.0, S - K)
        
        # Validate inputs
        if S <= 0 or K <= 0 or sigma <= 0:
            logger.warning(f"Invalid inputs: S={S}, K={K}, sigma={sigma}")
            return 0.0
        
        try:
            d1 = BlackScholes.d1(S, K, T, r, sigma)
            d2 = BlackScholes.d2(S, K, T, r, sigma)
            
            call_value = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            return max(0.0, call_value)  # Ensure non-negative
            
        except Exception as e:
            logger.error(f"Error calculating call price: {e}")
            return 0.0
    
    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate put option price."""
        # Handle expiration edge case
        if T <= 0:
            return max(0.0, K - S)
        
        # Validate inputs
        if S <= 0 or K <= 0 or sigma <= 0:
            logger.warning(f"Invalid inputs: S={S}, K={K}, sigma={sigma}")
            return 0.0
        
        try:
            d1 = BlackScholes.d1(S, K, T, r, sigma)
            d2 = BlackScholes.d2(S, K, T, r, sigma)
            
            put_value = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            return max(0.0, put_value)  # Ensure non-negative
            
        except Exception as e:
            logger.error(f"Error calculating put price: {e}")
            return 0.0
    
    @staticmethod
    def delta(S: float, K: float, T: float, r: float, sigma: float, 
              option_type: OptionType) -> float:
        """
        Calculate option delta (sensitivity to underlying price).
        
        Delta interpretation:
        - Call: 0 to 1 (0.5 = ATM)
        - Put: -1 to 0 (-0.5 = ATM)
        """
        if T <= 0:
            if option_type == OptionType.CALL:
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0
        
        try:
            d1 = BlackScholes.d1(S, K, T, r, sigma)
            
            if option_type == OptionType.CALL:
                return norm.cdf(d1)
            else:
                return norm.cdf(d1) - 1.0
                
        except Exception as e:
            logger.error(f"Error calculating delta: {e}")
            return 0.0
    
    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate gamma (rate of change of delta).
        
        Same for calls and puts.
        High gamma = delta changes rapidly.
        """
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        
        try:
            d1 = BlackScholes.d1(S, K, T, r, sigma)
            gamma_val = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            return max(0.0, gamma_val)
            
        except Exception as e:
            logger.error(f"Error calculating gamma: {e}")
            return 0.0
    
    @staticmethod
    def theta(S: float, K: float, T: float, r: float, sigma: float,
              option_type: OptionType) -> float:
        """
        Calculate theta (time decay) per day.
        
        Negative for long options (lose value over time).
        Positive for short options (gain value over time).
        
        Returns daily theta (divide Black-Scholes by 365).
        """
        if T <= 0:
            return 0.0
        
        try:
            d1 = BlackScholes.d1(S, K, T, r, sigma)
            d2 = BlackScholes.d2(S, K, T, r, sigma)
            
            # First term (same for both)
            term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
            
            if option_type == OptionType.CALL:
                term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
            else:
                term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
            
            # Return daily theta
            return (term1 + term2) / 365.0
            
        except Exception as e:
            logger.error(f"Error calculating theta: {e}")
            return 0.0
    
    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate vega (sensitivity to volatility) per 1% IV change.
        
        Same for calls and puts.
        Measures profit/loss for 1% change in IV.
        """
        if T <= 0:
            return 0.0
        
        try:
            d1 = BlackScholes.d1(S, K, T, r, sigma)
            vega_val = S * norm.pdf(d1) * np.sqrt(T) / 100.0  # Per 1% IV
            return max(0.0, vega_val)
            
        except Exception as e:
            logger.error(f"Error calculating vega: {e}")
            return 0.0
    
    @staticmethod
    def rho(S: float, K: float, T: float, r: float, sigma: float,
            option_type: OptionType) -> float:
        """
        Calculate rho (sensitivity to interest rate) per 1% rate change.
        
        Less important for short-dated options.
        """
        if T <= 0:
            return 0.0
        
        try:
            d2 = BlackScholes.d2(S, K, T, r, sigma)
            
            if option_type == OptionType.CALL:
                rho_val = K * T * np.exp(-r * T) * norm.cdf(d2) / 100.0
            else:
                rho_val = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100.0
            
            return rho_val
            
        except Exception as e:
            logger.error(f"Error calculating rho: {e}")
            return 0.0
    
    @staticmethod
    def calculate_all_greeks(S: float, K: float, T: float, r: float, sigma: float,
                            option_type: OptionType) -> Greeks:
        """
        Calculate all Greeks at once (more efficient).
        
        Returns:
            Greeks object with all values
        """
        return Greeks(
            delta=BlackScholes.delta(S, K, T, r, sigma, option_type),
            gamma=BlackScholes.gamma(S, K, T, r, sigma),
            theta=BlackScholes.theta(S, K, T, r, sigma, option_type),
            vega=BlackScholes.vega(S, K, T, r, sigma),
            rho=BlackScholes.rho(S, K, T, r, sigma, option_type)
        )
    
    @staticmethod
    def implied_volatility(price: float, S: float, K: float, T: float, r: float,
                          option_type: OptionType, max_iter: int = 100,
                          tolerance: float = 1e-5) -> Optional[float]:
        """
        Calculate implied volatility using Brent's method with Newton-Raphson fallback.
        
        Args:
            price: Market price of option
            S, K, T, r: Black-Scholes parameters
            option_type: CALL or PUT
            max_iter: Maximum iterations
            tolerance: Convergence tolerance
            
        Returns:
            Implied volatility (annualized), or None if calculation fails
        """
        if T <= 0:
            return None
        
        # Intrinsic value
        if option_type == OptionType.CALL:
            intrinsic = max(0, S - K)
        else:
            intrinsic = max(0, K - S)
        
        # Check if price is above intrinsic (must be)
        if price < intrinsic - 1e-6:
            logger.warning(f"Price {price} below intrinsic {intrinsic}")
            return None
        
        # Define objective function
        def objective(sigma):
            if option_type == OptionType.CALL:
                theo_price = BlackScholes.call_price(S, K, T, r, sigma)
            else:
                theo_price = BlackScholes.put_price(S, K, T, r, sigma)
            return theo_price - price
        
        # Try Brent's method first (robust)
        try:
            iv = brentq(objective, 0.001, 5.0, maxiter=max_iter, xtol=tolerance)
            return float(iv)
        except (ValueError, RuntimeError) as e:
            logger.debug(f"Brent's method failed: {e}, trying Newton-Raphson")
        
        # Fallback to Newton-Raphson
        sigma = 0.3  # Initial guess
        for i in range(max_iter):
            if option_type == OptionType.CALL:
                theo_price = BlackScholes.call_price(S, K, T, r, sigma)
            else:
                theo_price = BlackScholes.put_price(S, K, T, r, sigma)
            
            vega_val = BlackScholes.vega(S, K, T, r, sigma) * 100  # Convert to decimal
            
            if abs(vega_val) < 1e-10:
                logger.warning("Vega too small, IV calculation failed")
                return None
            
            # Newton-Raphson update
            sigma = sigma - (theo_price - price) / vega_val
            
            # Keep sigma in reasonable bounds
            sigma = max(0.001, min(sigma, 5.0))
            
            # Check convergence
            if abs(theo_price - price) < tolerance:
                return float(sigma)
        
        logger.warning(f"IV calculation did not converge after {max_iter} iterations")
        return None
    
    # Instance methods that use stored risk_free_rate
    
    def put_price(self, S: float, K: float, T: float, sigma: float) -> float:
        """Calculate put price using instance risk_free_rate."""
        return BlackScholes.put_price(S, K, T, self.risk_free_rate, sigma)
    
    def call_price_inst(self, S: float, K: float, T: float, sigma: float) -> float:
        """Calculate call price using instance risk_free_rate."""
        return BlackScholes.call_price(S, K, T, self.risk_free_rate, sigma)
    
    def calculate_all_greeks(self, S: float, K: float, T: float, sigma: float,
                            option_type: OptionType) -> Greeks:
        """Calculate all Greeks using instance risk_free_rate."""
        return BlackScholes.calculate_all_greeks(S, K, T, self.risk_free_rate, sigma, option_type)
