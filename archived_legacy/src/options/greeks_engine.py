"""
Greeks Engine
=============

Real-time options Greeks calculation and portfolio risk management.

Features:
- Black-Scholes Greeks: Delta, Gamma, Theta, Vega, Rho
- Portfolio-level Greek aggregation
- Dynamic hedging recommendations
- P&L attribution by Greek
- <100ms calculation latency

Uses analytical formulas for speed.
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy.stats import norm

logger = logging.getLogger(__name__)


@dataclass
class OptionGreeks:
    """Greeks for a single option position."""
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    underlying_price: float
    strike: float
    time_to_expiry: float
    iv: float
    option_type: str  # 'call' or 'put'


@dataclass
class PortfolioGreeks:
    """Aggregated Greeks for entire portfolio."""
    net_delta: float
    net_gamma: float
    net_theta: float
    net_vega: float
    net_rho: float
    num_positions: int
    total_notional: float


@dataclass
class HedgeRecommendation:
    """Hedging recommendation to neutralize Greeks."""
    action: str  # 'buy_stock', 'sell_stock', 'buy_option', 'sell_option'
    quantity: int
    symbol: str
    reason: str
    target_greek: str  # 'delta', 'gamma', 'vega'


class GreeksEngine:
    """
    Calculate and monitor option Greeks with real-time updates.
    
    Features:
    - Analytical Black-Scholes Greeks
    - Portfolio aggregation
    - Dynamic hedging recommendations
    - P&L decomposition by Greek
    """
    
    def __init__(self, risk_free_rate: float = 0.05):
        """
        Initialize Greeks engine.
        
        Args:
            risk_free_rate: Annual risk-free rate (default 5%)
        """
        self.risk_free_rate = risk_free_rate
        self.logger = logging.getLogger(__name__)
        
        # Hedging thresholds
        self.delta_threshold = 0.10  # Hedge if |net_delta| > 10
        self.gamma_threshold = 50.0  # Hedge if |net_gamma| > 50
        self.vega_threshold = 1000.0  # Hedge if |net_vega| > 1000
        
        self.logger.info(f"Initialized Greeks Engine (r={risk_free_rate:.2%})")
    
    def calculate_greeks(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str
    ) -> OptionGreeks:
        """
        Calculate option Greeks using Black-Scholes formulas.
        
        Args:
            S: Underlying price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Implied volatility
            option_type: 'call' or 'put'
            
        Returns:
            OptionGreeks with all calculated values
        """
        # Prevent division by zero
        if T <= 0:
            return self._zero_greeks(S, K, sigma, option_type)
        
        # Black-Scholes d1, d2
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Standard normal CDF and PDF
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        n_d1 = norm.pdf(d1)  # PDF for gamma, vega
        
        # Calculate Greeks
        if option_type.lower() == 'call':
            delta = N_d1
            theta = (-S * n_d1 * sigma / (2 * np.sqrt(T)) 
                     - r * K * np.exp(-r * T) * N_d2) / 365  # Daily theta
            rho = K * T * np.exp(-r * T) * N_d2 / 100  # 1% rate change
        else:  # put
            delta = N_d1 - 1
            theta = (-S * n_d1 * sigma / (2 * np.sqrt(T)) 
                     + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365  # Daily theta
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100  # 1% rate change
        
        # Gamma and Vega are same for calls and puts
        gamma = n_d1 / (S * sigma * np.sqrt(T))
        vega = S * n_d1 * np.sqrt(T) / 100  # 1% vol change
        
        return OptionGreeks(
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho,
            underlying_price=S,
            strike=K,
            time_to_expiry=T,
            iv=sigma,
            option_type=option_type
        )
    
    def _zero_greeks(self, S: float, K: float, sigma: float, option_type: str) -> OptionGreeks:
        """Return zero Greeks for expired options."""
        return OptionGreeks(
            delta=0.0,
            gamma=0.0,
            theta=0.0,
            vega=0.0,
            rho=0.0,
            underlying_price=S,
            strike=K,
            time_to_expiry=0.0,
            iv=sigma,
            option_type=option_type
        )
    
    def portfolio_greeks(self, positions: List[Dict]) -> PortfolioGreeks:
        """
        Calculate aggregated Greeks for portfolio.
        
        Args:
            positions: List of position dicts with:
                - symbol: Option symbol
                - quantity: Number of contracts (+ for long, - for short)
                - underlying_price: Current stock price
                - strike: Strike price
                - expiry: Expiration date (datetime)
                - iv: Implied volatility
                - option_type: 'call' or 'put'
                
        Returns:
            PortfolioGreeks with net exposures
        """
        net_delta = 0.0
        net_gamma = 0.0
        net_theta = 0.0
        net_vega = 0.0
        net_rho = 0.0
        total_notional = 0.0
        
        for pos in positions:
            # Calculate time to expiry
            if isinstance(pos['expiry'], datetime):
                T = (pos['expiry'] - datetime.now()).days / 365.0
            else:
                T = pos['expiry']  # Already in years
            
            T = max(T, 0)  # Can't be negative
            
            # Calculate Greeks for this position
            greeks = self.calculate_greeks(
                S=pos['underlying_price'],
                K=pos['strike'],
                T=T,
                r=self.risk_free_rate,
                sigma=pos['iv'],
                option_type=pos['option_type']
            )
            
            # Aggregate (weighted by quantity)
            qty = pos['quantity']
            net_delta += greeks.delta * qty * 100  # 100 shares per contract
            net_gamma += greeks.gamma * qty * 100
            net_theta += greeks.theta * qty
            net_vega += greeks.vega * qty
            net_rho += greeks.rho * qty
            
            # Notional = stock price * contracts * 100
            total_notional += abs(qty) * pos['underlying_price'] * 100
        
        return PortfolioGreeks(
            net_delta=net_delta,
            net_gamma=net_gamma,
            net_theta=net_theta,
            net_vega=net_vega,
            net_rho=net_rho,
            num_positions=len(positions),
            total_notional=total_notional
        )
    
    def hedge_recommendation(self, portfolio_greeks: PortfolioGreeks, underlying_price: float) -> List[HedgeRecommendation]:
        """
        Generate hedging recommendations based on portfolio Greeks.
        
        Args:
            portfolio_greeks: Current portfolio Greeks
            underlying_price: Current price of underlying
            
        Returns:
            List of hedge recommendations
        """
        recommendations = []
        
        # Delta hedge
        if abs(portfolio_greeks.net_delta) > self.delta_threshold:
            # Hedge with stock (delta = 1 per share)
            shares_needed = -portfolio_greeks.net_delta
            
            if shares_needed > 0:
                action = "buy_stock"
                reason = f"Portfolio is {-shares_needed:.0f} shares short-delta"
            else:
                action = "sell_stock"
                reason = f"Portfolio is {shares_needed:.0f} shares long-delta"
            
            recommendations.append(HedgeRecommendation(
                action=action,
                quantity=int(abs(shares_needed)),
                symbol="UNDERLYING",
                reason=reason,
                target_greek="delta"
            ))
        
        # Gamma hedge
        if abs(portfolio_greeks.net_gamma) > self.gamma_threshold:
            # Hedge gamma with ATM options (highest gamma)
            # This is simplified - in practice, optimize gamma/vega trade-off
            reason = f"Net gamma {portfolio_greeks.net_gamma:.1f} exceeds threshold {self.gamma_threshold}"
            
            if portfolio_greeks.net_gamma > 0:
                action = "sell_option"
            else:
                action = "buy_option"
            
            recommendations.append(HedgeRecommendation(
                action=action,
                quantity=1,  # placeholder
                symbol="ATM_OPTION",
                reason=reason,
                target_greek="gamma"
            ))
        
        # Vega hedge
        if abs(portfolio_greeks.net_vega) > self.vega_threshold:
            reason = f"Net vega {portfolio_greeks.net_vega:.1f} exceeds threshold {self.vega_threshold}"
            
            if portfolio_greeks.net_vega > 0:
                action = "sell_option"  # Long vega -> sell options
            else:
                action = "buy_option"  # Short vega -> buy options
            
            recommendations.append(HedgeRecommendation(
                action=action,
                quantity=1,  # placeholder
                symbol="OTM_OPTION",
                reason=reason,
                target_greek="vega"
            ))
        
        return recommendations
    
    def greeks_pnl_attribution(
        self,
        positions: List[Dict],
        price_changes: Dict[str, float],
        iv_changes: Dict[str, float],
        time_elapsed_days: float = 1.0
    ) -> Dict[str, float]:
        """
        Decompose P&L into Greek contributions.
        
        P&L ≈ Delta * ΔS + 0.5 * Gamma * (ΔS)² + Theta * Δt + Vega * Δσ
        
        Args:
            positions: List of positions
            price_changes: Dict mapping symbols to price changes
            iv_changes: Dict mapping symbols to IV changes
            time_elapsed_days: Days elapsed
            
        Returns:
            Dict with P&L attribution by Greek
        """
        delta_pnl = 0.0
        gamma_pnl = 0.0
        theta_pnl = 0.0
        vega_pnl = 0.0
        
        for pos in positions:
            # Get changes
            symbol = pos['symbol']
            dS = price_changes.get(symbol, 0.0)
            dIV = iv_changes.get(symbol, 0.0)
            
            # Calculate time to expiry
            if isinstance(pos['expiry'], datetime):
                T = (pos['expiry'] - datetime.now()).days / 365.0
            else:
                T = pos['expiry']
            
            T = max(T, 0)
            
            # Calculate Greeks
            greeks = self.calculate_greeks(
                S=pos['underlying_price'],
                K=pos['strike'],
                T=T,
                r=self.risk_free_rate,
                sigma=pos['iv'],
                option_type=pos['option_type']
            )
            
            qty = pos['quantity']
            
            # Delta contribution: Delta * ΔS * qty * 100
            delta_pnl += greeks.delta * dS * qty * 100
            
            # Gamma contribution: 0.5 * Gamma * (ΔS)² * qty * 100
            gamma_pnl += 0.5 * greeks.gamma * (dS ** 2) * qty * 100
            
            # Theta contribution: Theta * Δt * qty
            theta_pnl += greeks.theta * time_elapsed_days * qty
            
            # Vega contribution: Vega * Δσ * qty (vega is per 1% vol)
            vega_pnl += greeks.vega * dIV * qty
        
        total_pnl = delta_pnl + gamma_pnl + theta_pnl + vega_pnl
        
        return {
            'delta_pnl': delta_pnl,
            'gamma_pnl': gamma_pnl,
            'theta_pnl': theta_pnl,
            'vega_pnl': vega_pnl,
            'total_pnl': total_pnl,
            'attribution_check': total_pnl
        }
    
    def calculate_portfolio_stats(self, portfolio_greeks: PortfolioGreeks) -> Dict[str, float]:
        """Calculate portfolio risk statistics from Greeks."""
        return {
            'delta_risk_pct': (portfolio_greeks.net_delta / 
                              portfolio_greeks.total_notional * 100) if portfolio_greeks.total_notional > 0 else 0,
            'gamma_exposure': portfolio_greeks.net_gamma,
            'theta_daily': portfolio_greeks.net_theta,
            'vega_sensitivity': portfolio_greeks.net_vega,
            'positions_count': portfolio_greeks.num_positions
        }
