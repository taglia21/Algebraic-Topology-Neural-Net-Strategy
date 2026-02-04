"""
Greeks Portfolio Manager
=========================

Monitor and manage portfolio-level Greeks exposure with risk limits.

Portfolio Greeks Limits (per $100K capital):
- Delta: -20 to +20 (directional risk)
- Gamma: 0 to 5 (convexity risk)
- Theta: +30 to +70 (time decay income target)
- Vega: -150 to +50 (volatility risk)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum

from .utils.black_scholes import Greeks, OptionType
from .utils.constants import (
    MAX_PORTFOLIO_DELTA_PER_100K,
    MAX_PORTFOLIO_GAMMA_PER_100K,
    TARGET_PORTFOLIO_THETA_PER_100K,
    MAX_PORTFOLIO_VEGA_PER_100K,
    MIN_PORTFOLIO_THETA_PER_100K,
)

logger = logging.getLogger(__name__)


class GreeksViolationType(Enum):
    """Type of Greeks limit violation."""
    DELTA_HIGH = "delta_too_high"
    DELTA_LOW = "delta_too_low"
    GAMMA_HIGH = "gamma_too_high"
    THETA_LOW = "theta_too_low"
    VEGA_HIGH = "vega_too_high"
    MULTIPLE = "multiple_violations"


@dataclass
class PositionGreeks:
    """Greeks for a single option position."""
    symbol: str
    strike: float
    expiration: datetime
    option_type: OptionType
    quantity: int  # Positive for long, negative for short
    greeks: Greeks
    underlying_price: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def scaled_greeks(self) -> Greeks:
        """Get Greeks scaled by position quantity."""
        return self.greeks * self.quantity


@dataclass
class PortfolioGreeks:
    """Aggregated Greeks for entire portfolio."""
    total_delta: float
    total_gamma: float
    total_theta: float
    total_vega: float
    total_rho: float
    
    delta_per_100k: float
    gamma_per_100k: float
    theta_per_100k: float
    vega_per_100k: float
    
    account_value: float
    num_positions: int
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __str__(self) -> str:
        return (
            f"Portfolio Greeks (${self.account_value:,.0f}):\n"
            f"  Delta: {self.total_delta:+.2f} ({self.delta_per_100k:+.1f}/100K)\n"
            f"  Gamma: {self.total_gamma:+.3f} ({self.gamma_per_100k:+.2f}/100K)\n"
            f"  Theta: {self.total_theta:+.2f} ({self.theta_per_100k:+.1f}/100K)\n"
            f"  Vega:  {self.vega:+.2f} ({self.vega_per_100k:+.1f}/100K)\n"
            f"  Positions: {self.num_positions}"
        )


@dataclass
class GreeksViolation:
    """Greeks limit breach alert."""
    violation_type: GreeksViolationType
    metric_name: str
    current_value: float
    limit_value: float
    severity: str  # "warning", "critical"
    message: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class HedgeRecommendation:
    """Suggested hedge to bring Greeks within limits."""
    reason: str
    action: str  # "buy_calls", "sell_puts", "buy_stock", etc.
    symbol: str
    quantity: int
    strike: Optional[float] = None
    expiration: Optional[datetime] = None
    expected_delta_change: float = 0.0
    expected_gamma_change: float = 0.0
    expected_vega_change: float = 0.0


class GreeksManager:
    """
    Portfolio Greeks Manager.
    
    Tracks Greeks across all positions and enforces risk limits.
    Provides hedging recommendations when limits are breached.
    
    Usage:
        manager = GreeksManager(account_value=100000)
        manager.add_position(symbol='SPY', strike=450, ...)
        portfolio = manager.get_portfolio_greeks()
        violations = manager.check_limits()
    """
    
    def __init__(
        self,
        account_value: float,
        delta_limit: Optional[float] = None,
        gamma_limit: Optional[float] = None,
        theta_min: Optional[float] = None,
        vega_limit: Optional[float] = None
    ):
        """
        Initialize Greeks manager.
        
        Args:
            account_value: Total account value for scaling limits
            delta_limit: Max absolute delta per $100K (default from constants)
            gamma_limit: Max gamma per $100K (default from constants)
            theta_min: Min theta per $100K (default from constants)
            vega_limit: Max absolute vega per $100K (default from constants)
        """
        self.account_value = account_value
        
        # Risk limits (scaled to account size)
        self.scaling_factor = account_value / 100_000
        self.delta_limit = (delta_limit or MAX_PORTFOLIO_DELTA_PER_100K) * self.scaling_factor
        self.gamma_limit = (gamma_limit or MAX_PORTFOLIO_GAMMA_PER_100K) * self.scaling_factor
        self.theta_min = (theta_min or MIN_PORTFOLIO_THETA_PER_100K) * self.scaling_factor
        self.vega_limit = (vega_limit or abs(MAX_PORTFOLIO_VEGA_PER_100K)) * self.scaling_factor
        
        # Active positions
        self.positions: List[PositionGreeks] = []
        
        logger.info(
            f"Greeks Manager initialized (account=${account_value:,.0f}): "
            f"Delta limit=±{self.delta_limit:.1f}, Gamma limit={self.gamma_limit:.2f}, "
            f"Theta min={self.theta_min:.1f}, Vega limit=±{self.vega_limit:.1f}"
        )
    
    def add_position(
        self,
        symbol: str,
        strike: float,
        expiration: datetime,
        option_type: OptionType,
        quantity: int,
        greeks: Greeks,
        underlying_price: float
    ) -> None:
        """
        Add or update a position in the portfolio.
        
        Args:
            symbol: Underlying symbol
            strike: Option strike price
            expiration: Option expiration date
            option_type: CALL or PUT
            quantity: Number of contracts (+ for long, - for short)
            greeks: Option Greeks
            underlying_price: Current underlying price
        """
        # Check if position already exists (same symbol, strike, expiration, type)
        for i, pos in enumerate(self.positions):
            if (pos.symbol == symbol and pos.strike == strike and
                pos.expiration == expiration and pos.option_type == option_type):
                # Update existing position
                self.positions[i] = PositionGreeks(
                    symbol=symbol,
                    strike=strike,
                    expiration=expiration,
                    option_type=option_type,
                    quantity=quantity,
                    greeks=greeks,
                    underlying_price=underlying_price
                )
                logger.debug(f"Updated position: {symbol} {strike} {option_type.value}")
                return
        
        # Add new position
        position = PositionGreeks(
            symbol=symbol,
            strike=strike,
            expiration=expiration,
            option_type=option_type,
            quantity=quantity,
            greeks=greeks,
            underlying_price=underlying_price
        )
        self.positions.append(position)
        logger.debug(f"Added position: {symbol} {strike} {option_type.value} x{quantity}")
    
    def remove_position(
        self,
        symbol: str,
        strike: float,
        expiration: datetime,
        option_type: OptionType
    ) -> bool:
        """
        Remove a position from the portfolio.
        
        Returns:
            True if position was found and removed
        """
        for i, pos in enumerate(self.positions):
            if (pos.symbol == symbol and pos.strike == strike and
                pos.expiration == expiration and pos.option_type == option_type):
                removed = self.positions.pop(i)
                logger.info(f"Removed position: {symbol} {strike} {option_type.value}")
                return True
        
        logger.warning(f"Position not found: {symbol} {strike} {option_type.value}")
        return False
    
    def get_portfolio_greeks(self) -> PortfolioGreeks:
        """
        Calculate aggregated portfolio Greeks.
        
        Returns:
            PortfolioGreeks with totals and per-100K metrics
        """
        if not self.positions:
            return PortfolioGreeks(
                total_delta=0.0,
                total_gamma=0.0,
                total_theta=0.0,
                total_vega=0.0,
                total_rho=0.0,
                delta_per_100k=0.0,
                gamma_per_100k=0.0,
                theta_per_100k=0.0,
                vega_per_100k=0.0,
                account_value=self.account_value,
                num_positions=0
            )
        
        # Aggregate scaled Greeks
        total_greeks = sum(
            (pos.scaled_greeks() for pos in self.positions),
            start=Greeks(0, 0, 0, 0, 0)
        )
        
        # Per-100K metrics
        per_100k_factor = 100_000 / self.account_value if self.account_value > 0 else 0
        
        portfolio = PortfolioGreeks(
            total_delta=total_greeks.delta,
            total_gamma=total_greeks.gamma,
            total_theta=total_greeks.theta,
            total_vega=total_greeks.vega,
            total_rho=total_greeks.rho,
            delta_per_100k=total_greeks.delta * per_100k_factor,
            gamma_per_100k=total_greeks.gamma * per_100k_factor,
            theta_per_100k=total_greeks.theta * per_100k_factor,
            vega_per_100k=total_greeks.vega * per_100k_factor,
            account_value=self.account_value,
            num_positions=len(self.positions)
        )
        
        return portfolio
    
    def check_limits(self, warn_threshold: float = 0.8) -> List[GreeksViolation]:
        """
        Check if portfolio Greeks exceed risk limits.
        
        Args:
            warn_threshold: Fraction of limit to trigger warning (0.8 = 80%)
            
        Returns:
            List of violations (empty if all within limits)
        """
        violations = []
        portfolio = self.get_portfolio_greeks()
        
        # Check Delta
        abs_delta = abs(portfolio.total_delta)
        if abs_delta > self.delta_limit:
            violations.append(GreeksViolation(
                violation_type=GreeksViolationType.DELTA_HIGH if portfolio.total_delta > 0 else GreeksViolationType.DELTA_LOW,
                metric_name="Delta",
                current_value=portfolio.total_delta,
                limit_value=self.delta_limit if portfolio.total_delta > 0 else -self.delta_limit,
                severity="critical",
                message=f"Delta {portfolio.total_delta:+.1f} exceeds limit ±{self.delta_limit:.1f}"
            ))
        elif abs_delta > self.delta_limit * warn_threshold:
            violations.append(GreeksViolation(
                violation_type=GreeksViolationType.DELTA_HIGH if portfolio.total_delta > 0 else GreeksViolationType.DELTA_LOW,
                metric_name="Delta",
                current_value=portfolio.total_delta,
                limit_value=self.delta_limit if portfolio.total_delta > 0 else -self.delta_limit,
                severity="warning",
                message=f"Delta {portfolio.total_delta:+.1f} approaching limit ±{self.delta_limit:.1f}"
            ))
        
        # Check Gamma
        if portfolio.total_gamma > self.gamma_limit:
            violations.append(GreeksViolation(
                violation_type=GreeksViolationType.GAMMA_HIGH,
                metric_name="Gamma",
                current_value=portfolio.total_gamma,
                limit_value=self.gamma_limit,
                severity="critical",
                message=f"Gamma {portfolio.total_gamma:.2f} exceeds limit {self.gamma_limit:.2f}"
            ))
        elif portfolio.total_gamma > self.gamma_limit * warn_threshold:
            violations.append(GreeksViolation(
                violation_type=GreeksViolationType.GAMMA_HIGH,
                metric_name="Gamma",
                current_value=portfolio.total_gamma,
                limit_value=self.gamma_limit,
                severity="warning",
                message=f"Gamma {portfolio.total_gamma:.2f} approaching limit {self.gamma_limit:.2f}"
            ))
        
        # Check Theta (we WANT positive theta for premium selling)
        if portfolio.total_theta < self.theta_min:
            violations.append(GreeksViolation(
                violation_type=GreeksViolationType.THETA_LOW,
                metric_name="Theta",
                current_value=portfolio.total_theta,
                limit_value=self.theta_min,
                severity="warning" if portfolio.total_theta > 0 else "critical",
                message=f"Theta {portfolio.total_theta:+.1f} below target {self.theta_min:+.1f}"
            ))
        
        # Check Vega
        abs_vega = abs(portfolio.total_vega)
        if abs_vega > self.vega_limit:
            violations.append(GreeksViolation(
                violation_type=GreeksViolationType.VEGA_HIGH,
                metric_name="Vega",
                current_value=portfolio.total_vega,
                limit_value=self.vega_limit if portfolio.total_vega > 0 else -self.vega_limit,
                severity="critical",
                message=f"Vega {portfolio.total_vega:+.1f} exceeds limit ±{self.vega_limit:.1f}"
            ))
        elif abs_vega > self.vega_limit * warn_threshold:
            violations.append(GreeksViolation(
                violation_type=GreeksViolationType.VEGA_HIGH,
                metric_name="Vega",
                current_value=portfolio.total_vega,
                limit_value=self.vega_limit if portfolio.total_vega > 0 else -self.vega_limit,
                severity="warning",
                message=f"Vega {portfolio.total_vega:+.1f} approaching limit ±{self.vega_limit:.1f}"
            ))
        
        if violations:
            logger.warning(f"Greeks violations detected: {len(violations)} issues")
            for v in violations:
                logger.warning(f"  [{v.severity.upper()}] {v.message}")
        
        return violations
    
    def can_add_position(self, new_greeks: Greeks, quantity: int) -> Tuple[bool, Optional[str]]:
        """
        Check if adding a new position would violate limits.
        
        Args:
            new_greeks: Greeks of the proposed position
            quantity: Number of contracts (+ for long, - for short)
            
        Returns:
            (can_add: bool, reason: Optional[str])
        """
        current = self.get_portfolio_greeks()
        scaled_new = new_greeks * quantity
        
        # Projected portfolio Greeks
        new_delta = current.total_delta + scaled_new.delta
        new_gamma = current.total_gamma + scaled_new.gamma
        new_theta = current.total_theta + scaled_new.theta
        new_vega = current.total_vega + scaled_new.vega
        
        # Check limits
        if abs(new_delta) > self.delta_limit:
            return False, f"Would exceed delta limit: {new_delta:+.1f} > ±{self.delta_limit:.1f}"
        
        if new_gamma > self.gamma_limit:
            return False, f"Would exceed gamma limit: {new_gamma:.2f} > {self.gamma_limit:.2f}"
        
        if abs(new_vega) > self.vega_limit:
            return False, f"Would exceed vega limit: {new_vega:+.1f} > ±{self.vega_limit:.1f}"
        
        return True, None
    
    def suggest_hedge(self) -> Optional[HedgeRecommendation]:
        """
        Suggest a hedge to bring portfolio Greeks within limits.
        
        Returns:
            HedgeRecommendation or None if no hedge needed
        """
        violations = self.check_limits(warn_threshold=0.9)
        if not violations:
            return None
        
        portfolio = self.get_portfolio_greeks()
        
        # Priority: Fix delta first (most important), then vega, then gamma
        critical_violations = [v for v in violations if v.severity == "critical"]
        
        if not critical_violations:
            return None  # Only warnings, no urgent hedge needed
        
        # Handle delta violations
        delta_violation = next(
            (v for v in critical_violations if "Delta" in v.metric_name),
            None
        )
        if delta_violation:
            delta_excess = portfolio.total_delta
            
            if delta_excess > 0:
                # Too bullish: sell calls or buy puts or short stock
                return HedgeRecommendation(
                    reason=f"Reduce positive delta ({delta_excess:+.1f})",
                    action="sell_calls_or_buy_puts",
                    symbol="SPY",  # Example: use portfolio's main symbol
                    quantity=int(abs(delta_excess) / 50),  # Rough estimate
                    expected_delta_change=-abs(delta_excess) * 0.5
                )
            else:
                # Too bearish: buy calls or sell puts or buy stock
                return HedgeRecommendation(
                    reason=f"Reduce negative delta ({delta_excess:+.1f})",
                    action="buy_calls_or_sell_puts",
                    symbol="SPY",
                    quantity=int(abs(delta_excess) / 50),
                    expected_delta_change=abs(delta_excess) * 0.5
                )
        
        # Handle vega violations
        vega_violation = next(
            (v for v in critical_violations if "Vega" in v.metric_name),
            None
        )
        if vega_violation:
            vega_excess = portfolio.total_vega
            
            if vega_excess > 0:
                # Too much positive vega: close long options or sell short-dated
                return HedgeRecommendation(
                    reason=f"Reduce vega exposure ({vega_excess:+.1f})",
                    action="close_long_options",
                    symbol="SPY",
                    quantity=1,
                    expected_vega_change=-abs(vega_excess) * 0.3
                )
            else:
                # Too much negative vega: reduce short positions
                return HedgeRecommendation(
                    reason=f"Reduce negative vega ({vega_excess:+.1f})",
                    action="close_short_options",
                    symbol="SPY",
                    quantity=1,
                    expected_vega_change=abs(vega_excess) * 0.3
                )
        
        return None
    
    def update_account_value(self, new_value: float) -> None:
        """Update account value and rescale limits."""
        old_value = self.account_value
        self.account_value = new_value
        self.scaling_factor = new_value / 100_000
        
        # Rescale limits
        self.delta_limit = MAX_PORTFOLIO_DELTA_PER_100K * self.scaling_factor
        self.gamma_limit = MAX_PORTFOLIO_GAMMA_PER_100K * self.scaling_factor
        self.theta_min = MIN_PORTFOLIO_THETA_PER_100K * self.scaling_factor
        self.vega_limit = abs(MAX_PORTFOLIO_VEGA_PER_100K) * self.scaling_factor
        
        logger.info(
            f"Account value updated: ${old_value:,.0f} → ${new_value:,.0f} "
            f"(limits rescaled)"
        )
    
    def get_position_summary(self) -> str:
        """Get human-readable summary of all positions."""
        if not self.positions:
            return "No positions"
        
        lines = [f"Portfolio: {len(self.positions)} positions"]
        for pos in self.positions:
            scaled = pos.scaled_greeks()
            lines.append(
                f"  {pos.symbol} {pos.strike} {pos.option_type.value} x{pos.quantity:+d}: "
                f"Δ{scaled.delta:+.2f} Γ{scaled.gamma:+.3f} θ{scaled.theta:+.2f}"
            )
        
        portfolio = self.get_portfolio_greeks()
        lines.append(f"\n{portfolio}")
        
        return "\n".join(lines)
