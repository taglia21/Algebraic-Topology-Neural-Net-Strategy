"""
Theta Decay Optimization Engine
================================

Optimizes option entry and exit timing based on theta decay acceleration.

Key Insights:
- Theta decay is non-linear (accelerates near expiration)
- Optimal entry: 30-45 DTE (linear decay phase)
- Optimal exit: 14-21 DTE or 50% profit (before acceleration works against you)
- ATM options have highest theta (but also highest risk)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List
from enum import Enum
import logging

from .utils.black_scholes import BlackScholes, OptionType
from .utils.constants import (
    THETA_ACCELERATION_DTE,
    WHEEL_DTE_RANGE,
    SPREAD_DTE_RANGE,
    IC_DTE_RANGE,
    PROFIT_TARGET_PCT,
    TIME_EXIT_DTE,
    TRADING_DAYS_PER_YEAR,
)

logger = logging.getLogger(__name__)


class IVRegime(Enum):
    """IV environment classification."""
    LOW = "low"  # IV Rank < 30
    NORMAL = "normal"  # IV Rank 30-70
    HIGH = "high"  # IV Rank > 70
    EXTREME = "extreme"  # IV Rank > 90


class TrendDirection(Enum):
    """Market trend classification."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class ThetaMetrics:
    """Theta decay metrics for an option."""
    current_theta: float  # $ per day
    theta_decay_rate: float  # Rate of theta acceleration
    days_to_acceleration: int  # Days until theta accelerates
    optimal_exit_dte: int  # Recommended exit DTE
    expected_decay_total: float  # Expected total theta to collect
    theta_efficiency: float  # % of theoretical max theta (0-1)


@dataclass
class DTERecommendation:
    """DTE entry/exit recommendation."""
    entry_dte_min: int
    entry_dte_max: int
    exit_dte: int
    rationale: str
    expected_hold_days: int
    expected_theta_capture: float


class ThetaDecayEngine:
    """
    Theta decay optimization engine.
    
    Analyzes option theta decay characteristics and recommends
    optimal entry/exit timing for maximum theta capture.
    """
    
    def __init__(self, risk_free_rate: float = 0.05):
        """
        Initialize theta decay engine.
        
        Args:
            risk_free_rate: Risk-free rate for Black-Scholes
        """
        self.risk_free_rate = risk_free_rate
        logger.info("Theta Decay Engine initialized")
    
    def calculate_optimal_dte(
        self,
        iv_rank: float,
        trend: TrendDirection,
        volatility_regime: IVRegime,
        strategy_type: str = "premium_selling"
    ) -> DTERecommendation:
        """
        Calculate optimal DTE for entry and exit.
        
        Logic:
        - High IV (>70): Shorter DTE for faster decay, less vega risk
        - Medium IV (30-70): Standard DTE range
        - Low IV (<30): Longer DTE or skip (premium too small)
        
        Args:
            iv_rank: Current IV rank (0-100)
            trend: Market trend direction
            volatility_regime: Current IV regime
            strategy_type: Type of strategy ('premium_selling', 'directional', etc.)
            
        Returns:
            DTERecommendation with entry/exit guidance
        """
        # Base DTE ranges by strategy
        if strategy_type == "premium_selling":
            base_entry_min, base_entry_max = WHEEL_DTE_RANGE
        elif strategy_type == "spreads":
            base_entry_min, base_entry_max = SPREAD_DTE_RANGE
        elif strategy_type == "iron_condor":
            base_entry_min, base_entry_max = IC_DTE_RANGE
        else:
            base_entry_min, base_entry_max = 30, 45
        
        # Adjust based on IV regime
        if volatility_regime == IVRegime.HIGH or iv_rank > 70:
            # High IV: shorter DTE (faster decay, less vega risk)
            entry_min = max(21, base_entry_min - 5)
            entry_max = base_entry_max - 5
            exit_dte = max(14, TIME_EXIT_DTE - 3)
            rationale = "High IV: shorter DTE for faster theta capture, reduced vega risk"
            
        elif volatility_regime == IVRegime.LOW or iv_rank < 30:
            # Low IV: longer DTE or skip
            entry_min = base_entry_min + 10
            entry_max = base_entry_max + 15
            exit_dte = TIME_EXIT_DTE
            rationale = "Low IV: longer DTE needed for adequate premium, or consider skipping"
            
        else:
            # Normal IV: standard DTE
            entry_min = base_entry_min
            entry_max = base_entry_max
            exit_dte = TIME_EXIT_DTE
            rationale = "Normal IV: standard DTE range optimal"
        
        # Calculate expected holding period
        expected_hold_days = (entry_min + entry_max) // 2 - exit_dte
        
        # Estimate theta capture (rough approximation)
        expected_theta_capture = self._estimate_theta_capture(
            (entry_min + entry_max) // 2,
            exit_dte,
            iv_rank
        )
        
        return DTERecommendation(
            entry_dte_min=entry_min,
            entry_dte_max=entry_max,
            exit_dte=exit_dte,
            rationale=rationale,
            expected_hold_days=expected_hold_days,
            expected_theta_capture=expected_theta_capture
        )
    
    def calculate_theta_metrics(
        self,
        underlying_price: float,
        strike: float,
        dte: int,
        iv: float,
        option_type: OptionType
    ) -> ThetaMetrics:
        """
        Calculate detailed theta metrics for an option.
        
        Args:
            underlying_price: Current underlying price
            strike: Option strike price
            dte: Days to expiration
            iv: Implied volatility (annualized)
            option_type: CALL or PUT
            
        Returns:
            ThetaMetrics with decay analysis
        """
        T = dte / TRADING_DAYS_PER_YEAR
        
        # Calculate current theta
        current_theta = BlackScholes.theta(
            underlying_price, strike, T, self.risk_free_rate, iv, option_type
        )
        
        # Calculate theta in 1 day (to measure acceleration)
        if dte > 1:
            T_next = (dte - 1) / TRADING_DAYS_PER_YEAR
            next_theta = BlackScholes.theta(
                underlying_price, strike, T_next, self.risk_free_rate, iv, option_type
            )
            theta_decay_rate = (next_theta - current_theta) / current_theta if current_theta != 0 else 0
        else:
            theta_decay_rate = 0.0
        
        # Days until acceleration zone
        days_to_acceleration = max(0, dte - THETA_ACCELERATION_DTE)
        
        # Optimal exit DTE (before acceleration hurts)
        optimal_exit_dte = max(14, min(21, dte - 10))
        
        # Expected total decay if held to optimal exit
        expected_decay_total = self._calculate_expected_decay(
            underlying_price, strike, dte, optimal_exit_dte, iv, option_type
        )
        
        # Theta efficiency (how much of max theta we're capturing)
        # ATM has highest theta
        atm_theta = BlackScholes.theta(
            underlying_price, underlying_price, T, self.risk_free_rate, iv, option_type
        )
        theta_efficiency = abs(current_theta / atm_theta) if atm_theta != 0 else 0
        
        return ThetaMetrics(
            current_theta=current_theta,
            theta_decay_rate=theta_decay_rate,
            days_to_acceleration=days_to_acceleration,
            optimal_exit_dte=optimal_exit_dte,
            expected_decay_total=expected_decay_total,
            theta_efficiency=theta_efficiency
        )
    
    def project_decay_curve(
        self,
        underlying_price: float,
        strike: float,
        current_dte: int,
        target_dte: int,
        iv: float,
        option_type: OptionType,
        num_points: int = 30
    ) -> pd.DataFrame:
        """
        Project theta decay curve over time.
        
        Args:
            underlying_price: Current underlying price
            strike: Option strike
            current_dte: Current days to expiration
            target_dte: Target exit DTE
            iv: Implied volatility
            option_type: CALL or PUT
            num_points: Number of points in projection
            
        Returns:
            DataFrame with columns: [dte, days_elapsed, theta, option_value, cumulative_decay]
        """
        if current_dte <= target_dte:
            logger.warning(f"Current DTE {current_dte} <= target {target_dte}")
            return pd.DataFrame()
        
        # Create DTE range
        dte_range = np.linspace(current_dte, target_dte, num_points, dtype=int)
        
        data = []
        cumulative_decay = 0.0
        previous_value = None
        
        for dte in dte_range:
            T = dte / TRADING_DAYS_PER_YEAR
            
            # Calculate option value
            if option_type == OptionType.CALL:
                option_value = BlackScholes.call_price(
                    underlying_price, strike, T, self.risk_free_rate, iv
                )
            else:
                option_value = BlackScholes.put_price(
                    underlying_price, strike, T, self.risk_free_rate, iv
                )
            
            # Calculate theta
            theta = BlackScholes.theta(
                underlying_price, strike, T, self.risk_free_rate, iv, option_type
            )
            
            # Track cumulative decay
            if previous_value is not None:
                decay = previous_value - option_value
                cumulative_decay += decay
            
            data.append({
                'dte': dte,
                'days_elapsed': current_dte - dte,
                'theta': theta,
                'option_value': option_value,
                'cumulative_decay': cumulative_decay
            })
            
            previous_value = option_value
        
        df = pd.DataFrame(data)
        return df
    
    def should_exit_for_theta(
        self,
        current_dte: int,
        entry_dte: int,
        current_profit_pct: float,
        theta_captured_pct: float
    ) -> Tuple[bool, str]:
        """
        Determine if position should be exited based on theta considerations.
        
        Args:
            current_dte: Current days to expiration
            entry_dte: Days to expiration at entry
            current_profit_pct: Current profit as % of max (0-1)
            theta_captured_pct: % of expected theta already captured
            
        Returns:
            (should_exit: bool, reason: str)
        """
        # Exit if hit profit target
        if current_profit_pct >= PROFIT_TARGET_PCT:
            return True, f"Hit profit target ({current_profit_pct:.1%} of max)"
        
        # Exit if approaching acceleration zone
        if current_dte <= TIME_EXIT_DTE:
            return True, f"Approaching theta acceleration zone (DTE={current_dte})"
        
        # Exit if captured most of expected theta
        if theta_captured_pct >= 0.70:  # 70% of expected
            return True, f"Captured most expected theta ({theta_captured_pct:.1%})"
        
        # Exit if approaching expiration rapidly
        days_held = entry_dte - current_dte
        if days_held > 0:
            decay_rate = days_held / entry_dte
            if decay_rate > 0.75 and current_dte < 30:  # Held 75%+ of period
                return True, f"Held position long enough ({decay_rate:.1%} of period)"
        
        return False, "Continue holding for theta capture"
    
    def _estimate_theta_capture(
        self,
        entry_dte: int,
        exit_dte: int,
        iv_rank: float
    ) -> float:
        """
        Estimate total theta to capture over hold period.
        
        Rough approximation: Average daily theta * days held
        Adjusted for IV rank (higher IV = more theta)
        
        Returns:
            Expected theta capture (0-1, represents % of premium)
        """
        if entry_dte <= exit_dte:
            return 0.0
        
        days_held = entry_dte - exit_dte
        
        # Base theta capture: ~1-2% per day for ATM options
        # This is a rough estimate
        avg_daily_theta_pct = 0.015  # 1.5% per day
        
        # Adjust for IV (higher IV = more premium = more theta)
        iv_adjustment = 0.8 + (iv_rank / 100) * 0.4  # 0.8 to 1.2
        
        # Total capture (capped at 80% of premium)
        total_capture = min(0.80, avg_daily_theta_pct * days_held * iv_adjustment)
        
        return total_capture
    
    def _calculate_expected_decay(
        self,
        underlying_price: float,
        strike: float,
        current_dte: int,
        exit_dte: int,
        iv: float,
        option_type: OptionType
    ) -> float:
        """
        Calculate expected option value decay from current DTE to exit DTE.
        
        Assumes underlying price remains constant (theta isolation).
        
        Returns:
            Expected decay in dollar value
        """
        if current_dte <= exit_dte:
            return 0.0
        
        T_current = current_dte / TRADING_DAYS_PER_YEAR
        T_exit = exit_dte / TRADING_DAYS_PER_YEAR
        
        # Calculate option values
        if option_type == OptionType.CALL:
            current_value = BlackScholes.call_price(
                underlying_price, strike, T_current, self.risk_free_rate, iv
            )
            exit_value = BlackScholes.call_price(
                underlying_price, strike, T_exit, self.risk_free_rate, iv
            )
        else:
            current_value = BlackScholes.put_price(
                underlying_price, strike, T_current, self.risk_free_rate, iv
            )
            exit_value = BlackScholes.put_price(
                underlying_price, strike, T_exit, self.risk_free_rate, iv
            )
        
        # Expected decay (for short positions, this is profit)
        expected_decay = current_value - exit_value
        
        return max(0.0, expected_decay)
