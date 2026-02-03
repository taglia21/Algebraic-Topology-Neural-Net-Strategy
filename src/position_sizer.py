"""
Production-Ready Position Sizing Module
========================================

Kelly Criterion-based position sizing with volatility scaling and safety constraints.

Features:
- Kelly Criterion calculation from historical performance metrics
- Half-Kelly default for conservative risk management
- Volatility-based position scaling using ATR percentiles
- Confidence score integration for signal strength
- Portfolio-level constraints (max/min position sizes)
- Heat-adjusted sizing for drawdown periods

Author: Trading System
Version: 1.0.0
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class SizingMode(Enum):
    """Position sizing strategies."""
    FULL_KELLY = "full_kelly"
    HALF_KELLY = "half_kelly"
    QUARTER_KELLY = "quarter_kelly"
    FIXED_FRACTIONAL = "fixed_fractional"


@dataclass
class PerformanceMetrics:
    """Historical trading performance metrics for Kelly calculation."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_profit: float
    total_loss: float
    
    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades
    
    @property
    def avg_win(self) -> float:
        """Calculate average win amount."""
        if self.winning_trades == 0:
            return 0.0
        return self.total_profit / self.winning_trades
    
    @property
    def avg_loss(self) -> float:
        """Calculate average loss amount (positive value)."""
        if self.losing_trades == 0:
            return 0.0
        return abs(self.total_loss) / self.losing_trades
    
    @property
    def payoff_ratio(self) -> float:
        """Calculate payoff ratio (avg win / avg loss)."""
        if self.avg_loss == 0:
            return 0.0
        return self.avg_win / self.avg_loss
    
    @property
    def expectancy(self) -> float:
        """Calculate system expectancy."""
        return (self.win_rate * self.avg_win) - ((1 - self.win_rate) * self.avg_loss)


@dataclass
class SizingConfig:
    """Position sizing configuration."""
    # Kelly parameters
    sizing_mode: SizingMode = SizingMode.HALF_KELLY
    kelly_multiplier: float = 0.5  # 0.5 = half-Kelly, 1.0 = full Kelly
    
    # Portfolio constraints
    max_position_pct: float = 0.10  # 10% max position
    min_position_pct: float = 0.01  # 1% min position
    min_position_value: float = 100.0  # $100 minimum
    max_position_value: Optional[float] = None  # Optional cap
    
    # Volatility scaling
    use_volatility_scaling: bool = True
    vol_lookback_days: int = 20
    vol_percentile_low: float = 25.0  # Scale up if vol below this percentile
    vol_percentile_high: float = 75.0  # Scale down if vol above this percentile
    vol_scale_factor: float = 0.5  # How much to scale by volatility
    
    # Confidence integration
    min_confidence: float = 0.3  # Minimum confidence to size position
    confidence_power: float = 1.5  # Exponent for confidence scaling
    
    # Heat adjustment (reduce size during drawdowns)
    use_heat_adjustment: bool = True
    max_consecutive_losses: int = 3
    heat_reduction_factor: float = 0.5  # Reduce to 50% after max losses
    
    # Safety limits
    min_kelly_fraction: float = 0.01  # Don't size below 1%
    max_kelly_fraction: float = 0.25  # Cap Kelly at 25% even for full-Kelly


@dataclass
class PositionSize:
    """Position sizing result with detailed breakdown."""
    position_value: float
    position_pct: float
    kelly_fraction: float
    confidence_adjusted: float
    volatility_adjusted: float
    final_size: float
    sizing_factors: dict
    is_valid: bool
    rejection_reason: Optional[str] = None


class PositionSizer:
    """
    Kelly Criterion-based position sizer with advanced risk controls.
    
    Combines Kelly Criterion with volatility scaling, confidence scores,
    and dynamic heat adjustment for robust position sizing.
    """
    
    def __init__(self, config: Optional[SizingConfig] = None):
        """
        Initialize position sizer.
        
        Args:
            config: Sizing configuration (uses defaults if None)
        """
        self.config = config or SizingConfig()
        self.consecutive_losses = 0
        self.total_trades = 0
        
        # Default performance metrics (bootstrap values)
        self.default_metrics = PerformanceMetrics(
            total_trades=100,
            winning_trades=55,
            losing_trades=45,
            total_profit=11000,
            total_loss=-9000
        )
        
        logger.info(f"PositionSizer initialized with mode: {self.config.sizing_mode.value}")
    
    def calculate_kelly_fraction(self, win_rate: float, avg_win: float, 
                                 avg_loss: float) -> float:
        """
        Calculate Kelly Criterion fraction with CRITICAL division-by-zero protection.
        
        Kelly formula: f = (p * b - q) / b
        where:
        - p = win rate
        - q = 1 - p (loss rate)
        - b = payoff ratio (avg_win / avg_loss)
        
        Args:
            win_rate: Historical win rate (0.0 to 1.0)
            avg_win: Average winning trade amount
            avg_loss: Average losing trade amount (positive value)
            
        Returns:
            Kelly fraction (capped between min and max)
        """
        # Validate win_rate
        if not np.isfinite(win_rate) or win_rate <= 0 or win_rate >= 1:
            logger.warning(f"Invalid win_rate {win_rate}, using default")
            return self.config.min_kelly_fraction
        
        # CRITICAL: Validate avg_win (check for zero AND negative)
        if not np.isfinite(avg_win) or avg_win <= 0:
            logger.warning(f"Invalid avg_win {avg_win}, using default")
            return self.config.min_kelly_fraction
        
        # CRITICAL: Validate avg_loss (check for zero AND negative)
        if not np.isfinite(avg_loss) or avg_loss <= 0:
            logger.warning(f"Invalid avg_loss {avg_loss}, using default")
            return self.config.min_kelly_fraction
        
        # Calculate payoff ratio
        payoff_ratio = avg_win / avg_loss
        
        # CRITICAL: Additional safety check for payoff_ratio
        if not np.isfinite(payoff_ratio) or payoff_ratio == 0:
            logger.warning(f"Invalid payoff_ratio {payoff_ratio}, using default")
            return self.config.min_kelly_fraction
        
        # Kelly formula
        loss_rate = 1 - win_rate
        kelly_fraction = (win_rate * payoff_ratio - loss_rate) / payoff_ratio
        
        # Validate result before applying multiplier
        if not np.isfinite(kelly_fraction):
            logger.error(f"Kelly calculation produced invalid result: {kelly_fraction}")
            return self.config.min_kelly_fraction
        
        # Apply Kelly multiplier (half-Kelly, etc.)
        kelly_fraction *= self.config.kelly_multiplier
        
        # Apply safety bounds
        kelly_fraction = max(self.config.min_kelly_fraction, 
                           min(kelly_fraction, self.config.max_kelly_fraction))
        
        logger.debug(f"Kelly: win_rate={win_rate:.2%}, payoff={payoff_ratio:.2f}, "
                    f"kelly={kelly_fraction:.2%}")
        
        return kelly_fraction
    
    def calculate_volatility_scalar(self, current_volatility: float,
                                    historical_volatilities: np.ndarray) -> float:
        """
        Calculate position size scalar based on current volatility percentile.
        HIGH-SEVERITY FIX: Comprehensive array validation and invalid data filtering.
        
        Lower volatility -> scale up (opportunity)
        Higher volatility -> scale down (risk)
        
        Args:
            current_volatility: Current ATR or volatility measure
            historical_volatilities: Array of historical volatility values
            
        Returns:
            Volatility scalar (0.5 to 1.5)
        """
        if not self.config.use_volatility_scaling:
            return 1.0
        
        # Validate and clean historical data
        if not isinstance(historical_volatilities, np.ndarray):
            try:
                historical_volatilities = np.array(historical_volatilities)
            except Exception as e:
                logger.error(f"Cannot convert historical_volatilities to array: {e}")
                return 1.0
        
        # CRITICAL: Remove invalid values (NaN, inf, negative)
        valid_vols = historical_volatilities[
            np.isfinite(historical_volatilities) & (historical_volatilities > 0)
        ]
        
        if len(valid_vols) < 10:
            logger.warning(f"Insufficient valid volatility data: {len(valid_vols)} < 10")
            return 1.0
        
        # Validate current volatility
        if not np.isfinite(current_volatility) or current_volatility <= 0:
            logger.warning(f"Invalid current volatility: {current_volatility}")
            return 1.0
        
        # Calculate percentile of current volatility
        percentile = (valid_vols < current_volatility).sum() / len(valid_vols) * 100
        
        # Scale based on percentile
        if percentile < self.config.vol_percentile_low:
            # Low volatility - scale up
            scale = 1.0 + self.config.vol_scale_factor * (
                (self.config.vol_percentile_low - percentile) / self.config.vol_percentile_low
            )
        elif percentile > self.config.vol_percentile_high:
            # High volatility - scale down
            scale = 1.0 - self.config.vol_scale_factor * (
                (percentile - self.config.vol_percentile_high) / (100 - self.config.vol_percentile_high)
            )
        else:
            # Normal volatility - no adjustment
            scale = 1.0
        
        # Bound the scalar
        scale = max(0.5, min(1.5, scale))
        
        logger.debug(f"Volatility: percentile={percentile:.1f}, scalar={scale:.2f}")
        
        return scale
    
    def calculate_confidence_scalar(self, confidence: float) -> float:
        """
        Convert confidence score to position size scalar.
        
        Uses power function to emphasize high-confidence trades.
        
        Args:
            confidence: Confidence score (0.0 to 1.0)
            
        Returns:
            Confidence scalar (0.0 to 1.0)
        """
        if confidence < self.config.min_confidence:
            return 0.0
        
        # Normalize to 0-1 range above minimum confidence
        normalized = (confidence - self.config.min_confidence) / (1.0 - self.config.min_confidence)
        
        # Apply power function for non-linear scaling
        scalar = normalized ** self.config.confidence_power
        
        logger.debug(f"Confidence: raw={confidence:.2f}, scalar={scalar:.2f}")
        
        return scalar
    
    def calculate_heat_adjustment(self) -> float:
        """
        Calculate heat adjustment based on consecutive losses.
        
        Reduces position size during losing streaks to preserve capital.
        
        Returns:
            Heat scalar (0.5 to 1.0)
        """
        if not self.config.use_heat_adjustment:
            return 1.0
        
        if self.consecutive_losses >= self.config.max_consecutive_losses:
            logger.info(f"Heat adjustment active: {self.consecutive_losses} consecutive losses")
            return self.config.heat_reduction_factor
        
        return 1.0
    
    def size_position(self, portfolio_value: float, confidence: float = 1.0,
                     volatility_percentile: Optional[float] = None,
                     performance_metrics: Optional[PerformanceMetrics] = None) -> PositionSize:
        """
        Calculate optimal position size combining all factors.
        
        Args:
            portfolio_value: Current portfolio value
            confidence: Signal confidence (0.0 to 1.0)
            volatility_percentile: Current volatility percentile (0-100)
            performance_metrics: Historical performance for Kelly calculation
            
        Returns:
            PositionSize object with detailed breakdown
        """
        # Use provided or default metrics
        metrics = performance_metrics or self.default_metrics
        
        # Calculate base Kelly fraction
        kelly_fraction = self.calculate_kelly_fraction(
            metrics.win_rate,
            metrics.avg_win,
            metrics.avg_loss
        )
        
        # Calculate confidence scalar
        confidence_scalar = self.calculate_confidence_scalar(confidence)
        if confidence_scalar == 0.0:
            return PositionSize(
                position_value=0.0,
                position_pct=0.0,
                kelly_fraction=kelly_fraction,
                confidence_adjusted=0.0,
                volatility_adjusted=0.0,
                final_size=0.0,
                sizing_factors={},
                is_valid=False,
                rejection_reason=f"Confidence {confidence:.2f} below minimum {self.config.min_confidence:.2f}"
            )
        
        # Calculate volatility scalar
        vol_scalar = 1.0
        if volatility_percentile is not None:
            # Approximate volatility scalar from percentile
            if volatility_percentile < self.config.vol_percentile_low:
                vol_scalar = 1.0 + self.config.vol_scale_factor * 0.5
            elif volatility_percentile > self.config.vol_percentile_high:
                vol_scalar = 1.0 - self.config.vol_scale_factor * 0.5
        
        # Calculate heat adjustment
        heat_scalar = self.calculate_heat_adjustment()
        
        # Combine all factors
        confidence_adjusted = kelly_fraction * confidence_scalar
        volatility_adjusted = confidence_adjusted * vol_scalar
        heat_adjusted = volatility_adjusted * heat_scalar
        
        # Apply portfolio constraints
        position_pct = max(self.config.min_position_pct, 
                          min(heat_adjusted, self.config.max_position_pct))
        
        position_value = portfolio_value * position_pct
        
        # Apply absolute value constraints
        if position_value < self.config.min_position_value:
            return PositionSize(
                position_value=0.0,
                position_pct=0.0,
                kelly_fraction=kelly_fraction,
                confidence_adjusted=confidence_adjusted,
                volatility_adjusted=volatility_adjusted,
                final_size=0.0,
                sizing_factors={
                    "kelly": kelly_fraction,
                    "confidence": confidence_scalar,
                    "volatility": vol_scalar,
                    "heat": heat_scalar
                },
                is_valid=False,
                rejection_reason=f"Position value ${position_value:.2f} below minimum ${self.config.min_position_value:.2f}"
            )
        
        if self.config.max_position_value and position_value > self.config.max_position_value:
            position_value = self.config.max_position_value
            position_pct = position_value / portfolio_value
        
        sizing_factors = {
            "kelly_fraction": kelly_fraction,
            "confidence_scalar": confidence_scalar,
            "volatility_scalar": vol_scalar,
            "heat_scalar": heat_scalar,
            "win_rate": metrics.win_rate,
            "payoff_ratio": metrics.payoff_ratio,
            "expectancy": metrics.expectancy
        }
        
        logger.info(f"Position sized: ${position_value:,.2f} ({position_pct:.2%}) - "
                   f"Kelly={kelly_fraction:.2%}, Conf={confidence_scalar:.2f}, "
                   f"Vol={vol_scalar:.2f}, Heat={heat_scalar:.2f}")
        
        return PositionSize(
            position_value=position_value,
            position_pct=position_pct,
            kelly_fraction=kelly_fraction,
            confidence_adjusted=confidence_adjusted,
            volatility_adjusted=volatility_adjusted,
            final_size=position_pct,
            sizing_factors=sizing_factors,
            is_valid=True
        )
    
    def record_trade_result(self, is_winner: bool):
        """
        Record trade result for heat adjustment tracking.
        
        Args:
            is_winner: True if trade was profitable
        """
        self.total_trades += 1
        
        if is_winner:
            self.consecutive_losses = 0
            logger.debug(f"Win recorded, consecutive losses reset")
        else:
            self.consecutive_losses += 1
            logger.debug(f"Loss recorded, consecutive losses: {self.consecutive_losses}")
    
    def reset_heat(self):
        """Reset heat tracking (e.g., after successful recovery)."""
        self.consecutive_losses = 0
        logger.info("Heat tracking reset")
    
    def update_performance_metrics(self, metrics: PerformanceMetrics):
        """
        Update default performance metrics from recent trading history.
        
        Args:
            metrics: Updated performance metrics
        """
        self.default_metrics = metrics
        logger.info(f"Performance metrics updated: WR={metrics.win_rate:.2%}, "
                   f"Payoff={metrics.payoff_ratio:.2f}, Exp=${metrics.expectancy:.2f}")


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize position sizer
    sizer = PositionSizer()
    
    # Create sample performance metrics
    metrics = PerformanceMetrics(
        total_trades=100,
        winning_trades=60,
        losing_trades=40,
        total_profit=15000,
        total_loss=-10000
    )
    
    print(f"Performance: WR={metrics.win_rate:.2%}, Payoff={metrics.payoff_ratio:.2f}")
    
    # Calculate Kelly fraction
    kelly = sizer.calculate_kelly_fraction(metrics.win_rate, metrics.avg_win, metrics.avg_loss)
    print(f"Kelly fraction: {kelly:.2%}")
    
    # Size a position
    portfolio_value = 100000
    confidence = 0.8
    
    position = sizer.size_position(portfolio_value, confidence, 50.0, metrics)
    
    if position.is_valid:
        print(f"\nPosition sizing result:")
        print(f"  Value: ${position.position_value:,.2f}")
        print(f"  Percentage: {position.position_pct:.2%}")
        print(f"  Factors: {position.sizing_factors}")
    else:
        print(f"\nPosition rejected: {position.rejection_reason}")
    
    # Test heat adjustment
    print("\nTesting heat adjustment:")
    for i in range(5):
        sizer.record_trade_result(is_winner=False)
        heat = sizer.calculate_heat_adjustment()
        print(f"  After {i+1} losses: heat={heat:.2f}")
