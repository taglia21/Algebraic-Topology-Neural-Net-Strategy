"""Kelly Criterion Position Sizing with Volatility Adjustment.

Phase 4 Optimization: Replace conservative fixed-fractional with optimal Kelly sizing.
Uses Half-Kelly for safety with volatility-based position scaling.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class KellyResult:
    """Result of Kelly calculation."""
    full_kelly: float  # Full Kelly fraction
    half_kelly: float  # Conservative half-Kelly
    position_size_pct: float  # Recommended position size
    expected_return: float  # Expected per-trade return
    edge: float  # Win rate - loss rate weighted
    reason: str  # Explanation


class KellyPositionSizer:
    """
    Kelly Criterion-based position sizing with volatility adjustment.
    
    Kelly Formula: f* = (p * b - q) / b
    Where:
        f* = optimal fraction of bankroll
        p = probability of win
        q = probability of loss (1 - p)
        b = win/loss ratio (avg win / avg loss)
    
    We use Half-Kelly (f*/2) for practical safety.
    """
    
    def __init__(
        self,
        min_position_pct: float = 0.10,  # Minimum 10% position
        max_position_pct: float = 0.60,  # Maximum 60% position
        kelly_fraction: float = 0.50,    # Use half-Kelly
        volatility_scaling: bool = True,  # Scale by inverse volatility
        target_volatility: float = 0.15,  # 15% annual vol target
        min_trades_for_kelly: int = 20,   # Minimum trades before using Kelly
        default_position_pct: float = 0.25  # Default if not enough history
    ):
        self.min_position_pct = min_position_pct
        self.max_position_pct = max_position_pct
        self.kelly_fraction = kelly_fraction
        self.volatility_scaling = volatility_scaling
        self.target_volatility = target_volatility
        self.min_trades_for_kelly = min_trades_for_kelly
        self.default_position_pct = default_position_pct
        
        # Track trade history for Kelly estimation
        self.trade_returns = []
        
    def add_trade_result(self, return_pct: float):
        """Record a trade result for Kelly estimation."""
        self.trade_returns.append(return_pct)
    
    def calculate_kelly(self, 
                        win_rate: float = None,
                        avg_win: float = None,
                        avg_loss: float = None) -> KellyResult:
        """
        Calculate optimal Kelly fraction from trade history or provided stats.
        
        Args:
            win_rate: Optional override for win probability
            avg_win: Optional override for average winning trade
            avg_loss: Optional override for average losing trade
            
        Returns:
            KellyResult with position sizing recommendation
        """
        # Use provided stats or calculate from history
        if win_rate is None or avg_win is None or avg_loss is None:
            if len(self.trade_returns) < self.min_trades_for_kelly:
                return KellyResult(
                    full_kelly=0,
                    half_kelly=0,
                    position_size_pct=self.default_position_pct,
                    expected_return=0,
                    edge=0,
                    reason=f"Insufficient trades ({len(self.trade_returns)}/{self.min_trades_for_kelly}), using default"
                )
            
            returns = np.array(self.trade_returns)
            wins = returns[returns > 0]
            losses = returns[returns < 0]
            
            if len(wins) == 0 or len(losses) == 0:
                return KellyResult(
                    full_kelly=0,
                    half_kelly=0,
                    position_size_pct=self.min_position_pct,
                    expected_return=0,
                    edge=0,
                    reason="No wins or no losses in history"
                )
            
            win_rate = len(wins) / len(returns)
            avg_win = np.mean(wins)
            avg_loss = abs(np.mean(losses))
        
        # Calculate Kelly fraction
        # f* = (p * b - q) / b where b = avg_win / avg_loss
        p = win_rate
        q = 1 - p
        b = avg_win / avg_loss if avg_loss > 0 else 1
        
        if b <= 0:
            return KellyResult(
                full_kelly=0,
                half_kelly=0,
                position_size_pct=self.min_position_pct,
                expected_return=0,
                edge=0,
                reason="Invalid win/loss ratio"
            )
        
        full_kelly = (p * b - q) / b
        
        # Edge: expected value per bet
        edge = p * avg_win - q * avg_loss
        expected_return = edge
        
        # Apply Kelly fraction (half-Kelly for safety)
        half_kelly = full_kelly * self.kelly_fraction
        
        # Constrain to min/max
        position_size = np.clip(half_kelly, self.min_position_pct, self.max_position_pct)
        
        # Handle negative Kelly (edge is negative - don't bet)
        if full_kelly <= 0:
            return KellyResult(
                full_kelly=full_kelly,
                half_kelly=0,
                position_size_pct=0,
                expected_return=expected_return,
                edge=edge,
                reason=f"Negative edge ({edge:.4f}), no position recommended"
            )
        
        return KellyResult(
            full_kelly=full_kelly,
            half_kelly=half_kelly,
            position_size_pct=position_size,
            expected_return=expected_return,
            edge=edge,
            reason=f"Kelly: {full_kelly:.2%}, Half-Kelly: {half_kelly:.2%}"
        )
    
    def get_position_size(
        self,
        current_volatility: float,
        signal_strength: float = 1.0,
        regime: str = 'normal',
        win_rate: float = None,
        avg_win: float = None,
        avg_loss: float = None
    ) -> Tuple[float, str]:
        """
        Get volatility-adjusted position size.
        
        Args:
            current_volatility: Current asset volatility (annualized)
            signal_strength: Signal confidence (0-1)
            regime: Market regime ('bull', 'bear', 'normal', 'high_vol')
            win_rate: Optional win rate override
            avg_win: Optional average win override
            avg_loss: Optional average loss override
            
        Returns:
            Tuple of (position_size_pct, reason)
        """
        # Get base Kelly position
        kelly = self.calculate_kelly(win_rate, avg_win, avg_loss)
        base_position = kelly.position_size_pct
        
        if base_position <= 0:
            return 0, kelly.reason
        
        # Volatility scaling: reduce position when vol is high
        if self.volatility_scaling and current_volatility > 0:
            vol_scalar = self.target_volatility / current_volatility
            vol_scalar = np.clip(vol_scalar, 0.5, 2.0)  # Max 2x, min 0.5x
        else:
            vol_scalar = 1.0
        
        # Signal strength adjustment
        signal_scalar = np.clip(signal_strength, 0.5, 1.0)
        
        # Regime adjustment
        regime_scalars = {
            'bull': 1.2,      # Increase in bull markets
            'bear': 0.6,      # Reduce in bear markets
            'high_vol': 0.5,  # Reduce in high volatility
            'normal': 1.0
        }
        regime_scalar = regime_scalars.get(regime, 1.0)
        
        # Calculate final position
        final_position = base_position * vol_scalar * signal_scalar * regime_scalar
        final_position = np.clip(final_position, self.min_position_pct, self.max_position_pct)
        
        reason = (
            f"Kelly {kelly.half_kelly:.1%} × vol_adj {vol_scalar:.2f} × "
            f"signal {signal_scalar:.2f} × regime {regime_scalar:.2f} = {final_position:.1%}"
        )
        
        return final_position, reason


class DynamicCashAllocator:
    """
    Dynamic cash allocation based on market conditions.
    
    Phase 4: Stop sitting in 50%+ cash by default.
    New rule: Maximum 20% cash unless high volatility or bear market.
    """
    
    def __init__(
        self,
        min_cash_pct: float = 0.05,   # Always keep 5% cash
        max_cash_normal: float = 0.20, # Maximum 20% in normal conditions
        max_cash_high_vol: float = 0.40,  # Up to 40% in high vol
        max_cash_bear: float = 0.35,  # Up to 35% in bear market
        vix_high_threshold: float = 30,  # VIX above this = high vol
        ma_bear_threshold: float = 0  # 200-day MA slope < 0 = bear
    ):
        self.min_cash_pct = min_cash_pct
        self.max_cash_normal = max_cash_normal
        self.max_cash_high_vol = max_cash_high_vol
        self.max_cash_bear = max_cash_bear
        self.vix_high_threshold = vix_high_threshold
        self.ma_bear_threshold = ma_bear_threshold
    
    def get_max_cash_allocation(
        self,
        vix_level: float = None,
        ma_200_slope: float = None,
        drawdown_pct: float = 0
    ) -> Tuple[float, str]:
        """
        Determine maximum allowed cash allocation.
        
        Args:
            vix_level: Current VIX level
            ma_200_slope: Slope of 200-day MA (positive = uptrend)
            drawdown_pct: Current portfolio drawdown
            
        Returns:
            Tuple of (max_cash_pct, reason)
        """
        max_cash = self.max_cash_normal
        reasons = []
        
        # High volatility regime
        if vix_level and vix_level > self.vix_high_threshold:
            max_cash = max(max_cash, self.max_cash_high_vol)
            reasons.append(f"VIX={vix_level:.1f}>30")
        
        # Bear market regime
        if ma_200_slope and ma_200_slope < self.ma_bear_threshold:
            max_cash = max(max_cash, self.max_cash_bear)
            reasons.append("200MA declining")
        
        # Drawdown protection
        if drawdown_pct > 0.05:  # More than 5% drawdown
            max_cash = max(max_cash, 0.30)
            reasons.append(f"DD={drawdown_pct:.1%}")
        
        if not reasons:
            reasons.append("Normal conditions")
        
        return max_cash, " + ".join(reasons)
    
    def get_target_investment_pct(
        self,
        vix_level: float = None,
        ma_200_slope: float = None,
        drawdown_pct: float = 0
    ) -> Tuple[float, str]:
        """
        Get target investment percentage (1 - max_cash).
        
        Returns:
            Tuple of (target_investment_pct, reason)
        """
        max_cash, reason = self.get_max_cash_allocation(
            vix_level, ma_200_slope, drawdown_pct
        )
        target_investment = 1 - max_cash
        return target_investment, f"Target {target_investment:.0%} invested ({reason})"


class VolatilityTargeting:
    """
    Volatility targeting for consistent risk exposure.
    
    Scales position sizes to target a specific portfolio volatility level.
    """
    
    def __init__(
        self,
        target_volatility: float = 0.12,  # 12% target annual vol
        lookback_days: int = 20,  # Volatility estimation window
        vol_floor: float = 0.05,  # Minimum volatility assumption
        vol_cap: float = 0.50     # Maximum volatility assumption
    ):
        self.target_volatility = target_volatility
        self.lookback_days = lookback_days
        self.vol_floor = vol_floor
        self.vol_cap = vol_cap
    
    def calculate_exposure_scalar(
        self,
        current_volatility: float
    ) -> Tuple[float, str]:
        """
        Calculate exposure scalar to achieve target volatility.
        
        Args:
            current_volatility: Current realized/implied volatility
            
        Returns:
            Tuple of (exposure_scalar, reason)
        """
        # Bound volatility estimate
        vol = np.clip(current_volatility, self.vol_floor, self.vol_cap)
        
        # Scalar to achieve target vol
        scalar = self.target_volatility / vol
        
        # Practical limits
        scalar = np.clip(scalar, 0.5, 2.0)
        
        reason = f"Vol target: {self.target_volatility:.1%} / current {vol:.1%} = {scalar:.2f}x"
        
        return scalar, reason


if __name__ == "__main__":
    # Test Kelly position sizer
    print("Testing Kelly Position Sizer")
    print("=" * 50)
    
    sizer = KellyPositionSizer()
    
    # Add simulated trade history (60% win rate, 2:1 R:R)
    import random
    random.seed(42)
    for _ in range(50):
        if random.random() < 0.60:
            sizer.add_trade_result(0.02)  # 2% win
        else:
            sizer.add_trade_result(-0.01)  # 1% loss
    
    # Calculate Kelly
    result = sizer.calculate_kelly()
    print(f"Trade history: {len(sizer.trade_returns)} trades")
    print(f"Full Kelly: {result.full_kelly:.2%}")
    print(f"Half Kelly: {result.half_kelly:.2%}")
    print(f"Recommended position: {result.position_size_pct:.2%}")
    print(f"Edge per trade: {result.edge:.4f}")
    print(f"Reason: {result.reason}")
    
    # Test volatility-adjusted position
    print("\nVolatility-Adjusted Positions:")
    for vol in [0.10, 0.15, 0.20, 0.30]:
        pos, reason = sizer.get_position_size(
            current_volatility=vol,
            signal_strength=0.8,
            regime='normal'
        )
        print(f"  Vol {vol:.0%}: Position {pos:.1%}")
    
    # Test dynamic cash allocator
    print("\n" + "=" * 50)
    print("Testing Dynamic Cash Allocator")
    print("=" * 50)
    
    allocator = DynamicCashAllocator()
    
    scenarios = [
        (15, 0.01, 0),     # Normal: low VIX, uptrend, no DD
        (35, 0.01, 0),     # High vol: high VIX
        (20, -0.005, 0),   # Bear: declining MA
        (25, 0.01, 0.06),  # Drawdown protection
    ]
    
    for vix, ma_slope, dd in scenarios:
        invest_pct, reason = allocator.get_target_investment_pct(vix, ma_slope, dd)
        print(f"  VIX={vix}, MA_slope={ma_slope:.3f}, DD={dd:.0%}: {reason}")
    
    print("\nKelly Position Sizer tests complete!")
