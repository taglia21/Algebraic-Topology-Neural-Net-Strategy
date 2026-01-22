"""
Adaptive Kelly Position Sizing Module
======================================

V2.4 Profitability Enhancement - Optimal position sizing with risk constraints

Key Features:
1. Full/Half/Fractional Kelly - Configurable Kelly fraction (0.25-0.5x default)
2. Regime-Aware Adjustment - Reduce sizing in high-volatility regimes
3. Dynamic Rebalancing - Gradual position adjustment to reduce turnover
4. Correlation-Adjusted Sizing - Account for portfolio diversification
5. Max Drawdown Constraints - Hard limits on position risk

Research Basis:
- Kelly Criterion: f* = (p*b - q)/b for optimal growth
- Half-Kelly typically used for error in parameter estimation
- Regime-aware: Reduce 30-50% in crisis/high-vol periods

Target Performance:
- Increase risk-adjusted returns 20-30%
- Reduce max drawdown by limiting position sizes
- Maintain Sharpe 3.0+ with proper sizing
"""

import numpy as np
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from enum import Enum
from collections import deque
import warnings

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class KellyMode(Enum):
    """Kelly fraction modes."""
    FULL = "full"           # f* (highest growth, highest volatility)
    HALF = "half"           # 0.5 * f* (standard for uncertainty)
    QUARTER = "quarter"     # 0.25 * f* (conservative)
    FRACTIONAL = "fractional"  # Custom fraction
    ADAPTIVE = "adaptive"   # Regime-adjusted


class MarketRegime(Enum):
    """Market regime classifications."""
    BULL = "bull"           # Low vol, positive drift
    BEAR = "bear"           # Negative drift, high fear
    HIGH_VOL = "high_vol"   # Elevated volatility
    CRISIS = "crisis"       # Extreme volatility, correlations spike
    NORMAL = "normal"       # Baseline conditions


# Regime multipliers for position sizing
REGIME_MULTIPLIERS = {
    MarketRegime.BULL: 1.0,
    MarketRegime.NORMAL: 0.9,
    MarketRegime.HIGH_VOL: 0.6,
    MarketRegime.BEAR: 0.5,
    MarketRegime.CRISIS: 0.25,
}


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class KellyConfig:
    """Configuration for Adaptive Kelly Sizer."""
    
    # Kelly fraction parameters
    base_fraction: float = 0.5              # Half-Kelly default
    min_fraction: float = 0.1               # Minimum kelly fraction
    max_fraction: float = 1.0               # Maximum (full Kelly)
    
    # Position constraints
    max_position_pct: float = 0.20          # Max 20% in single position
    min_position_pct: float = 0.01          # Min 1% to take position
    max_total_leverage: float = 1.0         # No leverage by default
    
    # Risk constraints  
    max_portfolio_vol: float = 0.20         # Max 20% annual portfolio vol
    max_single_stock_vol: float = 0.50      # Max 50% vol for single stock
    target_daily_var: float = 0.02          # Target 2% daily VaR
    
    # Regime adjustment
    enable_regime_adjustment: bool = True
    vol_lookback_days: int = 20
    crisis_vol_threshold: float = 0.40      # 40% annualized vol = crisis
    high_vol_threshold: float = 0.25        # 25% annualized vol = high_vol
    
    # V3.0 Drawdown-Aware Scaling (NEW)
    enable_dd_scaling: bool = True
    dd_scale_start: float = 0.05            # Start scaling at 5% DD
    dd_scale_max: float = 0.15              # Full reduction at 15% DD
    dd_scale_min_factor: float = 0.25       # Min 25% of normal size at max DD
    dd_halt_threshold: float = 0.20         # Halt new positions at 20% DD
    
    # V3.0 Volatility Regime Scaling (NEW)
    enable_vol_regime_scaling: bool = True
    vol_median_lookback: int = 60           # 60-day median vol baseline
    vol_high_multiplier: float = 0.50       # Reduce 50% when vol > 2x median
    vol_extreme_multiplier: float = 0.25    # Reduce 75% when vol > 3x median
    
    # Correlation constraints
    enable_correlation_adjustment: bool = True
    max_correlated_exposure: float = 0.40   # Max 40% in correlated assets
    correlation_threshold: float = 0.7       # High correlation threshold
    
    # Rebalancing
    rebalance_threshold_pct: float = 0.05   # 5% deviation triggers rebalance
    max_daily_turnover: float = 0.20        # Max 20% daily turnover
    
    # Error estimation
    estimation_uncertainty: float = 0.3     # 30% uncertainty in parameters
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# KELLY CALCULATIONS
# =============================================================================

def kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    estimation_uncertainty: float = 0.3
) -> float:
    """
    Calculate Kelly fraction for betting sizing.
    
    f* = (p * b - q) / b
    
    Where:
    - p = probability of winning
    - q = 1 - p (probability of losing)
    - b = odds (avg_win / avg_loss)
    
    Args:
        win_rate: Probability of winning trade (0-1)
        avg_win: Average winning trade return
        avg_loss: Average losing trade return (positive number)
        estimation_uncertainty: Reduce fraction by this amount for uncertainty
        
    Returns:
        Optimal kelly fraction (0-1)
    """
    if avg_loss <= 0 or avg_win <= 0 or win_rate <= 0 or win_rate >= 1:
        return 0.0
        
    p = win_rate
    q = 1 - win_rate
    b = avg_win / avg_loss
    
    # Kelly formula
    kelly = (p * b - q) / b
    
    # Adjust for estimation uncertainty
    kelly *= (1 - estimation_uncertainty)
    
    # Clamp to valid range
    return max(0.0, min(1.0, kelly))


def kelly_from_sharpe(
    sharpe: float,
    volatility: float,
    estimation_uncertainty: float = 0.3
) -> float:
    """
    Approximate Kelly fraction from Sharpe ratio.
    
    For continuous returns: f* ≈ μ/σ² ≈ Sharpe/σ
    
    Args:
        sharpe: Annualized Sharpe ratio
        volatility: Annualized volatility
        estimation_uncertainty: Reduce for uncertainty
        
    Returns:
        Kelly fraction
    """
    if volatility <= 0:
        return 0.0
        
    kelly = sharpe / volatility
    kelly *= (1 - estimation_uncertainty)
    
    return max(0.0, min(2.0, kelly))  # Allow up to 2x for leveraged


def diversified_kelly(
    individual_kellys: np.ndarray,
    correlation_matrix: np.ndarray,
    max_leverage: float = 1.0
) -> np.ndarray:
    """
    Adjust Kelly fractions for portfolio diversification.
    
    High correlation = reduce total allocation.
    Low correlation = can allocate more.
    
    Args:
        individual_kellys: Kelly fractions for each asset
        correlation_matrix: Correlation matrix
        max_leverage: Maximum total leverage
        
    Returns:
        Diversified Kelly fractions
    """
    n = len(individual_kellys)
    if n == 0:
        return np.array([])
        
    if correlation_matrix is None or len(correlation_matrix) != n:
        return individual_kellys
        
    # Average pairwise correlation
    avg_corr = (np.sum(np.abs(correlation_matrix)) - n) / (n * n - n) if n > 1 else 0
    
    # Diversification benefit: higher with lower correlation
    div_factor = 1.0 / (1.0 + avg_corr)
    
    # Scale individual Kellys
    scaled = individual_kellys * div_factor
    
    # Enforce max leverage
    total = np.sum(scaled)
    if total > max_leverage:
        scaled *= max_leverage / total
        
    return scaled


# =============================================================================
# REGIME DETECTOR
# =============================================================================

class RegimeDetector:
    """
    Detect current market regime for sizing adjustment.
    """
    
    def __init__(self, config: KellyConfig):
        self.config = config
        self.vol_history: deque = deque(maxlen=config.vol_lookback_days)
        
    def update(self, daily_return: float):
        """Update with new daily return."""
        self.vol_history.append(daily_return)
        
    def detect_regime(
        self,
        current_vol: Optional[float] = None,
        vix: Optional[float] = None,
        market_return_20d: Optional[float] = None
    ) -> MarketRegime:
        """
        Detect current market regime.
        
        Args:
            current_vol: Current annualized volatility (optional)
            vix: Current VIX level (optional)
            market_return_20d: 20-day market return (optional)
            
        Returns:
            MarketRegime enum
        """
        # Calculate vol from history if not provided
        if current_vol is None and len(self.vol_history) >= 5:
            current_vol = np.std(list(self.vol_history)) * np.sqrt(252)
        elif current_vol is None:
            return MarketRegime.NORMAL
            
        # Use VIX if available
        if vix is not None:
            vix_vol = vix / 100 * np.sqrt(252) / np.sqrt(252)  # VIX is annualized
            if vix > 35:
                return MarketRegime.CRISIS
            elif vix > 25:
                return MarketRegime.HIGH_VOL
                
        # Check volatility thresholds
        if current_vol >= self.config.crisis_vol_threshold:
            return MarketRegime.CRISIS
        elif current_vol >= self.config.high_vol_threshold:
            return MarketRegime.HIGH_VOL
            
        # Check market direction
        if market_return_20d is not None:
            if market_return_20d < -0.10:  # -10% in 20 days
                return MarketRegime.BEAR
            elif market_return_20d > 0.05:  # +5% in 20 days
                return MarketRegime.BULL
                
        return MarketRegime.NORMAL
    
    def get_regime_multiplier(self, regime: Optional[MarketRegime] = None) -> float:
        """Get position sizing multiplier for regime."""
        if regime is None:
            regime = self.detect_regime()
        return REGIME_MULTIPLIERS.get(regime, 1.0)


# =============================================================================
# MAIN KELLY SIZER
# =============================================================================

class AdaptiveKellySizer:
    """
    Adaptive Kelly position sizing with regime awareness.
    
    Computes optimal position sizes considering:
    - Kelly criterion for growth optimization
    - Regime-based scaling
    - Correlation-based diversification
    - Risk constraints (max position, max vol, max drawdown)
    
    V3.0 Enhancements:
    - DD-aware scaling: Reduce position sizes as drawdown increases
    - Volatility regime scaling: Cut sizes when vol > 2x median
    - Halt new positions at max DD threshold
    """
    
    def __init__(self, config: Optional[KellyConfig] = None):
        self.config = config or KellyConfig()
        self.regime_detector = RegimeDetector(self.config)
        
        # Tracking
        self.sizing_history: deque = deque(maxlen=1000)
        self.current_positions: Dict[str, float] = {}
        self.portfolio_value: float = 1000000  # Default $1M
        
        # V3.0: Drawdown tracking
        self.high_water_mark: float = 1000000
        self.current_drawdown: float = 0.0
        
        # V3.0: Volatility history for median calculation
        self.vol_history: deque = deque(maxlen=self.config.vol_median_lookback)
        self.median_vol: float = 0.0
    
    def update_drawdown(self, current_portfolio_value: float) -> float:
        """
        V3.0: Update drawdown tracking.
        
        Args:
            current_portfolio_value: Current portfolio value
            
        Returns:
            Current drawdown as decimal (0.10 = 10%)
        """
        if current_portfolio_value > self.high_water_mark:
            self.high_water_mark = current_portfolio_value
            self.current_drawdown = 0.0
        elif self.high_water_mark > 0:
            self.current_drawdown = (self.high_water_mark - current_portfolio_value) / self.high_water_mark
        
        self.portfolio_value = current_portfolio_value
        return self.current_drawdown
    
    def get_dd_scaling_factor(self, current_dd: Optional[float] = None) -> float:
        """
        V3.0: Calculate position size multiplier based on current drawdown.
        
        Linear scale from 1.0 at dd_scale_start to dd_scale_min_factor at dd_scale_max.
        Returns 0.0 if DD exceeds halt threshold.
        
        Args:
            current_dd: Current drawdown (uses self.current_drawdown if None)
            
        Returns:
            Scaling factor 0.0 to 1.0
        """
        if not self.config.enable_dd_scaling:
            return 1.0
            
        dd = current_dd if current_dd is not None else self.current_drawdown
        
        # Halt new positions at threshold
        if dd >= self.config.dd_halt_threshold:
            logger.warning(f"DD {dd:.1%} >= halt threshold {self.config.dd_halt_threshold:.1%}, halting new positions")
            return 0.0
            
        # No scaling below start threshold
        if dd <= self.config.dd_scale_start:
            return 1.0
            
        # Linear interpolation
        dd_range = self.config.dd_scale_max - self.config.dd_scale_start
        dd_excess = min(dd - self.config.dd_scale_start, dd_range)
        scale_range = 1.0 - self.config.dd_scale_min_factor
        
        scaling = 1.0 - (dd_excess / dd_range) * scale_range
        
        logger.debug(f"DD scaling: dd={dd:.1%}, factor={scaling:.2f}")
        return max(self.config.dd_scale_min_factor, scaling)
    
    def update_vol_history(self, current_vol: float) -> None:
        """V3.0: Update volatility history for median calculation."""
        self.vol_history.append(current_vol)
        if len(self.vol_history) >= 5:
            self.median_vol = float(np.median(list(self.vol_history)))
    
    def get_vol_regime_scaling(self, current_vol: float) -> float:
        """
        V3.0: Calculate position size multiplier based on volatility regime.
        
        Compares current vol to median vol and applies scaling.
        
        Args:
            current_vol: Current annualized volatility
            
        Returns:
            Scaling factor 0.25 to 1.0
        """
        if not self.config.enable_vol_regime_scaling:
            return 1.0
            
        self.update_vol_history(current_vol)
        
        if self.median_vol <= 0 or len(self.vol_history) < 10:
            return 1.0  # Insufficient data
            
        vol_ratio = current_vol / self.median_vol
        
        if vol_ratio >= 3.0:
            # Extreme volatility - reduce 75%
            logger.warning(f"Extreme vol regime: {current_vol:.1%} = {vol_ratio:.1f}x median")
            return self.config.vol_extreme_multiplier
        elif vol_ratio >= 2.0:
            # High volatility - reduce 50%
            logger.info(f"High vol regime: {current_vol:.1%} = {vol_ratio:.1f}x median")
            return self.config.vol_high_multiplier
        else:
            return 1.0
        
    def compute_kelly(
        self,
        expected_return: float,
        volatility: float,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None
    ) -> float:
        """
        Compute base Kelly fraction.
        
        Can use either:
        1. Sharpe-based: from expected return and volatility
        2. Win/loss-based: from win rate and average win/loss
        """
        if win_rate is not None and avg_win is not None and avg_loss is not None:
            # Use win/loss formula
            return kelly_fraction(
                win_rate, avg_win, avg_loss,
                self.config.estimation_uncertainty
            )
        else:
            # Use Sharpe approximation
            sharpe = expected_return / volatility if volatility > 0 else 0
            return kelly_from_sharpe(
                sharpe, volatility,
                self.config.estimation_uncertainty
            )
    
    def compute_position_size(
        self,
        symbol: str,
        expected_return: float,
        volatility: float,
        current_price: float,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
        regime: Optional[MarketRegime] = None,
        correlation_with_portfolio: float = 0.0
    ) -> Dict[str, Any]:
        """
        Compute optimal position size for a single asset.
        
        Args:
            symbol: Asset symbol
            expected_return: Expected annualized return
            volatility: Annualized volatility
            current_price: Current asset price
            win_rate: Historical win rate (optional)
            avg_win: Average winning trade (optional)
            avg_loss: Average losing trade (optional)
            regime: Current market regime (optional, auto-detect)
            correlation_with_portfolio: Correlation with existing portfolio
            
        Returns:
            Position sizing recommendation
        """
        # Base Kelly
        base_kelly = self.compute_kelly(
            expected_return, volatility, win_rate, avg_win, avg_loss
        )
        
        # Apply Kelly mode
        if self.config.base_fraction < 1.0:
            kelly = base_kelly * self.config.base_fraction
        else:
            kelly = base_kelly
            
        # Regime adjustment
        if self.config.enable_regime_adjustment:
            if regime is None:
                regime = self.regime_detector.detect_regime(current_vol=volatility)
            regime_mult = self.regime_detector.get_regime_multiplier(regime)
            kelly *= regime_mult
        else:
            regime = MarketRegime.NORMAL
            regime_mult = 1.0
            
        # Correlation adjustment
        if self.config.enable_correlation_adjustment:
            # Reduce position if highly correlated with portfolio
            corr_mult = 1.0 - 0.5 * abs(correlation_with_portfolio)
            kelly *= max(0.5, corr_mult)
        else:
            corr_mult = 1.0
        
        # V3.0: Drawdown-aware scaling
        dd_mult = self.get_dd_scaling_factor()
        kelly *= dd_mult
        
        # V3.0: Volatility regime scaling
        vol_mult = self.get_vol_regime_scaling(volatility)
        kelly *= vol_mult
            
        # Apply volatility constraint
        max_by_vol = self.config.max_single_stock_vol / volatility if volatility > 0 else 1.0
        kelly = min(kelly, max_by_vol)
        
        # Apply position constraints
        kelly = max(kelly, self.config.min_fraction)
        kelly = min(kelly, self.config.max_fraction)
        
        position_pct = min(kelly, self.config.max_position_pct)
        
        # V3.0: Final check - don't take new positions if halted
        if dd_mult == 0.0:
            position_pct = 0.0
        
        # Compute dollar/share values
        position_value = self.portfolio_value * position_pct
        shares = int(position_value / current_price) if current_price > 0 else 0
        actual_value = shares * current_price
        actual_pct = actual_value / self.portfolio_value if self.portfolio_value > 0 else 0
        
        result = {
            'symbol': symbol,
            'base_kelly': base_kelly,
            'adjusted_kelly': kelly,
            'regime': regime.value,
            'regime_multiplier': regime_mult,
            'correlation_multiplier': corr_mult,
            'dd_multiplier': dd_mult,           # V3.0
            'vol_regime_multiplier': vol_mult,   # V3.0
            
            'position_pct': position_pct,
            'position_value': position_value,
            'shares': shares,
            'actual_value': actual_value,
            'actual_pct': actual_pct,
            
            'constraints': {
                'max_position_pct': self.config.max_position_pct,
                'max_by_vol': max_by_vol,
                'applied_fraction': self.config.base_fraction,
                'current_drawdown': self.current_drawdown,  # V3.0
                'halted': dd_mult == 0.0,                   # V3.0
            },
            
            'inputs': {
                'expected_return': expected_return,
                'volatility': volatility,
                'current_price': current_price,
            }
        }
        
        # Track
        self.sizing_history.append({
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'position_pct': position_pct,
            'regime': regime.value,
        })
        
        return result
    
    def compute_portfolio_sizes(
        self,
        assets: List[Dict[str, Any]],
        correlation_matrix: Optional[np.ndarray] = None,
        regime: Optional[MarketRegime] = None
    ) -> Dict[str, Any]:
        """
        Compute optimal position sizes for a portfolio of assets.
        
        Args:
            assets: List of asset dicts with keys:
                - symbol: Asset symbol
                - expected_return: Expected annualized return
                - volatility: Annualized volatility
                - current_price: Current price
                - win_rate, avg_win, avg_loss (optional)
            correlation_matrix: Correlation matrix (optional)
            regime: Current market regime (optional)
            
        Returns:
            Portfolio sizing recommendation
        """
        n = len(assets)
        if n == 0:
            return {'positions': [], 'total_allocation': 0}
            
        # Compute individual Kelly fractions
        individual_kellys = []
        for asset in assets:
            kelly = self.compute_kelly(
                asset.get('expected_return', 0),
                asset.get('volatility', 0.2),
                asset.get('win_rate'),
                asset.get('avg_win'),
                asset.get('avg_loss')
            )
            individual_kellys.append(kelly)
            
        individual_kellys = np.array(individual_kellys)
        
        # Apply diversification adjustment
        if correlation_matrix is not None:
            adjusted_kellys = diversified_kelly(
                individual_kellys, 
                correlation_matrix,
                self.config.max_total_leverage
            )
        else:
            adjusted_kellys = individual_kellys
            
        # Apply base fraction
        adjusted_kellys *= self.config.base_fraction
        
        # Regime adjustment
        if self.config.enable_regime_adjustment:
            if regime is None:
                avg_vol = np.mean([a.get('volatility', 0.2) for a in assets])
                regime = self.regime_detector.detect_regime(current_vol=avg_vol)
            regime_mult = self.regime_detector.get_regime_multiplier(regime)
            adjusted_kellys *= regime_mult
        else:
            regime = MarketRegime.NORMAL
            regime_mult = 1.0
            
        # Apply position constraints
        adjusted_kellys = np.clip(
            adjusted_kellys,
            self.config.min_fraction,
            self.config.max_position_pct
        )
        
        # Enforce total leverage
        total = np.sum(adjusted_kellys)
        if total > self.config.max_total_leverage:
            adjusted_kellys *= self.config.max_total_leverage / total
            total = self.config.max_total_leverage
            
        # Build positions
        positions = []
        for i, asset in enumerate(assets):
            pos_pct = adjusted_kellys[i]
            pos_value = self.portfolio_value * pos_pct
            price = asset.get('current_price', 100)
            shares = int(pos_value / price) if price > 0 else 0
            
            positions.append({
                'symbol': asset['symbol'],
                'base_kelly': individual_kellys[i],
                'adjusted_kelly': adjusted_kellys[i],
                'position_pct': pos_pct,
                'position_value': pos_value,
                'shares': shares,
            })
            
        return {
            'positions': positions,
            'total_allocation': total,
            'regime': regime.value,
            'regime_multiplier': regime_mult,
            'portfolio_value': self.portfolio_value,
        }
    
    def compute_rebalance_trades(
        self,
        target_positions: Dict[str, float],
        current_prices: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Compute trades needed to rebalance to target positions.
        
        Args:
            target_positions: Dict of symbol -> target weight (0-1)
            current_prices: Dict of symbol -> current price
            
        Returns:
            List of trades needed
        """
        trades = []
        total_turnover = 0.0
        
        for symbol, target_weight in target_positions.items():
            current_weight = self.current_positions.get(symbol, 0.0)
            delta = target_weight - current_weight
            
            # Check if rebalance is needed
            if abs(delta) < self.config.rebalance_threshold_pct:
                continue
                
            price = current_prices.get(symbol, 0)
            if price <= 0:
                continue
                
            trade_value = delta * self.portfolio_value
            shares = int(abs(trade_value / price))
            
            if shares > 0:
                trades.append({
                    'symbol': symbol,
                    'side': 'buy' if delta > 0 else 'sell',
                    'shares': shares,
                    'value': abs(trade_value),
                    'current_weight': current_weight,
                    'target_weight': target_weight,
                    'delta': delta,
                })
                total_turnover += abs(delta)
                
        # Check turnover constraint
        if total_turnover > self.config.max_daily_turnover:
            scale = self.config.max_daily_turnover / total_turnover
            for trade in trades:
                trade['shares'] = int(trade['shares'] * scale)
                trade['value'] *= scale
                trade['capped'] = True
                
        return trades
    
    def update_positions(self, positions: Dict[str, float]):
        """Update current position weights."""
        self.current_positions = positions.copy()
        
    def set_portfolio_value(self, value: float):
        """Set portfolio value for sizing calculations."""
        self.portfolio_value = value
        
    def get_sizing_stats(self) -> Dict[str, Any]:
        """Get sizing statistics."""
        if not self.sizing_history:
            return {'n_sizings': 0}
            
        history = list(self.sizing_history)
        avg_pct = np.mean([h['position_pct'] for h in history])
        
        regime_counts = {}
        for h in history:
            r = h['regime']
            regime_counts[r] = regime_counts.get(r, 0) + 1
            
        return {
            'n_sizings': len(history),
            'avg_position_pct': avg_pct,
            'regime_distribution': regime_counts,
        }
    
    # =========================================================================
    # V3.0 CONVENIENCE METHODS
    # =========================================================================
    
    def get_v30_position_scale(
        self,
        volatility: float,
        portfolio_value: Optional[float] = None
    ) -> float:
        """
        V3.0: Get combined position scaling factor for backtest integration.
        
        Combines DD-aware and volatility regime scaling into single factor.
        Call update_drawdown() first to set current DD state.
        
        Args:
            volatility: Current annualized volatility
            portfolio_value: Current portfolio value (optional, uses internal if None)
            
        Returns:
            Combined scaling factor 0.0 to 1.0
        """
        if portfolio_value is not None:
            self.update_drawdown(portfolio_value)
            
        dd_scale = self.get_dd_scaling_factor()
        vol_scale = self.get_vol_regime_scaling(volatility)
        
        return dd_scale * vol_scale
    
    def get_v30_status(self) -> Dict[str, Any]:
        """
        V3.0: Get current status summary for monitoring.
        
        Returns:
            Dict with current DD, scaling factors, and halt status
        """
        return {
            'current_drawdown': self.current_drawdown,
            'high_water_mark': self.high_water_mark,
            'portfolio_value': self.portfolio_value,
            'dd_scaling_factor': self.get_dd_scaling_factor(),
            'median_vol': self.median_vol,
            'vol_history_len': len(self.vol_history),
            'halted': self.current_drawdown >= self.config.dd_halt_threshold,
            'dd_scale_start': self.config.dd_scale_start,
            'dd_scale_max': self.config.dd_scale_max,
            'dd_halt_threshold': self.config.dd_halt_threshold,
        }


# =============================================================================
# V3.0 FACTORY FUNCTIONS
# =============================================================================

def create_v30_sizer(
    base_fraction: float = 0.25,  # Quarter-Kelly for safety
    dd_halt: float = 0.20,
    dd_scale_start: float = 0.05,
    dd_scale_max: float = 0.15,
) -> AdaptiveKellySizer:
    """
    V3.0: Create pre-configured sizer optimized for lower drawdown.
    
    Default configuration targets:
    - Max DD ~20% (halt at 20%, scale from 5-15%)
    - Conservative quarter-Kelly base
    - Full volatility regime scaling
    
    Args:
        base_fraction: Kelly fraction multiplier (default 0.25 = quarter-Kelly)
        dd_halt: DD threshold to halt new positions
        dd_scale_start: DD level to start reducing sizes
        dd_scale_max: DD level for maximum reduction
        
    Returns:
        Configured AdaptiveKellySizer
    """
    config = KellyConfig(
        base_fraction=base_fraction,
        max_position_pct=0.10,  # Max 10% per position (conservative)
        enable_dd_scaling=True,
        dd_scale_start=dd_scale_start,
        dd_scale_max=dd_scale_max,
        dd_scale_min_factor=0.25,
        dd_halt_threshold=dd_halt,
        enable_vol_regime_scaling=True,
        vol_high_multiplier=0.50,
        vol_extreme_multiplier=0.25,
    )
    return AdaptiveKellySizer(config)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Adaptive Kelly Sizer...")
    
    config = KellyConfig()
    sizer = AdaptiveKellySizer(config)
    sizer.set_portfolio_value(1_000_000)
    
    # Test single position sizing
    print("\n1. Testing single position sizing...")
    result = sizer.compute_position_size(
        symbol="AAPL",
        expected_return=0.15,  # 15% expected return
        volatility=0.25,       # 25% volatility
        current_price=175.0,
        win_rate=0.55,
        avg_win=0.02,
        avg_loss=0.015
    )
    print(f"   Symbol: {result['symbol']}")
    print(f"   Base Kelly: {result['base_kelly']:.2%}")
    print(f"   Adjusted Kelly: {result['adjusted_kelly']:.2%}")
    print(f"   Position %: {result['position_pct']:.2%}")
    print(f"   Shares: {result['shares']}")
    print(f"   Regime: {result['regime']}")
    
    # Test portfolio sizing
    print("\n2. Testing portfolio sizing...")
    assets = [
        {'symbol': 'AAPL', 'expected_return': 0.12, 'volatility': 0.25, 'current_price': 175},
        {'symbol': 'MSFT', 'expected_return': 0.10, 'volatility': 0.22, 'current_price': 380},
        {'symbol': 'GOOGL', 'expected_return': 0.14, 'volatility': 0.28, 'current_price': 140},
        {'symbol': 'AMZN', 'expected_return': 0.11, 'volatility': 0.30, 'current_price': 180},
    ]
    
    # Simple correlation matrix
    corr = np.array([
        [1.0, 0.7, 0.6, 0.65],
        [0.7, 1.0, 0.65, 0.6],
        [0.6, 0.65, 1.0, 0.7],
        [0.65, 0.6, 0.7, 1.0],
    ])
    
    portfolio = sizer.compute_portfolio_sizes(assets, corr)
    print(f"   Total allocation: {portfolio['total_allocation']:.2%}")
    print(f"   Regime: {portfolio['regime']}")
    for pos in portfolio['positions']:
        print(f"   {pos['symbol']}: {pos['position_pct']:.2%} ({pos['shares']} shares)")
    
    # Test regime detection
    print("\n3. Testing regime detection...")
    detector = RegimeDetector(config)
    
    # Add some volatile returns
    for r in [0.02, -0.03, 0.015, -0.025, 0.01, -0.02]:
        detector.update(r)
        
    regime = detector.detect_regime(current_vol=0.30)
    print(f"   Detected regime: {regime.value}")
    print(f"   Sizing multiplier: {detector.get_regime_multiplier(regime):.2f}")
    
    # Test Kelly calculations
    print("\n4. Testing Kelly formulas...")
    k1 = kelly_fraction(0.55, 0.02, 0.015)
    print(f"   Win rate 55%, 2% win, 1.5% loss: Kelly = {k1:.2%}")
    
    k2 = kelly_from_sharpe(2.0, 0.15)
    print(f"   Sharpe 2.0, 15% vol: Kelly = {k2:.2%}")
    
    # Test rebalancing
    print("\n5. Testing rebalance calculation...")
    sizer.update_positions({'AAPL': 0.15, 'MSFT': 0.20})
    trades = sizer.compute_rebalance_trades(
        target_positions={'AAPL': 0.20, 'MSFT': 0.15, 'GOOGL': 0.10},
        current_prices={'AAPL': 175, 'MSFT': 380, 'GOOGL': 140}
    )
    print(f"   Trades needed: {len(trades)}")
    for t in trades:
        print(f"   {t['side'].upper()} {t['shares']} {t['symbol']} (${t['value']:.0f})")
    
    print("\n✅ Adaptive Kelly Sizer tests passed!")
