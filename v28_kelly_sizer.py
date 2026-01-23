#!/usr/bin/env python3
"""
V28 Enhanced Kelly Criterion Position Sizing
=============================================
Advanced position sizing with fractional Kelly, regime adjustment,
and correlation-aware portfolio weights.

Features:
- Fractional Kelly with configurable fraction (default 0.25)
- Regime-adjusted position sizing
- Correlation-aware portfolio weight optimization
- Dynamic Kelly based on recent performance
- Risk-of-ruin constraints
- Maximum drawdown protection

Formula:
- Full Kelly: f* = (p * b - q) / b
- Fractional Kelly: f = fraction * f*
- Regime-Adjusted: f_adj = f * regime_multiplier
- Correlation-Adjusted: Final weights account for correlation
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('V28_Kelly')


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class PositionSizeMode(Enum):
    """Position sizing mode."""
    FULL_KELLY = "full_kelly"
    HALF_KELLY = "half_kelly"
    QUARTER_KELLY = "quarter_kelly"
    FIXED_FRACTIONAL = "fixed_fractional"
    VOLATILITY_SCALED = "volatility_scaled"


@dataclass
class KellyParameters:
    """Parameters for Kelly calculation."""
    win_rate: float  # Probability of winning
    avg_win: float   # Average winning trade return
    avg_loss: float  # Average losing trade return (positive value)
    win_loss_ratio: float  # avg_win / avg_loss
    expected_value: float  # p * avg_win - q * avg_loss
    edge: float      # Expected value / avg_loss


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""
    full_kelly_fraction: float
    adjusted_fraction: float
    position_size_pct: float
    position_size_dollars: float
    regime_multiplier: float
    correlation_adjustment: float
    volatility_adjustment: float
    kelly_parameters: KellyParameters
    constraints_applied: List[str]
    confidence_score: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'full_kelly_fraction': self.full_kelly_fraction,
            'adjusted_fraction': self.adjusted_fraction,
            'position_size_pct': self.position_size_pct,
            'position_size_dollars': self.position_size_dollars,
            'regime_multiplier': self.regime_multiplier,
            'correlation_adjustment': self.correlation_adjustment,
            'volatility_adjustment': self.volatility_adjustment,
            'kelly_parameters': {
                'win_rate': self.kelly_parameters.win_rate,
                'avg_win': self.kelly_parameters.avg_win,
                'avg_loss': self.kelly_parameters.avg_loss,
                'expected_value': self.kelly_parameters.expected_value
            },
            'constraints_applied': self.constraints_applied,
            'confidence_score': self.confidence_score,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class PortfolioWeightResult:
    """Result of portfolio weight optimization."""
    symbol: str
    raw_weight: float
    kelly_weight: float
    correlation_adjusted_weight: float
    final_weight: float
    position_size_dollars: float
    risk_contribution: float


# =============================================================================
# KELLY CALCULATOR
# =============================================================================

class KellyCalculator:
    """
    Core Kelly Criterion calculations.
    
    Supports multiple Kelly formulations:
    - Classic Kelly: f* = (p*b - q) / b
    - Win-Loss Kelly: f* = p - q/R where R = win/loss ratio
    - Generalized Kelly for continuous returns
    """
    
    @staticmethod
    def calculate_classic_kelly(
        win_rate: float,
        win_loss_ratio: float
    ) -> float:
        """
        Calculate classic Kelly fraction.
        
        f* = (p * b - q) / b = p - q/b
        
        Args:
            win_rate: Probability of win (0-1)
            win_loss_ratio: Average win / average loss
            
        Returns:
            Optimal Kelly fraction
        """
        p = win_rate
        q = 1 - p
        b = win_loss_ratio
        
        if b <= 0:
            return 0.0
        
        kelly = (p * b - q) / b
        return max(0.0, kelly)
    
    @staticmethod
    def calculate_continuous_kelly(
        expected_return: float,
        variance: float
    ) -> float:
        """
        Calculate Kelly for continuous returns.
        
        f* = μ / σ² (for log returns)
        
        Args:
            expected_return: Expected return (annualized)
            variance: Variance of returns (annualized)
            
        Returns:
            Optimal Kelly fraction
        """
        if variance <= 0:
            return 0.0
        
        return expected_return / variance
    
    @staticmethod
    def calculate_kelly_from_trades(trades: List[float]) -> Tuple[float, KellyParameters]:
        """
        Calculate Kelly from a list of trade returns.
        
        Args:
            trades: List of trade returns (positive = profit, negative = loss)
            
        Returns:
            Tuple of (kelly_fraction, parameters)
        """
        if len(trades) < 10:
            return 0.0, KellyParameters(0, 0, 0, 0, 0, 0)
        
        trades_arr = np.array(trades)
        wins = trades_arr[trades_arr > 0]
        losses = trades_arr[trades_arr < 0]
        
        if len(wins) == 0 or len(losses) == 0:
            return 0.0, KellyParameters(
                win_rate=len(wins) / len(trades_arr),
                avg_win=np.mean(wins) if len(wins) > 0 else 0,
                avg_loss=0,
                win_loss_ratio=0,
                expected_value=np.mean(trades_arr),
                edge=0
            )
        
        win_rate = len(wins) / len(trades_arr)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        expected_value = win_rate * avg_win - (1 - win_rate) * avg_loss
        edge = expected_value / avg_loss if avg_loss > 0 else 0
        
        params = KellyParameters(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            win_loss_ratio=win_loss_ratio,
            expected_value=expected_value,
            edge=edge
        )
        
        kelly = KellyCalculator.calculate_classic_kelly(win_rate, win_loss_ratio)
        
        return kelly, params


# =============================================================================
# REGIME-AWARE KELLY
# =============================================================================

class RegimeAwareKelly:
    """
    Adjust Kelly fraction based on market regime.
    
    Different regimes warrant different risk levels:
    - Bull + Low Vol: Full Kelly allowed
    - Bear/Crisis: Significantly reduced
    - High Vol: Volatility-scaled reduction
    """
    
    # Regime multipliers
    REGIME_MULTIPLIERS = {
        'bull': {'low_vol': 1.0, 'normal_vol': 0.85, 'high_vol': 0.60, 'extreme_vol': 0.30},
        'sideways': {'low_vol': 0.80, 'normal_vol': 0.70, 'high_vol': 0.50, 'extreme_vol': 0.25},
        'bear': {'low_vol': 0.50, 'normal_vol': 0.40, 'high_vol': 0.25, 'extreme_vol': 0.10},
        'crisis': {'low_vol': 0.20, 'normal_vol': 0.15, 'high_vol': 0.10, 'extreme_vol': 0.05}
    }
    
    def __init__(self):
        self.current_regime: str = 'sideways'
        self.current_vol_regime: str = 'normal_vol'
    
    def update_regime(self, market_regime: str, vol_regime: str):
        """Update current regime state."""
        self.current_regime = market_regime.lower()
        self.current_vol_regime = vol_regime.lower()
    
    def get_regime_multiplier(
        self,
        market_regime: str = None,
        vol_regime: str = None
    ) -> float:
        """Get position size multiplier for regime."""
        regime = market_regime or self.current_regime
        vol = vol_regime or self.current_vol_regime
        
        regime_map = self.REGIME_MULTIPLIERS.get(regime, self.REGIME_MULTIPLIERS['sideways'])
        return regime_map.get(vol, 0.5)
    
    def adjust_kelly(
        self,
        kelly_fraction: float,
        market_regime: str,
        vol_regime: str
    ) -> Tuple[float, float]:
        """
        Adjust Kelly fraction based on regime.
        
        Returns:
            Tuple of (adjusted_fraction, regime_multiplier)
        """
        multiplier = self.get_regime_multiplier(market_regime, vol_regime)
        adjusted = kelly_fraction * multiplier
        return adjusted, multiplier


# =============================================================================
# CORRELATION-AWARE WEIGHTS
# =============================================================================

class CorrelationAwareWeights:
    """
    Adjust portfolio weights based on correlation structure.
    
    High correlations reduce effective diversification,
    so position sizes should be reduced to maintain risk.
    """
    
    def __init__(
        self,
        max_portfolio_weight: float = 0.25,
        max_correlated_exposure: float = 0.40
    ):
        self.max_portfolio_weight = max_portfolio_weight
        self.max_correlated_exposure = max_correlated_exposure
    
    def calculate_correlation_penalty(
        self,
        symbol: str,
        correlation_matrix: pd.DataFrame,
        current_weights: Dict[str, float]
    ) -> float:
        """
        Calculate penalty factor based on correlation with existing positions.
        
        Args:
            symbol: Symbol to calculate penalty for
            correlation_matrix: Pairwise correlation matrix
            current_weights: Current portfolio weights
            
        Returns:
            Penalty factor (0-1, lower means more penalty)
        """
        if symbol not in correlation_matrix.index:
            return 1.0
        
        if not current_weights:
            return 1.0
        
        # Calculate weighted average correlation with existing positions
        total_corr_exposure = 0.0
        total_weight = 0.0
        
        for other_sym, weight in current_weights.items():
            if other_sym == symbol or weight <= 0:
                continue
            if other_sym not in correlation_matrix.columns:
                continue
            
            corr = correlation_matrix.loc[symbol, other_sym]
            total_corr_exposure += abs(corr) * weight
            total_weight += weight
        
        if total_weight == 0:
            return 1.0
        
        avg_corr = total_corr_exposure / total_weight
        
        # Convert correlation to penalty
        # High correlation -> low penalty factor (reduce size)
        penalty = 1.0 - (avg_corr * 0.5)  # Max 50% reduction at correlation = 1
        
        return max(0.3, min(1.0, penalty))
    
    def optimize_portfolio_weights(
        self,
        target_weights: Dict[str, float],
        correlation_matrix: pd.DataFrame,
        kelly_fractions: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Optimize portfolio weights considering correlations.
        
        Uses a simple iterative approach to balance Kelly sizing
        with correlation-based diversification.
        """
        symbols = list(target_weights.keys())
        n = len(symbols)
        
        if n == 0:
            return {}
        
        # Start with target weights
        weights = target_weights.copy()
        
        # Iterate to adjust for correlations
        for _ in range(3):  # 3 iterations usually sufficient
            new_weights = {}
            
            for sym in symbols:
                # Get correlation penalty
                other_weights = {s: w for s, w in weights.items() if s != sym}
                penalty = self.calculate_correlation_penalty(
                    sym, correlation_matrix, other_weights
                )
                
                # Adjust weight
                base_weight = target_weights[sym]
                kelly_weight = kelly_fractions.get(sym, 0.1)
                
                # Blend target with Kelly
                blended = 0.5 * base_weight + 0.5 * kelly_weight
                
                # Apply correlation penalty
                adjusted = blended * penalty
                
                # Apply max weight constraint
                adjusted = min(adjusted, self.max_portfolio_weight)
                
                new_weights[sym] = adjusted
            
            weights = new_weights
        
        # Normalize to sum to 1 (or less if constraints apply)
        total = sum(weights.values())
        if total > 1.0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights


# =============================================================================
# ENHANCED KELLY POSITION SIZER
# =============================================================================

class V28KellyPositionSizer:
    """
    Enhanced Kelly position sizer with all V28 features.
    
    Combines:
    - Fractional Kelly (default 0.25)
    - Regime-adjusted sizing
    - Correlation-aware weights
    - Volatility targeting
    - Drawdown protection
    """
    
    def __init__(
        self,
        kelly_fraction: float = 0.25,
        min_position_pct: float = 0.02,
        max_position_pct: float = 0.15,
        target_volatility: float = 0.15,
        max_drawdown_limit: float = 0.15,
        min_trades_for_kelly: int = 20,
        portfolio_value: float = 100000.0
    ):
        self.kelly_fraction = kelly_fraction
        self.min_position_pct = min_position_pct
        self.max_position_pct = max_position_pct
        self.target_volatility = target_volatility
        self.max_drawdown_limit = max_drawdown_limit
        self.min_trades_for_kelly = min_trades_for_kelly
        self.portfolio_value = portfolio_value
        
        # Components
        self.kelly_calc = KellyCalculator()
        self.regime_kelly = RegimeAwareKelly()
        self.corr_weights = CorrelationAwareWeights(max_portfolio_weight=max_position_pct)
        
        # History
        self.trade_history: List[float] = []
        self.sizing_history: List[PositionSizeResult] = []
        
        # Current state
        self.current_drawdown: float = 0.0
        self.peak_value: float = portfolio_value
    
    def add_trade(self, return_pct: float):
        """Record a completed trade."""
        self.trade_history.append(return_pct)
        if len(self.trade_history) > 500:
            self.trade_history = self.trade_history[-500:]
    
    def update_portfolio_value(self, value: float):
        """Update current portfolio value for drawdown tracking."""
        self.portfolio_value = value
        self.peak_value = max(self.peak_value, value)
        self.current_drawdown = (self.peak_value - value) / self.peak_value
    
    def update_regime(self, market_regime: str, vol_regime: str):
        """Update market regime for sizing adjustments."""
        self.regime_kelly.update_regime(market_regime, vol_regime)
    
    def calculate_position_size(
        self,
        symbol: str,
        signal_confidence: float = 0.5,
        volatility: float = None,
        market_regime: str = None,
        vol_regime: str = None,
        correlation_matrix: pd.DataFrame = None,
        current_positions: Dict[str, float] = None
    ) -> PositionSizeResult:
        """
        Calculate optimal position size.
        
        Args:
            symbol: Symbol to size
            signal_confidence: Confidence in the signal (0-1)
            volatility: Asset volatility (annualized)
            market_regime: Current market regime
            vol_regime: Current volatility regime
            correlation_matrix: Correlation matrix for portfolio
            current_positions: Current position weights
            
        Returns:
            PositionSizeResult with sizing details
        """
        constraints_applied = []
        
        # Calculate base Kelly from trade history
        if len(self.trade_history) >= self.min_trades_for_kelly:
            full_kelly, kelly_params = self.kelly_calc.calculate_kelly_from_trades(
                self.trade_history
            )
        else:
            # Default conservative sizing
            full_kelly = 0.10
            kelly_params = KellyParameters(
                win_rate=0.55,
                avg_win=0.02,
                avg_loss=0.015,
                win_loss_ratio=1.33,
                expected_value=0.004,
                edge=0.27
            )
            constraints_applied.append("insufficient_history")
        
        # Apply fractional Kelly
        kelly_sized = full_kelly * self.kelly_fraction
        
        # Regime adjustment
        if market_regime and vol_regime:
            regime_adjusted, regime_mult = self.regime_kelly.adjust_kelly(
                kelly_sized, market_regime, vol_regime
            )
        else:
            regime_adjusted = kelly_sized
            regime_mult = 1.0
        
        # Volatility adjustment
        if volatility and volatility > 0:
            vol_scalar = self.target_volatility / volatility
            vol_adjustment = np.clip(vol_scalar, 0.3, 2.0)
        else:
            vol_adjustment = 1.0
        
        vol_adjusted = regime_adjusted * vol_adjustment
        
        # Correlation adjustment
        if correlation_matrix is not None and current_positions:
            corr_penalty = self.corr_weights.calculate_correlation_penalty(
                symbol, correlation_matrix, current_positions
            )
        else:
            corr_penalty = 1.0
        
        corr_adjusted = vol_adjusted * corr_penalty
        
        # Apply signal confidence
        confidence_adjusted = corr_adjusted * signal_confidence
        
        # Drawdown protection
        if self.current_drawdown > self.max_drawdown_limit * 0.5:
            dd_penalty = 1.0 - (self.current_drawdown / self.max_drawdown_limit)
            dd_penalty = max(0.2, dd_penalty)
            confidence_adjusted *= dd_penalty
            constraints_applied.append("drawdown_protection")
        
        # Apply min/max constraints
        final_pct = np.clip(
            confidence_adjusted,
            self.min_position_pct,
            self.max_position_pct
        )
        
        if final_pct != confidence_adjusted:
            if final_pct == self.min_position_pct:
                constraints_applied.append("min_position")
            elif final_pct == self.max_position_pct:
                constraints_applied.append("max_position")
        
        # Calculate dollar amount
        position_dollars = final_pct * self.portfolio_value
        
        # Create result
        result = PositionSizeResult(
            full_kelly_fraction=full_kelly,
            adjusted_fraction=final_pct,
            position_size_pct=final_pct,
            position_size_dollars=position_dollars,
            regime_multiplier=regime_mult,
            correlation_adjustment=corr_penalty,
            volatility_adjustment=vol_adjustment,
            kelly_parameters=kelly_params,
            constraints_applied=constraints_applied,
            confidence_score=signal_confidence
        )
        
        # Store in history
        self.sizing_history.append(result)
        if len(self.sizing_history) > 100:
            self.sizing_history = self.sizing_history[-100:]
        
        return result
    
    def calculate_portfolio_weights(
        self,
        symbols: List[str],
        signals: Dict[str, float],
        volatilities: Dict[str, float],
        correlation_matrix: pd.DataFrame,
        market_regime: str = 'sideways',
        vol_regime: str = 'normal_vol'
    ) -> Dict[str, PortfolioWeightResult]:
        """
        Calculate optimal portfolio weights for multiple symbols.
        
        Args:
            symbols: List of symbols to weight
            signals: Signal strength for each symbol (-1 to 1)
            volatilities: Volatility for each symbol
            correlation_matrix: Pairwise correlations
            market_regime: Current market regime
            vol_regime: Current volatility regime
            
        Returns:
            Dictionary of PortfolioWeightResult per symbol
        """
        results = {}
        
        # Calculate raw Kelly for each symbol
        kelly_fractions = {}
        for sym in symbols:
            signal_strength = abs(signals.get(sym, 0))
            if signal_strength > 0.1:  # Minimum signal threshold
                # Use signal strength to adjust win rate assumption
                assumed_win_rate = 0.50 + signal_strength * 0.15
                assumed_wl_ratio = 1.2 + signal_strength * 0.3
                kelly = self.kelly_calc.calculate_classic_kelly(
                    assumed_win_rate, assumed_wl_ratio
                )
                kelly_fractions[sym] = kelly * self.kelly_fraction
            else:
                kelly_fractions[sym] = 0.0
        
        # Calculate equal-weight baseline
        active_symbols = [s for s in symbols if kelly_fractions.get(s, 0) > 0]
        n_active = len(active_symbols)
        
        if n_active == 0:
            return results
        
        base_weight = 1.0 / n_active
        target_weights = {s: base_weight for s in active_symbols}
        
        # Optimize with correlations
        optimized_weights = self.corr_weights.optimize_portfolio_weights(
            target_weights, correlation_matrix, kelly_fractions
        )
        
        # Calculate risk contributions
        vols = np.array([volatilities.get(s, 0.20) for s in active_symbols])
        weights = np.array([optimized_weights.get(s, 0) for s in active_symbols])
        
        # Simple marginal risk contribution
        # (More sophisticated: use covariance matrix)
        total_risk = np.sqrt(np.sum((weights * vols) ** 2))
        
        # Create results
        for sym in active_symbols:
            raw_weight = target_weights.get(sym, 0)
            kelly_weight = kelly_fractions.get(sym, 0)
            final_weight = optimized_weights.get(sym, 0)
            vol = volatilities.get(sym, 0.20)
            
            risk_contrib = (final_weight * vol) ** 2 / (total_risk ** 2 + 1e-8)
            
            results[sym] = PortfolioWeightResult(
                symbol=sym,
                raw_weight=raw_weight,
                kelly_weight=kelly_weight,
                correlation_adjusted_weight=final_weight,
                final_weight=final_weight,
                position_size_dollars=final_weight * self.portfolio_value,
                risk_contribution=risk_contrib
            )
        
        return results
    
    def get_sizing_stats(self) -> Dict[str, Any]:
        """Get statistics on recent sizing decisions."""
        if not self.sizing_history:
            return {}
        
        recent = self.sizing_history[-20:]
        
        return {
            'avg_position_pct': np.mean([r.position_size_pct for r in recent]),
            'avg_kelly_fraction': np.mean([r.full_kelly_fraction for r in recent]),
            'avg_regime_mult': np.mean([r.regime_multiplier for r in recent]),
            'avg_corr_adj': np.mean([r.correlation_adjustment for r in recent]),
            'constraints_freq': {
                'max_position': sum(1 for r in recent if 'max_position' in r.constraints_applied),
                'min_position': sum(1 for r in recent if 'min_position' in r.constraints_applied),
                'drawdown_protection': sum(1 for r in recent if 'drawdown_protection' in r.constraints_applied)
            }
        }


# =============================================================================
# MAIN / DEMO
# =============================================================================

def demo():
    """Demo the enhanced Kelly position sizer."""
    np.random.seed(42)
    
    logger.info("📊 V28 Enhanced Kelly Position Sizer Demo")
    logger.info("=" * 50)
    
    # Initialize sizer
    sizer = V28KellyPositionSizer(
        kelly_fraction=0.25,
        min_position_pct=0.02,
        max_position_pct=0.15,
        portfolio_value=100000.0
    )
    
    # Simulate trade history
    n_trades = 50
    wins = np.random.rand(n_trades) > 0.45  # 55% win rate
    for i in range(n_trades):
        if wins[i]:
            pnl = np.random.uniform(0.01, 0.05)  # 1-5% wins
        else:
            pnl = -np.random.uniform(0.005, 0.03)  # 0.5-3% losses
        sizer.add_trade(pnl)
    
    # Calculate position size for a new trade
    result = sizer.calculate_position_size(
        symbol='AAPL',
        signal_confidence=0.75,
        volatility=0.25,
        market_regime='bull',
        vol_regime='normal_vol'
    )
    
    logger.info(f"\n📈 Single Position Sizing Result:")
    logger.info(f"   Symbol: AAPL")
    logger.info(f"   Full Kelly: {result.full_kelly_fraction:.2%}")
    logger.info(f"   Fractional Kelly (0.25x): {result.full_kelly_fraction * 0.25:.2%}")
    logger.info(f"   Regime Multiplier: {result.regime_multiplier:.2f}")
    logger.info(f"   Volatility Adjustment: {result.volatility_adjustment:.2f}")
    logger.info(f"   Correlation Adjustment: {result.correlation_adjustment:.2f}")
    logger.info(f"   Final Position: {result.position_size_pct:.2%} (${result.position_size_dollars:,.0f})")
    if result.constraints_applied:
        logger.info(f"   Constraints Applied: {', '.join(result.constraints_applied)}")
    
    # Kelly parameters
    logger.info(f"\n📊 Kelly Parameters:")
    logger.info(f"   Win Rate: {result.kelly_parameters.win_rate:.1%}")
    logger.info(f"   Avg Win: {result.kelly_parameters.avg_win:.2%}")
    logger.info(f"   Avg Loss: {result.kelly_parameters.avg_loss:.2%}")
    logger.info(f"   Win/Loss Ratio: {result.kelly_parameters.win_loss_ratio:.2f}")
    logger.info(f"   Expected Value: {result.kelly_parameters.expected_value:.3%}")
    
    # Portfolio optimization demo
    logger.info(f"\n📊 Portfolio Weights Optimization:")
    
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META']
    signals = {'AAPL': 0.7, 'MSFT': 0.6, 'GOOGL': 0.5, 'NVDA': 0.8, 'META': 0.4}
    volatilities = {'AAPL': 0.25, 'MSFT': 0.22, 'GOOGL': 0.28, 'NVDA': 0.45, 'META': 0.35}
    
    # Create correlation matrix
    n = len(symbols)
    corr_matrix = np.eye(n)
    # Add some correlations
    corr_matrix[0, 1] = corr_matrix[1, 0] = 0.65  # AAPL-MSFT
    corr_matrix[0, 2] = corr_matrix[2, 0] = 0.55  # AAPL-GOOGL
    corr_matrix[3, 4] = corr_matrix[4, 3] = 0.45  # NVDA-META
    
    corr_df = pd.DataFrame(corr_matrix, index=symbols, columns=symbols)
    
    portfolio_results = sizer.calculate_portfolio_weights(
        symbols=symbols,
        signals=signals,
        volatilities=volatilities,
        correlation_matrix=corr_df,
        market_regime='bull',
        vol_regime='normal_vol'
    )
    
    for sym, result in sorted(portfolio_results.items(), key=lambda x: -x[1].final_weight):
        logger.info(f"   {sym}: {result.final_weight:.1%} "
                   f"(${result.position_size_dollars:,.0f}, "
                   f"risk contrib: {result.risk_contribution:.1%})")
    
    # Sizing stats
    stats = sizer.get_sizing_stats()
    logger.info(f"\n📊 Sizing Statistics:")
    logger.info(f"   Avg Position: {stats.get('avg_position_pct', 0):.2%}")
    logger.info(f"   Avg Kelly: {stats.get('avg_kelly_fraction', 0):.2%}")


if __name__ == '__main__':
    demo()
