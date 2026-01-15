"""Dynamic Position Optimizer for Phase 9.

Implements intelligent position sizing and portfolio construction:
1. Regime-adaptive Kelly criterion
2. Risk-parity allocation
3. Drawdown-aware scaling
4. Correlation-adjusted weights
5. Transaction cost optimization

Target: Optimal risk-adjusted position sizing
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class PositionRecommendation:
    """Position sizing recommendation for a ticker."""
    ticker: str
    
    # Raw position
    raw_weight: float
    kelly_weight: float
    
    # Risk-adjusted
    risk_parity_weight: float
    correlation_adjusted_weight: float
    
    # Final recommendation
    final_weight: float
    final_shares: int
    
    # Constraints
    max_weight: float
    min_weight: float
    
    # Risk metrics
    expected_return: float
    expected_volatility: float
    var_95: float  # Value at Risk 95%
    
    # Action
    action: str  # 'buy', 'sell', 'hold', 'rebalance'
    urgency: float  # 0-1


@dataclass
class PortfolioState:
    """Current portfolio state."""
    total_value: float
    cash: float
    invested: float
    
    positions: Dict[str, float]  # {ticker: value}
    weights: Dict[str, float]   # {ticker: weight}
    
    current_drawdown: float
    max_drawdown: float
    portfolio_beta: float
    portfolio_volatility: float


class KellyCalculator:
    """Compute Kelly criterion for position sizing."""
    
    def __init__(
        self,
        kelly_fraction: float = 0.25,  # Use 25% Kelly (conservative)
        max_position: float = 0.15,
        min_position: float = 0.02,
    ):
        self.kelly_fraction = kelly_fraction
        self.max_position = max_position
        self.min_position = min_position
    
    def compute_kelly_weight(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        confidence: float = 1.0,
    ) -> float:
        """
        Compute fractional Kelly weight.
        
        Kelly formula: f* = (bp - q) / b
        where b = avg_win/avg_loss, p = win_rate, q = 1-p
        """
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0
        
        b = abs(avg_win / avg_loss)  # Win/loss ratio
        p = win_rate
        q = 1 - p
        
        # Full Kelly
        full_kelly = (b * p - q) / b
        
        # Apply fractional Kelly
        kelly = full_kelly * self.kelly_fraction
        
        # Apply confidence scaling
        kelly *= confidence
        
        # Apply bounds
        return np.clip(kelly, self.min_position if kelly > 0 else 0, self.max_position)
    
    def compute_kelly_from_signal(
        self,
        signal: float,  # -1 to 1
        expected_return: float,  # Annualized
        volatility: float,  # Annualized
        confidence: float = 1.0,
    ) -> float:
        """
        Compute Kelly weight from signal strength and risk metrics.
        
        Uses simplified Kelly: f* = μ/σ² (for log returns)
        """
        if volatility == 0:
            return 0.0
        
        # Adjust expected return by signal strength
        adj_return = expected_return * abs(signal)
        
        # Kelly formula for continuous returns
        full_kelly = adj_return / (volatility ** 2)
        
        # Apply fractional Kelly
        kelly = full_kelly * self.kelly_fraction
        
        # Apply confidence
        kelly *= confidence
        
        # Apply bounds
        weight = np.clip(kelly, 0, self.max_position)
        
        # Sign based on signal
        return weight if signal >= 0 else -weight


class RiskParityOptimizer:
    """Risk parity portfolio optimization."""
    
    def __init__(
        self,
        target_volatility: float = 0.15,  # 15% annual portfolio vol
        max_iterations: int = 100,
    ):
        self.target_volatility = target_volatility
        self.max_iterations = max_iterations
    
    def optimize(
        self,
        tickers: List[str],
        volatilities: Dict[str, float],
        correlations: Optional[np.ndarray] = None,
        expected_returns: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Compute risk parity weights.
        
        Risk parity: Each position contributes equally to portfolio risk.
        """
        if not tickers or not volatilities:
            return {}
        
        n = len(tickers)
        vols = np.array([volatilities.get(t, 0.20) for t in tickers])
        
        # Initial weights: inverse volatility
        raw_weights = 1 / (vols + 1e-6)
        weights = raw_weights / np.sum(raw_weights)
        
        # If we have correlations, iterate to true risk parity
        if correlations is not None and correlations.shape == (n, n):
            weights = self._iterate_risk_parity(weights, vols, correlations)
        
        # Scale to target volatility
        portfolio_vol = self._compute_portfolio_vol(weights, vols, correlations)
        if portfolio_vol > 0:
            scale = self.target_volatility / portfolio_vol
            weights = np.clip(weights * scale, 0, 0.20)
            # Renormalize
            weights = weights / np.sum(weights)
        
        return {t: w for t, w in zip(tickers, weights)}
    
    def _iterate_risk_parity(
        self,
        initial_weights: np.ndarray,
        volatilities: np.ndarray,
        correlations: np.ndarray,
    ) -> np.ndarray:
        """Iterate to find true risk parity weights."""
        weights = initial_weights.copy()
        n = len(weights)
        
        for _ in range(self.max_iterations):
            # Compute covariance matrix
            cov = np.outer(volatilities, volatilities) * correlations
            
            # Compute marginal contributions
            portfolio_var = weights @ cov @ weights
            marginal = (cov @ weights) / np.sqrt(portfolio_var + 1e-10)
            
            # Risk contributions
            risk_contrib = weights * marginal
            total_risk = np.sum(risk_contrib)
            
            if total_risk == 0:
                break
            
            # Target: equal risk contribution
            target_contrib = total_risk / n
            
            # Update weights proportionally
            adjustment = target_contrib / (risk_contrib + 1e-10)
            weights = weights * adjustment
            
            # Normalize
            weights = weights / np.sum(weights)
        
        return weights
    
    def _compute_portfolio_vol(
        self,
        weights: np.ndarray,
        volatilities: np.ndarray,
        correlations: Optional[np.ndarray],
    ) -> float:
        """Compute portfolio volatility."""
        if correlations is None:
            # Assume zero correlation
            return np.sqrt(np.sum((weights * volatilities) ** 2))
        
        cov = np.outer(volatilities, volatilities) * correlations
        var = weights @ cov @ weights
        return np.sqrt(var)


class DrawdownScaler:
    """Scale positions based on drawdown."""
    
    def __init__(
        self,
        max_drawdown_threshold: float = 0.10,  # More aggressive - start scaling at 10%
        scale_power: float = 2.0,  # Faster reduction curve
        min_scale: float = 0.15,  # Allow going lower
        recovery_speed: float = 0.05,  # Slower recovery for safety
    ):
        self.max_dd_threshold = max_drawdown_threshold
        self.scale_power = scale_power
        self.min_scale = min_scale
        self.recovery_speed = recovery_speed
        
        self.peak_value = None
        self.last_scale = 1.0
    
    def compute_scale(
        self,
        current_value: float,
        peak_value: Optional[float] = None,
    ) -> float:
        """
        Compute position scale based on drawdown.
        
        Reduces position size as drawdown increases.
        """
        if peak_value is None:
            if self.peak_value is None:
                self.peak_value = current_value
            else:
                self.peak_value = max(self.peak_value, current_value)
            peak_value = self.peak_value
        
        if peak_value <= 0:
            return 1.0
        
        # Current drawdown
        drawdown = 1 - current_value / peak_value
        
        if drawdown <= 0:
            # At or above peak - use full scale with gradual recovery
            target_scale = 1.0
        elif drawdown < self.max_dd_threshold:
            # Gradual reduction using power function
            dd_ratio = drawdown / self.max_dd_threshold
            reduction = dd_ratio ** self.scale_power
            target_scale = max(self.min_scale, 1 - reduction * (1 - self.min_scale))
        else:
            # At or beyond max threshold
            target_scale = self.min_scale
        
        # Smooth transition (don't snap instantly)
        if target_scale > self.last_scale:
            # Recovering - move slowly
            scale = self.last_scale + self.recovery_speed * (target_scale - self.last_scale)
        else:
            # Reducing - move faster
            scale = 0.5 * self.last_scale + 0.5 * target_scale
        
        self.last_scale = scale
        return scale


class CorrelationAdjuster:
    """Adjust weights based on correlation structure."""
    
    def __init__(
        self,
        lookback_days: int = 60,
        max_correlation_penalty: float = 0.5,
    ):
        self.lookback_days = lookback_days
        self.max_penalty = max_correlation_penalty
    
    def compute_correlation_matrix(
        self,
        returns_data: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, List[str]]:
        """Compute correlation matrix from returns."""
        tickers = list(returns_data.keys())
        n = len(tickers)
        
        if n < 2:
            return np.eye(1), tickers
        
        # Build returns matrix
        returns_matrix = []
        for ticker in tickers:
            rets = returns_data[ticker]
            if len(rets) >= self.lookback_days:
                returns_matrix.append(rets[-self.lookback_days:])
            else:
                returns_matrix.append(rets)
        
        # Pad to same length
        max_len = max(len(r) for r in returns_matrix)
        padded = []
        for r in returns_matrix:
            if len(r) < max_len:
                r = np.concatenate([np.zeros(max_len - len(r)), r])
            padded.append(r)
        
        returns_matrix = np.array(padded)
        
        # Compute correlation
        corr = np.corrcoef(returns_matrix)
        corr = np.nan_to_num(corr, nan=0.0)
        
        return corr, tickers
    
    def adjust_weights(
        self,
        weights: Dict[str, float],
        correlation_matrix: np.ndarray,
        tickers: List[str],
    ) -> Dict[str, float]:
        """
        Adjust weights to penalize highly correlated positions.
        """
        if len(tickers) < 2:
            return weights
        
        n = len(tickers)
        adjusted = weights.copy()
        
        for i, ticker_i in enumerate(tickers):
            if ticker_i not in weights:
                continue
            
            # Compute average correlation with other holdings
            total_corr = 0
            count = 0
            for j, ticker_j in enumerate(tickers):
                if i != j and ticker_j in weights and weights[ticker_j] > 0:
                    total_corr += abs(correlation_matrix[i, j]) * weights[ticker_j]
                    count += weights[ticker_j]
            
            if count > 0:
                avg_corr = total_corr / count
                # Penalty increases with correlation
                penalty = min(self.max_penalty, avg_corr * 0.5)
                adjusted[ticker_i] = weights[ticker_i] * (1 - penalty)
        
        # Renormalize
        total = sum(adjusted.values())
        if total > 0:
            original_total = sum(weights.values())
            scale = original_total / total
            adjusted = {k: v * scale for k, v in adjusted.items()}
        
        return adjusted


class DynamicPositionOptimizer:
    """
    Complete dynamic position optimization system.
    
    Combines Kelly, risk parity, drawdown scaling, and correlation adjustment.
    """
    
    def __init__(
        self,
        kelly_fraction: float = 0.25,
        target_volatility: float = 0.15,
        max_position: float = 0.10,
        max_leverage: float = 1.0,
    ):
        self.kelly_calc = KellyCalculator(
            kelly_fraction=kelly_fraction,
            max_position=max_position,
        )
        self.risk_parity = RiskParityOptimizer(
            target_volatility=target_volatility,
        )
        self.drawdown_scaler = DrawdownScaler()
        self.correlation_adj = CorrelationAdjuster()
        
        self.max_leverage = max_leverage
        self.max_position = max_position
        
        # Portfolio state
        self.portfolio_state: Optional[PortfolioState] = None
        self.position_history = deque(maxlen=252)
    
    def optimize_positions(
        self,
        signals: Dict[str, float],  # {ticker: signal}
        expected_returns: Dict[str, float],
        volatilities: Dict[str, float],
        returns_data: Optional[Dict[str, np.ndarray]] = None,
        current_portfolio: Optional[PortfolioState] = None,
        regime_scale: float = 1.0,
    ) -> Dict[str, PositionRecommendation]:
        """
        Compute optimal positions for all tickers.
        
        Args:
            signals: {ticker: signal} where signal in [-1, 1]
            expected_returns: {ticker: annualized_return}
            volatilities: {ticker: annualized_volatility}
            returns_data: {ticker: returns_array} for correlation
            current_portfolio: Current portfolio state
            regime_scale: Regime-based position scale (from meta-strategy)
            
        Returns:
            {ticker: PositionRecommendation}
        """
        if current_portfolio:
            self.portfolio_state = current_portfolio
        
        recommendations = {}
        tickers = list(signals.keys())
        
        if not tickers:
            return recommendations
        
        # Step 1: Compute Kelly weights
        kelly_weights = {}
        for ticker in tickers:
            signal = signals.get(ticker, 0)
            exp_ret = expected_returns.get(ticker, 0.10)
            vol = volatilities.get(ticker, 0.20)
            
            kelly_w = self.kelly_calc.compute_kelly_from_signal(
                signal=signal,
                expected_return=exp_ret,
                volatility=vol,
                confidence=abs(signal),
            )
            kelly_weights[ticker] = kelly_w
        
        # Step 2: Compute risk parity weights
        rp_weights = self.risk_parity.optimize(
            tickers=tickers,
            volatilities=volatilities,
        )
        
        # Step 3: Blend Kelly and Risk Parity
        blended_weights = {}
        kelly_blend = 0.6  # 60% Kelly, 40% Risk Parity
        for ticker in tickers:
            kelly_w = kelly_weights.get(ticker, 0)
            rp_w = rp_weights.get(ticker, 0)
            
            # Only blend for positive Kelly positions
            if kelly_w > 0:
                blended_weights[ticker] = kelly_blend * kelly_w + (1 - kelly_blend) * rp_w
            else:
                blended_weights[ticker] = 0.0
        
        # Step 4: Apply correlation adjustment
        if returns_data:
            corr_matrix, corr_tickers = self.correlation_adj.compute_correlation_matrix(returns_data)
            blended_weights = self.correlation_adj.adjust_weights(
                blended_weights, corr_matrix, corr_tickers
            )
        
        # Step 5: Apply drawdown scaling
        if current_portfolio:
            dd_scale = self.drawdown_scaler.compute_scale(
                current_value=current_portfolio.total_value,
            )
        else:
            dd_scale = 1.0
        
        # Step 6: Apply regime scale
        total_scale = dd_scale * regime_scale
        
        # Step 7: Build recommendations
        for ticker in tickers:
            raw_weight = blended_weights.get(ticker, 0)
            final_weight = raw_weight * total_scale
            
            # Apply position limits
            final_weight = np.clip(final_weight, 0, self.max_position)
            
            # Determine action
            current_weight = current_portfolio.weights.get(ticker, 0) if current_portfolio else 0
            weight_diff = final_weight - current_weight
            
            if abs(weight_diff) < 0.01:
                action = 'hold'
                urgency = 0.1
            elif weight_diff > 0.02:
                action = 'buy'
                urgency = min(1.0, weight_diff * 5)
            elif weight_diff < -0.02:
                action = 'sell'
                urgency = min(1.0, abs(weight_diff) * 5)
            else:
                action = 'rebalance'
                urgency = 0.3
            
            # Compute shares
            if current_portfolio:
                total_value = current_portfolio.total_value
                target_value = total_value * final_weight
                # Note: would need price data to compute shares
                final_shares = 0  # Placeholder
            else:
                final_shares = 0
            
            vol = volatilities.get(ticker, 0.20)
            exp_ret = expected_returns.get(ticker, 0.10)
            
            recommendations[ticker] = PositionRecommendation(
                ticker=ticker,
                raw_weight=raw_weight,
                kelly_weight=kelly_weights.get(ticker, 0),
                risk_parity_weight=rp_weights.get(ticker, 0),
                correlation_adjusted_weight=blended_weights.get(ticker, 0),
                final_weight=final_weight,
                final_shares=final_shares,
                max_weight=self.max_position,
                min_weight=0.0,
                expected_return=exp_ret,
                expected_volatility=vol,
                var_95=final_weight * vol * 1.65,  # Approximate 95% VaR
                action=action,
                urgency=urgency,
            )
        
        return recommendations
    
    def get_portfolio_targets(
        self,
        recommendations: Dict[str, PositionRecommendation],
        cash_buffer: float = 0.05,
    ) -> Dict[str, float]:
        """Get target portfolio weights."""
        # Sum weights
        total_weight = sum(r.final_weight for r in recommendations.values())
        
        # Ensure we don't exceed leverage limit
        if total_weight > self.max_leverage:
            scale = self.max_leverage / total_weight
            return {t: r.final_weight * scale for t, r in recommendations.items()}
        
        # Ensure cash buffer
        max_invested = 1 - cash_buffer
        if total_weight > max_invested:
            scale = max_invested / total_weight
            return {t: r.final_weight * scale for t, r in recommendations.items()}
        
        return {t: r.final_weight for t, r in recommendations.items()}
