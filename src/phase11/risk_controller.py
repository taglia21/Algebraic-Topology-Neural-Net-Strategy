"""
Risk Controller for Phase 11
=============================

Comprehensive risk management with drawdown protection.

Features:
- Dynamic position sizing based on volatility
- Drawdown-triggered de-risking
- Sector and factor concentration limits
- Liquidity-based position limits
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    """Configuration for risk management."""
    # Drawdown thresholds
    dd_level_1: float = 0.05   # 5% DD -> reduce 15%
    dd_level_2: float = 0.10   # 10% DD -> reduce 30%
    dd_level_3: float = 0.15   # 15% DD -> reduce 50%
    
    # Position limits
    max_position: float = 0.05       # 5% max per position
    max_sector: float = 0.35         # 35% max per sector
    max_factor_exposure: float = 0.40  # 40% max for any factor tilt
    
    # Volatility scaling
    target_vol: float = 0.20         # 20% target portfolio vol
    vol_scale_factor: float = 0.5    # How much to scale by vol ratio
    
    # Liquidity limits  
    max_adv_pct: float = 0.05        # Max 5% of avg daily volume
    
    # Correlation limits
    max_correlation: float = 0.85    # Max 85% correlation between top holdings


class Phase11RiskController:
    """
    Risk controller for Phase 11 portfolios.
    
    Key principles:
    1. Cut losses quickly (drawdown protection)
    2. Let winners run (momentum alignment)
    3. Never exceed position/sector limits
    """
    
    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()
        
        # Track state
        self.peak_equity = 1.0
        self.current_equity = 1.0
        self.current_drawdown = 0.0
        
    def adjust_for_risk(
        self,
        target_weights: Dict[str, float],
        price_data: Dict[str, pd.DataFrame],
        sector_map: Dict[str, str],
        current_equity: float = 1.0,
    ) -> Tuple[Dict[str, float], Dict[str, any]]:
        """
        Adjust portfolio weights for risk.
        
        Args:
            target_weights: Proposed portfolio weights
            price_data: Historical price data
            sector_map: Ticker to sector mapping
            current_equity: Current portfolio value
            
        Returns:
            Tuple of (adjusted_weights, risk_metrics)
        """
        # Update drawdown state
        self._update_drawdown(current_equity)
        
        # Step 1: Apply drawdown adjustments
        weights = self._apply_drawdown_scaling(target_weights)
        
        # Step 2: Apply position limits
        weights = self._apply_position_limits(weights)
        
        # Step 3: Apply sector limits
        weights = self._apply_sector_limits(weights, sector_map)
        
        # Step 4: Volatility scaling
        weights = self._apply_vol_scaling(weights, price_data)
        
        # Step 5: Normalize weights
        total = sum(weights.values())
        if total > 1.0:
            weights = {k: v / total for k, v in weights.items()}
        
        # Compute risk metrics
        risk_metrics = {
            'current_drawdown': self.current_drawdown,
            'risk_scale': self._get_risk_scale(),
            'n_positions': len(weights),
            'max_position': max(weights.values()) if weights else 0,
            'total_exposure': sum(weights.values()),
        }
        
        logger.info(f"Risk adjustment: DD={self.current_drawdown:.1%}, "
                   f"scale={risk_metrics['risk_scale']:.0%}, "
                   f"exposure={risk_metrics['total_exposure']:.1%}")
        
        return weights, risk_metrics
    
    def _update_drawdown(self, current_equity: float):
        """Update drawdown tracking."""
        self.current_equity = current_equity
        self.peak_equity = max(self.peak_equity, current_equity)
        self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
    
    def _get_risk_scale(self) -> float:
        """Get risk scaling factor based on drawdown."""
        if self.current_drawdown >= self.config.dd_level_3:
            return 0.50  # 50% of target
        elif self.current_drawdown >= self.config.dd_level_2:
            return 0.70  # 70% of target
        elif self.current_drawdown >= self.config.dd_level_1:
            return 0.85  # 85% of target
        else:
            return 1.00  # Full allocation
    
    def _apply_drawdown_scaling(
        self, 
        weights: Dict[str, float],
    ) -> Dict[str, float]:
        """Scale positions based on current drawdown."""
        scale = self._get_risk_scale()
        
        if scale < 1.0:
            # Scale all positions down
            weights = {k: v * scale for k, v in weights.items()}
            logger.info(f"Drawdown scaling: {scale:.0%} (DD={self.current_drawdown:.1%})")
        
        return weights
    
    def _apply_position_limits(
        self, 
        weights: Dict[str, float],
    ) -> Dict[str, float]:
        """Enforce maximum position size."""
        limited = {}
        excess = 0.0
        
        for ticker, weight in weights.items():
            if weight > self.config.max_position:
                excess += weight - self.config.max_position
                limited[ticker] = self.config.max_position
            else:
                limited[ticker] = weight
        
        # Redistribute excess proportionally
        if excess > 0:
            under_limit = {k: v for k, v in limited.items() 
                         if v < self.config.max_position}
            if under_limit:
                total_under = sum(under_limit.values())
                for ticker in under_limit:
                    limited[ticker] += excess * (limited[ticker] / total_under)
        
        return limited
    
    def _apply_sector_limits(
        self,
        weights: Dict[str, float],
        sector_map: Dict[str, str],
    ) -> Dict[str, float]:
        """Enforce sector concentration limits."""
        # Compute sector exposures
        sector_weights = {}
        for ticker, weight in weights.items():
            sector = sector_map.get(ticker, 'Unknown')
            sector_weights[sector] = sector_weights.get(sector, 0) + weight
        
        # Check for violations
        limited = weights.copy()
        for sector, total in sector_weights.items():
            if total > self.config.max_sector:
                # Scale down proportionally
                scale = self.config.max_sector / total
                for ticker, weight in weights.items():
                    if sector_map.get(ticker, 'Unknown') == sector:
                        limited[ticker] = weight * scale
                        
                logger.info(f"Sector limit: {sector} reduced from "
                           f"{total:.1%} to {self.config.max_sector:.1%}")
        
        return limited
    
    def _apply_vol_scaling(
        self,
        weights: Dict[str, float],
        price_data: Dict[str, pd.DataFrame],
    ) -> Dict[str, float]:
        """Apply volatility-based position scaling."""
        # Compute portfolio volatility (simplified)
        vols = {}
        for ticker in weights:
            if ticker in price_data:
                df = price_data[ticker]
                if len(df) > 20:
                    prices = df['Close'].values if 'Close' in df.columns else df['close'].values
                    returns = np.diff(np.log(prices))
                    vols[ticker] = np.std(returns) * np.sqrt(252)
        
        if not vols:
            return weights
        
        # Estimate portfolio vol (simplified)
        weighted_vol = sum(weights.get(t, 0) * v for t, v in vols.items())
        
        if weighted_vol > self.config.target_vol * 1.5:
            # Reduce exposure
            scale = (self.config.target_vol / weighted_vol) ** self.config.vol_scale_factor
            weights = {k: v * scale for k, v in weights.items()}
            logger.info(f"Vol scaling: portfolio vol {weighted_vol:.1%} -> scaled by {scale:.1%}")
        
        return weights
    
    def check_position_liquidity(
        self,
        ticker: str,
        weight: float,
        portfolio_value: float,
        avg_daily_volume: float,
        avg_price: float,
    ) -> float:
        """Check if position size is appropriate for liquidity."""
        if avg_daily_volume <= 0 or avg_price <= 0:
            return weight
        
        # Dollar value of position
        position_value = weight * portfolio_value
        
        # Avg daily dollar volume
        daily_dollar_volume = avg_daily_volume * avg_price
        
        # Max position based on liquidity
        max_position_value = daily_dollar_volume * self.config.max_adv_pct
        
        if position_value > max_position_value:
            # Scale down
            new_weight = max_position_value / portfolio_value
            logger.warning(f"Liquidity limit: {ticker} reduced from "
                          f"{weight:.1%} to {new_weight:.1%}")
            return new_weight
        
        return weight
    
    def reset_peak(self):
        """Reset peak equity (e.g., after new capital injection)."""
        self.peak_equity = self.current_equity
        self.current_drawdown = 0.0


def compute_portfolio_stats(
    returns: pd.Series,
    risk_free_rate: float = 0.04,
) -> Dict[str, float]:
    """Compute comprehensive portfolio statistics."""
    if len(returns) < 20:
        return {}
    
    # Basic stats
    total_return = (1 + returns).prod() - 1
    n_years = len(returns) / 252
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    
    # Volatility
    ann_vol = returns.std() * np.sqrt(252)
    
    # Sharpe
    excess_return = cagr - risk_free_rate
    sharpe = excess_return / ann_vol if ann_vol > 0 else 0
    
    # Drawdown
    equity = (1 + returns).cumprod()
    rolling_max = equity.expanding().max()
    drawdown = (equity - rolling_max) / rolling_max
    max_dd = drawdown.min()
    
    # Sortino
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else ann_vol
    sortino = excess_return / downside_std if downside_std > 0 else 0
    
    return {
        'cagr': cagr,
        'volatility': ann_vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_dd,
        'total_return': total_return,
    }
