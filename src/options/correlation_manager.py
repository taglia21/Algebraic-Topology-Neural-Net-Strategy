"""
Portfolio Correlation and Concentration Risk Management
=======================================================

Millennium Capital Management-inspired correlation tracking system.
Prevents concentration risk through real-time monitoring of:
- Cross-position correlations
- Sector exposures
- Portfolio Greeks
- Value-at-Risk (VaR)

Features:
- Dynamic correlation matrix calculation
- Portfolio-level Greeks aggregation  
- Concentration risk alerts
- Hedge recommendations
- Monte Carlo VaR calculation
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging
from datetime import datetime, timedelta
import yfinance as yf
from scipy import stats


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class Position:
    """Trading position."""
    symbol: str
    quantity: int  # Number of contracts
    entry_price: float
    current_price: float
    strategy_type: str  # "credit_spread", "iron_condor", etc.
    
    # Greeks (per contract)
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    
    # Risk metrics
    notional_value: float = 0.0
    unrealized_pnl: float = 0.0
    sector: str = "unknown"
    
    def __post_init__(self):
        """Calculate derived fields."""
        if self.notional_value == 0.0:
            self.notional_value = abs(self.quantity * self.current_price * 100)
        if self.unrealized_pnl == 0.0:
            self.unrealized_pnl = (self.current_price - self.entry_price) * self.quantity * 100


@dataclass
class PortfolioGreeks:
    """Aggregated portfolio Greeks."""
    total_delta: float
    total_gamma: float
    total_theta: float
    total_vega: float
    delta_by_symbol: Dict[str, float]
    largest_delta_exposure: Tuple[str, float]  # (symbol, delta)
    net_directional_bias: str  # "bullish", "bearish", or "neutral"


@dataclass
class ConcentrationAlert:
    """Concentration risk alert."""
    alert_type: str  # "correlation", "sector", "single_name", "strategy"
    severity: str  # "low", "medium", "high", "critical"
    message: str
    affected_symbols: List[str]
    metric_value: float
    threshold: float
    timestamp: datetime


@dataclass
class HedgeRecommendation:
    """Hedge recommendation."""
    action: str  # "buy_puts", "sell_calls", "reduce_position", etc.
    symbol: str
    reasoning: str
    estimated_cost: float
    hedge_ratio: float
    priority: str  # "low", "medium", "high"


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================================
# CORRELATION MANAGER
# ============================================================================

class CorrelationManager:
    """
    Monitor cross-position correlations and portfolio Greeks.
    
    Key functions:
    1. Build correlation matrix between positions
    2. Calculate portfolio-level Greeks
    3. Detect concentration risk
    4. Generate hedge recommendations
    5. Calculate Value-at-Risk (VaR)
    """
    
    # Risk thresholds (Millennium-style)
    MAX_CORRELATION = 0.70  # Max correlation between any 2 positions
    MAX_SECTOR_EXPOSURE_PCT = 0.30  # Max 30% in any sector
    MAX_SINGLE_NAME_PCT = 0.10  # Max 10% in single stock
    MAX_STRATEGY_OVERLAP_PCT = 0.50  # Max 50% in one strategy type
    
    def __init__(self, lookback_days: int = 60):
        """
        Initialize correlation manager.
        
        Args:
            lookback_days: Days of price history for correlation (default 60)
        """
        self.lookback_days = lookback_days
        self.logger = logging.getLogger(__name__)
        self._price_cache: Dict[str, pd.Series] = {}
        self._last_cache_update = datetime.min
        
        self.logger.info(f"Initialized CorrelationManager (lookback={lookback_days})")
    
    async def build_correlation_matrix(
        self, 
        positions: List[Position]
    ) -> np.ndarray:
        """
        Build correlation matrix between positions.
        
        Args:
            positions: List of current positions
        
        Returns:
            Correlation matrix (N x N) where N = number of positions
        """
        if len(positions) == 0:
            return np.array([])
        
        if len(positions) == 1:
            return np.array([[1.0]])
        
        try:
            # Get unique symbols
            symbols = list(set([pos.symbol for pos in positions]))
            
            # Fetch price data for all symbols
            await self._update_price_cache(symbols)
            
            # Build returns matrix
            returns_list = []
            valid_symbols = []
            
            for symbol in symbols:
                if symbol in self._price_cache:
                    prices = self._price_cache[symbol]
                    returns = prices.pct_change().dropna()
                    if len(returns) > 10:  # Minimum data requirement
                        returns_list.append(returns)
                        valid_symbols.append(symbol)
            
            if len(returns_list) < 2:
                self.logger.warning("Insufficient data for correlation matrix")
                return np.eye(len(positions))
            
            # Align all returns series to common dates
            returns_df = pd.DataFrame({
                sym: ret for sym, ret in zip(valid_symbols, returns_list)
            }).dropna()
            
            # Calculate correlation matrix
            corr_matrix = returns_df.corr().values
            
            self.logger.info(
                f"Built correlation matrix ({corr_matrix.shape[0]}x{corr_matrix.shape[1]})"
            )
            
            return corr_matrix
        
        except Exception as e:
            self.logger.error(f"Failed to build correlation matrix: {e}", exc_info=True)
            # Return identity matrix as fallback
            return np.eye(len(positions))
    
    async def calculate_portfolio_greeks(
        self, 
        positions: List[Position]
    ) -> PortfolioGreeks:
        """
        Calculate aggregated portfolio Greeks.
        
        Args:
            positions: List of current positions
        
        Returns:
            PortfolioGreeks with total and per-symbol breakdowns
        """
        if len(positions) == 0:
            return PortfolioGreeks(
                total_delta=0.0,
                total_gamma=0.0,
                total_theta=0.0,
                total_vega=0.0,
                delta_by_symbol={},
                largest_delta_exposure=("", 0.0),
                net_directional_bias="neutral",
            )
        
        # Aggregate Greeks
        total_delta = 0.0
        total_gamma = 0.0
        total_theta = 0.0
        total_vega = 0.0
        delta_by_symbol: Dict[str, float] = {}
        
        for pos in positions:
            # Multiply by quantity to get position-level Greeks
            pos_delta = pos.delta * pos.quantity
            pos_gamma = pos.gamma * pos.quantity
            pos_theta = pos.theta * pos.quantity
            pos_vega = pos.vega * pos.quantity
            
            total_delta += pos_delta
            total_gamma += pos_gamma
            total_theta += pos_theta
            total_vega += pos_vega
            
            # Track delta by symbol
            if pos.symbol in delta_by_symbol:
                delta_by_symbol[pos.symbol] += pos_delta
            else:
                delta_by_symbol[pos.symbol] = pos_delta
        
        # Find largest delta exposure
        if delta_by_symbol:
            largest_symbol = max(delta_by_symbol.items(), key=lambda x: abs(x[1]))
            largest_delta_exposure = largest_symbol
        else:
            largest_delta_exposure = ("", 0.0)
        
        # Determine directional bias
        if total_delta > 10.0:
            bias = "bullish"
        elif total_delta < -10.0:
            bias = "bearish"
        else:
            bias = "neutral"
        
        greeks = PortfolioGreeks(
            total_delta=total_delta,
            total_gamma=total_gamma,
            total_theta=total_theta,
            total_vega=total_vega,
            delta_by_symbol=delta_by_symbol,
            largest_delta_exposure=largest_delta_exposure,
            net_directional_bias=bias,
        )
        
        self.logger.info(
            f"Portfolio Greeks: Δ={total_delta:.2f}, Γ={total_gamma:.2f}, "
            f"Θ={total_theta:.2f}, v={total_vega:.2f}"
        )
        
        return greeks
    
    def detect_concentration_risk(
        self, 
        positions: List[Position],
        portfolio_value: float,
        correlation_matrix: Optional[np.ndarray] = None,
    ) -> List[ConcentrationAlert]:
        """
        Detect concentration risks in portfolio.
        
        Checks:
        1. Correlation between positions
        2. Sector exposure
        3. Single-name exposure
        4. Strategy overlap
        
        Args:
            positions: Current positions
            portfolio_value: Total portfolio value
            correlation_matrix: Optional precomputed correlation matrix
        
        Returns:
            List of concentration alerts
        """
        alerts: List[ConcentrationAlert] = []
        
        if len(positions) == 0:
            return alerts
        
        # 1. Check correlation risk
        if correlation_matrix is not None and correlation_matrix.shape[0] > 1:
            # Get max off-diagonal correlation
            np.fill_diagonal(correlation_matrix, 0)  # Ignore self-correlation
            max_corr = np.max(np.abs(correlation_matrix))
            
            if max_corr > self.MAX_CORRELATION:
                # Find which positions are highly correlated
                max_idx = np.unravel_index(
                    np.argmax(np.abs(correlation_matrix)), 
                    correlation_matrix.shape
                )
                symbols = [pos.symbol for pos in positions]
                symbol1 = symbols[max_idx[0]] if max_idx[0] < len(symbols) else "unknown"
                symbol2 = symbols[max_idx[1]] if max_idx[1] < len(symbols) else "unknown"
                
                severity = self._get_severity(max_corr, self.MAX_CORRELATION)
                
                alerts.append(ConcentrationAlert(
                    alert_type="correlation",
                    severity=severity.value,
                    message=f"High correlation ({max_corr:.2f}) between {symbol1} and {symbol2}",
                    affected_symbols=[symbol1, symbol2],
                    metric_value=max_corr,
                    threshold=self.MAX_CORRELATION,
                    timestamp=datetime.now(),
                ))
        
        # 2. Check sector exposure
        sector_exposure: Dict[str, float] = {}
        for pos in positions:
            sector = pos.sector
            exposure = abs(pos.notional_value)
            sector_exposure[sector] = sector_exposure.get(sector, 0.0) + exposure
        
        for sector, exposure in sector_exposure.items():
            exposure_pct = exposure / portfolio_value
            if exposure_pct > self.MAX_SECTOR_EXPOSURE_PCT:
                severity = self._get_severity(exposure_pct, self.MAX_SECTOR_EXPOSURE_PCT)
                
                affected = [pos.symbol for pos in positions if pos.sector == sector]
                
                alerts.append(ConcentrationAlert(
                    alert_type="sector",
                    severity=severity.value,
                    message=f"Sector '{sector}' exposure at {exposure_pct:.1%} (max {self.MAX_SECTOR_EXPOSURE_PCT:.1%})",
                    affected_symbols=affected,
                    metric_value=exposure_pct,
                    threshold=self.MAX_SECTOR_EXPOSURE_PCT,
                    timestamp=datetime.now(),
                ))
        
        # 3. Check single-name exposure
        symbol_exposure: Dict[str, float] = {}
        for pos in positions:
            exposure = abs(pos.notional_value)
            symbol_exposure[pos.symbol] = symbol_exposure.get(pos.symbol, 0.0) + exposure
        
        for symbol, exposure in symbol_exposure.items():
            exposure_pct = exposure / portfolio_value
            if exposure_pct > self.MAX_SINGLE_NAME_PCT:
                severity = self._get_severity(exposure_pct, self.MAX_SINGLE_NAME_PCT)
                
                alerts.append(ConcentrationAlert(
                    alert_type="single_name",
                    severity=severity.value,
                    message=f"Symbol '{symbol}' exposure at {exposure_pct:.1%} (max {self.MAX_SINGLE_NAME_PCT:.1%})",
                    affected_symbols=[symbol],
                    metric_value=exposure_pct,
                    threshold=self.MAX_SINGLE_NAME_PCT,
                    timestamp=datetime.now(),
                ))
        
        # 4. Check strategy overlap
        strategy_exposure: Dict[str, float] = {}
        for pos in positions:
            exposure = abs(pos.notional_value)
            strategy_exposure[pos.strategy_type] = strategy_exposure.get(pos.strategy_type, 0.0) + exposure
        
        for strategy, exposure in strategy_exposure.items():
            exposure_pct = exposure / portfolio_value
            if exposure_pct > self.MAX_STRATEGY_OVERLAP_PCT:
                severity = self._get_severity(exposure_pct, self.MAX_STRATEGY_OVERLAP_PCT)
                
                affected = [pos.symbol for pos in positions if pos.strategy_type == strategy]
                
                alerts.append(ConcentrationAlert(
                    alert_type="strategy",
                    severity=severity.value,
                    message=f"Strategy '{strategy}' exposure at {exposure_pct:.1%} (max {self.MAX_STRATEGY_OVERLAP_PCT:.1%})",
                    affected_symbols=affected,
                    metric_value=exposure_pct,
                    threshold=self.MAX_STRATEGY_OVERLAP_PCT,
                    timestamp=datetime.now(),
                ))
        
        if alerts:
            self.logger.warning(f"Detected {len(alerts)} concentration alerts")
        else:
            self.logger.info("✓ No concentration risks detected")
        
        return alerts
    
    async def get_hedge_recommendations(
        self,
        positions: List[Position],
        portfolio_greeks: PortfolioGreeks,
        max_delta: float = 50.0,
    ) -> List[HedgeRecommendation]:
        """
        Generate hedge recommendations based on portfolio Greeks.
        
        Args:
            positions: Current positions
            portfolio_greeks: Aggregated Greeks
            max_delta: Maximum acceptable portfolio delta
        
        Returns:
            List of hedge recommendations
        """
        recommendations: List[HedgeRecommendation] = []
        
        # Check if delta hedging needed
        if abs(portfolio_greeks.total_delta) > max_delta:
            # Determine hedge direction
            if portfolio_greeks.total_delta > 0:
                # Too bullish, need to sell calls or buy puts
                action = "buy_puts"
                hedge_delta = -portfolio_greeks.total_delta
            else:
                # Too bearish, need to buy calls or sell puts
                action = "buy_calls"
                hedge_delta = -portfolio_greeks.total_delta
            
            # Find symbol with largest exposure
            largest_symbol, _ = portfolio_greeks.largest_delta_exposure
            
            # Estimate hedge cost (simplified)
            hedge_ratio = abs(hedge_delta / 100.0)  # Contracts needed
            estimated_cost = hedge_ratio * 1.0 * 100  # Assume $1.00 per contract
            
            # Determine priority
            excess_delta = abs(portfolio_greeks.total_delta) - max_delta
            if excess_delta > max_delta * 0.5:
                priority = "high"
            elif excess_delta > max_delta * 0.25:
                priority = "medium"
            else:
                priority = "low"
            
            recommendations.append(HedgeRecommendation(
                action=action,
                symbol=largest_symbol if largest_symbol else "SPY",
                reasoning=f"Portfolio delta {portfolio_greeks.total_delta:.2f} exceeds max {max_delta}",
                estimated_cost=estimated_cost,
                hedge_ratio=hedge_ratio,
                priority=priority,
            ))
        
        # Check for excessive vega exposure
        if abs(portfolio_greeks.total_vega) > 100.0:
            recommendations.append(HedgeRecommendation(
                action="reduce_vega",
                symbol="VXX",  # Use VIX products for vega hedge
                reasoning=f"High vega exposure: {portfolio_greeks.total_vega:.2f}",
                estimated_cost=0.0,
                hedge_ratio=abs(portfolio_greeks.total_vega / 100.0),
                priority="medium",
            ))
        
        if recommendations:
            self.logger.info(f"Generated {len(recommendations)} hedge recommendations")
        
        return recommendations
    
    def calculate_portfolio_var(
        self,
        positions: List[Position],
        correlation_matrix: Optional[np.ndarray] = None,
        confidence: float = 0.95,
        horizon_days: int = 1,
        simulations: int = 10000,
    ) -> float:
        """
        Calculate portfolio Value-at-Risk (VaR) using Monte Carlo.
        
        Args:
            positions: Current positions
            correlation_matrix: Position correlation matrix
            confidence: Confidence level (default 0.95 for 95% VaR)
            horizon_days: Time horizon in days
            simulations: Number of Monte Carlo simulations
        
        Returns:
            VaR in dollars (positive number = potential loss)
        """
        if len(positions) == 0:
            return 0.0
        
        try:
            # Extract position values
            position_values = np.array([pos.notional_value for pos in positions])
            
            # Estimate volatilities (simplified: use 20% annualized vol)
            annual_vols = np.ones(len(positions)) * 0.20
            daily_vols = annual_vols / np.sqrt(252)
            horizon_vols = daily_vols * np.sqrt(horizon_days)
            
            # Use identity matrix if no correlation provided
            if correlation_matrix is None or correlation_matrix.shape[0] != len(positions):
                correlation_matrix = np.eye(len(positions))
            
            # Build covariance matrix
            cov_matrix = np.outer(horizon_vols, horizon_vols) * correlation_matrix
            
            # Monte Carlo simulation
            np.random.seed(42)  # For reproducibility
            returns = np.random.multivariate_normal(
                mean=np.zeros(len(positions)),
                cov=cov_matrix,
                size=simulations,
            )
            
            # Calculate portfolio P&L for each simulation
            portfolio_pnl = returns @ position_values
            
            # VaR is the percentile loss
            var_percentile = 1 - confidence
            var = -np.percentile(portfolio_pnl, var_percentile * 100)
            
            self.logger.info(
                f"VaR (${confidence:.0%}, {horizon_days}d): ${var:,.0f}"
            )
            
            return var
        
        except Exception as e:
            self.logger.error(f"VaR calculation failed: {e}", exc_info=True)
            return 0.0
    
    async def _update_price_cache(self, symbols: List[str]) -> None:
        """
        Update price cache for symbols.
        
        Args:
            symbols: List of symbols to fetch
        """
        # Only update if cache is stale (more than 1 hour old)
        if datetime.now() - self._last_cache_update < timedelta(hours=1):
            return
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days + 10)
            
            for symbol in symbols:
                if symbol not in self._price_cache:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(start=start_date, end=end_date)
                    if len(data) > 0:
                        self._price_cache[symbol] = data['Close']
            
            self._last_cache_update = datetime.now()
            self.logger.debug(f"Updated price cache for {len(symbols)} symbols")
        
        except Exception as e:
            self.logger.error(f"Failed to update price cache: {e}")
    
    def _get_severity(self, value: float, threshold: float) -> AlertSeverity:
        """
        Determine alert severity based on threshold breach.
        
        Args:
            value: Current value
            threshold: Alert threshold
        
        Returns:
            AlertSeverity level
        """
        ratio = value / threshold
        
        if ratio < 1.0:
            return AlertSeverity.LOW
        elif ratio < 1.2:
            return AlertSeverity.MEDIUM
        elif ratio < 1.5:
            return AlertSeverity.HIGH
        else:
            return AlertSeverity.CRITICAL


# ============================================================================
# TESTING HELPER
# ============================================================================

async def test_correlation_manager():
    """Test the correlation manager."""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    manager = CorrelationManager()
    
    # Create sample positions
    positions = [
        Position(
            symbol="AAPL",
            quantity=10,
            entry_price=2.50,
            current_price=3.00,
            strategy_type="credit_spread",
            delta=0.30,
            gamma=0.05,
            theta=-0.10,
            vega=0.15,
            sector="Technology",
        ),
        Position(
            symbol="MSFT",
            quantity=8,
            entry_price=1.80,
            current_price=2.20,
            strategy_type="credit_spread",
            delta=0.25,
            gamma=0.04,
            theta=-0.08,
            vega=0.12,
            sector="Technology",
        ),
        Position(
            symbol="SPY",
            quantity=-5,
            entry_price=5.00,
            current_price=4.50,
            strategy_type="iron_condor",
            delta=-0.10,
            gamma=0.02,
            theta=-0.15,
            vega=0.20,
            sector="Index",
        ),
    ]
    
    print("\n" + "="*60)
    print("TESTING CORRELATION MANAGER")
    print("="*60)
    
    # Test 1: Build correlation matrix
    print("\n1. Building correlation matrix...")
    corr_matrix = await manager.build_correlation_matrix(positions)
    print(f"Correlation matrix shape: {corr_matrix.shape}")
    print(f"Correlation matrix:\n{corr_matrix}")
    
    # Test 2: Calculate portfolio Greeks
    print("\n2. Calculating portfolio Greeks...")
    greeks = await manager.calculate_portfolio_greeks(positions)
    print(f"Total Delta: {greeks.total_delta:.2f}")
    print(f"Total Gamma: {greeks.total_gamma:.2f}")
    print(f"Total Theta: {greeks.total_theta:.2f}")
    print(f"Total Vega: {greeks.total_vega:.2f}")
    print(f"Directional Bias: {greeks.net_directional_bias}")
    
    # Test 3: Detect concentration risk
    print("\n3. Detecting concentration risk...")
    alerts = manager.detect_concentration_risk(
        positions=positions,
        portfolio_value=50000.0,
        correlation_matrix=corr_matrix,
    )
    print(f"Found {len(alerts)} concentration alerts")
    for alert in alerts:
        print(f"  - {alert.severity.upper()}: {alert.message}")
    
    # Test 4: Get hedge recommendations
    print("\n4. Getting hedge recommendations...")
    hedges = await manager.get_hedge_recommendations(
        positions=positions,
        portfolio_greeks=greeks,
        max_delta=3.0,  # Low threshold to trigger hedge
    )
    print(f"Generated {len(hedges)} hedge recommendations")
    for hedge in hedges:
        print(f"  - {hedge.priority.upper()}: {hedge.action} {hedge.symbol}")
        print(f"    Reasoning: {hedge.reasoning}")
    
    # Test 5: Calculate VaR
    print("\n5. Calculating portfolio VaR...")
    var = manager.calculate_portfolio_var(
        positions=positions,
        correlation_matrix=corr_matrix,
        confidence=0.95,
    )
    print(f"95% 1-day VaR: ${var:,.0f}")
    
    # Validate
    assert corr_matrix.shape[0] == len(set([p.symbol for p in positions]))
    assert isinstance(greeks.total_delta, float)
    assert isinstance(alerts, list)
    assert isinstance(hedges, list)
    assert var >= 0.0
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_correlation_manager())
