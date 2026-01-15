"""
Enhanced Risk Manager - Production-Grade Risk Management

Implements research-backed risk controls:
1. Dynamic Drawdown-Based Position Sizing (Stanford "cushion" approach)
2. Volatility-Adjusted Allocation (constant portfolio vol targeting)
3. Transaction Cost Optimization (turnover penalty)
4. Multi-Level Stop-Loss (position, trailing, circuit breaker)

Target improvements:
- Reduce max drawdown from -29.7% to <-15%
- Improve Sharpe from 0.56 to >1.2
- Maintain CAGR >15%
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    """Configuration for risk management."""
    # Drawdown-based sizing (Phase 8.1: less aggressive)
    max_allowed_drawdown: float = 0.25  # 25% max DD before full reduction (was 0.20)
    min_position_scale: float = 0.80    # 80% minimum (was 0.25) - less aggressive
    dd_scaling_power: float = 1.5       # Softer scaling curve (was implicit 2.0)
    fast_recovery_threshold: float = 0.05  # Full allocation when DD <5%
    
    # Volatility targeting
    target_annual_vol: float = 0.15     # 15% annualized volatility target (was 0.12)
    vol_lookback_days: int = 20         # Rolling window for realized vol
    vol_rebalance_threshold: float = 0.25  # Rebalance when vol deviates >25% (was 0.20)
    
    # Transaction costs
    cost_per_trade_bps: float = 10      # 10 basis points per trade
    min_alpha_to_cost_ratio: float = 1.5  # Trade when alpha > 1.5x cost (was 2.0)
    max_turnover_per_rebal: float = 0.50  # 50% max turnover per rebalance (was 0.30)
    
    # Stop-loss levels (Phase 8.1: slightly relaxed)
    position_stop_loss: float = 0.10    # -10% position stop (was -8%)
    trailing_stop_pct: float = 0.12     # 12% trailing stop from peak (was 10%)
    circuit_breaker_dd: float = 0.22    # -22% portfolio circuit breaker (was -15%)
    
    # Position limits
    max_position_weight: float = 0.10   # 10% max single position
    max_sector_weight: float = 0.25     # 25% max sector concentration (was 0.30)
    
    # Regime-aware leverage (Phase 8.1 new)
    base_leverage_bull: float = 1.2     # Leverage in bull regime
    base_leverage_neutral: float = 1.0  # Leverage in neutral regime
    base_leverage_bear: float = 0.75    # Leverage in bear regime
    vix_circuit_breaker: float = 40.0   # Reduce to 0.5x if VIX > 40


@dataclass
class PositionState:
    """Track state for a single position."""
    ticker: str
    entry_price: float
    current_price: float
    peak_price: float
    weight: float
    sector: str
    
    @property
    def pnl_pct(self) -> float:
        """Current P&L percentage."""
        if self.entry_price <= 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price
    
    @property
    def drawdown_from_peak(self) -> float:
        """Drawdown from position peak."""
        if self.peak_price <= 0:
            return 0.0
        return (self.current_price - self.peak_price) / self.peak_price


class EnhancedRiskManager:
    """
    Production-grade risk management system.
    
    Key features:
    - Dynamically adjusts position sizes based on current drawdown
    - Targets constant portfolio volatility
    - Optimizes for transaction costs
    - Implements multi-level stop-loss protection
    """
    
    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()
        
        # Portfolio state tracking
        self.portfolio_value = 100.0
        self.peak_value = 100.0
        self.positions: Dict[str, PositionState] = {}
        
        # Historical tracking
        self.equity_history: List[float] = [100.0]
        self.returns_history: List[float] = []
        self.drawdown_history: List[float] = [0.0]
        self.vol_history: List[float] = []
        
        # Metrics tracking
        self.total_turnover = 0.0
        self.total_costs = 0.0
        self.stop_loss_triggers = 0
        self.circuit_breaker_triggers = 0
        self.trailing_stop_triggers = 0
        
        logger.info(f"Initialized EnhancedRiskManager: target_vol={self.config.target_annual_vol:.0%}, max_dd={self.config.max_allowed_drawdown:.0%}")
    
    def current_drawdown(self) -> float:
        """Calculate current portfolio drawdown from peak."""
        if self.peak_value <= 0:
            return 0.0
        return (self.portfolio_value - self.peak_value) / self.peak_value
    
    def realized_volatility(self, window: Optional[int] = None) -> float:
        """Calculate realized annualized volatility."""
        window = window or self.config.vol_lookback_days
        if len(self.returns_history) < window:
            return self.config.target_annual_vol  # Default to target
        
        recent_returns = self.returns_history[-window:]
        daily_vol = np.std(recent_returns)
        annual_vol = daily_vol * np.sqrt(252)
        return annual_vol
    
    def drawdown_position_scale(self) -> float:
        """
        Calculate position size scaling based on current drawdown.
        
        Phase 8.1 Enhanced:
        - Fast recovery: full size when DD < 5% (fast_recovery_threshold)
        - Softer curve: exponent = dd_scaling_power (1.5 vs 2.0)
        - Higher floor: min_position_scale = 0.80 (was 0.25)
        
        Formula: scale = min_scale + (1 - min_scale) * (1 - |dd|/max_dd)^power
        """
        dd = abs(self.current_drawdown())
        max_dd = self.config.max_allowed_drawdown
        
        # Fast recovery - full allocation when drawdown is minimal
        if dd < self.config.fast_recovery_threshold:
            return 1.0
        
        if dd >= max_dd:
            return self.config.min_position_scale
        
        # Softer curve with higher floor - less aggressive scaling
        min_scale = self.config.min_position_scale
        power = self.config.dd_scaling_power
        
        # Linear interpolation base with power curve
        dd_ratio = (dd - self.config.fast_recovery_threshold) / (max_dd - self.config.fast_recovery_threshold)
        scale = min_scale + (1 - min_scale) * ((1 - dd_ratio) ** power)
        
        return max(min_scale, scale)
    
    def volatility_scale(self) -> float:
        """
        Calculate position scaling to target constant portfolio volatility.
        
        Formula: scale = target_vol / realized_vol
        Bounded between 0.5 and 1.5 to prevent extreme adjustments.
        """
        realized_vol = self.realized_volatility()
        if realized_vol <= 0:
            return 1.0
        
        target = self.config.target_annual_vol
        scale = target / realized_vol
        
        # Bound scaling to prevent extreme adjustments
        return np.clip(scale, 0.5, 1.5)
    
    def combined_position_scale(self) -> float:
        """Combined scaling from drawdown and volatility."""
        dd_scale = self.drawdown_position_scale()
        vol_scale = self.volatility_scale()
        
        # Use geometric mean for balanced effect
        combined = np.sqrt(dd_scale * vol_scale)
        
        logger.debug(f"Position scale: DD={dd_scale:.2f}, Vol={vol_scale:.2f}, Combined={combined:.2f}")
        return combined
    
    def regime_leverage_scale(self, regime_state: Optional[Dict] = None) -> float:
        """
        Calculate leverage adjustment based on market regime.
        
        Phase 8.1: Regime-adaptive leverage
        - Bull market: 1.2x leverage (base_leverage_bull)
        - Neutral: 1.0x (base_leverage_neutral)
        - Bear market: 0.75x (base_leverage_bear)
        
        Args:
            regime_state: Dict with 'regime' key ('bull', 'neutral', 'bear')
                         and optional 'probabilities' (p_bull, p_neutral, p_bear)
        
        Returns:
            Leverage scale factor (0.75 to 1.2)
        """
        if regime_state is None:
            return 1.0
        
        regime = regime_state.get('regime', 'neutral')
        
        # Use regime probabilities for smooth transitions if available
        if 'probabilities' in regime_state:
            probs = regime_state['probabilities']
            p_bull = probs.get('bull', 0.0)
            p_neutral = probs.get('neutral', 1.0)
            p_bear = probs.get('bear', 0.0)
            
            # Weighted average of regime leverages
            leverage = (
                p_bull * self.config.base_leverage_bull +
                p_neutral * self.config.base_leverage_neutral +
                p_bear * self.config.base_leverage_bear
            )
            return leverage
        
        # Discrete regime mapping
        if regime == 'bull':
            return self.config.base_leverage_bull
        elif regime == 'bear':
            return self.config.base_leverage_bear
        else:
            return self.config.base_leverage_neutral
    
    def vix_circuit_breaker_active(self, vix_level: Optional[float] = None) -> bool:
        """
        Check if VIX circuit breaker is active.
        
        Phase 8.1: VIX-based risk reduction
        When VIX > vix_circuit_breaker (40), reduce exposure dramatically.
        """
        if vix_level is None:
            return False
        return vix_level >= self.config.vix_circuit_breaker
    
    def full_position_scale(
        self, 
        regime_state: Optional[Dict] = None,
        vix_level: Optional[float] = None
    ) -> float:
        """
        Calculate full position scale with all Phase 8.1 factors.
        
        Combines:
        - Drawdown scaling (soft curve, high floor)
        - Volatility targeting
        - Regime-aware leverage
        - VIX circuit breaker
        """
        # Start with DD and vol scaling
        base_scale = self.combined_position_scale()
        
        # Apply regime leverage
        regime_scale = self.regime_leverage_scale(regime_state)
        
        # VIX circuit breaker
        if self.vix_circuit_breaker_active(vix_level):
            logger.warning(f"VIX circuit breaker active: VIX={vix_level:.1f}")
            return 0.3  # Minimal exposure during extreme fear
        
        final_scale = base_scale * regime_scale
        
        logger.debug(f"Full scale: base={base_scale:.2f}, regime={regime_scale:.2f}, final={final_scale:.2f}")
        return np.clip(final_scale, 0.1, 1.5)

    def calculate_transaction_cost(
        self,
        old_weights: Dict[str, float],
        new_weights: Dict[str, float],
    ) -> float:
        """Calculate transaction cost for portfolio rebalance."""
        turnover = 0.0
        
        all_tickers = set(old_weights.keys()) | set(new_weights.keys())
        for ticker in all_tickers:
            old_w = old_weights.get(ticker, 0.0)
            new_w = new_weights.get(ticker, 0.0)
            turnover += abs(new_w - old_w)
        
        # One-way turnover (divide by 2 for round-trip)
        turnover = turnover / 2
        cost = turnover * (self.config.cost_per_trade_bps / 10000)
        
        return cost
    
    def optimize_for_turnover(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        expected_alphas: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Optimize target weights to balance alpha vs transaction costs.
        
        Only trades when expected alpha > min_alpha_to_cost_ratio * cost.
        Limits maximum turnover per rebalance.
        """
        optimized = current_weights.copy()
        
        # Calculate which trades are worth making
        all_tickers = set(current_weights.keys()) | set(target_weights.keys())
        trades = []
        
        for ticker in all_tickers:
            curr_w = current_weights.get(ticker, 0.0)
            tgt_w = target_weights.get(ticker, 0.0)
            delta = tgt_w - curr_w
            
            if abs(delta) < 0.005:  # Skip tiny trades
                continue
            
            # Estimate trade cost
            trade_cost = abs(delta) * (self.config.cost_per_trade_bps / 10000)
            
            # Get expected alpha (default to signal strength proxy)
            alpha = expected_alphas.get(ticker, 0.0) if expected_alphas else abs(delta)
            
            # Calculate cost-adjusted benefit
            benefit = alpha - trade_cost
            
            trades.append({
                'ticker': ticker,
                'delta': delta,
                'target': tgt_w,
                'benefit': benefit,
                'cost': trade_cost,
            })
        
        # Sort by benefit and apply turnover constraint
        trades.sort(key=lambda x: x['benefit'], reverse=True)
        
        remaining_turnover = self.config.max_turnover_per_rebal
        for trade in trades:
            if trade['benefit'] < trade['cost'] * self.config.min_alpha_to_cost_ratio:
                # Skip low benefit trades
                continue
            
            delta = trade['delta']
            if abs(delta) > remaining_turnover:
                # Partial trade to stay within turnover budget
                delta = remaining_turnover * np.sign(delta)
            
            current = optimized.get(trade['ticker'], 0.0)
            optimized[trade['ticker']] = current + delta
            remaining_turnover -= abs(delta)
            
            if remaining_turnover <= 0:
                break
        
        # Clean up near-zero positions
        optimized = {k: v for k, v in optimized.items() if v > 0.001}
        
        # Normalize weights
        total = sum(optimized.values())
        if total > 0:
            optimized = {k: v/total for k, v in optimized.items()}
        
        return optimized
    
    def check_stop_losses(
        self,
        prices: Dict[str, float],
    ) -> List[str]:
        """
        Check all stop-loss conditions.
        
        Returns list of tickers to exit.
        """
        exits = []
        
        # Check circuit breaker first
        if abs(self.current_drawdown()) >= self.config.circuit_breaker_dd:
            logger.warning(f"🚨 Circuit breaker triggered at {self.current_drawdown():.1%} drawdown!")
            self.circuit_breaker_triggers += 1
            return list(self.positions.keys())  # Exit all positions
        
        for ticker, pos in self.positions.items():
            if ticker not in prices:
                continue
            
            current_price = prices[ticker]
            
            # Update position state
            pos.current_price = current_price
            pos.peak_price = max(pos.peak_price, current_price)
            
            # Check position stop-loss (-8%)
            if pos.pnl_pct <= -self.config.position_stop_loss:
                logger.info(f"Stop-loss triggered for {ticker}: {pos.pnl_pct:.1%}")
                self.stop_loss_triggers += 1
                exits.append(ticker)
                continue
            
            # Check trailing stop (-10% from peak)
            if pos.drawdown_from_peak <= -self.config.trailing_stop_pct:
                logger.info(f"Trailing stop triggered for {ticker}: {pos.drawdown_from_peak:.1%} from peak")
                self.trailing_stop_triggers += 1
                exits.append(ticker)
        
        return exits
    
    def apply_position_limits(
        self,
        weights: Dict[str, float],
        sectors: Dict[str, str],
    ) -> Dict[str, float]:
        """Apply position and sector concentration limits."""
        adjusted = weights.copy()
        
        # Cap individual positions
        max_pos = self.config.max_position_weight
        excess_weight = 0.0
        
        for ticker in adjusted:
            if adjusted[ticker] > max_pos:
                excess_weight += adjusted[ticker] - max_pos
                adjusted[ticker] = max_pos
        
        # Redistribute excess to uncapped positions
        if excess_weight > 0:
            uncapped = [t for t in adjusted if adjusted[t] < max_pos]
            if uncapped:
                per_pos = excess_weight / len(uncapped)
                for ticker in uncapped:
                    adjusted[ticker] = min(adjusted[ticker] + per_pos, max_pos)
        
        # Check sector concentration
        sector_weights = {}
        for ticker, weight in adjusted.items():
            sector = sectors.get(ticker, 'Other')
            sector_weights[sector] = sector_weights.get(sector, 0) + weight
        
        max_sector = self.config.max_sector_weight
        for sector, sw in sector_weights.items():
            if sw > max_sector:
                # Scale down all positions in over-concentrated sector
                scale = max_sector / sw
                for ticker in adjusted:
                    if sectors.get(ticker, 'Other') == sector:
                        adjusted[ticker] *= scale
        
        # Normalize
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v/total for k, v in adjusted.items()}
        
        return adjusted
    
    def calculate_risk_adjusted_weights(
        self,
        raw_scores: Dict[str, float],
        current_weights: Dict[str, float],
        sectors: Dict[str, str],
        prices: Dict[str, pd.DataFrame],
        n_positions: int = 30,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Calculate risk-adjusted portfolio weights.
        
        Returns:
            (final_weights, metrics_dict)
        """
        # Step 1: Select top stocks by score
        sorted_tickers = sorted(raw_scores.keys(), key=lambda x: raw_scores[x], reverse=True)
        selected = sorted_tickers[:n_positions]
        
        # Step 2: Calculate equal-weight base
        base_weight = 1.0 / len(selected)
        target_weights = {t: base_weight for t in selected}
        
        # Step 3: Apply position scaling
        scale = self.combined_position_scale()
        cash_allocation = 1.0 - scale
        
        scaled_weights = {t: w * scale for t, w in target_weights.items()}
        
        # Step 4: Apply position and sector limits
        limited_weights = self.apply_position_limits(scaled_weights, sectors)
        
        # Step 5: Optimize for transaction costs
        final_weights = self.optimize_for_turnover(
            current_weights,
            limited_weights,
            expected_alphas=raw_scores,
        )
        
        # Calculate metrics
        turnover = sum(abs(final_weights.get(t, 0) - current_weights.get(t, 0)) 
                      for t in set(final_weights.keys()) | set(current_weights.keys())) / 2
        cost = turnover * (self.config.cost_per_trade_bps / 10000)
        
        self.total_turnover += turnover
        self.total_costs += cost
        
        metrics = {
            'dd_scale': self.drawdown_position_scale(),
            'vol_scale': self.volatility_scale(),
            'combined_scale': scale,
            'cash_allocation': cash_allocation,
            'turnover': turnover,
            'cost_bps': cost * 10000,
            'n_positions': len(final_weights),
        }
        
        return final_weights, metrics
    
    def update_portfolio_value(
        self,
        new_value: float,
    ):
        """Update portfolio value and tracking metrics."""
        if len(self.equity_history) > 0:
            prev_value = self.equity_history[-1]
            daily_return = (new_value - prev_value) / prev_value
            self.returns_history.append(daily_return)
        
        self.portfolio_value = new_value
        self.peak_value = max(self.peak_value, new_value)
        self.equity_history.append(new_value)
        self.drawdown_history.append(self.current_drawdown())
        
        # Update realized vol
        if len(self.returns_history) >= self.config.vol_lookback_days:
            self.vol_history.append(self.realized_volatility())
    
    def get_risk_metrics(self) -> Dict:
        """Get comprehensive risk metrics."""
        returns = np.array(self.returns_history) if self.returns_history else np.array([0])
        
        # Basic metrics
        total_return = (self.portfolio_value / 100) - 1
        n_days = len(self.equity_history)
        years = n_days / 252
        
        cagr = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        
        # Volatility
        annual_vol = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
        
        # Sharpe (risk-free = 0 for simplicity)
        sharpe = cagr / annual_vol if annual_vol > 0 else 0
        
        # Sortino (downside deviation)
        downside = returns[returns < 0]
        downside_vol = np.std(downside) * np.sqrt(252) if len(downside) > 1 else annual_vol
        sortino = cagr / downside_vol if downside_vol > 0 else 0
        
        # Max drawdown
        max_dd = min(self.drawdown_history) if self.drawdown_history else 0
        
        # Calmar
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        
        return {
            'total_return': total_return,
            'cagr': cagr,
            'annual_vol': annual_vol,
            'sharpe': sharpe,
            'sortino': sortino,
            'max_drawdown': max_dd,
            'calmar': calmar,
            'total_turnover': self.total_turnover,
            'total_costs_bps': self.total_costs * 10000,
            'stop_loss_triggers': self.stop_loss_triggers,
            'trailing_stop_triggers': self.trailing_stop_triggers,
            'circuit_breaker_triggers': self.circuit_breaker_triggers,
            'avg_realized_vol': np.mean(self.vol_history) if self.vol_history else 0,
        }
    
    def print_summary(self):
        """Print risk management summary."""
        metrics = self.get_risk_metrics()
        
        print("\n" + "="*60)
        print("ENHANCED RISK MANAGER SUMMARY")
        print("="*60)
        print(f"\n📊 PERFORMANCE:")
        print(f"  Total Return:   {metrics['total_return']:>8.1%}")
        print(f"  CAGR:           {metrics['cagr']:>8.1%}")
        print(f"  Annual Vol:     {metrics['annual_vol']:>8.1%}")
        print(f"  Sharpe:         {metrics['sharpe']:>8.2f}")
        print(f"  Sortino:        {metrics['sortino']:>8.2f}")
        print(f"  Max Drawdown:   {metrics['max_drawdown']:>8.1%}")
        print(f"  Calmar:         {metrics['calmar']:>8.2f}")
        
        print(f"\n💰 TRANSACTION COSTS:")
        print(f"  Total Turnover: {metrics['total_turnover']:>8.1%}")
        print(f"  Total Costs:    {metrics['total_costs_bps']:>8.1f} bps")
        
        print(f"\n🛑 STOP-LOSS TRIGGERS:")
        print(f"  Position Stops: {metrics['stop_loss_triggers']:>8}")
        print(f"  Trailing Stops: {metrics['trailing_stop_triggers']:>8}")
        print(f"  Circuit Breakers: {metrics['circuit_breaker_triggers']:>6}")
        
        print(f"\n📈 VOLATILITY TARGETING:")
        print(f"  Target Vol:     {self.config.target_annual_vol:>8.1%}")
        print(f"  Avg Realized:   {metrics['avg_realized_vol']:>8.1%}")
        
        print("="*60)


def test_risk_manager():
    """Test enhanced risk manager."""
    print("\n" + "="*60)
    print("Testing Enhanced Risk Manager")
    print("="*60)
    
    rm = EnhancedRiskManager()
    
    # Simulate some portfolio movements
    print("\nSimulating 20-day period...")
    
    np.random.seed(42)
    for day in range(20):
        # Random daily return
        daily_ret = np.random.normal(0.001, 0.015)
        new_value = rm.portfolio_value * (1 + daily_ret)
        rm.update_portfolio_value(new_value)
        
        # Check scaling
        dd = rm.current_drawdown()
        scale = rm.combined_position_scale()
        print(f"Day {day+1:2d}: Value={rm.portfolio_value:.2f}, DD={dd:+.1%}, Scale={scale:.2f}")
    
    rm.print_summary()


if __name__ == "__main__":
    test_risk_manager()
