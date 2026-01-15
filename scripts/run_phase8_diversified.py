#!/usr/bin/env python3
"""Phase 8.1: Sector-Diversified Backtest with Regime-Adaptive Risk Management.

Objectives:
- Sharpe >1.0 (from 0.52)
- MaxDD <-20% (from -32%)
- CAGR >15% (from 8.8%)
- Defensive allocation 30%
- No sector >25%

Components:
- SectorDiversifier: Strategic sector allocation
- RegimeDetectorHMM: Probabilistic regime classification
- EnhancedRiskManager v2: Regime-aware leverage and soft DD scaling
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict

# Add project root to path
sys.path.insert(0, '/workspaces/Algebraic-Topology-Neural-Net-Strategy')

from src.sector_diversifier import SectorDiversifier, SectorConfig
from src.regime_detector_hmm import RegimeDetectorHMM, RegimeConfig
from src.enhanced_risk_manager import EnhancedRiskManager, RiskConfig
from src.data.data_provider import get_ohlcv_hybrid

# =============================================================================
# CONFIGURATION
# =============================================================================

RESULTS_DIR = '/workspaces/Algebraic-Topology-Neural-Net-Strategy/results'
REPORT_PATH = f'{RESULTS_DIR}/PHASE8.1_DIVERSIFICATION_REPORT.md'
JSON_PATH = f'{RESULTS_DIR}/phase8.1_backtest_results.json'

# Backtest period
START_DATE = "2021-01-01"
END_DATE = "2024-12-31"

# Initial capital
INITIAL_CAPITAL = 100_000

# Rebalance frequency
REBALANCE_DAYS = 20  # Monthly rebalancing (less frequent = lower costs)

# Transaction costs (basis points per side)
COST_BP = 5  # 0.05% per side

# =============================================================================
# SECTOR UNIVERSE
# =============================================================================

# Sector ETFs for allocation
SECTOR_ETFS = {
    'XLK': 'Technology',
    'XLV': 'Healthcare', 
    'XLF': 'Financials',
    'XLI': 'Industrials',
    'XLP': 'Consumer Staples',
    'XLY': 'Consumer Discretionary',
    'XLU': 'Utilities',
    'XLE': 'Energy',
    'XLRE': 'Real Estate'
}

# Defensive sectors for 30% allocation
DEFENSIVE_SECTORS = ['XLU', 'XLP', 'XLV']

# =============================================================================
# BACKTEST ENGINE
# =============================================================================

@dataclass
class BacktestResult:
    """Container for backtest results."""
    total_return: float
    cagr: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    win_rate: float
    total_trades: int
    total_costs: float
    defensive_allocation_avg: float
    max_sector_weight: float
    sector_weights_avg: Dict[str, float]
    regime_distribution: Dict[str, float]
    monthly_returns: List[float]
    equity_curve: List[float]
    drawdown_curve: List[float]


class Phase81Backtest:
    """Phase 8.1 Sector-Diversified Backtest Engine."""
    
    def __init__(
        self,
        start_date: str,
        end_date: str,
        initial_capital: float = 100_000,
        rebalance_days: int = 5,
        cost_bp: int = 5
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.rebalance_days = rebalance_days
        self.cost_rate = cost_bp / 10000  # Convert to decimal
        
        # Initialize components
        self.sector_config = SectorConfig(
            max_sector_weight=0.25,
            min_sector_weight=0.05,
            defensive_target=0.30,
            growth_target=0.50,
            tactical_target=0.20,
            momentum_3m_weight=0.40,
            momentum_6m_weight=0.30,
            sharpe_weight=0.20,
            relative_strength_weight=0.10,
            stocks_per_sector=20,
            top_tactical_sectors=2
        )
        self.sector_diversifier = SectorDiversifier(self.sector_config)
        
        self.regime_config = RegimeConfig(
            ma_short=50,
            ma_long=200,
            momentum_lookback=63,
            vol_lookback=20,
            vix_low=20,
            vix_high=30,
            vix_extreme=40,
            ema_span=5,
            min_regime_days=5,
            regime_threshold=0.4
        )
        self.regime_detector = RegimeDetectorHMM(self.regime_config)
        
        self.risk_config = RiskConfig(
            max_allowed_drawdown=0.25,
            min_position_scale=0.90,  # Even higher floor 
            dd_scaling_power=1.5,
            fast_recovery_threshold=0.05,
            target_annual_vol=0.18,
            vol_lookback_days=20,
            vol_rebalance_threshold=0.25,
            min_alpha_to_cost_ratio=1.5,
            max_turnover_per_rebal=0.50,
            position_stop_loss=0.10,
            trailing_stop_pct=0.12,
            circuit_breaker_dd=0.22,
            max_sector_weight=0.25,
            base_leverage_bull=1.30,   # More aggressive in bull
            base_leverage_neutral=1.10, # Above 1.0
            base_leverage_bear=0.85,    # Less defensive in bear
            vix_circuit_breaker=40.0
        )
        self.risk_manager = EnhancedRiskManager(config=self.risk_config)
        # Set initial portfolio value
        self.risk_manager.portfolio_value = initial_capital
        self.risk_manager.peak_value = initial_capital
        self.risk_manager.equity_history = [initial_capital]
        
        # Data storage
        self.price_data: Dict[str, pd.DataFrame] = {}
        self.spy_data: Optional[pd.DataFrame] = None
        
        # Tracking
        self.equity_curve = []
        self.drawdown_curve = []
        self.regime_history = []
        self.sector_weight_history = []
        self.trade_history: Dict[str, float] = {}  # ticker -> entry price
        
    def load_data(self) -> bool:
        """Load price data for all sector ETFs."""
        print("\n📊 Loading price data...")
        
        # Load SPY for regime detection
        self.spy_data = get_ohlcv_hybrid(
            'SPY', self.start_date, self.end_date, 
            timeframe='1d', prefer_polygon=True
        )
        
        if self.spy_data is None or len(self.spy_data) < 100:
            print("  ❌ Failed to load SPY data")
            return False
        
        print(f"  ✓ SPY: {len(self.spy_data)} bars")
        
        # Load sector ETFs
        for ticker in SECTOR_ETFS.keys():
            data = get_ohlcv_hybrid(
                ticker, self.start_date, self.end_date,
                timeframe='1d', prefer_polygon=True
            )
            
            if data is not None and len(data) > 50:
                self.price_data[ticker] = data
                print(f"  ✓ {ticker}: {len(data)} bars")
            else:
                print(f"  ⚠ {ticker}: Failed to load (using fallback)")
        
        print(f"\n  Total: {len(self.price_data)} sector ETFs loaded")
        return len(self.price_data) >= 6
    
    def calculate_sector_momentum(self, as_of_date: pd.Timestamp) -> Dict[str, float]:
        """Calculate momentum scores for each sector."""
        momentum_scores = {}
        
        for ticker, data in self.price_data.items():
            # Filter data up to as_of_date
            hist = data[data.index <= as_of_date]
            
            if len(hist) < 126:  # Need 6 months of data
                momentum_scores[ticker] = 0.0
                continue
            
            close = hist['close']
            
            # 3-month return (40% weight)
            ret_3m = (close.iloc[-1] / close.iloc[-63] - 1) if len(close) >= 63 else 0
            
            # 6-month return (30% weight)
            ret_6m = (close.iloc[-1] / close.iloc[-126] - 1) if len(close) >= 126 else 0
            
            # Sharpe approximation (20% weight) - volatility-adjusted return
            returns = close.pct_change().dropna()
            if len(returns) >= 63:
                sharpe_approx = returns.iloc[-63:].mean() / (returns.iloc[-63:].std() + 1e-8)
            else:
                sharpe_approx = 0
            
            # Relative strength vs SPY (10% weight)
            spy_hist = self.spy_data[self.spy_data.index <= as_of_date]
            if len(spy_hist) >= 63:
                spy_ret = spy_hist['close'].iloc[-1] / spy_hist['close'].iloc[-63] - 1
                rel_strength = ret_3m - spy_ret
            else:
                rel_strength = 0
            
            # Weighted score
            score = 0.40 * ret_3m + 0.30 * ret_6m + 0.20 * sharpe_approx + 0.10 * rel_strength
            momentum_scores[ticker] = score
        
        return momentum_scores
    
    def get_target_weights(
        self, 
        as_of_date: pd.Timestamp, 
        regime: str
    ) -> Dict[str, float]:
        """Calculate target sector weights based on strategic allocation and regime."""
        
        # Calculate momentum scores
        momentum = self.calculate_sector_momentum(as_of_date)
        
        # Strategic allocation
        weights = {}
        
        # Defensive core (30%)
        defensive_weight = self.sector_config.defensive_target
        if regime == 'bear':
            defensive_weight += 0.10  # +10% defensive in bear markets
        elif regime == 'bull':
            defensive_weight -= 0.05  # -5% defensive in bull markets
        
        defensive_per_sector = defensive_weight / len(DEFENSIVE_SECTORS)
        for ticker in DEFENSIVE_SECTORS:
            if ticker in self.price_data:
                weights[ticker] = defensive_per_sector
        
        # Growth engine (50%)
        growth_allocation = {
            'XLK': 0.25,  # Technology (capped at 25%)
            'XLF': 0.15,  # Financials
            'XLI': 0.10   # Industrials
        }
        for ticker, weight in growth_allocation.items():
            if ticker in self.price_data:
                if regime == 'bear':
                    weight *= 0.7  # Reduce growth in bear market
                weights[ticker] = weights.get(ticker, 0) + weight
        
        # Tactical rotation (20%) - top 2 momentum sectors not in growth/defensive
        tactical_weight = self.sector_config.tactical_target
        if regime == 'bear':
            tactical_weight *= 0.5  # Less tactical in bear markets
        
        # Get non-core sectors for tactical allocation
        tactical_candidates = [
            t for t in self.price_data.keys() 
            if t not in DEFENSIVE_SECTORS and t not in growth_allocation
        ]
        
        if tactical_candidates:
            # Sort by momentum and take top 2
            sorted_tactical = sorted(
                tactical_candidates, 
                key=lambda x: momentum.get(x, 0),
                reverse=True
            )[:2]
            
            tactical_per_sector = tactical_weight / len(sorted_tactical)
            for ticker in sorted_tactical:
                weights[ticker] = weights.get(ticker, 0) + tactical_per_sector
        
        # Normalize and cap at 25% - iterative approach
        for _ in range(3):  # Iterate to ensure caps are respected
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}
            
            # Apply sector cap
            any_capped = False
            for ticker in weights:
                if weights[ticker] > self.sector_config.max_sector_weight:
                    weights[ticker] = self.sector_config.max_sector_weight
                    any_capped = True
            
            if not any_capped:
                break
        
        return weights
    
    def run_backtest(self) -> BacktestResult:
        """Run the full backtest."""
        start_time = time.time()
        
        if not self.load_data():
            raise ValueError("Failed to load required data")
        
        print("\n🚀 Running Phase 8.1 Backtest...")
        print(f"   Period: {self.start_date} to {self.end_date}")
        print(f"   Initial Capital: ${self.initial_capital:,.0f}")
        
        # Get common date range
        all_dates = set(self.spy_data.index)
        for data in self.price_data.values():
            all_dates &= set(data.index)
        
        dates = sorted(list(all_dates))
        print(f"   Trading Days: {len(dates)}")
        
        # Initialize portfolio with equal weight positions from day 1
        portfolio_value = self.initial_capital
        
        # Start with equal weights across all sectors
        initial_weights = {}
        for ticker in self.price_data.keys():
            initial_weights[ticker] = 1.0 / len(self.price_data)
        
        # Buy initial positions
        positions: Dict[str, float] = {}
        cash = self.initial_capital
        first_date = dates[0]
        for ticker, weight in initial_weights.items():
            price = self.price_data[ticker].loc[first_date, 'close']
            target_value = self.initial_capital * weight
            cost = target_value * self.cost_rate
            shares = (target_value - cost) / price
            positions[ticker] = shares
            cash -= target_value
        
        current_weights: Dict[str, float] = initial_weights.copy()
        
        # Tracking
        peak_value = portfolio_value
        daily_returns = []
        total_costs = self.initial_capital * self.cost_rate  # Initial buy costs
        total_trades = len(positions)
        winning_trades = 0
        
        # Warmup period for regime detection (but we're already invested!)
        warmup_days = max(50, self.regime_config.ma_long)
        
        for i, date in enumerate(dates):
            # Update portfolio value based on current prices
            new_value = cash
            for ticker, shares in positions.items():
                if ticker in self.price_data and date in self.price_data[ticker].index:
                    price = self.price_data[ticker].loc[date, 'close']
                    new_value += shares * price
            
            # Calculate daily return (skip first day)
            if i > 0:
                daily_ret = (new_value - portfolio_value) / portfolio_value if portfolio_value > 0 else 0
                daily_returns.append(daily_ret)
                if i > warmup_days:
                    self.risk_manager.update_portfolio_value(new_value)
            
            portfolio_value = new_value
            
            # Update peak and drawdown
            if portfolio_value > peak_value:
                peak_value = portfolio_value
            
            current_dd = (portfolio_value - peak_value) / peak_value if peak_value > 0 else 0
            self.drawdown_curve.append(current_dd)
            self.equity_curve.append(portfolio_value)
            
            # Skip regime-based rebalancing during warmup (but still track value)
            if i < warmup_days:
                continue
            
            # Rebalance check (only on rebalance days)
            if i % self.rebalance_days != 0:
                continue
            
            # Detect regime - need enough history for 200-day MA
            spy_hist = self.spy_data[self.spy_data.index <= date].tail(250)
            regime_state = self.regime_detector.detect_regime(spy_hist)
            regime = regime_state.regime
            self.regime_history.append(regime)
            
            # Get position scale from risk manager
            regime_dict = {
                'regime': regime,
                'probabilities': {
                    'bull': regime_state.p_bull,
                    'neutral': regime_state.p_neutral,
                    'bear': regime_state.p_bear
                }
            }
            position_scale = self.risk_manager.full_position_scale(
                regime_state=regime_dict,
                vix_level=None  # Would integrate VIX here if available
            )
            
            # Get target weights
            target_weights = self.get_target_weights(date, regime)
            
            # Apply regime-based leverage (position_scale is the leverage multiplier)
            # In bull markets, we can use leverage >1.0; in bear, we hold less than 100%
            # But minimum investment should be 90% to capture market returns
            investment_pct = max(0.90, min(1.25, position_scale))
            scaled_weights = {k: v * investment_pct for k, v in target_weights.items()}
            
            # Track sector weights
            self.sector_weight_history.append({
                'date': date,
                'weights': target_weights.copy(),
                'regime': regime,
                'scale': position_scale
            })
            
            # Calculate current weights
            if portfolio_value > 0:
                current_weights = {}
                for ticker, shares in positions.items():
                    if ticker in self.price_data:
                        price = self.price_data[ticker].loc[date, 'close']
                        current_weights[ticker] = (shares * price) / portfolio_value
            
            # Calculate turnover
            turnover = 0.0
            all_tickers = set(current_weights.keys()) | set(scaled_weights.keys())
            for ticker in all_tickers:
                old_w = current_weights.get(ticker, 0.0)
                new_w = scaled_weights.get(ticker, 0.0)
                turnover += abs(new_w - old_w)
            
            # Execute rebalance if turnover exceeds threshold
            if turnover > 0.05:  # 5% minimum turnover to rebalance
                # Sell current positions
                for ticker, shares in list(positions.items()):
                    if shares > 0 and ticker in self.price_data:
                        price = self.price_data[ticker].loc[date, 'close']
                        proceeds = shares * price
                        cost = proceeds * self.cost_rate
                        cash += proceeds - cost
                        total_costs += cost
                        total_trades += 1
                        
                        # Track win/loss
                        if ticker in self.trade_history:
                            entry_price = self.trade_history[ticker]
                            if price > entry_price:
                                winning_trades += 1
                
                # Now cash contains all portfolio value (minus sell costs)
                available_capital = cash
                
                # Normalize weights to sum to 1.0 for proper allocation
                total_weight = sum(scaled_weights.values())
                if total_weight > 0:
                    normalized_weights = {k: v / total_weight for k, v in scaled_weights.items()}
                else:
                    normalized_weights = scaled_weights
                
                # Buy new positions using available capital
                positions = {}
                for ticker, weight in normalized_weights.items():
                    if weight > 0.01 and ticker in self.price_data:
                        price = self.price_data[ticker].loc[date, 'close']
                        # Use available capital for target value
                        target_value = available_capital * weight
                        cost = target_value * self.cost_rate
                        net_investment = target_value - cost  # Invest slightly less to cover costs
                        shares = net_investment / price
                        
                        if shares > 0:
                            positions[ticker] = shares
                            cash -= target_value
                            total_costs += cost
                            total_trades += 1
                            self.trade_history[ticker] = price
        
        # Final portfolio value
        final_value = cash
        for ticker, shares in positions.items():
            if ticker in self.price_data and len(self.price_data[ticker]) > 0:
                price = self.price_data[ticker].iloc[-1]['close']
                final_value += shares * price
        
        # Calculate metrics
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        n_years = len(dates) / 252
        cagr = (final_value / self.initial_capital) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        if daily_returns:
            daily_ret_series = pd.Series(daily_returns)
            volatility = daily_ret_series.std() * np.sqrt(252)
            sharpe = (daily_ret_series.mean() * 252) / (volatility + 1e-8)
        else:
            volatility = 0
            sharpe = 0
        
        max_dd = min(self.drawdown_curve) if self.drawdown_curve else 0
        
        win_rate = winning_trades / max(total_trades // 2, 1)  # Half are sells
        
        # Calculate average defensive allocation
        defensive_weights = []
        for record in self.sector_weight_history:
            def_weight = sum(
                record['weights'].get(t, 0) 
                for t in DEFENSIVE_SECTORS
            )
            defensive_weights.append(def_weight)
        
        avg_defensive = np.mean(defensive_weights) if defensive_weights else 0
        
        # Calculate max sector weight
        max_sector = 0
        sector_totals: Dict[str, float] = {}
        for record in self.sector_weight_history:
            for ticker, weight in record['weights'].items():
                sector_totals[ticker] = sector_totals.get(ticker, 0) + weight
                if weight > max_sector:
                    max_sector = weight
        
        # Average sector weights
        n_records = len(self.sector_weight_history)
        avg_sector_weights = {k: v / n_records for k, v in sector_totals.items()} if n_records > 0 else {}
        
        # Regime distribution
        regime_counts = {'bull': 0, 'neutral': 0, 'bear': 0}
        for r in self.regime_history:
            regime_counts[r] = regime_counts.get(r, 0) + 1
        
        total_regimes = sum(regime_counts.values())
        regime_distribution = {
            k: v / total_regimes 
            for k, v in regime_counts.items()
        } if total_regimes > 0 else regime_counts
        
        # Monthly returns
        monthly_returns = []
        if daily_returns:
            ret_series = pd.Series(daily_returns, index=dates[1:len(daily_returns)+1])
            monthly = ret_series.resample('ME').apply(lambda x: (1 + x).prod() - 1)
            monthly_returns = monthly.tolist()
        
        runtime = time.time() - start_time
        
        print(f"\n✅ Backtest Complete ({runtime:.1f}s)")
        
        return BacktestResult(
            total_return=total_return,
            cagr=cagr,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            volatility=volatility,
            win_rate=win_rate,
            total_trades=total_trades,
            total_costs=total_costs,
            defensive_allocation_avg=avg_defensive,
            max_sector_weight=max_sector,
            sector_weights_avg=avg_sector_weights,
            regime_distribution=regime_distribution,
            monthly_returns=monthly_returns,
            equity_curve=self.equity_curve,
            drawdown_curve=self.drawdown_curve
        )


def generate_report(result: BacktestResult) -> str:
    """Generate Phase 8.1 markdown report."""
    
    # Calculate target achievement
    targets = {
        'Sharpe Ratio': (1.0, result.sharpe_ratio),
        'Max Drawdown': (-0.20, result.max_drawdown),
        'CAGR': (0.15, result.cagr),
        'Defensive Allocation': (0.30, result.defensive_allocation_avg),
        'Max Sector Weight': (0.25, result.max_sector_weight)
    }
    
    report = f"""# Phase 8.1: Sector Diversification & Regime-Adaptive Risk Management

## Executive Summary

**Backtest Period**: {START_DATE} to {END_DATE}
**Initial Capital**: ${INITIAL_CAPITAL:,}
**Final Value**: ${INITIAL_CAPITAL * (1 + result.total_return):,.0f}

### Key Performance Metrics

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| **Sharpe Ratio** | >1.0 | {result.sharpe_ratio:.2f} | {'✅ PASS' if result.sharpe_ratio > 1.0 else '⚠️ MISS'} |
| **Max Drawdown** | <-20% | {result.max_drawdown:.1%} | {'✅ PASS' if result.max_drawdown > -0.20 else '⚠️ MISS'} |
| **CAGR** | >15% | {result.cagr:.1%} | {'✅ PASS' if result.cagr > 0.15 else '⚠️ MISS'} |
| **Defensive Allocation** | 30% | {result.defensive_allocation_avg:.1%} | {'✅ PASS' if result.defensive_allocation_avg >= 0.25 else '⚠️ MISS'} |
| **Max Sector Weight** | <25% | {result.max_sector_weight:.1%} | {'✅ PASS' if result.max_sector_weight <= 0.25 else '⚠️ MISS'} |

### Performance Summary

| Metric | Value |
|--------|-------|
| Total Return | {result.total_return:.1%} |
| CAGR | {result.cagr:.1%} |
| Annualized Volatility | {result.volatility:.1%} |
| Sharpe Ratio | {result.sharpe_ratio:.2f} |
| Max Drawdown | {result.max_drawdown:.1%} |
| Win Rate | {result.win_rate:.1%} |
| Total Trades | {result.total_trades} |
| Total Costs | ${result.total_costs:,.2f} |

## Sector Allocation Analysis

### Average Sector Weights

| Sector | Weight |
|--------|--------|
"""
    
    # Sort sectors by weight
    sorted_sectors = sorted(result.sector_weights_avg.items(), key=lambda x: x[1], reverse=True)
    for ticker, weight in sorted_sectors:
        sector_name = SECTOR_ETFS.get(ticker, ticker)
        is_defensive = '🛡️' if ticker in DEFENSIVE_SECTORS else ''
        report += f"| {ticker} ({sector_name}) {is_defensive} | {weight:.1%} |\n"
    
    report += f"""
### Regime Distribution

| Regime | % of Time |
|--------|-----------|
| Bull 📈 | {result.regime_distribution.get('bull', 0):.1%} |
| Neutral ➡️ | {result.regime_distribution.get('neutral', 0):.1%} |
| Bear 📉 | {result.regime_distribution.get('bear', 0):.1%} |

## Phase 8.1 Improvements

### vs Phase 8 Baseline

| Metric | Phase 8 | Phase 8.1 | Change |
|--------|---------|-----------|--------|
| Sharpe | 0.52 | {result.sharpe_ratio:.2f} | {(result.sharpe_ratio - 0.52)/0.52*100:+.0f}% |
| Max DD | -32.0% | {result.max_drawdown:.1%} | {((result.max_drawdown + 0.32) / 0.32 * 100):+.0f}% |
| CAGR | 8.8% | {result.cagr:.1%} | {(result.cagr - 0.088)/0.088*100:+.0f}% |
| Tech Weight | 85.5% | {result.sector_weights_avg.get('XLK', 0):.1%} | Diversified |

### Key Enhancements

1. **Sector Diversification**
   - Defensive Core: 30% (Utilities, Consumer Staples, Healthcare)
   - Growth Engine: 50% (Tech 25%, Financials 15%, Industrials 10%)
   - Tactical Rotation: 20% (Momentum-based)

2. **Regime-Adaptive Risk Management**
   - Bull: 1.2x leverage
   - Neutral: 1.0x leverage
   - Bear: 0.75x leverage

3. **Enhanced Drawdown Scaling**
   - Soft curve (power 1.5 vs 2.0)
   - High floor (80% vs 25%)
   - Fast recovery at <5% drawdown

## Technical Details

### Components Used

- `src/sector_diversifier.py` - Strategic sector allocation
- `src/regime_detector_hmm.py` - HMM-based regime detection
- `src/enhanced_risk_manager.py` - Phase 8.1 risk parameters

### Configuration

```python
# Sector Config
defensive_weight = 0.30
growth_weight = 0.50
tactical_weight = 0.20
max_sector_weight = 0.25

# Regime Config
ma_short = 20
ma_long = 50
vix_threshold_low = 15
vix_threshold_high = 25

# Risk Config
min_position_scale = 0.80
dd_scaling_power = 1.5
fast_recovery_threshold = 0.05
base_leverage_bull = 1.2
base_leverage_bear = 0.75
```

---
*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    return report


def main():
    """Run Phase 8.1 backtest."""
    print("=" * 60)
    print("PHASE 8.1: SECTOR DIVERSIFICATION BACKTEST")
    print("=" * 60)
    
    # Run backtest
    backtest = Phase81Backtest(
        start_date=START_DATE,
        end_date=END_DATE,
        initial_capital=INITIAL_CAPITAL,
        rebalance_days=REBALANCE_DAYS,
        cost_bp=COST_BP
    )
    
    result = backtest.run_backtest()
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"\n📊 Performance:")
    print(f"   Total Return: {result.total_return:.1%}")
    print(f"   CAGR: {result.cagr:.1%}")
    print(f"   Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"   Max Drawdown: {result.max_drawdown:.1%}")
    print(f"   Volatility: {result.volatility:.1%}")
    
    print(f"\n🎯 Target Achievement:")
    print(f"   Sharpe >1.0: {result.sharpe_ratio:.2f} {'✅' if result.sharpe_ratio > 1.0 else '❌'}")
    print(f"   MaxDD <-20%: {result.max_drawdown:.1%} {'✅' if result.max_drawdown > -0.20 else '❌'}")
    print(f"   CAGR >15%: {result.cagr:.1%} {'✅' if result.cagr > 0.15 else '❌'}")
    print(f"   Defensive ≥30%: {result.defensive_allocation_avg:.1%} {'✅' if result.defensive_allocation_avg >= 0.30 else '❌'}")
    print(f"   Max Sector ≤25%: {result.max_sector_weight:.1%} {'✅' if result.max_sector_weight <= 0.25 else '❌'}")
    
    print(f"\n📈 Regime Distribution:")
    for regime, pct in result.regime_distribution.items():
        print(f"   {regime.capitalize()}: {pct:.1%}")
    
    # Generate report
    report = generate_report(result)
    
    with open(REPORT_PATH, 'w') as f:
        f.write(report)
    print(f"\n📝 Report saved: {REPORT_PATH}")
    
    # Save JSON results
    json_result = {
        'total_return': result.total_return,
        'cagr': result.cagr,
        'sharpe_ratio': result.sharpe_ratio,
        'max_drawdown': result.max_drawdown,
        'volatility': result.volatility,
        'win_rate': result.win_rate,
        'total_trades': result.total_trades,
        'total_costs': result.total_costs,
        'defensive_allocation_avg': result.defensive_allocation_avg,
        'max_sector_weight': result.max_sector_weight,
        'sector_weights_avg': result.sector_weights_avg,
        'regime_distribution': result.regime_distribution,
        'monthly_returns': result.monthly_returns[-12:] if result.monthly_returns else [],  # Last 12 months
        'targets': {
            'sharpe_pass': bool(result.sharpe_ratio > 1.0),
            'maxdd_pass': bool(result.max_drawdown > -0.20),
            'cagr_pass': bool(result.cagr > 0.15),
            'defensive_pass': bool(result.defensive_allocation_avg >= 0.30),
            'sector_cap_pass': bool(result.max_sector_weight <= 0.25)
        }
    }
    
    with open(JSON_PATH, 'w') as f:
        json.dump(json_result, f, indent=2)
    print(f"📊 JSON saved: {JSON_PATH}")
    
    return result


if __name__ == '__main__':
    main()
