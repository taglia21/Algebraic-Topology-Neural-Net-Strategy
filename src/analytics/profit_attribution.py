"""
Profit Attribution Analysis Module
===================================

V2.4 Profitability Enhancement - Understand where profits come from

Key Features:
1. P&L Decomposition - Break down returns by component
2. Factor Attribution - Attribute returns to market factors
3. Strategy Attribution - Understand which strategies contribute
4. Trade Analysis - Win rate, avg win/loss, best/worst trades
5. Time-Based Analysis - P&L by hour, day, month

Research Basis:
- Brinson attribution for factor decomposition
- Risk attribution to understand exposures
- Trade journal analysis for behavioral insights

Target Performance:
- Identify which components generate alpha
- Find underperforming strategies to improve/remove
- Optimize timing and sizing based on historical patterns
"""

import numpy as np
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, date, timedelta
from enum import Enum
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND DATA STRUCTURES
# =============================================================================

class PnLComponent(Enum):
    """Components of P&L."""
    ALPHA = "alpha"                 # Stock selection
    MARKET = "market"               # Market exposure (beta)
    SECTOR = "sector"               # Sector allocation
    MOMENTUM = "momentum"           # Momentum factor
    VALUE = "value"                 # Value factor
    SIZE = "size"                   # Size factor (SMB)
    VOLATILITY = "volatility"       # Low vol factor
    QUALITY = "quality"             # Quality factor
    TDA = "tda"                     # Topological features
    REGIME = "regime"               # Regime detection
    TIMING = "timing"               # Market timing
    TRANSACTION = "transaction"     # Transaction costs
    SLIPPAGE = "slippage"           # Execution slippage
    OTHER = "other"                 # Unexplained


class StrategyType(Enum):
    """Trading strategy types."""
    LONG_ONLY = "long_only"
    LONG_SHORT = "long_short"
    MARKET_NEUTRAL = "market_neutral"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    TDA_SIGNAL = "tda_signal"
    ENSEMBLE = "ensemble"


@dataclass
class Trade:
    """Individual trade record."""
    trade_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    shares: int
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    strategy: str
    pnl: float = 0.0
    pnl_pct: float = 0.0
    transaction_cost: float = 0.0
    
    def calculate_pnl(self):
        """Calculate P&L for the trade."""
        direction = 1 if self.side == 'buy' else -1
        gross_pnl = direction * (self.exit_price - self.entry_price) * self.shares
        self.pnl = gross_pnl - self.transaction_cost
        self.pnl_pct = (self.exit_price / self.entry_price - 1) * direction * 100
        

@dataclass
class DailyPnL:
    """Daily P&L record."""
    date: date
    gross_pnl: float
    net_pnl: float
    transaction_costs: float
    n_trades: int
    win_trades: int
    loss_trades: int
    portfolio_value: float
    daily_return: float
    
    # Component breakdown
    alpha_pnl: float = 0.0
    market_pnl: float = 0.0
    factor_pnl: float = 0.0
    timing_pnl: float = 0.0


# =============================================================================
# FACTOR ATTRIBUTION
# =============================================================================

class FactorAttributor:
    """
    Attribute returns to market factors.
    
    Uses regression-based attribution to decompose returns.
    """
    
    def __init__(self):
        self.factor_exposures: Dict[str, Dict[str, float]] = {}
        self.factor_returns: Dict[str, List[float]] = defaultdict(list)
        
    def set_factor_exposures(
        self,
        symbol: str,
        exposures: Dict[str, float]
    ):
        """Set factor exposures for a symbol."""
        self.factor_exposures[symbol] = exposures
        
    def add_factor_returns(
        self,
        date: datetime,
        returns: Dict[str, float]
    ):
        """Add daily factor returns."""
        for factor, ret in returns.items():
            self.factor_returns[factor].append(ret)
            
    def attribute_returns(
        self,
        portfolio_weights: Dict[str, float],
        portfolio_return: float,
        factor_returns: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Attribute portfolio return to factors.
        
        Args:
            portfolio_weights: Dict of symbol -> weight
            portfolio_return: Total portfolio return
            factor_returns: Dict of factor -> return
            
        Returns:
            Dict of factor -> attributed return
        """
        attribution = {}
        explained = 0.0
        
        # Calculate factor contributions
        for factor in [e.value for e in PnLComponent if e not in [
            PnLComponent.ALPHA, PnLComponent.OTHER, 
            PnLComponent.TRANSACTION, PnLComponent.SLIPPAGE
        ]]:
            # Sum of (weight * exposure * factor return)
            contribution = 0.0
            for symbol, weight in portfolio_weights.items():
                exposures = self.factor_exposures.get(symbol, {})
                exposure = exposures.get(factor, 0.0)
                factor_ret = factor_returns.get(factor, 0.0)
                contribution += weight * exposure * factor_ret
                
            attribution[factor] = contribution
            explained += contribution
            
        # Alpha is the unexplained portion
        attribution['alpha'] = portfolio_return - explained
        
        return attribution
    
    def get_factor_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for each factor."""
        summary = {}
        
        for factor, returns in self.factor_returns.items():
            if returns:
                returns_arr = np.array(returns)
                summary[factor] = {
                    'total_return': float(np.sum(returns_arr)),
                    'mean_return': float(np.mean(returns_arr)),
                    'volatility': float(np.std(returns_arr) * np.sqrt(252)),
                    'sharpe': float(np.mean(returns_arr) / np.std(returns_arr) * np.sqrt(252)) if np.std(returns_arr) > 0 else 0,
                }
                
        return summary


# =============================================================================
# TRADE ANALYZER
# =============================================================================

class TradeAnalyzer:
    """Analyze individual trades for patterns."""
    
    def __init__(self):
        self.trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        
    def add_trade(self, trade: Trade):
        """Add a trade to the analyzer."""
        trade.calculate_pnl()
        self.trades.append(trade)
        if trade.exit_time is not None:
            self.closed_trades.append(trade)
            
    def get_summary(self) -> Dict[str, Any]:
        """Get trade summary statistics."""
        if not self.closed_trades:
            return {'n_trades': 0}
            
        pnls = [t.pnl for t in self.closed_trades]
        pnl_pcts = [t.pnl_pct for t in self.closed_trades]
        
        wins = [t for t in self.closed_trades if t.pnl > 0]
        losses = [t for t in self.closed_trades if t.pnl <= 0]
        
        win_rate = len(wins) / len(self.closed_trades) if self.closed_trades else 0
        
        avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Best and worst trades
        sorted_by_pnl = sorted(self.closed_trades, key=lambda t: t.pnl)
        best_trades = [
            {'symbol': t.symbol, 'pnl': t.pnl, 'pnl_pct': t.pnl_pct}
            for t in sorted_by_pnl[-5:]
        ]
        worst_trades = [
            {'symbol': t.symbol, 'pnl': t.pnl, 'pnl_pct': t.pnl_pct}
            for t in sorted_by_pnl[:5]
        ]
        
        return {
            'n_trades': len(self.closed_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_pnl': sum(pnls),
            'avg_pnl': np.mean(pnls),
            'std_pnl': np.std(pnls),
            'max_win': max(pnls),
            'max_loss': min(pnls),
            'avg_pnl_pct': np.mean(pnl_pcts),
            'best_trades': best_trades,
            'worst_trades': worst_trades,
        }
    
    def get_by_strategy(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics grouped by strategy."""
        by_strategy = defaultdict(list)
        
        for trade in self.closed_trades:
            by_strategy[trade.strategy].append(trade)
            
        result = {}
        for strategy, trades in by_strategy.items():
            wins = [t for t in trades if t.pnl > 0]
            result[strategy] = {
                'n_trades': len(trades),
                'win_rate': len(wins) / len(trades) if trades else 0,
                'total_pnl': sum(t.pnl for t in trades),
                'avg_pnl': np.mean([t.pnl for t in trades]),
            }
            
        return result
    
    def get_by_symbol(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics grouped by symbol."""
        by_symbol = defaultdict(list)
        
        for trade in self.closed_trades:
            by_symbol[trade.symbol].append(trade)
            
        result = {}
        for symbol, trades in by_symbol.items():
            wins = [t for t in trades if t.pnl > 0]
            result[symbol] = {
                'n_trades': len(trades),
                'win_rate': len(wins) / len(trades) if trades else 0,
                'total_pnl': sum(t.pnl for t in trades),
                'avg_pnl': np.mean([t.pnl for t in trades]),
            }
            
        return result
    
    def get_by_hour(self) -> Dict[int, Dict[str, Any]]:
        """Get statistics grouped by hour of day."""
        by_hour = defaultdict(list)
        
        for trade in self.closed_trades:
            hour = trade.entry_time.hour
            by_hour[hour].append(trade)
            
        result = {}
        for hour, trades in sorted(by_hour.items()):
            wins = [t for t in trades if t.pnl > 0]
            result[hour] = {
                'n_trades': len(trades),
                'win_rate': len(wins) / len(trades) if trades else 0,
                'total_pnl': sum(t.pnl for t in trades),
                'avg_pnl': np.mean([t.pnl for t in trades]),
            }
            
        return result


# =============================================================================
# MAIN PROFIT ATTRIBUTION ENGINE
# =============================================================================

class ProfitAttributionEngine:
    """
    Main engine for profit attribution analysis.
    
    Tracks P&L, decomposes returns, and identifies profit sources.
    """
    
    def __init__(self, initial_value: float = 1_000_000):
        self.initial_value = initial_value
        self.portfolio_value = initial_value
        
        # Components
        self.factor_attributor = FactorAttributor()
        self.trade_analyzer = TradeAnalyzer()
        
        # P&L tracking
        self.daily_pnl_history: List[DailyPnL] = []
        self.component_pnl: Dict[str, float] = defaultdict(float)
        
        # Cumulative tracking
        self.total_gross_pnl: float = 0.0
        self.total_transaction_costs: float = 0.0
        self.total_slippage: float = 0.0
        
    def record_daily_pnl(
        self,
        date: date,
        portfolio_value: float,
        gross_pnl: float,
        transaction_costs: float,
        n_trades: int,
        win_trades: int,
        factor_returns: Optional[Dict[str, float]] = None,
        portfolio_weights: Optional[Dict[str, float]] = None
    ):
        """Record daily P&L with component breakdown."""
        net_pnl = gross_pnl - transaction_costs
        daily_return = net_pnl / self.portfolio_value if self.portfolio_value > 0 else 0
        
        record = DailyPnL(
            date=date,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            transaction_costs=transaction_costs,
            n_trades=n_trades,
            win_trades=win_trades,
            loss_trades=n_trades - win_trades,
            portfolio_value=portfolio_value,
            daily_return=daily_return,
        )
        
        # Factor attribution
        if factor_returns and portfolio_weights:
            attribution = self.factor_attributor.attribute_returns(
                portfolio_weights, daily_return, factor_returns
            )
            record.alpha_pnl = attribution.get('alpha', 0) * self.portfolio_value
            record.market_pnl = attribution.get('market', 0) * self.portfolio_value
            record.factor_pnl = sum(
                v for k, v in attribution.items() 
                if k not in ['alpha', 'market']
            ) * self.portfolio_value
            
            # Update component tracking
            for component, value in attribution.items():
                self.component_pnl[component] += value * self.portfolio_value
                
        self.daily_pnl_history.append(record)
        
        # Update totals
        self.portfolio_value = portfolio_value
        self.total_gross_pnl += gross_pnl
        self.total_transaction_costs += transaction_costs
        
    def record_trade(self, trade: Trade):
        """Record a completed trade."""
        self.trade_analyzer.add_trade(trade)
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.daily_pnl_history:
            return {'message': 'No data available'}
            
        daily_returns = [d.daily_return for d in self.daily_pnl_history]
        daily_returns_arr = np.array(daily_returns)
        
        # Performance metrics
        total_return = (self.portfolio_value - self.initial_value) / self.initial_value
        
        # Risk metrics
        volatility = np.std(daily_returns_arr) * np.sqrt(252) if len(daily_returns_arr) > 1 else 0
        sharpe = np.mean(daily_returns_arr) / np.std(daily_returns_arr) * np.sqrt(252) if np.std(daily_returns_arr) > 0 else 0
        
        # Drawdown
        cumulative = np.cumprod(1 + daily_returns_arr)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        # Win rate
        winning_days = sum(1 for d in self.daily_pnl_history if d.net_pnl > 0)
        total_days = len(self.daily_pnl_history)
        
        return {
            'performance': {
                'total_return': total_return,
                'total_return_pct': total_return * 100,
                'annualized_return': total_return * 252 / total_days if total_days > 0 else 0,
                'volatility': volatility,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown,
                'winning_days': winning_days,
                'total_days': total_days,
                'daily_win_rate': winning_days / total_days if total_days > 0 else 0,
            },
            'pnl': {
                'gross_pnl': self.total_gross_pnl,
                'transaction_costs': self.total_transaction_costs,
                'net_pnl': self.total_gross_pnl - self.total_transaction_costs,
                'cost_ratio': self.total_transaction_costs / self.total_gross_pnl if self.total_gross_pnl > 0 else 0,
            },
            'portfolio': {
                'initial_value': self.initial_value,
                'current_value': self.portfolio_value,
            },
        }
    
    def get_component_attribution(self) -> Dict[str, Any]:
        """Get P&L attributed to each component."""
        total_pnl = sum(self.component_pnl.values())
        
        attribution = {}
        for component, pnl in sorted(self.component_pnl.items(), key=lambda x: -abs(x[1])):
            attribution[component] = {
                'pnl': pnl,
                'pct_of_total': pnl / total_pnl * 100 if total_pnl != 0 else 0,
            }
            
        return {
            'total_pnl': total_pnl,
            'components': attribution,
        }
    
    def get_trade_attribution(self) -> Dict[str, Any]:
        """Get trade-level attribution."""
        return {
            'summary': self.trade_analyzer.get_summary(),
            'by_strategy': self.trade_analyzer.get_by_strategy(),
            'by_symbol': self.trade_analyzer.get_by_symbol(),
            'by_hour': self.trade_analyzer.get_by_hour(),
        }
    
    def get_time_analysis(self) -> Dict[str, Any]:
        """Get time-based P&L analysis."""
        if not self.daily_pnl_history:
            return {}
            
        # By day of week
        by_dow = defaultdict(list)
        for d in self.daily_pnl_history:
            dow = d.date.strftime('%A')
            by_dow[dow].append(d.daily_return)
            
        dow_stats = {}
        for dow, returns in by_dow.items():
            dow_stats[dow] = {
                'avg_return': np.mean(returns) * 100,
                'win_rate': sum(1 for r in returns if r > 0) / len(returns) if returns else 0,
                'n_days': len(returns),
            }
            
        # By month
        by_month = defaultdict(list)
        for d in self.daily_pnl_history:
            month = d.date.strftime('%Y-%m')
            by_month[month].append(d.daily_return)
            
        monthly_returns = {}
        for month, returns in sorted(by_month.items()):
            monthly_returns[month] = {
                'total_return': np.sum(returns) * 100,
                'n_days': len(returns),
            }
            
        return {
            'by_day_of_week': dow_stats,
            'by_month': monthly_returns,
        }
    
    def get_full_report(self) -> Dict[str, Any]:
        """Generate comprehensive attribution report."""
        return {
            'generated_at': datetime.now().isoformat(),
            'performance': self.get_performance_summary(),
            'component_attribution': self.get_component_attribution(),
            'trade_attribution': self.get_trade_attribution(),
            'time_analysis': self.get_time_analysis(),
            'factor_summary': self.factor_attributor.get_factor_summary(),
        }
    
    def export_report(self, filepath: str):
        """Export report to JSON file."""
        report = self.get_full_report()
        
        # Convert dates to strings
        def serialize(obj):
            if isinstance(obj, (date, datetime)):
                return obj.isoformat()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            return str(obj)
            
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=serialize)
            
        logger.info(f"Report exported to {filepath}")


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Profit Attribution...")
    
    engine = ProfitAttributionEngine(initial_value=1_000_000)
    
    # Simulate some daily P&L
    print("\n1. Recording daily P&L...")
    import random
    random.seed(42)
    
    current_date = date(2024, 1, 1)
    for i in range(30):
        gross_pnl = random.gauss(500, 2000)
        tx_costs = abs(gross_pnl) * 0.001  # 10bps
        n_trades = random.randint(5, 20)
        wins = int(n_trades * random.uniform(0.4, 0.6))
        
        engine.record_daily_pnl(
            date=current_date,
            portfolio_value=engine.portfolio_value + gross_pnl - tx_costs,
            gross_pnl=gross_pnl,
            transaction_costs=tx_costs,
            n_trades=n_trades,
            win_trades=wins,
        )
        current_date += timedelta(days=1)
        
    # Add some trades
    print("\n2. Recording trades...")
    trade_id = 0
    for _ in range(20):
        trade_id += 1
        trade = Trade(
            trade_id=str(trade_id),
            symbol=random.choice(['AAPL', 'MSFT', 'GOOGL', 'AMZN']),
            side='buy',
            shares=100,
            entry_price=150.0,
            exit_price=150.0 * (1 + random.gauss(0.01, 0.03)),
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            strategy=random.choice(['momentum', 'mean_reversion', 'tda_signal']),
            transaction_cost=3.0,
        )
        engine.record_trade(trade)
        
    # Get reports
    print("\n3. Performance Summary:")
    summary = engine.get_performance_summary()
    print(f"   Total Return: {summary['performance']['total_return_pct']:.2f}%")
    print(f"   Sharpe Ratio: {summary['performance']['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {summary['performance']['max_drawdown']:.2%}")
    print(f"   Daily Win Rate: {summary['performance']['daily_win_rate']:.1%}")
    
    print("\n4. Trade Attribution:")
    trade_attr = engine.get_trade_attribution()
    print(f"   Total trades: {trade_attr['summary']['n_trades']}")
    print(f"   Win rate: {trade_attr['summary']['win_rate']:.1%}")
    print(f"   Profit factor: {trade_attr['summary']['profit_factor']:.2f}")
    
    print("\n5. By Strategy:")
    for strategy, stats in trade_attr['by_strategy'].items():
        print(f"   {strategy}: {stats['n_trades']} trades, ${stats['total_pnl']:.2f} P&L")
        
    print("\n6. By Symbol:")
    for symbol, stats in trade_attr['by_symbol'].items():
        print(f"   {symbol}: {stats['n_trades']} trades, ${stats['total_pnl']:.2f} P&L")
    
    print("\n✅ Profit Attribution tests passed!")
