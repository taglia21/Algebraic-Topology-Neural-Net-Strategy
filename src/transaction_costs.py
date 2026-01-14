"""Transaction Cost Model for realistic trading cost estimation.

Implements commission, bid-ask spread, and slippage modeling to eliminate
optimistic bias in backtesting results.

Version 1.0: Full cost model with volatility-adjusted slippage.
"""

import os
import csv
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CostModel:
    """
    Transaction cost model for realistic trading simulation.
    
    Implements three cost components:
    1. Commission: Fixed + per-share fees
    2. Spread: Bid-ask spread cost based on typical market conditions
    3. Slippage: Market impact cost that scales with volatility
    
    Default Parameters (Interactive Brokers-like):
    - Commission: $1.00 per trade + $0.005 per share, min $1.00
    - Spread: 5 bps (basis points) = 0.05%
    - Slippage: 3 bps base, scaled by volatility
    """
    
    def __init__(
        self,
        commission_per_trade: float = 1.0,
        commission_per_share: float = 0.005,
        min_commission: float = 1.0,
        bid_ask_spread_bps: float = 5.0,
        slippage_bps: float = 3.0,
        log_path: str = '/workspaces/Algebraic-Topology-Neural-Net-Strategy/results/transaction_costs.csv'
    ):
        """
        Initialize cost model with configurable parameters.
        
        Args:
            commission_per_trade: Fixed commission per trade ($)
            commission_per_share: Per-share commission ($)
            min_commission: Minimum commission per trade ($)
            bid_ask_spread_bps: Bid-ask spread in basis points
            slippage_bps: Base slippage in basis points
            log_path: Path to save transaction cost log
        """
        self.commission_per_trade = commission_per_trade
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission
        self.bid_ask_spread_bps = bid_ask_spread_bps
        self.slippage_bps = slippage_bps
        self.log_path = log_path
        
        # Cumulative cost tracking
        self.total_costs = 0.0
        self.total_commission = 0.0
        self.total_spread_costs = 0.0
        self.total_slippage = 0.0
        self.num_trades = 0
        
        # Initialize log file
        self._init_log_file()
        
        logger.info(f"CostModel initialized: comm=${commission_per_trade}+${commission_per_share}/sh, "
                   f"spread={bid_ask_spread_bps}bps, slippage={slippage_bps}bps")
    
    def _init_log_file(self):
        """Initialize transaction cost log CSV with headers."""
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'date', 'ticker', 'trade_type', 'shares', 'price',
                    'atr', 'volatility_factor', 'commission', 'spread_cost',
                    'slippage', 'total_cost', 'fill_price'
                ])
    
    def _log_transaction(self, date: str, ticker: str, trade_type: str, shares: int,
                         price: float, atr: float, vol_factor: float, commission: float,
                         spread_cost: float, slippage: float, total_cost: float,
                         fill_price: float):
        """Log a single transaction's costs to CSV."""
        try:
            with open(self.log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    date,
                    ticker,
                    trade_type,
                    shares,
                    round(price, 4),
                    round(atr, 4),
                    round(vol_factor, 4),
                    round(commission, 4),
                    round(spread_cost, 4),
                    round(slippage, 4),
                    round(total_cost, 4),
                    round(fill_price, 4)
                ])
        except Exception as e:
            logger.warning(f"Failed to log transaction: {e}")
    
    def calculate_total_cost(
        self,
        shares: int,
        price: float,
        atr: float = 0.0,
        trade_type: str = 'entry',
        ticker: str = '',
        date: str = ''
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate total transaction cost for a trade.
        
        Formula:
        - Commission = max(min_commission, commission_per_trade + shares * commission_per_share)
        - Spread cost = shares * price * (bid_ask_spread_bps / 10000)
        - Slippage = shares * price * (slippage_bps / 10000) * volatility_factor
        - Volatility factor = min(2.0, atr / price * 100)
        - Total = commission + spread + slippage
        
        Args:
            shares: Number of shares traded
            price: Trade price
            atr: Average True Range for volatility adjustment
            trade_type: 'entry' or 'exit'
            ticker: Symbol for logging
            date: Date string for logging
            
        Returns:
            Tuple of (total_cost, breakdown_dict)
        """
        if shares <= 0 or price <= 0:
            return 0.0, {'commission': 0, 'spread': 0, 'slippage': 0, 'total': 0}
        
        # 1. Commission calculation
        commission = self.commission_per_trade + (shares * self.commission_per_share)
        commission = max(self.min_commission, commission)
        
        # 2. Spread cost (half spread paid on each side)
        notional_value = shares * price
        spread_cost = notional_value * (self.bid_ask_spread_bps / 10000.0)
        
        # 3. Slippage with volatility adjustment
        # Volatility factor: higher ATR relative to price = more slippage
        if atr > 0 and price > 0:
            volatility_factor = min(2.0, (atr / price) * 100.0)
        else:
            volatility_factor = 1.0  # Default factor if no ATR
        
        slippage = notional_value * (self.slippage_bps / 10000.0) * volatility_factor
        
        # Total cost
        total_cost = commission + spread_cost + slippage
        
        # Update cumulative tracking
        self.total_costs += total_cost
        self.total_commission += commission
        self.total_spread_costs += spread_cost
        self.total_slippage += slippage
        self.num_trades += 1
        
        # Log transaction
        fill_price = self.adjust_fill_price(price, total_cost, shares, trade_type)
        self._log_transaction(
            date, ticker, trade_type, shares, price, atr, volatility_factor,
            commission, spread_cost, slippage, total_cost, fill_price
        )
        
        breakdown = {
            'commission': round(commission, 4),
            'spread': round(spread_cost, 4),
            'slippage': round(slippage, 4),
            'total': round(total_cost, 4),
            'volatility_factor': round(volatility_factor, 4),
            'notional_value': round(notional_value, 2)
        }
        
        return total_cost, breakdown
    
    def adjust_fill_price(
        self,
        price: float,
        cost: float,
        shares: int,
        direction: str
    ) -> float:
        """
        Adjust fill price to account for transaction costs.
        
        Args:
            price: Market price
            cost: Total transaction cost
            shares: Number of shares
            direction: 'buy'/'entry' (price moves up) or 'sell'/'exit' (price moves down)
            
        Returns:
            Adjusted fill price
        """
        if shares <= 0:
            return price
        
        cost_per_share = cost / shares
        
        # Buy/entry: we pay more than market price
        # Sell/exit: we receive less than market price
        if direction.lower() in ['buy', 'entry']:
            adjusted = price + cost_per_share
        else:
            adjusted = price - cost_per_share
        
        return round(adjusted, 4)
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """
        Get cumulative cost summary.
        
        Returns:
            Dict with total costs by category
        """
        return {
            'total_costs': round(self.total_costs, 2),
            'total_commission': round(self.total_commission, 2),
            'total_spread_costs': round(self.total_spread_costs, 2),
            'total_slippage': round(self.total_slippage, 2),
            'num_trades': self.num_trades,
            'avg_cost_per_trade': round(self.total_costs / self.num_trades, 2) if self.num_trades > 0 else 0
        }
    
    def get_cost_as_pct_of_pnl(self, gross_pnl: float) -> float:
        """
        Calculate costs as percentage of gross P&L.
        
        Args:
            gross_pnl: Gross profit/loss before costs
            
        Returns:
            Costs as percentage of gross P&L (can be >100% if costs exceed profits)
        """
        if abs(gross_pnl) < 0.01:
            return 0.0 if self.total_costs < 0.01 else float('inf')
        
        return round((self.total_costs / abs(gross_pnl)) * 100, 2)
    
    def reset(self):
        """Reset cumulative cost tracking."""
        self.total_costs = 0.0
        self.total_commission = 0.0
        self.total_spread_costs = 0.0
        self.total_slippage = 0.0
        self.num_trades = 0


class CostScenario:
    """
    Predefined cost scenarios for sensitivity analysis.
    """
    
    @staticmethod
    def low_cost() -> Dict[str, float]:
        """Low cost scenario (discount broker, liquid ETFs)."""
        return {
            'commission_per_trade': 0.0,
            'commission_per_share': 0.005,
            'min_commission': 0.0,
            'bid_ask_spread_bps': 3.0,
            'slippage_bps': 2.0
        }
    
    @staticmethod
    def baseline() -> Dict[str, float]:
        """Baseline scenario (typical retail conditions)."""
        return {
            'commission_per_trade': 1.0,
            'commission_per_share': 0.005,
            'min_commission': 1.0,
            'bid_ask_spread_bps': 5.0,
            'slippage_bps': 3.0
        }
    
    @staticmethod
    def high_cost() -> Dict[str, float]:
        """High cost scenario (less liquid instruments)."""
        return {
            'commission_per_trade': 5.0,
            'commission_per_share': 0.01,
            'min_commission': 5.0,
            'bid_ask_spread_bps': 10.0,
            'slippage_bps': 5.0
        }
    
    @staticmethod
    def extreme() -> Dict[str, float]:
        """Extreme cost scenario (stress test)."""
        return {
            'commission_per_trade': 10.0,
            'commission_per_share': 0.02,
            'min_commission': 10.0,
            'bid_ask_spread_bps': 20.0,
            'slippage_bps': 10.0
        }
    
    @staticmethod
    def get_all_scenarios() -> Dict[str, Dict[str, float]]:
        """Get all predefined scenarios."""
        return {
            'low_cost': CostScenario.low_cost(),
            'baseline': CostScenario.baseline(),
            'high_cost': CostScenario.high_cost(),
            'extreme': CostScenario.extreme()
        }


# Convenience alias for direct import
COST_SCENARIOS = CostScenario.get_all_scenarios()


def estimate_roundtrip_cost(
    entry_price: float,
    shares: int,
    atr: float,
    cost_model: CostModel = None
) -> Dict[str, float]:
    """
    Estimate roundtrip (entry + exit) transaction costs.
    
    Args:
        entry_price: Entry price
        shares: Position size
        atr: ATR for volatility adjustment
        cost_model: CostModel instance (uses baseline if None)
        
    Returns:
        Dict with roundtrip cost estimate
    """
    if cost_model is None:
        cost_model = CostModel(**CostScenario.baseline())
    
    # Entry costs
    entry_cost, entry_breakdown = cost_model.calculate_total_cost(
        shares, entry_price, atr, 'entry'
    )
    
    # Exit costs (assume same price for estimate)
    exit_cost, exit_breakdown = cost_model.calculate_total_cost(
        shares, entry_price, atr, 'exit'
    )
    
    roundtrip_cost = entry_cost + exit_cost
    notional = shares * entry_price
    
    return {
        'entry_cost': round(entry_cost, 2),
        'exit_cost': round(exit_cost, 2),
        'roundtrip_cost': round(roundtrip_cost, 2),
        'cost_bps': round((roundtrip_cost / notional) * 10000, 2) if notional > 0 else 0,
        'breakeven_move_pct': round((roundtrip_cost / notional) * 100, 4) if notional > 0 else 0
    }


def test_cost_model():
    """Unit tests for CostModel."""
    print("\n" + "=" * 60)
    print("Testing CostModel")
    print("=" * 60)
    
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = os.path.join(tmpdir, "test_costs.csv")
        cm = CostModel(log_path=log_path)
        
        # Test 1: Basic cost calculation
        print("\nTest 1: Basic cost calculation")
        cost, breakdown = cm.calculate_total_cost(
            shares=100, price=100.0, atr=2.0, trade_type='entry',
            ticker='SPY', date='2024-01-01'
        )
        print(f"  100 shares @ $100, ATR=2.0")
        print(f"  Breakdown: {breakdown}")
        assert cost > 0, "Cost should be positive"
        assert breakdown['commission'] >= 1.0, "Commission should be >= $1.00"
        print("  ✓ Cost calculation working")
        
        # Test 2: Zero shares
        print("\nTest 2: Zero shares")
        cost, breakdown = cm.calculate_total_cost(
            shares=0, price=100.0, atr=2.0, trade_type='entry'
        )
        assert cost == 0, "Zero shares should have zero cost"
        print("  ✓ Zero shares handled correctly")
        
        # Test 3: Fill price adjustment
        print("\nTest 3: Fill price adjustment")
        buy_fill = cm.adjust_fill_price(100.0, 10.0, 100, 'buy')
        sell_fill = cm.adjust_fill_price(100.0, 10.0, 100, 'sell')
        print(f"  Market: $100, Cost: $10, Shares: 100")
        print(f"  Buy fill: ${buy_fill}, Sell fill: ${sell_fill}")
        assert buy_fill > 100.0, "Buy fill should be higher than market"
        assert sell_fill < 100.0, "Sell fill should be lower than market"
        print("  ✓ Fill price adjustment correct")
        
        # Test 4: Volatility scaling
        print("\nTest 4: Volatility scaling")
        cm_test = CostModel(log_path=os.path.join(tmpdir, "test2.csv"))
        
        # Low volatility
        cost_low, _ = cm_test.calculate_total_cost(
            shares=100, price=100.0, atr=0.5, trade_type='entry'
        )
        
        # High volatility
        cost_high, _ = cm_test.calculate_total_cost(
            shares=100, price=100.0, atr=5.0, trade_type='entry'
        )
        
        print(f"  Low ATR (0.5): ${cost_low:.4f}")
        print(f"  High ATR (5.0): ${cost_high:.4f}")
        assert cost_high > cost_low, "Higher ATR should mean higher costs"
        print("  ✓ Volatility scaling working")
        
        # Test 5: Cost summary
        print("\nTest 5: Cost summary")
        summary = cm.get_cost_summary()
        print(f"  Summary: {summary}")
        assert summary['num_trades'] >= 1
        assert summary['total_costs'] > 0
        print("  ✓ Cost summary working")
        
        # Test 6: Scenarios
        print("\nTest 6: Cost scenarios")
        scenarios = CostScenario.get_all_scenarios()
        for name, params in scenarios.items():
            cm_scenario = CostModel(**params, log_path=os.path.join(tmpdir, f"{name}.csv"))
            cost, _ = cm_scenario.calculate_total_cost(
                shares=100, price=100.0, atr=2.0, trade_type='entry'
            )
            print(f"  {name:12s}: ${cost:.4f}")
        print("  ✓ All scenarios working")
        
        # Test 7: Roundtrip estimate
        print("\nTest 7: Roundtrip cost estimate")
        estimate = estimate_roundtrip_cost(100.0, 100, 2.0)
        print(f"  Roundtrip estimate: {estimate}")
        assert estimate['roundtrip_cost'] > 0
        assert estimate['breakeven_move_pct'] > 0
        print("  ✓ Roundtrip estimation working")
    
    print("\n" + "=" * 60)
    print("All CostModel tests passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    test_cost_model()
