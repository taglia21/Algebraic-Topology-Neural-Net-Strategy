#!/usr/bin/env python3
"""
V36 Transaction Cost Model
==========================
Realistic transaction cost modeling for backtesting and live trading.

Cost Components:
- Slippage: 0.05% for liquid stocks (ADV > 5M), 0.15% for illiquid
- Commission: $0.005/share (configurable, Alpaca is $0 in production)
- Market Impact: sqrt(trade_size/ADV) * volatility * 0.1

Usage:
    model = TransactionCostModel()
    costs = model.calculate_costs('AAPL', 100, 150.0, 50_000_000, 0.25)
    trades_df = model.apply_to_backtest(trades_df)
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('V36_TxCosts')


@dataclass
class CostConfig:
    """Configuration for transaction cost model."""
    # Slippage parameters
    slippage_liquid: float = 0.0005      # 0.05% for liquid stocks
    slippage_illiquid: float = 0.0015    # 0.15% for illiquid stocks
    adv_threshold: float = 5_000_000     # ADV threshold for liquidity
    
    # Commission
    commission_per_share: float = 0.005  # $0.005/share for testing
    min_commission: float = 0.0          # Minimum commission per trade
    
    # Market impact (Almgren-Chriss style)
    impact_coefficient: float = 0.1      # Market impact multiplier


@dataclass
class TransactionCosts:
    """Container for calculated transaction costs."""
    slippage: float
    commission: float
    market_impact: float
    total_cost: float
    cost_bps: float  # Total cost in basis points
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'slippage': self.slippage,
            'commission': self.commission,
            'market_impact': self.market_impact,
            'total_cost': self.total_cost,
            'cost_bps': self.cost_bps
        }


class TransactionCostModel:
    """
    Transaction cost model for realistic backtesting.
    
    Models three cost components:
    1. Slippage: Fixed percentage based on liquidity tier
    2. Commission: Per-share cost (configurable)
    3. Market Impact: Square-root model based on participation rate
    
    Args:
        config: Cost configuration parameters
    
    Example:
        model = TransactionCostModel()
        costs = model.calculate_costs('AAPL', 100, 150.0, 50_000_000, 0.25)
        print(f"Total cost: ${costs['total_cost']:.2f}")
    """

    def __init__(self, config: Optional[CostConfig] = None):
        self.config = config or CostConfig()

    def _calculate_slippage(self, shares: int, price: float, adv: float) -> float:
        """
        Calculate slippage cost based on liquidity.
        
        Args:
            shares: Number of shares traded
            price: Price per share
            adv: Average daily volume
        
        Returns:
            Slippage cost in dollars
        """
        trade_value = abs(shares) * price
        
        if adv >= self.config.adv_threshold:
            slippage_rate = self.config.slippage_liquid
        else:
            slippage_rate = self.config.slippage_illiquid
        
        return trade_value * slippage_rate

    def _calculate_commission(self, shares: int) -> float:
        """
        Calculate commission cost.
        
        Args:
            shares: Number of shares traded
        
        Returns:
            Commission cost in dollars
        """
        commission = abs(shares) * self.config.commission_per_share
        return max(commission, self.config.min_commission)

    def _calculate_market_impact(
        self, shares: int, price: float, adv: float, volatility: float
    ) -> float:
        """
        Calculate market impact using square-root model.
        
        Impact = sqrt(trade_size / ADV) * volatility * coefficient
        
        Args:
            shares: Number of shares traded
            price: Price per share
            adv: Average daily volume (shares)
            volatility: Annualized volatility (decimal)
        
        Returns:
            Market impact cost in dollars
        """
        if adv <= 0:
            return 0.0
        
        trade_value = abs(shares) * price
        participation_rate = abs(shares) / adv
        
        # Square-root market impact model
        impact_rate = (
            np.sqrt(participation_rate) * 
            volatility * 
            self.config.impact_coefficient
        )
        
        return trade_value * impact_rate

    def calculate_costs(
        self,
        symbol: str,
        shares: int,
        price: float,
        adv: float,
        volatility: float
    ) -> Dict[str, float]:
        """
        Calculate all transaction costs for a trade.
        
        Args:
            symbol: Stock symbol (for logging)
            shares: Number of shares (positive=buy, negative=sell)
            price: Price per share
            adv: Average daily volume (shares)
            volatility: Annualized volatility (decimal, e.g., 0.25 for 25%)
        
        Returns:
            Dictionary with slippage, commission, market_impact, total_cost, cost_bps
        """
        if shares == 0:
            return TransactionCosts(0, 0, 0, 0, 0).to_dict()
        
        slippage = self._calculate_slippage(shares, price, adv)
        commission = self._calculate_commission(shares)
        market_impact = self._calculate_market_impact(shares, price, adv, volatility)
        
        total_cost = slippage + commission + market_impact
        trade_value = abs(shares) * price
        cost_bps = (total_cost / trade_value * 10000) if trade_value > 0 else 0
        
        costs = TransactionCosts(
            slippage=slippage,
            commission=commission,
            market_impact=market_impact,
            total_cost=total_cost,
            cost_bps=cost_bps
        )
        
        logger.debug(
            f"{symbol}: {shares} shares @ ${price:.2f} | "
            f"Costs: ${total_cost:.2f} ({cost_bps:.1f} bps)"
        )
        
        return costs.to_dict()

    def apply_to_backtest(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply transaction costs to a backtest trades DataFrame.
        
        Expected columns in trades_df:
        - symbol: Stock symbol
        - shares: Number of shares traded
        - price: Execution price
        - adv: Average daily volume (optional, defaults to 5M)
        - volatility: Annualized volatility (optional, defaults to 0.25)
        
        Adds columns:
        - slippage, commission, market_impact, total_cost, cost_bps
        - net_value: Trade value minus costs (for buys) or plus costs (for sells)
        
        Args:
            trades_df: DataFrame with trade records
        
        Returns:
            DataFrame with added cost columns
        """
        if trades_df.empty:
            return trades_df
        
        df = trades_df.copy()
        
        # Ensure required columns exist
        if 'adv' not in df.columns:
            df['adv'] = 5_000_000  # Default ADV
        if 'volatility' not in df.columns:
            df['volatility'] = 0.25  # Default 25% volatility
        
        # Calculate costs for each trade
        cost_records = []
        for _, row in df.iterrows():
            costs = self.calculate_costs(
                symbol=row.get('symbol', 'UNKNOWN'),
                shares=int(row['shares']),
                price=float(row['price']),
                adv=float(row['adv']),
                volatility=float(row['volatility'])
            )
            cost_records.append(costs)
        
        # Add cost columns
        costs_df = pd.DataFrame(cost_records)
        for col in costs_df.columns:
            df[col] = costs_df[col].values
        
        # Calculate net trade value
        df['gross_value'] = df['shares'] * df['price']
        df['net_value'] = df['gross_value'].abs() - df['total_cost']
        df.loc[df['shares'] > 0, 'net_value'] *= -1  # Buys are negative cash flow
        
        # Summary statistics
        total_costs = df['total_cost'].sum()
        avg_bps = df['cost_bps'].mean()
        logger.info(
            f"Applied costs to {len(df)} trades: "
            f"Total=${total_costs:,.2f}, Avg={avg_bps:.1f} bps"
        )
        
        return df


def main() -> None:
    """Example usage of TransactionCostModel."""
    model = TransactionCostModel()
    
    # Single trade example
    print("=" * 60)
    print("SINGLE TRADE COST CALCULATION")
    print("=" * 60)
    
    costs = model.calculate_costs(
        symbol='AAPL',
        shares=1000,
        price=150.0,
        adv=50_000_000,
        volatility=0.25
    )
    print(f"Trade: 1000 shares of AAPL @ $150")
    print(f"  Slippage:      ${costs['slippage']:.2f}")
    print(f"  Commission:    ${costs['commission']:.2f}")
    print(f"  Market Impact: ${costs['market_impact']:.2f}")
    print(f"  Total Cost:    ${costs['total_cost']:.2f} ({costs['cost_bps']:.1f} bps)")
    
    # Backtest example
    print("\n" + "=" * 60)
    print("BACKTEST APPLICATION")
    print("=" * 60)
    
    trades_df = pd.DataFrame({
        'symbol': ['AAPL', 'MSFT', 'NVDA', 'SMALL'],
        'shares': [1000, -500, 200, 5000],
        'price': [150.0, 380.0, 450.0, 25.0],
        'adv': [50_000_000, 30_000_000, 40_000_000, 500_000],
        'volatility': [0.25, 0.22, 0.45, 0.60]
    })
    
    result = model.apply_to_backtest(trades_df)
    print(result[['symbol', 'shares', 'price', 'total_cost', 'cost_bps']].to_string())


if __name__ == "__main__":
    main()
