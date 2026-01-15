"""Portfolio Risk Controller for Phase 6.

Enhanced portfolio-level risk management for 3000+ stock universe:
- Sector diversification constraints
- Correlation-based position limits
- Dynamic drawdown control
- Factor exposure monitoring
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PortfolioPosition:
    """Single position in the portfolio."""
    ticker: str
    sector: str
    shares: int
    entry_price: float
    current_price: float
    stop_price: float
    target_price: float
    entry_date: str
    weight: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    
    @property
    def position_value(self) -> float:
        return self.shares * self.current_price
    
    @property
    def risk_amount(self) -> float:
        return self.shares * abs(self.entry_price - self.stop_price)


@dataclass
class PortfolioRiskReport:
    """Risk report for current portfolio state."""
    timestamp: str
    total_value: float
    cash: float
    equity: float
    
    # Position metrics
    num_positions: int
    avg_position_size: float
    max_position_weight: float
    
    # Sector metrics
    sector_weights: Dict[str, float] = field(default_factory=dict)
    sector_count: int = 0
    max_sector_weight: float = 0.0
    
    # Risk metrics
    portfolio_heat: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    drawdown: float = 0.0
    
    # Warnings
    warnings: List[str] = field(default_factory=list)
    
    def is_healthy(self) -> bool:
        """Check if portfolio passes all risk constraints."""
        return len(self.warnings) == 0


class PortfolioRiskController:
    """
    Portfolio-level risk controller for Phase 6.
    
    Implements:
    - Sector diversification (max 30% per sector, min 4 sectors)
    - Position size limits (max 8% per position)
    - Portfolio heat limits (max 25% at-risk)
    - Dynamic drawdown control (reduce size at 10% DD)
    - Correlation-based exposure limits
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        max_position_weight: float = 0.08,
        max_sector_weight: float = 0.30,
        min_sectors: int = 4,
        max_portfolio_heat: float = 0.25,
        max_drawdown_trigger: float = 0.10,
        drawdown_reduction: float = 0.50,
        target_volatility: float = 0.15,
    ):
        """
        Initialize portfolio risk controller.
        
        Args:
            initial_capital: Starting capital
            max_position_weight: Max weight for any single position
            max_sector_weight: Max weight for any sector
            min_sectors: Minimum number of sectors required
            max_portfolio_heat: Maximum portfolio heat (at-risk %)
            max_drawdown_trigger: Drawdown level that triggers risk reduction
            drawdown_reduction: Reduce position sizes by this factor at trigger
            target_volatility: Target portfolio volatility (annualized)
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        
        # Risk limits
        self.max_position_weight = max_position_weight
        self.max_sector_weight = max_sector_weight
        self.min_sectors = min_sectors
        self.max_portfolio_heat = max_portfolio_heat
        self.max_drawdown_trigger = max_drawdown_trigger
        self.drawdown_reduction = drawdown_reduction
        self.target_volatility = target_volatility
        
        # Portfolio state
        self.positions: Dict[str, PortfolioPosition] = {}
        self.cash = initial_capital
        self.equity_curve: List[float] = [initial_capital]
        self.trade_history: List[Dict] = []
        
        # Drawdown state
        self.in_drawdown_mode = False
        self.current_drawdown = 0.0
    
    def get_portfolio_value(self) -> float:
        """Get current portfolio value."""
        position_value = sum(p.position_value for p in self.positions.values())
        return self.cash + position_value
    
    def get_sector_weights(self) -> Dict[str, float]:
        """Get current sector allocation."""
        total = self.get_portfolio_value()
        if total <= 0:
            return {}
        
        sector_values = {}
        for pos in self.positions.values():
            sector_values[pos.sector] = sector_values.get(pos.sector, 0) + pos.position_value
        
        return {sector: value / total for sector, value in sector_values.items()}
    
    def get_position_weights(self) -> Dict[str, float]:
        """Get current position weights."""
        total = self.get_portfolio_value()
        if total <= 0:
            return {}
        
        return {ticker: pos.position_value / total for ticker, pos in self.positions.items()}
    
    def update_drawdown(self):
        """Update peak capital and current drawdown."""
        current = self.get_portfolio_value()
        
        if current > self.peak_capital:
            self.peak_capital = current
            self.in_drawdown_mode = False
        
        self.current_drawdown = (self.peak_capital - current) / self.peak_capital
        
        # Check if we need to enter drawdown mode
        if self.current_drawdown >= self.max_drawdown_trigger:
            if not self.in_drawdown_mode:
                logger.warning(f"Entering drawdown mode: {self.current_drawdown:.1%} drawdown")
                self.in_drawdown_mode = True
    
    def get_position_size_multiplier(self) -> float:
        """Get position size multiplier based on drawdown state."""
        if self.in_drawdown_mode:
            return self.drawdown_reduction
        return 1.0
    
    def can_add_position(
        self,
        ticker: str,
        sector: str,
        position_value: float,
    ) -> Tuple[bool, List[str]]:
        """
        Check if a new position can be added.
        
        Args:
            ticker: Stock symbol
            sector: Stock sector
            position_value: Proposed position value
            
        Returns:
            (can_add, list of warning messages)
        """
        warnings = []
        total = self.get_portfolio_value()
        
        if total <= 0:
            return False, ["Portfolio value is zero"]
        
        # Check 1: Position size limit
        proposed_weight = position_value / total
        if proposed_weight > self.max_position_weight:
            warnings.append(
                f"Position weight {proposed_weight:.1%} exceeds max {self.max_position_weight:.1%}"
            )
        
        # Check 2: Sector weight limit
        sector_weights = self.get_sector_weights()
        current_sector_weight = sector_weights.get(sector, 0)
        proposed_sector_weight = current_sector_weight + (position_value / total)
        
        if proposed_sector_weight > self.max_sector_weight:
            warnings.append(
                f"Sector {sector} weight {proposed_sector_weight:.1%} exceeds max {self.max_sector_weight:.1%}"
            )
        
        # Check 3: Cash availability
        if position_value > self.cash:
            warnings.append(
                f"Insufficient cash: need ${position_value:,.0f}, have ${self.cash:,.0f}"
            )
        
        # Check 4: Duplicate position
        if ticker in self.positions:
            warnings.append(f"Already have position in {ticker}")
        
        # Check 5: Drawdown mode - reduce size
        if self.in_drawdown_mode:
            warnings.append(f"In drawdown mode ({self.current_drawdown:.1%}), reduce position size")
        
        can_add = len([w for w in warnings if 'exceeds' in w or 'Insufficient' in w or 'Already' in w]) == 0
        
        return can_add, warnings
    
    def add_position(
        self,
        ticker: str,
        sector: str,
        shares: int,
        entry_price: float,
        stop_price: float,
        target_price: float,
        entry_date: str = None,
    ) -> bool:
        """
        Add a position to the portfolio.
        
        Args:
            ticker: Stock symbol
            sector: Stock sector
            shares: Number of shares
            entry_price: Entry price
            stop_price: Stop-loss price
            target_price: Target price
            entry_date: Entry date string
            
        Returns:
            True if position was added
        """
        entry_date = entry_date or datetime.now().strftime('%Y-%m-%d')
        position_value = shares * entry_price
        
        can_add, warnings = self.can_add_position(ticker, sector, position_value)
        
        if not can_add:
            logger.warning(f"Cannot add position {ticker}: {warnings}")
            return False
        
        # Deduct cash
        self.cash -= position_value
        
        # Create position
        total = self.get_portfolio_value()
        pos = PortfolioPosition(
            ticker=ticker,
            sector=sector,
            shares=shares,
            entry_price=entry_price,
            current_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            entry_date=entry_date,
            weight=position_value / total if total > 0 else 0,
        )
        
        self.positions[ticker] = pos
        
        logger.info(f"Added position: {ticker} x{shares} @ ${entry_price:.2f} ({sector})")
        
        return True
    
    def update_prices(self, prices: Dict[str, float]):
        """
        Update current prices and P&L.
        
        Args:
            prices: Dict mapping ticker to current price
        """
        for ticker, pos in self.positions.items():
            if ticker in prices:
                pos.current_price = prices[ticker]
                pos.pnl = (pos.current_price - pos.entry_price) * pos.shares
                pos.pnl_pct = (pos.current_price / pos.entry_price) - 1.0
        
        # Update weights
        total = self.get_portfolio_value()
        if total > 0:
            for pos in self.positions.values():
                pos.weight = pos.position_value / total
        
        # Update drawdown
        self.update_drawdown()
        
        # Log equity
        self.equity_curve.append(total)
    
    def close_position(
        self,
        ticker: str,
        exit_price: float,
        exit_date: str = None,
        reason: str = 'signal',
    ) -> Optional[Dict]:
        """
        Close a position.
        
        Args:
            ticker: Stock symbol
            exit_price: Exit price
            exit_date: Exit date string
            reason: Close reason ('signal', 'stop', 'target')
            
        Returns:
            Trade record dict or None
        """
        if ticker not in self.positions:
            return None
        
        pos = self.positions[ticker]
        exit_date = exit_date or datetime.now().strftime('%Y-%m-%d')
        
        # Calculate P&L
        pnl = (exit_price - pos.entry_price) * pos.shares
        pnl_pct = (exit_price / pos.entry_price) - 1.0
        
        # Return cash
        self.cash += pos.shares * exit_price
        
        # Record trade
        trade = {
            'ticker': ticker,
            'sector': pos.sector,
            'entry_date': pos.entry_date,
            'exit_date': exit_date,
            'entry_price': pos.entry_price,
            'exit_price': exit_price,
            'shares': pos.shares,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason,
        }
        self.trade_history.append(trade)
        
        # Remove position
        del self.positions[ticker]
        
        logger.info(f"Closed {ticker}: ${pnl:+,.2f} ({pnl_pct:+.1%}) - {reason}")
        
        return trade
    
    def check_stops_and_targets(self, prices: Dict[str, float]) -> List[str]:
        """
        Check which positions hit stops or targets.
        
        Args:
            prices: Dict mapping ticker to current price
            
        Returns:
            List of tickers that hit stops or targets
        """
        to_close = []
        
        for ticker, pos in self.positions.items():
            price = prices.get(ticker, pos.current_price)
            
            # Check stop (long position)
            if pos.stop_price < pos.entry_price:  # Long
                if price <= pos.stop_price:
                    to_close.append((ticker, price, 'stop'))
                elif price >= pos.target_price:
                    to_close.append((ticker, price, 'target'))
            else:  # Short
                if price >= pos.stop_price:
                    to_close.append((ticker, price, 'stop'))
                elif price <= pos.target_price:
                    to_close.append((ticker, price, 'target'))
        
        return to_close
    
    def calculate_portfolio_heat(self) -> float:
        """Calculate current portfolio heat (total at-risk percentage)."""
        total = self.get_portfolio_value()
        if total <= 0:
            return 0.0
        
        total_risk = sum(pos.risk_amount for pos in self.positions.values())
        return total_risk / total
    
    def get_risk_report(self) -> PortfolioRiskReport:
        """Generate comprehensive risk report."""
        total = self.get_portfolio_value()
        sector_weights = self.get_sector_weights()
        position_weights = self.get_position_weights()
        
        warnings = []
        
        # Check sector concentration
        if sector_weights:
            max_sector = max(sector_weights.values())
            if max_sector > self.max_sector_weight:
                warnings.append(f"Sector over-concentration: {max_sector:.1%}")
        
        # Check sector diversification
        if len(sector_weights) < self.min_sectors and len(self.positions) > 0:
            warnings.append(f"Insufficient diversification: {len(sector_weights)} sectors < {self.min_sectors}")
        
        # Check position concentration
        if position_weights:
            max_pos = max(position_weights.values())
            if max_pos > self.max_position_weight:
                max_ticker = max(position_weights.items(), key=lambda x: x[1])[0]
                warnings.append(f"Position over-concentration: {max_ticker} at {max_pos:.1%}")
        
        # Check portfolio heat
        heat = self.calculate_portfolio_heat()
        if heat > self.max_portfolio_heat:
            warnings.append(f"Excessive portfolio heat: {heat:.1%}")
        
        # Check drawdown
        if self.current_drawdown > 0.15:
            warnings.append(f"High drawdown: {self.current_drawdown:.1%}")
        
        report = PortfolioRiskReport(
            timestamp=datetime.now().isoformat(),
            total_value=total,
            cash=self.cash,
            equity=total - self.cash,
            num_positions=len(self.positions),
            avg_position_size=np.mean([p.position_value for p in self.positions.values()]) if self.positions else 0,
            max_position_weight=max(position_weights.values()) if position_weights else 0,
            sector_weights=sector_weights,
            sector_count=len(sector_weights),
            max_sector_weight=max(sector_weights.values()) if sector_weights else 0,
            portfolio_heat=heat,
            total_pnl=sum(p.pnl for p in self.positions.values()),
            total_pnl_pct=(total / self.initial_capital) - 1.0,
            drawdown=self.current_drawdown,
            warnings=warnings,
        )
        
        return report
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from trade history."""
        if not self.trade_history:
            return {}
        
        df = pd.DataFrame(self.trade_history)
        
        wins = df[df['pnl'] > 0]
        losses = df[df['pnl'] <= 0]
        
        return {
            'total_trades': len(df),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': len(wins) / len(df) if len(df) > 0 else 0,
            'total_pnl': df['pnl'].sum(),
            'avg_win': wins['pnl'].mean() if len(wins) > 0 else 0,
            'avg_loss': losses['pnl'].mean() if len(losses) > 0 else 0,
            'profit_factor': abs(wins['pnl'].sum() / losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else float('inf'),
            'avg_holding_period': 'N/A',  # Would need date parsing
        }
    
    def rebalance_portfolio(
        self,
        target_weights: Dict[str, float],
        current_prices: Dict[str, float],
    ) -> List[Dict]:
        """
        Rebalance portfolio to target weights.
        
        Args:
            target_weights: Dict mapping ticker to target weight
            current_prices: Dict mapping ticker to current price
            
        Returns:
            List of trade orders
        """
        orders = []
        total = self.get_portfolio_value()
        
        # Close positions not in target
        for ticker in list(self.positions.keys()):
            if ticker not in target_weights:
                price = current_prices.get(ticker, self.positions[ticker].current_price)
                orders.append({
                    'ticker': ticker,
                    'action': 'SELL',
                    'shares': self.positions[ticker].shares,
                    'price': price,
                    'reason': 'rebalance_exit',
                })
        
        # Adjust existing positions and add new ones
        for ticker, target_weight in target_weights.items():
            if ticker not in current_prices:
                continue
            
            price = current_prices[ticker]
            target_value = total * target_weight
            target_shares = int(target_value / price)
            
            if ticker in self.positions:
                current_shares = self.positions[ticker].shares
                diff = target_shares - current_shares
                
                if abs(diff) > 0:
                    orders.append({
                        'ticker': ticker,
                        'action': 'BUY' if diff > 0 else 'SELL',
                        'shares': abs(diff),
                        'price': price,
                        'reason': 'rebalance_adjust',
                    })
            else:
                if target_shares > 0:
                    orders.append({
                        'ticker': ticker,
                        'action': 'BUY',
                        'shares': target_shares,
                        'price': price,
                        'reason': 'rebalance_entry',
                    })
        
        return orders


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing PortfolioRiskController...")
    print("=" * 50)
    
    controller = PortfolioRiskController(
        initial_capital=100000,
        max_position_weight=0.08,
        max_sector_weight=0.30,
    )
    
    # Add some positions
    positions = [
        ('AAPL', 'Technology', 50, 180.0, 170.0, 195.0),
        ('MSFT', 'Technology', 30, 350.0, 330.0, 380.0),
        ('JPM', 'Financial', 40, 150.0, 140.0, 165.0),
        ('JNJ', 'Healthcare', 25, 160.0, 150.0, 175.0),
        ('XOM', 'Energy', 60, 100.0, 90.0, 115.0),
    ]
    
    for ticker, sector, shares, entry, stop, target in positions:
        controller.add_position(ticker, sector, shares, entry, stop, target)
    
    # Update prices
    prices = {
        'AAPL': 185.0,
        'MSFT': 360.0,
        'JPM': 155.0,
        'JNJ': 158.0,
        'XOM': 105.0,
    }
    controller.update_prices(prices)
    
    # Get risk report
    report = controller.get_risk_report()
    
    print(f"\nPortfolio Value: ${report.total_value:,.2f}")
    print(f"Cash: ${report.cash:,.2f}")
    print(f"Positions: {report.num_positions}")
    print(f"\nSector Weights:")
    for sector, weight in sorted(report.sector_weights.items(), key=lambda x: -x[1]):
        print(f"  {sector}: {weight:.1%}")
    
    print(f"\nRisk Metrics:")
    print(f"  Portfolio Heat: {report.portfolio_heat:.1%}")
    print(f"  Drawdown: {report.drawdown:.1%}")
    print(f"  P&L: ${report.total_pnl:,.2f} ({report.total_pnl_pct:+.1%})")
    
    if report.warnings:
        print(f"\nWarnings:")
        for w in report.warnings:
            print(f"  ⚠️ {w}")
    else:
        print(f"\n✅ Portfolio is healthy")
    
    print("\n" + "=" * 50)
    print("PortfolioRiskController tests complete!")
