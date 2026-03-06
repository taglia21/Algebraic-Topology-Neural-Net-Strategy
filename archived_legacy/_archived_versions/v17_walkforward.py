#!/usr/bin/env python3
"""
V17.0 Walk-Forward Backtesting Engine
======================================
Realistic backtesting with:
- 12-month training, 3-month testing
- Rolling monthly (expanding window optional)
- Transaction costs: 10bps roundtrip
- Slippage: 5-20bps based on ADV
- Position limits and risk management

Targets:
- Sharpe: 1.5-3.0 (realistic)
- CAGR: 25-50%
- MaxDD: -15% to -25%

Red flags:
- Sharpe >5.0 = overfit
- CAGR >100% = overfit
"""

import os
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('V17_WalkForward')


@dataclass
class BacktestConfig:
    """Configuration for walk-forward backtest"""
    train_months: int = 12
    test_months: int = 3
    roll_months: int = 1          # Roll forward by 1 month
    expanding_window: bool = False # If True, training window expands
    
    # Transaction costs
    commission_bps: float = 5.0    # 5 bps per side
    slippage_bps_base: float = 5.0 # Base slippage
    slippage_bps_max: float = 20.0 # Max slippage for illiquid
    
    # Position limits
    max_position_size: float = 0.05   # 5% max per position
    max_gross_exposure: float = 2.0    # 200% gross max
    max_net_exposure: float = 1.0      # 100% net max
    
    # Risk management
    vol_target: float = 0.15           # 15% annual vol target
    max_drawdown_stop: float = 0.20    # Stop at 20% drawdown
    
    # Portfolio
    initial_capital: float = 1_000_000
    min_trade_value: float = 1000      # Minimum trade size


@dataclass
class Trade:
    """Represents a single trade"""
    date: str
    symbol: str
    side: str          # 'BUY' or 'SELL'
    shares: float
    price: float
    value: float
    commission: float
    slippage: float
    total_cost: float


@dataclass
class Position:
    """Represents a position"""
    symbol: str
    shares: float
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    weight: float


@dataclass 
class WalkForwardFold:
    """A single walk-forward fold"""
    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_days: int
    test_days: int
    metrics: Dict[str, float] = field(default_factory=dict)


class TransactionCostModel:
    """Models transaction costs including commission and slippage"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
    
    def calculate_costs(
        self,
        trade_value: float,
        adv: float,  # Average daily volume in dollars
        is_buy: bool
    ) -> Tuple[float, float]:
        """
        Calculate commission and slippage for a trade.
        
        Returns:
            Tuple of (commission, slippage)
        """
        # Commission: fixed bps
        commission = abs(trade_value) * self.config.commission_bps / 10000
        
        # Slippage: based on participation rate
        if adv > 0:
            participation = abs(trade_value) / adv
            # Slippage increases with participation rate
            slippage_bps = self.config.slippage_bps_base + \
                          (self.config.slippage_bps_max - self.config.slippage_bps_base) * \
                          min(participation * 10, 1.0)  # Cap at max
        else:
            slippage_bps = self.config.slippage_bps_max
        
        slippage = abs(trade_value) * slippage_bps / 10000
        
        return commission, slippage


class PortfolioTracker:
    """Tracks portfolio state during backtest"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.cash = config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.daily_values: List[Dict] = []
        self.cost_model = TransactionCostModel(config)
    
    def get_portfolio_value(self) -> float:
        """Get total portfolio value"""
        position_value = sum(p.market_value for p in self.positions.values())
        return self.cash + position_value
    
    def get_gross_exposure(self) -> float:
        """Get gross exposure (sum of absolute position values)"""
        pv = self.get_portfolio_value()
        if pv == 0:
            return 0
        position_value = sum(abs(p.market_value) for p in self.positions.values())
        return position_value / pv
    
    def update_prices(self, prices: Dict[str, float]):
        """Update position prices"""
        for symbol, pos in self.positions.items():
            if symbol in prices:
                pos.current_price = prices[symbol]
                pos.market_value = pos.shares * pos.current_price
                pos.unrealized_pnl = pos.market_value - (pos.shares * pos.avg_cost)
                pos.weight = pos.market_value / max(self.get_portfolio_value(), 1)
    
    def execute_trade(
        self,
        date: str,
        symbol: str,
        target_shares: float,
        price: float,
        adv: float
    ) -> Optional[Trade]:
        """
        Execute a trade to reach target shares.
        """
        current_shares = self.positions.get(symbol, Position(symbol, 0, 0, 0, 0, 0, 0)).shares
        delta_shares = target_shares - current_shares
        
        if abs(delta_shares) < 1:  # Minimum 1 share
            return None
        
        trade_value = delta_shares * price
        
        if abs(trade_value) < self.config.min_trade_value:
            return None
        
        is_buy = delta_shares > 0
        commission, slippage = self.cost_model.calculate_costs(trade_value, adv, is_buy)
        
        # Total cost (always negative impact)
        total_cost = abs(commission) + abs(slippage)
        
        # Update cash
        self.cash -= trade_value + total_cost
        
        # Update position
        if symbol in self.positions:
            pos = self.positions[symbol]
            if target_shares == 0:
                del self.positions[symbol]
            else:
                if is_buy:
                    # Average up
                    total_cost_basis = pos.shares * pos.avg_cost + delta_shares * price
                    pos.avg_cost = total_cost_basis / target_shares
                pos.shares = target_shares
                pos.current_price = price
                pos.market_value = target_shares * price
        else:
            if target_shares != 0:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    shares=target_shares,
                    avg_cost=price,
                    current_price=price,
                    market_value=target_shares * price,
                    unrealized_pnl=0,
                    weight=0
                )
        
        trade = Trade(
            date=date,
            symbol=symbol,
            side='BUY' if is_buy else 'SELL',
            shares=abs(delta_shares),
            price=price,
            value=trade_value,
            commission=commission,
            slippage=slippage,
            total_cost=total_cost
        )
        
        self.trades.append(trade)
        return trade
    
    def record_daily_value(self, date: str):
        """Record daily portfolio value"""
        pv = self.get_portfolio_value()
        self.daily_values.append({
            'date': date,
            'portfolio_value': pv,
            'cash': self.cash,
            'position_value': pv - self.cash,
            'n_positions': len(self.positions),
            'gross_exposure': self.get_gross_exposure()
        })


class WalkForwardEngine:
    """
    Walk-forward backtesting engine with realistic transaction costs.
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.folds: List[WalkForwardFold] = []
        self.all_trades: List[Trade] = []
        self.daily_equity: pd.DataFrame = None
        self.final_metrics: Dict[str, float] = {}
    
    def create_folds(
        self,
        dates: pd.DatetimeIndex,
        train_months: int = None,
        test_months: int = None,
        roll_months: int = None
    ) -> List[WalkForwardFold]:
        """
        Create walk-forward folds.
        """
        train_months = train_months or self.config.train_months
        test_months = test_months or self.config.test_months
        roll_months = roll_months or self.config.roll_months
        
        dates = pd.to_datetime(dates).sort_values()
        start_date = dates.min()
        end_date = dates.max()
        
        folds = []
        fold_id = 0
        
        current_train_start = start_date
        
        while True:
            train_end = current_train_start + pd.DateOffset(months=train_months)
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=test_months)
            
            # Check if test period is within data
            if test_end > end_date:
                # Truncate last fold
                test_end = end_date
                if test_start >= test_end:
                    break
            
            # Get actual trading days
            train_mask = (dates >= current_train_start) & (dates < train_end)
            test_mask = (dates >= test_start) & (dates < test_end)
            
            train_days = train_mask.sum()
            test_days = test_mask.sum()
            
            if train_days < 100 or test_days < 20:
                break
            
            fold = WalkForwardFold(
                fold_id=fold_id,
                train_start=str(current_train_start.date()),
                train_end=str(train_end.date()),
                test_start=str(test_start.date()),
                test_end=str(test_end.date()),
                train_days=train_days,
                test_days=test_days
            )
            folds.append(fold)
            
            fold_id += 1
            
            # Roll forward
            if self.config.expanding_window:
                # Expanding: keep train_start fixed
                pass
            else:
                current_train_start = current_train_start + pd.DateOffset(months=roll_months)
            
            train_end = train_end + pd.DateOffset(months=roll_months)
            
            if train_end > end_date:
                break
        
        self.folds = folds
        return folds
    
    def run_backtest(
        self,
        prices: pd.DataFrame,
        signals: pd.DataFrame,
        adv: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Run walk-forward backtest.
        
        Args:
            prices: DataFrame with columns [date, symbol, close]
            signals: DataFrame with columns [date, symbol, signal, alpha_score]
            adv: Optional DataFrame with [symbol, adv] for transaction costs
        """
        logger.info("🚀 Starting walk-forward backtest...")
        
        # Ensure date index
        prices = prices.copy()
        if 'date' not in prices.columns:
            prices = prices.reset_index()
        prices['date'] = pd.to_datetime(prices['date'])
        
        # Get unique dates
        dates = prices['date'].sort_values().unique()
        dates = pd.DatetimeIndex(dates)
        
        # Create folds
        if not self.folds:
            self.create_folds(dates)
        
        logger.info(f"   📊 Created {len(self.folds)} walk-forward folds")
        
        # Initialize portfolio
        portfolio = PortfolioTracker(self.config)
        
        # Default ADV if not provided
        if adv is None:
            # Estimate from prices
            if 'volume' in prices.columns:
                adv_df = prices.groupby('symbol').apply(
                    lambda x: (x['close'] * x['volume']).rolling(20).mean().iloc[-1]
                ).reset_index()
                adv_df.columns = ['symbol', 'adv']
            else:
                adv_df = pd.DataFrame({
                    'symbol': prices['symbol'].unique(),
                    'adv': 10_000_000  # Default $10M ADV
                })
        else:
            adv_df = adv
        
        # Run through each fold's test period
        all_test_dates = set()
        for fold in self.folds:
            test_start = pd.to_datetime(fold.test_start)
            test_end = pd.to_datetime(fold.test_end)
            
            test_dates = dates[(dates >= test_start) & (dates < test_end)]
            all_test_dates.update(test_dates)
        
        all_test_dates = sorted(all_test_dates)
        
        logger.info(f"   📅 Test period: {min(all_test_dates):%Y-%m-%d} to {max(all_test_dates):%Y-%m-%d}")
        logger.info(f"   📈 Trading days: {len(all_test_dates)}")
        
        # For each test date
        for date in all_test_dates:
            # Get prices for this date
            date_prices = prices[prices['date'] == date]
            
            if date_prices.empty:
                continue
            
            # Update portfolio prices
            price_dict = dict(zip(date_prices['symbol'], date_prices['close']))
            portfolio.update_prices(price_dict)
            
            # Get signals for this date
            if 'date' in signals.columns:
                date_signals = signals[signals['date'] == date]
            else:
                # Use latest signals
                date_signals = signals
            
            if not date_signals.empty:
                # Calculate target positions
                pv = portfolio.get_portfolio_value()
                
                for _, row in date_signals.iterrows():
                    symbol = row['symbol']
                    signal = row.get('signal', 0)
                    
                    if symbol not in price_dict:
                        continue
                    
                    price = price_dict[symbol]
                    
                    # Get ADV for this symbol
                    symbol_adv = adv_df[adv_df['symbol'] == symbol]['adv']
                    symbol_adv = symbol_adv.iloc[0] if len(symbol_adv) > 0 else 10_000_000
                    
                    # Calculate target weight
                    target_weight = signal * self.config.max_position_size
                    target_value = target_weight * pv
                    target_shares = target_value / price if price > 0 else 0
                    
                    # Execute trade
                    portfolio.execute_trade(
                        date=str(date.date()),
                        symbol=symbol,
                        target_shares=target_shares,
                        price=price,
                        adv=symbol_adv
                    )
            
            # Record daily value
            portfolio.record_daily_value(str(date.date()))
            
            # Check drawdown stop
            if len(portfolio.daily_values) > 1:
                pv = portfolio.get_portfolio_value()
                peak = max(d['portfolio_value'] for d in portfolio.daily_values)
                dd = (pv - peak) / peak
                
                if dd < -self.config.max_drawdown_stop:
                    logger.warning(f"   ⚠️ Max drawdown reached ({dd:.1%}). Stopping.")
                    break
        
        # Calculate metrics
        self.all_trades = portfolio.trades
        self.daily_equity = pd.DataFrame(portfolio.daily_values)
        
        if not self.daily_equity.empty:
            self.final_metrics = self._calculate_metrics()
        
        logger.info(f"   📊 Executed {len(self.all_trades)} trades")
        
        return {
            'metrics': self.final_metrics,
            'trades': len(self.all_trades),
            'daily_equity': self.daily_equity,
            'folds': len(self.folds)
        }
    
    def _calculate_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics"""
        if self.daily_equity.empty:
            return {}
        
        df = self.daily_equity.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Returns
        df['return'] = df['portfolio_value'].pct_change()
        returns = df['return'].dropna()
        
        if len(returns) < 2:
            return {}
        
        # Basic metrics
        total_return = (df['portfolio_value'].iloc[-1] / df['portfolio_value'].iloc[0]) - 1
        
        # Annualized metrics
        trading_days = len(returns)
        years = trading_days / 252
        
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Volatility
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)
        
        # Sharpe (assuming 0% risk-free)
        sharpe = (returns.mean() * 252) / annual_vol if annual_vol > 0 else 0
        
        # Sortino
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 1
        sortino = (returns.mean() * 252) / downside_vol if downside_vol > 0 else 0
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min()
        
        # Calmar
        calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate
        winning = returns[returns > 0]
        win_rate = len(winning) / len(returns) if len(returns) > 0 else 0
        
        # Transaction costs
        total_commission = sum(t.commission for t in self.all_trades)
        total_slippage = sum(t.slippage for t in self.all_trades)
        total_costs = total_commission + total_slippage
        
        # Turnover
        total_traded = sum(abs(t.value) for t in self.all_trades)
        avg_portfolio = df['portfolio_value'].mean()
        annualized_turnover = (total_traded / 2) / avg_portfolio / years if years > 0 else 0
        
        metrics = {
            'total_return': total_return,
            'cagr': cagr,
            'annual_volatility': annual_vol,
            'sharpe': sharpe,
            'sortino': sortino,
            'max_drawdown': max_drawdown,
            'calmar': calmar,
            'win_rate': win_rate,
            'trading_days': trading_days,
            'years': years,
            'total_trades': len(self.all_trades),
            'total_commission': total_commission,
            'total_slippage': total_slippage,
            'total_costs': total_costs,
            'cost_drag': total_costs / df['portfolio_value'].iloc[0],
            'annualized_turnover': annualized_turnover
        }
        
        # Quality checks
        metrics['is_realistic'] = (
            metrics['sharpe'] < 5.0 and
            metrics['cagr'] < 1.0 and
            metrics['max_drawdown'] > -0.50
        )
        
        return metrics
    
    def print_report(self):
        """Print backtest report"""
        if not self.final_metrics:
            print("No metrics to report. Run backtest first.")
            return
        
        m = self.final_metrics
        
        print("\n" + "=" * 60)
        print("📊 V17.0 WALK-FORWARD BACKTEST REPORT")
        print("=" * 60)
        
        print(f"\n📈 Performance:")
        print(f"   Total Return:     {m['total_return']:.1%}")
        print(f"   CAGR:             {m['cagr']:.1%}")
        print(f"   Annual Vol:       {m['annual_volatility']:.1%}")
        print(f"   Sharpe Ratio:     {m['sharpe']:.2f}")
        print(f"   Sortino Ratio:    {m['sortino']:.2f}")
        print(f"   Max Drawdown:     {m['max_drawdown']:.1%}")
        print(f"   Calmar Ratio:     {m['calmar']:.2f}")
        
        print(f"\n📊 Trading Statistics:")
        print(f"   Trading Days:     {m['trading_days']}")
        print(f"   Total Trades:     {m['total_trades']}")
        print(f"   Win Rate:         {m['win_rate']:.1%}")
        print(f"   Annual Turnover:  {m['annualized_turnover']:.1%}")
        
        print(f"\n💰 Transaction Costs:")
        print(f"   Total Commission: ${m['total_commission']:,.0f}")
        print(f"   Total Slippage:   ${m['total_slippage']:,.0f}")
        print(f"   Cost Drag:        {m['cost_drag']:.2%}")
        
        print(f"\n✅ Realism Check:")
        if m['is_realistic']:
            print("   ✓ Metrics within realistic bounds")
        else:
            print("   ⚠️ WARNING: Metrics may be overfit!")
            if m['sharpe'] >= 5.0:
                print(f"     - Sharpe {m['sharpe']:.2f} >= 5.0 (suspicious)")
            if m['cagr'] >= 1.0:
                print(f"     - CAGR {m['cagr']:.1%} >= 100% (suspicious)")
        
        print()


def main():
    """Test walk-forward engine"""
    print("\n" + "=" * 60)
    print("🔄 V17.0 WALK-FORWARD ENGINE TEST")
    print("=" * 60)
    
    # Load price data
    price_file = 'cache/v17_prices/v17_prices_latest.parquet'
    
    if not os.path.exists(price_file):
        logger.warning(f"Price file not found: {price_file}")
        logger.info("Run v17_data_pipeline.py first to generate price data")
        return None
    
    prices = pd.read_parquet(price_file)
    logger.info(f"📊 Loaded {len(prices)} price records")
    
    # Create synthetic signals for testing
    logger.info("📈 Generating synthetic momentum signals...")
    
    # Select top 50 symbols by liquidity
    symbol_volume = prices.groupby('symbol')['volume'].mean().sort_values(ascending=False)
    top_symbols = symbol_volume.head(50).index.tolist()
    
    # Filter prices
    prices_subset = prices[prices['symbol'].isin(top_symbols)].copy()
    
    # Calculate momentum signal for each date and symbol
    signals_list = []
    
    for symbol in top_symbols:
        sym_data = prices_subset[prices_subset['symbol'] == symbol].copy()
        if len(sym_data) < 63:
            continue
        
        sym_data = sym_data.sort_values('date')
        
        # 3-month momentum
        sym_data['momentum'] = sym_data['close'].pct_change(63)
        
        # Generate signal for each date (after enough history)
        for idx in range(63, len(sym_data)):
            row = sym_data.iloc[idx]
            mom = row['momentum']
            
            if pd.isna(mom):
                continue
            
            signals_list.append({
                'date': row['date'],
                'symbol': symbol,
                'signal': 1.0 if mom > 0.05 else (-1.0 if mom < -0.05 else 0.0),
                'alpha_score': abs(mom)
            })
    
    signals = pd.DataFrame(signals_list)
    signals['date'] = pd.to_datetime(signals['date'])
    logger.info(f"   Generated {len(signals)} signal observations for {len(top_symbols)} symbols")
    
    # Run backtest
    config = BacktestConfig(
        train_months=6,   # Shorter for testing
        test_months=3,
        roll_months=1,
        initial_capital=1_000_000
    )
    
    engine = WalkForwardEngine(config)
    result = engine.run_backtest(prices_subset, signals)
    
    # Print report
    engine.print_report()
    
    # Save results
    results_dir = Path('results/v17')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save equity curve
    if engine.daily_equity is not None and not engine.daily_equity.empty:
        engine.daily_equity.to_parquet(results_dir / 'v17_equity_curve.parquet', index=False)
    
    # Save metrics
    with open(results_dir / 'v17_backtest_metrics.json', 'w') as f:
        json.dump(engine.final_metrics, f, indent=2, default=str)
    
    logger.info(f"💾 Results saved to {results_dir}")
    
    return engine


if __name__ == "__main__":
    main()
