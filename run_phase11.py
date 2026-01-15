"""
Phase 11: Total Market Domination
=================================

Run the full Phase 11 backtest with:
- Multi-factor stock selection from entire US market
- 5-factor composite scoring
- Leveraged ETF amplification
- Dynamic risk management

Target: CAGR ≥ 28%, Max DD ≤ 22%, Alpha vs SPY ≥ +5%
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.phase11 import (
    UniverseManager,
    StockFilter,
    FactorEngine,
    FactorWeights,
    PortfolioConstructor,
    PortfolioConfig,
    SectorLeverageManager,
    LeverageConfig,
    Phase11RiskController,
    RiskConfig,
    compute_portfolio_stats,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Phase11Backtest:
    """
    Full Phase 11 backtest with multi-factor selection.
    """
    
    def __init__(
        self,
        start_date: str = '2022-01-01',
        end_date: str = '2025-01-01',
        initial_capital: float = 100_000,
        rebalance_freq: str = 'monthly',  # 'weekly' or 'monthly'
        n_stocks: int = 40,
        use_leverage: bool = True,
    ):
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.initial_capital = initial_capital
        self.rebalance_freq = rebalance_freq
        self.n_stocks = n_stocks
        self.use_leverage = use_leverage
        
        # Initialize components
        self.universe_manager = UniverseManager()
        self.factor_engine = FactorEngine()
        self.portfolio_constructor = PortfolioConstructor(
            PortfolioConfig(
                n_stocks=20,  # More concentrated - 20 stocks
                max_position_weight=0.12,  # Allow 12% per position
                min_position_weight=0.03,
                leverage_allocation=0.55 if use_leverage else 0.0,  # 55% leverage
            )
        )
        self.leverage_manager = SectorLeverageManager(
            LeverageConfig(
                max_leverage_allocation=0.60,  # Up to 60% in leveraged ETFs
                min_leverage_allocation=0.30,  # At least 30% always
            )
        )
        self.risk_controller = Phase11RiskController(
            RiskConfig(
                dd_level_1=0.10,   # Higher tolerance for DD
                dd_level_2=0.18,
                dd_level_3=0.25,
                target_vol=0.40,   # Higher target vol (very aggressive)
                max_position=0.12,  # Match portfolio config
            )
        )
        
        # Data storage
        self.price_data: Dict[str, pd.DataFrame] = {}
        self.universe: List[str] = []
        self.sector_map: Dict[str, str] = {}
        
        # Results
        self.equity_curve = []
        self.portfolio_history = []
        self.trade_log = []
        
    def fetch_data(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV data for tickers."""
        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance required: pip install yfinance")
            return {}
        
        data = {}
        
        # Fetch in batches
        batch_size = 50
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i+batch_size]
            logger.info(f"Fetching batch {i//batch_size + 1}: {len(batch)} tickers")
            
            try:
                df = yf.download(
                    batch,
                    start=self.start_date - timedelta(days=400),  # Extra for lookback
                    end=self.end_date,
                    progress=False,
                    group_by='ticker',
                )
                
                if len(batch) == 1:
                    # Single ticker format is different
                    ticker = batch[0]
                    if not df.empty:
                        data[ticker] = df
                else:
                    # Multi-ticker format
                    for ticker in batch:
                        try:
                            if ticker in df.columns.get_level_values(0):
                                ticker_df = df[ticker].dropna()
                                if len(ticker_df) > 100:
                                    data[ticker] = ticker_df
                        except Exception as e:
                            logger.debug(f"Error extracting {ticker}: {e}")
                            continue
                            
            except Exception as e:
                logger.warning(f"Batch fetch error: {e}")
                continue
        
        logger.info(f"Fetched data for {len(data)} tickers")
        return data
    
    def run(self) -> Dict:
        """Run the full backtest."""
        logger.info("=" * 60)
        logger.info("PHASE 11: TOTAL MARKET DOMINATION")
        logger.info("=" * 60)
        
        # Step 1: Get universe
        logger.info("Step 1: Building universe...")
        self.universe = self.universe_manager.get_tradeable_universe()
        logger.info(f"Universe: {len(self.universe)} tickers")
        
        # Step 2: Fetch data
        logger.info("Step 2: Fetching price data...")
        
        # For efficiency, prioritize main constituents + leveraged
        priority_tickers = self._get_priority_tickers()
        self.price_data = self.fetch_data(priority_tickers)
        
        if len(self.price_data) < 50:
            logger.error(f"Insufficient data: only {len(self.price_data)} tickers")
            return {'error': 'Insufficient data'}
        
        # Also fetch benchmark
        benchmark_data = self.fetch_data(['SPY', '^VIX'])
        self.price_data.update(benchmark_data)
        
        # Get sector map for fetched tickers
        self.sector_map = self.universe_manager.get_sector_map(list(self.price_data.keys()))
        
        # Step 3: Generate rebalance dates
        rebalance_dates = self._get_rebalance_dates()
        logger.info(f"Step 3: {len(rebalance_dates)} rebalance periods")
        
        # Step 4: Run backtest
        logger.info("Step 4: Running backtest...")
        equity = self.initial_capital
        positions: Dict[str, float] = {}
        
        for i, date in enumerate(rebalance_dates):
            try:
                # Get data up to this date
                date_data = self._get_data_to_date(date)
                
                if len(date_data) < 30:
                    continue
                
                # Get SPY data for relative strength
                spy_data = date_data.get('SPY')
                
                # Compute factors
                factor_df = self.factor_engine.compute_all_factors(
                    date_data, 
                    date=str(date.date()),
                    spy_data=spy_data
                )
                
                if factor_df.empty:
                    continue
                
                # Get VIX level
                vix_level = self._get_vix(date)
                
                # Get SPY trend
                spy_trend = 'up'
                if 'SPY' in date_data:
                    spy_prices = date_data['SPY']['Close']
                    spy_trend = self.leverage_manager.get_spy_trend(spy_prices)
                
                # Compute leverage weights
                current_dd = self.risk_controller.current_drawdown
                leverage_weights = {}
                if self.use_leverage:
                    leverage_weights = self.leverage_manager.compute_leverage_weights(
                        date_data, vix_level, spy_trend, current_dd
                    )
                
                # Construct portfolio
                target_weights = self.portfolio_constructor.construct_portfolio(
                    factor_df,
                    self.sector_map,
                    leverage_weights,
                    current_dd,
                )
                
                # Apply risk controls
                final_weights, risk_metrics = self.risk_controller.adjust_for_risk(
                    target_weights,
                    date_data,
                    self.sector_map,
                    equity,
                )
                
                # Simulate returns until next rebalance
                if i < len(rebalance_dates) - 1:
                    next_date = rebalance_dates[i + 1]
                    period_return = self._simulate_period(
                        final_weights, date, next_date
                    )
                    equity *= (1 + period_return)
                
                # Track
                self.equity_curve.append({
                    'date': date,
                    'equity': equity,
                    'n_positions': len(final_weights),
                    'leverage_pct': sum(v for k, v in final_weights.items() 
                                       if k in ['TQQQ', 'SPXL', 'SOXL', 'UPRO']),
                    'vix': vix_level,
                    'trend': spy_trend,
                    **risk_metrics,
                })
                
                positions = final_weights
                
                if i % 3 == 0:
                    logger.info(f"  {date.date()}: Equity=${equity:,.0f}, "
                               f"DD={current_dd:.1%}, {len(positions)} positions")
                    
            except Exception as e:
                logger.error(f"Error on {date}: {e}")
                continue
        
        # Step 5: Compute results
        results = self._compute_results()
        
        return results
    
    def _get_priority_tickers(self) -> List[str]:
        """Get priority tickers for data fetching."""
        priority = set()
        
        # S&P 500 + NASDAQ 100 components
        priority.update(self.universe_manager._get_sp500_tickers())
        priority.update(self.universe_manager._get_nasdaq100_tickers())
        
        # Leveraged ETFs
        priority.update([
            'TQQQ', 'SPXL', 'UPRO', 'SOXL', 'TECL', 'FAS', 'TNA',
            'QLD', 'SSO', 'UWM', 'ERX', 'CURE', 'LABU', 'FNGU',
        ])
        
        # Sector ETFs
        priority.update([
            'XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLY', 'XLP', 
            'XLU', 'XLB', 'XLRE', 'XLC', 'SMH', 'IBB',
        ])
        
        # Benchmark
        priority.update(['SPY', 'QQQ', 'IWM', '^VIX'])
        
        return list(priority)
    
    def _get_rebalance_dates(self) -> List[pd.Timestamp]:
        """Generate rebalance dates."""
        dates = pd.date_range(self.start_date, self.end_date, freq='B')
        
        if self.rebalance_freq == 'monthly':
            # First trading day of each month
            rebalance = []
            current_month = None
            for d in dates:
                if d.month != current_month:
                    rebalance.append(d)
                    current_month = d.month
            return rebalance
        else:  # weekly
            # Every Friday
            return [d for d in dates if d.dayofweek == 4]
    
    def _get_data_to_date(
        self, 
        date: pd.Timestamp,
    ) -> Dict[str, pd.DataFrame]:
        """Get price data up to a given date."""
        result = {}
        
        for ticker, df in self.price_data.items():
            if isinstance(df.index, pd.DatetimeIndex):
                mask = df.index <= date
                subset = df[mask]
                if len(subset) > 60:  # Need enough history
                    result[ticker] = subset
        
        return result
    
    def _get_vix(self, date: pd.Timestamp) -> float:
        """Get VIX level for a date."""
        if '^VIX' in self.price_data:
            vix_df = self.price_data['^VIX']
            if isinstance(vix_df.index, pd.DatetimeIndex):
                mask = vix_df.index <= date
                vix_prices = vix_df[mask]['Close']
                if len(vix_prices) > 5:
                    return vix_prices.rolling(5).mean().iloc[-1]
        return 18.0  # Default
    
    def _simulate_period(
        self,
        weights: Dict[str, float],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> float:
        """Simulate returns for a period."""
        total_return = 0.0
        total_weight = sum(weights.values())
        
        for ticker, weight in weights.items():
            if ticker in self.price_data:
                df = self.price_data[ticker]
                try:
                    # Get prices for period
                    mask = (df.index >= start_date) & (df.index <= end_date)
                    period_df = df[mask]
                    
                    if len(period_df) >= 2:
                        start_price = period_df['Close'].iloc[0]
                        end_price = period_df['Close'].iloc[-1]
                        ticker_return = (end_price / start_price) - 1
                        total_return += weight * ticker_return
                except Exception:
                    continue
        
        # Cash return for unallocated (assume 4% annual)
        cash_weight = max(0, 1.0 - total_weight)
        days = (end_date - start_date).days
        cash_return = (1.04 ** (days / 365)) - 1
        total_return += cash_weight * cash_return
        
        return total_return
    
    def _compute_results(self) -> Dict:
        """Compute backtest results."""
        if not self.equity_curve:
            return {'error': 'No equity data'}
        
        eq_df = pd.DataFrame(self.equity_curve)
        eq_df.set_index('date', inplace=True)
        
        # Compute returns
        eq_df['returns'] = eq_df['equity'].pct_change()
        
        # Calculate metrics directly from equity curve
        start_eq = self.initial_capital
        end_eq = eq_df['equity'].iloc[-1]
        
        # Years
        start_dt = eq_df.index[0]
        end_dt = eq_df.index[-1]
        years = (end_dt - start_dt).days / 365.25
        
        # CAGR
        cagr = (end_eq / start_eq) ** (1 / years) - 1 if years > 0 else 0
        
        # Max drawdown from equity curve
        equity_series = eq_df['equity']
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max
        max_dd = drawdown.min()
        
        # Volatility and Sharpe (monthly data)
        monthly_returns = eq_df['returns'].dropna()
        monthly_vol = monthly_returns.std()
        ann_vol = monthly_vol * np.sqrt(12)  # Monthly to annual
        
        excess_return = cagr - 0.04  # 4% risk-free
        sharpe = excess_return / ann_vol if ann_vol > 0 else 0
        
        # Sortino
        downside = monthly_returns[monthly_returns < 0]
        downside_vol = downside.std() * np.sqrt(12) if len(downside) > 0 else ann_vol
        sortino = excess_return / downside_vol if downside_vol > 0 else 0
        
        # Get benchmark returns for alpha
        spy_returns = self._get_benchmark_returns(eq_df.index)
        if spy_returns is not None and len(spy_returns) > 5:
            spy_total = (1 + spy_returns.dropna()).prod() - 1
            spy_cagr = (1 + spy_total) ** (1 / years) - 1 if years > 0 else 0
            alpha = cagr - spy_cagr
        else:
            # Approximate SPY CAGR ~10-12% 
            alpha = cagr - 0.10
        
        results = {
            'phase': 'Phase 11: Total Market Domination',
            'period': f"{self.start_date.date()} to {self.end_date.date()}",
            'initial_capital': self.initial_capital,
            'final_equity': end_eq,
            'cagr': cagr,
            'max_drawdown': max_dd,
            'volatility': ann_vol,
            'sharpe': sharpe,
            'sortino': sortino,
            'total_return': (end_eq / start_eq) - 1,
            'alpha_vs_spy': alpha,
            'n_rebalances': len(eq_df),
            'avg_positions': eq_df['n_positions'].mean(),
            'avg_leverage': eq_df.get('leverage_pct', pd.Series([0])).mean(),
        }
        
        return results
    
    def _get_benchmark_returns(
        self, 
        dates: pd.DatetimeIndex,
    ) -> Optional[pd.Series]:
        """Get SPY returns for comparison."""
        if 'SPY' not in self.price_data:
            return None
        
        spy_df = self.price_data['SPY']
        spy_df = spy_df[spy_df.index.isin(dates)]
        
        if len(spy_df) > 20:
            return spy_df['Close'].pct_change()
        return None


def main():
    """Run Phase 11 backtest."""
    parser = argparse.ArgumentParser(description='Phase 11: Total Market Domination')
    parser.add_argument('--start', default='2022-01-01', help='Start date')
    parser.add_argument('--end', default='2025-01-01', help='End date')
    parser.add_argument('--capital', type=float, default=100_000, help='Initial capital')
    parser.add_argument('--stocks', type=int, default=40, help='Number of stocks')
    parser.add_argument('--freq', default='monthly', choices=['weekly', 'monthly'])
    parser.add_argument('--no-leverage', action='store_true', help='Disable leverage')
    
    args = parser.parse_args()
    
    # Run backtest
    backtest = Phase11Backtest(
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        rebalance_freq=args.freq,
        n_stocks=args.stocks,
        use_leverage=not args.no_leverage,
    )
    
    results = backtest.run()
    
    # Display results
    print("\n" + "=" * 60)
    print("PHASE 11 RESULTS")
    print("=" * 60)
    
    if 'error' in results:
        print(f"Error: {results['error']}")
        return
    
    print(f"\nPeriod: {results['period']}")
    print(f"Initial Capital: ${results['initial_capital']:,.0f}")
    print(f"Final Equity: ${results['final_equity']:,.0f}")
    print(f"\nPerformance:")
    print(f"  CAGR: {results.get('cagr', 0):.1%}")
    print(f"  Max Drawdown: {results.get('max_drawdown', 0):.1%}")
    print(f"  Sharpe Ratio: {results.get('sharpe', 0):.2f}")
    print(f"  Sortino Ratio: {results.get('sortino', 0):.2f}")
    print(f"  Volatility: {results.get('volatility', 0):.1%}")
    print(f"  Alpha vs SPY: {results.get('alpha_vs_spy', 0):+.1%}")
    print(f"\nPortfolio:")
    print(f"  Avg Positions: {results.get('avg_positions', 0):.0f}")
    print(f"  Avg Leverage: {results.get('avg_leverage', 0):.1%}")
    print(f"  Rebalances: {results.get('n_rebalances', 0)}")
    
    # Check targets
    print("\n" + "=" * 60)
    print("TARGET CHECK")
    print("=" * 60)
    
    cagr = results.get('cagr', 0)
    max_dd = abs(results.get('max_drawdown', 1))
    alpha = results.get('alpha_vs_spy', 0)
    
    target_cagr = cagr >= 0.28
    target_dd = max_dd <= 0.22
    target_alpha = alpha >= 0.05
    
    print(f"  CAGR ≥ 28%: {'✓' if target_cagr else '✗'} ({cagr:.1%})")
    print(f"  Max DD ≤ 22%: {'✓' if target_dd else '✗'} ({max_dd:.1%})")
    print(f"  Alpha ≥ 5%: {'✓' if target_alpha else '✗'} ({alpha:+.1%})")
    
    targets_hit = sum([target_cagr, target_dd, target_alpha])
    print(f"\n  Targets Hit: {targets_hit}/3")
    
    # Save results
    results_path = Path(__file__).parent.parent / 'results' / 'phase11_results.json'
    results_path.parent.mkdir(exist_ok=True)
    
    # Make JSON serializable
    results_json = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                   for k, v in results.items()}
    
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_path}")
    
    return results


if __name__ == '__main__':
    results = main()
