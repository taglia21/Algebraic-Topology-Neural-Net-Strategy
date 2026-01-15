#!/usr/bin/env python3
"""Phase 6 Full Universe Backtest.

Scales the TDA+Momentum strategy from 100 → 500+ stocks with:
- Multi-stage filtering pipeline
- Parallel TDA computation
- Multi-factor stock scoring (TDA + Momentum + Quality + Value)
- Portfolio risk controls with sector diversification

Target Metrics:
- CAGR > 17%
- Sharpe > 1.0
- Max Drawdown < 17%
- 4+ sectors, none > 30%

Usage:
    python scripts/run_phase6_full_universe.py
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_provider import (
    fetch_universe_batch,
    compute_returns_batch,
    get_universe_statistics,
)
from src.data.universe_expansion import FullUniverseManager, BatchDataFetcher
from src.data.data_cache import get_data_cache
from src.tda_features import TDAFeatureGenerator
from src.tda_engine_batched import TDAEngineBatched
from src.multi_factor_selector import MultiFactorSelector
from src.portfolio_risk_controller import PortfolioRiskController
from src.regime_detector import MarketRegimeDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Phase6Backtest:
    """
    Phase 6 backtesting engine for full US stock universe.
    
    Pipeline:
    1. Load/expand universe (500+ stocks)
    2. Fetch data with caching
    3. Compute TDA features (parallel)
    4. Multi-factor stock selection
    5. Monthly rebalancing with risk controls
    6. Performance analysis
    """
    
    def __init__(
        self,
        start_date: str = "2020-01-01",
        end_date: str = "2025-01-01",
        initial_capital: float = 100000.0,
        n_stocks: int = 30,
        rebalance_frequency: str = "monthly",
        use_cache: bool = True,
        provider: str = "yfinance",  # Use yfinance for no API limits
        cache_dir: str = "./cache",
    ):
        """
        Initialize Phase 6 backtest.
        
        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Starting capital
            n_stocks: Number of stocks to hold
            rebalance_frequency: 'monthly', 'weekly', or 'quarterly'
            use_cache: Whether to use data caching
            provider: Data provider ('yfinance' or 'polygon')
            cache_dir: Cache directory
        """
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.n_stocks = n_stocks
        self.rebalance_frequency = rebalance_frequency
        self.use_cache = use_cache
        self.provider = provider
        self.cache_dir = cache_dir
        
        # Initialize components
        self.universe_manager = FullUniverseManager(
            start_date=start_date,
            end_date=end_date,
            cache_dir=cache_dir,
        )
        
        self.tda_engine = TDAEngineBatched(
            window=30,
            embedding_dim=3,
            max_workers=4,
            cache_dir=cache_dir,
        )
        
        self.selector = MultiFactorSelector(
            n_stocks=n_stocks,
            max_sector_weight=0.30,
            max_single_stock=0.08,
        )
        
        self.risk_controller = PortfolioRiskController(
            initial_capital=initial_capital,
            max_position_weight=0.08,
            max_sector_weight=0.30,
            min_sectors=4,
        )
        
        self.regime_detector = MarketRegimeDetector()
        
        # State
        self.universe: List[str] = []
        self.ohlcv_data: Dict[str, pd.DataFrame] = {}
        self.tda_features: Dict[str, pd.DataFrame] = {}
        self.equity_curve: List[Tuple[str, float]] = []
        self.holdings_history: List[Dict] = []
        
    def initialize_universe(self, max_stocks: int = None) -> List[str]:
        """
        Initialize and filter the stock universe.
        
        Args:
            max_stocks: Optional limit on universe size
            
        Returns:
            List of tradeable tickers
        """
        logger.info("=" * 60)
        logger.info("PHASE 6: Initializing Stock Universe")
        logger.info("=" * 60)
        
        # Stage 1-3 filtering
        universe = self.universe_manager.get_final_universe(max_stocks=max_stocks)
        
        # Log sector distribution
        distribution = self.universe_manager.get_sector_distribution()
        logger.info("\nSector distribution:")
        for sector, count in sorted(distribution.items(), key=lambda x: -x[1]):
            logger.info(f"  {sector}: {count} stocks")
        
        self.universe = universe
        return universe
    
    def fetch_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for the universe.
        
        Returns:
            Dict mapping ticker to OHLCV DataFrame
        """
        logger.info("=" * 60)
        logger.info("PHASE 6: Fetching Market Data")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Use batch fetching
        data = fetch_universe_batch(
            tickers=self.universe,
            start_date=self.start_date,
            end_date=self.end_date,
            provider=self.provider,
            use_cache=self.use_cache,
            cache_dir=self.cache_dir,
            rate_limit=5.0,  # 5 calls/second
        )
        
        elapsed = time.time() - start_time
        logger.info(f"Data fetch complete: {len(data)} tickers in {elapsed:.1f}s")
        
        # Validate data
        valid_data = {}
        for ticker, df in data.items():
            if len(df) >= 252:  # At least 1 year of data
                valid_data[ticker] = df
        
        logger.info(f"Valid data: {len(valid_data)}/{len(data)} tickers")
        
        self.ohlcv_data = valid_data
        return valid_data
    
    def compute_tda_features(self) -> Dict[str, pd.DataFrame]:
        """
        Compute TDA features for all stocks.
        
        Returns:
            Dict mapping ticker to TDA features DataFrame
        """
        logger.info("=" * 60)
        logger.info("PHASE 6: Computing TDA Features")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Compute rolling TDA features
        tda_data = self.tda_engine.compute_universe_tda(
            ohlcv_dict=self.ohlcv_data,
            batch_size=50,
            use_cache=self.use_cache,
        )
        
        elapsed = time.time() - start_time
        logger.info(f"TDA computation complete: {len(tda_data)} tickers in {elapsed:.1f}s")
        
        self.tda_features = tda_data
        return tda_data
    
    def get_rebalance_dates(self) -> List[str]:
        """Get list of rebalance dates based on frequency."""
        # Get a sample ticker's date range
        sample_ticker = list(self.ohlcv_data.keys())[0]
        dates = self.ohlcv_data[sample_ticker].index
        
        if self.rebalance_frequency == 'monthly':
            # First trading day of each month
            monthly = dates.to_series().groupby([dates.year, dates.month]).first()
            return [d.strftime('%Y-%m-%d') for d in monthly.values]
        
        elif self.rebalance_frequency == 'weekly':
            # Every Monday (or first available day)
            weekly = dates[dates.dayofweek == 0]
            return [d.strftime('%Y-%m-%d') for d in weekly]
        
        elif self.rebalance_frequency == 'quarterly':
            # First day of each quarter
            quarterly = dates.to_series().groupby([dates.year, dates.quarter]).first()
            return [d.strftime('%Y-%m-%d') for d in quarterly.values]
        
        else:
            raise ValueError(f"Unknown frequency: {self.rebalance_frequency}")
    
    def get_latest_tda_features(self, date: str) -> Dict[str, Dict[str, float]]:
        """
        Get TDA features as of a specific date.
        
        Args:
            date: Date string (YYYY-MM-DD)
            
        Returns:
            Dict mapping ticker to latest TDA features dict
        """
        date_ts = pd.Timestamp(date)
        features = {}
        
        for ticker, df in self.tda_features.items():
            if df.empty:
                continue
            
            # Get features up to this date
            if isinstance(df.index, pd.DatetimeIndex):
                mask = df.index <= date_ts
                if mask.any():
                    latest = df[mask].iloc[-1].to_dict()
                    features[ticker] = latest
            else:
                # Use last row if no date index
                features[ticker] = df.iloc[-1].to_dict()
        
        return features
    
    def get_sector_map(self) -> Dict[str, str]:
        """Get sector for each ticker."""
        return {t: self.universe_manager.get_sector(t) for t in self.universe}
    
    def detect_regime(self, date: str) -> str:
        """
        Detect market regime as of date.
        
        Args:
            date: Date string
            
        Returns:
            Regime string ('bull', 'bear', or 'neutral')
        """
        # Use SPY as market proxy
        if 'SPY' in self.ohlcv_data:
            spy_df = self.ohlcv_data['SPY']
            date_ts = pd.Timestamp(date)
            
            if isinstance(spy_df.index, pd.DatetimeIndex):
                mask = spy_df.index <= date_ts
                if mask.any():
                    spy_subset = spy_df[mask]
                    if len(spy_subset) >= 200:
                        close = spy_subset['close'].values
                        sma_50 = np.mean(close[-50:])
                        sma_200 = np.mean(close[-200:])
                        
                        if close[-1] > sma_50 > sma_200:
                            return 'bull'
                        elif close[-1] < sma_50 < sma_200:
                            return 'bear'
        
        return 'neutral'
    
    def get_ohlcv_as_of_date(self, date: str) -> Dict[str, pd.DataFrame]:
        """Get OHLCV data up to a specific date."""
        date_ts = pd.Timestamp(date)
        result = {}
        
        for ticker, df in self.ohlcv_data.items():
            if isinstance(df.index, pd.DatetimeIndex):
                mask = df.index <= date_ts
                if mask.any():
                    result[ticker] = df[mask]
        
        return result
    
    def get_prices_on_date(self, date: str) -> Dict[str, float]:
        """Get closing prices for a specific date."""
        date_ts = pd.Timestamp(date)
        prices = {}
        
        for ticker, df in self.ohlcv_data.items():
            try:
                if isinstance(df.index, pd.DatetimeIndex):
                    # Find closest date
                    idx = df.index.get_indexer([date_ts], method='ffill')[0]
                    if idx >= 0:
                        close_col = 'close' if 'close' in df.columns else 'Close'
                        prices[ticker] = df.iloc[idx][close_col]
            except Exception:
                pass
        
        return prices
    
    def run_backtest(self) -> Dict[str, Any]:
        """
        Run the full Phase 6 backtest.
        
        Returns:
            Dict with backtest results
        """
        logger.info("=" * 60)
        logger.info("PHASE 6: Running Backtest")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Get rebalance dates
        rebalance_dates = self.get_rebalance_dates()
        logger.info(f"Rebalance periods: {len(rebalance_dates)}")
        
        # Get sector map
        sector_map = self.get_sector_map()
        
        # Track portfolio
        holdings: Dict[str, Tuple[int, float, str]] = {}  # ticker: (shares, entry_price, sector)
        cash = self.initial_capital
        
        # Equity curve
        equity_curve = []
        holdings_history = []
        
        for i, date in enumerate(rebalance_dates):
            logger.info(f"\n{'='*40}")
            logger.info(f"Rebalance {i+1}/{len(rebalance_dates)}: {date}")
            
            # Get current prices
            prices = self.get_prices_on_date(date)
            if len(prices) < 50:
                logger.warning(f"Not enough price data for {date}, skipping")
                continue
            
            # Update portfolio value
            portfolio_value = cash
            for ticker, (shares, entry_price, sector) in holdings.items():
                if ticker in prices:
                    portfolio_value += shares * prices[ticker]
            
            equity_curve.append((date, portfolio_value))
            logger.info(f"Portfolio value: ${portfolio_value:,.2f}")
            
            # Get TDA features as of this date
            tda_features = self.get_latest_tda_features(date)
            logger.info(f"TDA features available: {len(tda_features)} stocks")
            
            # Detect regime
            regime = self.detect_regime(date)
            logger.info(f"Market regime: {regime}")
            
            # Get OHLCV data up to this date for factor scoring
            ohlcv_subset = self.get_ohlcv_as_of_date(date)
            
            # Select stocks using multi-factor scoring
            selected = self.selector.select_stocks(
                ohlcv_dict=ohlcv_subset,
                tda_features=tda_features,
                sector_map=sector_map,
                regime=regime,
            )
            
            if len(selected) == 0:
                logger.warning("No stocks selected, holding cash")
                continue
            
            # Get target weights
            target_weights = self.selector.compute_weights(selected, weighting='equal')
            
            # Log selections
            logger.info(f"Selected {len(selected)} stocks:")
            for s in selected[:5]:
                logger.info(f"  {s.ticker}: score={s.composite_score:.3f} ({s.sector})")
            
            # Calculate trades
            # First, sell positions not in new selection
            for ticker in list(holdings.keys()):
                if ticker not in target_weights:
                    shares, entry_price, sector = holdings[ticker]
                    if ticker in prices:
                        cash += shares * prices[ticker]
                        logger.info(f"  SELL {ticker}: {shares} shares @ ${prices[ticker]:.2f}")
                    del holdings[ticker]
            
            # Then, rebalance to target weights
            target_value = portfolio_value * 0.98  # Keep 2% cash buffer
            
            for ticker, weight in target_weights.items():
                if ticker not in prices:
                    continue
                
                price = prices[ticker]
                target_shares = int((target_value * weight) / price)
                
                if target_shares <= 0:
                    continue
                
                current_shares = holdings.get(ticker, (0, 0, ''))[0]
                diff = target_shares - current_shares
                
                if diff > 0:
                    # Buy more
                    cost = diff * price
                    if cost <= cash:
                        cash -= cost
                        sector = sector_map.get(ticker, 'Unknown')
                        holdings[ticker] = (current_shares + diff, price, sector)
                        logger.info(f"  BUY {ticker}: {diff} shares @ ${price:.2f}")
                elif diff < 0:
                    # Sell some
                    cash += abs(diff) * price
                    entry = holdings[ticker][1]
                    sector = holdings[ticker][2]
                    holdings[ticker] = (current_shares + diff, entry, sector)
                    logger.info(f"  SELL {ticker}: {abs(diff)} shares @ ${price:.2f}")
            
            # Log sector allocation
            sector_alloc = {}
            for ticker, (shares, entry, sector) in holdings.items():
                if ticker in prices:
                    value = shares * prices[ticker]
                    sector_alloc[sector] = sector_alloc.get(sector, 0) + value
            
            if sector_alloc:
                total_equity = sum(sector_alloc.values())
                logger.info(f"\nSector allocation:")
                for sector, value in sorted(sector_alloc.items(), key=lambda x: -x[1]):
                    pct = value / total_equity if total_equity > 0 else 0
                    logger.info(f"  {sector}: {pct:.1%}")
            
            holdings_history.append({
                'date': date,
                'holdings': dict(holdings),
                'cash': cash,
                'portfolio_value': portfolio_value,
            })
        
        # Final portfolio value
        final_prices = self.get_prices_on_date(rebalance_dates[-1])
        final_value = cash
        for ticker, (shares, entry_price, sector) in holdings.items():
            if ticker in final_prices:
                final_value += shares * final_prices[ticker]
        
        equity_curve.append((rebalance_dates[-1], final_value))
        
        elapsed = time.time() - start_time
        
        # Calculate performance metrics
        results = self.calculate_performance_metrics(equity_curve, holdings)
        results['elapsed_time'] = elapsed
        results['equity_curve'] = equity_curve
        results['holdings_history'] = holdings_history
        
        self.equity_curve = equity_curve
        self.holdings_history = holdings_history
        
        return results
    
    def calculate_performance_metrics(
        self,
        equity_curve: List[Tuple[str, float]],
        final_holdings: Dict,
    ) -> Dict[str, Any]:
        """Calculate performance metrics from equity curve."""
        if len(equity_curve) < 2:
            return {}
        
        # Convert to DataFrame
        df = pd.DataFrame(equity_curve, columns=['date', 'value'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Calculate returns
        df['return'] = df['value'].pct_change()
        
        # Basic metrics
        start_value = df['value'].iloc[0]
        end_value = df['value'].iloc[-1]
        total_return = (end_value / start_value) - 1
        
        # CAGR
        days = (df.index[-1] - df.index[0]).days
        years = days / 365.25
        cagr = (end_value / start_value) ** (1 / years) - 1 if years > 0 else 0
        
        # Volatility (annualized)
        # Convert monthly returns to annualized
        monthly_vol = df['return'].std()
        annualized_vol = monthly_vol * np.sqrt(12)  # Monthly to annual
        
        # Sharpe ratio
        risk_free_rate = 0.05 / 12  # ~5% annual, monthly
        excess_returns = df['return'] - risk_free_rate
        sharpe = (excess_returns.mean() / df['return'].std()) * np.sqrt(12) if df['return'].std() > 0 else 0
        
        # Max drawdown
        peak = df['value'].expanding().max()
        drawdown = (df['value'] - peak) / peak
        max_drawdown = drawdown.min()
        
        # Sector diversity
        sector_counts = {}
        for ticker, (shares, price, sector) in final_holdings.items():
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        
        return {
            'start_date': df.index[0].strftime('%Y-%m-%d'),
            'end_date': df.index[-1].strftime('%Y-%m-%d'),
            'years': years,
            'start_value': start_value,
            'end_value': end_value,
            'total_return': total_return,
            'cagr': cagr,
            'volatility': annualized_vol,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'num_positions': len(final_holdings),
            'num_sectors': len(sector_counts),
            'sector_counts': sector_counts,
        }
    
    def save_results(self, results: Dict, output_dir: str = "results"):
        """Save backtest results to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save summary JSON
        summary = {k: v for k, v in results.items() if k not in ['equity_curve', 'holdings_history']}
        summary_path = os.path.join(output_dir, 'phase6_backtest_results.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Results saved to {summary_path}")
        
        # Save equity curve
        if results.get('equity_curve'):
            eq_df = pd.DataFrame(results['equity_curve'], columns=['date', 'value'])
            eq_path = os.path.join(output_dir, 'phase6_equity_curve.csv')
            eq_df.to_csv(eq_path, index=False)
            logger.info(f"Equity curve saved to {eq_path}")
    
    def print_results(self, results: Dict):
        """Print formatted backtest results."""
        print("\n" + "=" * 60)
        print("PHASE 6 BACKTEST RESULTS")
        print("=" * 60)
        
        print(f"\nPeriod: {results.get('start_date')} to {results.get('end_date')}")
        print(f"Duration: {results.get('years', 0):.2f} years")
        
        print(f"\nPerformance Metrics:")
        print(f"  Total Return: {results.get('total_return', 0)*100:.2f}%")
        print(f"  CAGR: {results.get('cagr', 0)*100:.2f}%")
        print(f"  Volatility: {results.get('volatility', 0)*100:.2f}%")
        print(f"  Sharpe Ratio: {results.get('sharpe', 0):.2f}")
        print(f"  Max Drawdown: {results.get('max_drawdown', 0)*100:.2f}%")
        
        print(f"\nPortfolio Composition:")
        print(f"  Final Positions: {results.get('num_positions', 0)}")
        print(f"  Sectors Covered: {results.get('num_sectors', 0)}")
        
        if results.get('sector_counts'):
            print(f"\n  Sector Distribution:")
            for sector, count in sorted(results['sector_counts'].items(), key=lambda x: -x[1]):
                print(f"    {sector}: {count} positions")
        
        # Check against targets
        print("\n" + "-" * 60)
        print("Phase 6 Targets Check:")
        
        cagr = results.get('cagr', 0) * 100
        sharpe = results.get('sharpe', 0)
        max_dd = abs(results.get('max_drawdown', 0) * 100)
        n_sectors = results.get('num_sectors', 0)
        
        print(f"  CAGR > 17%: {cagr:.2f}% {'✅' if cagr > 17 else '❌'}")
        print(f"  Sharpe > 1.0: {sharpe:.2f} {'✅' if sharpe > 1.0 else '❌'}")
        print(f"  Max DD < 17%: {max_dd:.2f}% {'✅' if max_dd < 17 else '❌'}")
        print(f"  Sectors >= 4: {n_sectors} {'✅' if n_sectors >= 4 else '❌'}")
        
        print("\n" + "=" * 60)


def main():
    """Run Phase 6 backtest."""
    print("=" * 60)
    print("PHASE 6: Full Universe TDA+Momentum Strategy")
    print("=" * 60)
    
    # Initialize backtest
    backtest = Phase6Backtest(
        start_date="2020-01-01",
        end_date="2025-01-01",
        initial_capital=100000.0,
        n_stocks=30,
        rebalance_frequency="monthly",
        use_cache=True,
        provider="yfinance",
    )
    
    # Step 1: Initialize universe
    universe = backtest.initialize_universe(max_stocks=50)  # Start with 50 for quick testing
    print(f"\nUniverse: {len(universe)} stocks")
    
    # Step 2: Fetch data
    data = backtest.fetch_data()
    print(f"Data fetched: {len(data)} stocks")
    
    # Step 3: Compute TDA features
    tda_features = backtest.compute_tda_features()
    print(f"TDA features: {len(tda_features)} stocks")
    
    # Step 4: Run backtest
    results = backtest.run_backtest()
    
    # Step 5: Print and save results
    backtest.print_results(results)
    backtest.save_results(results)
    
    print(f"\nBacktest completed in {results.get('elapsed_time', 0):.1f} seconds")
    
    return results


if __name__ == "__main__":
    main()
