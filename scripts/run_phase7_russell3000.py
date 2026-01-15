"""
Phase 7: Russell 3000 Scalable Backtest.

Main orchestration script that:
1. Fetches Russell 3000 universe data
2. Applies multi-stage screening filters
3. Computes TDA features in parallel
4. Runs momentum + TDA backtest on screened universe
5. Generates comprehensive scalability report

Targets:
- Successfully screen Russell 3000 → 200-300 tradeable universe
- Backtest completes in <30 minutes
- CAGR > 12% on larger universe
- Sharpe > 0.8
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.russell3000_provider import Russell3000DataProvider
from src.universe_screener import UniverseScreener, ScreeningResult
from src.tda_engine_parallel import ParallelTDAEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Phase7Config:
    """Configuration for Phase 7 backtest."""
    # Universe
    universe_size: str = 'liquid'  # 'liquid' (~500), 'medium' (~1000), 'full'
    target_screened_size: int = 200  # Target universe after screening
    
    # Date range
    start_date: str = '2021-01-01'
    end_date: str = '2024-12-31'
    
    # Strategy
    momentum_weight: float = 0.70
    tda_weight: float = 0.30
    n_stocks: int = 30  # Portfolio size
    rebalance_frequency: str = 'monthly'  # 'weekly', 'monthly'
    
    # Screening thresholds
    min_dollar_volume: float = 5_000_000
    min_price: float = 5.0
    min_trading_days: int = 252
    momentum_lookback: int = 126  # 6 months
    
    # Parallel processing
    n_data_workers: int = 20
    n_tda_workers: int = 4
    tda_batch_size: int = 50
    
    # Costs
    transaction_cost_bps: float = 10  # 10 bps round-trip
    
    # Caching
    cache_dir: str = 'data/cache'
    use_cache: bool = True
    
    # Output
    results_dir: str = 'results'


@dataclass
class TimingStats:
    """Timing statistics for scalability analysis."""
    data_fetch_time: float = 0.0
    screening_time: float = 0.0
    tda_compute_time: float = 0.0
    backtest_time: float = 0.0
    total_time: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class BacktestResult:
    """Backtest results."""
    total_return: float
    cagr: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    n_trades: int
    avg_positions: float
    sector_distribution: Dict[str, float]
    
    # Comparison with Phase 6
    phase6_cagr: float = 0.20  # Reference
    phase6_sharpe: float = 1.20
    phase6_max_dd: float = -0.142
    
    def to_dict(self) -> Dict:
        return asdict(self)


class Phase7Backtester:
    """
    Phase 7 scalable backtester.
    
    Implements momentum + TDA strategy on large universe.
    """
    
    def __init__(self, config: Phase7Config):
        self.config = config
        
        # Initialize components
        self.data_provider = Russell3000DataProvider(
            cache_dir=config.cache_dir,
            n_workers=config.n_data_workers,
        )
        
        self.screener = UniverseScreener(
            min_dollar_volume=config.min_dollar_volume,
            min_price=config.min_price,
            min_trading_days=config.min_trading_days,
            momentum_lookback=config.momentum_lookback,
            target_universe_size=config.target_screened_size,
        )
        
        self.tda_engine = ParallelTDAEngine(
            n_workers=config.n_tda_workers,
            batch_size=config.tda_batch_size,
            cache_dir=os.path.join(config.cache_dir, 'tda'),
        )
        
        # State
        self.ohlcv_dict: Dict[str, pd.DataFrame] = {}
        self.screened_tickers: List[str] = []
        self.tda_features: Dict[str, Dict] = {}
        self.timing = TimingStats()
        
    def fetch_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch universe data."""
        logger.info("\n" + "="*60)
        logger.info("STAGE 1: Data Fetching")
        logger.info("="*60)
        
        start_time = time.time()
        
        self.ohlcv_dict = self.data_provider.fetch_universe(
            universe_size=self.config.universe_size,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            force_download=not self.config.use_cache,
        )
        
        self.timing.data_fetch_time = time.time() - start_time
        
        logger.info(f"Fetched {len(self.ohlcv_dict)} stocks in {self.timing.data_fetch_time:.1f}s")
        
        return self.ohlcv_dict
    
    def screen_universe(self) -> ScreeningResult:
        """Apply screening filters."""
        logger.info("\n" + "="*60)
        logger.info("STAGE 2: Universe Screening")
        logger.info("="*60)
        
        start_time = time.time()
        
        screening_result = self.screener.get_final_universe(
            self.ohlcv_dict,
            size=self.config.target_screened_size,
        )
        
        self.screened_tickers = screening_result.passed_tickers
        self.timing.screening_time = time.time() - start_time
        
        self.screener.print_screening_summary(screening_result)
        
        logger.info(f"Screening completed in {self.timing.screening_time:.1f}s")
        
        return screening_result
    
    def compute_tda_features(self) -> Dict[str, Dict]:
        """Compute TDA features for screened universe."""
        logger.info("\n" + "="*60)
        logger.info("STAGE 3: TDA Computation")
        logger.info("="*60)
        
        start_time = time.time()
        
        # Filter to screened tickers
        screened_ohlcv = {
            t: df for t, df in self.ohlcv_dict.items()
            if t in self.screened_tickers
        }
        
        self.tda_features = self.tda_engine.compute_batch_tda(
            screened_ohlcv,
            use_cache=self.config.use_cache,
        )
        
        self.timing.tda_compute_time = time.time() - start_time
        
        stats = self.tda_engine.get_compute_stats()
        logger.info(f"TDA computed for {len(self.tda_features)} stocks in {self.timing.tda_compute_time:.1f}s")
        logger.info(f"TDA success rate: {stats['success_rate']:.1%}")
        
        return self.tda_features
    
    def calculate_momentum_score(
        self,
        prices: pd.Series,
        lookbacks: List[int] = [21, 63, 126, 252],
    ) -> float:
        """Calculate multi-period momentum score."""
        if len(prices) < max(lookbacks):
            return 0.0
        
        scores = []
        weights = [0.4, 0.3, 0.2, 0.1]
        
        for lb, w in zip(lookbacks, weights):
            if len(prices) >= lb:
                ret = (prices.iloc[-1] / prices.iloc[-lb]) - 1
                scores.append(ret * w)
        
        return sum(scores) if scores else 0.0
    
    def get_tda_score(self, ticker: str) -> float:
        """Get TDA score for ticker."""
        if ticker in self.tda_features:
            return self.tda_features[ticker].get('tda_score', 0)
        return 0.0
    
    def calculate_combined_score(
        self,
        ticker: str,
        prices: pd.Series,
    ) -> float:
        """Calculate combined momentum + TDA score."""
        mom_score = self.calculate_momentum_score(prices)
        tda_score = self.get_tda_score(ticker)
        
        # Normalize TDA score (typically 0-1 range already)
        # Momentum score can be negative to positive
        
        combined = (
            self.config.momentum_weight * mom_score +
            self.config.tda_weight * (tda_score / 10)  # Scale TDA down
        )
        
        return combined
    
    def run_backtest(self) -> BacktestResult:
        """
        Run momentum + TDA backtest on screened universe.
        """
        logger.info("\n" + "="*60)
        logger.info("STAGE 4: Running Backtest")
        logger.info("="*60)
        
        start_time = time.time()
        
        # Get date range from data
        all_dates = set()
        for ticker in self.screened_tickers[:50]:  # Sample for dates
            if ticker in self.ohlcv_dict:
                df = self.ohlcv_dict[ticker]
                dates = df.index[
                    (df.index >= self.config.start_date) &
                    (df.index <= self.config.end_date)
                ]
                all_dates.update(dates)
        
        if not all_dates:
            logger.error("No trading dates found")
            return BacktestResult(0, 0, 0, 0, 0, 0, 0, {})
        
        trading_dates = sorted(list(all_dates))
        logger.info(f"Backtest period: {trading_dates[0]} to {trading_dates[-1]}")
        logger.info(f"Trading days: {len(trading_dates)}")
        
        # Initialize portfolio
        initial_capital = 100_000
        cash = initial_capital
        positions: Dict[str, Dict] = {}
        equity_curve = [initial_capital]
        trade_count = 0
        position_counts = []
        sector_exposures = []
        
        # Determine rebalance dates
        rebalance_dates = set()
        for dt in trading_dates:
            if isinstance(dt, str):
                dt = pd.Timestamp(dt)
            
            if self.config.rebalance_frequency == 'monthly':
                if dt.day <= 5:
                    rebalance_dates.add(dt)
            elif self.config.rebalance_frequency == 'weekly':
                if dt.weekday() == 0:  # Monday
                    rebalance_dates.add(dt)
        
        logger.info(f"Rebalance dates: {len(rebalance_dates)}")
        
        # Backtest loop
        for i, date in enumerate(trading_dates):
            # Skip first year for lookback
            if i < 252:
                equity_curve.append(cash)
                continue
            
            # Rebalance
            if date in rebalance_dates:
                # Score all screened stocks
                scores = {}
                for ticker in self.screened_tickers:
                    if ticker not in self.ohlcv_dict:
                        continue
                    
                    df = self.ohlcv_dict[ticker]
                    if date not in df.index:
                        continue
                    
                    prices = df['close'].loc[:date]
                    if len(prices) < 252:
                        continue
                    
                    score = self.calculate_combined_score(ticker, prices)
                    scores[ticker] = score
                
                # Rank and select top N
                ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                target_tickers = [t for t, s in ranked[:self.config.n_stocks] if s > 0]
                
                # Calculate current portfolio value
                portfolio_value = cash
                for ticker, pos in positions.items():
                    if ticker in self.ohlcv_dict:
                        df = self.ohlcv_dict[ticker]
                        if date in df.index:
                            portfolio_value += pos['shares'] * df['close'].loc[date]
                
                # Equal weight allocation
                target_weight = 1.0 / max(1, len(target_tickers))
                
                # Close positions not in target
                for ticker in list(positions.keys()):
                    if ticker not in target_tickers:
                        if ticker in self.ohlcv_dict:
                            df = self.ohlcv_dict[ticker]
                            if date in df.index:
                                price = df['close'].loc[date]
                                proceeds = positions[ticker]['shares'] * price
                                cost = proceeds * (self.config.transaction_cost_bps / 10000)
                                cash += proceeds - cost
                                trade_count += 1
                        
                        del positions[ticker]
                
                # Open/adjust positions
                for ticker in target_tickers:
                    if ticker not in self.ohlcv_dict:
                        continue
                    
                    df = self.ohlcv_dict[ticker]
                    if date not in df.index:
                        continue
                    
                    price = df['close'].loc[date]
                    target_value = portfolio_value * target_weight
                    target_shares = int(target_value / price)
                    
                    current_shares = positions.get(ticker, {}).get('shares', 0)
                    
                    if target_shares != current_shares:
                        delta = target_shares - current_shares
                        trade_value = abs(delta * price)
                        cost = trade_value * (self.config.transaction_cost_bps / 10000)
                        
                        if delta > 0:  # Buy
                            if cash >= trade_value + cost:
                                cash -= trade_value + cost
                                positions[ticker] = {
                                    'shares': target_shares,
                                    'entry_price': price,
                                }
                                trade_count += 1
                        else:  # Sell
                            cash += trade_value - cost
                            if target_shares == 0:
                                del positions[ticker]
                            else:
                                positions[ticker]['shares'] = target_shares
                            trade_count += 1
                
                position_counts.append(len(positions))
            
            # Calculate daily portfolio value
            portfolio_value = cash
            for ticker, pos in positions.items():
                if ticker in self.ohlcv_dict:
                    df = self.ohlcv_dict[ticker]
                    if date in df.index:
                        portfolio_value += pos['shares'] * df['close'].loc[date]
            
            equity_curve.append(portfolio_value)
        
        # Calculate metrics
        equity = pd.Series(equity_curve)
        returns = equity.pct_change().dropna()
        
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
        years = len(trading_dates) / 252
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        rolling_max = equity.expanding().max()
        drawdowns = (equity - rolling_max) / rolling_max
        max_dd = drawdowns.min()
        
        avg_positions = np.mean(position_counts) if position_counts else 0
        
        self.timing.backtest_time = time.time() - start_time
        
        # Get sector distribution
        sector_dist = {}
        for ticker in self.screened_tickers[:30]:
            sector = self.screener.get_sector(ticker)
            sector_dist[sector] = sector_dist.get(sector, 0) + 1
        total_sect = sum(sector_dist.values())
        sector_dist = {k: v/total_sect for k, v in sector_dist.items()}
        
        result = BacktestResult(
            total_return=total_return,
            cagr=cagr,
            sharpe=sharpe,
            max_drawdown=max_dd,
            win_rate=0.55,  # Placeholder
            n_trades=trade_count,
            avg_positions=avg_positions,
            sector_distribution=sector_dist,
        )
        
        logger.info(f"Backtest completed in {self.timing.backtest_time:.1f}s")
        
        return result
    
    def run_full_pipeline(self) -> Tuple[BacktestResult, TimingStats]:
        """
        Run full Phase 7 pipeline.
        
        Returns:
            Tuple of (backtest_result, timing_stats)
        """
        pipeline_start = time.time()
        
        print("\n" + "="*70)
        print("PHASE 7: SCALABLE UNIVERSE EXPANSION TO RUSSELL 3000")
        print("="*70)
        print(f"Universe size: {self.config.universe_size}")
        print(f"Target screened: {self.config.target_screened_size} stocks")
        print(f"Portfolio size: {self.config.n_stocks} stocks")
        print(f"Period: {self.config.start_date} to {self.config.end_date}")
        print("="*70)
        
        # Stage 1: Fetch data
        self.fetch_data()
        
        # Stage 2: Screen universe
        screening_result = self.screen_universe()
        
        # Stage 3: Compute TDA
        self.compute_tda_features()
        
        # Stage 4: Run backtest
        backtest_result = self.run_backtest()
        
        self.timing.total_time = time.time() - pipeline_start
        
        # Print results
        print("\n" + "="*70)
        print("PHASE 7 RESULTS")
        print("="*70)
        
        print(f"\n{'Performance Metrics':}")
        print("-"*40)
        print(f"  Total Return: {backtest_result.total_return:.1%}")
        print(f"  CAGR: {backtest_result.cagr:.1%}")
        print(f"  Sharpe Ratio: {backtest_result.sharpe:.2f}")
        print(f"  Max Drawdown: {backtest_result.max_drawdown:.1%}")
        print(f"  Trades: {backtest_result.n_trades}")
        print(f"  Avg Positions: {backtest_result.avg_positions:.1f}")
        
        print(f"\n{'Comparison vs Phase 6 (100 stocks)':}")
        print("-"*40)
        print(f"  {'Metric':<20} {'Phase 6':>12} {'Phase 7':>12} {'Delta':>12}")
        print(f"  {'CAGR':<20} {backtest_result.phase6_cagr:>11.1%} {backtest_result.cagr:>11.1%} "
              f"{backtest_result.cagr - backtest_result.phase6_cagr:>+11.1%}")
        print(f"  {'Sharpe':<20} {backtest_result.phase6_sharpe:>12.2f} {backtest_result.sharpe:>12.2f} "
              f"{backtest_result.sharpe - backtest_result.phase6_sharpe:>+12.2f}")
        print(f"  {'Max DD':<20} {backtest_result.phase6_max_dd:>11.1%} {backtest_result.max_drawdown:>11.1%}")
        
        print(f"\n{'Timing Statistics':}")
        print("-"*40)
        print(f"  Data Fetching: {self.timing.data_fetch_time:>8.1f}s")
        print(f"  Screening: {self.timing.screening_time:>8.1f}s")
        print(f"  TDA Computation: {self.timing.tda_compute_time:>8.1f}s")
        print(f"  Backtest: {self.timing.backtest_time:>8.1f}s")
        print(f"  {'TOTAL':} {self.timing.total_time:>8.1f}s")
        
        # Check targets
        print(f"\n{'Target Checks':}")
        print("-"*40)
        print(f"  CAGR > 12%: {backtest_result.cagr:.1%} {'✓' if backtest_result.cagr > 0.12 else '✗'}")
        print(f"  Sharpe > 0.8: {backtest_result.sharpe:.2f} {'✓' if backtest_result.sharpe > 0.8 else '✗'}")
        print(f"  Max DD > -20%: {backtest_result.max_drawdown:.1%} {'✓' if backtest_result.max_drawdown > -0.20 else '✗'}")
        print(f"  Total time < 30min: {self.timing.total_time/60:.1f}min {'✓' if self.timing.total_time < 1800 else '✗'}")
        
        return backtest_result, self.timing
    
    def save_results(
        self,
        backtest_result: BacktestResult,
        screening_result: ScreeningResult = None,
    ):
        """Save results to files."""
        results_dir = Path(self.config.results_dir)
        results_dir.mkdir(exist_ok=True)
        
        # Save JSON results
        results_dict = {
            'config': asdict(self.config),
            'backtest': backtest_result.to_dict(),
            'timing': self.timing.to_dict(),
            'data_stats': self.data_provider.get_fetch_stats(),
            'tda_stats': self.tda_engine.get_compute_stats(),
            'timestamp': datetime.now().isoformat(),
        }
        
        # Handle non-serializable types
        def clean_dict(d):
            cleaned = {}
            for k, v in d.items():
                if isinstance(v, (np.floating, np.integer)):
                    cleaned[k] = float(v)
                elif isinstance(v, dict):
                    cleaned[k] = clean_dict(v)
                elif isinstance(v, list):
                    cleaned[k] = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in v]
                else:
                    cleaned[k] = v
            return cleaned
        
        results_dict = clean_dict(results_dict)
        
        results_file = results_dir / 'phase7_russell3000_results.json'
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_file}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 7: Russell 3000 Scalable Backtest")
    parser.add_argument('--universe', choices=['liquid', 'medium', 'full', 'debug'], 
                       default='liquid', help='Universe size')
    parser.add_argument('--target-size', type=int, default=200, 
                       help='Target screened universe size')
    parser.add_argument('--portfolio-size', type=int, default=30,
                       help='Number of stocks in portfolio')
    parser.add_argument('--start', default='2021-01-01', help='Start date')
    parser.add_argument('--end', default='2024-12-31', help='End date')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching')
    
    args = parser.parse_args()
    
    config = Phase7Config(
        universe_size=args.universe,
        target_screened_size=args.target_size,
        n_stocks=args.portfolio_size,
        start_date=args.start,
        end_date=args.end,
        use_cache=not args.no_cache,
    )
    
    backtester = Phase7Backtester(config)
    backtest_result, timing = backtester.run_full_pipeline()
    backtester.save_results(backtest_result)
    
    return backtest_result


if __name__ == "__main__":
    main()
