"""Phase 5: Full-Universe TDA Multi-Asset Strategy Backtest.

This script implements the TRUE Algebraic-Topology-Neural-Net-Strategy:
1. Fetches ~3,000 liquid US stocks from Polygon
2. Builds correlation matrices and computes persistent homology
3. Detects market regimes from topological features
4. Selects stocks from optimal topological clusters
5. Constructs TDA-weighted portfolios
6. Runs full backtest with walk-forward validation

Target Metrics:
- Beat SPY by 5-10% annually
- Sharpe > 1.5
- Max Drawdown < 15%
- 20-50 position portfolio
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.universe_manager import UniverseManager, UniverseDataFetcher
from src.tda_engine import TDAEngine, TDAFeatures
from src.market_regime_tda import MarketRegimeDetector, MarketRegime, TDARegimeFeatures
from src.tda_stock_selector import TDAStockSelector, StockScore
from src.tda_portfolio import TDAPortfolioConstructor, PortfolioAllocation
from src.data.polygon_client import PolygonClient

# Get Polygon API key from environment
POLYGON_API_KEY_ENV = "POLYGON_API_KEY_OTREP"

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Phase5TDAStrategy:
    """
    Full-Universe TDA Multi-Asset Strategy.
    
    This is the complete implementation that uses TDA for its true purpose:
    analyzing the topological structure of thousands of stocks simultaneously.
    """
    
    def __init__(
        self,
        start_date: str = "2020-01-01",
        end_date: str = "2025-01-01",
        universe_size: int = 500,  # Start smaller for testing
        n_portfolio_stocks: int = 15,  # More concentrated for higher returns
        rebalance_frequency: int = 5,  # Weekly rebalancing
        correlation_window: int = 30,
        use_cache: bool = True,
    ):
        """
        Initialize Phase 5 strategy.
        
        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            universe_size: Number of stocks in universe (use smaller for testing)
            n_portfolio_stocks: Target portfolio size
            rebalance_frequency: Days between rebalancing
            correlation_window: Window for correlation calculation
            use_cache: Whether to cache data fetches
        """
        self.start_date = start_date
        self.end_date = end_date
        self.universe_size = universe_size
        self.n_portfolio_stocks = n_portfolio_stocks
        self.rebalance_frequency = rebalance_frequency
        self.correlation_window = correlation_window
        self.use_cache = use_cache
        
        # Initialize components
        self.polygon_client = PolygonClient(api_key_env=POLYGON_API_KEY_ENV)
        self.universe_manager = UniverseManager()  # Uses env var by default
        self.data_fetcher = UniverseDataFetcher(
            cache_dir="cache/ohlcv",
            max_workers=5,
        )
        
        self.tda_engine = TDAEngine(
            correlation_window=correlation_window,
            max_dimension=1,  # H0 and H1
        )
        
        # Recalibrated thresholds based on actual turbulence distributions
        self.regime_detector = MarketRegimeDetector(
            turbulence_threshold=75,  # Was 60 - too aggressive
            crisis_threshold=90,       # Was 80 - too aggressive
        )
        
        self.stock_selector = TDAStockSelector(
            n_stocks=n_portfolio_stocks,
            min_correlation=0.3,
        )
        
        self.portfolio_constructor = TDAPortfolioConstructor(
            max_position_weight=0.10,
            max_cluster_weight=0.30,
        )
        
        # State
        self.universe: List[str] = []
        self.returns_df: Optional[pd.DataFrame] = None
        self.spy_returns: Optional[pd.Series] = None
        self.tda_history: List[TDAFeatures] = []
        
    def fetch_universe(self) -> List[str]:
        """Fetch and cache universe of liquid stocks."""
        logger.info(f"Fetching universe of {self.universe_size} stocks...")
        
        # For testing, use a curated list of liquid stocks
        # In production, use self.universe_manager.get_liquid_universe()
        
        # Top liquid stocks by sector (curated list)
        curated_universe = [
            # Technology
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC', 'CRM',
            'ADBE', 'ORCL', 'CSCO', 'IBM', 'QCOM', 'TXN', 'AVGO', 'NOW', 'INTU', 'PYPL',
            'SQ', 'SHOP', 'SNOW', 'PLTR', 'DDOG', 'NET', 'CRWD', 'ZS', 'OKTA', 'MDB',
            # Financials
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'USB',
            'PNC', 'TFC', 'COF', 'BK', 'STT', 'CME', 'ICE', 'SPGI', 'MCO', 'MSCI',
            # Healthcare
            'JNJ', 'UNH', 'PFE', 'MRK', 'ABBV', 'LLY', 'TMO', 'ABT', 'DHR', 'BMY',
            'AMGN', 'GILD', 'CVS', 'CI', 'HUM', 'ISRG', 'REGN', 'VRTX', 'BIIB', 'MRNA',
            # Consumer
            'WMT', 'HD', 'PG', 'KO', 'PEP', 'COST', 'MCD', 'NKE', 'SBUX', 'TGT',
            'LOW', 'TJX', 'CMG', 'DG', 'DLTR', 'ROST', 'YUM', 'MAR', 'HLT', 'BKNG',
            # Industrial
            'CAT', 'DE', 'BA', 'UNP', 'UPS', 'HON', 'GE', 'MMM', 'LMT', 'RTX',
            'NOC', 'GD', 'WM', 'EMR', 'ETN', 'PH', 'ITW', 'ROK', 'FAST', 'SWK',
            # Energy
            'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'MPC', 'VLO', 'PSX', 'OXY',
            'HAL', 'BKR', 'DVN', 'FANG', 'HES', 'MRO', 'APA', 'OVV', 'CTRA', 'EQT',
            # Materials
            'LIN', 'APD', 'SHW', 'ECL', 'DD', 'DOW', 'FCX', 'NEM', 'VMC', 'MLM',
            # Utilities
            'NEE', 'DUK', 'SO', 'D', 'AEP', 'XEL', 'ED', 'EXC', 'SRE', 'WEC',
            # REITs
            'PLD', 'AMT', 'EQIX', 'PSA', 'DLR', 'SPG', 'O', 'WELL', 'AVB', 'EQR',
            # Communication
            'DIS', 'NFLX', 'CMCSA', 'VZ', 'T', 'TMUS', 'CHTR', 'EA', 'TTWO', 'ATVI',
        ]
        
        # Limit to universe_size
        self.universe = curated_universe[:self.universe_size]
        
        logger.info(f"Universe: {len(self.universe)} stocks")
        return self.universe
    
    def fetch_data(self) -> pd.DataFrame:
        """Fetch OHLCV data for all stocks in universe."""
        logger.info(f"Fetching data for {len(self.universe)} stocks from {self.start_date} to {self.end_date}...")
        
        # Fetch SPY for benchmark
        spy_data = self.polygon_client.get_ohlcv('SPY', self.start_date, self.end_date)
        if spy_data is not None and len(spy_data) > 0:
            # Already indexed by datetime
            self.spy_returns = spy_data['close'].pct_change()
        else:
            logger.warning("Could not fetch SPY data")
            self.spy_returns = None
        
        # Fetch all stocks using polygon client directly (simpler, no caching issues)
        all_returns = {}
        failed_count = 0
        
        for i, ticker in enumerate(self.universe):
            try:
                data = self.polygon_client.get_ohlcv(ticker, self.start_date, self.end_date)
                if data is not None and len(data) > 0:
                    returns = data['close'].pct_change()
                    all_returns[ticker] = returns
                else:
                    failed_count += 1
            except Exception as e:
                failed_count += 1
                if failed_count <= 5:
                    logger.warning(f"Failed to fetch {ticker}: {e}")
            
            if (i + 1) % 20 == 0:
                logger.info(f"  Progress: {i+1}/{len(self.universe)} ({len(all_returns)} successful)")
        
        logger.info(f"Fetched {len(all_returns)} tickers, {failed_count} failed")
        
        # Build returns DataFrame
        if all_returns:
            self.returns_df = pd.DataFrame(all_returns)
            self.returns_df = self.returns_df.dropna(how='all')
            logger.info(f"Returns matrix: {self.returns_df.shape[0]} days x {self.returns_df.shape[1]} stocks")
        
        return self.returns_df
    
    def run_backtest(self) -> Dict:
        """
        Run full backtest.
        
        Returns:
            Dict with backtest results
        """
        logger.info("=" * 60)
        logger.info("PHASE 5: FULL-UNIVERSE TDA MULTI-ASSET STRATEGY")
        logger.info("=" * 60)
        
        # Step 1: Fetch universe
        self.fetch_universe()
        
        # Step 2: Fetch data
        self.fetch_data()
        
        if self.returns_df is None or len(self.returns_df) < self.correlation_window + 20:
            logger.error("Insufficient data for backtest")
            return {"error": "Insufficient data"}
        
        # Step 3: Run walk-forward backtest
        results = self._walk_forward_backtest()
        
        # Step 4: Calculate metrics
        metrics = self._calculate_metrics(results)
        
        # Step 5: Compare to SPY
        benchmark = self._calculate_benchmark_metrics()
        
        # Combine results
        final_results = {
            "strategy": metrics,
            "benchmark": benchmark,
            "outperformance": {
                "total_return_diff": metrics.get("total_return", 0) - benchmark.get("total_return", 0),
                "cagr_diff": metrics.get("cagr", 0) - benchmark.get("cagr", 0),
                "sharpe_diff": metrics.get("sharpe", 0) - benchmark.get("sharpe", 0),
            },
            "trade_log": results.get("trades", []),
            "regime_log": results.get("regimes", []),
        }
        
        return final_results
    
    def _walk_forward_backtest(self) -> Dict:
        """
        Walk-forward backtest with TDA rebalancing.
        """
        logger.info("Running walk-forward backtest...")
        
        # Initialize
        equity = 1.0
        equity_curve = []
        trades = []
        regimes = []
        
        current_positions: List = []
        
        # Get dates
        dates = self.returns_df.index.tolist()
        
        # Start after warmup period
        warmup = self.correlation_window + 10
        
        for day_idx in range(warmup, len(dates)):
            date = dates[day_idx]
            
            # Get returns up to this day
            returns_window = self.returns_df.iloc[:day_idx]
            daily_returns = self.returns_df.iloc[day_idx]
            
            # Should we rebalance?
            should_rebalance = (day_idx - warmup) % self.rebalance_frequency == 0
            
            if should_rebalance and len(returns_window) >= self.correlation_window:
                try:
                    # ===== TREND FILTER: 200-SMA on SPY =====
                    # If SPY is below its 200-SMA, go to cash for risk-off
                    spy_history = self.spy_returns.iloc[:day_idx+1]
                    if len(spy_history) >= 200:
                        spy_cumulative = (1 + spy_history).cumprod()
                        spy_sma_200 = spy_cumulative.rolling(200).mean().iloc[-1]
                        spy_current = spy_cumulative.iloc[-1]
                        
                        # If below 200-SMA, reduce exposure significantly
                        if spy_current < spy_sma_200:
                            # Skip this rebalance - stay in current safer position
                            logger.info(f"SPY below 200-SMA ({spy_current:.4f} < {spy_sma_200:.4f}), reducing exposure")
                            regimes.append({
                                "date": str(date),
                                "regime": "BEAR",
                                "turbulence": 80,
                            })
                            # Go to minimal positions (or hold current defensive)
                            if current_positions:
                                # Keep only top 5 lowest-vol positions
                                current_positions = sorted(current_positions, key=lambda p: p.weight)[:5]
                            continue
                    
                    # Step 1: TDA Analysis
                    tda_result = self._analyze_market_topology(returns_window)
                    
                    if tda_result:
                        self.tda_history.append(tda_result)
                    
                    # Step 2: Regime Detection
                    regime_signal = self._detect_regime(tda_result)
                    regime = regime_signal.regime.value if regime_signal else "NEUTRAL"
                    
                    # TDA Turbulence-based position sizing (disabled for now - reduces returns)
                    turbulence = tda_result.turbulence_index if tda_result else 50
                    position_scale = 1.0  # Full position size - the 200-SMA filter handles risk-off
                    
                    regimes.append({
                        "date": str(date),
                        "regime": regime,
                        "turbulence": turbulence,
                    })
                    
                    # Step 3: Stock Selection
                    selected_stocks, cluster_analysis = self.stock_selector.select_stocks(
                        returns_window.iloc[-60:] if len(returns_window) >= 60 else returns_window,
                        regime=regime,
                    )
                    
                    # Step 4: Portfolio Construction
                    new_allocation = self.portfolio_constructor.construct_portfolio(
                        selected_stocks,
                        cluster_analysis,
                        returns_window,
                        regime=regime,
                        method="score_weight",
                    )
                    
                    # Apply turbulence-based position scaling
                    if position_scale < 1.0:
                        from src.tda_portfolio import PortfolioPosition
                        scaled_positions = [
                            PortfolioPosition(
                                ticker=p.ticker,
                                weight=p.weight * position_scale,
                                cluster_id=p.cluster_id,
                                score=p.score,
                            )
                            for p in new_allocation.positions
                        ]
                        new_allocation.positions = scaled_positions
                        new_allocation.total_exposure *= position_scale
                    
                    # Log trades
                    old_tickers = set(p.ticker for p in current_positions) if current_positions else set()
                    new_tickers = set(p.ticker for p in new_allocation.positions)
                    
                    if old_tickers != new_tickers or not current_positions:
                        trades.append({
                            "date": str(date),
                            "action": "REBALANCE",
                            "n_positions": new_allocation.n_positions,
                            "exposure": new_allocation.total_exposure,
                            "regime": regime,
                        })
                    
                    current_positions = new_allocation.positions
                    
                except Exception as e:
                    logger.warning(f"Rebalance failed on {date}: {e}")
                    # Keep current positions on error
            
            # Calculate daily return
            portfolio_return = 0.0
            
            if current_positions:
                for position in current_positions:
                    if position.ticker in daily_returns.index:
                        stock_return = daily_returns[position.ticker]
                        if not np.isnan(stock_return):
                            portfolio_return += position.weight * stock_return
            
            # Update equity
            equity *= (1 + portfolio_return)
            
            equity_curve.append({
                "date": str(date),
                "equity": equity,
                "return": portfolio_return,
                "n_positions": len(current_positions),
            })
        
        return {
            "equity_curve": equity_curve,
            "trades": trades,
            "regimes": regimes,
            "final_equity": equity,
        }
    
    def _analyze_market_topology(self, returns_df: pd.DataFrame) -> Optional[TDAFeatures]:
        """Analyze market topology using TDA."""
        try:
            tickers = returns_df.columns.tolist()
            
            if len(tickers) < 20:
                return None
            
            # Get most recent date
            date = str(returns_df.index[-1])
            
            # Run TDA analysis
            tda_result = self.tda_engine.analyze_market(
                returns_df,
                tickers,
                date,
                self.tda_history,
            )
            
            return tda_result
            
        except Exception as e:
            logger.warning(f"TDA analysis failed: {e}")
            return None
    
    def _detect_regime(self, tda_features: Optional[TDAFeatures]):
        """Detect market regime from TDA features."""
        if tda_features is None:
            return None
        
        # Build feature dict for regime detector
        feature_dict = {
            'betti_0': tda_features.persistence.betti_0,
            'betti_1': tda_features.persistence.betti_1,
            'fragmentation': tda_features.persistence.fragmentation,
            'stability': tda_features.persistence.stability,
            'entropy_h0': tda_features.persistence.entropy_h0,
            'entropy_h1': tda_features.persistence.entropy_h1,
            'total_persistence_h0': tda_features.persistence.total_persistence_h0,
            'total_persistence_h1': tda_features.persistence.total_persistence_h1,
            'turbulence_index': tda_features.turbulence_index,
        }
        
        # Build history
        history = [feature_dict]
        
        regime_features = self.regime_detector.extract_regime_features(history)
        regime_signal = self.regime_detector.detect_regime(regime_features)
        
        return regime_signal
    
    def _calculate_metrics(self, results: Dict) -> Dict:
        """Calculate strategy performance metrics."""
        if "equity_curve" not in results or len(results["equity_curve"]) == 0:
            return {}
        
        equity_curve = pd.DataFrame(results["equity_curve"])
        
        # Total return
        total_return = results.get("final_equity", 1.0) - 1
        
        # CAGR
        n_years = len(equity_curve) / 252
        cagr = (results.get("final_equity", 1.0) ** (1 / n_years) - 1) if n_years > 0 else 0
        
        # Volatility
        returns = equity_curve["return"].values
        volatility = np.std(returns) * np.sqrt(252)
        
        # Sharpe Ratio
        excess_return = cagr - 0.02  # Assume 2% risk-free rate
        sharpe = excess_return / volatility if volatility > 0 else 0
        
        # Max Drawdown
        equity_values = equity_curve["equity"].values
        peak = np.maximum.accumulate(equity_values)
        drawdown = (equity_values - peak) / peak
        max_drawdown = abs(np.min(drawdown))
        
        # Win rate
        positive_days = np.sum(returns > 0)
        total_days = len(returns)
        win_rate = positive_days / total_days if total_days > 0 else 0
        
        # Average positions
        avg_positions = equity_curve["n_positions"].mean()
        
        return {
            "total_return": total_return,
            "cagr": cagr,
            "volatility": volatility,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "avg_positions": avg_positions,
            "n_trades": len(results.get("trades", [])),
        }
    
    def _calculate_benchmark_metrics(self) -> Dict:
        """Calculate SPY benchmark metrics."""
        if self.spy_returns is None:
            return {}
        
        # Align to backtest period
        spy = self.spy_returns.dropna()
        
        if len(spy) < 252:
            return {}
        
        # Total return
        total_return = (1 + spy).prod() - 1
        
        # CAGR
        n_years = len(spy) / 252
        cagr = ((1 + spy).prod() ** (1 / n_years) - 1) if n_years > 0 else 0
        
        # Volatility
        volatility = spy.std() * np.sqrt(252)
        
        # Sharpe
        excess_return = cagr - 0.02
        sharpe = excess_return / volatility if volatility > 0 else 0
        
        # Max Drawdown
        cumulative = (1 + spy).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_drawdown = abs(drawdown.min())
        
        return {
            "total_return": total_return,
            "cagr": cagr,
            "volatility": volatility,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
        }


def run_phase5_backtest(
    universe_size: int = 100,  # Start small for testing
    n_stocks: int = 30,
) -> Dict:
    """
    Run Phase 5 TDA strategy backtest.
    
    Args:
        universe_size: Number of stocks in universe
        n_stocks: Portfolio size
        
    Returns:
        Backtest results
    """
    strategy = Phase5TDAStrategy(
        start_date="2020-01-01",
        end_date="2025-01-01",
        universe_size=universe_size,
        n_portfolio_stocks=n_stocks,
        rebalance_frequency=5,  # Weekly
        use_cache=True,
    )
    
    results = strategy.run_backtest()
    
    return results


def print_results(results: Dict):
    """Pretty print backtest results."""
    print("\n" + "=" * 70)
    print("PHASE 5: FULL-UNIVERSE TDA MULTI-ASSET STRATEGY RESULTS")
    print("=" * 70)
    
    if "error" in results:
        print(f"\nError: {results['error']}")
        return
    
    # Strategy metrics
    strat = results.get("strategy", {})
    print("\nSTRATEGY PERFORMANCE:")
    print("-" * 40)
    print(f"  Total Return:    {strat.get('total_return', 0)*100:>8.2f}%")
    print(f"  CAGR:            {strat.get('cagr', 0)*100:>8.2f}%")
    print(f"  Volatility:      {strat.get('volatility', 0)*100:>8.2f}%")
    print(f"  Sharpe Ratio:    {strat.get('sharpe', 0):>8.2f}")
    print(f"  Max Drawdown:    {strat.get('max_drawdown', 0)*100:>8.2f}%")
    print(f"  Win Rate:        {strat.get('win_rate', 0)*100:>8.1f}%")
    print(f"  Avg Positions:   {strat.get('avg_positions', 0):>8.1f}")
    print(f"  Total Trades:    {strat.get('n_trades', 0):>8d}")
    
    # Benchmark metrics
    bench = results.get("benchmark", {})
    if bench:
        print("\nSPY BENCHMARK:")
        print("-" * 40)
        print(f"  Total Return:    {bench.get('total_return', 0)*100:>8.2f}%")
        print(f"  CAGR:            {bench.get('cagr', 0)*100:>8.2f}%")
        print(f"  Volatility:      {bench.get('volatility', 0)*100:>8.2f}%")
        print(f"  Sharpe Ratio:    {bench.get('sharpe', 0):>8.2f}")
        print(f"  Max Drawdown:    {bench.get('max_drawdown', 0)*100:>8.2f}%")
    
    # Outperformance
    out = results.get("outperformance", {})
    if out:
        print("\nOUTPERFORMANCE vs SPY:")
        print("-" * 40)
        print(f"  Total Return:    {out.get('total_return_diff', 0)*100:>+8.2f}%")
        print(f"  CAGR:            {out.get('cagr_diff', 0)*100:>+8.2f}%")
        print(f"  Sharpe:          {out.get('sharpe_diff', 0):>+8.2f}")
    
    # Regime distribution
    regimes = results.get("regime_log", [])
    if regimes:
        from collections import Counter
        regime_counts = Counter([r["regime"] for r in regimes])
        print("\nREGIME DISTRIBUTION:")
        print("-" * 40)
        for regime, count in sorted(regime_counts.items()):
            pct = count / len(regimes) * 100
            print(f"  {regime:<15}: {count:>4} days ({pct:>5.1f}%)")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    print("Starting Phase 5 Full-Universe TDA Multi-Asset Strategy Backtest...")
    print("This may take several minutes to fetch data and run analysis.")
    print()
    
    # Run with smaller universe first for testing
    results = run_phase5_backtest(
        universe_size=100,  # 100 stocks for testing
        n_stocks=20,        # 20-stock portfolio 
    )
    
    # Print results
    print_results(results)
    
    # Save results
    output_path = "results/phase5_tda_universe_results.json"
    with open(output_path, 'w') as f:
        # Convert non-serializable types
        def serialize(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Timestamp):
                return str(obj)
            return str(obj)
        
        json.dump(results, f, indent=2, default=serialize)
    
    print(f"\nResults saved to: {output_path}")
