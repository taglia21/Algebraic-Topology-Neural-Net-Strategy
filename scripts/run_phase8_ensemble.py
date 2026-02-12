"""
Phase 8: Full Ensemble Backtest Script

Orchestrates the complete Phase 8 ensemble multi-factor system:
1. Data fetch from Russell 3000 (1000 liquid stocks)
2. GICS sector classification
3. Parallel TDA computation
4. Ensemble factor scoring (Momentum + TDA + Value + Quality)
5. Regime-adaptive weighting
6. Walk-forward backtesting

Targets: CAGR >18%, Sharpe >1.2, MaxDD <15%, Runtime <5 min
"""

import os
import sys
import time
import json
import logging
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.russell3000_provider import Russell3000DataProvider, LIQUID_UNIVERSE
from src.data.sector_mapper import GICSSectorMapper, MANUAL_SECTOR_MAP
from src.data.hybrid_provider import HybridDataProvider
from src.tda_engine_parallel import ParallelTDAEngine
from src.ensemble_factor_model import (
    EnsembleFactorModel, FactorWeights, REGIME_WEIGHTS
)
from src.universe_screener import UniverseScreener
from src.enhanced_risk_manager import EnhancedRiskManager, RiskConfig

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Phase8Config:
    """Configuration for Phase 8 backtest."""
    # Universe
    universe_size: int = 1000
    min_price: float = 5.0
    min_volume: float = 1_000_000
    
    # Dates
    start_date: str = "2021-01-01"
    end_date: str = "2024-01-01"
    
    # Ensemble weights (base, regime will override)
    momentum_weight: float = 0.35
    tda_weight: float = 0.25
    value_weight: float = 0.20
    quality_weight: float = 0.20
    
    # Portfolio
    n_positions: int = 30
    rebalance_days: int = 20
    equal_weight: bool = True
    
    # Risk
    max_sector_weight: float = 0.30
    stop_loss_pct: float = 0.15
    
    # TDA
    tda_window: int = 60
    
    # Parallel
    n_data_workers: int = 20
    n_tda_workers: int = 4
    
    # Mode
    use_regime_rotation: bool = True


@dataclass 
class Phase8Metrics:
    """Performance metrics from backtest."""
    total_return: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    avg_trade_return: float
    n_trades: int
    
    # Sector metrics
    n_sectors: int
    max_sector_concentration: float
    sector_other_pct: float
    
    # Regime stats
    regime_distribution: Dict[str, int]


@dataclass
class Phase8Result:
    """Full result from Phase 8 backtest."""
    config: Phase8Config
    metrics: Phase8Metrics
    
    # Timing
    data_fetch_seconds: float
    sector_map_seconds: float
    tda_compute_seconds: float
    backtest_seconds: float
    total_seconds: float
    
    # Universe stats
    total_tickers_fetched: int
    tickers_after_screen: int
    tickers_with_tda: int
    
    # Daily data
    equity_curve: List[float]
    dates: List[str]
    
    def to_dict(self) -> Dict:
        return {
            'config': asdict(self.config),
            'metrics': asdict(self.metrics),
            'timing': {
                'data_fetch_seconds': self.data_fetch_seconds,
                'sector_map_seconds': self.sector_map_seconds,
                'tda_compute_seconds': self.tda_compute_seconds,
                'backtest_seconds': self.backtest_seconds,
                'total_seconds': self.total_seconds,
            },
            'universe': {
                'total_fetched': self.total_tickers_fetched,
                'after_screen': self.tickers_after_screen,
                'with_tda': self.tickers_with_tda,
            },
        }
    
    def save(self, path: str):
        """Save result to JSON."""
        result_dict = self.to_dict()
        # Equity curve too large for JSON, save summary
        result_dict['equity_summary'] = {
            'start': self.equity_curve[0] if self.equity_curve else None,
            'end': self.equity_curve[-1] if self.equity_curve else None,
            'n_days': len(self.equity_curve),
        }
        with open(path, 'w') as f:
            json.dump(result_dict, f, indent=2)


class Phase8Backtester:
    """
    Main Phase 8 backtester implementing ensemble multi-factor strategy.
    """
    
    def __init__(self, config: Phase8Config):
        self.config = config
        
        # Initialize components - use HybridDataProvider for clean column handling
        self.data_provider = Russell3000DataProvider(
            n_workers=config.n_data_workers,
        )
        # Also keep a hybrid provider reference for single-stock fetches
        self.hybrid_provider = HybridDataProvider()
        
        self.sector_mapper = GICSSectorMapper()
        self.tda_engine = ParallelTDAEngine(
            n_workers=config.n_tda_workers,
        )
        self.screener = UniverseScreener()
        self.ensemble = EnsembleFactorModel(
            base_weights=FactorWeights(
                momentum=config.momentum_weight,
                tda=config.tda_weight,
                value=config.value_weight,
                quality=config.quality_weight,
            ),
            use_regime_rotation=config.use_regime_rotation,
        )
        
        # Enhanced risk manager with tuned parameters for practical use
        # Less aggressive scaling to maintain market exposure
        risk_config = RiskConfig(
            max_allowed_drawdown=0.30,       # Scale down as DD approaches 30%
            min_position_scale=0.60,         # Never reduce below 60% invested
            target_annual_vol=0.18,          # Higher vol target for growth stocks
            vol_rebalance_threshold=0.30,    # Only rebalance on 30% vol deviation
            position_stop_loss=0.12,         # -12% position stop
            trailing_stop_pct=0.15,          # 15% trailing stop from peak
            circuit_breaker_dd=0.35,         # -35% circuit breaker (extreme only)
            max_position_weight=1.0 / config.n_positions,
            max_sector_weight=config.max_sector_weight,
            cost_per_trade_bps=10,           # 10 bps per trade
            min_alpha_to_cost_ratio=1.0,     # Trade when alpha > cost
            max_turnover_per_rebal=0.60,     # Allow 60% turnover per rebalance
        )
        self.risk_manager = EnhancedRiskManager(risk_config)
        
        # Data storage
        self.prices: Dict[str, pd.DataFrame] = {}
        self.tda_features: Dict[str, Dict] = {}
        self.sectors: Dict[str, str] = {}
        self.spy_prices: Optional[pd.DataFrame] = None
        
        # Show provider status
        self.hybrid_provider.print_status()
        
        logger.info(f"Initialized Phase8Backtester with {config.universe_size} target positions")
    
    def fetch_data(self) -> Tuple[Dict[str, pd.DataFrame], float]:
        """Fetch all price data."""
        start = time.time()
        
        # Get liquid universe
        tickers = list(LIQUID_UNIVERSE)[:self.config.universe_size]
        
        logger.info(f"Fetching data for {len(tickers)} tickers...")
        
        self.prices = self.data_provider.fetch_batch_parallel(
            tickers=tickers,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
        )
        
        # Filter out empty/invalid data
        self.prices = {
            t: df for t, df in self.prices.items()
            if df is not None and len(df) >= 252
        }
        
        # Fetch SPY for regime detection using hybrid provider
        try:
            self.spy_prices = self.hybrid_provider.get_ohlcv(
                'SPY', self.config.start_date, self.config.end_date
            )
        except Exception as e:
            logger.warning(f"Could not fetch SPY: {e}, using default regime")
            self.spy_prices = None
            self.spy_prices = None
        
        elapsed = time.time() - start
        logger.info(f"Fetched {len(self.prices)} valid tickers in {elapsed:.1f}s")
        
        return self.prices, elapsed
    
    def classify_sectors(self) -> Tuple[Dict[str, str], float]:
        """Classify all tickers by sector."""
        start = time.time()
        
        tickers = list(self.prices.keys())
        self.sectors = self.sector_mapper.batch_classify(tickers)
        
        # Calculate sector distribution
        sector_counts = {}
        for sector in self.sectors.values():
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        
        # Log distribution
        logger.info("Sector distribution:")
        for sector, count in sorted(sector_counts.items(), key=lambda x: -x[1])[:5]:
            pct = count / len(tickers) * 100
            logger.info(f"  {sector}: {count} ({pct:.1f}%)")
        
        other_count = sector_counts.get('Other', 0) + sector_counts.get('Diversified', 0)
        other_pct = other_count / len(tickers) * 100
        logger.info(f"  Other/Diversified: {other_count} ({other_pct:.1f}%)")
        
        elapsed = time.time() - start
        return self.sectors, elapsed
    
    def compute_tda_features(self) -> Tuple[Dict[str, Dict], float]:
        """Compute TDA features for all tickers."""
        start = time.time()
        
        logger.info(f"Computing TDA features for {len(self.prices)} tickers...")
        
        self.tda_features = self.tda_engine.compute_batch_tda(
            ohlcv_dict=self.prices,
            use_cache=True,
        )
        
        success_count = sum(1 for v in self.tda_features.values() if v)
        logger.info(f"TDA computed for {success_count}/{len(self.prices)} tickers")
        
        elapsed = time.time() - start
        return self.tda_features, elapsed
    
    def _get_close_price(self, df: pd.DataFrame, date) -> Optional[float]:
        """Safely get close price handling multi-level columns and case variations."""
        try:
            if date not in df.index:
                return None
            
            # Handle multi-level columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                if 'Close' in df.columns.get_level_values(0):
                    return float(df.loc[date, 'Close'].iloc[0])
                elif 'Adj Close' in df.columns.get_level_values(0):
                    return float(df.loc[date, 'Adj Close'].iloc[0])
                elif 'close' in df.columns.get_level_values(0):
                    return float(df.loc[date, 'close'].iloc[0])
            else:
                # Try different case variations
                for col in ['Close', 'close', 'Adj Close', 'adj close']:
                    if col in df.columns:
                        return float(df.loc[date, col])
            
            return None
        except Exception:
            return None
    
    def run_backtest(self) -> Tuple[Phase8Metrics, List[float], List[str], float]:
        """
        Run the actual backtest.
        
        Returns:
            metrics, equity_curve, dates, elapsed_seconds
        """
        start = time.time()
        
        # Filter price data to config date range
        start_dt = pd.Timestamp(self.config.start_date)
        end_dt = pd.Timestamp(self.config.end_date)
        
        filtered_prices = {}
        for ticker, df in self.prices.items():
            # Filter to date range
            mask = (df.index >= start_dt) & (df.index <= end_dt)
            filtered_df = df[mask]
            if len(filtered_df) >= 60:  # Minimum 60 days of data
                filtered_prices[ticker] = filtered_df
        
        self.prices = filtered_prices
        
        # Get common date range
        all_dates = None
        for ticker, df in self.prices.items():
            if all_dates is None:
                all_dates = set(df.index)
            else:
                all_dates &= set(df.index)
        
        all_dates = sorted(list(all_dates))
        
        if len(all_dates) < 252:
            logger.error("Not enough common trading days")
            return None, [], [], 0
        
        # Initialize tracking
        equity = 100.0
        equity_curve = [equity]
        dates_used = [str(all_dates[0].date())]
        
        portfolio: Dict[str, float] = {}  # ticker -> shares
        portfolio_weights: Dict[str, float] = {}  # ticker -> weight
        cash = equity
        
        trade_returns = []
        regime_counts = {k: 0 for k in REGIME_WEIGHTS.keys()}
        sector_weights_history = []
        
        # Reset risk manager
        self.risk_manager.portfolio_value = 100.0
        self.risk_manager.peak_value = 100.0
        self.risk_manager.equity_history = [100.0]
        self.risk_manager.returns_history = []
        self.risk_manager.drawdown_history = [0.0]
        
        # Rebalance schedule
        rebalance_dates = all_dates[::self.config.rebalance_days]
        
        logger.info(f"Running backtest over {len(all_dates)} days with {len(rebalance_dates)} rebalances...")
        
        for i, date in enumerate(all_dates[1:], 1):
            date_str = str(date.date())
            
            # Calculate daily portfolio value
            prev_equity = equity
            equity = cash
            
            current_prices = {}
            for ticker, shares in portfolio.items():
                if ticker in self.prices:
                    df = self.prices[ticker]
                    price = self._get_close_price(df, date)
                    if price is not None:
                        equity += shares * price
                        current_prices[ticker] = price
            
            # Update risk manager with new value
            self.risk_manager.update_portfolio_value(equity)
            
            # Check stop losses
            exits = self.risk_manager.check_stop_losses(current_prices)
            if exits:
                for ticker in exits:
                    if ticker in portfolio:
                        shares = portfolio[ticker]
                        if ticker in current_prices:
                            sell_value = shares * current_prices[ticker]
                            cash += sell_value
                            del portfolio[ticker]
                            if ticker in portfolio_weights:
                                del portfolio_weights[ticker]
            
            equity_curve.append(equity)
            dates_used.append(date_str)
            
            # Check if rebalance day
            if date in rebalance_dates:
                # Compute ensemble scores
                scores_df = self.ensemble.compute_composite_score(
                    prices=self.prices,
                    tda_features=self.tda_features,
                    spy_prices=self.spy_prices,
                    end_date=date_str,
                )
                
                regime = self.ensemble.last_regime
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
                
                # Get scores as dict for risk manager
                raw_scores = scores_df['composite'].to_dict()
                
                # Calculate risk-adjusted weights
                final_weights, risk_metrics = self.risk_manager.calculate_risk_adjusted_weights(
                    raw_scores=raw_scores,
                    current_weights=portfolio_weights,
                    sectors=self.sectors,
                    prices=self.prices,
                    n_positions=self.config.n_positions,
                )
                
                # Apply additional sector constraint
                selected = list(final_weights.keys())
                sector_counts = {}
                max_per_sector = int(self.config.n_positions * self.config.max_sector_weight)
                
                final_selected = []
                for ticker in selected:
                    sector = self.sectors.get(ticker, 'Diversified')
                    current = sector_counts.get(sector, 0)
                    
                    if current < max_per_sector:
                        final_selected.append(ticker)
                        sector_counts[sector] = current + 1
                
                # Sell all current positions
                for ticker, shares in list(portfolio.items()):
                    if ticker in self.prices:
                        df = self.prices[ticker]
                        price = self._get_close_price(df, date)
                        if price is not None:
                            sell_value = shares * price
                            cash += sell_value
                            
                            # Track trade return (simplified)
                            trade_returns.append(sell_value / (shares * 100) - 1)  # rough
                
                portfolio = {}
                portfolio_weights = {}
                
                # Buy new positions with risk-adjusted weights
                if final_selected:
                    # Get position scale from risk manager
                    position_scale = self.risk_manager.combined_position_scale()
                    
                    # Calculate total investable (after risk scaling)
                    investable = cash * position_scale
                    remaining_cash = cash - investable
                    
                    # Use risk-adjusted weights if available, else equal weight
                    if final_weights:
                        total_weight = sum(final_weights.get(t, 0) for t in final_selected)
                        for ticker in final_selected:
                            weight = final_weights.get(ticker, 0) / total_weight if total_weight > 0 else 1.0/len(final_selected)
                            target_value = investable * weight
                            
                            if ticker in self.prices:
                                df = self.prices[ticker]
                                price = self._get_close_price(df, date)
                                if price is not None and price > 0:
                                    shares = target_value / price
                                    portfolio[ticker] = shares
                                    portfolio_weights[ticker] = weight
                                    cash -= target_value
                    else:
                        weight = 1.0 / len(final_selected)
                        target_value = investable * weight
                        
                        for ticker in final_selected:
                            if ticker in self.prices:
                                df = self.prices[ticker]
                                price = self._get_close_price(df, date)
                                if price is not None and price > 0:
                                    shares = target_value / price
                                    portfolio[ticker] = shares
                                    portfolio_weights[ticker] = weight
                                    cash -= target_value
        
        # Get risk manager metrics
        risk_metrics = self.risk_manager.get_risk_metrics()
        
        # Calculate metrics
        equity_series = pd.Series(equity_curve, index=pd.to_datetime(dates_used))
        returns = equity_series.pct_change().dropna()
        
        total_return = equity_curve[-1] / equity_curve[0] - 1
        n_years = len(all_dates) / 252
        cagr = (equity_curve[-1] / equity_curve[0]) ** (1/n_years) - 1
        
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        neg_returns = returns[returns < 0]
        sortino = returns.mean() / neg_returns.std() * np.sqrt(252) if len(neg_returns) > 0 and neg_returns.std() > 0 else 0
        
        # Max drawdown
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax
        max_dd = drawdown.min()
        
        calmar = cagr / abs(max_dd) if max_dd < 0 else 0
        
        win_rate = len([r for r in trade_returns if r > 0]) / max(1, len(trade_returns))
        avg_trade = np.mean(trade_returns) if trade_returns else 0
        
        # Sector stats
        sector_set = set(self.sectors.values())
        other_count = sum(1 for s in self.sectors.values() if s in ('Other', 'Diversified'))
        other_pct = other_count / max(1, len(self.sectors))

        # Compute max sector concentration from final portfolio weights
        sector_weights: dict[str, float] = {}
        for ticker, weight in portfolio_weights.items():
            sec = self.sectors.get(ticker, 'Other')
            sector_weights[sec] = sector_weights.get(sec, 0.0) + weight
        max_sec_conc = max(sector_weights.values()) if sector_weights else 0.0

        metrics = Phase8Metrics(
            total_return=total_return,
            cagr=cagr,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            win_rate=win_rate,
            avg_trade_return=avg_trade,
            n_trades=len(trade_returns),
            n_sectors=len(sector_set),
            max_sector_concentration=max_sec_conc,
            sector_other_pct=other_pct,
            regime_distribution=regime_counts,
        )
        
        elapsed = time.time() - start
        return metrics, equity_curve, dates_used, elapsed
    
    def run_full_backtest(self) -> Phase8Result:
        """Run complete Phase 8 backtest pipeline."""
        total_start = time.time()
        
        print("\n" + "="*70)
        print("PHASE 8: ENSEMBLE MULTI-FACTOR BACKTEST")
        print("="*70)
        print(f"Start: {self.config.start_date} | End: {self.config.end_date}")
        print(f"Universe: {self.config.universe_size} | Positions: {self.config.n_positions}")
        print("="*70)
        
        # Step 1: Fetch data
        print("\n[1/4] Fetching price data...")
        prices, data_time = self.fetch_data()
        print(f"  ✓ Fetched {len(prices)} tickers in {data_time:.1f}s")
        
        # Step 2: Classify sectors
        print("\n[2/4] Classifying sectors...")
        sectors, sector_time = self.classify_sectors()
        print(f"  ✓ Classified {len(sectors)} tickers in {sector_time:.1f}s")
        
        # Step 3: Compute TDA
        print("\n[3/4] Computing TDA features...")
        tda_features, tda_time = self.compute_tda_features()
        tda_success = sum(1 for v in tda_features.values() if v)
        print(f"  ✓ Computed TDA for {tda_success} tickers in {tda_time:.1f}s")
        
        # Step 4: Run backtest
        print("\n[4/4] Running backtest...")
        metrics, equity_curve, dates, backtest_time = self.run_backtest()
        print(f"  ✓ Backtest complete in {backtest_time:.1f}s")
        
        total_time = time.time() - total_start
        
        # Build result
        result = Phase8Result(
            config=self.config,
            metrics=metrics,
            data_fetch_seconds=data_time,
            sector_map_seconds=sector_time,
            tda_compute_seconds=tda_time,
            backtest_seconds=backtest_time,
            total_seconds=total_time,
            total_tickers_fetched=len(prices),
            tickers_after_screen=len(prices),
            tickers_with_tda=tda_success,
            equity_curve=equity_curve,
            dates=dates,
        )
        
        # Print summary
        self._print_summary(result)
        
        return result
    
    def _print_summary(self, result: Phase8Result):
        """Print backtest summary."""
        m = result.metrics
        
        print("\n" + "="*70)
        print("PHASE 8 RESULTS")
        print("="*70)
        
        # Performance
        print("\n📈 PERFORMANCE:")
        print(f"  Total Return:  {m.total_return:>8.1%}")
        print(f"  CAGR:          {m.cagr:>8.1%}  {'✅' if m.cagr > 0.18 else '⚠️'} (target: >18%)")
        print(f"  Sharpe Ratio:  {m.sharpe_ratio:>8.2f}  {'✅' if m.sharpe_ratio > 1.2 else '⚠️'} (target: >1.2)")
        print(f"  Sortino Ratio: {m.sortino_ratio:>8.2f}")
        print(f"  Max Drawdown:  {m.max_drawdown:>8.1%}  {'✅' if m.max_drawdown > -0.15 else '⚠️'} (target: >-15%)")
        print(f"  Calmar Ratio:  {m.calmar_ratio:>8.2f}")
        
        # Trading
        print("\n📊 TRADING:")
        print(f"  Win Rate:      {m.win_rate:>8.1%}")
        print(f"  Avg Trade:     {m.avg_trade_return:>8.2%}")
        print(f"  Total Trades:  {m.n_trades:>8}")
        
        # Risk management stats
        risk_stats = self.risk_manager.get_risk_metrics()
        print("\n🛡️ RISK MANAGEMENT:")
        print(f"  Total Turnover:    {risk_stats['total_turnover']:>6.1%}")
        print(f"  Total Costs:       {risk_stats['total_costs_bps']:>6.1f} bps")
        print(f"  Stop-Loss Hits:    {risk_stats['stop_loss_triggers']:>6}")
        print(f"  Trailing Stops:    {risk_stats['trailing_stop_triggers']:>6}")
        print(f"  Circuit Breakers:  {risk_stats['circuit_breaker_triggers']:>6}")
        print(f"  Avg Realized Vol:  {risk_stats['avg_realized_vol']:>6.1%}")
        
        # Sector
        print("\n🏢 SECTOR DIVERSIFICATION:")
        print(f"  Sectors Used:      {m.n_sectors:>8}")
        print(f"  'Other' Pct:       {m.sector_other_pct:>8.1%}  {'✅' if m.sector_other_pct < 0.05 else '⚠️'} (target: <5%)")
        
        # Regime
        print("\n🌡️ REGIME DISTRIBUTION:")
        for regime, count in m.regime_distribution.items():
            if count > 0:
                print(f"  {regime:<12}: {count:>4}")
        
        # Timing
        print("\n⏱️ TIMING:")
        print(f"  Data Fetch:    {result.data_fetch_seconds:>6.1f}s")
        print(f"  Sector Map:    {result.sector_map_seconds:>6.1f}s")
        print(f"  TDA Compute:   {result.tda_compute_seconds:>6.1f}s")
        print(f"  Backtest:      {result.backtest_seconds:>6.1f}s")
        print(f"  TOTAL:         {result.total_seconds:>6.1f}s  {'✅' if result.total_seconds < 300 else '⚠️'} (target: <5 min)")
        
        # Summary
        targets_met = sum([
            m.cagr > 0.18,
            m.sharpe_ratio > 1.2,
            m.max_drawdown > -0.15,
            m.sector_other_pct < 0.05,
            result.total_seconds < 300,
        ])
        
        print("\n" + "="*70)
        if targets_met == 5:
            print("🏆 ALL TARGETS MET! Phase 8 Successful!")
        else:
            print(f"⚠️ {targets_met}/5 targets met. Further optimization needed.")
        print("="*70)


def run_phase8_backtest(
    universe_size: int = 500,
    n_positions: int = 30,
    quick_mode: bool = False,
) -> Phase8Result:
    """
    Run Phase 8 backtest with specified parameters.
    
    Args:
        universe_size: Number of stocks in universe
        n_positions: Number of positions to hold
        quick_mode: If True, use smaller dataset for faster testing
    """
    if quick_mode:
        config = Phase8Config(
            universe_size=100,
            n_positions=20,
            start_date="2022-01-01",  # 2 years for proper backtest
            end_date="2024-01-01",
            rebalance_days=21,
        )
    else:
        # Full production backtest
        config = Phase8Config(
            universe_size=universe_size,
            n_positions=n_positions,
            start_date="2021-06-01",  # 3 years including bull and bear
            end_date="2024-06-01",
            rebalance_days=21,
        )
    
    backtester = Phase8Backtester(config)
    result = backtester.run_full_backtest()
    
    # Save results
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    result.save(str(results_dir / 'phase8_ensemble_results.json'))
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Phase 8 Ensemble Backtest')
    parser.add_argument('--universe', type=int, default=500, help='Universe size')
    parser.add_argument('--positions', type=int, default=30, help='Number of positions')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    
    args = parser.parse_args()
    
    result = run_phase8_backtest(
        universe_size=args.universe,
        n_positions=args.positions,
        quick_mode=args.quick,
    )
