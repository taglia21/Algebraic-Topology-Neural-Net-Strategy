"""
Asset-Level Performance Filtering for Dynamic Asset Selection.

V2.0: Iteration 2 Optimization
Implements rolling performance tracking per asset to dynamically adjust allocation
or pause trading for consistently underperforming assets.

Key Features:
- Rolling Sharpe ratio calculation per asset
- Asset eligibility checking based on performance
- Volatility percentile-based filtering
- Dynamic allocation adjustment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AssetPerformance:
    """Rolling performance metrics for a single asset."""
    ticker: str
    trade_returns: deque = field(default_factory=lambda: deque(maxlen=60))
    daily_returns: deque = field(default_factory=lambda: deque(maxlen=180))  # ~6 months
    volatility_history: deque = field(default_factory=lambda: deque(maxlen=252))  # 1 year
    win_count: int = 0
    loss_count: int = 0
    total_pnl: float = 0.0
    
    @property
    def num_trades(self) -> int:
        return len(self.trade_returns)
    
    @property
    def win_rate(self) -> float:
        total = self.win_count + self.loss_count
        return self.win_count / total if total > 0 else 0.5
    
    @property
    def rolling_sharpe(self) -> float:
        """Calculate rolling Sharpe ratio from daily returns."""
        if len(self.daily_returns) < 20:
            return 0.0
        
        returns = np.array(self.daily_returns)
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        
        if std_ret <= 0:
            return 0.0
        
        # Annualized Sharpe
        return (mean_ret / std_ret) * np.sqrt(252)
    
    @property
    def avg_trade_return(self) -> float:
        if len(self.trade_returns) == 0:
            return 0.0
        return np.mean(list(self.trade_returns))
    
    @property
    def volatility_percentile(self) -> float:
        """Current volatility as percentile of historical."""
        if len(self.volatility_history) < 20:
            return 0.5  # Neutral if insufficient history
        
        vol_array = np.array(self.volatility_history)
        current_vol = vol_array[-1]
        percentile = np.sum(vol_array < current_vol) / len(vol_array)
        return percentile


class AssetFilter:
    """
    Dynamic asset selection based on rolling performance.
    
    Trading Rules:
    - If asset Sharpe < min_sharpe: reduce allocation
    - If asset Sharpe < 0 persistently: pause trading
    - If volatility > 70th percentile: reduce position size
    
    Prevents capital drain from consistently underperforming assets.
    """
    
    def __init__(
        self,
        min_sharpe_threshold: float = 0.3,
        pause_sharpe_threshold: float = 0.0,
        min_trades_for_evaluation: int = 5,
        volatility_reduction_percentile: float = 0.70,
        volatility_pause_percentile: float = 0.90,
        lookback_days: int = 180
    ):
        """
        Initialize AssetFilter.
        
        Args:
            min_sharpe_threshold: Reduce allocation if Sharpe below this
            pause_sharpe_threshold: Pause trading if Sharpe below this
            min_trades_for_evaluation: Minimum trades before applying filter
            volatility_reduction_percentile: Reduce size if vol above this
            volatility_pause_percentile: Pause if vol above this
            lookback_days: Days of history to track
        """
        self.min_sharpe_threshold = min_sharpe_threshold
        self.pause_sharpe_threshold = pause_sharpe_threshold
        self.min_trades_for_evaluation = min_trades_for_evaluation
        self.volatility_reduction_percentile = volatility_reduction_percentile
        self.volatility_pause_percentile = volatility_pause_percentile
        self.lookback_days = lookback_days
        
        # Performance tracking per asset
        self.asset_performance: Dict[str, AssetPerformance] = {}
        
        # Filter statistics
        self.filter_stats = {
            'total_checks': 0,
            'allowed': 0,
            'reduced_allocation': 0,
            'paused': 0,
            'volatility_reduced': 0,
            'volatility_paused': 0
        }
        
        logger.info(f"AssetFilter initialized: min_sharpe={min_sharpe_threshold}, "
                   f"pause_sharpe={pause_sharpe_threshold}, vol_reduce={volatility_reduction_percentile:.0%}")
    
    def register_asset(self, ticker: str):
        """Register an asset for tracking."""
        if ticker not in self.asset_performance:
            self.asset_performance[ticker] = AssetPerformance(ticker=ticker)
            logger.debug(f"Registered asset: {ticker}")
    
    def record_trade(self, ticker: str, trade_return: float, is_win: bool):
        """
        Record a completed trade for an asset.
        
        Args:
            ticker: Asset ticker
            trade_return: Percentage return of the trade
            is_win: Whether trade was profitable
        """
        if ticker not in self.asset_performance:
            self.register_asset(ticker)
        
        perf = self.asset_performance[ticker]
        perf.trade_returns.append(trade_return)
        perf.total_pnl += trade_return
        
        if is_win:
            perf.win_count += 1
        else:
            perf.loss_count += 1
    
    def record_daily_return(self, ticker: str, daily_return: float):
        """Record a daily return for Sharpe calculation."""
        if ticker not in self.asset_performance:
            self.register_asset(ticker)
        
        self.asset_performance[ticker].daily_returns.append(daily_return)
    
    def record_volatility(self, ticker: str, volatility: float):
        """Record current volatility for percentile calculation."""
        if ticker not in self.asset_performance:
            self.register_asset(ticker)
        
        self.asset_performance[ticker].volatility_history.append(volatility)
    
    def check_asset_eligibility(
        self,
        ticker: str,
        current_volatility: float = None
    ) -> Tuple[bool, float, str]:
        """
        Check if an asset is eligible for trading.
        
        Args:
            ticker: Asset ticker
            current_volatility: Current realized volatility (optional)
            
        Returns:
            - eligible: Whether to trade (True/False)
            - size_multiplier: Position size multiplier (0.0 to 1.5)
            - reason: Explanation string
        """
        self.filter_stats['total_checks'] += 1
        
        if ticker not in self.asset_performance:
            self.register_asset(ticker)
            return True, 1.0, "New asset - trade normally"
        
        perf = self.asset_performance[ticker]
        
        # Check volatility first (if provided)
        if current_volatility is not None:
            perf.volatility_history.append(current_volatility)
            vol_pct = perf.volatility_percentile
            
            if vol_pct >= self.volatility_pause_percentile:
                self.filter_stats['volatility_paused'] += 1
                return False, 0.0, f"Volatility at {vol_pct:.0%} percentile - PAUSED"
            
            if vol_pct >= self.volatility_reduction_percentile:
                self.filter_stats['volatility_reduced'] += 1
                # Scale down linearly from 1.0 at 70% to 0.5 at 90%
                reduction = (vol_pct - self.volatility_reduction_percentile) / \
                           (self.volatility_pause_percentile - self.volatility_reduction_percentile)
                size_mult = max(0.5, 1.0 - 0.5 * reduction)
                return True, size_mult, f"High volatility ({vol_pct:.0%} pctl) - size reduced to {size_mult:.1%}"
        
        # Need minimum trades before evaluating performance
        if perf.num_trades < self.min_trades_for_evaluation:
            self.filter_stats['allowed'] += 1
            return True, 1.0, f"Insufficient trades ({perf.num_trades}) - trade normally"
        
        # Check rolling Sharpe
        sharpe = perf.rolling_sharpe
        
        if sharpe < self.pause_sharpe_threshold:
            self.filter_stats['paused'] += 1
            return False, 0.0, f"Sharpe {sharpe:.2f} < {self.pause_sharpe_threshold} - PAUSED"
        
        if sharpe < self.min_sharpe_threshold:
            self.filter_stats['reduced_allocation'] += 1
            # Linear reduction from 1.0 at min_sharpe to 0.5 at pause_sharpe
            range_size = self.min_sharpe_threshold - self.pause_sharpe_threshold
            position_in_range = (sharpe - self.pause_sharpe_threshold) / range_size if range_size > 0 else 0.5
            size_mult = 0.5 + 0.5 * position_in_range
            return True, size_mult, f"Sharpe {sharpe:.2f} - allocation reduced to {size_mult:.0%}"
        
        self.filter_stats['allowed'] += 1
        
        # Good performance - possibly increase allocation
        if sharpe > 1.0:
            size_mult = min(1.25, 1.0 + (sharpe - 1.0) * 0.25)
            return True, size_mult, f"Strong Sharpe {sharpe:.2f} - size increased to {size_mult:.0%}"
        
        return True, 1.0, f"Sharpe {sharpe:.2f} - trade normally"
    
    def get_asset_ranking(self) -> List[Tuple[str, float]]:
        """
        Get assets ranked by rolling Sharpe ratio.
        
        Returns:
            List of (ticker, sharpe) tuples sorted by Sharpe descending
        """
        rankings = []
        for ticker, perf in self.asset_performance.items():
            rankings.append((ticker, perf.rolling_sharpe))
        
        return sorted(rankings, key=lambda x: x[1], reverse=True)
    
    def get_optimal_weights(
        self,
        tickers: List[str],
        min_weight: float = 0.05,
        max_weight: float = 0.50
    ) -> Dict[str, float]:
        """
        Calculate optimal portfolio weights based on rolling performance.
        
        Uses performance-weighted allocation:
        - Assets with Sharpe > 0.5 get higher weights
        - Assets with Sharpe < 0 get minimum weight or excluded
        
        Args:
            tickers: List of tickers to weight
            min_weight: Minimum weight per asset
            max_weight: Maximum weight per asset
            
        Returns:
            Dict of ticker -> weight (sums to 1.0)
        """
        sharpe_scores = {}
        
        for ticker in tickers:
            if ticker in self.asset_performance:
                sharpe = self.asset_performance[ticker].rolling_sharpe
            else:
                sharpe = 0.0  # No history - neutral
            
            # Transform Sharpe to positive score
            # Negative Sharpe -> very low score
            # Sharpe 0 -> score 1
            # Sharpe 1 -> score 2
            score = max(0.1, 1.0 + sharpe)
            sharpe_scores[ticker] = score
        
        # Normalize to weights
        total_score = sum(sharpe_scores.values())
        weights = {}
        
        for ticker, score in sharpe_scores.items():
            raw_weight = score / total_score if total_score > 0 else 1.0 / len(tickers)
            weights[ticker] = max(min_weight, min(max_weight, raw_weight))
        
        # Re-normalize to sum to 1.0
        weight_sum = sum(weights.values())
        weights = {t: w / weight_sum for t, w in weights.items()}
        
        return weights
    
    def get_filter_summary(self) -> Dict[str, Any]:
        """Get summary of filter activity."""
        summary = {
            'filter_stats': self.filter_stats.copy(),
            'asset_performance': {}
        }
        
        for ticker, perf in self.asset_performance.items():
            summary['asset_performance'][ticker] = {
                'num_trades': perf.num_trades,
                'win_rate': round(perf.win_rate, 4),
                'rolling_sharpe': round(perf.rolling_sharpe, 4),
                'avg_trade_return': round(perf.avg_trade_return, 4),
                'total_pnl': round(perf.total_pnl, 4),
                'volatility_percentile': round(perf.volatility_percentile, 4)
            }
        
        return summary
    
    def print_summary(self):
        """Print summary of asset filter performance."""
        print("\n" + "=" * 60)
        print("ASSET FILTER SUMMARY")
        print("=" * 60)
        
        print(f"\n  Filter Activity:")
        print(f"    Total checks: {self.filter_stats['total_checks']}")
        print(f"    Allowed: {self.filter_stats['allowed']}")
        print(f"    Reduced allocation: {self.filter_stats['reduced_allocation']}")
        print(f"    Paused (Sharpe): {self.filter_stats['paused']}")
        print(f"    Volatility reduced: {self.filter_stats['volatility_reduced']}")
        print(f"    Volatility paused: {self.filter_stats['volatility_paused']}")
        
        print(f"\n  Asset Performance:")
        print(f"    {'Ticker':<8} {'Trades':>8} {'Win Rate':>10} {'Sharpe':>10} {'Avg Ret':>10}")
        print("    " + "-" * 48)
        
        for ticker, perf in self.asset_performance.items():
            print(f"    {ticker:<8} {perf.num_trades:>8} {perf.win_rate:>10.1%} "
                  f"{perf.rolling_sharpe:>10.2f} {perf.avg_trade_return:>9.2%}")
        
        print("=" * 60)


def test_asset_filter():
    """Test AssetFilter functionality."""
    print("\nTesting AssetFilter...")
    
    af = AssetFilter(
        min_sharpe_threshold=0.3,
        pause_sharpe_threshold=0.0,
        min_trades_for_evaluation=3
    )
    
    # Register assets
    for ticker in ['SPY', 'QQQ', 'IWM']:
        af.register_asset(ticker)
    
    # Simulate trades
    # SPY: good performer
    for _ in range(5):
        af.record_trade('SPY', 0.02, True)
        af.record_daily_return('SPY', 0.001)
    for i in range(20):
        af.record_daily_return('SPY', np.random.normal(0.001, 0.01))
    
    # QQQ: mediocre performer
    for _ in range(3):
        af.record_trade('QQQ', 0.01, True)
    for _ in range(3):
        af.record_trade('QQQ', -0.015, False)
    for i in range(20):
        af.record_daily_return('QQQ', np.random.normal(-0.0005, 0.015))
    
    # IWM: poor performer
    for _ in range(6):
        af.record_trade('IWM', -0.02, False)
    for i in range(20):
        af.record_daily_return('IWM', np.random.normal(-0.002, 0.02))
    
    # Check eligibility
    print("\nEligibility checks:")
    for ticker in ['SPY', 'QQQ', 'IWM']:
        eligible, size_mult, reason = af.check_asset_eligibility(ticker)
        print(f"  {ticker}: eligible={eligible}, size_mult={size_mult:.2f}, reason={reason}")
    
    # Get rankings
    print("\nAsset rankings:")
    for ticker, sharpe in af.get_asset_ranking():
        print(f"  {ticker}: Sharpe={sharpe:.2f}")
    
    # Get optimal weights
    weights = af.get_optimal_weights(['SPY', 'QQQ', 'IWM'])
    print("\nOptimal weights:")
    for ticker, weight in weights.items():
        print(f"  {ticker}: {weight:.1%}")
    
    af.print_summary()
    
    print("\n  ✓ All AssetFilter tests passed")


if __name__ == "__main__":
    test_asset_filter()
